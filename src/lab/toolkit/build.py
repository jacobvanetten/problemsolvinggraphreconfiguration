"""Load exchange networks and derive networkx views (spec 7.1)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import networkx as nx
import yaml

from lab.schemas import HUMAN_KINDS, Network


def load_network(path: str | Path) -> Network:
    path = Path(path)
    if path.suffix == ".graphml":
        return from_graphml(path)
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    return Network.model_validate(data)


def from_graphml(path: Path) -> Network:
    """Map a bare graphml onto the schema. Missing fields get conservative defaults."""
    g = nx.read_graphml(path)
    nodes = [{"id": str(n), "label": d.get("label", str(n)), "kind": d.get("kind", "organisation")}
             for n, d in g.nodes(data=True)]
    edges = []
    for i, (u, v, d) in enumerate(g.edges(data=True), 1):
        edges.append({"id": d.get("id", f"e{i:03d}"), "source": str(u), "target": str(v),
                      "contribution": d.get("contribution", d.get("label", "unspecified")),
                      "flow_type": d.get("flow_type", "material"), "form": d.get("form", "gift"),
                      "evidence": [{"source": str(path.name), "confidence": "hypothesis"}]})
    return Network.model_validate({"nodes": nodes, "edges": edges})


def check_size(net: Network, max_nodes_warn: int = 300) -> list[str]:
    notes = []
    if len(net.nodes) > max_nodes_warn:
        msg = f"network has {len(net.nodes)} nodes (warn above {max_nodes_warn}); cycle search may be slow"
        warnings.warn(msg)
        notes.append(msg)
    if not (30 <= len(net.nodes) <= 80):
        notes.append(f"node count {len(net.nodes)} outside the target 30-80")
    if not (60 <= len(net.edges) <= 250):
        notes.append(f"edge count {len(net.edges)} outside the target 60-250")
    return notes


def to_multidigraph(net: Network) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in net.nodes:
        g.add_node(n.id, label=n.label, kind=n.kind, human=n.kind in HUMAN_KINDS)
    for e in net.edges:
        g.add_edge(e.source, e.target, key=e.id, id=e.id, flow_type=e.flow_type, form=e.form,
                   reciprocity=e.reciprocity, valency=dict(e.valency))
    return g


def to_simple_digraph(net: Network) -> nx.DiGraph:
    """Collapse parallel edges; keep the ids on the collapsed edge."""
    g = nx.DiGraph()
    for n in net.nodes:
        g.add_node(n.id, kind=n.kind)
    for e in net.edges:
        if g.has_edge(e.source, e.target):
            g[e.source][e.target]["ids"].append(e.id)
        else:
            g.add_edge(e.source, e.target, ids=[e.id], flow_type=e.flow_type, form=e.form,
                       reciprocity=e.reciprocity)
    return g


def to_undirected(net: Network) -> nx.Graph:
    return to_simple_digraph(net).to_undirected()
