"""Analogue catalogue and matching (spec 7.7): theme score plus motif score."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx
import yaml
from networkx.algorithms.isomorphism import DiGraphMatcher
from pydantic import BaseModel, Field

from lab.schemas import Motif, Network

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "analogues"


class Analogue(BaseModel):
    id: str
    name: str
    description: str
    themes: list[str]
    nodes: list[str]
    edges: list[dict[str, Any]]  # {source, target, flow_type?, form?}
    status: str = "starter"  # starter | proposed
    extra: dict[str, Any] = Field(default_factory=dict)

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        for e in self.edges:
            g.add_edge(e["source"], e["target"], form=e.get("form"), flow_type=e.get("flow_type"))
        return g


def load_catalogue(extra_dir: Path | None = None) -> list[Analogue]:
    files = sorted(DATA_DIR.glob("*.yaml"))
    if extra_dir and extra_dir.exists():
        files += sorted(extra_dir.glob("*.yaml"))
    return [Analogue.model_validate(yaml.safe_load(f.read_text())) for f in files]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{4,}", text.lower())}


def theme_score(theme_label: str, a: Analogue) -> float:
    """Token overlap between the theme and the analogue's themes/description. Placeholder for
    embedding or LLM judgement in a real run."""
    tt = _tokens(theme_label)
    if not tt:
        return 0.0
    at = _tokens(" ".join(a.themes) + " " + a.description)
    return round(len(tt & at) / len(tt), 4)


def motif_graph(net: Network, m: Motif) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(m.nodes)
    for eid in m.edges:
        e = net.edge(eid)
        g.add_edge(e.source, e.target, form=e.form, flow_type=e.flow_type)
    return g


def motif_score(motif_g: nx.DiGraph, analogue_g: nx.DiGraph) -> float:
    """1.0 for isomorphic structure, 0.6 when the motif embeds in the analogue, else 0."""
    if motif_g.number_of_edges() == 0:
        return 0.0
    if analogue_g.number_of_nodes() < motif_g.number_of_nodes() or analogue_g.number_of_edges() < motif_g.number_of_edges():
        return 0.0
    if (analogue_g.number_of_nodes() == motif_g.number_of_nodes()
            and analogue_g.number_of_edges() == motif_g.number_of_edges()
            and nx.weisfeiler_lehman_graph_hash(motif_g) == nx.weisfeiler_lehman_graph_hash(analogue_g)
            and nx.is_isomorphic(motif_g, analogue_g)):
        return 1.0
    matcher = DiGraphMatcher(analogue_g, motif_g)
    return 0.6 if matcher.subgraph_is_monomorphic() else 0.0


def find_analogues(net: Network, catalogue: list[Analogue], theme_id: str | None = None,
                   motif_pattern: str | None = None, motifs: list[Motif] | None = None,
                   k: int = 5) -> list[dict[str, Any]]:
    theme = next((t for t in net.themes if t.id == theme_id), None) if theme_id else None
    if theme_id and theme is None:
        raise KeyError(f"unknown theme {theme_id!r}")
    cands = [m for m in (motifs or []) if motif_pattern and m.pattern == motif_pattern and m.edges]
    rows = []
    for a in catalogue:
        ts = theme_score(theme.label, a) if theme else None
        ms = None
        if cands:
            ms = max(motif_score(motif_graph(net, m), a.graph()) for m in cands)
        parts = [x for x in (ts, ms) if x is not None]
        score = sum(parts) / len(parts) if parts else 0.0
        rows.append({"analogue": a.id, "name": a.name, "theme_score": ts, "motif_score": ms,
                     "score": round(score, 4), "status": a.status})
    rows.sort(key=lambda r: (-r["score"], r["analogue"]))
    return rows[:k]
