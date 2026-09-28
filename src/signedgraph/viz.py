"""Interactive HTML rendering with pyvis."""

from __future__ import annotations

from pathlib import Path

from pyvis.network import Network

from signedgraph.schema import SignedGraph

POSITIVE_COLOR = "#2b8a3e"
NEGATIVE_COLOR = "#c92a2a"


def to_pyvis(graph: SignedGraph) -> Network:
    """Green edges are +1, red dashed edges are -1; hovering an edge shows its passage."""
    net = Network(directed=True, height="700px", width="100%", cdn_resources="in_line")
    for n in graph.nodes:
        net.add_node(n.id, label=n.display)
    for e in graph.edges:
        positive = e.sign == 1
        title = e.passage if e.citation is None else f"{e.passage}\n— {e.citation}"
        net.add_edge(
            e.source,
            e.target,
            label="+" if positive else "−",
            title=title,
            color=POSITIVE_COLOR if positive else NEGATIVE_COLOR,
            dashes=not positive,
            arrows="to",
        )
    return net


def write_html(graph: SignedGraph, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(to_pyvis(graph).generate_html(), encoding="utf-8")
    return path
