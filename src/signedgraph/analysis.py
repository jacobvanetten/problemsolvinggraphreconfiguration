"""Path signs, cycle polarity and edge-flip reports for a :class:`SignedGraph`.

The sign of a path or cycle is the product of its edge signs. All enumeration is
over *simple* paths/cycles, which is exponential in the worst case: fine for
hand-built or LLM-extracted graphs of tens of nodes, not for large networks.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from math import prod

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ConfigDict

from signedgraph.schema import Sign, SignedGraph


class NetEffect(StrEnum):
    """Aggregate influence of one node on another across all simple paths."""

    POSITIVE = "positive"  # every path is +1
    NEGATIVE = "negative"  # every path is -1
    MIXED = "mixed"  # paths of both signs: the net effect is ambiguous
    NONE = "none"  # no path


class Polarity(StrEnum):
    REINFORCING = "reinforcing"  # cycle sign +1
    BALANCING = "balancing"  # cycle sign -1


class SignedPath(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[str, ...]
    sign: Sign
    passages: tuple[str, ...]

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.nodes, self.nodes[1:]))


class Cycle(BaseModel):
    """A simple cycle. ``nodes`` starts at its smallest id and does not repeat it at the end."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[str, ...]
    sign: Sign
    passages: tuple[str, ...]

    @property
    def polarity(self) -> Polarity:
        return Polarity.REINFORCING if self.sign == 1 else Polarity.BALANCING

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        ring = self.nodes + self.nodes[:1]
        return tuple(zip(ring, ring[1:]))


class CycleChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[str, ...]
    before: Polarity
    after: Polarity


class EffectChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    before: NetEffect
    after: NetEffect
    paths_through_edge: int
    total_paths: int


class FlipReport(BaseModel):
    """What changes when the edge ``source -> target`` has its sign reversed.

    ``cycle_changes`` lists every cycle through the edge (each one flips polarity).
    ``effect_changes`` lists only node pairs whose :class:`NetEffect` changed; pairs
    whose paths all flip together still count (positive <-> negative), while pairs
    that stay ``mixed`` are omitted even if some of their paths flipped.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    old_sign: Sign
    new_sign: Sign
    cycle_changes: tuple[CycleChange, ...]
    effect_changes: tuple[EffectChange, ...]

    def effect_changes_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [c.model_dump(mode="json") for c in self.effect_changes],
            columns=list(EffectChange.model_fields),
        )

    def summary(self) -> str:
        lines = [f"Flip {self.source} -> {self.target}: {self.old_sign:+d} => {self.new_sign:+d}"]
        if not self.cycle_changes and not self.effect_changes:
            lines.append("  no cycle or net-effect changes")
        for c in self.cycle_changes:
            lines.append(f"  cycle {' -> '.join(c.nodes)}: {c.before} => {c.after}")
        for e in self.effect_changes:
            lines.append(
                f"  effect {e.source} on {e.target}: {e.before} => {e.after}"
                f" ({e.paths_through_edge}/{e.total_paths} paths use the edge)"
            )
        return "\n".join(lines)


def _passages(g: nx.DiGraph, edges: Iterable[tuple[str, str]]) -> tuple[str, ...]:
    return tuple(g.edges[u, v]["passage"] for u, v in edges)


def _sign(g: nx.DiGraph, edges: Iterable[tuple[str, str]]) -> Sign:
    return prod(g.edges[u, v]["sign"] for u, v in edges)  # type: ignore[return-value]


def _require_node(graph: SignedGraph, node: str) -> None:
    if node not in {n.id for n in graph.nodes}:
        raise KeyError(f"unknown node {node!r}")


def path_signs(
    graph: SignedGraph, source: str, target: str, cutoff: int | None = None
) -> list[SignedPath]:
    """All simple paths ``source -> target`` (at most ``cutoff`` edges) with their signs.

    Paths are sorted by length, then lexicographically, so output is deterministic.
    """
    _require_node(graph, source)
    _require_node(graph, target)
    if source == target:
        return []
    g = graph.to_networkx()
    out = []
    for nodes in nx.all_simple_paths(g, source, target, cutoff=cutoff):
        edges = list(zip(nodes, nodes[1:]))
        out.append(SignedPath(nodes=tuple(nodes), sign=_sign(g, edges), passages=_passages(g, edges)))
    return sorted(out, key=lambda p: (len(p.nodes), p.nodes))


def net_effect(paths: Iterable[SignedPath]) -> NetEffect:
    signs = {p.sign for p in paths}
    if not signs:
        return NetEffect.NONE
    if signs == {1}:
        return NetEffect.POSITIVE
    if signs == {-1}:
        return NetEffect.NEGATIVE
    return NetEffect.MIXED


def cycles(graph: SignedGraph) -> list[Cycle]:
    """All simple cycles with their sign and polarity, in canonical rotation."""
    g = graph.to_networkx()
    out = []
    for raw in nx.simple_cycles(g):
        i = raw.index(min(raw))
        nodes = tuple(raw[i:] + raw[:i])
        ring = nodes + nodes[:1]
        edges = list(zip(ring, ring[1:]))
        out.append(Cycle(nodes=nodes, sign=_sign(g, edges), passages=_passages(g, edges)))
    return sorted(out, key=lambda c: (len(c.nodes), c.nodes))


def cycles_dataframe(graph: SignedGraph) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"nodes": " -> ".join(c.nodes), "length": len(c.nodes), "sign": c.sign, "polarity": str(c.polarity)}
            for c in cycles(graph)
        ],
        columns=["nodes", "length", "sign", "polarity"],
    )


def flip_report(graph: SignedGraph, source: str, target: str) -> FlipReport:
    """Report cycle-polarity and net-effect changes caused by flipping one edge."""
    old = graph.edge(source, target)
    flipped = graph.with_flipped_edge(source, target)
    edge = (source, target)

    after_cycles = {c.nodes: c for c in cycles(flipped)}
    cycle_changes = tuple(
        CycleChange(nodes=c.nodes, before=c.polarity, after=after_cycles[c.nodes].polarity)
        for c in cycles(graph)
        if edge in c.edges
    )

    # Only pairs (a, b) with a ->* source and target ->* b can have a path using the edge.
    g = graph.to_networkx()
    starts = sorted(nx.ancestors(g, source) | {source})
    ends = sorted(nx.descendants(g, target) | {target})
    effect_changes = []
    for a in starts:
        for b in ends:
            if a == b:
                continue
            before = path_signs(graph, a, b)
            through = sum(edge in p.edges for p in before)
            if through == 0:
                continue
            after = path_signs(flipped, a, b)
            e_before, e_after = net_effect(before), net_effect(after)
            if e_before != e_after:
                effect_changes.append(
                    EffectChange(
                        source=a,
                        target=b,
                        before=e_before,
                        after=e_after,
                        paths_through_edge=through,
                        total_paths=len(before),
                    )
                )

    return FlipReport(
        source=source,
        target=target,
        old_sign=old.sign,
        new_sign=-old.sign,  # type: ignore[arg-type]
        cycle_changes=cycle_changes,
        effect_changes=tuple(effect_changes),
    )
