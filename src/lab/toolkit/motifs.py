"""Structural motif detection: the structural twins of themes (spec 6.4)."""

from __future__ import annotations

import networkx as nx

from lab.schemas import Motif, Network
from lab.toolkit.build import to_simple_digraph


def list_motifs(net: Network, cycle_max_len: int = 6, scale: str | None = None) -> list[Motif]:
    out: list[Motif] = []
    n = 0

    def add(label: str, pattern: str, nodes: list[str], edges: list[str], sc: str) -> None:
        nonlocal n
        n += 1
        out.append(Motif(id=f"X{n:02d}", label=label, pattern=pattern, nodes=nodes, edges=edges, scale=sc))

    for e in sorted(net.edges, key=lambda e: e.id):
        if e.reciprocity in ("pending", "refused") and not e.counter_edge:
            add(f"{e.source} gives '{e.contribution}' to {e.target}; nothing comes back ({e.reciprocity})",
                "open_reciprocity_loop", [e.source, e.target], [e.id], "local")
    for e in sorted(net.edges, key=lambda e: e.id):
        vals = set(e.valency.values())
        if len(vals) > 1:
            add(f"Actors disagree on the sign of {e.id} ({e.source} to {e.target})",
                "valency_conflict", [e.source, e.target], [e.id], "local")
    sd = to_simple_digraph(net)
    cycles = sorted((c for c in nx.simple_cycles(sd, length_bound=cycle_max_len) if len(c) >= 3),
                    key=lambda c: (len(c), sorted(c)))
    for c in cycles:
        ids = [sd[c[i]][c[(i + 1) % len(c)]]["ids"][0] for i in range(len(c))]
        add(f"Exchange cycle through {' > '.join(c)}", "indirect_reciprocity_cycle", list(c), ids, "global")
    for nd in sorted(net.nodes, key=lambda x: x.id):
        if nd.anomaly:
            add(f"{nd.id} does not fit its category: {nd.anomaly}", "anomaly", [nd.id], [], "local")
        if nd.aggregates:
            add(f"{nd.id} lumps {nd.aggregates}", "lumped_actor", [nd.id], [], "local")
        if nd.mediator:
            add(f"{nd.id} transforms what passes through it", "mediator", [nd.id], [], "local")
    out += list(net.motifs)
    return [m for m in out if scale is None or m.scale == scale]
