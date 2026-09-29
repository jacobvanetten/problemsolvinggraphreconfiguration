"""Local and global network properties (spec 7.2 to 7.4)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from lab.schemas import Network
from lab.toolkit.build import to_multidigraph, to_simple_digraph, to_undirected
from lab.toolkit.perspectives import get_perspective, unspoken_actors


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def local_metrics(net: Network, ego_radius: int = 2) -> dict[str, dict[str, Any]]:
    md = to_multidigraph(net)
    sd = to_simple_digraph(net)
    ug = to_undirected(net)
    clustering = nx.clustering(ug)
    cons = nx.constraint(ug) if ug.number_of_edges() else {}
    eff = nx.effective_size(ug) if ug.number_of_edges() else {}
    out: dict[str, dict[str, Any]] = {}
    for n in net.nodes:
        flows: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        for _, _, d in md.out_edges(n.id, data=True):
            flows[d["flow_type"]]["out"] += 1
        for _, _, d in md.in_edges(n.id, data=True):
            flows[d["flow_type"]]["in"] += 1
        outs = list(sd.successors(n.id))
        direct = sum(1 for v in outs if sd.has_edge(v, n.id))
        p = get_perspective(net, n.id, ego_radius)
        c = cons.get(n.id)
        out[n.id] = {
            "degree_by_flow": {k: dict(v) for k, v in sorted(flows.items())},
            "dyadic_reciprocity": _r(direct / len(outs)) if outs else None,
            "indirect_edges": sum(1 for e in net.edges if e.source == n.id and e.reciprocity == "indirect"),
            "open_obligations": len(p.open_obligations),
            "owed": len(p.owed),
            "valency_sum": p.perceived_valency_sum,
            "clustering": _r(clustering.get(n.id)),
            "constraint": None if c is None or c != c else _r(c),
            "effective_size": _r(eff.get(n.id)),
            "ego1": sorted(nx.ego_graph(ug, n.id, 1).nodes),
            "ego_r": sorted(nx.ego_graph(ug, n.id, ego_radius).nodes),
        }
    return out


def _cycle_edges(sd: nx.DiGraph, lo: int, hi: int) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for cyc in nx.simple_cycles(sd, length_bound=hi):
        if len(cyc) < lo:
            continue
        for i, u in enumerate(cyc):
            seen.add((u, cyc[(i + 1) % len(cyc)]))
    return seen


def balance_by_perspective(net: Network) -> dict[str, dict[str, Any]]:
    """Structural balance on the signed, undirected projection of each perspective's valencies."""
    holders = sorted({a for e in net.edges for a in e.valency})
    res: dict[str, dict[str, Any]] = {}
    for h in holders:
        g = nx.Graph()
        for e in net.edges:
            if h in e.valency:
                s = e.valency[h]
                if g.has_edge(e.source, e.target):
                    s = g[e.source][e.target]["sign"] * s  # parallel signed edges multiply
                g.add_edge(e.source, e.target, sign=s)
        tri = bal = 0
        for tset in _triangles(g):
            a, b, c = tset
            prod = g[a][b]["sign"] * g[b][c]["sign"] * g[a][c]["sign"]
            tri += 1
            bal += prod > 0
        res[h] = {"signed_edges": g.number_of_edges(), "triangles": tri,
                  "balanced_fraction": _r(bal / tri) if tri else None}
    return res


def _triangles(g: nx.Graph) -> list[tuple[str, str, str]]:
    out = set()
    for u in g:
        for v in g[u]:
            for w in g[v]:
                if w != u and g.has_edge(u, w):
                    out.add(tuple(sorted((u, v, w))))
    return sorted(out)


def global_metrics(net: Network, cycle_max_len: int = 6, seed: int = 1) -> dict[str, Any]:
    sd = to_simple_digraph(net)
    ug = sd.to_undirected()
    n = ug.number_of_nodes()
    e_pairs = set(sd.edges)
    direct = {p for p in e_pairs if (p[1], p[0]) in e_pairs}
    gen = _cycle_edges(sd, 3, cycle_max_len)
    degs = [d for _, d in ug.degree()]
    if n > 2 and degs:
        mx = max(degs)
        cent = sum(mx - d for d in degs) / ((n - 1) * (n - 2))
    else:
        cent = 0.0
    if ug.number_of_edges():
        comms = nx.community.louvain_communities(ug, seed=seed)
        comm_sorted = sorted(sorted(c) for c in comms)
    else:
        comm_sorted = [[x] for x in sorted(ug.nodes)]
    btw = nx.betweenness_centrality(sd)
    top = sorted(btw.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    m = max(len(e_pairs), 1)
    return {
        "nodes": n, "edges": len(net.edges),
        "density": _r(nx.density(sd)),
        "components": nx.number_weakly_connected_components(sd) if n else 0,
        "communities": comm_sorted,
        "degree_centralisation": _r(cent),
        "reciprocity_direct": _r(len(direct) / m),
        "reciprocity_generalised": _r(len(gen) / m),
        "structural_balance": balance_by_perspective(net),
        "bottlenecks": [(k, _r(v)) for k, v in top if v > 0],
        "unspoken_actors": unspoken_actors(net),
    }


def get_metrics(net: Network, scope: str = "global", cycle_max_len: int = 6, ego_radius: int = 2,
                seed: int = 1) -> dict[str, Any]:
    if scope == "global":
        return global_metrics(net, cycle_max_len, seed)
    if scope == "local":
        return local_metrics(net, ego_radius)
    if scope == "all":
        return {"global": global_metrics(net, cycle_max_len, seed), "local": local_metrics(net, ego_radius)}
    net.node(scope)  # scope is an actor id
    return local_metrics(net, ego_radius)[scope]


def most_central(net: Network, k: int = 3) -> list[tuple[str, int]]:
    ug = to_undirected(net)
    return sorted(ug.degree, key=lambda kv: (-kv[1], kv[0]))[:k]


def most_constrained(net: Network, k: int = 3) -> list[tuple[str, float]]:
    ug = to_undirected(net)
    if not ug.number_of_edges():
        return []
    c = {n: v for n, v in nx.constraint(ug).items() if v == v}
    return sorted(((n, round(v, 3)) for n, v in c.items()), key=lambda kv: (-kv[1], kv[0]))[:k]
