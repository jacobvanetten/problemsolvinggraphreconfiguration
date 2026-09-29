"""Generic edits on a copy of the network, plus the consequence report (spec 7.5).

Every named operator compiles to a list of edits and goes through `apply_edits`, so an
operator and its hand-written edit list give the same report by construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from lab.schemas import Edge, Network, Node
from lab.toolkit.metrics import global_metrics, local_metrics
from lab.toolkit.perspectives import get_perspective


class ConsequenceReport(BaseModel):
    chain: list[dict[str, Any]] = Field(default_factory=list)
    diff: dict[str, list[str]]
    global_changes: dict[str, list[Any]]  # metric -> [before, after]
    local_changes: dict[str, dict[str, list[Any]]]  # actor -> metric -> [before, after]
    affected_perspectives: list[str]
    narratives: dict[str, str]
    local_global_disagreements: list[str]
    reading: dict[str, Any] | None = None  # read-only operators (shift_scale, take_perspective)
    network: Network = Field(exclude=True)

    def summary(self) -> str:
        head = (f"+nodes {self.diff['nodes_added']} -nodes {self.diff['nodes_removed']} "
                f"+edges {self.diff['edges_added']} -edges {self.diff['edges_removed']} "
                f"~edges {self.diff['edges_changed']}")
        gl = "; ".join(f"{k}: {a}->{b}" for k, (a, b) in self.global_changes.items()) or "no global change"
        return f"{head}\nglobal: {gl}\nagreement flags: {len(self.local_global_disagreements)}"


def _edit_op(net: Network, ed: dict[str, Any]) -> Network:
    op = ed["op"]
    nodes = list(net.nodes)
    edges = list(net.edges)
    if op == "add_node":
        nodes.append(Node.model_validate(ed["node"]))
    elif op == "remove_node":
        nid = ed["id"]
        net.node(nid)
        nodes = [n for n in nodes if n.id != nid]
        edges = [e for e in edges if nid not in (e.source, e.target)]
    elif op == "update_node":
        nodes = [n.model_copy(update=ed["fields"]) if n.id == ed["id"] else n for n in nodes]
        net.node(ed["id"])
    elif op == "add_edge":
        edges.append(Edge.model_validate(ed["edge"]))
    elif op == "remove_edge":
        net.edge(ed["id"])
        edges = [e for e in edges if e.id != ed["id"]]
    elif op == "update_edge":
        net.edge(ed["id"])
        edges = [e.model_copy(update=ed["fields"]) if e.id == ed["id"] else e for e in edges]
    else:
        raise ValueError(f"unknown edit op {op!r}")
    # rebuild so all validators run again (models are copied, not mutated)
    return Network.model_validate(
        {"nodes": [n.model_dump() for n in nodes], "edges": [e.model_dump() for e in edges],
         "themes": [t.model_dump() for t in net.themes], "motifs": [m.model_dump() for m in net.motifs],
         "links": [k.model_dump() for k in net.links]}
    )


def apply_edits_raw(net: Network, edits: list[dict[str, Any]]) -> Network:
    out = net
    for ed in edits:
        out = _edit_op(out, ed)
    return out


def _diff(a: Network, b: Network) -> dict[str, list[str]]:
    an = {n.id: n for n in a.nodes}
    bn = {n.id: n for n in b.nodes}
    ae = {e.id: e for e in a.edges}
    be = {e.id: e for e in b.edges}
    return {
        "nodes_added": sorted(set(bn) - set(an)),
        "nodes_removed": sorted(set(an) - set(bn)),
        "nodes_changed": sorted(i for i in set(an) & set(bn) if an[i] != bn[i]),
        "edges_added": sorted(set(be) - set(ae)),
        "edges_removed": sorted(set(ae) - set(be)),
        "edges_changed": sorted(i for i in set(ae) & set(be) if ae[i] != be[i]),
    }


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def consequence_report(before: Network, after: Network, chain: list[dict[str, Any]] | None = None,
                       cycle_max_len: int = 6, ego_radius: int = 2, seed: int = 1) -> ConsequenceReport:
    diff = _diff(before, after)
    gb, ga = global_metrics(before, cycle_max_len, seed), global_metrics(after, cycle_max_len, seed)
    gchg = {k: [gb[k], ga[k]] for k in ("density", "components", "degree_centralisation",
                                        "reciprocity_direct", "reciprocity_generalised")
            if gb[k] != ga[k]}
    if [len(c) for c in gb["communities"]] != [len(c) for c in ga["communities"]]:
        gchg["communities"] = [len(gb["communities"]), len(ga["communities"])]
    if gb["structural_balance"] != ga["structural_balance"]:
        gchg["structural_balance"] = [gb["structural_balance"], ga["structural_balance"]]
    lb, la = local_metrics(before, ego_radius), local_metrics(after, ego_radius)
    lchg: dict[str, dict[str, list[Any]]] = {}
    keys = ("open_obligations", "owed", "valency_sum", "dyadic_reciprocity", "effective_size", "constraint")
    for actor in sorted(set(lb) | set(la)):
        b, a = lb.get(actor, {}), la.get(actor, {})
        ch = {k: [b.get(k), a.get(k)] for k in keys if b.get(k) != a.get(k)}
        if ch:
            lchg[actor] = ch
    touched = set(diff["nodes_added"]) | set(diff["nodes_removed"]) | set(diff["nodes_changed"])
    for eid in diff["edges_added"] + diff["edges_changed"]:
        e = after.edge(eid)
        touched |= {e.source, e.target}
    for eid in diff["edges_removed"]:
        e = before.edge(eid)
        touched |= {e.source, e.target}
    affected = sorted(set(lchg) | {t for t in touched if t in {n.id for n in after.nodes}})
    narratives: dict[str, str] = {}
    for actor in affected:
        if actor not in {n.id for n in after.nodes}:
            continue
        pa = get_perspective(after, actor, ego_radius)
        pb = get_perspective(before, actor, ego_radius) if actor in {n.id for n in before.nodes} else None
        gain = sorted(set(pa.receives) - set(pb.receives)) if pb else pa.receives
        lose = sorted(set(pb.receives) - set(pa.receives)) if pb else []
        new_obl = sorted(set(pa.open_obligations) - set(pb.open_obligations)) if pb else pa.open_obligations
        narratives[actor] = (
            f"{actor} gains {gain or 'nothing new'}; loses {lose or 'nothing'}; "
            f"new obligation {new_obl or 'none'}. [hypothesis: structural reading, not lived view]"
        )
    # local vs global: flag actors whose local direction opposes the global direction
    g_dir = _sign(sum((ga[k] or 0) - (gb[k] or 0) for k in ("reciprocity_direct", "reciprocity_generalised")))
    dis = []
    for actor, ch in lchg.items():
        if "open_obligations" in ch or "owed" in ch:
            b, a = lb.get(actor, {}), la.get(actor, {})
            l_dir = _sign((b.get("open_obligations", 0) + b.get("owed", 0)) - (a.get("open_obligations", 0) + a.get("owed", 0)))
            if g_dir and l_dir and l_dir != g_dir:
                dis.append(f"{actor}: open-loop load {'falls' if l_dir > 0 else 'rises'} while network reciprocity "
                           f"{'rises' if g_dir > 0 else 'falls'} (flagged, not resolved)")
    return ConsequenceReport(chain=chain or [], diff=diff, global_changes=gchg, local_changes=lchg,
                             affected_perspectives=affected, narratives=narratives,
                             local_global_disagreements=dis, network=after)


def apply_edits(net: Network, edits: list[dict[str, Any]], **kw: Any) -> ConsequenceReport:
    return consequence_report(net, apply_edits_raw(net, edits), chain=[{"edits": edits}], **kw)
