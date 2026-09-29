"""Named network operators (spec 7.5). Each compiles to edits on a copy."""

from __future__ import annotations

from typing import Any

from lab.schemas import Network
from lab.toolkit.edits import ConsequenceReport, apply_edits_raw, consequence_report
from lab.toolkit.motifs import list_motifs
import networkx as nx

from lab.toolkit.build import to_undirected

READ_ONLY = {"shift_scale", "take_perspective"}


def _fresh_id(net: Network, prefix: str, used: set[str]) -> str:
    i = 1
    existing = {e.id for e in net.edges} | {n.id for n in net.nodes} | used
    while f"{prefix}{i:03d}" in existing:
        i += 1
    return f"{prefix}{i:03d}"


def compile_operator(net: Network, op: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    a = args
    if op == "flip_valency":
        e = net.edge(a["edge"])
        p = a["perspective"]
        if p not in e.valency:
            raise ValueError(f"edge {e.id} has no valency for {p} (unknown, cannot flip)")
        return [{"op": "update_edge", "id": e.id, "fields": {"valency": {**e.valency, p: -e.valency[p]}}}]
    if op == "close_loop":
        e = net.edge(a["edge"])
        c = a["counter"]
        cid = c.get("id") or _fresh_id(net, "e", set())
        new = {"id": cid, "source": e.target, "target": e.source, "contribution": c["contribution"],
               "flow_type": c.get("flow_type", "recognition"), "form": c.get("form", "gift"),
               "reciprocity": "reciprocated", "counter_edge": e.id,
               "evidence": [{"source": "operator:close_loop", "confidence": "hypothesis"}]}
        return [{"op": "add_edge", "edge": new},
                {"op": "update_edge", "id": e.id, "fields": {"reciprocity": "reciprocated", "counter_edge": cid}}]
    if op == "open_loop":
        e = net.edge(a["edge"])
        out: list[dict[str, Any]] = [
            {"op": "update_edge", "id": e.id, "fields": {"reciprocity": "pending", "counter_edge": None}}]
        if e.counter_edge:
            net.edge(e.counter_edge)
            out.append({"op": "remove_edge", "id": e.counter_edge})
        return out
    if op == "insert_mediator":
        e = net.edge(a["edge"])
        m = a["mediator"]
        out = []
        mid = m if isinstance(m, str) else m["id"]
        if isinstance(m, str):
            out.append({"op": "update_node", "id": mid, "fields": {"mediator": True}})
        else:
            out.append({"op": "add_node", "node": {**m, "mediator": True}})
        e1, e2 = _fresh_id(net, "e", set()), None
        e2 = _fresh_id(net, "e", {e1})
        base = e.model_dump(exclude={"id", "source", "target", "counter_edge", "reciprocity"})
        if e.counter_edge and any(x.id == e.counter_edge for x in net.edges):
            out.append({"op": "update_edge", "id": e.counter_edge, "fields": {"counter_edge": None, "reciprocity": "pending"}})
        out += [{"op": "remove_edge", "id": e.id},
                {"op": "add_edge", "edge": {**base, "id": e1, "source": e.source, "target": mid, "reciprocity": "unknown"}},
                {"op": "add_edge", "edge": {**base, "id": e2, "source": mid, "target": e.target, "reciprocity": "unknown"}}]
        return out
    if op == "split_actor":
        node = net.node(a["node"])
        parts = a["parts"]
        pids = [p["id"] for p in parts]
        out = [{"op": "add_node", "node": {**{"kind": node.kind, "classification": node.classification},
                                          **p, "label": p.get("label", p["id"]),
                                          "evidence": [{"source": "operator:split_actor", "confidence": "hypothesis"}]}}
               for p in parts]
        used: set[str] = set()
        for e in net.edges:
            if node.id not in (e.source, e.target):
                continue
            for pid in pids:
                eid = _fresh_id(net, "e", used)
                used.add(eid)
                d = e.model_dump(exclude={"id", "counter_edge"})
                d["valency"] = {(pid if k == node.id else k): v for k, v in d["valency"].items()}
                if d["expectation"] and d["expectation"]["holder"] == node.id:
                    d["expectation"]["holder"] = pid
                if d["expectation"] and d["expectation"].get("addressee") == node.id:
                    d["expectation"]["addressee"] = pid
                if e.reciprocity == "indirect":
                    d["reciprocity"] = "unknown"
                d["source"] = pid if e.source == node.id else e.source
                d["target"] = pid if e.target == node.id else e.target
                out.append({"op": "add_edge", "edge": {**d, "id": eid}})
        out.append({"op": "remove_node", "id": node.id})
        return out
    if op == "merge_actors":
        ids = list(a["nodes"])
        into = a.get("into") or "_".join(ids)
        first = net.node(ids[0])
        for i in ids:
            net.node(i)
        out = [{"op": "add_node", "node": {"id": into, "label": a.get("label", into), "kind": first.kind,
                                            "classification": first.classification, "aggregates": ids,
                                            "evidence": [{"source": "operator:merge_actors", "confidence": "hypothesis"}]}}]
        used: set[str] = set()
        for e in net.edges:
            if e.source in ids and e.target in ids:
                continue
            if e.source not in ids and e.target not in ids:
                continue
            eid = _fresh_id(net, "e", used)
            used.add(eid)
            d = e.model_dump(exclude={"id", "counter_edge"})
            d["source"] = into if e.source in ids else e.source
            d["target"] = into if e.target in ids else e.target
            d["valency"] = {(into if k in ids else k): v for k, v in d["valency"].items()}
            if d["expectation"] and d["expectation"]["holder"] in ids:
                d["expectation"]["holder"] = into
            if d["expectation"] and d["expectation"].get("addressee") in ids:
                d["expectation"]["addressee"] = into
            if e.reciprocity == "indirect":
                d["reciprocity"] = "unknown"
            out.append({"op": "add_edge", "edge": {**d, "id": eid}})
        out += [{"op": "remove_node", "id": i} for i in ids]
        return out
    if op == "reclassify":
        net.node(a["node"])
        return [{"op": "update_node", "id": a["node"], "fields": {"classification": a["category"], "anomaly": None}}]
    if op == "rebind":
        hit = [e for e in net.edges if e.source == a["from"] and e.contribution == a["contribution"]]
        if not hit:
            raise ValueError(f"no contribution {a['contribution']!r} given by {a['from']}")
        net.node(a["to"])
        e = hit[0]
        f: dict[str, Any] = {"source": a["to"]}
        if e.expectation and e.expectation.holder == a["from"]:
            f["expectation"] = e.expectation.model_copy(update={"holder": a["to"]})
        return [{"op": "update_edge", "id": e.id, "fields": f}]
    if op == "extend_boundary":
        out = [{"op": "add_node", "node": {**n, "evidence": n.get("evidence") or [{"source": "operator:extend_boundary", "confidence": "hypothesis"}]}}
               for n in a["actors"]]
        out += [{"op": "add_edge", "edge": e} for e in a.get("edges", [])]
        return out
    if op in READ_ONLY:
        return []
    raise ValueError(f"unknown operator {op!r}")


def _reading(net: Network, op: str, a: dict[str, Any], cycle_max_len: int, ego_radius: int) -> dict[str, Any] | None:
    if op == "shift_scale":
        t, scale = a["target"], a["scale"]
        ms = [m for m in list_motifs(net, cycle_max_len) if t in m.nodes or t in m.id]
        return {"target": t, "scale": scale, "motifs_at_scale": [m.id for m in ms if m.scale == scale],
                "motifs_other_scale": [m.id for m in ms if m.scale != scale]}
    if op == "take_perspective":
        actor = a["actor"]
        ug = to_undirected(net)
        dist = nx.single_source_shortest_path_length(ug, actor)
        ranked = sorted(list_motifs(net, cycle_max_len), key=lambda m: (min((dist.get(n, 99) for n in m.nodes), default=99), m.id))
        return {"actor": actor, "ranked_motifs": [(m.id, min((dist.get(n, 99) for n in m.nodes), default=99)) for m in ranked]}
    return None


def apply_operators(net: Network, chain: list[dict[str, Any]], cycle_max_len: int = 6,
                    ego_radius: int = 2, seed: int = 1) -> ConsequenceReport:
    """chain: [{"op": "flip_valency", "args": {...}}, ...]. Applied in order to a copy."""
    cur = net
    reading = None
    for step in chain:
        op, args = step["op"], step.get("args", {})
        edits = compile_operator(cur, op, args)
        cur = apply_edits_raw(cur, edits)
        r = _reading(cur, op, args, cycle_max_len, ego_radius)
        reading = r or reading
    rep = consequence_report(net, cur, chain=chain, cycle_max_len=cycle_max_len, ego_radius=ego_radius, seed=seed)
    rep.reading = reading
    return rep
