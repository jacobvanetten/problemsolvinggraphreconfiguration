"""World Café schedule (spec 9 Phase 4). Greedy with seeded restarts; fails loudly."""

from __future__ import annotations

import random

import networkx as nx
from dataclasses import dataclass


class ScheduleError(RuntimeError):
    pass


@dataclass
class TableInfo:
    id: str
    authors: set[str]
    executors: set[str]  # Trial 1 executors (core members)


@dataclass
class Params:
    core: list[str]
    newcomers: list[str]
    core_per_table: int = 2
    new_per_table: int = 2
    executed_eligible_from_round: int = 3  # 0 = strict: never visit an executed table
    pv: str = "PV"


# schedule[round][table] = {"core": [...], "newcomers": [...]}
Schedule = dict[int, dict[str, dict[str, list[str]]]]


def _core_ok(p: Params, t: TableInfo, person: str, rnd: int) -> bool:
    if person in t.authors:
        return False
    if person in t.executors:
        return p.executed_eligible_from_round > 0 and rnd >= p.executed_eligible_from_round
    return True


def plan(tables: list[TableInfo], params: Params, rounds: list[int], seed: int,
         history: dict[str, set[str]] | None = None, restarts: int = 400) -> Schedule:
    history = history or {}
    last_err = "no attempt"
    for r in range(restarts):
        rng = random.Random(f"{seed}:{r}")
        try:
            return _attempt(tables, params, rounds, rng, {k: set(v) for k, v in history.items()})
        except ScheduleError as exc:
            last_err = str(exc)
    raise ScheduleError(f"no schedule satisfies the constraints after {restarts} restarts ({last_err}). "
                        "Options: fewer rounds, fewer core visits per table, or relax executed-table exclusion.")


def _attempt(tables: list[TableInfo], p: Params, rounds: list[int], rng: random.Random,
             visited: dict[str, set[str]]) -> Schedule:
    out: Schedule = {}
    load: dict[str, int] = {x: 0 for x in [*p.core, *p.newcomers]}
    for rnd in rounds:
        slots = {t.id: {"core": [], "newcomers": []} for t in tables}
        rload = {x: 0 for x in load}
        tmap = {t.id: t for t in tables}
        # core: a max-flow matching persons (quota) to tables (capacity), so no greedy dead ends
        need_total = len(tables) * p.core_per_table
        order = list(p.core)
        rng.shuffle(order)
        quota = {x: need_total // len(order) + (1 if i < need_total % len(order) else 0) for i, x in enumerate(order)}
        g = nx.DiGraph()
        for x in order:
            g.add_edge("S", ("p", x), capacity=quota[x])
        tl = [t for t in tables]
        rng.shuffle(tl)
        for t in tl:
            g.add_edge(("t", t.id), "T", capacity=p.core_per_table)
            cand = [x for x in order if _core_ok(p, t, x, rnd) and t.id not in visited.get(x, set())]
            for x in cand:
                g.add_edge(("p", x), ("t", t.id), capacity=1)
        val, flow = nx.maximum_flow(g, "S", "T")
        if val != need_total:
            raise ScheduleError(f"round {rnd}: only {val} of {need_total} core visits can be placed")
        for x in order:
            for node, f in flow[("p", x)].items():
                if f:
                    tid = node[1]
                    slots[tid]["core"].append(x)
                    visited.setdefault(x, set()).add(tid)
                    load[x] += 1
                    rload[x] += 1
        for tid, s in slots.items():
            if len(s["core"]) != p.core_per_table:
                raise ScheduleError(f"round {rnd}: table {tid} got {len(s['core'])} core visitors")
        # newcomers: least total load first (PV wins ties), max 2 per round, never the same table twice
        tids = [t.id for t in tables]
        rng.shuffle(tids)
        for tid in tids:
            for _ in range(p.new_per_table):
                cands = [x for x in p.newcomers if tid not in visited.get(x, set()) and rload[x] < 2
                         and x not in slots[tid]["newcomers"]]
                if not cands:
                    raise ScheduleError(f"round {rnd}: no newcomer free for table {tid}")
                rng.shuffle(cands)
                cands.sort(key=lambda x: (load[x], rload[x], x != p.pv))
                x = cands[0]
                slots[tid]["newcomers"].append(x)
                visited.setdefault(x, set()).add(tid)
                load[x] += 1
                rload[x] += 1
        out[rnd] = slots
    return out


def validate(schedule: Schedule, tables: list[TableInfo], p: Params) -> list[str]:
    """Return a list of violations (empty = schedule satisfies Phase 4's constraints)."""
    bad: list[str] = []
    tmap = {t.id: t for t in tables}
    seen: dict[str, set[str]] = {}
    tot: dict[str, int] = {x: 0 for x in [*p.core, *p.newcomers]}
    new_at_table: dict[str, list[str]] = {t.id: [] for t in tables}
    for rnd, tabs in schedule.items():
        cvis: dict[str, int] = {}
        for tid, s in tabs.items():
            if len(s["core"]) != p.core_per_table or len(s["newcomers"]) != p.new_per_table:
                bad.append(f"r{rnd} {tid}: wrong visitor counts")
            for c in s["core"]:
                if not _core_ok(p, tmap[tid], c, rnd):
                    bad.append(f"r{rnd}: {c} visits {tid} which they authored/executed")
                cvis[c] = cvis.get(c, 0) + 1
            for x in [*s["core"], *s["newcomers"]]:
                if tid in seen.get(x, set()):
                    bad.append(f"{x} visits {tid} twice")
                seen.setdefault(x, set()).add(tid)
                tot[x] += 1
            new_at_table[tid] += s["newcomers"]
        want = len(tables) * p.core_per_table
        lo, hi = want // len(p.core), -(-want // len(p.core))
        for c in p.core:
            if not lo <= cvis.get(c, 0) <= hi:
                bad.append(f"r{rnd}: core {c} made {cvis.get(c, 0)} visits")
    nv = [tot[x] for x in p.newcomers]
    if nv and max(nv) - min(nv) > 1:
        bad.append(f"newcomer visits unbalanced: {dict(zip(p.newcomers, nv))}")
    return bad
