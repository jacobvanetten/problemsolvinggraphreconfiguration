"""Phase 4: World Café on methods (spec 9)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lab.cafe.drift import drift_report
from lab.cafe.ledger import Ledger
from lab.cafe.schedule import Params, ScheduleError, TableInfo, plan, validate
from lab.methods.store import add_version, check_protocol, current, load_history
from lab.phases.common import active_methods, best_frame, trial_dir
from lab.run import Run
from lab.schemas import CafeItems, MethodProtocol, Reception, Revision, Text, words

QUESTIONS = [
    "What in this method did work in the trial, and what did not? What from your practice could replace a weak step?",
    "Whose perspective does this method fail to reach, and at what scale does it fail? Could a team of 4 to 10 run it?",
    "Which component from another table, or from your own practice, would most strengthen this method?",
]


class MergeProposal(BaseModel):
    pairs: list[list[str]] = []


class Agree(BaseModel):
    agree: bool
    reason: str


def _tables(run: Run) -> list[TableInfo]:
    reg = run.registry()
    out = []
    for m in active_methods(run):
        v = reg[m]
        out.append(TableInfo(m, set(v["authors"]), set(v["executors"])))
    return out


def _params(run: Run) -> Params:
    c = run.cfg
    return Params(core=run.p("methods", "cycle.json").exists() and run.read_json(run.p("methods", "cycle.json")) or run.core_ids,
                  newcomers=[v for v in c.people.visitors_enabled if v in run.people],
                  core_per_table=c.cafe.core_visitors_per_table, new_per_table=c.cafe.newcomer_visitors_per_table,
                  executed_eligible_from_round=c.cafe.executed_tables_eligible_from_round)


def _protocol_brief(run: Run, mid: str) -> dict[str, str]:
    p = current(run, mid)
    return {"name": p.name, "purpose": p.purpose}


def _visitor_check(run: Run, persona_group: str, pid: str):
    def check(obj: CafeItems) -> None:
        for it in obj.items:
            if words(it.text) > run.cfg.cafe.item_max_words:
                raise ValueError("item too long")
        if persona_group == "visitor":
            if obj.must_produce_in is None:
                raise ValueError("visitor integrity: the must_produce element is missing")
        if pid == "PV" and not obj.reversal_of:
            raise ValueError("PV's challenge must be a reversal of the method's central step (set reversal_of)")
    return check


def run_visit(run: Run, ledger: Ledger, rnd: int, mid: str, visitor: str, host: str, seen: dict[str, list[str]]) -> None:
    p = run.persona(visitor)
    reg = run.registry()
    proto = current(run, mid)
    t1 = trial_dir(run, "T1", mid)
    frames = (t1 / "frames.json").read_text() if (t1 / "frames.json").exists() else "{}"
    aar = (t1 / "aar.md").read_text() if (t1 / "aar.md").exists() else ""
    own = [m for m, v in reg.items() if visitor in v["authors"]]
    graft = {"own_methods": {m: _protocol_brief(run, m) for m in own}} if p.group == "core" else {"own_practice": p.practice}
    if rnd > 1:
        graft["tables_visited_earlier"] = {m: _protocol_brief(run, m) for m in seen.get(visitor, [])}
    ctx = {"round": rnd, "round_question": QUESTIONS[min(rnd, len(QUESTIONS)) - 1], "table": mid, "protocol": proto.render_md(),
           "trial1_frames": frames, "trial1_aar": aar, "graft_sources": graft, "must_produce": p.must_produce,
           "your_group": p.group}
    got = run.llm.call(task="cafe_items", schema=CafeItems, persona=p, context=ctx,
                       check=_visitor_check(run, p.group, visitor)).obj
    ids = [ledger.add(giver=visitor, group=p.group, table=mid, rnd=rnd, type_=i.type, text=i.text) for i in got.items]
    if all(ledger.get(i)["response"] for i in ids):
        return
    hp = run.persona(host) if host in run.people else run.persona("FA")
    rec = run.llm.call(task="reception", schema=Reception, persona=hp,
                       context={"table": mid, "round": rnd, "protocol": proto.render_md(),
                                "items": [{"id": i, "type": ledger.get(i)["type"], "text": ledger.get(i)["text"]} for i in ids]},
                       check=lambda r: _all_answered(r, ids)).obj
    for r in rec.responses:
        ledger.respond(r.item_id, r.decision, r.reason)
    ledger.save()


def _all_answered(rec: Reception, ids: list[str]) -> None:
    got = {r.item_id for r in rec.responses}
    if got != set(ids):
        raise ValueError(f"every item needs a response; missing {sorted(set(ids) - got)}, unknown {sorted(got - set(ids))}")


def revise_table(run: Run, ledger: Ledger, rnd: int, mid: str, host: str) -> None:
    stage = f"cafe_r{rnd}"
    if any(v.stage == stage for v in load_history(run, mid)):
        return
    items = ledger.for_table_round(mid, rnd)
    usable = {i["id"] for i in items if i["response"] in ("adopt", "adapt")}
    proto = current(run, mid)
    pcheck = check_protocol(run)

    def check(rev: Revision) -> None:
        pcheck(rev)
        for c in rev.changelog:
            bad = [x for x in c.ledger_ids if x not in usable]
            if bad:
                raise ValueError(f"changelog names ledger items that were not adopted or adapted here: {bad}")

    persona = run.persona(host) if host in run.people else run.persona("FA")
    rev = run.llm.call(task="revision", schema=Revision, persona=persona,
                       context={"protocol": proto.model_dump(), "ledger_items_with_responses": items,
                                "limits": {"max_words": run.cfg.method.max_words, "trial_max_calls": run.cfg.trials.max_calls}},
                       check=check).obj
    add_version(run, mid, rev.protocol, stage, rev.changelog)
    for c in rev.changelog:
        for x in c.ledger_ids:
            ledger.mark_used(x)
    ledger.save()


def maybe_merge(run: Run, after_round: int, harvest: str) -> bool:
    active = active_methods(run)
    prop = run.llm.call(task="merge_check", schema=MergeProposal, persona=run.persona("FA"),
                        context={"harvest": harvest, "methods": {m: _protocol_brief(run, m) for m in active}}).obj
    reg = run.registry()
    merged = False
    for pair in prop.pairs:
        if len(pair) != 2 or any(m not in active for m in pair) or len(active) - 1 < run.cfg.cafe.min_methods_for_ranking:
            continue
        votes = [run.llm.call(task="merge_agree", schema=Agree, persona=run.persona(reg[m]["authors"][0] if reg[m]["authors"][0] in run.people else "FA"),
                              context={"merge": pair, "protocols": {x: current(run, x).model_dump() for x in pair}}).obj.agree
                 for m in pair]
        if not all(votes):
            continue
        new = f"M{max(int(k[1:]) for k in reg if k[1:].isdigit()) + 1:02d}"
        lead = run.persona(reg[pair[0]]["authors"][0]) if reg[pair[0]]["authors"][0] in run.people else run.persona("FA")
        proto = run.llm.call(task="merge_methods", schema=MethodProtocol, persona=lead,
                             context={"protocols": {x: current(run, x).model_dump() for x in pair},
                                      "limits": {"max_words": run.cfg.method.max_words, "trial_max_calls": run.cfg.trials.max_calls}},
                             check=check_protocol(run)).obj
        add_version(run, new, proto, f"merged_after_r{after_round}")
        authors = sorted(set(reg[pair[0]]["authors"]) | set(reg[pair[1]]["authors"]))
        reg[new] = {"pair_index": None, "authors": authors, "executors": reg[pair[0]]["executors"],
                    "source": "agent", "active": True, "merged_from": pair}
        for m in pair:
            reg[m]["active"] = False
            reg[m]["merged_into"] = new
        run.save_registry(reg)
        active = active_methods(run)
        merged = True
    return merged


def run_phase(run: Run) -> bool:
    cfg = run.cfg.cafe
    ledger = Ledger(run)
    tables = _tables(run)
    params = _params(run)
    rounds = list(range(1, cfg.rounds + 1))
    sched = plan(tables, params, rounds, run.cfg.seed, restarts=cfg.schedule_restarts)
    bad = validate(sched, tables, params)
    if bad:
        raise ScheduleError("; ".join(bad[:5]))
    seen: dict[str, list[str]] = {}
    for rnd in rounds:
        for tid in sorted(sched[rnd]):
            if tid not in active_methods(run):
                continue  # merged away after the plan was made
            reg = run.registry()
            authors = reg[tid]["authors"]
            host = authors[(rnd - 1) % len(authors)]
            vis = sched[rnd][tid]["core"] + sched[rnd][tid]["newcomers"]
            for v in vis:
                run_visit(run, ledger, rnd, tid, v, host, seen)
                seen.setdefault(v, []).append(tid)
            revise_table(run, ledger, rnd, tid, host)
        stat = ledger.uptake()
        h = run.llm.call(task="harvest", schema=Text, persona=run.persona("FA"),
                         context={"round": rnd, "ledger": ledger.items, "uptake": stat, "protocols": {m: _protocol_brief(run, m) for m in active_methods(run)}},
                         check=run.cap_words(600)).obj.text
        ds = run.llm.call(task="fixation_note", schema=Text, persona=run.persona("DS"), context={"harvest": h}).obj.text
        run.write_text(run.p("cafe", f"harvest_R{rnd}.md"), f"{h}\n\n## Note on fixation in the lab (DS)\n{ds}\n")
        if rnd == cfg.allow_merge_after_round and rnd < cfg.rounds:
            if maybe_merge(run, rnd, h):
                tables = _tables(run)
                hist = {k: set(v) for k, v in seen.items()}
                reg = run.registry()
                for m, v in reg.items():  # a merged table inherits its parents' visits
                    if "merged_from" in v:
                        for person, tabs in hist.items():
                            if set(v["merged_from"]) & tabs:
                                tabs.add(m)
                rest = list(range(rnd + 1, cfg.rounds + 1))
                sched.update(plan(tables, params, rest, run.cfg.seed + 7, hist, cfg.schedule_restarts))
    run.write_json(run.p("cafe", "schedule.json"), {str(k): v for k, v in sched.items()})
    up = ledger.uptake()
    run.write_json(run.p("cafe", "uptake.json"), up)
    if ledger.unanswered():
        raise RuntimeError(f"ledger items without a host response: {ledger.unanswered()[:5]}")
    texts: dict[str, list[str]] = {}
    for i in ledger.items:
        texts.setdefault(i["giver"], []).append(i["text"])
    run.write_json(run.p("cafe", "drift.json"), drift_report(texts))
    return True
