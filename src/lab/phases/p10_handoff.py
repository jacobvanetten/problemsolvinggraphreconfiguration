"""Phase 10: final reflection and handoff (spec 9)."""

from __future__ import annotations

import json
from collections import Counter

from pydantic import BaseModel

from lab.phases.common import all_hypotheses
from lab.phases.p8_integration import composite_check, render
from lab.run import Run
from lab.schemas import Composite, DifferenceMap, Text


class FinalReflection(BaseModel):
    generalises: str
    depends_on_case_A: str
    lab_process: str
    toolkit: str
    next_run: str


def toolkit_usage(run: Run) -> dict[str, int]:
    c: Counter[str] = Counter()
    for f in list(run.p("trials").rglob("trace.jsonl")) + list(run.p("heldout").rglob("trace.jsonl")):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "tool":
                c[r["tool"]] += 1
    return dict(c)


def run_phase(run: Run) -> bool:
    usage = toolkit_usage(run)
    comp = Composite.model_validate_json(run.p("integration", "composite.json").read_text())
    cmp_ = run.read_json(run.p("heldout", "comparison.json"))
    ledger_up = run.read_json(run.p("cafe", "uptake.json"))
    notes = {}
    for pid in [*run.core_ids, "PE"]:
        notes[pid] = run.llm.call(task="reflection", schema=Text, persona=run.persona(pid),
                                  context={"heldout": cmp_, "toolkit_usage": usage, "uptake": ledger_up},
                                  check=run.cap_words(300)).obj.text
    integ = run.persona("IN")
    fr = run.llm.call(task="final_reflection", schema=FinalReflection, persona=integ, model="integrator",
                      context={"notes": notes, "heldout": cmp_, "toolkit_usage": usage, "uptake": ledger_up,
                               "drift": run.read_json(run.p("cafe", "drift.json"))}).obj
    run.write_text(run.p("reflection", "final_reflection.md"), "\n\n".join(
        f"## {h}\n{getattr(fr, k)}" for k, h in [("generalises", "What generalises beyond pearl millet"),
                                                 ("depends_on_case_A", "What depended on case A"),
                                                 ("lab_process", "The lab as a process, including the café and newcomers"),
                                                 ("toolkit", "Which toolkit parts earned their place"),
                                                 ("next_run", "What should change in the next run")]) +
        "\n\n## Toolkit calls seen in traces\n" + ("\n".join(f"- {k}: {v}" for k, v in sorted(usage.items())) or "- none") + "\n\n" +
        "## Individual notes\n" + "\n\n".join(f"### {p}\n{t}" for p, t in notes.items()) + "\n")
    dm = DifferenceMap.model_validate_json(run.p("discussion", "difference_map.json").read_text())
    sel = json.loads(run.p("discussion", "selection.json").read_text())["selected"]
    rev = run.llm.call(task="revise_composite", schema=Composite, persona=integ, model="integrator",
                       context={"composite": comp.model_dump(), "heldout": cmp_, "final_reflection": fr.model_dump(),
                                "limits": {"max_words": run.cfg.method.max_words, "trial_max_calls": run.cfg.trials.max_calls}},
                       check=composite_check(run, sel, dm)).obj
    dissent = run.read_json(run.p("integration", "objections.json"))["dissent"]
    run.write_text(run.p("handoff", "method_v1.md"), render(rev, dissent))
    hyps = all_hypotheses(run)
    impl = run.llm.call(task="implementation_notes", schema=Text, persona=run.persona("FA"),
                        context={"method": rev.model_dump(), "toolkit_usage": usage,
                                 "invented_operators_seen": [t for t in usage if t == "apply_edits"]}).obj.text
    run.write_text(run.p("handoff", "implementation_notes.md"), impl + "\n")
    fq = run.llm.call(task="field_questions", schema=Text, persona=run.persona("FA"), context={"hypotheses": hyps}).obj.text
    run.write_text(run.p("handoff", "field_questions.md"),
                   fq + "\n\n## All hypotheses about human actors recorded in trials (unverified, simulated)\n" +
                   ("\n".join(f"- [{h['trial']} {h['step']}] {h['text']}" for h in hyps) or "- none recorded") + "\n")
    return run.checkpoint("H6", "Accept the handoff in handoff/")
