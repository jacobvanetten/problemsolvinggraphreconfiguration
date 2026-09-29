"""Phase 8: modular composite, objection round (spec 9)."""

from __future__ import annotations

import json

from pydantic import BaseModel

from lab.methods.store import check_protocol, current, load_history
from lab.phases.common import trial_dir
from lab.run import Run
from lab.schemas import Composite, DifferenceMap, Objection, words


class Reply(BaseModel):
    member: str
    reply: str
    resolved: bool


class IntegratorAnswer(BaseModel):
    replies: list[Reply]
    revised: Composite | None = None


def composite_check(run: Run, selected: list[str], dm: DifferenceMap):
    pcheck = check_protocol(run)

    def check(c: Composite) -> None:
        pcheck(c)
        for m in c.modules:
            bad = [s for s in m.source_methods if s not in selected]
            if bad:
                raise ValueError(f"module {m.name}: unknown source methods {bad}")
            if not m.inputs or not m.outputs:
                raise ValueError(f"module {m.name} needs inputs and outputs")
        texts = c.open_choices + c.resolutions
        for p in dm.pairs:
            if p.tag == "exclusive" and not any(p.a in t and p.b in t for t in texts):
                raise ValueError(f"exclusive pair {p.a} vs {p.b} is neither resolved nor named as an open choice")
    return check


def render(c: Composite, dissent: list[dict]) -> str:
    L = [c.protocol.render_md(), "", "# Modules"]
    for m in c.modules:
        L += [f"## {m.name}", f"- steps: {', '.join(m.step_ids) or 'n/a'}", f"- inputs: {', '.join(m.inputs)}",
              f"- outputs: {', '.join(m.outputs)}", f"- lineage (methods): {', '.join(m.source_methods)}",
              f"- lineage (café ledger): {', '.join(m.ledger_items) or 'none'}",
              f"- lineage (trial evidence): {', '.join(m.trial_evidence) or 'none'}", ""]
    L += ["# Failed modules"] + [f"- {f.claim} [{f.cite}]" for f in c.failed_modules] or ["- none"]
    L += ["", "# Resolved choices"] + [f"- {x}" for x in c.resolutions]
    L += ["", "# Open choices for the human team"] + ([f"- {x}" for x in c.open_choices] or ["- none"])
    L += ["", "# Dissent log"] + ([f"- {d['member']}: {d['objection']} (integrator: {d['reply']})" for d in dissent] or ["- none"])
    return "\n".join(L) + "\n"


def _aars(run: Run, sel: list[str]) -> dict[str, str]:
    root = run.p("trials")
    return {str(p.relative_to(run.dir)): p.read_text() for p in sorted(root.rglob("aar.md"))
            if p.relative_to(root).parts[1] in sel}


def run_phase(run: Run) -> bool:
    sel = json.loads(run.p("discussion", "selection.json").read_text())["selected"]
    dm = DifferenceMap.model_validate_json(run.p("discussion", "difference_map.json").read_text())
    ledger = json.loads(run.p("cafe", "ledger.json").read_text())
    ctx = {"selected_methods": {m: current(run, m).model_dump() for m in sel},
           "histories": {m: [{"stage": v.stage, "changelog": v.changelog} for v in load_history(run, m)] for m in sel},
           "aars": _aars(run, sel),
           "ledger": [i for i in ledger if i["table"] in sel], "difference_map": dm.model_dump(),
           "rankings": {n: json.loads(run.p("ranking", n, "option_summaries.json").read_text()) for n in ("frames", "methods")},
           "toolkit": ["get_metrics", "get_perspective", "list_motifs", "apply_operators", "apply_edits", "find_analogues", "get_evidence"],
           "limits": {"max_words": run.cfg.method.max_words, "trial_max_calls": run.cfg.trials.max_calls}}
    integ = run.persona("IN")
    comp = run.llm.call(task="composite", schema=Composite, persona=integ, model="integrator", context=ctx,
                        check=composite_check(run, sel, dm)).obj
    objections = {}
    for pid in run.core_ids:
        objections[pid] = run.llm.call(task="objection", schema=Objection, persona=run.persona(pid),
                                       context={"composite": comp.model_dump()},
                                       check=lambda o: (_ for _ in ()).throw(ValueError("over 150 words")) if words(o.text) > 150 else None).obj.text
    ans = run.llm.call(task="integrator_answer", schema=IntegratorAnswer, persona=integ, model="integrator",
                       context={"composite": comp.model_dump(), "objections": objections},
                       check=lambda a: composite_check(run, sel, dm)(a.revised) if a.revised else None).obj
    if ans.revised:
        comp = ans.revised
    replies = {r.member: r for r in ans.replies}
    dissent = [{"member": p, "objection": objections[p], "reply": replies[p].reply if p in replies else "no reply"}
               for p in objections if p not in replies or not replies[p].resolved]
    run.write_json(run.p("integration", "composite.json"), comp)
    run.write_json(run.p("integration", "objections.json"), {"objections": objections, "replies": [r.model_dump() for r in ans.replies], "dissent": dissent})
    run.write_text(run.p("integration", "composite_method.md"), render(comp, dissent))
    return run.checkpoint("H5", "Accept, edit or reject integration/composite_method.md (edit composite.json with source: human) before the held-out test")
