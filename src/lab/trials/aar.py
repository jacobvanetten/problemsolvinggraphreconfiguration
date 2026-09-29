"""After-action review (spec 8.3). Claims must cite the trace."""

from __future__ import annotations

import re
from typing import Any

from lab.methods.executor import TrialResult
from lab.run import Run
from lab.schemas import AAR, MethodProtocol, Text


def valid_cites(protocol: MethodProtocol, res: TrialResult) -> tuple[set[str], int]:
    steps = {s.id for s in protocol.steps}
    interps = {f"interp:{i['step']}" for i in res.interpretations}
    return steps | interps, len(res.trace)


def cite_checker(protocol: MethodProtocol, res: TrialResult):
    ids, n = valid_cites(protocol, res)

    def ok(cite: str) -> bool:
        for part in re.split(r"[,;]\s*", cite):
            part = part.strip()
            m = re.fullmatch(r"trace:(\d+)", part)
            if m:
                if not 1 <= int(m.group(1)) <= n:
                    return False
            elif part not in ids:
                return False
        return True

    def check(a: AAR) -> None:
        for sec in (a.steps_worked, a.steps_no_work, a.stuck_points, a.surprises, a.proposed_changes):
            for c in sec:
                if not ok(c.cite):
                    raise ValueError(f"citation {c.cite!r} does not match a step id, trace line or interpretation note")
    return check


def trace_summary(res: TrialResult) -> dict[str, Any]:
    return {"calls_used": res.calls_used, "cut_off": res.cut_off, "stuck": res.stuck,
            "trace_lines": len(res.trace),
            "steps": [{"line": t["line"], **{k: t.get(k) for k in ("step", "kind", "stuck", "interpretation")}}
                      for t in res.trace if t["kind"] in ("step", "stuck", "cut_off", "step_failed")],
            "tool_calls": [{"line": t["line"], "step": t.get("step"), "tool": t.get("tool")}
                           for t in res.trace if t["kind"] == "tool"]}


def run_aar(run: Run, protocol: MethodProtocol, res: TrialResult, intent: str, exec_notes: str, ethnography: str,
            prior_aar: AAR | None = None, changelog: list | None = None) -> AAR:
    fa, pe = run.persona("FA"), run.persona("PE")
    base = {"protocol": protocol.render_md(), "trace_summary": trace_summary(res),
            "frames": res.frames.model_dump() if res.frames else None,
            "interpretation_notes": res.interpretations}
    answers = {
        "designers": run.llm.call(task="aar_designers", schema=Text, persona=fa,
                                  context={**base, "designers_statement_of_intent": intent}).obj.text,
        "executors": run.llm.call(task="aar_executors", schema=Text, persona=fa,
                                  context={**base, "executor_notes": exec_notes}).obj.text,
        "ethnographer": run.llm.call(task="aar_ethnographer", schema=Text, persona=pe,
                                     context={**base, "ethnography": ethnography}).obj.text,
    }
    ctx = {**base, "answers_to_the_four_questions": answers, "valid_citation_ids": sorted(valid_cites(protocol, res)[0]),
           "valid_trace_lines": f"trace:1 to trace:{len(res.trace)}"}
    if prior_aar is not None:
        ctx["trial1_aar"] = prior_aar.model_dump()
        ctx["revision_changelog"] = changelog or []
        ctx["extra_instruction"] = "Add a comparison section: what changed from Trial 1 and whether the revisions fixed what they meant to fix."
    return run.llm.call(task="aar", schema=AAR, persona=fa, context=ctx, check=cite_checker(protocol, res)).obj
