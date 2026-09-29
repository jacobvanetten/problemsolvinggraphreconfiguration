"""One trial = one method, one case, one executor team (spec 8.2)."""

from __future__ import annotations

from pathlib import Path

from lab.methods.executor import Team, TrialResult, run_protocol
from lab.run import Run
from lab.schemas import AAR, MethodProtocol, Text, words
from lab.trials.aar import run_aar


def run_trial(run: Run, protocol: MethodProtocol, team: Team, case: str, out_dir: Path, intent: str,
              prior_aar: AAR | None = None, changelog: list | None = None) -> tuple[TrialResult, AAR]:
    res = run_protocol(run, protocol, team, case, run.cfg.trials.max_calls)
    lead = team.members[0]
    notes = run.llm.call(task="executor_notes", schema=Text, persona=lead, model="executors",
                         context={"protocol": protocol.render_md(), "trace_summary": [t for t in res.trace if t["kind"] != "tool"][:60],
                                  "interpretations": res.interpretations}, check=run.cap_words(300)).obj.text
    ethno = run.llm.call(task="ethnography", schema=Text, persona=run.persona("PE"),
                         context={"trace": res.trace, "protocol_step_ids": [s.id for s in protocol.steps]},
                         check=run.cap_words(400)).obj.text
    aar = run_aar(run, protocol, res, intent, notes, ethno, prior_aar, changelog)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.jsonl").write_text("".join(__import__("json").dumps(t, default=str) + "\n" for t in res.trace))
    run.write_json(out_dir / "frames.json", res.frames)
    run.write_json(out_dir / "interpretations.json", res.interpretations)
    run.write_json(out_dir / "summary.json", {"team": res.team, "calls_used": res.calls_used, "cut_off": res.cut_off,
                                              "stuck": res.stuck,
                                              "step_outputs": {k: v.model_dump() for k, v in res.steps.items()}})
    run.write_text(out_dir / "executor_notes.md", notes + "\n")
    run.write_text(out_dir / "ethnography.md", ethno + "\n")
    run.write_text(out_dir / "aar.md", aar.render_md())
    run.write_json(out_dir / "aar.json", aar)
    return res, aar
