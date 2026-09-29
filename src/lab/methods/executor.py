"""Runs a protocol step by step (spec 8.2). Executors see only the protocol, the case files,
the network and the tools. Never designers' notes, lens notes or café material."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lab.llm import LabError
from lab.run import Run
from lab.schemas import FrameSet, MethodProtocol, Persona, Step, StepResult, Text
from lab.toolkit import Toolbox

RX_ROLES = [("breeder", "crop breeder"), ("agronomist", "agronomist"), ("socio-economist", "socio-economist"),
            ("gender specialist", "gender specialist"), ("field technician", "field technician")]


@dataclass
class Team:
    label: str
    kind: str  # cross | reference
    members: list[Persona]


def cross_team(run: Run, pair: tuple[str, str]) -> Team:
    return Team(f"{pair[0]}+{pair[1]}", "cross", [run.persona(pair[0]), run.persona(pair[1])])


def reference_team(run: Run) -> Team:
    rx = run.persona("RX")
    members = [Persona(id=f"RX-{i}", group="role", title=f"Reference team {title}", lens=f"{title}. Member of a plain national breeding team; no special theory.", voice="Plain, practical")
               for i, (_, title) in enumerate(RX_ROLES)]
    return Team("RX", "reference", members)


def _tokens(s: str) -> set[str]:
    return {t[:5] for t in re.findall(r"[a-z]{4,}", s.lower())}


def map_roles(roles: list[str], team: Team) -> dict[str, Persona]:
    """Best keyword fit between a step role and member title, lens and practice. Ties by order."""
    out: dict[str, Persona] = {}
    load = {m.id: 0 for m in team.members}
    for r in roles:
        rt = _tokens(r)
        scored = sorted(team.members, key=lambda m: (-len(rt & _tokens(f"{m.title} {m.lens} {m.practice}")),
                                                     load[m.id], m.id))
        out[r] = scored[0]
        load[scored[0].id] += 1
    return out


@dataclass
class TrialResult:
    team: str
    steps: dict[str, StepResult] = field(default_factory=dict)
    frames: FrameSet | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    interpretations: list[dict[str, str]] = field(default_factory=list)
    calls_used: int = 0
    cut_off: bool = False
    stuck: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, str]] = field(default_factory=list)


def _refs(step: Step) -> set[str]:
    return set(re.findall(r"\bS\d+\b", " ".join(step.inputs) + " " + step.action))


def run_protocol(run: Run, protocol: MethodProtocol, team: Team, case: str, cap: int,
                 toolbox: Toolbox | None = None) -> TrialResult:
    files = run.case_files(case)
    tb = toolbox or run.toolbox(case)
    res = TrialResult(team=team.label)
    proto_md = protocol.render_md()

    def tline(kind: str, **kw: Any) -> None:
        res.trace.append({"line": len(res.trace) + 1, "kind": kind, **kw})

    for step in protocol.steps:
        if res.calls_used >= cap - 1:  # keep one call for the frames
            res.cut_off = True
            tline("cut_off", step=step.id, reason=f"trial cap {cap} reached")
            continue
        who = map_roles(step.roles, team)
        lead = next(iter(who.values()))
        prior = {sid: r.output for sid, r in res.steps.items() if sid in _refs(step)}
        ctx = {"protocol": proto_md, "case_files": files, "step": step.model_dump(),
               "network_hint": "Use the tools to read the network. Only the actor list is pasted here.",
               "network_actors": [{"id": n.id, "label": n.label, "kind": n.kind} for n in tb.net.nodes],
               "outputs_of_earlier_steps_you_may_use": prior, "role_assignment": {r: p.id for r, p in who.items()}}
        ok = False
        for attempt in (1, 2):
            if res.calls_used >= cap - 1:
                res.cut_off = True
                break
            res.calls_used += 1
            try:
                cr = run.llm.call(task="exec_step", schema=StepResult, context=ctx, persona=lead, model="executors",
                                  toolbox=tb, tool_names=step.tools, max_tool_calls=min(step.max_calls, run.cfg.tools.max_calls_per_turn))
            except LabError as exc:
                tline("step_failed", step=step.id, attempt=attempt, error=str(exc)[:300])
                continue
            for t in cr.tool_trace:
                tline("tool", step=step.id, **{k: v for k, v in t.items() if k != "result"},
                      result_digest=str(t.get("result"))[:200])
            sr = cr.obj
            tline("step", step=step.id, attempt=attempt, executor=lead.id, stuck=sr.stuck, output=sr.output[:400],
                  interpretation=sr.interpretation)
            if sr.stuck:
                continue
            res.steps[step.id] = sr
            res.hypotheses += [dict(step=step.id, **h.model_dump()) for h in sr.hypotheses]
            if sr.interpretation:
                res.interpretations.append({"step": step.id, "note": sr.interpretation})
            ok = True
            break
        if not ok:
            res.stuck.append(step.id)
            res.steps[step.id] = StepResult(output="STUCK: no valid output after two attempts", stuck=True)
            tline("stuck", step=step.id)
    # frames
    lead = team.members[0]
    fctx = {"protocol": proto_md, "case_files": {"paradox.md": files["paradox.md"]},
            "step_outputs": {sid: r.output for sid, r in res.steps.items()}, "frames_wanted": run.cfg.trials.frames_per_trial}
    res.calls_used += 1
    cr = run.llm.call(task="frames", schema=FrameSet, context=fctx, persona=lead, model="executors")
    res.frames = cr.obj
    tline("frames", n=len(cr.obj.frames), best=cr.obj.best)
    return res
