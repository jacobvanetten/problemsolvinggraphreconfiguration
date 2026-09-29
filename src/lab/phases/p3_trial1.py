"""Phase 3: Trial 1 and after-action reviews (spec 9)."""

from __future__ import annotations

import yaml

from lab.methods.executor import cross_team
from lab.methods.store import add_version, check_protocol, current
from lab.phases.common import free_pair_for, load_aar, pairs_of, trial_dir
from lab.run import Run
from lab.schemas import MethodProtocol, Revision
from lab.trials.runner import run_trial


def register_human_methods(run: Run) -> None:
    hdir = run.p("methods", "human")
    if not hdir.exists():
        return
    reg = run.registry()
    pairs = pairs_of(run)
    for f in sorted(hdir.glob("*.yaml")):
        mid = f.stem.upper()
        if mid in reg:
            continue
        proto = MethodProtocol.model_validate(yaml.safe_load(f.read_text()))
        proto.validate_limits(run.cfg.method.max_words, run.cfg.trials.max_calls)
        add_version(run, mid, proto, "proposal", source="human")
        ex = free_pair_for(set(), pairs, len(reg))
        reg[mid] = {"pair_index": None, "authors": ["human"], "executors": list(ex), "source": "human", "active": True}
    run.save_registry(reg)


def intent_text(run: Run, mid: str) -> str:
    f = run.p("methods", f"{mid}.intent.json")
    if not f.exists():
        return "Human-written method; intent is the protocol's purpose."
    d = run.read_json(f)
    return f"Draft A: {d['draft_a']}\nDraft B: {d['draft_b']}"


def run_phase(run: Run) -> bool:
    register_human_methods(run)
    reg = run.registry()
    for mid in sorted(reg):
        if not reg[mid].get("active", True):
            continue
        out = trial_dir(run, "T1", mid)
        proto = current(run, mid)
        if load_aar(out / "aar.json") is None:
            team = cross_team(run, tuple(reg[mid]["executors"]))  # type: ignore[arg-type]
            run_trial(run, proto, team, "A", out, intent_text(run, mid))
        if not any(v.stage == "trial1_fix" for v in __import__("lab.methods.store", fromlist=["x"]).load_history(run, mid)):
            aar = load_aar(out / "aar.json")
            frames_md = (out / "frames.json").read_text()
            lead = "human" if reg[mid]["authors"] == ["human"] else reg[mid]["authors"][0]
            persona = run.persona(lead) if lead != "human" else run.persona("FA")
            rev = run.llm.call(task="revise_stuck", schema=Revision, persona=persona,
                               context={"protocol": proto.model_dump(), "trial1_frames": frames_md,
                                        "aar": aar.model_dump(), "limits": {"max_words": run.cfg.method.max_words,
                                                                            "trial_max_calls": run.cfg.trials.max_calls}},
                               check=check_protocol(run)).obj
            add_version(run, mid, rev.protocol, "trial1_fix", rev.changelog)
    return run.checkpoint("H3", "Read the AARs in trials/T1, add observations, stop a method that clearly fails (set active:false in methods/registry.json)")
