"""Phase 5: Trial 2, by the Trial 1 cross pair and by the reference team (spec 9)."""

from __future__ import annotations

from lab.methods.executor import cross_team, reference_team
from lab.methods.store import current, load_history
from lab.phases.common import active_methods, free_pair_for, load_aar, pairs_of, trial_dir
from lab.phases.p3_trial1 import intent_text
from lab.run import Run
from lab.trials.runner import run_trial


def run_phase(run: Run) -> bool:
    reg = run.registry()
    pairs = pairs_of(run)
    for mid in active_methods(run):
        proto = current(run, mid)
        hist = load_history(run, mid)
        changelog = [c for v in hist if v.stage != "proposal" for c in v.changelog]
        t1 = load_aar(trial_dir(run, "T1", mid) / "aar.json")
        authors = set(reg[mid]["authors"])
        ex = tuple(reg[mid]["executors"])
        if set(ex) & authors:  # merged methods: pick a pair with no author among them
            ex = free_pair_for(authors, pairs, 0)
        teams = {"cross": cross_team(run, ex), "reference": reference_team(run)}  # type: ignore[arg-type]
        for kind in run.cfg.trials.trial2_executors:
            out = trial_dir(run, "T2", mid, kind)
            if load_aar(out / "aar.json") is None:
                run_trial(run, proto, teams[kind], "A", out, intent_text(run, mid), prior_aar=t1, changelog=changelog)
    return True
