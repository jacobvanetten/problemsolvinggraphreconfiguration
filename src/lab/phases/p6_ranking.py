"""Phase 6: two tricot studies, frames first (blind), then methods (spec 11)."""

from __future__ import annotations

import json
from typing import Any

from lab.phases.common import (active_methods, best_frame, load_criteria, persona_ids_terms, trial_dir)
from lab.ranking.dossier import build_digest, frame_worth, frame_worth_text
from lab.ranking.studies import run_study
from lab.methods.store import current
from lab.run import Run


def forbidden_for(run: Run, mids: list[str]) -> dict[str, set[str]]:
    reg = run.registry()
    return {m: {a for a in reg[m]["authors"] if a in run.people} | set(reg[m]["executors"]) for m in mids}


def frame_units(run: Run, trial: str = "T2") -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    units: dict[str, tuple[str, str]] = {}
    umap: dict[str, str] = {}
    i = 0
    for mid in active_methods(run):
        for sub in sorted(p.name for p in trial_dir(run, trial, mid).iterdir() if p.is_dir()):
            f = trial_dir(run, trial, mid, sub) / "frames.json"
            if not f.exists():
                continue
            i += 1
            uid = f"F{i:02d}"
            stmt, sees = best_frame(f)
            units[uid] = (uid, f"{stmt}\n\nWhat this frame lets the team see: {sees}")
            umap[uid] = mid
    return units, umap


def run_phase(run: Run) -> bool:
    mids = active_methods(run)
    reg = run.registry()
    paradox = run.case_files("A")["paradox.md"]
    # Study 1: frames, blind
    units, umap = frame_units(run)
    run.write_json(run.p("ranking", "frames", "unit_map.json"), umap)
    terms = {"placeholders": {"own": "[this method]", "other": "[another method]", "global": "[a team member]"},
             "units": {u: [current(run, m).name, "=" + m] for u, m in umap.items()},
             "others": [], "global": persona_ids_terms(run)}
    fs = run.cfg.ranking.frames
    st1 = run_study(run, "frames", units, terms, fs, load_criteria(run, "frames"),
                    "a member of a multidisciplinary breeding design team judging frames for case A",
                    {u: forbidden_for(run, [m])[m] for u, m in umap.items()},
                    extra_ctx=lambda c: {"paradox.md": paradox} if c == "shift" else {})
    fw = frame_worth(st1.worths(), umap)
    run.write_json(run.p("ranking", "frames", "frame_worth_by_method.json"), fw)
    # Study 2: methods
    munits: dict[str, tuple[str, str]] = {}
    mterms_units: dict[str, list[str]] = {}
    for mid in mids:
        proto = current(run, mid)
        material: dict[str, Any] = {"aars": {}, "executor_notes": {}, "ethnography": {}, "calls": {}}
        for trial in ("T1", "T2"):
            base = trial_dir(run, trial, mid)
            dirs = [base] if trial == "T1" else [base / k for k in run.cfg.trials.trial2_executors]
            for d in dirs:
                if (d / "aar.md").exists():
                    key = f"{trial}/{d.name}"
                    material["aars"][key] = (d / "aar.md").read_text()
                    material["executor_notes"][key] = (d / "executor_notes.md").read_text()
                    material["ethnography"][key] = (d / "ethnography.md").read_text()
                    material["calls"][key] = json.loads((d / "summary.json").read_text())["calls_used"]
        digest = build_digest(run, mid, proto, material)
        run.write_text(run.p("ranking", "methods", "dossiers", f"{mid}.digest.md"), digest + "\n")
        fwt = frame_worth_text(fw.get(mid, {})) if mid in fw else "No frame worth available."
        text = f"{proto.render_md()}\n\n# Trial digest\n{digest}\n\n# {fwt}\n"
        munits[mid] = (mid, text)
        mterms_units[mid] = [proto.name, "=" + mid]
    mterms = {"placeholders": {"own": "[this method]", "other": "[another method]", "global": "[a team member]"},
              "units": mterms_units, "others": [], "global": persona_ids_terms(run)}
    st2 = run_study(run, "methods", munits, mterms, run.cfg.ranking.methods, load_criteria(run, "methods"),
                    "a member of a multidisciplinary team designing a reframing method for breeding teams. Judge from your own discipline.",
                    forbidden_for(run, mids))
    return run.checkpoint("H4", "Read ranking/frames and ranking/methods; change criteria and rerun, or choose methods via ranking/methods/selected_override.json")
