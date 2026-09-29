"""Phase 9: held-out test on case B (spec 9, 10.2)."""

from __future__ import annotations

import json
import math


from lab.methods.executor import cross_team, reference_team
from lab.methods.store import current
from lab.phases.common import best_frame, load_aar, load_criteria, pairs_of, persona_ids_terms
from lab.phases.p0_field import build_field
from lab.phases.p6_ranking import forbidden_for
from lab.ranking.studies import run_study
from lab.run import Run
from lab.schemas import AAR, Composite, Text
from lab.trials.runner import run_trial


def leakage_check(run: Run) -> dict:
    c = run.cfg.cases
    if not (c.leakage_check and c.B_kind == "dorst_disguised"):
        return {"run": False, "reason": "case B is not a disguised Dorst case or check disabled"}
    text = "\n\n".join(run.case_files("B").values())
    ans = run.llm.call(task="leakage_probe", schema=Text, context={"case": text}).obj.text
    hit = ans.strip().lower() not in ("unknown", "")
    return {"run": True, "identified_source": hit, "answer": ans,
            "consequence": "comparison with Dorst's frame is weak evidence" if hit else "no leakage detected by this probe"}


def pick_cross_pair(run: Run, sel_methods: list[str], comp: Composite) -> tuple[str, str]:
    pairs = pairs_of(run)
    reg = run.registry()
    main = {s for m in comp.modules for s in m.source_methods}
    authoring = {tuple(reg[m]["authors"]) for m in main if m in reg}
    ok = [p for p in pairs if tuple(p) not in authoring and not (set(p) & {a for m in main for a in reg[m]["authors"]})] \
        or [p for p in pairs if tuple(p) not in authoring]
    return ok[run.rng("heldout_pair").randrange(len(ok))]


def run_phase(run: Run) -> bool:
    run.seal.unseal()
    run.write_text(run.p("state", "unsealed"), "case B unsealed for Phase 9\n")
    sel = json.loads(run.p("discussion", "selection.json").read_text())
    comb = sel["combined"]
    comp = Composite.model_validate_json(run.p("integration", "composite.json").read_text())
    build_field(run, "B")
    run.write_json(run.p("heldout", "leakage_check.json"), leakage_check(run))
    top = sorted(comb, key=lambda m: -comb[m]["combined_log_worth"])[:run.cfg.heldout.compare_top_n]
    reg = run.registry()
    runs: dict[str, tuple[str, str]] = {}  # unit id -> (label, dir key)
    todo = [("COMPOSITE", comp.protocol, "reference", reference_team(run))]
    todo.append(("COMPOSITE", comp.protocol, "cross", cross_team(run, pick_cross_pair(run, sel["selected"], comp))))
    for m in top:
        todo.append((m, current(run, m), "reference", reference_team(run)))
    aars: dict[str, AAR] = {}
    for label, proto, kind, team in todo:
        out = run.p("heldout", "T3", label, kind)
        a = load_aar(out / "aar.json")
        if a is None:
            _, a = run_trial(run, proto, team, "B", out, f"Held-out test of {label} on case B. Intent: as in the protocol's purpose.")
        aars[f"{label}/{kind}"] = a
    units, umap = {}, {}
    for i, (label, _, kind, _) in enumerate(todo, 1):
        uid = f"H{i:02d}"
        stmt, sees = best_frame(run.p("heldout", "T3", label, kind) / "frames.json")
        units[uid] = (uid, f"{stmt}\n\nWhat this frame lets the team see: {sees}")
        umap[uid] = f"{label}/{kind}"
    forb = {}
    for u, k in umap.items():
        m = k.split("/")[0]
        if m == "COMPOSITE":
            forb[u] = set()  # authored by the integrator only
        else:
            forb[u] = forbidden_for(run, [m])[m]
    terms = {"placeholders": {"own": "[this method]", "other": "[another method]", "global": "[a team member]"},
             "units": {u: [current(run, k.split("/")[0]).name if k.startswith("M") else comp.protocol.name] for u, k in umap.items()},
             "others": [], "global": persona_ids_terms(run)}
    fs = run.cfg.ranking.frames.model_copy(update={"max_reps": min(run.cfg.ranking.frames.max_reps, 6)})
    paradox = run.case_files("B")["paradox.md"]
    st = run_study(run, "heldout_frames", units, terms, fs, load_criteria(run, "frames"),
                   "a member of a multidisciplinary breeding design team judging frames for case B", forb,
                   extra_ctx=lambda c: {"paradox.md": paradox} if c == "shift" else {})
    w = st.worths()
    w["run"] = w["unit_id"].map(umap)
    run.write_text(run.p("heldout", "frame_worths.csv"), w.to_csv(index=False))
    # comparison
    lines = ["# Held-out comparison (case B)", "",
             "Frames ranked blind, by criterion. Estimates carry quasi-SEs; differences under about two combined quasi-SEs are not clear.", ""]
    verdict: dict[str, dict] = {}
    for c in sorted(w["criterion"].unique()):
        g = w[w["criterion"] == c]
        comp_rows = g[g["run"].str.startswith("COMPOSITE")]
        par_rows = g[~g["run"].str.startswith("COMPOSITE")]
        cw, cq = comp_rows["log_worth"].mean(), math.sqrt(float((comp_rows["quasi_se"] ** 2).sum())) / len(comp_rows)
        best = par_rows.sort_values("log_worth", ascending=False).iloc[0]
        d = cw - best["log_worth"]
        se = math.sqrt(cq ** 2 + best["quasi_se"] ** 2)
        state = ("composite at least as good" if d >= 0 else "composite WORSE than its best parent") + \
                (" (difference is clear)" if abs(d) >= 2 * se else " (difference is not clear)")
        verdict[c] = {"composite_mean_log_worth": round(float(cw), 3), "best_parent": best["run"],
                      "best_parent_log_worth": round(float(best["log_worth"]), 3), "difference": round(float(d), 3),
                      "combined_qse": round(float(se), 3), "reading": state}
        lines.append(f"- **{c}**: composite {round(float(cw), 3)} vs best parent {best['run']} {round(float(best['log_worth']), 3)}: {state}")
    # which modules travelled
    travelled: dict[str, str] = {}
    for m in comp.modules:
        worked = any(cl.cite in m.step_ids or any(s == cl.cite for s in m.step_ids)
                     for k, a in aars.items() if k.startswith("COMPOSITE") for cl in a.steps_worked)
        idle = any(cl.cite in m.step_ids for k, a in aars.items() if k.startswith("COMPOSITE") for cl in a.steps_no_work)
        travelled[m.name] = "travelled" if worked and not idle else ("did no work on case B" if idle else "no evidence either way")
    lines += ["", "## Modules on case B (from the AARs)"] + [f"- {k}: {v}" for k, v in travelled.items()]
    run.write_json(run.p("heldout", "comparison.json"), {"verdict": verdict, "modules": travelled, "top_parents": top})
    run.write_text(run.p("heldout", "comparison.md"), "\n".join(lines) + "\n")
    return True
