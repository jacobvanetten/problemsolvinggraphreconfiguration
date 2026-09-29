"""Phase 7: differences between the strongest methods (spec 9)."""

from __future__ import annotations

import json
import math
from itertools import combinations

import pandas as pd
from pydantic import BaseModel

from lab.methods.store import current
from lab.phases.common import trial_dir
from lab.run import Run
from lab.schemas import DifferenceMap, words

ROWS = ["core_representation", "entry_point", "how_themes_are_found", "how_perspectives_are_reached", "how_analogues_are_found",
        "operators_used_or_invented", "local_and_global_handling", "use_of_the_paradox", "role_of_humans", "cost_in_calls",
        "failures_seen_in_trials", "strength_and_weakness_themes"]


class DifferenceColumn(BaseModel):
    rows: dict[str, str]


class Statement(BaseModel):
    text: str
    weakness_of: str
    special_check: str | None = None


def combined(worths: pd.DataFrame) -> pd.DataFrame:
    g = worths.groupby("unit_id")
    df = pd.DataFrame({"combined_log_worth": g["log_worth"].mean(),
                       "combined_qse": g["quasi_se"].apply(lambda s: math.sqrt(float((s ** 2).sum())) / len(s))})
    return df.sort_values("combined_log_worth", ascending=False)


def non_dominated(worths: pd.DataFrame) -> list[str]:
    piv = worths.pivot(index="unit_id", columns="criterion", values="worth")
    out = []
    for u in piv.index:
        dom = any((piv.loc[v] >= piv.loc[u]).all() and (piv.loc[v] > piv.loc[u]).any() for v in piv.index if v != u)
        if not dom:
            out.append(u)
    return sorted(out)


def select(run: Run, worths: pd.DataFrame) -> dict:
    ov = run.p("ranking", "methods", "selected_override.json")
    comb = combined(worths)
    order = list(comb.index)
    nd = non_dominated(worths)
    if ov.exists():
        sel = json.loads(ov.read_text())["selected"]
        note = "selected by humans at H4"
    else:
        sel = order[:3] + [m for m in order if m in nd and m not in order[:3]]
        sel = sel[:run.cfg.phase7.max_selected]
        note = "top 3 by combined worth, plus non-dominated methods, capped"
    pairs = {}
    for a, b in combinations(sel, 2):
        d = comb.loc[a, "combined_log_worth"] - comb.loc[b, "combined_log_worth"]
        se = math.sqrt(comb.loc[a, "combined_qse"] ** 2 + comb.loc[b, "combined_qse"] ** 2)
        pairs[f"{a}|{b}"] = {"difference": round(float(d), 4), "combined_qse": round(float(se), 4),
                             "clear": bool(abs(d) >= 2 * se)}
    return {"selected": sel, "rule": note, "non_dominated": nd, "combined": json.loads(comb.round(4).to_json(orient="index")),
            "pairwise_clarity": pairs,
            "caution": "Differences under about two combined quasi-SEs are not clear."}


def run_phase(run: Run) -> bool:
    worths = pd.read_csv(run.p("ranking", "methods", "worths.csv"))
    sel = select(run, worths)
    mids = sel["selected"]
    run.write_json(run.p("discussion", "selection.json"), sel)
    summaries = json.loads(run.p("ranking", "methods", "option_summaries.json").read_text())
    div = json.loads(run.p("ranking", "methods", "divergence.json").read_text())
    fa = run.persona("FA")
    cols = {}
    for m in mids:
        aars = {str(d.relative_to(run.dir)): (d / "aar.md").read_text()
                for d in [trial_dir(run, "T1", m), *[p for p in trial_dir(run, "T2", m).iterdir() if p.is_dir()]] if (d / "aar.md").exists()}
        cols[m] = run.llm.call(
            task="difference_column", schema=DifferenceColumn, persona=fa,
            context={"method": m, "rows": ROWS, "protocol": current(run, m).model_dump(), "aars": aars,
                     "tricot_summaries": summaries.get(m, {})},
            check=lambda c: _rows_ok(c)).obj.rows
    matrix = "| row | " + " | ".join(mids) + " |\n|---|" + "---|" * len(mids) + "\n" + "\n".join(
        f"| {r} | " + " | ".join(cols[m][r].replace("|", "/") for m in mids) + " |" for r in ROWS)
    run.write_text(run.p("discussion", "difference_matrix.md"), matrix + "\n")
    first = [x for x in div["diverging_members"] if x in run.core_ids]
    order = first + [x for x in run.core_ids if x not in first]
    stmts = {}
    for pid in order:
        def chk(s: Statement, pid=pid) -> None:
            if words(s.text) > 300:
                raise ValueError("statement over 300 words")
            if s.weakness_of not in mids:
                raise ValueError(f"weakness_of must name a selected method: {mids}")
            if pid in ("PL", "SY") and not s.special_check:
                raise ValueError("PL checks logical incompatibility and SY checks system-level conflicts: fill special_check")
        stmts[pid] = run.llm.call(task="statement", schema=Statement, persona=run.persona(pid),
                                  context={"selection": sel, "matrix": cols, "diverged_first": first,
                                           "special": {"PL": "check for logical incompatibility between method steps",
                                                       "SY": "check for conflicts at system level"}.get(pid)},
                                  check=chk).obj
    run.write_text(run.p("discussion", "statements.md"), "\n\n".join(
        f"## {p} (weakness named in {s.weakness_of})\n{s.text}" + (f"\n\nSpecial check: {s.special_check}" if s.special_check else "")
        for p, s in stmts.items()) + "\n")

    def mcheck(d: DifferenceMap) -> None:
        for p in d.pairs:
            if not p.sources:
                raise ValueError("every pair needs sources")
            if p.a.split(":")[0] not in mids or p.b.split(":")[0] not in mids:
                raise ValueError("components are named '<method id>:<component>' and must come from the selected methods")
    dm = run.llm.call(task="difference_map", schema=DifferenceMap, persona=fa,
                      context={"selected": mids, "matrix": cols, "statements": {p: s.model_dump() for p, s in stmts.items()}},
                      check=mcheck).obj
    run.write_json(run.p("discussion", "difference_map.json"), dm)
    run.write_text(run.p("discussion", "difference_map.md"), "\n".join(
        f"- [{p.tag}] {p.a} vs {p.b}: {p.reason} (sources: {', '.join(p.sources)})" for p in dm.pairs) + "\n")
    for name in ("frames", "methods"):  # raw judgements are deleted only now (spec 11.1)
        from lab.ranking.tricot_bridge import Study
        st = Study(run, name, run.cfg.ranking.frames if name == "frames" else run.cfg.ranking.methods, [], "")
        if (st.dir / "judgements.jsonl").exists():
            st.purge()
    return True


def _rows_ok(c: DifferenceColumn) -> None:
    missing = [r for r in ROWS if not c.rows.get(r, "").strip()]
    if missing:
        raise ValueError(f"missing rows: {missing}")
