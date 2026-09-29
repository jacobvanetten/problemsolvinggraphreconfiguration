"""Method units for Study 2 (spec 11.3) and per-method frame worth from Study 1 (spec 11.2)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from lab.run import Run
from lab.schemas import MethodProtocol, Text


from pydantic import BaseModel, Field


class DigestVerdict(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)


def frame_worth(worths: pd.DataFrame, unit_method: dict[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean log-worth of a method's frames per criterion. SE of the mean from the quasi-standard errors,
    treating the frames' estimates as independent (an approximation, stated in the dossier)."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    df = worths.copy()
    df["method"] = df["unit_id"].map(unit_method)
    for (m, c), g in df.dropna(subset=["method"]).groupby(["method", "criterion"]):
        n = len(g)
        out.setdefault(m, {})[c] = {"mean_log_worth": round(float(g["log_worth"].mean()), 4),
                                    "se": round(math.sqrt(float((g["quasi_se"] ** 2).sum())) / n, 4), "n_frames": n}
    return out


def frame_worth_text(fw: dict[str, dict[str, float]]) -> str:
    rows = [f"- {c}: mean log-worth {v['mean_log_worth']} +/- {v['se']} (quasi-SE, {v['n_frames']} frames)"
            for c, v in sorted(fw.items())]
    return "Blind frame worth from Study 1 (approximate SE):\n" + "\n".join(rows)


def build_digest(run: Run, mid: str, protocol: MethodProtocol, trial_material: dict[str, Any]) -> str:
    fa, kg = run.persona("FA"), run.persona("KG")
    ctx = {"method": mid, "protocol": protocol.render_md(), "trials": trial_material,
           "template": "Sections: worked, did no work, stuck points, revisions T1 to T2, calls, executor notes, ethnographer's observation"}
    digest = run.llm.call(task="digest", schema=Text, persona=fa, context=ctx, check=run.cap_words(600)).obj.text
    for _ in range(2):
        v = run.llm.call(task="digest_check", schema=DigestVerdict, persona=kg,
                         context={"digest": digest, "aars": trial_material.get("aars")}).obj
        if v.ok:
            return digest
        digest = run.llm.call(task="digest", schema=Text, persona=fa, context={**ctx, "issues_to_fix": v.issues},
                              check=run.cap_words(600)).obj.text
    return digest
