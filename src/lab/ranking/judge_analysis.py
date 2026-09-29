"""Divergence analysis (spec 11.5): how each member's judgements sit under the pooled Plackett-Luce fit."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def pl_loglik(ranking: list[str], worth: dict[str, float]) -> float:
    ll = 0.0
    for i in range(len(ranking) - 1):
        rest = ranking[i:]
        ll += math.log(worth[ranking[i]] / sum(worth[u] for u in rest))
    return ll


def divergence(judgements: list[dict[str, Any]], judges_csv: Path, worths: pd.DataFrame) -> dict[str, Any]:
    who: dict[tuple[int, str], str] = {}
    with judges_csv.open() as f:
        for r in csv.DictReader(f):
            who[(int(r["block"]), r["criterion"])] = r["member"]
    w = {(r.criterion, r.unit_id): float(r.worth) for r in worths.itertuples()}
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for j in judgements:
        m = who.get((int(j["block"]), j["criterion"]))
        if m is None:
            continue
        worth = {u: w[(j["criterion"], u)] for u in j["ranking"]}
        per[(m, j["criterion"])].append(pl_loglik(j["ranking"], worth))
    table = {f"{m}|{c}": round(sum(v) / len(v), 4) for (m, c), v in sorted(per.items())}
    by_member: dict[str, list[float]] = defaultdict(list)
    for (m, _), v in per.items():
        by_member[m].append(sum(v) / len(v))
    means = {m: sum(v) / len(v) for m, v in by_member.items()}
    vals = list(means.values())
    mu = sum(vals) / len(vals) if vals else 0.0
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / max(len(vals) - 1, 1)) if len(vals) > 1 else 0.0
    flagged = sorted((m for m, x in means.items() if sd > 0 and (x - mu) / sd < -1.0), key=lambda m: means[m])
    return {"mean_loglik_by_member_criterion": table,
            "mean_loglik_by_member": {m: round(x, 4) for m, x in sorted(means.items())},
            "diverging_members": flagged,
            "note": "Lower log-likelihood = reads the options differently from the pooled fit. Describes disciplines; does not rank members. "
                    "Members flagged when more than 1 SD below the member mean."}
