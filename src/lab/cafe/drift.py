"""Persona drift check (spec 14): report only. Bag-of-words vectors, no external embedding service."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

DIM = 512


def embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def drift_report(texts_by_person: dict[str, list[str]]) -> dict:
    vecs = {p: [embed(t) for t in ts] for p, ts in texts_by_person.items() if ts}
    cent = {p: [sum(col) / len(vs) for col in zip(*vs)] for p, vs in vecs.items()}
    out = {}
    for p, vs in vecs.items():
        closer = 0
        for v in vs:
            own = cos(v, cent[p])
            other = max((cos(v, c) for q, c in cent.items() if q != p), default=-1.0)
            closer += other > own
        out[p] = {"texts": len(vs), "closer_to_other_centroid": closer, "share": round(closer / len(vs), 3)}
    flagged = sorted(p for p, r in out.items() if r["share"] > 0.5)
    return {"per_person": out, "flagged": flagged, "note": "Report only. Flagged = more than half of a person's texts sit nearer another person's centroid."}
