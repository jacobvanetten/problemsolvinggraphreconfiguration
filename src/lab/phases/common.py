"""Helpers shared by phases."""

from __future__ import annotations

import json
from pathlib import Path


from lab.run import Run
from lab.schemas import AAR, FrameSet

DATA = Path(__file__).resolve().parent.parent / "data"


def load_criteria(run: Run, which: str) -> list[dict[str, str]]:
    return json.loads((DATA / f"criteria_{which}.json").read_text())


def pairs_of(run: Run) -> list[tuple[str, str]]:
    f = run.p("methods", "pairs.json")
    if f.exists():
        return [tuple(x) for x in run.read_json(f)]  # type: ignore[misc]
    raise RuntimeError("pairs.json missing; run phase 2")


def active_methods(run: Run) -> list[str]:
    reg = run.registry()
    return sorted(m for m, v in reg.items() if v.get("active", True))


def trial_dir(run: Run, trial: str, mid: str, sub: str | None = None) -> Path:
    d = run.p("trials", trial, mid)
    return d / sub if sub else d


def load_aar(path: Path) -> AAR | None:
    return AAR.model_validate_json(path.read_text()) if path.exists() else None


def best_frame(path: Path) -> tuple[str, str]:
    fs = FrameSet.model_validate_json(path.read_text())
    f = fs.frames[fs.best]
    return f.statement, f.sees


def persona_ids_terms(run: Run) -> list[str]:
    out: list[str] = []
    for p in run.people.values():
        out.append(p.title)
        out.append("=" + p.id)
    return out


def free_pair_for(authors: set[str], pairs: list[tuple[str, str]], prefer: int) -> tuple[str, str]:
    """A core pair sharing no member with the authors, preferring the pair at index `prefer`."""
    n = len(pairs)
    for k in range(n):
        pr = pairs[(prefer + k) % n]
        if not (set(pr) & authors):
            return pr
    raise RuntimeError("no executor pair is free of the authors")


def all_hypotheses(run: Run) -> list[dict[str, str]]:
    """Every claim labelled hypothesis in any trial summary (feeds field_questions)."""
    out: list[dict[str, str]] = []
    for f in sorted(run.p("trials").rglob("summary.json")) + sorted(run.p("heldout").rglob("summary.json")):
        d = json.loads(f.read_text())
        for sid, r in d.get("step_outputs", {}).items():
            for h in r.get("hypotheses", []):
                out.append({"trial": str(f.parent.relative_to(run.dir)), "step": sid, **h})
    return out
