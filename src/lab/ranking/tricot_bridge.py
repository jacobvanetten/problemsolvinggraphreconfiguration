"""Bridge to the tricot-ranking skill's CLI (spec 11). The skill is used as is, never reimplemented."""

from __future__ import annotations

import csv
import glob
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from lab.config import StudyCfg
from lab.llm import LabError
from lab.run import Run
from lab.schemas import Judgement


def find_skill(run: Run) -> Path:
    p = run.cfg.ranking.skill_path
    if p:
        q = Path(p)
    else:
        hits = sorted(glob.glob(str(Path.home() / ".claude/skills/*/*/tricot-ranking"))
                      + glob.glob(str(Path.home() / ".claude/skills/tricot-ranking")))
        if not hits:
            raise LabError("tricot-ranking skill not found; set ranking.skill_path in config")
        q = Path(hits[0])
    if not (q / "scripts" / "tricot.py").exists():
        raise LabError(f"{q} has no scripts/tricot.py")
    return q


def judge_prompt_text(skill: Path) -> str:
    """The skill's judge instructions, verbatim (the blockquote under 'Judge subagent prompt')."""
    md = (skill / "references" / "claude-code.md").read_text()
    sec = md.split("## Judge subagent prompt", 1)[1].split("\n## ", 1)[0]
    lines = [ln[2:] if ln.startswith("> ") else ln.lstrip(">") for ln in sec.splitlines() if ln.startswith(">")]
    return "\n".join(lines).strip()


@dataclass
class Assigner:
    """Round-robin judge assignment per criterion. Skips members who authored or executed a method in the block."""

    members: list[str]
    forbidden_by_unit: dict[str, set[str]]
    pointer: dict[str, int] = field(default_factory=dict)
    conflicts: int = 0

    def pick(self, criterion: str, units: list[str]) -> tuple[str, bool]:
        """Next member in rotation who authored or executed nothing in the block. When every member has a
        conflict (the design makes it impossible), take the member with the fewest conflicting units, rotation
        breaking ties, and flag it."""
        start = self.pointer.get(criterion, 0)
        n = len(self.members)
        order = [self.members[(start + k) % n] for k in range(n)]

        def clashes(m: str) -> int:
            return sum(1 for u in units if m in self.forbidden_by_unit.get(u, set()))

        for k, m in enumerate(order):
            if clashes(m) == 0:
                self.pointer[criterion] = (start + k + 1) % n
                return m, False
        k, m = min(enumerate(order), key=lambda km: (clashes(km[1]), km[0]))
        self.pointer[criterion] = (start + k + 1) % n
        self.conflicts += 1
        return m, True


class Study:
    def __init__(self, run: Run, name: str, cfg: StudyCfg, criteria: list[dict[str, str]], frame: str):
        self.run, self.name, self.cfg, self.criteria, self.frame = run, name, cfg, criteria, frame
        self.dir = run.p("ranking", name, "study")
        self.out = run.p("ranking", name)
        self.skill = find_skill(run)
        self.script = self.skill / "scripts" / "tricot.py"
        self.cmd_count = 0

    def tricot(self, *args: str, check: bool = True) -> str:
        r = subprocess.run([sys.executable, str(self.script), *args], capture_output=True, text=True,
                           cwd=str(self.script.parent))
        self.cmd_count += 1
        if check and r.returncode != 0:
            raise LabError(f"tricot {' '.join(args[:2])} failed: {(r.stderr or r.stdout)[-600:]}")
        return r.stdout if r.returncode == 0 else "RC!=0: " + r.stdout + r.stderr

    def setup(self, units: dict[str, tuple[str, str]], terms: dict[str, Any]) -> dict[str, Any]:
        """units: uid -> (title, text). Writes units, criteria, terms; init, anonymize, feasibility."""
        if (self.dir / "study.json").exists():
            return json.loads((self.dir / "study.json").read_text())["feasibility"]
        uroot = self.out / "units_in"
        uroot.mkdir(parents=True, exist_ok=True)
        with (uroot.parent / "units.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["unit_id", "title", "text"])
            w.writeheader()
            for uid, (title, text) in units.items():
                w.writerow({"unit_id": uid, "title": title, "text": text})
        crit = self.out / "criteria.json"
        crit.write_text(json.dumps(self.criteria, indent=1))
        (self.out / "terms.json").write_text(json.dumps(terms, indent=1))
        args = ["init", "--study", str(self.dir), "--units", str(uroot.parent / "units.csv"), "--criteria", str(crit),
                "--goal", self.cfg.goal, "--precision", self.cfg.precision, "--max-reps", str(self.cfg.max_reps),
                "--seed", str(self.run.cfg.seed), "--frame", self.frame]
        if self.cfg.goal == "topk":
            args += ["--top-k", str(self.cfg.top_k)]
        self.tricot(*args)
        self.tricot("anonymize", "--study", str(self.dir), "--terms", str(self.out / "terms.json"))
        self.tricot("feasibility", "--study", str(self.dir))
        feas = json.loads((self.dir / "study.json").read_text())["feasibility"]
        return feas

    def recommended(self) -> str:
        return json.loads((self.dir / "study.json").read_text())["feasibility"].get("recommended", "")

    def blocks(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        with (self.dir / "blocks.csv").open() as f:
            for r in csv.DictReader(f):
                out.setdefault(int(r["block_id"]), []).append(r["unit_id"])
        return out

    def judge_all(self, members: list[str], forbidden: dict[str, set[str]],
                  judge_fn: Callable[[str, str, str, int, str], Judgement]) -> dict[str, Any]:
        """Orchestrator loop (references/claude-code.md). judge_fn(member, criterion, show_text, block, err) -> Judgement."""
        blocks = self.blocks()
        assigner = Assigner(members, forbidden)
        log_path = self.out / "judges.csv"
        new_file = not log_path.exists()
        assess_log: list[str] = []
        n_judged = 0
        with log_path.open("a", newline="") as lf:
            w = csv.writer(lf)
            if new_file:
                w.writerow(["block", "criterion", "member", "timestamp", "conflict"])
            while True:
                pend = json.loads(self.tricot("pending", "--study", str(self.dir), "--limit", "8").strip().splitlines()[-1])
                st = pend["status"]
                if st == "complete":
                    break
                if st == "assess":
                    assess_log.append(self.tricot("assess", "--study", str(self.dir)))
                    continue
                if st == "digest":
                    raise LabError("tricot asked for digests; the spec requires full texts. Stop and report.")
                for t in pend["tasks"]:
                    b, c = int(t["block"]), t["criterion"]
                    member, conflict = assigner.pick(c, blocks[b])
                    show = self.tricot("show", "--study", str(self.dir), "--block", str(b), "--criterion", c)
                    err = ""
                    for attempt in (1, 2):
                        j = judge_fn(member, c, show, b, err)
                        jf = self.out / "judgements_pending" / f"b{b}_{c}.json"
                        jf.parent.mkdir(exist_ok=True)
                        jf.write_text(j.model_dump_json())
                        r = self.tricot("record", "--study", str(self.dir), "--file", str(jf), check=False)
                        if not r.startswith("RC!=0:"):
                            break
                        err = r[-500:]
                    else:
                        raise LabError(f"tricot rejected judgement for block {b} {c}: {err}")
                    w.writerow([b, c, member, time.strftime("%Y-%m-%dT%H:%M:%S"), int(conflict)])
                    lf.flush()
                    n_judged += 1
        (self.out / "assessments.txt").write_text("\n\n".join(assess_log))
        return {"judgements": n_judged, "assessments": len(assess_log), "conflicts": assigner.conflicts}

    def fit(self, npseudo: float = 0.5) -> pd.DataFrame:
        self.tricot("fit", "--study", str(self.dir), "--npseudo", str(npseudo))
        return self.worths()

    def worths(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "worths.csv")

    def summarise_all(self, summarise_fn: Callable[[str, str], dict[str, Any]]) -> None:
        for _ in range(500):
            out = self.tricot("mentions", "--study", str(self.dir), "--unit", "next")
            if "ALL UNITS SUMMARISED" in out:
                return
            uid = re.search(r"=== UNIT (\S+)", out).group(1)  # type: ignore[union-attr]
            payload = summarise_fn(uid, out)
            f = self.out / "summaries_pending" / f"{uid}.json"
            f.parent.mkdir(exist_ok=True)
            f.write_text(json.dumps(payload))
            self.tricot("summarise", "--study", str(self.dir), "--unit", uid, "--file", str(f))
        raise LabError("summarise loop did not finish")

    def export(self) -> None:
        for name in ("worths.csv", "vcov.json", "fit_summary.json", "option_summaries.json"):
            src = self.dir / name
            if src.exists():
                (self.out / name).write_text(src.read_text())

    def purge(self) -> None:
        self.tricot("purge", "--study", str(self.dir))

    def judgements(self) -> list[dict[str, Any]]:
        f = self.dir / "judgements.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
