"""lab run | phase | resume | status | estimate | dry-run (spec 4)."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

import yaml

from lab.config import Config, load_config
from lab.run import Run

PHASES = [(0, "p0_field"), (1, "p1_lenses"), (2, "p2_proposals"), (3, "p3_trial1"), (4, "p4_cafe"), (5, "p5_trial2"),
          (6, "p6_ranking"), (7, "p7_differences"), (8, "p8_integration"), (9, "p9_heldout"), (10, "p10_handoff")]
DATA = Path(__file__).parent / "data"


def run_phases(run: Run, only: int | None = None) -> int:
    for n, mod in PHASES:
        if only is not None and n != only:
            continue
        if run.phase_done(n) and only is None:
            continue
        run.llm.phase = n
        print(f"phase {n} ({mod}) ...", flush=True)
        ok = importlib.import_module(f"lab.phases.{mod}").run_phase(run)
        if not ok:
            print(f"phase {n} finished; waiting at a human checkpoint. See state/*.waiting. "
                  f"Create state/<id>.ok then `lab resume`.")
            run.mark_done(n)  # work is done; only the checkpoint is pending
            return n
        run.mark_done(n)
    return -1


def resume(run: Run) -> int:
    """Interactive: a phase that stopped at a checkpoint is marked done; the next resume moves on
    only when its state/<id>.ok exists."""
    waiting = sorted((run.dir / "state").glob("H*.waiting")) if (run.dir / "state").exists() else []
    for w in waiting:
        if not (run.dir / "state" / f"{w.stem}.ok").exists():
            print(f"checkpoint {w.stem} still waiting: {w.read_text().strip()}")
            return 1
    return 0 if run_phases(run) == -1 else 1


def status(run: Run) -> None:
    done = [n for n, _ in PHASES if run.phase_done(n)]
    print(f"run {run.cfg.run_id} in {run.dir}\nphases done: {done}\n" + str(run.llm.totals()))


def estimate(cfg: Config) -> dict[str, tuple[int, int]]:
    c = cfg
    nm = 9
    visits = c.cafe.rounds * nm * (c.cafe.core_visitors_per_table + c.cafe.newcomer_visitors_per_table)
    t2 = nm * len(c.trials.trial2_executors)
    est = {"0 field A": (10, 40), "1 lens notes": (10, 10), "2 proposals": (27, 27),
           "3 trial 1 + AARs": (nm * 8, nm * c.trials.max_calls + 40),
           "4 café": (visits * 2 + nm * c.cafe.rounds, visits * 2 + nm * c.cafe.rounds + 2 * c.cafe.rounds + 10),
           "5 trial 2 + AARs": (t2 * 8, t2 * c.trials.max_calls + 70),
           "6 ranking": (60, 160 + 250 + 40), "7 differences": (15, 15), "8 integration": (15, 15),
           "9 held-out": (60, 4 * c.trials.max_calls + 80), "10 reflection": (15, 15)}
    return est


def print_estimate(cfg: Config) -> None:
    est = estimate(cfg)
    for k, (lo, hi) in est.items():
        print(f"{k:22s} {lo:5d} to {hi:5d} calls")
    print(f"{'total':22s} {sum(v[0] for v in est.values()):5d} to {sum(v[1] for v in est.values()):5d} calls "
          "(upper bound assumes every trial uses its full cap; tricot judging is often cut short by `assess`)")


def write_dry_workspace(root: Path) -> Path:
    """Toy cases, config and distance matrix for a mock run."""
    root.mkdir(parents=True, exist_ok=True)
    for case in ("A", "B"):
        dst = root / "cases" / case
        dst.mkdir(parents=True, exist_ok=True)
        for f in (DATA / "toy_cases" / case).iterdir():
            shutil.copy(f, dst / f.name)
    (root / "config").mkdir(exist_ok=True)
    src = Path(__file__).resolve().parents[2] / "config" / "discipline_distance.csv"
    shutil.copy(src if src.exists() else DATA / "discipline_distance.csv", root / "config" / "discipline_distance.csv")
    cfg = {"run_id": "dry_run", "mode": "auto", "seed": 1, "backend": "mock",
           "cases": {"A": "cases/A", "B": "cases/B", "B_kind": "non_agricultural", "leakage_check": False},
           "people": {"distance_matrix": "config/discipline_distance.csv"},
           "ranking": {"frames": {"goal": "order", "max_reps": 9}, "methods": {"goal": "topk", "top_k": 3, "max_reps": 12}}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return root / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "resume", "status", "estimate"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="config.yaml")
        p.add_argument("--run-dir", default=None)
    p = sub.add_parser("phase")
    p.add_argument("n", type=int)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--run-dir", default=None)
    p = sub.add_parser("dry-run")
    p.add_argument("--dir", default="dry_run_workspace")
    a = ap.parse_args(argv)
    if a.cmd == "dry-run":
        cfgp = write_dry_workspace(Path(a.dir).resolve())
        cfg = load_config(cfgp)
        run = Run(cfg, cfgp.parent / "runs" / cfg.run_id)
        code = run_phases(run)
        status(run)
        return 0 if code == -1 else 1
    cfg = load_config(a.config)
    if a.cmd == "estimate":
        print_estimate(cfg)
        return 0
    run = Run(cfg, Path(a.run_dir) if a.run_dir else cfg.base_dir / "runs" / cfg.run_id)
    if a.cmd == "status":
        status(run)
        return 0
    if a.cmd == "phase":
        return 0 if run_phases(run, a.n) == -1 else 1
    if a.cmd == "resume":
        return resume(run)
    if cfg.mode == "interactive":
        print_estimate(cfg)
        if input("Proceed? [y/N] ").strip().lower() != "y":
            return 1
    return 0 if run_phases(run) == -1 else 1


if __name__ == "__main__":
    sys.exit(main())
