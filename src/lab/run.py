"""Run state: file-based, resumable (spec 4). Phases read and write files under the run directory."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from lab.cases.seal import Seal
from lab.config import Config
from lab.llm import LLM, AnthropicBackend
from lab.schemas import Persona, words
from lab.toolkit import Toolbox
from lab.toolkit.analogues import load_catalogue
from lab.toolkit.build import load_network

PEOPLE_DIR = Path(__file__).parent / "people"
HUMAN_MARK = "<!-- source: human -->"


def load_people() -> dict[str, Persona]:
    out: dict[str, Persona] = {}
    for sub in ("core", "visitors", "roles"):
        for f in sorted((PEOPLE_DIR / sub).glob("*.yaml")):
            p = Persona.model_validate(yaml.safe_load(f.read_text()))
            out[p.id] = p
    return out


class Run:
    def __init__(self, cfg: Config, run_dir: Path, llm: LLM | None = None):
        self.cfg = cfg
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.people = load_people()
        self.core_ids = [i for i, p in self.people.items() if p.group == "core"]
        self.seal = Seal(cfg.resolve(cfg.cases.B))
        if (self.dir / "state" / "unsealed").exists():
            self.seal.unseal()
        self.llm = llm or self._make_llm()
        self.log: list[str] = []

    def _make_llm(self) -> LLM:
        from lab.mock import MockBackend

        backend = MockBackend(self.cfg.seed, self.people) if self.cfg.backend == "mock" else AnthropicBackend()
        m = self.cfg.model
        return LLM(backend, self.dir / "logs" / "calls.jsonl",
                   {"default": m.default, "integrator": m.integrator, "executors": m.executors, "judges": m.judges},
                   self.cfg.context_token_cap, seal_check=self.seal.check)

    # ---- files
    def p(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def rng(self, label: str) -> random.Random:
        return random.Random(f"{self.cfg.seed}:{label}")

    def _human_locked(self, path: Path) -> bool:
        if not path.exists():
            return False
        head = path.read_text()[:400]
        if path.suffix == ".json":
            try:
                d = json.loads(path.read_text())
                return isinstance(d, dict) and d.get("source") == "human"
            except json.JSONDecodeError:
                return False
        return HUMAN_MARK in head

    def write_text(self, path: Path, text: str) -> bool:
        """Never overwrites files a human has marked source: human."""
        if self._human_locked(path):
            self.note(f"kept human file {path.relative_to(self.dir)}")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return True

    def write_json(self, path: Path, obj: Any) -> bool:
        if isinstance(obj, BaseModel):
            obj = json.loads(obj.model_dump_json())
        return self.write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def note(self, msg: str) -> None:
        self.log.append(msg)
        d = self.dir / "logs"
        d.mkdir(exist_ok=True)
        with (d / "run.log").open("a") as f:
            f.write(msg + "\n")

    # ---- state
    def state_file(self, phase: int) -> Path:
        return self.p("state", f"p{phase}.done")

    def phase_done(self, phase: int) -> bool:
        return self.state_file(phase).exists()

    def mark_done(self, phase: int) -> None:
        self.state_file(phase).parent.mkdir(parents=True, exist_ok=True)
        self.state_file(phase).write_text("done\n")

    def checkpoint(self, cid: str, note: str) -> bool:
        """interactive: pause (returns False until state/<id>.ok exists). auto: log and continue."""
        self.note(f"checkpoint {cid}: {note}")
        if self.cfg.mode == "auto":
            return True
        ok = self.p("state", f"{cid}.ok")
        if ok.exists():
            return True
        ok.parent.mkdir(parents=True, exist_ok=True)
        self.p("state", f"{cid}.waiting").write_text(note + "\nEdit files, then run `lab resume` after creating "
                                                      f"state/{cid}.ok\n")
        return False

    # ---- domain helpers
    def case_dir(self, case: str) -> Path:
        d = self.cfg.resolve(self.cfg.cases.A if case == "A" else self.cfg.cases.B)
        assert d is not None
        return d

    def case_files(self, case: str) -> dict[str, str]:
        d = self.case_dir(case)
        out = {}
        for name in ("archaeology.md", "paradox.md", "context.md"):
            f = d / name
            out[name] = f.read_text() if f.exists() else ""
        return out

    def network(self, case: str):
        return load_network(self.p("field", case, "network.json"))

    def toolbox(self, case: str) -> Toolbox:
        c = self.cfg
        extra = self.p("catalogue_proposed")
        return Toolbox(self.network(case), c.network.cycle_search_max_len, c.network.ego_radius, c.seed,
                       load_catalogue(extra), c.tools.max_calls_per_turn)

    def persona(self, pid: str) -> Persona:
        return self.people[pid]

    def registry(self) -> dict[str, Any]:
        f = self.p("methods", "registry.json")
        return json.loads(f.read_text()) if f.exists() else {}

    def save_registry(self, reg: dict[str, Any]) -> None:
        self.write_json(self.p("methods", "registry.json"), reg)

    def cap_words(self, n: int):
        def check(obj: Any) -> None:
            t = getattr(obj, "text", None)
            if t is not None and words(t) > n:
                raise ValueError(f"text has {words(t)} words, cap {n}")
        return check
