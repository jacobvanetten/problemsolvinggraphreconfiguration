"""Sealing and leakage check for the held-out case B (spec 10.2)."""

from __future__ import annotations

import re
from pathlib import Path

from lab.llm import LabError

SHINGLE = 6  # words


def _norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Seal:
    """Holds n-gram fingerprints of case B files. `check` raises if a prompt contains any of them."""

    def __init__(self, case_dir: Path | None = None, extra_strings: list[str] | None = None):
        self.shingles: set[tuple[str, ...]] = set()
        self.planted = [s.lower() for s in (extra_strings or [])]
        self.active = True
        if case_dir and case_dir.exists():
            for f in sorted(case_dir.rglob("*")):
                if f.is_file() and f.suffix in (".md", ".txt", ".yaml", ".yml", ".json", ".graphml"):
                    self.add_text(f.read_text(errors="ignore"))

    def add_text(self, text: str) -> None:
        w = _norm(text)
        for i in range(len(w) - SHINGLE + 1):
            self.shingles.add(tuple(w[i:i + SHINGLE]))

    def leaks(self, prompt: str) -> list[str]:
        hits = [p for p in self.planted if p in prompt.lower()]
        w = _norm(prompt)
        for i in range(len(w) - SHINGLE + 1):
            if tuple(w[i:i + SHINGLE]) in self.shingles:
                hits.append(" ".join(w[i:i + SHINGLE]))
                break
        return hits

    def check(self, prompt: str) -> None:
        if not self.active:
            return
        hits = self.leaks(prompt)
        if hits:
            raise LabError(f"case B content found in a sealed-phase prompt: {hits[0]!r}")

    def unseal(self) -> None:
        self.active = False
