"""Pairing cycle, executor mapping (spec 9 Phase 2, 8.2)."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

DEFAULT_CYCLE = ["QG", "SA", "PL", "SY", "PD", "KG", "CA", "CB", "DS"]


def load_distance(path: Path) -> dict[frozenset[str], float]:
    d: dict[frozenset[str], float] = {}
    with path.open() as f:
        rows = list(csv.reader(f))
    head = rows[0][1:]
    for r in rows[1:]:
        for h, v in zip(head, r[1:]):
            if r[0] != h and v != "":
                d[frozenset((r[0], h))] = float(v)
    return d


def cycle_score(cycle: list[str], dist: dict[frozenset[str], float]) -> float:
    return sum(dist[frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))] for i in range(len(cycle)))


def cycle_max_distance(members: list[str], dist: dict[frozenset[str], float]) -> list[str]:
    """Hamiltonian cycle with the largest summed neighbour distance. Ties go to the default order,
    then to the lexicographically first cycle (deterministic)."""
    first, rest = members[0], sorted(members[1:])
    best: tuple[float, list[str]] | None = None
    for perm in itertools.permutations(rest):
        if perm[0] > perm[-1]:
            continue  # mirror image
        cyc = [first, *perm]
        sc = cycle_score(cyc, dist)
        if best is None or sc > best[0] + 1e-9:
            best = (sc, cyc)
    assert best is not None
    if set(members) == set(DEFAULT_CYCLE) and abs(cycle_score(DEFAULT_CYCLE, dist) - best[0]) < 1e-9:
        return list(DEFAULT_CYCLE)
    return best[1]


def make_pairs(cycle: list[str]) -> list[tuple[str, str]]:
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def executor_pair_index(i: int, n: int) -> int:
    """Pair i + 4 (mod n) executes the method authored by pair i (no shared member for n = 9)."""
    return (i + 4) % n


def executors_for(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    n = len(pairs)
    return [pairs[executor_pair_index(i, n)] for i in range(n)]
