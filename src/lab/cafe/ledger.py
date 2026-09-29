"""Café ledger (cafe/ledger.json): who gave what to which table and what happened to it."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from lab.run import Run


class Ledger:
    def __init__(self, run: Run):
        self.run = run
        self.path = run.p("cafe", "ledger.json")
        self.items: list[dict[str, Any]] = run.read_json(self.path) if self.path.exists() else []

    def save(self) -> None:
        self.run.write_json(self.path, self.items)

    def get(self, item_id: str) -> dict[str, Any]:
        for i in self.items:
            if i["id"] == item_id:
                return i
        raise KeyError(item_id)

    def has(self, item_id: str) -> bool:
        return any(i["id"] == item_id for i in self.items)

    def add(self, *, giver: str, group: str, table: str, rnd: int, type_: str, text: str) -> str:
        iid = f"L-R{rnd}-{table}-{giver}-{type_}"
        if not self.has(iid):
            self.items.append({"id": iid, "giver": giver, "group": group, "table": table, "round": rnd,
                               "type": type_, "text": text, "response": None, "reason": None, "in_next_version": False})
        return iid

    def respond(self, item_id: str, decision: str, reason: str) -> None:
        it = self.get(item_id)
        it["response"], it["reason"] = decision, reason

    def mark_used(self, item_id: str) -> None:
        self.get(item_id)["in_next_version"] = True

    def for_table_round(self, table: str, rnd: int) -> list[dict[str, Any]]:
        return [i for i in self.items if i["table"] == table and i["round"] == rnd]

    def unanswered(self) -> list[str]:
        return [i["id"] for i in self.items if i["response"] is None]

    def uptake(self) -> dict[str, dict[str, dict[str, float]]]:
        """Share of items adopted/adapted and carried into the next version, per giver group and per table.
        Reported only; never used in the ranking."""
        out: dict[str, dict[str, dict[str, float]]] = {"by_group": {}, "by_table": {}}
        for key, name in (("group", "by_group"), ("table", "by_table")):
            bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for i in self.items:
                bucket[i[key]].append(i)
            for k, its in sorted(bucket.items()):
                n = len(its)
                out[name][k] = {"items": n,
                                "accepted": round(sum(i["response"] in ("adopt", "adapt") for i in its) / n, 3),
                                "in_next_version": round(sum(i["in_next_version"] for i in its) / n, 3)}
        return out
