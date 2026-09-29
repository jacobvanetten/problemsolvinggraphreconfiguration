"""Network toolkit exposed to methods (spec 7)."""

from __future__ import annotations

import json
from typing import Any, Callable

from lab.schemas import Network
from lab.toolkit.analogues import Analogue, find_analogues as _find, load_catalogue
from lab.toolkit.edits import ConsequenceReport, apply_edits as _apply_edits
from lab.toolkit.metrics import get_metrics as _metrics
from lab.toolkit.motifs import list_motifs as _motifs
from lab.toolkit.operators import apply_operators as _apply_ops
from lab.toolkit.perspectives import get_perspective as _persp


class ToolCapExceeded(RuntimeError):
    pass


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "get_metrics", "description": "Local or global network metrics. scope: 'global', 'local', 'all' or an actor id.",
     "input_schema": {"type": "object", "properties": {"scope": {"type": "string"}}, "required": ["scope"]}},
    {"name": "get_perspective", "description": "Ego view of one actor.",
     "input_schema": {"type": "object", "properties": {"actor_id": {"type": "string"}}, "required": ["actor_id"]}},
    {"name": "list_motifs", "description": "Structural motifs, optionally at one scale (local|global).",
     "input_schema": {"type": "object", "properties": {"scale": {"type": "string"}}}},
    {"name": "apply_operators", "description": "Apply a chain of named operators to a copy. chain: [{op, args}].",
     "input_schema": {"type": "object", "properties": {"chain": {"type": "array", "items": {"type": "object"}}}, "required": ["chain"]}},
    {"name": "apply_edits", "description": "Generic edits on a copy: add_node, remove_node, update_node, add_edge, remove_edge, update_edge.",
     "input_schema": {"type": "object", "properties": {"edits": {"type": "array", "items": {"type": "object"}}}, "required": ["edits"]}},
    {"name": "find_analogues", "description": "Rank catalogue analogues by theme and motif match.",
     "input_schema": {"type": "object", "properties": {"theme_id": {"type": "string"}, "motif_pattern": {"type": "string"}, "k": {"type": "integer"}}}},
    {"name": "get_evidence", "description": "Evidence entries for a node or edge id.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


class Toolbox:
    """Read-only on the master network. Every call is recorded; a per-turn cap is enforced."""

    def __init__(self, net: Network, cycle_max_len: int = 6, ego_radius: int = 2, seed: int = 1,
                 catalogue: list[Analogue] | None = None, cap: int = 8):
        self.net = net
        self.cycle_max_len, self.ego_radius, self.seed = cycle_max_len, ego_radius, seed
        self.catalogue = catalogue if catalogue is not None else load_catalogue()
        self.cap = cap
        self.calls_this_turn = 0

    def new_turn(self) -> None:
        self.calls_this_turn = 0

    def specs(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        return [s for s in TOOL_SPECS if names is None or s["name"] in names]

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if self.calls_this_turn >= self.cap:
            raise ToolCapExceeded(f"tool call cap {self.cap} reached this turn")
        self.calls_this_turn += 1
        fn: Callable[..., Any] = getattr(self, f"_t_{name}", None)  # type: ignore[assignment]
        if fn is None:
            raise KeyError(f"unknown tool {name!r}")
        return _jsonable(fn(**args))

    # -- tools
    def _t_get_metrics(self, scope: str = "global"):
        return _metrics(self.net, scope, self.cycle_max_len, self.ego_radius, self.seed)

    def _t_get_perspective(self, actor_id: str):
        return _persp(self.net, actor_id, self.ego_radius)

    def _t_list_motifs(self, scale: str | None = None):
        return _motifs(self.net, self.cycle_max_len, scale)

    def _t_apply_operators(self, chain: list[dict[str, Any]]):
        return _apply_ops(self.net, chain, self.cycle_max_len, self.ego_radius, self.seed)

    def _t_apply_edits(self, edits: list[dict[str, Any]]):
        return _apply_edits(self.net, edits, cycle_max_len=self.cycle_max_len, ego_radius=self.ego_radius, seed=self.seed)

    def _t_find_analogues(self, theme_id: str | None = None, motif_pattern: str | None = None, k: int = 5):
        return _find(self.net, self.catalogue, theme_id, motif_pattern, _motifs(self.net, self.cycle_max_len), k)

    def _t_get_evidence(self, id: str):  # noqa: A002
        for coll in (self.net.nodes, self.net.edges):
            for x in coll:
                if x.id == id:
                    return [ev.model_dump() for ev in x.evidence]
        raise KeyError(f"unknown id {id!r}")


def _jsonable(x: Any) -> Any:
    if isinstance(x, ConsequenceReport):
        return json.loads(x.model_dump_json())
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, list):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x
