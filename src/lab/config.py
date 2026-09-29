"""Run configuration (spec 13)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    default: str = "claude-sonnet-5-5"
    integrator: str = "claude-opus-5-5"
    executors: str = "claude-sonnet-5-5"
    judges: str = "claude-sonnet-5-5"


class CasesCfg(BaseModel):
    A: str = "cases/A"
    B: str = "cases/B"
    B_kind: str = "non_agricultural"
    leakage_check: bool = False


class PeopleCfg(BaseModel):
    visitors_enabled: list[str] = Field(
        default_factory=lambda: ["ST1", "ST2", "ST3", "TM", "PW", "VA", "MU", "EC", "TZ", "GD", "SF", "PV"]
    )
    pairing: str = "cycle_max_distance"
    distance_matrix: str | None = None


class NetworkCfg(BaseModel):
    max_nodes_warn: int = 300
    cycle_search_max_len: int = 6
    ego_radius: int = 2


class MethodCfg(BaseModel):
    max_words: int = 1500


class TrialsCfg(BaseModel):
    max_calls: int = 30
    frames_per_trial: int = 3
    trial2_executors: list[str] = Field(default_factory=lambda: ["cross", "reference"])


class CafeCfg(BaseModel):
    rounds: int = 3
    core_visitors_per_table: int = 2
    newcomer_visitors_per_table: int = 2
    item_max_words: int = 80
    allow_merge_after_round: int = 2
    min_methods_for_ranking: int = 6
    # Spec 9 Phase 4 cannot be met with 3 rounds x 2 visits: each core member has only 5 tables
    # they neither authored nor executed but needs 6 distinct visits. Executed tables become
    # eligible from this round on (authored tables never). Set to 0 to insist on the strict rule.
    executed_tables_eligible_from_round: int = 3
    schedule_restarts: int = 400


class ToolsCfg(BaseModel):
    max_calls_per_turn: int = 8


class StudyCfg(BaseModel):
    goal: str
    precision: str = "medium"
    max_reps: int = 9
    top_k: int | None = None


class RankingCfg(BaseModel):
    skill_path: str | None = None
    frames: StudyCfg = StudyCfg(goal="order", max_reps=9)
    methods: StudyCfg = StudyCfg(goal="topk", top_k=3, max_reps=12)


class Phase7Cfg(BaseModel):
    max_selected: int = 4


class HeldoutCfg(BaseModel):
    compare_top_n: int = 2


class Config(BaseModel):
    run_id: str = "reframing_lab_v1"
    language: str = "en"
    mode: str = "auto"  # interactive | auto
    seed: int = 1
    backend: str = "mock"  # mock | anthropic
    model: ModelCfg = ModelCfg()
    cases: CasesCfg = CasesCfg()
    people: PeopleCfg = PeopleCfg()
    network: NetworkCfg = NetworkCfg()
    method: MethodCfg = MethodCfg()
    trials: TrialsCfg = TrialsCfg()
    cafe: CafeCfg = CafeCfg()
    tools: ToolsCfg = ToolsCfg()
    ranking: RankingCfg = RankingCfg()
    phase7: Phase7Cfg = Phase7Cfg()
    heldout: HeldoutCfg = HeldoutCfg()
    context_token_cap: int = 12000
    # resolved at load time
    base_dir: Path = Path(".")

    def resolve(self, p: str | None) -> Path | None:
        if p is None:
            return None
        q = Path(p)
        return q if q.is_absolute() else (self.base_dir / q)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    cfg = Config.model_validate(data)
    return cfg.model_copy(update={"base_dir": path.parent.resolve()})
