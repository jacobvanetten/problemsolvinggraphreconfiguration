"""Pydantic models for the lab. Structured LLM outputs are validated against these."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Confidence = Literal["low", "medium", "high", "hypothesis"]
NodeKind = Literal[
    "human_person", "human_group", "organisation", "organism", "variety",
    "material", "environment", "artefact", "institution", "information",
]
FlowType = Literal[
    "material", "labour", "knowledge", "money", "risk", "recognition",
    "nutrients", "data", "seed", "harm",
]
Form = Literal[
    "gift", "market", "obligation", "extraction", "mutualism", "script",
    "predation", "redistribution",
]
ExpectationKind = Literal["intentional", "functional", "attributed"]
Horizon = Literal["immediate", "season", "years", "generation"]
Reciprocity = Literal[
    "reciprocated", "pending", "refused", "indirect", "not_expected", "unknown"
]
HUMAN_KINDS = {"human_person", "human_group"}


def words(text: str) -> int:
    return len(text.split())


class Claim(BaseModel):
    """A statement about people, places or crops, labelled by evidence status (spec 14)."""

    text: str = Field(min_length=1)
    label: Literal["sourced", "hypothesis"]


class Evidence(BaseModel):
    source: str
    confidence: Confidence = "hypothesis"


class Expectation(BaseModel):
    content: str
    holder: str
    addressee: str | None = None
    kind: ExpectationKind
    horizon: Horizon = "season"


class Node(BaseModel):
    id: str
    label: str
    kind: NodeKind
    classification: str | None = None
    anomaly: str | None = None
    aggregates: list[str] = Field(default_factory=list)
    mediator: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    id: str
    source: str
    target: str
    contribution: str
    flow_type: FlowType
    form: Form
    expectation: Expectation | None = None
    reciprocity: Reciprocity = "unknown"
    counter_edge: str | None = None
    valency: dict[str, Literal[1, -1]] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _indirect_needs_trace(self) -> Edge:
        if self.reciprocity == "indirect" and not (
            self.counter_edge or self.extra.get("path_note")
        ):
            raise ValueError(f"edge {self.id}: indirect reciprocity needs counter_edge or extra.path_note")
        if self.expectation and self.expectation.kind == "intentional":
            pass  # non-human intentions are checked at network level (needs node kinds)
        return self


class Theme(BaseModel):
    id: str
    label: str
    kind: Literal["universal", "situated"] = "universal"
    grounding: list[str] = Field(default_factory=list)


class Motif(BaseModel):
    id: str
    label: str
    pattern: str
    nodes: list[str]
    edges: list[str] = Field(default_factory=list)
    scale: Literal["local", "global"] = "local"


class ThemeMotifLink(BaseModel):
    theme: str
    motif: str
    relation: str


class Network(BaseModel):
    """Exchange network. Working copies are made with model_copy / rebuild, never mutated in place."""

    nodes: list[Node]
    edges: list[Edge]
    themes: list[Theme] = Field(default_factory=list)
    motifs: list[Motif] = Field(default_factory=list)
    links: list[ThemeMotifLink] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Network:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids")
        idset = set(ids)
        eids = [e.id for e in self.edges]
        if len(eids) != len(set(eids)):
            raise ValueError("duplicate edge ids")
        kinds = {n.id: n.kind for n in self.nodes}
        for e in self.edges:
            if e.counter_edge and e.counter_edge not in eids:
                raise ValueError(f"edge {e.id}: counter_edge {e.counter_edge} does not exist")
        for e in self.edges:
            if e.source not in idset or e.target not in idset:
                raise ValueError(f"edge {e.id} has unknown endpoint")
            if e.source == e.target:
                raise ValueError(f"edge {e.id} is a self-loop")
            x = e.expectation
            if x and x.kind == "intentional" and kinds.get(x.holder) not in HUMAN_KINDS | {"organisation", "institution"}:
                raise ValueError(f"edge {e.id}: non-human expectation must be functional or attributed (no false agency)")
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"unknown node {node_id!r}")

    def edge(self, edge_id: str) -> Edge:
        for e in self.edges:
            if e.id == edge_id:
                return e
        raise KeyError(f"unknown edge {edge_id!r}")


class Perspective(BaseModel):
    actor: str
    ego_radius: int
    gives: list[str]
    receives: list[str]
    open_obligations: list[str]
    owed: list[str]
    perceived_valency_sum: int
    blind_to: list[str]
    annotation: Claim


# ---------------------------------------------------------------- people

class Persona(BaseModel):
    id: str
    group: Literal["core", "visitor", "role"]
    title: str
    lens: str = ""
    practice: str = ""
    core_questions: list[str] = Field(default_factory=list)
    favoured_moves: list[str] = Field(default_factory=list)
    must_produce: list[str] = Field(default_factory=list)
    blind_spots: list[str] = Field(default_factory=list)
    voice: str = ""

    @model_validator(mode="after")
    def _rules(self) -> Persona:
        if self.group in ("core", "visitor") and not self.must_produce:
            raise ValueError(f"{self.id}: must_produce is required")
        if self.group == "visitor" and not self.practice:
            raise ValueError(f"{self.id}: visitors need a concrete practice")
        return self


# ---------------------------------------------------------------- methods

class StepOutputSpec(BaseModel):
    type: str
    schema_hint: str = ""


class Step(BaseModel):
    id: str
    name: str
    roles: list[str]
    inputs: list[str] = Field(default_factory=list)
    action: str
    tools: list[str] = Field(default_factory=list)
    output: StepOutputSpec
    done_when: str
    max_calls: int = Field(default=4, ge=1)


class MethodProtocol(BaseModel):
    name: str
    purpose: str
    network_stance: Literal["uses", "extends", "rejects"]
    stance_reason: str
    representations: str
    steps: list[Step]
    how_themes: str
    how_perspectives: str
    how_analogues: str
    how_scales: str
    how_paradox: str
    how_frame: str
    human_role: str
    stopping_rules: str
    failure_modes: str
    budget_calls: int = Field(ge=1)
    example_frame: str | None = None

    @field_validator("name")
    @classmethod
    def _name_len(cls, v: str) -> str:
        if words(v) > 8:
            raise ValueError("name is longer than 8 words")
        return v

    @model_validator(mode="after")
    def _steps(self) -> MethodProtocol:
        if not self.steps:
            raise ValueError("protocol needs at least one step")
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        return self

    def text_words(self) -> int:
        return sum(
            words(getattr(self, f))
            for f in ("purpose", "stance_reason", "representations", "how_themes", "how_perspectives",
                      "how_analogues", "how_scales", "how_paradox", "how_frame", "human_role",
                      "stopping_rules", "failure_modes")
        )

    def validate_limits(self, max_words: int, trial_cap: int) -> None:
        if self.text_words() > max_words:
            raise ValueError(f"protocol has {self.text_words()} words, cap {max_words}")
        step_sum = sum(s.max_calls for s in self.steps)
        if self.budget_calls > trial_cap:
            raise ValueError(f"budget {self.budget_calls} exceeds trial cap {trial_cap}")
        if step_sum > self.budget_calls:
            raise ValueError(f"steps need {step_sum} calls, budget is {self.budget_calls}")

    def render_md(self, authors_hidden: bool = True) -> str:
        out = [f"# {self.name}", "", f"**Purpose and claim.** {self.purpose}", "",
               f"**Network stance ({self.network_stance}).** {self.stance_reason}", "",
               f"**Representations.** {self.representations}", "", "## Steps", ""]
        for s in self.steps:
            out += [f"### {s.id}. {s.name}", f"- roles: {', '.join(s.roles)}",
                    f"- inputs: {', '.join(s.inputs)}", f"- action: {s.action}",
                    f"- tools: {', '.join(s.tools) or 'none'}",
                    f"- output: {s.output.type} ({s.output.schema_hint})",
                    f"- done when: {s.done_when}", f"- max calls: {s.max_calls}", ""]
        for h, v in [("How themes are found", self.how_themes),
                     ("How perspectives are reached", self.how_perspectives),
                     ("How analogues are found", self.how_analogues),
                     ("Local and global checks", self.how_scales),
                     ("Use of the paradox", self.how_paradox),
                     ("How a frame is formed and stated", self.how_frame),
                     ("Role of the human team", self.human_role),
                     ("Stopping rules", self.stopping_rules),
                     ("Expected failure modes", self.failure_modes)]:
            out += [f"## {h}", v, ""]
        out += [f"## Budget", f"{self.budget_calls} calls", ""]
        if self.example_frame:
            out += ["## Example frame from Trial 1", self.example_frame, ""]
        return "\n".join(out)


class Version(BaseModel):
    version: int
    stage: str
    protocol: MethodProtocol
    changelog: list[dict[str, Any]] = Field(default_factory=list)
    source: Literal["agent", "human"] = "agent"


# ---------------------------------------------------------------- trials

class StepResult(BaseModel):
    output: str
    interpretation: str | None = None
    stuck: bool = False
    hypotheses: list[Claim] = Field(default_factory=list)  # claims about people, places or crops


class Frame(BaseModel):
    statement: str
    sees: str
    evidence_steps: list[str] = Field(default_factory=list)

    @field_validator("statement")
    @classmethod
    def _form(cls, v: str) -> str:
        if not re.search(r"\bas if\b", v, re.I) or "then" not in v.lower():
            raise ValueError("frame must have the form 'If the problem situation is approached as if it is X, then Y'")
        return v

    @field_validator("sees")
    @classmethod
    def _sees_len(cls, v: str) -> str:
        if words(v) > 150:
            raise ValueError("'sees' statement exceeds 150 words")
        return v


class FrameSet(BaseModel):
    frames: list[Frame] = Field(min_length=1, max_length=3)
    best: int = 0

    @model_validator(mode="after")
    def _best(self) -> FrameSet:
        if not 0 <= self.best < len(self.frames):
            raise ValueError("best index out of range")
        return self


class Text(BaseModel):
    text: str = Field(min_length=1)


class CitedClaim(BaseModel):
    claim: str
    cite: str = Field(min_length=1, description="step id, trace line or interpretation note")


class AAR(BaseModel):
    steps_worked: list[CitedClaim]
    steps_no_work: list[CitedClaim]
    stuck_points: list[CitedClaim]
    surprises: list[CitedClaim]
    frame_quality: str
    method_quality: str
    attribution: str = Field(description="which issues belong to method, case or executors")
    proposed_changes: list[CitedClaim]
    comparison: str | None = None

    def render_md(self) -> str:
        def sec(title: str, items: list[CitedClaim]) -> str:
            body = "\n".join(f"- {c.claim} [{c.cite}]" for c in items) or "- none"
            return f"## {title}\n{body}\n"
        return "\n".join([
            sec("Steps that did work", self.steps_worked),
            sec("Steps that did no work", self.steps_no_work),
            sec("Stuck points and interpretations", self.stuck_points),
            sec("Surprises", self.surprises),
            f"## Frame quality versus method quality\n- Frame: {self.frame_quality}\n- Method: {self.method_quality}\n- Attribution: {self.attribution}\n",
            sec("Proposed changes", self.proposed_changes),
            f"## Comparison with Trial 1\n{self.comparison}\n" if self.comparison else "",
        ])


# ---------------------------------------------------------------- café

ItemType = Literal["addition", "challenge", "graft"]


class CafeItem(BaseModel):
    type: ItemType
    text: str

    @field_validator("text")
    @classmethod
    def _len(cls, v: str) -> str:
        if words(v) > 80:
            raise ValueError("item exceeds 80 words")
        return v


class CafeItems(BaseModel):
    items: list[CafeItem]
    must_produce_in: ItemType | None = None
    reversal_of: str | None = None

    @model_validator(mode="after")
    def _three(self) -> CafeItems:
        if sorted(i.type for i in self.items) != ["addition", "challenge", "graft"]:
            raise ValueError("exactly one addition, one challenge and one graft are required")
        return self


class Response(BaseModel):
    item_id: str
    decision: Literal["adopt", "adapt", "decline"]
    reason: str = Field(min_length=1)


class Reception(BaseModel):
    responses: list[Response]


class Change(BaseModel):
    change: str
    ledger_ids: list[str] = Field(default_factory=list)


class Revision(BaseModel):
    protocol: MethodProtocol
    changelog: list[Change]


# ---------------------------------------------------------------- integration

class Module(BaseModel):
    name: str
    step_ids: list[str] = Field(default_factory=list)
    inputs: list[str]
    outputs: list[str]
    source_methods: list[str]
    ledger_items: list[str] = Field(default_factory=list)
    trial_evidence: list[str] = Field(default_factory=list)


class Composite(BaseModel):
    protocol: MethodProtocol
    modules: list[Module] = Field(min_length=1)
    failed_modules: list[CitedClaim] = Field(default_factory=list)
    open_choices: list[str] = Field(default_factory=list)
    resolutions: list[str] = Field(default_factory=list)


class Difference(BaseModel):
    a: str
    b: str
    tag: Literal["complementary", "compatible", "in_tension", "exclusive"]
    reason: str
    sources: list[str]


class DifferenceMap(BaseModel):
    pairs: list[Difference]


class Objection(BaseModel):
    text: str


class Judgement(BaseModel):
    """Mirror of the tricot judgement template."""

    block: int
    criterion: str
    options: dict[str, dict[str, list[str]]]
    reasoning: str
    ranking: list[str]
