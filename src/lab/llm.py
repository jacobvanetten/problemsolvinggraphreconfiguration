"""LLM adapter: one interface, a mock backend for tests and an Anthropic backend for real runs.

Agents are stateless calls with explicit context bundles. Structured outputs are JSON validated
with pydantic; on failure the call is retried once with the error, then fails loudly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ValidationError

from lab.schemas import Persona
from lab.toolkit import Toolbox, ToolCapExceeded

T = TypeVar("T", bound=BaseModel)
PROMPT_DIR = Path(__file__).parent / "prompts"
_TASKS = yaml.safe_load((PROMPT_DIR / "tasks.yaml").read_text())
_ENV = Environment(loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined, keep_trailing_newline=True)
METHOD_FIRST_TASKS = {"method_draft", "method_response", "method_protocol", "cafe_items", "reception", "revision",
                      "harvest", "statement", "difference_map", "difference_column", "composite", "objection"}
METHOD_REMINDER = ("Reminder: the object of this task is the METHOD, not the case. "
                   "Comments about case A content must say what they imply for the method.")


class LabError(RuntimeError):
    pass


@dataclass
class BackendResult:
    text: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(Protocol):
    def run(self, *, task: str, model: str, prompt: str, ctx: dict[str, Any], schema: type[BaseModel],
            tool_specs: list[dict[str, Any]], call_tool: Callable[[str, dict[str, Any]], Any] | None,
            persona_id: str | None, attempt: int) -> BackendResult: ...


@dataclass
class CallResult:
    obj: Any
    tool_trace: list[dict[str, Any]]
    call_id: int


def render_context(ctx: dict[str, Any], cap_tokens: int) -> str:
    """Render the bundle as sections; if over the cap, truncate the longest sections first."""
    parts = {k: v if isinstance(v, str) else json.dumps(v, indent=1, default=str, ensure_ascii=False)
             for k, v in ctx.items()}
    budget = cap_tokens * 4
    while sum(len(v) for v in parts.values()) > budget and parts:
        k = max(parts, key=lambda x: len(parts[x]))
        over = sum(len(v) for v in parts.values()) - budget
        keep = max(200, len(parts[k]) - over - 40)
        if keep >= len(parts[k]):
            break
        parts[k] = parts[k][:keep] + "\n[truncated to fit the context cap]"
    return "\n\n".join(f"## {k}\n{v}" for k, v in parts.items())


def persona_block(p: Persona | None) -> str:
    if p is None:
        return "You are a careful, neutral assistant."
    lines = [f"You are {p.title} ({p.id})."]
    for label, val in [("Lens", p.lens), ("Practice", p.practice), ("Voice", p.voice)]:
        if val:
            lines.append(f"{label}: {val}")
    if p.core_questions:
        lines.append("Core questions: " + " | ".join(p.core_questions))
    if p.favoured_moves:
        lines.append("Favoured moves: " + ", ".join(p.favoured_moves))
    if p.must_produce:
        lines.append("You must produce: " + " | ".join(p.must_produce))
    if p.blind_spots:
        lines.append("Known blind spots: " + "; ".join(p.blind_spots))
    return "\n".join(lines)


def extract_json(text: str) -> Any:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(t[i:j + 1])


class LLM:
    def __init__(self, backend: Backend, log_path: Path, models: dict[str, str], context_token_cap: int = 12000,
                 pricing: dict[str, tuple[float, float]] | None = None,
                 seal_check: Callable[[str], None] | None = None):
        self.backend = backend
        self.log_path = log_path
        self.models = models
        self.cap = context_token_cap
        self.pricing = pricing or {}
        self.seal_check = seal_check
        self.phase: int | None = None
        self.n_calls = self._count_existing()

    def _count_existing(self) -> int:
        if self.log_path.exists():
            return sum(1 for _ in self.log_path.open())
        return 0

    def build_prompt(self, task: str, persona: Persona | None, context: dict[str, Any], schema: type[BaseModel],
                     error: str | None = None) -> str:
        return _ENV.get_template("base.j2").render(
            persona=persona_block(persona), task=task, instruction=_TASKS[task],
            reminder=METHOD_REMINDER if task in METHOD_FIRST_TASKS else "",
            context=render_context(context, self.cap),
            schema=json.dumps(schema.model_json_schema(), separators=(",", ":")), error=error)

    def call(self, *, task: str, schema: type[T], context: dict[str, Any], persona: Persona | None = None,
             model: str = "default", toolbox: Toolbox | None = None, tool_names: list[str] | None = None,
             check: Callable[[T], None] | None = None, max_tool_calls: int | None = None) -> CallResult:
        model_id = self.models.get(model, model)
        error: str | None = None
        for attempt in (1, 2):
            prompt = self.build_prompt(task, persona, context, schema, error)
            if self.seal_check:
                self.seal_check(prompt)
            trace: list[dict[str, Any]] = []
            calls_left = [max_tool_calls if max_tool_calls is not None else (toolbox.cap if toolbox else 0)]

            def call_tool(name: str, args: dict[str, Any]) -> Any:
                if toolbox is None or calls_left[0] <= 0:
                    raise ToolCapExceeded("tool call cap reached")
                calls_left[0] -= 1
                try:
                    res = toolbox.call(name, args)
                    trace.append({"tool": name, "args": args, "result": res})
                    return res
                except Exception as exc:
                    trace.append({"tool": name, "args": args, "error": str(exc)})
                    raise

            if toolbox:
                toolbox.new_turn()
            res = self.backend.run(task=task, model=model_id, prompt=prompt, ctx=context, schema=schema,
                                   tool_specs=toolbox.specs(tool_names) if toolbox else [],
                                   call_tool=call_tool if toolbox else None,
                                   persona_id=persona.id if persona else None, attempt=attempt)
            self.n_calls += 1
            ok = True
            obj = None
            try:
                obj = schema.model_validate(extract_json(res.text))
                if check:
                    check(obj)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                ok = False
                error = str(exc)[:1500]
            self._log(task, persona, model_id, res, ok, attempt, len(trace))
            if ok:
                return CallResult(obj, trace, self.n_calls)
        raise LabError(f"task {task!r} failed validation twice: {error}")

    def _log(self, task: str, persona: Persona | None, model_id: str, res: BackendResult, ok: bool, attempt: int,
             n_tools: int = 0) -> None:
        price = self.pricing.get(model_id)
        cost = None if price is None else round((res.input_tokens * price[0] + res.output_tokens * price[1]) / 1e6, 6)
        rec = {"n": self.n_calls, "phase": self.phase, "task": task, "persona": persona.id if persona else None,
               "model": model_id, "attempt": attempt, "ok": ok, "input_tokens": res.input_tokens,
               "output_tokens": res.output_tokens, "cost_usd": cost,
               "tool_calls": n_tools}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def totals(self) -> dict[str, Any]:
        if not self.log_path.exists():
            return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": None}
        rows = [json.loads(line) for line in self.log_path.open()]
        costs = [r["cost_usd"] for r in rows]
        return {"calls": len(rows), "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 4),
                "failed_validations": sum(1 for r in rows if not r["ok"])}


class AnthropicBackend:
    """Real backend: Anthropic messages API with a tool-use loop. Not exercised without an API key."""

    def __init__(self, max_tokens: int = 8000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens

    def run(self, *, task: str, model: str, prompt: str, ctx: dict[str, Any], schema: type[BaseModel],
            tool_specs: list[dict[str, Any]], call_tool: Callable[[str, dict[str, Any]], Any] | None,
            persona_id: str | None, attempt: int) -> BackendResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        trace: list[dict[str, Any]] = []
        tin = tout = 0
        while True:
            kw: dict[str, Any] = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
            if tool_specs:
                kw["tools"] = tool_specs
            resp = self.client.messages.create(**kw)
            tin += resp.usage.input_tokens
            tout += resp.usage.output_tokens
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses or call_tool is None:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return BackendResult(text, trace, tin, tout)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for u in uses:
                try:
                    out = json.dumps(call_tool(u.name, dict(u.input)), default=str)
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": out[:20000]})
                except Exception as exc:  # cap reached or bad args: tell the model, let it finish
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": f"error: {exc}", "is_error": True})
            messages.append({"role": "user", "content": results})
