"""Method registry and version history (methods/M01.md, M01.history.json)."""

from __future__ import annotations

from lab.run import Run
from lab.schemas import MethodProtocol, Version


def history_path(run: Run, mid: str):
    return run.p("methods", f"{mid}.history.json")


def load_history(run: Run, mid: str) -> list[Version]:
    f = history_path(run, mid)
    if not f.exists():
        return []
    return [Version.model_validate(v) for v in run.read_json(f)]


def add_version(run: Run, mid: str, protocol: MethodProtocol, stage: str, changelog: list | None = None,
                source: str = "agent") -> Version:
    hist = load_history(run, mid)
    # idempotent on resume: replace an existing version of the same stage
    hist = [h for h in hist if h.stage != stage]
    v = Version(version=len(hist) + 1, stage=stage, protocol=protocol,
                changelog=[c if isinstance(c, dict) else c.model_dump() for c in (changelog or [])], source=source)
    hist.append(v)
    run.write_json(history_path(run, mid), [h.model_dump() for h in hist])
    run.write_text(run.p("methods", f"{mid}.md"), protocol.render_md())
    return v


def current(run: Run, mid: str) -> MethodProtocol:
    hist = load_history(run, mid)
    if not hist:
        raise KeyError(f"no versions for {mid}")
    return hist[-1].protocol


def check_protocol(run: Run):
    cfg = run.cfg

    def check(obj) -> None:
        proto = obj.protocol if hasattr(obj, "protocol") else obj
        proto.validate_limits(cfg.method.max_words, cfg.trials.max_calls)
    return check
