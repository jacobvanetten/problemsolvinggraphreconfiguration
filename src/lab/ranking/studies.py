"""Runs one tricot study end to end with persona judges (spec 11.1)."""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel

from lab.config import StudyCfg
from lab.llm import LabError
from lab.ranking.judge_analysis import divergence
from lab.ranking.tricot_bridge import Study, judge_prompt_text
from lab.run import Run
from lab.schemas import Judgement


class SummaryPayload(BaseModel):
    criteria: dict[str, dict[str, list[dict[str, Any]]]]


def judge_check(labels: list[str], block: int, criterion: str):
    def check(j: Judgement) -> None:
        if j.block != block or j.criterion != criterion:
            raise ValueError("block or criterion does not match the task")
        if sorted(j.ranking) != sorted(labels) or set(j.options) != set(labels):
            raise ValueError(f"options and ranking must use exactly the labels {labels}, no ties")
    return check


def run_study(run: Run, name: str, units: dict[str, tuple[str, str]], terms: dict[str, Any], scfg: StudyCfg,
              criteria: list[dict[str, str]], frame: str, forbidden: dict[str, set[str]],
              extra_ctx: Callable[[str], dict[str, Any]] | None = None, members: list[str] | None = None) -> Study:
    st = Study(run, name, scfg, criteria, frame)
    feas = st.setup(units, terms)
    rec = st.recommended()
    if rec.startswith("digest"):
        raise LabError(f"tricot feasibility recommends {rec!r} for study {name}: digest mode is exploratory. Stop and report.")
    run.note(f"study {name}: feasibility recommends {rec}")
    instr = judge_prompt_text(st.skill)
    members = members or run.core_ids

    def judge_fn(member: str, criterion: str, show: str, block: int, err: str) -> Judgement:
        labels = sorted(set(re.findall(r"OPTION ([A-Z])\b", show)))
        prompt_instr = (instr.replace("STUDY", name).replace("BLOCK", str(block)).replace("CRITERION", criterion)
                        .replace("OUT", "your reply") + "\n(The output of `show` is provided below in place of running it.)")
        ctx = {"judge_instructions": prompt_instr, "show_output": show, "block": block, "criterion": criterion}
        if extra_ctx:
            ctx.update(extra_ctx(criterion))
        if err:
            ctx["record_error"] = err
        return run.llm.call(task="judge", schema=Judgement, persona=run.persona(member), model="judges", context=ctx,
                            check=judge_check(labels, block, criterion)).obj

    stats = st.judge_all(members, forbidden, judge_fn)
    st.fit()

    def summarise_fn(uid: str, mentions_text: str) -> dict[str, Any]:
        ids = set(re.findall(r"\[(b\d+-[sw]\d+)\]", mentions_text))

        def check(p: SummaryPayload) -> None:
            for cid, entry in p.criteria.items():
                for key in ("strengths", "weaknesses"):
                    items = entry.get(key, [])
                    if len(items) > 3:
                        raise ValueError("at most 3 themes per list")
                    for t in items:
                        if len(str(t.get("theme", ""))) > 80 or not t.get("mentions"):
                            raise ValueError("theme too long or cites no mentions")
                        if any(m not in ids for m in t["mentions"]):
                            raise ValueError("cites a mention id that does not exist")
        return run.llm.call(task="summarise", schema=SummaryPayload, persona=run.persona("FA"),
                            context={"unit": uid, "mentions": mentions_text}, check=check).obj.model_dump()

    st.summarise_all(summarise_fn)
    st.export()
    div = divergence(st.judgements(), st.out / "judges.csv", st.worths())
    run.write_json(st.out / "divergence.json", div)
    run.write_json(st.out / "run_stats.json", {**stats, "feasibility_recommended": rec, "tricot_cli_calls": st.cmd_count})
    return st
