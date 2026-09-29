import json
from pathlib import Path

import pytest
import yaml

from lab.cases.seal import Seal
from lab.cli import PHASES, run_phases, resume
from lab.llm import LabError
from lab.methods.executor import cross_team, map_roles, reference_team, run_protocol
from lab.people.pairing import DEFAULT_CYCLE
from lab.ranking.tricot_bridge import Assigner
from lab.schemas import AAR, CitedClaim, MethodProtocol
from lab.trials.aar import cite_checker

FIX = Path("tests/fixtures")


def _toy(run):
    """Toy protocol, and the case A network in the run directory."""
    import shutil

    from lab.phases.p0_field import build_field

    build_field(run, "A")
    return MethodProtocol.model_validate(yaml.safe_load((FIX / "toy_method.yaml").read_text()))


def test_executor_records_interpretation_and_uses_invented_operator(make_run):
    run = make_run()
    proto = _toy(run)
    res = run_protocol(run, proto, cross_team(run, ("CA", "SA")), "A", 30)
    assert [i["step"] for i in res.interpretations] == ["S3"]  # the deliberately ambiguous step
    tools = [t["tool"] for t in res.trace if t["kind"] == "tool"]
    assert "apply_edits" in tools and "apply_operators" in tools
    assert res.frames and res.calls_used == 5 and not res.stuck and not res.cut_off
    assert len(run.network("A").edges) == 31  # master network untouched by operators
    assert all(r.stuck is False for r in res.steps.values())


def test_step_that_fails_twice_is_marked_stuck_and_run_continues(make_run):
    run = make_run()
    proto = _toy(run)
    steps = [s.model_copy(update={"action": s.action + " [[FAIL]]"}) if s.id == "S2" else s for s in proto.steps]
    res = run_protocol(run, proto.model_copy(update={"steps": steps}), cross_team(run, ("CA", "SA")), "A", 30)
    assert res.stuck == ["S2"] and res.steps["S2"].stuck
    assert "S3" in res.steps and "S4" in res.steps and res.frames  # later steps still ran
    assert res.calls_used == 1 + 2 + 1 + 1 + 1  # S1, S2 twice, S3, S4, frames


def test_trial_cap_cuts_off_and_is_recorded(make_run):
    run = make_run()
    proto = _toy(run)
    res = run_protocol(run, proto, cross_team(run, ("CA", "SA")), "A", 3)
    assert res.cut_off and res.calls_used <= 3
    assert any(t["kind"] == "cut_off" for t in res.trace)


def test_role_mapping_and_reference_team(make_run):
    run = make_run()
    m = map_roles(["breeder", "geneticist"], cross_team(run, ("QG", "CB")))
    assert m["breeder"].id == "CB"
    rx = reference_team(run)
    assert len(rx.members) == 5
    assert map_roles(["gender specialist"], rx)["gender specialist"].title.endswith("gender specialist")


def test_aar_claims_must_cite_trace(make_run):
    run = make_run()
    proto = _toy(run)
    res = run_protocol(run, proto, cross_team(run, ("CA", "SA")), "A", 30)
    check = cite_checker(proto, res)
    base = dict(steps_worked=[], steps_no_work=[], stuck_points=[], surprises=[], frame_quality="f", method_quality="m",
                attribution="a", proposed_changes=[])
    check(AAR(**{**base, "steps_worked": [CitedClaim(claim="x", cite="S1"), CitedClaim(claim="y", cite="trace:2")]}))
    check(AAR(**{**base, "stuck_points": [CitedClaim(claim="x", cite="interp:S3")]}))
    for bad in ("S9", "trace:999", "interp:S1", "my hunch"):
        with pytest.raises(ValueError):
            check(AAR(**{**base, "surprises": [CitedClaim(claim="x", cite=bad)]}))


def test_seal_catches_case_b_content_before_phase_9(make_run):
    run = make_run()
    run.seal.check("Case A: pearl millet, women processors and the seed lender.")  # fine
    planted = "the harbourmaster's fog log is kept on paper and read by nobody outside the harbour office"
    with pytest.raises(LabError, match="case B"):
        run.llm.build_prompt  # sanity: adapter exists
        run.seal.check(f"Some context. {planted}. More context.")
    with pytest.raises(LabError):  # also through the adapter, so no phase can bypass it
        run.llm.call(task="lens_synthesis", schema=__import__("lab.schemas", fromlist=["Text"]).Text,
                     context={"notes": planted})
    run.seal.unseal()
    run.seal.check(planted)  # allowed after Phase 9 starts
    assert Seal(None, ["planted marker xyz"]).leaks("has PLANTED MARKER XYZ inside") == ["planted marker xyz"]


def test_judge_assignment_avoids_conflicts_where_possible():
    forb = {"u1": {"QG", "SA"}, "u2": {"PL"}, "u3": {"SY", "PD"}}
    a = Assigner(list(DEFAULT_CYCLE), forb)
    seen = []
    for _ in range(20):
        who, conflict = a.pick("c", ["u1", "u2", "u3"])
        assert not conflict and who not in {"QG", "SA", "PL", "SY", "PD"}
        seen.append(who)
    assert set(seen) == {"KG", "CA", "CB", "DS"}  # everyone eligible takes a turn
    # impossible block: everyone forbidden for something -> least-conflicted member, flagged
    forb2 = {"a": set(DEFAULT_CYCLE), "b": {"QG"}}
    who, conflict = Assigner(list(DEFAULT_CYCLE), forb2).pick("c", ["a", "b"])
    assert conflict and who != "QG"  # QG clashes with both units, all others with one


def test_ranking_bridge_random_judges_give_flat_worths_and_assess_stops_early(make_run):
    from lab.config import StudyCfg
    from lab.phases.common import load_criteria
    from lab.ranking.studies import run_study

    run = make_run()
    units = {f"U{i}": (f"U{i}", f"Unit {i}: mock text about a method, number {i}.") for i in range(1, 7)}
    st = run_study(run, "flat", units, {"units": {}, "others": [], "global": []}, StudyCfg(goal="order", max_reps=12),
                   load_criteria(run, "frames")[:1], "a judge", {})
    stats = json.loads((st.out / "run_stats.json").read_text())
    full = 2 * 12  # blocks per round x max reps, one criterion
    assert stats["assessments"] >= 2 and stats["judgements"] < full
    assert "stopped early" in (st.out / "assessments.txt").read_text()
    w = st.worths()
    assert w["log_worth"].abs().max() < 1.5  # random judges: no unit stands out
    assert (st.out / "option_summaries.json").exists()


def _tree(root: Path, sub: list[str]) -> dict[str, bytes]:
    out = {}
    for s in sub:
        for f in sorted((root / s).rglob("*")):
            if f.is_file():
                out[str(f.relative_to(root))] = f.read_bytes()
    return out


@pytest.mark.slow
def test_interactive_checkpoints_and_human_edits(make_run):
    run = make_run(mode="interactive")
    assert run_phases(run) == 0  # stops at H1 after phase 0
    assert resume(run) == 1  # still waiting: no H1.ok
    net = run.p("field", "A", "network.json")
    d = json.loads(net.read_text())
    d["source"] = "human"
    d["nodes"][0]["label"] = "Edited by a human"
    net.write_text(json.dumps(d))
    assert not run.write_json(net, {"nodes": [], "edges": []})  # never overwritten
    assert json.loads(net.read_text())["nodes"][0]["label"] == "Edited by a human"
    run.p("state", "H1.ok").write_text("ok")
    assert resume(run) == 1 and run.phase_done(1)  # moved on to the next checkpoint (H2)


@pytest.mark.slow
def test_human_method_joins_trials_and_cafe(make_run):
    run = make_run()
    hdir = run.p("methods", "human")
    hdir.mkdir(parents=True)
    (hdir / "H01.yaml").write_text((FIX / "toy_method.yaml").read_text())
    for n, mod in PHASES[:5]:
        run.llm.phase = n
        __import__(f"lab.phases.{mod}", fromlist=["x"]).run_phase(run)
    reg = run.registry()
    assert reg["H01"]["source"] == "human" and (run.p("trials", "T1", "H01", "aar.md")).exists()
    ledger = json.loads(run.p("cafe", "ledger.json").read_text())
    assert any(i["table"] == "H01" for i in ledger) and all(i["response"] for i in ledger)


@pytest.mark.slow
def test_merge_after_round_two_keeps_at_least_six_methods(make_run):
    from lab.mock import MockBackend

    run = make_run()
    run.llm.backend = MockBackend(1, run.people, merge_pairs=[["M01", "M02"]])
    for n, mod in PHASES[:5]:
        run.llm.phase = n
        __import__(f"lab.phases.{mod}", fromlist=["x"]).run_phase(run)
    reg = run.registry()
    active = [m for m, v in reg.items() if v["active"]]
    assert len(active) == 8 and "M10" in active and reg["M01"]["merged_into"] == "M10"
    sched = json.loads(run.p("cafe", "schedule.json").read_text())
    assert "M10" in sched["3"] and "M01" not in sched["3"]


@pytest.mark.slow
@pytest.mark.parametrize("boundary", [1, 3])
def test_resume_from_phase_boundary_reproduces_uninterrupted_run(make_run, boundary):
    def go(run, phases):
        for n, mod in PHASES:
            if n in phases:
                run.llm.phase = n
                __import__(f"lab.phases.{mod}", fromlist=["x"]).run_phase(run)
                run.mark_done(n)

    straight = make_run()
    go(straight, range(0, 6))
    split = make_run()
    go(split, range(0, boundary + 1))
    from lab.run import Run

    split2 = Run(split.cfg, split.dir)  # a fresh process would rebuild everything from files
    go(split2, range(boundary + 1, 6))
    subs = ["field", "lenses", "methods", "trials", "cafe"]
    a, b = _tree(straight.dir, subs), _tree(split2.dir, subs)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)


@pytest.mark.slow
def test_full_mock_dry_run_completes_and_validates(make_run):
    run = make_run()
    assert run_phases(run) == -1
    d = run.dir
    for f in ["field/A/network.json", "field/A/field_map.md", "lenses/synthesis.md", "methods/M01.md", "methods/M01.history.json",
              "trials/T1/M01/trace.jsonl", "trials/T1/M01/frames.json", "trials/T1/M01/aar.md", "trials/T2/M01/cross/aar.md",
              "trials/T2/M01/reference/aar.md", "cafe/ledger.json", "cafe/harvest_R1.md", "cafe/harvest_R3.md",
              "ranking/frames/worths.csv", "ranking/frames/vcov.json", "ranking/frames/fit_summary.json", "ranking/frames/option_summaries.json",
              "ranking/methods/worths.csv", "discussion/difference_map.json", "discussion/difference_map.md",
              "integration/composite_method.md", "heldout/comparison.md", "reflection/final_reflection.md",
              "handoff/method_v1.md", "handoff/implementation_notes.md", "handoff/field_questions.md", "logs/calls.jsonl"]:
        assert (d / f).exists(), f
    # every protocol validates and fits its budget
    for f in sorted((d / "methods").glob("M??.history.json")):
        for v in json.loads(f.read_text()):
            MethodProtocol.model_validate(v["protocol"]).validate_limits(1500, 30)
    # blind study 1: frame units carry no method ids in the text judges saw
    import csv
    txt = " ".join(r["text"] for r in csv.DictReader((d / "ranking/frames/study/units.csv").open()))
    assert "M01" not in txt
    # raw judgements were purged only after Phase 7
    assert not (d / "ranking/frames/study/judgements.jsonl").exists()
    # calls log has one line per call and totals
    n = sum(1 for _ in (d / "logs/calls.jsonl").open())
    assert n == run.llm.totals()["calls"] > 500
    # held-out comparison reports every criterion, including negative results
    cmp_ = json.loads((d / "heldout/comparison.json").read_text())
    assert set(cmp_["verdict"]) == {"shift", "fruit", "reach"}
