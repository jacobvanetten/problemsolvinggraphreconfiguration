"""Phase 2: pair method proposals (spec 9)."""

from __future__ import annotations

from lab.methods.store import add_version, check_protocol
from lab.people.pairing import cycle_max_distance, load_distance, make_pairs, executors_for
from lab.phases.common import DATA
from lab.phases.p1_lenses import BRIEF
from lab.run import Run
from lab.schemas import MethodProtocol, Text


def run_phase(run: Run) -> bool:
    dm = run.cfg.resolve(run.cfg.people.distance_matrix) if run.cfg.people.distance_matrix else None
    if dm is None or not dm.exists():
        raise FileNotFoundError("people.distance_matrix is required (config/discipline_distance.csv)")
    cycle = cycle_max_distance(run.core_ids, load_distance(dm))
    pairs = make_pairs(cycle)
    run.write_json(run.p("methods", "pairs.json"), pairs)
    run.write_json(run.p("methods", "cycle.json"), cycle)
    execs = executors_for(pairs)
    reg = run.registry()
    field_map = run.p("field", "A", "field_map.md").read_text()
    synth = run.p("lenses", "synthesis.md").read_text()
    for i, (a, b) in enumerate(pairs):
        mid = f"M{i + 1:02d}"
        if mid in reg and run.p("methods", f"{mid}.history.json").exists():
            continue
        base = {"brief": BRIEF, "lens_synthesis": synth, "case_A_field_map": field_map}
        da = run.llm.call(task="method_draft", schema=Text, persona=run.persona(a), context=base,
                          check=run.cap_words(500)).obj.text
        db = run.llm.call(task="method_response", schema=Text, persona=run.persona(b),
                          context={**base, "partner_draft": da}, check=run.cap_words(900)).obj.text
        proto = run.llm.call(task="method_protocol", schema=MethodProtocol, persona=run.persona(a),
                             context={**base, "draft_a": da, "draft_b_with_response": db,
                                      "limits": {"max_words": run.cfg.method.max_words, "trial_max_calls": run.cfg.trials.max_calls,
                                                 "budget_rule": "sum of step max_calls <= budget_calls <= trial_max_calls"}},
                             check=check_protocol(run)).obj
        add_version(run, mid, proto, "proposal")
        run.write_json(run.p("methods", f"{mid}.intent.json"), {"authors": [a, b], "draft_a": da, "draft_b": db})
        reg[mid] = {"pair_index": i, "authors": [a, b], "executors": list(execs[i]), "source": "agent", "active": True}
        run.save_registry(reg)
    return run.checkpoint("H2", "Add human-written methods as methods/human/*.yaml (MethodProtocol fields); they join the trials and café")
