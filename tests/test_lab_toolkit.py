import pytest

from lab.toolkit import Toolbox, ToolCapExceeded
from lab.toolkit.analogues import Analogue, find_analogues, load_catalogue, motif_graph, motif_score
from lab.toolkit.build import load_network
from lab.toolkit.edits import apply_edits
from lab.toolkit.metrics import global_metrics, local_metrics
from lab.toolkit.motifs import list_motifs
from lab.toolkit.operators import apply_operators
from lab.toolkit.perspectives import get_perspective, unspoken_actors

NET = load_network("tests/fixtures/toy_millet_network.yaml")


def test_global_metrics_hand_checked():
    g = global_metrics(NET)
    assert (g["nodes"], g["edges"], g["components"]) == (17, 31, 1)
    assert "striga" in g["unspoken_actors"] and "women_processors" not in g["unspoken_actors"]
    assert g["reciprocity_direct"] < g["reciprocity_generalised"] <= 1


def test_local_open_obligations():
    loc = local_metrics(NET)
    # male_heads receives e005, e015, e025, e027 with reciprocity pending
    assert loc["male_heads"]["open_obligations"] == 4
    # pm_improved is owed for e005 (pending) and e006 (refused)
    assert loc["pm_improved"]["owed"] == 2


def test_motifs_find_toy_structure():
    ms = list_motifs(NET)
    pats = {m.pattern for m in ms}
    assert {"open_reciprocity_loop", "indirect_reciprocity_cycle", "anomaly", "valency_conflict", "lumped_actor"} <= pats
    cyc = [m for m in ms if m.pattern == "indirect_reciprocity_cycle"]
    assert any(set(m.nodes) == {"livestock", "soil", "pm_landrace"} for m in cyc)
    assert [m for m in ms if m.pattern == "anomaly"][0].nodes == ["young_men_labour"]


def test_close_loop_operator():
    r = apply_operators(NET, [{"op": "close_loop", "args": {"edge": "e005", "counter": {"contribution": "seed money"}}}])
    assert len(r.diff["edges_added"]) == 1 and r.diff["edges_changed"] == ["e005"]
    assert r.local_changes["male_heads"]["open_obligations"] == [4, 3]
    assert r.local_changes["pm_improved"]["owed"] == [2, 1]
    assert "male_heads" in r.affected_perspectives and "pm_improved" in r.narratives


def test_open_loop_operator_lowers_reciprocity():
    r = apply_operators(NET, [{"op": "open_loop", "args": {"edge": "e002"}}])
    assert r.diff["edges_removed"] == ["e003"] and r.diff["edges_changed"] == ["e002"]
    b, a = r.global_changes["reciprocity_direct"]
    assert a < b


def test_flip_valency_and_unknown_perspective():
    r = apply_operators(NET, [{"op": "flip_valency", "args": {"edge": "e001", "perspective": "women_processors"}}])
    assert r.diff["edges_changed"] == ["e001"] and not r.diff["edges_added"]
    assert r.network.edge("e001").valency["women_processors"] == -1
    with pytest.raises(ValueError, match="no valency"):
        apply_operators(NET, [{"op": "flip_valency", "args": {"edge": "e001", "perspective": "striga"}}])


def test_split_and_merge_actors():
    r = apply_operators(NET, [{"op": "split_actor", "args": {"node": "young_men_labour", "parts": [{"id": "young_men_local"}, {"id": "young_men_migrant"}]}}])
    assert r.diff["nodes_removed"] == ["young_men_labour"] and r.diff["nodes_added"] == ["young_men_local", "young_men_migrant"]
    assert len(r.diff["edges_added"]) == 4 and len(r.diff["edges_removed"]) == 2
    m = apply_operators(NET, [{"op": "merge_actors", "args": {"nodes": ["cowpea", "soil"], "into": "fertility_system"}}])
    assert "fertility_system" in m.diff["nodes_added"] and m.network.node("fertility_system").aggregates == ["cowpea", "soil"]


def test_insert_mediator_and_reclassify():
    r = apply_operators(NET, [{"op": "insert_mediator", "args": {"edge": "e005", "mediator": "extension_agent"}}])
    assert r.diff["edges_removed"] == ["e005"] and len(r.diff["edges_added"]) == 2
    c = apply_operators(NET, [{"op": "reclassify", "args": {"node": "young_men_labour", "category": "migrant workers"}}])
    n = c.network.node("young_men_labour")
    assert n.anomaly is None and n.classification == "migrant workers"


def test_apply_edits_matches_named_operator():
    named = apply_operators(NET, [{"op": "reclassify", "args": {"node": "young_men_labour", "category": "migrant workers"}}])
    edits = apply_edits(NET, [{"op": "update_node", "id": "young_men_labour", "fields": {"classification": "migrant workers", "anomaly": None}}])
    assert named.diff == edits.diff and named.global_changes == edits.global_changes
    assert named.local_changes == edits.local_changes and named.narratives == edits.narratives
    assert named.network == edits.network


def test_read_only_operators_do_not_change_network():
    r = apply_operators(NET, [{"op": "take_perspective", "args": {"actor": "women_processors"}}])
    assert not any(r.diff.values()) and r.reading["ranked_motifs"]
    s = apply_operators(NET, [{"op": "shift_scale", "args": {"target": "male_heads", "scale": "global"}}])
    assert s.reading["scale"] == "global"


def test_operators_compose_on_a_copy():
    chain = [{"op": "close_loop", "args": {"edge": "e005", "counter": {"contribution": "seed money"}}},
             {"op": "open_loop", "args": {"edge": "e002"}}]
    r = apply_operators(NET, chain)
    assert len(r.chain) == 2 and r.diff["edges_removed"] == ["e003"]
    assert len(NET.edges) == 31  # master untouched


def test_bad_inputs():
    with pytest.raises(KeyError):
        get_perspective(NET, "nobody")
    with pytest.raises(ValueError):
        apply_operators(NET, [{"op": "teleport", "args": {}}])
    with pytest.raises(KeyError):
        apply_operators(NET, [{"op": "open_loop", "args": {"edge": "zzz"}}])


def test_perspective_is_labelled_hypothesis():
    p = get_perspective(NET, "women_processors")
    assert p.annotation.label == "hypothesis" and "e001" in p.receives


def test_toolbox_cap_and_readonly():
    tb = Toolbox(NET, cap=2)
    tb.call("get_metrics", {"scope": "global"})
    tb.call("get_evidence", {"id": "e001"})
    with pytest.raises(ToolCapExceeded):
        tb.call("list_motifs", {})
    tb.new_turn()
    assert tb.call("list_motifs", {})
    assert len(tb.net.edges) == 31
    with pytest.raises(KeyError):
        tb.call("no_such_tool", {})


def test_analogue_matcher_finds_planted_motif():
    ms = [m for m in list_motifs(NET) if m.pattern == "indirect_reciprocity_cycle" and set(m.nodes) == {"livestock", "soil", "pm_landrace"}]
    assert ms
    planted = Analogue(id="planted", name="Planted ring", description="a three-party ring", themes=["x"],
                       nodes=["a", "b", "c"], edges=[{"source": "a", "target": "b"}, {"source": "b", "target": "c"}, {"source": "c", "target": "a"}], status="proposed")
    decoy = Analogue(id="decoy", name="Pair", description="two parties", themes=["y"], nodes=["a", "b"],
                     edges=[{"source": "a", "target": "b"}])
    assert motif_score(motif_graph(NET, ms[0]), planted.graph()) == 1.0
    assert motif_score(motif_graph(NET, ms[0]), decoy.graph()) == 0.0
    rows = find_analogues(NET, [decoy, planted], motif_pattern="indirect_reciprocity_cycle", motifs=ms, k=2)
    assert rows[0]["analogue"] == "planted"


def test_catalogue_has_twenty_starters_and_theme_match():
    cat = load_catalogue()
    assert len(cat) == 20
    rows = find_analogues(NET, cat, theme_id="T01", k=3)
    assert rows[0]["theme_score"] > 0
