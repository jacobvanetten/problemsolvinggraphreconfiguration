import pytest

from signedgraph import (
    NetEffect,
    Polarity,
    cycles,
    cycles_dataframe,
    flip_report,
    net_effect,
    path_signs,
    toy_graph,
)

R1 = ("crop_yield", "food_price", "hunger", "labour_productivity")
B1 = ("crop_yield", "food_price", "farm_income", "fertiliser_use")


@pytest.fixture
def g():
    return toy_graph()


def test_path_signs_rainfall_to_hunger(g):
    paths = path_signs(g, "rainfall", "hunger")
    assert [(p.nodes, p.sign) for p in paths] == [
        (("rainfall", "flooding", "hunger"), 1),
        (("rainfall", "crop_yield", "food_price", "hunger"), -1),
    ]
    assert paths[1].passages == tuple(
        g.edge(u, v).passage for u, v in paths[1].edges
    )
    assert net_effect(paths) is NetEffect.MIXED


def test_path_signs_cutoff_and_trivial_cases(g):
    assert [p.nodes for p in path_signs(g, "rainfall", "hunger", cutoff=2)] == [
        ("rainfall", "flooding", "hunger")
    ]
    assert path_signs(g, "hunger", "rainfall") == []
    assert net_effect(path_signs(g, "hunger", "rainfall")) is NetEffect.NONE
    assert path_signs(g, "hunger", "hunger") == []
    with pytest.raises(KeyError):
        path_signs(g, "rainfall", "nowhere")


def test_net_effect_single_sign(g):
    assert net_effect(path_signs(g, "crop_yield", "food_price")) is NetEffect.NEGATIVE
    # rainfall reaches food_price via yields (-) and via floods -> hunger -> labour (+)
    assert net_effect(path_signs(g, "rainfall", "food_price")) is NetEffect.MIXED
    assert net_effect(path_signs(g, "farm_income", "crop_yield")) is NetEffect.POSITIVE


def test_cycles_and_polarity(g):
    found = cycles(g)
    assert [(c.nodes, c.sign, c.polarity) for c in found] == [
        (B1, -1, Polarity.BALANCING),
        (R1, 1, Polarity.REINFORCING),
    ]
    assert len(found[0].passages) == 4
    df = cycles_dataframe(g)
    assert list(df["polarity"]) == ["balancing", "reinforcing"]


def test_acyclic_graph_has_no_cycles(g):
    acyclic = g.model_copy(
        update={"edges": tuple(e for e in g.edges if e.key != ("crop_yield", "food_price"))}
    )
    assert cycles(acyclic) == []


def test_flip_on_shared_edge_flips_both_loops(g):
    report = flip_report(g, "crop_yield", "food_price")
    assert (report.old_sign, report.new_sign) == (-1, 1)
    assert {(c.nodes, c.before, c.after) for c in report.cycle_changes} == {
        (B1, Polarity.BALANCING, Polarity.REINFORCING),
        (R1, Polarity.REINFORCING, Polarity.BALANCING),
    }


def test_flip_food_price_to_hunger(g):
    report = flip_report(g, "food_price", "hunger")
    assert [(c.nodes, c.before, c.after) for c in report.cycle_changes] == [
        (R1, Polarity.REINFORCING, Polarity.BALANCING)
    ]
    changes = {(c.source, c.target): c for c in report.effect_changes}

    direct = changes[("food_price", "hunger")]
    assert (direct.before, direct.after) == (NetEffect.POSITIVE, NetEffect.NEGATIVE)

    rain = changes[("rainfall", "hunger")]
    assert (rain.before, rain.after) == (NetEffect.MIXED, NetEffect.POSITIVE)
    assert (rain.paths_through_edge, rain.total_paths) == (1, 2)

    # Pairs whose paths never use the edge are not reported.
    assert ("rainfall", "flooding") not in changes
    assert ("farm_income", "crop_yield") not in changes

    # Every reported change actually differs, and matches a recomputation.
    flipped = g.with_flipped_edge("food_price", "hunger")
    for c in report.effect_changes:
        assert c.before != c.after
        assert net_effect(path_signs(flipped, c.source, c.target)) is c.after


def test_flip_edge_outside_cycles(g):
    report = flip_report(g, "flooding", "hunger")
    assert report.cycle_changes == ()
    changes = {(c.source, c.target): (c.before, c.after) for c in report.effect_changes}
    assert changes[("rainfall", "hunger")] == (NetEffect.MIXED, NetEffect.NEGATIVE)
    assert changes[("flooding", "hunger")] == (NetEffect.POSITIVE, NetEffect.NEGATIVE)


def test_flip_report_summary_and_dataframe(g):
    report = flip_report(g, "food_price", "hunger")
    text = report.summary()
    assert text.startswith("Flip food_price -> hunger: +1 => -1")
    assert "reinforcing => balancing" in text
    df = report.effect_changes_dataframe()
    assert len(df) == len(report.effect_changes)
    assert set(df["after"]) <= {e.value for e in NetEffect}


def test_flip_missing_edge_raises(g):
    with pytest.raises(KeyError):
        flip_report(g, "hunger", "rainfall")
