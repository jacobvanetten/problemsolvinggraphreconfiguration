import pytest

from signedgraph import NetEffect, cycles, flip_report, net_effect, path_signs, seeded_toy_graph, toy_graph


@pytest.fixture
def g():
    return seeded_toy_graph()


def test_seed_extends_baseline():
    base, seeded = toy_graph(), seeded_toy_graph()
    assert set(base.edges) < set(seeded.edges)
    assert {n.id for n in seeded.nodes} - {n.id for n in base.nodes} == {"improved_seed", "grain_quality"}


def test_seed_adds_no_loops(g):
    assert [c.nodes for c in cycles(g)] == [c.nodes for c in cycles(toy_graph())]


def test_seed_leaves_baseline_effects_unchanged(g):
    base = toy_graph()
    ids = [n.id for n in base.nodes]
    for a in ids:
        for b in ids:
            if a != b:
                assert net_effect(path_signs(g, a, b)) is net_effect(path_signs(base, a, b))


def test_seed_on_food_price_is_mixed(g):
    paths = path_signs(g, "improved_seed", "food_price")
    assert [(p.nodes, p.sign) for p in paths] == [
        (("improved_seed", "crop_yield", "food_price"), -1),
        (("improved_seed", "grain_quality", "food_price"), 1),
    ]
    assert net_effect(paths) is NetEffect.MIXED


@pytest.mark.parametrize(
    "target, expected",
    [
        ("grain_quality", NetEffect.POSITIVE),
        ("crop_yield", NetEffect.MIXED),  # direct +, but back through price and the loops too
        ("hunger", NetEffect.MIXED),
        ("farm_income", NetEffect.MIXED),
        ("labour_productivity", NetEffect.MIXED),
        ("rainfall", NetEffect.NONE),
    ],
)
def test_seed_effects(g, target, expected):
    assert net_effect(path_signs(g, "improved_seed", target)) is expected


def test_without_price_premium_seed_lowers_prices(g):
    report = flip_report(g, "grain_quality", "food_price")
    changes = {(c.source, c.target): (c.before, c.after) for c in report.effect_changes}
    assert changes[("improved_seed", "food_price")] == (NetEffect.MIXED, NetEffect.NEGATIVE)
    assert changes[("improved_seed", "hunger")] == (NetEffect.MIXED, NetEffect.NEGATIVE)
    assert report.cycle_changes == ()
