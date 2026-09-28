import pytest

from signedgraph import NetEffect, cycles, flip_report, net_effect, path_signs, seeded_toy_graph, toy_graph

NEW_NODES = {"improved_seed", "grain_quality", "farm_gate_price", "premium_supply", "premium_price"}


@pytest.fixture
def g():
    return seeded_toy_graph()


def test_seed_extends_baseline():
    base, seeded = toy_graph(), seeded_toy_graph()
    assert set(base.edges) < set(seeded.edges)
    assert {n.id for n in seeded.nodes} - {n.id for n in base.nodes} == NEW_NODES


def test_seed_adds_no_loops(g):
    assert [c.nodes for c in cycles(g)] == [c.nodes for c in cycles(toy_graph())]


def test_seed_leaves_baseline_effects_unchanged(g):
    base = toy_graph()
    ids = [n.id for n in base.nodes]
    for a in ids:
        for b in ids:
            if a != b:
                assert net_effect(path_signs(g, a, b)) is net_effect(path_signs(base, a, b))


def test_quality_on_farm_gate_price_is_mixed(g):
    paths = path_signs(g, "grain_quality", "farm_gate_price")
    assert [(p.nodes, p.sign) for p in paths] == [
        (("grain_quality", "farm_gate_price"), 1),
        (("grain_quality", "premium_supply", "premium_price", "farm_gate_price"), -1),
    ]
    assert net_effect(paths) is NetEffect.MIXED


def test_seed_on_food_price_is_mixed(g):
    paths = path_signs(g, "improved_seed", "food_price")
    assert [(p.nodes, p.sign) for p in paths] == [
        (("improved_seed", "crop_yield", "food_price"), -1),
        (
            ("improved_seed", "grain_quality", "farm_gate_price", "farm_income",
             "fertiliser_use", "crop_yield", "food_price"),
            -1,
        ),
        (
            ("improved_seed", "grain_quality", "premium_supply", "premium_price",
             "farm_gate_price", "farm_income", "fertiliser_use", "crop_yield", "food_price"),
            1,
        ),
    ]
    assert net_effect(paths) is NetEffect.MIXED


@pytest.mark.parametrize(
    "source, target, expected",
    [
        ("improved_seed", "grain_quality", NetEffect.POSITIVE),
        ("improved_seed", "premium_supply", NetEffect.POSITIVE),
        ("improved_seed", "premium_price", NetEffect.NEGATIVE),
        ("improved_seed", "farm_gate_price", NetEffect.MIXED),
        ("improved_seed", "crop_yield", NetEffect.MIXED),
        ("improved_seed", "hunger", NetEffect.MIXED),
        ("improved_seed", "rainfall", NetEffect.NONE),
        ("grain_quality", "farm_income", NetEffect.MIXED),
        ("grain_quality", "hunger", NetEffect.MIXED),
        ("premium_price", "farm_income", NetEffect.POSITIVE),
        ("premium_supply", "farm_gate_price", NetEffect.NEGATIVE),
    ],
)
def test_seed_effects(g, source, target, expected):
    assert net_effect(path_signs(g, source, target)) is expected


def test_without_premium_pass_through_quality_clearly_helps_farmers(g):
    # If farmers' price did not track the premium price (e.g. fixed contracts).
    cut = g.model_copy(
        update={"edges": tuple(e for e in g.edges if e.key != ("premium_price", "farm_gate_price"))}
    )
    for target in ["farm_gate_price", "farm_income", "fertiliser_use", "crop_yield"]:
        assert net_effect(path_signs(cut, "grain_quality", target)) is NetEffect.POSITIVE
    assert net_effect(path_signs(cut, "grain_quality", "hunger")) is NetEffect.NEGATIVE


def test_flip_premium_pass_through(g):
    report = flip_report(g, "premium_price", "farm_gate_price")
    assert report.cycle_changes == ()
    changes = {(c.source, c.target): (c.before, c.after) for c in report.effect_changes}
    assert changes[("grain_quality", "farm_gate_price")] == (NetEffect.MIXED, NetEffect.POSITIVE)
    assert changes[("improved_seed", "hunger")] == (NetEffect.MIXED, NetEffect.NEGATIVE)
