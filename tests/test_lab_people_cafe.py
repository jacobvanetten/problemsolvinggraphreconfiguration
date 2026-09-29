from collections import Counter
from pathlib import Path

import pytest

from lab.cafe.schedule import Params, ScheduleError, TableInfo, plan, validate
from lab.people.pairing import (DEFAULT_CYCLE, cycle_max_distance, cycle_score, executors_for, load_distance, make_pairs)
from lab.run import load_people

DIST = load_distance(Path("config/discipline_distance.csv"))
NEW = ["ST1", "ST2", "ST3", "TM", "PW", "VA", "MU", "EC", "TZ", "GD", "SF", "PV"]


def _tables():
    pairs = make_pairs(DEFAULT_CYCLE)
    ex = executors_for(pairs)
    return [TableInfo(f"M{i+1:02d}", set(pairs[i]), set(ex[i])) for i in range(9)]


def test_people_files():
    p = load_people()
    assert sum(x.group == "core" for x in p.values()) == 9
    assert sum(x.group == "visitor" for x in p.values()) == 12
    assert {"FA", "PE", "IN", "RX"} <= set(p)
    assert all(x.practice for x in p.values() if x.group == "visitor")


def test_pairing_cycle_uses_everyone_twice_and_is_optimal():
    ids = sorted(DEFAULT_CYCLE)
    cyc = cycle_max_distance(ids, DIST)
    assert cyc == DEFAULT_CYCLE
    assert cycle_score(cyc, DIST) == 45.0  # max possible with distances capped at 5
    pairs = make_pairs(cyc)
    counts = Counter(m for pr in pairs for m in pr)
    assert set(counts.values()) == {2} and len(pairs) == 9


def test_executors_share_no_member_with_authors():
    pairs = make_pairs(DEFAULT_CYCLE)
    for a, e in zip(pairs, executors_for(pairs)):
        assert not set(a) & set(e)
    assert executors_for(pairs)[0] == ("PD", "KG")  # spec table: M1 by PD + KG
    assert executors_for(pairs)[5] == ("QG", "SA")  # M6 by QG + SA


@pytest.mark.parametrize("seed", range(1, 51))
def test_cafe_schedule_constraints(seed):
    t = _tables()
    p = Params(core=DEFAULT_CYCLE, newcomers=NEW)
    s = plan(t, p, [1, 2, 3], seed)
    assert validate(s, t, p) == []
    counts = Counter(x for r in s.values() for tb in r.values() for x in tb["newcomers"])
    assert set(counts.values()) <= {4, 5} and sum(counts.values()) == 54


def test_strict_rule_is_infeasible_and_fails_loudly():
    """Spec inconsistency: 6 core visits per member over 3 rounds, but only 5 tables are neither authored nor executed."""
    t = _tables()
    p = Params(core=DEFAULT_CYCLE, newcomers=NEW, executed_eligible_from_round=0)
    with pytest.raises(ScheduleError):
        plan(t, p, [1, 2, 3], 1, restarts=20)
    plan(t, p, [1, 2], 1, restarts=50)  # two rounds are fine


def test_pv_gets_the_most_visits():
    t = _tables()
    p = Params(core=DEFAULT_CYCLE, newcomers=NEW)
    s = plan(t, p, [1, 2, 3], 3)
    counts = Counter(x for r in s.values() for tb in r.values() for x in tb["newcomers"])
    assert counts["PV"] == 5
