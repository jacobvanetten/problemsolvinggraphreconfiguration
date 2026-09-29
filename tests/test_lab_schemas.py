import pytest
from pydantic import ValidationError

from lab.schemas import Edge, Frame, MethodProtocol, Network, Persona
from lab.toolkit.build import load_network
import yaml

NET = "tests/fixtures/toy_millet_network.yaml"


def test_network_round_trip():
    n = load_network(NET)
    again = Network.model_validate_json(n.model_dump_json())
    assert again == n
    assert len(n.nodes) == 17 and len(n.edges) == 31


def test_indirect_reciprocity_needs_trace():
    with pytest.raises(ValidationError):
        Edge(id="x", source="a", target="b", contribution="c", flow_type="material", form="gift", reciprocity="indirect")


def test_no_false_agency():
    d = load_network(NET).model_dump()
    d["edges"][0]["expectation"] = {"content": "x", "holder": "striga", "kind": "intentional"}
    with pytest.raises(ValidationError, match="false agency|no false agency"):
        Network.model_validate(d)


def test_counter_edge_must_exist_and_self_loops_rejected():
    d = load_network(NET).model_dump()
    d["edges"][1]["counter_edge"] = "nope"
    with pytest.raises(ValidationError):
        Network.model_validate(d)
    d = load_network(NET).model_dump()
    d["edges"][0]["target"] = d["edges"][0]["source"]
    with pytest.raises(ValidationError):
        Network.model_validate(d)


def test_persona_rules():
    with pytest.raises(ValidationError):
        Persona(id="V", group="visitor", title="t", must_produce=["x"])  # visitor without practice
    with pytest.raises(ValidationError):
        Persona(id="C", group="core", title="t")  # core without must_produce


def test_frame_form():
    Frame(statement="If the problem situation is approached as if it is a ring, then debts become visible.", sees="x")
    with pytest.raises(ValidationError):
        Frame(statement="Make the seed better.", sees="x")


def test_protocol_limits():
    p = MethodProtocol.model_validate(yaml.safe_load(open("tests/fixtures/toy_method.yaml")))
    p.validate_limits(1500, 30)
    with pytest.raises(ValueError, match="cap"):
        p.validate_limits(1500, 10)
    with pytest.raises(ValueError, match="words"):
        p.validate_limits(5, 30)
