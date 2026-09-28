import pandas as pd
import pytest
from pydantic import ValidationError

from signedgraph import Node, SignedEdge, SignedGraph, toy_graph


def _edge(s="a", t="b", sign=1, passage="a raises b"):
    return SignedEdge(source=s, target=t, sign=sign, passage=passage)


@pytest.mark.parametrize("sign", [0, 2, -2, "+"])
def test_sign_must_be_plus_or_minus_one(sign):
    with pytest.raises(ValidationError):
        _edge(sign=sign)


def test_passage_is_required_and_non_empty():
    with pytest.raises(ValidationError):
        SignedEdge(source="a", target="b", sign=1)
    with pytest.raises(ValidationError):
        _edge(passage="")


def test_edges_must_reference_known_nodes():
    with pytest.raises(ValidationError, match="unknown node"):
        SignedGraph(nodes=(Node(id="a"),), edges=(_edge(),))


def test_duplicate_edges_and_self_loops_rejected():
    nodes = (Node(id="a"), Node(id="b"))
    with pytest.raises(ValidationError, match="duplicate edge"):
        SignedGraph(nodes=nodes, edges=(_edge(), _edge(sign=-1)))
    with pytest.raises(ValidationError, match="self-loop"):
        SignedGraph(nodes=nodes, edges=(_edge("a", "a"),))


def test_duplicate_node_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate node"):
        SignedGraph(nodes=(Node(id="a"), Node(id="a")))


def test_flip_returns_new_graph_and_leaves_original_untouched():
    g = toy_graph()
    flipped = g.with_flipped_edge("food_price", "hunger")
    assert g.edge("food_price", "hunger").sign == 1
    assert flipped.edge("food_price", "hunger").sign == -1
    assert flipped.edge("food_price", "hunger").passage == g.edge("food_price", "hunger").passage
    with pytest.raises(KeyError):
        g.with_flipped_edge("hunger", "rainfall")


def test_networkx_round_trip():
    g = toy_graph()
    rebuilt = SignedGraph.from_networkx(g.to_networkx())
    assert set(rebuilt.nodes) == set(g.nodes)
    assert set(rebuilt.edges) == set(g.edges)


def test_node_without_label_round_trips_through_networkx():
    g = SignedGraph(nodes=(Node(id="a"), Node(id="b")), edges=(_edge(),))
    assert SignedGraph.from_networkx(g.to_networkx()) == g


def test_dataframe_round_trip():
    g = toy_graph()
    df = g.edges_dataframe()
    assert list(df.columns) == ["source", "target", "sign", "passage", "citation"]
    assert len(df) == len(g.edges)
    rebuilt = SignedGraph.from_edges_dataframe(df, nodes=list(g.nodes))
    assert rebuilt == g


def test_dataframe_creates_missing_nodes_and_handles_nan_citation():
    df = pd.DataFrame([{"source": "x", "target": "y", "sign": -1, "passage": "x lowers y", "citation": float("nan")}])
    g = SignedGraph.from_edges_dataframe(df)
    assert {n.id for n in g.nodes} == {"x", "y"}
    assert g.edge("x", "y").citation is None
