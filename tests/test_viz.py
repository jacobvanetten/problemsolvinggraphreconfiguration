from signedgraph import toy_graph
from signedgraph.viz import NEGATIVE_COLOR, POSITIVE_COLOR, to_pyvis, write_html


def test_pyvis_colors_and_passages(tmp_path):
    g = toy_graph()
    net = to_pyvis(g)
    assert len(net.nodes) == len(g.nodes)
    assert len(net.edges) == len(g.edges)
    by_key = {(e["from"], e["to"]): e for e in net.edges}
    assert by_key[("crop_yield", "food_price")]["color"] == NEGATIVE_COLOR
    assert by_key[("rainfall", "crop_yield")]["color"] == POSITIVE_COLOR
    assert g.edge("rainfall", "crop_yield").passage in by_key[("rainfall", "crop_yield")]["title"]

    out = write_html(g, tmp_path / "toy.html")
    assert "crop_yield" in out.read_text()
