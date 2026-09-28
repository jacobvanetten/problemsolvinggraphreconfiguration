"""A small hand-built graph used in tests and examples.

Structure (signs in brackets):

    rainfall -[+]-> crop_yield -[-]-> food_price -[+]-> hunger
    rainfall -[+]-> flooding   -[+]-> hunger
    hunger -[-]-> labour_productivity -[+]-> crop_yield          (reinforcing loop R1)
    food_price -[+]-> farm_income -[+]-> fertiliser_use -[+]-> crop_yield  (balancing loop B1)

So ``rainfall`` has a *mixed* effect on ``hunger`` (via yields: -, via floods: +).

``seeded_toy_graph()`` adds an improved seed variety that raises both yield and
grain quality; quality carries a price premium:

    improved_seed -[+]-> crop_yield
    improved_seed -[+]-> grain_quality -[+]-> food_price

The seed has no incoming edges, so the loops are unchanged, but its effect on
``food_price`` is *mixed* (more supply: -, premium quality: +), and so is its effect
on everything downstream of price.
"""

from signedgraph.schema import Node, SignedEdge, SignedGraph

_EDGES = [
    ("rainfall", "crop_yield", 1, "Good rains in the spring raised the harvest well above the five-year average."),
    ("crop_yield", "food_price", -1, "Abundant harvests pushed grain prices down in every regional market."),
    ("food_price", "hunger", 1, "When prices climbed, poorer households cut meals and hunger spread."),
    ("rainfall", "flooding", 1, "Heavy rain swelled the river until it burst its banks."),
    ("flooding", "hunger", 1, "Floods destroyed stored grain, leaving families without food."),
    ("hunger", "labour_productivity", -1, "Hungry workers could not sustain a full day in the fields."),
    ("labour_productivity", "crop_yield", 1, "Where labourers worked longer, more of the crop was brought in."),
    ("food_price", "farm_income", 1, "Higher grain prices lifted what farmers earned per sack."),
    ("farm_income", "fertiliser_use", 1, "With money in hand, farmers bought more fertiliser."),
    ("fertiliser_use", "crop_yield", 1, "Fertilised plots yielded noticeably more grain."),
]

_SEED_EDGES = [
    ("improved_seed", "crop_yield", 1, "Fields sown with the new seed variety produced markedly more grain per hectare."),
    ("improved_seed", "grain_quality", 1, "The new variety also gave fuller, cleaner kernels than the old landrace."),
    ("grain_quality", "food_price", 1, "Buyers paid a premium for the better grain, lifting market prices."),
]


def _build(edges: list[tuple[str, str, int, str]]) -> SignedGraph:
    ids = sorted({end for s, t, *_ in edges for end in (s, t)})
    return SignedGraph(
        nodes=tuple(Node(id=i, label=i.replace("_", " ")) for i in ids),
        edges=tuple(
            SignedEdge(source=s, target=t, sign=sign, passage=p, citation="toy corpus")
            for s, t, sign, p in edges
        ),
    )


def toy_graph() -> SignedGraph:
    return _build(_EDGES)


def seeded_toy_graph() -> SignedGraph:
    """``toy_graph()`` plus an improved seed that raises yield and grain quality."""
    return _build(_EDGES + _SEED_EDGES)
