"""Ego views of the network (spec 6.3)."""

from __future__ import annotations

import networkx as nx

from lab.schemas import HUMAN_KINDS, Claim, Network, Perspective
from lab.toolkit.build import to_undirected


def valency_holders(net: Network) -> set[str]:
    """Actors from whose standpoint someone has recorded a valency. Expectation holders do not count:
    a non-human's (attributed) expectation is not a perspective anyone speaks for."""
    return {a for e in net.edges for a in e.valency}


def unspoken_actors(net: Network) -> list[str]:
    """Non-human actors that no perspective in the network speaks for."""
    holders = valency_holders(net)
    return sorted(n.id for n in net.nodes if n.kind not in HUMAN_KINDS and n.id not in holders)


def get_perspective(net: Network, actor: str, ego_radius: int = 2) -> Perspective:
    net.node(actor)
    ego = nx.ego_graph(to_undirected(net), actor, radius=ego_radius)
    visible = set(ego.nodes)
    gives = [e.id for e in net.edges if e.source == actor]
    receives = [e.id for e in net.edges if e.target == actor]
    open_obl = [e.id for e in net.edges if e.target == actor and e.reciprocity in ("pending", "refused")]
    owed = [e.id for e in net.edges if e.source == actor and e.reciprocity in ("pending", "refused")]
    val = sum(e.valency[actor] for e in net.edges if actor in e.valency)
    blind = sorted(n.id for n in net.nodes if n.id not in visible)
    return Perspective(
        actor=actor, ego_radius=ego_radius, gives=gives, receives=receives,
        open_obligations=open_obl, owed=owed, perceived_valency_sum=val, blind_to=blind,
        annotation=Claim(
            text=f"Computed from network structure: {actor} gives {len(gives)}, receives {len(receives)}, "
                 f"owes {len(open_obl)}, is owed {len(owed)}. Lived view is not established.",
            label="hypothesis",
        ),
    )
