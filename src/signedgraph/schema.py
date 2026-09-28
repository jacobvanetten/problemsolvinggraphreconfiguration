"""Pydantic schema for signed directed graphs whose edges cite a source passage."""

from __future__ import annotations

from typing import Literal

import networkx as nx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sign = Literal[1, -1]


class Node(BaseModel):
    """A concept in the graph. ``id`` is the stable key; ``label`` is for display."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    label: str | None = None

    @property
    def display(self) -> str:
        return self.label or self.id


class SignedEdge(BaseModel):
    """A directed causal claim ``source -> target`` with a sign and its evidence.

    ``sign`` is +1 (more source -> more target) or -1 (more source -> less target).
    ``passage`` is the verbatim text the claim was extracted from; it is required,
    because every edge must be traceable to its evidence.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    sign: Sign
    passage: str = Field(min_length=1)
    citation: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.target)

    def flipped(self) -> SignedEdge:
        return self.model_copy(update={"sign": -self.sign})


class SignedGraph(BaseModel):
    """A signed digraph: at most one edge per ordered node pair, no self-loops."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[Node, ...] = ()
    edges: tuple[SignedEdge, ...] = ()

    @model_validator(mode="after")
    def _check_integrity(self) -> SignedGraph:
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids")
        known = set(ids)
        seen: set[tuple[str, str]] = set()
        for e in self.edges:
            for end in e.key:
                if end not in known:
                    raise ValueError(f"edge {e.source}->{e.target} references unknown node {end!r}")
            if e.source == e.target:
                raise ValueError(f"self-loop on {e.source!r} is not allowed")
            if e.key in seen:
                raise ValueError(f"duplicate edge {e.source}->{e.target}")
            seen.add(e.key)
        return self

    def edge(self, source: str, target: str) -> SignedEdge:
        for e in self.edges:
            if e.key == (source, target):
                return e
        raise KeyError(f"no edge {source}->{target}")

    def with_flipped_edge(self, source: str, target: str) -> SignedGraph:
        """Return a new graph with the sign of ``source -> target`` reversed."""
        self.edge(source, target)  # raise KeyError early if absent
        edges = tuple(e.flipped() if e.key == (source, target) else e for e in self.edges)
        return self.model_copy(update={"edges": edges})

    def to_networkx(self) -> nx.DiGraph:
        """Build a ``nx.DiGraph``; edge attributes are ``sign``, ``passage``, ``citation``."""
        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.id, label=n.label)
        for e in self.edges:
            g.add_edge(e.source, e.target, sign=e.sign, passage=e.passage, citation=e.citation)
        return g

    @classmethod
    def from_networkx(cls, g: nx.DiGraph) -> SignedGraph:
        nodes = tuple(Node(id=str(n), label=d.get("label")) for n, d in g.nodes(data=True))
        edges = tuple(
            SignedEdge(
                source=str(u),
                target=str(v),
                sign=d["sign"],
                passage=d["passage"],
                citation=d.get("citation"),
            )
            for u, v, d in g.edges(data=True)
        )
        return cls(nodes=nodes, edges=edges)

    def edges_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [e.model_dump() for e in self.edges],
            columns=["source", "target", "sign", "passage", "citation"],
        )

    @classmethod
    def from_edges_dataframe(cls, df: pd.DataFrame, nodes: list[Node] | None = None) -> SignedGraph:
        """Build a graph from a dataframe of edges; missing nodes are created from ids."""
        records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
        edges = tuple(SignedEdge.model_validate(r) for r in records)
        node_list = list(nodes or [])
        known = {n.id for n in node_list}
        for e in edges:
            for end in e.key:
                if end not in known:
                    node_list.append(Node(id=end))
                    known.add(end)
        return cls(nodes=tuple(node_list), edges=edges)
