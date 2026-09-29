"""Phase 0: field for a case (spec 7.1, 9). Also used for case B in Phase 9."""

from __future__ import annotations

from pydantic import BaseModel, Field

from lab.llm import LabError
from lab.run import Run
from lab.schemas import Edge, Network, Node
from lab.toolkit.build import check_size, load_network
from lab.toolkit.edits import apply_edits_raw
from lab.toolkit.render import field_map_md


class Extraction(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class EditProposal(BaseModel):
    edits: list[dict] = Field(default_factory=list)
    notes: str = ""


def chunks(text: str, size: int = 3000) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def import_or_extract(run: Run, case: str) -> Network:
    d = run.case_dir(case)
    for name in ("network.json", "network.yaml", "network.graphml"):
        if (d / name).exists():
            return load_network(d / name)
    docs = [f for f in sorted((d / "documents").glob("*")) if f.suffix in (".md", ".txt")] if (d / "documents").exists() else []
    texts = list(run.case_files(case).items()) + [(f.name, f.read_text()) for f in docs]
    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}
    for name, text in texts:
        for i, ch in enumerate(chunks(text)):
            if not ch.strip():
                continue
            ex = run.llm.call(task="extract_network", schema=Extraction, persona=run.persona("KG"),
                              context={"source": f"{name}#chunk{i}", "text": ch}).obj
            for n in ex.nodes:
                nodes.setdefault(n.id, n)
            for e in ex.edges:
                edges.setdefault(e.id, e)
    if not nodes:
        raise LabError(f"case {case}: no network file and extraction found no actors")
    return Network(nodes=list(nodes.values()), edges=[e for e in edges.values() if e.source in nodes and e.target in nodes])


def build_field(run: Run, case: str) -> Network:
    net = import_or_extract(run, case)
    for pid in ("KG", "SA", "SY"):  # provenance and duplicates, split aggregates, add biophysical actors
        prop = run.llm.call(task="network_review", schema=EditProposal, persona=run.persona(pid),
                            context={"case": case, "network": net.model_dump(mode="json", exclude={"themes", "motifs", "links"}),
                                     "role": {"KG": "provenance and duplicates", "SA": "split aggregated human actors",
                                              "SY": "add biophysical actors and feedbacks"}[pid]}).obj
        if prop.edits:
            try:
                net = apply_edits_raw(net, prop.edits)
            except Exception as exc:  # a bad proposal must not corrupt the network
                run.note(f"{pid} edits for case {case} rejected: {exc}")
        run.write_text(run.p("field", case, f"review_{pid}.md"), prop.notes + "\n")
    for n in check_size(net, run.cfg.network.max_nodes_warn):
        run.note(f"case {case}: {n}")
    run.write_json(run.p("field", case, "network.json"), net)
    run.write_text(run.p("field", case, "field_map.md"),
                   field_map_md(net, case, run.cfg.network.cycle_search_max_len, run.cfg.seed))
    return net


def run_phase(run: Run) -> bool:
    if not run.p("field", "A", "network.json").exists():
        build_field(run, "A")
    return run.checkpoint("H1", "Correct the case A network in field/A/network.json (mark source: human to protect edits)")
