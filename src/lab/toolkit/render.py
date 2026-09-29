"""Human-readable renderings (spec 7.1: the facilitator's field map)."""

from __future__ import annotations

from lab.schemas import Network
from lab.toolkit.metrics import global_metrics, most_central, most_constrained
from lab.toolkit.motifs import list_motifs


def field_map_md(net: Network, case: str, cycle_max_len: int = 6, seed: int = 1) -> str:
    g = global_metrics(net, cycle_max_len, seed)
    label = {n.id: n.label for n in net.nodes}
    L = [f"# Field map, case {case}", "",
         f"{g['nodes']} actors, {g['edges']} exchanges. Evidence status: see edge `evidence` fields; "
         "unsourced additions are hypotheses.", "", "## Global properties",
         f"- density {g['density']}, components {g['components']}, communities {len(g['communities'])}, "
         f"degree centralisation {g['degree_centralisation']}",
         f"- reciprocity: direct {g['reciprocity_direct']}, generalised (cycles 3 to {cycle_max_len}) {g['reciprocity_generalised']}",
         "- structural balance by perspective: " + (", ".join(
             f"{k}: {v['balanced_fraction']} ({v['triangles']} triangles)" for k, v in g["structural_balance"].items()) or "none"),
         "", "## Most central actors"] + [f"- {label[n]} ({n}), degree {d}" for n, d in most_central(net)]
    L += ["", "## Most constrained actors (Burt)"] + [f"- {label[n]} ({n}), constraint {c}" for n, c in most_constrained(net)]
    L += ["", "## Flow bottlenecks"] + [f"- {n}: betweenness {b}" for n, b in g["bottlenecks"]]
    ms = list_motifs(net, cycle_max_len)
    for pat, title in [("open_reciprocity_loop", "Open reciprocity loops"), ("indirect_reciprocity_cycle", "Exchange cycles"),
                       ("valency_conflict", "Conflicting valencies"), ("anomaly", "Anomalies"), ("lumped_actor", "Lumped actors")]:
        L += ["", f"## {title}"] + ([f"- {m.id}: {m.label}" for m in ms if m.pattern == pat] or ["- none"])
    L += ["", "## Actors no one speaks for"] + ([f"- {label[a]} ({a})" for a in g["unspoken_actors"]] or ["- none"])
    return "\n".join(L) + "\n"
