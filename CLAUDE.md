# CLAUDE.md

Signed directed graphs of causal claims, each claim traceable to the text passage it
came from. Used to explore problem solving as graph reconfiguration: what changes in
the system's behaviour when one claim (edge sign) is reversed.

## Tooling

- Python >= 3.12, managed with **uv**. Never use `pip install` directly.
  - `uv sync` — install deps into `.venv`
  - `uv run pytest` — run tests (must pass before committing)
  - `uv add <pkg>` / `uv add --dev <pkg>` — add dependencies (updates `uv.lock`; commit it)
- Runtime deps: networkx, pydantic, pyvis, pandas, numpy, scipy, pyyaml, jinja2, anthropic. Dev: pytest.
- Two packages in `src/` (src layout): `signedgraph` and `lab`; tests in `tests/`.
- `uv run pytest -m "not slow"` is the fast suite (~30 s); plain `uv run pytest` also runs the full mock pipeline (minutes).

## Layout

| Module | Purpose |
|---|---|
| `schema.py` | Pydantic models `Node`, `SignedEdge`, `SignedGraph`; networkx and pandas conversion |
| `analysis.py` | `path_signs`, `net_effect`, `cycles`, `cycles_dataframe`, `flip_report` |
| `viz.py` | pyvis rendering (`to_pyvis`, `write_html`) |
| `toy.py` | `toy_graph()` — the small reference graph; `seeded_toy_graph()` — the same plus an improved seed |

## Schema conventions

- **`SignedGraph` is the source of truth.** It is an immutable pydantic model; networkx
  graphs are derived views (`to_networkx()`), built on demand and never stored.
- **Edges**: `SignedEdge(source, target, sign, passage, citation=None)`.
  - `source` / `target` are **node ids** (graph endpoints), not bibliographic sources.
  - `sign` is exactly `+1` or `-1` (`Sign = Literal[1, -1]`). `+1`: more source → more
    target. `-1`: more source → less target. No zero, no weights.
  - `passage` is required and non-empty: the verbatim text the claim was taken from.
    Never invent or paraphrase passages; bibliographic info goes in `citation`.
- **Graph invariants** (enforced by a validator): unique node ids, every edge endpoint is
  a declared node, at most one edge per ordered pair, no self-loops. Contradictory claims
  about the same pair must be resolved before building the graph, not stored as two edges.
- **Immutability**: all models are `frozen=True`. To change a graph, build a new one
  (e.g. `graph.with_flipped_edge(u, v)`, `graph.model_copy(update=...)`).
- In networkx views, edge attributes are `sign`, `passage`, `citation`; node attribute is `label`.
- Node ids are `snake_case` strings; `label` is optional human-readable text.

## Analysis conventions

- Sign of a path or cycle = **product** of its edge signs.
- Only **simple** paths and cycles are enumerated (networkx `all_simple_paths`,
  `simple_cycles`). This is exponential in the worst case — fine for tens of nodes.
  Use `cutoff` on `path_signs` for bigger graphs; do not silently add caching or sampling.
- `NetEffect` of A on B across all paths: `positive` (all +), `negative` (all −),
  `mixed` (both — ambiguous, don't collapse it into a sum), `none` (no path).
- Cycle polarity: sign +1 → `reinforcing`, sign −1 → `balancing`.
- Cycles are reported in canonical rotation (start at the lexicographically smallest
  node id, first node not repeated at the end).
- All list outputs are deterministically sorted (length, then node tuple) so tests can
  compare exact lists.
- `flip_report(graph, u, v)` reports: every cycle through the edge (each flips polarity),
  and every node pair whose `NetEffect` changes. Pairs that stay `mixed` are omitted even
  if some of their paths flipped. It also exposes `summary()` and
  `effect_changes_dataframe()`.
- Results are pydantic models; tabular views are pandas DataFrames (`*_dataframe`).

## Visualisation

`viz.to_pyvis`: green solid edges = +1, red dashed edges = −1, edge hover shows the
passage (and citation). Generated `.html` files are git-ignored.

## Tests

- Tests use `toy_graph()`; its structure is documented in `toy.py`. It has one
  reinforcing loop (via hunger/labour), one balancing loop (via farm income/fertiliser),
  and a mixed effect of `rainfall` on `hunger`. If you change it, update the expected
  values in tests deliberately rather than regenerating them from output.
- Every new analysis function gets tests with hand-checked expected values on the toy
  graph plus edge cases (no path, unknown node, acyclic graph).

## Method design lab (`src/lab/`)

Simulated multidisciplinary team designs, trials, ranks and integrates reframing methods (spec: `reframing_step_spec.md` v0.2).
Operating procedure: `docs/lab_procedure.md`. Entry point: `uv run lab ...` (`dry-run`, `run`, `resume`, `phase`, `status`, `estimate`).

- Independent of `signedgraph`: it uses its own exchange-network schema (`lab/schemas.py`, `lab/toolkit/`).
- Every LLM call goes through `lab/llm.py` (`LLM.call`): persona, task, context bundle, schema; JSON validated with pydantic,
  one retry with the error, then `LabError`. `backend: mock` (`lab/mock.py`) is deterministic and templated: it exercises tool use
  against the real network, the validators and the ranking bridge, but its content and its random judgements carry no information.
- State is files under `runs/<run_id>/`; phases are `lab/phases/p0..p10`. A finished phase writes `state/p<n>.done`.
  Files marked `source: human` are never overwritten (`Run.write_text`).
- Ranking calls the `tricot-ranking` skill CLI (`ranking/tricot_bridge.py`); it is never reimplemented. Raw judgements are purged only after Phase 7.
- Case B is sealed (`cases/seal.py`) until Phase 9: the adapter refuses any prompt containing its text.
- Operators compile to edit lists and go through `apply_edits`, so a named operator and its hand-written edits give the same report.
- Toy fixtures (`tests/fixtures/`, `lab/data/toy_cases/`) are illustrative; all their edges are `hypothesis`. Changing them means updating hand-checked test values.
