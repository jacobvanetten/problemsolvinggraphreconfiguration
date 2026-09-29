# Procedure: running the method design lab

Derived from `reframing_step_spec.md` v0.2. The code lives in `src/lab/`; this file is the operating procedure.

## 0. What you need

| For | You need |
|---|---|
| A mock dry run (tests the pipeline, toolkit and validators; content is templated) | Nothing. `uv sync` |
| A real run | `ANTHROPIC_API_KEY`, `backend: anthropic` in `config.yaml`, case A files, the `tricot-ranking` skill on disk |

Case A: `cases/A/archaeology.md`, `paradox.md`, `context.md`, optional `documents/`, optional `network.json|yaml|graphml`.
Case B (held out): same layout in `cases/B/`. Its files are fingerprinted at start; any prompt containing them before Phase 9 raises.

## 1. Commands

```bash
uv sync
uv run lab dry-run --dir dry_run_workspace      # full mock run on toy cases, ~5 min
uv run lab estimate --config config.yaml        # call estimate before a real run
uv run lab run --config config.yaml             # interactive: pauses at H1..H6
uv run lab resume --config config.yaml          # after creating state/H<n>.ok
uv run lab phase 4 --config config.yaml         # re-run one phase (skips finished work)
uv run lab status --config config.yaml          # phases done, calls, tokens
uv run pytest -m "not slow"                     # 2 minutes; drop the filter for the full mock pipeline tests
```

## 2. Phases

| # | Does | Key outputs | Checkpoint |
|---|---|---|---|
| 0 | Import or extract the case A exchange network; KG, SA, SY review; field map | `field/A/network.json`, `field_map.md` | H1 |
| 1 | Nine independent lens notes; merged synthesis | `lenses/` | |
| 2 | Pairing cycle from the distance matrix; 9 pair proposals (3 calls each) | `methods/M01..M09.md`, `.history.json` | H2 |
| 3 | Trial 1 by cross pair (pair i+4), PE observation, four-question AAR, first fix | `trials/T1/` | H3 |
| 4 | Three café rounds, ledger with host response to every item, revision per table per round, harvests, merge check, drift check | `cafe/` | |
| 5 | Trial 2 by cross pair and by reference team; AAR with comparison | `trials/T2/` | |
| 6 | Tricot study 1 (frames, blind), then study 2 (methods) with dossiers | `ranking/` | H4 |
| 7 | Selection, difference matrix, discussion, difference map; then purge raw judgements | `discussion/` | |
| 8 | Composite with modules, lineage, failed modules; objection round; dissent log | `integration/` | H5 |
| 9 | Unseal B; composite and top 2 methods on B; blind frame study; comparison | `heldout/` | |
| 10 | Final reflection; revised composite as v1; implementation notes; field questions | `reflection/`, `handoff/` | H6 |

State is files. A phase that is `done` under `runs/<run_id>/state/` is skipped. Inside a phase, finished trials, versions and ledger items are skipped on re-run.

## 3. Human checkpoints (interactive mode)

The run stops after the phase, writes `state/H<n>.waiting`. Edit files, then create `state/H<n>.ok` and `lab resume`.
Files whose JSON has `"source": "human"` (or Markdown starting with `<!-- source: human -->`) are never overwritten.

- H1 correct `field/A/network.json`. H2 add `methods/human/H01.yaml` (a `MethodProtocol`). H3 set `active: false` in `methods/registry.json` to stop a method.
- H4 write `ranking/methods/selected_override.json` (`{"selected": ["M02", ...]}`) to choose Phase 7 methods. H5 edit `integration/composite.json`. H6 accept.

## 4. Things to check in the outputs

1. `ranking/*/run_stats.json` `conflicts`: blocks where every core member had authored or executed a method in the block. Those judgements went to the member with fewest clashes.
2. `ranking/*/assessments.txt`: a study that stopped early with "flat at two consecutive assessments" says its criterion barely separated units.
3. `heldout/comparison.md`: read the negative results first.
4. `cafe/uptake.json` is descriptive. It is not used in any ranking.
5. `cafe/drift.json`: persona drift report. Report only.
6. Every claim about a human actor in `handoff/field_questions.md` is a hypothesis from a simulation.

## 5. Known deviations from the spec (decisions to confirm)

1. **Café constraints are infeasible as written** with 3 rounds and 2 core visits per table: each core member has 5 tables they neither authored nor executed but needs 6 visits. `cafe.executed_tables_eligible_from_round: 3` lets members visit tables they executed (never authored) in round 3 only. Set it to 0 for the strict rule; the scheduler then fails loudly with 3 rounds and works with 2.
2. **Judge exclusion** is impossible for some blocks (about a quarter to a third in mock runs). See section 4, item 1.
3. **Discipline distance matrix** (`config/discipline_distance.csv`) is my judgement, chosen so the spec's default cycle is optimal. Spec open question 1 applies.
4. `tricot.py feasibility` recommends `full_in_chat` on small studies. Both full modes are accepted; a digest mode stops the run.
5. Toy fixtures are illustrative and every edge is a hypothesis.
