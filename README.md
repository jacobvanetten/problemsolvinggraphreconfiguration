# problemsolvinggraphreconfiguration
Explore problem solving as graph reconfiguration combining LLM and knowledge graphs

## Quick start

```bash
uv sync
uv run pytest
```

```python
from signedgraph import toy_graph, path_signs, net_effect, cycles, flip_report
from signedgraph.viz import write_html

g = toy_graph()
paths = path_signs(g, "rainfall", "hunger")      # each path with sign and passages
net_effect(paths)                                 # NetEffect.MIXED
[(c.nodes, c.polarity) for c in cycles(g)]        # reinforcing / balancing loops
print(flip_report(g, "food_price", "hunger").summary())
write_html(g, "toy.html")                         # interactive pyvis view
```

See `CLAUDE.md` for schema and analysis conventions.

## Method design lab

`src/lab/` is a simulated team that designs, trials, ranks and integrates reframing methods. See `docs/lab_procedure.md`.

```bash
uv run lab dry-run --dir dry_run_workspace   # full mock run on toy cases (templated content; tests the machinery)
uv run lab estimate --config config.yaml     # call estimate for a real run
```
