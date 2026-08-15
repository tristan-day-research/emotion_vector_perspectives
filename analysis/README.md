# `analysis/`

Notebooks for analysing run results. Intentionally empty for now.

Notebooks are not the pipeline — they read what the pipeline wrote. Keep fitting
logic in `core/` so a notebook and a script cannot disagree about what a direction
is.

Loading a finished run:

```python
from core.activation_store import ActivationStore
from core.directions import DirectionSet
from core import plotting

run = "outputs/qwen2.5-32b_10emo_all-layers"

directions = DirectionSet.load(f"{run}/results/directions")
store = ActivationStore(f"{run}/activations")

directions.emotions          # emotion order for every matrix
directions.layer_indices     # layers available
v = directions.direction("angry", layer=32)     # (hidden,), unit norm

# Held-out activations for one split, one layer
rows = store.subset(source="emotion", splits=["test"])
acts = store.load_layer(32, rows)                # (n, hidden) float32
scores = directions.score(acts, layer=32, mode="centered_dot")   # (n, n_emotions)

plotting.apply_style()       # same validated palette as the pipeline plots
```

`store.index` is the full per-example index (example id, emotion, topic, split,
token counts, chunk location). `store.skipped()` returns anything excluded during
extraction, with reasons.

Reach for `directions.matrix(layer, "direction_unprojected")` when you want to check
whether a result depends on the neutral-PC removal step.
