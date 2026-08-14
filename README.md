# Emotion Vector Perspectives

A mechanistic-interpretability pipeline for extracting **emotion directions** from
the residual stream of a language model, reproducing the vector-construction
method from Anthropic's *Emotion Concepts and their Function in a Large Language
Model*, in a model-agnostic way.

This repository currently implements only the **baseline**: activation extraction
and emotion-direction construction, plus held-out validation. It is deliberately
structured so that the later experiments — whether emotion representations are
bound to the *experiencer* (default assistant / alternative first-person persona /
third-person character), trained linear probes, cross-condition probe transfer, and
causal steering — can be built on top without reworking these interfaces.

---

## Attribution and provenance

**What is ours and what is not** — please keep this distinction when citing or
extending this work.

| Component | Origin |
|---|---|
| Method: mean-difference emotion vectors, 50-token pooling offset, neutral-PC removal at 50% variance | **Anthropic** — Sofroniew et al., *Emotion Concepts and their Function in a Large Language Model* (2026) |
| Dataset: 205,200 emotional stories + 1,200 neutral stories (171 emotions × 100 topics × 12) | **Ryan Codrai** — [`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes) on Hugging Face, CC-BY-4.0, generated with Gemini 3.1 Pro Preview |
| Methodological reference for a concrete implementation of the above | **Ryan Codrai** — [`RyanCodrai/gemma-emotional-probes`](https://github.com/RyanCodrai/gemma-emotional-probes) |
| The 171-word emotion list in [data/emotions_171.txt](data/emotions_171.txt) | Transcribed from Anthropic's paper appendix |
| All code in this repository | **Ours**, written fresh |

### On the upstream repository

`RyanCodrai/gemma-emotional-probes` was used **as a methodological reference only**.
At the time of writing it carries **no open-source licence**, so none of its code was
copied, adapted, or vendored here. Every module in this repository is an independent
implementation written from the method description in Anthropic's paper and from the
upstream README's account of the procedure. Differences are intentional:

* model-agnostic (upstream targets Gemma 4 E4B; our default is Qwen2.5-32B and any
  `transformers` decoder-only checkpoint works)
* topic-level train/validation/test splitting, which upstream does not do
* padding-agnostic pooling, resumable sharded extraction, R2 mirroring,
  per-run provenance records, and held-out evaluation with stability checks

The **dataset** is a different matter: it is CC-BY-4.0, so we load it directly from
the Hub and credit it. We do not regenerate stories.

### Citations

```bibtex
@article{sofroniew2026emotions,
  title  = {Emotion Concepts and their Function in a Large Language Model},
  author = {Sofroniew, Nicholas and others},
  journal = {Transformer Circuits Thread},
  year   = {2026},
  url    = {https://transformer-circuits.pub/2026/emotions/index.html}
}

@misc{codrai2026emotionprobes,
  title  = {Emotion Probes Dataset},
  author = {Codrai, Ryan},
  year   = {2026},
  publisher = {Hugging Face},
  doi    = {10.57967/hf/8303},
  url    = {https://huggingface.co/datasets/ryancodrai/emotion-probes},
  note   = {CC-BY-4.0}
}

@misc{codrai2026gemmaprobes,
  title  = {gemma-emotional-probes},
  author = {Codrai, Ryan},
  year   = {2026},
  howpublished = {\url{https://github.com/RyanCodrai/gemma-emotional-probes}},
  note   = {Methodological reference; no licence at time of use, no code reused}
}
```

---

## Method

For each layer, and for each emotion, the pipeline computes a
**mean-difference direction with neutral-PC removal**:

1. **Centroid** — mean pooled residual-stream activation over that emotion's
   training stories.
2. **Global emotion mean** — mean across the *n* emotion centroids, giving each
   emotion equal weight (not each story, so unequal story counts cannot tilt it).
3. **Mean difference** — subtract the global mean from each centroid.
4. **Neutral PCA** — centre the neutral-story activations and take their SVD,
   separately at each layer.
5. **Threshold** — keep the fewest neutral PCs explaining ≥ 50% of neutral variance.
6. **Projection** — project each mean-difference vector off that subspace.
7. **Normalisation** — scale to unit L2 norm.

Pooling follows the paper: residual-stream activations are averaged over all real
token positions **after the first 50**, at which point the emotional content of a
story should be apparent.

> **Naming discipline.** These are `emotion_direction` /
> `mean_difference_direction` objects. They are **not trained probes** — nothing is
> fitted against labels, no objective is optimised. When logistic-regression probes
> are added later they live separately, and their numbers must never be presented
> as if they came from these directions. `directions_metadata.json` records
> `"is_trained_probe": false` for exactly this reason.

Both the projected direction *and* the unprojected mean-difference vector are
saved and evaluated, because the paper notes its qualitative findings hold either
way and that is worth verifying on a different model.

---

## Dataset (verified, not assumed)

Loaded through `datasets` from `ryancodrai/emotion-probes`. Structure confirmed by
inspection rather than taken from the README:

| File | Rows | Columns |
|---|---|---|
| `expression/stories.parquet` | 205,200 | `emotion`, `topic`, `story` |
| `expression/neutral_stories.parquet` | 1,200 | `topic`, `story` |

Validated facts:

* **171 emotions**, exactly matching Anthropic's appendix list (zero symmetric difference)
* **100 topics**
* **12 stories per (emotion, topic)** — uniformly, for all 17,100 pairs; 1,200 per emotion
* **1,200 neutral stories** = the *same* 100 topics × 12
* stories are 93–170 words (≈120–230 tokens), so `max_length=512` truncates nothing
  and no story is short enough to be dropped by the 50-token offset

The dataset also ships `deflection/` files (239,400 dialogues + 1,200 neutral
dialogues) for the paper's emotion-*deflection* vectors. This baseline does not use
them.

`--dry-run` re-validates all of this on every run and writes the result into the run
record, so a structural change upstream shows up as a warning rather than as silently
different numbers.

### Splitting

Splits are by **topic**, never by individual story. Twelve stories share one topic
and are near-paraphrases of one scenario; splitting by story would put variants of
the same scenario in both train and eval and inflate every metric. With the default
70/15/15 over 100 topics: **70 / 15 / 15 topics**. Because the neutral stories reuse
the same topics, one assignment partitions both tables consistently.

Directions used in held-out analyses are built from **training topics only**.
`full_dataset=true` builds them from everything, for the final artefact after
evaluation is complete — write those to a separate subdirectory so they can never be
mistaken for held-out results.

---

## Layout

```
analysis/                 (empty) notebooks for analysing results
core/                     code shared across the whole project
  activation_store.py     resumable chunked storage + per-layer streaming reads
  dataset.py              HF loading, validation, topic-level splitting
  directions.py           the direction maths + DirectionSet load/save interface
  model_utils.py          model-agnostic loading, layer resolution, text prep
  paths.py                canonical paths
  plotting.py             shared, validated plot palette
  pooling.py              offset-aware, padding-agnostic mean pooling
  provenance.py           environment capture, run records
  r2.py                   Cloudflare R2 (S3-compatible) mirroring
  seeds.py                deterministic seeding
data/                     dataset cache + the 171-emotion reference list (git-ignored except the list)
experiments/              (empty) mech-interp experiments
extract_emotion_vectors/
  vector_extraction_config.py   ← the config you edit
  extract_activations.py        stage 1
  compute_directions.py         stage 2
  evaluate_directions.py        stage 3
outputs/                  (empty) run outputs
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Validate config + dataset, estimate storage. Does not download the model.
python -m extract_emotion_vectors.extract_activations --dry-run

# 2. Benchmark on a GPU with a handful of examples first.
python -m extract_emotion_vectors.extract_activations --limit 256

# 3. Full extraction (resumable — just re-run if it dies).
python -m extract_emotion_vectors.extract_activations

# 4. Directions from the training split.
python -m extract_emotion_vectors.compute_directions

# 5. Held-out validation + plots.
python -m extract_emotion_vectors.evaluate_directions
```

Everything is driven by
[extract_emotion_vectors/vector_extraction_config.py](extract_emotion_vectors/vector_extraction_config.py).
Override any field without editing the file:

```bash
python -m extract_emotion_vectors.extract_activations \
    --set emotions=joyful,sad,angry,calm \
    --set stories_per_emotion=300 \
    --set layer_spec=evenly_spaced:14 \
    --set batch_size=8
```

### The config

The main knobs, all in `VectorExtractionConfig`:

* `emotions` — which of the 171 to extract (default: 10 spanning the affective
  circumplex and several of the paper's clusters)
* `stories_per_emotion` — topic-stratified subsample; 1200 = all
* `layer_spec` — `"all"`, `"blocks"`, `"evenly_spaced:14"`, `"range:20:50:2"`, or an
  explicit list
* `model_name`, `model_revision`, `dtype`, `quantization` (`None`/`"4bit"`/`"8bit"`)
* `max_length`, `token_offset`, `min_pooled_tokens`, `use_chat_template`
* `batch_size`, `activation_dtype`, `chunk_size`
* `neutral_variance_threshold` — the 50% neutral-PC threshold
* `split_seed`, `split_proportions`, `full_dataset`
* `r2_sync`, `r2_threshold_gib`, `delete_local_after_sync`
* `eval_splits`, `eval_bootstrap_n`, `eval_layer_spec`

**On quantisation:** it is off by default and should stay off. 4/8-bit weights
perturb the residual stream we are trying to measure. Qwen2.5-32B in bf16 is ~65 GiB
of weights and fits one 80 GiB A100/H100 unquantised. Never compare directions
across quantisation settings.

---

## Storage and RunPod

Every run's activations are kept, since re-running a 32B model to recover a layer
you skipped costs far more than the disk.

Per pooled activation: `n_layers × hidden_size × 2` bytes (bf16). For Qwen2.5-32B
(65 hidden states, `hidden_size=5120`) at the default 10 emotions × 1,200 stories +
1,200 neutral = 13,200 examples:

```
65 × 5120 × 2 B  =  650 KiB / example
13,200 examples  ≈  8.2 GiB
```

That exceeds a comfortable laptop footprint, so **Cloudflare R2 mirroring is
built in**. `r2_sync="auto"` (the default) uploads each chunk as it is written when
credentials exist *and* the estimated size passes `r2_threshold_gib` (5 GiB) — and
tells you plainly which branch it took. `--dry-run` prints the exact estimate before
you commit any GPU time.

```bash
cp .env.example .env      # fill in R2_ACCOUNT_ID / keys / bucket
python -m core.r2 check
python -m core.r2 push outputs/<run>/activations --prefix runs/<run>/activations
python -m core.r2 pull  outputs/<run>/activations --prefix runs/<run>/activations
```

Set `delete_local_after_sync=true` when the pod disk cannot hold the whole run; the
small index parquets stay behind so resume still works, and you `pull` before fitting
directions.

Set `HF_HOME=/workspace/hf_cache` on RunPod so a 65 GiB checkpoint lands on the
volume rather than the container's root disk.

### Multi-GPU

Shards are independent processes over a deterministic round-robin partition, so
every shard sees a balanced mix of emotions, topics and splits:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m extract_emotion_vectors.extract_activations \
      --num-shards 4 --shard-index $i --set device_map=None &
done
wait
```

Each shard writes its own `shardNNN/` directory; stage 2 reads all of them.

---

## Evaluation

`evaluate_directions.py` reports, per layer:

* **top-1 emotion classification** on held-out topics, against chance `1/n_emotions`
* **one-vs-rest AUROC** per emotion and macro-averaged, skipped where degenerate
* **bootstrap stability** — resample training topics with replacement, refit
  centroids, cosine to the reference direction (the neutral subspace is held fixed
  across resamples; that is a deliberate cost/benefit choice, recorded in
  `summary.json`)
* **split-half agreement** — two disjoint halves of the training topics, each given
  a fully independent refit including its own neutral subspace

Scores are reported under three rules — raw `dot`, `centered_dot` (after subtracting
the layer's global emotion mean; the rule matched to how the directions were built,
and the headline number), and `cosine` (as the paper plots) — and for both the
projected and unprojected vectors.

Outputs: `metrics_by_layer.csv`, `metrics_by_emotion.csv`, `stability_bootstrap.csv`,
`stability_split_half.csv`, `confusion_*.csv`, `emotion_cosine_layer*.csv`,
`summary.json`, and four plots.

---

## Reproducibility

Every stage writes `run_config*.txt` (human-readable) and `run_manifest*.json`
(machine-readable) into its output directory, containing the full resolved config,
resolved layer indices, model and dataset commit shas, dataset validation counts,
topic lists per split, token-length statistics, package versions, GPU model, git
commit and dirty state, and the exact command line. A run is reproducible from its
own output directory alone.

**Incompatible runs abort rather than mix.** The activations directory carries a
fingerprint of everything that changes what a stored vector *means* — model + sha,
dtype, quantisation, layers, pooling offset, max length, chat-template settings,
dataset sha, split seed. Re-running with any of those changed raises
`IncompatibleRunError` with a field-by-field diff; `--overwrite` is the explicit
opt-in.

Deliberately *not* in the fingerprint: `emotions`, `stories_per_emotion`,
`neutral_stories`. Those select which examples to extract, not what a vector means,
so you can add emotions to the config and re-run to **extend** the same run.

**Caveats.** Seeds are set for python/numpy/torch and all structural randomness
(splits, subsampling, bootstraps) uses explicitly derived generators, so splits are
bit-identical across machines. Activations are not bit-identical across different
GPU models, batch sizes, or attention kernels — reduction order changes. Pooling
reduces in fp32 to keep this well below the level that matters for directions.

**Skipped examples are recorded, never silently dropped:** `shardNNN/skipped.jsonl`
holds the example id, token counts, and the reason.

---

## Design notes worth knowing

**Padding.** `hidden[:, 50:seq_len]` is only correct for right-padded batches. We
rank tokens by position *among real tokens* via the cumulative sum of the attention
mask, which is correct for left, right, or any padding. `python -m core.pooling`
runs a self-test asserting left/right equivalence against an explicit reference.

**Memory.** Activations are pooled **on-device**; only `(batch × hidden)` per
selected layer ever moves to CPU. Token-level activations for all layers are never
transferred.

**Storage layout.** One tensor per layer per chunk, not one `(n, layers, hidden)`
blob — both direction fitting and evaluation iterate layer-by-layer over the whole
dataset, so this lets them stream one layer at a time.

**Crash safety.** Tensors are written first, the index parquet second, both by
atomic rename. A chunk without its index is an incomplete write and is discarded on
resume, so a killed pod never corrupts a run. Resume is by example id, so it
survives changes to shard count or ordering.

---

## What comes next (not implemented)

The interfaces exist for these; the experiments do not.

1. **Experiencer binding** — does the emotion representation depend on *who* is
   feeling it: the default assistant, an alternative first-person persona, or a
   third-person character? `use_chat_template` / `chat_role` / `text_prefix` /
   `text_suffix` are the hooks, and they are part of the activation fingerprint, so
   each condition is forced into its own run rather than being silently pooled with
   another.
2. **Trained probes** — logistic regression on the same pooled activations, kept
   strictly separate from these fixed directions.
3. **Cross-condition transfer** — fit in one experiencer condition, evaluate in
   another. `DirectionSet.score` already takes arbitrary activations.
4. **Causal steering** — add `α · direction` at a layer and measure the behavioural
   effect. `DirectionSet.direction(emotion, layer)` returns the unit vector.
