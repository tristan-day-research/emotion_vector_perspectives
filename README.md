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
outputs/                  (empty) run outputs, one dir per run:
  <run_name>/
    activations/          large pooled activations (mirrored to R2, never pulled)
    results/              everything small + pullable by `pull-results`
      run_config_*.txt    provenance records
      directions/         directions.safetensors, layer_summary.csv
      evaluation/         metrics CSVs, summary.json, plots
run.py                    stage router for the shared scripts/ RunPod workflow
.runpod.env               pod config overrides for the shared scripts/ workflow
```

The `activations/` vs `results/` split is what lets `pull-results` fetch the
artefacts you want to look at while leaving 8 GiB on the pod.

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

## Running on RunPod via the shared `scripts/` workflow

This repo is driven by the generic, **project-agnostic** scripts in
`../scripts/` (`set_pod.sh`, `sync_up.sh`, `run_experiment.sh`,
`pull_results.sh`, exposed as `set-pod` / `sync-up` / `run-experiment` /
`pull-results`). They aren't specific to any one project — they sync
whatever directory you `cd` into, and run one fixed `RUN_ENTRYPOINT` with
your arguments appended. There are no per-project wrapper commands; the
same four commands work here and in every other project.

[`run.py`](run.py) is the entrypoint they call — a router that takes a **stage
name** first and forwards everything else verbatim, so the `--set field=value`
convention is unchanged:

```bash
run-experiment extract_activations --dry-run
run-experiment extract_activations --limit 256
run-experiment all --set stories_per_emotion=300     # all three stages
run-experiment r2 push outputs/<run>/activations --prefix runs/<run>/activations
```

The job runs in tmux on the pod, so it survives your laptop disconnecting.
`run-experiment --status` / `--attach` / `--stop` manage it.

One-time setup — this repo's [`.runpod.env`](.runpod.env) already carries the
settings below, checked in and shared by anyone who clones this repo:

```bash
cd ~/dev/code/emotion_vector_perspectives   # cd here first — every command
                                             # below acts on your cwd
set-pod ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

If your entrypoint needs a secret (e.g. `HF_TOKEN` for a gated model), put it
in `../scripts/r2.env` instead of anywhere in this repo — that file is
already forwarded to the pod and sourced before every run (see the R2
section below), so an `export HF_TOKEN=...` line there reaches this
project's job with no extra wiring.

### Settings in `.runpod.env` that are load-bearing

```bash
REQUIREMENTS_FILE="requirements.txt"
INSTALL_CMD="pip install -r $REQUIREMENTS_FILE"
RUN_ENTRYPOINT="HF_HOME=/workspace/hf_cache PYTHONUNBUFFERED=1 python run.py"
RESULTS_SUBDIR="outputs"
```

**`RESULTS_SUBDIR="outputs"`** — `pull-results` fetches exactly
`$RESULTS_SUBDIR/*/results/***` and excludes any `activations/` directory. Our runs
are `outputs/<run>/results/…` with `activations/` as a sibling, so this pulls all
33-odd directions/metrics/plots/run-records and leaves the 8 GiB behind. The
shared default `"experiments"` (this project doesn't have that folder) would
silently pull **nothing**.

**`RUN_ENTRYPOINT="HF_HOME=/workspace/hf_cache PYTHONUNBUFFERED=1 python run.py"`** —
`HF_HOME` must be set here, not in the pod's shell profile: the job is launched
through `tmux new-session` with a non-login shell that never reads `~/.bashrc`.
Without it, the 65 GiB Qwen checkpoint lands on the container's root disk and is
lost on every pod stop. (Verified that a `VAR=value`-prefixed entrypoint survives
the workflow's `printf %q` arg escaping.)

**R2 credentials come from `../scripts/r2.env`, but the bucket does not.**
`run-experiment` forwards that file to the pod and sources it, so mirroring needs
no RunPod-UI env-var step. It exports `R2_ENDPOINT` (`core/r2.py` accepts
`R2_ENDPOINT`, `R2_ENDPOINT_URL`, or `R2_ACCOUNT_ID`).

> **Do not change `R2_BUCKET` in `scripts/r2.env`.** That file is shared with
> `persona_introspection`, which reads `os.environ["R2_BUCKET"]` with no fallback and
> expects `persona-activations`; editing it there would silently redirect that
> project's uploads. Because `run_experiment.sh` sources `/tmp/r2.env` *before* running
> `RUN_ENTRYPOINT`, an assignment on the entrypoint wins for this project only —
> add it to this repo's `.runpod.env`:
>
> ```bash
> RUN_ENTRYPOINT="HF_HOME=/workspace/hf_cache R2_BUCKET=emotion-vector-perspectives PYTHONUNBUFFERED=1 python run.py"
> ```
>
> Shared credentials, per-project bucket. Your R2 API token must be scoped to
> **both** buckets (or be a second token) — one scoped only to `persona-activations`
> cannot write here.

For local use on your Mac, `source ../scripts/r2.env` then override just the bucket:
`R2_BUCKET=emotion-vector-perspectives python run.py r2 ls --prefix story-activations/`.

### Typical pod session

```bash
cd ~/dev/code/emotion_vector_perspectives
run-experiment extract_activations --dry-run    # confirm 8.18 GiB, no model download
run-experiment extract_activations --limit 256  # real throughput number
run-experiment all                              # full pipeline, resumable
pull-results                                    # directions + metrics + plots, no activations
```

Activations stay on the pod and go to R2. `pull-results
outputs/<run>/activations` overrides that if you really want them locally.

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

Bucket `emotion-vector-perspectives`; story activations land under
`story-activations/<run_name>/` (set by `config.r2_root` / `config.r2_prefix`).

Credentials live in `../scripts/r2.env` — one file, forwarded to the pod on every
run, so there is no RunPod-UI env-var step and it survives pod wipes.

```bash
python run.py r2 check                      # confirm bucket + endpoint
python run.py r2 ls --prefix story-activations/
python run.py r2 pull outputs/<run>/activations \
    --prefix story-activations/<run_name>    # bring a finished run back
```

### Activations live in R2, not on disk (default)

`delete_local_after_sync=true` is the **default**: each `.safetensors` chunk is
deleted locally as soon as its upload is *verified* (object present, size matches).
A finished run leaves ~8 GiB in R2 and only megabytes on the pod.

What stays local, deliberately:

* the per-chunk **index parquets** — tiny, and what lets a resumed run know which
  examples are done without querying R2
* **`manifest.json`** — without it the activations are uninterpretable

Nothing is ever deleted without a verified remote copy. If R2 is unconfigured or an
upload fails, local files are kept and the failure is reported; the final sweep and
any later `r2 push` retry it. A truncated upload is caught by the size check and the
local copy survives — covered by
[`tests/test_r2_only_flow.py`](tests/test_r2_only_flow.py).

Because stages 2 and 3 read activations from disk, a fresh machine needs a pull:

```bash
python run.py r2 pull outputs/<run>/activations --prefix story-activations/<run>
```

`run.py all` does this for you between extraction and direction fitting. If you run
a stage directly and the chunks are absent, you get a `MissingChunkError` that prints
the exact `r2 pull` command rather than a bare `FileNotFoundError`.

Set `delete_local_after_sync=false` to keep local copies too — the right choice when
the disk can hold the run and you want to refit repeatedly without downloading.

Set `HF_HOME=/workspace/hf_cache` on RunPod so a 65 GiB checkpoint lands on the
volume rather than the container's root disk.

### Sharing activations with collaborators

Pick by what the recipient needs. Bucket is private; nothing below makes it public.

**1. Read-only API token — the default for anyone running code.**
Cloudflare → R2 → *Manage R2 API Tokens* → *Create API Token*, permissions
**Object Read only**, scoped to **only** `emotion-vector-perspectives`. Issue **one
token per person** so you can revoke individually, and send it through a password
manager or Signal — never Slack, email, or a commit.

They then need no write access and cannot damage the run:

```bash
export R2_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com"
export R2_ACCESS_KEY_ID="<their-read-only-key>"
export R2_SECRET_ACCESS_KEY="<their-read-only-secret>"
export R2_BUCKET="emotion-vector-perspectives"

python run.py r2 ls --prefix story-activations/
python run.py r2 pull outputs/<run_name>/activations \
    --prefix story-activations/<run_name>
```

Then `compute_directions` / `evaluate_directions` run locally against the pulled
activations, or `ActivationStore` opens them in a notebook.

**2. Presigned URLs — for a handful of files, no account needed.**
Time-limited HTTPS links, max 7 days (R2's cap):

```bash
python run.py r2 share --prefix story-activations/<run_name>/results --expires-hours 72
```

Each URL is a **secret** — it embeds a signature from your key and grants read
access until it expires. Per-object, so it refuses to mint more than `--max-urls`
(default 50) and points you at option 1 for a whole 8 GiB run.

**3. Share the directions, not the activations — usually the right answer.**
`results/directions/directions.safetensors` is a few MB versus 8 GiB, and it is what
downstream analysis actually consumes. Attach it to a release, commit it to a
separate data repo, or presign it. Collaborators only need raw activations to refit
directions or train probes.

**4. A Cloudflare account invite** (R2 → *Manage* → member with R2 read) if a
teammate wants dashboard access. Heaviest option; only for someone co-administering
storage.

> **Reproducibility note.** Activations are only interpretable alongside their
> `manifest.json` (model sha, layers, pooling offset, dtype) and the run records
> under `results/`. `r2 pull` of a full run prefix brings the manifest along; if you
> hand over a subset, include the manifest or the recipient cannot verify what they
> have. `ActivationStore` refuses to open a directory without it.

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

## Pipeline validation (smoke test, not a result)

Before spending GPU hours on a 32B model, the whole three-stage pipeline was run on
CPU with `EleutherAI/pythia-160m` (12 layers, `hidden_size=768`), the default
10-emotion list, 120 stories per emotion and 300 neutral stories:

| | |
|---|---|
| top-1 accuracy, held-out validation topics | **0.746** (chance 0.100) |
| macro one-vs-rest AUROC | **0.939** |
| best layer | 6 of 12 (mid-network, as the paper reports) |
| bootstrap direction stability | mean cosine 0.947 |
| split-half agreement | mean cosine 0.716, min 0.370 |
| neutral PCs removed | 8–28 depending on layer |

The emotion×emotion cosine geometry is interpretable even at 160M parameters:
afraid/ashamed/sad group together, afraid↔proud is strongly negative, calm↔angry is
negative.

This is a check that the pipeline extracts real signal, **not** a research result —
160M parameters, a fraction of the stories, and no attempt at tuning. Split-half
agreement in particular is limited by having only ~35 topics per half; expect it to
rise with the full 1,200 stories per emotion.

Reproduce with:

```bash
python -m extract_emotion_vectors.extract_activations \
    --set model_name=EleutherAI/pythia-160m --set dtype=float32 \
    --set stories_per_emotion=120 --set neutral_stories=300 \
    --set run_name=pythia160m_10emo --set layer_spec=all --set device_map=None
# then compute_directions and evaluate_directions with the same --set flags
```

(`dtype=float32` because bf16 matmuls are roughly 30× slower on CPU.)

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
