# Emotion Vector Perspectives

Mechanistic-interpretability pipelines for **emotion representations** in the
residual stream of a language model. Two experiments share one set of primitives
(`core/`): model-agnostic loading, offset-aware pooling, chunked resumable
activation storage, R2 mirroring, provenance records.

| Experiment | Question | Model | Status |
|---|---|---|---|
| **1. Emotion directions** (`extract_emotion_vectors/`) | Can a mean-difference direction per emotion be recovered, and does it survive held-out topics? | Qwen2.5-32B | baseline implemented |
| **2. Emotion-space PCA + J-lens** (`emotion_pca_jlens/`) | Do the *principal axes* of emotion space recover the affective circumplex (valence × arousal), and what does each axis read out as? | Qwen3-32B | Phase 0 implemented (gated) |

Experiment 1 reproduces the vector-construction method from Anthropic's *Emotion
Concepts and their Function in a Large Language Model* in a model-agnostic way:
activation extraction, direction construction, held-out validation. It is
structured so later work — experiencer binding (default assistant / alternative
first-person persona / third-person character), trained linear probes,
cross-condition transfer, causal steering — can be built on top without reworking
these interfaces.

Experiment 2 asks a different question of the same representations. Instead of one
direction per emotion, it takes one vector per emotion, runs PCA *across*
emotions, and interprets the top principal components with the **Jacobian lens** —
an interpretability tool that reads which vocabulary tokens a residual-stream
direction is disposed to make the model say. Its later phases then split each
emotion vector into the part the model can verbalise and the ~90% remainder it
cannot, and ask which of the two actually moves the model's behaviour — the point
being that welfare assessment built on self-report would miss anything driven by the
remainder. It runs one phase at a time, each ending at a gate meant to be read by a
human before the next runs. See
[Experiment 2](#experiment-2-emotion-space-pca-and-the-jacobian-lens-qwen3-32b).

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
| Method + reference implementation: the **Jacobian lens** (Experiment 2) | **Anthropic** — [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), Apache-2.0, companion code for *Verbalizable Representations Form a Global Workspace in Language Models* (2026) |
| Pre-fitted J-lens tensors (`J_l` per layer, per model) | **Neuronpedia** — [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) on Hugging Face, MIT |
| Phase 0 gate prompts in [emotion_pca_jlens/gate_prompts.py](emotion_pca_jlens/gate_prompts.py) | **Anthropic** — vendored verbatim from `data/experiments/probe-swap.json` and `data/evaluations/lens-eval-association.json` in the above repo, Apache-2.0 |
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
  jlens_lens.py           Jacobian-lens resolve/download/inspect + readout (exp 2)
  model_utils.py          model-agnostic loading, layer resolution, text prep
  paths.py                canonical paths
  plotting.py             shared, validated plot palette
  pooling.py              offset-aware, padding-agnostic mean pooling
  provenance.py           environment capture, run records
  r2.py                   Cloudflare R2 (S3-compatible) mirroring
  seeds.py                deterministic seeding
data/                     dataset cache + the 171-emotion reference list (git-ignored except the list)
experiments/              (empty) mech-interp experiments
extract_emotion_vectors/  EXPERIMENT 1 — emotion directions
  vector_extraction_config.py   ← the config you edit
  extract_activations.py        stage 1
  compute_directions.py         stage 2
  evaluate_directions.py        stage 3
emotion_pca_jlens/        EXPERIMENT 2 — emotion-space PCA + J-lens
  pca_jlens_config.py           ← the config you edit
  gate_prompts.py               Phase 0 gate prompts (vendored from Anthropic)
  phase0_lens_gate.py           phase 0: load + verify the lens (GATE)
  phase1_stimuli.py             phase 1: circumplex stimulus set (GATE)
outputs/                  (empty) run outputs, one dir per run:
  <run_name>/
    activations/          large pooled activations (mirrored to R2, never pulled)
    results/              everything small + pullable by `pull-results`
      run_config_*.txt    provenance records
      directions/         directions.safetensors, layer_summary.csv
      evaluation/         metrics CSVs, summary.json, plots
      phases/             experiment 2: one record per phase gate
run.py                    stage router for the shared scripts/ RunPod workflow
.runpod.env               pod config overrides for the shared scripts/ workflow
```

The `activations/` vs `results/` split is what lets `pull-results` fetch the
artefacts you want to look at while leaving 8 GiB on the pod.

---

## Quick start (Experiment 1)

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

### The config (Experiment 1)

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

## Experiment 2: emotion-space PCA and the Jacobian lens (Qwen3-32B)

**Question.** Experiment 1 asks what direction each emotion occupies. This asks
what the *axes* of emotion space are. Take one vector per emotion, mean-centre
across emotions, run PCA, and interpret the top principal components with the
Jacobian lens.

**Prediction.** After mean-centring, PC1/PC2 should read out as **valence**
(good/bad) and **arousal** (calm/intense) — the affective circumplex.

### Why the Jacobian lens

The J-lens reads out what a residual-stream vector is disposed to make the model
*say*. It linearly transports the vector at layer `l` into the final-layer basis
and decodes it with the model's own unembedding:

```
lens_l(h) = unembed(J_l @ h),   J_l = E[∂h_final / ∂h_l]
```

We do not fit `J_l` ourselves. Pre-fitted tensors come from
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens);
the readout code path is Anthropic's own
[`jlens`](https://github.com/anthropics/jacobian-lens), pinned in
`requirements.txt` to the commit its conventions were read from.

### Conventions verified, not assumed

Three things about the lens are easy to get silently wrong. All three were read
out of the reference implementation's source and tests rather than inferred, and
are recorded in [core/jlens_lens.py](core/jlens_lens.py):

1. **What `unembed` is.** It is `lm_head(final_norm(h))` — the model's *own*
   final norm with its learned weight, not an ad-hoc normalisation — followed by
   `final_logit_softcapping` where the config has one (Gemma-2 does; Qwen3 does
   not). Softcapping is monotonic, so it cannot reorder top-k.
2. **Layer indexing is by residual block, not hidden state.** `jlens` hooks
   `model.layers[l]`, so `l` means *the output of block `l`*. This repo's
   `core.model_utils` / `core.pooling` index the `output_hidden_states` tuple,
   where `0` is the embedding output. They differ by one:
   `hidden_state_index = block_index + 1`. Use
   `core.jlens_lens.hidden_state_index()` rather than writing `+1` inline.
3. **The final block has no fitted `J`.** `source_layers = 0 … n_layers-2`:
   `J_l` maps *output of block l* to *output of the last block*, so the last
   block is the transport target, not a source. The library's own
   `tests/test_fitting.py` pins this (`source_layers == [0,1,2]` for a 4-layer
   model, `J_2 == I + W_3`). Confirmed independently by file arithmetic: the
   Qwen3-32B lens is `5120² × 4 bytes × 63 = 6,606,028,800`, matching the actual
   6,606,048,498-byte file — fp32, blocks 0–62.

   **Consequence:** the "J ≈ identity at late layers" check must run at block
   **62**, not 63. At 63 there is no `J`.

A fourth property, used by the PC readout: the transport is linear and
`final_norm` is an RMSNorm, so the readout of a *bare* direction is
scale-invariant. `+PC` and `−PC` are therefore well defined without choosing a
step size. Phase 0 verifies this numerically rather than trusting the algebra.

### Model choice

`Qwen/Qwen3-32B` is Qwen3's **post-trained (instruct)** release. The base model
is the separate `Qwen/Qwen3-32B-Base`, which has no published lens. A Jacobian is
a function of the weights, so a lens cannot be moved between the two —
`resolve_lens_artifact` verifies the fit's recorded `hf_model_name` and **refuses**
a mismatch rather than producing plausible nonsense. Quantisation is rejected for
the same reason: it perturbs the residual stream the lens was fitted against.

### Phases, each ending at a gate

Every phase stops, prints a diagnostic, and waits for a human to read it. Nothing
runs end-to-end. The point of the gates is that each phase can invalidate
everything after it, and all four failures are quiet ones — a lens off by one
layer, a stimulus set that cannot express an arousal axis, vectors too noisy to
have a covariance structure, or a PC that is real but not lexicalised.

| Phase | One line | Collects activations? | GPU? | Runtime | Status |
|---|---|---|---|---|---|
| **0** | Load and verify the J-lens | No | yes | ~20 min / ~5 min cached | implemented |
| **1** | Choose and assemble the stimuli | No | **no** | 2–5 min | implemented |
| **2** | One residual vector per emotion | **Yes → R2** | yes | 10–25 min | not written |
| **3** | PCA across emotions | No | **no** | seconds | not written |
| **4** | J-lens the principal components | No | yes | ~5 min | not written |
| 5 | Optional structural extensions | varies | varies | varies | not written |
| **6** | Split each vector into reportable + remainder | No | yes | minutes | not written |
| **7** | Build two measurement channels | No | yes | ~1 h + review | not written |
| **8** | Steer under 4 conditions, measure both channels | No | yes | 2–4 h | not written |
| 9 | The re-entry clamp (decisive control) | No | yes | ~1 h | not written |

The shape of it: **0 validates the instrument, 1 the stimuli, 2 the measurement,
3 is the structural result and 4 its interpretation; then 6 decomposes, 7 builds the
behavioural readout, 8 is the functional result and 9 the decisive control.** Phases
3–4 map the *reportable* structure of emotion; 8–9 ask whether that structure is
what actually moves the model — see
[Phases 6–9](#phases-69-from-structure-to-function). Only Phase 2 ever extracts
activations; everything after it reuses those vectors and the Phase 0 lens.

**Phase 0 — load and verify the J-lens.** `python run.py phase0`
Resolves which pre-fitted lens belongs to this checkpoint and refuses a mismatch;
downloads it (6.6 GiB) and reports its real tensor shapes, dtypes and fitted layer
range; cross-checks those against the model config; then confirms the readout
formula from the loaded objects rather than from documentation. Two gates follow:
**GATE A** checks that an implied concept the prompt never names surfaces at mid
layers, on prompts whose expected readout Anthropic published; **GATE B** checks
that at the top of the stack the lens collapses towards the plain logit lens and
the model's real next token. Also runs an R2 round-trip so Phase 2 cannot discover
a broken bucket after spending GPU time. *Collects no activations — it reads out 12
short prompts and writes a few KB of run record.* If this fails, the lens is loaded
wrong and nothing downstream means anything.

**Phase 1 — choose and assemble the stimuli.** `python run.py phase1`
Prints the emotion set grouped by circumplex quadrant **first**, before touching
any data, because coverage is the thing worth disagreeing with and it costs nothing
to check. Then selects the matching stories, verifies on the assembled table that
every emotion really is written about the same topics in the same proportions, and
checks that stimulus lengths clear the pooling offset. Gate output is a few
examples per emotion plus per-emotion counts. Writes
`results/phases/phase1_stimuli.parquet` with `[emotion, quadrant, text]` plus
provenance columns. *No model, no GPU.* The failure this catches: a set with no
low-arousal emotions can never show an arousal axis, no matter what the model does.

**Phase 2 — one residual vector per emotion.** *(not written)*
Runs each stimulus through the model and mean-pools the post-MLP residual stream at
a target block in the middle third, excluding the first 50 tokens, then averages
across all stimuli of an emotion. Gate is **split-half reliability**: refit each
emotion's vector from two disjoint halves of its *topics* and report the cosine
between them. Above ~0.9 the vector is trustworthy; around 0.6 means more stimuli
are needed before PCA can mean anything. Halving by topic rather than by story
matters — twelve stories share one scenario, so a story-level split would leak it
across halves and inflate the cosine. *This is the only phase that collects
activations; they mirror to R2 as they are written.*

**Phase 3 — PCA across emotions.** *(not written)*
Stacks the emotion vectors into a matrix (rows = emotions), **mean-centres across
emotions**, and runs PCA. The centring is the load-bearing step: without it PC1 is
just overall affect magnitude, which is trivially true and tells you nothing. Gate
is the variance-explained table plus the headline PC1–PC2 scatter, labelled by
emotion and coloured by quadrant — the point being to eyeball whether the
circumplex is visually there before spending anything on interpreting it. *No GPU;
runs in seconds on the saved vectors.*

**Phase 4 — J-lens the principal components.**
Each PC is a unit direction in residual space, so it can be fed straight to the
lens. Reads out both the `+PC` and `−PC` ends — an axis has two, and valence should
read pleasant one way and unpleasant the other. Output is a table of PC index,
variance explained, and top-k tokens per end. Expect PC3 onward to get murky, and
the gate reports that honestly rather than straining to interpret noise.

The phase runs as **two sequential gates**, and A must pass before B means
anything:

* **GATE A — is the instrument working?** A murky readout has two completely
  different causes, and the tokens alone cannot tell them apart: either the PC is
  real but *not lexicalised* (no single vocabulary token for it — a fact about the
  vocabulary), or the *lens is too weak* to verbalise anything at this block (in
  which case the readout says nothing either way). That second branch is live, not
  hypothetical: the published `qwen3-32b` lens is an interrupted fit, 80 prompts,
  stopping at `mean_rel_change` 0.026 against its own 0.002 threshold. So before
  reading a single PC, the lens is pointed at 16 directions whose answer is already
  known — each fitted emotion's own vector — and the rank of *its own word* is
  recorded. Ground truth nobody chose: the words come from the dataset and the
  Phase 1 design. If the lens cannot find "sad" in the sad vector, nothing in
  GATE B is evidence about anything. The hand-built a priori valence/arousal axes
  from Phase 3 are read out in the same block as a second known-answer control.
* **GATE B — what do the PCs say?** Top-k tokens per end, scored rather than
  eyeballed. The 16 anchor words are used as **probes**: the lens scores all
  ~152k vocabulary tokens, and you look only at where the anchors landed, then ask
  whether the 8 pleasant ones outrank the 8 unpleasant ones (likewise activated vs
  deactivated). That ordering is summarised as an AUROC with a permutation
  p-value — necessary because with 8 words a side a chance AUROC has a standard
  error near 0.15, so a bare 0.75 threshold would mint lexicalised axes out of
  noise. Each PC also carries its Phase 3 split-half stability and label
  correlations, so a murky readout can be attributed rather than puzzled over.

Reading the GATE B table:

| Field | Plain reading |
|---|---|
| `best_axis_strength` | AUROC — how cleanly the axis sorts the probes. 0.5 is chance; with 8 a side, ignore anything under ~0.85. |
| `p_best_axis` / `alpha` | Permutation null. `alpha` is Bonferroni-corrected: 0.05 for the pre-registered PC1–PC2, 0.0083 for PC3+, which are stamped `exploratory: True` and are leads, not findings. |
| `sign_agrees_with_phase3` | The cross-check neither phase can do alone. Phase 3 measures *where emotions sit* (geometry, no language); Phase 4 measures *what the axis says* (language, no geometry). If pleasant emotions have negative PC scores, the `−` end must read pleasant. Nothing forces this to line up, so agreement is real evidence. |
| `jaccard_with_logit_lens` | Overlap with the plain logit lens on the same direction. Near 1.0 means the transport did nothing and this is a logit-lens result wearing a J-lens label. |
| `effective_tokens` | Entropy of the readout. A value near 1 on a whitespace-topped list means the lens had nothing to say, not that it said something subtle. |

Two traps the table is built to avoid. First, `unembed` is odd, so
`AUROC(−PC) = 1 − AUROC(+PC)` is *arithmetic* — "one end reads pleasant and the
other unpleasant" is guaranteed and cannot be evidence. Only the `+` end is
scored; what the `−` end genuinely contributes is its token list, and whether
those read as antonyms is the part a human has to judge. Second, GATE A scores
only English casing/space variants of each word, so a **correct non-English
readout counts as a miss** — on `qwen3-32b` the lens frequently verbalises in
Chinese (愤怒 for `angry`, 无聊 for `bored`, 恐惧/恐慌 for `terrified`), which
depresses the GATE A hit rate without the lens being wrong. Read `passed: false`
as *unproven*, not *refuted*, and check the token lists before concluding the lens
is dead.

**Phase 5 — optional structural extensions.** *(not written; only if 0–4 are clean)*
(a) **Layer sweep** — repeat extraction, PCA and lensing at several blocks to show
how the structure emerges and dissolves with depth. Free in compute if Phase 2
stored all layers, which it does. (b) **Perspective axis** — re-frame a subset of
stimuli as self ("you") vs other ("a person") and test whether a perspective axis
appears roughly orthogonal to the emotion axes, and what it lenses to. (c)
**Within-emotion PCA** — as a contrast, PCA inside a single emotion should recover
topic and scenario axes rather than affect, which is the argument for why the
cross-emotion design is the right one.

---

### Phases 6–9: from structure to function

**The premise.** Everything in Phases 0–5 is, by construction, the part of emotion
the model is *disposed to verbalise* — that is what a lens reads. The workspace
paper reports that the J-space carries only roughly **6–10% of a concept vector's
variance**; about 90% lies outside it. So every Phase 2 emotion vector has a large
**non-reportable remainder** that Phases 0–5 simply ignore.

**The question.** Does emotion influence the model's behaviour through its
reportable component, its non-reportable remainder, or both?

**Why it matters.** If behaviour is driven by the non-reportable part, then welfare
assessment built on self-report misses what emotion actually *does*. That is the
payoff, and it is why the structural phases were the setup rather than the result.

**Reuse discipline, which is what makes this a one-day project.** Phases 6–9 reuse
the Phase 2 emotion vectors and the Phase 0 lens directly. **No re-extraction, and
no fitting `J_ℓ` from scratch** — the pre-fitted lens is the entire reason this fits
in a day. Fitting one costs a backward pass per prompt over hundreds of prompts.

**Phase 6 — split each vector into reportable + remainder.** *(GPU · GATE)*
For each Phase 2 emotion vector `v` at the target block, find `v_J`, the part the
lens can express as tokens: a **sparse nonnegative** combination of the top-k
(k ≈ 16–25) lens dictionary directions reconstructing `v` by gradient pursuit. The
remainder is `v_⊥ = v − v_J`. Produce norm-matched `v`, `v_J`, `v_⊥`, and a
matched-norm random direction as control.
*Gate:* per-emotion reconstruction fraction **against matched-norm random
directions** — ratio and Monte-Carlo p-value, Bonferroni-corrected across emotions,
at both `k = 16` and `k = 25`. That comparison is the only claim the method supports:
`k` atoms from a large pool reconstruct some share of *any* direction, so a fraction
without its null measures the pursuit's degrees of freedom. `v_J` should also be
**small, ~5–15%**; if it comes out large, the decomposition or the lens is wrong, not
the theory. Plus the top tokens `v_J` decomposes into, which should read as the
emotion — reported, not gated.

> **Open design question, now settled.** The `jlens` API has no "dictionary" — a
> fitted lens is a per-layer matrix `J_ℓ`, and `JacobianLens` exposes only `jacobians`,
> `transport` and `apply` (verified against the pinned commit). The dictionary has to be
> *constructed*, and the atom for token `t` is `d_t = J_ℓᵀ (g ⊙ w_t)`, unit-normalised —
> as this paragraph originally said, for a better reason than it gave. The lens logit is
>
> ```
> logit_t = ⟨w_t, g ⊙ (J h / rms(J h))⟩ = ⟨g ⊙ w_t, J h⟩ / rms(J h) = ⟨Jᵀ u_t, h⟩ / rms(J h)
> ```
>
> so `Jᵀ u_t` **is** the measurement weight vector — the direction the lens *reads*
> token `t` with — and `1/rms(J h)` is a positive scalar common to every token. Anything
> orthogonal to every `Jᵀ u_t` moves no logit at all, which makes `span{Jᵀ u_t}` the
> verbalizable subspace by construction. **Reportability is a reading question.**
>
> `J_ℓ⁻¹ (g ⊙ w_t)` is a different direction: the one that most efficiently *writes*
> token `t`. It is a real question, kept runnable behind `--set write_space=true` as a
> labelled ablation, and it is not this one. The RMSNorm's learned gain `g` is absorbed
> exactly; its input-dependent `1/rms` cannot reorder logits.
>
> There is deliberately **no "does lensing an atom return its own token" check** — that
> is a property of write directions, and read directions have no reason to satisfy it.
> What is checked instead is the identity above, exactly: the atoms must reproduce the
> lens's own logits up to the positive scalar, which confirms `J_ℓ`'s stored orientation
> rather than assuming it. The **gate** is the reconstruction fraction against
> `n_random_controls` matched-norm random directions — ratio and Monte-Carlo p-value per
> emotion, Bonferroni-corrected across emotions — reported at both `k = 16` and `k = 25`.
> Reconstruction error and atom coherence are reported alongside.
>
> **What `v_⊥` is not.** Under sparse nonnegative coding the reconstructable set is
> `{Σ cᵢdᵢ : cᵢ ≥ 0, |support| ≤ k}` — a *union of cones*, not a linear subspace. So
> `v_⊥` means "not captured by this sparse approximation, at this `k`, from this pool",
> never "intrinsically unverbalizable". Both `k` are printed side by side because the
> boundary moves with `k`. A sentence of the form "the model cannot verbalise this
> component" is not supported by this stage.

**Phase 7 — build two measurement channels, kept strictly apart.** *(GPU · GATE)*
Pick 2–3 emotions with clean Phase 6 decompositions; an arousal-heavy negative one
(`anxious`) is the best behavioural probe.

- **Report channel** — five randomized, position-balanced choices among the selected
  emotions and “none,” requiring exactly one letter and scored mechanically. This is
  the report manipulation check Phase 8 reuses; no report judge is involved.
- **Behaviour channel** — four unrelated prompts from the prespecified Phase 8 family
  (`risk` by default), containing no emotion language and scored mechanically.

*Gate:* this is the phase most likely to silently confound the entire result. If
affect words leak into the behaviour rubric, "behaviour tracks emotion" collapses
into "the judge saw emotion words twice." The gate must **print both rubrics in
full** and explicitly confirm the behaviour rubric contains zero affect vocabulary.
It also hard-gates verified thinking-off rendering, ≥95% EOS completion, ≤10% invalid
responses per family, and known-answer scorer controls. A flat unsteered baseline is
diagnostic rather than a failure: consistent “none” reports can have maximal upward
room under steering.

**Phase 8 — compact preregistered steering test.** *(GPU · GATE)*
For each chosen emotion, steer at the target block at `α = 0, 0.5, 1.0` under
`v`, equal-norm `v_J`, equal-norm `v_⊥`, and five matched-norm random directions,
measuring **both** channels under each. The report channel reuses Phase 7's five
randomized exact-choice prompts. The behavioural outcome is one prespecified
four-prompt family (`risk` by default), rather than searching all families for the
largest result.

- **Fluency/perplexity check at every strength**, so a behavioural change is not
  degradation in disguise.
- **Specificity control:** re-run once with a topic vector (e.g. "ocean") in place
  of the emotion, showing any dissociation is not generic to any concept at all.
- Headline comparison: whole/readable/remainder versus the five-random distribution,
  across report and behaviour.

*Gate:* print the grid and the controls, and report with the **re-entry caveat**
intact — a `v_⊥` behavioural effect could still route through the workspace by
having downstream layers re-derive the concept. Phase 8 does not rule that out.

> Two things the brief leaves open, decided in
> [emotion_pca_jlens/phase8_steer.py](emotion_pca_jlens/phase8_steer.py). **The
> behavioural outcome is prespecified:** the default is risk, with its four prompt
> variants. Other families require an explicit override and are exploratory.
> **`α = 0` is generated once per concept** and copied across conditions with
> `shared_baseline` set on every copy: at zero strength every condition is the same
> unsteered model. Perplexity is measured under the
> *unsteered* model — only a model that was not perturbed can say whether the text is
> degraded. Completion and per-family format validity are also hard cell gates. The
> fixed `α = 1` comparison reports the full-vector manipulation check, `v_J` versus
> `v_⊥` report contrast, `v_⊥` versus all five random behavioural effects, the 25%
> report-silence threshold, and the dissociation statistic `D`.

**Phase 9 — the re-entry clamp.** *(decisive control; OPTIONAL — ask first, only if
6–8 are clean and time remains)*
Re-run `v_⊥` → behaviour while **clamping the emotion's J-lens coordinates to
clean-pass values at every position and layer**, so the concept cannot re-enter the
workspace and re-report itself.
Fiddly, and the verification comes first: confirm the clamp drives the report
channel to ~baseline while leaving unrelated J-space content intact, and print that
verification before trusting any behavioural number from this phase.
*Decisive cell:* `v_⊥`, J-space clamped, behaviour channel. If behaviour still
shifts while report is suppressed, that is an emotional state steering action
without being reportable.

> Three decisions in
> [emotion_pca_jlens/phase9_clamp.py](emotion_pca_jlens/phase9_clamp.py), all forced by
> the brief rather than chosen.
>
> **What gets clamped.** The lens reads token `t` at block `ℓ` with `J_ℓᵀ(g ⊙ w_t)`
> (Phase 6), so the emotion's J-space at that block is the span of those directions over
> its token set — the word's own single-token variants plus the atoms `v_J` selected —
> and the clamp is `h ← h − A_ℓ(A_ℓᵀh) + A_ℓ c_clean`. A **different subspace per
> block**, because each block has its own `J_ℓ`. Every fitted block by default; a run
> with a narrower `clamp_blocks` is marked NOT DECISIVE, since a partial clamp cannot
> tell "the effect bypassed the workspace" from "it re-entered at block 41".
>
> **What "clean-pass" means once the tokens diverge.** A steered model writes different
> text, so there is no position-by-position correspondence to clamp *to*. Clamping only
> the prompt leaves generation unclamped; clamping generated position `i` to the clean
> run's position `i` compares different sentences. So the two runs advance in lockstep,
> **one decode step at a time, with the clean run following the steered run's token** —
> its coordinates then answer "on this exact prefix, without the perturbation". The cost
> is two forward passes per step and one prompt at a time, and the clamp is registered
> *after* the steering hook so that at the one block carrying both, the residual held
> fixed is the post-steering one.
>
> **The verification gates the result.** Three checks print first: the clamp must be the
> identity at `α = 0` (numerically, per block and position — text equality is reported
> with the bf16 caveat rather than required); it must remove at least
> `clamp_min_report_suppression` of the report lift that steering with `v` produces; and
> it must leave unrelated J-space intact, measured as the rank correlation of the lens
> readout over control tokens outside the clamped set, plus the share of the clean
> residual's variance the subspace holds. Fail any and the decisive cell prints but is
> marked not interpretable, with which check failed naming the fix.
>
> Phase 9 also refuses to run when Phase 8's record shows no undegraded `v_⊥`
> behavioural movement above its random control: it is a control for a result, and a
> control for nothing costs hours to confirm a null Phase 8 already reported.

### Practical notes for Phases 6–9

Three things that will need deciding, flagged now rather than discovered later:

- **Phases 7–9 need *generation*, not forward passes.** Phase 2 costs one forward
  pass per stimulus; these cost ~150 sequential decode steps per sample, across
  4 conditions × 4 strengths × 2 channels × N prompts. That is the expensive part
  of the whole project — hence the 2–4 h estimate for Phase 8 — and it is why
  Phase 7 restricts to 2–3 emotions rather than all 16.
- **The LLM judge should not be the model being steered.** Using Qwen3-32B to
  score its own steered outputs confounds the measurement: the steering perturbs
  the judge as well as the subject. An external judge (a Claude model via API) is
  the right call, and this repo has no API client yet — a new dependency and a
  cost line, not a free step.
- **Chat template mismatch.** Phase 2 extracts from raw story text
  (`use_chat_template=False`, matching how the lens was fitted). Phases 7–9 need
  chat-formatted prompts, because the behaviour channel is a conversation. Steering
  vectors are generally taken to transfer across this boundary, but it is an
  assumption being made, not a verified property, and belongs in the writeup.

### Hardware and runtime

**One H100 80GB is enough for every phase.** Qwen3-32B in bf16 is ~66 GiB of
weights, which leaves ~14 GiB for activations — comfortable at `batch_size=8`.

Two constraints worth knowing before you rent anything:

- **≥80 GiB VRAM is mandatory, and there is no quantisation escape hatch.** The
  lens was fitted on the unquantised weights, so 4-bit loading would invalidate
  the transport — `config.validate()` rejects it. A 40–48 GiB card (A100-40GB,
  L40S) cannot hold this model in bf16.
- **More GPUs: shard, don't `device_map="auto"`.** Auto-sharding pipelines a
  single stream across cards and leaves most of them idle. Run one process per
  GPU over disjoint data instead (`CUDA_VISIBLE_DEVICES=N ... --num-shards 4
  --shard-index N --set device_map=None`), which scales close to linearly.

Also provision ~64 GiB host RAM (the lens is 6.6 GiB upcast to fp32 on the CPU)
and a **≥200 GiB volume** with `HF_HOME` pointed at it — the checkpoint is ~66 GiB
and is lost on every pod stop otherwise.

The runtimes in the phase table above are for the default 16-emotion set, and are
**estimates from FLOP arithmetic, not measurements** (~66 GFLOP/token forward,
35–45% MFU → ~1,000–3,000 tok/s on one H100). Get a real number in two minutes with
the `--limit` benchmark before committing to the 171-emotion run. Only Phase 2
scales with the number of emotions:

| Phase 2 scale | Stimuli | Estimated wall-clock, 1×H100 |
|---|---|---|
| 16 emotions + neutral, 400 each | 6,800 | **10–25 min** |
| All 171 + neutral, 200 each | 34,400 | **1.5–3.5 h** (÷ N with `--num-shards N`) |

Storage is not a constraint here, which is why all layers are kept: at
`5120 × 2 B = 10 KiB` per stimulus per layer, all 65 hidden states cost **~4.4 GiB**
for the 16-emotion run and **~22 GiB** for the 171-emotion run. `output_hidden_states`
returns every layer anyway, so keeping them costs no extra compute and makes the
Phase 5 layer sweep free rather than a re-run.

### 16 emotions or all 171?

Run the 16 first, then the 171. They answer different halves of the question.

The 16-emotion default is a **balanced 4-per-quadrant design**, which makes the a
priori valence and arousal contrasts *exactly* orthogonal and mean-zero (Phase 1
asserts this). That gives a legible scatter and a clean alignment test.

But it has a real statistical limit, and it is the reason to run 171 as well:
after mean-centring, `n` emotion centroids span a space of rank `n − 1`. Sixteen
points in 5,120 dimensions make PC1/PC2 explain a large variance fraction **almost
by construction** — so "PC1+PC2 = 60% of variance" from the 16-run is not by itself
evidence of anything. With 171 emotions that number becomes interpretable, and a
circumplex is the standard psychometric finding at that scale.

```bash
python run.py phase1                              # the balanced 16
python run.py phase1 --set emotions=all --set stories_per_emotion=200
```

`emotions=all` uses an **anchor design**: PCA is fitted over all 171 emotions, and
valence/arousal alignment is scored against the 16 labelled anchors only. The other
155 words are carried as `unlabelled` rather than hand-labelled, because assigning
valence and arousal to 155 words by eye would invent precision that is not there.
Published VAD norms (e.g. Warriner et al. 2013) would be the right way to label all
171 if that becomes worth doing.

### Phase 0 collects no activations

Worth stating plainly, because the stage does load the model: Phase 0 runs 12
short gate prompts, reads logits out of the lens, and writes **only** a
text/JSON run record of a few kilobytes to `results/phases/`. Nothing is pooled
and nothing is stored. Activation extraction begins in Phase 2.

What Phase 0 proves before any of that is worth building:

- **GATE A — concept readout.** On prompts whose expected readout is published,
  does an implied concept the prompt never names surface at mid layers? The
  prompts are vendored from Anthropic's `probe-swap.json` (the canonical sport
  items: *"the sport invented in Springfield, Massachusetts"* → `basketball`)
  and `lens-eval-association.json` (vignettes implying `grief`, `anger`,
  `shame`, `lonely`, `relief` without naming them). Ground truth we did not
  choose ourselves — a prompt we invented would let a plausible-looking readout
  pass as a correct one. The emotion set is this experiment in miniature.
  The gate also tracks the *answer* token's rank per block: a correctly-loaded
  lens shows the concept peaking mid-stack and the answer taking over late, and
  that crossover is harder to fake than either alone.
- **GATE B — late-layer identity.** At block 62 the transport spans one residual
  block, so `‖J−I‖_F/‖I‖_F` should be small and the three readouts — J-lens,
  plain logit lens, and the model's real output — should converge.

If these fail, the lens is loaded wrong and nothing downstream is valid.

### Storage: its own R2 folder

Experiment 2's activations go to **`pca-jlens-activations/<run_name>/`** in bucket
`emotion-vector-perspectives` — deliberately a separate top-level folder from
Experiment 1's `story-activations/`. Different model, different layer-index
convention, different analysis; a resumed run keys off the prefix and must not
find incompatible vectors there.

Two differences from Experiment 1's storage defaults, both chosen so activations
cannot be silently lost on an ephemeral pod:

- `r2_sync = True`, not `"auto"`. `"auto"` keeps activations local whenever the
  estimated size falls under a threshold, and "silently local" is the failure
  mode that loses a run to a pod teardown. With `True`, extraction aborts up
  front on missing credentials instead of after the GPU time is spent.
- `--set r2_sync=<anything unrecognised>` is an **error**. The shared coercion
  helper maps any unrecognised string to `False`, so `r2_sync=ture` would quietly
  disable the mirror; this config refuses to guess.

Phase 0 runs an R2 **preflight** before touching the lens: it uploads a small
marker to the prefix, confirms its size remotely, downloads it back and compares
bytes. An env-var check alone would not catch a wrong bucket, a bad endpoint or a
key without write permission — all of which look fine until the first PUT.

### Quick start (Experiment 2)

```bash
pip install -r requirements.txt   # installs jlens from its pinned commit

# Phase 0. Lens facts + R2 preflight + config cross-check; no model weights.
python run.py phase0 --dry-run
python run.py phase0                    # the real gate: GATE A and GATE B

# Phase 1. No model, no GPU. Coverage first, then the table.
python run.py phase1 --coverage-only    # emotion set by quadrant; no dataset
python run.py phase1                    # assemble + gate the stimulus set

# Overrides work as everywhere else.
python run.py phase0 --set topk=20 --set gate_blocks=evenly_spaced:16
python run.py phase1 --set emotions=all --set stories_per_emotion=200
```

`--dry-run` is a gate before the gate. It resolves, downloads and describes the
lens (6.6 GiB file, ~6.6 GiB host RAM to read) and cross-checks `d_model`, `J`
shape and fitted block range against the model config — catching a wrong lens
before ~65 GiB of weights load. Budget ~80 GiB VRAM for bf16 weights; the
Jacobians stay in host RAM, and transport runs there as a matrix–vector product,
so no 100 MiB `J_l` is copied to the GPU per readout.

Everything is driven by
[emotion_pca_jlens/pca_jlens_config.py](emotion_pca_jlens/pca_jlens_config.py).

### Caveats to state up front in any writeup

Not buried at the end. Each one bounds a claim that is otherwise easy to overstate.

1. **"Reportable" means "verbalisable as these single tokens."** The J-lens decodes
   one vocabulary token at a time, so a state can register as non-reportable simply
   by not being lexicalised. Absence of a readout is not absence of structure.
   Phase 0 reports explicitly when an expected word is not a single token, rather
   than scoring it as a miss.
2. **A lens reading of "anxious" is a disposition to say "anxious"** — not proof the
   model is anxious. This is precisely why the behavioural channel carries the
   weight of any non-reportability claim: for the structural phases, the honest
   claim is that emotion *representations* are circumplex-organised, not that the
   model feels along those axes.
3. **Without Phase 9, a `v_⊥` behavioural effect is not distinguished from workspace
   re-entry.** Downstream layers could re-derive the concept from the remainder and
   route it back through the workspace. If the clamp is not run, state that as a
   limitation plainly — do not gloss it.
4. **`v_⊥` is defined by an approximation, not by the model.** It is the residual of a
   `k`-sparse nonnegative code, whose reachable set is a union of cones rather than a
   linear subspace, so it means "missed at this `k`, from this pool" — never
   "intrinsically unverbalizable". Raising `k` moves the boundary and could move any
   result that rests on it.

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

Secrets go in this repo's own git-ignored `r2.env` (see
[Credentials: `r2.env`](#credentials-r2env)), including `HF_TOKEN` for a gated
model. Run `./push_r2_env.sh` once per pod to copy it over.

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

**R2 credentials come from this repo's `r2.env`.** See
[Credentials: `r2.env`](#credentials-r2env) below. `core/r2.py` accepts
`R2_ENDPOINT`, `R2_ENDPOINT_URL`, or `R2_ACCOUNT_ID` for the endpoint.

On the pod, run `./push_r2_env.sh` once — `sync-up` honours `.gitignore`, so a
normal sync deliberately will not carry credentials over the wire.

> **Why the repo's `r2.env` overrides the environment.** `run-experiment` also
> forwards the *shared* `../scripts/r2.env` to `/tmp/r2.env` and sources it before
> the entrypoint, and that file sets `R2_BUCKET=persona-activations` for
> `persona_introspection`, which reads `os.environ["R2_BUCKET"]` with no fallback.
> If the inherited value won, this project would upload 8 GiB into the wrong
> bucket. So `core/env_file.py` lets the repo-local file win, and prints the
> override rather than doing it silently. No `RUN_ENTRYPOINT` bucket hack needed.

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

Bucket `emotion-vector-perspectives`, with one top-level folder per experiment
family (set by `config.r2_root` / `config.r2_prefix`):

| Folder | Experiment | `r2_sync` default |
|---|---|---|
| `story-activations/<run_name>/` | 1 — emotion directions (Qwen2.5-32B) | `"auto"` (threshold-gated) |
| `pca-jlens-activations/<run_name>/` | 2 — PCA + J-lens (Qwen3-32B) | `True` (always) |

Separate folders are not cosmetic: a resumed run keys off its prefix, and the two
experiments' vectors are not interchangeable (different model, and Experiment 2
indexes layers by residual block rather than by hidden state). Experiment 2
defaults to `r2_sync=True` rather than `"auto"` so activations can never be left
on an ephemeral pod — see
[Experiment 2 storage](#storage-its-own-r2-folder).

### Credentials: `r2.env`

One git-ignored file in the repo root carries everything — R2 keys, bucket,
endpoint, and `HF_TOKEN`:

```bash
cp r2.env.example r2.env     # then fill in the blanks
python run.py r2 check       # confirms bucket + endpoint, and names the file used
```

Every entry point loads it automatically ([core/env_file.py](core/env_file.py)) —
no `source`, no `export`, and nothing to keep in sync between two files. It stays
valid shell, so `set -a; . r2.env; set +a` still works if you want the variables in
your own shell.

Search order, first existing file wins: `$R2_ENV_FILE`, `r2.env`, `.env`,
`/tmp/r2.env` (where the shared workflow lands creds on a pod),
`../scripts/r2.env`. Values from that file **override** the inherited environment
(see the warning above for why); `R2_ENV_FILE=none` opts out entirely.

To onboard a teammate, run `./share_with_teammate.sh` — it prints per-person,
read-only setup instructions. Never commit `r2.env` or paste it into Slack.

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

## Phase 4 reads out in Chinese, and what that did to GATE A

This is the largest interpretive correction the project has made, so it is written up
here rather than left in a notebook. The full analysis is
[`analysis/results_notebook.ipynb`](analysis/results_notebook.ipynb) sections 5e–5i,
exported to [`analysis/RESULTS.md`](analysis/RESULTS.md), with an audit trail for
independent checking in [`analysis/VERIFICATION.md`](analysis/VERIFICATION.md).

### The observation

Between 42% and 55% of the top-12 tokens in every Phase 4 lens readout are CJK script,
against 23–32% Latin. The stimuli are English stories, the anchors are English emotion
words, the probe set is English — and the lens reads the directions out in Chinese.

The obvious explanation is "Qwen3-32B is a Chinese model." That is not sufficient:
**96–97% of the Latin tokens in these readouts are whole words** (` failed`, ` sorrow`,
` panic`), not subword fragments. The lens is not being forced into Chinese because
English tokenises badly; it selects dense whole words when it selects Latin at all, and
still prefers Chinese for content. Why the block-31 readout has that preference is **not
diagnosed here** and should not be asserted.

### Two confounds, not one

GATE A asks: *does the emotion's own English word appear in its vector's top-12 lens
tokens?* It scored 0/14 and 5/114. Two independent things could produce that:

1. **Script** — the direction names its concept in Chinese, so no English lemma can
   reach the top-12 however good the direction is.
2. **Exact-lemma matching** — the readout offers ` sorrow` where the test demands
   ` sad`. A near-synonym is not a hit.

Translating the output only addresses the first. Both were scored separately.

### Why GATE A failed while GATE B passed

The resolution needs no translation data at all, and it is the single most useful
sentence in this section. The a-priori `+valence` axis separates the pleasant from the
unpleasant anchors **perfectly** — AUROC 1.00, no overlap — while ranking its best
English probe word at **1,927 out of 151,936**, with zero probes inside GATE A's top-12
window. Same story for `+arousal` at AUROC 0.96.

So the two Phase 4 tests were never measuring the same thing:

| test | kind | effect of a foreign-language readout |
| --- | --- | --- |
| **GATE A** | absolute containment — "is the lemma in the top 12?" | destroyed; the top-12 fills with Chinese first |
| **GATE B / AUROC** | relative ordering within a fixed English probe set | immune; burying all 14 probes by a common factor leaves the comparison intact |

**The direction knows about valence in English. It just does not say it in English.**
That one claim explains the whole pattern.

### Re-scoring GATE A in three tiers

Same top-12 containment rule, varying only what counts as the emotion's name:

| tier | what counts | 16 run | 171 run |
| --- | --- | --- | --- |
| T1 | exact English lemma (GATE A as specified) | 0% | 3% |
| T2 | English near-synonym | 6% | 8% |
| T3 | Chinese translation | **62%** | **25%** |

The objection is immediate: T3 gives each emotion 3–5 candidates where T1 gives one
lemma, so of course the rate rises. A permutation null answers exactly that — it keeps
every list intact, same contents and same length, and only shuffles **which list scores
which emotion's readout**. Chance under that null is 10.9% and 1.7% against observed
62.5% and 25.1%, p = 0.0005 over 2000 permutations. Scoring each readout against every
*other* emotion's list agrees (7.9%, 1.6%). The lists are not doing the work.

The translation table is [`analysis/zh_en_glossary.py`](analysis/zh_en_glossary.py). It
is **hand-written and is the only non-derived input in the analysis.** It was written
from the 171 emotion words alone and saved before any matching ran; matching is exact
set membership, never fuzzy; and the full table is printed in `RESULTS.md` so its
generosity can be audited and re-scored. The module docstring records the
pre-commitment protocol and discloses what the author had already seen.

### A denominator problem worth knowing about independently

Checked against the real Qwen3-32B tokenizer — cached locally at 11 MB, **no model
weights** — only 59% of the Chinese candidates are single tokens, and a multi-token
candidate can never appear in a top-12 *token* list. Restricting to the Chinese-testable
set moves the 171-run rate from 25.1% to 28.3%.

The more important finding has nothing to do with translation: **GATE A structurally
could not score 57 of the 171 emotions (33%)**, because their English lemma is not a
single token — and 43 of those 57 *do* have a single-token Chinese form. Their exclusion
is a fact about English orthography, not about their directions. Any GATE A rate should
carry the denominator it was computed on.

### What this changes, and what it does not

Changed: the readable summary of Phase 4. "The lens cannot name emotion directions"
becomes "the lens names them, in Chinese, and the gate asked in English."

**Not** changed, and none of these may be softened on the strength of the above:

* **Phases 1–3 are untouched.** Stimuli, emotion vectors, split-half reliability, PCA,
  circumplex recovery and the cross-run PC correspondence are computed from activations.
  The lens is not involved at any point in them.
* **The recorded GATE A verdict is still FAILED.** The re-scoring depends on a
  hand-written translation table, so it is reported alongside the gate, never
  substituted for it.
* **This is containment, not rank.** `phase4_readouts.csv` persisted only the top 12
  tokens per direction, so there is no Chinese analogue of `own_word_rank`.
* The under-converged 80-prompt lens, the effective dimensionality of 9.8, and PC1's
  affect-presence contamination on the 171 run are all exactly as they were.

Separately, and **not** because of anything above, Phase 6 was re-run in **read space**
and its verdict reversed. The lens score for token `t` is `u_tᵀ J h`, which regroups as
`(Jᵀu_t)ᵀ h` — so the direction the readout is linear in is `Jᵀu_t`, and that is what
`atom_mode: read` builds. The earlier gate asked "does lensing an atom return its own
token?", which is a **write**-direction property (`J⁺u_t`) that a read dictionary has no
reason to satisfy; the write-space run passes it 24/24, which is the tell. The read
construction is validated instead by a score-identity check holding at r > 0.9999.

The result is real, above chance, and small: **3.0%** and **2.3%** readable at k = 16
against random-direction controls of 0.65% and 0.64% (n = 500) — ratios of **4.6×** and
**3.6×**. Raising k from 16 to 25 moves the fraction by at most ~1e-3, so the ceiling is a
property of the pool and the vector, not the sparsity budget. Three caveats travel with
it: 30.9% of atoms exceed the 0.5 interchangeability threshold, so specific token
attributions are weak; "remainder" means outside the k-sparse *nonnegative* span of this
pool at this k — a union of cones, not a subspace — and so is **not** evidence of
anything being intrinsically unverbalizable; and `own_word_atom_rank` is null for
essentially every emotion, so the readable part is not the emotion's own word.

One number must not be read at face value: the 171-run gate records 0/172 emotions
beating its Bonferroni null. That is a **resolution limit** — 500 permutations floor the
p-value at 0.00200 while Bonferroni over 172 demands 0.00029, so no effect size could
clear it (it needs ≈3,440 permutations). All 172 clear the uncorrected 0.05; the 16-run
test is resolvable and passes 17/17.

The write-space run is kept under `results/phases/write_space_ablation/` as an
*ablation*, not a superseded attempt: it answers "what residual would make the model emit
token `t`", which is the question relevant to steering.

### Fixing it properly — no GPU required

A rank-based cross-lingual GATE A is a small computation. The readout is
`logits = lm_head(final_norm(J_l @ h))`, so it needs two tensors that are not in
`outputs/`:

| tensor | shape | size | where |
| --- | --- | --- | --- |
| `J[31]` | 5120 × 5120 fp32 | ~105 MB | one entry inside the 6.6 GB `Qwen3-32B_jacobian_lens.pt` |
| `lm_head.weight` + final-norm gain | 151936 × 5120 bf16 | ~1.6 GB | one shard of the Qwen3-32B safetensors |

The emotion vectors are already local in `phase2_emotion_vectors.safetensors` and the
tokenizer is already local. What remains is **one matrix–vector product and a top-k on
CPU** — seconds, no GPU, no 32B forward pass, no pod. The cost is ~1.7 GB of bandwidth,
not hardware.

Two follow-ups this specifies, neither yet run:

1. **A real cross-lingual GATE A** with Chinese token ranks, replacing the containment
   proxy above.
2. **A Chinese probe set for the AUROC tests.** Every Phase 4 ordering statistic uses
   English probes that sit at rank 10⁵. If the model's block-31 lexicalisation is
   Chinese, Chinese probes should separate at least as well and probably better — a
   cheap, falsifiable prediction that this analysis makes and does not test.

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

**Experiment 2, phases 2–9.** Phases 0 and 1 are implemented; each gate decides
whether the next is worth writing. Phases 6–9 (the functional experiment) are
specified in [Phases 6–9](#phases-69-from-structure-to-function) and deliberately
not coded yet — they depend on Phase 6's variance split coming out as expected.
Planned shape for the structural remainder:

1. **Phase 2 — vectors.** Mean-pool the post-block residual at a target block in
   the middle third, one vector per emotion; gate on split-half reliability
   (cosine > 0.9 trustworthy, ~0.6 means more data needed). Halve by *topic*, not
   by story: 12 stories share one scenario, so a story-level split would leak it
   across halves and inflate the cosine.
2. **Phase 3 — PCA.** Mean-centre across emotions *before* PCA — without it PC1 is
   just overall affect magnitude and the circumplex cannot appear. Gate on the
   variance table and the PC1–PC2 scatter.
3. **Phase 4 — lens the PCs.** Read out `+PC` and `−PC` for the top 3–5
   components. Expect PC3+ to get murky, and report that rather than straining to
   interpret noise.

Phase 1 chose to select stimuli from
[`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes)
rather than generate vignettes from templates. Three reasons, in
[emotion_pca_jlens/phase1_stimuli.py](emotion_pca_jlens/phase1_stimuli.py): the
dataset is topic-matched across emotions *by construction*; it has 1,200 stories
per emotion where a template scheme would give ~40, which the split-half gate
needs; and its stories are long enough to survive the 50-token pooling offset,
where 2–4 sentence vignettes (~40–60 tokens) would be **skipped entirely**. That
last one would have quietly emptied the dataset.

`remove_neutral_pcs` is available but **off** by default: projecting off the top
neutral-story PCs is right for isolating a single emotion direction (Experiment 1)
but could strip the very cross-emotion axes Experiment 2 is looking for. It
belongs as a robustness check, not a default.

**Experiment 1 extensions.** The interfaces exist for these; the experiments do not.

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
