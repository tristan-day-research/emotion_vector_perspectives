"""Configuration for activation extraction and emotion-direction construction.

Edit ``CONFIG`` at the bottom of this file, or override individual fields on the
command line (``--set batch_size=8 --set layer_spec=evenly_spaced:14``).

Every run writes ``run_config.txt`` / ``run_manifest.json`` containing the fully
resolved configuration, so a run is always reproducible from its own output
directory regardless of what this file says later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from core import paths

# --------------------------------------------------------------------------- #
# Which emotions to extract
# --------------------------------------------------------------------------- #
# The dataset covers all 171 emotion words from Anthropic's appendix (see
# data/emotions_171.txt). Extracting all of them costs 171x more GPU time than
# you need for a weekend sprint, so pick a subset that spans the affective
# circumplex (valence x arousal) and several of the paper's emotion clusters --
# that way cosine-similarity structure and top-1 classification are both
# informative rather than degenerate.
#
# The default 10: four positive (two high-arousal, two low-arousal), one
# high-arousal neutral-valence, five negative drawn from distinct clusters
# (Hostile Anger, Anxious Fear, Sadness, Shame, Depleted Disengagement).
DEFAULT_EMOTIONS = [
    "joyful",       # positive, high arousal   (Exuberant Joy)
    "grateful",     # positive, mid arousal    (Compassionate Gratitude)
    "calm",         # positive, low arousal    (Peaceful Contentment)
    "proud",        # positive, self-focused   (Competitive Pride)
    "surprised",    # neutral valence, high arousal
    "afraid",       # negative, high arousal   (Anxious Fear)
    "angry",        # negative, high arousal   (Hostile Anger)
    "sad",          # negative, low arousal    (Sadness)
    "ashamed",      # negative, self-focused   (Shame)
    "bored",        # negative, low arousal    (Depleted Disengagement)
]


@dataclass
class VectorExtractionConfig:
    """All parameters for one extraction + direction-construction run."""

    # ---------------------------------------------------------------- run --- #
    run_name: str = "qwen2.5-32b_10emo_all-layers"
    """Output directory name under ``outputs/``. Change it to start a fresh run."""

    seed: int = 0
    """Global RNG seed (torch/numpy/python)."""

    # -------------------------------------------------------------- model --- #
    model_name: str = "Qwen/Qwen2.5-32B"
    """Any decoder-only ``transformers`` checkpoint. Nothing below is Qwen-specific."""

    model_revision: str | None = None
    """Pin to a commit sha for reproducibility. ``None`` = ``main`` (resolved sha
    is still recorded in the run record)."""

    dtype: str = "bfloat16"
    """Weight/compute dtype. bf16 is what the checkpoint ships in and is right on
    any modern GPU.

    Note for CPU smoke tests: bf16 matmuls fall back to very slow paths on CPU
    (roughly 30x slower in our measurements). Set ``dtype=float32`` when running a
    small model on CPU to sanity-check the pipeline.
    """

    quantization: str | None = None
    """``None``, ``"4bit"`` or ``"8bit"`` (bitsandbytes).

    Off by default and worth keeping off: quantisation perturbs the residual
    stream we are measuring. Qwen2.5-32B in bf16 needs ~65 GiB of weights, so it
    fits one 80 GiB A100/H100 unquantised. Use 4bit only to fit a larger model on
    smaller hardware, and never compare directions across quantisation settings.
    """

    device_map: str | None = "auto"
    """``"auto"`` shards across visible GPUs via accelerate. ``None`` = single device."""

    attn_implementation: str | None = None
    """e.g. ``"sdpa"``, ``"flash_attention_2"``. ``None`` lets transformers choose."""

    trust_remote_code: bool = False

    # --------------------------------------------------------------- data --- #
    emotions: list[str] = field(default_factory=lambda: list(DEFAULT_EMOTIONS))
    """Emotions to extract. ``None`` means all 171 (~205k stories: don't, yet)."""

    stories_per_emotion: int | None = 1200
    """Stories per emotion, topic-stratified. 1200 = all of them.

    Reduce for a faster first pass; 300 (3 per topic) still gives stable
    centroids in our split-half check. Values below 100 cannot cover every topic
    and will unbalance the split proportions -- you'll get a warning.
    """

    neutral_stories: int | None = 1200
    """Neutral stories for nuisance-PC removal. 1200 = all. The PCA is fitted on
    the *training-topic* neutrals only (840 of 1200 at a 70/15/15 split)."""

    dataset_revision: str | None = None
    """Pin the HF dataset revision. ``None`` = ``main`` (sha recorded regardless)."""

    # ------------------------------------------------------------- splits --- #
    split_seed: int = 12345
    """Seed for the topic-level train/validation/test partition."""

    split_proportions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    """Fractions of *topics* (not stories) per split. Must sum to 1."""

    full_dataset: bool = False
    """Direction construction uses all splits instead of train only.

    For the final directions you ship, *after* held-out evaluation is done. Keep
    ``False`` while evaluating: directions used in held-out analyses must be built
    from training topics alone. Extraction always records true split labels, so
    flipping this flag needs no re-extraction.
    """

    # ------------------------------------------------------- tokenization --- #
    max_length: int = 512
    """Truncation length. Stories are 93-170 words (~120-230 tokens), so 512
    truncates nothing; it only bounds worst-case memory."""

    add_special_tokens: bool = True
    """Tokenizer default. Note any BOS token counts as a real token and therefore
    against ``token_offset``."""

    use_chat_template: bool = False
    """Wrap each story in the model's chat template.

    ``False`` matches the paper: activations are taken over raw story text. This
    hook exists for the later experiencer-binding experiments (assistant /
    first-person persona / third-person character), which need a conversation.
    Requires an instruct checkpoint.
    """

    chat_role: str = "user"
    chat_add_generation_prompt: bool = False

    text_prefix: str = ""
    text_suffix: str = ""
    """Optional raw-string framing applied before tokenization. Empty = verbatim
    story text. Part of the compatibility fingerprint."""

    # ------------------------------------------------------------ pooling --- #
    token_offset: int = 50
    """Leading real (non-pad) tokens to exclude before mean-pooling.

    Anthropic: "averaging across all token positions within each story, beginning
    with the 50th token (at which point the emotional content should be
    apparent)". Padding-agnostic -- see :mod:`core.pooling`.
    """

    min_pooled_tokens: int = 10
    """Minimum tokens remaining after the offset for an example to be kept.
    Shorter examples are skipped and logged to ``skipped.jsonl``. No story in this
    dataset is short enough to trip this, but a truncated/filtered variant could be."""

    pooling: str = "mean_after_offset"
    """Recorded for provenance; ``mean_after_offset`` is the only implementation."""

    # ------------------------------------------------------------- layers --- #
    layer_spec: object = "all"
    """Which residual-stream layers to save.

    ``"all"``               every hidden state, ``0..n_layers`` (65 for Qwen2.5-32B)
    ``"blocks"``            block outputs only, ``1..n_layers``
    ``"evenly_spaced:14"``  14 evenly spaced layers (the paper's RSA sweep)
    ``"range:20:50:2"``     half-open range with a step
    ``[0, 16, 32, 48, 64]`` explicit hidden-state indices (negatives allowed)

    Index 0 is the embedding output; index ``i`` is the residual stream after
    block ``i``. ``"all"`` is the default because re-running a 32B model to get a
    layer you skipped is far more expensive than the disk -- run ``--dry-run`` to
    see the size before committing.
    """

    # --------------------------------------------------------- extraction --- #
    batch_size: int = 16
    """Sequences per forward pass. ``output_hidden_states=True`` holds every layer
    on the GPU for the batch: roughly ``batch x seq x hidden x (n_layers+1) x 2``
    bytes (~1 GiB at batch 16 for Qwen2.5-32B on these stories). Lower it if you
    OOM after the weights load."""

    activation_dtype: str = "bfloat16"
    """Storage dtype for pooled activations. Pooling always reduces in fp32; only
    the saved value is cast. bf16 halves the footprint and is well within the
    noise of the directions. Use ``"float32"`` for numerically delicate analyses."""

    chunk_size: int = 512
    """Examples per activation chunk file. At 65 layers x 5120 dims x bf16 this is
    ~341 MiB/chunk; also the granularity of resume and R2 upload."""

    log_every_batches: int = 20

    # ------------------------------------------------ direction construction - #
    neutral_variance_threshold: float = 0.50
    """Keep the smallest number of neutral PCs explaining at least this much
    neutral variance, then project emotion vectors off that subspace. 0.50 is the
    paper's value."""

    # ------------------------------------------------------------ storage --- #
    r2_sync: object = "auto"
    """Mirror activation chunks to Cloudflare R2 as they are written.

    ``"auto"``  enable when R2 credentials exist *and* the estimated output
                exceeds ``r2_threshold_gib`` (the default config does: ~9 GiB)
    ``True``    always (errors if credentials are missing)
    ``False``   never
    """

    r2_threshold_gib: float = 5.0
    r2_prefix: str | None = None
    """Bucket key prefix. ``None`` -> ``runs/<run_name>/activations``."""

    delete_local_after_sync: bool = False
    """Delete each local ``.safetensors`` chunk once uploaded. Index parquets stay,
    so resume still works. Turn on when the pod disk cannot hold the whole run;
    remember ``python -m core.r2 pull`` before fitting directions."""

    # --------------------------------------------------------- evaluation --- #
    eval_splits: tuple[str, ...] = ("validation", "test")
    eval_bootstrap_n: int = 50
    """Bootstrap resamples (over training *topics*) for direction stability."""

    eval_layer_spec: object = "evenly_spaced:14"
    """Layers to evaluate. Evaluation reads every stored layer it is asked for;
    ``"all"`` works but produces a lot of rows."""

    make_plots: bool = True

    # ------------------------------------------------------------ helpers --- #

    def __post_init__(self) -> None:
        self.split_proportions = tuple(float(p) for p in self.split_proportions)
        self.eval_splits = tuple(self.eval_splits)
        if self.emotions is not None:
            self.emotions = list(self.emotions)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the config is usable."""
        problems: list[str] = []

        if abs(sum(self.split_proportions) - 1.0) > 1e-9:
            problems.append(f"split_proportions must sum to 1, got {sum(self.split_proportions)}")
        if len(self.split_proportions) != 3:
            problems.append("split_proportions must have three entries")
        if self.split_proportions[0] <= 0:
            problems.append("the train proportion must be > 0")

        if self.token_offset < 0:
            problems.append("token_offset must be >= 0")
        if self.max_length <= self.token_offset:
            problems.append(
                f"max_length ({self.max_length}) must exceed token_offset ({self.token_offset}), "
                "otherwise every example is skipped"
            )
        if self.min_pooled_tokens < 1:
            problems.append("min_pooled_tokens must be >= 1")
        if self.max_length - self.token_offset < self.min_pooled_tokens:
            problems.append(
                f"max_length - token_offset = {self.max_length - self.token_offset} < "
                f"min_pooled_tokens = {self.min_pooled_tokens}"
            )

        if not 0 < self.neutral_variance_threshold < 1:
            problems.append("neutral_variance_threshold must be in (0, 1)")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.chunk_size < 1:
            problems.append("chunk_size must be >= 1")
        if self.stories_per_emotion is not None and self.stories_per_emotion < 1:
            problems.append("stories_per_emotion must be >= 1 or None")
        if self.neutral_stories is not None and self.neutral_stories < 2:
            problems.append("neutral_stories must be >= 2 (PCA needs at least two samples) or None")
        if self.activation_dtype not in ("bfloat16", "float16", "float32"):
            problems.append(f"unsupported activation_dtype {self.activation_dtype!r}")
        if self.quantization not in (None, "none", "4bit", "8bit"):
            problems.append(f"unsupported quantization {self.quantization!r}")

        if self.emotions is not None:
            if len(self.emotions) != len(set(self.emotions)):
                problems.append("emotions contains duplicates")
            if len(self.emotions) < 2:
                problems.append(
                    "at least two emotions are required: the direction is a difference from "
                    "the mean across emotion centroids, which is undefined for one emotion"
                )
            try:
                known = set(paths.load_emotions_171())
            except OSError:
                known = set()
            unknown = sorted(set(self.emotions) - known) if known else []
            if unknown:
                problems.append(
                    f"emotions not in data/emotions_171.txt: {unknown} "
                    "(the dataset check at load time is authoritative)"
                )

        if self.use_chat_template and self.token_offset > 0:
            # Not an error, but easy to get wrong: template tokens are consumed by
            # the offset, so the effective story offset shifts.
            pass

        return problems

    def resolved_r2_prefix(self) -> str:
        return self.r2_prefix or f"runs/{self.run_name}/activations"

    @property
    def output_dir(self) -> Path:
        return paths.run_dir(self.run_name)

    @property
    def results_dir(self) -> Path:
        """Small, pullable artefacts: directions, metrics, plots, run records.

        The shared ``scripts/`` workflow's ``pull-results`` fetches exactly
        ``<RESULTS_SUBDIR>/*/results/***`` and excludes any ``activations/``
        directory. Putting everything except activations under a ``results/`` level
        is what makes ``pull-results-emotionvectors`` work with zero flags -- and
        with the wrong layout it silently pulls *nothing*, which is worse than an
        error. Verified against the real rsync filters.
        """
        return self.output_dir / "results"

    @property
    def activations_dir(self) -> Path:
        """Large activation chunks: deliberately a sibling of ``results/``.

        Kept outside ``results/`` so ``pull-results`` never drags 8 GiB over the
        wire; activations travel via R2 instead.
        """
        return self.output_dir / "activations"

    @property
    def directions_dir(self) -> Path:
        return self.results_dir / "directions"

    @property
    def eval_dir(self) -> Path:
        return self.results_dir / "evaluation"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["split_proportions"] = list(self.split_proportions)
        out["eval_splits"] = list(self.eval_splits)
        return out

    def fingerprint(self, resolved_layers: list[int], hidden_size: int, model_sha: str,
                    dataset_sha: str) -> dict:
        """Settings that change the *meaning* of a stored activation vector.

        Two runs sharing a fingerprint may share an activations directory; a
        mismatch aborts extraction rather than mixing incomparable vectors.

        Deliberately excluded: ``emotions``, ``stories_per_emotion``,
        ``neutral_stories``. Those select *which* examples to extract, not what a
        vector means, so you can add emotions to a config and resume the same run
        to extend it. ``split_seed``/``split_proportions`` *are* included, because
        the split label is recorded alongside every activation.
        """
        return {
            "model_name": self.model_name,
            "model_sha": model_sha,
            "dtype": self.dtype,
            "quantization": self.quantization or "none",
            "layers": [int(l) for l in resolved_layers],
            "hidden_size": int(hidden_size),
            "activation_dtype": self.activation_dtype,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens,
            "use_chat_template": self.use_chat_template,
            "chat_role": self.chat_role if self.use_chat_template else "",
            "chat_add_generation_prompt": (
                self.chat_add_generation_prompt if self.use_chat_template else False
            ),
            "text_prefix": self.text_prefix,
            "text_suffix": self.text_suffix,
            "token_offset": self.token_offset,
            "min_pooled_tokens": self.min_pooled_tokens,
            "pooling": self.pooling,
            "dataset_id": "ryancodrai/emotion-probes",
            "dataset_sha": dataset_sha,
            "split_seed": self.split_seed,
            "split_proportions": list(self.split_proportions),
        }

    def with_overrides(self, overrides: dict[str, object]) -> "VectorExtractionConfig":
        """Return a copy with ``overrides`` applied, coercing to field types."""
        valid = {f.name: f for f in fields(self)}
        coerced: dict[str, object] = {}
        for key, value in overrides.items():
            if key not in valid:
                raise KeyError(
                    f"unknown config field {key!r}. Valid fields: {sorted(valid)}"
                )
            coerced[key] = _coerce(key, value, getattr(self, key))
        return replace(self, **coerced)


def _coerce(key: str, value: object, current: object) -> object:
    """Coerce a string CLI override to the type of the current field value."""
    if not isinstance(value, str):
        return value
    text = value.strip()

    if key in ("layer_spec", "eval_layer_spec"):
        # These are intentionally polymorphic: "all", "evenly_spaced:14", or a list.
        if text.startswith("["):
            return json.loads(text)
        return text
    if key == "r2_sync":
        if text.lower() in ("auto", ""):
            return "auto"
        return text.lower() in ("1", "true", "yes", "on")
    if text.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key}: cannot parse {value!r} as a boolean")
    if isinstance(current, tuple):
        parsed = json.loads(text) if text.startswith("[") else [p for p in text.split(",") if p]
        return tuple(type(current[0])(p) if current else p for p in parsed)
    if isinstance(current, list) or key == "emotions":
        if text.startswith("["):
            return json.loads(text)
        return [p.strip() for p in text.split(",") if p.strip()]
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if current is None:
        # Fields typed `X | None` that currently hold None: int-like names we know.
        if key in ("stories_per_emotion", "neutral_stories"):
            return int(text)
    return text


#: The configuration used when no ``--config-json`` is supplied.
CONFIG = VectorExtractionConfig()
