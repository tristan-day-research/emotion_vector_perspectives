"""Phase 2 (GATE): one pooled residual vector per emotion, gated on reliability.

What this stage does
--------------------
Forward-passes every Phase 1 stimulus, mean-pools the residual stream over the
real tokens after ``token_offset``, stores the pooled activations for *every*
hidden state, and averages them into one vector per emotion at ``target_block``.
Then it stops at a gate: the split-half reliability of those vectors.

This is the only phase that collects activations; they mirror to R2 as they are
written. The small artefact everything downstream consumes is
``results/phases/phase2_emotion_vectors.safetensors`` plus its JSON sidecar.

Layer indexing -- the thing to get silently wrong
-------------------------------------------------
Two conventions are in play and they differ by one:

* ``target_block`` is a **residual-block** index (the J-lens convention), because
  the vector has to be readable by the lens in Phase 4.
* ``output_hidden_states`` and :mod:`core.pooling` index the hidden-state tuple,
  where ``0`` is the embedding output.

``hidden_state_index = block_index + 1``, always via
:func:`core.jlens_lens.hidden_state_index` and never written inline. The lens is
fitted only for blocks ``0 .. n_layers-2`` (0..62 for Qwen3-32B's 64 blocks), so a
``target_block`` outside that range is **rejected up front**: a vector the lens
cannot read is useless to every phase after this one, and the failure would
otherwise surface in Phase 4 after the GPU time was spent.
``target_block=None`` resolves to the middle of the fitted range -- the same block
Phase 0 prints as its suggestion -- and is reported in both conventions.

Why every hidden state is stored
--------------------------------
``output_hidden_states=True`` materialises all of them anyway, so keeping them
costs disk (~4.4 GiB for the 16-emotion run, ~22 GiB for 171) but no extra
compute. That turns the Phase 5 layer sweep into a re-read instead of another
pass of a 32B model. Consequently ``target_block`` is *not* in the activation
fingerprint -- see :meth:`PCAJLensConfig.fingerprint`.

The gate: split-half by TOPIC, on mean-centred vectors
------------------------------------------------------
Two choices, both load-bearing, and the second is the one that would quietly
render the gate useless:

1. **Halve by topic, not by story.** Twelve stories share one topic and are
   near-paraphrases of one scenario, so a story-level split puts variants of the
   same scenario in both halves, and the cosine measures paraphrase similarity
   rather than reliability. The topic partition is drawn once from the *full*
   Phase 1 table, so every shard derives the same one.
2. **Compare mean-centred vectors.** Raw pooled residuals share a large
   layer-wide common component, so a raw split-half cosine sits at ~0.999 for
   every emotion and the gate cannot fail -- it would pass on stimuli that carry
   no emotion signal at all. Phase 3 runs PCA on the *mean-centred* vectors, so
   the honest reliability number is the one measured on what Phase 3 consumes.
   Each half is centred by its own cross-emotion mean, keeping the two halves
   independent refits (the discipline ``evaluate_directions.split_half_agreement``
   follows when it gives each half its own neutral subspace). Both numbers are
   printed; the centred one is the gate.

Above ~0.9 centred, the vectors are trustworthy. Around 0.6 means more stimuli
are needed before the PCA can mean anything -- raise ``stories_per_emotion`` and
re-run, rather than reading a circumplex out of noise.

How the emotion vectors are accumulated, and why not by re-reading the chunks
----------------------------------------------------------------------------
Per-emotion sums are accumulated in host RAM as extraction streams past, and
written as small per-shard partials (``phase2_partials/shardNNN.*``, ~14 MiB at
171 emotions). Re-reading the target layer out of the activation store afterwards
is the obvious alternative and is wrong here for two independent reasons:
``delete_local_after_sync=True`` is the default, so the chunk tensors are in R2
only by the time extraction ends; and with ``--num-shards`` the sums have to be
combined across processes regardless. Only two accumulators per emotion are
needed -- one per topic half -- because the full vector is their weighted sum, so
this costs megabytes rather than a re-download.

A resumed run folds the already-stored stimuli back in: from this shard's prior
partial when its contribution count matches the store (the clean-shutdown case,
which needs no chunks at all), and by re-reading the target layer from the store
otherwise. Whichever path is taken, aggregation asserts that every stimulus in
scope is either accumulated or explicitly skipped, and **refuses to write the
vectors** when it is not. An emotion vector that is quietly the mean of a subset
is the failure this arrangement exists to prevent, because nothing downstream
could detect it.

Usage::

    python run.py phase2 --dry-run       # stimuli + storage estimate; no weights
    python run.py phase2 --limit 256     # real throughput number first
    python run.py phase2                 # the gate
    python run.py phase2 --set target_block=40 --set batch_size=4

    # 171 emotions across two GPUs, one process each over disjoint stimuli
    for i in 0 1; do
      CUDA_VISIBLE_DEVICES=$i python run.py phase2 \
          --num-shards 2 --shard-index $i --set device_map=None &
    done; wait
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import activation_store, env_file, jlens_lens, model_utils, paths, provenance
from core.activation_store import (
    ActivationStore,
    ActivationWriter,
    estimate_storage_bytes,
    human_bytes,
    init_or_check_manifest,
)
from core.pooling import pool_hidden_states
from core.seeds import rng_for, set_global_seeds

# Reused rather than reimplemented: `assign_shard` defines the round-robin shard
# partition, and the two token-length helpers define what --dry-run reports. A
# second copy of either would be the thing that drifts, and a shard partition that
# disagreed between the two experiments would be a silent overlap.
from extract_emotion_vectors.extract_activations import (
    assign_shard,
    estimate_tokens_without_tokenizer,
    token_length_stats,
)
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config

# The literal Phase 1 writes into the `emotion` column for neutral rows -- it is
# not NA there, deliberately, so groupby/plotting treat neutral as a category.
# Imported rather than re-spelled so the two stages cannot drift apart.
from emotion_pca_jlens.phase1_stimuli import NEUTRAL_QUADRANT

RULE = "=" * 78
THIN = "-" * 78

#: Directory of per-shard accumulator partials, under the phase output dir.
PARTIALS_DIR_NAME = "phase2_partials"

#: Columns Phase 1 promises. Checked rather than assumed: a stimulus table missing
#: `topic_id` would make the split-half gate silently halve by something else.
REQUIRED_STIMULUS_COLUMNS = (
    "emotion", "quadrant", "text", "valence", "arousal", "family",
    "source", "example_id", "topic", "topic_id", "story_idx", "split",
    "content_sha1",
)

#: Saved vectors are float32: a few MB at 171 emotions, and Phase 3's PCA plus
#: Phase 6's decomposition both want more headroom than bf16 leaves.
VECTOR_DTYPE = np.float32


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Phase 2 gate: one pooled residual vector per emotion, "
            "gated on split-half reliability."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the stimuli, resolve layers, estimate storage; never load the model",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N stimuli per shard -- for GPU benchmarking. Artefacts "
             "go to a benchmark subdirectory so a partial mean cannot be mistaken "
             "for the real emotion vectors",
    )
    p.add_argument("--num-shards", type=int, default=1, help="split the stimuli into N shards")
    p.add_argument("--shard-index", type=int, default=0, help="which shard this process handles")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing incompatible activations directory (and its partials) "
             "instead of aborting. Single-process only: with --num-shards this would "
             "delete the other shards' work",
    )
    p.add_argument(
        "--config-json", type=Path, default=None, help="JSON file of config overrides"
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field; repeatable",
    )
    return p


# --------------------------------------------------------------------------- #
# Phase 1's stimulus table
# --------------------------------------------------------------------------- #

def read_stimuli(config: PCAJLensConfig) -> pd.DataFrame:
    """Load ``phase1_stimuli.parquet``, or explain how to produce it."""
    path = config.stimuli_path
    if not path.exists():
        raise SystemExit(
            f"no stimulus table at\n  {path}\n\n"
            "Phase 2 consumes Phase 1's output rather than re-selecting stories, so "
            "the two stages cannot disagree about what was measured.\n"
            "Run it first:\n\n"
            "  python run.py phase1\n"
        )
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_STIMULUS_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing columns {missing}.\n"
            "It was probably written by an older Phase 1; re-run `python run.py phase1`."
        )
    if df.empty:
        raise SystemExit(f"{path} has no rows; re-run `python run.py phase1`.")
    if df["emotion"].isna().any():
        # Phase 1 fills neutral rows with the literal "neutral". An NA here would
        # make groupby drop those rows and the neutral vector would vanish silently.
        raise SystemExit(
            f"{path} has NA in `emotion` for "
            f"{int(df['emotion'].isna().sum())} rows. Phase 1 labels neutral rows "
            f"{NEUTRAL_QUADRANT!r} precisely so this cannot happen; re-run phase1."
        )
    dupes = int(df["example_id"].duplicated().sum())
    if dupes:
        raise SystemExit(
            f"{path} has {dupes} duplicate example_id values. Resume keys off "
            "example_id, so duplicates would be extracted once and averaged twice."
        )
    # Stable order, independent of however Phase 1 happened to sort: the shard
    # partition is positional, so a reordered table would reshuffle the shards and
    # a resumed multi-GPU run would re-extract rows another shard already owns.
    return df.sort_values("example_id").reset_index(drop=True)


def stimuli_fingerprint(df: pd.DataFrame) -> dict:
    """Content hash of the stimulus set, for the activation fingerprint.

    Hashes ``(example_id, content_sha1)`` pairs *sorted*, so re-ordering the table
    is not a change while altering any stimulus's text -- or adding/removing one --
    is. The counts alongside the digest are redundant but make
    ``IncompatibleRunError``'s field-by-field diff readable: "16 emotions became
    172" is a diagnosis, a changed hex digest is not.
    """
    keys = sorted(
        f"{eid}\x1f{sha}" for eid, sha in zip(df["example_id"], df["content_sha1"])
    )
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return {
        "sha256": digest,
        "n_stimuli": int(len(df)),
        "n_groups": int(df["emotion"].nunique()),
    }


def scope_table(examples: pd.DataFrame, num_shards: int, limit: int | None) -> pd.DataFrame:
    """The stimuli the *whole* run (all shards together) is expected to cover.

    Needed because completeness is checked across shards but each process only
    holds its own. With ``--limit`` every shard truncates its own partition, so the
    scope is the union of those prefixes rather than the first ``limit`` rows.
    """
    if limit is None:
        return examples
    parts = [assign_shard(examples, num_shards, i).iloc[:limit] for i in range(num_shards)]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Layer resolution
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TargetLayer:
    """The block Phase 2 averages at, in both index conventions."""

    block: int
    hidden_state: int
    n_layers: int
    max_lens_block: int
    resolved_from: str

    def describe(self) -> str:
        return jlens_lens.describe_block(self.block, self.n_layers)


def resolve_target_layer(config: PCAJLensConfig, n_layers: int) -> TargetLayer:
    """Resolve ``config.target_block`` and refuse a block the lens cannot read.

    ``None`` means "middle of the stack", resolved against the *fitted* block range
    rather than the full depth, so the default can never land on the one block with
    no ``J_l``. It reproduces Phase 0's printed suggestion
    (``fitted[len(fitted) // 2]``) from the model config alone, which is why Phase 2
    does not need to load the 6.6 GiB lens to agree with it.
    """
    highest = jlens_lens.max_lens_block(n_layers)
    if highest < 0:
        raise SystemExit(
            f"a {n_layers}-block model has no lens-readable block "
            "(J_l is fitted for 0..n_layers-2)"
        )
    if config.target_block is None:
        block, origin = highest // 2, "None -> middle of the fitted block range"
    else:
        block, origin = int(config.target_block), "config.target_block"
        if not 0 <= block <= highest:
            raise SystemExit(
                f"target_block={block} is outside the lens's fitted range 0..{highest} "
                f"(= 0..n_layers-2 for this {n_layers}-block model).\n"
                f"Block {n_layers - 1} is the transport *target*, not a source, so it "
                "has no J_l.\n"
                "A vector at a block the lens cannot read is useless to Phase 4 and "
                "everything after it, so this aborts here rather than there."
            )
    return TargetLayer(
        block=block,
        hidden_state=jlens_lens.hidden_state_index(block),
        n_layers=n_layers,
        max_lens_block=highest,
        resolved_from=origin,
    )


# --------------------------------------------------------------------------- #
# The topic-level split-half partition
# --------------------------------------------------------------------------- #

def topic_halves(df: pd.DataFrame, seed: int) -> tuple[dict[int, int], dict]:
    """Partition *topics* into two halves. Returns ``(topic_id -> 0|1, report)``.

    Halving by topic rather than by story is the whole point: 12 stories share one
    topic and are near-paraphrases of one scenario, so a story-level split leaks the
    scenario into both halves and the cosine measures paraphrase similarity instead
    of reliability.

    Derived from the full stimulus table with a seed-derived generator, so every
    shard computes the identical partition without communicating -- if they
    disagreed, the two "halves" would overlap and the gate would silently pass.
    """
    topics = np.asarray(sorted(int(t) for t in df["topic_id"].unique()))
    rng = rng_for(seed, "phase2_split_half")
    permuted = topics[rng.permutation(len(topics))]
    cut = len(permuted) // 2
    half_a = {int(t) for t in permuted[:cut]}
    mapping = {int(t): (0 if int(t) in half_a else 1) for t in topics}
    counts = df["topic_id"].map(mapping).value_counts().to_dict()
    return mapping, {
        "n_topics": int(len(topics)),
        "n_topics_half_a": cut,
        "n_topics_half_b": int(len(permuted) - cut),
        "n_stimuli_half_a": int(counts.get(0, 0)),
        "n_stimuli_half_b": int(counts.get(1, 0)),
        "seed": seed,
        "derived_from": "sorted topic_id of the full Phase 1 table",
    }


# --------------------------------------------------------------------------- #
# The accumulator and its on-disk partials
# --------------------------------------------------------------------------- #

@dataclass
class VectorAccumulator:
    """Per-(emotion, topic-half) sums of the target block's pooled activation.

    Two halves per emotion is all that is needed: the full emotion vector is their
    count-weighted mean, so nothing is lost by never storing a third accumulator.

    Sums are float64 while the addends are the bf16 values that went to disk. That
    pairing is deliberate: fp64 makes the reduction order irrelevant across shards
    and resumes, and summing the *stored* values means re-deriving a vector from the
    chunks later reproduces this run exactly rather than almost.
    """

    emotions: list[str]
    d_model: int
    sums: np.ndarray    # (n_emotions, 2, d_model) float64
    counts: np.ndarray  # (n_emotions, 2) int64

    @classmethod
    def empty(cls, emotions: list[str], d_model: int) -> "VectorAccumulator":
        return cls(
            emotions=list(emotions),
            d_model=int(d_model),
            sums=np.zeros((len(emotions), 2, d_model), dtype=np.float64),
            counts=np.zeros((len(emotions), 2), dtype=np.int64),
        )

    @property
    def n_contributed(self) -> int:
        return int(self.counts.sum())

    def add(self, rows: np.ndarray, halves: np.ndarray, vectors: np.ndarray) -> None:
        """Fold in ``vectors[i]`` at ``(rows[i], halves[i])``."""
        if vectors.shape[1] != self.d_model:
            raise ValueError(
                f"vectors have {vectors.shape[1]} dims, accumulator has {self.d_model}"
            )
        np.add.at(self.sums, (rows, halves), vectors.astype(np.float64, copy=False))
        np.add.at(self.counts, (rows, halves), 1)

    def merge(self, other: "VectorAccumulator") -> None:
        if other.emotions != self.emotions or other.d_model != self.d_model:
            raise ValueError(
                "cannot merge accumulators over different emotion sets or dimensions:\n"
                f"  self : {len(self.emotions)} emotions, d_model={self.d_model}\n"
                f"  other: {len(other.emotions)} emotions, d_model={other.d_model}"
            )
        self.sums += other.sums
        self.counts += other.counts


def partials_dir(out_dir: Path) -> Path:
    return Path(out_dir) / PARTIALS_DIR_NAME


def _partial_paths(directory: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard{shard_index:03d}"
    return directory / f"{stem}.safetensors", directory / f"{stem}.json"


def save_partial(
    directory: Path,
    shard_index: int,
    accumulator: VectorAccumulator,
    meta: dict,
) -> tuple[Path, Path]:
    """Write one shard's accumulator, tensors first and JSON second.

    Same commit discipline as :class:`~core.activation_store.ActivationWriter`: the
    JSON is the marker that the pair is complete, so a partial killed mid-write is
    ignored by :func:`load_partials` rather than read as zeros.
    """
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors_path, meta_path = _partial_paths(directory, shard_index)
    meta_path.unlink(missing_ok=True)

    tmp_tensors = tensors_path.with_suffix(".safetensors.tmp")
    save_file(
        {"sums": accumulator.sums, "counts": accumulator.counts},
        str(tmp_tensors),
        metadata={"emotions": json.dumps(accumulator.emotions)},
    )
    os.replace(tmp_tensors, tensors_path)

    payload = {
        "shard_index": int(shard_index),
        "emotions": accumulator.emotions,
        "d_model": accumulator.d_model,
        "n_contributed": accumulator.n_contributed,
        **meta,
    }
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return tensors_path, meta_path


def load_partial(directory: Path, shard_index: int) -> tuple[VectorAccumulator, dict] | None:
    """Read one shard's partial, or ``None`` if it is absent or incomplete."""
    from safetensors.numpy import load_file

    tensors_path, meta_path = _partial_paths(directory, shard_index)
    if not (tensors_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensors_path))
    accumulator = VectorAccumulator(
        emotions=list(meta["emotions"]),
        d_model=int(meta["d_model"]),
        sums=np.asarray(tensors["sums"], dtype=np.float64),
        counts=np.asarray(tensors["counts"], dtype=np.int64),
    )
    return accumulator, meta


def load_partials(directory: Path, num_shards: int) -> tuple[list[int], list[int]]:
    """``(present, missing)`` shard indices for an expected shard count."""
    present = [i for i in range(num_shards) if load_partial(directory, i) is not None]
    return present, [i for i in range(num_shards) if i not in present]


# --------------------------------------------------------------------------- #
# Skipped stimuli
# --------------------------------------------------------------------------- #

def skipped_records(activations_dir: Path) -> list[dict]:
    """Skipped-stimulus records across shards, deduplicated by example id.

    ``skipped.jsonl`` is append-only, so a resumed run can re-record an id; the
    count that matters is distinct ids. Read from disk rather than tracked in
    memory so the number survives a resume, and so it is the same number a human
    would get by reading the files.
    """
    seen: dict[str, dict] = {}
    for path in sorted(Path(activations_dir).glob("shard*/skipped.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                seen.setdefault(str(record.get("example_id")), record)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# R2 decision
# --------------------------------------------------------------------------- #

def decide_r2(config: PCAJLensConfig, estimated_bytes: int) -> tuple[bool, str]:
    """Resolve ``config.r2_sync`` into a decision + reason, aborting up front.

    Deliberately not the mean-difference pipeline's ``decide_r2``: that one tells
    you to "set r2_sync=False", which is exactly the wrong advice here. Experiment
    2's activations always belong in R2, so a missing credential is a thing to fix
    before the model loads -- not after an hour of GPU time, on a pod whose disk
    disappears at teardown.
    """
    from core.r2 import r2_available

    available, reason = r2_available()
    gib = estimated_bytes / 1024**3

    if config.r2_sync is False:
        return False, "disabled in config (activations will exist only on this disk)"
    if config.r2_sync is True:
        if not available:
            raise SystemExit(
                f"r2_sync=True but R2 is not usable: {reason}\n\n"
                "Aborting before the model loads. This experiment defaults to "
                "r2_sync=True because\n'silently local' is the failure mode that "
                "loses a run to a pod teardown.\n"
                "Fill in r2.env (see r2.env.example), or `python run.py r2 check` to "
                "diagnose."
            )
        return True, "enabled in config"
    if gib < config.r2_threshold_gib:
        return False, (
            f"auto: estimated {gib:.2f} GiB is below the "
            f"{config.r2_threshold_gib} GiB threshold"
        )
    if not available:
        return False, (
            f"auto: estimated {gib:.2f} GiB exceeds {config.r2_threshold_gib} GiB but "
            f"R2 is unusable ({reason}); keeping activations local only"
        )
    return True, (
        f"auto: estimated {gib:.2f} GiB exceeds {config.r2_threshold_gib} GiB and R2 "
        "is configured"
    )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract(
    config: PCAJLensConfig,
    todo: pd.DataFrame,
    model,
    tokenizer,
    layers: list[int],
    target: TargetLayer,
    accumulator: VectorAccumulator,
    half_of_topic: dict[int, int],
    shard_index: int,
    on_chunk_written=None,
) -> dict:
    """Forward-pass ``todo``, store pooled activations, fold the target block in.

    Structurally the mean-difference pipeline's extraction loop, with one addition:
    the target block's pooled vector is accumulated per (emotion, topic-half) as it
    goes. Storing and accumulating in the same pass is what makes the emotion
    vectors independent of whether the chunks survive locally.
    """
    import torch

    device = model_utils.model_input_device(model)
    out_dtype = model_utils.torch_dtype(config.activation_dtype)
    emotion_row = {emotion: i for i, emotion in enumerate(accumulator.emotions)}

    n_batches = int(np.ceil(len(todo) / config.batch_size))
    n_written = n_skipped = 0
    n_tokens_seen = 0
    t_start = time.time()

    with ActivationWriter(
        activations_dir=config.activations_dir,
        shard_index=shard_index,
        layers=layers,
        dtype=config.activation_dtype,
        chunk_size=config.chunk_size,
        on_chunk_written=on_chunk_written,
    ) as writer, torch.inference_mode():

        for batch_i in range(n_batches):
            batch = todo.iloc[batch_i * config.batch_size : (batch_i + 1) * config.batch_size]
            texts = model_utils.prepare_texts(
                batch["text"].tolist(),
                tokenizer,
                use_chat_template=config.use_chat_template,
            )
            encoded = tokenizer(
                texts,
                add_special_tokens=config.add_special_tokens,
                truncation=True,
                max_length=config.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # Pool on-device, then move only (batch x hidden) per selected layer.
            # Token-level activations for 65 layers never touch the host.
            pooled = pool_hidden_states(
                outputs.hidden_states,
                attention_mask,
                layers=layers,
                offset=config.token_offset,
                min_pooled_tokens=config.min_pooled_tokens,
                out_dtype=out_dtype,
            )
            del outputs

            keep = pooled.keep.numpy()
            n_tokens_seen += int(attention_mask.sum().item())

            records: list[dict] = []
            skipped: list[dict] = []
            for row_i, (_, row) in enumerate(batch.iterrows()):
                entry = {
                    "example_id": row["example_id"],
                    "source": row["source"],
                    "emotion": row["emotion"],
                    "topic": row["topic"],
                    "topic_id": int(row["topic_id"]),
                    "story_idx": int(row["story_idx"]),
                    "split": row["split"],
                    "content_sha1": row["content_sha1"],
                    "n_tokens": int(pooled.n_real_tokens[row_i]),
                    "n_pooled_tokens": int(pooled.n_pooled_tokens[row_i]),
                }
                if keep[row_i]:
                    records.append(entry)
                else:
                    skipped.append({
                        **entry,
                        "reason": (
                            f"only {int(pooled.n_pooled_tokens[row_i])} tokens remain "
                            f"after token_offset={config.token_offset}; "
                            f"min_pooled_tokens={config.min_pooled_tokens}"
                        ),
                    })

            if skipped:
                writer.record_skipped(skipped)
                n_skipped += len(skipped)

            if records:
                keep_idx = np.flatnonzero(keep)
                writer.add(
                    records,
                    {
                        layer: tensor[torch.from_numpy(keep_idx)]
                        for layer, tensor in pooled.pooled.items()
                    },
                )
                # Accumulate the *stored* dtype's values, upcast for the sum only.
                vectors = (
                    pooled.pooled[target.hidden_state][torch.from_numpy(keep_idx)]
                    .to(torch.float32).cpu().numpy()
                )
                accumulator.add(
                    np.asarray([emotion_row[r["emotion"]] for r in records], dtype=np.int64),
                    np.asarray([half_of_topic[r["topic_id"]] for r in records], dtype=np.int64),
                    vectors,
                )
                n_written += len(records)

            if (batch_i + 1) % config.log_every_batches == 0 or batch_i + 1 == n_batches:
                done = min((batch_i + 1) * config.batch_size, len(todo))
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9)
                print(
                    f"  [{done:>7,}/{len(todo):,}] "
                    f"{rate:6.1f} ex/s | {n_tokens_seen / max(elapsed, 1e-9):8.0f} tok/s | "
                    f"elapsed {elapsed / 60:5.1f}m | eta {eta / 60:5.1f}m | "
                    f"skipped {n_skipped}",
                    flush=True,
                )

    elapsed = time.time() - t_start
    return {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_requested": len(todo),
        "elapsed_s": round(elapsed, 2),
        "examples_per_s": round(n_written / max(elapsed, 1e-9), 3),
        "tokens_per_s": round(n_tokens_seen / max(elapsed, 1e-9), 1),
        "bytes_on_disk": sum(
            p.stat().st_size for p in Path(config.activations_dir).rglob("*") if p.is_file()
        ),
        "shard_index": shard_index,
        "finished": provenance.utc_timestamp(),
    }


def fold_stored_rows(
    config: PCAJLensConfig,
    accumulator: VectorAccumulator,
    example_ids: set[str],
    target: TargetLayer,
    half_of_topic: dict[int, int],
) -> int:
    """Fold already-extracted stimuli into ``accumulator`` by re-reading the store.

    The slow resume path, used when this shard's prior partial does not match what
    the store holds (a crash mid-run, or a changed ``--num-shards``). Reads only the
    target layer, so it streams one 5120-float row per stimulus rather than all 65.

    Requires the chunk tensors locally; :class:`~core.activation_store.ActivationStore`
    raises :class:`MissingChunkError` with the exact ``r2 pull`` command otherwise,
    which is a better outcome than guessing.
    """
    store = ActivationStore(config.activations_dir)
    rows = store.index[store.index["example_id"].isin(example_ids)]
    if rows.empty:
        return 0
    emotion_row = {emotion: i for i, emotion in enumerate(accumulator.emotions)}
    unknown = sorted(set(rows["emotion"]) - set(emotion_row))
    if unknown:
        raise SystemExit(
            f"stored activations carry emotions absent from the current stimulus "
            f"set: {unknown}.\nThe activations directory belongs to a different "
            "stimulus set; use a new run_name or --overwrite."
        )
    activations = store.load_layer(target.hidden_state, rows)
    accumulator.add(
        np.asarray([emotion_row[e] for e in rows["emotion"]], dtype=np.int64),
        np.asarray([half_of_topic[int(t)] for t in rows["topic_id"]], dtype=np.int64),
        activations,
    )
    return len(rows)


# --------------------------------------------------------------------------- #
# Emotion vectors and the reliability table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Reliability:
    """One emotion's split-half diagnostics."""

    emotion: str
    n_expected: int
    n_used: int
    n_half_a: int
    n_half_b: int
    norm: float
    cosine_raw: float | None
    cosine_centered: float | None

    def as_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "n_expected": self.n_expected,
            "n_used": self.n_used,
            "n_half_a": self.n_half_a,
            "n_half_b": self.n_half_b,
            "norm": self.norm,
            "cosine_raw": self.cosine_raw,
            "cosine_centered": self.cosine_centered,
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return None if denom == 0.0 else float(np.dot(a, b) / denom)


def build_vectors(
    accumulator: VectorAccumulator,
    expected_counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[Reliability], dict]:
    """Emotion vectors and their split-half reliability.

    Returns ``(full, halves, rows, report)`` where ``full`` is
    ``(n_emotions, d_model)`` and ``halves`` is ``(n_emotions, 2, d_model)``.

    The full vector is the count-weighted mean of the two halves, which is exactly
    the mean over all of that emotion's stimuli -- so nothing is approximated by
    only ever having accumulated halves.

    Both cosines are reported per emotion:

    ``cosine_raw``       between the two half-vectors as they are. Expect ~0.999
                         for everything: pooled residuals at one layer share a
                         large common component, so this number cannot fail and is
                         printed only so that is visible rather than mistaken for a
                         result.
    ``cosine_centered``  after subtracting each half's own cross-emotion mean. This
                         is the gate. It is what Phase 3 runs PCA on, and each half
                         being centred by its own mean keeps the halves independent.

    The centring mean is taken over the *emotional* rows only. Neutral is the
    circumplex origin rather than one of the axes' constituents, so folding it into
    the mean would drag the origin by ``1/(n+1)``; its own centred cosine is still
    reported, measured against that same mean.
    """
    counts = accumulator.counts
    n_emotions = len(accumulator.emotions)
    totals = counts.sum(axis=1)

    full = np.zeros((n_emotions, accumulator.d_model), dtype=np.float64)
    have = totals > 0
    if have.any():
        full[have] = accumulator.sums.sum(axis=1)[have] / totals[have][:, None]

    halves = np.zeros_like(accumulator.sums)
    for h in (0, 1):
        ok = counts[:, h] > 0
        if ok.any():
            halves[ok, h] = accumulator.sums[ok, h] / counts[ok, h][:, None]

    both = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    emotional = np.asarray([e != NEUTRAL_QUADRANT for e in accumulator.emotions])
    # Centre on the emotions that have *both* halves, so the two half-means are
    # taken over the same emotion set. Over different sets, part of the cosine drop
    # would be a shifted origin rather than unreliability.
    centre_rows = emotional & both
    centred = np.zeros_like(halves)
    can_centre = int(centre_rows.sum()) >= 2
    if can_centre:
        for h in (0, 1):
            centred[:, h] = halves[:, h] - halves[centre_rows, h].mean(axis=0)

    rows: list[Reliability] = []
    for i, emotion in enumerate(accumulator.emotions):
        rows.append(Reliability(
            emotion=emotion,
            n_expected=int(expected_counts.get(emotion, 0)),
            n_used=int(totals[i]),
            n_half_a=int(counts[i, 0]),
            n_half_b=int(counts[i, 1]),
            norm=float(np.linalg.norm(full[i])),
            cosine_raw=_cosine(halves[i, 0], halves[i, 1]) if both[i] else None,
            cosine_centered=(
                _cosine(centred[i, 0], centred[i, 1]) if both[i] and can_centre else None
            ),
        ))

    scored = [r.cosine_centered for r in rows if r.cosine_centered is not None]
    emotional_scored = [
        r.cosine_centered for r in rows
        if r.cosine_centered is not None and r.emotion != NEUTRAL_QUADRANT
    ]
    report = {
        "n_emotions": n_emotions,
        "n_scored": len(scored),
        "centering_rows": [
            e for i, e in enumerate(accumulator.emotions) if centre_rows[i]
        ],
        "centering_applied": can_centre,
        "min_cosine_centered": min(scored) if scored else None,
        "mean_cosine_centered": float(np.mean(scored)) if scored else None,
        "median_cosine_centered": float(np.median(scored)) if scored else None,
        "min_cosine_centered_emotional": min(emotional_scored) if emotional_scored else None,
        "mean_cosine_raw": (
            float(np.mean([r.cosine_raw for r in rows if r.cosine_raw is not None]))
            if any(r.cosine_raw is not None for r in rows) else None
        ),
        "worst": [
            r.emotion for r in
            sorted((r for r in rows if r.cosine_centered is not None),
                   key=lambda r: r.cosine_centered)[:5]
        ],
    }
    return full, halves, rows, report


def save_vectors(
    out_dir: Path,
    full: np.ndarray,
    halves: np.ndarray,
    emotions: list[str],
    metadata: dict,
    vectors_name: str,
    meta_name: str,
) -> tuple[Path, Path]:
    """Write the emotion vectors plus the sidecar that makes them interpretable.

    Both half-vectors are saved alongside the full one, so the reliability check can
    be redone -- or done at a different centring -- without re-running a 32B model.

    A few MB, under ``results/``, and readable with nothing but ``safetensors`` and
    ``json``: Phase 3 runs on a laptop CPU and must never need an activation chunk.
    """
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / vectors_name
    meta_path = out_dir / meta_name

    save_file(
        {
            "emotion_vectors": np.ascontiguousarray(full, dtype=VECTOR_DTYPE),
            "emotion_vectors_half_a": np.ascontiguousarray(halves[:, 0], dtype=VECTOR_DTYPE),
            "emotion_vectors_half_b": np.ascontiguousarray(halves[:, 1], dtype=VECTOR_DTYPE),
        },
        str(vectors_path),
        metadata={
            "emotions": json.dumps(emotions),
            "target_block": str(metadata["target"]["block"]),
            "target_hidden_state": str(metadata["target"]["hidden_state"]),
            "block_index_convention": "jlens residual block; hidden_state = block + 1",
            "dtype": np.dtype(VECTOR_DTYPE).name,
            "row_order": "emotion_vectors[i] belongs to emotions[i]",
        },
    )
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return vectors_path, meta_path


# --------------------------------------------------------------------------- #
# Aggregation across shards
# --------------------------------------------------------------------------- #

def aggregate_shards(
    directory: Path,
    shard_indices: list[int],
    emotions: list[str],
    d_model: int,
) -> tuple[VectorAccumulator, list[dict]]:
    """Sum every shard's partial into one accumulator.

    Raises:
        ValueError: If a partial was written against a different emotion set or
            dimension -- naming the shard, because "delete the partials and re-run"
            is only actionable if you know which one disagrees.
    """
    combined = VectorAccumulator.empty(emotions, d_model)
    metas: list[dict] = []
    for index in shard_indices:
        loaded = load_partial(directory, index)
        if loaded is None:  # pragma: no cover - callers check presence first
            raise ValueError(f"shard {index}'s partial vanished between listing and read")
        part, meta = loaded
        try:
            combined.merge(part)
        except ValueError as exc:
            raise ValueError(f"shard {index}: {exc}") from exc
        metas.append(meta)
    return combined, metas


def seed_accumulator(
    config: PCAJLensConfig,
    resumed: set[str],
    prior: tuple[VectorAccumulator, dict] | None,
    emotions: list[str],
    d_model: int,
    target: TargetLayer,
    half_of_topic: dict[int, int],
) -> tuple[VectorAccumulator, str]:
    """Account for stimuli extracted by an earlier run. Returns ``(acc, source)``.

    Fast path: this shard's prior partial, when its contribution count matches what
    the store holds. That equality is the whole check -- the partial is written once
    per clean shutdown, so matching counts mean it covers exactly the stored rows,
    and it needs no chunk files at all.

    Slow path: re-read the target layer from the store. Used after a crash mid-run
    or a ``--num-shards`` change, and it needs the chunks locally.
    """
    accumulator = VectorAccumulator.empty(emotions, d_model)
    if not resumed:
        if prior is not None:
            print("  a partial exists but the store has none of this shard's stimuli; "
                  "ignoring it as stale")
        return accumulator, "nothing to fold in (fresh shard)"

    if (
        prior is not None
        and prior[0].emotions == emotions
        and prior[0].d_model == d_model
        and prior[0].n_contributed == len(resumed)
    ):
        accumulator.merge(prior[0])
        return accumulator, (
            f"this shard's prior partial ({prior[0].n_contributed:,} stimuli); "
            "no chunks needed"
        )

    why = (
        "no prior partial" if prior is None
        else f"prior partial holds {prior[0].n_contributed:,} stimuli, store has "
             f"{len(resumed):,}"
    )
    print(f"  prior partial not usable ({why}); re-reading the target layer from the store")
    n_folded = fold_stored_rows(config, accumulator, resumed, target, half_of_topic)
    return accumulator, f"re-read {n_folded:,} stored stimuli from the activation store"


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def print_header(config: PCAJLensConfig, args, out_dir: Path) -> None:
    print(RULE)
    print(f"PHASE 2 GATE -- emotion vectors   run '{config.run_name}'")
    print(RULE)
    print(f"model      : {config.model_name} ({config.dtype})")
    print(f"stimuli    : {config.stimuli_path}")
    print(f"outputs    : {out_dir}")
    print(f"activations: {config.activations_dir}")
    print(f"r2 folder  : {config.resolved_r2_prefix()}")
    print()
    print("This is the ONLY phase that collects activations. Chunks mirror to R2 as")
    print("they are written; the small artefact Phase 3 reads is a few MB under")
    print("results/, so Phase 3 never needs an activation chunk.")
    if args.limit is not None:
        print()
        print(f"  --limit {args.limit}: a BENCHMARK. Its emotion vectors are the mean of an")
        print("  arbitrary prefix of the stimuli, not a result, so they are written to")
        print(f"  {out_dir.name}/ where nothing downstream will pick them up.")
        print("  Consider a throwaway --set run_name=... too: these stimuli land in the")
        print("  real activations directory and the full run then has to resume over them.")
    print()


def print_stimuli_summary(
    stimuli: pd.DataFrame,
    fingerprint: dict,
    emotions: list[str],
    per_emotion: dict[str, int],
) -> None:
    print(RULE)
    print("STEP 1  Phase 1's stimulus set")
    print(RULE)
    print(f"stimuli        : {len(stimuli):,} rows, {len(emotions)} emotion groups")
    print(f"  emotional    : {int((stimuli['source'] == 'emotion').sum()):,}")
    print(f"  neutral      : {int((stimuli['source'] == 'neutral').sum()):,}"
          f"   (labelled {NEUTRAL_QUADRANT!r} in the emotion column, not NA)")
    print(f"  topics       : {stimuli['topic_id'].nunique()}")
    print("  splits       : "
          + ", ".join(f"{k}={v:,}" for k, v in stimuli.groupby("split").size().items())
          + "   (all are used: nothing is fitted then held out here)")
    print(f"  content hash : {fingerprint['sha256'][:16]}...")
    seen = sorted(set(per_emotion.values()))
    shown = seen if len(seen) <= 4 else f"{seen[:2]}..{seen[-2:]}"
    verdict = "uniform" if len(seen) == 1 else "UNEVEN -- emotions carry different weight"
    print(f"  per group    : {shown}   ({verdict})")


def print_layer_plan(config: PCAJLensConfig, arch, layers: list[int], target: TargetLayer) -> bool:
    """Print the layer plan. Returns ``False`` if the target block is not stored."""
    print()
    print(RULE)
    print("STEP 2  Resolve the target block and the layers to store")
    print(RULE)
    print(f"model        : {arch.architectures}  n_layers={arch.n_layers}  "
          f"hidden_size={arch.hidden_size}")
    print(f"model sha    : {arch.resolved_sha}")
    print()
    print("the two layer-index conventions, which differ by one:")
    print(f"  jlens block index   : output of residual block l, 0..{arch.n_layers - 1}")
    print(f"  hidden-state index  : output_hidden_states, 0..{arch.n_hidden_states - 1} "
          "(0 = embeddings)")
    print("  conversion          : hidden_state_index = block_index + 1")
    print(f"  lens covers blocks  : 0..{target.max_lens_block} (= n_layers - 2); block "
          f"{arch.n_layers - 1} is the transport target, not a source")
    print()
    print(f"TARGET BLOCK : {target.describe()}")
    print(f"               resolved from {target.resolved_from}")
    print(f"               block index (jlens / lens / Phase 4) : {target.block}")
    print(f"               hidden-state index (pooling / store)  : {target.hidden_state}")
    print(f"               readable by the lens                  : "
          f"{target.block <= target.max_lens_block}")
    print()
    print(f"layers stored ({len(layers)}): "
          f"{layers if len(layers) <= 20 else f'{layers[:6]} ... {layers[-4:]}'}")
    print(f"  layer_spec={config.layer_spec!r}. All of them because output_hidden_states "
          "returns")
    print("  every layer anyway: no extra compute, and the Phase 5 layer sweep becomes a")
    print("  re-read instead of a second pass of a 32B model.")
    if arch.n_hidden_states - 1 in layers:
        print(f"  NOTE hidden state {arch.n_hidden_states - 1} is the last one, which "
              "transformers returns")
        print("       *after* the final norm, and it has no fitted J_l. Stored for "
              "completeness; do not lens it.")
    if target.hidden_state in layers:
        return True
    print()
    print(f"ABORTED: the target block's hidden state ({target.hidden_state}) is not among "
          "the stored layers.", file=sys.stderr)
    print(f"  layer_spec={config.layer_spec!r} resolved to {layers}.\n"
          "  Either widen it (layer_spec=all) or pick a target_block it covers.\n"
          "  Not auto-added on purpose: `layers` is part of the activation fingerprint, "
          "so\n"
          "  quietly extending it would make two runs with the same layer_spec "
          "incompatible\n"
          "  and destroy the free layer sweep.", file=sys.stderr)
    return False


def print_storage_plan(
    config: PCAJLensConfig,
    args,
    n_stimuli: int,
    n_shard: int,
    n_scope: int,
    storage: dict,
    use_r2: bool,
    r2_reason: str,
) -> None:
    print(f"shard {args.shard_index + 1}/{args.num_shards}: {n_shard:,} of "
          f"{n_stimuli:,} stimuli"
          + (f" (limited to {args.limit})" if args.limit is not None else ""))
    print(f"storage    : {human_bytes(storage['bytes_per_stimulus'])}/stimulus x "
          f"{n_shard:,} = {human_bytes(storage['estimated_bytes_shard'])} this shard")
    print(f"             {human_bytes(storage['estimated_bytes_total'])} for the whole run "
          f"({n_scope:,} stimuli, {config.activation_dtype})")
    print(f"R2 mirror  : {use_r2} ({r2_reason})")
    if use_r2:
        print(f"  prefix   : s3://{os.environ.get('R2_BUCKET')}/{config.resolved_r2_prefix()}")
        print(f"  delete local chunks after a verified upload: "
              f"{config.delete_local_after_sync}")
    elif config.r2_sync is not False:
        print("  WARNING activations will live only on this disk. On an ephemeral pod")
        print("          they are lost at teardown and cost the full GPU time to redo.")


def print_token_fit(config: PCAJLensConfig, tok_stats: dict) -> None:
    needed = config.token_offset + config.min_pooled_tokens
    print()
    print(RULE)
    print("STEP 4  Token lengths vs the pooling offset")
    print(RULE)
    print("token lengths (sampled): " + ", ".join(f"{k}={v}" for k, v in tok_stats.items()))
    print(f"  pooling keeps real tokens {config.token_offset + 1}.. ; a stimulus needs "
          f">= {needed} (min observed {tok_stats['min']:g})")
    if tok_stats["min"] < needed:
        print("  NOTE some stimuli will be skipped for length. Phase 1 predicted zero, so")
        print("       this is a disagreement to resolve before trusting the vectors.")
    if tok_stats["n_at_max_length"]:
        print(f"  NOTE {tok_stats['n_at_max_length']} sampled stimuli hit "
              f"max_length={config.max_length} and were truncated")


def print_half_plan(config: PCAJLensConfig, half_report: dict) -> None:
    print()
    print(RULE)
    print("STEP 5  The split-half partition (by TOPIC, not by story)")
    print(RULE)
    print(f"topics            : {half_report['n_topics']} -> "
          f"{half_report['n_topics_half_a']} + {half_report['n_topics_half_b']}")
    print(f"stimuli per half  : {half_report['n_stimuli_half_a']:,} + "
          f"{half_report['n_stimuli_half_b']:,}")
    print(f"derived from      : {half_report['derived_from']}, seed={config.seed}")
    print("                    (identical in every shard without communication -- if the")
    print("                     shards disagreed the halves would overlap and the gate")
    print("                     would pass on leaked scenarios)")
    print()
    print("Twelve stories share one topic and are near-paraphrases of one scenario, so a")
    print("story-level split would put variants of the same scenario in both halves and")
    print("measure paraphrase similarity rather than reliability.")


def print_skipped(skipped: list[dict]) -> None:
    """The loud report. Phase 1 predicted zero of these, so a nonzero count is news."""
    print()
    print("  SKIPPED STIMULI -- Phase 1 predicted zero of these.")
    print("  Its length check is tokenizer-free (~1.35 tokens/word) and the real")
    print("  tokenizer disagrees. The per-emotion counts are what matter: an uneven")
    print("  loss tilts the comparison between emotions rather than just shrinking it.")
    by_emotion: dict[str, int] = {}
    for record in skipped:
        key = str(record.get("emotion"))
        by_emotion[key] = by_emotion.get(key, 0) + 1
    for emotion, n in sorted(by_emotion.items()):
        print(f"    {emotion:<16} {n:>6,}")
    for record in skipped[:5]:
        print(f"    e.g. {record.get('example_id')}: {record.get('reason')}")
    if len(skipped) > 5:
        print(f"    ... and {len(skipped) - 5} more (full records in shard*/skipped.jsonl)")


def print_gate_explanation(reliability: dict) -> None:
    print()
    print(RULE)
    print("GATE  Split-half reliability per emotion")
    print(RULE)
    print("Each emotion's vector refitted from two disjoint halves of the TOPICS, then")
    print("compared by cosine. Two columns, because only one of them can fail:")
    print()
    print("  cos raw     : the two half-vectors as they are. Pooled residuals at one")
    print("                layer share a large common component, so this sits near")
    print("                0.999 for everything -- printed so that is visible rather")
    print("                than mistaken for a result.")
    print("  cos centred : after subtracting each half's own cross-emotion mean. This is")
    print("                what Phase 3 runs PCA on, so it is the gate. Each half being")
    print("                centred by its own mean keeps the two halves independent.")
    print()
    print(f"The centring mean is over {len(reliability['centering_rows'])} emotional rows. "
          "Neutral is the circumplex")
    print("origin rather than one of the axes' constituents, so it is excluded from that")
    print("mean -- but still scored against it.")
    print()


def print_reliability(rows: list[Reliability], report: dict, threshold: float) -> None:
    # 16 wide fits the longest word in data/emotions_171.txt ("grief-stricken", 14)
    # with room to spare, so columns do not shift on the 171-emotion run.
    width = max([16, *(len(r.emotion) + 2 for r in rows)])
    print(f"{'emotion':<{width}}{'n':>7}{'half A':>8}{'half B':>8}"
          f"{'|v|':>10}{'cos raw':>10}{'cos centred':>13}")
    print(THIN)
    for row in sorted(rows, key=lambda r: r.emotion):
        raw = "     n/a" if row.cosine_raw is None else f"{row.cosine_raw:8.4f}"
        centred = "      n/a" if row.cosine_centered is None else f"{row.cosine_centered:9.4f}"
        flag = ""
        if row.cosine_centered is not None and row.cosine_centered < threshold:
            flag = "  <- below threshold"
        if row.n_used != row.n_expected:
            short = row.n_expected - row.n_used
            flag += f"  [{short} stimul{'us' if short == 1 else 'i'} missing]"
        print(f"{row.emotion:<{width}}{row.n_used:>7,}{row.n_half_a:>8,}{row.n_half_b:>8,}"
              f"{row.norm:>10.2f}{raw:>10}{centred:>13}{flag}")
    print(THIN)
    if report["min_cosine_centered"] is None:
        print("  no emotion had stimuli in both halves; the gate could not be scored.")
        return
    print(f"  centred split-half cosine: min {report['min_cosine_centered']:.4f}, "
          f"mean {report['mean_cosine_centered']:.4f}, "
          f"median {report['median_cosine_centered']:.4f}   "
          f"({report['n_scored']} of {report['n_emotions']} scored)")
    print(f"  raw (uncentred) mean     : {report['mean_cosine_raw']:.4f}  "
          "<- near 1 by construction; not the gate")
    print(f"  weakest emotions         : {', '.join(report['worst'])}")


def print_verdict(
    config: PCAJLensConfig,
    target: TargetLayer,
    reliability: dict,
    counts: dict,
    artifacts: dict[str, object],
    use_r2: bool,
    bytes_on_disk: int,
) -> bool:
    """Print the verdict block. Returns whether the reliability gate passed."""
    min_centered = reliability["min_cosine_centered"]
    gate_pass = (
        min_centered is not None
        and min_centered >= config.split_half_min_cosine
        and reliability["n_scored"] == reliability["n_emotions"]
    )
    print()
    print(RULE)
    print("PHASE 2 VERDICT")
    print(RULE)
    if min_centered is None:
        print("  split-half reliability : NOT SCORED (no emotion had both halves)")
    else:
        print(f"  split-half reliability : {'PASS' if gate_pass else 'REVIEW'} "
              f"(min {min_centered:.3f}, mean "
              f"{reliability['mean_cosine_centered']:.3f}, threshold "
              f"{config.split_half_min_cosine:.2f})")
    print(f"  stimuli accounted for  : {'PASS' if not counts['skipped'] else 'REVIEW'} "
          f"({counts['accumulated']:,} accumulated, {counts['skipped']:,} skipped, "
          "0 unaccounted)")
    print(f"  target block           : {target.describe()}")
    print(f"  activations            : {human_bytes(bytes_on_disk)} local, mirrored to "
          f"{config.resolved_r2_prefix() if use_r2 else 'nowhere (r2 off)'}")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<8}: {path}" if label else f"  {'':<8}  {path}")
    print()
    if min_centered is not None and min_centered < config.split_half_min_cosine:
        print("  Below threshold. The brief's reading: ~0.9 is trustworthy, ~0.6 means more")
        print("  stimuli are needed before PCA can mean anything. The fix is more stimuli")
        print("  per emotion, not a lower threshold:")
        print("    python run.py phase1 --set stories_per_emotion=800")
        print("    python run.py phase2 --set stories_per_emotion=800 --set run_name=<new>")
        print("  (a changed stimulus set needs a new run_name -- see the fingerprint.)")
        print()
    print("  The threshold is a summary, not the judgement. Read the per-emotion table:")
    print("  one weak emotion is a stimulus problem for that word, while a uniformly low")
    print("  column means the target block carries little emotion structure and a")
    print("  different block is the thing to try.")
    print()
    print("STOPPING at the Phase 2 gate, as agreed. Phase 3 (PCA) has not run.")
    print(RULE)
    return gate_pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    # r2.env carries HF_TOKEN as well as the R2 credentials, so it must be loaded
    # before the tokenizer/model are fetched -- and this stage must work standalone
    # (`python -m emotion_pca_jlens.phase2_vectors`), not only via run.py.
    env_file.load_env_file()

    cache_dir = paths.hf_cache_dir()
    out_dir = (
        config.phase_dir if args.limit is None
        else config.phase_dir / f"benchmark_limit{args.limit}"
    )
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index:03d}"
    print_header(config, args, out_dir)

    # --- the stimulus set ------------------------------------------------ #
    stimuli = read_stimuli(config)
    fingerprint_stimuli = stimuli_fingerprint(stimuli)
    emotions = sorted(stimuli["emotion"].unique().tolist())
    per_emotion = {str(k): int(v) for k, v in stimuli.groupby("emotion").size().items()}
    print_stimuli_summary(stimuli, fingerprint_stimuli, emotions, per_emotion)

    # --- layers ---------------------------------------------------------- #
    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )
    layers = model_utils.resolve_layers(config.layer_spec, arch.n_hidden_states)
    target = resolve_target_layer(config, arch.n_layers)
    if not print_layer_plan(config, arch, layers, target):
        return 2

    # --- shard, scope, storage, R2 --------------------------------------- #
    print()
    print(RULE)
    print("STEP 3  Shard, storage estimate, and the R2 destination")
    print(RULE)
    shard = assign_shard(stimuli, args.num_shards, args.shard_index)
    if args.limit is not None:
        shard = shard.iloc[: args.limit].reset_index(drop=True)
    scope = scope_table(stimuli, args.num_shards, args.limit)
    # Per-emotion counts the gate compares against come from the *scope*, not the
    # whole table: under --limit the run only intends to cover a prefix, so the full
    # count would flag every emotion as short and bury a real shortfall.
    per_emotion_scope = {str(k): int(v) for k, v in scope.groupby("emotion").size().items()}

    nbytes = model_utils.dtype_nbytes(config.activation_dtype)
    storage = {
        "bytes_per_stimulus": len(layers) * arch.hidden_size * nbytes,
        "estimated_bytes_shard": estimate_storage_bytes(
            len(shard), len(layers), arch.hidden_size, nbytes),
        "estimated_bytes_total": estimate_storage_bytes(
            len(scope), len(layers), arch.hidden_size, nbytes),
    }
    use_r2, r2_reason = decide_r2(config, storage["estimated_bytes_total"])
    print_storage_plan(config, args, len(stimuli), len(shard), len(scope),
                       storage, use_r2, r2_reason)

    # --- tokens vs the pooling offset ------------------------------------ #
    tokenizer = None
    try:
        tokenizer = model_utils.load_tokenizer(
            config.model_name, config.model_revision, cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        if not args.dry_run:
            raise
        print(f"\nWARNING tokenizer unavailable ({exc});")
        print("        estimating token lengths from word counts instead")

    sample = (shard if len(shard) else stimuli)["text"].tolist()
    if tokenizer is not None:
        tok_stats = token_length_stats(
            model_utils.prepare_texts(
                sample, tokenizer, use_chat_template=config.use_chat_template
            ),
            tokenizer, config.max_length, config.add_special_tokens, seed=config.seed,
        )
    else:
        tok_stats = estimate_tokens_without_tokenizer(sample, config.max_length)
    print_token_fit(config, tok_stats)

    # --- the topic halving ----------------------------------------------- #
    half_of_topic, half_report = topic_halves(stimuli, config.seed)
    print_half_plan(config, half_report)

    # --- run record ------------------------------------------------------ #
    fingerprint = config.fingerprint(
        layers, arch.hidden_size, arch.resolved_sha, fingerprint_stimuli
    )
    sections: dict = {
        "run": {
            "stage": "phase2_vectors",
            "run_name": config.run_name,
            "dry_run": args.dry_run,
            "limit": args.limit,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "output_dir": str(out_dir),
        },
        "config": config.to_dict(),
        "resolved": {
            "target_block": target.block,
            "target_hidden_state": target.hidden_state,
            "target_resolved_from": target.resolved_from,
            "highest_lens_block": target.max_lens_block,
            "layers": layers,
            "n_layers_stored": len(layers),
            "model_n_layers": arch.n_layers,
            "hidden_size": arch.hidden_size,
            "model_sha": arch.resolved_sha,
            "model_architectures": list(arch.architectures),
            "r2_sync": use_r2,
            "r2_reason": r2_reason,
            "r2_prefix": config.resolved_r2_prefix() if use_r2 else None,
            "r2_bucket": os.environ.get("R2_BUCKET") if use_r2 else None,
            # Which credentials file was in play: the bucket is not a config field,
            # so without this a record cannot say where the activations went.
            "env_file": str(record.path) if (record := env_file.loaded()) and record.path else None,
            **storage,
        },
        "stimuli": {
            "path": str(config.stimuli_path),
            "n_rows": len(stimuli),
            "n_in_scope": len(scope),
            "n_this_shard": len(shard),
            "emotions": emotions,
            "per_emotion_counts": per_emotion,
            "per_split_counts": {
                str(k): int(v) for k, v in stimuli.groupby("split").size().items()
            },
            **fingerprint_stimuli,
        },
        "split_half": half_report,
        "token_stats": tok_stats,
        "fingerprint": fingerprint,
    }

    if args.dry_run:
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run",
            title=f"PHASE 2 DRY RUN -- {config.run_name}",
            sections=sections,
            txt_name=f"phase2_dry_run{suffix}.txt",
            json_name=f"phase2_dry_run{suffix}.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: stimuli, layers and storage validated; no weights loaded.")
        print(RULE)
        print(f"  target block : {target.describe()}")
        print(f"  storage      : {human_bytes(storage['estimated_bytes_total'])} for the "
              f"run ({human_bytes(storage['estimated_bytes_shard'])} this shard)")
        print(f"  R2           : {use_r2} -- {r2_reason}")
        print(f"  records      : {txt_path}")
        print(f"                 {json_path}")
        print()
        print("Re-run without --dry-run to extract. Get a throughput number first:")
        print("  python run.py phase2 --limit 256")
        print(RULE)
        return 0

    # --- resume ----------------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 6  Resume state")
    print(RULE)
    partials = partials_dir(out_dir)
    if args.overwrite and partials.exists():
        # init_or_check_manifest deletes the activations directory below; the
        # partials are derived from it and would otherwise be a stale accumulator
        # silently double-counting into the new run.
        print(f"--overwrite: removing stale accumulator partials in {partials}")
        shutil.rmtree(partials)

    try:
        init_or_check_manifest(
            config.activations_dir,
            fingerprint=fingerprint,
            extra={
                "created": provenance.utc_timestamp(),
                "run_name": config.run_name,
                "stage": "phase2_vectors",
                "layers": layers,
                "hidden_size": arch.hidden_size,
                "activation_dtype": config.activation_dtype,
                "token_offset": config.token_offset,
                # Recorded, not fingerprinted: the store holds every layer, so the
                # target is an analysis choice rather than part of what a stored
                # vector means.
                "target_block": target.block,
                "target_hidden_state": target.hidden_state,
                "packages": provenance.package_versions(),
            },
            allow_overwrite=args.overwrite,
        )
    except activation_store.IncompatibleRunError as exc:
        sys.stdout.flush()
        print(f"\nABORTED -- incompatible existing run\n\n{exc}\n", file=sys.stderr)
        print("If the stimuli_* fields differ, the stimulus set changed. That is not an\n"
              "extension of this run: every emotion vector is a mean over exactly one\n"
              "stimulus set, so mixing two would silently change what each vector is a\n"
              "mean of. Use a new run_name.\n", file=sys.stderr)
        return 3

    already = activation_store.completed_example_ids(config.activations_dir)
    shard_ids = set(shard["example_id"])
    resumed = shard_ids & already
    # Skipped stimuli are absent from the chunk index, so resuming on `already`
    # alone would re-process them every run -- loading 65 GiB of weights to re-derive
    # the same "too short after the offset" verdict. They cannot become eligible
    # without changing token_offset/min_pooled_tokens/max_length, and all three are
    # in the fingerprint, so a change aborts above rather than arriving here.
    # Excluding them is what makes re-running a finished shard a real no-op.
    permanently_skipped = shard_ids & {
        str(r.get("example_id")) for r in skipped_records(config.activations_dir)
    }
    todo = shard[~shard["example_id"].isin(already | permanently_skipped)].reset_index(drop=True)

    try:
        accumulator, fold_source = seed_accumulator(
            config, resumed, load_partial(partials, args.shard_index),
            emotions, arch.hidden_size, target, half_of_topic,
        )
    except activation_store.MissingChunkError as exc:
        print(f"\nABORTED -- cannot rebuild the accumulator\n\n{exc}\n", file=sys.stderr)
        print("Alternatively re-extract from scratch with --overwrite.", file=sys.stderr)
        return 3

    print(f"already stored (all shards) : {len(already):,}")
    print(f"already stored (this shard) : {len(resumed):,}")
    if permanently_skipped:
        print(f"already skipped (too short) : {len(permanently_skipped):,}  "
              "(not retried: token_offset is fingerprinted, so it cannot have changed)")
    print(f"still to extract            : {len(todo):,}")
    print(f"accumulator seeded from     : {fold_source}")

    # --- extract ---------------------------------------------------------- #
    stats: dict = {
        "n_written": 0, "n_skipped": 0, "n_requested": 0, "elapsed_s": 0.0,
        "bytes_on_disk": sum(
            p.stat().st_size for p in config.activations_dir.rglob("*") if p.is_file()
        ),
        "shard_index": args.shard_index,
        "note": "nothing to extract for this shard",
    }
    if todo.empty:
        print("\nNothing left to extract for this shard; recomputing the vectors only.")
    else:
        print()
        print(RULE)
        print(f"STEP 7  Extract {len(todo):,} stimuli")
        print(RULE)
        print(f"Loading {config.model_name} ({config.dtype}, "
              f"device_map={config.device_map}) ...")
        t_load = time.time()
        model = model_utils.load_model(
            config.model_name,
            revision=config.model_revision,
            cache_dir=cache_dir,
            dtype=config.dtype,
            device_map=config.device_map,
            quantization=config.quantization,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        )
        print(f"  loaded in {time.time() - t_load:.0f}s; input device "
              f"{model_utils.model_input_device(model)}")

        on_chunk = None
        if use_r2:
            from core.r2 import make_chunk_uploader

            on_chunk = make_chunk_uploader(
                config.resolved_r2_prefix(),
                config.activations_dir,
                delete_local=config.delete_local_after_sync,
            )

        stats = extract(
            config=config, todo=todo, model=model, tokenizer=tokenizer, layers=layers,
            target=target, accumulator=accumulator, half_of_topic=half_of_topic,
            shard_index=args.shard_index, on_chunk_written=on_chunk,
        )
        print()
        print(f"  written : {stats['n_written']:,}")
        print(f"  skipped : {stats['n_skipped']:,}")
        print(f"  elapsed : {stats['elapsed_s'] / 60:.1f} min "
              f"({stats['examples_per_s']:.1f} stimuli/s, {stats['tokens_per_s']:.0f} tok/s)")
        print(f"  on disk : {human_bytes(stats['bytes_on_disk'])}")

    # The partial is the durable form of the accumulator, so it is written before
    # the R2 sweep: a sweep that hangs must not be what loses it.
    tensors_path, _ = save_partial(
        partials, args.shard_index, accumulator,
        meta={
            "run_name": config.run_name,
            "num_shards": args.num_shards,
            "limit": args.limit,
            "target_block": target.block,
            "target_hidden_state": target.hidden_state,
            "fold_source": fold_source,
            "written": provenance.utc_timestamp(),
            "extraction": stats,
        },
    )
    print(f"\naccumulator partial: {tensors_path.name} "
          f"({accumulator.n_contributed:,} stimuli)")

    provenance.write_run_record(
        out_dir,
        title=f"PHASE 2 EXTRACTION -- {config.run_name}",
        sections={"result": stats, "run": sections["run"], "fingerprint": fingerprint},
        txt_name=f"phase2_extract{suffix}.txt",
        json_name=f"phase2_extract{suffix}.json",
    )

    if use_r2:
        print("\nFinal R2 sweep (catching anything a failed upload left behind) ...")
        from core.r2 import R2Client

        result = R2Client.from_env().sync_up(
            config.activations_dir, config.resolved_r2_prefix(),
            delete_local=False, verbose=False,
        )
        print(f"  uploaded {result['uploaded']}, already present {result['skipped']}, "
              f"{result['bytes'] / 1024**3:.2f} GiB")

    # --- combine the shards ----------------------------------------------- #
    print()
    print(RULE)
    print("STEP 8  Combine the shards into one vector per emotion")
    print(RULE)
    present, missing = load_partials(partials, args.num_shards)
    print(f"accumulator partials present: {present}")
    if missing:
        print(f"still missing               : {missing}")
        print()
        print(RULE)
        print(f"{len(missing)} of {args.num_shards} shards have not finished. An emotion")
        print("vector is a mean over ALL stimuli, so it cannot be written yet. Whichever")
        print("shard finishes last will aggregate and print the gate.")
        print(f"  this shard's partial: {tensors_path}")
        print(RULE)
        return 0

    try:
        combined, shard_meta = aggregate_shards(
            partials, present, emotions, arch.hidden_size
        )
    except ValueError as exc:
        print(f"\nABORTED -- incompatible accumulator partials\n\n{exc}\n", file=sys.stderr)
        print("They were written against different stimulus sets. Delete\n"
              f"{partials}\nand re-run every shard.", file=sys.stderr)
        return 3

    # Intersected with the scope because skipped.jsonl outlives a --limit run: a
    # stimulus skipped by an earlier, wider run is not this run's business.
    scope_ids = set(scope["example_id"])
    skipped = [
        r for r in skipped_records(config.activations_dir)
        if r.get("example_id") in scope_ids
    ]
    n_accumulated = combined.n_contributed
    n_unaccounted = len(scope) - n_accumulated - len(skipped)

    print(f"stimuli in scope : {len(scope):,}")
    print(f"accumulated      : {n_accumulated:,}")
    print(f"skipped (short)  : {len(skipped):,}")
    print(f"unaccounted      : {n_unaccounted:,}")
    if skipped:
        print_skipped(skipped)

    if n_unaccounted != 0:
        print()
        print("ABORTED: the accumulator does not account for every stimulus in scope.",
              file=sys.stderr)
        print(f"  {n_unaccounted} stimuli are neither accumulated nor recorded as "
              "skipped.\n"
              "  Every shard reported a partial, so this is a bookkeeping inconsistency\n"
              "  -- most likely a process killed between a chunk commit and its partial\n"
              "  update, or a --num-shards change mid-run.\n"
              "  Refusing to write emotion vectors: a mean over a silently truncated\n"
              "  stimulus set is undetectable downstream.\n"
              "  Rebuild from the chunks, or re-extract:\n"
              f"    python run.py r2 pull {config.activations_dir} "
              f"--prefix {config.resolved_r2_prefix()}\n"
              f"    rm -r {partials} && python run.py phase2\n"
              "  or start clean: python run.py phase2 --overwrite", file=sys.stderr)
        return 3

    # --- the gate --------------------------------------------------------- #
    full, halves, rows, reliability = build_vectors(combined, per_emotion_scope)
    print_gate_explanation(reliability)
    print_reliability(rows, reliability, config.split_half_min_cosine)

    counts = {
        "in_scope": len(scope),
        "accumulated": n_accumulated,
        "skipped": len(skipped),
        "per_emotion_expected": per_emotion_scope,
    }
    metadata = {
        "run": sections["run"],
        "target": {
            "block": target.block,
            "hidden_state": target.hidden_state,
            "description": target.describe(),
            "n_layers": arch.n_layers,
            "highest_lens_block": target.max_lens_block,
            "resolved_from": target.resolved_from,
            "convention": "block index is jlens/residual-block; hidden_state = block + 1",
        },
        "emotions": combined.emotions,
        "row_order": "emotion_vectors[i] belongs to emotions[i]",
        "d_model": combined.d_model,
        "dtype": np.dtype(VECTOR_DTYPE).name,
        "labels": {
            str(emotion): {
                "quadrant": str(group["quadrant"].iloc[0]),
                "valence": int(group["valence"].iloc[0]),
                "arousal": int(group["arousal"].iloc[0]),
                "family": str(group["family"].iloc[0]),
                "source": str(group["source"].iloc[0]),
            }
            for emotion, group in stimuli.groupby("emotion", sort=True)
        },
        "counts": counts,
        "split_half": {
            **half_report,
            "threshold": config.split_half_min_cosine,
            "summary": reliability,
            "per_emotion": [row.as_dict() for row in rows],
        },
        "fingerprint": fingerprint,
        "stimuli": sections["stimuli"],
        "shards": shard_meta,
        "written": provenance.utc_timestamp(),
    }
    vectors_path, vectors_meta_path = save_vectors(
        out_dir, full, halves, combined.emotions, metadata,
        vectors_name=config.emotion_vectors_path.name,
        meta_name=config.emotion_vectors_meta_path.name,
    )
    reliability_csv = out_dir / "phase2_split_half.csv"
    pd.DataFrame([row.as_dict() for row in rows]).to_csv(reliability_csv, index=False)

    sections["result"] = {
        "n_accumulated": n_accumulated,
        "n_skipped": len(skipped),
        "shards": present,
    }
    sections["split_half"] = metadata["split_half"]
    sections["artifacts"] = {
        "emotion_vectors": str(vectors_path),
        "emotion_vectors_metadata": str(vectors_meta_path),
        "split_half_csv": str(reliability_csv),
        "partials_dir": str(partials),
        "activations_dir": str(config.activations_dir),
    }
    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 2 GATE -- {config.run_name}",
        sections=sections,
        txt_name="phase2_gate.txt",
        json_name="phase2_gate.json",
    )

    print_verdict(
        config, target, reliability, counts,
        artifacts={
            "vectors": vectors_path,
            "metadata": vectors_meta_path,
            "table": reliability_csv,
            "records": txt_path,
            "": json_path,   # continuation of "records"; both are one record pair
        },
        use_r2=use_r2,
        bytes_on_disk=stats.get("bytes_on_disk", 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
