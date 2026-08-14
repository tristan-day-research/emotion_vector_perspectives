"""Stage 1: extract pooled residual-stream activations for every selected story.

Usage
-----
::

    # Validate config + estimate storage, no model download
    python -m extract_emotion_vectors.extract_activations --dry-run

    # Small GPU benchmark
    python -m extract_emotion_vectors.extract_activations --limit 256

    # Full run
    python -m extract_emotion_vectors.extract_activations

    # Four GPUs, four independent shards (one process per GPU)
    CUDA_VISIBLE_DEVICES=0 python -m extract_emotion_vectors.extract_activations \
        --num-shards 4 --shard-index 0 --set device_map=None
    ...

Re-running the same command resumes: examples already present in a completed
chunk are skipped. Changing anything that affects what an activation *means*
aborts instead of mixing incompatible vectors (see
``VectorExtractionConfig.fingerprint``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import activation_store, dataset, model_utils, paths, provenance
from core.activation_store import (
    ActivationWriter,
    estimate_storage_bytes,
    human_bytes,
    init_or_check_manifest,
)
from core.pooling import pool_hidden_states
from core.seeds import set_global_seeds
from extract_emotion_vectors.vector_extraction_config import CONFIG, VectorExtractionConfig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract pooled residual-stream activations for emotional and neutral stories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config, load + validate the dataset, estimate storage; never loads the model",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N examples (per shard) -- for GPU benchmarking",
    )
    p.add_argument("--num-shards", type=int, default=1, help="split the dataset into N shards")
    p.add_argument("--shard-index", type=int, default=0, help="which shard this process handles")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing incompatible activations directory instead of aborting",
    )
    p.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="JSON file of config overrides (keys must be config field names)",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a single config field; repeatable",
    )
    return p


def load_config(args: argparse.Namespace) -> VectorExtractionConfig:
    overrides: dict[str, object] = {}
    if args.config_json:
        overrides.update(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value
    config = CONFIG.with_overrides(overrides) if overrides else CONFIG

    problems = config.validate()
    if problems:
        print("Configuration problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(2)
    return config


# --------------------------------------------------------------------------- #
# Sharding
# --------------------------------------------------------------------------- #

def assign_shard(examples: pd.DataFrame, num_shards: int, shard_index: int) -> pd.DataFrame:
    """Deterministic round-robin over the globally sorted example table.

    Round-robin (rather than contiguous blocks) keeps every shard's mix of
    emotions, topics and splits balanced, so a partially finished sweep is still
    usable for a coarse look at all emotions.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    if num_shards == 1:
        return examples.reset_index(drop=True)
    mask = (np.arange(len(examples)) % num_shards) == shard_index
    return examples.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Token statistics (used by --dry-run and to sanity-check the offset)
# --------------------------------------------------------------------------- #

def token_length_stats(
    texts: list[str],
    tokenizer,
    max_length: int,
    add_special_tokens: bool,
    sample: int = 400,
    seed: int = 0,
) -> dict:
    """Token-length distribution over a deterministic sample of ``texts``."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))[: min(sample, len(texts))]
    lengths = [
        len(tokenizer(texts[int(i)], add_special_tokens=add_special_tokens,
                      truncation=True, max_length=max_length)["input_ids"])
        for i in idx
    ]
    arr = np.asarray(lengths)
    return {
        "n_sampled": int(arr.size),
        "min": int(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "n_at_max_length": int((arr >= max_length).sum()),
    }


def estimate_tokens_without_tokenizer(texts: list[str], max_length: int) -> dict:
    """Fallback when the tokenizer cannot be loaded (offline dry-run)."""
    words = np.asarray([len(t.split()) for t in texts], dtype=float)
    approx = np.minimum(words * 1.35, max_length)  # ~1.35 tokens/word for English prose
    return {
        "n_sampled": int(approx.size),
        "min": float(approx.min()),
        "p5": float(np.percentile(approx, 5)),
        "median": float(np.median(approx)),
        "p95": float(np.percentile(approx, 95)),
        "max": float(approx.max()),
        "mean": float(approx.mean()),
        "n_at_max_length": int((approx >= max_length).sum()),
        "note": "estimated from word counts (tokenizer unavailable); ~1.35 tokens/word",
    }


# --------------------------------------------------------------------------- #
# R2 decision
# --------------------------------------------------------------------------- #

def decide_r2(config: VectorExtractionConfig, estimated_bytes: int) -> tuple[bool, str]:
    """Resolve ``config.r2_sync`` ("auto"/True/False) into a decision + reason."""
    from core.r2 import r2_available

    available, reason = r2_available()
    gib = estimated_bytes / 1024**3

    if config.r2_sync is False:
        return False, "disabled in config"
    if config.r2_sync is True:
        if not available:
            raise SystemExit(
                f"r2_sync=True but R2 is not usable: {reason}\n"
                "Fill in .env (see .env.example) or set r2_sync=False."
            )
        return True, "enabled in config"
    # "auto"
    if gib < config.r2_threshold_gib:
        return False, (
            f"auto: estimated {gib:.2f} GiB is below the {config.r2_threshold_gib} GiB threshold"
        )
    if not available:
        return False, (
            f"auto: estimated {gib:.2f} GiB exceeds the {config.r2_threshold_gib} GiB threshold, "
            f"but R2 is unusable ({reason}); keeping activations local only"
        )
    return True, f"auto: estimated {gib:.2f} GiB exceeds {config.r2_threshold_gib} GiB and R2 is configured"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)

    cache_dir = paths.hf_cache_dir()
    print("=" * 78)
    print(f"Emotion vector extraction -- run '{config.run_name}'")
    print("=" * 78)
    print(f"model      : {config.model_name} (revision={config.model_revision or 'main'})")
    print(f"output dir : {config.output_dir}")
    print(f"hf cache   : {cache_dir}")
    print()

    # --- dataset -------------------------------------------------------- #
    print("Loading dataset (ryancodrai/emotion-probes, expression/*.parquet) ...")
    dataset_sha = dataset.dataset_revision(config.dataset_revision)
    examples, topic_split, report = dataset.build_example_table(
        emotions=config.emotions,
        stories_per_emotion=config.stories_per_emotion,
        neutral_stories=config.neutral_stories,
        split_seed=config.split_seed,
        split_proportions=config.split_proportions,
        revision=config.dataset_revision,
        cache_dir=cache_dir,
    )
    print(dataset.format_validation_report(report))
    print()
    print(dataset.summarise_examples(examples))
    print(
        f"  topic split (seed={topic_split.seed}, proportions={topic_split.proportions}): "
        + ", ".join(f"{k}={v}" for k, v in topic_split.counts.items())
        + " topics"
    )
    print()

    # --- architecture / layers ------------------------------------------ #
    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )
    layers = model_utils.resolve_layers(config.layer_spec, arch.n_hidden_states)
    print(
        f"Architecture: {arch.architectures or 'unknown'}  "
        f"n_layers={arch.n_layers}  hidden_size={arch.hidden_size}  "
        f"hidden_states=0..{arch.n_hidden_states - 1}"
    )
    print(f"Layers selected ({len(layers)}): {layers if len(layers) <= 20 else f'{layers[:8]} ... {layers[-4:]}'}")
    print(f"Model sha  : {arch.resolved_sha}")
    print(f"Dataset sha: {dataset_sha}")
    print()

    # --- shard ---------------------------------------------------------- #
    shard = assign_shard(examples, args.num_shards, args.shard_index)
    if args.limit is not None:
        shard = shard.iloc[: args.limit].reset_index(drop=True)
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: {len(shard):,} of {len(examples):,} examples"
        + (f" (limited to {args.limit})" if args.limit is not None else "")
    )

    # --- storage estimate ----------------------------------------------- #
    nbytes = model_utils.dtype_nbytes(config.activation_dtype)
    shard_bytes = estimate_storage_bytes(len(shard), len(layers), arch.hidden_size, nbytes)
    total_bytes = estimate_storage_bytes(len(examples), len(layers), arch.hidden_size, nbytes)
    per_example = len(layers) * arch.hidden_size * nbytes
    print(
        f"Storage    : {human_bytes(per_example)}/example x {len(shard):,} "
        f"= {human_bytes(shard_bytes)} this shard; {human_bytes(total_bytes)} for the full run"
    )

    use_r2, r2_reason = decide_r2(config, total_bytes)
    print(f"R2 mirror  : {use_r2} ({r2_reason})")
    if use_r2:
        print(f"  prefix   : s3://{__import__('os').environ.get('R2_BUCKET')}/{config.resolved_r2_prefix()}")
        print(f"  delete local chunks after upload: {config.delete_local_after_sync}")
    print()

    # --- tokenizer / token stats ---------------------------------------- #
    tokenizer = None
    try:
        tokenizer = model_utils.load_tokenizer(
            config.model_name, config.model_revision, cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        if not args.dry_run:
            raise
        print(f"WARNING: tokenizer unavailable ({exc}); estimating token lengths from word counts")

    sample_stories = shard["story"].tolist() if len(shard) else examples["story"].tolist()
    if tokenizer is not None:
        texts = model_utils.prepare_texts(
            sample_stories, tokenizer,
            use_chat_template=config.use_chat_template,
            chat_role=config.chat_role,
            chat_add_generation_prompt=config.chat_add_generation_prompt,
            prefix=config.text_prefix, suffix=config.text_suffix,
        )
        tok_stats = token_length_stats(
            texts, tokenizer, config.max_length, config.add_special_tokens, seed=config.seed
        )
    else:
        tok_stats = estimate_tokens_without_tokenizer(sample_stories, config.max_length)

    print("Token lengths (sampled): " + ", ".join(f"{k}={v}" for k, v in tok_stats.items()))
    at_risk = config.token_offset + config.min_pooled_tokens
    print(
        f"  pooling keeps real tokens {config.token_offset + 1}.. ; examples need "
        f">= {at_risk} tokens (min observed {tok_stats['min']:g})"
    )
    if tok_stats["min"] < at_risk:
        print("  NOTE: some examples will be skipped for insufficient length (logged to skipped.jsonl)")
    if tok_stats["n_at_max_length"]:
        print(
            f"  NOTE: {tok_stats['n_at_max_length']} sampled examples hit max_length="
            f"{config.max_length} and were truncated"
        )
    print()

    # --- run record ----------------------------------------------------- #
    fingerprint = config.fingerprint(layers, arch.hidden_size, arch.resolved_sha, dataset_sha)
    sections = {
        "run": {
            "run_name": config.run_name,
            "stage": "extract_activations",
            "dry_run": args.dry_run,
            "limit": args.limit,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "output_dir": str(config.output_dir),
        },
        "config": config.to_dict(),
        "resolved": {
            "layers": layers,
            "n_layers_selected": len(layers),
            "hidden_size": arch.hidden_size,
            "model_n_layers": arch.n_layers,
            "model_architectures": list(arch.architectures),
            "model_sha": arch.resolved_sha,
            "model_config_dtype": arch.config_dtype,
            "dataset_sha": dataset_sha,
            "dataset_id": dataset.HF_DATASET_ID,
            "dataset_files": [dataset.EXPRESSION_STORIES_FILE, dataset.EXPRESSION_NEUTRAL_FILE],
            "r2_sync": use_r2,
            "r2_reason": r2_reason,
            "r2_prefix": config.resolved_r2_prefix() if use_r2 else None,
            "bytes_per_example": per_example,
            "estimated_bytes_shard": shard_bytes,
            "estimated_bytes_total": total_bytes,
        },
        "dataset_validation": report,
        "examples": {
            "n_total": len(examples),
            "n_this_shard": len(shard),
            "n_emotional": int((examples["source"] == "emotion").sum()),
            "n_neutral": int((examples["source"] == "neutral").sum()),
            "emotions": sorted(examples.loc[examples["source"] == "emotion", "emotion"].unique().tolist()),
            "per_split_stories": {
                k: int(v) for k, v in examples.groupby("split").size().items()
            },
            "per_split_topics": topic_split.counts,
            "train_topics": topic_split.topics_for("train"),
            "validation_topics": topic_split.topics_for("validation"),
            "test_topics": topic_split.topics_for("test"),
        },
        "token_stats": tok_stats,
        "fingerprint": fingerprint,
    }

    record_dir = config.output_dir if not args.dry_run else config.output_dir / "dry_run"
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index:03d}"
    txt_path, json_path = provenance.write_run_record(
        record_dir,
        title=f"EXTRACTION RUN RECORD -- {config.run_name}",
        sections=sections,
        txt_name=f"run_config_extract{suffix}.txt",
        json_name=f"run_manifest_extract{suffix}.json",
    )
    print(f"Run record written:\n  {txt_path}\n  {json_path}")

    if args.dry_run:
        print("\n--dry-run: configuration and dataset validated, model not loaded.")
        if report["warnings"]:
            print("Dataset warnings above should be understood before a real run.")
        return 0

    # --- resume --------------------------------------------------------- #
    try:
        init_or_check_manifest(
            config.activations_dir,
            fingerprint=fingerprint,
            extra={
                "created": provenance.utc_timestamp(),
                "run_name": config.run_name,
                "layers": layers,
                "hidden_size": arch.hidden_size,
                "activation_dtype": config.activation_dtype,
                "pooling": config.pooling,
                "token_offset": config.token_offset,
                "packages": provenance.package_versions(),
            },
            allow_overwrite=args.overwrite,
        )
    except activation_store.IncompatibleRunError as exc:
        # An expected operator error, not a crash: report it plainly.
        sys.stdout.flush()
        print(f"\nABORTED -- incompatible existing run\n\n{exc}\n", file=sys.stderr)
        return 3

    already = activation_store.completed_example_ids(config.activations_dir)
    todo = shard[~shard["example_id"].isin(already)].reset_index(drop=True)
    if already:
        print(f"\nResuming: {len(already):,} examples already stored, {len(todo):,} remaining.")
    if todo.empty:
        print("Nothing left to extract for this shard.")
        return 0

    # --- model ---------------------------------------------------------- #
    print(f"\nLoading model {config.model_name} ...")
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
    device = model_utils.model_input_device(model)
    print(f"  loaded in {time.time() - t_load:.1f}s; input device {device}")

    on_chunk = None
    if use_r2:
        from core.r2 import make_chunk_uploader

        on_chunk = make_chunk_uploader(
            config.resolved_r2_prefix(),
            config.activations_dir,
            delete_local=config.delete_local_after_sync,
        )

    stats = extract(
        config=config,
        todo=todo,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        shard_index=args.shard_index,
        on_chunk_written=on_chunk,
    )

    # --- final report --------------------------------------------------- #
    print()
    print("=" * 78)
    print(f"Extraction complete for shard {args.shard_index}")
    print(f"  written  : {stats['n_written']:,} examples")
    print(f"  skipped  : {stats['n_skipped']:,}")
    print(f"  elapsed  : {stats['elapsed_s'] / 60:.1f} min "
          f"({stats['examples_per_s']:.1f} examples/s)")
    print(f"  on disk  : {human_bytes(stats['bytes_on_disk'])}")
    print("=" * 78)

    provenance.write_run_record(
        config.output_dir,
        title=f"EXTRACTION RESULT -- {config.run_name}",
        sections={
            "result": stats,
            "run": sections["run"],
            "fingerprint": fingerprint,
        },
        txt_name=f"run_result_extract{suffix}.txt",
        json_name=f"run_result_extract{suffix}.json",
    )

    if use_r2:
        print("\nFinal R2 sweep (catching anything a failed upload left behind) ...")
        from core.r2 import R2Client

        result = R2Client.from_env().sync_up(
            config.activations_dir,
            config.resolved_r2_prefix(),
            delete_local=False,
            verbose=False,
        )
        print(f"  uploaded {result['uploaded']}, already present {result['skipped']}, "
              f"{result['bytes'] / 1024**3:.2f} GiB")

    return 0


def extract(
    config: VectorExtractionConfig,
    todo: pd.DataFrame,
    model,
    tokenizer,
    layers: list[int],
    shard_index: int,
    on_chunk_written=None,
) -> dict:
    """Run the forward passes and write pooled activations."""
    import torch

    device = model_utils.model_input_device(model)
    out_dtype = model_utils.torch_dtype(config.activation_dtype)

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
                batch["story"].tolist(),
                tokenizer,
                use_chat_template=config.use_chat_template,
                chat_role=config.chat_role,
                chat_add_generation_prompt=config.chat_add_generation_prompt,
                prefix=config.text_prefix,
                suffix=config.text_suffix,
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
            # Pool on-device, then transfer only (batch x hidden) per selected
            # layer. Never move token-level activations to CPU.
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

            records = []
            skipped = []
            for row_i, (_, row) in enumerate(batch.iterrows()):
                entry = {
                    "example_id": row["example_id"],
                    "source": row["source"],
                    "emotion": None if pd.isna(row["emotion"]) else row["emotion"],
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
                            f"only {int(pooled.n_pooled_tokens[row_i])} tokens remain after "
                            f"token_offset={config.token_offset}; "
                            f"min_pooled_tokens={config.min_pooled_tokens}"
                        ),
                    })

            if skipped:
                writer.record_skipped(skipped)
                n_skipped += len(skipped)

            if records:
                keep_idx = torch.from_numpy(np.flatnonzero(keep))
                writer.add(
                    records,
                    {layer: tensor[keep_idx] for layer, tensor in pooled.pooled.items()},
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
    bytes_on_disk = sum(
        p.stat().st_size for p in Path(config.activations_dir).rglob("*") if p.is_file()
    )
    return {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_requested": len(todo),
        "elapsed_s": round(elapsed, 2),
        "examples_per_s": round(n_written / max(elapsed, 1e-9), 3),
        "tokens_per_s": round(n_tokens_seen / max(elapsed, 1e-9), 1),
        "bytes_on_disk": bytes_on_disk,
        "shard_index": shard_index,
        "finished": provenance.utc_timestamp(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
