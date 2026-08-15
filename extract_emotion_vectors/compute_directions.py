"""Stage 2: build emotion directions from stored activations.

Reads pooled activations from ``outputs/<run>/activations`` and writes directions,
centroids, global means and neutral PCA artefacts to
``outputs/<run>/results/directions`` (under ``results/`` so ``pull-results`` fetches
them and skips the large activations).

Usage
-----
::

    python -m extract_emotion_vectors.compute_directions
    python -m extract_emotion_vectors.compute_directions --dry-run
    python -m extract_emotion_vectors.compute_directions --set full_dataset=true \
        --output-subdir directions_full

By default only **training-split topics** contribute, so directions are safe to
use in held-out analyses. ``full_dataset=true`` uses every split -- for the final
directions, after evaluation is finished. Keep those in a separate subdirectory so
you can never accidentally evaluate them on their own training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core import model_utils, provenance
from core.activation_store import ActivationStore, human_bytes
from core.directions import DirectionSet, fit_layer_directions
from core.seeds import set_global_seeds
from extract_emotion_vectors.vector_extraction_config import VectorExtractionConfig
from extract_emotion_vectors.extract_activations import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute mean-difference emotion directions with neutral-PC removal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be fitted, without loading activations")
    p.add_argument("--limit", type=int, default=None,
                   help="use at most N stories per emotion (debugging)")
    p.add_argument("--layers", default=None,
                   help="layer spec override, e.g. 'all', 'evenly_spaced:14', '[16,32,48]'")
    p.add_argument("--output-subdir", default="directions",
                   help="subdirectory of the run directory to write into")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p


def select_rows(
    store: ActivationStore,
    config: VectorExtractionConfig,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rows contributing to the directions: emotional stories and neutral stories.

    Both are restricted to the same split set, so the neutral nuisance subspace is
    estimated from the same topics as the emotion centroids.
    """
    splits = None if config.full_dataset else ["train"]

    emotional = store.subset(source="emotion", splits=splits, emotions=config.emotions)
    neutral = store.subset(source="neutral", splits=splits)

    if emotional.empty:
        raise SystemExit(
            "No emotional activations found for the requested emotions/splits. "
            f"Stored emotions: {sorted(store.subset(source='emotion')['emotion'].unique())}"
        )
    if neutral.empty:
        raise SystemExit(
            "No neutral activations found. Neutral stories are required for the "
            "nuisance-PC removal step; re-run extraction with neutral_stories set."
        )

    if limit is not None:
        emotional = emotional.groupby("emotion", sort=False).head(limit).reset_index(drop=True)

    emotions = sorted(emotional["emotion"].unique().tolist())
    return emotional, neutral, emotions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)

    out_dir = config.results_dir / args.output_subdir

    print("=" * 78)
    print(f"Emotion direction construction -- run '{config.run_name}'")
    print("=" * 78)

    store = ActivationStore(config.activations_dir)
    fingerprint = store.manifest.get("fingerprint", {})
    print(f"activations : {config.activations_dir}")
    print(f"  examples  : {len(store.index):,}  ({human_bytes(store.total_bytes())} on disk)")
    print(f"  layers    : {len(store.layers)} stored, hidden_size={store.hidden_size}")
    print(f"  model     : {fingerprint.get('model_name')} @ {fingerprint.get('model_sha')}")
    print(f"  pooling   : {fingerprint.get('pooling')} (offset {fingerprint.get('token_offset')})")

    skipped = store.skipped()
    if len(skipped):
        print(f"  skipped   : {len(skipped)} examples during extraction")

    layer_spec = args.layers if args.layers is not None else config.layer_spec
    requested = model_utils.resolve_layers(layer_spec, max(store.layers) + 1)
    layers = [l for l in requested if l in store.layers]
    missing = [l for l in requested if l not in store.layers]
    if missing:
        print(f"  NOTE: {len(missing)} requested layers were never extracted and are "
              f"skipped: {missing if len(missing) <= 12 else f'{missing[:8]} ...'}")
    if not layers:
        raise SystemExit(f"none of the requested layers {requested} are present in the store")

    emotional, neutral, emotions = select_rows(store, config, args.limit)
    split_label = "ALL SPLITS (full_dataset=true)" if config.full_dataset else "train split only"

    print()
    print(f"Fitting from  : {split_label}")
    print(f"  emotions    : {len(emotions)} -> {emotions}")
    print(f"  stories     : {len(emotional):,} "
          f"({emotional.groupby('emotion').size().min()}-{emotional.groupby('emotion').size().max()} per emotion)")
    print(f"  topics      : {emotional['topic'].nunique()}")
    print(f"  neutral     : {len(neutral):,} stories over {neutral['topic'].nunique()} topics")
    print(f"  layers      : {len(layers)} -> "
          f"{layers if len(layers) <= 20 else f'{layers[:8]} ... {layers[-4:]}'}")
    print(f"  neutral PCs : smallest count explaining >= "
          f"{config.neutral_variance_threshold:.0%} of neutral variance")
    print(f"  output      : {out_dir}")
    print()

    if config.full_dataset:
        print("!! full_dataset=true: these directions have seen validation and test topics.")
        print("!! Do not report held-out metrics computed with them.")
        print()

    if args.dry_run:
        print("--dry-run: nothing fitted.")
        return 0

    per_emotion_rows = {e: emotional[emotional["emotion"] == e] for e in emotions}
    # Load each layer once for *all* emotions and slice in memory. Loading per
    # emotion would re-read every chunk n_emotions times per layer -- ~100 GiB of
    # redundant reads over a 65-layer Qwen run.
    emotion_labels = emotional["emotion"].to_numpy()
    emotion_positions = {e: np.flatnonzero(emotion_labels == e) for e in emotions}

    results = []
    for i, layer in enumerate(layers):
        all_acts = store.load_layer(layer, emotional)
        emotion_acts = {e: all_acts[pos] for e, pos in emotion_positions.items()}
        neutral_acts = store.load_layer(layer, neutral)
        result = fit_layer_directions(
            layer=layer,
            emotion_activations=emotion_acts,
            neutral_activations=neutral_acts,
            variance_threshold=config.neutral_variance_threshold,
        )
        results.append(result)
        print(
            f"  layer {layer:>3}  neutral PCs={result.neutral.n_components:>3} "
            f"(cum var {result.neutral.cumulative_variance:.3f})  "
            f"residual after projection: "
            f"{result.residual_fraction.mean():.3f} mean, {result.residual_fraction.min():.3f} min"
            f"   [{i + 1}/{len(layers)}]",
            flush=True,
        )

    metadata = {
        "run_name": config.run_name,
        "stage": "compute_directions",
        "method": "mean_difference_with_neutral_pc_removal",
        "is_trained_probe": False,
        "method_steps": [
            "per-emotion mean of pooled training activations (centroid)",
            "global mean across emotion centroids (equal weight per emotion)",
            "mean-difference = centroid - global mean",
            "PCA/SVD of centred neutral activations at this layer",
            f"keep smallest k with cumulative EVR >= {config.neutral_variance_threshold}",
            "project mean-difference vectors off the neutral-PC subspace",
            "unit-normalise",
        ],
        "full_dataset": config.full_dataset,
        "splits_used": ["train", "validation", "test"] if config.full_dataset else ["train"],
        "neutral_variance_threshold": config.neutral_variance_threshold,
        "n_emotional_stories": int(len(emotional)),
        "n_neutral_stories": int(len(neutral)),
        "n_topics_emotional": int(emotional["topic"].nunique()),
        "n_topics_neutral": int(neutral["topic"].nunique()),
        "hidden_size": store.hidden_size,
        "activation_fingerprint": fingerprint,
        "config": config.to_dict(),
        "limit": args.limit,
    }
    direction_set = DirectionSet.from_layer_results(results, metadata)
    directions_path, metadata_path = direction_set.save(out_dir)

    summary = pd.DataFrame(
        [
            {
                "layer": r.layer,
                "n_neutral_pcs": r.neutral.n_components,
                "neutral_cumulative_variance": r.neutral.cumulative_variance,
                "n_neutral_samples": r.neutral.n_samples,
                "mean_difference_norm_mean": float(
                    np.linalg.norm(r.mean_difference, axis=1).mean()
                ),
                "centroid_norm_mean": float(np.linalg.norm(r.centroids, axis=1).mean()),
                "global_mean_norm": float(np.linalg.norm(r.global_mean)),
                "residual_fraction_mean": float(r.residual_fraction.mean()),
                "residual_fraction_min": float(r.residual_fraction.min()),
                "mean_pairwise_cosine": float(
                    _mean_offdiag(r.direction @ r.direction.T)
                ),
                "mean_pairwise_cosine_unprojected": float(
                    _mean_offdiag(r.direction_unprojected @ r.direction_unprojected.T)
                ),
            }
            for r in results
        ]
    )
    summary_path = out_dir / "layer_summary.csv"
    summary.to_csv(summary_path, index=False)

    provenance.write_run_record(
        out_dir,
        title=f"DIRECTION CONSTRUCTION RECORD -- {config.run_name}",
        sections={
            "run": {
                "run_name": config.run_name,
                "stage": "compute_directions",
                "output_dir": str(out_dir),
                "layers": layers,
                "emotions": emotions,
                "full_dataset": config.full_dataset,
                "limit": args.limit,
            },
            "method": {k: metadata[k] for k in ("method", "is_trained_probe", "method_steps",
                                                "neutral_variance_threshold", "splits_used")},
            "counts": {
                "n_emotional_stories": metadata["n_emotional_stories"],
                "n_neutral_stories": metadata["n_neutral_stories"],
                "n_topics_emotional": metadata["n_topics_emotional"],
                "n_topics_neutral": metadata["n_topics_neutral"],
                "stories_per_emotion": {
                    e: int(len(rows)) for e, rows in per_emotion_rows.items()
                },
                "n_skipped_at_extraction": int(len(skipped)),
            },
            "config": config.to_dict(),
            "activation_fingerprint": fingerprint,
            "per_layer_summary": summary.round(5).to_dict(orient="records"),
        },
        txt_name="run_config_directions.txt",
        json_name="run_manifest_directions.json",
    )

    print()
    print("Wrote:")
    print(f"  {directions_path}")
    print(f"  {metadata_path}")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'run_config_directions.txt'}")
    ks = [r.neutral.n_components for r in results]
    print(f"\nNeutral PCs kept across layers: min={min(ks)} median={int(np.median(ks))} max={max(ks)}")
    return 0


def _mean_offdiag(matrix: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a square matrix."""
    n = matrix.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


if __name__ == "__main__":
    raise SystemExit(main())
