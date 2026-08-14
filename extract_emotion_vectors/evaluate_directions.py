"""Stage 3: lightweight held-out validation of the emotion directions.

What this does
--------------
* **Held-out scoring** -- dot products of validation/test activations against each
  direction, giving top-1 emotion classification and one-vs-rest AUROC.
* **Bootstrap stability** -- resample training *topics* with replacement, refit
  centroids, and report cosine similarity to the reference direction.
* **Split-half agreement** -- split the training topics into two disjoint halves,
  fit directions independently on each, and compare.

What this is not
----------------
These are **fixed mean-difference directions**, not trained probes. No objective
is optimised against the labels here; top-1 accuracy is a sanity check on the
directions, not a claim about the best achievable decoder. Logistic-regression
probes are a separate, later comparison and must be reported separately.

Usage
-----
::

    python -m extract_emotion_vectors.evaluate_directions
    python -m extract_emotion_vectors.evaluate_directions --dry-run
    python -m extract_emotion_vectors.evaluate_directions --layers all --no-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import model_utils, provenance
from core.activation_store import ActivationStore
from core.directions import DirectionSet, mean_difference_directions
from core.seeds import rng_for, set_global_seeds
from extract_emotion_vectors.extract_activations import load_config

SCORE_MODES = ("dot", "centered_dot", "cosine")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Held-out validation and stability checks for emotion directions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="report the plan without loading activations")
    p.add_argument("--limit", type=int, default=None, help="cap stories per emotion per split")
    p.add_argument("--layers", default=None, help="layer spec override (default: config.eval_layer_spec)")
    p.add_argument("--directions-subdir", default="directions",
                   help="which directions to evaluate")
    p.add_argument("--output-subdir", default="evaluation")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    emotions: list[str],
) -> tuple[dict, list[dict]]:
    """Top-1 accuracy and one-vs-rest AUROC from a ``(n, n_emotions)`` score matrix.

    AUROC is computed per emotion (that emotion's score column, that emotion vs
    all others) and skipped as not meaningful where only one class is present.
    """
    from sklearn.metrics import roc_auc_score

    predicted = np.argmax(scores, axis=1)
    truth = np.array([emotions.index(l) for l in labels])
    accuracy = float((predicted == truth).mean())

    per_emotion: list[dict] = []
    aurocs = []
    for i, emotion in enumerate(emotions):
        positive = truth == i
        n_pos = int(positive.sum())
        row = {
            "emotion": emotion,
            "n": n_pos,
            "top1_recall": float((predicted[positive] == i).mean()) if n_pos else float("nan"),
            "top1_precision": (
                float((truth[predicted == i] == i).mean()) if int((predicted == i).sum()) else float("nan")
            ),
        }
        if 0 < n_pos < len(truth):
            auroc = float(roc_auc_score(positive.astype(int), scores[:, i]))
            row["auroc"] = auroc
            aurocs.append(auroc)
        else:
            row["auroc"] = float("nan")  # degenerate: only one class present
        per_emotion.append(row)

    summary = {
        "top1_accuracy": accuracy,
        "chance_accuracy": 1.0 / len(emotions),
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "n_examples": int(len(truth)),
        "n_emotions": len(emotions),
    }
    return summary, per_emotion


def confusion_counts(scores: np.ndarray, labels: np.ndarray, emotions: list[str]) -> pd.DataFrame:
    predicted = np.argmax(scores, axis=1)
    truth = np.array([emotions.index(l) for l in labels])
    matrix = np.zeros((len(emotions), len(emotions)), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1
    return pd.DataFrame(matrix, index=emotions, columns=emotions)


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #

def bootstrap_stability(
    train_rows: pd.DataFrame,
    neutral_rows: pd.DataFrame,
    store: ActivationStore,
    layer: int,
    emotions: list[str],
    reference: np.ndarray,
    reference_centroids: np.ndarray,
    n_bootstrap: int,
    variance_threshold: float,
    seed: int,
) -> list[dict]:
    """Resample training topics with replacement; measure direction/centroid drift.

    The neutral subspace is fitted **once** on the full training neutrals and held
    fixed across resamples, so what we measure is sampling noise in the emotion
    centroids rather than noise in the nuisance estimate. Refitting the SVD per
    resample would multiply runtime by ``n_bootstrap`` for a second-order effect.
    """
    # One read per layer, sliced per emotion in memory (see compute_directions).
    labels = train_rows["emotion"].to_numpy()
    all_acts = store.load_layer(layer, train_rows)
    acts = {e: all_acts[labels == e] for e in emotions}
    topics = {e: train_rows.loc[labels == e, "topic"].to_numpy() for e in emotions}
    neutral_acts = store.load_layer(layer, neutral_rows)

    from core.directions import fit_neutral_subspace, project_out, unit_normalize

    subspace = fit_neutral_subspace(neutral_acts, variance_threshold)
    unique_topics = np.array(sorted(train_rows["topic"].unique()))

    direction_cos = np.zeros((n_bootstrap, len(emotions)))
    centroid_cos = np.zeros((n_bootstrap, len(emotions)))

    for b in range(n_bootstrap):
        rng = rng_for(seed, "bootstrap", layer, b)
        sampled = unique_topics[rng.integers(0, len(unique_topics), size=len(unique_topics))]
        # Multiplicity of each topic in the resample -> weights on that topic's rows.
        multiplicity = pd.Series(sampled).value_counts()

        centroids = np.empty((len(emotions), store.hidden_size), dtype=np.float64)
        for i, emotion in enumerate(emotions):
            weights = pd.Series(topics[emotion]).map(multiplicity).fillna(0.0).to_numpy()
            total = weights.sum()
            if total == 0:  # pathological resample; fall back to unweighted
                centroids[i] = acts[emotion].mean(axis=0, dtype=np.float64)
            else:
                centroids[i] = (weights @ acts[emotion].astype(np.float64)) / total

        boot_directions = unit_normalize(project_out(centroids - centroids.mean(axis=0),
                                                     subspace.components))
        direction_cos[b] = np.sum(boot_directions * reference, axis=1)
        centroid_cos[b] = _rowwise_cosine(centroids, reference_centroids)

    return [
        {
            "layer": layer,
            "emotion": emotion,
            "n_bootstrap": n_bootstrap,
            "direction_cosine_mean": float(direction_cos[:, i].mean()),
            "direction_cosine_std": float(direction_cos[:, i].std()),
            "direction_cosine_p05": float(np.percentile(direction_cos[:, i], 5)),
            "direction_cosine_min": float(direction_cos[:, i].min()),
            "centroid_cosine_mean": float(centroid_cos[:, i].mean()),
            "centroid_cosine_std": float(centroid_cos[:, i].std()),
        }
        for i, emotion in enumerate(emotions)
    ]


def split_half_agreement(
    train_rows: pd.DataFrame,
    neutral_rows: pd.DataFrame,
    store: ActivationStore,
    layer: int,
    emotions: list[str],
    variance_threshold: float,
    seed: int,
) -> list[dict]:
    """Fit directions on two disjoint halves of the training topics and compare.

    Each half gets its own neutral subspace (only two extra SVDs per layer), so
    this is a genuinely independent refit of the whole construction.
    """
    topics = np.array(sorted(train_rows["topic"].unique()))
    rng = rng_for(seed, "split_half", layer)
    permuted = topics[rng.permutation(len(topics))]
    halves = (set(permuted[: len(permuted) // 2]), set(permuted[len(permuted) // 2 :]))

    fitted = []
    for half in halves:
        rows = train_rows[train_rows["topic"].isin(half)]
        neutral = neutral_rows[neutral_rows["topic"].isin(half)]
        labels = rows["emotion"].to_numpy()
        half_acts = store.load_layer(layer, rows)
        acts = {e: half_acts[labels == e] for e in emotions}
        if any(a.shape[0] == 0 for a in acts.values()) or neutral.shape[0] < 2:
            return []
        order, directions = mean_difference_directions(
            acts, store.load_layer(layer, neutral), variance_threshold
        )
        assert list(order) == emotions
        fitted.append(directions)

    cosine = np.sum(fitted[0] * fitted[1], axis=1)
    return [
        {
            "layer": layer,
            "emotion": emotion,
            "split_half_cosine": float(cosine[i]),
            "n_topics_half_a": len(halves[0]),
            "n_topics_half_b": len(halves[1]),
        }
        for i, emotion in enumerate(emotions)
    ]


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.where((a_norm * b_norm) == 0, 1.0, a_norm * b_norm)
    return np.sum(a * b, axis=1) / denom


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def make_plots(
    out_dir: Path,
    layer_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    split_half: pd.DataFrame,
    layer_summary: pd.DataFrame | None,
    cosine_matrix: tuple[int, list[str], np.ndarray] | None,
) -> list[Path]:
    """Diagnostics: metrics vs layer, stability vs layer, neutral PCs, cosine heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core import plotting

    plotting.apply_style()
    written: list[Path] = []

    # -- top-1 accuracy and macro AUROC vs layer, one figure each, series = split #
    # One row per (layer, split): pick a single score mode AND a single vector kind,
    # otherwise every layer contributes several points and the line zig-zags.
    primary = layer_metrics[
        (layer_metrics["score_mode"] == "centered_dot") & (layer_metrics["kind"] == "direction")
    ]
    for metric, label, chance_col in (
        ("top1_accuracy", "Top-1 accuracy", "chance_accuracy"),
        ("macro_auroc", "Macro one-vs-rest AUROC", None),
    ):
        if primary.empty or primary[metric].isna().all():
            continue
        fig, ax = plt.subplots()
        for i, split in enumerate(sorted(primary["split"].unique())):
            rows = primary[primary["split"] == split].sort_values("layer")
            colour = plotting.SERIES[i % len(plotting.SERIES)]
            ax.plot(rows["layer"], rows[metric], color=colour, label=split)
            plotting.label_line_end(ax, rows["layer"].to_numpy(), rows[metric].to_numpy(),
                                    split, colour)
        if chance_col is not None:
            chance = float(primary[chance_col].iloc[0])
            ax.axhline(chance, color=plotting.INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
            ax.annotate("chance", xy=(0.01, chance), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=plotting.INK_MUTED)
        else:
            ax.axhline(0.5, color=plotting.INK_MUTED, linestyle=(0, (4, 3)), linewidth=1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        plotting.finish(
            fig, ax,
            title=f"{label} of mean-difference emotion directions",
            subtitle="held-out topics; scores are centred dot products (not a trained probe)",
            xlabel="hidden-state layer", ylabel=label,
            integer_x=True, right_margin=0.12,
        )
        written.append(plotting.save(fig, out_dir / f"{metric}_by_layer.png"))

    # -- direction stability vs layer ------------------------------------- #
    if not split_half.empty or not stability.empty:
        fig, ax = plt.subplots()
        # Per-emotion detail as unlabelled hairlines: identity lives in the CSV,
        # not in a fourth-plus categorical hue.
        if not split_half.empty:
            for emotion, rows in split_half.groupby("emotion"):
                rows = rows.sort_values("layer")
                ax.plot(rows["layer"], rows["split_half_cosine"],
                        color=plotting.GRIDLINE, linewidth=0.9, zorder=1)
            mean_sh = split_half.groupby("layer")["split_half_cosine"].mean().sort_index()
            ax.plot(mean_sh.index, mean_sh.to_numpy(), color=plotting.SERIES[0],
                    label="split-half agreement (mean)", zorder=3)
            plotting.label_line_end(ax, mean_sh.index.to_numpy(), mean_sh.to_numpy(),
                                    "split-half", plotting.SERIES[0])
        if not stability.empty:
            mean_boot = stability.groupby("layer")["direction_cosine_mean"].mean().sort_index()
            ax.plot(mean_boot.index, mean_boot.to_numpy(), color=plotting.SERIES[1],
                    label="bootstrap stability (mean)", zorder=3)
            plotting.label_line_end(ax, mean_boot.index.to_numpy(), mean_boot.to_numpy(),
                                    "bootstrap", plotting.SERIES[1])
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        plotting.finish(
            fig, ax,
            title="Direction stability across resamples of the training topics",
            subtitle="cosine similarity to the reference direction; gray hairlines are individual emotions",
            xlabel="hidden-state layer", ylabel="cosine similarity",
            integer_x=True, right_margin=0.14,
        )
        written.append(plotting.save(fig, out_dir / "direction_stability_by_layer.png"))

    # -- neutral PCs removed vs layer (single series) ---------------------- #
    if layer_summary is not None and not layer_summary.empty:
        fig, ax = plt.subplots()
        rows = layer_summary.sort_values("layer")
        ax.plot(rows["layer"], rows["n_neutral_pcs"], color=plotting.SERIES[0])
        plotting.finish(
            fig, ax,
            title="Neutral principal components projected out, by layer",
            subtitle="fewest PCs explaining >= 50% of neutral-story variance",
            xlabel="hidden-state layer", ylabel="number of PCs removed",
            integer_x=True,
        )
        written.append(plotting.save(fig, out_dir / "neutral_pcs_by_layer.png"))

    # -- emotion x emotion cosine similarity heatmap ----------------------- #
    if cosine_matrix is not None:
        layer, emotions, matrix = cosine_matrix
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        image = ax.imshow(matrix, cmap=plotting.diverging_cmap(), vmin=-1, vmax=1)
        ax.set_xticks(range(len(emotions)), emotions, rotation=45, ha="right")
        ax.set_yticks(range(len(emotions)), emotions)
        ax.grid(False)
        ax.set_xticks(np.arange(len(emotions) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(emotions) + 1) - 0.5, minor=True)
        # 2px surface gap between cells.
        ax.grid(which="minor", color=plotting.SURFACE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
        bar.set_label("cosine similarity", color=plotting.INK_SECONDARY, fontsize=8.5)
        bar.outline.set_visible(False)
        ax.set_title(f"Emotion direction cosine similarity (layer {layer})",
                     loc="left", pad=10, color=plotting.INK_PRIMARY)
        fig.tight_layout()
        written.append(plotting.save(fig, out_dir / f"emotion_cosine_heatmap_layer{layer:03d}.png"))

    return written


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)

    directions_dir = config.output_dir / args.directions_subdir
    out_dir = config.output_dir / args.output_subdir

    print("=" * 78)
    print(f"Direction evaluation -- run '{config.run_name}'")
    print("=" * 78)

    direction_set = DirectionSet.load(directions_dir)
    store = ActivationStore(config.activations_dir)
    emotions = list(direction_set.emotions)

    if direction_set.metadata.get("full_dataset"):
        print("!! These directions were built with full_dataset=true: they have already seen")
        print("!! validation and test topics. Held-out metrics below are NOT held out.")
        print()

    layer_spec = args.layers if args.layers is not None else config.eval_layer_spec
    requested = model_utils.resolve_layers(layer_spec, max(store.layers) + 1)
    layers = [l for l in requested if l in direction_set.layer_indices and l in store.layers]
    if not layers:
        raise SystemExit(
            f"no evaluatable layers: requested {requested}, directions have "
            f"{direction_set.layer_indices}, store has {store.layers}"
        )

    train_rows = store.subset(source="emotion", splits=["train"], emotions=emotions)
    train_neutral = store.subset(source="neutral", splits=["train"])
    eval_rows = {
        split: store.subset(source="emotion", splits=[split], emotions=emotions)
        for split in config.eval_splits
    }
    if args.limit is not None:
        for split, rows in eval_rows.items():
            eval_rows[split] = (
                rows.groupby("emotion", sort=False).head(args.limit).reset_index(drop=True)
            )

    print(f"directions : {directions_dir}")
    print(f"  method   : {direction_set.metadata.get('method')} "
          f"(trained probe: {direction_set.metadata.get('is_trained_probe')})")
    print(f"  emotions : {len(emotions)} -> {emotions}")
    print(f"  layers   : {len(layers)} -> {layers}")
    print(f"training   : {len(train_rows):,} stories, {train_rows['topic'].nunique()} topics "
          f"({len(train_neutral):,} neutral)")
    for split, rows in eval_rows.items():
        print(f"{split:<11}: {len(rows):,} stories, {rows['topic'].nunique()} topics")
    print(f"bootstrap  : {config.eval_bootstrap_n} resamples of the training topics")
    print(f"output     : {out_dir}")
    print()

    empty = [s for s, rows in eval_rows.items() if rows.empty]
    if empty:
        print(f"NOTE: no stored activations for split(s) {empty}; they will be skipped.")

    if args.dry_run:
        print("--dry-run: nothing evaluated.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    layer_metric_rows: list[dict] = []
    emotion_metric_rows: list[dict] = []
    stability_rows: list[dict] = []
    split_half_rows: list[dict] = []
    confusions: dict[str, pd.DataFrame] = {}

    for i, layer in enumerate(layers):
        for split, rows in eval_rows.items():
            if rows.empty:
                continue
            activations = store.load_layer(layer, rows)
            labels = rows["emotion"].to_numpy()
            for mode in SCORE_MODES:
                scores = direction_set.score(activations, layer, mode=mode)
                summary, per_emotion = classification_metrics(scores, labels, emotions)
                layer_metric_rows.append(
                    {"layer": layer, "split": split, "score_mode": mode,
                     "kind": "direction", **summary}
                )
                emotion_metric_rows += [
                    {"layer": layer, "split": split, "score_mode": mode,
                     "kind": "direction", **row}
                    for row in per_emotion
                ]
                # Same metrics for the unprojected vectors: the paper notes its
                # qualitative findings survive without neutral-PC removal, and this
                # is the cheapest way to check that on our model.
                scores_raw = direction_set.score(activations, layer, mode=mode,
                                                 kind="direction_unprojected")
                summary_raw, per_emotion_raw = classification_metrics(scores_raw, labels, emotions)
                layer_metric_rows.append(
                    {"layer": layer, "split": split, "score_mode": mode,
                     "kind": "direction_unprojected", **summary_raw}
                )
                emotion_metric_rows += [
                    {"layer": layer, "split": split, "score_mode": mode,
                     "kind": "direction_unprojected", **row}
                    for row in per_emotion_raw
                ]

            if split == config.eval_splits[0]:
                confusions[f"layer{layer:03d}_{split}"] = confusion_counts(
                    direction_set.score(activations, layer, mode="centered_dot"), labels, emotions
                )

        if config.eval_bootstrap_n > 0 and not train_rows.empty:
            stability_rows += bootstrap_stability(
                train_rows, train_neutral, store, layer, emotions,
                reference=direction_set.matrix(layer, "direction"),
                reference_centroids=direction_set.matrix(layer, "centroid"),
                n_bootstrap=config.eval_bootstrap_n,
                variance_threshold=config.neutral_variance_threshold,
                seed=config.seed,
            )
        split_half_rows += split_half_agreement(
            train_rows, train_neutral, store, layer, emotions,
            variance_threshold=config.neutral_variance_threshold, seed=config.seed,
        )

        headline = [
            r for r in layer_metric_rows
            if r["layer"] == layer and r["score_mode"] == "centered_dot"
            and r["kind"] == "direction" and r["split"] == config.eval_splits[0]
        ]
        sh = [r["split_half_cosine"] for r in split_half_rows if r["layer"] == layer]
        message = f"  layer {layer:>3}"
        if headline:
            message += (f"  top1={headline[0]['top1_accuracy']:.3f} "
                        f"auroc={headline[0]['macro_auroc']:.3f}")
        if sh:
            message += f"  split-half cos={np.mean(sh):.3f}"
        print(message + f"   [{i + 1}/{len(layers)}]", flush=True)

    layer_metrics = pd.DataFrame(layer_metric_rows)
    emotion_metrics = pd.DataFrame(emotion_metric_rows)
    stability = pd.DataFrame(stability_rows)
    split_half = pd.DataFrame(split_half_rows)

    layer_metrics.to_csv(out_dir / "metrics_by_layer.csv", index=False)
    emotion_metrics.to_csv(out_dir / "metrics_by_emotion.csv", index=False)
    if not stability.empty:
        stability.to_csv(out_dir / "stability_bootstrap.csv", index=False)
    if not split_half.empty:
        split_half.to_csv(out_dir / "stability_split_half.csv", index=False)
    for name, matrix in confusions.items():
        matrix.to_csv(out_dir / f"confusion_{name}.csv")

    # -- best layer + cosine geometry ------------------------------------- #
    headline = layer_metrics[
        (layer_metrics["score_mode"] == "centered_dot")
        & (layer_metrics["kind"] == "direction")
        & (layer_metrics["split"] == config.eval_splits[0])
    ]
    best_layer = int(headline.loc[headline["top1_accuracy"].idxmax(), "layer"]) if not headline.empty else layers[len(layers) // 2]
    matrix = direction_set.matrix(best_layer, "direction")
    cosine = matrix @ matrix.T
    pd.DataFrame(cosine, index=emotions, columns=emotions).to_csv(
        out_dir / f"emotion_cosine_layer{best_layer:03d}.csv"
    )

    summary = {
        "run_name": config.run_name,
        "stage": "evaluate_directions",
        "directions_dir": str(directions_dir),
        "method": direction_set.metadata.get("method"),
        "is_trained_probe": False,
        "note": (
            "Fixed mean-difference directions with neutral-PC removal. These metrics "
            "characterise those directions; they are not the output of a trained probe "
            "and should not be compared to probe accuracy without saying so."
        ),
        "emotions": emotions,
        "layers_evaluated": layers,
        "eval_splits": list(config.eval_splits),
        "n_train_stories": int(len(train_rows)),
        "n_train_topics": int(train_rows["topic"].nunique()) if not train_rows.empty else 0,
        "n_eval_stories": {s: int(len(r)) for s, r in eval_rows.items()},
        "chance_accuracy": 1.0 / len(emotions),
        "best_layer_by_top1": best_layer,
        "headline": (
            headline.loc[headline["layer"] == best_layer]
            .drop(columns=["kind"])
            .to_dict(orient="records")
        ),
        "bootstrap": {
            "n_resamples": config.eval_bootstrap_n,
            "unit": "training topics, sampled with replacement",
            "neutral_subspace": "held fixed across resamples (fitted on all training neutrals)",
            "mean_direction_cosine": (
                float(stability["direction_cosine_mean"].mean()) if not stability.empty else None
            ),
        },
        "split_half": {
            "mean_cosine": float(split_half["split_half_cosine"].mean()) if not split_half.empty else None,
            "min_cosine": float(split_half["split_half_cosine"].min()) if not split_half.empty else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    layer_summary_path = directions_dir / "layer_summary.csv"
    layer_summary = pd.read_csv(layer_summary_path) if layer_summary_path.exists() else None

    plots: list[Path] = []
    if config.make_plots and not args.no_plots:
        try:
            plots = make_plots(
                out_dir, layer_metrics, stability, split_half, layer_summary,
                cosine_matrix=(best_layer, emotions, cosine),
            )
        except Exception as exc:
            print(f"WARNING: plotting failed ({exc}); CSV/JSON results were still written")

    provenance.write_run_record(
        out_dir,
        title=f"EVALUATION RECORD -- {config.run_name}",
        sections={
            "run": {
                "run_name": config.run_name,
                "stage": "evaluate_directions",
                "directions_subdir": args.directions_subdir,
                "output_dir": str(out_dir),
                "layers_evaluated": layers,
                "limit": args.limit,
            },
            "summary": summary,
            "config": config.to_dict(),
            "artefacts": [str(p) for p in plots],
        },
        txt_name="run_config_evaluation.txt",
        json_name="run_manifest_evaluation.json",
    )

    print()
    print("=" * 78)
    if not headline.empty:
        best = headline.loc[headline["layer"] == best_layer].iloc[0]
        print(f"Best layer by top-1 ({config.eval_splits[0]}): {best_layer}")
        print(f"  top-1 accuracy : {best['top1_accuracy']:.3f} "
              f"(chance {best['chance_accuracy']:.3f})")
        print(f"  macro AUROC    : {best['macro_auroc']:.3f}")
    if not split_half.empty:
        print(f"Split-half direction agreement: mean cos "
              f"{split_half['split_half_cosine'].mean():.3f}, "
              f"min {split_half['split_half_cosine'].min():.3f}")
    if not stability.empty:
        print(f"Bootstrap direction stability : mean cos "
              f"{stability['direction_cosine_mean'].mean():.3f}")
    print(f"\nResults in {out_dir}")
    for path in sorted(out_dir.glob("*")):
        print(f"  {path.name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
