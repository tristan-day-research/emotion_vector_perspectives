"""Phase 3b: do valence and arousal behave like reusable components?

This is a supervised geometric follow-up to Phase 3, not another PCA.  It uses the
balanced 2 x 2 anchor design to ask whether the *same* valence displacement appears
at high and low arousal, whether the same arousal displacement appears at positive
and negative valence, and whether their combination is approximately additive.

No model or activation chunks are loaded.  Phase 2's saved full and split-half
emotion vectors are sufficient::

    python run.py phase3b

The analysis is deliberately limited to the 16 preregistered anchors.  The other 155
emotion words have no valence/arousal labels and are not silently hand-labelled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, provenance
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config
from emotion_pca_jlens.phase3_pca import EmotionSpace, read_emotion_vectors

RULE = "=" * 78
THIN = "-" * 78
QUADRANTS = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 3b: cross-validated valence/arousal compositionality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--output-subdir", default="phase3b_compositionality")
    parser.add_argument("--n-resamples", type=int, default=1000,
                        help="balanced-label permutations and within-quadrant bootstraps")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def factorial_components(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray
) -> dict[str, np.ndarray]:
    """Least-squares coefficients for mu + V*gV + A*gA + V*A*gVA."""
    design = np.column_stack([
        np.ones(len(matrix)), valence, arousal, valence * arousal,
    ])
    beta = np.linalg.lstsq(design, matrix, rcond=None)[0]
    return {
        "mean": beta[0], "valence": beta[1], "arousal": beta[2],
        "interaction": beta[3],
    }


def quadrant_centroids(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for v, a in QUADRANTS:
        rows = matrix[(valence == v) & (arousal == a)]
        if len(rows) < 2:
            raise ValueError(
                f"quadrant valence={v:+d}, arousal={a:+d} has {len(rows)} rows; "
                "the compositionality test needs at least two"
            )
        out[(v, a)] = rows.mean(axis=0)
    return out


def geometry_metrics(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray
) -> dict[str, float]:
    q = quadrant_centroids(matrix, valence, arousal)
    hp, hn, lp, ln = q[(1, 1)], q[(-1, 1)], q[(1, -1)], q[(-1, -1)]
    a_pos, a_neg = hp - lp, hn - ln
    v_high, v_low = hp - hn, lp - ln
    components = factorial_components(matrix, valence, arousal)
    main_norm = float(np.hypot(
        np.linalg.norm(components["valence"]),
        np.linalg.norm(components["arousal"]),
    ))
    interaction_norm = float(np.linalg.norm(components["interaction"]))
    parallelogram = hp + ln - hn - lp
    contrast_scale = float(np.mean([
        np.linalg.norm(a_pos), np.linalg.norm(a_neg),
        np.linalg.norm(v_high), np.linalg.norm(v_low),
    ]))
    arousal_reuse = cosine(a_pos, a_neg)
    valence_reuse = cosine(v_high, v_low)
    return {
        "arousal_contrast_cosine": arousal_reuse,
        "valence_contrast_cosine": valence_reuse,
        "mean_contrast_cosine": float(np.nanmean([arousal_reuse, valence_reuse])),
        "interaction_to_main_ratio": interaction_norm / max(main_norm, 1e-12),
        "parallelogram_residual_ratio": (
            float(np.linalg.norm(parallelogram)) / max(contrast_scale, 1e-12)
        ),
        "valence_component_norm": float(np.linalg.norm(components["valence"])),
        "arousal_component_norm": float(np.linalg.norm(components["arousal"])),
        "interaction_component_norm": interaction_norm,
    }


def leave_one_out(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray,
    *, include_interaction: bool,
) -> pd.DataFrame:
    """Predict each held-out centered vector from components fitted on the other 15."""
    rows: list[dict] = []
    for held_out in range(len(matrix)):
        keep = np.arange(len(matrix)) != held_out
        columns = [np.ones(int(keep.sum())), valence[keep], arousal[keep]]
        if include_interaction:
            columns.append(valence[keep] * arousal[keep])
        design = np.column_stack(columns)
        beta = np.linalg.lstsq(design, matrix[keep], rcond=None)[0]
        actual = matrix[held_out] - beta[0]
        prediction = valence[held_out] * beta[1] + arousal[held_out] * beta[2]
        if include_interaction:
            prediction = prediction + valence[held_out] * arousal[held_out] * beta[3]
        denom = float(np.dot(actual, actual))
        rows.append({
            "held_out": held_out,
            "model": "valence+arousal+interaction" if include_interaction
                     else "valence+arousal",
            "cosine": cosine(actual, prediction),
            "r2_centered": (
                1.0 - float(np.dot(actual - prediction, actual - prediction)) / denom
                if denom > 0 else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def cv_summary(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.concat([
        leave_one_out(matrix, valence, arousal, include_interaction=False),
        leave_one_out(matrix, valence, arousal, include_interaction=True),
    ], ignore_index=True)
    summary: dict[str, float] = {}
    for model, group in frame.groupby("model"):
        key = "additive" if model == "valence+arousal" else "with_interaction"
        summary[f"{key}_mean_cosine"] = float(group["cosine"].mean())
        summary[f"{key}_mean_r2"] = float(group["r2_centered"].mean())
    return frame, summary


def balanced_labels(valence: np.ndarray, arousal: np.ndarray) -> np.ndarray:
    return np.column_stack([valence, arousal])


def resampling_controls(
    matrix: np.ndarray, valence: np.ndarray, arousal: np.ndarray,
    n_resamples: int, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced-label permutation null and within-quadrant bootstrap."""
    permutations: list[dict] = []
    bootstraps: list[dict] = []
    labels = balanced_labels(valence, arousal)
    groups = {
        (v, a): np.flatnonzero((valence == v) & (arousal == a))
        for v, a in QUADRANTS
    }
    for sample in range(n_resamples):
        perm = labels[rng.permutation(len(labels))]
        geom = geometry_metrics(matrix, perm[:, 0], perm[:, 1])
        _, cv = cv_summary(matrix, perm[:, 0], perm[:, 1])
        permutations.append({"sample": sample, **geom, **cv})

        indices = np.concatenate([
            rng.choice(groups[key], size=len(groups[key]), replace=True)
            for key in QUADRANTS
        ])
        boot_geom = geometry_metrics(matrix[indices], valence[indices], arousal[indices])
        bootstraps.append({"sample": sample, **boot_geom})
    return pd.DataFrame(permutations), pd.DataFrame(bootstraps)


def empirical_p(null: pd.Series, observed: float, *, smaller: bool = False) -> float:
    values = null.dropna().to_numpy(dtype=float)
    extreme = values <= observed if smaller else values >= observed
    return float((int(extreme.sum()) + 1) / (len(values) + 1))


def ci(frame: pd.DataFrame, key: str) -> list[float]:
    return [float(x) for x in frame[key].quantile([0.025, 0.975]).to_list()]


def anchor_arrays(space: EmotionSpace, matrix: np.ndarray | None = None):
    mask = space.anchor_mask()
    emotions = list(np.asarray(space.emotions)[mask])
    values = space.matrix[mask] if matrix is None else matrix[mask]
    valence = space.label_array("valence")[mask].astype(int)
    arousal = space.label_array("arousal")[mask].astype(int)
    if len(values) != 16 or set(zip(valence, arousal)) != set(QUADRANTS):
        raise SystemExit(
            f"expected the balanced 16-anchor design; found {len(values)} labelled "
            f"anchors with cells {sorted(set(zip(valence, arousal)))}"
        )
    return emotions, values, valence, arousal


def make_plot(
    out_path: Path, observed: dict, permutations: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = ["arousal_contrast_cosine", "valence_contrast_cosine"]
    labels = ["arousal", "valence"]
    values = [observed[m] for m in metrics]
    errors = np.asarray([
        [values[i] - ci(bootstrap, m)[0] for i, m in enumerate(metrics)],
        [ci(bootstrap, m)[1] - values[i] for i, m in enumerate(metrics)],
    ])
    axes[0].bar(labels, values, color=["#2f78d0", "#eb6032"], yerr=errors, capsize=4)
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_ylabel("cosine between the same contrast\nin the other context")
    axes[0].set_title("Component reuse")

    axes[1].hist(permutations["mean_contrast_cosine"], bins=30, color="0.75")
    axes[1].axvline(observed["mean_contrast_cosine"], color="#eb6032", lw=2)
    axes[1].set_title("Reuse vs shuffled labels")
    axes[1].set_xlabel("mean contrast cosine")

    axes[2].hist(permutations["additive_mean_cosine"], bins=30, color="0.75")
    axes[2].axvline(observed["additive_mean_cosine"], color="#2f78d0", lw=2)
    axes[2].set_title("Held-out reconstruction")
    axes[2].set_xlabel("mean held-out cosine")
    fig.suptitle("Do valence and arousal behave like reusable emotion components?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_resamples < 99:
        raise SystemExit("--n-resamples must be at least 99 for a useful empirical null")
    config: PCAJLensConfig = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()
    vectors_path = args.vectors or config.emotion_vectors_path
    meta_path = (vectors_path.with_name(config.emotion_vectors_meta_path.name)
                 if args.vectors else config.emotion_vectors_meta_path)
    out_dir = config.phase_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    space = read_emotion_vectors(vectors_path, meta_path)
    emotions, matrix, valence, arousal = anchor_arrays(space)
    geometry = geometry_metrics(matrix, valence, arousal)
    heldout, cv = cv_summary(matrix, valence, arousal)
    heldout["emotion"] = [emotions[int(i)] for i in heldout["held_out"]]
    observed = {**geometry, **cv}

    _, half_a, _, _ = anchor_arrays(space, space.half_a)
    _, half_b, _, _ = anchor_arrays(space, space.half_b)
    comp_a = factorial_components(half_a, valence, arousal)
    comp_b = factorial_components(half_b, valence, arousal)
    split_half = {
        f"{name}_component_cosine": cosine(comp_a[name], comp_b[name])
        for name in ("valence", "arousal", "interaction")
    }

    rng = rng_for(config.seed, "phase3b-compositionality")
    permutations, bootstraps = resampling_controls(
        matrix, valence, arousal, args.n_resamples, rng
    )
    tests = {
        "mean_contrast_cosine": {
            "observed": observed["mean_contrast_cosine"],
            "bootstrap_95": ci(bootstraps, "mean_contrast_cosine"),
            "permutation_p": empirical_p(
                permutations["mean_contrast_cosine"],
                observed["mean_contrast_cosine"],
            ),
        },
        "additive_heldout_cosine": {
            "observed": observed["additive_mean_cosine"],
            "permutation_p": empirical_p(
                permutations["additive_mean_cosine"],
                observed["additive_mean_cosine"],
            ),
        },
        "interaction_ratio": {
            "observed": observed["interaction_to_main_ratio"],
            "bootstrap_95": ci(bootstraps, "interaction_to_main_ratio"),
            "permutation_p_smaller": empirical_p(
                permutations["interaction_to_main_ratio"],
                observed["interaction_to_main_ratio"], smaller=True,
            ),
        },
    }
    evidence = bool(
        tests["mean_contrast_cosine"]["permutation_p"] < 0.05
        and tests["additive_heldout_cosine"]["permutation_p"] < 0.05
        and split_half["valence_component_cosine"] > 0.5
        and split_half["arousal_component_cosine"] > 0.5
    )

    components = factorial_components(matrix, valence, arousal)
    from safetensors.numpy import save_file
    component_path = out_dir / "phase3b_components.safetensors"
    save_file({k: np.asarray(v, dtype=np.float32) for k, v in components.items()},
              str(component_path))
    heldout_path = out_dir / "phase3b_heldout.csv"
    null_path = out_dir / "phase3b_permutation_null.csv"
    bootstrap_path = out_dir / "phase3b_bootstrap.csv"
    heldout.to_csv(heldout_path, index=False)
    permutations.to_csv(null_path, index=False)
    bootstraps.to_csv(bootstrap_path, index=False)
    figure = make_plot(
        out_dir / "phase3b_compositionality.png", observed, permutations, bootstraps
    )

    sections = {
        "run": {"stage": "phase3b_compositionality", "run_name": config.run_name,
                "output_dir": str(out_dir)},
        "source": {"vectors": str(vectors_path), "metadata": str(meta_path),
                   "content_sha256": space.content_sha256},
        "design": {"anchors": emotions, "n_anchors": len(emotions),
                   "n_resamples": args.n_resamples,
                   "model": "mu + valence*gV + arousal*gA (+ interaction*gVA)"},
        "geometry": geometry,
        "heldout": cv,
        "split_half": split_half,
        "tests": tests,
        "verdict": {"compositional_evidence": evidence,
                    "criterion": "both permutation p<.05 and valence/arousal split-half cos>.5"},
        "artifacts": {"components": str(component_path), "heldout": str(heldout_path),
                      "permutation_null": str(null_path), "bootstrap": str(bootstrap_path),
                      "figure": str(figure)},
    }
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 3b COMPOSITIONALITY -- {config.run_name}",
        sections=sections, txt_name="phase3b_gate.txt", json_name="phase3b_gate.json",
    )

    print(RULE)
    print(f"PHASE 3b -- reusable valence/arousal components   run '{config.run_name}'")
    print(RULE)
    print(f"anchors: {len(emotions)}; permutations/bootstraps: {args.n_resamples}")
    print(THIN)
    print(f"arousal contrast reused across valence : {geometry['arousal_contrast_cosine']:+.3f}")
    print(f"valence contrast reused across arousal : {geometry['valence_contrast_cosine']:+.3f}")
    print(f"mean contrast cosine                  : {geometry['mean_contrast_cosine']:+.3f} "
          f"(permutation p={tests['mean_contrast_cosine']['permutation_p']:.4f})")
    print(f"interaction / main-effect norm        : {geometry['interaction_to_main_ratio']:.3f}")
    print(f"additive held-out cosine              : {cv['additive_mean_cosine']:+.3f} "
          f"(permutation p={tests['additive_heldout_cosine']['permutation_p']:.4f})")
    print(f"with-interaction held-out cosine      : {cv['with_interaction_mean_cosine']:+.3f}")
    print(f"split-half component cosine           : valence "
          f"{split_half['valence_component_cosine']:+.3f}, arousal "
          f"{split_half['arousal_component_cosine']:+.3f}")
    print(THIN)
    print("VERDICT: " + ("evidence for reusable components" if evidence else
                         "criteria not all met; report as mixed/null"))
    print(f"figure : {figure}")
    print(f"record : {txt_path}\n         {json_path}")
    print(RULE)
    return 0 if evidence else 3


if __name__ == "__main__":
    raise SystemExit(main())
