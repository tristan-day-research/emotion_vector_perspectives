"""Phase 3 (GATE): PCA across emotion vectors -- are the axes the circumplex?

What this stage does
--------------------
Reads Phase 2's emotion vectors, stacks them into a matrix (rows = emotions),
mean-centres **across emotions**, and runs PCA. The gate is the
variance-explained table plus the PC1-PC2 scatter, so a human can see whether the
affective circumplex is there before anything is spent interpreting it.

No model, no GPU, no activation chunks. It reads two small files that
``pull-results`` already fetches, and runs in seconds -- which is the whole reason
Phase 2 writes the vectors out separately instead of leaving them implicit in
22 GiB of activations.

Mean-centring is the load-bearing step
--------------------------------------
Without it PC1 is overall affect magnitude: every emotion vector is a pooled
residual at one layer, so they share a large common component, and the first
axis of the uncentred matrix is just "how big is this vector". That is trivially
true and says nothing about emotion. ``mean_center`` is settable only so the gate
can *show* that failure rather than assert it.

Neutral is projected in, not fitted
-----------------------------------
The neutral centroid is the circumplex origin, so including it in the fit would
add a point differing from the emotions along "is there any affect at all" -- and
PC1 would partly become that contrast, reintroducing exactly what centring
removes. Instead it is projected into the fitted space, where landing near the
origin is a check that the centring did what it claims. If neutral lands far out
along PC1, PC1 is affect magnitude and the gate says so.

Three things that stop "PC1+PC2 = 60%" being mistaken for evidence
-----------------------------------------------------------------
At n=16 the variance-explained figure cannot fail, and reporting it alone would be
the quiet failure this gate exists to catch. After centring, ``n`` centroids span
rank ``n-1``, so two of fifteen components carry a large share by construction.
Three checks are reported alongside it:

1. **A null band.** Random directions with the observed norms, centred and
   decomposed the same way, ``pca_null_samples`` times. This answers "large
   compared to what" for this ``n`` and ``d``. Necessary, not sufficient: clearing
   the null says the points are not isotropic, not that the structure is affect.
2. **Split-half PC stability.** Phase 2 saved each emotion's vector refitted from
   two disjoint halves of the *topics*, so the PCA can be refitted independently
   per half and the components compared by |cosine| (the sign of a PC is
   arbitrary). This is the check that separates real covariance structure from
   noise, and unlike variance explained it *can* fail at small n. It is free.
3. **Alignment with the a priori axes.** Valence and arousal contrasts built from
   the labelled anchors, compared with the PCs two ways -- the correlation of PC
   scores with the labels, and the principal angles between the PC1-PC2 plane and
   the valence-arousal plane. The second is the honest summary of "did we recover
   the circumplex", because a circumplex is a claim about a *plane*, not about
   which axis came out first.

The labels are never fitted against: PCA is unsupervised and never sees a
quadrant. That is what keeps the alignment an independent check rather than the
answer.

PC signs are canonical, not oriented to the prediction
------------------------------------------------------
Each component's largest-magnitude entry is made positive. That is deterministic
across re-runs and, crucially, *not* chosen to make the alignment look good --
flipping PCs to match the a priori axes would let an arbitrary sign read as a
discovery. The alignment table carries signs, so "PC1 correlates -0.95 with
valence" is as informative as +0.95, and Phase 4 reads out both ends of every axis
anyway.

Usage::

    python run.py phase3                       # the gate
    python run.py phase3 --set n_pcs_to_report=15
    python run.py phase3 --set mean_center=false    # to see why centring matters
    # A Phase 5 layer sweep points at another Phase 2 artefact:
    python run.py phase3 --vectors outputs/<run>/results/phases/block40/\
phase2_emotion_vectors.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, jlens_lens, provenance
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config
from emotion_pca_jlens.phase1_stimuli import (
    NEUTRAL_QUADRANT,
    QUADRANT_ORDER,
    UNLABELLED_QUADRANT,
)

RULE = "=" * 78
THIN = "-" * 78

#: Components are saved as float32: a few MB at 171 emotions, and Phase 4 hands
#: them to a lens that upcasts to fp32 regardless.
PC_DTYPE = np.float32

#: Variance-share gap below which PC1 and PC2 count as near-degenerate, i.e. free
#: to rotate within their shared plane between refits. Two components 2 percentage
#: points apart are not separately identified by 16 noisy centroids, so a low
#: per-axis split-half cosine at that gap says nothing about whether the *plane* is
#: real. Judged on the gap rather than a formal test because the relevant question
#: is only "could these two have swapped".
DEGENERACY_GAP = 0.02


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 3 gate: PCA across emotion vectors, gated on the variance "
                    "table, PC stability and the circumplex scatter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vectors",
        type=Path,
        default=None,
        help="Phase 2 emotion-vector safetensors to read (default: this run's). Its "
             "JSON sidecar must sit alongside it. Exists so the Phase 5 layer sweep "
             "can point at a different block's vectors without a config edit",
    )
    p.add_argument(
        "--output-subdir",
        default=None,
        help="write artefacts into this subdirectory of results/phases/ instead of "
             "alongside the vectors (use when re-running variants)",
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
# Reading Phase 2's artefact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EmotionSpace:
    """Phase 2's emotion vectors plus everything needed to interpret the rows."""

    emotions: list[str]
    matrix: np.ndarray            # (n_emotions, d_model) float64
    half_a: np.ndarray            # (n_emotions, d_model) float64
    half_b: np.ndarray
    labels: dict[str, dict]       # emotion -> quadrant/valence/arousal/family/source
    target_block: int
    target_hidden_state: int
    n_layers: int
    metadata: dict                # Phase 2's full sidecar
    vectors_path: Path
    content_sha256: str

    @property
    def d_model(self) -> int:
        return int(self.matrix.shape[1])

    def quadrant(self, emotion: str) -> str:
        return str(self.labels.get(emotion, {}).get("quadrant", UNLABELLED_QUADRANT))

    def is_neutral(self, emotion: str) -> bool:
        return emotion == NEUTRAL_QUADRANT

    def label_array(self, key: str) -> np.ndarray:
        """``valence``/``arousal`` per row; 0 for neutral and unlabelled words."""
        return np.asarray(
            [float(self.labels.get(e, {}).get(key, 0) or 0) for e in self.emotions]
        )

    def anchor_mask(self) -> np.ndarray:
        """Rows with an a priori circumplex position -- what alignment is scored on.

        With ``emotions="all"`` the other 155 words are ``unlabelled`` by design:
        the PCA is fitted on all of them for a well-determined covariance
        structure, and only the balanced 16 anchors carry valence/arousal labels
        worth scoring against. Hand-labelling 155 words would invent precision.
        """
        return np.asarray([self.quadrant(e) in QUADRANT_ORDER for e in self.emotions])

    def fit_mask(self, include_neutral: bool) -> np.ndarray:
        """Rows the PCA is fitted on."""
        if include_neutral:
            return np.ones(len(self.emotions), dtype=bool)
        return np.asarray([not self.is_neutral(e) for e in self.emotions])


def read_emotion_vectors(vectors_path: Path, meta_path: Path) -> EmotionSpace:
    """Load Phase 2's vectors, or explain how to produce them.

    Touches no activation chunk by construction: everything comes from these two
    files, which is what lets Phase 3 run on a laptop while the 22 GiB of
    activations stay in R2.
    """
    from safetensors.numpy import load_file

    if not vectors_path.exists():
        raise SystemExit(
            f"no emotion vectors at\n  {vectors_path}\n\n"
            "Phase 3 consumes Phase 2's output. Run it first:\n\n"
            "  python run.py phase2 --dry-run    # check the plan\n"
            "  python run.py phase2              # extract + the reliability gate\n"
        )
    if not meta_path.exists():
        raise SystemExit(
            f"{vectors_path} has no sidecar at\n  {meta_path}\n\n"
            "Row i is only an emotion because the sidecar says so, so Phase 3 will "
            "not guess.\nRe-run `python run.py phase2` (it writes both together), or "
            "pull the whole\nresults/ tree rather than the one file."
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = load_file(str(vectors_path))
    missing = [
        k for k in ("emotion_vectors", "emotion_vectors_half_a", "emotion_vectors_half_b")
        if k not in tensors
    ]
    if missing:
        raise SystemExit(
            f"{vectors_path} is missing {missing}. It was written by an older Phase 2; "
            "re-run it."
        )

    emotions = list(metadata["emotions"])
    matrix = np.asarray(tensors["emotion_vectors"], dtype=np.float64)
    if matrix.shape[0] != len(emotions):
        raise SystemExit(
            f"{vectors_path} has {matrix.shape[0]} rows but the sidecar names "
            f"{len(emotions)} emotions; the pair is inconsistent."
        )
    if len(emotions) < 3:
        raise SystemExit(
            f"only {len(emotions)} emotion vectors. After centring, n centroids span "
            "rank n-1, so\nthree is the minimum that can yield a two-dimensional "
            "plane to look for a circumplex in."
        )
    target = metadata.get("target", {})
    return EmotionSpace(
        emotions=emotions,
        matrix=matrix,
        half_a=np.asarray(tensors["emotion_vectors_half_a"], dtype=np.float64),
        half_b=np.asarray(tensors["emotion_vectors_half_b"], dtype=np.float64),
        labels=dict(metadata.get("labels", {})),
        target_block=int(target.get("block", -1)),
        target_hidden_state=int(target.get("hidden_state", -1)),
        n_layers=int(target.get("n_layers", 0)),
        metadata=metadata,
        vectors_path=vectors_path,
        # Identifies the exact numbers this PCA was fitted on, so a Phase 4 readout
        # can be tied back to one Phase 2 run rather than to "whatever was there".
        content_sha256=hashlib.sha256(
            np.ascontiguousarray(tensors["emotion_vectors"]).tobytes()
        ).hexdigest(),
    )


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PCAResult:
    """A fitted PCA over emotion centroids."""

    mean: np.ndarray                     # (d_model,) the centring mean
    components: np.ndarray               # (rank, d_model) unit rows
    explained_variance: np.ndarray       # (rank,)
    explained_variance_ratio: np.ndarray # (rank,)
    scores: np.ndarray                   # (n_fit_rows, rank)
    fit_emotions: list[str]
    mean_centered: bool

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    @property
    def cumulative_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    @property
    def participation_ratio(self) -> float:
        """``(sum L)^2 / sum L^2`` -- effective number of dimensions used.

        One number for "is the variance in a couple of axes or spread over all of
        them", which the top-2 share alone cannot say: 2 of 15 components at 60%
        looks the same whether the remaining 40% sits in one axis or thirteen.
        """
        lam = self.explained_variance
        total = float(lam.sum())
        return float(total**2 / float((lam**2).sum())) if total > 0 else 0.0

    def project(self, matrix: np.ndarray) -> np.ndarray:
        """Scores of arbitrary rows in this fitted space (e.g. the neutral centroid)."""
        return (np.asarray(matrix, dtype=np.float64) - self.mean) @ self.components.T


def _canonical_signs(components: np.ndarray) -> np.ndarray:
    """Make each component's largest-magnitude entry positive.

    A PC's sign is arbitrary, so it has to be pinned to something or two runs on
    near-identical inputs print mirrored scatters. Deliberately *not* pinned by
    agreement with the a priori valence/arousal axes: that would turn an arbitrary
    sign into an apparent finding.
    """
    flip = np.take_along_axis(
        components, np.abs(components).argmax(axis=1)[:, None], axis=1
    ).ravel() < 0
    return np.where(flip[:, None], -components, components)


def fit_pca(
    matrix: np.ndarray,
    emotions: list[str],
    mean_center: bool = True,
) -> PCAResult:
    """PCA over the rows of ``matrix`` by SVD.

    Truncated at the numerical rank, which after centring is ``n_rows - 1``:
    components beyond it have zero variance and would print as meaningless noise
    directions in the table.
    """
    X = np.asarray(matrix, dtype=np.float64)
    mean = X.mean(axis=0) if mean_center else np.zeros(X.shape[1], dtype=np.float64)
    centred = X - mean

    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    rank = int((singular > max(singular[0], 0.0) * 1e-10).sum()) if singular.size else 0
    rank = max(rank, 1)
    singular, components = singular[:rank], _canonical_signs(components[:rank])

    variance = singular**2 / max(X.shape[0] - 1, 1)
    total = float(variance.sum())
    return PCAResult(
        mean=mean,
        components=components,
        explained_variance=variance,
        explained_variance_ratio=(
            variance / total if total > 0 else np.zeros_like(variance)
        ),
        scores=centred @ components.T,
        fit_emotions=list(emotions),
        mean_centered=mean_center,
    )


def principal_angle_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between two subspaces, largest first.

    ``a`` and ``b`` are ``(k, d)`` with orthonormal rows (PCA components are). The
    singular values of ``a @ b.T`` are exactly those cosines: 1 means the subspaces
    share a direction, 0 means that direction of one is orthogonal to all of the
    other. Used instead of comparing PC1 to PC1, because a circumplex is a claim
    about a *plane* -- if valence and arousal come out as PC1/PC2 rotated 30
    degrees within their own plane, per-axis cosines look poor while the plane is
    recovered perfectly.
    """
    cosines = np.linalg.svd(np.asarray(a) @ np.asarray(b).T, compute_uv=False)
    return np.clip(cosines, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Check 1: is the variance share large compared to anything?
# --------------------------------------------------------------------------- #

def null_variance_band(
    n_rows: int,
    d_model: int,
    norms: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict:
    """Variance-explained percentiles for random directions with the observed norms.

    The point of the null: after centring, ``n`` points span rank ``n-1``, so at
    n=16 two components hold a large share whatever the data. Random unit
    directions scaled to the *observed* norms, centred and decomposed identically,
    say how large "large" has to be before it means anything.

    Norms are matched rather than assumed equal because an emotion with a much
    larger vector would dominate the covariance, and a null that ignored that would
    be easier to beat than the real thing.

    Only the eigenvalues are needed, so this goes through the ``n x n`` Gram matrix
    rather than a full SVD of ``n x d``. What it is *not*: a test that the
    structure is affective. It bounds one alternative -- isotropy -- and nothing
    more.
    """
    rank = max(n_rows - 1, 1)
    analytic = 1.0 / rank  # isotropic expectation per component
    report = {
        "n_samples": int(n_samples),
        "n_rows": int(n_rows),
        "d_model": int(d_model),
        "rank": rank,
        "analytic_isotropic_per_pc": analytic,
        "analytic_isotropic_top2": min(2.0 * analytic, 1.0),
    }
    if n_samples < 1:
        report["note"] = "disabled (pca_null_samples=0)"
        return report

    rng = rng_for(seed, "phase3_null", n_rows, d_model)
    scale = np.asarray(norms, dtype=np.float64).reshape(-1, 1)
    top1 = np.empty(n_samples)
    top2 = np.empty(n_samples)
    for i in range(n_samples):
        draw = rng.normal(size=(n_rows, d_model))
        draw *= scale / np.linalg.norm(draw, axis=1, keepdims=True)
        draw -= draw.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(draw @ draw.T)[::-1]
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        total = float(eigenvalues.sum())
        if total <= 0:  # pragma: no cover - a degenerate draw
            top1[i] = top2[i] = 0.0
            continue
        top1[i] = eigenvalues[0] / total
        top2[i] = float(eigenvalues[:2].sum()) / total

    for name, values in (("pc1", top1), ("top2", top2)):
        report[name] = {
            "p5": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return report


# --------------------------------------------------------------------------- #
# Check 2: do the PCs survive an independent refit?
# --------------------------------------------------------------------------- #

def pc_stability(space: EmotionSpace, fit_rows: np.ndarray, config: PCAJLensConfig) -> dict:
    """Refit the PCA on each topic half and compare the components.

    Free, because Phase 2 saved every emotion's vector refitted from two disjoint
    halves of its *topics*. This is the check that can actually fail: variance
    explained cannot, at small n.

    Per-PC |cosine| is reported with a caveat that matters. Matching PC *k* of one
    half to PC *k* of the other is only meaningful when their eigenvalues are well
    separated -- two components with nearly equal variance are free to swap or mix
    between refits, and a low cosine then reflects that degeneracy rather than
    noise. The variance gap to the next component is reported beside each cosine so
    a low value can be read correctly, and the top-2 *plane* comparison is immune
    to the swap.
    """
    full = fit_pca(space.matrix[fit_rows], list(np.asarray(space.emotions)[fit_rows]),
                   mean_center=config.mean_center)
    halves = [
        fit_pca(half[fit_rows], list(np.asarray(space.emotions)[fit_rows]),
                mean_center=config.mean_center)
        for half in (space.half_a, space.half_b)
    ]
    k = min(config.n_pcs_to_report, full.rank, halves[0].rank, halves[1].rank)

    rows: list[dict] = []
    ratios = full.explained_variance_ratio
    for i in range(k):
        gap = float(ratios[i] - ratios[i + 1]) if i + 1 < full.rank else float(ratios[i])
        rows.append({
            "pc": i + 1,
            "explained_variance_ratio": float(ratios[i]),
            "variance_gap_to_next": gap,
            # |cos|: a PC's sign is arbitrary, so a -1 is perfect agreement.
            "cos_half_a_vs_half_b": abs(float(
                halves[0].components[i] @ halves[1].components[i]
            )),
            "cos_full_vs_half_a": abs(float(full.components[i] @ halves[0].components[i])),
            "cos_full_vs_half_b": abs(float(full.components[i] @ halves[1].components[i])),
        })

    report: dict = {"per_pc": rows, "n_compared": k}
    if k >= 2:
        plane = principal_angle_cosines(halves[0].components[:2], halves[1].components[:2])
        report["top2_plane_cosines"] = [float(c) for c in plane]
        report["top2_plane_angles_deg"] = [
            float(np.degrees(np.arccos(c))) for c in plane
        ]
    stable = [r["cos_half_a_vs_half_b"] for r in rows]
    report["min_cosine"] = min(stable) if stable else None
    report["n_stable"] = sum(1 for c in stable if c >= config.pc_stability_min_cosine)
    report["threshold"] = config.pc_stability_min_cosine

    # Separate the two claims the top-2 columns can support, because a balanced
    # circumplex makes them come apart. If valence and arousal carry *equal*
    # variance -- which the 4-per-quadrant design encourages, and which a real
    # circumplex would produce -- then PC1 and PC2 are free to rotate within their
    # plane between refits. Per-axis cosines then look poor while the plane is
    # recovered exactly. Judging on the per-axis column alone would flag that
    # success as a failure, so both are recorded and the verdict reads them
    # together.
    top2 = rows[:2]
    report["axes_identified"] = bool(
        len(top2) == 2 and min(r["cos_half_a_vs_half_b"] for r in top2)
        >= config.pc_stability_min_cosine
    )
    report["top2_degenerate"] = bool(
        len(top2) == 2 and top2[0]["variance_gap_to_next"] < DEGENERACY_GAP
    )
    report["plane_stable"] = bool(
        min(report.get("top2_plane_cosines", [0.0])) >= config.pc_stability_min_cosine
    )
    return report


# --------------------------------------------------------------------------- #
# Check 3: do the PCs line up with the a priori circumplex axes?
# --------------------------------------------------------------------------- #

def _contrast_axes(
    centred: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    anchors: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """``(axes, reason)``: unit valence/arousal contrasts over already-centred rows.

    ``valence = mean(pleasant anchors) - mean(unpleasant)``, likewise arousal. All
    four arrays index the same rows.
    """
    axes: dict[str, np.ndarray] = {}
    for key, labels in (("valence", valence), ("arousal", arousal)):
        high = anchors & (labels > 0)
        low = anchors & (labels < 0)
        if not (high.any() and low.any()):
            return None, f"anchors do not span both signs of {key}"
        axis = centred[high].mean(axis=0) - centred[low].mean(axis=0)
        norm = float(np.linalg.norm(axis))
        if norm == 0:  # pragma: no cover - identical group means
            return None, f"{key} contrast is zero"
        axes[key] = axis / norm
    return axes, None


def apriori_axes(space: EmotionSpace, pca: PCAResult, anchors: np.ndarray) -> dict:
    """Valence and arousal contrast directions built from the labelled anchors.

    Balanced 4-per-quadrant labels make the two contrasts orthogonal *in label
    space* by construction (Phase 1 asserts it) -- but the resulting residual
    directions need not be, and their cosine is an empirical fact worth printing.
    If it is large, "PC1 is valence" and "PC2 is arousal" are not separable claims.

    Returned as unit directions so Phase 4 can lens them: reading out the a priori
    valence axis is a useful control on the PC readouts, since a lens that cannot
    verbalise a hand-built valence contrast is unlikely to be trusted on PC1.

    Note what these axes are *not* good for on their own. They are group means of
    the same matrix the PCA was fitted on, so a PC and a contrast can agree partly
    because they absorbed the same noise. Measured that way on a synthetic set with
    signal-to-noise near 1, the plane agreement reads 0.99 while the true plane is
    recovered at only 0.92 -- an optimistic bias of exactly the size that would
    make a null result look like a circumplex. :func:`crossfit_alignment` is the
    number to trust; this one is reported beside it and labelled within-sample.
    """
    out: dict = {"available": False}
    if not anchors.any():
        out["reason"] = "no anchors have circumplex labels"
        return out

    axes, reason = _contrast_axes(
        space.matrix - pca.mean,
        space.label_array("valence"),
        space.label_array("arousal"),
        anchors,
    )
    if axes is None:
        out["reason"] = reason
        return out
    out.update({
        "available": True,
        "valence_axis": axes["valence"],
        "arousal_axis": axes["arousal"],
        "cos_valence_arousal": float(axes["valence"] @ axes["arousal"]),
    })
    return out


def crossfit_alignment(
    space: EmotionSpace,
    fit_rows: np.ndarray,
    anchors: np.ndarray,
    config: PCAJLensConfig,
) -> dict:
    """Alignment with the a priori plane, measured out of sample.

    Builds the valence/arousal contrasts from one topic half and the PCs from the
    *other*, then compares the two planes. Both halves describe the same emotions
    but carry independent noise, so agreement can no longer come from a shared
    noise realisation the way the within-sample number can -- and Phase 2 saved the
    halves, so this costs nothing.

    Read it as a lower bound. Noise attenuates both sides, exactly as it does a
    split-half reliability, so the true alignment is at least this good. That
    asymmetry is the right way round for a gate: it cannot manufacture a circumplex
    that is not there.
    """
    out: dict = {"available": False}
    if not anchors.any():
        out["reason"] = "no anchors have circumplex labels"
        return out

    rows = fit_rows
    names = list(np.asarray(space.emotions)[rows])
    valence, arousal = (space.label_array(k)[rows] for k in ("valence", "arousal"))
    anchor_rows = anchors[rows]

    folds: list[dict] = []
    for axis_half, pc_half, tag in (
        (space.half_a, space.half_b, "axes_from_A_pcs_from_B"),
        (space.half_b, space.half_a, "axes_from_B_pcs_from_A"),
    ):
        pca_half = fit_pca(pc_half[rows], names, mean_center=config.mean_center)
        if pca_half.rank < 2:
            out["reason"] = "a half has rank < 2"
            return out
        source = axis_half[rows]
        centred = source - source.mean(axis=0) if config.mean_center else source
        axes, reason = _contrast_axes(centred, valence, arousal, anchor_rows)
        if axes is None:
            out["reason"] = reason
            return out
        plane = principal_angle_cosines(
            pca_half.components[:2], np.vstack([axes["valence"], axes["arousal"]])
        )
        folds.append({
            "fold": tag,
            "plane_cosines": [float(c) for c in plane],
            "plane_angles_deg": [float(np.degrees(np.arccos(c))) for c in plane],
            "plane_mean_cosine": float(np.mean(plane)),
            "cos_pc1_valence": float(pca_half.components[0] @ axes["valence"]),
            "cos_pc1_arousal": float(pca_half.components[0] @ axes["arousal"]),
            "cos_pc2_valence": float(pca_half.components[1] @ axes["valence"]),
            "cos_pc2_arousal": float(pca_half.components[1] @ axes["arousal"]),
        })

    out.update({
        "available": True,
        "folds": folds,
        "plane_mean_cosine": float(np.mean([f["plane_mean_cosine"] for f in folds])),
        "worst_plane_angle_deg": float(
            max(a for f in folds for a in f["plane_angles_deg"])
        ),
    })
    return out


def alignment(
    space: EmotionSpace,
    pca: PCAResult,
    fit_rows: np.ndarray,
    anchors: np.ndarray,
    axes: dict,
    n_report: int,
) -> tuple[list[dict], dict]:
    """Per-PC alignment with valence and arousal, two independent ways.

    Score correlation answers "does this PC order the emotions the way the labels
    do"; direction cosine answers "does this PC point along the hand-built
    contrast". They can disagree -- a PC can order anchors correctly while pointing
    mostly elsewhere in the 5,120-dimensional space -- and the disagreement is
    informative, so both are reported rather than one being picked.

    Correlations are computed on anchors only. With ``emotions="all"`` the other
    155 words have no labels to correlate against, and scoring them as 0 would
    quietly drag every correlation toward nothing.
    """
    # Scores are indexed over fit rows; map the anchor mask into that indexing.
    anchor_in_fit = anchors[fit_rows]
    valence = space.label_array("valence")[fit_rows][anchor_in_fit]
    arousal = space.label_array("arousal")[fit_rows][anchor_in_fit]
    k = min(n_report, pca.rank)

    def correlate(scores: np.ndarray, labels: np.ndarray) -> float | None:
        if scores.size < 3 or np.std(scores) == 0 or np.std(labels) == 0:
            return None
        return float(np.corrcoef(scores, labels)[0, 1])

    rows: list[dict] = []
    for i in range(k):
        column = pca.scores[anchor_in_fit, i]
        row = {
            "pc": i + 1,
            "explained_variance_ratio": float(pca.explained_variance_ratio[i]),
            "r_valence": correlate(column, valence),
            "r_arousal": correlate(column, arousal),
            "cos_valence": None,
            "cos_arousal": None,
        }
        if axes.get("available"):
            row["cos_valence"] = float(pca.components[i] @ axes["valence_axis"])
            row["cos_arousal"] = float(pca.components[i] @ axes["arousal_axis"])
        rows.append(row)

    summary: dict = {"n_anchors": int(anchor_in_fit.sum()), "n_pcs_scored": k}
    if axes.get("available") and pca.rank >= 2:
        # The headline: a circumplex is a claim about a plane. If valence and
        # arousal sit in the PC1-PC2 plane but rotated within it, per-axis cosines
        # look mediocre while the plane is recovered exactly -- and it is the plane
        # that Phase 4 then interprets.
        plane = principal_angle_cosines(
            pca.components[:2],
            np.vstack([axes["valence_axis"], axes["arousal_axis"]]),
        )
        summary["plane_cosines"] = [float(c) for c in plane]
        summary["plane_angles_deg"] = [float(np.degrees(np.arccos(c))) for c in plane]
        summary["plane_mean_cosine"] = float(np.mean(plane))
    best = {}
    for key in ("r_valence", "r_arousal"):
        scored = [(abs(r[key]), r["pc"]) for r in rows if r[key] is not None]
        best[key] = max(scored)[1] if scored else None
    summary["best_pc_for_valence"] = best["r_valence"]
    summary["best_pc_for_arousal"] = best["r_arousal"]
    return rows, summary


# --------------------------------------------------------------------------- #
# Artefacts
# --------------------------------------------------------------------------- #

def save_pcs(
    out_dir: Path,
    pca: PCAResult,
    space: EmotionSpace,
    axes: dict,
    metadata: dict,
    pcs_name: str,
    meta_name: str,
) -> tuple[Path, Path]:
    """Write the principal axes plus the sidecar that makes them interpretable.

    The a priori valence/arousal axes go in the same file deliberately: Phase 4
    should read them out beside the PCs, because a lens that cannot verbalise a
    hand-built valence contrast is not evidence about PC1 either way.

    Scores cover *every* emotion row, including any not fitted (neutral by
    default), so a Phase 3 scatter can be redrawn without re-reading the vectors.
    """
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    pcs_path = out_dir / pcs_name
    meta_path = out_dir / meta_name

    tensors = {
        "components": np.ascontiguousarray(pca.components, dtype=PC_DTYPE),
        "explained_variance": np.ascontiguousarray(pca.explained_variance, dtype=PC_DTYPE),
        "explained_variance_ratio": np.ascontiguousarray(
            pca.explained_variance_ratio, dtype=PC_DTYPE
        ),
        "mean": np.ascontiguousarray(pca.mean, dtype=PC_DTYPE),
        "scores_all_emotions": np.ascontiguousarray(
            pca.project(space.matrix), dtype=PC_DTYPE
        ),
    }
    if axes.get("available"):
        tensors["valence_axis"] = np.ascontiguousarray(axes["valence_axis"], dtype=PC_DTYPE)
        tensors["arousal_axis"] = np.ascontiguousarray(axes["arousal_axis"], dtype=PC_DTYPE)

    save_file(
        tensors,
        str(pcs_path),
        metadata={
            "emotions": json.dumps(space.emotions),
            "fit_emotions": json.dumps(pca.fit_emotions),
            "target_block": str(space.target_block),
            "target_hidden_state": str(space.target_hidden_state),
            "block_index_convention": "jlens residual block; hidden_state = block + 1",
            "mean_centered": str(pca.mean_centered),
            "rank": str(pca.rank),
            "dtype": np.dtype(PC_DTYPE).name,
            "row_order": "components[i] is PC i+1; scores_all_emotions[j] is emotions[j]",
        },
    )
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return pcs_path, meta_path


def write_tables(
    out_dir: Path,
    space: EmotionSpace,
    pca: PCAResult,
    fit_rows: np.ndarray,
    anchors: np.ndarray,
    variance_rows: list[dict],
    alignment_rows: list[dict],
    stability: dict,
) -> dict[str, Path]:
    """CSVs of every number that appears in a figure.

    Required, not optional: ``core.plotting``'s palette clears the
    colour-vision-deficiency separation floors only for its first three
    categorical slots, and the accompanying table is what satisfies the contrast
    requirement for the rest. It is also what makes the scatter's 172 unlabelled
    points readable at all.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_all = pca.project(space.matrix)
    k = min(scores_all.shape[1], max(len(variance_rows), 2))
    scores = pd.DataFrame(
        {f"pc{i + 1}": scores_all[:, i] for i in range(k)}
    )
    scores.insert(0, "emotion", space.emotions)
    scores.insert(1, "quadrant", [space.quadrant(e) for e in space.emotions])
    scores.insert(2, "valence", space.label_array("valence"))
    scores.insert(3, "arousal", space.label_array("arousal"))
    scores.insert(4, "is_anchor", anchors)
    scores.insert(5, "in_pca_fit", fit_rows)
    scores.insert(6, "norm", np.linalg.norm(space.matrix, axis=1))

    paths = {
        "variance_csv": out_dir / "phase3_variance.csv",
        "scores_csv": out_dir / "phase3_scores.csv",
        "alignment_csv": out_dir / "phase3_alignment.csv",
        "stability_csv": out_dir / "phase3_pc_stability.csv",
    }
    pd.DataFrame(variance_rows).to_csv(paths["variance_csv"], index=False)
    scores.to_csv(paths["scores_csv"], index=False)
    pd.DataFrame(alignment_rows).to_csv(paths["alignment_csv"], index=False)
    pd.DataFrame(stability["per_pc"]).to_csv(paths["stability_csv"], index=False)
    return paths


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
# The palette in core.plotting assigns categorical hues from three validated slots
# only. Four quadrants plus neutral would need five, so quadrant is encoded on two
# channels instead of one: hue carries valence, marker fill carries arousal. That
# respects the palette *and* reads better for a 2x2 design -- the valence and
# arousal separations become independently visible, which is exactly what the gate
# is asking a human to judge.

def _quadrant_style(quadrant: str) -> dict:
    """Marker style for one quadrant: hue = valence, fill = arousal."""
    from core import plotting

    if quadrant == NEUTRAL_QUADRANT:
        return {"color": plotting.INK_MUTED, "marker": "D", "filled": True}
    if quadrant not in QUADRANT_ORDER:
        return {"color": plotting.BASELINE, "marker": ".", "filled": True}
    positive = quadrant.endswith("-P")
    high_arousal = quadrant.startswith("HA")
    return {
        "color": plotting.SERIES[0] if positive else plotting.SERIES[1],
        "marker": "o",
        "filled": high_arousal,
    }


def plot_scree(out_dir: Path, pca: PCAResult, null: dict) -> Path:
    """Variance explained per PC and cumulative, against the isotropic null band."""
    import matplotlib.pyplot as plt

    from core import plotting

    k = min(pca.rank, 20)
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots()
    if null.get("pc1"):
        # A band, not a fourth hue: "what random directions would give" is context,
        # not a series to compare like with like.
        ax.axhspan(null["pc1"]["p5"], null["pc1"]["p95"], color=plotting.GRIDLINE,
                   zorder=0)
        ax.annotate("isotropic null, PC1 (5-95%)", xy=(k, null["pc1"]["p95"]),
                    xytext=(-2, 3), textcoords="offset points", ha="right",
                    fontsize=8, color=plotting.INK_MUTED)
    ax.plot(x, pca.explained_variance_ratio[:k], color=plotting.SERIES[0],
            marker="o", label="per component")
    ax.plot(x, pca.cumulative_ratio[:k], color=plotting.SERIES[1],
            marker="o", label="cumulative")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")
    plotting.finish(
        fig, ax,
        title="Variance explained by the principal axes of emotion space",
        subtitle=f"{pca.rank} components (rank = n_emotions - 1 after centring); "
                 f"effective dimensionality {pca.participation_ratio:.1f}",
        xlabel="principal component", ylabel="fraction of variance",
        integer_x=True,
    )
    return plotting.save(fig, out_dir / "phase3_variance_explained.png")


def _place_labels(ax, points: list[tuple[float, float, str]], min_gap_pt: float = 10.0) -> None:
    """Annotate scatter points, nudging labels apart where they would collide.

    Near-synonyms inside a quadrant land almost on top of each other -- four words
    for "calm" are close together by design -- so without this the gate's headline
    figure prints them as an illegible smear. Same idea as
    ``core.plotting.label_line_ends``, but collisions are tested in both axes: two
    labels at the same height on opposite sides of the plot do not overlap and must
    not be pushed apart.

    Requires the axes transform to be settled, so callers fix the limits first.
    """
    from core import plotting

    figure = ax.get_figure()
    placed: list[tuple[float, float, float]] = []  # (x_pt, y_pt, offset_pt)
    for x, y, text in sorted(points, key=lambda item: item[1]):
        px, py = (v / figure.dpi * 72.0 for v in ax.transData.transform((x, y)))
        offset = 0.0
        for prev_x, prev_y, prev_offset in placed:
            if abs(px - prev_x) > 62.0:  # far apart horizontally: no collision
                continue
            gap = (py + offset) - (prev_y + prev_offset)
            if abs(gap) < min_gap_pt:
                offset += min_gap_pt - gap
        placed.append((px, py, offset))
        ax.annotate(
            text, xy=(x, y), xytext=(6, offset + 3), textcoords="offset points",
            fontsize=8.5, color=plotting.INK_SECONDARY, va="center", zorder=4,
            # A leader line only where the label had to travel, so the common case
            # stays clean.
            arrowprops=(
                {"arrowstyle": "-", "color": plotting.GRIDLINE, "linewidth": 0.6,
                 "shrinkA": 0, "shrinkB": 2}
                if abs(offset) > min_gap_pt else None
            ),
        )


def plot_pc_scatter(
    out_dir: Path,
    space: EmotionSpace,
    pca: PCAResult,
    anchors: np.ndarray,
    axes: dict,
    align_summary: dict,
) -> Path | None:
    """The headline: emotions in the PC1-PC2 plane, with the a priori axes overlaid.

    Equal aspect ratio, because a circumplex is a statement about *angles* between
    emotions. A stretched axis would make an ellipse look like a circle or the
    reverse, which is the one thing this figure exists to let a human judge.

    The a priori valence and arousal axes are drawn as arrows projected into the
    plane, so the prediction sits on top of the data instead of in a table
    somewhere else.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from core import plotting

    if pca.rank < 2:
        # Only reachable if every centroid is collinear, but the figure is the whole
        # point of this gate, so it fails loudly rather than with an IndexError.
        print("  NOTE rank < 2, so there is no PC1-PC2 plane to plot.")
        return None
    scores = pca.project(space.matrix)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.grid(False)
    # A square, symmetric range with equal aspect, fixed *before* anything is drawn.
    # Equal aspect is not cosmetic here: a circumplex is a claim about angles between
    # emotions, and a stretched axis turns a ring into an ellipse or the reverse --
    # the one thing this figure exists to let a human judge. Fixing the limits up
    # front also settles the transform that _place_labels needs.
    # Extra headroom for the emotion labels, which extend right of their points.
    span = float(np.abs(scores[:, :2]).max()) * 1.32 or 1.0
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color=plotting.BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0, color=plotting.BASELINE, linewidth=0.8, zorder=1)

    # Label the words a reader can act on. With emotions="all" that is the anchors
    # plus neutral; 155 overlapping labels would make the figure unreadable and the
    # scores CSV already carries every one of them.
    small_set = len(space.emotions) <= 24
    to_label: list[tuple[float, float, str]] = []
    for i, emotion in enumerate(space.emotions):
        style = _quadrant_style(space.quadrant(emotion))
        unlabelled = space.quadrant(emotion) == UNLABELLED_QUADRANT
        ax.plot(
            [scores[i, 0]], [scores[i, 1]],
            marker=style["marker"],
            markersize=3.0 if unlabelled else 7.0,
            color=style["color"] if style["filled"] else plotting.SURFACE,
            markeredgecolor=style["color"],
            markeredgewidth=1.4,
            zorder=2 if unlabelled else 3,
        )
        if anchors[i] or space.is_neutral(emotion) or small_set:
            to_label.append((float(scores[i, 0]), float(scores[i, 1]), emotion))
    _place_labels(ax, to_label)

    if axes.get("available"):
        for name, key in (("valence", "valence_axis"), ("arousal", "arousal_axis")):
            projected = pca.components[:2] @ axes[key]
            length = float(np.linalg.norm(projected))
            if length == 0:  # pragma: no cover - an axis orthogonal to the plane
                continue
            end = projected / length * span * 0.80
            ax.annotate(
                "", xy=(end[0], end[1]), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": plotting.INK_MUTED,
                            "linewidth": 1.1, "shrinkA": 0, "shrinkB": 0},
                zorder=5,
            )
            # Labelled at the arrow's midpoint, offset perpendicular to it. At the
            # tip the text runs outside the frame -- these arrows reach ~80% of a
            # square axis, so any outward-aligned label extends past the edge, and
            # the figure is cropped to the artist bounds on save.
            perpendicular = np.array([-projected[1], projected[0]]) / length
            ax.annotate(
                f"a priori {name}  ({length:.2f} in plane)",
                xy=(end[0] * 0.55, end[1] * 0.55),
                xytext=tuple(perpendicular * 10.0),
                textcoords="offset points",
                ha="center", va="center",
                fontsize=8, color=plotting.INK_MUTED, zorder=5,
            )

    handles = [
        Line2D([], [], marker="o", linestyle="", color=plotting.SERIES[0],
               markeredgecolor=plotting.SERIES[0], label="pleasant, activated"),
        Line2D([], [], marker="o", linestyle="", color=plotting.SURFACE,
               markeredgecolor=plotting.SERIES[0], label="pleasant, deactivated"),
        Line2D([], [], marker="o", linestyle="", color=plotting.SERIES[1],
               markeredgecolor=plotting.SERIES[1], label="unpleasant, activated"),
        Line2D([], [], marker="o", linestyle="", color=plotting.SURFACE,
               markeredgecolor=plotting.SERIES[1], label="unpleasant, deactivated"),
    ]
    if any(space.is_neutral(e) for e in space.emotions):
        handles.append(Line2D([], [], marker="D", linestyle="", color=plotting.INK_MUTED,
                              label="neutral (projected, not fitted)"))
    # Count genuinely unlabelled words, not `~anchors`: neutral is not an anchor
    # either, and lumping it in reported "1 words" on a set that had none.
    n_unlabelled = sum(
        space.quadrant(e) == UNLABELLED_QUADRANT for e in space.emotions
    )
    if n_unlabelled:
        handles.append(Line2D(
            [], [], marker=".", linestyle="", color=plotting.BASELINE,
            label=f"unlabelled ({n_unlabelled} word{'' if n_unlabelled == 1 else 's'})",
        ))
    # Below the axes, not loc="best": a circumplex leaves the centre of the plot
    # empty of *points*, so "best" puts the legend exactly where the a priori arrows
    # and the neutral marker are.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8)

    plane = align_summary.get("plane_mean_cosine")
    subtitle = (f"PC1 {pca.explained_variance_ratio[0]:.0%} + "
                f"PC2 {pca.explained_variance_ratio[1]:.0%} of variance")
    if plane is not None:
        subtitle += f"; PC1-PC2 plane vs a priori plane, mean cos {plane:.2f}"
    plotting.finish(
        fig, ax,
        title="Emotion vectors in the plane of their first two principal axes",
        subtitle=subtitle,
        xlabel="PC1", ylabel="PC2",
    )
    return plotting.save(fig, out_dir / "phase3_pc1_pc2_scatter.png")


def plot_alignment(out_dir: Path, alignment_rows: list[dict]) -> Path | None:
    """PC x {valence, arousal} correlations and cosines, as a signed heatmap."""
    import matplotlib.pyplot as plt

    from core import plotting

    columns = [
        ("r_valence", "r (scores, valence)"),
        ("r_arousal", "r (scores, arousal)"),
        ("cos_valence", "cos (PC, valence axis)"),
        ("cos_arousal", "cos (PC, arousal axis)"),
    ]
    usable = [c for c in columns if any(r[c[0]] is not None for r in alignment_rows)]
    if not usable or not alignment_rows:
        return None

    matrix = np.array([
        [np.nan if row[key] is None else row[key] for key, _ in usable]
        for row in alignment_rows
    ])
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(alignment_rows) + 2.2))
    image = ax.imshow(matrix, cmap=plotting.diverging_cmap(), vmin=-1, vmax=1,
                      aspect="auto")
    ax.set_xticks(range(len(usable)), [label for _, label in usable],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(alignment_rows)), [f"PC{r['pc']}" for r in alignment_rows])
    ax.grid(False)
    ax.set_xticks(np.arange(len(usable) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(alignment_rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=plotting.SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            # Near-black on the saturated ends of the diverging ramp is about 2.7:1,
            # below the 4.5:1 floor; the surface colour is ~5.5:1 there. Switch at
            # the point the ramp stops being pale.
            saturated = abs(matrix[i, j]) > 0.55
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color=plotting.SURFACE if saturated else plotting.INK_PRIMARY)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    bar.set_label("signed alignment", color=plotting.INK_SECONDARY, fontsize=8.5)
    bar.outline.set_visible(False)
    ax.set_title("Alignment of each PC with the a priori circumplex axes",
                 loc="left", pad=10, color=plotting.INK_PRIMARY)
    fig.tight_layout()
    return plotting.save(fig, out_dir / "phase3_alignment.png")


def make_plots(
    out_dir: Path,
    space: EmotionSpace,
    pca: PCAResult,
    anchors: np.ndarray,
    axes: dict,
    alignment_rows: list[dict],
    align_summary: dict,
    null: dict,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    from core import plotting

    plotting.apply_style()
    written = [plot_scree(out_dir, pca, null)]
    for figure in (plot_pc_scatter(out_dir, space, pca, anchors, axes, align_summary),
                   plot_alignment(out_dir, alignment_rows)):
        if figure is not None:
            written.append(figure)
    return written


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def print_header(config: PCAJLensConfig, space: EmotionSpace, out_dir: Path) -> None:
    print(RULE)
    print(f"PHASE 3 GATE -- PCA across emotions   run '{config.run_name}'")
    print(RULE)
    print(f"vectors : {space.vectors_path}")
    print(f"outputs : {out_dir}")
    print()
    print("No model, no GPU, no activation chunks: this reads two small files under")
    print("results/ and runs in seconds. That separation is why Phase 2 writes the")
    print("vectors out rather than leaving them implicit in 22 GiB of activations.")
    print()


def print_space_summary(space: EmotionSpace) -> None:
    print(RULE)
    print("STEP 1  What Phase 2 handed over")
    print(RULE)
    anchors = space.anchor_mask()
    norms = np.linalg.norm(space.matrix, axis=1)
    print(f"emotion vectors : {len(space.emotions)} x {space.d_model}")
    print(f"  anchors       : {int(anchors.sum())} with a priori circumplex labels")
    unlabelled = int(sum(space.quadrant(e) == UNLABELLED_QUADRANT for e in space.emotions))
    if unlabelled:
        print(f"  unlabelled    : {unlabelled} (anchor design: fitted, not scored "
              "for alignment)")
    print(f"  neutral row   : "
          f"{'present (projected, not fitted)' if any(space.is_neutral(e) for e in space.emotions) else 'absent'}")
    described = (
        jlens_lens.describe_block(space.target_block, space.n_layers)
        if space.n_layers else f"block {space.target_block}"
    )
    print(f"  target block  : {described}")
    print(f"  vector norms  : min {norms.min():.2f}, median {np.median(norms):.2f}, "
          f"max {norms.max():.2f}  (CV {norms.std() / max(norms.mean(), 1e-12):.3f})")
    if norms.std() / max(norms.mean(), 1e-12) > 0.15:
        print("    NOTE norms differ by more than 15%. PCA is on raw vectors, so a")
        print("         large-norm emotion carries more weight than a small one. Worth")
        print("         re-checking any conclusion against unit-normalised rows.")
    print(f"  content hash  : {space.content_sha256[:16]}...")

    # Carry Phase 2's verdict forward. Running PCA on vectors that failed their own
    # reliability gate is precisely the mistake the gates exist to prevent, and it
    # is invisible from inside Phase 3 unless it is printed.
    summary = space.metadata.get("split_half", {}).get("summary", {})
    threshold = space.metadata.get("split_half", {}).get("threshold")
    minimum = summary.get("min_cosine_centered")
    print()
    if minimum is None:
        print("  Phase 2 reliability : not scored -- treat every number below as "
              "provisional.")
    else:
        verdict = "PASS" if threshold is not None and minimum >= threshold else "REVIEW"
        print(f"  Phase 2 reliability : {verdict}  (min centred split-half cosine "
              f"{minimum:.3f}"
              + (f", threshold {threshold:.2f})" if threshold is not None else ")"))
        if verdict == "REVIEW":
            print("    The vectors did not clear their own gate. A PCA of noisy centroids")
            print("    still returns axes -- they are just axes of the noise. Read Phase 2's")
            print("    table before reading anything below as structure.")


def print_variance_table(pca: PCAResult, rows: list[dict]) -> None:
    print()
    print(RULE)
    print("STEP 2  Variance explained")
    print(RULE)
    print(f"mean-centred across emotions: {pca.mean_centered}")
    if not pca.mean_centered:
        print("  WARNING mean_center=False. PC1 is now dominated by the component every")
        print("          emotion vector shares, i.e. overall affect magnitude. This is")
        print("          the failure the centring exists to prevent; the axes below are")
        print("          not comparable with a centred run.")
    print(f"rank                        : {pca.rank}  (= n_fitted - 1 after centring)")
    print(f"effective dimensionality    : {pca.participation_ratio:.2f}  "
          "((sum L)^2 / sum L^2)")
    print()
    print(f"{'PC':>4}{'variance':>12}{'share':>9}{'cumulative':>13}"
          f"{'null p50':>11}{'null p95':>11}")
    print(THIN)
    for row in rows:
        null_p50 = "" if row["null_p50"] is None else f"{row['null_p50']:>10.1%}"
        null_p95 = "" if row["null_p95"] is None else f"{row['null_p95']:>10.1%}"
        print(f"{row['pc']:>4}{row['explained_variance']:>12.4g}"
              f"{row['explained_variance_ratio']:>9.1%}{row['cumulative_ratio']:>13.1%}"
              f"{null_p50:>11}{null_p95:>11}")
    print(THIN)


def print_null_note(pca: PCAResult, null: dict) -> None:
    if not null.get("pc1"):
        print(f"  null band disabled (pca_null_samples={null['n_samples']}); "
              f"isotropic reference is {null['analytic_isotropic_top2']:.1%} for two "
              "components.")
        return
    top2 = float(pca.cumulative_ratio[1]) if pca.rank >= 2 else float(pca.cumulative_ratio[0])
    band = null["top2"]
    print(f"  PC1+PC2 observed {top2:.1%} vs isotropic null "
          f"{band['p5']:.1%}-{band['p95']:.1%} "
          f"(median {band['p50']:.1%}, max of {null['n_samples']} draws {band['max']:.1%})")
    if top2 > band["max"]:
        print("  Above every null draw: the centroids are not isotropically arranged.")
    elif top2 > band["p95"]:
        print("  Above the 95th percentile of the null, but not above every draw.")
    else:
        print("  INSIDE the null band. At this n, two components hold this much variance")
        print("  by construction -- there is no structural claim to make here.")
    print()
    print("  What this does and does not settle. It bounds one alternative -- that the")
    print(f"  {null['n_rows']} centroids are isotropic in {null['d_model']} dimensions -- "
          "and nothing more.")
    print("  It is not evidence that the axes are valence and arousal; that is what the")
    print("  alignment check and Phase 4's readout are for. And at small n the honest")
    print("  reading of the share itself is still the README's: with 16 emotions PC1/PC2")
    print("  are large almost by construction, and the number only becomes interpretable")
    print("  on the 171-emotion run.")


def print_stability(stability: dict) -> None:
    print()
    print(RULE)
    print("STEP 3  Do the axes survive an independent refit? (split-half by topic)")
    print(RULE)
    print("Phase 2 saved each emotion's vector refitted from two disjoint halves of its")
    print("topics, so the whole PCA can be refitted per half at no cost. Unlike variance")
    print("explained, this check can fail -- which is what makes it worth reading.")
    print()
    print(f"{'PC':>4}{'share':>9}{'gap to next':>13}{'|cos| A vs B':>15}"
          f"{'|cos| full-A':>14}{'|cos| full-B':>14}")
    print(THIN)
    for row in stability["per_pc"]:
        flag = "" if row["cos_half_a_vs_half_b"] >= stability["threshold"] else "  <- unstable"
        degenerate = "  (near-degenerate)" if row["variance_gap_to_next"] < 0.01 else ""
        print(f"{row['pc']:>4}{row['explained_variance_ratio']:>9.1%}"
              f"{row['variance_gap_to_next']:>13.1%}"
              f"{row['cos_half_a_vs_half_b']:>15.3f}"
              f"{row['cos_full_vs_half_a']:>14.3f}{row['cos_full_vs_half_b']:>14.3f}"
              f"{flag}{degenerate}")
    print(THIN)
    if "top2_plane_angles_deg" in stability:
        angles = stability["top2_plane_angles_deg"]
        print(f"  top-2 PLANE across halves: principal angles "
              f"{angles[0]:.1f} deg, {angles[1]:.1f} deg "
              f"(cos {stability['top2_plane_cosines'][0]:.3f}, "
              f"{stability['top2_plane_cosines'][1]:.3f})")
        print("    Read this before the per-PC column. Two components with nearly equal")
        print("    variance are free to swap or mix between refits, so a low per-PC cosine")
        print("    at a small variance gap is degeneracy, not noise -- but the plane they")
        print("    span is unaffected, which is why the plane is the robust statement.")
    print(f"  {stability['n_stable']}/{stability['n_compared']} components at "
          f"|cos| >= {stability['threshold']:.2f}")


def print_alignment(rows: list[dict], summary: dict, axes: dict,
                    crossfit: dict) -> None:
    print()
    print(RULE)
    print("STEP 4  Alignment with the a priori valence and arousal axes")
    print(RULE)
    if not axes.get("available"):
        print(f"  not scored: {axes.get('reason', 'no labelled anchors')}")
        print("  The PCA above is unaffected -- alignment is a check on it, not an input.")
        return
    print(f"Scored on {summary['n_anchors']} labelled anchors. The PCA never saw a label:")
    print("it is unsupervised, which is what keeps this an independent check rather than")
    print("the answer. Signs are meaningful here but arbitrary in the PCs themselves --")
    print("r = -0.95 is as strong as +0.95.")
    print()
    print(f"cos(a priori valence, a priori arousal) = {axes['cos_valence_arousal']:+.3f}")
    if abs(axes["cos_valence_arousal"]) > 0.3:
        print("  These two contrasts are NOT close to orthogonal in residual space, even")
        print("  though the 4-per-quadrant design makes them orthogonal in label space.")
        print("  'PC1 is valence' and 'PC2 is arousal' are then not separable claims.")
    print()
    print(f"{'PC':>4}{'share':>9}{'r valence':>12}{'r arousal':>12}"
          f"{'cos valence':>14}{'cos arousal':>14}")
    print(THIN)
    for row in rows:
        def fmt(key: str, width: int) -> str:
            return "n/a".rjust(width) if row[key] is None else f"{row[key]:>+{width}.3f}"
        print(f"{row['pc']:>4}{row['explained_variance_ratio']:>9.1%}"
              f"{fmt('r_valence', 12)}{fmt('r_arousal', 12)}"
              f"{fmt('cos_valence', 14)}{fmt('cos_arousal', 14)}")
    print(THIN)
    print(f"  best PC for valence: PC{summary['best_pc_for_valence']}   "
          f"best PC for arousal: PC{summary['best_pc_for_arousal']}")
    if "plane_angles_deg" in summary:
        angles = summary["plane_angles_deg"]
        print()
        print("  PC1-PC2 plane vs the a priori valence-arousal plane. A circumplex is a")
        print("  claim about a *plane*, not about which axis came out first: valence and")
        print("  arousal rotated within the PC1-PC2 plane still recover it, and per-axis")
        print("  cosines would understate that. Two versions, and the order matters --")
        print()
        if crossfit.get("available"):
            print(f"    CROSS-FIT   mean cosine {crossfit['plane_mean_cosine']:.3f}  "
                  f"(worst angle {crossfit['worst_plane_angle_deg']:.1f} deg)")
            print("                axes from one topic half, PCs from the other. This is")
            print("                the number to trust, and it is a LOWER bound: noise")
            print("                attenuates both sides, so the true alignment is at")
            print("                least this good.")
        else:
            print(f"    CROSS-FIT   unavailable: {crossfit.get('reason', 'unknown')}")
        print(f"    within-sample mean cosine {summary['plane_mean_cosine']:.3f} "
              f"(angles {angles[0]:.1f}, {angles[1]:.1f} deg)")
        print("                optimistically biased, and not by a little: the contrasts")
        print("                are group means of the same matrix the PCA was fitted on,")
        print("                so both absorb the same noise. On a synthetic set with")
        print("                signal-to-noise near 1 this reads 0.99 where the true")
        print("                plane is recovered at 0.92. Reported for comparison only.")


def print_neutral_check(space: EmotionSpace, pca: PCAResult, fit_rows: np.ndarray) -> None:
    if fit_rows.all():
        print()
        print("  neutral was INCLUDED in the fit (include_neutral_in_pca=True), so the")
        print("  origin check below is unavailable and PC1 may partly be affect magnitude.")
        return
    neutral = [i for i, e in enumerate(space.emotions) if space.is_neutral(e)]
    if not neutral:
        return
    scores = pca.project(space.matrix)
    spread = float(np.linalg.norm(scores[fit_rows][:, :2], axis=1).mean())
    point = scores[neutral[0], :2]
    distance = float(np.linalg.norm(point))
    print()
    print(f"  neutral projects to (PC1, PC2) = ({point[0]:+.3g}, {point[1]:+.3g}); "
          f"|.| = {distance:.3g}")
    print(f"  mean |.| of the fitted emotions = {spread:.3g}  "
          f"-> ratio {distance / max(spread, 1e-12):.2f}")
    if distance > spread:
        print("    Neutral sits FURTHER from the origin than a typical emotion. PC1 is")
        print("    then partly 'how much affect is present', which is the contrast the")
        print("    centring was meant to remove. Treat PC1's interpretation with care.")
    else:
        print("    Near the origin, as the circumplex predicts: neutral is the absence of")
        print("    affect rather than a direction in it. The centring did what it claims.")


def print_verdict(
    pca: PCAResult,
    null: dict,
    stability: dict,
    align_summary: dict,
    crossfit: dict,
    space: EmotionSpace,
    artifacts: dict[str, object],
) -> bool:
    top2 = float(pca.cumulative_ratio[min(1, pca.rank - 1)])
    beats_null = bool(null.get("top2") and top2 > null["top2"]["p95"])
    identified = stability.get("axes_identified", False)
    plane_only = stability.get("plane_stable", False) and stability.get("top2_degenerate")
    stable = bool(identified or plane_only)
    plane = crossfit.get("plane_mean_cosine")
    within = align_summary.get("plane_mean_cosine")
    phase2 = space.metadata.get("split_half", {}).get("summary", {})
    phase2_threshold = space.metadata.get("split_half", {}).get("threshold")
    phase2_min = phase2.get("min_cosine_centered")

    print()
    print(RULE)
    print("PHASE 3 VERDICT")
    print(RULE)
    print(f"  variance in PC1+PC2   : {top2:.1%}  "
          + ("above the null band" if beats_null else "NOT above the null band")
          + (f" ({null['top2']['p95']:.1%} at p95)" if null.get("top2") else ""))
    per_axis = (
        f"{min(r['cos_half_a_vs_half_b'] for r in stability['per_pc'][:2]):.3f}"
        if len(stability["per_pc"]) >= 2 else "n/a"
    )
    print(f"  top-2 PC stability    : {'PASS' if stable else 'REVIEW'}  "
          f"(per-axis |cos| {per_axis}, threshold {stability['threshold']:.2f})")
    if plane_only and not identified:
        plane = min(stability.get("top2_plane_cosines", [0.0]))
        print(f"                          The PLANE is stable (|cos| {plane:.3f}) while the")
        print("                          individual axes are not identified: PC1 and PC2")
        print("                          differ by less than "
              f"{DEGENERACY_GAP:.0%} of variance, so they are")
        print("                          free to rotate within it between refits. That is")
        print("                          what a balanced circumplex looks like, not a")
        print("                          failure -- but do not call PC1 'the valence axis'.")
    if plane is None:
        reason = crossfit.get("reason", "no labelled anchors")
        print(f"  circumplex alignment  : NOT SCORED ({reason})")
        if within is not None:
            print(f"                          within-sample only: {within:.3f}")
    else:
        print(f"  circumplex alignment  : cross-fit mean cosine {plane:.3f} between the")
        print("                          PC1-PC2 plane and the a priori valence-arousal")
        print(f"                          plane (within-sample {within:.3f}, biased high)")
    if phase2_min is not None and phase2_threshold is not None:
        print(f"  Phase 2 vectors       : "
              f"{'PASS' if phase2_min >= phase2_threshold else 'REVIEW'} "
              f"(min split-half {phase2_min:.3f})")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<9}: {path}" if label else f"  {'':<9}  {path}")
    print()
    print("  Look at phase3_pc1_pc2_scatter.png before reading any number above. The")
    print("  question it answers is the one a table cannot: are the four quadrants in")
    print("  four places, and do the a priori arrows point along the spread? A high")
    print("  variance share with the quadrants interleaved is not a circumplex.")
    print()
    if len(pca.fit_emotions) <= 24:
        print(f"  Then run the 171-emotion design. With {len(pca.fit_emotions)} emotions the")
        print("  variance share is large almost by construction, so it is the *stability*")
        print("  and *alignment* rows above that carry the weight here. The share only")
        print("  becomes interpretable at 171, where a circumplex is also the standard")
        print("  psychometric finding:")
        print("    python run.py phase1 --set emotions=all --set stories_per_emotion=200 \\")
        print("        --set run_name=<new>")
        print("    python run.py phase2 --set emotions=all --set stories_per_emotion=200 \\")
        print("        --set run_name=<new>")
        print()
    print("STOPPING at the Phase 3 gate, as agreed. Phase 4 (lensing the PCs) has not run.")
    print(RULE)
    return bool(beats_null and stable)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    if config.remove_neutral_pcs:
        print(
            "remove_neutral_pcs=True is not implemented in Phase 3.\n\n"
            "The neutral subspace is an SVD of the neutral *stories*, and Phase 3 only\n"
            "has their centroid -- fitting it needs the activation chunks, which this\n"
            "phase deliberately never touches. It belongs in Phase 5 beside the layer\n"
            "sweep, as a robustness check rather than a default.\n"
            "Re-run with --set remove_neutral_pcs=false.",
            file=sys.stderr,
        )
        return 2

    vectors_path = args.vectors or config.emotion_vectors_path
    meta_path = (
        vectors_path.with_name(config.emotion_vectors_meta_path.name)
        if args.vectors else config.emotion_vectors_meta_path
    )
    out_dir = vectors_path.parent
    if args.output_subdir:
        out_dir = config.phase_dir / args.output_subdir

    space = read_emotion_vectors(vectors_path, meta_path)
    print_header(config, space, out_dir)
    print_space_summary(space)

    # --- fit -------------------------------------------------------------- #
    fit_rows = space.fit_mask(config.include_neutral_in_pca)
    if int(fit_rows.sum()) < 3:
        print(f"\nABORTED: only {int(fit_rows.sum())} rows to fit; need at least 3.",
              file=sys.stderr)
        return 3
    anchors = space.anchor_mask()
    pca = fit_pca(
        space.matrix[fit_rows],
        list(np.asarray(space.emotions)[fit_rows]),
        mean_center=config.mean_center,
    )

    null = null_variance_band(
        n_rows=int(fit_rows.sum()),
        d_model=space.d_model,
        norms=np.linalg.norm(space.matrix[fit_rows] - pca.mean, axis=1),
        n_samples=config.pca_null_samples,
        seed=config.seed,
    )
    n_report = min(config.n_pcs_to_report, pca.rank)
    variance_rows = [
        {
            "pc": i + 1,
            "explained_variance": float(pca.explained_variance[i]),
            "explained_variance_ratio": float(pca.explained_variance_ratio[i]),
            "cumulative_ratio": float(pca.cumulative_ratio[i]),
            "null_p50": null["pc1"]["p50"] if i == 0 and null.get("pc1") else None,
            "null_p95": null["pc1"]["p95"] if i == 0 and null.get("pc1") else None,
        }
        for i in range(n_report)
    ]
    print_variance_table(pca, variance_rows)
    print_null_note(pca, null)
    print_neutral_check(space, pca, fit_rows)

    stability = pc_stability(space, fit_rows, config)
    print_stability(stability)

    axes = apriori_axes(space, pca, anchors)
    alignment_rows, align_summary = alignment(
        space, pca, fit_rows, anchors, axes, config.n_pcs_to_report
    )
    crossfit = crossfit_alignment(space, fit_rows, anchors, config)
    print_alignment(alignment_rows, align_summary, axes, crossfit)

    # --- artefacts -------------------------------------------------------- #
    tables = write_tables(
        out_dir, space, pca, fit_rows, anchors,
        variance_rows, alignment_rows, stability,
    )
    figures: list[Path] = []
    if config.make_plots:
        figures = make_plots(
            out_dir, space, pca, anchors, axes, alignment_rows, align_summary, null,
        )

    metadata = {
        "run": {"stage": "phase3_pca", "run_name": config.run_name,
                "output_dir": str(out_dir)},
        "source": {
            "emotion_vectors": str(vectors_path),
            "emotion_vectors_metadata": str(meta_path),
            "content_sha256": space.content_sha256,
            "phase2_fingerprint": space.metadata.get("fingerprint", {}),
            "phase2_split_half": space.metadata.get("split_half", {}).get("summary", {}),
        },
        "target": space.metadata.get("target", {}),
        "emotions": space.emotions,
        "fit_emotions": pca.fit_emotions,
        "row_order": "components[i] is PC i+1; scores_all_emotions[j] is emotions[j]",
        # Carried forward from Phase 2 so this sidecar is self-describing: Phase 4
        # needs valence/arousal to test whether a PC's two ends read out as opposite
        # affect, and deriving them from the circumplex table there instead would
        # silently disagree if that table were edited between phases.
        "labels": space.labels,
        "pca": {
            "mean_centered": pca.mean_centered,
            "include_neutral_in_pca": config.include_neutral_in_pca,
            "rank": pca.rank,
            "d_model": space.d_model,
            "participation_ratio": pca.participation_ratio,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio],
            "cumulative_ratio": [float(v) for v in pca.cumulative_ratio],
        },
        "null_band": {k: v for k, v in null.items()},
        "pc_stability": stability,
        "alignment": {
            "per_pc": alignment_rows,
            "summary": align_summary,
            "crossfit": crossfit,
            "cos_valence_arousal": axes.get("cos_valence_arousal"),
            "axes_available": bool(axes.get("available")),
            "reason": axes.get("reason"),
        },
        "artifacts": {
            **{k: str(v) for k, v in tables.items()},
            "figures": [str(p) for p in figures],
        },
        "written": provenance.utc_timestamp(),
    }
    pcs_path, pcs_meta_path = save_pcs(
        out_dir, pca, space, axes, metadata,
        pcs_name=config.pcs_path.name,
        meta_name=config.pcs_meta_path.name,
    )

    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 3 GATE -- {config.run_name}",
        sections={
            "run": metadata["run"],
            "config": config.to_dict(),
            "source": metadata["source"],
            "pca": metadata["pca"],
            "null_band": metadata["null_band"],
            "pc_stability": stability,
            "alignment": metadata["alignment"],
            "artifacts": {**metadata["artifacts"], "pcs": str(pcs_path)},
        },
        txt_name="phase3_gate.txt",
        json_name="phase3_gate.json",
    )

    print_verdict(
        pca, null, stability, align_summary, crossfit, space,
        artifacts={
            "pcs": pcs_path,
            "metadata": pcs_meta_path,
            "tables": ", ".join(p.name for p in tables.values()),
            "figures": ", ".join(p.name for p in figures) or "(disabled)",
            "records": txt_path,
            "": json_path,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
