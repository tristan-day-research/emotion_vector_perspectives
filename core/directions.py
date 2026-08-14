"""Mean-difference emotion directions with neutral-PC removal.

Method (Anthropic, Sofroniew et al. 2026; same construction as Ryan Codrai's
``gemma-emotional-probes``), per layer:

1. ``centroid[e]``  = mean pooled activation over stories for emotion ``e``
2. ``global_mean``  = mean over the ``n_emotions`` centroids (equal weight per
   emotion, *not* per story -- unbalanced story counts must not tilt it)
3. ``mean_diff[e]`` = ``centroid[e] - global_mean``
4. centre the neutral activations, take their SVD at this layer
5. ``k`` = fewest neutral PCs explaining >= ``variance_threshold`` of neutral variance
6. project ``mean_diff[e]`` off that ``k``-dimensional subspace
7. unit-normalise

The result is a **mean-difference direction**, not a trained probe: no labels are
fitted, no objective is optimised. Logistic-regression probes, when we add them,
live separately and must not be conflated with these.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DIRECTIONS_FILE = "directions.safetensors"
METADATA_FILE = "directions_metadata.json"


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def unit_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Scale to unit L2 norm along ``axis``; near-zero vectors are left at zero."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    safe = np.where(norms < eps, 1.0, norms)
    out = vectors / safe
    return np.where(norms < eps, 0.0, out)


@dataclass(frozen=True)
class NeutralSubspace:
    """Nuisance subspace estimated from emotionally neutral text at one layer."""

    mean: np.ndarray            # (hidden,)
    components: np.ndarray      # (k, hidden), orthonormal rows
    explained_variance_ratio: np.ndarray  # (n_components_total,)
    n_components: int
    cumulative_variance: float
    n_samples: int


def fit_neutral_subspace(
    neutral: np.ndarray,
    variance_threshold: float = 0.50,
) -> NeutralSubspace:
    """PCA/SVD of centred neutral activations; keep the PCs reaching the threshold.

    Args:
        neutral: ``(n_samples, hidden)`` pooled neutral activations.
        variance_threshold: fraction of neutral variance the kept PCs must explain.
    """
    if neutral.ndim != 2:
        raise ValueError(f"neutral must be 2-D, got shape {neutral.shape}")
    if neutral.shape[0] < 2:
        raise ValueError(f"need >= 2 neutral samples for PCA, got {neutral.shape[0]}")
    if not 0 < variance_threshold < 1:
        raise ValueError("variance_threshold must be in (0, 1)")

    X = np.asarray(neutral, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean

    # Economy SVD: Vt rows are the principal directions, S**2 the (unnormalised)
    # explained variances.
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    variance = S**2
    total = variance.sum()
    if total <= 0:
        raise ValueError("neutral activations have zero variance")
    evr = variance / total
    cumulative = np.cumsum(evr)
    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k = min(k, Vt.shape[0])

    return NeutralSubspace(
        mean=mean.astype(np.float32),
        components=np.ascontiguousarray(Vt[:k]).astype(np.float32),
        explained_variance_ratio=evr.astype(np.float32),
        n_components=k,
        cumulative_variance=float(cumulative[k - 1]),
        n_samples=int(X.shape[0]),
    )


def project_out(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of each row of ``vectors`` inside ``span(basis)``.

    ``basis`` rows are assumed orthonormal (true for SVD output), so the
    projection is ``V^T V v`` and no inverse is needed.
    """
    if basis.size == 0:
        return np.asarray(vectors, dtype=np.float64)
    V = np.asarray(basis, dtype=np.float64)
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim == 1:
        return X - V.T @ (V @ X)
    return X - (X @ V.T) @ V


# --------------------------------------------------------------------------- #
# One layer
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LayerDirections:
    """Everything produced for a single layer."""

    layer: int
    emotions: tuple[str, ...]
    centroids: np.ndarray            # (n_emotions, hidden)
    global_mean: np.ndarray          # (hidden,)
    mean_difference: np.ndarray      # (n_emotions, hidden), before projection/normalisation
    direction: np.ndarray            # (n_emotions, hidden), projected + unit norm  <- use this
    direction_unprojected: np.ndarray  # (n_emotions, hidden), unit norm, no neutral removal
    neutral: NeutralSubspace
    counts: dict[str, int]
    residual_fraction: np.ndarray    # (n_emotions,) ||projected|| / ||mean_diff||


def fit_layer_directions(
    layer: int,
    emotion_activations: Mapping[str, np.ndarray],
    neutral_activations: np.ndarray,
    variance_threshold: float = 0.50,
) -> LayerDirections:
    """Build mean-difference directions with neutral-PC removal for one layer.

    Args:
        layer: hidden-state index (recorded, not used in the maths).
        emotion_activations: ``{emotion: (n_stories, hidden)}``. Story counts may
            differ between emotions; each emotion still gets equal weight in the
            global mean.
        neutral_activations: ``(n_neutral, hidden)``.
        variance_threshold: neutral variance fraction to project out.
    """
    emotions = tuple(sorted(emotion_activations))
    if len(emotions) < 2:
        raise ValueError(
            "at least two emotions are required: subtracting the mean across emotion "
            "centroids is meaningless with one emotion"
        )

    hidden = int(neutral_activations.shape[1])
    centroids = np.empty((len(emotions), hidden), dtype=np.float64)
    counts: dict[str, int] = {}
    for i, emotion in enumerate(emotions):
        acts = np.asarray(emotion_activations[emotion])
        if acts.ndim != 2 or acts.shape[1] != hidden:
            raise ValueError(
                f"{emotion}: expected (n, {hidden}) activations, got {acts.shape}"
            )
        if acts.shape[0] == 0:
            raise ValueError(f"{emotion}: no activations to average at layer {layer}")
        centroids[i] = acts.mean(axis=0, dtype=np.float64)
        counts[emotion] = int(acts.shape[0])

    # Equal weight per emotion.
    global_mean = centroids.mean(axis=0)
    mean_difference = centroids - global_mean

    neutral = fit_neutral_subspace(neutral_activations, variance_threshold)
    projected = project_out(mean_difference, neutral.components)

    md_norms = np.linalg.norm(mean_difference, axis=1)
    proj_norms = np.linalg.norm(projected, axis=1)
    residual_fraction = np.divide(
        proj_norms, md_norms, out=np.zeros_like(proj_norms), where=md_norms > 0
    )

    return LayerDirections(
        layer=int(layer),
        emotions=emotions,
        centroids=centroids.astype(np.float32),
        global_mean=global_mean.astype(np.float32),
        mean_difference=mean_difference.astype(np.float32),
        direction=unit_normalize(projected).astype(np.float32),
        direction_unprojected=unit_normalize(mean_difference).astype(np.float32),
        neutral=neutral,
        counts=counts,
        residual_fraction=residual_fraction.astype(np.float32),
    )


def mean_difference_directions(
    emotion_activations: Mapping[str, np.ndarray],
    neutral_activations: np.ndarray | None = None,
    variance_threshold: float = 0.50,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Convenience wrapper returning just ``(emotions, unit directions)``.

    Used by evaluation's bootstrap and split-half checks, where only the final
    directions matter. Pass ``neutral_activations=None`` to skip PC removal.
    """
    emotions = tuple(sorted(emotion_activations))
    centroids = np.stack(
        [np.asarray(emotion_activations[e]).mean(axis=0, dtype=np.float64) for e in emotions]
    )
    mean_diff = centroids - centroids.mean(axis=0)
    if neutral_activations is not None:
        subspace = fit_neutral_subspace(neutral_activations, variance_threshold)
        mean_diff = project_out(mean_diff, subspace.components)
    return emotions, unit_normalize(mean_diff).astype(np.float32)


# --------------------------------------------------------------------------- #
# Multi-layer container: save / load
# --------------------------------------------------------------------------- #

class DirectionSet:
    """All layers' directions, with the metadata needed to interpret them.

    This is the interface later stages (probe comparison, cross-condition
    transfer, causal steering) should use to obtain a direction:

        >>> ds = DirectionSet.load("outputs/<run>/directions")
        >>> v = ds.direction("angry", layer=32)      # (hidden,), unit norm
        >>> scores = ds.score(activations, layer=32)  # (n, n_emotions)
    """

    def __init__(self, layers: dict[int, dict[str, np.ndarray]], metadata: dict):
        self._layers = layers
        self.metadata = metadata
        self.emotions: list[str] = list(metadata["emotions"])
        self.layer_indices: list[int] = sorted(layers)
        self._emotion_index = {e: i for i, e in enumerate(self.emotions)}

    # -- access ------------------------------------------------------------ #

    def matrix(self, layer: int, kind: str = "direction") -> np.ndarray:
        """``(n_emotions, hidden)`` matrix of the requested kind."""
        if layer not in self._layers:
            raise KeyError(f"layer {layer} not available; have {self.layer_indices}")
        arrays = self._layers[layer]
        if kind not in arrays:
            raise KeyError(f"{kind!r} not stored; have {sorted(arrays)}")
        return arrays[kind]

    def direction(self, emotion: str, layer: int, kind: str = "direction") -> np.ndarray:
        if emotion not in self._emotion_index:
            raise KeyError(f"unknown emotion {emotion!r}; have {self.emotions}")
        return self.matrix(layer, kind)[self._emotion_index[emotion]]

    def global_mean(self, layer: int) -> np.ndarray:
        return self.matrix(layer, "global_mean")

    def score(
        self,
        activations: np.ndarray,
        layer: int,
        mode: str = "dot",
        kind: str = "direction",
    ) -> np.ndarray:
        """Score activations against every emotion direction.

        Modes:
            ``dot``           raw dot product with the unit direction
            ``centered_dot``  dot product after subtracting the layer's global
                              emotion mean (the decision rule matched to how the
                              directions were built)
            ``cosine``        cosine similarity, as plotted in the paper
        """
        A = np.asarray(activations, dtype=np.float32)
        if A.ndim == 1:
            A = A[None, :]
        D = self.matrix(layer, kind)

        if mode == "dot":
            return A @ D.T
        if mode == "centered_dot":
            return (A - self.global_mean(layer)) @ D.T
        if mode == "cosine":
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return (A / norms) @ D.T
        raise ValueError(f"unknown score mode {mode!r}")

    # -- persistence ------------------------------------------------------- #

    @staticmethod
    def _key(kind: str, layer: int) -> str:
        return f"{kind}/layer_{layer:03d}"

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        from safetensors.numpy import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, np.ndarray] = {}
        for layer, arrays in self._layers.items():
            for kind, array in arrays.items():
                tensors[self._key(kind, layer)] = np.ascontiguousarray(array)

        directions_path = out_dir / DIRECTIONS_FILE
        save_file(
            tensors,
            str(directions_path),
            metadata={
                "emotions": json.dumps(self.emotions),
                "layers": json.dumps(self.layer_indices),
                "kinds": json.dumps(sorted({k for a in self._layers.values() for k in a})),
            },
        )
        metadata_path = out_dir / METADATA_FILE
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )
        return directions_path, metadata_path

    @classmethod
    def load(cls, in_dir: str | Path) -> "DirectionSet":
        from safetensors.numpy import load_file

        in_dir = Path(in_dir)
        metadata = json.loads((in_dir / METADATA_FILE).read_text(encoding="utf-8"))
        tensors = load_file(str(in_dir / DIRECTIONS_FILE))

        layers: dict[int, dict[str, np.ndarray]] = {}
        for key, array in tensors.items():
            kind, layer_part = key.rsplit("/", 1)
            layer = int(layer_part.removeprefix("layer_"))
            layers.setdefault(layer, {})[kind] = array
        return cls(layers, metadata)

    @classmethod
    def from_layer_results(
        cls,
        results: Sequence[LayerDirections],
        metadata: dict,
    ) -> "DirectionSet":
        """Assemble from :func:`fit_layer_directions` outputs."""
        if not results:
            raise ValueError("no layer results to assemble")
        emotions = results[0].emotions
        for result in results:
            if result.emotions != emotions:
                raise ValueError("layer results disagree on the emotion ordering")

        layers = {
            r.layer: {
                "direction": r.direction,
                "direction_unprojected": r.direction_unprojected,
                "mean_difference": r.mean_difference,
                "centroid": r.centroids,
                "global_mean": r.global_mean,
                "neutral_mean": r.neutral.mean,
                "neutral_components": r.neutral.components,
                "neutral_explained_variance_ratio": r.neutral.explained_variance_ratio,
                "residual_fraction": r.residual_fraction,
            }
            for r in results
        }
        merged = {
            **metadata,
            "emotions": list(emotions),
            "layers": [r.layer for r in results],
            "per_layer": {
                str(r.layer): {
                    "n_neutral_pcs": r.neutral.n_components,
                    "neutral_cumulative_variance": r.neutral.cumulative_variance,
                    "n_neutral_samples": r.neutral.n_samples,
                    "story_counts": r.counts,
                    "residual_fraction": {
                        e: float(f) for e, f in zip(r.emotions, r.residual_fraction)
                    },
                }
                for r in results
            },
        }
        return cls(layers, merged)
