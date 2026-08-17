"""Phase 6 (GATE): split each emotion vector into a reportable part and a remainder.

What this stage does
--------------------
For each Phase 2 emotion vector ``v`` at the target block, it finds ``v_J`` -- the part
the lens can read as tokens -- as a sparse nonnegative combination of ``k`` lens
dictionary directions, fitted by gradient pursuit, at each ``k`` in
``dict_atom_counts``. The remainder is ``v_perp = v - v_J``. It writes norm-matched
``v``, ``v_J``, ``v_perp`` and a matched-norm random control at ``n_dict_atoms``, which
is what Phase 8 steers with.

Depends on Phase 2's vectors and the Phase 0 lens. **Not on Phase 3, 4 or 5** --
Phase 3's PCA and Phase 5's extensions are a different question about the same
vectors, so this can run whether or not they have. Phase 3's artefact is read when
present, only to quote the circumplex context in the gate.

The dictionary: read directions, because reportability is a reading question
----------------------------------------------------------------------------
``jlens`` has no dictionary: a fitted lens is a per-layer matrix ``J``, and the
dictionary has to be constructed. The atom for token ``t`` is ``d_t = J^T (g * w_t)``,
unit-normalised.

That is not a convenient approximation, it is an identity. ``unembed`` is
``lm_head(final_norm(h))``, so for an RMSNorm the lens logit for token ``t`` at residual
``h`` is::

    logit_t = <w_t, g * (Jh / rms(Jh))>
            = <g * w_t, Jh> / rms(Jh)
            = <J^T u_t, h> / rms(Jh)          with u_t = g * w_t

So ``J^T u_t`` **is** the measurement weight vector -- the direction the lens reads
token ``t`` with -- and ``1/rms(Jh)`` is a positive scalar that scales every token's
logit together. Two consequences do the work of this whole stage:

* Any ``h`` orthogonal to every ``J^T u_t`` changes **no logit at all**. So
  ``span{J^T u_t : t in V}`` is the verbalizable subspace, by construction rather than
  by argument. Note what that does *not* buy: with a 150k vocabulary in 5,120 dimensions
  the span is generically the whole space, so the subspace alone carves out nothing. The
  work is done by the sparse nonnegative approximation from a restricted pool, which is
  what the gate measures and what the cones note below is about.
* The gain ``g`` is absorbed exactly. Dropping it would make the dictionary disagree
  with the lens it represents, by whatever the model's learned per-channel scale does.

``J^+ u_t`` is a *different* direction: the one that most efficiently **writes** token
``t``. It is available behind ``write_space`` as a labelled ablation, because "which part
of an emotion vector would most efficiently produce its words" is a real question -- just
not this one. Conflating the two is an easy mistake to make twice, which is why the
ablation stays runnable and labelled instead of being deleted.

**There is no "does lensing an atom return its own token" check here, deliberately.**
That is a property of write directions. A read direction ``J^T u_t`` has no reason to
satisfy it, and treating its failure as a defect is what argued this stage into the wrong
construction once already.

The gate: the reconstruction fraction against a random null
-----------------------------------------------------------
The only claim the method supports. For each emotion, the fraction of variance the sparse
code captures is reported against the identical decomposition run on ``n_random_controls``
matched-norm **random** directions, with the ratio and a Monte-Carlo p-value. ``k`` atoms
chosen from a pool of ``dict_pool_size`` have real degrees of freedom: if a structureless
direction reaches the same fraction, the number measures the pursuit's flexibility and
nothing about emotion. "8% of the variance" has no referent on its own, and the brief's
5-15% expectation cannot supply one.

Reported at both ``k = 16`` and ``k = 25`` -- the paper's settings -- because of the next
paragraph.

What v_perp is not
------------------
Under sparse nonnegative coding the reconstructable set is ``{sum c_i d_i : c_i >= 0,
|support| <= k}``: a **union of cones**, one per choice of ``k`` atoms from the pool, not
a linear subspace. The verbalizable *subspace* above is linear; this approximation to it
is not, and it is the approximation that gets measured.

So ``v_perp`` means **"not captured by this sparse approximation, at this k, from this
pool"**. It never means "intrinsically unverbalizable". Raising ``k``, widening the pool,
or dropping the nonnegativity constraint all move the boundary, which is exactly why both
``k`` values are reported side by side rather than one being chosen. Any downstream
sentence of the form "the model cannot verbalise this component" is unsupported by this
stage; the supportable sentence is "this component is outside the ``k``-sparse nonnegative
span of the tokens the vector is most disposed to say".

Norm matching, and why all four directions get it
-------------------------------------------------
Phase 8 adds these to the residual stream and compares behaviour across
conditions. A steering effect scales with the perturbation's size, so unmatched
norms would make "``v_perp`` moves behaviour more than ``v_J``" a statement about
``v_perp`` being longer. All four are scaled to ``||v||``, including the random
control, and the raw norms are recorded so the scaling is visible rather than
implicit.

Usage::

    python run.py phase6 --dry-run    # lens vs vectors; no model weights
    python run.py phase6              # the gate, at k=16 and k=25
    python run.py phase6 --set n_dict_atoms=25          # save the k=25 split instead
    python run.py phase6 --set write_space=true         # the J^+ ablation, labelled
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config
from emotion_pca_jlens.phase1_stimuli import NEUTRAL_QUADRANT
from emotion_pca_jlens.phase3_pca import read_emotion_vectors
from emotion_pca_jlens.phase4_lens_pcs import (
    PUBLISHED_INTERRUPTED_N_PROMPTS,
    crosscheck_lens,
    resolve_lens,
)

RULE = "=" * 78
THIN = "-" * 78

#: Atoms sampled for the read-identity check and the coherence report.
DICTIONARY_PROBE_ATOMS = 24

#: How closely the raw atoms must reproduce the lens's own logits for the identity
#: ``logit_t = <J^T u_t, h> / rms(Jh)`` to be confirmed on this lens. Exact algebra, so the
#: correlation should be 1 to float32 precision; anything lower means ``jacobians[block]``
#: is not oriented the way this code reads it, and then every atom is a unit vector
#: pointing somewhere unrelated to its token -- invisible to every shape check. This is a
#: correctness assertion about the code, not a claim about the science, which is why it
#: aborts rather than being reported.
READ_IDENTITY_MIN_CORRELATION = 0.999

#: |cosine| between atoms above which they are near-duplicates. **Reported, not gated.**
#: Two atoms this aligned make the pursuit's choice between them arbitrary, so the tokens
#: ``v_J`` names stop being attributable -- but most of the coherence here is intrinsic to
#: the pool: it is one direction's own top tokens, whose unembedding rows are aligned with
#: that direction and therefore with each other. So it is a number to read, not a gate.
DICTIONARY_MAX_COHERENCE = 0.5

#: Family-wise error rate for the gate's p-values, Bonferroni-corrected across emotions.
#: The same convention Phase 4 uses for its exploratory PCs: one threshold, corrected for
#: the number of emotions tested, so a single emotion clearing 0.05 out of sixteen is not
#: read as a result.
GATE_ALPHA = 0.05

#: Random-control fraction above which the decomposition has too many degrees of
#: freedom for the dimensionality, and no reportable fraction can be read. At
#: d=5120 with the defaults the control should land near a percent; a large value
#: means the atom count or the pool has grown until the pursuit spans the space, and
#: then a *small* emotion fraction would be as meaningless as a large one.
DEGENERATE_CONTROL_FRACTION = 0.25

#: Saved as float32: Phase 8 hands these to a forward pass that upcasts anyway.
VECTOR_DTYPE = np.float32


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 6 gate: split each emotion vector into the part the lens "
                    "can verbalise and the remainder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="cross-check the lens against the vectors and report the plan; never "
             "loads model weights",
    )
    p.add_argument(
        "--vectors", type=Path, default=None,
        help="Phase 2 emotion-vector safetensors to decompose (default: this run's)",
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
# The dictionary
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Dictionary:
    """A pool of lens dictionary atoms for one direction, at one block."""

    token_ids: np.ndarray       # (n_pool,) vocabulary ids, best-first
    atoms: np.ndarray           # (n_pool, d_model) unit rows
    raw: np.ndarray             # (n_pool, d_model) before normalisation
    raw_norms: np.ndarray       # (n_pool,) ||d_t|| before normalisation
    gain_absorbed: bool
    mode: str                   # "read" (J^T) or "write" (J^+)


def unembed_parts(readout) -> tuple[np.ndarray, np.ndarray | None]:
    """``(W_U, g)`` -- the LM head weight and the final norm's learned gain.

    Reaches into ``HFLensModel``'s private attributes, exactly as
    ``LensReadout.unembed_description`` does and for the same reason: the point is to
    use what the library will *really* apply, and there is no public accessor.
    ``requirements.txt`` pins ``jlens`` to the commit these names were read from.

    ``g`` is ``None`` when the final norm has no learned weight, which is reported
    rather than silently treated as ones -- the dictionary would then disagree with
    the lens by whatever the gain does.
    """
    import torch

    head = readout.model._lm_head.weight.detach().to(torch.float32).cpu().numpy()
    norm_weight = getattr(readout.model._final_norm, "weight", None)
    gain = (
        None if norm_weight is None
        else norm_weight.detach().to(torch.float32).cpu().numpy().reshape(-1)
    )
    return head, gain


@dataclass(frozen=True)
class Transport:
    """``J`` at one block, factored once into the pullback every atom needs.

    Factored once and reused: the emotions, the probe and all ``n_random_controls``
    share it, so a 5120x5120 SVD is paid for a single time rather than per direction.
    The SVD factors are kept so :meth:`retruncate` can rebuild the pullback at another
    ``rcond`` for the cost of one matmul -- which is what makes the ``rcond`` sweep at
    the gate affordable instead of five re-runs of the phase.
    """

    block: int
    pullback: np.ndarray        # (d, d) truncated pseudo-inverse of J
    singular_values: np.ndarray
    rank: int                   # singular values kept
    condition_number: float     # s_max / s_min over the whole spectrum
    effective_condition: float  # s_max / s_min over the kept part
    rcond: float
    device: str
    roundtrip_cosine: float
    factors: tuple = field(repr=False, default=())   # (U, s, V^T) as torch CPU tensors

    def summary(self) -> dict:
        return {
            "block": self.block, "rank": self.rank, "d_model": self.pullback.shape[0],
            "condition_number": self.condition_number,
            "effective_condition": self.effective_condition,
            "rcond": self.rcond, "device": self.device,
            "roundtrip_cosine": self.roundtrip_cosine,
            "singular_value_max": float(self.singular_values[0]),
            "singular_value_min": float(self.singular_values[-1]),
        }

    def retruncate(self, rcond: float, readout=None) -> "Transport":
        """The same SVD, truncated at a different ``rcond``. No refactorisation."""
        return _truncate(self.factors, self.block, rcond, self.device, readout)


def _truncate(factors: tuple, block: int, rcond: float, device: str, readout) -> Transport:
    """Build ``J^+`` from a finished SVD, dropping singular values below the cutoff."""
    left, values, right = factors
    singular = values.numpy()
    keep = max(int((singular > rcond * singular[0]).sum()), 1)
    # J = U S V^T, so J^+ = V S^-1 U^T over the kept directions.
    pullback = (
        right[:keep].transpose(0, 1) / values[:keep]
    ) @ left[:, :keep].transpose(0, 1)

    roundtrip = float("nan")
    if readout is not None:
        # A mid-spectrum left singular vector: inside the kept range, so this isolates
        # orientation and arithmetic from truncation loss (which the atom-validity check
        # measures end to end), and not the leading mode, which would round-trip even
        # through a badly conditioned inverse.
        probe = left[:, keep // 2]
        transported = readout.lens.transport(pullback @ probe, block).float().cpu()
        denominator = float(transported.norm() * probe.norm())
        roundtrip = (
            float(transported @ probe / denominator) if denominator > 0 else 0.0
        )
    return Transport(
        block=block,
        pullback=pullback.numpy(),
        singular_values=singular,
        rank=keep,
        condition_number=float(singular[0] / max(singular[-1], np.finfo(np.float32).tiny)),
        effective_condition=float(singular[0] / singular[keep - 1]),
        rcond=rcond,
        device=device,
        roundtrip_cosine=roundtrip,
        factors=factors,
    )


def factor_transport(readout, block: int, rcond: float) -> Transport:
    """SVD of ``J`` at ``block``, truncated, as the pullback ``J^+``.

    An atom must satisfy ``J d_t = u_t``, so building one is a solve against ``J``, not a
    multiplication by ``J^T``. The inverse is **truncated**, and that is not a numerical
    nicety -- it is the main free parameter of the construction. ``J^+`` amplifies each
    right-singular direction by ``1/s``, so the directions ``J`` nearly annihilates
    dominate every atom; since atoms are unit-normalised afterwards, the amplification
    does not wash out, it *becomes* the atom. Two symptoms follow, and only the second is
    caught by the atom-validity check:

    * atoms collapse towards the same few small-``s`` directions, so distinct tokens get
      near-duplicate atoms -- measured by :func:`dictionary_coherence`;
    * eventually the atoms stop lensing back to their own tokens at all.

    Every singular value below ``rcond * s_max`` is dropped (numpy's ``pinv``
    convention), which is why ``dict_pinv_rcond`` is a config field and why the gate
    prints ``cond(J)``, the retained rank and the coherence beside the check.

    Runs on GPU when there is one, and falls back to the CPU rather than failing: the
    model is already resident under ``device_map="auto"``, so a 5120x5120 SVD can lose
    the memory race, and one slow factorisation is better than no dictionary.

    ``roundtrip_cosine`` checks the result against ``jlens``'s own ``transport`` rather
    than against the orientation this code assumes ``jacobians[block]`` is stored in. A
    transposed ``J`` would give a pullback that is wrong in a way no shape check catches:
    every atom would still be a unit vector of the right length, just pointing somewhere
    unrelated. A cosine near 1 says the pullback really does invert the transport the
    lens applies.
    """
    import torch

    matrix = torch.as_tensor(readout.lens.jacobians[block], dtype=torch.float32).cpu()
    device = "cpu"
    if torch.cuda.is_available():
        try:
            left, values, right = torch.linalg.svd(matrix.cuda(), full_matrices=False)
            device = "cuda"
        except RuntimeError:  # OOM against the resident model, or no cuSOLVER workspace
            left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    else:
        left, values, right = torch.linalg.svd(matrix, full_matrices=False)

    factors = (left.float().cpu(), values.float().cpu(), right.float().cpu())
    return _truncate(factors, block, rcond, device, readout)


def dictionary_coherence(dictionary: Dictionary, threshold: float) -> dict:
    """Mutual |cosine| between atoms -- how attributable the selected tokens are.

    Two atoms 90% the same direction make the pursuit's choice between them arbitrary, so
    the reconstruction's *tokens* -- the part that makes ``v_J`` mean "the part the lens
    can verbalise" rather than "16 numbers" -- are not attributable, even where the
    fraction itself beats the null.

    **Reported, not gated.** Most of the coherence here is intrinsic to the pool rather
    than to the construction: the pool is one direction's own top tokens, and their
    unembedding rows are aligned with that direction and therefore with each other,
    whatever ``J`` does. Under ``write_space`` it has a second cause -- ``1/s``
    amplification pulling every atom towards the smallest-``s`` directions -- which is
    what ``dict_pinv_rcond`` controls.
    """
    atoms = dictionary.atoms
    if len(atoms) < 2:
        return {"n_atoms": len(atoms)}
    gram = np.abs(atoms @ atoms.T)
    np.fill_diagonal(gram, 0.0)
    upper = gram[np.triu_indices(len(atoms), k=1)]
    return {
        "n_atoms": len(atoms),
        "mean": float(upper.mean()),
        "max": float(upper.max()),
        "frac_above_threshold": float((upper > threshold).mean()),
        "threshold": threshold,
    }


def build_dictionary(
    readout,
    direction: np.ndarray,
    block: int,
    pool_size: int,
    head: np.ndarray,
    gain: np.ndarray | None,
    transport: Transport | None = None,
) -> Dictionary:
    """Atoms for the ``pool_size`` tokens this direction is most disposed to say.

    ``d_t = J^T (g * w_t)``, unit-normalised: the direction the lens *reads* token ``t``
    with, since ``logit_t = <J^T u_t, h> / rms(Jh)``. See the module docstring.

    The pool comes from the direction's own lens readout, so the candidate set *is* "the
    tokens ``v`` wants to say" -- which is what makes the reconstruction a reportable part
    rather than an arbitrary sparse approximation of ``v``.

    Atoms are unit-normalised, so selection in the pursuit is a correlation with the
    residual rather than a contest between long and short atoms, and the coefficients carry
    the magnitudes. The raw rows are kept for :func:`verify_read_directions`, which needs
    the unnormalised weights to check the identity.

    ``transport`` switches to the ``write_space`` ablation: ``d_t = J^+ (g * w_t)``, the
    direction that most efficiently *writes* ``t``. ``None`` -- the default -- is the read
    construction, and it needs no factorisation of ``J`` at all.
    """
    import torch

    logits = readout.direction_logits(direction, block, use_jacobian=True)
    pool = min(pool_size, logits.numel())
    token_ids = torch.topk(logits, pool).indices.cpu().numpy()

    rows = head[token_ids]
    if gain is not None:
        rows = rows * gain[None, :]

    if transport is None:
        # d_t = J^T (g * w_t). `rows` holds u_t as ROWS, so `rows @ J` gives (J^T u_t)^T
        # per row -- one matmul for the whole pool.
        jacobian = torch.as_tensor(
            readout.lens.jacobians[block], dtype=torch.float32
        ).cpu().numpy()
        raw = rows @ jacobian
        mode = "read"
    else:
        if transport.block != block:
            raise ValueError(
                f"transport was factored at block {transport.block}, asked for {block}"
            )
        raw = rows @ transport.pullback.T
        mode = "write"

    norms = np.linalg.norm(raw, axis=1)
    keep = norms > 0
    return Dictionary(
        token_ids=token_ids[keep],
        atoms=raw[keep] / norms[keep][:, None],
        raw=raw[keep],
        raw_norms=norms[keep],
        gain_absorbed=gain is not None,
        mode=mode,
    )


def verify_read_directions(
    readout, dictionary: Dictionary, block: int, rng, n_probes: int = 8
) -> dict:
    """Confirm ``logit_t = <d_t, h> / rms(Jh)`` on this lens, for random ``h``.

    The identity that *defines* a read direction, checked rather than assumed. For a random
    residual ``h``, the lens's own logits over the pool's tokens must be exactly
    proportional to ``raw_atoms @ h`` -- the constant of proportionality being the positive
    scalar ``1/rms(Jh)``, which is why this is a correlation and not a difference.

    It exists because the orientation of ``jacobians[block]`` is an assumption: ``jlens``
    exposes ``transport``, and whether the stored matrix is ``J`` or ``J^T`` is not
    something a shape check can tell you. Get it backwards and every atom is still a unit
    vector of the right length, just pointing somewhere unrelated to its token.

    This is **not** the removed "does lensing an atom return its own token" check. That
    asked a question about write directions, and read directions have no reason to satisfy
    it. This asks whether the atoms are the lens's own measurement weights, which is exact
    linear algebra -- so a failure means the code is wrong, not that the finding is weak.
    """
    d_model = dictionary.atoms.shape[1]
    correlations: list[float] = []
    for _ in range(n_probes):
        h = rng.normal(size=d_model)
        logits = readout.direction_logits(h, block, use_jacobian=True).numpy()
        predicted = dictionary.raw @ h
        observed = logits[dictionary.token_ids]
        if predicted.std() == 0 or observed.std() == 0:  # pragma: no cover
            correlations.append(0.0)
            continue
        correlations.append(float(np.corrcoef(predicted, observed)[0, 1]))
    values = np.asarray(correlations)
    return {
        "n_probes": n_probes,
        "min_correlation": float(values.min()),
        "mean_correlation": float(values.mean()),
        "holds": bool(values.min() >= READ_IDENTITY_MIN_CORRELATION),
        "threshold": READ_IDENTITY_MIN_CORRELATION,
        "mode": dictionary.mode,
        "gain_absorbed": dictionary.gain_absorbed,
    }


# --------------------------------------------------------------------------- #
# Gradient pursuit
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Decomposition:
    """One vector split into its reportable part and the remainder."""

    support: list[int]          # indices into the dictionary pool
    token_ids: list[int]
    coefficients: list[float]
    reportable: np.ndarray      # v_J
    remainder: np.ndarray       # v_perp = v - v_J
    frac_reportable: float
    frac_remainder: float
    reconstruction_error: float
    cos_parts: float            # cos(v_J, v_perp); ~0 if the fit is a projection
    n_iterations: int


def nonneg_gradient_pursuit(
    vector: np.ndarray, atoms: np.ndarray, k: int, steps: int
) -> tuple[np.ndarray, list[int]]:
    """Sparse nonnegative reconstruction of ``vector`` from unit ``atoms``.

    Greedy selection with a gradient-based coefficient update, which is what
    "gradient pursuit" names: each round adds the atom most correlated with the
    current residual, then re-fits *all* selected coefficients by projected gradient
    descent under ``c >= 0``.

    Nonnegativity is the brief's constraint and it is doing real work: a signed
    combination could reach any direction in the span of ``k`` atoms, including ones
    the lens would read as the opposite tokens. Restricted to the nonnegative cone,
    ``v_J`` can only be built out of tokens the vector is actually disposed to say.

    The step size is ``1/L`` with ``L`` the Gram matrix's spectral norm -- the
    standard guarantee for projected gradient on a quadratic -- so ``steps`` only has
    to be large enough to converge and is not a tuned parameter. Selection stops
    early if no remaining atom has positive correlation with the residual, because in
    the nonnegative cone such an atom cannot reduce the error.
    """
    residual = np.asarray(vector, dtype=np.float64).copy()
    support: list[int] = []
    coefficients = np.zeros(0, dtype=np.float64)

    for _ in range(min(k, atoms.shape[0])):
        scores = atoms @ residual
        if support:
            scores[support] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]) or scores[best] <= 0:
            break
        support.append(best)

        selected = atoms[support]
        gram = selected @ selected.T
        projections = selected @ vector
        spectral = float(np.linalg.norm(gram, 2))
        step = 1.0 / spectral if spectral > 0 else 0.0
        coefficients = np.concatenate([coefficients, [0.0]])
        for _ in range(steps):
            coefficients = np.maximum(
                coefficients - step * (gram @ coefficients - projections), 0.0
            )
        residual = vector - selected.T @ coefficients

    return coefficients, support


def decompose_vector(
    vector: np.ndarray, dictionary: Dictionary, k: int, steps: int
) -> Decomposition:
    """Split one vector, and report how well the split actually holds."""
    coefficients, support = nonneg_gradient_pursuit(
        vector, dictionary.atoms, k, steps
    )
    reportable = (
        dictionary.atoms[support].T @ coefficients if support
        else np.zeros_like(vector)
    )
    remainder = vector - reportable
    total = float(vector @ vector)
    denominator = float(np.linalg.norm(reportable) * np.linalg.norm(remainder))
    return Decomposition(
        support=[int(i) for i in support],
        token_ids=[int(dictionary.token_ids[i]) for i in support],
        coefficients=[float(c) for c in coefficients],
        reportable=reportable,
        remainder=remainder,
        frac_reportable=float(reportable @ reportable / total) if total > 0 else 0.0,
        frac_remainder=float(remainder @ remainder / total) if total > 0 else 0.0,
        reconstruction_error=(
            float(np.linalg.norm(remainder) / np.linalg.norm(vector))
            if total > 0 else 0.0
        ),
        # A least-squares projection leaves the residual orthogonal to the fitted
        # part. The nonnegativity constraint can break that, and by how much is worth
        # knowing: if these are not near-orthogonal the two "fractions" do not
        # partition the variance and cannot be read as competing shares.
        cos_parts=(
            float(reportable @ remainder / denominator) if denominator > 0 else 0.0
        ),
        n_iterations=len(support),
    )


def match_norm(direction: np.ndarray, target_norm: float) -> np.ndarray:
    """Scale ``direction`` to ``target_norm``; zeros stay zero."""
    norm = float(np.linalg.norm(direction))
    return direction if norm == 0 else direction * (target_norm / norm)


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #

def random_control_fractions(
    readout, config: PCAJLensConfig, block: int, target_norm: float,
    d_model: int, head: np.ndarray, gain: np.ndarray | None, rng,
    transport: Transport | None, atom_counts: Sequence[int],
) -> dict[int, dict]:
    """The identical decomposition on matched-norm random directions, per ``k``.

    **This is the gate**, not a footnote to it. ``k`` atoms drawn from a pool of
    ``dict_pool_size`` are a real number of degrees of freedom, so some share of *any*
    direction is reconstructable; without this null, "``v_J`` is 8% of the variance"
    cannot be told apart from "the pursuit has 16 knobs".

    Each control gets its own pool, from its own lens readout, so it is the identical
    procedure rather than the emotion pool applied to noise. One draw is decomposed at
    every ``k``, which shares the expensive part -- the pool build -- across the ``k``
    values and keeps the nulls paired across them.

    Returns ``{k: {"n", "mean", "p95", "max", "fractions"}}``. The full distribution is
    kept because :func:`p_value_vs_random` needs it; the summary statistics alone cannot
    produce a p-value.
    """
    draws = max(config.n_random_controls, 0)
    fractions: dict[int, list[float]] = {int(k): [] for k in atom_counts}
    for index in range(draws):
        draw = match_norm(rng.normal(size=d_model), target_norm)
        pool = build_dictionary(
            readout, draw, block, config.dict_pool_size, head, gain, transport
        )
        for k in fractions:
            fractions[k].append(
                decompose_vector(draw, pool, k, config.pursuit_steps).frac_reportable
            )
        if draws >= 50 and (index + 1) % 50 == 0:
            print(f"    {index + 1}/{draws} random controls", flush=True)

    out: dict[int, dict] = {}
    for k, values in fractions.items():
        if not values:
            out[k] = {"n": 0}
            continue
        array = np.asarray(values)
        out[k] = {
            "n": len(values),
            "mean": float(array.mean()),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
            "fractions": [float(v) for v in array],
        }
    return out


def p_value_vs_random(observed: float, control: dict) -> float | None:
    """Monte-Carlo p-value for one emotion's fraction against the random null.

    ``(1 + #{random >= observed}) / (1 + n)`` -- the add-one estimator, which is the
    unbiased form for a randomisation test and, unlike the bare proportion, never returns
    exactly 0 for an effect that simply beat every draw taken. The floor is ``1/(1+n)``,
    so ``n_random_controls`` sets the resolution and is reported beside every p-value.

    A randomisation test against matched-norm random directions rather than a label
    permutation: the decomposition has no labels to permute. The null being tested is
    "this vector's reconstructable share is what an unstructured direction of the same
    norm gets", which is the null the design actually supports.
    """
    values = control.get("fractions")
    if not values:
        return None
    array = np.asarray(values)
    return float((1 + int((array >= observed).sum())) / (1 + array.size))


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def significance_threshold(n_emotions: int) -> float:
    """Bonferroni-corrected alpha across the emotions tested."""
    return GATE_ALPHA / max(n_emotions, 1)


def print_decomposition_table(
    rows: list[dict], controls: dict[int, dict], config: PCAJLensConfig,
    atom_counts: Sequence[int],
) -> None:
    """The gate: each emotion's reconstruction fraction against the random null."""
    print()
    print(RULE)
    print("GATE  Reconstruction fraction vs matched-norm random directions")
    print(RULE)
    print("The only claim this method supports. A sparse nonnegative code with k atoms")
    print("from a pool of " + f"{config.dict_pool_size}" + " reconstructs some share of")
    print("ANY direction, so the fraction is read against the share it gets from an")
    print("unstructured direction of the same norm -- as a ratio and a p-value, per")
    print("emotion, at each k.")
    print()
    if config.write_space:
        print("write_space ABLATION: atoms are J^+ (g*w_t), the directions that most")
        print("efficiently WRITE each token, not the directions the lens reads them with.")
        print("Nothing below is a statement about reportability.")
        print()

    alpha = significance_threshold(len(rows))
    print(f"significance: p < {alpha:.4f}  "
          f"(alpha {GATE_ALPHA} Bonferroni-corrected over {len(rows)} emotions)")
    floors = {
        int(k): 1.0 / (1 + controls.get(int(k), {}).get("n", 0))
        for k in atom_counts
    }
    print(f"p-value floor: {max(floors.values()):.4f} at "
          f"n_random_controls={config.n_random_controls}"
          + ("   TOO COARSE for the corrected alpha -- raise n_random_controls"
             if max(floors.values()) > alpha else ""))
    for k in atom_counts:
        control = controls.get(int(k), {})
        if not control.get("n"):
            print(f"k={k:<3}   chance baseline NOT MEASURED (n_random_controls=0); the "
                  "fractions below have no referent")
            continue
        print(f"k={k:<3}   random null: mean {control['mean']:.1%}, "
              f"p95 {control['p95']:.1%}, max {control['max']:.1%} "
              f"over {control['n']} draws")
        if control["mean"] > DEGENERATE_CONTROL_FRACTION:
            print(f"        DEGENERATE at this k: a random direction already reaches "
                  f"{control['mean']:.0%}, so")
            print(f"        {k} atoms from {config.dict_pool_size} candidates can "
                  "reconstruct almost anything and NO")
            print("        fraction at this k is interpretable -- not a small one and not a")
            print("        large one. Lower k or dict_pool_size.")
    print()

    width = max([16, *(len(r["emotion"]) + 2 for r in rows)])
    header = f"{'emotion':<{width}}{'|v|':>9}"
    for k in atom_counts:
        header += f"{f'frac k={k}':>11}{'x null':>8}{'p':>8}"
    header += f"{'own word':>10}"
    print(header)
    print(THIN)
    for row in sorted(rows, key=lambda r: r["emotion"]):
        line = f"{row['emotion']:<{width}}{row['norm']:>9.2f}"
        for k in atom_counts:
            at_k = row["per_k"][str(k)]
            control = controls.get(int(k), {})
            ratio = (f"{at_k['frac_reportable'] / control['mean']:>7.1f}x"
                     if control.get("mean") else "    n/a")
            p_value = at_k.get("p_value")
            mark = "" if p_value is None or p_value >= alpha else "*"
            shown = "     n/a" if p_value is None else f"{p_value:>7.4f}"
            line += f"{at_k['frac_reportable']:>11.1%}{ratio:>8}{shown}{mark:<1}"
        own = ("n/a" if row["own_word_atom_rank"] is None
               else f"#{row['own_word_atom_rank']}")
        line += f"{own:>9}"
        print(line)
    print(THIN)
    print(f"  * = p < {alpha:.4f}, the Bonferroni-corrected threshold")
    for k in atom_counts:
        values = np.asarray([r["per_k"][str(k)]["frac_reportable"] for r in rows])
        significant = sum(
            1 for r in rows
            if (r["per_k"][str(k)].get("p_value") or 1.0) < alpha
        )
        print(f"  k={k:<3} fraction: min {values.min():.1%}, "
              f"median {np.median(values):.1%}, max {values.max():.1%};  "
              f"{significant}/{len(rows)} beat the null")
    worst_cos = max(
        abs(r["per_k"][str(k)]["cos_parts"]) for r in rows for k in atom_counts
    )
    print(f"  |cos(v_J, v_perp)|: max {worst_cos:.3f}")
    if worst_cos > 0.2:
        print("    The two parts are not close to orthogonal, so their fractions do not")
        print("    partition the variance and must not be read as competing shares. The")
        print("    nonnegativity constraint is binding hard here.")
    print("  'own word' is where the emotion's own token sits among the atoms selected at "
          f"k={config.n_dict_atoms},")
    print("  by coefficient. Reported, not gated: a vector can be well reconstructed by")
    print("  related words without its own name being among them.")
    print()
    print(f"  How the fraction moves with k is the point of printing both. It rises with k")
    print("  by construction, because the reconstructable set GROWS with k -- see the")
    print("  cones note in the verdict.")


def print_verdict(
    rows: list[dict], controls: dict[int, dict], identity: dict, coherence: dict,
    transport: Transport | None, lens_warnings: list[str], config: PCAJLensConfig,
    atom_counts: Sequence[int], artifacts: dict[str, object],
) -> None:
    alpha = significance_threshold(len(rows))
    primary = int(config.n_dict_atoms)
    beat_null = {
        int(k): [r for r in rows if (r["per_k"][str(k)].get("p_value") or 1.0) < alpha]
        for k in atom_counts
    }
    degenerate = [
        int(k) for k in atom_counts
        if controls.get(int(k), {}).get("mean", 0.0) > DEGENERATE_CONTROL_FRACTION
    ]
    in_range = [
        r for r in rows
        if 0.01 <= r["per_k"][str(primary)]["frac_reportable"] <= config.frac_j_expected_max
    ]
    with_word = [r for r in rows if r["own_word_atom_rank"] is not None]

    print()
    print(RULE)
    print("PHASE 6 VERDICT")
    print(RULE)
    if config.write_space:
        print("  MODE: write_space ABLATION (atoms = J^+ (g*w_t)). These are the directions")
        print("  that most efficiently WRITE each token, not the ones the lens reads them")
        print("  with, so nothing here is a statement about reportability. Run without")
        print("  --set write_space=true for that.")
        print()
    print(f"  read identity holds        : {identity['holds']}  "
          f"(min corr {identity['min_correlation']:.6f} over "
          f"{identity['n_probes']} random h,")
    print(f"                               threshold {identity['threshold']}; confirms "
          "the atoms ARE the lens's")
    print("                               own measurement weights, so J's stored "
          "orientation is not assumed)")
    print(f"  gain absorbed              : {identity['gain_absorbed']}")
    if transport is not None:
        print(f"  J^+ truncation             : rank {transport.rank}/"
              f"{transport.pullback.shape[0]} at rcond={transport.rcond:g}, "
              f"cond(J) {transport.condition_number:.3g}")
    for k in atom_counts:
        control = controls.get(int(k), {})
        values = np.asarray([r["per_k"][str(k)]["frac_reportable"] for r in rows])
        marker = "  <- saved" if int(k) == primary else ""
        print(f"  k={k:<3} beats the null       : {len(beat_null[int(k)])}/{len(rows)} "
              f"emotions at p < {alpha:.4f}   "
              f"(median {np.median(values):.1%} vs null "
              f"{control.get('mean', float('nan')):.1%}){marker}")
    print(f"  fraction in the brief's band: {len(in_range)}/{len(rows)} emotions inside "
          f"1-{config.frac_j_expected_max:.0%} at k={primary}")
    if degenerate:
        print(f"  degrees of freedom         : REVIEW -- the random null exceeds "
              f"{DEGENERATE_CONTROL_FRACTION:.0%} at k={degenerate},")
        print("                               so no fraction at those k is interpretable "
              "either way")
    print(f"  atom coherence (reported)  : mean "
          f"{coherence.get('mean', float('nan')):.3f}, max "
          f"{coherence.get('max', float('nan')):.3f}"
          + (f"   above the {DICTIONARY_MAX_COHERENCE:g} ceiling"
             if coherence.get("max", 0.0) > DICTIONARY_MAX_COHERENCE else ""))
    if coherence.get("max", 0.0) > DICTIONARY_MAX_COHERENCE:
        print("                               atoms that aligned are interchangeable, so")
        print("                               the TOKENS v_J names are not attributable")
        print("                               even where the fraction beats the null.")
        print("                               Not gated: most of it comes from the pool")
        print("                               being one direction's own top tokens.")
    print(f"  v_J selects the own token  : {len(with_word)}/{len(rows)} at k={primary} "
          "(reported, not gated)")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<10}: {path}" if label else f"  {'':<10}  {path}")
    print()

    print(RULE)
    print("  WHAT v_perp IS NOT -- carry this with every number above")
    print(RULE)
    print("  Under sparse nonnegative coding the reconstructable set is")
    print("    { sum c_i d_i : c_i >= 0, |support| <= k }")
    print("  which is a UNION OF CONES -- one per choice of k atoms from the pool -- and")
    print("  NOT a linear subspace. The verbalizable subspace itself is linear: it is")
    print("  span{J^T u_t}, and any direction orthogonal to all of it moves no logit. But")
    print("  what gets measured here is a k-sparse nonnegative approximation inside that")
    print("  span, and the approximation is what the fraction is about.")
    print()
    print("  So v_perp means: NOT CAPTURED BY THIS SPARSE APPROXIMATION, AT THIS k, FROM")
    print("  THIS POOL. It never means 'intrinsically unverbalizable'. Raising k, widening")
    print("  dict_pool_size, or dropping the nonnegativity constraint each move the")
    print(f"  boundary -- which is why both k={atom_counts[0]} and k={atom_counts[-1]} are "
          "printed side by side")
    print("  rather than one being chosen for you.")
    print()
    print("  A sentence of the form 'the model cannot verbalise this component' is not")
    print("  supported by this stage. The supportable sentence is 'this component lies")
    print("  outside the k-sparse nonnegative span of the tokens the vector is most")
    print("  disposed to say'.")
    print()
    if not any(beat_null[int(k)] for k in atom_counts):
        print("  NO emotion beat the random null at any k. That is the gate failing: the")
        print("  reconstruction has measured the pursuit's degrees of freedom and nothing")
        print("  about emotion. Do not read the fractions as small-and-therefore-")
        print("  interesting; there is no effect here to be small.")
        print()
    if max(
        np.asarray([r["per_k"][str(primary)]["frac_reportable"] for r in rows]).max(), 0.0
    ) > config.frac_j_expected_max:
        print(f"  Some fractions exceed {config.frac_j_expected_max:.0%} at k={primary}. "
              "Per the brief that points at")
        print("  the decomposition or the lens rather than at the theory: check the null")
        print("  above, then whether the lens is the converged one (`run.py refit_lens`),")
        print("  before revising any claim about how much of emotion is verbalisable.")
        print()
    if lens_warnings:
        print("  The lens warnings in STEP 2 bound all of this. Read them again.")
        print()
    print("  What Phase 8 does with the output. The four norm-matched directions per")
    print("  emotion are steering conditions, and the point of the split is that if")
    print("  behaviour moves under v_perp while the report channel stays quiet, an")
    print("  emotional state is acting without being reportable. Phase 6 only makes the")
    print("  split; it is not evidence for that on its own -- and note a v_perp effect")
    print("  would still not rule out the concept being re-derived downstream, which is")
    print("  what Phase 9's clamp exists for.")
    print()
    print("STOPPING at the Phase 6 gate, as agreed. Phase 7 has not run.")
    print(RULE)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    cache_dir = paths.hf_cache_dir()
    vectors_path = args.vectors or config.emotion_vectors_path
    meta_path = (
        vectors_path.with_name(config.emotion_vectors_meta_path.name)
        if args.vectors else config.emotion_vectors_meta_path
    )
    out_dir = vectors_path.parent

    space = read_emotion_vectors(vectors_path, meta_path)
    block = space.target_block
    print(RULE)
    print(f"PHASE 6 GATE -- reportable / remainder split   run '{config.run_name}'")
    print(RULE)
    print(f"model   : {config.model_name} ({config.dtype})")
    print(f"vectors : {vectors_path}")
    print(f"outputs : {out_dir}")
    print()
    print("Reads Phase 2's vectors and the Phase 0 lens. Independent of Phases 3-5, so")
    print("it runs whether or not the PCA and its extensions have.")
    print()
    print(RULE)
    print("STEP 1  The vectors to split")
    print(RULE)
    fitted = [e for e in space.emotions if e != NEUTRAL_QUADRANT]
    described = (
        jlens_lens.describe_block(block, space.n_layers) if space.n_layers
        else f"block {block}"
    )
    norms = np.linalg.norm(space.matrix, axis=1)
    print(f"emotions       : {len(space.emotions)} ({len(fitted)} non-neutral)")
    print(f"target block   : {described}")
    print(f"vector norms   : min {norms.min():.2f}, median {np.median(norms):.2f}, "
          f"max {norms.max():.2f}")
    phase2_min = space.metadata.get("split_half", {}).get("summary", {}).get(
        "min_cosine_centered"
    )
    if phase2_min is not None:
        print(f"Phase 2 min split-half cosine: {phase2_min:.3f}  "
              "(a noisy vector splits into a noisy v_J and a noisy remainder)")
    print(f"plan           : {config.n_dict_atoms} atoms from a pool of "
          f"{config.dict_pool_size}, {config.pursuit_steps} pursuit steps,")
    print(f"                 {config.n_random_controls} random controls")

    lens_path, lens_report = resolve_lens(config, cache_dir)
    print()
    print(RULE)
    print("STEP 2  The lens")
    print(RULE)
    print("reading the lens checkpoint (loads it into host RAM) ...")
    description = jlens_lens.describe_lens_checkpoint(lens_path)
    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )

    class _Axes:
        """Minimal shim so Phase 4's lens cross-check can be reused verbatim."""

        d_model = space.d_model
        target_block = block
        n_layers = space.n_layers

    problems, lens_warnings = crosscheck_lens(description, _Axes(), arch)
    print(f"source         : {lens_report['source']}  ({lens_report['path']})")
    print(f"d_model        : {description.d_model}   (vectors: {space.d_model})")
    print(f"fitted blocks  : {description.source_layers[0]}.."
          f"{description.source_layers[-1]}   (vectors at {block})")
    print(f"prompts fitted : {description.n_prompts}")
    print()
    if problems:
        print("  MISMATCH:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  OK  the lens covers this block at this dimension.")
    for warning in lens_warnings:
        import textwrap

        print("\n  WARNING:")
        for line in textwrap.wrap(warning, 72):
            print(f"    {line}")

    sections: dict = {
        "run": {"stage": "phase6_decompose", "run_name": config.run_name,
                "dry_run": args.dry_run, "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "vectors": {
            "path": str(vectors_path), "target_block": block,
            "n_emotions": len(space.emotions), "d_model": space.d_model,
            "content_sha256": space.content_sha256,
            "phase2_fingerprint": space.metadata.get("fingerprint", {}),
        },
        "lens": {**lens_report, "n_prompts": description.n_prompts,
                 "problems": problems, "warnings": lens_warnings},
    }

    if args.dry_run:
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run", title=f"PHASE 6 DRY RUN -- {config.run_name}",
            sections=sections, txt_name="phase6_dry_run.txt",
            json_name="phase6_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: lens and vectors cross-checked; no weights loaded.")
        print(f"  records : {txt_path}")
        print(f"            {json_path}")
        print()
        print("  atoms will be J^T (g * w_t) -- the directions the lens READS each token")
        print(f"  with -- at k={list(config.dict_atom_counts)}, saving k="
              f"{config.n_dict_atoms}. The gate is each emotion's")
        print(f"  reconstruction fraction against {config.n_random_controls} matched-norm")
        print("  random directions, with a ratio and a Monte-Carlo p-value; both need the")
        print("  lens loaded, so nothing about it can be previewed here.")
        if config.write_space:
            print()
            print("  write_space is SET: the ablation will use J^+ (g * w_t) instead, the")
            print("  directions that most efficiently WRITE each token. Not a statement")
            print(f"  about reportability. dict_pinv_rcond={config.dict_pinv_rcond:g}.")
        if problems:
            print("\n  Fix the MISMATCH first; the decomposition would be meaningless.")
        print(RULE)
        return 0 if not problems else 3

    if problems:
        print("\nABORTED: the lens does not fit these vectors (see MISMATCH above).",
              file=sys.stderr)
        return 3

    print()
    print(RULE)
    print(f"Loading {config.model_name} ({config.dtype}) ...")
    print(RULE)
    t0 = time.time()
    tokenizer = model_utils.load_tokenizer(
        config.model_name, config.model_revision, cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    hf_model = model_utils.load_model(
        config.model_name, revision=config.model_revision, cache_dir=cache_dir,
        dtype=config.dtype, device_map=config.device_map,
        quantization=config.quantization,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    print(f"  weights loaded in {time.time() - t0:.0f}s")
    readout = jlens_lens.LensReadout.build(hf_model, tokenizer, lens_path)
    head, gain = unembed_parts(readout)
    print(f"  unembedding    : {head.shape}")
    print(f"  final-norm gain: "
          + ("absorbed into the atoms" if gain is not None
             else "ABSENT -- atoms will disagree with the lens by whatever it does"))

    # --- the dictionary --------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 3  The dictionary: read directions")
    print(RULE)
    print("logit_t = <w_t, g * (Jh/rms(Jh))> = <J^T (g*w_t), h> / rms(Jh), so J^T (g*w_t)")
    print("IS the weight vector the lens reads token t with, and anything orthogonal to")
    print("every one of them moves no logit at all. That makes span{J^T u_t} the")
    print("verbalizable subspace by construction. Atoms are those weights, unit-normalised.")
    print()
    print("There is NO 'does lensing an atom return its own token' check here. That is a")
    print("property of WRITE directions (J^+ u_t), and a read direction has no reason to")
    print("satisfy it. What is checked instead is the identity above, exactly.")
    atom_counts = tuple(int(k) for k in config.dict_atom_counts)
    transport: Transport | None = None
    if config.write_space:
        print()
        print("  write_space ABLATION: atoms will be J^+ (g*w_t) instead -- the directions")
        print("  that most efficiently WRITE each token. A different question, kept")
        print("  runnable and labelled so the two cannot be conflated again.")
        t0 = time.time()
        transport = factor_transport(readout, block, config.dict_pinv_rcond)
        print(f"  SVD of J            : {transport.pullback.shape} on "
              f"{transport.device} in {time.time() - t0:.0f}s")
        print(f"  cond(J)             : {transport.condition_number:.4g}"
              f"   -> {transport.effective_condition:.4g} after truncation")
        print(f"  rank kept           : {transport.rank}/{transport.pullback.shape[0]} "
              f"at dict_pinv_rcond={transport.rcond:g}")
        print(f"  round-trip cos      : {transport.roundtrip_cosine:.4f} against jlens's "
              "own transport")

    probe_direction = space.matrix[0] - space.matrix.mean(axis=0)
    probe_dictionary = build_dictionary(
        readout, probe_direction, block, config.dict_pool_size, head, gain, transport,
    )
    identity_rng = rng_for(config.seed, "phase6_read_identity")
    identity = verify_read_directions(readout, probe_dictionary, block, identity_rng)
    coherence = dictionary_coherence(probe_dictionary, DICTIONARY_MAX_COHERENCE)
    print()
    print(f"  mode                : {probe_dictionary.mode}  "
          f"({'J^+ (g*w_t)' if transport else 'J^T (g*w_t)'})")
    print(f"  pool                : {len(probe_dictionary.token_ids)} atoms")
    print(f"  gain absorbed       : {identity['gain_absorbed']}")
    print(f"  read identity       : min corr {identity['min_correlation']:.6f} over "
          f"{identity['n_probes']} random h")
    print(f"                        (lens logits vs <d_t, h>; the ratio 1/rms(Jh) is a")
    print("                        positive scalar, hence a correlation. Confirms J's")
    print("                        stored orientation rather than assuming it.)")
    print(f"  atom coherence      : mean {coherence.get('mean', float('nan')):.3f}, "
          f"max {coherence.get('max', float('nan')):.3f}")
    print(f"                        (|cos| between atoms. Above "
          f"{DICTIONARY_MAX_COHERENCE:g} they are near-duplicates and")
    print("                        which the pursuit picks is arbitrary, so the tokens v_J")
    print("                        names stop being attributable. Reported, not gated:")
    print("                        the pool is one direction's own top tokens, whose rows")
    print("                        are aligned with it and so with each other regardless.)")
    if not identity["holds"] and not config.write_space:
        print()
        print("ABORTED: the read identity does not hold, so these atoms are not the lens's",
              file=sys.stderr)
        print("measurement weights. Either jacobians[block] is stored transposed relative to",
              file=sys.stderr)
        print("what this code reads, or the lens's unembed differs from lm_head(final_norm).",
              file=sys.stderr)
        print("Every atom would still be a unit vector of the right length, pointing",
              file=sys.stderr)
        print("somewhere unrelated to its token -- which no shape check would catch. This is",
              file=sys.stderr)
        print("a code fault, not a weak finding.", file=sys.stderr)
        return 3

    # --- decompose --------------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 4  Split each vector")
    print(RULE)
    mean = space.matrix[
        np.asarray([e != NEUTRAL_QUADRANT for e in space.emotions])
    ].mean(axis=0)
    rng = rng_for(config.seed, "phase6_decompose")
    rows: list[dict] = []
    saved: dict[str, list[np.ndarray]] = {
        "v": [], "v_reportable": [], "v_remainder": [], "v_random": []
    }
    order: list[str] = []

    for i, emotion in enumerate(space.emotions):
        # Mean-centred, matching what Phase 3 runs PCA on and what Phase 4 lensed:
        # a raw centroid is dominated by the component every story shares at this
        # layer, and its "reportable part" would be that component's tokens.
        vector = space.matrix[i] - mean
        norm = float(np.linalg.norm(vector))
        # One pool per emotion, shared across every k: the pool is a property of the
        # vector's own readout, not of the atom budget, so rebuilding it per k would
        # cost a vocabulary top-k each time and change nothing.
        dictionary = build_dictionary(
            readout, vector, block, config.dict_pool_size, head, gain, transport
        )
        variants = readout.single_token_variants(emotion)
        own_ids = set(variants.values())

        # String keys: json turns int keys into strings, and a reader round-tripping the
        # sidecar must see the same shape the run produced.
        per_k: dict[str, dict] = {}
        primary_result = None
        for k in atom_counts:
            result = decompose_vector(vector, dictionary, k, config.pursuit_steps)
            per_k[str(k)] = {
                "k": k,
                "frac_reportable": result.frac_reportable,
                "frac_remainder": result.frac_remainder,
                "reconstruction_error": result.reconstruction_error,
                "cos_parts": result.cos_parts,
                "n_atoms": result.n_iterations,
                # The ids, not only their decoded forms: Phase 9 clamps the read
                # directions of these exact tokens, and re-encoding a decoded string is
                # not guaranteed to return the id it came from.
                "token_ids": [int(t) for t in result.token_ids],
                "top_atom_tokens": [tokenizer.decode([t]) for t in result.token_ids[:8]],
                "coefficients": result.coefficients[:8],
            }
            if k == int(config.n_dict_atoms):
                primary_result = result
        assert primary_result is not None   # validate() guarantees the primary k is present

        own_rank = next(
            (rank for rank, tid in enumerate(primary_result.token_ids)
             if tid in own_ids), None
        )
        rows.append({
            "emotion": emotion, "norm": norm,
            "own_word_atom_rank": own_rank,
            "own_word_single_token": bool(variants),
            "saved_k": int(config.n_dict_atoms),
            "per_k": per_k,
        })
        order.append(emotion)
        saved["v"].append(match_norm(vector, norm))
        saved["v_reportable"].append(match_norm(primary_result.reportable, norm))
        saved["v_remainder"].append(match_norm(primary_result.remainder, norm))
        saved["v_random"].append(match_norm(rng.normal(size=space.d_model), norm))
        print(f"  {emotion:<16} "
              + "  ".join(f"k={k} {per_k[str(k)]['frac_reportable']:>6.1%}" for k in atom_counts)
              + "   " + " ".join(repr(t) for t in per_k[str(atom_counts[0])]["top_atom_tokens"][:5]),
              flush=True)

    print()
    print(f"  {config.n_random_controls} matched-norm random controls, each decomposed at "
          f"every k ...", flush=True)
    controls = random_control_fractions(
        readout, config, block, float(np.median([r["norm"] for r in rows])),
        space.d_model, head, gain, rng, transport, atom_counts,
    )
    for row in rows:
        for k in atom_counts:
            row["per_k"][str(k)]["p_value"] = p_value_vs_random(
                row["per_k"][str(k)]["frac_reportable"], controls.get(k, {})
            )
    print_decomposition_table(rows, controls, config, atom_counts)

    # --- save -------------------------------------------------------------- #
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            key: np.ascontiguousarray(np.vstack(value), dtype=VECTOR_DTYPE)
            for key, value in saved.items()
        },
        str(out_dir / config.decomposition_path.name),
        metadata={
            "emotions": json.dumps(order),
            "target_block": str(block),
            "target_hidden_state": str(jlens_lens.hidden_state_index(block)),
            "row_order": "row i of every tensor belongs to emotions[i]",
            "norm_matched": "all four are scaled to ||v|| for that emotion",
            "centred": "vectors are mean-centred across the non-neutral emotions",
            "dtype": np.dtype(VECTOR_DTYPE).name,
            # Travel with the tensors, not only in the sidecar: these four arrays are
            # what Phase 8 steers with, and k and the mode are what make v_reportable
            # mean anything. v_perp is "outside this k-sparse code", never
            # "unverbalizable" -- a reader holding only the safetensors must be told.
            "saved_k": str(config.n_dict_atoms),
            "reported_k": json.dumps([int(k) for k in atom_counts]),
            "atom_mode": probe_dictionary.mode,
            "v_remainder_means": "outside the k-sparse nonnegative span of this pool at "
                                 "the saved k; NOT intrinsically unverbalizable",
        },
    )
    table_path = out_dir / "phase6_decomposition.csv"
    pd.DataFrame([
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
        for row in rows
    ]).to_csv(table_path, index=False)

    gate = {
        "alpha": GATE_ALPHA,
        "alpha_bonferroni": significance_threshold(len(rows)),
        "n_emotions": len(rows),
        "p_value_floor": 1.0 / (1 + config.n_random_controls),
        "beat_null": {
            str(k): [
                r["emotion"] for r in rows
                if (r["per_k"][str(k)].get("p_value") or 1.0)
                < significance_threshold(len(rows))
            ]
            for k in atom_counts
        },
    }
    # Random-control distributions are dropped from the record: 200 floats per k is
    # noise in a file meant to be read, and the p-values computed from them are kept.
    control_summary = {
        str(k): {key: value for key, value in controls.get(k, {}).items()
                 if key != "fractions"}
        for k in atom_counts
    }
    metadata = {
        **sections,
        "read_identity": identity,
        "coherence": coherence,
        "atom_mode": probe_dictionary.mode,
        "write_space": bool(config.write_space),
        "reported_k": [int(k) for k in atom_counts],
        "saved_k": int(config.n_dict_atoms),
        "gate": gate,
        "v_remainder_means": "outside the k-sparse nonnegative span of this pool at the "
                             "saved k; NOT intrinsically unverbalizable -- the "
                             "reconstructable set is a union of cones, not a subspace",
        "transport": transport.summary() if transport is not None else None,
        "random_control": control_summary,
        "per_emotion": rows,
        "emotions": order,
        "written": provenance.utc_timestamp(),
    }
    (out_dir / config.decomposition_meta_path.name).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    sections["read_identity"] = identity
    sections["coherence"] = coherence
    sections["atom_mode"] = probe_dictionary.mode
    sections["write_space"] = bool(config.write_space)
    sections["reported_k"] = [int(k) for k in atom_counts]
    sections["saved_k"] = int(config.n_dict_atoms)
    sections["v_remainder_means"] = metadata["v_remainder_means"]
    sections["gate"] = gate
    if transport is not None:
        sections["transport"] = transport.summary()
    sections["random_control"] = control_summary
    sections["per_emotion"] = rows
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 6 GATE -- {config.run_name}",
        sections=sections, txt_name="phase6_gate.txt", json_name="phase6_gate.json",
    )

    print_verdict(
        rows, controls, identity, coherence, transport, lens_warnings, config,
        atom_counts, artifacts={
            "directions": out_dir / config.decomposition_path.name,
            "metadata": out_dir / config.decomposition_meta_path.name,
            "table": table_path,
            "records": txt_path,
            "": json_path,
        },
    )
    # Exit 3 when nothing beat the random null at any k, matching the other gates'
    # failure code: the reconstruction then measured the pursuit's degrees of freedom and
    # nothing about emotion, and Phase 7 must not chain onto it. Note it is NOT keyed on
    # the fraction being small or large -- only on whether it is distinguishable from
    # what an unstructured direction of the same norm gets.
    beat_any = any(gate["beat_null"][str(k)] for k in atom_counts)
    return 0 if beat_any else 3


if __name__ == "__main__":
    raise SystemExit(main())
