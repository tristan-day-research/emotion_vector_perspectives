"""Phase 6 (GATE): split each emotion vector into a reportable part and a remainder.

What this stage does
--------------------
For each Phase 2 emotion vector ``v`` at the target block, it finds ``v_J`` -- the
part the lens can express as tokens -- as a sparse nonnegative combination of
``n_dict_atoms`` lens dictionary directions, fitted by gradient pursuit. The
remainder is ``v_perp = v - v_J``. It writes norm-matched ``v``, ``v_J``,
``v_perp`` and a matched-norm random control, which is what Phase 8 steers with.

Depends on Phase 2's vectors and the Phase 0 lens. **Not on Phase 3, 4 or 5** --
Phase 3's PCA and Phase 5's extensions are a different question about the same
vectors, so this can run whether or not they have. Phase 3's artefact is read when
present, only to quote the circumplex context in the gate.

The gate is a variance fraction, and it needs a baseline
--------------------------------------------------------
``v_J`` should be **small** -- the workspace paper reports roughly 6-10% of a
concept vector's variance inside J-space, and the brief expects 5-15%. A large
value means the decomposition or the lens is wrong, not that the theory is.

But "8% of the variance" has no referent on its own, so the identical
decomposition is run on matched-norm **random** directions. ``k`` atoms chosen from
a pool of ``dict_pool_size`` have real degrees of freedom: if a structureless
direction also reaches 8%, the number measures the pursuit's flexibility rather
than anything about emotion. The control is the headline's denominator, and it is
the one thing the brief's 5-15% expectation cannot supply.

Answering the open design question rather than assuming it
----------------------------------------------------------
``jlens`` has no dictionary: a fitted lens is a per-layer matrix ``J``, and the
dictionary has to be constructed. It is the pullback of unembedding rows through the
transport -- and getting the pullback right means inverting ``J``, not transposing it.

The derivation, since the README's sketch (``d_t = J^T w_t``) is wrong:

* **The final norm's learned gain is absorbable exactly.** ``unembed`` is
  ``lm_head(final_norm(h))``, and for an RMSNorm the lens logit for token ``t`` is
  ``<w_t, g * (Jh / rms(Jh))>`` = ``<g * w_t, Jh> / rms(Jh)``. Writing
  ``u_t = g * w_t``, that is ``<u_t, Jh> / rms(Jh)``. Dropping ``g`` would make the
  dictionary disagree with the lens it is supposed to represent, by exactly the amount
  the model's learned per-channel scale varies.
* **The normalisation is by ``rms(Jh)``, not by ``||h||``, and that is what decides
  the answer.** Maximising ``<u_t, Jh> / rms(Jh)`` over ``h`` needs ``Jh`` parallel to
  ``u_t``, so the atom is ``d_t = J^-1 u_t``. ``J^T u_t`` is the maximiser of the
  *unnormalised* ``<u_t, Jh>`` per unit ``||h||``, which is a different problem;
  ``J^T = J^-1`` only for an orthogonal ``J``, and an averaged Jacobian is not
  orthogonal. The transpose was the original construction here and it failed the
  atom-validity check below -- lensing an atom did not return its own token, which is
  what a mislabelled dictionary looks like from the outside.
* **``J`` is inverted by truncated SVD**, not solved exactly: see
  :func:`factor_transport`. An exact inverse would amplify the directions ``J`` nearly
  annihilates by ``1/s`` and let numerical noise dominate every atom.
* **The input-dependent part of the norm cannot be absorbed and does not need to be.**
  ``1 / rms(Jh)`` is a positive scalar. It scales every logit for a given ``h``
  equally, so it cannot change which tokens rank highest, and Phase 0 already verified
  numerically that a bare direction's readout is magnitude-free. It affects comparisons
  of magnitude *between* different ``h``, which the pursuit never makes.

Whether the pullback is a dictionary is measured, not argued.
:func:`verify_dictionary` lenses a sample of atoms and reports, per atom, whether its own
token comes back top-1 and where it ranks. That is the check the transpose failed, and it
is what gates: if the median rank is not 0, **no reportable fraction is printed** and the
stage exits 3, because a variance share out of mislabelled directions is not a
measurement of reportability and it reads exactly like one. Phases 7 and 8 read
``dictionary_valid`` from the sidecar and refuse in turn -- Phase 7 ranks emotions by that
fraction and Phase 8 steers with the split. Reconstruction error is reported beside the
split for the same reason.

:func:`dictionary_coherence` reports |cos| *between* atoms alongside it, and does not
gate. Two atoms that aligned are interchangeable, so the tokens ``v_J`` names stop being
attributable even where the fraction is sound -- but part of the coherence comes from the
pool being one direction's own top tokens, whose unembedding rows are aligned with that
direction and therefore with each other, whatever the pullback does. So it is a number to
read against the ``dict_pinv_rcond`` sweep, not a second gate.

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
    python run.py phase6              # the gate
    python run.py phase6 --set n_dict_atoms=25 --set dict_pool_size=1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

#: Atoms sampled for the self-readout check that answers the open design question.
DICTIONARY_PROBE_ATOMS = 24

#: The atom-validity check passes when the median probed atom lenses back to its own
#: token at rank 0, and nearly all of them land in the top 10. Rank 0 for the median is
#: the demanding half: with a correct pullback ``J d_t = u_t`` exactly, so top-1 is what
#: the algebra predicts and anything less says the inversion is losing the token. The
#: top-10 floor catches the tail, where near-duplicate tokens (" sad" against "sad")
#: can legitimately outrank an atom's own id without the atom being wrong.
DICTIONARY_VALID_MEDIAN_RANK = 0
DICTIONARY_VALID_MIN_TOP10 = 0.9

#: |cosine| between atoms above which they are near-duplicates. **Reported, not gated.**
#: Two atoms this aligned make the pursuit's choice between them arbitrary, so the tokens
#: ``v_J`` names are not attributable -- but part of the coherence is intrinsic to the
#: pool rather than to the pullback: the pool is one direction's top tokens, and their
#: unembedding rows are aligned with that direction and so with each other. On a synthetic
#: lens the coherence stayed near 0.7 at every ``dict_pinv_rcond``, which is why this
#: ceiling flags a number for the operator to read against the sweep instead of blocking
#: the fraction. See :func:`dictionary_coherence`.
DICTIONARY_MAX_COHERENCE = 0.5

#: ``rcond`` values swept and reported when the dictionary fails, so one run shows the
#: whole rank / coherence / self-rank trade-off instead of five.
RCOND_SWEEP = (1e-3, 1e-2, 3e-2, 1e-1, 2e-1)

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
    raw_norms: np.ndarray       # (n_pool,) ||J^+ (g * w_t)|| before normalisation
    gain_absorbed: bool


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
    """Mutual |cosine| between atoms -- the failure the self-readout check cannot see.

    A dictionary of near-duplicate atoms can still pass :func:`verify_dictionary`: each
    atom lenses back to its own token, because the token's own unembedding row is what
    built it. But if two atoms are 90% the same direction, the pursuit's choice between
    them is arbitrary, and the reconstruction's *tokens* -- the part that makes ``v_J``
    mean "the part the lens can verbalise" rather than "20 numbers" -- are then not
    attributable at all.

    Under-truncation is what causes it: ``1/s`` amplification pulls every atom towards
    the same smallest-``s`` directions. So this is the diagnostic that says
    ``dict_pinv_rcond`` is too low, in the regime where the self-ranks still look fine.
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
    transport: Transport,
) -> Dictionary:
    """Atoms for the ``pool_size`` tokens this direction is most disposed to say.

    The pool comes from the direction's own lens readout, so the candidate set *is*
    "the tokens ``v`` wants to say" -- which is what makes the reconstruction the
    reportable part rather than merely a sparse approximation of ``v``.

    Every atom in the pool is built in one matmul against the pre-factored pullback,
    which is why :func:`factor_transport` is called by the caller and not here: the
    factorisation is the expensive part and it does not depend on the direction.

    Atoms are unit-normalised. Selection in the pursuit is then a correlation with
    the residual rather than a contest between long and short atoms, and the
    coefficients carry the magnitudes.
    """
    import torch

    if transport.block != block:
        raise ValueError(
            f"transport was factored at block {transport.block}, asked for {block}"
        )
    logits = readout.direction_logits(direction, block, use_jacobian=True)
    pool = min(pool_size, logits.numel())
    token_ids = torch.topk(logits, pool).indices.cpu().numpy()

    rows = head[token_ids]
    if gain is not None:
        rows = rows * gain[None, :]
    # d_t = J^+ (g * w_t), batched over the pool: (n, d) @ (d, d).
    atoms = rows @ transport.pullback.T
    norms = np.linalg.norm(atoms, axis=1)
    keep = norms > 0
    return Dictionary(
        token_ids=token_ids[keep],
        atoms=atoms[keep] / norms[keep][:, None],
        raw_norms=norms[keep],
        gain_absorbed=gain is not None,
    )


def verify_dictionary(readout, dictionary: Dictionary, block: int, n_probe: int) -> dict:
    """Does lensing atom ``d_t`` actually put token ``t`` on top?

    The test for whether the pullback is a dictionary, and the test for whether the
    inversion in :func:`factor_transport` was the right one -- the transpose
    construction this replaced failed exactly here. If these ranks are large, the
    decomposition below is reconstructing ``v`` out of directions that do not mean what
    their labels say, and no variance split can be read off it.

    Unchanged from the transpose era on purpose: it is the before-and-after measurement,
    so changing what it computes would destroy the comparison. ``per_atom`` is added
    beside the summary, not in place of it.
    """
    probe = min(n_probe, len(dictionary.token_ids))
    if probe == 0:  # pragma: no cover - an empty pool
        return {"checked": 0}
    step = max(1, len(dictionary.token_ids) // probe)
    picks = list(range(0, len(dictionary.token_ids), step))[:probe]

    ranks: list[int] = []
    for i in picks:
        logits = readout.direction_logits(dictionary.atoms[i], block, use_jacobian=True)
        order = np.argsort(-logits.numpy())
        ranks.append(int(np.where(order == dictionary.token_ids[i])[0][0]))
    ranks_array = np.asarray(ranks)
    return {
        "checked": probe,
        "median_self_rank": float(np.median(ranks_array)),
        "max_self_rank": int(ranks_array.max()),
        "frac_rank_zero": float((ranks_array == 0).mean()),
        "frac_in_top10": float((ranks_array < 10).mean()),
        "gain_absorbed": dictionary.gain_absorbed,
        "per_atom": [
            {"pool_index": int(i), "token_id": int(dictionary.token_ids[i]),
             "self_rank": rank, "self_top1": rank == 0}
            for i, rank in zip(picks, ranks)
        ],
    }


def dictionary_is_valid(check: dict) -> bool:
    """Whether the atom-validity check passed, and therefore whether a fraction exists.

    Two conditions, both from the check as it stands: the median probed atom lenses back
    to its own token at rank 0, and nearly all of them land in the top 10. See
    :data:`DICTIONARY_VALID_MEDIAN_RANK`.

    Coherence deliberately does **not** gate. It is reported beside this, and it matters,
    but part of it is intrinsic to the pool rather than to the pullback -- see
    :data:`DICTIONARY_MAX_COHERENCE` -- so gating on it would block the fraction for a
    reason unrelated to the construction being tested.
    """
    if not check.get("checked"):
        return False
    return (
        check.get("median_self_rank", 1e9) <= DICTIONARY_VALID_MEDIAN_RANK
        and check.get("frac_in_top10", 0.0) >= DICTIONARY_VALID_MIN_TOP10
    )


def sweep_rcond(readout, transport: Transport, direction: np.ndarray, block: int,
                config: PCAJLensConfig, head: np.ndarray,
                gain: np.ndarray | None) -> list[dict]:
    """Rank, coherence and self-rank at each :data:`RCOND_SWEEP` value.

    Printed when the dictionary fails, because ``dict_pinv_rcond`` is a trade-off and the
    operator cannot see which way to move it from one number. Cheap: the SVD is reused,
    so each row costs one matmul plus one dictionary build.
    """
    rows: list[dict] = []
    for rcond in RCOND_SWEEP:
        candidate = transport.retruncate(rcond)
        dictionary = build_dictionary(
            readout, direction, block, config.dict_pool_size, head, gain, candidate
        )
        check = verify_dictionary(readout, dictionary, block, DICTIONARY_PROBE_ATOMS)
        coherence = dictionary_coherence(dictionary, DICTIONARY_MAX_COHERENCE)
        rows.append({
            "rcond": rcond, "rank": candidate.rank,
            "effective_condition": candidate.effective_condition,
            "frac_rank_zero": check.get("frac_rank_zero"),
            "median_self_rank": check.get("median_self_rank"),
            "coherence_mean": coherence.get("mean"),
            "coherence_max": coherence.get("max"),
            "valid": dictionary_is_valid(check),
        })
    return rows


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
    transport: Transport,
) -> dict:
    """The same decomposition on matched-norm random directions.

    The denominator for the headline. ``k`` atoms drawn from a pool of
    ``dict_pool_size`` are a real number of degrees of freedom, so some fraction of
    *any* direction is reconstructable; without this, "``v_J`` is 8% of the variance"
    cannot be told apart from "the pursuit has 20 knobs".

    Each control gets its own pool, from its own lens readout, so it is the identical
    procedure rather than the emotion pool applied to noise.
    """
    fractions: list[float] = []
    for _ in range(max(config.n_random_controls, 0)):
        draw = rng.normal(size=d_model)
        draw = match_norm(draw, target_norm)
        pool = build_dictionary(
            readout, draw, block, config.dict_pool_size, head, gain, transport
        )
        result = decompose_vector(
            draw, pool, config.n_dict_atoms, config.pursuit_steps
        )
        fractions.append(result.frac_reportable)
    if not fractions:
        return {"n": 0}
    values = np.asarray(fractions)
    return {
        "n": len(fractions),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def print_decomposition_table(
    rows: list[dict], control: dict, config: PCAJLensConfig, valid: bool
) -> None:
    print()
    print(RULE)
    print("GATE  How much of each emotion vector can the lens verbalise?")
    print(RULE)
    if not valid:
        print("WITHHELD. The atom-validity check did not pass, so lensing an atom does")
        print("not reliably return its own token, and there is no such thing as the")
        print("reportable fraction of a decomposition into mislabelled directions. The")
        print("numbers exist in the CSV for diagnosis; they are not printed here because")
        print("a printed percentage gets quoted.")
        print()
        print("What to do with it: read J's condition number and retained rank in STEP 3.")
        print("A low rank means dict_pinv_rcond is discarding range the atoms need; a")
        print("huge condition number with a high rank means the opposite. Change one and")
        print("re-run before revising any claim about how much of emotion is verbalisable.")
        print()
        width = max([16, *(len(r["emotion"]) + 2 for r in rows)])
        print(f"{'emotion':<{width}}{'|v|':>9}{'recon err':>11}{'atoms':>7}"
              f"{'own word':>10}")
        print(THIN)
        for row in sorted(rows, key=lambda r: r["emotion"]):
            own = ("n/a" if row["own_word_atom_rank"] is None
                   else f"#{row['own_word_atom_rank']}")
            print(f"{row['emotion']:<{width}}{row['norm']:>9.2f}"
                  f"{row['reconstruction_error']:>11.3f}{row['n_atoms']:>7}{own:>10}")
        print(THIN)
        return
    print("The expectation is a SMALL reportable fraction: the workspace paper puts")
    print("~6-10% of a concept vector's variance inside J-space and the brief expects")
    print("5-15%. A large value means the decomposition or the lens is wrong, not that")
    print("the theory is.")
    print()
    if control.get("n"):
        print(f"Chance baseline: {config.n_dict_atoms} atoms from a pool of "
              f"{config.dict_pool_size} recover")
        print(f"  {control['mean']:.1%} of a matched-norm RANDOM direction "
              f"(p95 {control['p95']:.1%}, max {control['max']:.1%}).")
        print("  Read every fraction below against that. If they are not clearly above")
        print("  it, the split has measured the pursuit's degrees of freedom.")
        if control["mean"] > DEGENERATE_CONTROL_FRACTION:
            print()
            print(f"  DEGENERATE: a random direction already reaches "
                  f"{control['mean']:.0%}. With {config.n_dict_atoms} atoms")
            print(f"  chosen from {config.dict_pool_size} candidates in "
                  f"{len(rows) and ''}{config.dict_pool_size} dimensions' worth of")
            print("  freedom, the pursuit can reconstruct almost anything, so NO reportable")
            print("  fraction below is interpretable -- not a small one and not a large")
            print("  one. Lower n_dict_atoms and dict_pool_size until the control is small")
            print("  relative to the emotions, then re-read the table.")
    else:
        print("Chance baseline: NOT MEASURED (n_random_controls=0). The fractions below")
        print("  have no referent -- some share of any direction is reconstructable.")
    print()
    width = max([16, *(len(r["emotion"]) + 2 for r in rows)])
    print(f"{'emotion':<{width}}{'|v|':>9}{'frac v_J':>10}{'x chance':>10}"
          f"{'frac v_perp':>13}{'recon err':>11}{'atoms':>7}{'own word':>10}")
    print(THIN)
    for row in sorted(rows, key=lambda r: r["emotion"]):
        ratio = (
            f"{row['frac_reportable'] / control['mean']:>9.1f}x"
            if control.get("mean") else "      n/a"
        )
        own = (
            "n/a" if row["own_word_atom_rank"] is None
            else f"#{row['own_word_atom_rank']}"
        )
        print(f"{row['emotion']:<{width}}{row['norm']:>9.2f}"
              f"{row['frac_reportable']:>10.1%}{ratio:>10}"
              f"{row['frac_remainder']:>13.1%}{row['reconstruction_error']:>11.3f}"
              f"{row['n_atoms']:>7}{own:>10}")
    print(THIN)
    fractions = np.asarray([r["frac_reportable"] for r in rows])
    print(f"  reportable fraction: min {fractions.min():.1%}, "
          f"median {np.median(fractions):.1%}, max {fractions.max():.1%}")
    worst_cos = max(abs(r["cos_parts"]) for r in rows)
    print(f"  |cos(v_J, v_perp)|: max {worst_cos:.3f}")
    if worst_cos > 0.2:
        print("    The two parts are not close to orthogonal, so their fractions do not")
        print("    partition the variance and must not be read as competing shares. The")
        print("    nonnegativity constraint is binding hard here.")
    print("  'own word' is where the emotion's own token sits among the selected atoms,")
    print("  by coefficient. v_J should decompose into words that read as the emotion.")


def print_verdict(
    rows: list[dict], control: dict, dictionary_check: dict, coherence: dict,
    transport: Transport, lens_warnings: list[str], config: PCAJLensConfig,
    artifacts: dict[str, object],
) -> None:
    fractions = np.asarray([r["frac_reportable"] for r in rows])
    in_range = [
        r for r in rows
        if 0.01 <= r["frac_reportable"] <= config.frac_j_expected_max
    ]
    above_chance = (
        [r for r in rows if r["frac_reportable"] > 3 * control["mean"]]
        if control.get("mean") else []
    )
    dictionary_ok = dictionary_is_valid(dictionary_check)
    degenerate = control.get("mean", 0.0) > DEGENERATE_CONTROL_FRACTION
    with_word = [r for r in rows if r["own_word_atom_rank"] is not None]

    print()
    print(RULE)
    print("PHASE 6 VERDICT")
    print(RULE)
    print(f"  atoms lens to their token  : {'PASS' if dictionary_ok else 'FAIL'}  "
          f"({dictionary_check.get('frac_rank_zero', 0):.0%} top-1, "
          f"{dictionary_check.get('frac_in_top10', 0):.0%} in the top 10,")
    print(f"                               median rank "
          f"{dictionary_check.get('median_self_rank')}, worst "
          f"{dictionary_check.get('max_self_rank')})")
    print(f"  atom coherence (reported)  : mean "
          f"{coherence.get('mean', float('nan')):.3f}, max "
          f"{coherence.get('max', float('nan')):.3f}"
          + ("   ABOVE the "
             f"{DICTIONARY_MAX_COHERENCE:g} ceiling"
             if coherence.get("max", 0.0) > DICTIONARY_MAX_COHERENCE else ""))
    if coherence.get("max", 0.0) > DICTIONARY_MAX_COHERENCE:
        print("                               atoms that aligned are interchangeable, so")
        print("                               the TOKENS v_J names are not attributable")
        print("                               even where the fraction is. Does not gate:")
        print("                               part of it comes from the pool being one")
        print("                               direction's top tokens, not from J^+.")
    print(f"  J inversion                : rank {transport.rank}/"
          f"{transport.pullback.shape[0]} kept at rcond={transport.rcond:g}, "
          f"cond(J) {transport.condition_number:.3g}")
    print(f"                               round-trip cos "
          f"{transport.roundtrip_cosine:.4f} against jlens's own transport")
    if not dictionary_ok:
        print("  reportable fraction        : WITHHELD -- see below")
    else:
        print(f"  reportable fraction        : {len(in_range)}/{len(rows)} emotions "
              f"inside 1-{config.frac_j_expected_max:.0%} "
              f"(median {np.median(fractions):.1%})")
        if control.get("mean"):
            print(f"  above the chance baseline  : {len(above_chance)}/{len(rows)} at "
                  f"more than 3x the random control's {control['mean']:.1%}")
        if degenerate:
            print("  degrees of freedom         : REVIEW -- the random control reaches "
                  f"{control['mean']:.0%}, so the")
            print("                               fractions above are uninterpretable "
                  "either way")
    print(f"  v_J reads as the emotion   : {len(with_word)}/{len(rows)} selected the "
          "emotion's own token as an atom")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<10}: {path}" if label else f"  {'':<10}  {path}")
    print()
    if not dictionary_ok:
        print("  The atom-validity check gates everything above it, and it FAILED, so no")
        print("  reportable fraction is reported. An atom is d_t = J^+(g*w_t) and is a")
        print("  dictionary entry only if lensing it returns token t; where it does not,")
        print("  v_J is a reconstruction out of directions that do not mean what their")
        print("  labels say, and its variance share is not a measure of reportability.")
        print("  Report this as a property of the construction, not as a fraction.")
        print()
        print("  The knob is dict_pinv_rcond, and STEP 3 printed the sweep over it. Poor")
        print("  self-ranks with a retained rank near d_model mean near-null directions")
        print("  are being amplified into every atom -- raise it. Poor self-ranks with a")
        print("  rank far below d_model mean the truncation is discarding range the")
        print("  atoms need -- lower it. Read the coherence column beside both.")
        print()
        print("  Phases 7 and 8 read frac_reportable from this run's sidecar, and both")
        print("  refuse to proceed while dictionary_valid is false. That is deliberate:")
        print("  Phase 7 picks its emotions by this number and Phase 8 steers with the")
        print("  split, so a mislabelled dictionary would propagate silently.")
        print()
    if dictionary_ok and fractions.max() > config.frac_j_expected_max:
        print(f"  Some fractions exceed {config.frac_j_expected_max:.0%}. Per the brief that")
        print("  points at the decomposition or the lens rather than at the theory:")
        print("  check the dictionary row above, then whether the lens is the converged")
        print("  one (`run.py refit_lens`), before revising any claim about how much of")
        print("  emotion is verbalisable.")
        print()
    if lens_warnings:
        print("  The lens warnings in STEP 2 bound all of this. Read them again.")
        print()
    print("  What Phase 8 does with the output. The four norm-matched directions per")
    print("  emotion are steering conditions, and the point of the split is that if")
    print("  behaviour moves under v_perp while the report channel stays quiet, an")
    print("  emotional state is acting without being reportable. Phase 6 only makes")
    print("  the split; it is not evidence for that on its own -- and note a v_perp")
    print("  effect would still not rule out the concept being re-derived downstream,")
    print("  which is what Phase 9's clamp exists for.")
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
        print(f"  atoms will be J^+ (g * w_t) with dict_pinv_rcond="
              f"{config.dict_pinv_rcond:g}, so J's SVD")
        print("  and its condition number are only available once the lens is loaded.")
        print("  The atom-validity check runs there, and gates whether any reportable")
        print("  fraction is reported at all.")
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

    # --- the open design question, measured ------------------------------- #
    print()
    print(RULE)
    print("STEP 3  Is the pullback actually a dictionary?")
    print(RULE)
    print("An atom must satisfy J d_t = g * w_t, so it is J^-1 (g * w_t) and not")
    print("J^T (g * w_t) -- those agree only for an orthogonal J, and an averaged")
    print("Jacobian is not orthogonal. J is inverted by SVD with small singular values")
    print("truncated; the check below is what says the inversion was the right one.")
    t0 = time.time()
    transport = factor_transport(readout, block, config.dict_pinv_rcond)
    print(f"  SVD of J            : {transport.pullback.shape} on {transport.device} "
          f"in {time.time() - t0:.0f}s, once for every atom in the run")
    print(f"  singular values     : max {transport.singular_values[0]:.4g}, "
          f"min {transport.singular_values[-1]:.4g}")
    print(f"  cond(J)             : {transport.condition_number:.4g}"
          f"   -> {transport.effective_condition:.4g} after truncation")
    print(f"  rank kept           : {transport.rank}/{transport.pullback.shape[0]} "
          f"at dict_pinv_rcond={transport.rcond:g}")
    print(f"  round-trip cos      : {transport.roundtrip_cosine:.4f}  "
          "(J^+ inverts jlens's own transport, so the")
    print("                        stored orientation of J is confirmed, not assumed)")
    if transport.roundtrip_cosine < 0.99:
        print("    LOW. Either the SVD is losing precision or jacobians[block] is stored")
        print("    transposed relative to what this code assumes. Every atom below would")
        print("    then be a unit vector pointing somewhere unrelated to its token, which")
        print("    is invisible to every shape check.")
    probe_direction = space.matrix[0] - space.matrix.mean(axis=0)
    probe_dictionary = build_dictionary(
        readout, probe_direction, block, config.dict_pool_size, head, gain, transport,
    )
    dictionary_check = verify_dictionary(
        readout, probe_dictionary, block, DICTIONARY_PROBE_ATOMS
    )
    coherence = dictionary_coherence(probe_dictionary, DICTIONARY_MAX_COHERENCE)
    dictionary_valid = dictionary_is_valid(dictionary_check)
    print(f"  atoms probed        : {dictionary_check['checked']}")
    print(f"  own token at rank 0 : {dictionary_check['frac_rank_zero']:.0%}")
    print(f"  own token in top 10 : {dictionary_check['frac_in_top10']:.0%}")
    print(f"  median / worst rank : {dictionary_check['median_self_rank']:.0f} / "
          f"{dictionary_check['max_self_rank']}")
    print(f"  gain absorbed       : {dictionary_check['gain_absorbed']}")
    print(f"  atom coherence      : mean {coherence.get('mean', float('nan')):.3f}, "
          f"max {coherence.get('max', float('nan')):.3f} over "
          f"{coherence.get('n_atoms')} atoms")
    print(f"                        (|cos| between atoms. Above "
          f"{DICTIONARY_MAX_COHERENCE:g} they are near-duplicates and which")
    print("                        one the pursuit picks is arbitrary, so the tokens v_J")
    print("                        names stop being attributable. Reported, not gated:")
    print("                        the pool is one direction's top tokens, whose rows are")
    print("                        aligned with it and so with each other regardless.)")
    print(f"  VALID               : {dictionary_valid}"
          + ("" if dictionary_valid else
             f"   (needs median rank <= {DICTIONARY_VALID_MEDIAN_RANK} and "
             f">= {DICTIONARY_VALID_MIN_TOP10:.0%} in the top 10; coherence is\n                        reported above but does not gate)"))
    print()
    print("  per atom: rank of its own token in its own lens readout")
    for entry in dictionary_check.get("per_atom", []):
        token = tokenizer.decode([entry["token_id"]])
        print(f"    {token!r:<18} rank {entry['self_rank']:>6}"
              f"   {'top-1' if entry['self_top1'] else ''}")
    sweep: list[dict] = []
    if not dictionary_valid:
        print()
        print("  FAILED. No reportable fraction will be printed below -- a variance share")
        print("  out of mislabelled or interchangeable directions is not a measurement,")
        print("  and it reads like one. The split is still computed and written, for")
        print("  diagnosis only.")
        print()
        print("  dict_pinv_rcond is the knob. Swept here on the same SVD, so this is the")
        print("  whole trade-off -- rank, conditioning, self-ranks and coherence -- from")
        print("  one run rather than five:")
        sweep = sweep_rcond(readout, transport, probe_direction, block, config, head, gain)
        print(f"    {'rcond':>8}{'rank':>7}{'eff cond':>10}{'top-1':>8}"
              f"{'median':>8}{'coh mean':>10}{'coh max':>9}{'valid':>8}")
        for row in sweep:
            print(f"    {row['rcond']:>8.0e}{row['rank']:>7}"
                  f"{row['effective_condition']:>10.1f}"
                  f"{row['frac_rank_zero']:>8.0%}{row['median_self_rank']:>8.0f}"
                  f"{row['coherence_mean']:>10.3f}{row['coherence_max']:>9.3f}"
                  f"{('yes' if row['valid'] else 'no'):>8}")
        workable = [row["rcond"] for row in sweep if row["valid"]]
        if workable:
            print(f"    -> try --set dict_pinv_rcond={min(workable):g}, the least "
                  "truncation that passes")
        else:
            print("    -> nothing in the sweep passes. That points at the lens rather")
            print("       than at this knob: check whether it is the converged one")
            print("       (`run.py refit_lens`) and re-read the STEP 2 warnings.")

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
        dictionary = build_dictionary(
            readout, vector, block, config.dict_pool_size, head, gain, transport
        )
        result = decompose_vector(
            vector, dictionary, config.n_dict_atoms, config.pursuit_steps
        )
        variants = readout.single_token_variants(emotion)
        own_ids = set(variants.values())
        own_rank = next(
            (rank for rank, tid in enumerate(result.token_ids) if tid in own_ids), None
        )
        tokens = [
            tokenizer.decode([tid]) for tid in result.token_ids[:8]
        ]
        rows.append({
            "emotion": emotion, "norm": norm,
            "frac_reportable": result.frac_reportable,
            "frac_remainder": result.frac_remainder,
            "reconstruction_error": result.reconstruction_error,
            "cos_parts": result.cos_parts,
            "n_atoms": result.n_iterations,
            "own_word_atom_rank": own_rank,
            "own_word_single_token": bool(variants),
            "top_atom_tokens": tokens,
            "coefficients": result.coefficients[:8],
        })
        order.append(emotion)
        saved["v"].append(match_norm(vector, norm))
        saved["v_reportable"].append(match_norm(result.reportable, norm))
        saved["v_remainder"].append(match_norm(result.remainder, norm))
        saved["v_random"].append(match_norm(rng.normal(size=space.d_model), norm))
        print(f"  {emotion:<16} v_J {result.frac_reportable:>6.1%}  "
              f"{result.n_iterations:>2} atoms  " + " ".join(repr(t) for t in tokens[:6]),
              flush=True)

    control = random_control_fractions(
        readout, config, block, float(np.median([r["norm"] for r in rows])),
        space.d_model, head, gain, rng, transport,
    )
    print_decomposition_table(rows, control, config, dictionary_valid)

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
            # Travels with the tensors, not only in the sidecar: these four arrays are
            # what Phase 8 steers with, and a reader who has the safetensors without
            # the JSON must still be told the split is uninterpretable.
            "dictionary_valid": str(dictionary_valid),
        },
    )
    table_path = out_dir / "phase6_decomposition.csv"
    pd.DataFrame([
        {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
        for row in rows
    ]).to_csv(table_path, index=False)

    metadata = {
        **sections,
        "dictionary": dictionary_check,
        "coherence": coherence,
        "rcond_sweep": sweep,
        # Top level, not nested in "dictionary": Phases 7 and 8 read this to decide
        # whether frac_reportable means anything, and a consumer should not have to know
        # which sub-object the gate happened to put it in.
        "dictionary_valid": dictionary_valid,
        "transport": transport.summary(),
        "random_control": control,
        "per_emotion": rows,
        "emotions": order,
        "written": provenance.utc_timestamp(),
    }
    (out_dir / config.decomposition_meta_path.name).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    sections["dictionary"] = dictionary_check
    sections["coherence"] = coherence
    sections["rcond_sweep"] = sweep
    sections["dictionary_valid"] = dictionary_valid
    sections["transport"] = transport.summary()
    sections["random_control"] = control
    sections["per_emotion"] = rows
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 6 GATE -- {config.run_name}",
        sections=sections, txt_name="phase6_gate.txt", json_name="phase6_gate.json",
    )

    print_verdict(
        rows, control, dictionary_check, coherence, transport, lens_warnings, config,
        artifacts={
            "directions": out_dir / config.decomposition_path.name,
            "metadata": out_dir / config.decomposition_meta_path.name,
            "table": table_path,
            "records": txt_path,
            "": json_path,
        },
    )
    # Exit 3 on an invalid dictionary, matching the other gates' failure code. Printing
    # "do not read this" and exiting 0 would leave Phase 7 free to rank emotions by a
    # number the gate has just disowned.
    return 0 if dictionary_valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
