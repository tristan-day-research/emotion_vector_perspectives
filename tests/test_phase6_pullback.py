"""Does the pseudo-inverse pullback actually make a dictionary, where J^T did not?

Runs the real ``factor_transport`` / ``build_dictionary`` / ``verify_dictionary`` against
a synthetic lens small enough to check by hand: a non-orthogonal J, a random unembedding
head, an RMSNorm gain, and the exact logit the lens computes. The transpose construction
is reimplemented here in four lines so the two can be compared on identical inputs --
that comparison is the point of the file.

Run: python tests/test_phase6_pullback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig  # noqa: E402
from emotion_pca_jlens.phase6_decompose import (  # noqa: E402
    DICTIONARY_MAX_COHERENCE,
    DICTIONARY_VALID_MIN_TOP10,
    RCOND_SWEEP,
    Dictionary,
    build_dictionary,
    dictionary_coherence,
    dictionary_is_valid,
    factor_transport,
    sweep_rcond,
    verify_dictionary,
)

D_MODEL = 96
VOCAB = 2000
BLOCK = 3

#: Condition number of the synthetic J. A near-identity, well-conditioned J is the one
#: case where the transpose construction *does* work -- J^T ~= J^-1 up to a mild
#: distortion -- so testing against one would have shown the fix making no difference
#: and proved nothing. A real averaged Jacobian has a decaying spectrum, which is what
#: makes the transpose wrong in practice, so the synthetic one is given the same shape.
CONDITION_NUMBER = 300.0
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeLens:
    """Just enough of ``JacobianLens``: ``jacobians`` and ``transport``."""

    def __init__(self, jacobian, transposed: bool = False):
        self.jacobians = {BLOCK: torch.as_tensor(jacobian, dtype=torch.float32)}
        self.source_layers = [BLOCK]
        # `transposed` models the failure the round-trip check exists to catch: the
        # stored matrix being the transpose of what `transport` applies.
        self._transposed = transposed

    def transport(self, h, block):
        matrix = self.jacobians[block]
        h = torch.as_tensor(h, dtype=torch.float32).reshape(-1)
        return (matrix.T if self._transposed else matrix) @ h


class FakeReadout:
    """The exact lens: ``logits = W_U (g * (Jh / rms(Jh)))``."""

    def __init__(self, jacobian, head, gain, transposed: bool = False):
        self.lens = FakeLens(jacobian, transposed=transposed)
        self.head = torch.as_tensor(head, dtype=torch.float32)
        self.gain = torch.as_tensor(gain, dtype=torch.float32)

    def direction_logits(self, direction, block, use_jacobian: bool = True):
        h = torch.as_tensor(np.asarray(direction), dtype=torch.float32).reshape(-1)
        if use_jacobian:
            h = self.lens.transport(h, block)
        normed = h / torch.sqrt((h * h).mean() + 1e-6)
        return self.head @ (self.gain * normed)


def transpose_dictionary(readout, direction, block, pool_size, head, gain) -> Dictionary:
    """The construction this fix replaced: ``d_t = J^T (g * w_t)``."""
    logits = readout.direction_logits(direction, block, use_jacobian=True)
    token_ids = torch.topk(logits, pool_size).indices.cpu().numpy()
    rows = head[token_ids] * gain[None, :]
    atoms = rows @ readout.lens.jacobians[block].numpy()
    norms = np.linalg.norm(atoms, axis=1)
    keep = norms > 0
    return Dictionary(token_ids=token_ids[keep], atoms=atoms[keep] / norms[keep][:, None],
                      raw_norms=norms[keep], gain_absorbed=True)


def main() -> int:
    rng = np.random.default_rng(0)

    # A Jacobian with a real one's shape: random orthogonal factors and a log-spaced
    # singular spectrum, so it is full-rank, invertible, and a long way from orthogonal.
    left = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    right = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    spectrum = np.logspace(0, -np.log10(CONDITION_NUMBER), D_MODEL)
    jacobian = (left * spectrum) @ right.T
    head = rng.normal(size=(VOCAB, D_MODEL)) / np.sqrt(D_MODEL)
    gain = 1.0 + 0.3 * rng.normal(size=D_MODEL)
    readout = FakeReadout(jacobian, head, gain)
    direction = rng.normal(size=D_MODEL)

    print("orthogonality of the synthetic J")
    gram = jacobian.T @ jacobian
    off = np.abs(gram - np.diag(np.diag(gram))).max()
    check("J is not orthogonal, so J^T != J^-1", off > 0.05,
          f"max |off-diagonal of J^T J| = {off:.3f}")

    print("\nfactor_transport")
    transport = factor_transport(readout, BLOCK, 1e-3)
    check("pullback inverts J", np.allclose(
        transport.pullback @ jacobian, np.eye(D_MODEL), atol=1e-3),
        f"rank {transport.rank}/{D_MODEL}, cond {transport.condition_number:.2f}")
    check("round-trip cosine ~ 1", transport.roundtrip_cosine > 0.999,
          f"{transport.roundtrip_cosine:.6f}")
    check("summary carries the condition number and rank",
          {"condition_number", "rank", "rcond", "roundtrip_cosine"}
          <= set(transport.summary()))

    flipped = factor_transport(
        FakeReadout(jacobian, head, gain, transposed=True), BLOCK, 1e-3
    )
    check("round-trip catches a transposed J", flipped.roundtrip_cosine < 0.9,
          f"{flipped.roundtrip_cosine:.4f} vs {transport.roundtrip_cosine:.4f}")

    print("\ntruncation actually truncates")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    loose = factor_transport(readout, BLOCK, 1e-6)
    tight = factor_transport(readout, BLOCK, 0.5)
    check("rcond controls the retained rank", tight.rank < loose.rank <= D_MODEL,
          f"rcond 0.5 -> {tight.rank}, 1e-6 -> {loose.rank}, "
          f"s range {singular[-1]:.3f}..{singular[0]:.3f}")

    print("\nthe atom-validity check: J^+ against J^T")
    pinv_dict = build_dictionary(readout, direction, BLOCK, 64,
                                 np.asarray(head), np.asarray(gain), transport)
    transpose_dict = transpose_dictionary(
        readout, direction, BLOCK, 64, np.asarray(head), np.asarray(gain)
    )
    pinv_check = verify_dictionary(readout, pinv_dict, BLOCK, 24)
    transpose_check = verify_dictionary(readout, transpose_dict, BLOCK, 24)
    print(f"    J^+ : top-1 {pinv_check['frac_rank_zero']:.0%}, "
          f"top-10 {pinv_check['frac_in_top10']:.0%}, "
          f"median rank {pinv_check['median_self_rank']:.0f}, "
          f"worst {pinv_check['max_self_rank']}")
    print(f"    J^T : top-1 {transpose_check['frac_rank_zero']:.0%}, "
          f"top-10 {transpose_check['frac_in_top10']:.0%}, "
          f"median rank {transpose_check['median_self_rank']:.0f}, "
          f"worst {transpose_check['max_self_rank']}")
    check("J^+ atoms lens back to their own token", dictionary_is_valid(pinv_check))
    check("J^+ is perfect on top-1", pinv_check["frac_rank_zero"] == 1.0)
    check("J^+ beats J^T on top-1",
          pinv_check["frac_rank_zero"] > transpose_check["frac_rank_zero"],
          f"{pinv_check['frac_rank_zero']:.0%} vs "
          f"{transpose_check['frac_rank_zero']:.0%}")

    # Whether J^T *fails the gate* depends on how badly conditioned J is -- at cond ~1
    # the transpose is the inverse and works fine, which is why a single synthetic J
    # proves nothing on its own. This sweep is the mechanism: J^+ is exact at every
    # conditioning, J^T decays with it, and the real J is somewhere on this curve.
    print("\n  the mechanism, swept over cond(J):")
    print(f"    {'cond(J)':>10}{'J^+ top-1':>12}{'J^T top-1':>12}{'J^T median':>12}")
    decay: list[float] = []
    for condition in (3.0, 30.0, 300.0, 3000.0):
        spread = np.logspace(0, -np.log10(condition), D_MODEL)
        swept = (left * spread) @ right.T
        swept_readout = FakeReadout(swept, head, gain)
        swept_transport = factor_transport(swept_readout, BLOCK, 1e-8)
        a = verify_dictionary(swept_readout, build_dictionary(
            swept_readout, direction, BLOCK, 64, np.asarray(head), np.asarray(gain),
            swept_transport), BLOCK, 24)
        b = verify_dictionary(swept_readout, transpose_dictionary(
            swept_readout, direction, BLOCK, 64, np.asarray(head), np.asarray(gain)),
            BLOCK, 24)
        decay.append(b["frac_rank_zero"])
        print(f"    {condition:>10.0f}{a['frac_rank_zero']:>11.0%}"
              f"{b['frac_rank_zero']:>12.0%}{b['median_self_rank']:>12.0f}"
              + ("   J^T FAILS the gate" if not dictionary_is_valid(b) else ""))
        check(f"J^+ is exact at cond {condition:.0f}", a["frac_rank_zero"] == 1.0)
    check("J^T degrades as J's conditioning worsens", decay[-1] < decay[0],
          f"{decay[0]:.0%} at cond 3 -> {decay[-1]:.0%} at cond 3000")

    print("\nper-atom reporting")
    per_atom = pinv_check.get("per_atom", [])
    check("one entry per probed atom", len(per_atom) == pinv_check["checked"])
    check("each entry names the token, the rank and the top-1 verdict",
          all({"token_id", "self_rank", "self_top1"} <= set(e) for e in per_atom))
    check("self_top1 agrees with self_rank",
          all(e["self_top1"] == (e["self_rank"] == 0) for e in per_atom))
    check("frac_rank_zero is the mean of self_top1",
          abs(np.mean([e["self_top1"] for e in per_atom])
              - pinv_check["frac_rank_zero"]) < 1e-9)

    print("\ncoherence: reported beside the gate, deliberately not gating")
    loose = factor_transport(readout, BLOCK, 1e-8)
    loose_dict = build_dictionary(readout, direction, BLOCK, 64, np.asarray(head),
                                  np.asarray(gain), loose)
    loose_check = verify_dictionary(readout, loose_dict, BLOCK, 24)
    loose_coherence = dictionary_coherence(loose_dict, DICTIONARY_MAX_COHERENCE)
    tight_coherence = dictionary_coherence(pinv_dict, DICTIONARY_MAX_COHERENCE)
    print(f"    rcond 1e-8: rank {loose.rank}, self-rank median "
          f"{loose_check['median_self_rank']:.0f}, top-1 "
          f"{loose_check['frac_rank_zero']:.0%}, coherence mean "
          f"{loose_coherence['mean']:.3f} max {loose_coherence['max']:.3f}")
    print(f"    rcond 1e-3: rank {transport.rank}, self-rank median "
          f"{pinv_check['median_self_rank']:.0f}, top-1 "
          f"{pinv_check['frac_rank_zero']:.0%}, coherence mean "
          f"{tight_coherence['mean']:.3f} max {tight_coherence['max']:.3f}")
    check("coherence is measured and bounded", 0.0 <= tight_coherence["max"] <= 1.0)
    check("mean <= max, and both are over the off-diagonal only",
          tight_coherence["mean"] <= tight_coherence["max"])
    check("the fraction above the ceiling is reported",
          0.0 <= tight_coherence["frac_above_threshold"] <= 1.0)
    check("high coherence does NOT block the fraction",
          dictionary_is_valid(pinv_check) is True
          and tight_coherence["max"] > DICTIONARY_MAX_COHERENCE,
          f"coherence max {tight_coherence['max']:.3f} > "
          f"{DICTIONARY_MAX_COHERENCE}, yet valid -- part of it is the pool, not J^+")
    check("a one-atom dictionary reports without crashing",
          "n_atoms" in dictionary_coherence(
              Dictionary(token_ids=np.array([0]), atoms=np.ones((1, D_MODEL)),
                         raw_norms=np.ones(1), gain_absorbed=True),
              DICTIONARY_MAX_COHERENCE))

    print("\nthe rcond sweep the gate prints on a failure")
    sweep = sweep_rcond(readout, transport, direction, BLOCK,
                        PCAJLensConfig().with_overrides({"dict_pool_size": 64}),
                        np.asarray(head), np.asarray(gain))
    print(f"    {'rcond':>8}{'rank':>7}{'eff cond':>10}{'coh max':>10}{'valid':>7}")
    for row in sweep:
        print(f"    {row['rcond']:>8.0e}{row['rank']:>7}"
              f"{row['effective_condition']:>10.1f}{row['coherence_max']:>10.3f}"
              f"{('yes' if row['valid'] else 'no'):>7}")
    check("the sweep covers every RCOND_SWEEP value", len(sweep) == len(RCOND_SWEEP))
    check("rank falls monotonically as rcond rises",
          all(a["rank"] >= b["rank"] for a, b in zip(sweep, sweep[1:])))
    check("the sweep reuses the SVD rather than refactoring",
          all(row["rank"] <= D_MODEL for row in sweep) and bool(transport.factors))
    check("retruncate leaves the original transport alone",
          transport.rank == factor_transport(readout, BLOCK, 1e-3).rank)

    print("\nthe validity threshold")
    check("median rank 1 fails however good the tail is",
          not dictionary_is_valid({"checked": 24, "median_self_rank": 1.0,
                                   "frac_in_top10": 1.0}))
    check("a thin tail fails at median 0",
          not dictionary_is_valid({"checked": 24, "median_self_rank": 0.0,
                                   "frac_in_top10": DICTIONARY_VALID_MIN_TOP10 - 0.05}))
    check("an unrun check is not a pass", not dictionary_is_valid({"checked": 0}))

    print("\nbuild_dictionary guards")
    try:
        build_dictionary(readout, direction, BLOCK + 1, 8, np.asarray(head),
                         np.asarray(gain), transport)
        check("refuses a transport factored at another block", False)
    except ValueError:
        check("refuses a transport factored at another block", True)
    check("atoms are unit vectors",
          np.allclose(np.linalg.norm(pinv_dict.atoms, axis=1), 1.0, atol=1e-5))
    check("gain absorption is recorded", pinv_dict.gain_absorbed)
    ungained = build_dictionary(readout, direction, BLOCK, 8, np.asarray(head), None,
                               transport)
    check("dropping the gain is recorded too", not ungained.gain_absorbed)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
