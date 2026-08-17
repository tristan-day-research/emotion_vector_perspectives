"""Are the atoms the lens's own measurement weights, and is the gate the right gate?

The claim under test is an identity, not a heuristic: the lens logit for token ``t`` at
residual ``h`` is ``<J^T u_t, h> / rms(Jh)``, so ``J^T u_t`` is what the lens reads ``t``
with. That is checked against a synthetic lens whose logits are computed the long way
round, on a badly conditioned non-orthogonal ``J`` where ``J^T`` and ``J^+`` are far
apart -- so "read" and "write" atoms are genuinely different objects here and cannot be
passing for the same reason.

Also checks the two things the gate now rests on: the random-null p-value, and the
k-dependence that makes ``v_perp`` a statement about an approximation rather than about
the model.

Run: python tests/test_phase6_dictionary.py
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
    GATE_ALPHA,
    Dictionary,
    build_dictionary,
    decompose_vector,
    dictionary_coherence,
    factor_transport,
    match_norm,
    p_value_vs_random,
    random_control_fractions,
    significance_threshold,
    unembed_parts,
    verify_read_directions,
)

D_MODEL = 96
VOCAB = 2000
BLOCK = 3

#: Condition number of the synthetic J. Deliberately far from 1: at cond ~ 1 the transpose
#: and the pseudo-inverse coincide, and every test below would pass for the wrong reason.
CONDITION_NUMBER = 300.0
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeLens:
    """Just enough of ``JacobianLens``: ``jacobians`` and ``transport``."""

    def __init__(self, jacobian, transposed: bool = False):
        matrix = torch.as_tensor(jacobian, dtype=torch.float32)
        # Two blocks, so the "transport factored at another block" guard can be
        # exercised against a block that exists rather than against a KeyError.
        self.jacobians = {BLOCK: matrix, BLOCK + 1: matrix.clone()}
        self.source_layers = [BLOCK, BLOCK + 1]
        # `transposed` models the orientation bug the read-identity check exists to catch.
        self._transposed = transposed

    def transport(self, h, block):
        matrix = self.jacobians[block]
        h = torch.as_tensor(h, dtype=torch.float32).reshape(-1)
        return (matrix.T if self._transposed else matrix) @ h


class FakeModel:
    """``unembed`` the long way round: ``W_U (g * (x / rms(x)))``."""

    def __init__(self, head, gain):
        self.d_model = D_MODEL
        self.n_layers = 6
        self.tokenizer = None
        self._lm_head = type("H", (), {"weight": torch.as_tensor(head, dtype=torch.float32)})()
        self._final_norm = type("N", (), {"weight": torch.as_tensor(gain, dtype=torch.float32)})()
        self._logit_softcap = None

    def unembed(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        normed = x / torch.sqrt((x * x).mean() + 1e-8)
        return self._lm_head.weight @ (self._final_norm.weight * normed)


class FakeReadout:
    def __init__(self, jacobian, head, gain, transposed: bool = False):
        self.lens = FakeLens(jacobian, transposed=transposed)
        self.model = FakeModel(head, gain)

    def direction_logits(self, direction, block, use_jacobian: bool = True):
        h = torch.as_tensor(np.asarray(direction), dtype=torch.float32).reshape(-1)
        if use_jacobian:
            h = self.lens.transport(h, block)
        return self.model.unembed(h)


def main() -> int:
    rng = np.random.default_rng(0)
    left = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    right = np.linalg.qr(rng.normal(size=(D_MODEL, D_MODEL)))[0]
    spectrum = np.logspace(0, -np.log10(CONDITION_NUMBER), D_MODEL)
    jacobian = (left * spectrum) @ right.T
    head_raw = rng.normal(size=(VOCAB, D_MODEL)) / np.sqrt(D_MODEL)
    gain_raw = 1.0 + 0.3 * rng.normal(size=D_MODEL)
    readout = FakeReadout(jacobian, head_raw, gain_raw)
    head, gain = unembed_parts(readout)
    direction = rng.normal(size=D_MODEL)

    print("the synthetic lens")
    gram = jacobian.T @ jacobian
    off = np.abs(gram - np.diag(np.diag(gram))).max()
    check("J is not orthogonal, so J^T and J^+ are different objects", off > 0.05,
          f"max |off-diagonal of J^T J| = {off:.3f}, cond {CONDITION_NUMBER:.0f}")
    check("the gain is recovered from the model, not assumed to be ones",
          gain is not None and np.allclose(gain, gain_raw))

    print("\nread atoms are the lens's measurement weights")
    read = build_dictionary(readout, direction, BLOCK, 64, head, gain)
    check("mode is recorded as read", read.mode == "read")
    check("atoms are unit vectors",
          np.allclose(np.linalg.norm(read.atoms, axis=1), 1.0, atol=1e-5))
    check("the raw weights are kept, not just the normalised ones",
          read.raw.shape == read.atoms.shape
          and not np.allclose(read.raw, read.atoms))
    # The identity, by hand: raw atom i must literally equal J^T (g * w_{t_i}).
    expected = (head[read.token_ids] * gain[None, :]) @ jacobian
    check("raw atom == J^T (g * w_t), elementwise",
          np.allclose(read.raw, expected, atol=1e-5))
    identity = verify_read_directions(readout, read, BLOCK, np.random.default_rng(1))
    check("the identity holds against the lens's own logits", identity["holds"],
          f"min corr {identity['min_correlation']:.8f} over {identity['n_probes']} probes")
    check("the correlation is 1 to float precision, not merely high",
          identity["min_correlation"] > 0.9999,
          f"{identity['min_correlation']:.8f}")

    print("\n  the identity catches a transposed J, which no shape check would")
    flipped = FakeReadout(jacobian, head_raw, gain_raw, transposed=True)
    flipped_dict = build_dictionary(flipped, direction, BLOCK, 64, head, gain)
    flipped_identity = verify_read_directions(
        flipped, flipped_dict, BLOCK, np.random.default_rng(1)
    )
    check("a transposed J fails the identity", not flipped_identity["holds"],
          f"min corr {flipped_identity['min_correlation']:.4f} vs "
          f"{identity['min_correlation']:.4f}")
    check("its atoms are still unit vectors of the right shape, hence undetectable "
          "any other way",
          flipped_dict.atoms.shape == read.atoms.shape
          and np.allclose(np.linalg.norm(flipped_dict.atoms, axis=1), 1.0, atol=1e-5))

    print("\n  anything orthogonal to every read direction moves no logit")
    all_reads = (head * gain[None, :]) @ jacobian
    basis = np.linalg.svd(all_reads, full_matrices=True)[2]
    rank = np.linalg.matrix_rank(all_reads)
    if rank < D_MODEL:
        null_vector = basis[rank]
        logits = readout.direction_logits(null_vector, BLOCK).numpy()
        check("a null-space direction produces no logit spread",
              float(np.abs(logits).max()) < 1e-3, f"max |logit| {np.abs(logits).max():.2e}")
    else:
        # With VOCAB >> d_model the read directions span the space, so there is no null
        # space to probe -- report it rather than skipping silently.
        check("read directions span the residual space at this vocab size",
              rank == D_MODEL,
              f"rank {rank}/{D_MODEL} of {VOCAB} read directions: no null space exists "
              "here, so the orthogonality claim is vacuous at this size, not false")

    print("\nwrite_space is a different dictionary, not a better one")
    transport = factor_transport(readout, BLOCK, 1e-2)
    write = build_dictionary(readout, direction, BLOCK, 64, head, gain, transport)
    check("mode is recorded as write", write.mode == "write")
    overlap = np.abs(np.sum(read.atoms * write.atoms, axis=1))
    check("write atoms differ from read atoms", float(overlap.max()) < 0.99,
          f"max |cos(read_t, write_t)| = {overlap.max():.3f}, "
          f"median {np.median(overlap):.3f}")
    write_identity = verify_read_directions(
        readout, write, BLOCK, np.random.default_rng(1)
    )
    check("write atoms do NOT satisfy the read identity -- as expected",
          not write_identity["holds"],
          f"min corr {write_identity['min_correlation']:.4f}; they answer the write "
          "question, so this is a label, not a defect")
    check("build_dictionary rejects a transport factored at another block",
          _raises(lambda: build_dictionary(
              readout, direction, BLOCK + 1, 8, head, gain, transport), ValueError))

    print("\nthe gate: reconstruction fraction against a random null")
    config = PCAJLensConfig().with_overrides({
        # 400 controls: the assertion below is against the Bonferroni-corrected
# threshold for 16 emotions (0.0031), and fewer draws could not reach it
# however strong the planted structure -- the p-value floors at 1/(n+1).
        "dict_pool_size": 64, "n_random_controls": 400, "pursuit_steps": 60,
    })
    atom_counts = (16, 25)
    # A vector planted inside the code's reach: a nonnegative mix of read atoms plus a
    # smaller residual. Its fraction must beat the null; an unstructured draw must not.
    planted = read.atoms[:8].T @ np.full(8, 1.0)
    noise = rng.normal(size=D_MODEL)
    noise -= (noise @ planted) * planted / (planted @ planted)
    planted = planted + noise * (0.5 * np.linalg.norm(planted) / np.linalg.norm(noise))
    controls = random_control_fractions(
        readout, config, BLOCK, float(np.linalg.norm(planted)), D_MODEL, head, gain,
        np.random.default_rng(2), None, atom_counts,
    )
    check("the null is measured at every k", set(controls) == set(atom_counts))
    check("the null distribution is kept, not just its summary",
          all(len(controls[k]["fractions"]) == 400 for k in atom_counts))

    pool = build_dictionary(readout, planted, BLOCK, config.dict_pool_size, head, gain)
    print(f"    {'k':>4}{'planted':>10}{'null mean':>11}{'p':>9}{'random p':>10}")
    for k in atom_counts:
        observed = decompose_vector(planted, pool, k, config.pursuit_steps).frac_reportable
        p_planted = p_value_vs_random(observed, controls[k])
        # A fresh unstructured draw put through the same machinery: its p-value should be
        # unremarkable, which is what says the test is testing structure and not the code.
        draw = match_norm(np.random.default_rng(3).normal(size=D_MODEL),
                          float(np.linalg.norm(planted)))
        draw_pool = build_dictionary(readout, draw, BLOCK, config.dict_pool_size, head, gain)
        p_draw = p_value_vs_random(
            decompose_vector(draw, draw_pool, k, config.pursuit_steps).frac_reportable,
            controls[k],
        )
        print(f"    {k:>4}{observed:>10.1%}{controls[k]['mean']:>11.1%}"
              f"{p_planted:>9.4f}{p_draw:>10.4f}")
        check(f"k={k}: the planted vector beats the null",
              p_planted < significance_threshold(16))
        check(f"k={k}: an unstructured draw does not",
              p_draw > significance_threshold(16), f"p={p_draw:.4f}")

    print("\n  the p-value's shape")
    fake = {"fractions": [0.1] * 99}
    check("add-one estimator, so a clean win is not reported as p=0",
          p_value_vs_random(0.9, fake) == 1 / 100, f"{p_value_vs_random(0.9, fake)}")
    check("the floor is 1/(1+n), which is why n_random_controls matters",
          p_value_vs_random(0.9, {"fractions": [0.1] * 15}) == 1 / 16)
    check("a fraction at the null's level gets p ~ 1",
          p_value_vs_random(0.1, fake) == 1.0)
    check("no controls means no p-value, rather than a fabricated one",
          p_value_vs_random(0.5, {"n": 0}) is None)
    check("the default n_random_controls resolves the corrected alpha",
          1 / (1 + PCAJLensConfig().n_random_controls) < significance_threshold(16),
          f"floor {1 / (1 + PCAJLensConfig().n_random_controls):.4f} < "
          f"alpha {significance_threshold(16):.4f}")
    check("alpha is Bonferroni-corrected over the emotions",
          significance_threshold(16) == GATE_ALPHA / 16)

    print("\n  the fraction rises with k, which is why both are reported")
    at_k = {
        k: decompose_vector(planted, pool, k, config.pursuit_steps).frac_reportable
        for k in (4, 16, 25, 40)
    }
    print("    " + "  ".join(f"k={k}: {v:.1%}" for k, v in at_k.items()))
    check("monotone in k, so v_perp is a statement about k and not about the model",
          all(at_k[a] <= at_k[b] + 1e-9
              for a, b in zip(sorted(at_k), sorted(at_k)[1:])))
    check("the reachable set genuinely grows -- k=40 captures more than k=4",
          at_k[40] > at_k[4], f"{at_k[4]:.1%} -> {at_k[40]:.1%}")

    print("\ncoherence is reported, not gated")
    coherence = dictionary_coherence(read, DICTIONARY_MAX_COHERENCE)
    check("mean <= max, over the off-diagonal only",
          coherence["mean"] <= coherence["max"])
    check("the fraction above the ceiling is reported",
          0.0 <= coherence["frac_above_threshold"] <= 1.0,
          f"mean {coherence['mean']:.3f}, max {coherence['max']:.3f}")
    check("a one-atom dictionary reports without crashing",
          "n_atoms" in dictionary_coherence(
              Dictionary(token_ids=np.array([0]), atoms=np.ones((1, D_MODEL)),
                         raw=np.ones((1, D_MODEL)), raw_norms=np.ones(1),
                         gain_absorbed=True, mode="read"),
              DICTIONARY_MAX_COHERENCE))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


def _raises(thunk, exception) -> bool:
    try:
        thunk()
    except exception:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
