"""Phase 9 (GATE): the re-entry clamp -- the decisive control, if there is one to run.

What this stage does
--------------------
Re-runs Phase 8's ``v_perp`` steering while **holding the emotion's J-lens coordinates at
clean-pass values at every position and every layer**, so the concept cannot be
re-derived downstream and re-enter the workspace to report itself. The decisive cell is
``v_perp``, clamped, behaviour channel: if behaviour still shifts while the report
channel is suppressed, that is an emotional state steering action without being
reportable.

Phase 8 could not distinguish that from re-entry. This can, and it is the only phase that
can, which is also why every number it produces is worthless unless the clamp is first
shown to work.

What is clamped, exactly
------------------------
The lens reads token ``t`` at block ``l`` with the weight vector ``J_l^T (g * w_t)`` --
see Phase 6. So the emotion's J-space at block ``l`` is
``span{J_l^T (g * w_t) : t in T}`` for its token set ``T``, and "clamping its
coordinates" means, at every position::

    h  <-  h - A_l (A_l^T h) + A_l c_clean

with ``A_l`` an orthonormal basis of that span. Outside ``span(A_l)`` the residual is
untouched -- that is linear algebra, not an empirical claim. What *is* empirical is
whether ``span(A_l)`` is small enough to be a concept rather than a large slice of the
model, which is what the collateral check measures.

Every fitted block by default. Re-entry is downstream layers re-deriving the concept, so
a clamp that skips layers leaves exactly the room it needs; a partial clamp cannot tell
"the effect bypassed the workspace" from "the effect re-entered at block 41". Runs with
a narrower ``clamp_blocks`` are marked NOT DECISIVE.

The hard part: what "clean-pass" means once the tokens diverge
--------------------------------------------------------------
A steered model generates different tokens from an unsteered one, so after the prompt
there is no position-by-position correspondence to clamp *to*. Three wrong answers and
the one this uses:

* Clamping only the prompt leaves generation -- where the behaviour is -- unclamped.
* Clamping generated position ``i`` to the clean run's position ``i`` compares different
  tokens, so the target is a coordinate for a sentence the model never wrote.
* Teacher-forcing the clean run onto the steered text and calling that the baseline is
  circular: the text is what the clamp was supposed to influence.

What Phase 9 does instead is a **paired counterfactual, one decode step at a time.** The
steered run chooses each token; the clean run is then advanced over *that same token*, and
its coordinates at the new position become the clamp target for the steered run's next
step. The clean pass is therefore always answering "what would these coordinates be, on
this exact prefix, without the perturbation" -- which is the counterfactual the clamp
wants and the only one that is well defined.

The cost is two forward passes per decode step, and one prompt at a time rather than a
batch: the two runs need separate KV caches advanced in lockstep, and batching that with
left padding is a second chance to get the position bookkeeping wrong for no scientific
gain. Phase 9 is one decisive cell, not a dose-response curve.

The verification comes first, and it can refuse the result
-----------------------------------------------------------
Three checks print before any behavioural number, and the gate will not interpret the
decisive cell unless the first two pass:

1. **No-op.** With no steering, the clamp replaces the clean coordinates with the clean
   coordinates, so it must be the identity. Checked numerically (does the residual's
   coordinate actually land on the target) and behaviourally (is the text unchanged).
   This catches implementation faults, which is most of what can go wrong here.
2. **Suppression.** Steering with ``v`` -- which has a real J-component -- raises the
   report channel. The clamp must bring it back down, by at least
   ``clamp_min_report_suppression`` of the lift. A clamp that does not suppress is not
   clamping, and the cell below it means nothing.
3. **Collateral.** Unrelated J-space content must survive. Measured as the rank
   correlation of the lens readout over **control tokens** -- tokens outside the clamped
   set -- between the clamped and unclamped steered runs, plus the share of the clean
   residual's norm that lies inside the clamped subspace. A clamp that suppresses the
   report channel by flattening the whole residual passes check 2 and proves nothing.

Usage::

    python run.py phase9 --dry-run     # preconditions, subspace size, cost; no weights
    python run.py phase9 --verify-only # the three checks, no behavioural grid
    python run.py phase9               # verification then the decisive cell
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens import phase8_steer as p8
from emotion_pca_jlens.channel_prompts import (
    BEHAVIOUR_TASKS,
    REPORT_PROMPTS,
    invalid_rates,
)
from emotion_pca_jlens.pca_jlens_config import (
    PCAJLensConfig,
    load_config,
    resolve_block_spec,
)

RULE = "=" * 78
THIN = "-" * 78

#: The two steering conditions Phase 9 runs, each clamped and unclamped. ``v`` is here
#: for the suppression check rather than for its own sake: it is the condition with a
#: real J-component, so it is the one that can show the clamp working at all.
CONDITIONS: tuple[str, ...] = ("v", "v_remainder")

#: How a (condition, clamp) pair is named in the tables. Folded into one string so
#: Phase 8's aggregation functions can be reused unchanged -- they group by
#: ``condition``, and a separate boolean column would need a fork of all of them.
def cell_name(condition: str, clamped: bool) -> str:
    return f"{condition}|clamp" if clamped else condition


CELL_LABELS: dict[str, str] = {
    "v": "v                the whole vector, unclamped",
    "v|clamp": "v + clamp        the suppression check",
    "v_remainder": "v_perp           Phase 8's condition, unclamped",
    "v_remainder|clamp": "v_perp + clamp   THE DECISIVE CELL",
}

#: Control tokens for the collateral check: sampled from the vocabulary outside the
#: clamped set. Enough that a rank correlation over them is stable, few enough that the
#: readout comparison is one cheap pass.
N_CONTROL_TOKENS = 512

#: Singular-value floor, relative to the largest, for keeping a direction in the clamp
#: basis. The read directions for related tokens are correlated -- that is why they are a
#: concept and not a random set -- so the raw stack is rank-deficient, and orthonormalising
#: without a floor would put numerical noise into the basis and clamp arbitrary directions.
BASIS_RANK_RCOND = 1e-4

#: Single-stream decode throughput for the time estimate; the estimate then doubles it
#: for the paired clean pass. Order-of-magnitude arithmetic, not a measurement: batch-1
#: decode of a 32B model is bandwidth-bound, and this is the number that makes Phase 9's
#: cost visible before it is paid rather than after.
TOKENS_PER_SECOND = 25.0

#: Roles a generated row can have. ``noop`` rows are the alpha=0 CLAMPED cell: they exist
#: for the no-op check and are kept out of the grid, because a second alpha=0 condition
#: would give the shift-from-baseline arithmetic two candidate baselines to pick between.
ROLE_BASELINE, ROLE_NOOP, ROLE_CELL = "baseline", "noop", "cell"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 9 gate: clamp the emotion's J-space coordinates and re-run "
                    "v_perp -> behaviour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="check the preconditions from Phases 6-8, report the clamp's shape and the "
             "cost; never loads model weights",
    )
    p.add_argument(
        "--verify-only", action="store_true",
        help="run the three verification checks and stop. The right first run: if the "
             "clamp does not verify, the behavioural grid is hours spent on nothing",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip judge scoring. Leaves the report channel unscored, which the "
             "suppression check needs, so the verification cannot pass",
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
# The clamped subspace
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClampBasis:
    """Orthonormal bases for one emotion's J-space, one per clamped block."""

    emotion: str
    token_ids: list[int]
    tokens: list[str]
    bases: dict[int, np.ndarray] = field(repr=False, default_factory=dict)
    ranks: dict[int, int] = field(default_factory=dict)

    @property
    def blocks(self) -> list[int]:
        return sorted(self.bases)

    def summary(self) -> dict:
        ranks = [self.ranks[b] for b in self.blocks]
        return {
            "emotion": self.emotion,
            "n_tokens": len(self.token_ids),
            "tokens": self.tokens,
            "n_blocks": len(self.bases),
            "blocks": [self.blocks[0], self.blocks[-1]] if self.bases else [],
            "rank_min": int(min(ranks)) if ranks else 0,
            "rank_max": int(max(ranks)) if ranks else 0,
            "rank_median": float(np.median(ranks)) if ranks else 0.0,
        }


def clamp_token_ids(
    readout, meta: dict, emotion: str, saved_k: str, count: int
) -> tuple[list[int], list[str]]:
    """The token set whose read directions define the emotion's J-space.

    The emotion word's own single-token variants first, then the atoms Phase 6's ``v_J``
    actually selected, best-first by coefficient. Those are not arbitrary: they are the
    tokens the decomposition says this vector is disposed to say, so they are the concept
    as the lens sees it rather than as a thesaurus sees it.

    Missing own-word variants is a real finding about the vocabulary rather than an error
    -- the J-lens is single-token -- so it is recorded and the atom tokens carry the set.
    """
    ids: list[int] = []
    variants = readout.single_token_variants(emotion)
    ids.extend(int(i) for i in variants.values())

    row = next(
        (r for r in meta.get("per_emotion", []) if r.get("emotion") == emotion), {}
    )
    at_k = row.get("per_k", {}).get(saved_k, {})
    for token_id in at_k.get("token_ids", []) or []:
        if int(token_id) not in ids:
            ids.append(int(token_id))
    ids = ids[:count]
    return ids, [readout.tokenizer.decode([i]) for i in ids]


def build_clamp_basis(
    readout, emotion: str, token_ids: list[int], tokens: list[str],
    blocks: list[int], head: np.ndarray, gain: np.ndarray | None,
) -> ClampBasis:
    """One orthonormal basis per block for ``span{J_l^T (g * w_t)}``.

    Per block, because each block has its own ``J_l`` and therefore its own read
    directions: the emotion's J-space is not one subspace transported around, it is a
    different subspace at every layer. Clamping with a single block's basis everywhere
    would hold the wrong directions fixed at 62 of the 63 layers.

    Orthonormalised by SVD with a relative floor, not by Gram-Schmidt: the read
    directions of related tokens are correlated -- that is what makes them a concept --
    so the stack is rank-deficient, and a basis built without a floor would spend its
    last columns on numerical noise and clamp arbitrary directions.
    """
    import torch

    rows = head[token_ids]
    if gain is not None:
        rows = rows * gain[None, :]

    bases: dict[int, np.ndarray] = {}
    ranks: dict[int, int] = {}
    for block in blocks:
        jacobian = torch.as_tensor(
            readout.lens.jacobians[block], dtype=torch.float32
        ).cpu().numpy()
        directions = rows @ jacobian          # (m, d): row i is J_l^T (g * w_{t_i})
        left, singular, _ = np.linalg.svd(directions.T, full_matrices=False)
        keep = max(int((singular > BASIS_RANK_RCOND * singular[0]).sum()), 1)
        bases[block] = np.ascontiguousarray(left[:, :keep], dtype=np.float32)
        ranks[block] = keep
    return ClampBasis(
        emotion=emotion, token_ids=list(token_ids), tokens=list(tokens),
        bases=bases, ranks=ranks,
    )


def residual_share(basis: ClampBasis, hidden: dict[int, np.ndarray]) -> dict:
    """Fraction of the clean residual's squared norm inside the clamped subspace.

    The blunt collateral number. The clamp holds ``span(A_l)`` fixed, so this is
    literally how much of the model it is holding still. A concept should be a small
    share; a large one means the "clamp" is a lobotomy and any downstream null is
    explained by that rather than by the workspace.
    """
    shares: dict[int, float] = {}
    for block, matrix in basis.bases.items():
        residual = hidden.get(block)
        if residual is None:
            continue
        inside = residual @ matrix
        total = float((residual * residual).sum())
        shares[block] = float((inside * inside).sum() / total) if total > 0 else 0.0
    if not shares:
        return {"n_blocks": 0}
    values = np.asarray(list(shares.values()))
    return {
        "n_blocks": len(shares),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "per_block": {str(b): shares[b] for b in sorted(shares)},
    }


# --------------------------------------------------------------------------- #
# The clamp, and paired generation
# --------------------------------------------------------------------------- #

class Clamp:
    """Hooks that capture clean J-coordinates and hold a steered run at them.

    One instance wraps one model and one basis. ``capture()`` and ``apply()`` are
    context managers over the *same* hook set, switched by mode, so the two passes cannot
    drift apart in which blocks they touch -- a capture on 63 blocks and an apply on 62
    would silently leave one layer free, which is precisely the hole this phase exists to
    close.
    """

    def __init__(self, model, basis: ClampBasis):
        import torch

        self.model = model
        self.basis = basis
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is None:
            raise SystemExit(
                "cannot reach model.model.layers to clamp: this architecture does not "
                "expose the block list the lens convention indexes."
            )
        missing = [b for b in basis.blocks if b >= len(layers)]
        if missing:
            raise SystemExit(f"blocks {missing} are beyond model.model.layers")
        self.layers = layers
        # Host copies; moved to each block's own device once, on first use. Re-sending a
        # (d, rank) matrix per forward call would be ~0.5 MB x 63 blocks x 300 steps of
        # pure PCIe traffic per prompt, which would dominate the phase.
        self._host = {
            block: torch.as_tensor(matrix) for block, matrix in basis.bases.items()
        }
        self._device: dict[int, "torch.Tensor"] = {}
        self.captured: dict[int, "torch.Tensor"] = {}
        self.targets: dict[int, "torch.Tensor"] = {}
        self._deviation: "torch.Tensor | None" = None
        self._mode = "off"
        self._handles: list = []

    def _matrix(self, block: int, like):
        """The block's basis, resident on the residual's device in fp32."""
        import torch

        cached = self._device.get(block)
        if cached is None or cached.device != like.device:
            cached = self._host[block].to(device=like.device, dtype=torch.float32)
            self._device[block] = cached
        return cached

    @property
    def deviation(self) -> float:
        """Worst |landed - target| since :meth:`apply` was entered. Syncs once, here."""
        return 0.0 if self._deviation is None else float(self._deviation)

    def _hook(self, block: int):
        import torch

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            matrix = self._matrix(block, hidden)
            flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
            coordinates = flat @ matrix
            if self._mode == "capture":
                self.captured[block] = coordinates.detach()
                return output
            target = self.targets.get(block)
            if target is None:
                return output
            target = target.to(device=coordinates.device, dtype=coordinates.dtype)
            steered = (flat + (target - coordinates) @ matrix.T).to(hidden.dtype)
            # How far the coordinate actually LANDS from its target. Accumulated on device
            # and read once, in `deviation`: a .item() per block per step would be ~19k
            # GPU syncs per prompt. The arithmetic is exact in fp32 but the stream is
            # bf16, so the no-op check has to be told what rounding costs.
            worst = (steered.to(torch.float32) @ matrix - target).abs().max()
            self._deviation = (
                worst if self._deviation is None
                else torch.maximum(self._deviation, worst)
            )
            steered = steered.reshape(hidden.shape)
            return (steered,) + tuple(output[1:]) if is_tuple else steered

        return hook

    @contextlib.contextmanager
    def _hooked(self, mode: str):
        self._mode = mode
        self._handles = [
            self.layers[block].register_forward_hook(self._hook(block))
            for block in self.basis.blocks
        ]
        try:
            yield self
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            self._mode = "off"

    def capture(self):
        """Record ``A_l^T h`` at every clamped block, changing nothing."""
        self.captured = {}
        return self._hooked("capture")

    def apply(self):
        """Hold ``A_l^T h`` at :attr:`targets` at every clamped block."""
        return self._hooked("apply")

    def reset_deviation(self) -> None:
        self._deviation = None


@dataclass
class Generation:
    """One completion, plus what the paired run measured while producing it."""

    text: str
    n_new_tokens: int
    clamp_deviation: float
    finished: bool = True

    @property
    def truncated(self) -> bool:
        return not self.finished


def paired_generate(
    model, tokenizer, config: PCAJLensConfig, prompt: str, block: int,
    vector: np.ndarray | None, alpha: float, clamp: Clamp | None,
) -> Generation:
    """Generate one completion, clamping J-coordinates to a paired clean pass.

    The loop, which is the whole idea:

    1. Prefill the **clean** model on the prompt, capturing ``A_l^T h`` at every clamped
       block and position.
    2. Prefill the **steered** model on the same prompt with the clamp applying those
       coordinates, and take its next token.
    3. Advance the clean model over *that token* -- the steered run's choice, not its own
       -- capturing the new position's coordinates.
    4. Advance the steered model over the same token with the clamp applying them.
    5. Repeat.

    Step 3 is the load-bearing one. The clean pass never generates; it is driven by the
    steered run's token stream, so its coordinates always answer "on this exact prefix,
    without the perturbation" rather than "in some other sentence at the same offset".

    Greedy, one prompt at a time, both by design -- see the module docstring.
    """
    import torch

    device = model_utils.model_input_device(model)
    text = model_utils.prepare_texts(
        [prompt], tokenizer, use_chat_template=True, chat_add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
    )[0]
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    def forward(input_ids, cache, clamped: bool):
        """One steered forward pass, clamped after the steering at the shared block.

        **Order matters and is not the obvious one.** PyTorch runs forward hooks in
        registration order, so the steering context has to be entered *first* for the
        clamp to run *second*. At the one block that carries both, the residual the clamp
        holds must be the post-steering residual -- otherwise the steering vector's own
        J-component slips through unclamped at exactly the block it was injected, and
        the clamp only catches it from the next layer up.
        """
        with p8.steering(model, block, vector, alpha, config.steer_positions):
            if not clamped or clamp is None:
                return model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            with clamp.apply():
                return model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    def advance_clean(input_ids, cache):
        """Advance the unsteered run and make its new coordinates the clamp target."""
        with clamp.capture():
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        clamp.targets = dict(clamp.captured)
        return out.past_key_values

    generated: list[int] = []
    finished = False
    clean_cache = None
    steered_cache = None
    with torch.inference_mode():
        if clamp is not None:
            clean_cache = advance_clean(ids, clean_cache)
        out = forward(ids, steered_cache, clamped=clamp is not None)
        steered_cache = out.past_key_values

        for index in range(config.generation_max_new_tokens):
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            # Stop before the forward pass whose logits nobody will read. Trivial for one
            # prompt and not for a phase that pays two passes per step.
            if token == tokenizer.eos_token_id:
                finished = True
                break
            if index == config.generation_max_new_tokens - 1:
                break
            step = torch.tensor([[token]], device=device)
            if clamp is not None:
                # The clean run follows the STEERED run's token, never its own. This is
                # the line that makes the clamp target a counterfactual rather than a
                # coordinate from some other sentence at the same offset.
                clean_cache = advance_clean(step, clean_cache)
            out = forward(step, steered_cache, clamped=clamp is not None)
            steered_cache = out.past_key_values

    return Generation(
        text=tokenizer.decode(generated, skip_special_tokens=True),
        n_new_tokens=len(generated),
        clamp_deviation=0.0 if clamp is None else clamp.deviation,
        finished=finished,
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def control_token_ids(vocab_size: int, exclude: list[int], rng, n: int) -> list[int]:
    """Vocabulary ids outside the clamped set, for the collateral check."""
    excluded = set(int(i) for i in exclude)
    pool = rng.permutation(vocab_size)
    return [int(i) for i in pool if int(i) not in excluded][:n]


def collateral(
    model, tokenizer, config: PCAJLensConfig, readout, block: int,
    vector: np.ndarray, alpha: float, clamp: Clamp, controls: list[int], prompt: str,
) -> dict:
    """Does the clamp leave unrelated J-space content intact?

    Compares the lens readout over **control tokens** -- ids outside the clamped set --
    between the steered-unclamped and steered-clamped runs, on the same prompt, at the
    same strength. Spearman rather than Pearson: what matters is whether the model's
    ordering over unrelated vocabulary survived, not whether the logits kept their scale,
    and the clamp changes the residual's norm by construction.

    Measured at ``alpha > 0`` on purpose. At zero strength the clamp is the identity and
    the correlation would be 1 by arithmetic, which would look like a passing check while
    testing nothing.
    """
    import torch

    device = model_utils.model_input_device(model)
    text = model_utils.prepare_texts(
        [prompt], tokenizer, use_chat_template=True, chat_add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
    )[0]
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    hidden_state = jlens_lens.hidden_state_index(block)

    def final_residual(clamped: bool) -> np.ndarray:
        with torch.inference_mode():
            if not clamped:
                with p8.steering(model, block, vector, alpha, config.steer_positions):
                    out = model(input_ids=ids, output_hidden_states=True)
                return out.hidden_states[hidden_state][0, -1].float().cpu().numpy()
            with clamp.capture():
                model(input_ids=ids)
            clamp.targets = dict(clamp.captured)
            with clamp.apply():
                with p8.steering(model, block, vector, alpha, config.steer_positions):
                    out = model(input_ids=ids, output_hidden_states=True)
            return out.hidden_states[hidden_state][0, -1].float().cpu().numpy()

    unclamped = readout.direction_logits(final_residual(False), block).numpy()
    clamped = readout.direction_logits(final_residual(True), block).numpy()
    index = np.asarray(controls, dtype=np.int64)
    spearman = _spearman(unclamped[index], clamped[index])
    return {
        "n_control_tokens": len(controls),
        "spearman": spearman,
        "disturbance": 1.0 - spearman,
        "ceiling": config.clamp_max_collateral,
        "holds": bool(1.0 - spearman <= config.clamp_max_collateral),
        "alpha": alpha,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what makes this Spearman and not a guess.

    ``argsort(argsort(x))`` gives tied values *arbitrary distinct* ranks, and here the
    tie that matters is the one a broken clamp produces: a readout flattened to a
    constant would get ranks ``0..n-1`` and correlate spuriously with anything, so the
    collateral check would pass exactly when it should fail.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in scipy for one number.

    Zero when either side has no rank variance -- an all-tied readout has no ordering to
    correlate, and calling that 0 rather than 1 is what makes the collateral check able
    to fail.
    """
    rank_a = _rank(np.asarray(a, dtype=np.float64))
    rank_b = _rank(np.asarray(b, dtype=np.float64))
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.linalg.norm(rank_a) * np.linalg.norm(rank_b))
    return float(rank_a @ rank_b / denominator) if denominator > 0 else 0.0


def suppression(baseline: float | None, steered: float | None,
                clamped: float | None, floor: float) -> dict:
    """How much of the ``v``-induced report lift the clamp removed.

    ``(steered - clamped) / (steered - baseline)``. Undefined when steering did not raise
    the report channel in the first place, and that is reported as undefined rather than
    as a pass: with no lift to remove, the clamp has not been shown to do anything, and
    the decisive cell has no license.
    """
    if baseline is None or steered is None or clamped is None:
        return {"defined": False, "reason": "a report score is missing"}
    lift = steered - baseline
    if lift <= 0:
        return {
            "defined": False,
            "reason": f"steering with v did not raise the report channel "
                      f"(baseline {baseline:.2f} -> {steered:.2f}), so there is no lift "
                      "for the clamp to remove",
            "baseline": baseline, "steered": steered, "clamped": clamped,
        }
    removed = (steered - clamped) / lift
    return {
        "defined": True,
        "baseline": baseline, "steered": steered, "clamped": clamped,
        "lift": lift, "removed": float(removed), "floor": floor,
        "holds": bool(removed >= floor),
    }


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

def run_cells(
    model, tokenizer, config: PCAJLensConfig, block: int,
    concept: p8.Directions, clamp: Clamp, strengths: list[float],
) -> list[dict]:
    """Generate every (condition, clamp, strength) cell for one emotion.

    ``alpha = 0`` is generated twice, not once: unclamped it is the plain baseline, and
    clamped it is the no-op check. Those are the same thing *if the implementation is
    right*, which is exactly what makes running both worth one extra cell.
    """
    report_prompts = list(REPORT_PROMPTS)
    behaviour_prompts = [task.prompt for task in BEHAVIOUR_TASKS]
    rows: list[dict] = []

    cells: list[tuple[str, bool, float, str]] = []
    for alpha in strengths:
        if alpha == 0.0:
            cells.append((CONDITIONS[0], False, 0.0, ROLE_BASELINE))
            cells.append((CONDITIONS[0], True, 0.0, ROLE_NOOP))
            continue
        for condition in CONDITIONS:
            for clamped in (False, True):
                cells.append((condition, clamped, alpha, ROLE_CELL))

    for condition, clamped, alpha, role in cells:
        vector = concept.vectors[condition]
        clamp.reset_deviation()
        t0 = time.time()
        generations: list[Generation] = []
        for prompt in report_prompts + behaviour_prompts:
            generations.append(paired_generate(
                model, tokenizer, config, prompt, block, vector, alpha,
                clamp if clamped else None,
            ))
        # Outside every hook: fluency is judged by the unmodified model, or it measures
        # the perturbation rather than the text.
        perplexities = p8.perplexity(
            model, tokenizer, config, [g.text for g in generations]
        )
        common = {
            "concept": concept.name, "kind": concept.kind,
            "report_emotion": concept.report_emotion,
            "condition": cell_name(condition, clamped),
            "base_condition": condition, "clamped": clamped, "alpha": alpha,
            "role": role, "shared_baseline": role == ROLE_BASELINE,
        }
        for i, prompt in enumerate(report_prompts):
            rows.append({**common, "channel": "report", "family": "report",
                         "prompt": prompt, "response": generations[i].text,
                         "finished": generations[i].finished,
                         "n_new_tokens": generations[i].n_new_tokens,
                         "perplexity": perplexities[i],
                         "clamp_deviation": generations[i].clamp_deviation})
        for j, task in enumerate(BEHAVIOUR_TASKS):
            index = len(report_prompts) + j
            rows.append({**common, "channel": "behaviour", "family": task.family,
                         "prompt": task.prompt, "response": generations[index].text,
                         "finished": generations[index].finished,
                         "n_new_tokens": generations[index].n_new_tokens,
                         "perplexity": perplexities[index],
                         "clamp_deviation": generations[index].clamp_deviation})
        label = f"{concept.name}/{cell_name(condition, clamped)}/a={alpha:g}"
        worst = max((g.clamp_deviation for g in generations), default=0.0)
        done = sum(1 for g in generations if g.finished) / max(len(generations), 1)
        print(f"  {label:<38} {len(generations):>3} generations in "
              f"{time.time() - t0:>6.0f}s   {done:>4.0%} EOS"
              + (f"   worst |landed-target| {worst:.3g}" if clamped else "")
              + ("   [no-op check]" if role == ROLE_NOOP else ""), flush=True)
    return rows


def expand_baseline(rows: list[dict]) -> list[dict]:
    """Copy the alpha=0 unclamped rows to every cell name, and drop the no-op rows.

    Two things at once, both about keeping one baseline. Phase 8's shift-from-baseline
    arithmetic finds the alpha=0 row inside each (concept, family) group, so a second
    alpha=0 *condition* -- which the no-op cell is -- would leave it choosing between two
    baselines by position. The no-op rows stay in the generations CSV, where the check
    that needs them reads them; they are simply not a cell of the grid.
    """
    baseline = [r for r in rows if r.get("role") == ROLE_BASELINE]
    graded = [r for r in rows if r.get("role") != ROLE_NOOP]
    names = [cell_name(c, k) for c in CONDITIONS for k in (False, True)]
    expanded = [r for r in graded if r.get("role") != ROLE_BASELINE]
    for row in baseline:
        for name in names:
            expanded.append({**row, "condition": name})
    return expanded


def noop_check(rows: list[dict], config: PCAJLensConfig) -> dict:
    """Is the clamp the identity when there is nothing to clamp away?

    Two readings of the same claim. The numeric one -- how far the residual's coordinate
    lands from its target -- is the real check, since it holds per position and per block.
    The textual one is the consequence anyone would notice, and it is reported with the
    bf16 caveat rather than asserted: the correction is computed in fp32 and written back
    into a bf16 stream, so exact equality is not owed, and a single flipped token late in
    a greedy decode is rounding rather than a bug.
    """
    plain = {r["prompt"]: r for r in rows if r.get("role") == ROLE_BASELINE}
    clamped = [r for r in rows if r.get("role") == ROLE_NOOP]
    if not plain or not clamped:
        return {"checked": 0}
    identical = sum(
        1 for r in clamped if plain.get(r["prompt"], {}).get("response") == r["response"]
    )
    deviations = [float(r.get("clamp_deviation") or 0.0) for r in clamped]
    return {
        "checked": len(clamped),
        "identical_texts": identical,
        "frac_identical": identical / len(clamped),
        "max_coordinate_deviation": max(deviations) if deviations else 0.0,
        # bf16 has ~3 decimal digits; a coordinate on a residual of norm ~100 lands
        # within ~1e-1 at worst. Above that the correction is not being applied.
        "numeric_holds": bool(max(deviations) < 1.0) if deviations else False,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def print_design(
    config: PCAJLensConfig, emotions: list[str], block: int, fitted: list[int],
    blocks: list[int], bases: dict[str, ClampBasis], strengths: list[float],
    n_prompts: int, n_generations: int, decisive: bool,
) -> None:
    print(f"model       : {config.model_name} ({config.dtype})")
    print(f"emotions    : {emotions}")
    print(f"steer block : {block} (hidden state "
          f"{jlens_lens.hidden_state_index(block)}), from Phase 6's record")
    print(f"clamp blocks: {len(blocks)} of {len(fitted)} fitted "
          f"({blocks[0]}..{blocks[-1]})   spec={config.clamp_blocks!r}")
    if decisive:
        print("              every fitted block, which is what makes the cell decisive:")
        print("              re-entry is downstream layers re-deriving the concept, so a")
        print("              clamp that skips layers leaves it the room it needs.")
    else:
        print("              NOT DECISIVE -- a partial clamp cannot tell 'the effect")
        print("              bypassed the workspace' from 'it re-entered at an unclamped")
        print("              block'. Useful for layer attribution, not for the headline.")
    print(f"clamped span: span{{J_l^T (g * w_t)}} over {config.clamp_token_count} tokens "
          "max, per block")
    for name, basis in bases.items():
        info = basis.summary()
        print(f"              {name:<12} {info['n_tokens']:>2} tokens, rank "
              f"{info['rank_min']}-{info['rank_max']} "
              f"(median {info['rank_median']:.0f}) of d_model")
        print(f"                           {' '.join(repr(t) for t in info['tokens'][:8])}")
    print(f"conditions  : {len(CONDITIONS)} x {{unclamped, clamped}}")
    for condition in CONDITIONS:
        for clamped in (False, True):
            print(f"                {CELL_LABELS[cell_name(condition, clamped)]}")
    print(f"strengths   : {strengths}  as multiples of ||v||")
    print(f"prompts     : {n_prompts} per cell "
          f"({len(REPORT_PROMPTS)} report + {len(BEHAVIOUR_TASKS)} behaviour)")
    print()
    print(f"generations : {n_generations:,} at {config.generation_max_new_tokens} new "
          "tokens, ONE prompt at a time")
    print("              batch 1 and two forward passes per decode step: the clean and")
    print("              steered runs need separate KV caches advanced in lockstep over")
    print("              the same tokens. That is the price of a defined clamp target.")
    hours = (
        n_generations * config.generation_max_new_tokens * 2 / TOKENS_PER_SECOND / 3600
    )
    print(f"              ~{hours:.1f} h at ~{TOKENS_PER_SECOND:.0f} tok/s single-stream "
          "-- arithmetic, not a measurement")


def print_verification(
    noop: dict, suppress: dict, collateral_result: dict, shares: dict,
    config: PCAJLensConfig,
) -> bool:
    """The three checks, before any behavioural number. Returns whether to interpret."""
    print()
    print(RULE)
    print("VERIFICATION -- the clamp has to be shown to work before it proves anything")
    print(RULE)

    numeric = noop.get("numeric_holds", False)
    print(f"1. no-op at alpha=0        : "
          f"{'PASS' if numeric else 'FAIL'}   coordinates land within "
          f"{noop.get('max_coordinate_deviation', float('nan')):.4g} of target")
    print(f"                             text identical for "
          f"{noop.get('identical_texts', 0)}/{noop.get('checked', 0)} prompts")
    if numeric and noop.get("frac_identical", 1.0) < 1.0:
        print("                             the correction is fp32 written into a bf16")
        print("                             stream, so a late token flip is rounding")
        print("                             rather than a bug -- the numeric check is")
        print("                             the one that gates.")
    if not numeric:
        print("                             the correction is NOT landing on the target.")
        print("                             That is an implementation fault, not a")
        print("                             finding: nothing below can be read.")

    if suppress.get("defined"):
        holds = suppress["holds"]
        print(f"2. report suppression      : {'PASS' if holds else 'FAIL'}   "
              f"{suppress['removed']:.0%} of the v-induced lift removed "
              f"(floor {suppress['floor']:.0%})")
        print(f"                             baseline {suppress['baseline']:.2f} -> "
              f"v {suppress['steered']:.2f} -> v+clamp {suppress['clamped']:.2f}")
    else:
        holds = False
        print("2. report suppression      : UNDEFINED")
        print(f"                             {suppress.get('reason', '')}")

    collateral_ok = collateral_result.get("holds", False)
    if not collateral_result:
        print("3. unrelated J-space intact : NOT MEASURED (no non-zero strength ran)")
    else:
        print(f"3. unrelated J-space intact : "
              f"{'PASS' if collateral_ok else 'FAIL'}   control-token readout Spearman "
              f"{collateral_result['spearman']:.4f}")
        print(f"                             disturbance "
              f"{collateral_result['disturbance']:.4f} vs ceiling "
              f"{config.clamp_max_collateral:g} over "
              f"{collateral_result['n_control_tokens']} tokens, at alpha="
              f"{collateral_result['alpha']:g}")
    if shares.get("n_blocks"):
        print(f"                             clamped subspace holds "
              f"{shares['mean']:.1%} of the clean residual's variance on average, "
              f"{shares['max']:.1%} at worst")
        if shares["max"] > 0.5:
            print("                             OVER HALF at some block: this is not a")
            print("                             concept being held fixed, it is most of")
            print("                             the model. Lower clamp_token_count.")

    interpretable = bool(numeric and holds and collateral_ok)
    print()
    print(f"  clamp verified: {interpretable}")
    if not interpretable:
        print("  The decisive cell below is NOT interpretable. Which check failed says")
        print("  what to do: a no-op failure is a bug, a suppression failure means the")
        print("  clamped span misses how the concept re-enters (raise clamp_token_count),")
        print("  and a collateral failure means the span is too large (lower it). The")
        print("  numbers are still printed, because a run you cannot interpret is still a")
        print("  run you should be able to diagnose.")
    return interpretable


def print_grid(table: pd.DataFrame, summary: pd.DataFrame, concept: str) -> None:
    print()
    print(THIN)
    print(f"{concept}")
    print(THIN)
    subset = summary[summary["concept"] == concept]
    strengths = sorted(s for s in subset["alpha"].unique() if s > 0)
    order = [cell_name(c, k) for c in CONDITIONS for k in (False, True)]
    header = "    " + f"{'cell':<18}" + "".join(f"{f'a={a:g}':>10}" for a in strengths)

    print("  shift from the alpha=0 baseline in grid-SD units (mean |z| over families)")
    for channel in ("report", "behaviour"):
        print(f"    {channel} channel")
        print(header)
        for cell in order:
            rows = subset[(subset["channel"] == channel) & (subset["condition"] == cell)]
            line = f"    {cell:<18}"
            for alpha in strengths:
                at = rows[rows["alpha"] == alpha]
                if at.empty or pd.isna(at["abs_z"].iloc[0]):
                    line += f"{'-':>10}"
                    continue
                mark = "*" if bool(at["degraded"].iloc[0]) else " "
                line += f"{at['abs_z'].iloc[0]:>9.2f}{mark}"
            print(line)
    print()
    print("  raw scores per family")
    for family, note in p8.FAMILY_NOTES:
        rows = table[(table["concept"] == concept) & (table["family"] == family)]
        if rows.empty or rows["n_scored"].sum() == 0:
            continue
        print(f"    {family}  --  {note}")
        print(header)
        for cell in order:
            at_cell = rows[rows["condition"] == cell]
            line = f"    {cell:<18}"
            for alpha in strengths:
                at = at_cell[at_cell["alpha"] == alpha]
                if at.empty or pd.isna(at["score"].iloc[0]):
                    line += f"{'-':>10}"
                    continue
                mark = "*" if bool(at["degraded"].iloc[0]) else " "
                line += f"{at['score'].iloc[0]:>9.2f}{mark}"
            print(line)


def print_verdict(
    summary: pd.DataFrame, config: PCAJLensConfig, emotions: list[str],
    interpretable: bool, decisive: bool, phase8: dict, artifacts: dict,
) -> None:
    print()
    print(RULE)
    print("PHASE 9 VERDICT -- the decisive cell")
    print(RULE)

    def largest(concept: str, cell: str, channel: str) -> float | None:
        values = summary.loc[
            (summary["concept"] == concept) & (summary["condition"] == cell)
            & (summary["channel"] == channel) & (summary["alpha"] > 0)
            & (~summary["degraded"]), "abs_z"
        ].dropna()
        return None if values.empty else float(values.max())

    for concept in emotions:
        print(f"  {concept}   largest |z| over the undegraded strengths")
        for cell in (cell_name(c, k) for c in CONDITIONS for k in (False, True)):
            parts = []
            for channel in ("report", "behaviour"):
                value = largest(concept, cell, channel)
                parts.append(
                    f"{channel} " + ("   n/a" if value is None else f"{value:6.2f}")
                )
            print(f"    {cell:<18} {' | '.join(parts)}")
        behaviour = largest(concept, cell_name("v_remainder", True), "behaviour")
        report = largest(concept, cell_name("v_remainder", True), "report")
        unclamped = largest(concept, cell_name("v_remainder", False), "behaviour")
        if not interpretable:
            print("    NOT INTERPRETABLE -- the clamp did not verify. See above.")
        elif behaviour is None or report is None:
            print("    the decisive cell was not scored in both channels, so it cannot "
                  "be read")
        else:
            print(f"    DECISIVE CELL: v_perp + clamp -> behaviour {behaviour:.2f}, "
                  f"report {report:.2f}")
            if unclamped is not None:
                print(f"      against v_perp unclamped -> behaviour {unclamped:.2f}: "
                      f"{'survives' if behaviour >= 0.5 * unclamped else 'collapses'} "
                      "the clamp")
            print("      Behaviour moving while report stays suppressed is the result:")
            print("      an emotional state steering action without being reportable,")
            print("      with re-entry ruled out. Behaviour collapsing with the clamp is")
            print("      the opposite finding -- the Phase 8 effect WAS routed through")
            print("      the workspace -- and it is just as publishable.")

    print()
    print(f"  clamp verified   : {interpretable}")
    print(f"  clamp decisive   : {decisive}"
          + ("" if decisive else "   (clamp_blocks is not 'all')"))
    print(f"  phase 8 v_perp   : {phase8.get('note', 'not summarised')}")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<9}: {path}")
    print()
    print(RULE)
    print("  WHAT THIS STILL DOES NOT SETTLE")
    print(RULE)
    print("  The clamp holds the emotion's J-space coordinates over a token set. A")
    print("  concept the lens reads through tokens outside that set can still re-enter,")
    print("  and clamp_token_count is where that judgement was made -- the tokens are")
    print("  printed above so the set can be argued with.")
    print()
    print("  v_perp remains the residual of a k-sparse nonnegative code, whose reachable")
    print("  set is a union of cones rather than a linear subspace. It means 'missed by")
    print("  that approximation at that k and pool', never 'intrinsically")
    print("  unverbalizable'. A clamped behavioural effect is an effect of a direction")
    print("  the sparse code did not capture, which is narrower than it sounds.")
    print()
    print("  And the two limits that bound every phase from 7 on: the steering vectors")
    print("  were built from raw story text and applied to chat-formatted prompts, and a")
    print("  lens readout is a disposition to say a word, so the report channel measures")
    print("  reportability rather than felt experience.")
    print()
    print("This is the last phase. There is no Phase 10 to defer anything to.")
    print(RULE)


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #

def read_phase8(config: PCAJLensConfig) -> tuple[dict, Path, dict]:
    """Phase 8's gate record, plus whether there is a result for Phase 9 to explain.

    Phase 9 is a control, and a control for nothing is hours of GPU time. So the
    precondition is not merely "Phase 8 ran" but "Phase 8 found a ``v_perp`` behavioural
    movement that survives its own fluency check" -- which is exactly the claim the clamp
    exists to test.
    """
    gate = config.phase_dir / "phase8_steering" / "phase8_gate.json"
    if not gate.exists():
        raise SystemExit(
            f"no Phase 8 record at\n  {gate}\n\n"
            "Phase 9 is the control for Phase 8's v_perp result. Run it first:\n\n"
            "  python run.py phase8\n"
        )
    record = json.loads(gate.read_text(encoding="utf-8"))
    if record.get("thinking", {}).get("resolved", "on") != "off":
        raise SystemExit(
            f"{gate} was produced with thinking mode "
            f"{record.get('thinking', {}).get('resolved', 'unrecorded')}.\n\n"
            "Its v_perp effect is then a measurement over truncated reasoning traces, so "
            "there is\nnothing here for the clamp to explain. Re-run Phase 8 with "
            "enable_thinking=false (the\ndefault):\n\n  python run.py phase8\n"
        )
    frame = pd.DataFrame(record.get("summary", []))
    finding: dict = {"note": "Phase 8's summary is empty", "has_effect": False}
    if not frame.empty:
        rows = frame[
            (frame["condition"] == "v_remainder") & (frame["channel"] == "behaviour")
            & (frame["alpha"] > 0) & (~frame["degraded"].astype(bool))
        ]
        values = rows["abs_z"].dropna() if "abs_z" in rows else pd.Series(dtype=float)
        best = float(values.max()) if not values.empty else 0.0
        random_rows = frame[
            (frame["condition"] == "v_random") & (frame["channel"] == "behaviour")
            & (frame["alpha"] > 0) & (~frame["degraded"].astype(bool))
        ]
        random_values = (
            random_rows["abs_z"].dropna() if "abs_z" in random_rows
            else pd.Series(dtype=float)
        )
        random_best = float(random_values.max()) if not random_values.empty else 0.0
        finding = {
            "v_perp_behaviour_z": best,
            "random_behaviour_z": random_best,
            "has_effect": bool(best > 0 and best > random_best),
            "note": f"v_perp behaviour |z| {best:.2f} vs random {random_best:.2f}",
        }
    return record, gate, finding


def read_phase6(config: PCAJLensConfig) -> dict:
    """Phase 6's sidecar, refusing the ``write_space`` ablation."""
    path = config.decomposition_meta_path
    if not path.exists():
        raise SystemExit(
            f"no decomposition metadata at\n  {path}\n\n"
            "Phase 9 clamps the read directions of the tokens Phase 6's v_J selected. "
            "Run it\nfirst:\n\n  python run.py phase6\n"
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("write_space"):
        raise SystemExit(
            f"{path} was written by a write_space run.\n\n"
            "Its v_J selected the tokens a vector most efficiently WRITES, not the ones "
            "the lens\nreads it with, so clamping their read directions would clamp the "
            "wrong subspace.\nRe-run Phase 6 without the ablation:\n\n"
            "  python run.py phase6\n"
        )
    return meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    out_dir = config.phase_dir / "phase9_clamp"
    cache_dir = paths.hf_cache_dir()
    rng = rng_for(config.seed, "phase9")

    print(RULE)
    print(f"PHASE 9 GATE -- the re-entry clamp   run '{config.run_name}'")
    print(RULE)

    record, record_path, finding = read_phase8(config)
    meta = read_phase6(config)
    emotions = (
        list(config.channel_emotions) if config.channel_emotions
        else list(record.get("phase7", {}).get("emotions", []))
    )
    if not emotions:
        raise SystemExit(f"{record_path} names no emotions; re-run Phase 8.")
    directions, block = p8.load_emotion_directions(config, emotions)
    strengths = [float(a) for a in config.clamp_strengths]

    print(f"phase 8     : {record_path.name}")
    print(f"              {finding['note']}")
    if not finding["has_effect"]:
        print()
        print("There is no v_perp behavioural effect in Phase 8's record for this clamp",
              file=sys.stderr)
        print("to explain. Phase 9 is a control, and a control for nothing costs hours",
              file=sys.stderr)
        print("of GPU time to confirm a null that Phase 8 already reported. Read Phase",
              file=sys.stderr)
        print("8's grid first; if its v_perp column did move, re-check whether those",
              file=sys.stderr)
        print("cells were marked degraded.", file=sys.stderr)
        return 3

    # --- the clamp's shape, before any weights --------------------------------- #
    from emotion_pca_jlens.phase4_lens_pcs import resolve_lens

    lens_path, lens_report = resolve_lens(config, cache_dir)
    description = jlens_lens.describe_lens_checkpoint(lens_path)
    fitted = list(description.source_layers)
    blocks = resolve_block_spec(config.clamp_blocks, fitted)
    decisive = set(blocks) == set(fitted)
    n_prompts = len(REPORT_PROMPTS) + len(BEHAVIOUR_TASKS)
    n_cells = 2 + (len(strengths) - 1) * len(CONDITIONS) * 2
    n_generations = len(directions) * n_cells * n_prompts

    sections: dict = {
        "run": {"stage": "phase9_clamp", "run_name": config.run_name,
                "dry_run": args.dry_run, "verify_only": args.verify_only,
                "no_judge": args.no_judge, "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "phase8": {"record": str(record_path), **finding},
        "phase6": {"saved_k": meta.get("saved_k"), "atom_mode": meta.get("atom_mode")},
        "lens": lens_report,
        "design": {
            "steer_block": block, "clamp_blocks": blocks,
            "n_fitted_blocks": len(fitted), "decisive": decisive,
            "conditions": [cell_name(c, k) for c in CONDITIONS for k in (False, True)],
            "strengths": strengths, "prompts_per_cell": n_prompts,
            "cells_per_emotion": n_cells, "n_generations": n_generations,
        },
    }

    if args.dry_run:
        print(f"model       : {config.model_name} ({config.dtype})")
        print(f"emotions    : {emotions}")
        print(f"steer block : {block}")
        print(f"clamp blocks: {len(blocks)} of {len(fitted)} fitted "
              f"({blocks[0]}..{blocks[-1]})   decisive={decisive}")
        print(f"strengths   : {strengths}")
        print(f"cells       : {n_cells} per emotion, {n_generations:,} generations")
        print("              the token set and the subspace rank need the lens's")
        print("              unembedding, so they are only available with weights loaded.")
        hours = (n_generations * config.generation_max_new_tokens * 2
                 / TOKENS_PER_SECOND / 3600)
        print(f"              ~{hours:.1f} h at ~{TOKENS_PER_SECOND:.0f} tok/s "
              "single-stream, two passes per step")
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run", title=f"PHASE 9 DRY RUN -- {config.run_name}",
            sections=sections, txt_name="phase9_dry_run.txt",
            json_name="phase9_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: preconditions checked, cost estimated, no weights.")
        print(f"  records : {txt_path}")
        print(f"            {json_path}")
        print()
        print("Verify the clamp before paying for the grid:")
        print("  python run.py phase9 --verify-only")
        print(RULE)
        return 0

    # --- load ------------------------------------------------------------------ #
    print()
    print(RULE)
    print(f"Loading {config.model_name} ...")
    print(RULE)
    t0 = time.time()
    tokenizer = model_utils.load_tokenizer(
        config.model_name, config.model_revision, cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    model = model_utils.load_model(
        config.model_name, revision=config.model_revision, cache_dir=cache_dir,
        dtype=config.dtype, device_map=config.device_map,
        quantization=config.quantization,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    print(f"  loaded in {time.time() - t0:.0f}s")
    thinking = model_utils.thinking_flag_effect(tokenizer)
    resolved_thinking = (
        "off" if not config.enable_thinking and thinking["supported"]
        else "on" if thinking["supported"] else "unsupported"
    )
    print(f"  thinking mode  : {resolved_thinking.upper()}   (requested "
          f"enable_thinking={config.enable_thinking}; template responds: "
          f"{thinking['supported']})")
    if resolved_thinking != "off":
        print(f"    {thinking.get('reason') or 'thinking is ON'} -- the clamp would be")
        print("    holding coordinates over a reasoning trace rather than an answer.")
    readout = jlens_lens.LensReadout.build(model, tokenizer, lens_path)
    from emotion_pca_jlens.phase6_decompose import unembed_parts

    head, gain = unembed_parts(readout)

    print()
    print(RULE)
    print("STEP 1  Build the clamped subspace, one basis per block")
    print(RULE)
    saved_k = str(meta.get("saved_k", config.n_dict_atoms))
    bases: dict[str, ClampBasis] = {}
    for concept in directions:
        t0 = time.time()
        token_ids, tokens = clamp_token_ids(
            readout, meta, concept.name, saved_k, config.clamp_token_count
        )
        if not token_ids:
            raise SystemExit(
                f"no clamp tokens for {concept.name!r}: neither a single-token variant of "
                "the word\nnor any v_J atom token was available. Phase 6's sidecar may "
                "predate token_ids\nbeing recorded per k -- re-run Phase 6."
            )
        bases[concept.name] = build_clamp_basis(
            readout, concept.name, token_ids, tokens, blocks, head, gain
        )
        info = bases[concept.name].summary()
        print(f"  {concept.name:<14} {info['n_tokens']:>2} tokens -> rank "
              f"{info['rank_min']}-{info['rank_max']} per block, "
              f"{len(blocks)} blocks in {time.time() - t0:.0f}s", flush=True)
        print(f"                 {' '.join(repr(t) for t in tokens[:10])}")

    print()
    print_design(config, emotions, block, fitted, blocks, bases, strengths,
                 n_prompts, n_generations, decisive)

    # --- generate -------------------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 2  Generate, clamping to a paired clean pass")
    print(RULE)
    print("Two forward passes per decode step: the clean run is advanced over the")
    print("STEERED run's tokens, so its coordinates answer 'on this exact prefix, without")
    print("the perturbation'. One prompt at a time -- see the module docstring.")
    grid_t0 = time.time()
    rows: list[dict] = []
    collateral_results: dict[str, dict] = {}
    share_results: dict[str, dict] = {}
    nonzero = [a for a in strengths if a > 0]
    for concept in directions:
        clamp = Clamp(model, bases[concept.name])
        rows.extend(run_cells(model, tokenizer, config, block, concept, clamp, strengths))
        if nonzero:
            collateral_results[concept.name] = collateral(
                model, tokenizer, config, readout, block, concept.vectors["v"],
                nonzero[0], clamp,
                control_token_ids(head.shape[0], bases[concept.name].token_ids,
                                  rng, N_CONTROL_TOKENS),
                REPORT_PROMPTS[0],
            )
            share_results[concept.name] = _clean_residual_share(
                model, tokenizer, config, bases[concept.name], REPORT_PROMPTS[0]
            )
    print(f"  {len(rows):,} generations in {(time.time() - grid_t0) / 60:.1f} min")

    # --- score and verify ------------------------------------------------------ #
    print()
    print(RULE)
    print("STEP 3  Scoring")
    print(RULE)
    usage = p8.score_grid(rows, config, use_judge=not args.no_judge)

    frame = pd.DataFrame(rows)
    graded = pd.DataFrame(expand_baseline(rows))
    table = p8.family_table(graded, config)
    summary = p8.channel_summary(table)
    fluency = p8.cell_fluency(graded, config)

    primary = directions[0].name
    noop = noop_check([r for r in rows if r["concept"] == primary], config)
    reports = table[(table["family"] == "report") & (table["concept"] == primary)]

    def report_score(cell: str, alpha: float) -> float | None:
        at = reports[(reports["condition"] == cell) & (reports["alpha"] == alpha)]
        if at.empty or pd.isna(at["score"].iloc[0]):
            return None
        return float(at["score"].iloc[0])

    suppress = suppression(
        report_score(cell_name("v", False), 0.0),
        report_score(cell_name("v", False), nonzero[0]) if nonzero else None,
        report_score(cell_name("v", True), nonzero[0]) if nonzero else None,
        config.clamp_min_report_suppression,
    )
    interpretable = print_verification(
        noop, suppress, collateral_results.get(primary, {}),
        share_results.get(primary, {}), config,
    )

    if not args.verify_only:
        print()
        print(RULE)
        print("GATE  The decisive cell")
        print(RULE)
        print("Read the v_perp+clamp row against v_perp alone. Behaviour surviving the")
        print("clamp while report stays suppressed is the result; behaviour collapsing")
        print("with it means Phase 8's effect was routed through the workspace after all.")
        for concept in directions:
            print_grid(table, summary, concept.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "phase9_generations.csv"
    grid_path = out_dir / "phase9_grid.csv"
    frame.to_csv(generations_path, index=False)
    table.to_csv(grid_path, index=False)

    sections["clamp"] = {name: basis.summary() for name, basis in bases.items()}
    sections["verification"] = {
        "noop": noop, "suppression": suppress, "collateral": collateral_results,
        "residual_share": share_results, "interpretable": interpretable,
        "decisive": decisive,
    }
    sections["grid"] = table.to_dict(orient="records")
    sections["summary"] = summary.to_dict(orient="records")
    sections["fluency"] = fluency.to_dict(orient="records")
    sections["judge_usage"] = usage
    sections["invalid_rates"] = invalid_rates(rows)
    sections["completion"] = {
        "rate": sum(1 for r in rows if r.get("finished")) / max(len(rows), 1),
        "n": len(rows),
    }
    sections["thinking"] = {
        "requested": config.enable_thinking, "template_effect": thinking,
        "resolved": resolved_thinking,
    }
    sections["caveats"] = {
        "token_set": "the clamp holds the emotion's J-space over clamp_token_count "
                     "tokens; a concept the lens reads through tokens outside that set "
                     "can still re-enter",
        "v_perp_is_k_dependent": "v_perp is the residual of a k-sparse nonnegative code, "
                                 "a union of cones rather than a subspace, so it means "
                                 "'missed at this k and pool', never 'intrinsically "
                                 "unverbalizable'",
        "chat_template": "directions were built from raw story text and applied to "
                         "chat-formatted prompts; transfer is assumed, not verified",
        "reportability": "a lens readout is a disposition to say a word, so the report "
                         "channel measures reportability rather than felt experience",
    }
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 9 GATE -- {config.run_name}",
        sections=sections, txt_name="phase9_gate.txt", json_name="phase9_gate.json",
    )

    if args.verify_only:
        print()
        print(RULE)
        print("--verify-only complete: the three checks ran, the grid was not read.")
        print(f"  records : {txt_path}")
        print(RULE)
        return 0 if interpretable else 3

    print_verdict(
        summary, config, emotions, interpretable, decisive, finding,
        artifacts={"grid": grid_path, "raw": generations_path, "records": txt_path,
                   "": json_path},
    )
    return 0 if interpretable else 3


def _clean_residual_share(
    model, tokenizer, config: PCAJLensConfig, basis: ClampBasis, prompt: str,
) -> dict:
    """Share of the clean residual's variance inside the clamped subspace, per block."""
    import torch

    device = model_utils.model_input_device(model)
    text = model_utils.prepare_texts(
        [prompt], tokenizer, use_chat_template=True, chat_add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
    )[0]
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = model(input_ids=ids, output_hidden_states=True)
    hidden = {
        block: outputs.hidden_states[jlens_lens.hidden_state_index(block)][0]
               .float().cpu().numpy()
        for block in basis.blocks
    }
    return residual_share(basis, hidden)


if __name__ == "__main__":
    raise SystemExit(main())
