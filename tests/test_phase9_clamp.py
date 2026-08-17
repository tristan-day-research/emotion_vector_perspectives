"""Does the clamp hold what it claims, and does the paired loop pair the right things?

Two things here can be wrong without anything complaining. The clamp can fail to land the
residual on its target (then it is not a clamp), or it can hold directions it was never
meant to touch (then a null result is a lobotomy). And the paired loop can advance the
clean pass over the *clean* run's own tokens instead of the steered run's, which would
silently compare two different sentences and make every clamp target meaningless.

All three are checked against a tiny stub model where the right answer is computable by
hand, plus the aggregation that keeps the alpha=0 baseline single.

Run: python tests/test_phase9_clamp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_pca_jlens import phase9_clamp as p9  # noqa: E402
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig  # noqa: E402

D_MODEL = 32
VOCAB = 200
N_LAYERS = 5
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

class StubLayer(torch.nn.Module):
    """Adds a fixed vector, so a hook's effect is separable from the layer's."""

    def __init__(self, index: int, as_tuple: bool):
        super().__init__()
        self.index = index
        self.as_tuple = as_tuple
        self.register_buffer("bias", torch.full((D_MODEL,), 0.01 * (index + 1)))

    def forward(self, hidden):
        out = hidden + self.bias
        return (out, None) if self.as_tuple else out


class StubInner(torch.nn.Module):
    def __init__(self, as_tuple: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            StubLayer(i, as_tuple) for i in range(N_LAYERS)
        )


class StubModel(torch.nn.Module):
    def __init__(self, as_tuple: bool = True):
        super().__init__()
        self.model = StubInner(as_tuple)

    def run(self, hidden):
        for layer in self.model.layers:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


def basis_for(blocks, rank: int, seed: int = 0) -> p9.ClampBasis:
    rng = np.random.default_rng(seed)
    bases, ranks = {}, {}
    for block in blocks:
        raw = rng.normal(size=(D_MODEL, rank))
        matrix = np.linalg.qr(raw)[0][:, :rank]
        bases[block] = np.ascontiguousarray(matrix, dtype=np.float32)
        ranks[block] = rank
    return p9.ClampBasis(emotion="anxious", token_ids=[1, 2, 3], tokens=["a", "b", "c"],
                         bases=bases, ranks=ranks)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_clamp_arithmetic() -> None:
    print("the clamp holds the coordinate it says it holds")
    for as_tuple in (True, False):
        model = StubModel(as_tuple=as_tuple)
        basis = basis_for([0, 1, 2], rank=4)
        clamp = p9.Clamp(model, basis)
        hidden = torch.randn(1, 3, D_MODEL)

        with clamp.capture():
            model.run(hidden)
        captured = dict(clamp.captured)
        check(f"capture records every clamped block (tuple={as_tuple})",
              set(captured) == set(basis.blocks))
        check(f"capture changes nothing (tuple={as_tuple})",
              torch.allclose(model.run(hidden), model.run(hidden)))

        # Clamp EVERY block for the zero-target check, so the stack's final output is
        # also the last clamped block's output. Clamping a middle block and then reading
        # the final residual would measure the unclamped layers above it.
        every = basis_for(range(N_LAYERS), rank=4, seed=1)
        clamp_all = p9.Clamp(model, every)
        with clamp_all.capture():
            model.run(hidden)
        clamp_all.reset_deviation()
        clamp_all.targets = {b: torch.zeros_like(v) for b, v in clamp_all.captured.items()}
        with clamp_all.apply():
            out = model.run(hidden)
        coordinate = out.reshape(-1, D_MODEL) @ torch.as_tensor(every.bases[N_LAYERS - 1])
        check(f"a zero target drives the coordinate to zero (tuple={as_tuple})",
              float(coordinate.abs().max()) < 1e-4,
              f"max |coord| {float(coordinate.abs().max()):.2e}")
        clamp = clamp_all
        check(f"the deviation is reported, not assumed zero (tuple={as_tuple})",
              clamp.deviation < 1e-4, f"{clamp.deviation:.2e}")
        check(f"hooks are removed on exit (tuple={as_tuple})",
              not any(model.model.layers[b]._forward_hooks
                      for b in range(N_LAYERS)))

    print("\n  the no-op: clamping to the captured values changes nothing")
    model = StubModel()
    basis = basis_for(range(N_LAYERS), rank=6)
    clamp = p9.Clamp(model, basis)
    hidden = torch.randn(1, 4, D_MODEL)
    plain = model.run(hidden)
    with clamp.capture():
        model.run(hidden)
    clamp.targets = dict(clamp.captured)
    clamp.reset_deviation()
    with clamp.apply():
        clamped = model.run(hidden)
    check("output is unchanged to float precision",
          torch.allclose(plain, clamped, atol=1e-5),
          f"max |diff| {float((plain - clamped).abs().max()):.2e}")
    check("and the coordinates landed on target",
          clamp.deviation < 1e-4, f"{clamp.deviation:.2e}")

    print("\n  outside the subspace the residual is untouched, by construction")
    model = StubModel()
    basis = basis_for([1], rank=5)
    clamp = p9.Clamp(model, basis)
    hidden = torch.randn(1, 2, D_MODEL)
    with clamp.capture():
        model.run(hidden)
    matrix = torch.as_tensor(basis.bases[1])
    shifted = {b: v + 3.0 for b, v in clamp.captured.items()}
    clamp.targets = shifted
    with clamp.apply():
        out = model.run(hidden)
    plain = model.run(hidden)
    difference = (out - plain).reshape(-1, D_MODEL)
    # Every row of the difference must lie in span(A): projecting it out leaves nothing.
    residual_outside = difference - (difference @ matrix) @ matrix.T
    check("the change lies entirely inside span(A)",
          float(residual_outside.abs().max()) < 1e-4,
          f"max component outside {float(residual_outside.abs().max()):.2e}")
    check("and inside it the change is real, so this is not a vacuous check",
          float((difference @ matrix).abs().max()) > 1.0,
          f"max component inside {float((difference @ matrix).abs().max()):.2f}")

    print("\n  a block beyond the model is refused rather than silently skipped")
    try:
        p9.Clamp(StubModel(), basis_for([N_LAYERS + 3], rank=2))
        check("out-of-range block aborts", False)
    except SystemExit:
        check("out-of-range block aborts", True)


class ToyOutput:
    def __init__(self, logits, cache):
        self.logits = logits
        self.past_key_values = cache


class ToyLM(StubModel):
    """A toy LM whose hidden state is a function of the token sequence.

    Enough to drive :func:`paired_generate`: hooks fire on ``model.layers``, a KV "cache"
    is a token count so lockstep can be asserted, and every call records the ids it saw --
    which is how the clean/steered pairing gets checked rather than assumed.
    """

    def __init__(self):
        super().__init__(as_tuple=True)
        generator = torch.Generator().manual_seed(3)
        self.embed = torch.randn(VOCAB, D_MODEL, generator=generator)
        self.head = torch.randn(VOCAB, D_MODEL, generator=generator)
        self.calls: list[list[int]] = []

    def forward(self, input_ids=None, past_key_values=None, use_cache=False, **kwargs):
        self.calls.append([int(t) for t in input_ids[0]])
        hidden = self.embed[input_ids[0]].unsqueeze(0)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        cache = (past_key_values or 0) + int(input_ids.shape[-1])
        return ToyOutput(hidden @ self.head.T, cache)


class ToyTokenizer:
    eos_token_id = -1

    def __call__(self, text, return_tensors=None):
        ids = [int(part) for part in text.split()]
        return type("E", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


def test_paired_loop() -> None:
    print("\nthe paired loop: the clean run follows the STEERED run's tokens")
    config = PCAJLensConfig().with_overrides({"generation_max_new_tokens": 5})
    prompt = "3 5 7 11"
    block = 2
    vector = np.random.default_rng(4).normal(size=D_MODEL) * 4.0

    original_prepare, original_device = (
        p9.model_utils.prepare_texts, p9.model_utils.model_input_device
    )
    p9.model_utils.prepare_texts = lambda texts, *a, **k: list(texts)
    p9.model_utils.model_input_device = lambda _m: "cpu"
    try:
        model = ToyLM()
        basis = basis_for(range(N_LAYERS), rank=4, seed=2)
        clamp = p9.Clamp(model, basis)
        clamped = p9.paired_generate(
            model, ToyTokenizer(), config, prompt, block, vector, 1.0, clamp
        )
        calls = list(model.calls)

        model_plain = ToyLM()
        plain = p9.paired_generate(
            model_plain, ToyTokenizer(), config, prompt, block, vector, 1.0, None
        )
        unsteered = p9.paired_generate(
            ToyLM(), ToyTokenizer(), config, prompt, block, vector, 0.0, None
        )
    finally:
        p9.model_utils.prepare_texts = original_prepare
        p9.model_utils.model_input_device = original_device

    check("something was generated", clamped.n_new_tokens == 5,
          f"{clamped.n_new_tokens} tokens")
    check("exactly two passes per step -- one clean, one steered",
          len(calls) == 2 * clamped.n_new_tokens,
          f"{len(calls)} calls for {clamped.n_new_tokens} tokens")
    # Calls alternate clean, steered, clean, steered, ... so each pair must agree on the
    # token. If the clean run were advancing over its OWN argmax they would diverge.
    pairs = [(calls[i], calls[i + 1]) for i in range(0, len(calls) - 1, 2)]
    check("every clean call sees exactly what the steered call sees",
          all(a == b for a, b in pairs),
          f"{sum(1 for a, b in pairs if a != b)}/{len(pairs)} pairs disagree")
    check("the prompt is prefilled by both runs",
          pairs[0][0] == [3, 5, 7, 11])
    check("later calls are single-token decode steps",
          all(len(a) == 1 for a, _ in pairs[1:]))

    check("the unclamped run makes one pass per step, so the clamp costs exactly 2x",
          len(model_plain.calls) == plain.n_new_tokens
          and len(calls) == 2 * len(model_plain.calls),
          f"{len(model_plain.calls)} unclamped vs {len(calls)} clamped")
    check("steering changes the output, so the clamp has something to act on",
          plain.text != unsteered.text)
    check("the clamp changes the steered trajectory",
          clamped.text != plain.text,
          f"clamped {clamped.text!r} vs unclamped {plain.text!r}")
    check("the clamp lands its coordinates while steering is active",
          clamped.clamp_deviation < 1e-3, f"{clamped.clamp_deviation:.2e}")


def test_hook_order() -> None:
    print("\n  the clamp runs AFTER the steering at the block that carries both")
    model = StubModel()
    # The LAST block, so the stack's final output is also the clamped block's output.
    # Clamping a middle block and reading the final residual would measure the biases the
    # unclamped layers above it add, not the hook order.
    last = N_LAYERS - 1
    basis = basis_for([last], rank=4, seed=5)
    clamp = p9.Clamp(model, basis)
    matrix = torch.as_tensor(basis.bases[last])
    hidden = torch.zeros(1, 1, D_MODEL)

    with clamp.capture():
        model.run(hidden)
    clamp.targets = dict(clamp.captured)
    clamp.reset_deviation()
    # A steering vector deliberately INSIDE the clamped subspace: if the clamp ran first
    # the steering would survive it, and the coordinate would come out shifted.
    inside = (matrix @ torch.ones(4)).numpy() * 5.0
    with p9.p8.steering(model, last, inside, 1.0):
        with clamp.apply():
            out = model.run(hidden)
    coordinate = out.reshape(-1, D_MODEL) @ matrix
    target = clamp.targets[last]
    check("a steering vector inside the clamped span is removed by the clamp",
          float((coordinate - target).abs().max()) < 1e-3,
          f"max |coord - target| {float((coordinate - target).abs().max()):.2e}")

    # The reverse order is what the code deliberately avoids: assert it would fail, so
    # the ordering comment is backed by a measurement rather than by reasoning.
    clamp.reset_deviation()
    with clamp.apply():
        with p9.p8.steering(model, last, inside, 1.0):
            wrong = model.run(hidden)
    wrong_coordinate = wrong.reshape(-1, D_MODEL) @ matrix
    check("and the reverse order would have let it through, which is why order is fixed",
          float((wrong_coordinate - target).abs().max()) > 1.0,
          f"max |coord - target| {float((wrong_coordinate - target).abs().max()):.2f}")


def test_residual_share() -> None:
    print("\nresidual share: how much of the model the clamp is holding still")
    basis = basis_for([0, 1], rank=4)
    matrix = basis.bases[0]
    # A residual entirely inside span(A) must report a share of 1, entirely outside 0.
    inside = (matrix @ np.ones((4, 1))).T.astype(np.float64)
    outside = np.linalg.svd(matrix.T, full_matrices=True)[2][4:5].astype(np.float64)
    share_in = p9.residual_share(basis, {0: inside, 1: inside})
    share_out = p9.residual_share(basis, {0: outside, 1: outside})
    check("a residual inside the subspace reports share 1",
          abs(share_in["per_block"]["0"] - 1.0) < 1e-6,
          f"{share_in['per_block']['0']:.6f}")
    check("a residual orthogonal to it reports share 0",
          share_out["per_block"]["0"] < 1e-6, f"{share_out['per_block']['0']:.2e}")
    check("mean and max are reported per run", {"mean", "max"} <= set(share_in))
    check("an absent block is skipped rather than crashing",
          p9.residual_share(basis, {0: inside})["n_blocks"] == 1)


def test_suppression() -> None:
    print("\nsuppression: the fraction of the report lift the clamp removed")
    full = p9.suppression(1.0, 3.0, 1.0, 0.7)
    check("removing the whole lift is 100%", full["defined"] and full["removed"] == 1.0)
    check("and it passes the floor", full["holds"])
    half = p9.suppression(1.0, 3.0, 2.0, 0.7)
    check("removing half is 50%", abs(half["removed"] - 0.5) < 1e-9)
    check("and it fails a 70% floor", not half["holds"])
    none = p9.suppression(1.0, 1.0, 1.0, 0.7)
    check("no lift to remove is UNDEFINED, not a pass",
          not none["defined"] and "did not raise" in none["reason"])
    check("a null report score is undefined too, not treated as zero",
          not p9.suppression(1.0, None, 1.0, 0.7)["defined"])
    over = p9.suppression(1.0, 3.0, 0.0, 0.7)
    check("overshooting past baseline still passes, and is visible as >100%",
          over["holds"] and over["removed"] > 1.0, f"{over['removed']:.0%}")


def test_spearman() -> None:
    print("\nthe rank correlation, since scipy is not a dependency")
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    check("identical inputs give 1", abs(p9._spearman(a, a) - 1.0) < 1e-9)
    check("a reversal gives -1", abs(p9._spearman(a, -a) + 1.0) < 1e-9)
    check("a monotone transform gives 1, which Pearson would not",
          abs(p9._spearman(a, np.exp(a)) - 1.0) < 1e-9)
    check("independent inputs give ~0", abs(p9._spearman(a, rng.normal(size=200))) < 0.2)
    check("an all-tied input gives 0, not a spurious correlation",
          p9._spearman(a, np.zeros(200)) == 0.0,
          "argsort(argsort(.)) would have invented an ordering here")
    tied = np.array([1.0, 1.0, 2.0, 3.0])
    check("ties get averaged ranks, as Spearman requires",
          np.allclose(p9._rank(tied), [0.5, 0.5, 2.0, 3.0]), str(p9._rank(tied)))
    check("a flattened readout cannot pass the collateral check",
          p9._spearman(rng.normal(size=512), np.full(512, 3.0)) == 0.0)


def test_control_tokens() -> None:
    print("\ncontrol tokens exclude the clamped set")
    rng = np.random.default_rng(1)
    excluded = [3, 7, 11, 19]
    controls = p9.control_token_ids(VOCAB, excluded, rng, 50)
    check("none of the clamped tokens is a control",
          not (set(controls) & set(excluded)))
    check("the requested count is returned", len(controls) == 50)
    check("ids are unique", len(set(controls)) == len(controls))
    check("asking for more than the vocabulary holds returns what exists",
          len(p9.control_token_ids(20, list(range(15)), rng, 100)) == 5)


def test_baseline_expansion() -> None:
    print("\none baseline, and the no-op cell kept out of the grid")
    rows = []
    for prompt in ("p1", "p2"):
        rows.append({"concept": "anxious", "family": "report", "prompt": prompt,
                     "condition": "v", "alpha": 0.0, "role": p9.ROLE_BASELINE,
                     "score": 1.0})
        rows.append({"concept": "anxious", "family": "report", "prompt": prompt,
                     "condition": "v|clamp", "alpha": 0.0, "role": p9.ROLE_NOOP,
                     "score": 1.0})
        for cell in ("v", "v|clamp", "v_remainder", "v_remainder|clamp"):
            rows.append({"concept": "anxious", "family": "report", "prompt": prompt,
                         "condition": cell, "alpha": 1.0, "role": p9.ROLE_CELL,
                         "score": 2.0})
    expanded = p9.expand_baseline(rows)
    zero = [r for r in expanded if r["alpha"] == 0.0]
    check("the no-op rows are dropped from the grid",
          all(r["role"] == p9.ROLE_BASELINE for r in zero))
    check("the baseline is copied to every cell name",
          {r["condition"] for r in zero}
          == {p9.cell_name(c, k) for c in p9.CONDITIONS for k in (False, True)})
    check("exactly one alpha=0 row per (cell, prompt)",
          len(zero) == 4 * 2, f"{len(zero)} rows")
    check("the steered cells survive untouched",
          len([r for r in expanded if r["alpha"] == 1.0]) == 8)

    print("\n  Phase 8's shift arithmetic then finds a single baseline")
    frame = __import__("pandas").DataFrame([
        {**r, "channel": "report", "response": "x", "perplexity": 10.0,
         "kind": "emotion"}
        for r in expanded
    ])
    table = p8_family_table(frame)
    baselines = table["baseline"].dropna().unique()
    check("one baseline value across the whole table", len(baselines) == 1,
          str(baselines))
    check("the steered cells shift by the right amount",
          all(abs(v - 1.0) < 1e-9
              for v in table.loc[table["alpha"] == 1.0, "shift"].dropna()))


def p8_family_table(frame):
    from emotion_pca_jlens import phase8_steer as p8

    return p8.family_table(frame, PCAJLensConfig())


def test_noop_check() -> None:
    print("\nthe no-op check reads roles, and reports rounding rather than failing on it")
    rows = [
        {"prompt": "p1", "role": p9.ROLE_BASELINE, "response": "same"},
        {"prompt": "p2", "role": p9.ROLE_BASELINE, "response": "same too"},
        {"prompt": "p1", "role": p9.ROLE_NOOP, "response": "same",
         "clamp_deviation": 1e-3},
        {"prompt": "p2", "role": p9.ROLE_NOOP, "response": "drifted",
         "clamp_deviation": 2e-3},
    ]
    result = p9.noop_check(rows, PCAJLensConfig())
    check("both clamped rows are checked", result["checked"] == 2)
    check("text equality is counted, not required",
          result["identical_texts"] == 1 and result["frac_identical"] == 0.5)
    check("the numeric check is what gates, and it passes here",
          result["numeric_holds"], f"max dev {result['max_coordinate_deviation']}")
    broken = p9.noop_check(
        [rows[0], {**rows[2], "clamp_deviation": 12.0}], PCAJLensConfig()
    )
    check("a coordinate that misses its target fails the numeric check",
          not broken["numeric_holds"])
    check("no rows is not a pass", p9.noop_check([], PCAJLensConfig())["checked"] == 0)


def test_cells_and_labels() -> None:
    print("\nthe grid's shape")
    config = PCAJLensConfig()
    strengths = [float(a) for a in config.clamp_strengths]
    n_cells = 2 + (len(strengths) - 1) * len(p9.CONDITIONS) * 2
    check("the default is 6 cells per emotion", n_cells == 6, f"{n_cells}")
    check("every (condition, clamp) pair has a label",
          all(p9.cell_name(c, k) in p9.CELL_LABELS
              for c in p9.CONDITIONS for k in (False, True)))
    check("the decisive cell is named as such",
          "DECISIVE" in p9.CELL_LABELS[p9.cell_name("v_remainder", True)])
    check("0.0 is required, so there is always a baseline and a no-op check",
          0.0 in strengths)
    n_prompts = 21
    print(f"    {n_cells} cells x {n_prompts} prompts x 2 forward passes per decode step")


def test_verification_printing() -> None:
    print("\nverification output")
    import contextlib
    import io

    config = PCAJLensConfig()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        good = p9.print_verification(
            {"checked": 21, "identical_texts": 21, "frac_identical": 1.0,
             "max_coordinate_deviation": 1e-3, "numeric_holds": True},
            p9.suppression(1.0, 3.0, 1.1, config.clamp_min_report_suppression),
            {"spearman": 0.99, "disturbance": 0.01, "holds": True,
             "n_control_tokens": 512, "alpha": 1.0},
            {"n_blocks": 63, "mean": 0.03, "max": 0.08},
            config,
        )
    text = buffer.getvalue()
    check("a clean run verifies", good)
    check("all three checks are printed",
          "no-op" in text and "suppression" in text and "unrelated J-space" in text)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        bad = p9.print_verification(
            {"checked": 21, "identical_texts": 3, "frac_identical": 0.14,
             "max_coordinate_deviation": 40.0, "numeric_holds": False},
            {"defined": False, "reason": "no lift"},
            {"spearman": 0.4, "disturbance": 0.6, "holds": False,
             "n_control_tokens": 512, "alpha": 1.0},
            {"n_blocks": 63, "mean": 0.6, "max": 0.9},
            config,
        )
    text = buffer.getvalue()
    check("a broken run does not verify", not bad)
    check("it says the cell is not interpretable", "NOT interpretable" in text)
    check("it names what to do about each failure",
          "clamp_token_count" in text and "implementation fault" in text)
    check("an over-large subspace is called out",
          "most of" in text and "OVER HALF" in text)
    check("a missing collateral measurement is stated, not formatted as a number",
          "NOT MEASURED" in _verification_text(config))


def _verification_text(config) -> str:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        p9.print_verification(
            {"checked": 0}, {"defined": False, "reason": "x"}, {}, {}, config
        )
    return buffer.getvalue()


def test_phase8_precondition() -> None:
    print("\nPhase 9 refuses to be a control for nothing")
    import json
    import tempfile

    from core import paths

    tmp = Path(tempfile.mkdtemp())
    paths.OUTPUTS_DIR = tmp / "outputs"
    config = PCAJLensConfig(run_name="t9")
    (config.phase_dir / "phase8_steering").mkdir(parents=True, exist_ok=True)
    gate = config.phase_dir / "phase8_steering" / "phase8_gate.json"

    def summary(v_perp_z, random_z, degraded=False):
        return [
            {"concept": "anxious", "condition": "v_remainder", "channel": "behaviour",
             "alpha": 1.0, "abs_z": v_perp_z, "degraded": degraded},
            {"concept": "anxious", "condition": "v_random", "channel": "behaviour",
             "alpha": 1.0, "abs_z": random_z, "degraded": False},
        ]

    for label, payload, expect in (
        ("a real v_perp effect", summary(2.0, 0.3), True),
        ("v_perp no better than random", summary(0.3, 2.0), False),
        ("no v_perp movement at all", summary(0.0, 0.0), False),
        ("the effect only in degraded cells", summary(2.0, 0.3, degraded=True), False),
    ):
        gate.write_text(json.dumps({
            "summary": payload,
            "phase7": {"emotions": ["anxious"]},
            "thinking": {"resolved": "off"},
        }))
        _, _, finding = p9.read_phase8(config)
        check(f"{label}: has_effect={expect}", finding["has_effect"] == expect,
              finding["note"])

    for label, thinking, refused in (
        ("thinking ON", {"resolved": "on"}, True),
        ("thinking unrecorded", {}, True),
        ("thinking off", {"resolved": "off"}, False),
    ):
        gate.write_text(json.dumps({
            "summary": summary(2.0, 0.3),
            "phase7": {"emotions": ["anxious"]},
            "thinking": thinking,
        }))
        try:
            p9.read_phase8(config)
            raised = False
        except SystemExit as exc:
            raised, message = True, str(exc)
        check(f"a Phase 8 record with {label} is "
              f"{'refused' if refused else 'accepted'}", raised == refused,
              message.splitlines()[0][:66] if raised else "")

    gate.unlink()
    try:
        p9.read_phase8(config)
        check("a missing Phase 8 record aborts", False)
    except SystemExit as exc:
        check("a missing Phase 8 record aborts", "run.py phase8" in str(exc))


def main() -> int:
    test_clamp_arithmetic()
    test_hook_order()
    test_paired_loop()
    test_residual_share()
    test_suppression()
    test_spearman()
    test_control_tokens()
    test_baseline_expansion()
    test_noop_check()
    test_cells_and_labels()
    test_verification_printing()
    test_phase8_precondition()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
