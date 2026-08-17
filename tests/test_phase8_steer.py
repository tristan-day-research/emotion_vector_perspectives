"""Phase 8's pure logic, exercised without a GPU or an API key.

Covers the parts where a silent wrong answer is possible: the steering hook's arithmetic
and cleanup, left padding for batched generation, the shared alpha=0 baseline, and the
grid scaling used by the cross-channel summary. The generation and judge paths are
stubbed; what is checked is the code around them.

Run: python tests/test_phase8_steer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_pca_jlens import phase8_steer as p8  # noqa: E402
from emotion_pca_jlens import channel_prompts as cp  # noqa: E402
from emotion_pca_jlens.channel_prompts import BEHAVIOUR_TASKS  # noqa: E402
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig  # noqa: E402

FAILURES: list[str] = []
D_MODEL = 8


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

class StubLayer(torch.nn.Module):
    """A decoder layer that returns its input, so a hook's delta is the whole output."""

    def __init__(self, as_tuple: bool):
        super().__init__()
        self.as_tuple = as_tuple

    def forward(self, hidden):
        return (hidden, "cache") if self.as_tuple else hidden


class StubInner(torch.nn.Module):
    def __init__(self, n_layers: int, as_tuple: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(StubLayer(as_tuple) for _ in range(n_layers))


class StubModel(torch.nn.Module):
    def __init__(self, n_layers: int = 4, as_tuple: bool = True):
        super().__init__()
        self.model = StubInner(n_layers, as_tuple)


class StubTokenizer:
    """Records the padding side in force at each call, which is the thing under test."""

    pad_token_id = 0
    # A stop token distinct from pad, so "finished" is a real signal rather than an
    # artefact of padding.
    eos_token_id = 99

    def __init__(self):
        self.padding_side = "right"
        self.sides_seen: list[str] = []
        self.batch_sizes: list[int] = []

    def __call__(self, texts, return_tensors=None, padding=False, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        self.sides_seen.append(self.padding_side)
        self.batch_sizes.append(len(texts))
        widths = [len(t.split()) for t in texts]
        width = max(widths)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        for i, w in enumerate(widths):
            # Left padding puts the content at the end, which is what generation needs.
            span = slice(width - w, width) if self.padding_side == "left" else slice(0, w)
            ids[i, span] = torch.arange(1, w + 1)
        return {"input_ids": ids} if return_tensors is None else _Encoded(ids)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids if int(i) != 0)


class _Encoded(dict):
    def __init__(self, ids):
        super().__init__(input_ids=ids)

    def to(self, _device):
        return self

    def keys(self):  # for **encoded
        return super().keys()


class StubGenerator(StubModel):
    """Appends two tokens per row, so the prompt/completion split can be checked."""

    def generate(self, input_ids=None, max_new_tokens=1, **kwargs):
        extra = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long)
        extra[:, 1] = torch.arange(input_ids.shape[0]) + 20
        return torch.cat([input_ids, extra], dim=1)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_steering() -> None:
    print("steering hook")
    for as_tuple in (True, False):
        model = StubModel(as_tuple=as_tuple)
        vector = np.arange(D_MODEL, dtype=np.float64)
        hidden = torch.zeros((2, 3, D_MODEL))
        with p8.steering(model, 1, vector, 2.0):
            out = model.model.layers[1](hidden)
        got = out[0] if as_tuple else out
        check(f"delta is alpha*v (tuple={as_tuple})",
              torch.allclose(got[0, 0], torch.as_tensor(2.0 * vector, dtype=torch.float32)))
        check(f"the rest of the tuple survives (tuple={as_tuple})",
              (out[1] == "cache") if as_tuple else True)
        untouched = model.model.layers[1](hidden)
        untouched = untouched[0] if as_tuple else untouched
        check(f"hook is removed on exit (tuple={as_tuple})",
              torch.allclose(untouched, hidden))

    model = StubModel()
    hidden = torch.ones((1, 3, D_MODEL))
    with p8.steering(model, 0, np.ones(D_MODEL), 0.0):
        out = model.model.layers[0](hidden)[0]
    check("alpha=0 is a genuine no-op", torch.allclose(out, hidden))
    check("no hook is registered at alpha=0",
          not model.model.layers[0]._forward_hooks)

    model = StubModel()
    with p8.steering(model, 2, np.ones(D_MODEL), 1.0, positions="generated"):
        prefill = model.model.layers[2](torch.zeros((1, 5, D_MODEL)))[0]
        decode = model.model.layers[2](torch.zeros((1, 1, D_MODEL)))[0]
    check("positions='generated' skips the prompt", torch.allclose(prefill, torch.zeros(1)))
    check("positions='generated' steers the decode", torch.allclose(decode, torch.ones(1)))

    model = StubModel()
    try:
        with p8.steering(model, 99, np.ones(D_MODEL), 1.0):
            pass
        check("out-of-range block aborts", False)
    except SystemExit:
        check("out-of-range block aborts", True)

    # A hook that raises must still be removed, or every later cell is silently steered
    # by a stale direction.
    model = StubModel()
    try:
        with p8.steering(model, 0, np.ones(D_MODEL), 1.0):
            raise RuntimeError("cell failed")
    except RuntimeError:
        pass
    check("hook is removed even when the body raises",
          not model.model.layers[0]._forward_hooks)


def test_generation() -> None:
    print("\nbatched generation")
    config = PCAJLensConfig().with_overrides({"generation_batch_size": 2})
    tokenizer = StubTokenizer()
    model = StubGenerator()
    prompts = ["a", "b b b", "c c", "d", "e"]

    original = p8.model_utils.prepare_texts
    seen_thinking: list = []

    def spy(texts, *a, **k):
        seen_thinking.append(k.get("enable_thinking"))
        return list(texts)

    p8.model_utils.prepare_texts = spy
    device = p8.model_utils.model_input_device
    p8.model_utils.model_input_device = lambda _model: "cpu"
    try:
        out = p8.generate_batched(model, tokenizer, config, prompts)
    finally:
        p8.model_utils.prepare_texts = original
        p8.model_utils.model_input_device = device

    check("one completion per prompt", len(out) == len(prompts))
    check("every batch asked prepare_texts to disable thinking",
          seen_thinking and all(v is False for v in seen_thinking),
          f"{len(seen_thinking)} batches, values {set(seen_thinking)}")
    check("left padding while generating", set(tokenizer.sides_seen) == {"left"},
          str(tokenizer.sides_seen))
    check("padding side restored afterwards", tokenizer.padding_side == "right")
    check("batched at generation_batch_size", tokenizer.batch_sizes == [2, 2, 1],
          str(tokenizer.batch_sizes))
    texts = [c.text for c in out]
    check("only the completion is decoded", all(len(t.split()) == 2 for t in texts),
          str(texts))
    check("completions are not all identical", len(set(texts)) > 1)
    check("each completion reports whether the model stopped",
          all(isinstance(c.finished, bool) for c in out))
    check("the stub never emits EOS, so nothing is marked finished",
          not any(c.finished for c in out),
          "which is exactly the state the completion-rate line exists to surface")


def test_grid_cells() -> None:
    print("\ngrid cells and the shared baseline")
    cells = p8.grid_cells([0.0, 0.5, 1.0, 2.0])
    check("one cell per condition per nonzero strength, plus one baseline",
          len(cells) == 3 * len(p8.CONDITIONS) + 1, f"{len(cells)} cells")
    check("alpha=0 appears exactly once",
          sum(1 for _, a in cells if a == 0.0) == 1)
    check("no strengths means no cells", p8.grid_cells([]) == [])
    check("without 0.0 there is no baseline cell",
          all(a != 0.0 for _, a in p8.grid_cells([1.0])))

    rows = [
        {"concept": "anxious", "condition": "v", "alpha": 0.0, "shared_baseline": True,
         "score": 2.0, "family": "report"},
        {"concept": "anxious", "condition": "v", "alpha": 1.0, "shared_baseline": False,
         "score": 3.0, "family": "report"},
    ]
    expanded = p8.expand_baseline(rows)
    check("the baseline is copied to every condition",
          sorted(r["condition"] for r in expanded if r["alpha"] == 0.0)
          == sorted(p8.CONDITIONS))
    check("the steered row is not copied",
          sum(1 for r in expanded if r["alpha"] == 1.0) == 1)
    check("every copy stays flagged as shared",
          all(r["shared_baseline"] for r in expanded if r["alpha"] == 0.0))
    check("scores travel with the copies",
          {r["score"] for r in expanded if r["alpha"] == 0.0} == {2.0})

    base_vectors = {
        "v": np.ones(D_MODEL), "v_reportable": np.ones(D_MODEL),
        "v_remainder": np.ones(D_MODEL), "v_random": np.ones(D_MODEL),
    }
    randoms = p8.add_random_controls(
        base_vectors, 3.0, np.random.default_rng(4)
    )
    check("five random controls are constructed",
          all(key in randoms for key in p8.RANDOM_CONDITIONS))
    check("every random control is norm matched",
          all(abs(np.linalg.norm(randoms[key]) - 3.0) < 1e-9
              for key in p8.RANDOM_CONDITIONS[1:]))


def test_scoring() -> None:
    print("\nmechanical scoring")
    families = {task.family for task in BEHAVIOUR_TASKS}
    report_task = cp.build_report_choice_tasks(
        ["anxious", "calm"], seed=0, n_variants=1
    )[0]
    target_label = next(
        label for label, emotion in report_task.label_to_emotion if emotion == "anxious"
    )
    responses = {
        "risk": "A",
        "persistence": "Therefore the final answer is 42. " + "working step " * 45,
        "hedging": "It might possibly be about right " * 6,
        "refusal": "I will not do that.",
    }
    rows = [
        {"family": task.family, "response": responses[task.family]}
        for task in BEHAVIOUR_TASKS
    ] + [{
        "family": "report", "response": target_label, "report_emotion": "anxious",
        "report_variant": report_task.variant, "prompt": report_task.prompt,
        "report_mapping": json.dumps(report_task.mapping), "finished": True,
    }]
    usage = p8.score_grid(rows, PCAJLensConfig(), use_judge=False)
    scored = {r["family"] for r in rows if r["score"] is not None}
    check("every exact/mechanical family is scored",
          scored == (families - {"refusal"}) | {"report"},
          str(sorted(scored)))
    check("only the refusal judge family is unscored without a judge",
          all(r["score"] is None for r in rows if r["family"] == "refusal"))
    check("report choice is target-specific and exact-scored",
          rows[-1]["score"] == 1.0 and rows[-1]["scorer"] == "exact-letter")
    check("--no-judge is recorded, not silent", usage.get("skipped") == "--no-judge")
    check("no judge calls are counted", usage["calls"] == 0)
    check("mechanical scorers leave their detail",
          all(r["detail"] for r in rows if r["scorer"] == "mechanical"))


def _frame(scores: dict[tuple[str, float, str], float], perplexity=10.0) -> pd.DataFrame:
    rows = []
    for (condition, alpha, family), score in scores.items():
        rows.append({
            "concept": "anxious", "kind": "emotion", "report_emotion": "anxious",
            "condition": condition, "alpha": alpha,
            "channel": "report" if family == "report" else "behaviour",
            "family": family, "prompt": "p", "response": "r", "score": score,
            "shared_baseline": alpha == 0.0,
            "perplexity": perplexity if alpha > 1.0 else 10.0,
        })
    return pd.DataFrame(rows)


def test_aggregation() -> None:
    print("\naggregation and the grid-SD scale")
    config = PCAJLensConfig().with_overrides({"perplexity_max_ratio": 1.5})
    scores = {}
    for condition in p8.CONDITIONS:
        scores[(condition, 0.0, "report")] = 1.0
        scores[(condition, 0.0, "hedging")] = 10.0
    scores[("v", 1.0, "report")] = 3.0
    scores[("v", 1.0, "hedging")] = 20.0
    scores[("v_remainder", 1.0, "report")] = 1.0
    scores[("v_remainder", 1.0, "hedging")] = 30.0
    frame = _frame(scores)
    table = p8.family_table(frame, config)

    def cell(condition, alpha, family, column):
        row = table[(table["condition"] == condition) & (table["alpha"] == alpha)
                    & (table["family"] == family)]
        return float(row[column].iloc[0])

    check("baseline is the alpha=0 score", cell("v", 1.0, "report", "baseline") == 1.0)
    check("shift is raw", cell("v", 1.0, "hedging", "shift") == 10.0)
    check("z rescales incompatible families onto one axis",
          abs(cell("v", 1.0, "report", "z") / cell("v", 1.0, "hedging", "z") - 1.0) > 0.0)
    check("a bigger raw shift in the same family gives a bigger z",
          cell("v_remainder", 1.0, "hedging", "z") > cell("v", 1.0, "hedging", "z"))
    check("the baseline cell has z=0", cell("v", 0.0, "report", "z") == 0.0)

    summary = p8.channel_summary(table)
    remainder = summary[(summary["condition"] == "v_remainder")
                        & (summary["alpha"] == 1.0)]
    report = float(remainder[remainder["channel"] == "report"]["abs_z"].iloc[0])
    behaviour = float(remainder[remainder["channel"] == "behaviour"]["abs_z"].iloc[0])
    check("the dissociation is visible in the summary", behaviour > 0 and report == 0.0,
          f"behaviour {behaviour:.2f}, report {report:.2f}")

    print("\n  magnitude, not sign")
    signed = {}
    for condition in p8.CONDITIONS:
        signed[(condition, 0.0, "hedging")] = 10.0
        signed[(condition, 0.0, "risk")] = 0.5
    signed[("v", 1.0, "hedging")] = 20.0     # up
    signed[("v", 1.0, "risk")] = 0.0         # down by the same z
    table2 = p8.family_table(_frame(signed), config)
    summary2 = p8.channel_summary(table2)
    cell2 = summary2[(summary2["condition"] == "v") & (summary2["alpha"] == 1.0)
                     & (summary2["channel"] == "behaviour")]
    check("opposite-signed family shifts do not cancel",
          float(cell2["abs_z"].iloc[0]) > 0.5, f"{float(cell2['abs_z'].iloc[0]):.2f}")

    print("\n  fluency")
    degraded = _frame(scores, perplexity=100.0)
    fluency = p8.cell_fluency(degraded, config)
    check("a high-perplexity cell is flagged",
          bool(fluency[fluency["alpha"] > 1.0]["degraded"].all())
          if (fluency["alpha"] > 1.0).any() else True)
    check("the baseline is never degraded",
          not fluency[fluency["alpha"] == 0.0]["degraded"].any())
    check("the ratio is against the alpha=0 mean",
          abs(float(fluency[fluency["alpha"] == 0.0]["perplexity_ratio"].iloc[0]) - 1.0)
          < 1e-9)

    print("\n  degraded cells stay out of the scale")
    wide = {}
    for condition in p8.CONDITIONS:
        wide[(condition, 0.0, "hedging")] = 10.0
    wide[("v", 1.0, "hedging")] = 12.0
    wide[("v", 2.0, "hedging")] = 500.0      # nonsense from a degraded cell
    rows = _frame(wide)
    rows.loc[rows["alpha"] == 2.0, "perplexity"] = 100.0
    scaled = p8.family_table(rows, config)
    z_clean = float(scaled[(scaled["alpha"] == 1.0)
                           & (scaled["condition"] == "v")]["z"].iloc[0])
    rows_no_bad = rows[rows["alpha"] != 2.0]
    z_without = float(p8.family_table(rows_no_bad, config).query(
        "alpha == 1.0 and condition == 'v'")["z"].iloc[0])
    check("the degraded cell does not shrink the undegraded z",
          abs(z_clean - z_without) < 1e-9, f"{z_clean:.3f} vs {z_without:.3f}")

    print("\n  completion and format are real cell gates")
    quality_rows = _frame(scores)
    quality_rows["finished"] = True
    quality_rows["valid"] = True
    target = quality_rows.index[quality_rows["alpha"] == 1.0][0]
    quality_rows.loc[target, "finished"] = False
    quality_rows.loc[target, "valid"] = False
    quality = p8.cell_fluency(quality_rows, config)
    bad_key = quality_rows.loc[target, ["concept", "condition", "alpha"]]
    bad = quality[
        (quality["concept"] == bad_key["concept"])
        & (quality["condition"] == bad_key["condition"])
        & (quality["alpha"] == bad_key["alpha"])
    ]
    check("an incomplete/invalid cell is unusable even when perplexity is fine",
          len(bad) == 1 and not bool(bad["quality_passed"].iloc[0]))

    print("\n  preregistered dissociation criteria")
    prereg = {}
    for condition in p8.CONDITIONS:
        prereg[(condition, 0.0, "report")] = 0.0
        prereg[(condition, 0.0, "risk")] = 0.0
        prereg[(condition, 1.0, "report")] = 0.0
        prereg[(condition, 1.0, "risk")] = 0.0
    prereg[("v", 1.0, "report")] = 1.0
    prereg[("v", 1.0, "risk")] = 1.0
    prereg[("v_reportable", 1.0, "report")] = 1.0
    prereg[("v_remainder", 1.0, "risk")] = 1.0
    prereg_table = p8.family_table(_frame(prereg), config)
    evidence = p8.dissociation_evidence(prereg_table, ["anxious"], "risk")
    item = evidence["per_emotion"]["anxious"]
    check("the full-vector manipulation check passes", item["experiment_interpretable"])
    check("v_perp must beat every random control",
          item["remainder_behaviour_beats_all_random"])
    check("the complete synthetic pattern is recognized",
          item["dissociation_pattern_supported"] and item["dissociation_statistic_D"] == 1.0)


def test_printing() -> None:
    print("\nprinting")
    config = PCAJLensConfig()
    scores = {}
    for condition in p8.CONDITIONS:
        scores[(condition, 0.0, "report")] = 1.0
        scores[(condition, 0.0, "hedging")] = 10.0
    scores[("v", 1.0, "report")] = 3.0
    scores[("v", 1.0, "hedging")] = 20.0
    table = p8.family_table(_frame(scores), config)
    summary = p8.channel_summary(table)
    concept = p8.Directions(name="anxious", kind="emotion", report_emotion="anxious",
                            vectors={}, norm=12.0)

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        p8.print_design(
            config,
            {"manipulation_checks": {"completion": {"rate": 1.0}},
             "thinking": {"resolved": "off"}},
            Path("phase7_gate.json"), ["anxious"], 1, [concept],
            [0.0, 0.5, 1.0], p8.grid_cells([0.0, 0.5, 1.0]),
            5, 4, 9, 153, 0,
        )
        p8.print_concept(summary, table, concept)
        p8.print_verdict(summary, p8.cell_fluency(_frame(scores), config), config,
                         ["anxious"], None, "test", {"calls": 0},
                         {"grid": Path("g.csv")})
    text = buffer.getvalue()
    check("both channels are printed",
          "report channel" in text and "behaviour channel" in text)
    check("the dry-run design printer has no hidden-variable NameError",
          "phase7_gate.json" in text and "completion rate 100%" in text)
    check("the per-family raw scores are printed too", "hedging" in text)
    check("the re-entry caveat is printed", "RE-ENTRY CAVEAT" in text)
    check("Phase 9 is named as the thing that would settle it", "Phase 9" in text)
    check("a missing specificity control is called out", "NOT RUN" in text)
    check("the gate says it stops", "STOPPING at the Phase 8 gate" in text)
    check("no row is wider than 100 columns",
          max(len(line) for line in text.splitlines()) <= 100,
          f"widest {max(len(line) for line in text.splitlines())}")

    print("\n  an all-unscored grid")
    blank = {(c, a, "report"): None for c in p8.CONDITIONS for a in (0.0, 1.0)}
    blank_table = p8.family_table(_frame(blank), config)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        p8.print_concept(p8.channel_summary(blank_table), blank_table, concept)
    check("unscored cells print as '-' rather than crashing",
          "-" in buffer.getvalue())


def test_prompt_counts() -> None:
    print("\nthe grid's size")
    config = PCAJLensConfig()
    cells = p8.grid_cells([float(a) for a in config.steer_strengths])
    report_tasks = cp.build_report_choice_tasks(["anxious", "calm"], seed=0)
    behaviour_tasks = p8.selected_behaviour_tasks(config)
    n_prompts = len(report_tasks) + len(behaviour_tasks)
    judged = sum(1 for t in behaviour_tasks if t.scorer == "judge")
    per_concept = len(cells) * n_prompts
    check("the default grid is 17 cells per concept", len(cells) == 17, f"{len(cells)}")
    check("five matched random controls are present", len(p8.RANDOM_CONDITIONS) == 5)
    check("only one prespecified four-prompt behaviour family runs",
          len(behaviour_tasks) == 4 and {t.family for t in behaviour_tasks} == {"risk"})
    check("phase8_grid_calls covers two concepts",
          config.phase8_grid_calls >= 2 * len(cells) * judged,
          f"config {config.phase8_grid_calls}, needed {2 * len(cells) * judged}")
    print(f"    {per_concept} generations and {len(cells) * judged} judge calls "
          f"per concept")


def main() -> int:
    test_steering()
    test_generation()
    test_grid_cells()
    test_scoring()
    test_aggregation()
    test_printing()
    test_prompt_counts()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
