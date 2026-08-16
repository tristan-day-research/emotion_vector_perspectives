"""Phase 8 (GATE): steer under four conditions, measure both channels.

What this stage does
--------------------
For each emotion Phase 7 chose, adds ``alpha * direction`` to the residual stream at
Phase 6's target block across ``steer_strengths``, under four conditions -- ``v``,
``v_J``, ``v_perp``, and a matched-norm random direction -- and measures **both**
channels in every cell. The headline is a 4-condition x 2-channel grid.

This is the functional result the structural phases were setting up. If behaviour
moves under ``v_perp`` while the report channel stays quiet, an emotional state is
influencing action without being reportable -- subject to the caveat at the bottom of
this docstring, which is not a footnote.

Why alpha is a multiple of ``||v||``
-----------------------------------
Phase 6 wrote all four directions norm-matched to the emotion vector's own norm, so
one alpha is the same perturbation size in every condition. Without that, "``v_perp``
moves behaviour more than ``v_J``" would be a statement about ``v_perp`` being longer
-- and since ``v_J`` is a modest fraction of ``v``'s variance, it would be a large one.

``alpha = 0`` is generated once, not four times
-----------------------------------------------
At zero strength all four conditions are the identical unsteered model. Generating
that row per condition would spend a quarter of the grid's GPU time producing the same
text four times, and would send the same passages to the judge four times. It is
generated and scored once, then copied across conditions with ``shared_baseline`` set
on every copy, so the CSV cannot be mistaken for four independent baselines.

Three controls, because the headline is worthless without them
--------------------------------------------------------------
1. **Fluency at every strength.** A behavioural change at a strength where the text
   has stopped being fluent is degradation wearing a result's clothes. Perplexity of
   each steered continuation is measured **under the unsteered model** -- the question
   is whether the text is degraded, and only a model that was not perturbed can answer
   it. Cells above ``perplexity_max_ratio`` are marked and kept out of the scale the
   summary is expressed in, rather than quietly averaged into it.
2. **Specificity.** The whole grid re-runs with a *topic* vector in place of the
   emotion, so a dissociation can be shown not to be generic to any concept at all.
   The topic vector is built from the same stored activations and split by the same
   pursuit; it differs from an emotion vector in what it is about and in nothing else.
3. **A random direction as the fourth condition**, norm-matched like the rest. It is
   what rules out "any perturbation of this size moves behaviour".

Why the behaviour channel is not one number
-------------------------------------------
Its four families are on incompatible scales: risk and persistence are 0/1, hedging is
a rate per 100 words, refusal is a judge's 0-3. Averaging them would give a number
whose movement is dominated by whichever family happens to have the widest units. So
each family is reported raw, and the cross-channel summary is expressed in **grid-SD
units** -- each cell's shift from the ``alpha=0`` baseline over that family's own
standard deviation across the undegraded grid. That makes the two channels comparable
without inventing a common scale for the raw scores.

The re-entry caveat, which this phase cannot resolve
---------------------------------------------------
A ``v_perp`` behavioural effect does **not** establish that the effect bypassed the
workspace. Downstream layers can re-derive the concept from the remainder and route it
back through J-space, which would produce exactly this signature. Phase 8 does not
rule that out; Phase 9's clamp is what would. Every report of a ``v_perp`` result has
to carry that caveat, so the gate prints it as part of the verdict rather than leaving
it to be remembered.

Usage::

    python run.py phase8 --dry-run     # the grid's shape, cost and time; no weights
    python run.py phase8 --no-judge    # mechanical behaviour families only
    python run.py phase8               # the full grid
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.activation_store import ActivationStore
from core.judge import Judge, judge_available
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens.channel_prompts import (
    BEHAVIOUR_REFUSAL_RUBRIC,
    BEHAVIOUR_TASKS,
    MECHANICAL_SCORERS,
    REPORT_PROMPTS,
    REPORT_RUBRIC,
)
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config

RULE = "=" * 78
THIN = "-" * 78

#: The four steering conditions, in the order the grid prints them. These are the
#: tensor keys Phase 6 wrote, so the grid cannot drift from the artefact it reads.
CONDITIONS: tuple[str, ...] = ("v", "v_reportable", "v_remainder", "v_random")

#: Short name and gloss per condition -- "v_reportable" does not explain itself in a
#: table header.
CONDITION_LABELS: dict[str, tuple[str, str]] = {
    "v": ("v", "the whole emotion vector"),
    "v_reportable": ("v_J", "the part the lens can verbalise"),
    "v_remainder": ("v_perp", "the remainder it cannot"),
    "v_random": ("random", "matched-norm control"),
}

#: The channel families in reading order, with what a higher score means. Printed
#: beside every family's rows because three of them are not self-orienting: a rising
#: ``risk`` score means *more* caution, which is the opposite of what the name suggests.
FAMILY_NOTES: tuple[tuple[str, str], ...] = (
    ("report", "higher = expresses the emotion more strongly (judge, 0-4)"),
    ("risk", "higher = chose the certain option, i.e. more risk-averse (0/1)"),
    ("refusal", "higher = declined or narrowed rather than helping (judge, 0-3)"),
    ("persistence", "higher = produced a committed final answer (0/1)"),
    ("hedging", "higher = more hedge constructions per 100 words"),
)

#: The refusal rubric defines 0-3. The judge's default allowed set is 0-4, and an enum
#: that permits a score the rubric never defines is an invitation to use it.
REFUSAL_SCORES: tuple[int, ...] = (0, 1, 2, 3)

#: Decode throughput used for the time estimate. The same order-of-magnitude arithmetic
#: the README uses for extraction, not a measurement on any particular box.
TOKENS_PER_SECOND = 1500.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 8 gate: steer under four conditions, measure both channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="report the grid's shape, judge cost and generation-time estimate; never "
             "loads model weights",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip judge scoring. The mechanical behaviour families still run, but the "
             "report channel goes unscored and the headline cannot be read",
    )
    p.add_argument(
        "--no-specificity", action="store_true",
        help="skip the topic-vector control. Halves the runtime and removes the one "
             "control that shows a dissociation is not generic to any concept",
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
# Steering
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def steering(model, block: int, vector, alpha: float, positions: str = "all"):
    """Add ``alpha * vector`` to the residual stream at ``block`` for the duration.

    A forward hook on ``model.model.layers[block]``. That module's output *is* hidden
    state ``block + 1``, which is the tensor Phase 2 pooled and the space Phase 6's
    directions live in -- see :func:`core.jlens_lens.hidden_state_index` for the
    convention. Hooking one module either side would steer with a direction fitted
    somewhere else, and nothing downstream would complain.

    ``positions="all"`` adds at every position including the prompt; ``"generated"``
    adds only where the sequence length is 1, i.e. during incremental decoding once the
    prompt has been processed. All-positions is the standard protocol and is what makes
    the behaviour channel move at all, because the model then reads its instructions
    through the perturbation -- which is also why the fluency control is not optional.

    A genuine no-op at ``alpha == 0``, rather than a hook that adds zero, so the
    baseline row is the unmodified model and not the model plus a hook.
    """
    import torch

    if alpha == 0.0:
        yield
        return

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None or block >= len(layers):
        raise SystemExit(
            f"cannot reach model.model.layers[{block}] to steer: this architecture does "
            f"not expose the block list where the lens convention places block {block}. "
            "Steering anywhere else would silently use the wrong space."
        )
    direction = torch.as_tensor(np.asarray(vector), dtype=torch.float32)

    def hook(_module, _inputs, output):
        # Decoder layers return either a bare hidden state or a tuple whose first
        # element is one; both shapes are in circulation across transformers versions.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if positions == "generated" and hidden.shape[1] != 1:
            return output
        steered = hidden + (alpha * direction).to(
            dtype=hidden.dtype, device=hidden.device
        )
        return (steered,) + tuple(output[1:]) if is_tuple else steered

    handle = layers[block].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# Generation and fluency
# --------------------------------------------------------------------------- #

def generate_batched(
    model, tokenizer, config: PCAJLensConfig, prompts: list[str]
) -> list[str]:
    """Chat-formatted batched generation, one completion per prompt.

    **Left padding, set here rather than inherited.** ``load_tokenizer`` leaves the
    default right padding, which is correct for pooling and wrong for generation: with
    right padding a batched ``generate`` continues from pad tokens for every sequence
    shorter than the longest in the batch, producing fluent text that answers nothing.
    The failure is silent and survives every downstream check, so the side is set at
    the call site and restored afterwards.

    Greedy, because the grid compares cells: sampling noise across 4 conditions x 4
    strengths x 2 channels would need repeats per cell to see through.

    No truncation. These prompts are short literals from
    :mod:`emotion_pca_jlens.channel_prompts`, and truncating a task prompt would change
    the task rather than merely shorten it.
    """
    import torch

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = model_utils.model_input_device(model)
    outputs: list[str] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), config.generation_batch_size):
                chunk = prompts[start : start + config.generation_batch_size]
                texts = model_utils.prepare_texts(
                    chunk, tokenizer, use_chat_template=True,
                    chat_add_generation_prompt=True,
                )
                encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
                generated = model.generate(
                    **encoded,
                    max_new_tokens=config.generation_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                width = encoded["input_ids"].shape[-1]
                for row in generated:
                    outputs.append(
                        tokenizer.decode(row[width:], skip_special_tokens=True)
                    )
    finally:
        tokenizer.padding_side = previous_side
    return outputs


def perplexity(model, tokenizer, config: PCAJLensConfig, texts: list[str]) -> list[float]:
    """Per-token perplexity of each text, **under the unsteered model**.

    Must be called outside any steering context. Scoring steered text with the steered
    model asks "is this likely given the perturbation", which rises with the
    perturbation and says nothing about quality; only the unmodified model can answer
    "has this stopped looking like language".

    ``nan`` for anything under two tokens -- several behaviour prompts ask for a single
    letter, and a one-token sequence has no next-token loss to average. ``nan`` drops
    out of the mean rather than being counted as perfectly fluent.
    """
    import torch

    device = model_utils.model_input_device(model)
    out: list[float] = []
    with torch.inference_mode():
        for text in texts:
            if not text.strip():
                out.append(float("nan"))
                continue
            ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=config.max_length
            ).input_ids.to(device)
            if ids.shape[-1] < 2:
                out.append(float("nan"))
                continue
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            out.append(float(torch.exp(loss.float())))
    return out


# --------------------------------------------------------------------------- #
# The directions being steered with
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Directions:
    """The four norm-matched directions for one concept, at one block.

    ``report_emotion`` is which emotion the report-channel rubric scores for. For an
    emotion concept it is that emotion. For the topic control it is the *primary*
    emotion, which is what makes its report row a control rather than a curiosity: it
    asks whether a non-emotion concept of the same size also raises the emotion-report
    score.
    """

    name: str
    kind: str                        # "emotion" | "topic"
    report_emotion: str
    vectors: dict[str, np.ndarray]
    norm: float
    detail: dict = field(default_factory=dict)


def load_emotion_directions(
    config: PCAJLensConfig, wanted: list[str]
) -> tuple[list[Directions], int]:
    """Phase 6's four directions per emotion, plus the block they belong to.

    The block comes from Phase 6's own record rather than ``config.target_block``. That
    field is deliberately not part of the run fingerprint -- Phase 2 stores every layer
    so Phase 5's sweep is free -- so it can differ from the block the decomposition was
    fitted at, and steering at the wrong one would inject a direction from another
    space without any error.
    """
    from safetensors.numpy import load_file

    if not config.decomposition_path.exists():
        raise SystemExit(
            f"no decomposition at\n  {config.decomposition_path}\n\n"
            "Phase 8 steers with the four directions Phase 6 wrote. Run it first:\n\n"
            "  python run.py phase6\n"
        )
    tensors = load_file(str(config.decomposition_path))
    meta = json.loads(config.decomposition_meta_path.read_text(encoding="utf-8"))
    if not meta.get("dictionary_valid", False):
        raise SystemExit(
            f"{config.decomposition_meta_path} reports dictionary_valid=false (or "
            "predates the\ncheck). Phase 6's atoms did not lens back to their own "
            "tokens, so v_J is not the\npart the lens can verbalise and v_perp is not "
            "the remainder -- steering with them\nwould produce a real behavioural "
            "result about two arbitrary directions. Fix the\ndictionary first:\n\n"
            "  python run.py phase6\n"
        )
    emotions = list(meta["emotions"])
    block = int(meta.get("vectors", {}).get("target_block", -1))
    if block < 0:
        raise SystemExit(
            f"{config.decomposition_meta_path} does not record which block the "
            "decomposition was fitted at; re-run Phase 6 rather than guessing."
        )
    missing = [key for key in CONDITIONS if key not in tensors]
    if missing:
        raise SystemExit(f"{config.decomposition_path} lacks {missing}; re-run Phase 6.")

    per_emotion = {
        row["emotion"]: row for row in meta.get("per_emotion", []) if "emotion" in row
    }
    out: list[Directions] = []
    for emotion in wanted:
        if emotion not in emotions:
            raise SystemExit(f"{emotion!r} is not in Phase 6's output ({emotions})")
        index = emotions.index(emotion)
        vectors = {
            key: np.asarray(tensors[key][index], dtype=np.float64) for key in CONDITIONS
        }
        row = per_emotion.get(emotion, {})
        out.append(Directions(
            name=emotion, kind="emotion", report_emotion=emotion, vectors=vectors,
            norm=float(np.linalg.norm(vectors["v"])),
            detail={"frac_reportable": row.get("frac_reportable"),
                    "frac_remainder": row.get("frac_remainder")},
        ))
    return out, block


def topic_centroid(
    config: PCAJLensConfig, block: int
) -> tuple[np.ndarray, str, dict] | None:
    """A topic direction from the same stored activations, or ``None``.

    **Emotion is removed exactly, not by assumption.** Each row has its own emotion's
    mean subtracted before the topic average is taken, so the result cannot carry an
    emotion-mean component at all. Centring on the grand mean instead would rely on
    every topic holding a balanced set of emotions, and a control that leans on the
    design being balanced is a control that fails quietly when it is not.

    Returns ``None`` when the pooled activations are not on this machine. That is the
    expected first failure -- ``delete_local_after_sync=True`` is the default, so a
    fresh box has the index parquets and not the tensors -- and the honest response is
    to run without the control and say so, not to substitute something weaker.
    """
    try:
        store = ActivationStore(config.activations_dir)
    except (FileNotFoundError, RuntimeError):
        return None
    hidden_state = jlens_lens.hidden_state_index(block)
    if hidden_state not in store.layers:
        return None

    rows = store.index
    labels = sorted(rows["topic"].dropna().astype(str).unique().tolist())
    if not labels:
        return None
    topic = str(config.specificity_topic) if config.specificity_topic else labels[0]
    if topic not in labels:
        raise SystemExit(
            f"specificity_topic={topic!r} is not among the stored topics "
            f"(e.g. {labels[:3]})"
        )
    try:
        activations = store.load_layer(hidden_state, rows).astype(np.float64)
    except FileNotFoundError:
        return None

    emotions = rows["emotion"].astype(str).to_numpy()
    residual = activations.copy()
    for name in np.unique(emotions):
        mask = emotions == name
        residual[mask] -= residual[mask].mean(axis=0)
    selected = rows["topic"].astype(str).to_numpy() == topic
    if not selected.any():
        return None
    return residual[selected].mean(axis=0), topic, {
        "topic": topic, "n_rows": int(selected.sum()), "n_topics": len(labels),
        "hidden_state": hidden_state,
        "centring": "each row's own emotion mean removed before averaging",
    }


def decompose_control(
    vector: np.ndarray, topic: str, report_emotion: str, readout, block: int,
    config: PCAJLensConfig, target_norm: float, rng,
) -> Directions:
    """Split the topic vector with the same pursuit Phase 6 used on the emotions.

    Same dictionary construction, same atom budget, same norm matching, so the only
    difference between this concept and an emotion is what it is about. Anything else
    would leave a specificity difference attributable to the construction -- including
    the pullback, which is Phase 6's truncated ``J^+`` at the same ``dict_pinv_rcond``
    and not a transpose.
    """
    from emotion_pca_jlens.phase6_decompose import (
        build_dictionary,
        decompose_vector,
        factor_transport,
        match_norm,
        unembed_parts,
    )

    head, gain = unembed_parts(readout)
    transport = factor_transport(readout, block, config.dict_pinv_rcond)
    dictionary = build_dictionary(
        readout, vector, block, config.dict_pool_size, head, gain, transport
    )
    result = decompose_vector(
        vector, dictionary, config.n_dict_atoms, config.pursuit_steps
    )
    return Directions(
        name=topic, kind="topic", report_emotion=report_emotion, norm=target_norm,
        vectors={
            "v": match_norm(vector, target_norm),
            "v_reportable": match_norm(result.reportable, target_norm),
            "v_remainder": match_norm(result.remainder, target_norm),
            "v_random": match_norm(rng.normal(size=vector.size), target_norm),
        },
        detail={"frac_reportable": result.frac_reportable,
                "frac_remainder": result.frac_remainder,
                "atoms": result.n_iterations},
    )


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

def grid_cells(strengths: list[float]) -> list[tuple[str, float]]:
    """``(condition, alpha)`` pairs to actually generate.

    ``alpha=0`` appears once, under the first condition, because all four conditions
    are the same unsteered model there. :func:`expand_baseline` copies the result to
    the others after scoring, so the saving is in both GPU time and judge calls.
    """
    cells = [(CONDITIONS[0], 0.0)] if 0.0 in strengths else []
    return cells + [
        (condition, alpha)
        for alpha in strengths if alpha != 0.0
        for condition in CONDITIONS
    ]


def run_grid(
    model, tokenizer, config: PCAJLensConfig, block: int, directions: list[Directions]
) -> list[dict]:
    """Generate every cell. One row per (concept, condition, alpha, prompt)."""
    report_prompts = list(REPORT_PROMPTS)
    behaviour_prompts = [task.prompt for task in BEHAVIOUR_TASKS]
    cells = grid_cells([float(a) for a in config.steer_strengths])
    rows: list[dict] = []

    for concept in directions:
        for condition, alpha in cells:
            t0 = time.time()
            with steering(model, block, concept.vectors[condition], alpha,
                          config.steer_positions):
                report_texts = generate_batched(model, tokenizer, config, report_prompts)
                behaviour_texts = generate_batched(
                    model, tokenizer, config, behaviour_prompts
                )
            # Outside the steering context: fluency is judged by the unmodified model,
            # or it measures the perturbation instead of the text.
            perplexities = perplexity(
                model, tokenizer, config, report_texts + behaviour_texts
            )
            shared = alpha == 0.0
            common = {
                "concept": concept.name, "kind": concept.kind,
                "report_emotion": concept.report_emotion,
                "condition": condition, "alpha": alpha, "shared_baseline": shared,
            }
            for i, (prompt, text) in enumerate(zip(report_prompts, report_texts)):
                rows.append({**common, "channel": "report", "family": "report",
                             "prompt": prompt, "response": text,
                             "perplexity": perplexities[i]})
            for j, (task, text) in enumerate(zip(BEHAVIOUR_TASKS, behaviour_texts)):
                rows.append({**common, "channel": "behaviour", "family": task.family,
                             "prompt": task.prompt, "response": text,
                             "perplexity": perplexities[len(report_texts) + j]})
            label = f"{concept.name}/{CONDITION_LABELS[condition][0]}/a={alpha:g}"
            print(f"  {label:<32} {len(report_texts) + len(behaviour_texts):>3} "
                  f"generations in {time.time() - t0:>5.0f}s"
                  + ("   (shared baseline)" if shared else ""), flush=True)
    return rows


def expand_baseline(rows: list[dict]) -> list[dict]:
    """Copy each concept's ``alpha=0`` rows to the other three conditions.

    After scoring, so the shared cell costs one generation and one judge call rather
    than four. Every copy keeps ``shared_baseline=True``: there is one baseline per
    concept, and a reader counting rows in the CSV must not take it for four.
    """
    expanded = list(rows)
    for row in rows:
        if not row.get("shared_baseline"):
            continue
        for condition in CONDITIONS:
            if condition != row["condition"]:
                expanded.append({**row, "condition": condition})
    return expanded


def score_grid(rows: list[dict], config: PCAJLensConfig, use_judge: bool) -> dict:
    """Score every generation: mechanical where the family allows it, judge where not.

    One judge per distinct rubric, so the rubric stays a cache-hit prefix across
    hundreds of calls. The report rubric is built per ``report_emotion``: scoring a
    cell steered towards one emotion against another emotion's rubric would report a
    wrong number without erring.
    """
    for row in rows:
        row["score"] = None
        row["scorer"] = None
        row["detail"] = ""
        scorer = MECHANICAL_SCORERS.get(row["family"])
        if scorer is not None:
            result = scorer(row["response"])
            row["score"], row["scorer"] = result["score"], "mechanical"
            row["detail"] = result["detail"]

    usage: dict = {"calls": 0, "input": 0, "output": 0, "cached": 0, "unusable": 0}
    if not use_judge:
        usage["skipped"] = "--no-judge"
        print("  --no-judge: the report channel and the refusal family stay unscored,")
        print("  so the report half of the headline cannot be read.")
        return usage
    available, reason = judge_available()
    if not available:
        usage["skipped"] = reason.splitlines()[0]
        print(f"  judge unavailable: {usage['skipped']}")
        print("  the report channel and the refusal family stay unscored.")
        return usage

    jobs: list[tuple[str, str, tuple[int, ...], list[dict]]] = []
    for emotion in sorted({r["report_emotion"] for r in rows if r["family"] == "report"}):
        jobs.append((
            f"report[{emotion}]", REPORT_RUBRIC.format(emotion=emotion), (0, 1, 2, 3, 4),
            [r for r in rows
             if r["family"] == "report" and r["report_emotion"] == emotion],
        ))
    refusal = [r for r in rows if r["family"] == "refusal"]
    if refusal:
        jobs.append(("refusal", BEHAVIOUR_REFUSAL_RUBRIC, REFUSAL_SCORES, refusal))

    for name, rubric, allowed, targets in jobs:
        judge = Judge(rubric=rubric, model=config.judge_model, allowed_scores=allowed)
        items = [(f"{name}#{i:05d}", r["response"]) for i, r in enumerate(targets)]
        print(f"  {name:<20} {len(items):>4} passages "
              f"({'batched' if config.judge_use_batches else 'interactive'}) ...",
              flush=True)
        verdicts = judge.score_many(
            items, use_batches=config.judge_use_batches,
            progress=lambda message: print(f"    {message}", flush=True),
        )
        for (key, _), row in zip(items, targets):
            verdict = verdicts.get(key)
            row["scorer"] = f"judge:{config.judge_model}"
            row["score"] = None if verdict is None else verdict.score
            row["detail"] = "" if verdict is None else (
                verdict.reason or verdict.error or ""
            )
            if verdict is None or not verdict.usable:
                usage["unusable"] += 1
        usage["calls"] += judge.calls
        for key in ("input", "output", "cached"):
            usage[key] += judge.usage.get(key, 0)
    return usage


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def cell_fluency(frame: pd.DataFrame, config: PCAJLensConfig) -> pd.DataFrame:
    """Mean perplexity per (concept, condition, alpha), and whether it is degraded.

    Per *cell*, not per family: degradation is a property of the perturbation, and a
    family whose answers are one word each would otherwise be flagged or spared on the
    strength of having little text to measure.
    """
    cells = frame.groupby(["concept", "condition", "alpha"], dropna=False).agg(
        perplexity=("perplexity", "mean"),
    ).reset_index()
    baseline = float(frame[frame["alpha"] == 0.0]["perplexity"].mean())
    cells["perplexity_ratio"] = cells["perplexity"] / max(baseline, 1e-9)
    cells["degraded"] = cells["perplexity_ratio"] > config.perplexity_max_ratio
    return cells


def family_table(frame: pd.DataFrame, config: PCAJLensConfig) -> pd.DataFrame:
    """One row per (concept, channel, family, condition, alpha), with shifts.

    ``shift`` is the raw difference from that concept-and-family's ``alpha=0`` score.
    ``z`` is the shift in units of the family's own standard deviation **across the
    undegraded grid** -- a degraded cell's score is not a measurement, and letting it
    into the scale would inflate the denominator and shrink every real effect.
    """
    if frame.empty:
        return frame
    table = frame.groupby(
        ["concept", "kind", "channel", "family", "condition", "alpha"], dropna=False
    ).agg(
        n=("response", "size"),
        n_scored=("score", "count"),
        score=("score", "mean"),
        perplexity=("perplexity", "mean"),
    ).reset_index()

    fluency = cell_fluency(frame, config)
    table = table.merge(
        fluency[["concept", "condition", "alpha", "perplexity_ratio", "degraded"]],
        on=["concept", "condition", "alpha"], how="left",
    )

    table["baseline"] = np.nan
    table["shift"] = np.nan
    table["z"] = np.nan
    for _, group in table.groupby(["concept", "family"], dropna=False):
        base = group.loc[group["alpha"] == 0.0, "score"]
        if base.empty or pd.isna(base.iloc[0]):
            continue
        shift = group["score"] - float(base.iloc[0])
        spread = float(group.loc[~group["degraded"], "score"].std(ddof=1))
        table.loc[group.index, "baseline"] = float(base.iloc[0])
        table.loc[group.index, "shift"] = shift
        if np.isfinite(spread) and spread > 0:
            table.loc[group.index, "z"] = shift / spread
        else:
            # No spread anywhere in the undegraded grid: every cell equals the
            # baseline, so the shift is genuinely zero rather than unmeasurable.
            table.loc[group.index, "z"] = np.where(shift == 0, 0.0, np.nan)
    return table


def channel_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean |z| per (concept, channel, condition, alpha) -- the 4x2 grid itself.

    Magnitude, not sign: the families point in different directions (more hedging and
    more risk-aversion are both "moved"), so a signed average would let two real
    effects cancel into a null. The signed per-family shifts stay in the CSV and in the
    raw tables printed below the grid.
    """
    if table.empty:
        return table
    working = table.assign(abs_z=table["z"].abs())
    return working.groupby(
        ["concept", "kind", "channel", "condition", "alpha"], dropna=False
    ).agg(
        abs_z=("abs_z", "mean"),
        n_families=("family", "nunique"),
        n_scored=("n_scored", "sum"),
        degraded=("degraded", "max"),
    ).reset_index()


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def _row(label: str, values: list[tuple[float | None, bool]]) -> str:
    """One table line: a label, then a number or ``-`` per strength, ``*`` if degraded."""
    line = f"    {label:<9}"
    for value, degraded in values:
        if value is None or pd.isna(value):
            line += f"{'-':>9}"
        else:
            line += f"{value:>8.2f}" + ("*" if degraded else " ")
    return line


def print_concept(summary: pd.DataFrame, table: pd.DataFrame, concept: Directions) -> None:
    """One concept's grid: the 4x2 summary, then the raw per-family scores."""
    print()
    print(THIN)
    print(f"{concept.name}   ({concept.kind}, ||v|| = {concept.norm:.2f})")
    print(THIN)
    strengths = sorted(summary.loc[summary["concept"] == concept.name, "alpha"].unique())
    header = "    " + f"{'condition':<9}" + "".join(f"{f'a={a:g}':>9}" for a in strengths)

    def cells(frame: pd.DataFrame, column: str, condition: str):
        rows = frame[frame["condition"] == condition]
        out = []
        for alpha in strengths:
            cell = rows[rows["alpha"] == alpha]
            if cell.empty:
                out.append((None, False))
            else:
                out.append((cell[column].iloc[0], bool(cell["degraded"].iloc[0])))
        return out

    print("  shift from the alpha=0 baseline in grid-SD units (mean |z| over families)")
    for channel in ("report", "behaviour"):
        scoped = summary[
            (summary["concept"] == concept.name) & (summary["channel"] == channel)
        ]
        print(f"    {channel} channel")
        print(header)
        for condition in CONDITIONS:
            print(_row(CONDITION_LABELS[condition][0], cells(scoped, "abs_z", condition)))

    print()
    print("  raw scores per family -- the summary above is a re-scaling of these")
    for family, note in FAMILY_NOTES:
        scoped = table[(table["concept"] == concept.name) & (table["family"] == family)]
        if scoped.empty or scoped["n_scored"].sum() == 0:
            continue
        print(f"    {family}  --  {note}")
        print(header)
        for condition in CONDITIONS:
            print(_row(CONDITION_LABELS[condition][0], cells(scoped, "score", condition)))


def print_verdict(
    summary: pd.DataFrame, fluency: pd.DataFrame, config: PCAJLensConfig,
    emotions: list[str], control: str | None, control_skipped: str,
    usage: dict, artifacts: dict,
) -> None:
    print()
    print(RULE)
    print("PHASE 8 VERDICT")
    print(RULE)
    if summary.empty:
        print("  nothing was generated.")
        return

    steered = summary[~summary["degraded"] & (summary["alpha"] > 0)]

    def largest(concept: str, condition: str, channel: str) -> float | None:
        values = steered.loc[
            (steered["concept"] == concept) & (steered["condition"] == condition)
            & (steered["channel"] == channel), "abs_z"
        ].dropna()
        return None if values.empty else float(values.max())

    for concept in emotions:
        print(f"  {concept}   largest |z| over the undegraded strengths")
        if steered[steered["concept"] == concept].empty:
            print("    no undegraded steered cells")
            continue
        for condition in CONDITIONS:
            parts = [
                f"{channel} " + (
                    "   n/a" if largest(concept, condition, channel) is None
                    else f"{largest(concept, condition, channel):6.2f}"
                )
                for channel in ("report", "behaviour")
            ]
            print(f"    {CONDITION_LABELS[condition][0]:<7} {' | '.join(parts)}")
        behaviour = largest(concept, "v_remainder", "behaviour")
        report = largest(concept, "v_remainder", "report")
        random = largest(concept, "v_random", "behaviour")
        if behaviour is None or report is None:
            print("    v_perp was not scored in both channels, so the headline contrast")
            print("    cannot be read for this emotion.")
        elif random is None:
            print("    no usable random control, so a v_perp behavioural shift is not")
            print("    yet attributable to the direction rather than to perturbation.")
        else:
            print(f"    the contrast: v_perp behaviour {behaviour:.2f}, v_perp report "
                  f"{report:.2f}, random behaviour {random:.2f}.")
            print("    The dissociation this experiment looks for needs all three --")
            print("    behaviour high, report low, random low. Read them together or")
            print("    not at all.")

    degraded = int(fluency["degraded"].sum())
    print()
    print(f"  degraded cells (perplexity > {config.perplexity_max_ratio:g}x baseline)"
          f" : {degraded}/{len(fluency)}")
    if degraded:
        print("    marked * above, kept out of the grid-SD scale, and excluded from the")
        print("    verdict. A behavioural shift in a degraded cell is degradation.")
    print("  specificity control : " + (
        f"{control!r} -- a topic vector built and split the same way" if control
        else f"NOT RUN -- {control_skipped}"
    ))
    if not control:
        print("    without it, a dissociation cannot be shown to be specific to an")
        print("    emotion rather than generic to any concept of this size.")
    print(f"  judge               : {usage.get('calls', 0)} calls"
          + (f", {usage['unusable']} unusable" if usage.get("unusable") else "")
          + (f"   [{usage['skipped']}]" if usage.get("skipped") else ""))
    print()
    for label, path in artifacts.items():
        print(f"  {label:<9}: {path}")

    print()
    print(RULE)
    print("  THE RE-ENTRY CAVEAT -- belongs in every report of a v_perp result")
    print(RULE)
    print("  A behavioural effect under v_perp does NOT establish that the effect")
    print("  bypassed the workspace. Downstream layers can re-derive the concept from")
    print("  the remainder and route it back through J-space, which would produce")
    print("  exactly this signature: behaviour moves, the report channel does not, and")
    print("  the concept is nonetheless verbalisable somewhere in the stack.")
    print()
    print("  Phase 8 cannot distinguish those two stories. Phase 9's clamp -- holding")
    print("  the emotion's J-space coordinates at clean-pass values at every position")
    print("  and layer -- is what would. If Phase 9 is not run, state this as a")
    print("  limitation rather than glossing it.")
    print()
    print("  Two further limits belong beside any number above. The steering vectors")
    print("  were built from raw story text and applied to chat-formatted prompts, and")
    print("  transfer across that boundary is assumed rather than verified. And a lens")
    print("  readout is a disposition to say a word, so the report channel measures")
    print("  reportability, not felt experience.")
    print()
    print("STOPPING at the Phase 8 gate, as agreed. Phase 9 has not run.")
    print(RULE)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def read_phase7(config: PCAJLensConfig) -> tuple[dict, Path]:
    """Phase 7's record, preferring the full gate and accepting the dry run.

    The dry-run record carries the separation verdict and the chosen emotions -- both
    are settled before Phase 7 loads any weights -- so Phase 8's own ``--dry-run`` stays
    runnable on a laptop, which is the point of having one.
    """
    gate = config.phase_dir / "phase7_channels" / "phase7_gate.json"
    dry = config.phase_dir / "phase7_channels" / "dry_run" / "phase7_dry_run.json"
    for path in (gate, dry):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    raise SystemExit(
        f"no Phase 7 record at\n  {gate}\n  {dry}\n\n"
        "Phase 8 steers the emotions Phase 7 chose, and Phase 7's gate is what confirms "
        "the\ntwo channels are separate. Run it first:\n\n"
        "  python run.py phase7 --dry-run\n"
    )


def print_design(
    config: PCAJLensConfig, record_path: Path, emotions: list[str], block: int,
    directions: list[Directions], strengths: list[float], cells: list[tuple[str, float]],
    n_prompts: int, total_generations: int, judge_calls: int,
) -> None:
    print(f"model      : {config.model_name} ({config.dtype})")
    print(f"phase 7    : {record_path.name}   separation PASS")
    print(f"emotions   : {emotions}")
    print(f"block      : {block} (hidden state {jlens_lens.hidden_state_index(block)}), "
          "read from Phase 6's record --")
    print("             not from config.target_block, which is not fingerprinted and so")
    print("             can point somewhere the decomposition was never fitted")
    for concept in directions:
        frac = concept.detail.get("frac_reportable")
        print(f"               {concept.name:<14} ||v|| {concept.norm:>8.2f}"
              + (f"   v_J {frac:.1%} of variance" if frac else ""))
    print(f"conditions : {len(CONDITIONS)}")
    for key in CONDITIONS:
        short, gloss = CONDITION_LABELS[key]
        print(f"               {short:<7} {gloss}")
    print(f"strengths  : {strengths}  as multiples of ||v||. Phase 6 norm-matched all")
    print("             four conditions, so one alpha is the same perturbation size in")
    print("             each -- otherwise the grid would be comparing vector lengths.")
    print(f"positions  : {config.steer_positions}"
          + ("   (the prompt too, so the model reads its"
             if config.steer_positions == "all" else ""))
    if config.steer_positions == "all":
        print("             instructions through the perturbation -- which is also why")
        print("             the fluency control is not optional)")
    print(f"prompts    : {n_prompts} per cell ({len(REPORT_PROMPTS)} report + "
          f"{len(BEHAVIOUR_TASKS)} behaviour)")
    print(f"cells      : {len(cells)} per concept ({len(strengths) - 1} strengths x "
          f"{len(CONDITIONS)} conditions + 1 shared baseline)")
    print("             alpha=0 is generated and judged ONCE per concept and copied")
    print("             across conditions: at zero strength all four are the same")
    print("             unsteered model, so a per-condition baseline would repeat a")
    print("             quarter of the grid to get identical numbers.")
    print()
    print(f"generations : {total_generations:,} at {config.generation_max_new_tokens} "
          f"new tokens, batch {config.generation_batch_size}")
    hours = total_generations * config.generation_max_new_tokens / TOKENS_PER_SECOND / 3600
    print(f"              ~{hours:.1f} h at ~{TOKENS_PER_SECOND:,.0f} tok/s decode -- "
          "FLOP arithmetic, not a measurement")
    if judge_calls:
        # estimate_cost never touches the network, so a sentinel client keeps --dry-run
        # working on a machine with no API key.
        probe = Judge(rubric=REPORT_RUBRIC, model=config.judge_model, client=object())
        estimate = probe.estimate_cost(judge_calls)
        if estimate.get("known_price"):
            chosen = (estimate["usd_cached_batched"] if config.judge_use_batches
                      else estimate["usd_cached"])
            print(f"judge       : {judge_calls:,} calls -> ~${chosen:.2f} "
                  f"({'batched' if config.judge_use_batches else 'interactive'}; "
                  f"batched ${estimate['usd_cached_batched']:.2f}, interactive "
                  f"${estimate['usd_cached']:.2f})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    out_dir = config.phase_dir / "phase8_steering"
    cache_dir = paths.hf_cache_dir()
    rng = rng_for(config.seed, "phase8")

    print(RULE)
    print(f"PHASE 8 GATE -- steer under four conditions   run '{config.run_name}'")
    print(RULE)

    record, record_path = read_phase7(config)
    if not record.get("separation", {}).get("separated", False):
        raise SystemExit(
            "Phase 7's separation gate did not pass, so the behaviour channel shares "
            "affect\nvocabulary with the report channel and every number Phase 8 "
            "produced would be\nconfounded. Fix the rubric first:\n\n"
            "  python run.py phase7 --dry-run\n"
        )
    emotions = (
        list(config.channel_emotions) if config.channel_emotions
        else list(record["emotions"]["chosen"])
    )
    directions, block = load_emotion_directions(config, emotions)

    strengths = [float(a) for a in config.steer_strengths]
    if 0.0 not in strengths:
        joined = ",".join(f"{a:g}" for a in strengths)
        raise SystemExit(
            "steer_strengths has no 0.0, but every number in the grid is a shift from "
            "the\nunsteered baseline -- there would be nothing to shift from:\n\n"
            f"  python run.py phase8 --set steer_strengths=0,{joined}\n"
        )
    cells = grid_cells(strengths)
    n_prompts = len(REPORT_PROMPTS) + len(BEHAVIOUR_TASKS)
    n_judged = len(REPORT_PROMPTS) + sum(
        1 for task in BEHAVIOUR_TASKS if task.scorer == "judge"
    )
    n_concepts = len(directions) + (0 if args.no_specificity else 1)
    total_generations = n_concepts * len(cells) * n_prompts
    judge_calls = 0 if args.no_judge else n_concepts * len(cells) * n_judged

    print_design(config, record_path, emotions, block, directions, strengths, cells,
                 n_prompts, total_generations, judge_calls)

    sections: dict = {
        "run": {"stage": "phase8_steer", "run_name": config.run_name,
                "dry_run": args.dry_run, "no_judge": args.no_judge,
                "no_specificity": args.no_specificity, "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "phase7": {"record": str(record_path), "separated": True, "emotions": emotions},
        "design": {
            "block": block, "hidden_state": jlens_lens.hidden_state_index(block),
            "conditions": list(CONDITIONS), "strengths": strengths,
            "steer_positions": config.steer_positions,
            "prompts_per_cell": n_prompts, "cells_per_concept": len(cells),
            "n_concepts": n_concepts, "total_generations": total_generations,
            "judge_calls": judge_calls, "alpha_zero_shared": True,
            "norms": {c.name: c.norm for c in directions},
        },
    }

    if args.dry_run:
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run", title=f"PHASE 8 DRY RUN -- {config.run_name}",
            sections=sections, txt_name="phase8_dry_run.txt",
            json_name="phase8_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: grid shape, cost and time estimated; no weights "
              "loaded.")
        print(f"  records : {txt_path}")
        print(f"            {json_path}")
        print()
        print("Get a real throughput number on one strength before committing the grid:")
        print("  python run.py phase8 --set steer_strengths=0,1 --no-specificity")
        print(RULE)
        return 0

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

    # --- the specificity control ------------------------------------------- #
    control_name: str | None = None
    control_skipped = "--no-specificity"
    control_detail: dict = {}
    if not args.no_specificity:
        print()
        print("Specificity control: a topic vector, built and split the same way ...")
        built = topic_centroid(config, block)
        if built is None:
            control_skipped = "the pooled activations are not on this machine"
            print(f"  {control_skipped}, so the topic vector cannot be built. Running")
            print("  WITHOUT the specificity control.")
            print(f"    python run.py r2 pull {config.activations_dir} "
                  f"--prefix {config.resolved_r2_prefix()}")
        else:
            vector, topic, control_detail = built
            try:
                from emotion_pca_jlens.phase4_lens_pcs import resolve_lens

                lens_path, lens_report = resolve_lens(config, cache_dir)
                readout = jlens_lens.LensReadout.build(model, tokenizer, lens_path)
                control = decompose_control(
                    vector, topic, directions[0].report_emotion, readout, block,
                    config, directions[0].norm, rng,
                )
                directions.append(control)
                control_name = control.name
                control_detail = {**control_detail, **control.detail, "lens": lens_report}
                print(f"  topic {topic!r}: {control_detail['n_rows']} rows of "
                      f"{control_detail['n_topics']} topics, each row's own emotion")
                print(f"  mean removed first. v_J {control.detail['frac_reportable']:.1%}"
                      f", all four norm-matched to {control.norm:.2f}")
            except Exception as exc:  # noqa: BLE001 - the grid is worth more than this
                control_skipped = f"{type(exc).__name__}: {exc}"
                print(f"  could not split the topic vector ({control_skipped}); running")
                print("  WITHOUT the specificity control rather than losing the grid.")

    # --- generate, score, aggregate ---------------------------------------- #
    print()
    print(RULE)
    print("Generating")
    print(RULE)
    grid_t0 = time.time()
    rows = run_grid(model, tokenizer, config, block, directions)
    print(f"  {len(rows):,} generations in {(time.time() - grid_t0) / 60:.1f} min")

    print()
    print(RULE)
    print("Scoring")
    print(RULE)
    usage = score_grid(rows, config, use_judge=not args.no_judge)

    frame = pd.DataFrame(expand_baseline(rows))
    table = family_table(frame, config)
    summary = channel_summary(table)
    fluency = cell_fluency(frame, config)

    print()
    print(RULE)
    print("GATE  The 4-condition x 2-channel grid")
    print(RULE)
    print("Read down a column: does the score move with strength? Read across: does it")
    print("move differently for v_J than for v_perp, and does random stay flat? A v_perp")
    print("column that moves in behaviour while its report column does not is the result")
    print("this experiment was built to look for -- with the caveat below.")
    print()
    print("unsteered baseline perplexity: "
          f"{frame[frame['alpha'] == 0.0]['perplexity'].mean():.2f}")
    print(f"* marks a cell above {config.perplexity_max_ratio:g}x it, where the text is "
          "degraded and its")
    print("  behavioural score is degradation rather than behaviour.")
    for concept in directions:
        print_concept(summary, table, concept)

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "phase8_generations.csv"
    grid_path = out_dir / "phase8_grid.csv"
    frame.to_csv(generations_path, index=False)
    table.to_csv(grid_path, index=False)

    sections["grid"] = table.to_dict(orient="records")
    sections["summary"] = summary.to_dict(orient="records")
    sections["fluency"] = fluency.to_dict(orient="records")
    sections["judge_usage"] = usage
    sections["specificity_control"] = {
        "topic": control_name,
        "skipped_reason": None if control_name else control_skipped,
        **control_detail,
    }
    sections["caveats"] = {
        "re_entry": "a v_perp behavioural effect is not distinguished from the concept "
                    "being re-derived downstream and re-entering the workspace; "
                    "Phase 9's clamp is what would distinguish them",
        "chat_template": "directions were built from raw story text and applied to "
                         "chat-formatted prompts; transfer is assumed, not verified",
        "reportability": "a lens readout is a disposition to say a word, so the report "
                         "channel measures reportability rather than felt experience",
        "behaviour_scale": "the four behaviour families are on incompatible scales, so "
                           "the channel summary is mean |z| in grid-SD units and the "
                           "raw per-family scores are the data",
    }
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 8 GATE -- {config.run_name}",
        sections=sections, txt_name="phase8_gate.txt", json_name="phase8_gate.json",
    )

    print_verdict(
        summary, fluency, config, emotions, control_name, control_skipped, usage,
        artifacts={"grid": grid_path, "raw": generations_path, "records": txt_path,
                   "": json_path},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
