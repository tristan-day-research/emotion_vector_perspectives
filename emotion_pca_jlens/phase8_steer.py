"""Phase 8 (GATE): steer whole/readable/remainder plus five random controls.

What this stage does
--------------------
For each emotion Phase 7 chose, adds ``alpha * direction`` to the residual stream at
Phase 6's target block at the preregistered positive strengths under ``v``, ``v_J``,
``v_perp``, and five matched-norm random directions, measuring both channels in every
cell.  The report channel reuses Phase 7's randomized exact-choice prompts; the
behaviour channel runs one prespecified four-prompt family.

This is the functional result the structural phases were setting up. If behaviour
moves under ``v_perp`` while the report channel stays quiet, an emotional state is
influencing action without being reportable -- subject to the caveat at the bottom of
this docstring, which is not a footnote.

Why alpha is a multiple of ``||v||``
-----------------------------------
Phase 6 wrote the three concept directions norm-matched to the emotion vector's own
norm, and Phase 8 norm-matches each random control, so
one alpha is the same perturbation size in every condition. Without that, "``v_perp``
moves behaviour more than ``v_J``" would be a statement about ``v_perp`` being longer
-- and since ``v_J`` is a modest fraction of ``v``'s variance, it would be a large one.

``alpha = 0`` is generated once
-------------------------------
At zero strength all conditions are the identical unsteered model. Generating
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
3. **Five random directions**, each norm-matched like the rest. They provide a small
   perturbation distribution rather than asking one noisy random vector to carry the
   entire control.

Why one behaviour family
------------------------
Searching four families and reporting whichever moved most is outcome selection.
Phase 8 therefore runs one prespecified family (risk by default) with all four prompt
variants. Other families require an explicit override and are exploratory.

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
    python run.py phase8 --no-judge    # enough for the default risk-family run
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
    Completion,
    ReportChoiceTask,
    completion_rate,
    invalid_rates,
    judge_precheck,
    score_report_choice,
)
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config

RULE = "=" * 78
THIN = "-" * 78

#: The four steering conditions, in the order the grid prints them. These are the
#: tensor keys Phase 6 wrote, so the grid cannot drift from the artefact it reads.
RANDOM_CONDITIONS: tuple[str, ...] = (
    "v_random", "v_random_2", "v_random_3", "v_random_4", "v_random_5",
)
CONDITIONS: tuple[str, ...] = (
    "v", "v_reportable", "v_remainder", *RANDOM_CONDITIONS,
)

#: Short name and gloss per condition -- "v_reportable" does not explain itself in a
#: table header.
CONDITION_LABELS: dict[str, tuple[str, str]] = {
    "v": ("v", "the whole emotion vector"),
    "v_reportable": ("v_J", "the part the lens can verbalise"),
    "v_remainder": ("v_perp", "the remainder it cannot"),
    "v_random": ("random", "matched-norm control"),
    "v_random_2": ("random2", "matched-norm control"),
    "v_random_3": ("random3", "matched-norm control"),
    "v_random_4": ("random4", "matched-norm control"),
    "v_random_5": ("random5", "matched-norm control"),
}

#: The channel families in reading order, with what a higher score means. Printed
#: beside every family's rows because three of them are not self-orienting: a rising
#: ``risk`` score means *more* caution, which is the opposite of what the name suggests.
FAMILY_NOTES: tuple[tuple[str, str], ...] = (
    ("report", "higher = selected the steered emotion (exact choice, 0/1)"),
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
        description="Phase 8 gate: steer under three concept and five random controls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="report the grid's shape, judge cost and generation-time estimate; never "
             "loads model weights",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip judge scoring. Exact report choices and mechanical behaviour still "
             "run; only a selected refusal family would be unavailable",
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
) -> list[Completion]:
    """Chat-formatted batched generation, one completion per prompt.

    **Left padding, set here rather than inherited.** ``load_tokenizer`` leaves the
    default right padding, which is correct for pooling and wrong for generation: with
    right padding a batched ``generate`` continues from pad tokens for every sequence
    shorter than the longest in the batch, producing fluent text that answers nothing.
    The failure is silent and survives every downstream check, so the side is set at
    the call site and restored afterwards.

    Greedy, because the grid compares cells: sampling noise across the conditions,
    strengths and channels would need repeats per cell to see through.

    ``enable_thinking`` is forwarded from the config and is **off** by default: Qwen3's
    template otherwise opens a ``<think>`` block, the budget goes on reasoning, and a
    truncated reasoning trace scores as if it were an answer. Returns
    :class:`Completion`, because whether the model stopped or ran out of budget is not
    recoverable from the text and every scorer needs it.

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
                    enable_thinking=config.enable_thinking,
                )
                encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
                generated = model.generate(
                    **encoded,
                    max_new_tokens=config.generation_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                width = encoded["input_ids"].shape[-1]
                stops = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
                for row in generated:
                    ids = [int(t) for t in row[width:]]
                    outputs.append(Completion(
                        text=tokenizer.decode(row[width:], skip_special_tokens=True),
                        # A run that stopped emitted a stop token; one that hit the cap
                        # did not. Right-padding is impossible here (the side is left), so
                        # a trailing pad token means generate() finished early.
                        finished=any(i in stops for i in ids),
                        n_new_tokens=len([i for i in ids if i not in stops]),
                    ))
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
    """The norm-matched concept and random directions at one block.

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


def add_random_controls(
    vectors: dict[str, np.ndarray], target_norm: float, rng,
) -> dict[str, np.ndarray]:
    """Return ``vectors`` with five independent matched-norm random controls.

    Phase 6 saved one random direction.  It is retained as the first control and four
    more are generated deterministically for Phase 8.  One random vector is a noisy
    anecdote; five give a small empirical perturbation distribution against which the
    remainder's behavioural effect can be compared.
    """
    out = dict(vectors)
    for key in RANDOM_CONDITIONS:
        if key in out:
            continue
        draw = np.asarray(rng.normal(size=next(iter(out.values())).size), dtype=np.float64)
        norm = float(np.linalg.norm(draw))
        if norm == 0:
            raise RuntimeError("sampled a zero random direction")
        out[key] = draw * (target_norm / norm)
    return out


def load_emotion_directions(
    config: PCAJLensConfig, wanted: list[str]
) -> tuple[list[Directions], int]:
    """Phase 6's directions plus five random controls and their block.

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
            "Phase 8 steers with the directions Phase 6 wrote. Run it first:\n\n"
            "  python run.py phase6\n"
        )
    tensors = load_file(str(config.decomposition_path))
    meta = json.loads(config.decomposition_meta_path.read_text(encoding="utf-8"))
    if meta.get("write_space"):
        raise SystemExit(
            f"{config.decomposition_meta_path} was written by a write_space run.\n\n"
            "Those v_J / v_perp are split by which directions most efficiently WRITE the "
            "tokens,\nnot by which the lens reads them with, so a behavioural "
            "dissociation between them\nwould not be about reportability. Re-run Phase 6 "
            "without the ablation:\n\n  python run.py phase6\n"
        )
    emotions = list(meta["emotions"])
    block = int(meta.get("vectors", {}).get("target_block", -1))
    if block < 0:
        raise SystemExit(
            f"{config.decomposition_meta_path} does not record which block the "
            "decomposition was fitted at; re-run Phase 6 rather than guessing."
        )
    source_conditions = ("v", "v_reportable", "v_remainder", "v_random")
    missing = [key for key in source_conditions if key not in tensors]
    if missing:
        raise SystemExit(f"{config.decomposition_path} lacks {missing}; re-run Phase 6.")

    saved_k = str(meta.get("saved_k", config.n_dict_atoms))
    per_emotion = {
        row["emotion"]: row for row in meta.get("per_emotion", []) if "emotion" in row
    }
    out: list[Directions] = []
    for emotion in wanted:
        if emotion not in emotions:
            raise SystemExit(f"{emotion!r} is not in Phase 6's output ({emotions})")
        index = emotions.index(emotion)
        vectors = {
            key: np.asarray(tensors[key][index], dtype=np.float64)
            for key in source_conditions
        }
        vector_norm = float(np.linalg.norm(vectors["v"]))
        vectors = add_random_controls(
            vectors, vector_norm, rng_for(config.seed, f"phase8-random:{emotion}")
        )
        # The saved k, not any k: the tensors above are the split at exactly that k, and
        # the fraction rises with k, so quoting another one would mislabel these vectors.
        row = per_emotion.get(emotion, {})
        at_k = row.get("per_k", {}).get(saved_k, row)
        out.append(Directions(
            name=emotion, kind="emotion", report_emotion=emotion, vectors=vectors,
            norm=vector_norm,
            detail={"frac_reportable": at_k.get("frac_reportable"),
                    "frac_remainder": at_k.get("frac_remainder"),
                    "p_value_vs_random": at_k.get("p_value"), "k": saved_k},
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

    Same read-direction atoms, same ``k``, same norm matching, so the only difference
    between this concept and an emotion is what it is about. Anything else would leave a
    specificity difference attributable to the construction. ``k`` is ``n_dict_atoms``,
    the one Phase 6 saved its tensors at -- the emotions Phase 8 steers with were split at
    that ``k``, so the control has to be too.
    """
    from emotion_pca_jlens.phase6_decompose import (
        build_dictionary,
        decompose_vector,
        match_norm,
        unembed_parts,
    )

    head, gain = unembed_parts(readout)
    dictionary = build_dictionary(
        readout, vector, block, config.dict_pool_size, head, gain
    )
    result = decompose_vector(
        vector, dictionary, config.n_dict_atoms, config.pursuit_steps
    )
    return Directions(
        name=topic, kind="topic", report_emotion=report_emotion, norm=target_norm,
        vectors=add_random_controls({
            "v": match_norm(vector, target_norm),
            "v_reportable": match_norm(result.reportable, target_norm),
            "v_remainder": match_norm(result.remainder, target_norm),
            "v_random": match_norm(rng.normal(size=vector.size), target_norm),
        }, target_norm, rng),
        detail={"frac_reportable": result.frac_reportable,
                "frac_remainder": result.frac_remainder,
                "atoms": result.n_iterations},
    )


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

def grid_cells(strengths: list[float]) -> list[tuple[str, float]]:
    """``(condition, alpha)`` pairs to actually generate.

    ``alpha=0`` appears once, under the first condition, because all conditions
    are the same unsteered model there. :func:`expand_baseline` copies the result to
    the others after scoring, so the saving is in both GPU time and judge calls.
    """
    cells = [(CONDITIONS[0], 0.0)] if 0.0 in strengths else []
    return cells + [
        (condition, alpha)
        for alpha in strengths if alpha != 0.0
        for condition in CONDITIONS
    ]


def selected_behaviour_tasks(config: PCAJLensConfig):
    """The prespecified behavioural family, with all four prompt variants."""
    tasks = tuple(
        task for task in BEHAVIOUR_TASKS
        if task.family == config.phase8_behaviour_family
    )
    if len(tasks) < 4:
        raise SystemExit(
            f"phase8_behaviour_family={config.phase8_behaviour_family!r} has only "
            f"{len(tasks)} prompts; at least four are required"
        )
    return tasks


def run_grid(
    model, tokenizer, config: PCAJLensConfig, block: int, directions: list[Directions],
    report_tasks: list[ReportChoiceTask], behaviour_tasks,
) -> list[dict]:
    """Generate every cell. One row per (concept, condition, alpha, prompt)."""
    report_prompts = [task.prompt for task in report_tasks]
    behaviour_prompts = [task.prompt for task in behaviour_tasks]
    cells = grid_cells([float(a) for a in config.steer_strengths])
    rows: list[dict] = []

    for concept in directions:
        for condition, alpha in cells:
            t0 = time.time()
            with steering(model, block, concept.vectors[condition], alpha,
                          config.steer_positions):
                report_out = generate_batched(model, tokenizer, config, report_prompts)
                behaviour_out = generate_batched(
                    model, tokenizer, config, behaviour_prompts
                )
            # Outside the steering context: fluency is judged by the unmodified model,
            # or it measures the perturbation instead of the text.
            perplexities = perplexity(
                model, tokenizer, config,
                [c.text for c in report_out] + [c.text for c in behaviour_out],
            )
            shared = alpha == 0.0
            common = {
                "concept": concept.name, "kind": concept.kind,
                "report_emotion": concept.report_emotion,
                "condition": condition, "alpha": alpha, "shared_baseline": shared,
            }
            for i, (task, out) in enumerate(zip(report_tasks, report_out)):
                rows.append({**common, "channel": "report", "family": "report",
                             "prompt": task.prompt, "response": out.text,
                             "report_variant": task.variant,
                             "report_mapping": json.dumps(task.mapping),
                             "finished": out.finished,
                             "n_new_tokens": out.n_new_tokens,
                             "perplexity": perplexities[i]})
            for j, (task, out) in enumerate(zip(behaviour_tasks, behaviour_out)):
                rows.append({**common, "channel": "behaviour", "family": task.family,
                             "prompt": task.prompt, "response": out.text,
                             "finished": out.finished,
                             "n_new_tokens": out.n_new_tokens,
                             "perplexity": perplexities[len(report_out) + j]})
            everything = list(report_out) + list(behaviour_out)
            done = completion_rate(everything)
            label = f"{concept.name}/{CONDITION_LABELS[condition][0]}/a={alpha:g}"
            print(f"  {label:<32} {len(everything):>3} "
                  f"generations in {time.time() - t0:>5.0f}s"
                  f"   {done['rate']:>4.0%} ended in EOS"
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
        row["valid"] = False
        row["reason"] = ""
        truncated = not bool(row.get("finished", True))
        if row["family"] == "report":
            mapping = json.loads(row["report_mapping"])
            task = ReportChoiceTask(
                variant=int(row["report_variant"]), prompt=row["prompt"],
                label_to_emotion=tuple(mapping.items()),
            )
            result = score_report_choice(
                row["response"], task, target_emotion=row["report_emotion"],
                truncated=truncated,
            )
            row["score"], row["scorer"] = result["score"], "exact-letter"
            row["detail"], row["valid"] = result["detail"], bool(result["valid"])
            row["reason"] = result.get("reason", "")
            row["choice_label"] = result.get("choice_label")
            row["choice_emotion"] = result.get("choice_emotion")
            row["is_none"] = result.get("is_none")
            continue
        scorer = MECHANICAL_SCORERS.get(row["family"])
        if scorer is not None:
            result = scorer(row["response"], truncated=truncated)
            row["score"], row["scorer"] = result["score"], "mechanical"
            row["detail"], row["valid"] = result["detail"], bool(result["valid"])
            row["reason"] = result.get("reason", "")
            continue
        # Judge families get the same precheck the mechanical ones do, BEFORE any call is
        # paid for: a response that never reached an answer must not be scored 3 for
        # "declined", nor 0 for "expresses none of this emotion". It expressed nothing
        # because it said nothing.
        precheck = judge_precheck(row["response"], truncated=truncated)
        if precheck is not None:
            row["scorer"] = "precheck"
            row["detail"], row["reason"] = precheck["detail"], precheck["reason"]

    usage: dict = {"calls": 0, "input": 0, "output": 0, "cached": 0, "unusable": 0}
    if not use_judge:
        usage["skipped"] = "--no-judge"
        print("  --no-judge: exact report choices and mechanical behaviour remain scored;")
        print("  only the refusal family would be unavailable if it was selected.")
        return usage
    if not any(r["family"] == "refusal" and r["scorer"] != "precheck" for r in rows):
        usage["not_required"] = True
        print("  no judge calls required: report and the prespecified behaviour family")
        print("  are both scored mechanically.")
        return usage
    available, reason = judge_available()
    if not available:
        usage["skipped"] = reason.splitlines()[0]
        print(f"  judge unavailable: {usage['skipped']}")
        print("  the report channel and the refusal family stay unscored.")
        return usage

    def sendable(family_rows: list[dict]) -> list[dict]:
        return [r for r in family_rows if r["scorer"] != "precheck"]

    jobs: list[tuple[str, str, tuple[int, ...], list[dict]]] = []
    refusal = sendable([r for r in rows if r["family"] == "refusal"])
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
            row["valid"] = bool(verdict is not None and verdict.usable)
            row["detail"] = "" if verdict is None else (
                verdict.reason or verdict.error or ""
            )
            if not row["valid"]:
                row["reason"] = (
                    "no verdict" if verdict is None
                    else (verdict.error or "unusable verdict")
                )
                usage["unusable"] += 1
        usage["calls"] += judge.calls
        for key in ("input", "output", "cached"):
            usage[key] += judge.usage.get(key, 0)
    return usage


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def cell_fluency(frame: pd.DataFrame, config: PCAJLensConfig) -> pd.DataFrame:
    """Per-cell fluency, completion and format validity.

    A cell is usable only if its text remains fluent, at least
    ``min_completion_rate`` generations reach EOS, and no task family exceeds the
    invalid-rate ceiling.  This turns the old end-of-run warning into a real gate.
    """
    working = frame.copy()
    working["_finished"] = (
        working["finished"].fillna(False).astype(bool)
        if "finished" in working else True
    )
    working["_valid"] = (
        working["valid"].fillna(False).astype(bool)
        if "valid" in working else True
    )
    cells = working.groupby(["concept", "condition", "alpha"], dropna=False).agg(
        perplexity=("perplexity", "mean"),
        completion_rate=("_finished", "mean"),
    ).reset_index()
    family_invalid = working.groupby(
        ["concept", "condition", "alpha", "family"], dropna=False
    )["_valid"].mean().reset_index(name="valid_rate")
    worst = family_invalid.groupby(
        ["concept", "condition", "alpha"], dropna=False
    )["valid_rate"].min().reset_index()
    worst["max_family_invalid_rate"] = 1.0 - worst["valid_rate"]
    cells = cells.merge(
        worst.drop(columns=["valid_rate"]),
        on=["concept", "condition", "alpha"], how="left",
    )
    baseline = float(frame[frame["alpha"] == 0.0]["perplexity"].mean())
    cells["perplexity_ratio"] = cells["perplexity"] / max(baseline, 1e-9)
    cells["fluency_passed"] = cells["perplexity_ratio"] <= config.perplexity_max_ratio
    cells["completion_passed"] = (
        cells["completion_rate"] >= config.min_completion_rate
    )
    cells["format_passed"] = (
        cells["max_family_invalid_rate"] <= config.max_invalid_rate
    )
    cells["quality_passed"] = (
        cells["fluency_passed"] & cells["completion_passed"] & cells["format_passed"]
    )
    cells["degraded"] = ~cells["quality_passed"]
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
    # Invalid rows carry score=None already, so `count` and `mean` skip them -- but the
    # invalid COUNT is kept per cell, because "this cell averaged 0.8 over 2 of 4
    # responses" is a different claim from "0.8 over 4".
    working = frame.copy()
    if "valid" in working:
        working.loc[~working["valid"].astype(bool), "score"] = np.nan
    table = working.groupby(
        ["concept", "kind", "channel", "family", "condition", "alpha"], dropna=False
    ).agg(
        n=("response", "size"),
        n_scored=("score", "count"),
        score=("score", "mean"),
        perplexity=("perplexity", "mean"),
    ).reset_index()
    if "valid" in working:
        invalid_counts = working.assign(
            invalid=~working["valid"].astype(bool)
        ).groupby(
            ["concept", "kind", "channel", "family", "condition", "alpha"], dropna=False
        )["invalid"].sum().reset_index(name="n_invalid")
        table = table.merge(
            invalid_counts,
            on=["concept", "kind", "channel", "family", "condition", "alpha"],
            how="left",
        )

    fluency = cell_fluency(frame, config)
    table = table.merge(
        fluency[[
            "concept", "condition", "alpha", "perplexity_ratio", "degraded",
            "completion_rate", "max_family_invalid_rate", "quality_passed",
        ]],
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
    """Mean |z| per (concept, channel, condition, alpha).

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


def dissociation_evidence(
    table: pd.DataFrame, emotions: list[str], behaviour_family: str,
    alpha: float = 1.0,
) -> dict:
    """Evaluate the preregistered contrasts at one fixed positive strength.

    This does not select the largest effect after seeing the grid.  It compares the
    alpha=1 cells against the shared baseline and the five random controls, and records
    separately whether the manipulation was valid and whether the dissociation
    hypothesis was supported.
    """
    results: dict[str, dict] = {}

    def value(concept: str, condition: str, family: str, strength: float):
        cell = table[
            (table["concept"] == concept) & (table["condition"] == condition)
            & (table["family"] == family) & (table["alpha"] == strength)
        ]
        if cell.empty or pd.isna(cell["score"].iloc[0]):
            return None, False
        return float(cell["score"].iloc[0]), bool(cell["quality_passed"].iloc[0])

    for emotion in emotions:
        base_r, base_r_ok = value(emotion, "v", "report", 0.0)
        base_b, base_b_ok = value(emotion, "v", behaviour_family, 0.0)
        report: dict[str, float | None] = {}
        behaviour: dict[str, float | None] = {}
        quality: dict[str, bool] = {}
        for condition in CONDITIONS:
            r, r_ok = value(emotion, condition, "report", alpha)
            b, b_ok = value(emotion, condition, behaviour_family, alpha)
            report[condition] = None if r is None or base_r is None else r - base_r
            behaviour[condition] = None if b is None or base_b is None else b - base_b
            quality[condition] = bool(r_ok and b_ok)

        random_report = [report[c] for c in RANDOM_CONDITIONS if report[c] is not None]
        random_behaviour = [
            behaviour[c] for c in RANDOM_CONDITIONS if behaviour[c] is not None
        ]
        dr_v = report["v"]
        dr_j = report["v_reportable"]
        dr_p = report["v_remainder"]
        db_v = behaviour["v"]
        db_p = behaviour["v_remainder"]
        full_manipulation = bool(
            base_r_ok and base_b_ok and quality["v"] and dr_v is not None and dr_v > 0
        )
        readable_privileged = bool(
            dr_j is not None and dr_p is not None and dr_j > dr_p
        )
        remainder_beats_random = bool(
            db_p is not None and len(random_behaviour) == len(RANDOM_CONDITIONS)
            and abs(db_p) > max(abs(x) for x in random_behaviour)
        )
        remainder_report_silent = bool(
            dr_v not in (None, 0.0) and dr_p is not None
            and abs(dr_p) < 0.25 * abs(dr_v)
        )
        no_degradation = bool(
            base_r_ok and base_b_ok and all(quality.values())
        )
        d_stat = None
        if db_v not in (None, 0.0) and dr_v not in (None, 0.0) and db_p is not None \
                and dr_p is not None:
            d_stat = db_p / db_v - dr_p / dr_v
        results[emotion] = {
            "alpha": alpha,
            "baseline": {"report": base_r, "behaviour": base_b},
            "report_deltas": report,
            "behaviour_deltas": behaviour,
            "random_report_deltas": random_report,
            "random_behaviour_deltas": random_behaviour,
            "full_vector_manipulation_passed": full_manipulation,
            "readable_component_report_privileged": readable_privileged,
            "remainder_behaviour_beats_all_random": remainder_beats_random,
            "remainder_report_silent_25pct": remainder_report_silent,
            "all_required_cells_quality_passed": no_degradation,
            "dissociation_statistic_D": d_stat,
            "experiment_interpretable": bool(full_manipulation and no_degradation),
            "dissociation_pattern_supported": bool(
                full_manipulation and readable_privileged and remainder_beats_random
                and remainder_report_silent and no_degradation
            ),
        }
    return {
        "primary_alpha": alpha,
        "behaviour_family": behaviour_family,
        "report_silence_threshold": 0.25,
        "per_emotion": results,
        "any_interpretable": any(
            item["experiment_interpretable"] for item in results.values()
        ),
    }


def print_dissociation_evidence(evidence: dict) -> None:
    print()
    print(RULE)
    print("PREREGISTERED DISSOCIATION CHECK (alpha=1, no best-cell selection)")
    print(RULE)
    for emotion, item in evidence["per_emotion"].items():
        print(f"  {emotion}")
        print(f"    full v moves target report : "
              f"{'PASS' if item['full_vector_manipulation_passed'] else 'FAIL'}")
        print(f"    v_J report > v_perp report: "
              f"{'PASS' if item['readable_component_report_privileged'] else 'NO'}")
        print(f"    v_perp behaviour > 5 random: "
              f"{'PASS' if item['remainder_behaviour_beats_all_random'] else 'NO'}")
        print(f"    v_perp report < 25% of v  : "
              f"{'PASS' if item['remainder_report_silent_25pct'] else 'NO'}")
        print(f"    required cells usable     : "
              f"{'PASS' if item['all_required_cells_quality_passed'] else 'FAIL'}")
        d_stat = item["dissociation_statistic_D"]
        print("    D statistic               : "
              + ("undefined" if d_stat is None else f"{d_stat:+.3f}"))
        print(f"    interpretation            : "
              f"{'pattern supported' if item['dissociation_pattern_supported'] else 'not supported'}"
              + ("" if item["experiment_interpretable"] else " (manipulation invalid)"))


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
    """One concept's cross-channel summary, then the raw family scores."""
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
        random_values = [
            largest(concept, condition, "behaviour")
            for condition in RANDOM_CONDITIONS
        ]
        random_values = [value for value in random_values if value is not None]
        random = max(random_values) if random_values else None
        if behaviour is None or report is None:
            print("    v_perp was not scored in both channels, so the headline contrast")
            print("    cannot be read for this emotion.")
        elif random is None:
            print("    no usable random control, so a v_perp behavioural shift is not")
            print("    yet attributable to the direction rather than to perturbation.")
        else:
            print(f"    the contrast: v_perp behaviour {behaviour:.2f}, v_perp report "
                  f"{report:.2f}, largest of 5 random behaviours {random:.2f}.")
            print("    The dissociation this experiment looks for needs all three --")
            print("    behaviour high, report low, random low. Read them together or")
            print("    not at all.")

    degraded = int(fluency["degraded"].sum())
    print()
    print(f"  unusable cells (fluency, completion, or format gate failed)"
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
    print("  WHAT v_perp IS. Phase 6 built it as the residual of a k-sparse nonnegative")
    print("  code, and that code's reachable set is a union of cones, not a linear")
    print("  subspace. So v_perp is what that approximation missed AT THAT k, FROM THAT")
    print("  POOL -- not an intrinsically unverbalizable component. A behavioural effect")
    print("  under it is an effect of a direction the sparse code did not capture, which")
    print("  is a weaker and more precise claim than 'an effect of something the model")
    print("  cannot report'. Raising k would move the boundary and could move this result.")
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


def report_tasks_from_phase7(record: dict) -> list[ReportChoiceTask]:
    """Reconstruct the exact report prompts and mappings Phase 7 validated."""
    channels = record.get("channels", {})
    if channels.get("report_protocol") != "randomized_choice_v1":
        raise SystemExit(
            "Phase 7 did not validate the randomized exact-choice report protocol.\n\n"
            "Phase 8 must measure the same channel Phase 7 calibrated; re-run:\n\n"
            "  python run.py phase7\n"
        )
    raw = channels.get("report_choice_tasks") or []
    tasks = [
        ReportChoiceTask(
            variant=int(item["variant"]), prompt=str(item["prompt"]),
            label_to_emotion=tuple(item["label_to_emotion"].items()),
        )
        for item in raw
    ]
    if len(tasks) < 5:
        raise SystemExit(
            f"Phase 7 recorded only {len(tasks)} report-choice variants; five are required. "
            "Re-run python run.py phase7."
        )
    return tasks


def print_design(
    config: PCAJLensConfig, phase7_record: dict, record_path: Path,
    emotions: list[str], block: int,
    directions: list[Directions], strengths: list[float], cells: list[tuple[str, float]],
    n_report_prompts: int, n_behaviour_prompts: int,
    n_prompts: int, total_generations: int, judge_calls: int,
) -> None:
    checks = phase7_record.get("manipulation_checks", {})
    thinking_record = phase7_record.get("thinking", {})
    print(f"model      : {config.model_name} ({config.dtype})")
    print(f"phase 7    : {record_path.name}   separation PASS, "
          f"manipulation checks PASS")
    print(f"             thinking was {thinking_record.get('resolved', 'unrecorded')} "
          f"there; completion rate "
          f"{checks.get('completion', {}).get('rate', float('nan')):.0%}")
    print(f"thinking   : requested enable_thinking={config.enable_thinking} "
          "(resolved once the tokenizer loads)")
    print(f"emotions   : {emotions}")
    print(f"block      : {block} (hidden state {jlens_lens.hidden_state_index(block)}), "
          "read from Phase 6's record --")
    print("             not from config.target_block, which is not fingerprinted and so")
    print("             can point somewhere the decomposition was never fitted")
    for concept in directions:
        frac = concept.detail.get("frac_reportable")
        p_value = concept.detail.get("p_value_vs_random")
        print(f"               {concept.name:<14} ||v|| {concept.norm:>8.2f}"
              + (f"   v_J {frac:.1%} at k={concept.detail.get('k')}" if frac else "")
              + (f", p={p_value:.4f} vs random" if p_value else ""))
    print(f"conditions : {len(CONDITIONS)} (whole, readable, remainder, five random)")
    for key in CONDITIONS:
        short, gloss = CONDITION_LABELS[key]
        print(f"               {short:<7} {gloss}")
    print(f"strengths  : {strengths}  as multiples of ||v||. Phase 6 norm-matched all")
    print("             all conditions, so one alpha is the same perturbation size in")
    print("             each -- otherwise the grid would be comparing vector lengths.")
    print(f"positions  : {config.steer_positions}"
          + ("   (the prompt too, so the model reads its"
             if config.steer_positions == "all" else ""))
    if config.steer_positions == "all":
        print("             instructions through the perturbation -- which is also why")
        print("             the fluency control is not optional)")
    print(f"prompts    : {n_prompts} per cell ({n_report_prompts} exact-choice report + "
          f"{n_behaviour_prompts} prespecified behaviour)")
    print(f"cells      : {len(cells)} per concept ({len(strengths) - 1} strengths x "
          f"{len(CONDITIONS)} conditions + 1 shared baseline)")
    print("             alpha=0 is generated and judged ONCE per concept and copied")
    print("             across conditions: at zero strength they are the same")
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
        probe = Judge(
            rubric=BEHAVIOUR_REFUSAL_RUBRIC,
            model=config.judge_model, client=object(), allowed_scores=(0, 1, 2, 3),
        )
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
    print(f"PHASE 8 GATE -- three concept + five random conditions   run '{config.run_name}'")
    print(RULE)

    record, record_path = read_phase7(config)
    if not record.get("separation", {}).get("separated", False):
        raise SystemExit(
            "Phase 7's separation gate did not pass, so the behaviour channel shares "
            "affect\nvocabulary with the report channel and every number Phase 8 "
            "produced would be\nconfounded. Fix the rubric first:\n\n"
            "  python run.py phase7 --dry-run\n"
        )
    # Phase 7's manipulation checks gate here too. They previously *reported* that report,
    # risk and persistence had no dynamic range, exited 0, and Phase 8 ran anyway -- hours
    # of grid spent measuring families that could not move. A record without the checks at
    # all is a pre-fix Phase 7 run and is refused for the same reason.
    checks = record.get("manipulation_checks")
    if not (checks or {}).get("passed", False):
        detail = "the record predates the checks" if checks is None else (
            f"invalid over ceiling: {checks.get('families_over_invalid_ceiling')}; "
            f"completion passed: {checks.get('completion_passed')}; "
            f"thinking passed: {checks.get('thinking_passed')}"
        )
        raise SystemExit(
            f"Phase 7's manipulation checks did not pass ({detail}).\n\n"
            "An incomplete or mostly unscoreable channel has no valid measurement to "
            "steer.\nPhase 8 refuses to spend the generation budget on it. Re-run:\n\n"
            "  python run.py phase7\n"
        )
    thinking_record = record.get("thinking", {})
    if thinking_record.get("resolved") != "off":
        raise SystemExit(
            "Phase 7 did not verify thinking mode OFF.\n\n"
            "Its baseline scores are then measurements of truncated reasoning traces, and "
            "Phase 8\nwould be steering against them. Re-run Phase 7 with "
            "enable_thinking=false (the\ndefault):\n\n  python run.py phase7\n"
        )
    phase7_emotions = list(record["emotions"]["chosen"])
    if config.channel_emotions and list(config.channel_emotions) != phase7_emotions:
        raise SystemExit(
            f"Phase 8 channel_emotions={list(config.channel_emotions)} do not match the "
            f"Phase 7 report choices {phase7_emotions}. Re-run Phase 7 with the desired "
            "emotions; Phase 8 will not measure a different channel than it calibrated."
        )
    emotions = phase7_emotions
    report_tasks = report_tasks_from_phase7(record)
    behaviour_tasks = selected_behaviour_tasks(config)
    directions, block = load_emotion_directions(config, emotions)

    strengths = [float(a) for a in config.steer_strengths]
    if 0.0 not in strengths:
        joined = ",".join(f"{a:g}" for a in strengths)
        raise SystemExit(
            "steer_strengths has no 0.0, but every number in the grid is a shift from "
            "the\nunsteered baseline -- there would be nothing to shift from:\n\n"
            f"  python run.py phase8 --set steer_strengths=0,{joined}\n"
        )
    if 1.0 not in strengths:
        raise SystemExit(
            "steer_strengths must include the preregistered alpha=1 comparison; "
            "Phase 8 will not choose the best strength after seeing the results."
        )
    cells = grid_cells(strengths)
    n_prompts = len(report_tasks) + len(behaviour_tasks)
    n_judged = sum(1 for task in behaviour_tasks if task.scorer == "judge")
    n_concepts = len(directions) + (0 if args.no_specificity else 1)
    total_generations = n_concepts * len(cells) * n_prompts
    judge_calls = 0 if args.no_judge else n_concepts * len(cells) * n_judged

    print_design(
        config, record, record_path, emotions, block, directions, strengths, cells,
        len(report_tasks), len(behaviour_tasks), n_prompts, total_generations,
        judge_calls,
    )

    sections: dict = {
        "run": {"stage": "phase8_steer", "run_name": config.run_name,
                "dry_run": args.dry_run, "no_judge": args.no_judge,
                "no_specificity": args.no_specificity, "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "phase7": {"record": str(record_path), "separated": True, "emotions": emotions},
        "design": {
            "block": block, "hidden_state": jlens_lens.hidden_state_index(block),
            "conditions": list(CONDITIONS), "strengths": strengths,
            "report_protocol": "randomized_choice_v1",
            "report_tasks": [task.to_dict() for task in report_tasks],
            "behaviour_family": config.phase8_behaviour_family,
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
    thinking = model_utils.thinking_flag_effect(tokenizer)
    resolved_thinking = (
        "off" if (
            not config.enable_thinking and thinking["supported"]
            and not thinking.get("disabled_opens_think_block", True)
        )
        else "on" if thinking["supported"] else "unsupported"
    )
    print(f"  thinking mode  : {resolved_thinking.upper()}   (requested "
          f"enable_thinking={config.enable_thinking}; template responds: "
          f"{thinking['supported']})")
    if resolved_thinking != "off":
        raise SystemExit(
            f"Phase 8 cannot verify thinking mode OFF ({resolved_thinking}). "
            f"{thinking.get('reason') or 'the rendered prompt leaves a reasoning block open'}\n"
            "Aborting before generation."
        )

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
                      f", all directions norm-matched to {control.norm:.2f}")
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
    rows = run_grid(
        model, tokenizer, config, block, directions, report_tasks, behaviour_tasks
    )
    print(f"  {len(rows):,} generations in {(time.time() - grid_t0) / 60:.1f} min")

    print()
    print(RULE)
    print("Scoring")
    print(RULE)
    usage = score_grid(rows, config, use_judge=not args.no_judge)

    invalid = invalid_rates(rows)
    done = completion_rate([
        Completion(text=r["response"], finished=bool(r.get("finished", True)),
                   n_new_tokens=int(r.get("n_new_tokens") or 0))
        for r in rows
    ])
    print()
    print(f"  completion rate : {done['rate']:.1%} ({done['finished']}/{done['n']}) ended "
          f"in EOS; median {done['median_new_tokens']:.0f} new tokens")
    print(f"  {'family':<14}{'n':>6}{'invalid':>9}  top reason")
    for family in sorted(invalid):
        bucket = invalid[family]
        print(f"  {family:<14}{bucket['n']:>6}{bucket['rate']:>8.0%}  "
              + bucket["top_reason"][:44])
    worst = max((b["rate"] for b in invalid.values()), default=0.0)
    if worst > config.max_invalid_rate:
        print(f"  WARNING: {worst:.0%} invalid exceeds the {config.max_invalid_rate:.0%} "
              "ceiling Phase 7 was gated on.")
        print("  Steering can legitimately break format compliance -- that is what the")
        print("  fluency check is for -- but a cell scored over a handful of valid")
        print("  responses is not a measurement. n_invalid is in the grid CSV per cell.")

    frame = pd.DataFrame(expand_baseline(rows))
    table = family_table(frame, config)
    summary = channel_summary(table)
    fluency = cell_fluency(frame, config)

    print()
    print(RULE)
    print("GATE  The 8-condition x 2-channel grid")
    print(RULE)
    print("Read down a column: does the score move with strength? Read across: does it")
    print("move differently for v_J than for v_perp, and does random stay flat? A v_perp")
    print("column that moves in behaviour while its report column does not is the result")
    print("this experiment was built to look for -- with the caveat below.")
    print()
    print("unsteered baseline perplexity: "
          f"{frame[frame['alpha'] == 0.0]['perplexity'].mean():.2f}")
    print(f"* marks a cell that failed fluency ({config.perplexity_max_ratio:g}x), "
          "completion, or format validity;")
    print("  its behavioural score is not treated as a measurement.")
    for concept in directions:
        print_concept(summary, table, concept)
    evidence = dissociation_evidence(
        table, emotions, config.phase8_behaviour_family, alpha=1.0
    )
    print_dissociation_evidence(evidence)

    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "phase8_generations.csv"
    grid_path = out_dir / "phase8_grid.csv"
    frame.to_csv(generations_path, index=False)
    table.to_csv(grid_path, index=False)

    sections["grid"] = table.to_dict(orient="records")
    sections["summary"] = summary.to_dict(orient="records")
    sections["fluency"] = fluency.to_dict(orient="records")
    sections["judge_usage"] = usage
    sections["invalid_rates"] = invalid
    sections["completion"] = done
    sections["dissociation_evidence"] = evidence
    sections["thinking"] = {
        "requested": config.enable_thinking, "template_effect": thinking,
        "resolved": resolved_thinking,
    }
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
        "v_perp_is_k_dependent": "v_perp is the residual of a k-sparse nonnegative code, "
                                 "whose reachable set is a union of cones rather than a "
                                 "linear subspace, so it means 'missed by this "
                                 "approximation at this k and pool', never "
                                 "'intrinsically unverbalizable'",
        "behaviour_scope": f"{config.phase8_behaviour_family} was prespecified as the "
                           "single primary behavioural family; other families were not "
                           "searched for a larger effect",
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
    if not evidence["any_interpretable"]:
        print()
        print("PHASE 8 EXIT 3: no emotion passed both the full-vector manipulation check")
        print("and the per-cell quality gates. The saved outputs are diagnostic, but they")
        print("do not support a report/behaviour dissociation claim.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
