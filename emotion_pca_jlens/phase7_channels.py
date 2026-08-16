"""Phase 7 (GATE): build the two measurement channels and prove they are separate.

What this stage does
--------------------
Chooses 2-3 emotions with clean Phase 6 decompositions, assembles the report and
behaviour channels, and validates them **unsteered** -- generations, scores, and an
activation-level report-availability probe. It steers nothing; Phase 8 does that.

The gate is the one that cannot be delegated to a threshold
-----------------------------------------------------------
This is the phase most likely to silently confound the whole result. If affect
vocabulary leaks into the behaviour rubric, "behaviour tracks emotion" collapses
into "the judge saw emotion words twice", and nothing downstream can recover from
it. So the gate **prints both rubrics in full**, mechanically confirms the behaviour
side contains zero affect vocabulary, and stops for a human to read them.

The mechanical check is necessary and not sufficient, and the gate says so. A
wordlist cannot catch a behaviour rubric that is affect-free in vocabulary while
still asking an affect question -- "how guarded does the answer sound" contains no
listed term and is a leak. That is what the printed rubrics are for.

Three things beyond the brief, each closing a hole the brief leaves open
-----------------------------------------------------------------------
1. **Baseline dynamic range.** A rubric that scores every unsteered response
   identically has no room to show a steering effect, and Phase 8 would spend hours
   discovering that. The gate reports the unsteered spread per task family and flags
   families with none.
2. **Mechanical scorers wherever the task allows.** Three of the four behaviour
   families are scored by code. A regex over hedge words is auditable for
   affect-blindness in a way a judge prompt is not, and it means the gate runs with
   no API key at all -- the separation check does not depend on paying for a judge.
3. **The judge is external, and priced.** Scoring the steered model with itself
   would let the steering perturb the judge as well as the subject. The judge runs
   on a different model through the API, and the gate prints the cost arithmetic for
   Phase 8's much larger grid rather than leaving it to be discovered.

The chat-template boundary, stated rather than smoothed over
-----------------------------------------------------------
Phase 2 extracted from raw story text with ``use_chat_template=False``, matching how
the lens was fitted. This channel is a conversation, so its prompts are
chat-formatted. Steering vectors are generally taken to transfer across that
boundary, but that is an assumption being made here, not a property anyone verified
-- it belongs in the writeup, and the gate prints it every run so it cannot quietly
become invisible.

Usage::

    python run.py phase7 --dry-run     # rubrics, separation check, cost; no weights
    python run.py phase7 --no-judge    # everything the mechanical scorers can do
    python run.py phase7               # the full gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.judge import DEFAULT_JUDGE_MODEL, Judge, judge_available
from core.seeds import set_global_seeds
from emotion_pca_jlens.channel_prompts import (
    AFFECT_ALLOWLIST,
    BEHAVIOUR_REFUSAL_RUBRIC,
    BEHAVIOUR_TASKS,
    MECHANICAL_SCORERS,
    REPORT_PROMPTS,
    REPORT_RUBRIC,
    affect_hits,
    affect_vocabulary,
    channel_texts,
)
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config

RULE = "=" * 78
THIN = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 7 gate: build the report and behaviour channels and prove "
                    "they are separate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="run the separation check, print both rubrics, and estimate judge cost; "
             "never loads model weights",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip every judge call. The separation gate does not need one, so this "
             "is the right mode for validating rubrics before paying for scoring",
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
# Choosing the emotions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """One emotion ranked for use as a steering probe."""

    emotion: str
    frac_reportable: float
    own_word_atom_rank: int | None
    quadrant: str
    arousal: int
    valence: int
    reasons: list[str]

    @property
    def usable(self) -> bool:
        return not self.reasons


def read_decomposition(config: PCAJLensConfig) -> dict:
    """Phase 6's sidecar, or an explanation of how to produce it."""
    path = config.decomposition_meta_path
    if not path.exists():
        raise SystemExit(
            f"no decomposition metadata at\n  {path}\n\n"
            "Phase 7 picks its emotions by how cleanly Phase 6 split them. Run it "
            "first:\n\n  python run.py phase6\n"
        )
    decomposition = json.loads(path.read_text(encoding="utf-8"))
    # Phase 7 ranks emotions by frac_reportable, so it must not run on a decomposition
    # whose own gate refused to report one. Absent means a pre-check Phase 6 run, which
    # is treated as invalid rather than as permission.
    if not decomposition.get("dictionary_valid", False):
        raise SystemExit(
            f"{path} reports dictionary_valid=false (or predates the check).\n\n"
            "Phase 6's atom-validity check did not pass, so lensing an atom does not\n"
            "reliably return its own token and frac_reportable is a variance share out "
            "of\nmislabelled directions. Phase 7 ranks emotions by exactly that number. "
            "Fix the\ndictionary first -- Phase 6's verdict names the knob:\n\n"
            "  python run.py phase6\n"
        )
    return decomposition


def rank_candidates(decomposition: dict, config: PCAJLensConfig) -> list[Candidate]:
    """Rank emotions by how usable their Phase 6 decomposition is.

    The brief asks for 2-3 emotions with clean decompositions and names an
    arousal-heavy negative one as the best behavioural probe -- risk aversion,
    refusal rate, persistence and hedging are all things anxiety plausibly moves,
    where a low-arousal positive emotion gives the behaviour channel very little to
    detect. So arousal and valence break ties, but cleanliness gates: an emotion
    whose ``v_J`` is at chance has no meaningful reportable part to contrast against
    its remainder, and steering it would compare two arbitrary directions.
    """
    control = decomposition.get("random_control", {})
    chance = control.get("mean")
    labels = {}
    for row in decomposition.get("per_emotion", []):
        labels[row["emotion"]] = row

    candidates: list[Candidate] = []
    for emotion, row in labels.items():
        if emotion == "neutral":
            continue
        reasons: list[str] = []
        frac = float(row.get("frac_reportable", 0.0))
        if chance and frac <= 3 * chance:
            reasons.append(
                f"v_J is {frac:.1%}, not clearly above the {chance:.1%} random control"
            )
        if frac > config.frac_j_expected_max:
            reasons.append(f"v_J is {frac:.1%}, above the expected ceiling")
        if row.get("own_word_atom_rank") is None:
            reasons.append("v_J did not select the emotion's own token as an atom")
        meta = decomposition.get("labels", {}).get(emotion, {})
        candidates.append(Candidate(
            emotion=emotion, frac_reportable=frac,
            own_word_atom_rank=row.get("own_word_atom_rank"),
            quadrant=str(meta.get("quadrant", "?")),
            arousal=int(meta.get("arousal", 0) or 0),
            valence=int(meta.get("valence", 0) or 0),
            reasons=reasons,
        ))
    # Usable first, then arousal-heavy negatives, then by how far v_J sits above
    # chance -- a cleaner split gives Phase 8 a sharper contrast.
    return sorted(
        candidates,
        key=lambda c: (
            not c.usable, -(c.arousal > 0 and c.valence < 0), -c.frac_reportable,
        ),
    )


# --------------------------------------------------------------------------- #
# THE GATE: are the channels separate?
# --------------------------------------------------------------------------- #

def check_separation() -> dict:
    """Affect vocabulary per channel. The behaviour side must have none.

    Deliberately asymmetric: the report channel is *supposed* to be full of affect
    terms, so finding them there is confirmation the check works rather than a
    failure. Only the behaviour side's count gates.
    """
    stems = affect_vocabulary()
    texts = channel_texts()
    report: dict = {
        "n_affect_stems": len(stems),
        "emotion_list_loaded": len(stems) > 120,
        "allowlist": AFFECT_ALLOWLIST,
        "per_group": {},
    }
    for group, items in texts.items():
        hits = [hit for item in items for hit in affect_hits(item, stems)]
        report["per_group"][group] = {
            "n_items": len(items),
            "n_hits": len(hits),
            "hits": [
                {"stem": h.stem, "matched": h.matched, "context": h.context}
                for h in hits[:20]
            ],
        }
    behaviour_groups = [g for g in texts if g.startswith("behaviour")]
    report["behaviour_hits"] = sum(
        report["per_group"][g]["n_hits"] for g in behaviour_groups
    )
    report["report_hits"] = sum(
        report["per_group"][g]["n_hits"] for g in texts if g.startswith("report")
    )
    report["separated"] = report["behaviour_hits"] == 0
    return report


def print_separation_gate(report: dict) -> None:
    print()
    print(RULE)
    print("GATE  Are the two channels strictly apart?")
    print(RULE)
    print("The failure this catches: if affect vocabulary reaches the behaviour side,")
    print("'behaviour tracks emotion' becomes 'the judge saw emotion words twice', and")
    print("no later analysis recovers from it. The check is asymmetric on purpose --")
    print("the report channel SHOULD be full of affect terms.")
    print()
    print(f"affect stems checked : {report['n_affect_stems']} "
          f"(171-word list loaded: {report['emotion_list_loaded']})")
    if not report["emotion_list_loaded"]:
        print("  WARNING data/emotions_171.txt did not load, so the check is running on")
        print("          the general stem list alone -- narrower than intended. See")
        print("          core.paths.load_emotions_171 for why that is usually a sync")
        print("          problem rather than a missing download.")
    print(f"allowlisted words    : {sorted(AFFECT_ALLOWLIST)} "
          "(each one is a hole in the check)")
    print()
    print(f"{'channel group':<28}{'items':>7}{'affect hits':>13}   expectation")
    print(THIN)
    for group, data in report["per_group"].items():
        expectation = (
            "expected, this is the emotion channel" if group.startswith("report")
            else "MUST be zero"
        )
        print(f"{group:<28}{data['n_items']:>7}{data['n_hits']:>13}   {expectation}")
    print(THIN)
    if report["separated"]:
        print("  OK  the behaviour channel contains zero affect vocabulary, including")
        print("      its rubric and the source of its mechanical scorers.")
    else:
        print("  FAILED: affect vocabulary in the behaviour channel:")
        for group, data in report["per_group"].items():
            if group.startswith("behaviour") and data["n_hits"]:
                for hit in data["hits"]:
                    print(f"    [{group}] {hit['matched']!r} (stem {hit['stem']!r})")
                    print(f"        ...{hit['context']}...")
    print()
    print("  This check is necessary and NOT sufficient. A wordlist cannot catch a")
    print("  rubric that is affect-free in vocabulary while still asking an affect")
    print("  question -- 'how guarded does the answer sound' contains no listed term")
    print("  and is exactly the leak. Read both rubrics below yourself.")


def print_rubrics(emotion: str) -> None:
    """Print both rubrics in full. The brief requires this and it cannot be a summary."""
    print()
    print(RULE)
    print("BOTH RUBRICS IN FULL -- read them, this gate cannot be a threshold")
    print(RULE)
    print()
    print(THIN)
    print("REPORT CHANNEL rubric (affect vocabulary here is correct)")
    print(THIN)
    print(REPORT_RUBRIC.format(emotion=emotion))
    print(THIN)
    print("BEHAVIOUR CHANNEL rubric (must contain no affect vocabulary)")
    print(THIN)
    print(BEHAVIOUR_REFUSAL_RUBRIC)
    print(THIN)
    print("What to look for in the behaviour rubric: does any line ask about how the")
    print("response *sounds* rather than what it *did*? 'Reluctant', 'guarded',")
    print("'hesitant' are affect judgements wearing behavioural labels. The rubric")
    print("above scores only whether the task was carried out, and explicitly tells")
    print("the judge to ignore tone.")


# --------------------------------------------------------------------------- #
# The activation-level report-availability probe
# --------------------------------------------------------------------------- #

def report_availability(
    model, tokenizer, config: PCAJLensConfig, texts: list[str],
    direction: np.ndarray, hidden_state: int,
) -> dict:
    """Cosine of the residual against the emotion's lens tokens, per token position.

    The second half of the report channel, and the half a judge cannot provide: the
    judge reads what the model *said*, this reads whether the emotion was available
    to say. The direction is Phase 6's ``v_J`` -- the part of the emotion vector the
    lens can express as tokens -- so a high cosine means the residual carries
    verbalisable emotion content at that position, whether or not it was verbalised.

    Pooled over real token positions with no offset, unlike Phase 2: a 50-token
    offset is right for a 200-token story and would discard most of a short reply.
    """
    import torch

    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    probe = torch.as_tensor(unit, dtype=torch.float32)
    device = model_utils.model_input_device(model)
    rows: list[dict] = []

    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=config.max_length
            )
            outputs = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                output_hidden_states=True, use_cache=False,
            )
            hidden = outputs.hidden_states[hidden_state][0].to(torch.float32).cpu()
            del outputs
            norms = hidden.norm(dim=-1).clamp_min(1e-12)
            cosines = (hidden @ probe) / norms
            rows.append({
                "n_tokens": int(hidden.shape[0]),
                "mean_cosine": float(cosines.mean()),
                "max_cosine": float(cosines.max()),
                "frac_positive": float((cosines > 0).float().mean()),
            })
    frame = pd.DataFrame(rows)
    return {
        "n_texts": len(rows),
        "mean_cosine": float(frame["mean_cosine"].mean()),
        "max_cosine": float(frame["max_cosine"].max()),
        "per_text": rows,
    }


# --------------------------------------------------------------------------- #
# Generation and scoring
# --------------------------------------------------------------------------- #

def generate(model, tokenizer, config: PCAJLensConfig, prompts: list[str]) -> list[str]:
    """Chat-formatted generation, one completion per prompt.

    Chat-formatted because the behaviour channel is a conversation -- which crosses
    the boundary Phase 2's raw-text extraction sat on. See the module docstring: the
    assumption that steering transfers across it is an assumption.
    """
    import torch

    device = model_utils.model_input_device(model)
    outputs: list[str] = []
    with torch.inference_mode():
        for prompt in prompts:
            text = model_utils.prepare_texts(
                [prompt], tokenizer, use_chat_template=True,
                chat_add_generation_prompt=True,
            )[0]
            encoded = tokenizer(text, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=config.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion = generated[0][encoded["input_ids"].shape[-1]:]
            outputs.append(tokenizer.decode(completion, skip_special_tokens=True))
    return outputs


def score_behaviour(
    responses: dict[str, str], judge: Judge | None
) -> tuple[list[dict], dict]:
    """Score every behaviour response. Mechanical where possible, judge where not.

    ``judge`` must carry :data:`BEHAVIOUR_REFUSAL_RUBRIC`, not the report rubric. Handing
    it the report judge would score "did this response do the task" against a rubric
    that asks "does this express the emotion" -- a number in the right range, on the
    right scale, measuring the wrong thing, and it is the report channel leaking into
    the behaviour channel that the separation gate exists to prevent.
    """
    rows: list[dict] = []
    for task in BEHAVIOUR_TASKS:
        response = responses.get(task.prompt)
        if response is None:
            continue
        if task.scorer == "mechanical":
            result = MECHANICAL_SCORERS[task.family](response)
            rows.append({
                "family": task.family, "scorer": "mechanical",
                "score": result["score"], "detail": result["detail"],
                "prompt": task.prompt, "note": task.note,
            })
            continue
        if judge is None:
            rows.append({
                "family": task.family, "scorer": "judge (skipped)",
                "score": None, "detail": "no judge available",
                "prompt": task.prompt, "note": task.note,
            })
            continue
        verdict = judge.score(response, label=task.family)
        rows.append({
            "family": task.family, "scorer": "judge",
            "score": verdict.score,
            "detail": verdict.reason or verdict.error or "",
            "prompt": task.prompt, "note": task.note,
        })
    usage = {"calls": judge.calls, **judge.usage} if judge else {"calls": 0}
    return rows, usage


def dynamic_range(rows: list[dict]) -> dict:
    """Unsteered spread per family. No spread means Phase 8 has nothing to detect.

    The check the brief does not ask for and Phase 8 cannot do without: a rubric that
    scores every baseline response identically leaves no room for a steering effect,
    and discovering that after a 2-4 hour grid is the expensive way to learn it.
    """
    frame = pd.DataFrame([r for r in rows if r["score"] is not None])
    if frame.empty:
        return {"scored": False}
    out: dict = {"scored": True, "per_family": {}}
    for family, group in frame.groupby("family"):
        values = group["score"].astype(float)
        out["per_family"][str(family)] = {
            "n": int(len(values)),
            "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()),
            "spread": float(values.max() - values.min()),
            "has_range": bool(values.max() > values.min()),
        }
    out["families_without_range"] = [
        f for f, d in out["per_family"].items() if not d["has_range"]
    ]
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    out_dir = config.phase_dir / "phase7_channels"
    print(RULE)
    print(f"PHASE 7 GATE -- two measurement channels   run '{config.run_name}'")
    print(RULE)
    print(f"model    : {config.model_name} ({config.dtype})")
    print(f"outputs  : {out_dir}")
    print()
    print("Builds and validates the channels UNSTEERED. Steering is Phase 8.")
    print()
    print("Chat-template boundary, stated every run: Phase 2 extracted from raw story")
    print("text (use_chat_template=False, matching how the lens was fitted); these")
    print("prompts are chat-formatted because the behaviour channel is a conversation.")
    print("Steering vectors are generally taken to transfer across that boundary. That")
    print("is an assumption, not a verified property, and belongs in the writeup.")
    print()

    # --- the emotions ------------------------------------------------------ #
    decomposition = read_decomposition(config)
    ranked = rank_candidates(decomposition, config)
    print(RULE)
    print("STEP 1  Which emotions have clean enough Phase 6 decompositions?")
    print(RULE)
    print(f"{'emotion':<16}{'quadrant':>10}{'frac v_J':>10}{'own token':>11}   verdict")
    print(THIN)
    for candidate in ranked[:12]:
        own = "-" if candidate.own_word_atom_rank is None else f"#{candidate.own_word_atom_rank}"
        verdict = "usable" if candidate.usable else "; ".join(candidate.reasons)
        print(f"{candidate.emotion:<16}{candidate.quadrant:>10}"
              f"{candidate.frac_reportable:>10.1%}{own:>11}   {verdict}")
    print(THIN)

    if config.channel_emotions is None:
        chosen = [c.emotion for c in ranked if c.usable][: config.n_channel_emotions]
    else:
        chosen = list(config.channel_emotions)
    unknown = sorted(set(chosen) - {c.emotion for c in ranked})
    if unknown:
        raise SystemExit(f"channel_emotions {unknown} are not in Phase 6's output")
    if not chosen:
        print("\nABORTED: no emotion has a usable Phase 6 decomposition. Phase 8 would",
              file=sys.stderr)
        print("be steering with directions Phase 6 could not validate; fix Phase 6 first.",
              file=sys.stderr)
        return 3
    print(f"chosen: {chosen}")
    arousal_negative = [
        c.emotion for c in ranked
        if c.emotion in chosen and c.arousal > 0 and c.valence < 0
    ]
    print(f"  arousal-heavy negative among them: {arousal_negative or 'NONE'}")
    if not arousal_negative:
        print("  NOTE the brief names an arousal-heavy negative emotion (anxious) as the")
        print("       best behavioural probe -- risk aversion, refusal, persistence and")
        print("       hedging are things it plausibly moves. Without one, a null result")
        print("       in Phase 8 is weak evidence: it may be the probe, not the theory.")

    # --- THE GATE ---------------------------------------------------------- #
    separation = check_separation()
    print_separation_gate(separation)
    print_rubrics(chosen[0])

    # --- the judge --------------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 2  The judge")
    print(RULE)
    available, reason = judge_available()
    judge: Judge | None = None
    behaviour_judge: Judge | None = None
    print("External by necessity: scoring the steered model with itself would let the")
    print("steering perturb the judge as well as the subject, so a score shift could be")
    print("the judge moving rather than the behaviour.")
    print()
    print(f"judge model : {config.judge_model}")
    print(f"available   : {available}" + ("" if available else f"  ({reason})"))
    if args.no_judge:
        print("  --no-judge: skipping every judge call. The separation gate above does")
        print("  not need one, so this is the right mode for validating rubrics first.")
    elif available:
        judge = Judge(rubric=REPORT_RUBRIC.format(emotion=chosen[0]),
                      model=config.judge_model)
        # A second judge, because the two channels have different rubrics AND different
        # scales: the refusal rubric defines 0-3, and leaving the default 0-4 enum in
        # place would let the judge return a score the rubric never defines.
        behaviour_judge = Judge(rubric=BEHAVIOUR_REFUSAL_RUBRIC,
                                model=config.judge_model,
                                allowed_scores=(0, 1, 2, 3))
        estimate = judge.estimate_cost(n_calls=len(REPORT_PROMPTS) + len(BEHAVIOUR_TASKS))
        grid = judge.estimate_cost(n_calls=config.phase8_grid_calls)
        if estimate.get("known_price"):
            print(f"  this phase  : ~${estimate['usd_cached']:.3f} "
                  f"({estimate['n_calls']} calls, rubric cached)")
            print(f"  Phase 8 grid: ~${grid['usd_cached']:.2f} interactive, "
                  f"~${grid['usd_cached_batched']:.2f} via the Batches API")
            print(f"                ({grid['n_calls']} calls at "
                  f"phase8_grid_calls; batching halves it and Phase 8 is not")
            print("                latency-sensitive, so it should batch)")
    else:
        print("  Continuing without a judge. Mechanical scorers cover 3 of the 4")
        print("  behaviour families, and the separation gate is unaffected.")

    sections: dict = {
        "run": {"stage": "phase7_channels", "run_name": config.run_name,
                "dry_run": args.dry_run, "no_judge": args.no_judge,
                "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "emotions": {
            "chosen": chosen,
            "arousal_heavy_negative": arousal_negative,
            "ranked": [
                {"emotion": c.emotion, "frac_reportable": c.frac_reportable,
                 "quadrant": c.quadrant, "usable": c.usable, "reasons": c.reasons}
                for c in ranked
            ],
        },
        "separation": separation,
        "channels": {
            "report_prompts": list(REPORT_PROMPTS),
            "report_rubric": REPORT_RUBRIC,
            "behaviour_rubric": BEHAVIOUR_REFUSAL_RUBRIC,
            "behaviour_tasks": [
                {"family": t.family, "scorer": t.scorer, "prompt": t.prompt}
                for t in BEHAVIOUR_TASKS
            ],
        },
        "judge": {"model": config.judge_model, "available": available,
                  "reason": reason, "used": judge is not None},
        "chat_template_boundary": (
            "Phase 2 extracted with use_chat_template=False; these prompts are "
            "chat-formatted. Transfer across that boundary is assumed, not verified."
        ),
    }

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run", title=f"PHASE 7 DRY RUN -- {config.run_name}",
            sections=sections, txt_name="phase7_dry_run.txt",
            json_name="phase7_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: rubrics printed, separation checked, no weights loaded.")
        print(f"  separation : {'PASS' if separation['separated'] else 'FAILED'}")
        print(f"  records    : {txt_path}")
        print(f"               {json_path}")
        print(RULE)
        return 0 if separation["separated"] else 3

    if not separation["separated"]:
        print("\nABORTED: the channels are not separate. Fix the behaviour rubric or",
              file=sys.stderr)
        print("prompts before generating anything -- every number after this point would",
              file=sys.stderr)
        print("be confounded.", file=sys.stderr)
        return 3

    # --- baseline generations ---------------------------------------------- #
    cache_dir = paths.hf_cache_dir()
    print()
    print(RULE)
    print(f"STEP 3  Baseline (unsteered) generations")
    print(RULE)
    print(f"Loading {config.model_name} ...")
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
    print(f"  generating {len(REPORT_PROMPTS)} report + {len(BEHAVIOUR_TASKS)} "
          f"behaviour completions at {config.generation_max_new_tokens} new tokens "
          "each ...")
    t0 = time.time()
    report_texts = generate(model, tokenizer, config, list(REPORT_PROMPTS))
    behaviour_prompts = [t.prompt for t in BEHAVIOUR_TASKS]
    behaviour_texts = generate(model, tokenizer, config, behaviour_prompts)
    print(f"  generated in {time.time() - t0:.0f}s")

    behaviour_rows, judge_usage = score_behaviour(
        dict(zip(behaviour_prompts, behaviour_texts)), behaviour_judge
    )
    report_rows: list[dict] = []
    for prompt, text in zip(REPORT_PROMPTS, report_texts):
        verdict = judge.score(text, label=chosen[0]) if judge else None
        report_rows.append({
            "prompt": prompt,
            "score": None if verdict is None else verdict.score,
            "detail": "" if verdict is None else (verdict.reason or verdict.error or ""),
            "response": text,
        })
    if judge is not None:
        # Both judges' calls, or the cost line would report only the behaviour half.
        judge_usage = {
            "calls": judge_usage.get("calls", 0) + judge.calls,
            **{key: judge_usage.get(key, 0) + value for key, value in judge.usage.items()},
        }

    # --- activation-level report availability ------------------------------ #
    print()
    print(RULE)
    print("STEP 4  Activation-level report availability")
    print(RULE)
    availability = {}
    try:
        from safetensors.numpy import load_file

        tensors = load_file(str(config.decomposition_path))
        emotions = json.loads(decomposition["emotions"]) if isinstance(
            decomposition.get("emotions"), str
        ) else decomposition["emotions"]
        block = int(decomposition.get("vectors", {}).get("target_block", -1))
        hidden_state = jlens_lens.hidden_state_index(block)
        print("The half a judge cannot give you: the judge reads what the model SAID,")
        print("this reads whether the emotion was available to say. Cosine of the")
        print(f"residual at block {block} (hidden state {hidden_state}) against v_J --")
        print("the part of the emotion vector")
        print("the lens can express as tokens.")
        for emotion in chosen:
            index = emotions.index(emotion)
            availability[emotion] = report_availability(
                model, tokenizer, config, report_texts,
                np.asarray(tensors["v_reportable"][index], dtype=np.float64),
                hidden_state,
            )
            print(f"  {emotion:<14} mean cos {availability[emotion]['mean_cosine']:+.4f}, "
                  f"max {availability[emotion]['max_cosine']:+.4f} "
                  f"over {availability[emotion]['n_texts']} report responses")
        print()
        print("  These are BASELINE values. They are the reference Phase 8's steered")
        print("  values move against; on their own they say nothing.")
    except Exception as exc:
        availability = {"available": False, "reason": str(exc)}
        print(f"  unavailable: {exc}")

    # --- dynamic range ----------------------------------------------------- #
    ranges = dynamic_range(behaviour_rows + [
        {"family": "report", "score": r["score"]} for r in report_rows
    ])
    print()
    print(RULE)
    print("STEP 5  Does the unsteered baseline leave room for an effect?")
    print(RULE)
    print("A rubric that scores every baseline response identically has no room to show")
    print("a steering effect, and Phase 8's grid is 2-4 hours. Cheaper to know now.")
    print()
    if ranges.get("scored"):
        print(f"{'family':<16}{'n':>4}{'min':>8}{'max':>8}{'mean':>8}{'spread':>9}")
        print(THIN)
        for family, data in sorted(ranges["per_family"].items()):
            flag = "" if data["has_range"] else "   <- no range at baseline"
            print(f"{family:<16}{data['n']:>4}{data['min']:>8.2f}{data['max']:>8.2f}"
                  f"{data['mean']:>8.2f}{data['spread']:>9.2f}{flag}")
        print(THIN)
        if ranges["families_without_range"]:
            print(f"  {ranges['families_without_range']} scored identically on every")
            print("  baseline prompt. Either the prompts are too similar or the scorer is")
            print("  too coarse; a steering effect there would be invisible. Fix before")
            print("  Phase 8 rather than reading a null result out of it.")
    else:
        print("  nothing scored (no judge, and mechanical families produced no scores)")

    # --- save + verdict ---------------------------------------------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(behaviour_rows).to_csv(out_dir / "phase7_behaviour.csv", index=False)
    pd.DataFrame(report_rows).to_csv(out_dir / "phase7_report.csv", index=False)
    sections["baseline"] = {
        "report": report_rows, "behaviour": behaviour_rows,
        "dynamic_range": ranges, "report_availability": availability,
        "judge_usage": judge_usage,
    }
    txt_path, json_path = provenance.write_run_record(
        out_dir, title=f"PHASE 7 GATE -- {config.run_name}",
        sections=sections, txt_name="phase7_gate.txt", json_name="phase7_gate.json",
    )

    print()
    print(RULE)
    print("PHASE 7 VERDICT")
    print(RULE)
    print(f"  channels separate    : {'PASS' if separation['separated'] else 'FAILED'} "
          f"({separation['behaviour_hits']} affect terms in the behaviour channel, "
          f"{separation['report_hits']} in the report channel as expected)")
    print(f"  emotions chosen      : {chosen}"
          + ("" if arousal_negative else "   (no arousal-heavy negative -- see above)"))
    print(f"  baseline range       : "
          f"{'PASS' if ranges.get('scored') and not ranges['families_without_range'] else 'REVIEW'}")
    print(f"  judge                : "
          f"{config.judge_model if judge else 'not used'}"
          + (f", {judge_usage['calls']} calls" if judge else ""))
    print()
    print(f"  report   : {out_dir / 'phase7_report.csv'}")
    print(f"  behaviour: {out_dir / 'phase7_behaviour.csv'}")
    print(f"  records  : {txt_path}")
    print(f"             {json_path}")
    print()
    print("  Now read the two rubrics printed above, and the baseline responses in the")
    print("  CSVs. The mechanical check confirms no affect *word* crosses into the")
    print("  behaviour channel; only you can confirm no affect *question* does. That is")
    print("  why this gate stops here and cannot be a threshold.")
    print()
    print("STOPPING at the Phase 7 gate, as agreed. Phase 8 (steering) has not run.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
