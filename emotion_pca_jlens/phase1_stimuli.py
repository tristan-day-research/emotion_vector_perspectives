"""Phase 1 (GATE): assemble the emotion stimulus set and verify its coverage.

What this stage produces
------------------------
One row per stimulus, with the columns the brief asks for -- ``emotion``,
``quadrant``, ``text`` -- plus the provenance columns that make the set
reproducible (``example_id``, ``topic``, ``topic_id``, ``story_idx``, ``split``,
``content_sha1``). Written to ``results/phases/phase1_stimuli.parquet``, which is
what Phase 2 reads. No model is loaded and no activations are collected.

Where the text comes from, and why not from templates
-----------------------------------------------------
The stimuli are selected from `ryancodrai/emotion-probes
<https://huggingface.co/datasets/ryancodrai/emotion-probes>`_ (CC-BY-4.0), which
this repo already loads for Experiment 1: 171 emotions x the same 100 topics x 12
stories, plus 1,200 neutral stories on those same 100 topics.

Selecting from it rather than generating vignettes from templates buys three
things that matter to the result:

1. **Topic matching is structural, not aspirational.** Every emotion is written
   about the *same* 100 topics. With ``stories_per_emotion`` a multiple of 100,
   each emotion draws exactly the same number of stories per topic, so subject
   matter is balanced across emotions by construction rather than by care. This
   is the property that lets a direction be read as encoding emotion rather than
   scenario -- and :func:`verify_topic_matching` checks it on the real table
   instead of trusting the argument.
2. **Enough data for the reliability gate.** 1,200 stories per emotion are
   available against the ~40 a template scheme would produce. Phase 2 gates on
   split-half cosine, and 40 stimuli would likely fail it for reasons that have
   nothing to do with whether the representation exists.
3. **The lengths actually fit the pooling.** This is the one that would have
   quietly broken the experiment. Pooling excludes the first
   ``token_offset = 50`` real tokens, so a 2-4 sentence vignette (~40-60 tokens)
   would leave nothing to average and be *skipped*. These stories run 93-170
   words (~125-230 tokens), leaving 75-180 tokens pooled per stimulus.
   :func:`report_length_fit` prints that margin.

The trade is that the stimuli are longer than the 2-4 sentences the brief
imagined, and they are third-person narrative prose rather than minimal vignettes.

The circumplex labels are not used by the analysis
--------------------------------------------------
``quadrant`` / ``valence`` / ``arousal`` in :data:`DEFAULT_CIRCUMPLEX_SET` are *a
priori* labels from the affective-circumplex literature. They are used only to
(a) verify quadrant coverage here, (b) colour the Phase 3 scatter, and (c) score
how well the discovered PCs line up with them. The PCA itself is unsupervised and
never sees them, so they stay an independent check rather than becoming the
answer.

Usage::

    python run.py phase1                        # build, gate, stop
    python run.py phase1 --coverage-only        # emotion table only, no dataset
    python run.py phase1 --set stories_per_emotion=200
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from core import dataset, env_file, paths, provenance
from core.seeds import set_global_seeds
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config

RULE = "=" * 78
THIN = "-" * 78

NEUTRAL_QUADRANT = "neutral"
UNLABELLED_QUADRANT = "unlabelled"


# --------------------------------------------------------------------------- #
# The circumplex design
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CircumplexEmotion:
    """One emotion with its a priori circumplex position.

    Attributes:
        emotion: Word as it appears in ``data/emotions_171.txt``.
        quadrant: ``HA-P`` / ``HA-N`` / ``LA-P`` / ``LA-N``.
        valence: ``+1`` pleasant, ``-1`` unpleasant.
        arousal: ``+1`` activated, ``-1`` deactivated.
        family: Sub-family within the quadrant. Not used by the analysis; it is
            here so the within-quadrant spread is visible at the gate, and
            because a PC beyond the first two may well separate these rather
            than being noise.
    """

    emotion: str
    quadrant: str
    valence: int
    arousal: int
    family: str


#: A balanced 4-per-quadrant design over the two circumplex axes.
#:
#: Balance is deliberate and load-bearing. With four emotions in every
#: valence x arousal cell, the a priori valence and arousal contrasts are
#: *orthogonal by construction*. If PC1 and PC2 turn out to align with them, that
#: cannot be an artefact of having sampled more high-arousal than low-arousal
#: emotions -- which is exactly the confound the brief flags when it says an
#: all-high-arousal set can never show an arousal axis.
#:
#: Within each quadrant the four words are drawn from different sub-families
#: rather than being intensity variants of one word, so the set spans each
#: quadrant instead of sampling one point in it four times.
DEFAULT_CIRCUMPLEX_SET: tuple[CircumplexEmotion, ...] = (
    # High arousal, positive valence -- activated pleasant.
    CircumplexEmotion("excited", "HA-P", +1, +1, "anticipatory joy"),
    CircumplexEmotion("elated", "HA-P", +1, +1, "elevated mood"),
    CircumplexEmotion("ecstatic", "HA-P", +1, +1, "peak joy"),
    CircumplexEmotion("thrilled", "HA-P", +1, +1, "delighted excitement"),
    # High arousal, negative valence -- activated unpleasant.
    CircumplexEmotion("anxious", "HA-N", -1, +1, "anticipatory fear"),
    CircumplexEmotion("terrified", "HA-N", -1, +1, "acute fear"),
    CircumplexEmotion("angry", "HA-N", -1, +1, "hostile anger"),
    CircumplexEmotion("furious", "HA-N", -1, +1, "intense anger"),
    # Low arousal, positive valence -- deactivated pleasant.
    CircumplexEmotion("content", "LA-P", +1, -1, "contentment"),
    CircumplexEmotion("serene", "LA-P", +1, -1, "serenity"),
    CircumplexEmotion("calm", "LA-P", +1, -1, "calm"),
    CircumplexEmotion("relaxed", "LA-P", +1, -1, "relaxation"),
    # Low arousal, negative valence -- deactivated unpleasant.
    CircumplexEmotion("sad", "LA-N", -1, -1, "sadness"),
    CircumplexEmotion("gloomy", "LA-N", -1, -1, "dysphoric mood"),
    CircumplexEmotion("bored", "LA-N", -1, -1, "disengagement"),
    CircumplexEmotion("weary", "LA-N", -1, -1, "depletion"),
)

#: Display order for quadrants: positive above negative, activated before
#: deactivated. Matches how a circumplex is normally drawn.
QUADRANT_ORDER = ("HA-P", "HA-N", "LA-P", "LA-N")

QUADRANT_LABELS = {
    "HA-P": "high arousal, positive  (activated pleasant)",
    "HA-N": "high arousal, negative  (activated unpleasant)",
    "LA-P": "low arousal, positive   (deactivated pleasant)",
    "LA-N": "low arousal, negative   (deactivated unpleasant)",
    NEUTRAL_QUADRANT: "neutral                 (circumplex origin)",
    UNLABELLED_QUADRANT: "unlabelled              (no circumplex position)",
}


def resolve_emotion_set(config: PCAJLensConfig) -> list[CircumplexEmotion]:
    """The emotions this run will use, with circumplex labels attached.

    ``config.emotions``:

    * ``None``  -- the balanced 16 of :data:`DEFAULT_CIRCUMPLEX_SET`.
    * ``"all"`` -- all 171 words, labelled where the table knows them. The other
      155 become ``unlabelled`` at the origin *by design*: PCA is fitted on all
      171 for a well-determined covariance structure, and validated against the
      16 balanced anchors. Hand-labelling 155 words here would invent precision
      that is not there.
    * a list    -- an explicit set, in the given order.

    An unknown word becomes ``unlabelled`` rather than an error, so exploring a
    new emotion does not require editing the table first. The gate says so
    plainly, because Phase 3 cannot colour those points.
    """
    if config.emotions is None:
        return list(DEFAULT_CIRCUMPLEX_SET)
    if isinstance(config.emotions, str) and config.emotions != "all":
        # config.validate() already rejects this, so it only fires for a caller
        # constructing a config directly. Worth guarding: list("sad") is
        # ['s','a','d'], which would fail far away with three unknown "emotions".
        raise ValueError(
            f"emotions={config.emotions!r}: the only string form is 'all'; "
            "pass a list for an explicit set"
        )
    names = (
        paths.load_emotions_171() if config.emotions == "all" else list(config.emotions)
    )
    table = {entry.emotion: entry for entry in DEFAULT_CIRCUMPLEX_SET}
    return [
        table.get(
            name,
            CircumplexEmotion(name, UNLABELLED_QUADRANT, 0, 0, "not in circumplex table"),
        )
        for name in names
    ]


def anchors(entries: list[CircumplexEmotion]) -> list[CircumplexEmotion]:
    """The circumplex-labelled subset -- what Phase 3 validates its PCs against."""
    return [e for e in entries if e.quadrant in QUADRANT_ORDER]


def format_coverage(
    entries: list[CircumplexEmotion], anchor_design: bool = False
) -> tuple[str, list[str]]:
    """Render the emotion set grouped by quadrant. Returns (text, warnings).

    Printed *before* any dataset work, which is the order the brief asks for:
    coverage is the thing to disagree with, and it costs nothing to check first.

    ``anchor_design`` (set for ``emotions="all"``) means unlabelled words are
    intentional, so their presence is reported as design rather than warned about.
    Coverage and balance are then judged over the anchors alone, which is what the
    validation actually uses.
    """
    lines: list[str] = []
    warnings: list[str] = []
    by_quadrant: dict[str, list[CircumplexEmotion]] = {}
    for entry in entries:
        by_quadrant.setdefault(entry.quadrant, []).append(entry)

    ordered = [q for q in QUADRANT_ORDER if q in by_quadrant]
    ordered += [q for q in by_quadrant if q not in QUADRANT_ORDER]

    for quadrant in ordered:
        members = by_quadrant[quadrant]
        label = QUADRANT_LABELS.get(quadrant, quadrant)
        lines.append(f"  {quadrant:<12} {label}   n={len(members)}")
        if quadrant == UNLABELLED_QUADRANT and len(members) > 12:
            # 155 words would bury the part worth reading.
            names = [e.emotion for e in members]
            lines.append(f"      {', '.join(names[:12])},")
            lines.append(f"      ... and {len(names) - 12} more (see the run record)")
        else:
            for entry in members:
                lines.append(
                    f"      {entry.emotion:<12} valence={entry.valence:+d} "
                    f"arousal={entry.arousal:+d}   {entry.family}"
                )
        lines.append("")

    missing = [q for q in QUADRANT_ORDER if q not in by_quadrant]
    if missing:
        warnings.append(
            f"quadrants with no emotions: {missing}. An axis cannot emerge along a "
            "contrast the stimulus set does not span -- with no low-arousal emotions "
            "there is no arousal axis to find."
        )

    sizes = {q: len(by_quadrant[q]) for q in QUADRANT_ORDER if q in by_quadrant}
    if sizes and len(set(sizes.values())) > 1:
        warnings.append(
            f"unbalanced quadrants {sizes}: the a priori valence and arousal contrasts "
            "are no longer orthogonal, so a PC aligning with one of them may partly "
            "reflect the sampling rather than the representation."
        )

    unlabelled = [e.emotion for e in entries if e.quadrant == UNLABELLED_QUADRANT]
    if unlabelled and anchor_design:
        lines.append(
            f"  DESIGN: {len(entries) - len(unlabelled)} labelled anchors, "
            f"{len(unlabelled)} unlabelled.\n"
            f"          PCA is fitted on all {len(entries)} emotions, which is what makes\n"
            "          the variance-explained figures interpretable; valence/arousal\n"
            "          alignment is scored against the balanced anchors only."
        )
    elif unlabelled:
        warnings.append(
            f"no circumplex position for {unlabelled}: Phase 3 cannot colour these by "
            "quadrant and they are excluded from the valence/arousal alignment score. "
            "Add them to DEFAULT_CIRCUMPLEX_SET to fix."
        )
    return "\n".join(lines), warnings


# --------------------------------------------------------------------------- #
# Building the table
# --------------------------------------------------------------------------- #

def build_stimulus_table(config: PCAJLensConfig, entries: list[CircumplexEmotion]):
    """Assemble the stimulus table. Returns ``(df, topic_split, dataset_report)``."""
    import pandas as pd

    examples, topic_split, report = dataset.build_example_table(
        emotions=[e.emotion for e in entries],
        stories_per_emotion=config.stories_per_emotion,
        neutral_stories=config.neutral_stories if config.include_neutral else None,
        split_seed=config.split_seed,
        split_proportions=config.split_proportions,
        revision=config.dataset_revision,
        cache_dir=paths.hf_cache_dir(),
        include_neutral=config.include_neutral,
    )

    labels = {e.emotion: e for e in entries}

    def label_for(row) -> tuple[str, int, int, str]:
        if row["source"] == "neutral":
            return NEUTRAL_QUADRANT, 0, 0, "neutral"
        entry = labels[row["emotion"]]
        return entry.quadrant, entry.valence, entry.arousal, entry.family

    attached = pd.DataFrame(
        [label_for(row) for _, row in examples.iterrows()],
        columns=["quadrant", "valence", "arousal", "family"],
        index=examples.index,
    )
    out = examples.join(attached)
    # `emotion` is NA for neutral rows in the loader. Give them an explicit label
    # so groupby/plotting treat neutral as its own category instead of dropping it.
    out["emotion"] = out["emotion"].fillna(NEUTRAL_QUADRANT)
    # The brief's column name; the loader calls it `story`.
    out = out.rename(columns={"story": "text"})

    columns = [
        "emotion", "quadrant", "text", "valence", "arousal", "family",
        "source", "example_id", "topic", "topic_id", "story_idx", "split",
        "content_sha1",
    ]
    return out[columns].reset_index(drop=True), topic_split, report


def verify_topic_matching(df) -> dict:
    """Check that every emotion really is written about the same topics, equally.

    The whole argument for this stimulus set is that subject matter is balanced
    across emotions. That follows from the dataset's structure *and* from
    ``stories_per_emotion`` dividing the topic count -- so it is checked on the
    assembled table rather than assumed from the arguments.
    """
    emotional = df[df["source"] == "emotion"]
    if emotional.empty:
        return {"checked": False, "reason": "no emotional stimuli"}

    counts = emotional.pivot_table(
        index="emotion", columns="topic_id", values="example_id",
        aggfunc="count", fill_value=0,
    )
    per_emotion_topics = {e: int((row > 0).sum()) for e, row in counts.iterrows()}
    distinct_cells = sorted({int(v) for v in counts.to_numpy().ravel()})
    identical_topic_sets = len(set(map(tuple, (counts > 0).to_numpy().tolist()))) == 1
    return {
        "checked": True,
        "n_topics_total": int(counts.shape[1]),
        "topics_per_emotion": per_emotion_topics,
        "all_emotions_same_topics": identical_topic_sets,
        "distinct_per_cell_counts": distinct_cells,
        "exactly_matched": identical_topic_sets and len(distinct_cells) == 1,
    }


def report_length_fit(df, config: PCAJLensConfig) -> dict:
    """Word-length distribution vs the pooling offset.

    Tokenizer-free on purpose: Phase 1 must not need the model. ~1.35 tokens per
    English word is the same approximation ``extract_activations --dry-run`` uses,
    and the margin here is wide enough that the approximation is not doing any
    load-bearing work.
    """
    import numpy as np

    words = df["text"].str.split().str.len().to_numpy()
    approx_tokens = words * 1.35
    need = config.token_offset + config.min_pooled_tokens
    return {
        "words_min": int(words.min()),
        "words_median": float(np.median(words)),
        "words_max": int(words.max()),
        "approx_tokens_min": float(approx_tokens.min()),
        "approx_tokens_median": float(np.median(approx_tokens)),
        "token_offset": config.token_offset,
        "min_pooled_tokens": config.min_pooled_tokens,
        "tokens_needed": need,
        "approx_pooled_min": float(approx_tokens.min() - config.token_offset),
        "n_at_risk": int((approx_tokens < need).sum()),
    }


# --------------------------------------------------------------------------- #
# Gate output
# --------------------------------------------------------------------------- #

def print_examples(df, per_emotion: int, quadrants: list[str] | None = None) -> None:
    """Show a few stimuli per emotion, from different topics.

    ``quadrants`` restricts which groups are shown. With ``emotions="all"`` the
    full listing would be ~500 vignettes, which nobody reads; the anchors plus
    neutral are the set worth eyeballing.
    """
    present = set(df["quadrant"])
    order = [q for q in QUADRANT_ORDER if q in present] + [
        q for q in (NEUTRAL_QUADRANT, UNLABELLED_QUADRANT) if q in present
    ]
    if quadrants is not None:
        order = [q for q in order if q in quadrants]
    for quadrant in order:
        block = df[df["quadrant"] == quadrant]
        print()
        print(THIN)
        print(f"{quadrant}  {QUADRANT_LABELS.get(quadrant, '')}")
        print(THIN)
        for emotion, rows in block.groupby("emotion", sort=True):
            print(f"\n{emotion}  (n={len(rows)})")
            # Spread the shown examples across the table rather than taking the
            # first few, which would all come from topic 0.
            step = max(1, len(rows) // per_emotion)
            for _, row in rows.iloc[::step].head(per_emotion).iterrows():
                text = " ".join(row["text"].split())
                print(f"  [{row['topic']}]")
                print(f"    {text[:260]}{'...' if len(text) > 260 else ''}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1 gate: assemble and verify the emotion stimulus set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--coverage-only",
        action="store_true",
        help="print the emotion set grouped by quadrant and stop; touches no dataset",
    )
    p.add_argument(
        "--config-json", type=Path, default=None, help="JSON file of config overrides"
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field; repeatable",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    print(RULE)
    print(f"PHASE 1 GATE -- emotion stimulus set   run '{config.run_name}'")
    print(RULE)
    print(f"source  : {dataset.HF_DATASET_ID} (CC-BY-4.0)")
    print(f"output  : {config.stimuli_path}")
    print()
    print("This stage loads no model and collects no activations. It selects text.")
    print()

    # --- coverage first: the thing to disagree with, and it is free --------- #
    entries = resolve_emotion_set(config)
    anchor_design = config.emotions == "all"
    anchor_set = anchors(entries)
    print(RULE)
    print(f"STEP 1  Circumplex coverage: {len(entries)} emotions"
          + (" + neutral" if config.include_neutral else ""))
    print(RULE)
    print("Quadrant labels are a priori, from the affective-circumplex literature.")
    print("The PCA in Phase 3 is unsupervised and never sees them -- they are used")
    print("only to check coverage here, colour the scatter, and score alignment.")
    if anchor_design:
        print()
        print(f"emotions='all': PCA over {len(entries)} emotions, validated against")
        print(f"{len(anchor_set)} labelled anchors. Rationale: after mean-centring, n")
        print("centroids span rank n-1, so with only 16 emotions PC1/PC2 would explain a")
        print("large variance fraction by construction. 171 makes that number mean")
        print("something; the anchors keep the valence/arousal test balanced.")
    print()
    coverage, warnings = format_coverage(entries, anchor_design=anchor_design)
    print(coverage)

    known = set(paths.load_emotions_171())
    unknown = sorted({e.emotion for e in entries} - known)
    if unknown:
        print(f"  NOT IN data/emotions_171.txt: {unknown}")
        print("  The dataset check below is authoritative, but these will almost")
        print("  certainly fail it.")
        warnings.append(f"emotions absent from the 171-word list: {unknown}")

    if warnings:
        print("  WARNINGS:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  OK  all four quadrants covered, equal size, every emotion labelled.")

    sections: dict = {
        "run": {"stage": "phase1_stimuli", "run_name": config.run_name,
                "coverage_only": args.coverage_only},
        "config": config.to_dict(),
        "emotion_set": {
            "n_emotions": len(entries),
            "n_anchors": len(anchor_set),
            "anchor_design": anchor_design,
            "include_neutral": config.include_neutral,
            "emotions": [
                {"emotion": e.emotion, "quadrant": e.quadrant, "valence": e.valence,
                 "arousal": e.arousal, "family": e.family}
                for e in entries
            ],
            "warnings": warnings,
        },
    }

    if args.coverage_only:
        provenance.write_run_record(
            config.phase_dir, title=f"PHASE 1 COVERAGE -- {config.run_name}",
            sections=sections, txt_name="phase1_coverage.txt",
            json_name="phase1_coverage.json",
        )
        print()
        print(RULE)
        print("--coverage-only: emotion set printed, no dataset loaded.")
        print("Re-run without the flag to assemble the stimulus table.")
        print(RULE)
        return 0

    if unknown:
        print("\nABORTED: unknown emotion words; fix the set before loading the dataset.",
              file=sys.stderr)
        return 3

    # --- assemble ---------------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 2  Load the dataset and assemble the stimulus table")
    print(RULE)
    dataset_sha = dataset.dataset_revision(config.dataset_revision)
    df, topic_split, report = build_stimulus_table(config, entries)
    print(dataset.format_validation_report(report))
    print()
    print(f"stimuli assembled : {len(df):,} rows")
    print(f"  emotional       : {int((df['source'] == 'emotion').sum()):,} "
          f"across {df.loc[df['source'] == 'emotion', 'emotion'].nunique()} emotions")
    if config.include_neutral:
        print(f"  neutral         : {int((df['source'] == 'neutral').sum()):,}")
    print(f"  topic split     : "
          + ", ".join(f"{k}={v}" for k, v in topic_split.counts.items()) + " topics")
    print(f"  dataset sha     : {dataset_sha}")

    # --- topic matching ---------------------------------------------------- #
    print()
    print(RULE)
    print("STEP 3  Verify topic matching (the property the design rests on)")
    print(RULE)
    matching = verify_topic_matching(df)
    print(f"topics in the set              : {matching.get('n_topics_total')}")
    print(f"every emotion has same topics  : {matching.get('all_emotions_same_topics')}")
    print(f"distinct stories-per-cell      : {matching.get('distinct_per_cell_counts')}")
    if matching.get("exactly_matched"):
        print("\n  OK  exactly topic-matched: same topics, same count in every cell.")
        print("      Subject matter is balanced across emotions by construction, so a")
        print("      difference between emotion vectors is not a difference in topic.")
    else:
        print("\n  NOT exactly matched. Emotions differ in topic coverage or in stories")
        print("  per topic, so part of any between-emotion difference is subject matter.")
        print(f"  Set stories_per_emotion to a multiple of "
              f"{matching.get('n_topics_total')} to fix.")

    # --- lengths vs pooling ------------------------------------------------ #
    print()
    print(RULE)
    print("STEP 4  Stimulus length vs the pooling offset")
    print(RULE)
    lengths = report_length_fit(df, config)
    print(f"words per stimulus       : min {lengths['words_min']}, "
          f"median {lengths['words_median']:.0f}, max {lengths['words_max']}")
    print(f"approx tokens (x1.35)    : min {lengths['approx_tokens_min']:.0f}, "
          f"median {lengths['approx_tokens_median']:.0f}")
    print(f"pooling needs            : > {lengths['tokens_needed']} tokens "
          f"(token_offset={lengths['token_offset']} + "
          f"min_pooled_tokens={lengths['min_pooled_tokens']})")
    print(f"shortest leaves          : ~{lengths['approx_pooled_min']:.0f} tokens pooled")
    print(f"stimuli at risk of skip  : {lengths['n_at_risk']}")
    if lengths["n_at_risk"] == 0:
        print("\n  OK  every stimulus clears the offset with room to spare.")
        print("      Worth noting why this matters: a 2-4 sentence vignette (~40-60")
        print("      tokens) would fall *below* the 50-token offset and be skipped")
        print("      entirely. Narrative-length stimuli are a requirement of this")
        print("      pooling scheme, not a stylistic preference.")
    else:
        print(f"\n  {lengths['n_at_risk']} stimuli may be skipped in Phase 2. Lower")
        print("  token_offset or drop the short stimuli.")

    # --- save -------------------------------------------------------------- #
    config.phase_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.stimuli_path, index=False)
    sample_path = config.phase_dir / "phase1_stimuli_sample.csv"
    df.groupby("emotion", sort=True).head(5).to_csv(sample_path, index=False)

    sections["dataset"] = {
        "dataset_id": dataset.HF_DATASET_ID,
        "dataset_sha": dataset_sha,
        "validation": report,
        "n_rows": len(df),
        "n_emotional": int((df["source"] == "emotion").sum()),
        "n_neutral": int((df["source"] == "neutral").sum()),
        "per_emotion_counts": {
            str(k): int(v) for k, v in df.groupby("emotion").size().items()
        },
        "per_quadrant_counts": {
            str(k): int(v) for k, v in df.groupby("quadrant").size().items()
        },
        "per_split_counts": {
            str(k): int(v) for k, v in df.groupby("split").size().items()
        },
        "topic_split": topic_split.counts,
    }
    sections["topic_matching"] = matching
    sections["length_fit"] = lengths
    sections["artifacts"] = {
        "stimuli_parquet": str(config.stimuli_path),
        "sample_csv": str(sample_path),
    }

    # --- the gate: examples + counts --------------------------------------- #
    print()
    print(RULE)
    show = None
    if len(entries) > 24:
        show = list(QUADRANT_ORDER) + [NEUTRAL_QUADRANT]
        print(f"GATE  {config.gate_examples_per_emotion} examples per emotion, "
              f"anchors + neutral only")
        print(RULE)
        print(f"{len(entries)} emotions would be ~{len(entries) * 3} vignettes to read.")
        print("Showing the labelled anchors and neutral; every stimulus is in the")
        print("parquet, and phase1_stimuli_sample.csv has 5 per emotion for all of them.")
    else:
        print(f"GATE  {config.gate_examples_per_emotion} examples per emotion, "
              "spread across topics")
        print(RULE)
    print_examples(df, config.gate_examples_per_emotion, quadrants=show)

    print()
    print(RULE)
    print("PER-EMOTION COUNTS")
    print(RULE)
    counts = df.groupby(["quadrant", "emotion"], sort=True).size()
    for (quadrant, emotion), n in counts.items():
        print(f"  {quadrant:<12} {emotion:<14} {n:>6,}")
    print(f"  {'':<12} {'TOTAL':<14} {len(df):>6,}")

    txt_path, json_path = provenance.write_run_record(
        config.phase_dir, title=f"PHASE 1 GATE -- {config.run_name}",
        sections=sections, txt_name="phase1_gate.txt", json_name="phase1_gate.json",
    )

    print()
    print(RULE)
    print("PHASE 1 VERDICT")
    print(RULE)
    coverage_ok = not warnings
    print(f"  quadrant coverage : {'PASS' if coverage_ok else 'REVIEW'} "
          f"({len(entries)} emotions"
          + (" + neutral" if config.include_neutral else "") + ")")
    print(f"  topic matching    : "
          f"{'PASS' if matching.get('exactly_matched') else 'REVIEW'}")
    print(f"  length vs pooling : "
          f"{'PASS' if lengths['n_at_risk'] == 0 else 'REVIEW'}")
    print(f"\n  stimuli : {config.stimuli_path}")
    print(f"  sample  : {sample_path}")
    print(f"  records : {txt_path}")
    print(f"            {json_path}")
    print()
    print("  Read the examples above before continuing. What matters is whether the")
    print("  text for each emotion reads as that emotion *without* leaning on the")
    print("  emotion word itself, and whether topics genuinely recur across emotions.")
    print()
    print("STOPPING at the Phase 1 gate. Phase 2 (extraction) has not run.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
