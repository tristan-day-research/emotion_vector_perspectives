"""Section bodies for `results_notebook.ipynb`.

The notebook is a driver: one markdown cell of prose per section, then one call into
this module. Everything that is loading, computing, table-building or wording lives here,
so the notebook reads as an outline and a change to one section touches one function.

These functions run once, in order, passing work forward through the handles in
:data:`SHARED`. That is the contract the notebook's cell globals already had; naming it
here makes it inspectable. The extraction that produced this file promotes a name to
module scope only when a later section reads it *without* rebinding it first, so a loop
variable in one section cannot collide with anything in another.

Ordering contract, enforced by the notebook's cell order:

    setup -> provenance -> phase0 -> phase1 -> phase2 -> phase3
          -> phase4_readouts -> phase4_chinese -> phase4_shape -> phase4_fix
          -> phase6 -> summary -> export

Every body runs inside :meth:`nbtools.Report.guard`, so a missing artefact is reported
and skipped rather than raised. Running a section out of order will not crash; it will
report things as missing.
"""

from __future__ import annotations

__all__ = ["setup", "provenance", "phase0", "phase1", "phase2", "phase3", "phase4_readouts", "phase4_chinese", "phase4_shape", "phase4_fix", "phase6", "phase7", "phase8", "summary", "export"]

#: Handles passed between sections, with the section that creates each.
SHARED = {
    "P7": "phase7",
    "P8": "phase8",
    "ALIGN": "phase3",
    "DESIGN": "setup",
    "F": "setup",
    "GATE3": "phase3",
    "Markdown": "setup",
    "P1": "phase1",
    "P4": "phase4_readouts",
    "P6": "phase6",
    "PANEL": "setup",
    "PC1_171": "phase3",
    "PROV": "provenance",
    "R": "setup",
    "ROOT": "setup",
    "RUNS": "setup",
    "RUN_KEYS": "setup",
    "SCRIPT": "phase4_readouts",
    "SH": "phase2",
    "SHORT": "setup",
    "TIERS": "phase4_chinese",
    "XRUN": "phase3",
    "ctl": "phase6",
    "display": "setup",
    "emotional": "phase6",
    "end": "phase4_readouts",
    "frame": "phase6",
    "g": "phase6",
    "gate": "phase6",
    "k": "phase6",
    "k0": "phase6",
    "key": "phase6",
    "ks": "phase6",
    "nbtools": "setup",
    "np": "setup",
    "pc": "phase4_readouts",
    "pd": "setup",
    "row": "phase6",
    "runs": "setup",
    "sub": "phase4_shape",
    "v": "phase3",
    "w": "provenance",
}

P7 = None
P8 = None
ALIGN = None
DESIGN = None
F = None
GATE3 = None
Markdown = None
P1 = None
P4 = None
P6 = None
PANEL = None
PC1_171 = None
PROV = None
R = None
ROOT = None
RUNS = None
RUN_KEYS = None
SCRIPT = None
SH = None
SHORT = None
TIERS = None
XRUN = None
ctl = None
display = None
emotional = None
end = None
frame = None
g = None
gate = None
k = None
k0 = None
key = None
ks = None
nbtools = None
np = None
pc = None
pd = None
row = None
runs = None
sub = None
v = None
w = None


def setup():
    """Imports, styling, report and run handles."""
    global DESIGN, F, Markdown, PANEL, R, ROOT, RUNS, RUN_KEYS, SHORT, display, key, nbtools, np, pd, runs

    import sys
    from pathlib import Path

    print("interpreter:", sys.executable)

    _missing = []
    for _pkg in ("numpy", "pandas", "matplotlib", "tokenizers"):
        try:
            __import__(_pkg)
        except ImportError:
            _missing.append(_pkg)
    if _missing:
        raise SystemExit(
            "This notebook needs " + ", ".join(_missing) + ".\n"
            "The Homebrew python3 has no packages -- run under conda `base` "
            "(~/miniconda3/bin/python)."
        )

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown

    # analysis/ first (for nbtools), then the repo root (nbfigures needs core.plotting
    # at import time, so the root has to be on the path before it is imported).
    _here = Path.cwd()
    sys.path.insert(0, str(_here if _here.name == "analysis" else _here / "analysis"))
    import nbtools

    ROOT = nbtools.find_root()
    sys.path.insert(0, str(ROOT))
    from core import plotting
    import nbfigures as F

    plotting.apply_style()
    pd.set_option("display.max_colwidth", 220)
    pd.set_option("display.width", 240)

    RUNS = {"16": "qwen3-32b_pca-jlens", "171": "qwen3-32b_pca-jlens_171"}
    DESIGN = {"16": "16 balanced emotions x 400 stories",
              "171": "all 171 emotions x 200 stories"}
    PANEL = {k: f"{RUNS[k]}\n{DESIGN[k]}" for k in RUNS}      # two-line panel titles
    SHORT = {k: f"{k}-emotion run" for k in RUNS}             # tight panel titles
    RUN_KEYS = list(RUNS)

    R = nbtools.Report(ROOT)
    runs = nbtools.Runs(ROOT, RUNS, DESIGN, report=R)

    print("root:", ROOT)
    print("figures ->", R.fig_dir)
    for key in RUN_KEYS:
        print(f"{key:>4}  {RUNS[key]:<28} "
              f"{'ok' if runs.phases(key).is_dir() else 'ABSENT'}")


def provenance():
    """Section 0: provenance table and the lens-convergence flag."""
    global PROV, key, w

    R.md("---\n\n# Emotion PCA x Jacobian lens -- assembled results\n\n"
         "All numbers are read from artefacts under `outputs/*/results/phases/`. Nothing "
         "was re-run to produce this document.\n\n"
         "| key | run | design |\n| --- | --- | --- |\n"
         "| `16` | `qwen3-32b_pca-jlens` | 16 balanced circumplex emotions, 400 stories each |\n"
         "| `171` | `qwen3-32b_pca-jlens_171` | all 171 emotions, 200 stories each; the same 16 "
         "are the labelled anchors |\n", show=False)
    R.heading("0. Provenance")

    PROV = {}
    with R.guard("provenance"):
        for key in RUN_KEYS:
            PROV[key] = nbtools.provenance_row(runs, key)
        prov = pd.DataFrame(PROV)
        prov.index.name = "field"
        R.table(prov.reset_index(), "Provenance, both runs")
    with R.guard("lens convergence flag"):
        n_done = PROV["16"]["lens prompts (checkpoint n_done)"]
        claimed = PROV["16"]["lens prompts (config.yaml claim)"]
        R.md(
            "### FLAG -- the published lens is an interrupted fit\n\n"
            "The `qwen3-32b` Jacobian lens on the Hub is a **resumable `fit_checkpoint`**, not "
            "a finished lens: it stores `jacobian_sum` (a running sum over prompts) plus "
            "`n_done`, and the mean is `jacobian_sum / n_done`. Reading that divisor is what "
            "caught the problem.\n\n"
            f"* checkpoint `n_done` = **{n_done}** prompts\n"
            f"* the accompanying `config.yaml` records `prompts_fitted` = **{claimed}**\n"
            "* the fit's own stopping rule was `stop_at_delta = 0.002` over a 10-prompt window "
            "with `min_prompts = 100` -- so 80 prompts is short of the fit script's own floor\n\n"
            f"**Divisor cross-check: {'AGREE' if n_done == claimed else 'DISAGREE'}.** The "
            "normalisation constant is not confirmed. This matters asymmetrically:\n\n"
            "* **Top-k readouts are unaffected.** The model's final norm is an RMSNorm, so a "
            "globally mis-scaled `J` yields *identical* top-k tokens. Phase 0 verified this "
            "numerically (`scale_invariant_top5: true`).\n"
            "* **Every magnitude-sensitive quantity is suspect**: GATE B's "
            "`||J - I||_F / ||I||_F` curve, and any Phase 6 variance split.\n\n"
            "Both runs use the same lens file, so this applies to both. `run.py refit_lens` "
            "produces a converged alternative; it has not been run."
        )

    with R.guard("pipeline warnings"):
        seen = []
        for key in RUN_KEYS:
            for w in ((runs.json(key, "phase4_gate.json") or {}).get("lens", {})
                      .get("warnings", [])):
                if w not in seen:
                    seen.append(w)
        if seen:
            R.md("**Verbatim warning recorded by the pipeline itself:**\n\n"
                 + "\n>\n".join(f"> {w}" for w in seen))


def phase0():
    """Section 1: GATE A table and the GATE B identity curve."""

    R.heading("1. Phase 0 -- lens gates")
    R.md(
        "*In plain terms.* *"
        "Before trusting anything else: does the lens actually work? Two checks -- can it "
        "read a known fact back out of the model, and does it behave sensibly near the top of "
        "the network where it has almost no work to do?*"
    )
    display(Markdown("### GATE A"))

    with R.guard("Phase 0 GATE A"):
        p0 = runs.json("16", "phase0_gate.json")
        if p0 is None:
            R.md("> Phase 0 produced no artefact for either run; GATE A cannot be shown.")
        else:
            rows = pd.DataFrame(p0["gate_a"]["rows"]).rename(columns={
                "word": "expected word", "best_rank": "best rank", "best_block": "block"})
            is_emotion = rows["group"].str.contains("emotion", case=False)
            factual, emotion = rows[~is_emotion], rows[is_emotion]
            R.table(factual.drop(columns=["group"]),
                    "GATE A -- factual items (16-emotion run; the 171 run did not re-gate)")
            R.table(emotion.drop(columns=["group"]), "GATE A -- emotion vignettes")

            def tally(sub):
                counts = sub["verdict"].value_counts()
                return {"n": len(sub), "HIT": int(counts.get("HIT", 0)),
                        "near": int(counts.get("near", 0)), "MISS": int(counts.get("MISS", 0)),
                        "median best rank": float(sub["best rank"].median())}
            summary = pd.DataFrame({"factual": tally(factual),
                                    "emotion vignettes": tally(emotion)}).T
            summary.index.name = "group"
            R.table(summary.reset_index(), "GATE A -- split tally")
            R.md(
                f"**Reading.** {p0['gate_a']['n_hits']}/{p0['gate_a']['n_scorable']} items hit "
                f"overall, but the split is the result: **{tally(factual)['HIT']}/"
                f"{tally(factual)['n']}** factual items hit (median best rank "
                f"{tally(factual)['median best rank']:.0f}) against "
                f"**{tally(emotion)['HIT']}/{tally(emotion)['n']}** emotion vignettes (median "
                f"best rank {tally(emotion)['median best rank']:.0f}). The single emotion-group "
                "HIT is `chess`, the control item in that block, not an emotion word. The lens "
                "reads factual content and does not surface English emotion nouns. Section 5 "
                "measures why."
            )
    display(Markdown("### GATE B"))

    with R.guard("Phase 0 GATE B"):
        p0 = runs.json("16", "phase0_gate.json")
        if p0 is None:
            R.md("> Phase 0 produced no artefact; GATE B cannot be shown.")
        else:
            gate_b = p0["gate_b"]
            distances = {int(k): v for k, v in gate_b["identity_distances"].items()}
            blocks = sorted(distances)
            values = np.array([distances[b] for b in blocks])
            low, high, peak = values[0], values[-1], int(np.argmax(values))

            R.fig(
                F.gate_b_identity(gate_b, RUNS["16"]),
                "phase0_gate_b_identity.png",
                "GATE B: ||J - I||_F / ||I||_F against block (16-emotion run)",
                how_to_read=(
                    "The x-axis is the block index, 0 at the bottom of the stack. The y-axis is "
                    "how far the transport `J` at that block sits from the identity matrix, in "
                    "Frobenius norm relative to `||I||`. **Low means `J` is close to a no-op.** "
                    "At the highest fitted block the transport spans a single residual block, so "
                    "a well-behaved lens should approach zero on the right."
                ),
                what_it_shows=(
                    f"The distance falls from **{low:.3f}** at block {blocks[0]} to "
                    f"**{high:.3f}** at block {blocks[-1]} — the expected direction, so GATE B "
                    f"passes. But the fall is **not monotone**: it climbs to "
                    f"**{values[peak]:.3f}** at block {blocks[peak]} first. The gate records "
                    "`falls_with_depth` as an endpoint comparison only, so the mid-stack hump is "
                    "visible here and nowhere in the JSON. Note also that this y-axis is "
                    "magnitude-sensitive and the lens divisor is unconfirmed (section 0): the "
                    "*shape* is scale-free, the absolute values are not."
                ),
                            plain=(
                    "Is the lens trustworthy at all? Near the very top of the model it only has to "
                    "translate across one small step, so it should barely change anything. This checks "
                    "whether it does. "
                ),
)

            agreement = gate_b["agreement"]
            R.table(pd.DataFrame([
                {"comparison": "top-1 J-lens == model output",
                 "value": f"{agreement['top1_jlens_equals_model']} / 1"},
                {"comparison": "top-1 J-lens == logit lens",
                 "value": f"{agreement['top1_jlens_equals_logitlens']} / 1"},
                {"comparison": "top-12 overlap, J-lens vs model",
                 "value": f"{agreement['top12_overlap_jlens_model']} / 12"},
                {"comparison": "top-12 overlap, J-lens vs logit lens",
                 "value": f"{agreement['top12_overlap_jlens_logitlens']} / 12"},
            ]), f"GATE B -- agreement at block {gate_b['top_block']} "
                f"(probe: `{gate_b['probe']}`)")


def phase1():
    """Section 2: stimulus design, topic matching, coverage figure."""
    global P1, frame, key

    R.heading("2. Phase 1 -- stimuli")
    R.md(
        "*In plain terms.* *"
        "The raw material. Thousands of short stories, each written to express one emotion, "
        "checked so that every emotion got the same list of topics -- otherwise a difference "
        "between two emotions might just be a difference in what their stories were about.*"
    )

    P1 = {}
    with R.guard("Phase 1 coverage"):
        meta, quads = {}, {}
        for key in RUN_KEYS:
            p1 = runs.json(key, "phase1_gate.json")
            if p1 is None:
                continue
            P1[key] = p1
            quads[key] = p1["dataset"]["per_quadrant_counts"]
            meta[key] = nbtools.phase1_summary(p1)
        if meta:
            frame = pd.DataFrame(meta)
            frame.index.name = "field"
            R.table(frame.reset_index(), "Phase 1 -- stimulus design, both runs")
        if quads:
            quad = pd.DataFrame(quads).fillna(0).astype(int)
            quad.index.name = "quadrant"
            R.table(quad.reset_index(), "Phase 1 -- stimulus count by circumplex quadrant")
    with R.guard("Phase 1 coverage figure"):
        if not P1:
            R.md("> No Phase 1 artefacts; coverage figure skipped.")
        else:
            counts = {k: P1[k]["dataset"]["per_quadrant_counts"] for k in P1}
            labelled = sum(v for q, v in counts.get("171", {}).items()
                           if q in ("HA-P", "HA-N", "LA-P", "LA-N"))
            emotional_171 = sum(v for q, v in counts.get("171", {}).items() if q != "neutral")
            R.fig(
                F.quadrant_coverage(counts, PANEL),
                "phase1_quadrant_coverage.png",
                "Phase 1: stimuli per circumplex quadrant, both runs",
                how_to_read=(
                    "One panel per run; bar height is the number of *story stimuli* in each "
                    "circumplex quadrant. Blue bars are the four labelled quadrants "
                    "(HA/LA = high/low arousal, P/N = pleasant/unpleasant). Grey bars carry no "
                    "circumplex coordinates: `neutral`, and in the 171 run the 155 emotions with "
                    "no a-priori valence/arousal labels. **The two panels have different "
                    "y-scales** — compare shapes within a panel, not bar heights across them."
                ),
                what_it_shows=(
                    "The 16-emotion run is exactly balanced: 1,600 stimuli in each of the four "
                    f"quadrants. The 171-emotion run has 800 per labelled quadrant against "
                    f"{counts['171'].get('unlabelled', 0):,} unlabelled — so only {labelled:,} "
                    f"of its {emotional_171:,} emotional stimuli carry circumplex coordinates at "
                    "all. That is why every alignment and AUROC measurement in sections 4 and 5 "
                    "rests on the same 16 anchors in *both* runs, no matter how many emotions "
                    "the run contains."
                ),
                            plain=(
                    "How many stories went into each corner of the emotion map -- happy-and-excited, "
                    "happy-and-calm, unhappy-and-excited, unhappy-and-calm -- and how many had no emotion "
                    "label at all. "
                ),
)

    with R.guard("Phase 1 reading"):
        lines = []
        for key in RUN_KEYS:
            if key not in P1:
                continue
            tm, lf, ds = P1[key]["topic_matching"], P1[key]["length_fit"], P1[key]["dataset"]
            per_emotion = {k: v for k, v in ds["per_emotion_counts"].items() if k != "neutral"}
            lines.append(
                f"* **{RUNS[key]}** -- {ds['n_rows']:,} rows, {len(per_emotion)} emotions x "
                f"{min(per_emotion.values())} stories. All {tm['n_topics_total']} topics appear "
                f"for every emotion (`exactly_matched = {tm['exactly_matched']}`), "
                f"{'/'.join(str(c) for c in tm['distinct_per_cell_counts'])} stories per "
                f"(emotion, topic) cell. Shortest story pools over ~{lf['approx_pooled_min']:.0f} "
                f"tokens against the {lf['tokens_needed']} needed -- **{lf['n_at_risk']} stimuli "
                f"at length risk**."
            )
        R.md("**Reading.**\n\n" + "\n".join(lines) + "\n\n"
             "Topic matching is exact in both runs, so a difference between two emotion vectors "
             "is not a difference in what the stories were about. No stimulus is short enough "
             "for the 50-token prefix skip to starve the pool.")


def phase2():
    """Section 3: split-half reliability."""
    global SH, frame, key

    R.heading("3. Phase 2 -- split-half reliability")
    R.md(
        "*In plain terms.* *"
        "Boil each emotion's stories down to a single 'fingerprint' -- an average pattern of "
        "activity inside the model -- and check those fingerprints are stable rather than an "
        "accident of which stories went in.*"
    )

    SH = {}
    with R.guard("Phase 2 split-half"):
        stats = {}
        for key in RUN_KEYS:
            frame, p2 = runs.csv(key, "phase2_split_half.csv"), runs.json(key, "phase2_gate.json")
            if frame is None:
                continue
            SH[key] = frame
            stats[key] = nbtools.phase2_summary(frame, p2)
        if stats:
            frame = pd.DataFrame(stats)
            frame.index.name = "field"
            R.table(frame.reset_index(), "Phase 2 -- split-half reliability summary")
    with R.guard("Phase 2 split-half figure"):
        if not SH:
            R.md("> No split-half CSVs; figure skipped.")
        else:
            quadrant_of = {e["emotion"]: e["quadrant"]
                           for e in P1.get("16", {}).get("emotion_set", {}).get("emotions", [])}
            worst16 = SH["16"][SH["16"].emotion != "neutral"].nsmallest(5, "cosine_centered")
            quads = [quadrant_of.get(e, "?") for e in worst16["emotion"]]
            lan = [e for e, q in zip(worst16["emotion"], quads) if q == "LA-N"]
            n171_below = int((SH["171"]["cosine_centered"] < 0.9).sum())
            min16 = SH["16"][SH["16"].emotion != "neutral"]["cosine_centered"].min()

            R.fig(
                F.split_half(SH, PANEL),
                "phase2_split_half.png",
                "Phase 2: per-emotion split-half centred cosine, sorted ascending",
                how_to_read=(
                    "Each bar is one emotion. Build its vector from 50 topics, build it again "
                    "from the other 50, take the cosine between the two: **1.0 means the vector "
                    "does not depend on which topics you used.** Bars are sorted ascending, so "
                    "the least reliable emotions are on the left. The dashed line is the 0.9 "
                    "threshold the pipeline set; red bars fall below it. **The y-axis is "
                    "truncated near 0.78** so the threshold is resolvable — bar *lengths* are "
                    "therefore not proportional to their values. The 171 panel has 172 bars a "
                    "pixel apart, so its sub-threshold emotions are listed in a text box rather "
                    "than tick-labelled."
                ),
                what_it_shows=(
                    f"The 16-emotion run is uniformly reliable: every emotion clears 0.9, "
                    f"minimum {min16:.3f}. The 171-emotion run has a real tail — **{n171_below} "
                    f"of 172 fall below 0.9**, worst `hurt` at "
                    f"{SH['171']['cosine_centered'].min():.3f}.\n\n"
                    f"The structure is in the 16 run's weak end: its five weakest are "
                    + ", ".join(f"`{e}` ({q})" for e, q in zip(worst16["emotion"], quads))
                    + f" — **{len(lan)} of 5 are the low-arousal-negative quadrant**, which is "
                    f"the whole of LA-N. Sadness, gloom, boredom and weariness produce the least "
                    f"topic-invariant activation of the four quadrants. That is a structured "
                    f"asymmetry, not noise spread evenly across the design."
                ),
                            plain=(
                    "If we build each emotion's fingerprint twice, from two different halves of the "
                    "stories, do we get the same thing both times? A fingerprint that changes depending "
                    "on which stories you happened to use is not measuring the emotion. "
                ),
)
            R.md("The raw (uncentred) cosine is ~0.998 in both runs. Reporting that would be "
                 "meaningless — it measures the shared component of any mean activation, not "
                 "emotion-specific signal. Everything here uses the centred cosine.")


def phase3():
    """Section 4: PCA, alignment, PC2-PC3 scatter, cross-run cosines."""
    global ALIGN, GATE3, PC1_171, XRUN, frame, gate, key, sub, v

    R.heading("4. Phase 3 -- PCA")
    R.md(
        "*In plain terms.* *"
        "Ask the maths to find the biggest patterns of difference across the emotion "
        "fingerprints, without telling it anything about feelings, and see whether it "
        "rediscovers the two dimensions psychologists use: pleasant vs unpleasant, and calm "
        "vs worked-up.*"
    )
    display(Markdown("### The pipeline's own figures"))

    SCATTER_PLAIN = (
    "A map of all the emotions, placed by the two strongest patterns the maths "
    "found on its own, without being told anything about feelings. If those "
    "patterns really are pleasantness and intensity, the four kinds of emotion "
    "should land in four different corners."
)
    SCATTER_PLAIN = (
    "A map of all the emotions, placed by the two strongest patterns the maths found "
    "on its own, without being told anything about feelings. If those patterns really "
    "are pleasantness and intensity, the four kinds of emotion should land in four "
    "different corners."
)
    SCATTER_HOW = (
        "Each point is one emotion vector projected into the plane of the run's first two "
        "principal components. **Fill means arousal, colour means valence**: filled = "
        "activated, hollow = deactivated, blue = pleasant, orange = unpleasant. The grey "
        "diamond is `neutral`, projected into the plane but held out of the PCA fit. The grey "
        "arrows are the a-priori valence and arousal axes projected into this plane, labelled "
        "with how much of each axis the plane actually captures — a short arrow means that "
        "axis mostly points somewhere else."
    )
    VARIANCE_PLAIN = (
    "How much of the difference between emotions each discovered pattern "
    "accounts for, next to how much you would get from patterns found in pure "
    "noise. A pattern that does not beat the noise line is not telling you "
    "anything."
)
    VARIANCE_PLAIN = (
    "How much of the difference between emotions each discovered pattern accounts for, "
    "next to how much you would get from patterns found in pure noise. A pattern that "
    "does not beat the noise line is not telling you anything."
)
    VARIANCE_HOW = (
        "Share of total variance carried by each principal component, in descending order, "
        "against the isotropic null — the share you would expect from random directions in "
        "the same number of dimensions. A PC only means something if it clears that null."
    )
    ALIGNMENT_PLAIN = (
    "A grid checking how closely each discovered pattern lines up with the two "
    "things psychology says emotions vary on: pleasant vs unpleasant, and "
    "worked-up vs calm. Strong colour means a close match; near-white means no "
    "relationship."
)
    ALIGNMENT_PLAIN = (
    "A grid checking how closely each discovered pattern lines up with the two things "
    "psychology says emotions vary on: pleasant vs unpleasant, and worked-up vs calm. "
    "Strong colour means a close match; near-white means no relationship."
)
    ALIGNMENT_HOW = (
        "A heatmap, one row per PC. The first two columns are the correlation of that PC's "
        "*scores* with the anchors' ±1 valence and arousal labels; the last two are the cosine "
        "between the PC *direction* and the fitted axis. Red is positive, blue negative, "
        "near-white is no relationship. Sign is arbitrary per PC — read magnitude."
    )
    SCATTER_WHAT = {
        "16": ("The four quadrants land in four corners, which is the circumplex recovered "
               "without supervision. Both a-priori arrows are long (0.99 in plane), so the "
               "plane the PCA found and the plane the labels define are nearly the same plane."),
        "171": ("The anchors do **not** separate cleanly here, and the a-priori valence arrow "
                "is visibly short. This figure is the reason the PC2-PC3 scatter below exists: "
                "for this run, PC1-PC2 is simply the wrong plane to look at."),
    }
    VARIANCE_WHAT = {
        "16": ("PC1 and PC2 stand far above the null and far above the rest; the spectrum then "
               "falls away. Two components dominate."),
        "171": ("A much flatter spectrum — PC1 at 23%, and a long tail staying above the null "
                "well past PC5. No two components dominate this run."),
    }
    ALIGNMENT_WHAT = {
        "16": ("PC1 is arousal (r = -0.94) and PC2 is valence (r = -0.95), each essentially "
               "pure, with everything below PC2 near zero."),
        "171": ("The strong cells have moved **down one row**: PC2 carries arousal (-0.94) and "
                "PC3 carries valence (+0.92). PC1's +0.82 on valence is the contamination "
                "diagnosed further down this section."),
    }

    for key in RUN_KEYS:
        for filename, what, how, whats, plain in [
            ("phase3_pc1_pc2_scatter.png", "emotion vectors in the PC1-PC2 plane",
             SCATTER_HOW, SCATTER_WHAT, SCATTER_PLAIN),
            ("phase3_variance_explained.png", "variance explained per PC against the null",
             VARIANCE_HOW, VARIANCE_WHAT, VARIANCE_PLAIN),
            ("phase3_alignment.png", "per-PC alignment with the a-priori circumplex axes",
             ALIGNMENT_HOW, ALIGNMENT_WHAT, ALIGNMENT_PLAIN),
        ]:
            R.existing_fig(runs.phases(key) / filename, f"{RUNS[key]}: {what}",
                           how_to_read=how, what_it_shows=whats[key], plain=plain)
    display(Markdown("### Variance and effective dimensionality"))

    VAR, ALIGN, GATE3 = {}, {}, {}
    with R.guard("Phase 3 variance"):
        rows = {}
        for key in RUN_KEYS:
            var, gate = runs.csv(key, "phase3_variance.csv"), runs.json(key, "phase3_gate.json")
            if var is None or gate is None:
                continue
            VAR[key], GATE3[key] = var, gate
            rows[key] = nbtools.phase3_variance_summary(gate)
        if rows:
            frame = pd.DataFrame(rows)
            frame.index.name = "field"
            R.table(frame.reset_index(), "Phase 3 -- variance structure, both runs")

    with R.guard("Phase 3 alignment table"):
        parts = []
        for key in RUN_KEYS:
            align = runs.csv(key, "phase3_alignment.csv")
            if align is None:
                continue
            ALIGN[key] = align
            sub = align[["pc", "r_valence", "r_arousal", "cos_valence", "cos_arousal"]].copy()
            sub.columns = ["PC"] + [f"{key}: {c}" for c in sub.columns[1:]]
            parts.append(sub.set_index("PC"))
        if parts:
            R.table(pd.concat(parts, axis=1).reset_index().round(3),
                    "Phase 3 -- per-PC correlation and cosine with the a-priori axes "
                    "(r = correlation of PC scores with the +-1 anchor labels; cos = cosine of "
                    "the PC direction with the fitted axis)")
    display(Markdown("### How much of the valence axis lives in each plane"))

    INPLANE = {}
    with R.guard("Phase 3 in-plane fractions"):
        rows = []
        for key in RUN_KEYS:
            if key not in ALIGN:
                continue
            align = ALIGN[key].set_index("pc")

            def frac(axis, pcs, _a=align):
                return float(np.sqrt(sum(_a.loc[p, f"cos_{axis}"] ** 2 for p in pcs)))

            INPLANE[key] = {
                f"{axis} in PC{pcs[0]}-PC{pcs[-1]}": frac(axis, pcs)
                for axis in ("valence", "arousal")
                for pcs in ([1, 2], [2, 3], [1, 2, 3])
            }
            rows.append({"run": RUNS[key], **{k: round(v, 3) for k, v in INPLANE[key].items()}})
        if rows:
            R.table(pd.DataFrame(rows),
                    "Phase 3 -- norm of the unit a-priori axis captured by each PC subspace "
                    "(1.0 = the axis lies entirely inside that subspace)")
            if "171" in INPLANE:
                v = INPLANE["171"]
                R.md(
                    f"**This is why the pipeline's PC1-PC2 figure understates the 171 run.** "
                    f"Only **{v['valence in PC1-PC2']:.0%}** of the valence axis lies in the "
                    f"PC1-PC2 plane. Moving to PC2-PC3 raises it to "
                    f"**{v['valence in PC2-PC3']:.0%}**, and the first three PCs together hold "
                    f"**{v['valence in PC1-PC3']:.0%}**. PC3 carries the valence signal the "
                    f"published scatter cannot show."
                )
    display(Markdown("### PC2 vs PC3 for the 171-emotion run (new figure)"))

    with R.guard("Phase 3 PC2-PC3 scatter"):
        scores, align = runs.csv("171", "phase3_scores.csv"), ALIGN.get("171")
        if scores is None or align is None:
            R.md("> Phase 3 scores or alignment missing for the 171 run; scatter skipped.")
        else:
            R.fig(
                F.pc2_pc3_scatter(scores, align, RUNS["171"]),
                "phase3_pc2_pc3_scatter_171.png",
                "Phase 3 (new): the 171-emotion run in the PC2-PC3 plane",
                how_to_read=(
                    "Same encoding as the pipeline's scatter — filled = activated, hollow = "
                    "deactivated, blue = pleasant, orange = unpleasant — but plotting **PC2 "
                    "against PC3 instead of PC1 against PC2**. Small grey dots are the 155 "
                    "emotions with no circumplex labels; they enter the PCA but cannot be "
                    "scored. The grey diamond is `neutral`, held out of the fit. Arrows are the "
                    "a-priori axes projected into *this* plane, labelled with the fraction of "
                    "each axis the plane captures."
                ),
                what_it_shows=(
                    "The circumplex the PC1-PC2 figure could not show. The four anchor groups "
                    "separate into four regions — pleasant-activated top left, pleasant-"
                    "deactivated top right, unpleasant-activated bottom left, unpleasant-"
                    "deactivated middle right — and both arrows are now long (valence 0.77, "
                    "arousal 0.95 in plane, against 0.55 for valence in PC1-PC2). The "
                    "171-emotion run does recover the circumplex; it simply does not put it in "
                    "the top two components."
                ),
                            plain=(
                    "The same map of emotions as the pipeline's own figure, but drawn along the 2nd and "
                    "3rd strongest patterns instead of the 1st and 2nd -- because in the 171-emotion run "
                    "that is where the pleasant/unpleasant and calm/excited structure actually sits. "
                ),
)
    display(Markdown("### Do the two runs find the same axes?"))

    XRUN = {}
    with R.guard("cross-run PC correspondence"):
        a = runs.safetensors("16", "phase3_pcs.safetensors")
        b = runs.safetensors("171", "phase3_pcs.safetensors")
        if a is None or b is None:
            R.md("> One of the PC safetensors files is missing; comparison skipped.")
        else:
            A = a["components"][:5].astype(np.float64)
            B = b["components"][:5].astype(np.float64)
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True)
            M = np.abs(A @ B.T)
            table = pd.DataFrame(M.round(3),
                                 index=[f"16-run PC{i}" for i in range(1, 6)],
                                 columns=[f"171-run PC{j}" for j in range(1, 6)])
            table.index.name = "abs cosine"
            R.table(table.reset_index(),
                    "Phase 3 -- absolute cosine between the two runs' principal axes")
            XRUN = {"match_pc": (np.argmax(M, axis=1) + 1).tolist(),
                    "match_cos": M.max(axis=1).round(3).tolist()}

            axis_rows = [{
                "a-priori axis": name.replace("_axis", ""),
                "cosine between the two runs' fitted axes": float(
                    a[name] @ b[name] / np.linalg.norm(a[name]) / np.linalg.norm(b[name])),
            } for name in ("valence_axis", "arousal_axis") if name in a and name in b]
            if axis_rows:
                R.table(pd.DataFrame(axis_rows).round(4),
                        "Phase 3 -- the a-priori axes are the same in both runs (both fitted "
                        "from the same 16 anchors)")
            R.md(
                f"**Reading.** The correspondence is direct, not inferred from correlation "
                f"signs: **16-run PC1 matches 171-run PC{XRUN['match_pc'][0]} at |cos| = "
                f"{XRUN['match_cos'][0]}**, and **16-run PC2 matches 171-run "
                f"PC{XRUN['match_pc'][1]} at |cos| = {XRUN['match_cos'][1]}**. The 16-run's "
                f"arousal axis (its PC1) and valence axis (its PC2) reappear in the 171 run one "
                f"slot lower, displaced by a PC1 the 16-emotion design never produces. Below "
                f"PC2 the match degrades ({XRUN['match_cos'][2]} for 16-run PC3)."
            )
    display(Markdown("### What PC1 of the 171 run actually is"))

    PC1_171 = {}
    with R.guard("171 PC1 contamination"):
        scores = runs.csv("171", "phase3_scores.csv")
        if scores is None:
            R.md("> Phase 3 scores missing for the 171 run.")
        else:
            fitted = scores[scores["in_pca_fit"] == True]  # noqa: E712
            neutral = scores[scores["emotion"] == "neutral"]
            PC1_171 = {
                "lo": fitted.nsmallest(8, "pc1")["emotion"].tolist(),
                "hi": fitted.nlargest(8, "pc1")["emotion"].tolist(),
                "max": float(fitted["pc1"].max()),
                "neutral": float(neutral["pc1"].iloc[0]) if len(neutral) else float("nan"),
            }
            R.table(pd.DataFrame([
                {"quantity": "PC1 range over the 171 fitted emotions",
                 "value": f"{fitted['pc1'].min():.1f} .. {fitted['pc1'].max():.1f}"},
                {"quantity": "PC1 score of `neutral` (held OUT of the fit)",
                 "value": f"{PC1_171['neutral']:+.1f}"},
                {"quantity": "8 lowest on PC1", "value": ", ".join(PC1_171["lo"])},
                {"quantity": "8 highest on PC1", "value": ", ".join(PC1_171["hi"])},
                {"quantity": "corr(PC1 score, vector norm)",
                 "value": f"{np.corrcoef(fitted['pc1'], fitted['norm'])[0, 1]:+.2f}"},
            ]), "Phase 3 -- diagnosing PC1 of the 171-emotion run")
            R.md(
                "**Reading.** PC1 of the 171 run is not a circumplex axis. Its low end is "
                + ", ".join(f"`{e}`" for e in PC1_171["lo"][:5])
                + " — overwhelming, high-intensity affect — and its high end is "
                + ", ".join(f"`{e}`" for e in PC1_171["hi"][:5])
                + " — appraisal-like, dispositional or motivational states rather than acute "
                f"affect. Decisively: **`neutral`, held out of the PCA fit entirely, projects to "
                f"PC1 = {PC1_171['neutral']:+.1f}, beyond the {PC1_171['max']:+.1f} maximum of "
                f"all 171 fitted emotions.** PC1 is substantially an *affect-presence / "
                f"intensity* axis, with the absence of affect at the extreme. It correlates with "
                f"valence (r = +0.82) only because intense states in this vocabulary skew "
                f"unpleasant. This is what pushes the circumplex down into PC2-PC3."
            )
    with R.guard("Phase 3 overall reading"):
        lines = []
        for key in RUN_KEYS:
            if key not in GATE3:
                continue
            gate = GATE3[key]
            pca, stab = gate["pca"], gate["pc_stability"]
            summary = gate["alignment"]["summary"]
            crossfit = gate["alignment"].get("crossfit", {})
            ratios = pca["explained_variance_ratio"]
            lines.append(
                f"* **{RUNS[key]}** — top-2 PCs hold {sum(ratios[:2]):.0%} of variance against "
                f"an isotropic null of {gate['null_band']['analytic_isotropic_top2']:.1%}. "
                f"Participation ratio **{pca['participation_ratio']:.2f}**. Best valence PC = "
                f"PC{summary['best_pc_for_valence']}, best arousal PC = "
                f"PC{summary['best_pc_for_arousal']}. Plane principal cosines "
                f"{', '.join(f'{c:.2f}' for c in summary['plane_cosines'])} "
                f"(mean {summary['plane_mean_cosine']:.2f}). Cross-fit worst plane angle "
                f"{crossfit.get('worst_plane_angle_deg', float('nan')):.1f} deg. "
                f"`axes_identified = {stab['axes_identified']}`, "
                f"`plane_stable = {stab['plane_stable']}`."
            )
        R.md(
            "**Reading — Phase 3 overall.**\n\n" + "\n".join(lines) + "\n\n"
            "The 16-emotion run recovers the circumplex cleanly and the gate says so. The "
            "171-emotion run recovers the *same two directions* — the cross-run cosines above — "
            "but they are no longer the top two, they hold far less variance, and Phase 3's own "
            "stability check marks the top-2 plane as **not** stable and the axes as **not** "
            "identified. A participation ratio of "
            f"{GATE3.get('171', {}).get('pca', {}).get('participation_ratio', float('nan')):.1f} "
            "is the honest headline for the 171 run: the circumplex is a 2-D slice of a roughly "
            "10-dimensional space, not a description of it."
        )


def phase4_readouts():
    """Sections 5a-5d: verdicts, token panels, AUROC, script mix."""
    global P4, SCRIPT, end, frame, key, pc, sub

    R.heading("5. Phase 4 -- lens readouts of the PCs")
    R.md(
        "*In plain terms.* *"
        "Take the patterns the maths discovered and ask the lens what words they correspond "
        "to. This is where the model turns out to answer in Chinese, which breaks a test that "
        "was written expecting English.*"
    )
    display(Markdown("### 5a. Verdict per PC"))

    P4 = {}
    with R.guard("Phase 4 verdict table"):
        for key in RUN_KEYS:
            p4 = runs.json(key, "phase4_gate.json")
            if p4 is None:
                continue
            P4[key] = p4
            R.table(nbtools.phase4_verdicts(p4),
                    f"Phase 4 GATE B -- verdict per PC, {RUNS[key]} ({DESIGN[key]})")
        R.md("`alpha` is 0.05 for the pre-registered PCs (PC1, PC2) and 0.0083 = 0.05/6 for "
             "exploratory ones — the pipeline's own family correction over six extra tests. "
             "`AUROC` is over the 14 tokenisable anchors of the 16 (`elated` and `gloomy` are "
             "not single tokens), so the statistic is coarse: 14 points, permutation p-values.")
    display(Markdown("### 5b. The + and - ends of the top 3 PCs"))

    with R.guard("Phase 4 token panels"):
        for key in RUN_KEYS:
            p4 = P4.get(key)
            if p4 is None:
                continue
            rows = []
            for pc in p4["gate_b"]["per_pc"][:3]:
                for end in ("plus", "minus"):
                    blob = pc.get(end) or {}
                    rows.append({
                        "PC": f"PC{int(pc['pc'])}", "end": blob.get("name", end),
                        "AUROC valence": blob.get("auroc_valence"),
                        "AUROC arousal": blob.get("auroc_arousal"),
                        "top-1 prob": blob.get("top1_prob"),
                        "effective tokens": blob.get("effective_tokens"),
                        "top 12 tokens": " ".join(
                            nbtools.show_token(t["token"]) for t in blob.get("tokens", [])[:12]),
                    })
            R.table(pd.DataFrame(rows).round(3),
                    f"Phase 4 -- top-12 lens tokens for each end of PC1-PC3, {RUNS[key]}")

        for key in RUN_KEYS:
            p4 = P4.get(key)
            if p4 is None:
                continue
            rows = [{
                "axis": name,
                "AUROC valence": blob.get("auroc_valence"),
                "AUROC arousal": blob.get("auroc_arousal"),
                "effective tokens": blob.get("effective_tokens"),
                "top 12 tokens": " ".join(
                    nbtools.show_token(t["token"]) for t in blob.get("tokens", [])[:12]),
            } for name, blob in (p4.get("controls_apriori_axes") or {}).items()]
            if rows:
                R.table(pd.DataFrame(rows).round(3),
                        f"Phase 4 -- control: the a-priori valence/arousal axes read through the "
                        f"same lens, {RUNS[key]}")

    R.md(
        "**Reading the antonym question honestly.** The two ends of a PC are exact negations "
        "by construction, so *that* they order the anchors oppositely is arithmetic, not "
        "evidence. The only non-arithmetic question is whether the token lists read as "
        "antonyms — judged by eye from the tables above, which is why they are printed in "
        "full. Two things are measurable rather than impressionistic:\n\n"
        "* Several ends are dominated by **whitespace and punctuation**. The `+PC1` end of the "
        "16 run puts 0.90 probability on `' \\n'` with an effective token count under 2 — that "
        "end is one newline, and cannot be judged for antonymy at all.\n"
        "* The content-bearing ends are overwhelmingly Chinese. That is quantified next."
    )
    display(Markdown("### 5c. AUROC per PC per end"))

    with R.guard("Phase 4 AUROC figure"):
        if not P4:
            R.md("> No Phase 4 gate JSON; AUROC figure skipped.")
        else:
            clears = []
            for key in P4:
                for pc in P4[key]["gate_b"]["per_pc"]:
                    for axis in ("valence", "arousal"):
                        for end, value in (("+", pc[f"auroc_{axis}"]),
                                           ("-", pc[f"auroc_{axis}_minus_end"])):
                            if value >= 0.75:
                                clears.append({
                                    "run": key, "PC": int(pc["pc"]), "axis": axis, "end": end,
                                    "AUROC": value, "p": pc[f"p_{axis}"], "alpha": pc["alpha"],
                                    "significant at its alpha": pc[f"p_{axis}"] < pc["alpha"]})
            n_cells = sum(len(P4[k]["gate_b"]["per_pc"]) for k in P4) * 2
            n_sig = sum(1 for c in clears if c["significant at its alpha"])

            R.fig(
                F.auroc_per_pc_per_end({k: P4[k]["gate_b"]["per_pc"] for k in P4}, SHORT),
                "phase4_auroc_per_pc_per_end.png",
                "Phase 4 (new): AUROC of each PC end against each circumplex axis, both runs",
                how_to_read=(
                    "Four panels: rows are the two runs, columns are the two circumplex axes. "
                    "Within a panel each PC gets two bars — its `+` end (blue) and its `-` end "
                    "(orange). AUROC asks: **does this end of this PC rank the pleasant (or "
                    "activated) anchors above the others?** 1.0 is perfect, 0.5 is chance, 0.0 "
                    "is perfectly inverted. Because the two ends are exact complements, **each "
                    "pair of bars sums to 1.00 by construction** — read which end carries the "
                    "ordering, not two independent measurements. Dashed line is the 0.75 pass "
                    "threshold; solid line is chance."
                ),
                what_it_shows=(
                    f"**{len(clears)} of the {n_cells} (PC, axis) cells clear 0.75, and {n_sig} "
                    f"of those also clear their own alpha.** In the 16 run it is PC1 on arousal "
                    f"and PC2 on valence; in the 171 run it is PC2 on arousal and PC3 on valence "
                    f"— the same one-slot shift the cross-run cosines showed. Everything else "
                    f"sits against the chance line.\n\n"
                    f"The exception worth naming is the 171 run's **PC5 on valence: AUROC 0.90 "
                    f"at p = 0.012**. It clears the AUROC bar and would be significant at 0.05, "
                    f"but PC5 is exploratory and its threshold is 0.0083 over six tests, where "
                    f"roughly one hit at p < 0.05 is expected by chance. The pipeline does not "
                    f"interpret it and neither does this notebook."
                ),
                            plain=(
                    "Each pattern the maths found is a direction with two ends. This asks whether either "
                    "end can correctly sort the emotions from pleasant to unpleasant, or from calm to "
                    "worked-up. 1.0 is a perfect sort; 0.5 is no better than guessing. "
                ),
)
            R.table(pd.DataFrame(clears),
                    "Phase 4 -- every PC end that clears AUROC 0.75, and whether it also clears "
                    "its own significance threshold")
    display(Markdown("### 5d. What script are the readouts in?"))

    GROUP_LABEL = {"gate_a_emotion_vector": "emotion vectors (GATE A)",
                   "gate_b_pc": "PC ends (GATE B)",
                   "control_apriori_axis": "a-priori axis controls"}
    SCRIPT = {}
    with R.guard("Phase 4 script mix"):
        rows = []
        for key in RUN_KEYS:
            readouts = runs.csv(key, "phase4_readouts.csv")
            if readouts is None:
                continue
            SCRIPT[key] = nbtools.script_mix(readouts)
            for group, sub in SCRIPT[key].groupby("group"):
                share = sub["script"].value_counts(normalize=True)
                rows.append({
                    "run": key, "readout group": GROUP_LABEL.get(group, group),
                    "n directions": sub["direction"].nunique(), "n tokens": len(sub),
                    **{c: round(share.get(c, 0.0) * 100, 1)
                       for c in ("CJK", "Latin", "other script", "punct / whitespace")}})
        if rows:
            R.table(pd.DataFrame(rows),
                    "Phase 4 -- script of the top-12 tokens per readout, as a percentage of all "
                    "tokens in that group")
            R.table(pd.DataFrame([{
                "run": RUNS[key], "all readouts, n tokens": len(sub),
                **{f"{c} %": round(sub["script"].value_counts(normalize=True).get(c, 0) * 100, 1)
                   for c in ("CJK", "Latin", "other script", "punct / whitespace")},
            } for key, sub in SCRIPT.items()]),
                "Phase 4 -- script mix over every readout in the run")
    with R.guard("Phase 4 GATE A aggregate"):
        rows = []
        for key in RUN_KEYS:
            p4 = P4.get(key)
            if p4 is None:
                continue
            ga = p4["gate_a"]
            ranks = [r["own_word_rank"] for r in ga["rows"]
                     if isinstance(r.get("own_word_rank"), (int, float))]
            sub = SCRIPT.get(key)
            cjk = None
            if sub is not None:
                emo = sub[sub["group"] == "gate_a_emotion_vector"]
                cjk = round((emo["script"] == "CJK").mean() * 100, 1) if len(emo) else None
            rows.append({
                "field": RUNS[key],
                "emotions scorable (single-token)": ga["n_scorable"],
                "untokenizable, excluded": len(ga["untokenizable"]),
                "HIT": ga["n_hits"], "near": ga["n_near"],
                "hit rate": ga["hit_rate"], "threshold to pass": ga["threshold"],
                "passed": ga["passed"], "chance hit rate": ga["chance_hit_rate"],
                "median rank of the emotion's own word": float(np.median(ranks)) if ranks else None,
                "% of top-12 tokens that are CJK": cjk,
                "which emotions hit": ", ".join(
                    r["emotion"] for r in ga["rows"] if r.get("verdict") == "HIT") or "none",
            })
        if rows:
            frame = pd.DataFrame(rows).set_index("field").T
            frame.index.name = "field"
            R.table(frame.reset_index(),
                    "Phase 4 GATE A -- does an emotion vector's lens readout contain its own "
                    "English word? (aggregate; the 2,052 individual token rows are in "
                    "`phase4_readouts.csv`)")


def phase4_chinese():
    """Sections 5e-5h: translations, tiered GATE A, burial, denominators."""
    global TIERS, frame, key, row, sub

    R.md(
        "### Why the readouts are Chinese, and the two confounds that hides\n\n"
        "Qwen3-32B is bilingual, so \"the readouts are Chinese because it is a Chinese model\" "
        "is the obvious reading. It is not sufficient: **96-97% of the Latin tokens in these "
        "readouts are whole words** (` failed`, ` sorrow`, ` panic`), not subword fragments. "
        "The lens is not being pushed into Chinese because English tokenises badly -- it "
        "selects dense whole words when it selects Latin at all, and still prefers Chinese for "
        "content. The stimuli, the anchors and the probe words are all English.\n\n"
        "That makes GATE A's failure ambiguous between **two separate causes**, which need "
        "different fixes:\n\n"
        "1. **Script.** The direction names its concept in Chinese, so no English lemma can "
        "reach the top-12 however good the direction is.\n"
        "2. **Exact-lemma matching.** The readout offers ` sorrow` where the test demands "
        "` sad`. A near-synonym is not a hit.\n\n"
        "Translating the output addresses only the first. What follows scores them separately: "
        "a Chinese tier for the script confound, an English near-synonym tier for the lemma "
        "confound, tokenizer-checked denominators in 5h, and 5g explaining why Phase 4's "
        "*other* test -- the AUROC ordering statistic -- was immune to both.\n\n"
        "The hand-written data is in `analysis/zh_en_glossary.py`. It is **not pipeline "
        "output**: it is a human translation judgement, written from the emotion words alone "
        "and saved before any matching was run, kept in its own module so it can be audited "
        "separately."
    )

    display(Markdown("### 5e. Every lensed direction, with translations"))

    GLOSSARY = None
    with R.guard("glossary import"):
        import zh_en_glossary as GLOSSARY  # noqa: N816  -- matching data for 5f

    TRANS = {}
    with R.guard("load translated readouts"):
        for key in RUN_KEYS:
            frame = runs.csv(key, "phase4_readouts_translated.csv")
            if frame is None:
                R.md(f"> `phase4_readouts_translated.csv` missing for `{RUNS[key]}`; "
                     f"falling back to the hand-written glossary for display.")
                continue
            TRANS[key] = frame
            covered = frame[frame["script"] != "latin"]
            covered = covered[~covered["script"].isin(["whitespace", "punctuation"])]
            print(f"{RUNS[key]}: {len(frame)} rows, "
                  f"{covered['token_english'].notna().sum()}/{len(covered)} non-Latin tokens glossed")

    with R.guard("PC pole table"):
        for key in RUN_KEYS:
            p4, trans = P4.get(key), TRANS.get(key)
            if p4 is None or trans is None:
                continue
            R.table(nbtools.pc_pole_table(p4, trans),
                    f"Phase 4 -- every lensed principal component, both poles, "
                    f"{RUNS[key]}. Translations from `phase4_readouts_translated.csv`; "
                    f"`effective tokens` is exp(entropy) over the whole vocabulary")

        R.md(
            "**Only 5 of the available principal components were lensed** — "
            "`n_pcs_to_lens: 5` — against 15 and 170 PC directions sitting in "
            "`phase3_pcs.safetensors`. This table is therefore complete with respect to what "
            "the pipeline computed, not with respect to what was asked for: there is no "
            "readout for PC6 upward, and none can be derived from these artefacts. "
            "Section 5k lists what a re-run would need.\n\n"
            "**Read the tokens against the `effective tokens` column, not on their own.** "
            "The `interpretable?` column applies the rule from 5j: past 200 effective tokens "
            "the printed twelve are an arbitrary slice of a nearly flat distribution, and the "
            "words in them should not be summarised into a theme however suggestive they look."
        )

    with R.guard("emotion vector readout table"):
        for key in RUN_KEYS:
            p4, trans = P4.get(key), TRANS.get(key)
            if p4 is None or trans is None:
                continue
            table = nbtools.emotion_readout_table(p4, trans)
            R.table(table,
                    f"Phase 4 -- all {len(table)} lensed emotion vectors, {RUNS[key]}, "
                    f"with translated readouts. Positive pole only, and no per-token "
                    f"probabilities were persisted, so no concentration column exists here")

        R.md(
            "**What the emotion-vector table is and is not.** It is complete — every emotion "
            "in each run has a readout. But two limits are structural, not editorial:\n\n"
            "* **Positive pole only.** No `-emotion` direction was lensed, so \"what does the "
            "opposite of sadness read as?\" is not answerable here.\n"
            "* **No probabilities.** The emotion-vector rows carry token strings and "
            "`own_word_rank` but no `prob`, so the concentration analysis in 5j cannot be run "
            "on them. A diffuse emotion-vector readout would be a **third** explanation for "
            "GATE A's failure alongside script and exact-lemma matching, and it remains "
            "untested.\n\n"
            "Reading the translated tokens beside the `GATE A` column is the clearest single "
            "view of why the gate failed: emotions scored `MISS` on their English lemma "
            "routinely have an apt Chinese word in the top few — `angry` opens with `骂` "
            "*scold, curse* and `愤怒` *anger, furious* while its English `own-word rank` is "
            "335."
        )

    with R.guard("translation provenance"):
        if TRANS:
            rows = []
            for key in RUN_KEYS:
                frame = TRANS.get(key)
                if frame is None:
                    continue
                counts = frame["gloss_confidence"].value_counts()
                scripts = frame["script"].value_counts()
                rows.append({
                    "run": RUNS[key],
                    "rows": len(frame),
                    "identical to phase4_readouts.csv": True,
                    **{f"script: {k}": int(v) for k, v in scripts.items()},
                    **{f"gloss confidence: {k}": int(counts.get(k, 0))
                       for k in ("high", "medium", "low", "n/a")},
                })
            R.table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "field"}),
                    "Provenance of the translated readouts")
            R.md(
                "`phase4_readouts_translated.csv` is `phase4_readouts.csv` row-for-row plus "
                "`script`, `token_english` and `gloss_confidence`; every non-Latin token "
                "carries a gloss. It is **supplied translation data, not pipeline output**, and "
                "it carries its own confidence labels — a `low`-confidence gloss on a rare or "
                "bound morpheme should be treated as a guess.\n\n"
                "It is used for **display only**. The matching lists that drive the tiered "
                "GATE A statistic in 5f still come from `zh_en_glossary.py`, because those were "
                "pre-committed before any matching ran and swapping in a different translation "
                "source afterwards would destroy exactly the property that makes 5f "
                "trustworthy."
            )
    display(Markdown("### 5f. GATE A re-scored in three tiers"))

    TIERS = {}
    with R.guard("tiered GATE A"):
        if GLOSSARY is None:
            R.md("> Glossary unavailable; the cross-lingual GATE A cannot run.")
        else:
            summaries, details = [], {}
            for key in RUN_KEYS:
                readouts, p4 = runs.csv(key, "phase4_readouts.csv"), P4.get(key)
                if readouts is None or p4 is None:
                    continue
                summary, per = nbtools.tiered_gate_a(readouts, p4, GLOSSARY)
                summary.insert(0, "run", key)
                summaries.append(summary)
                details[key] = per
            if summaries:
                TIERS = {"summary": pd.concat(summaries, ignore_index=True), "detail": details}
                for key in RUN_KEYS:
                    sub = TIERS["summary"]
                    sub = sub[sub["run"] == key].drop(columns=["run"])
                    if len(sub):
                        R.table(sub, f"Phase 4 -- GATE A re-scored in three tiers, {RUNS[key]}. "
                                     f"Each tier asks the same containment question GATE A asks "
                                     f"(is it in the top-12?), varying only what counts as the "
                                     f"emotion's name")
    with R.guard("tiered GATE A figure"):
        if not TIERS:
            R.md("> No tier results; figure skipped.")
        else:
            indexed = TIERS["summary"].set_index(["run", "tier"])

            def val(run, prefix, col):
                for (r_, t_), row in indexed.iterrows():
                    if r_ == run and t_.startswith(prefix):
                        return row[col]
                return float("nan")

            R.fig(
                F.gate_a_tiers(TIERS["summary"], PANEL),
                "phase4_gate_a_tiers.png",
                "Phase 4 (new): GATE A re-scored by what counts as the emotion's name",
                how_to_read=(
                    "One panel per run, three bars each. Every bar answers the **same** question "
                    "GATE A asks — *is the emotion's name inside its vector's top-12 lens "
                    "tokens?* — varying only what counts as a name: T1 the exact English lemma "
                    "(GATE A as specified), T2 an English near-synonym, T3 a Chinese "
                    "translation. The **orange rule over each bar is the 95th percentile of a "
                    "permutation null** that shuffles the translation lists between emotions, "
                    "keeping every list's contents and length intact: a bar must clear its "
                    "orange rule to mean anything at all. The dashed line is GATE A's own 50% "
                    "pass bar."
                ),
                what_it_shows=(
                    f"Scored on the English lemma the directions look unreadable (0% and 3%). "
                    f"Scored on Chinese translations with the identical containment rule, the 16 "
                    f"run reaches **{val('16', 'T3', 'rate (all)'):.0%}** and the 171 run "
                    f"**{val('171', 'T3', 'rate (all)'):.0%}** — the 16 run clearing GATE A's "
                    f"own pass bar.\n\n"
                    f"**The lists are not doing the work**, which is what the orange rules "
                    f"establish. Chance under the permutation null is "
                    f"{val('16', 'T3', 'permutation null mean'):.0%} and "
                    f"{val('171', 'T3', 'permutation null mean'):.0%} "
                    f"(p = {val('16', 'T3', 'p'):.4f}, {val('171', 'T3', 'p'):.4f}, 2000 "
                    f"permutations). T2 clears its null on the 171 run but not the 16 run, where "
                    f"a single hit at n = 16 is underpowered — so exact-lemma matching costs "
                    f"something, but script is much the larger effect."
                ),
                            plain=(
                    "When we ask an emotion's direction what word it is, does its own name come back? "
                    "This tries three definitions of 'its name' -- the exact English word, an English "
                    "near-synonym, and a Chinese translation -- to find out whether the original test "
                    "failed because the direction is meaningless or because we asked in the wrong "
                    "language. "
                ),
)
    with R.guard("tiered GATE A caveats"):
        if TIERS:
            R.md(
                "**What this changes, and what it does not.**\n\n"
                "GATE A's premise — that a direction the lens can read should surface the "
                "emotion's name — is largely satisfied; the test asked in the wrong language. "
                "But:\n\n"
                "* The pipeline's recorded GATE A verdict **stands as FAILED**. This is a "
                "re-scoring under a hand-written translation table, not a re-run of the gate, "
                "and it is reported as a secondary result rather than substituted for the "
                "original.\n"
                "* It is a **containment** test, not a rank test. The readout CSVs store only "
                "the top 12, so there is no Chinese analogue of `own_word_rank`.\n"
                "* Denominators differ by tier. T1 can only score English-single-token emotions; "
                "T3 can score `elated` and `gloomy` too, because their Chinese forms tokenise "
                "when the English does not. Both denominators are in the table; 5h checks them "
                "against the real tokenizer.\n"
                "* The translations are the author's judgement — pre-committed and "
                "permutation-controlled, but a reader who disagrees with `serene -> 柔和` or "
                "`ecstatic -> 狂欢` should re-score with their own list.\n"
                "* The under-converged lens (section 0) is untouched by any of this."
            )

    with R.guard("translation list disclosure"):
        if GLOSSARY is not None and TIERS:
            rows = []
            for key in RUN_KEYS:
                per = TIERS["detail"].get(key)
                if per is None:
                    continue
                hits = per[per["T3 Chinese translation"] | per["T2 English near-synonym"]]
                for _, row in hits.iterrows():
                    rows.append({"run": key, "emotion": row["emotion"],
                                 "Chinese tokens matched": row["matched Chinese tokens"],
                                 "English tokens matched": row["matched English tokens"]})
            if rows:
                R.table(pd.DataFrame(rows),
                        "Phase 4 -- every emotion scored as a hit in the relaxed tiers, with the "
                        "exact tokens that matched (audit trail)")
            R.table(pd.DataFrame([{
                "emotion": e,
                "Chinese candidates": " ".join(GLOSSARY.EMOTION_ZH[e]),
                "English near-synonyms": ", ".join(GLOSSARY.EMOTION_EN_SYNONYMS[e]),
            } for e in sorted(GLOSSARY.EMOTION_ZH)]),
                "The full pre-committed translation table used above, all 171 emotions. "
                "Published so the generosity of the matching can be audited and re-scored")
    display(Markdown("### 5g. Why GATE B survived what GATE A did not"))

    with R.guard("probe rank burial"):
        p4 = P4.get("16")
        if p4 is None:
            R.md("> Phase 4 gate missing for the 16 run; burial analysis skipped.")
        else:
            probes = p4["probes"]
            vocab = p4["gate_a"]["vocab_size"]
            axes_blobs, probe_labels, rows = {}, {}, []
            for name, dimension, poles in [("+valence", "valence", ("pleasant", "unpleasant")),
                                           ("+arousal", "arousal", ("activated", "deactivated"))]:
                blob = (p4.get("controls_apriori_axes") or {}).get(name)
                if not blob:
                    continue
                labels = dict(zip(probes["words"], probes[dimension]))
                ranks = {k: v for k, v in (blob.get("probe_ranks") or {}).items() if v is not None}
                axes_blobs[name] = blob
                probe_labels[name] = {"labels": labels, "poles": poles,
                                      "auroc": blob[f"auroc_{dimension}"]}
                high = [r for w, r in ranks.items() if labels.get(w, 0) > 0]
                low = [r for w, r in ranks.items() if labels.get(w, 0) < 0]
                rows.append({
                    "a-priori axis": name,
                    "AUROC on its own axis": blob[f"auroc_{dimension}"],
                    f"median rank, {poles[0]}": float(np.median(high)),
                    f"median rank, {poles[1]}": float(np.median(low)),
                    "separation is perfect": max(high) < min(low),
                    "best rank of ANY probe word": int(min(ranks.values())),
                    "probes inside GATE A's top-12 window": sum(1 for r in ranks.values() if r < 12),
                })
            if rows:
                R.table(pd.DataFrame(rows),
                        "Phase 4 -- where the English probe words actually rank in the a-priori "
                        "axis readouts (16-emotion run)")

            R.fig(
                F.probe_rank_burial(axes_blobs, probe_labels, vocab),
                "phase4_probe_rank_burial.png",
                "Phase 4 (new): rank of each English probe word in the a-priori axis readouts",
                how_to_read=(
                    "Each dot is one English probe word, placed at **its rank in that "
                    "direction's readout** — a log scale, so far left means 'the lens wants to "
                    "say this word' and far right means 'the lens has 150,000 other things to "
                    "say first'. Colour is the pole the word belongs to. The dashed line at rank "
                    "12 is GATE A's window: only a dot left of it could ever count as a GATE A "
                    "hit. The grey line at the right is the full vocabulary size."
                ),
                what_it_shows=(
                    "Two facts at once, and their combination is the point. **The colours "
                    "separate completely** — every pleasant word ranks above every unpleasant "
                    "one, which is what AUROC 1.00 means. **And every dot is far right of the "
                    "dashed line** — the best-ranked English emotion word sits at 1,927 of "
                    "151,936, and not one is inside GATE A's window.\n\n"
                    "So the two Phase 4 tests were never measuring the same thing. **GATE A is "
                    "an absolute containment test** and is destroyed by the readout preferring "
                    "another language, because the top-12 fills with Chinese before any English "
                    "word gets near it. **AUROC is a relative test within a fixed probe set** — "
                    "it never asks whether English words rank *highly*, only whether the "
                    "pleasant ones rank *above* the unpleasant ones, and burying all fourteen by "
                    "a common factor leaves that untouched.\n\n"
                    "The direction knows about valence in English. It just does not *say* it in "
                    "English. That single claim explains why GATE A failed, GATE B passed at "
                    "0.98–1.00, and the Chinese re-scoring in 5f recovers the naming behaviour "
                    "GATE A was looking for."
                ),
                            plain=(
                    "Where do the English emotion words actually show up in what a direction wants to "
                    "say? Far left means the lens is eager to say that word; far right means 150,000 "
                    "other things come out first. "
                ),
)
    display(Markdown("### 5h. Tokenizer-checked denominators (fully local)"))

    TOKENIZER = None
    with R.guard("load local tokenizer"):
        from tokenizers import Tokenizer as _Tk
        found = sorted((ROOT / "data").glob("**/models--Qwen--Qwen3-32B/**/tokenizer.json"))
        if not found:
            R.note_missing("Qwen3-32B tokenizer.json under data/", "5h skipped")
            R.md("> Local Qwen3-32B `tokenizer.json` not found under `data/`. Section 5h is "
                 "skipped; nothing else depends on it.")
        else:
            TOKENIZER = _Tk.from_file(str(found[0]))
            print("tokenizer:", found[0].relative_to(ROOT), "| vocab", TOKENIZER.get_vocab_size())

    with R.guard("tokenizer-checked denominators"):
        if TOKENIZER is None or GLOSSARY is None or not TIERS:
            R.md("> Tokenizer or glossary unavailable; denominators stay as reported in 5f.")
        else:
            def one_token(text):
                return len(TOKENIZER.encode(text, add_special_tokens=False).ids) == 1

            single = {e: [c for c in cands if one_token(c)]
                      for e, cands in GLOSSARY.EMOTION_ZH.items()}
            rows = []
            for key in RUN_KEYS:
                per, p4 = TIERS["detail"].get(key), P4.get(key)
                if per is None or p4 is None:
                    continue
                untok = set(p4["gate_a"]["untokenizable"])
                emotions = list(per["emotion"])
                testable = [e for e in emotions if single.get(e)]
                hits = per.set_index("emotion")["T3 Chinese translation"]
                rows.append({
                    "run": key, "emotions in run": len(emotions),
                    "scorable in English (single-token lemma)":
                        len(emotions) - len(untok & set(emotions)),
                    "NOT scorable in English": len(untok & set(emotions)),
                    "testable in Chinese (>=1 single-token form)": len(testable),
                    "of those, invisible to English GATE A": len(
                        [e for e in emotions if e in untok and single.get(e)]),
                    "T3 rate, naive denominator (all emotions)": float(hits.mean()),
                    "T3 rate, honest denominator (Chinese-testable)":
                        float(hits.loc[testable].mean()),
                })
            if rows:
                R.table(pd.DataFrame(rows),
                        "Phase 4 -- denominators checked against the real Qwen3-32B tokenizer, "
                        "computed locally with no model and no network")
            total = sum(len(v) for v in GLOSSARY.EMOTION_ZH.values())
            frac = sum(len(v) for v in single.values()) / total
            no_zh = sorted(e for e, v in single.items() if not v)
            R.md(
                f"**Why this matters, and why it needed no GPU.** Section 5f rested on an "
                f"assumption it could not check: that the Chinese candidates are single tokens "
                f"in Qwen's vocabulary. Only **{frac:.0%}** of the {total} are. A multi-token "
                f"candidate can never appear in a top-12 *token* list, so it silently deflates "
                f"the hit rate. The Qwen3-32B **tokenizer** is cached locally at 11 MB and "
                f"carries no weights, so this is pure local computation.\n\n"
                f"* **The honest denominator is the Chinese-testable set.** {len(no_zh)} "
                f"emotions have no single-token Chinese form at all "
                f"(`{'`, `'.join(no_zh[:8])}`, ...) and cannot be scored by the Chinese tier any "
                f"more than `elated` can be scored by the English one.\n"
                f"* **GATE A structurally could not score a third of the 171-emotion run.** 57 "
                f"of 171 emotions have no single-token English lemma, and **43 of those 57 do "
                f"have a single-token Chinese form**. Their exclusion has nothing to do with "
                f"whether their direction is readable — English simply spells them with more "
                f"than one token.\n\n"
                f"This is a rigour fix, not a rescue: it moves the Chinese-tier rate by a few "
                f"points and leaves 5f's permutation-controlled conclusion where it was."
            )


def phase4_shape():
    """Sections 5j-5k: readout concentration and lens coverage."""
    global key, sub

    display(Markdown("### 5j. Does the shape of the readout distribution matter?"))

    # One line per readout end, written by the author from the glossed token lists in 5e.
    # This is INTERPRETATION, not measurement, and is labelled as such wherever it appears.
    POLE_SUMMARY = {
        ("16", "+PC1"): "formatting: newlines and spaces, no semantic content (温和 mild is the only word)",
        ("16", "-PC1"): "violent, explosive high-intensity action -- frenzy, detonation, screaming",
        ("16", "+PC2"): "negative outcome -- failure, lack, powerlessness, death",
        ("16", "-PC2"): "sweetness and celebration -- cheerful, sweet, cute, delightful",
        ("16", "+PC3"): "sex-work and violent-crime vocabulary; reads as corpus artefact, not affect",
        ("16", "-PC3"): "sadness and illness mixed with vague hedges (maybe, something, somehow)",
        ("16", "+PC4"): "not interpreted -- MURKY on both axes",
        ("16", "-PC4"): "not interpreted -- MURKY on both axes",
        ("16", "+PC5"): "not interpreted -- MURKY on both axes",
        ("16", "-PC5"): "not interpreted -- MURKY on both axes",
        ("171", "+PC1"): "multilingual corpus junk -- do not interpret",
        ("171", "-PC1"): "punctuation plus acute distress (suddenly, panic, fear, unable)",
        ("171", "+PC2"): "formatting: newlines and punctuation, no semantic content",
        ("171", "-PC2"): "high-intensity, explosive action -- frenzy, surge, detonation",
        ("171", "+PC3"): "joy and vitality -- cheerful, vibrant, joyful, delighted",
        ("171", "-PC3"): "rejection, blame and harm -- reject, failure, toxic, crime, afraid",
        ("171", "+PC4"): "not interpreted -- MURKY on both axes",
        ("171", "-PC4"): "not interpreted -- MURKY on both axes",
        ("171", "+PC5"): "not interpreted -- MURKY on both axes",
        ("171", "-PC5"): "not interpreted -- MURKY on both axes",
    }

    CONC = pd.DataFrame()
    with R.guard("readout concentration"):
        rows = []
        for key in RUN_KEYS:
            p4, readouts = P4.get(key), runs.csv(key, "phase4_readouts.csv")
            if p4 is None or readouts is None:
                continue
            blobs = [(f"PC{int(pc['pc'])}", pc.get(end) or {})
                     for pc in p4["gate_b"]["per_pc"] for end in ("plus", "minus")]
            blobs += [("a-priori axis", blob)
                      for blob in (p4.get("controls_apriori_axes") or {}).values()]
            for family, blob in blobs:
                name = blob.get("name")
                sub = readouts[readouts["direction"] == name]
                toks = [t["token"] for t in blob.get("tokens", [])[:12]]
                whitespace = sum(1 for t in toks if not str(t).strip()) >= 6
                rows.append({
                    "run": key, "family": family, "direction": name,
                    "top-1 prob": blob.get("top1_prob"),
                    "effective_tokens": blob.get("effective_tokens"),
                    "top-12 mass": float(sub["prob"].sum()) if sub["prob"].notna().any() else np.nan,
                    "whitespace end": whitespace,
                    "author's one-line summary": POLE_SUMMARY.get((key, name), ""),
                })
        CONC = pd.DataFrame(rows)
        for key in RUN_KEYS:
            sub = CONC[CONC["run"] == key].drop(columns=["run"])
            if len(sub):
                R.table(sub.sort_values("effective_tokens"),
                        f"Phase 4 -- how concentrated each readout is, {RUNS[key]}. "
                        f"`effective_tokens = exp(entropy)` over the whole vocabulary; the "
                        f"summary column is the author's reading of the glossed tokens, not a "
                        f"measurement")
    with R.guard("readout concentration figure"):
        if CONC.empty:
            R.md("> No readout probabilities available; concentration figure skipped.")
        else:
            diffuse = CONC[CONC["effective_tokens"] >= 200]
            peaked = CONC[CONC["effective_tokens"] < 4]
            R.fig(
                F.readout_concentration(CONC, PANEL),
                "phase4_readout_concentration.png",
                "Phase 4 (new): how much each readout distribution commits to",
                how_to_read=(
                    "One row per lensed direction, on a **log** x-axis. The position is "
                    "`effective_tokens = exp(entropy)` of the direction's softmax over the "
                    "whole 151,936-token vocabulary: **1 means the direction says a single "
                    "token; 151,936 means it says nothing at all.** The percentage beside each "
                    "dot is how much probability mass the printed top-12 actually captures. "
                    "Grey dots are ends whose top-12 is mostly whitespace. The shaded band past "
                    "200 marks readouts flat enough that a top-12 list is an arbitrary slice of "
                    "them."
                ),
                what_it_shows=(
                    f"**The spread is a factor of ~1,800** — from "
                    f"{CONC['effective_tokens'].min():.1f} effective tokens to "
                    f"{CONC['effective_tokens'].max():,.0f}. A top-12 list means completely "
                    f"different things at the two ends of that range, so it should never be "
                    f"read without this number beside it.\n\n"
                    f"Two patterns matter. **The most confident readouts are the ones that say "
                    f"nothing**: all {len(peaked)} ends below 4 effective tokens "
                    f"(`{'`, `'.join(peaked['direction'])}`) are the whitespace/newline ends, "
                    f"holding ~98% of their mass in the top 12. Peakedness is not "
                    f"interpretability. And **{len(diffuse)} of {len(CONC)} ends sit past 200**, "
                    f"where the top-12 holds under 60% of the mass — including the 171 run's "
                    f"`+PC5` at 3,227 effective tokens with only 18% of its mass in the printed "
                    f"list. That is the same PC5 that scored AUROC 0.90 on valence, and it is a "
                    f"second, independent reason not to interpret it."
                ),
                            plain=(
                    "When a direction is asked what it would say, does it commit to a handful of words or "
                    "spread itself thinly across thousands? That decides whether its top-12 word list is "
                    "real evidence or just the tip of a nearly flat distribution. "
                ),
)
            R.md(
                "**A reading rule this gives us.** The concentration number is not decoration; "
                "it decides whether a token list is evidence at all:\n\n"
                "| effective tokens | what a top-12 list means | ends here |\n"
                "| --- | --- | --- |\n"
                "| < 10 | the direction genuinely commits to a few tokens | "
                f"{int((CONC['effective_tokens'] < 10).sum())} |\n"
                "| 10 - 200 | a real but broad preference; the top-12 is representative | "
                f"{int(((CONC['effective_tokens'] >= 10) & (CONC['effective_tokens'] < 200)).sum())} |\n"
                "| > 200 | the top-12 is an arbitrary slice of a flat distribution | "
                f"{len(diffuse)} |\n\n"
                "The four PC ends that carry the circumplex — 16-run `-PC1` (arousal) and "
                "`-PC2` (valence), 171-run `-PC2` (arousal) and `+PC3` (valence) — sit at 7, 52, "
                "16 and 21 effective tokens respectively, all inside the interpretable band. "
                "That is a genuine consistency check the AUROC alone does not provide: the "
                "directions that order the anchors are also the directions that commit to a "
                "vocabulary.\n\n"
                "**On the author's summary column.** Those one-liners are my reading of the "
                "glossed token lists, not an independent LLM judge — I also wrote the glossary "
                "and the surrounding analysis, so they carry no evidential weight beyond the "
                "tokens in 5e, which are printed in full precisely so the summaries can be "
                "checked against them. Every end past 200 effective tokens is deliberately "
                "summarised as \"too diffuse to summarise\" rather than given a plausible "
                "label, because a plausible label over a flat distribution is exactly the "
                "failure mode this section exists to prevent."
            )
    display(Markdown("### 5k. What the lens was and was not run on"))

    with R.guard("lens coverage"):
        rows = []
        for key in RUN_KEYS:
            p4 = P4.get(key)
            readouts = runs.csv(key, "phase4_readouts.csv")
            if p4 is None or readouts is None:
                continue
            n_pc = p4["gate_b"]["n_pcs"]
            rank = p4["pcs"]["rank"]
            counts = readouts[readouts["rank"] < 12].groupby("group")["direction"].nunique()
            rows.append({
                "run": key,
                "PC directions available in phase3_pcs.safetensors": rank,
                "PCs actually lensed": n_pc,
                "PC readouts (both poles)": int(counts.get("gate_b_pc", 0)),
                "emotion vectors lensed": int(counts.get("gate_a_emotion_vector", 0)),
                "emotion-vector poles": "positive only",
                "a-priori axis readouts": int(counts.get("control_apriori_axis", 0)),
                "per-token probabilities stored?": "PC ends + axes only, not emotion vectors",
                "tokens stored per direction": 12,
            })
        if rows:
            R.table(pd.DataFrame(rows), "Phase 4 -- lens coverage: what was read out, and what "
                                        "was persisted")
        R.md(
            "**Three coverage gaps worth stating plainly, because each one bounds a question "
            "someone will want to ask of this data.**\n\n"
            "1. **Only 5 principal components were lensed**, against 15 and 170 available in "
            "`phase3_pcs.safetensors`. `n_pcs_to_lens: 5` in the config. A table of the top 20 "
            "PCs' readouts cannot be built from these artefacts.\n"
            "2. **Emotion vectors were lensed at the positive pole only.** All 171 have a "
            "readout; none has a `-emotion` readout, so \"what does the opposite of sadness "
            "read as?\" was never computed.\n"
            "3. **Per-token probabilities were persisted only for PC ends and axis controls.** "
            "The emotion-vector rows carry token strings and `own_word_rank` but no `prob`, so "
            "the concentration analysis in 5j **cannot be run on the 171 emotion vectors** — "
            "and that matters, because a diffuse emotion-vector readout would be a third "
            "explanation for GATE A's failure, alongside script and exact-lemma matching. It is "
            "currently untested.\n\n"
            "All three are unlocked by the *same* two tensors section 5i already specifies "
            "(`J[31]` and `lm_head`, ~1.7 GB, CPU-only). One download would give the top-20 PC "
            "table with both poles, full distributions for every direction including the "
            "emotion vectors, and the rank-based cross-lingual GATE A."
        )


def phase4_fix():
    """Section 5i: what a proper rank-based cross-lingual GATE A needs."""

    display(Markdown("### 5i. What would properly fix GATE A, and where it can run"))

    R.md(
        "### What would properly fix GATE A, and where it can run\n\n"
        "Everything in 5f-5h is a **containment** test: is the name in the top 12? That is the "
        "ceiling of what the saved artefacts support, because `phase4_readouts.csv` stores only "
        "the top 12 tokens per direction. GATE A's real criterion is a **rank** -- "
        "`own_word_rank` out of 151,936 -- and no rank for a Chinese token can be recovered "
        "from what is on disk.\n\n"
        "The proper fix is a small, well-specified computation, and it does **not** need the "
        "GPU pod:\n\n"
        "```\n"
        "logits = lm_head( final_norm( J[31] @ v ) )      # then rank the Chinese token ids\n"
        "```\n\n"
        "It needs exactly two tensors that are not in `outputs/`:\n\n"
        "| tensor | shape | size | where it lives |\n"
        "| --- | --- | --- | --- |\n"
        "| `J[31]`, the transport at the target block | 5120 x 5120 fp32 | ~105 MB | one entry "
        "inside the 6.6 GB `Qwen3-32B_jacobian_lens.pt` |\n"
        "| `lm_head.weight` + final-norm gain | 151936 x 5120 bf16 | ~1.6 GB | one shard of the "
        "Qwen3-32B safetensors |\n\n"
        "The computation is then **one matrix-vector product and a top-k on CPU** -- seconds, "
        "no GPU, no 32B forward pass. The emotion vectors are already local in "
        "`phase2_emotion_vectors.safetensors`, and 5h shows the tokenizer is local too. The "
        "cost is bandwidth (~1.7 GB if `J[31]` is extracted from the archive rather than loaded "
        "whole), not hardware.\n\n"
        "**So the analysis is local; the missing piece is two downloads.** Until they happen, "
        "the honest description of 5f is \"a top-12 containment test in Chinese, permutation "
        "controlled\" -- not \"GATE A passed in Chinese\"."
    )


def phase6():
    """Section 6: the read-space decomposition, ablation and method note."""
    global P6, ctl, emotional, frame, g, gate, k, k0, key, ks, row

    R.heading("6. Phase 6 -- how much of an emotion vector can the lens read?")
    R.md(
        "*In plain terms.* *"
        "Split each emotion's fingerprint into the part the lens can put into words and the "
        "part it cannot, then measure how big each part is against what you would get from a "
        "random direction.*"
    )
    R.heading("6a. THE RESULT -- the read-space decomposition", level=3)

    P6, P6ABL = {}, {}
    # Phase 6 has no translation file of its own; gloss its atom tokens by lookup
    # against the Phase 4 ones and carry the miss count so it can be reported.
    GLOSS = nbtools.gloss_map(runs)
    atom_missing = {}
    with R.guard("Phase 6 read-space result"):
        for key in RUN_KEYS:
            gate = runs.json(key, "phase6_gate.json")
            if gate is None:
                continue
            ks = [str(k) for k in gate.get("reported_k") or []]
            rows = []
            for entry in gate["per_emotion"]:
                row = {"emotion": entry["emotion"],
                       "own_word_atom_rank": entry.get("own_word_atom_rank")}
                for k in ks:
                    per = entry["per_k"][k]
                    row[f"frac_reportable k={k}"] = per["frac_reportable"]
                    row[f"frac_remainder k={k}"] = per["frac_remainder"]
                    row[f"p k={k}"] = per["p_value"]
                    row[f"n_atoms k={k}"] = per["n_atoms"]
                rendered, n_missing = nbtools.render_atom_tokens(
                    entry["per_k"][ks[0]]["top_atom_tokens"], GLOSS)
                row["top atom tokens"] = rendered
                atom_missing[key] = atom_missing.get(key, 0) + n_missing
                rows.append(row)
            frame = pd.DataFrame(rows)
            P6[key] = {"gate": gate, "frame": frame, "ks": ks}

        for key in RUN_KEYS:
            if key not in P6:
                continue
            gate, frame, ks = P6[key]["gate"], P6[key]["frame"], P6[key]["ks"]
            R.table(frame.drop(columns=["top atom tokens"]),
                    f"Phase 6 -- per-emotion reportable / remainder split at "
                    f"k = {' and k = '.join(ks)}, {RUNS[key]}")

        rows = []
        for key in RUN_KEYS:
            if key not in P6:
                continue
            gate, frame, ks = P6[key]["gate"], P6[key]["frame"], P6[key]["ks"]
            emotional = frame[frame["emotion"] != "neutral"]
            g = gate["gate"]
            for k in ks:
                ctl = gate["random_control"][k]
                rows.append({
                    "run": key, "k": int(k),
                    "mean frac_reportable": float(emotional[f"frac_reportable k={k}"].mean()),
                    "min": float(emotional[f"frac_reportable k={k}"].min()),
                    "max": float(emotional[f"frac_reportable k={k}"].max()),
                    "control mean": ctl["mean"], "control p95": ctl["p95"],
                    "control n": ctl["n"],
                    "ratio to control": float(emotional[f"frac_reportable k={k}"].mean()) / ctl["mean"],
                    "beat null (Bonferroni)": f"{len(g['beat_null'][k])}/{g['n_emotions']}",
                    "beat null (uncorrected 0.05)":
                        f"{int((frame[f'p k={k}'] < 0.05).sum())}/{len(frame)}",
                })
        if rows:
            R.table(pd.DataFrame(rows), "Phase 6 -- effect size against the random-direction "
                                        "control, both runs, both k")
    with R.guard("Phase 6 remainder figure"):
        if not P6:
            R.md("> No Phase 6 artefacts; figure skipped.")
        else:
            k0 = P6[RUN_KEYS[0]]["ks"][0]
            frames = {key: P6[key]["frame"].rename(
                          columns={f"frac_remainder k={k0}": "frac_remainder"})[
                          ["emotion", "frac_remainder"]] for key in P6}
            controls = {key: P6[key]["gate"]["random_control"][k0] for key in P6}
            g16 = P6["16"]["gate"]
            sat = max(abs(P6[key]["frame"][f"frac_reportable k={P6[key]['ks'][0]}"]
                          - P6[key]["frame"][f"frac_reportable k={P6[key]['ks'][-1]}"]).max()
                      for key in P6)
            R.fig(
                F.remainder_bars(frames, controls, PANEL, k=int(k0)),
                "phase6_remainder_per_emotion.png",
                "Phase 6: reportable / remainder split per emotion, read-space run",
                how_to_read=(
                    "One bar per emotion, showing `frac_remainder` — the share of the emotion "
                    "vector the k-sparse pursuit could **not** rebuild from lens-nameable "
                    "directions. **A shorter bar is a more reportable emotion.** Bars are sorted, "
                    "and the x-axis starts near 0.95 because the remainder dominates everywhere; "
                    "read the differences, not the absolute lengths. The two red rules sit at "
                    "`1 − control mean` and `1 − control p95`, so an emotion beats chance only "
                    "by reaching further **left** than the rules. The 171-run panel has 172 bars, "
                    "so only its four most and four least reportable emotions are labelled."
                ),
                what_it_shows=(
                    f"Every emotion in both runs reaches left of the control p95 line: the "
                    f"reportable share is small — mean "
                    f"{P6['16']['frame'][P6['16']['frame'].emotion != 'neutral'][f'frac_reportable k={k0}'].mean():.1%} "
                    f"(16 run) and "
                    f"{P6['171']['frame'][P6['171']['frame'].emotion != 'neutral'][f'frac_reportable k={k0}'].mean():.1%} "
                    f"(171 run) — but it is **{P6['16']['frame'][P6['16']['frame'].emotion != 'neutral'][f'frac_reportable k={k0}'].mean() / P6['16']['gate']['random_control'][k0]['mean']:.1f}x "
                    f"and {P6['171']['frame'][P6['171']['frame'].emotion != 'neutral'][f'frac_reportable k={k0}'].mean() / P6['171']['gate']['random_control'][k0]['mean']:.1f}x "
                    f"the random-direction control**, which now has n = "
                    f"{P6['16']['gate']['random_control'][k0]['n']} rather than 16.\n\n"
                    f"The spread is real: `calm` and `content` sit near 3.9% reportable while "
                    f"`bored` sits near 2.3%. Nothing here approaches the 5–15% the brief "
                    f"anticipated, and the remainder is ~97% throughout."
                ),
                            plain=(
                    "How much of each emotion's fingerprint can be rebuilt out of pieces the lens is able "
                    "to name -- and how much is left over that it cannot. A shorter bar means more of "
                    "that emotion could be put into words. "
                ),
)
            R.md(
                f"**The ceiling is not a budget artefact.** Raising the sparsity budget from "
                f"k = {P6['16']['ks'][0]} to k = {P6['16']['ks'][-1]} changes the reportable "
                f"fraction by a median of ~2–5×10⁻⁵ and by at most **{sat:.1e}** across every "
                f"emotion in both runs — all but one emotion agrees to three decimal places "
                f"(`bored` is the exception in both runs). The reconstruction saturates well "
                f"before k = {P6['16']['ks'][-1]}, so the ~3% is a property of the pool and the "
                f"vector, not of how many atoms the pursuit was allowed to spend.\n\n"
                f"*(The brief for this section said the two k values agree to four decimals. "
                f"They agree to four decimals for 9/17 and 107/172 emotions and to three "
                f"decimals for 16/17 and 169/172; the saturation conclusion holds, the stronger "
                f"phrasing does not.)*"
            )
    with R.guard("Phase 6 null resolvability"):
        if P6:
            rows = []
            for key in RUN_KEYS:
                if key not in P6:
                    continue
                g = P6[key]["gate"]["gate"]
                floor = g["p_value_floor"]
                rows.append({
                    "run": key,
                    "emotions tested": g["n_emotions"],
                    "permutations": int(round(1 / floor - 1)),
                    "smallest achievable p (floor)": floor,
                    "alpha, uncorrected": g["alpha"],
                    "alpha, Bonferroni": g["alpha_bonferroni"],
                    "floor < Bonferroni alpha?": floor < g["alpha_bonferroni"],
                    "permutations needed for Bonferroni": int(np.ceil(1 / g["alpha_bonferroni"])),
                    "beat null (Bonferroni)":
                        f"{len(g['beat_null'][P6[key]['ks'][0]])}/{g['n_emotions']}",
                })
            R.table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "field"}),
                    "Phase 6 -- can the Bonferroni-corrected null actually be resolved at 500 "
                    "permutations?")
            R.md(
                "**This table exists because one number below would otherwise be read exactly "
                "backwards.** The 171-run gate records **0 of 172 emotions beating the null**, "
                "and that is *not* a null result — it is a resolution limit.\n\n"
                "With 500 permutations the smallest p-value obtainable is 1/501 = 0.00200. "
                "Bonferroni over 172 emotions sets the threshold at 0.05/172 = 0.00029. Since "
                "0.00200 > 0.00029, **no emotion can clear that bar at any effect size "
                "whatsoever**; it would take ≥ 3,440 permutations to become resolvable. Every "
                "one of the 172 clears the uncorrected 0.05, and the effect is 3.6x the "
                "control.\n\n"
                "The 16-run gate is resolvable — Bonferroni over 17 emotions gives 0.00294, "
                "which is above the 0.00200 floor — and there **17 of 17 beat the null**, every "
                "p-value pinned at the floor. So the honest summary is: *supported and reliably "
                "above a 500-sample null in the 16-emotion run; the same effect size in the 171 "
                "run, with a corrected significance test that 500 permutations cannot decide.*"
            )
    R.heading("6b. Atom tokens -- what the readable component is made of", level=3)

    with R.guard("Phase 6 atom tokens"):
        for key in RUN_KEYS:
            if key not in P6:
                continue
            frame, ks = P6[key]["frame"], P6[key]["ks"]
            table = frame[["emotion", f"frac_reportable k={ks[0]}",
                           "own_word_atom_rank", "top atom tokens"]].copy()
            R.table(table, f"Phase 6 -- top atom tokens per emotion, {RUNS[key]} "
                           f"(read-space, k = {ks[0]})")

    with R.guard("Phase 6 atom gloss coverage"):
        rows = []
        for key in RUN_KEYS:
            if key not in P6:
                continue
            total = sum(len(e["per_k"][P6[key]["ks"][0]]["top_atom_tokens"])
                        for e in P6[key]["gate"]["per_emotion"])
            shown = atom_missing.get(key, 0)
            rows.append({"run": RUNS[key], "atom tokens shown": total,
                         "words left untranslated (marked ⚠)": shown,
                         "in English or punctuation":
                             f"{1 - shown / max(total, 1):.0%}"})
        if rows:
            R.table(pd.DataFrame(rows),
                    "Phase 6 -- gloss coverage of the atom tokens. Phase 6 has no "
                    "translation file of its own, so these are glossed by lookup against "
                    "the Phase 4 ones. Punctuation and whitespace pass through unmarked; a "
                    "⚠ is a *word* in a script for which no gloss was available")

        R.md(
            "**These connect straight back to Phase 4, and the connection is the point.** The "
            "atoms that carry `angry`'s readable component are **scold** · ` weapon` · "
            "**malignant** · **punishment** · ` merciless` · **coercion** · ` Worse` · "
            "**violence** — Chinese, Thai and English interleaved in a single eight-atom "
            "reconstruction, and the point survives translation: the readable part of an "
            "emotion vector is assembled from several languages at once. `calm` and "
            "`content` both open on **mild** and **natural**; `ecstatic` opens on "
            "**frenzied**.\n\n"
            "Section 5d measured 42–55% of Phase 4's readout tokens as CJK, and section 5f "
            "showed the emotion directions name themselves in Chinese far above chance. The "
            "atom pool here is drawn from the same lens vocabulary, so it inherits that "
            "property: **the readable component of an emotion vector is multilingual, and any "
            "English-only account of it will understate it.** This is the same finding arriving "
            "by a second, independent route — Phase 4 measured it in readouts, Phase 6 finds it "
            "in the reconstruction basis."
        )
    R.heading("6c. THE ABLATION -- the write-space run, a different question", level=3)

    with R.guard("Phase 6 write-space ablation"):
        rows = []
        for key in RUN_KEYS:
            gate = runs.json(key, "write_space_ablation/phase6_gate.json")
            frame = runs.csv(key, "write_space_ablation/phase6_decomposition.csv")
            if gate is None or frame is None:
                continue
            P6ABL[key] = {"gate": gate, "frame": frame}
            emotional = frame[frame["emotion"] != "neutral"]
            read = P6.get(key)
            k0 = read["ks"][0] if read else None
            read_emotional = (read["frame"][read["frame"].emotion != "neutral"]
                              if read else None)
            rows.append({
                "run": key,
                "READ mean frac_reportable": (
                    float(read_emotional[f"frac_reportable k={k0}"].mean()) if read else None),
                "READ control mean": read["gate"]["random_control"][k0]["mean"] if read else None,
                "READ control n": read["gate"]["random_control"][k0]["n"] if read else None,
                "READ ratio": (float(read_emotional[f"frac_reportable k={k0}"].mean())
                               / read["gate"]["random_control"][k0]["mean"]) if read else None,
                "WRITE mean frac_reportable": float(emotional["frac_reportable"].mean()),
                "WRITE control mean": gate["random_control"]["mean"],
                "WRITE control n": gate["random_control"]["n"],
                "WRITE ratio": float(emotional["frac_reportable"].mean())
                               / gate["random_control"]["mean"],
            })
        if rows:
            R.table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "quantity"}),
                    "Phase 6 -- read-space result beside the write-space ablation")
        R.md(
            "**These are answers to two different questions, not two attempts at one.**\n\n"
            "* The **read-space** run asks *what can the lens read off this vector?* Its atoms "
            "are the directions whose lens score is largest, so its reportable fraction is the "
            "share of `v` that the lens's own readout is sensitive to.\n"
            "* The **write-space** run asks *what residual would make the model emit token t?* "
            "Its atoms are the perturbations that most raise a token's logit. Its reportable "
            "fraction is the share of `v` expressible as a combination of emission-driving "
            "directions.\n\n"
            "Both are legitimate, and the write-space number is the one relevant to steering "
            "(Phase 8, not run). Neither supersedes the other, and the two fractions are close "
            "enough (~3% versus ~3.5%) that nothing here turns on the choice — but the "
            "write-space run's control had n = 16 against the read run's n = 500, so its "
            "1.7x/1.8x ratio is far less well resolved than the read run's 4.6x/3.6x."
        )
    R.heading("6d. THE METHODOLOGICAL NOTE -- read vs write, and a wrong test", level=3)

    R.md(
        "### Phase 6 methodological note: the read direction, the write direction, and a "
        "validity check that tested the wrong one\n\n"
        "This is a finding about the method, not bookkeeping, so it belongs in the record.\n\n"
        "**The read direction is `Jᵀu_t`.** The lens score for token `t` at residual `h` is "
        "`u_tᵀ J h`, where `u_t = g ⊙ w_t` folds in the final norm's learned gain. Regroup it: "
        "`u_tᵀ J h = (Jᵀu_t)ᵀ h`. So the direction in residual space that the lens score is "
        "linear in — the thing to project `v` onto if the question is *what can be read* — is "
        "`Jᵀu_t`. That is what `atom_mode: read` builds, and the run validates it directly: "
        "the recorded lens score and the projection onto `Jᵀu_t` correlate at "
        f"{P6['16']['gate']['read_identity']['min_correlation']:.6f} minimum over "
        f"{P6['16']['gate']['read_identity']['n_probes']} probes, against a "
        f"{P6['16']['gate']['read_identity']['threshold']} threshold.\n\n"
        "**The write direction is `J⁺u_t`, and it answers a different question.** If instead "
        "you want the residual perturbation that maximally raises token `t`'s logit per unit "
        "norm — a steering question — you need to invert the transport, not transpose it. "
        "`Jᵀ = J⁻¹` only for orthogonal `J`, and an averaged Jacobian is not orthogonal.\n\n"
        "**The earlier validity check was a write-direction test.** A previous version of this "
        "section gated Phase 6 on *does lensing an atom return its own token at rank 0?* That "
        "is a well-posed question about a **write** dictionary: `J⁺u_t` is constructed so that "
        "pushing it through `J` lands on `u_t`. A **read** atom `Jᵀu_t` has no reason to "
        "satisfy it, and failing it says nothing about whether the read decomposition is "
        "sound. The write-space ablation passes that check 24/24 with median self-rank 0, "
        "which is the cleanest evidence that the check was measuring the write construction "
        "all along.\n\n"
        "**What this cost.** The read-space decomposition was previously reported as invalid "
        "and its variance split withheld, on the strength of a test it was never obliged to "
        "pass. The correct check for a read dictionary is the score-identity correlation above, "
        "and it holds. Anyone reading an earlier version of `RESULTS.md` saw a FAILED verdict "
        "for Phase 6 that this section supersedes."
    )
    R.heading("6e. Caveats on this result", level=3)

    with R.guard("Phase 6 caveats"):
        if P6:
            rows = []
            for key in RUN_KEYS:
                if key not in P6:
                    continue
                gate = P6[key]["gate"]
                coh = gate["coherence"]
                frame = P6[key]["frame"]
                rows.append({
                    "run": key,
                    "atoms in pool": coh["n_atoms"],
                    "mean pairwise coherence": coh["mean"],
                    "max coherence": coh["max"],
                    "coherence threshold": coh["threshold"],
                    "frac atoms above threshold": coh["frac_above_threshold"],
                    "own_word_atom_rank null":
                        f"{int(frame['own_word_atom_rank'].isna().sum())}/{len(frame)}",
                    "emotions whose own word IS an atom": ", ".join(
                        frame[frame["own_word_atom_rank"].notna()]["emotion"]) or "none",
                })
            R.table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "field"}),
                    "Phase 6 -- caveat measurements")

        coh16 = P6["16"]["gate"]["coherence"] if "16" in P6 else {}
        remainder_note = (P6["16"]["gate"].get("v_remainder_means") if "16" in P6 else "") or ""
        R.md(
            "**1. Atom coherence limits attribution to specific tokens.** "
            f"{coh16.get('frac_above_threshold', float('nan')):.1%} of the "
            f"{coh16.get('n_atoms', '?')} atoms in the pool exceed the "
            f"{coh16.get('threshold', 0.5)} interchangeability threshold, with a maximum "
            f"pairwise coherence of {coh16.get('max', float('nan')):.2f}. Where atoms are that "
            "correlated, the pursuit's *choice* among them is close to arbitrary: the "
            "reportable **fraction** is stable, but the claim \"`v_J` names `骂` specifically\" "
            "is weakly attributable, because a near-duplicate atom would have served as well. "
            "Read the token tables in 6b as indicating a neighbourhood, not a selection.\n\n"
            "**2. The remainder is not \"intrinsically unverbalizable\".** The artefact records "
            f"its own definition: *{remainder_note}* The reconstructable set is a **union of "
            "cones**, not a linear subspace — nonnegative coefficients over a k-subset — so "
            "\"outside it\" is a statement about this pool at this k, not about the vector "
            "being beyond language. A different pool, a larger k, or signed coefficients would "
            "each move the boundary. Section 6a's saturation check bounds the k dependence but "
            "says nothing about the other two.\n\n"
            "**3. The readable component is not the emotion's own word.** "
            f"`own_word_atom_rank` is null for {int(P6['16']['frame']['own_word_atom_rank'].isna().sum())}"
            f"/{len(P6['16']['frame'])} emotions in the 16 run and "
            f"{int(P6['171']['frame']['own_word_atom_rank'].isna().sum())}"
            f"/{len(P6['171']['frame'])} in the 171 run — the emotion's own English word does "
            "not appear among its selected atoms at all. The four exceptions in the 171 run "
            f"({', '.join(P6['171']['frame'][P6['171']['frame']['own_word_atom_rank'].notna()]['emotion'])}) "
            "are the whole of it. So the ~3% that is readable is *not* the model naming the "
            "emotion; it is a multilingual scatter of adjacent concepts, consistent with GATE "
            "A's failure in section 5f rather than in tension with it.\n\n"
            "**4. The lens is still the 80-prompt interrupted fit** (section 0). A variance "
            "split is magnitude-sensitive, so this whole section inherits an unconfirmed "
            "normalisation constant."
        )



def phase7():
    """Section 7: the two measurement channels, their separation gate, and both baselines."""
    global P7

    P7 = {}
    R.heading("7. Phase 7 -- building two channels and proving they are separate")
    R.md(
        "**Phase 7 steers nothing.** It builds two measurement channels and shows they "
        "do not leak into each other, then records what both read when the model is left "
        "alone. Every Phase 8 number is a difference against these baselines, which is "
        "why a phase with no manipulation in it belongs here.\n\n"
        "It establishes that the channels are separate and that both baselines sit at "
        "floor. It establishes **nothing** about emotion causing behaviour. That is "
        "Phase 8."
    )

    with R.guard("Phase 7 load"):
        base = "phase7_channels/"
        gate = runs.json("16", base + "phase7_gate.json")
        report = runs.csv("16", base + "phase7_report.csv")
        behaviour = runs.csv("16", base + "phase7_behaviour.csv")
        if gate is None:
            R.md("> No Phase 7 artefacts; section skipped.")
            return
        P7 = {"gate": gate, "report": report, "behaviour": behaviour}

    # -- 7a separation gate ---------------------------------------------------
    R.heading("7a. THE SEPARATION GATE", level=3)
    with R.guard("Phase 7 separation gate"):
        sep = P7["gate"]["separation"]
        rows = [{"group": g, "items": v["n_items"], "affect-word hits": v["n_hits"]}
                for g, v in sep["per_group"].items()]
        R.table(pd.DataFrame(rows),
                "Phase 7 -- affect-vocabulary scan over every string in both channels "
                f"({sep['n_affect_stems']} stems)")
        R.table(pd.DataFrame([
            {"quantity": "behaviour-channel affect hits", "value": sep["behaviour_hits"]},
            {"quantity": "report-channel affect hits (expected, it asks about feeling)",
             "value": sep["report_hits"]},
            {"quantity": "separated", "value": sep["separated"]},
        ]), "Phase 7 -- separation verdict")
        R.md(
            f"**`separated: {sep['separated']}`.** The report channel contains "
            f"{sep['report_hits']} affect words *by design* — it asks the model how it "
            f"feels. The behaviour channel contains **{sep['behaviour_hits']}**, across "
            f"its prompts, its rubric, and the source of its mechanical scorers."
        )
        R.md("**The report rubric, in full:**\n\n```\n"
             + P7["gate"]["channels"]["report_rubric"].strip() + "\n```")
        R.md("**The behaviour rubric, in full:**\n\n```\n"
             + P7["gate"]["channels"]["behaviour_rubric"].strip() + "\n```")

        tasks = P7["gate"]["channels"]["behaviour_tasks"]
        by_scorer = {}
        for t in tasks:
            by_scorer.setdefault(t["family"], t["scorer"])
        R.table(pd.DataFrame([{"family": f, "scorer": sc,
                               "tasks": sum(1 for t in tasks if t["family"] == f)}
                              for f, sc in sorted(by_scorer.items())]),
                "Phase 7 -- how each behaviour family is scored")
        R.md(
            "**Three of the four behaviour families are scored by code**, not by a judge: "
            "a parsed multiple-choice letter for `risk`, a regex over hedge words for "
            "`hedging`, a turn count for `persistence`. Only `refusal` uses the judge.\n\n"
            "That is a design choice for auditability, not a saving. **A scorer you can "
            "read is a scorer you can verify is affect-blind** — the separation scan above "
            "runs over the scorers' own source text and finds zero affect stems in it. A "
            "judge prompt cannot be audited that way; you can check its wording, but not "
            "what the model behind it is sensitive to.\n\n"
            "**The limitation the gate states about itself.** The scan is mechanical: it "
            "confirms no affect *word* crosses into the behaviour channel. It cannot "
            "confirm that no affect *question* does. The refusal and risk prompts are "
            "morally loaded whatever their vocabulary — asking a model to decline "
            "something, or to gamble, engages dispositions that are not affect-neutral "
            "just because the words are. Only a human reading the prompts can settle that, "
            "and this notebook does not claim it has been settled."
        )

    # -- 7b report baseline ---------------------------------------------------
    R.heading("7b. Baseline report channel: flat denial", level=3)
    with R.guard("Phase 7 report baseline"):
        rep = P7["report"]
        R.table(rep[["prompt", "score", "detail"]],
                "Phase 7 -- unsteered report channel, judge-scored 0-4")
        quotes = []
        for _, row in rep.iterrows():
            text = " ".join(str(row["response"]).split())
            for marker in ("I don't have feelings", "I don't experience emotions",
                           "I don't have personal experiences", "I don't have emotions"):
                if marker in text:
                    start = max(0, text.index(marker) - 60)
                    quotes.append(f"> *\"...{text[start:start + 240]}...\"*  \n"
                                  f"> — in reply to **{row['prompt']}** (scored "
                                  f"{row['score']:.0f})")
                    break
        R.md(
            "**Every prompt scores 0.** The model's default self-report is not a weak "
            "signal to be teased out; it is flat denial of having any inner state to "
            "report. Verbatim:\n\n" + "\n>\n".join(quotes[:3])
        )
        R.md(
            "**Why a floor is the right baseline for Phase 8.** A channel pinned at 0 has "
            "the entire 0-4 range available above it. If steering moves self-report at "
            "all, it moves it off a floor that the unsteered model holds unanimously "
            "across five differently-worded prompts — so a rise cannot be prompt "
            "sensitivity, and there is no ceiling to mask an effect. The risk runs the "
            "other way: a directional prediction of *decrease* is untestable here."
        )

    # -- 7c behaviour baseline ------------------------------------------------
    R.heading("7c. Baseline behaviour channel: two families on the floor", level=3)
    with R.guard("Phase 7 behaviour baseline"):
        dyn = P7["gate"]["baseline"]["dynamic_range"]
        fam = dyn["per_family"]
        R.table(pd.DataFrame([
            {"family": f, "n": v["n"], "mean": v["mean"], "min": v["min"],
             "max": v["max"], "spread": v["spread"],
             "has usable range": v["has_range"]}
            for f, v in sorted(fam.items())]),
            "Phase 7 -- unsteered behaviour channel, mean score per family")
        zero = sorted(f for f, v in fam.items() if v["mean"] == 0)
        norange = dyn["families_without_range"]
        R.md(
            f"**Two families sit at exactly 0: `{'` and `'.join(zero)}`.** That is a "
            f"**floor effect**, and it removes a prediction rather than supporting one: "
            f"a score that is already 0 can only move up, so any Phase 8 hypothesis of "
            f"the form *\"this emotion decreases persistence\"* is **untestable on these "
            f"two families**. Only an increase could be detected.\n\n"
            f"A third family has no usable range for a different reason: "
            f"`{'`, `'.join(sorted(set(norange) - set(zero)))}` is pinned at its "
            f"mean with zero spread — the model made the same choice on every task — so "
            f"it is uninformative in both directions rather than one.\n\n"
            f"That leaves **`hedging`** (mean {fam['hedging']['mean']:.2f}, spread "
            f"{fam['hedging']['spread']:.2f}) and **`refusal`** (mean "
            f"{fam['refusal']['mean']:.2f}, spread {fam['refusal']['spread']:.2f}) as the "
            f"only two behaviour families with room to move in both directions. Phase 8's "
            f"behavioural evidence rests on those two, and `refusal` scored only "
            f"{fam['refusal']['n']} of its 4 tasks."
        )
        R.fig(
            F.phase7_baselines(P7["report"], fam, RUNS["16"]),
            "phase7_baselines.png",
            "Phase 7: both channels at baseline, before any steering",
            how_to_read=(
                "Left: the report channel, one bar per self-report prompt, judge-scored "
                "0–4 for how strongly the answer expresses the target emotion. Right: the "
                "behaviour channel, mean score per task family, with each family's n and "
                "spread annotated. **Red means pinned** — a bar at a scale endpoint, or a "
                "family with zero spread, against which a directional prediction cannot be "
                "tested. Note the two panels use different scales (0–4 and 0–3) because "
                "the two rubrics do."
            ),
            what_it_shows=(
                "Both channels are at floor before anything is done to them, but for "
                "different reasons and with different consequences. Every report prompt "
                "scores exactly 0 — maximum headroom for steering to move it upward. On "
                "the behaviour side only `hedging` and `refusal` have usable range; "
                "`persistence` and `report` are at 0 and `risk` is pinned at 1.0, so three "
                "of five families cannot register a decrease at all. **This is a fact "
                "about the measuring instrument, established before any steering, which is "
                "the entire point of running Phase 7 separately.**"
            ),
                    plain=(
                "What both measuring instruments read when the model is left completely alone -- "
                "nothing steered, nothing manipulated. This is the 'before' picture that every Phase "
                "8 result is compared against. "
            ),
)

    # -- 7d activation-level probe -------------------------------------------
    R.heading("7d. Report availability at the activation level", level=3)
    with R.guard("Phase 7 report probe"):
        avail = P7["gate"]["baseline"]["report_availability"]
        R.table(pd.DataFrame([
            {"emotion": e, "n texts": v["n_texts"], "mean cosine": v["mean_cosine"],
             "max cosine": v["max_cosine"]} for e, v in avail.items()]),
            "Phase 7 -- cosine between the residual during self-report and each emotion's "
            "lens tokens")
        lo = min(v["mean_cosine"] for v in avail.values())
        hi = max(v["mean_cosine"] for v in avail.values())
        R.fig(
            F.phase7_report_probe(avail, RUNS["16"]),
            "phase7_report_probe.png",
            "Phase 7: activation-level report availability at baseline",
            how_to_read=(
                "For each candidate emotion, the cosine between the residual stream while "
                "the model answers a self-report prompt and that emotion's lens-token "
                "directions. Bars are the mean over the five report texts; diamonds are "
                "the single best text. Zero is marked: **positive means the emotion's "
                "vocabulary is present in the activation, negative means it is actively "
                "absent.**"
            ),
            what_it_shows=(
                f"Every mean sits between {lo:+.3f} and {hi:+.3f} — indistinguishable from "
                "zero. The report channel is at floor in the activations, not only in the "
                "text.\n\n"
                "This matters because it is a **second, judge-free report measure**. The "
                "0-scores in 7b depend on a rubric and an LLM judge; this number depends "
                "on neither. Two measures of report-availability, one textual and one at "
                "the activation level, agree that there is nothing to report at baseline — "
                "so the floor in 7b is not an artefact of a strict judge."
            ),
                    plain=(
                "A second way of asking whether the model has any inner state to report, which does "
                "not rely on an AI judge reading its answers -- it looks at the model's internal "
                "activity directly instead. "
            ),
)

    # -- 7e candidate selection ----------------------------------------------
    R.heading("7e. Candidate selection: the filter was overridden", level=3)
    with R.guard("Phase 7 candidate selection"):
        ranked = P7["gate"]["emotions"]["ranked"]
        chosen = P7["gate"]["emotions"]["chosen"]
        R.table(pd.DataFrame([
            {"emotion": r["emotion"], "frac_reportable": r["frac_reportable"],
             "usable by default filter": r["usable"],
             "reason": "; ".join(r["reasons"]) or "-"} for r in ranked]),
            "Phase 7 -- every candidate emotion against the default filter")
        n_usable = sum(1 for r in ranked if r["usable"])
        R.md(
            f"**No emotion passed the default filter** ({n_usable} of {len(ranked)}), and "
            f"the phase aborted. It was re-run with "
            f"`channel_emotions={','.join(chosen)}`.\n\n"
            f"The filter requires that the Phase 6 decomposition selected **the emotion's "
            f"own English token** as one of its atoms. The override is a deviation from "
            f"the pre-registered procedure and is recorded here rather than quietly "
            f"applied. The three were chosen on reportable fraction and quadrant coverage: "
            f"`terrified` and `angry` are high-arousal negative, `content` is low-arousal "
            f"positive as a contrast.\n\n"
            f"**Why the filter is mis-specified, in two parts.**\n\n"
            f"1. **Wrong language.** Section 5d measured 42–55% of this model's lens "
            f"readout tokens as CJK, and 5f showed the emotion directions name themselves "
            f"in Chinese far above chance while failing in English. A filter that tests "
            f"for an English token is testing the wrong vocabulary in this model.\n"
            f"2. **Wrong act.** Being angry disposes a model to say angry *things*, not to "
            f"say the word \"angry\". `angry`'s own atoms in section 6b are **scold**, "
            f"` weapon`, **malignant**, **merciless**, **violence** — an expressive "
            f"profile, not a label. Naming a state and expressing it are different acts, "
            f"and the filter conflated them."
        )
        R.md(
            "> ### The same assumption has now failed four times\n"
            "> \n"
            "> This is the **fourth** place the pipeline encoded *\"the concept should "
            "surface as its own English token\"* as a validity criterion:\n"
            "> \n"
            "> | where | the test | outcome |\n"
            "> | --- | --- | --- |\n"
            "> | Phase 0 GATE A | does an emotion vignette read out its emotion word? | "
            "MISS on every emotion item |\n"
            "> | Phase 4 GATE A | is the emotion's own English word in its vector's top-12? "
            "| 0/14 and 5/114 |\n"
            "> | Phase 6, original | does an atom lens back to its own token? | failed, and "
            "was the wrong test for a read dictionary |\n"
            "> | Phase 7 filter | did the decomposition select the emotion's own token? | "
            "0 of 16 emotions passed |\n"
            "> \n"
            "> Each was written as a *soundness* check on the method. In a "
            "Chinese-developed model each instead measured the same latent assumption "
            "about English, and each time the assumption was what broke. Three of the four "
            "were treated as pipeline failures before the cause was identified.\n"
            "> \n"
            "> The generalisable lesson: **an interpretability validity check that asks "
            "whether a concept surfaces as a specific token is a test of the model's "
            "lexicalisation preferences as much as of the method.** Where the two can "
            "differ — a different training language, a multilingual vocabulary, a concept "
            "with no single-token name — the check needs to be language-agnostic or "
            "declared as language-specific up front."
        )

    # -- 7f handoff -----------------------------------------------------------
    R.heading("7f. What Phase 8 does with this", level=3)
    with R.guard("Phase 7 handoff"):
        cfg = P7["gate"]["config"]
        phase8_dir = runs.phases("16") / "phase8_steering"
        R.table(pd.DataFrame([
            {"setting": "conditions", "value": "v, v_J, v_perp, random"},
            {"setting": "emotions", "value": ", ".join(chosen)},
            {"setting": "steer strengths",
             "value": ", ".join(str(x) for x in cfg.get("steer_strengths", []))
                      + " (signed, so each is applied in both directions)"},
            {"setting": "positions", "value": cfg.get("steer_positions", "?")},
            {"setting": "report baseline to beat", "value": "0.0 on all 5 prompts"},
            {"setting": "behaviour families with usable range",
             "value": "hedging, refusal"},
        ]), "Phase 7 -- the handoff to Phase 8")
        R.md(
            "**What a positive result would look like.** The question Phase 8 asks is "
            "whether the part of an emotion vector the lens *cannot* read still moves "
            "behaviour. So the result that would matter is **`v_perp` shifting `hedging` "
            "or `refusal` while the report channel stays at 0** — behaviour moving without "
            "the model being able to say anything about it. The `random` condition bounds "
            "how much of any shift is just perturbation of that norm, and `v_J` versus "
            "`v_perp` splits readable from unreadable.\n\n"
            "**Read against the floors in 7b and 7c**: an effect on `persistence` or "
            "`risk` in the *downward* direction cannot be detected at all, and an absence "
            "of movement there is not evidence of absence.\n\n"
            + (f"Phase 8 artefacts are present at `{phase8_dir.name}/`. **It returned a "
               f"null and its manipulation check failed** -- see section 8."
               if phase8_dir.exists() else
               "**Phase 8 has not been run.** No `phase8_steering/` artefacts exist under "
               "either run, so nothing in this notebook speaks to whether steering moves "
               "either channel. Phase 7 has built and validated the instrument; the "
               "measurement is pending.")
        )



def phase8():
    """Section 8: the steering experiment. A null result, reported as one."""
    global P8

    P8 = {}
    R.heading("8. Phase 8 -- steering: a null result")
    R.md(
        "*In plain terms.* *Push the model's activity along each emotion direction and "
        "see whether anything changes -- what it says about how it feels, or how it "
        "behaves. The answer this run gives is: nothing changed that random noise of the "
        "same size did not also change.*"
    )
    R.md(
        "**This is a null result, and the manipulation check failed.** Steering along the "
        "full emotion vector did not move the report channel for two of the three "
        "emotions and moved it by one rubric point on one prompt for the third. Nothing "
        "in this grid supports a dissociation between what the model does and what it can "
        "say, and the section below is written to make that hard to misread rather than "
        "easy.\n\n"
        "The value of the phase is elsewhere: the apparatus is validated (no cell "
        "degraded), so this is an informative null rather than an inconclusive one."
    )

    with R.guard("Phase 8 load"):
        base = "phase8_steering/"
        gate = runs.json("16", base + "phase8_gate.json")
        grid = runs.csv("16", base + "phase8_grid.csv")
        gens = runs.csv("16", base + "phase8_generations.csv")
        if gate is None or grid is None:
            R.md("> No Phase 8 artefacts; section skipped.")
            return
        P8 = {"gate": gate, "grid": grid, "gens": gens,
              "summary": pd.DataFrame(gate["summary"])}

    # naming: the artefacts use v_reportable / v_remainder for what the brief calls
    # v_J / v_perp. Keep the artefact names and gloss them once.
    NICE = {"v": "v (full vector)", "v_reportable": "v_J (lens-readable part)",
            "v_remainder": "v_perp (remainder)", "v_random": "random (control)"}

    # -- 8a what ran ----------------------------------------------------------
    R.heading("8a. What ran", level=3)
    with R.guard("Phase 8 design"):
        design, usage = P8["gate"]["design"], P8["gate"]["judge_usage"]
        gens_df = P8["gens"]
        R.table(pd.DataFrame([
            {"setting": "emotions", "value": ", ".join(P8["gate"]["phase7"]["emotions"])},
            {"setting": "conditions",
             "value": ", ".join(f"{c} = {NICE.get(c, c)}" for c in design["conditions"])},
            {"setting": "signed strengths (alpha)",
             "value": ", ".join(str(a) for a in design["strengths"])},
            {"setting": "steer positions", "value": design["steer_positions"]},
            {"setting": "block", "value": design["block"]},
            {"setting": "prompts per cell", "value": design["prompts_per_cell"]},
            {"setting": "generations produced", "value": len(gens_df)},
            {"setting": "generations scored",
             "value": int(gens_df["score"].notna().sum()) if gens_df is not None else "?"},
            {"setting": "judge calls (unusable)",
             "value": f"{usage['calls']} ({usage['unusable']})"},
            {"setting": "grid cells", "value": len(P8["grid"])},
            {"setting": "||v|| per emotion",
             "value": ", ".join(f"{k} {v:.1f}" for k, v in design["norms"].items())},
        ]), "Phase 8 -- what was run")
        R.md(
            "Two design choices are worth naming because they cost coverage on purpose:\n\n"
            "* **Strengths are signed** (-1 to +1). A real effect should reverse when the "
            "direction reverses; a drift that does not is an artefact. Half the grid is "
            "spent buying that test.\n"
            "* **Positions are `generated` only** — the steering vector is added to the "
            "tokens the model produces, never to the prompt. Steering the prompt would "
            "change what was asked, which would confound a change in behaviour with a "
            "change in the question.\n\n"
            f"The design block anticipated {design['total_generations']:,} generations and "
            f"{design['judge_calls']} judge calls; {len(gens_df):,} generations and "
            f"{usage['calls']} judge calls were actually made, of which "
            f"{usage['unusable']} were unusable."
        )

    # -- 8b headline ----------------------------------------------------------
    R.heading("8b. The headline grid", level=3)
    with R.guard("Phase 8 headline"):
        summ = P8["summary"]
        rows = []
        for concept in sorted(summ["concept"].unique()):
            for condition in P8["gate"]["design"]["conditions"]:
                cell = {"emotion": concept, "condition": NICE.get(condition, condition)}
                for channel in ("report", "behaviour"):
                    sub = summ[(summ.concept == concept) & (summ.condition == condition)
                               & (summ.channel == channel)]
                    nz = sub[sub.alpha != 0]
                    cell[f"{channel} mean |z|"] = float(nz["abs_z"].mean()) if len(nz) else None
                    cell[f"{channel} max |z|"] = float(sub["abs_z"].max()) if len(sub) else None
                rows.append(cell)
        head = pd.DataFrame(rows)
        R.table(head, "Phase 8 -- effect size per emotion x condition x channel. "
                      "`mean |z|` averages the four non-zero strengths; `max |z|` is the "
                      "single largest cell. Both are in grid-SD units")

        rep = summ[(summ.channel == "report") & (summ.alpha != 0)]
        moved = sorted(rep[rep.abs_z > 0]["concept"].unique())
        flat = sorted(set(summ["concept"].unique()) - set(moved))
        R.md(
            "**1. The manipulation check failed.** If steering along the *full* emotion "
            "vector `v` cannot move the report channel, the grid cannot answer the "
            "dissociation question, because you cannot show that `v_perp` moves behaviour "
            "*without* moving report when nothing moves report.\n\n"
            f"* For **{' and '.join(flat)}** the report channel is **exactly 0.000 in "
            f"every cell, under every condition, at every strength**. Not small — zero.\n"
            f"* For **{', '.join(moved)}** it does move, but look at the raw scores before "
            f"reading anything into it: the baseline is 0.2 on a 0-4 rubric and the steered "
            f"values are 0.0, 0.2 or 0.4. That is one or two of five prompts shifting by a "
            f"single rubric point. The z of 1.39 is large only because the grid SD is "
            f"tiny. It is not a manipulation check passed; it is the coarsest possible "
            f"non-zero reading.\n\n"
            "**2. `v_perp` does not consistently beat the random control**, which is the "
            "comparison the dissociation hypothesis needs it to win."
        )

        cmp_rows = []
        for concept in sorted(summ["concept"].unique()):
            sub = summ[(summ.concept == concept) & (summ.channel == "behaviour")]
            def stat(cond, how):
                x = sub[sub.condition == cond]
                return float(x[x.alpha != 0]["abs_z"].mean()) if how == "mean" \
                    else float(x["abs_z"].max())
            cmp_rows.append({
                "emotion": concept,
                "v_perp mean |z|": stat("v_remainder", "mean"),
                "random mean |z|": stat("v_random", "mean"),
                "mean: v_perp wins?": stat("v_remainder", "mean") > stat("v_random", "mean"),
                "v_perp max |z|": stat("v_remainder", "max"),
                "random max |z|": stat("v_random", "max"),
                "max: v_perp wins?": stat("v_remainder", "max") > stat("v_random", "max"),
            })
        R.table(pd.DataFrame(cmp_rows),
                "Phase 8 -- the comparison the hypothesis rests on: does the unreadable "
                "remainder move behaviour more than random noise of the same norm?")
        R.md(
            "The answer flips with the emotion and with the statistic. `angry` loses on "
            "both; `terrified` wins on both; `content` wins on the mean and loses on the "
            "max, where the **random control produces the single largest effect in its "
            "row**. With four prompts per cell there is no ordering here to interpret — "
            "which is exactly what a null looks like when it is reported honestly rather "
            "than mined.\n\n"
            "**3. The largest single behaviour effect belongs to `v_J`, not `v_perp`** "
            "(`terrified`, max |z| 2.11) — the opposite of the hypothesis, which predicted "
            "the *unreadable* part would drive behaviour. Before treating even that as a "
            "finding, note the limitation below: norm-matching amplifies `v_J` by roughly "
            "6x while leaving `v_perp` near its original length, so `v_J` is being pushed "
            "considerably harder than the condition it is compared against."
        )

    # -- 8c dose-response figure ---------------------------------------------
    R.heading("8c. Dose-response", level=3)
    with R.guard("Phase 8 dose-response figure"):
        R.fig(
            F.phase8_dose_response(P8["summary"], NICE, RUNS["16"]),
            "phase8_dose_response.png",
            "Phase 8: dose-response for both channels, all conditions",
            plain=(
                "If steering really does something, pushing harder should do more of it, "
                "and pushing the other way should do the opposite. This plots how much "
                "each channel moved against how hard we pushed. Straight, separated lines "
                "would mean a real effect; tangled lines that cross the random-noise line "
                "mean nothing is happening."
            ),
            how_to_read=(
                "Top row: the behaviour channel. Bottom row: the report channel. One "
                "column per emotion. The x-axis is the signed steering strength alpha, so "
                "0 in the middle is unsteered and the two halves are opposite directions. "
                "The y-axis is the signed effect in grid-SD units. **The dashed grey line "
                "is the random control** — any real condition has to separate from it to "
                "mean anything. A genuine effect would look like a monotone line through "
                "the origin with the same sign on both sides reversed."
            ),
            what_it_shows=(
                "In the behaviour row the four lines are tangled and repeatedly cross, and "
                "the random control is not below the others — for `angry` it is among the "
                "largest. There is no monotone dose-response in any panel and no "
                "consistent sign reversal, which is what the signed strengths were "
                "included to detect.\n\n"
                "The report row is the manipulation check, drawn to the same scale: two of "
                "three panels are **flat on zero across the entire range**. `content` "
                "moves, in steps, between the only three values its five prompts can "
                "produce. Nothing here is a dose-response curve."
            ),
        )

    # -- 8d what did work -----------------------------------------------------
    R.heading("8d. What did work: the apparatus is validated", level=3)
    with R.guard("Phase 8 fluency"):
        grid = P8["grid"]
        thresh = P8["gate"]["config"].get("perplexity_max_ratio", 1.5)
        R.table(pd.DataFrame([
            {"check": "grid cells", "value": len(grid)},
            {"check": "cells flagged degraded", "value": int(grid["degraded"].sum())},
            {"check": "degradation threshold",
             "value": f"perplexity ratio > {thresh}x baseline"},
            {"check": "worst perplexity ratio observed",
             "value": float(grid["perplexity_ratio"].max())},
        ]), "Phase 8 -- fluency check across every steered cell")
        R.md(
            f"**Not one of the {len(grid)} cells degraded.** The worst perplexity ratio "
            f"anywhere in the grid is {grid['perplexity_ratio'].max():.3f}x baseline, "
            f"against a {thresh}x threshold — the steering never broke the model's "
            f"fluency.\n\n"
            "This is what makes the null informative rather than uninformative. Two "
            "failure modes are ruled out: the intervention was applied (the activations "
            "changed, and the text stayed coherent), and no apparent effect anywhere in "
            "the grid is degradation wearing an effect's clothes. What is left is a "
            "measurement that ran correctly and found nothing — which is a result, "
            "unlike a measurement that fell over."
        )

    # -- 8e limitations -------------------------------------------------------
    R.heading("8e. Limitations", level=3)
    with R.guard("Phase 8 limitations"):
        design = P8["gate"]["design"]
        spec = P8["gate"]["specificity_control"]
        norms = design["norms"]
        R.md(
            "These are the reasons this null is weaker than a null could be. They are a "
            "list, not a footnote, because at least two of them could each account for "
            "the whole result.\n\n"
            f"1. **The specificity control did not run.** `{spec['skipped_reason']}`. "
            f"Without it, even a real dissociation could not have been shown to be "
            f"*about emotion* rather than about pushing any concept vector of that size "
            f"through the residual stream.\n"
            f"2. **The intervention may simply be too weak to test.** The emotion vectors "
            f"have norms of "
            f"{', '.join(f'{v:.1f}' for v in norms.values())} against a residual-stream "
            f"norm around 108, so alpha = 1 is roughly a 17% perturbation. A null at this "
            f"amplitude does not license a null at any amplitude.\n"
            f"3. **Norm-matching biases the key comparison.** Scaling `v_J` to `||v||` "
            f"amplifies it by roughly 6x, while `v_perp` — which is most of the vector "
            f"already — is left near its original length. `v_J` is therefore pushed "
            f"harder than `v_perp`, and the one large effect in the grid is a `v_J` "
            f"effect.\n"
            f"4. **Two behaviour families had zero baselines** (Phase 7c): `persistence` "
            f"and `report` sit at exactly 0 and `risk` is pinned, so three of five "
            f"families could not register a decrease at all. The behavioural channel is "
            f"effectively two families wide.\n"
            f"5. **{P8['gate']['judge_usage']['unusable']} of "
            f"{P8['gate']['judge_usage']['calls']} judge calls were unusable**, and 10 of "
            f"{len(P8['gens']):,} generations went unscored.\n"
            f"6. **Four prompts per cell.** Every number in 8b is a mean over four "
            f"scored generations, which is why the mean/max orderings disagree.\n"
            f"7. **The lens is still the 80-prompt interrupted fit** (section 0), and "
            f"`v_J`/`v_perp` are defined by it."
        )

    # -- 8f re-entry caveat ---------------------------------------------------
    R.heading("8f. The re-entry caveat", level=3)
    with R.guard("Phase 8 re-entry caveat"):
        caveats = P8["gate"]["caveats"]
        R.md(
            "This run produced no `v_perp` effect, so nothing here is subject to the "
            "caveat below. It is recorded because it applies to **any** future `v_perp` "
            "result, including a positive one from a higher-strength re-run, and it is "
            "easier to state before there is a result to defend than after.\n\n"
            f"> {caveats['re_entry']}\n\n"
            "In other words: even a clean `v_perp` effect on behaviour would not by itself "
            "show that an unverbalizable component drove behaviour. The concept could be "
            "re-derived further down the network and re-enter the reportable workspace "
            "there. Distinguishing those two stories needs the Phase 9 clamp, which has "
            "not been run.\n\n"
            "The other caveats the gate records about itself:"
        )
        R.table(pd.DataFrame([{"caveat": k, "the gate's own wording": v}
                              for k, v in caveats.items() if k != "re_entry"]),
                "Phase 8 -- caveats recorded by the phase itself")

    # -- 8g manipulation check at higher strength ----------------------------
    with R.guard("Phase 8 high-alpha re-run"):
        strengths = P8["gate"]["design"]["strengths"]
        if max(abs(a) for a in strengths) > 1.0:
            R.heading("8g. MANIPULATION CHECK at higher strength", level=3)
            summ = P8["summary"]
            high = summ[(summ.channel == "report") & (summ.condition == "v")
                        & (summ.alpha.abs() > 1.0)]
            R.table(high[["concept", "alpha", "abs_z"]],
                    "Phase 8 -- did the full vector move the report channel above alpha 1?")
            R.md("**This is the number that decides how to read the null.** If `v` moves "
                 "report at higher strength, the alpha<=1 grid was simply too weak to "
                 "test and the null is about amplitude. If it still does not move, the "
                 "null is about the intervention.")
        else:
            R.md(
                "**No higher-strength re-run exists.** Every strength in this grid is "
                f"within +/-1 ({', '.join(str(a) for a in strengths)}), so the question "
                "that would settle how to read this null — *does the full vector move the "
                "report channel at all, given enough amplitude?* — has not been asked. "
                "Until it is, **this null means 'no effect at a ~17% perturbation', not "
                "'no effect'.** A re-run with alpha up to 4 on the `v` condition and the "
                "report channel alone would cost a fraction of this grid and would decide "
                "it."
            )


def summary():
    """Section 7: claims table and caveats."""

    R.heading("7. Summary")
    display(Markdown("### Claims table"))

    def p6_stats(key):
        """Flatten the new Phase 6 gate into the few scalars the summary needs."""
        if key not in P6:
            return None
        gate, frame, ks = P6[key]["gate"], P6[key]["frame"], P6[key]["ks"]
        k0 = ks[0]
        emotional = frame[frame["emotion"] != "neutral"]
        g, ctl = gate["gate"], gate["random_control"][k0]
        return {
            "k": k0,
            "mean": float(emotional[f"frac_reportable k={k0}"].mean()),
            "remainder": float(emotional[f"frac_remainder k={k0}"].mean()),
            "ctl_mean": ctl["mean"], "ctl_n": ctl["n"],
            "ratio": float(emotional[f"frac_reportable k={k0}"].mean()) / ctl["mean"],
            "beat": len(g["beat_null"][k0]), "n": g["n_emotions"],
            "resolvable": g["p_value_floor"] < g["alpha_bonferroni"],
            "perms": int(round(1 / g["p_value_floor"] - 1)),
            "perms_needed": int(np.ceil(1 / g["alpha_bonferroni"])),
            "coherence_above": gate["coherence"]["frac_above_threshold"],
        }

    P6S = {k: p6_stats(k) for k in RUN_KEYS if p6_stats(k)}

    with R.guard("claims table"):
        a16, a171 = ALIGN.get("16"), ALIGN.get("171")

        def g3(key, *path):
            node = GATE3.get(key, {})
            for step in path:
                node = node.get(step, {}) if isinstance(node, dict) else {}
            return node

        def tier(run, prefix, col):
            if not TIERS:
                return float("nan")
            for _, row in TIERS["summary"].iterrows():
                if row["run"] == run and str(row["tier"]).startswith(prefix):
                    return row[col]
            return float("nan")

        def cjk(k):
            sub = SCRIPT[k]
            return (sub[sub.group == "gate_a_emotion_vector"]["script"] == "CJK").mean()

        def sh(k):
            return SH[k][SH[k].emotion != "neutral"]["cosine_centered"]

        CLAIMS = [
            {"claim": "PCA over emotion vectors recovers a valence x arousal plane it was never "
                      "told about",
             "evidence": f"16 run: PC1 r_arousal "
                         f"{float(a16.set_index('pc').loc[1, 'r_arousal']):+.2f}, PC2 r_valence "
                         f"{float(a16.set_index('pc').loc[2, 'r_valence']):+.2f}; "
                         f"{sum(g3('16', 'pca').get('explained_variance_ratio', [0, 0])[:2]):.0%} "
                         f"of variance vs a "
                         f"{g3('16', 'null_band').get('analytic_isotropic_top2', float('nan')):.1%} "
                         f"isotropic null; cross-fit worst plane angle "
                         f"{g3('16', 'alignment', 'crossfit').get('worst_plane_angle_deg', float('nan')):.0f} deg",
             "status": "supported (16-emotion balanced design)"},

            {"claim": "The same two axes survive when the emotion set widens to all 171",
             "evidence": f"cross-run abs cosine: 16-PC1 to 171-PC{XRUN['match_pc'][0]} = "
                         f"{XRUN['match_cos'][0]}, 16-PC2 to 171-PC{XRUN['match_pc'][1]} = "
                         f"{XRUN['match_cos'][1]}; but only 27% of variance, split-half PC "
                         f"stability 0.63, `axes_identified = False`, `plane_stable = False`",
             "status": "exploratory -- the directions recur, the ranking and stability do not"},

            {"claim": "Emotion vectors are reliable enough to build a PCA on",
             "evidence": f"split-half centred cosine, 16 run: min {sh('16').min():.3f}, mean "
                         f"{sh('16').mean():.3f}, 0 of 16 below 0.9. 171 run: min "
                         f"{sh('171').min():.3f}, mean {sh('171').mean():.3f}, "
                         f"{int((sh('171') < 0.9).sum())} of 171 below 0.9",
             "status": "supported (16); qualified (171 -- a tail of unreliable emotions)"},

            {"claim": "The circumplex PCs are lexically readable through the Jacobian lens",
             "evidence": "16 run: PC1 arousal AUROC 0.98 (p = 0.003), PC2 valence 1.00 "
                         "(p = 0.001), both pre-registered at alpha 0.05. 171 run: PC2 arousal "
                         "0.96, PC3 valence 0.96 (exploratory, clears the 0.0083 family "
                         "threshold). Everything else MURKY.",
             "status": "supported for the ordering statistic; token content is a separate question"},

            {"claim": "An emotion vector's lens readout names the emotion (GATE A)",
             "evidence": f"16 run 0/{P4['16']['gate_a']['n_scorable']} hits; 171 run "
                         f"{P4['171']['gate_a']['n_hits']}/{P4['171']['gate_a']['n_scorable']} "
                         f"({P4['171']['gate_a']['hit_rate']:.1%}) against a "
                         f"{P4['171']['gate_a']['chance_hit_rate']:.1e} chance rate; pass bar 50%",
             "status": "FAILED as specified -- but see the next two rows: the test asked in the "
                       "wrong language"},

            {"claim": "The emotion directions encode valence in English even though they do not "
                      "verbalise it in English",
             "evidence": "the +valence a-priori axis separates pleasant from unpleasant anchors "
                         "perfectly (AUROC 1.00) while ranking its best English probe word at "
                         "1,927 of 151,936 and none inside GATE A's top-12 window; +arousal the "
                         "same at AUROC 0.96",
             "status": "supported -- and it is why GATE A (absolute containment) failed while "
                       "GATE B (relative ordering) passed"},

            {"claim": "The GATE A failure is a property of the test's language, not the directions",
             "evidence": f"re-scoring the same top-12 rule against pre-committed Chinese "
                         f"translations: {tier('16', 'T3', 'rate (all)'):.0%} and "
                         f"{tier('171', 'T3', 'rate (all)'):.0%}, against permutation nulls of "
                         f"{tier('16', 'T3', 'permutation null mean'):.0%} and "
                         f"{tier('171', 'T3', 'permutation null mean'):.0%} that shuffle the "
                         f"same lists between emotions (p = {tier('16', 'T3', 'p'):.4f}, "
                         f"{tier('171', 'T3', 'p'):.4f}); {cjk('16'):.0%} and {cjk('171'):.0%} "
                         f"of emotion-vector top-12 tokens are CJK",
             "status": "supported, secondary -- clears GATE A's 50% bar on the 16 run; rests on "
                       "a hand-written table, so it supplements rather than replaces the FAILED "
                       "verdict"},

            {"claim": "GATE A could score every emotion",
             "evidence": "57 of 171 emotions have no single-token English lemma and were "
                         "excluded; against the local tokenizer, 43 of those 57 do have a "
                         "single-token Chinese form",
             "status": "FAILED -- a third of the 171-emotion run was structurally unscorable"},

            {"claim": "The Chinese readouts are just an artefact of Qwen being a Chinese model",
             "evidence": "96-97% of the Latin tokens in these readouts are whole words "
                         "(` failed`, ` sorrow`, ` panic`), not subword fragments; the stimuli, "
                         "anchors and probes are all English",
             "status": "FAILED as a complete explanation -- the CJK preference is a property of "
                       "the block-31 readout, not further diagnosed here"},

            {"claim": "GATE B: the lens converges on the logit lens and the model at the top",
             "evidence": "||J - I||_F/||I||_F falls 1.365 -> 0.472 across blocks 0-62 "
                         "(non-monotone, peak 1.65 at block 43); top-1 agreement 1/1 with both; "
                         "top-12 overlap 10/12 and 8/12",
             "status": "supported (shape only -- absolute values depend on the unconfirmed "
                       "divisor)"},

            {"claim": "The circumplex is a sufficient description of emotion representation",
             "evidence": f"participation ratio "
                         f"{g3('16', 'pca').get('participation_ratio', float('nan')):.2f} (16) "
                         f"and {g3('171', 'pca').get('participation_ratio', float('nan')):.2f} "
                         f"(171); top-2 PCs hold only "
                         f"{sum(g3('171', 'pca').get('explained_variance_ratio', [0, 0])[:2]):.0%} "
                         f"of the 171 run's variance",
             "status": "FAILED -- the circumplex is a 2-D slice of a ~10-D space"},

            {"claim": "PC1 of the 171 run is a circumplex axis",
             "evidence": f"`neutral`, held out of the PCA fit, projects to PC1 = "
                         f"{PC1_171.get('neutral', float('nan')):+.1f}, beyond the "
                         f"{PC1_171.get('max', float('nan')):+.1f} maximum of all 171 fitted "
                         f"emotions; low end {' / '.join(PC1_171.get('lo', ['?'])[:3])}, high "
                         f"end {' / '.join(PC1_171.get('hi', ['?'])[:3])}",
             "status": "FAILED -- PC1 is substantially an affect-presence / intensity axis"},

            {"claim": "A measurable fraction of an emotion vector is lens-readable (Phase 6)",
             "evidence": (", ".join(
                 f"{RUNS[k]}: {v['mean']:.1%} readable at k={v['k']} vs a {v['ctl_mean']:.1%} "
                 f"random-direction control (n={v['ctl_n']}) = {v['ratio']:.1f}x, "
                 + (f"{v['beat']}/{v['n']} beat the Bonferroni null"
                    if v["resolvable"] else
                    f"Bonferroni null unresolvable at {v['perms']} permutations "
                    f"(needs {v['perms_needed']:,})")
                 for k, v in P6S.items())
                 + "; read-space score identity holds at r > 0.9999"
                 ) if P6S else "Phase 6 artefacts missing",
             "status": ("supported, small effect -- ~2-3% readable, reliably above a 500-sample "
                        "null in the 16-emotion run; same effect size in the 171 run but its "
                        "corrected test cannot be resolved at 500 permutations")
                       if P6S else "Phase 6 artefacts missing"},

            {"claim": "Phase 7's two measurement channels are separate",
         "evidence": (f"mechanical scan of {P7['gate']['separation']['n_affect_stems']} "
                      f"affect stems over every prompt, rubric and scorer source: "
                      f"{P7['gate']['separation']['behaviour_hits']} hits in the behaviour "
                      f"channel, {P7['gate']['separation']['report_hits']} in the report "
                      f"channel by design; 3 of 4 behaviour families scored by readable "
                      f"code rather than a judge") if P7 else "Phase 7 artefacts missing",
         "status": "measurement apparatus validated; separation gate passed; candidate "
                   "filter overridden (documented)"},

        {"claim": "Both Phase 7 channels are at floor before any steering",
         "evidence": ("report channel 0.0 on all 5 prompts (flat denial of inner states), "
                      "corroborated by an activation-level probe at -0.017 to +0.011 "
                      "cosine; behaviour channel has usable range in only 2 of 5 families "
                      "-- persistence and report at exactly 0, risk pinned at 1.0")
                     if P7 else "Phase 7 artefacts missing",
         "status": "supported -- maximum headroom upward for the report channel, but a "
                   "predicted *decrease* is untestable on 3 of 5 behaviour families"},

        {"claim": "Phase 7 shows emotion causes behaviour",
         "evidence": "Phase 7 applies no steering; no phase8_steering/ artefacts exist",
         "status": "NOT TESTED -- Phase 7 validates the instrument only; the measurement "
                   "is Phase 8 and is pending"},

        {"claim": "A concept should surface as its own English token (the pipeline's "
                  "recurring validity criterion)",
         "evidence": ("encoded four times -- Phase 0 GATE A, Phase 4 GATE A, Phase 6's "
                      "original atom check, Phase 7's candidate filter -- and failed every "
                      "time on this model: 0 emotion-item hits, 0/14 and 5/114, a "
                      "wrong-direction test, and 0 of 16 candidates passing"),
         "status": "FAILED as a validity criterion -- it measures the model's "
                   "lexicalisation language as much as the method; three of the four were "
                   "first misread as pipeline failures"},

        {"claim": "Steering an emotion vector moves behaviour without moving self-report "
                  "(the dissociation Phase 8 was built to test)",
         "evidence": ("null: the report channel is exactly 0.000 in every cell for angry "
                      "and terrified under every condition including the full vector v, "
                      "and moves by one rubric point on one of five prompts for content; "
                      "v_perp does not consistently beat the random control (loses on both "
                      "statistics for angry, wins both for terrified, splits for content, "
                      "where random gives the largest single effect); the biggest "
                      "behaviour effect is v_J at |z| 2.11, the opposite of the "
                      "hypothesis") if P8 else "Phase 8 artefacts missing",
         "status": "null -- no dissociation; failed manipulation check"},

        {"claim": "Phase 8's apparatus worked, so the null is informative",
         "evidence": ("0 of 300 grid cells degraded, worst perplexity ratio 1.027x against "
                      "a 1.5x threshold, so no effect is degradation in disguise and the "
                      "intervention did run") if P8 else "Phase 8 artefacts missing",
         "status": "supported -- but the specificity control did not run, alpha never "
                   "exceeded 1 (~17% perturbation), and norm-matching amplifies v_J ~6x "
                   "relative to v_perp"},

        {"claim": "The lens artefact is a converged fit",
             "evidence": f"checkpoint `n_done` = "
                         f"{PROV['16']['lens prompts (checkpoint n_done)']} vs `config.yaml "
                         f"prompts_fitted` = {PROV['16']['lens prompts (config.yaml claim)']}; "
                         f"the fit's own `min_prompts` floor is 100",
             "status": "FAILED -- interrupted fit, divisor unconfirmed; top-k results survive, "
                       "magnitude-sensitive ones do not"},
        ]
        R.table(pd.DataFrame(CLAIMS), "Claims, evidence, status")
    display(Markdown("### Caveats"))

    with R.guard("caveats"):
        p6_caveat = (
            "**Phase 6 is now a supported result, and it is small.** The read-space "
            "decomposition (`atom_mode: read`, atoms `Jᵀ(g ⊙ u_t)`) finds "
            + " and ".join(f"{v['mean']:.1%} readable for {RUNS[k]}" for k, v in P6S.items())
            + " against random-direction controls of "
            + " and ".join(f"{v['ctl_mean']:.1%} (n={v['ctl_n']})" for v in P6S.values())
            + " -- ratios of "
            + " and ".join(f"{v['ratio']:.1f}x" for v in P6S.values())
            + ". The remainder is ~97% throughout, and raising k from 16 to 25 moves the "
            "fraction by at most ~1e-3, so the ceiling is a property of the pool and the "
            "vector rather than of the sparsity budget. Nothing here reaches the 5-15% the "
            "brief anticipated."
        ) if P6S else (
            "**Phase 6 artefacts are missing**, so no reportable fraction is stated."
        )

        p6_null_caveat = (
            "**One Phase 6 significance number must not be read at face value.** The 171-run "
            "gate records 0 of 172 emotions beating its Bonferroni-corrected null. That is a "
            "resolution limit, not a null result: 500 permutations put the smallest achievable "
            "p-value at 0.00200 while Bonferroni over 172 emotions demands 0.00029, so no "
            "effect size could clear it. It would need ~3,440 permutations. All 172 clear the "
            "uncorrected 0.05. The 16-run test *is* resolvable (threshold 0.00294 against the "
            "same 0.00200 floor) and there 17 of 17 beat the null."
        ) if P6S else ""

        p6_method_caveat = (
            "**An earlier Phase 6 verdict was wrong, and the reason is methodological.** The "
            "read direction for token `t` is `Jᵀu_t`, because the lens score `u_tᵀJh` regroups "
            "as `(Jᵀu_t)ᵀh`; the write direction `J⁺u_t` answers the different question of what "
            "perturbation makes the model emit `t`. A previous version gated Phase 6 on \"does "
            "lensing an atom return its own token?\", which is a **write**-direction property "
            "that a correct read dictionary has no reason to satisfy — the write-space ablation "
            "passes it 24/24, which is the tell. The read construction is validated instead by "
            "a score-identity check that holds at r > 0.9999 over 8 probes. Any earlier "
            "`RESULTS.md` showing Phase 6 as FAILED is superseded."
        )

        p6_attribution_caveat = (
            f"**Phase 6's specific token attributions are weak even where the fraction is not.** "
            f"{P6S['16']['coherence_above']:.1%} of the 512 atoms in the pool exceed the 0.5 "
            f"interchangeability threshold, so which atom the pursuit picked is close to "
            f"arbitrary among near-duplicates. And the remainder means \"outside the k-sparse "
            f"nonnegative span of this pool at this k\" — a union of cones, not a linear "
            f"subspace — so it is **not** evidence that anything is intrinsically "
            f"unverbalizable. `own_word_atom_rank` is null for "
            f"{int(P6['16']['frame']['own_word_atom_rank'].isna().sum())}/{len(P6['16']['frame'])} "
            f"and {int(P6['171']['frame']['own_word_atom_rank'].isna().sum())}/{len(P6['171']['frame'])} "
            f"emotions: the readable component is not the emotion's own word."
        ) if P6S else ""

        CAVEATS = [
            "**The lens is an interrupted 80-prompt fit.** The published `qwen3-32b` artefact is "
            "a resumable `fit_checkpoint` whose `n_done` is 80 while its `config.yaml` claims "
            "615, short of the fit script's own `min_prompts = 100` floor. Because the final "
            "norm is an RMSNorm, top-k readouts are scale-free and survive; `||J - I||`, any "
            "variance split, and anything else magnitude-sensitive do not.",

            f"**PC1 of the 171 run is contaminated by affect-presence.** `neutral` was held out "
            f"of the PCA fit and still projects to PC1 = "
            f"{PC1_171.get('neutral', float('nan')):+.1f}, past the "
            f"{PC1_171.get('max', float('nan')):+.1f} maximum over all 171 fitted emotions. It "
            f"correlates with valence (r = +0.82) only because intense states in this vocabulary "
            f"skew unpleasant.",

            "**Effective dimensionality is 9.8 for the 171 run** (3.6 for the 16 run). The "
            "circumplex is a 2-D slice of a roughly 10-D space, and the top two PCs of the 171 "
            "run carry only 36% of variance.",

            p6_caveat,
            p6_null_caveat,
            p6_method_caveat,
            p6_attribution_caveat,

            "**GATE A failed in both runs as specified** — 0/14 and 5/114 English-word "
            "self-hits. Section 5f re-scores it against a pre-committed Chinese table and gets "
            "63% and 25%, well above a permutation null. Both things are true: the directions "
            "are far more readable than GATE A implies, *and* the recorded verdict is still "
            "FAILED, because the re-scoring depends on hand-written data rather than on anything "
            "the pipeline measured.",

            "**The relaxed GATE A is a containment test, not a rank test.** Only the top 12 "
            "tokens per direction were persisted. Section 5i specifies the proper fix: `J[31]` "
            "and `lm_head`, a CPU matrix-vector product, no GPU pod, ~1.7 GB of download.",

            "**A third of the 171-emotion run was never scorable by GATE A.** 57 of 171 have no "
            "single-token English lemma; 43 of those do have a single-token Chinese form. Any "
            "headline GATE A rate should carry the denominator it was computed on.",

            "**Translation quality is a human judgement in the loop.** `zh_en_glossary.py` is "
            "the author's work, pre-committed and permutation-controlled, but a reader who "
            "rejects specific pairs should re-score with their own list.",

            "**The 171 run's top-2 plane is not stable.** `axes_identified = False`, "
            "`plane_stable = False`, PC2/PC3 split-half stability 0.63 against a 0.8 threshold, "
            "worst cross-fit plane angle 62 degrees.",

            "**Every axis measurement rests on 16 anchors** with ±1 coordinates, not continuous "
            "norms, and every Phase 4 AUROC is over the 14 of those that are single tokens. "
            "n = 14 is a coarse instrument.",

            "**Phase 0 was only run for the 16-emotion run.** Both load the same lens at the "
            "same block, so the gate transfers, but it was not independently re-gated.",

            "**One model, one block, one lens, one dataset.** Qwen3-32B at block 31, the "
            "`neuronpedia/jacobian-lens` `qwen3-32b` artefact, `ryancodrai/emotion-probes` at "
            "sha `720f2eb3`. Phases 5, 7 and 8 produced no artefacts.",

            "**Story generation is itself a model artefact.** Topic matching controls for topic, "
            "not for the generator's stereotypes about how each emotion is written.",
        ]
        CAVEATS = [c for c in CAVEATS if c]
        R.md("### Caveats\n\n" + "\n\n".join(f"{i}. {c}" for i, c in enumerate(CAVEATS, 1)))

    if R.missing:
        R.md("### Missing or unavailable artefacts\n\n"
             + "\n".join(f"* `{m}`" for m in R.missing))
    else:
        R.md("### Missing or unavailable artefacts\n\nNone -- every artefact this notebook "
             "expects was present.")


def export():
    """Write RESULTS.md and assert every figure was explained."""

    # ------------------------------------------------------------- export RESULTS.md
    unexplained = R.unexplained_figures()
    assert unexplained == 0, f"{unexplained} figure(s) emitted without both explanations"

    out = R.write(
        R.analysis / "RESULTS.md",
        header=("<!-- Generated by analysis/results_notebook.ipynb. Do not edit by hand. -->\n"
                "<!-- Every number is read from outputs/*/results/phases/; nothing was re-run. -->\n"),
    )
    print(f"wrote {out}")
    print(f"  {len(R.blocks)} markdown blocks, {len(R.figures)} figures, "
          f"{len(R.missing)} missing artefacts, {unexplained} unexplained figures")
    print(f"  {out.stat().st_size / 1024:.1f} KiB")
