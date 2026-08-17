"""Plot builders for `results_notebook.ipynb`.

Every function here takes already-loaded data and returns a bare matplotlib ``Figure``.
None of them save, display, caption, or explain -- that is :meth:`nbtools.Report.fig`'s
job, and keeping the split means a figure's *prose* lives next to the argument it
supports in the notebook while its *drawing code* lives here.

Styling comes from :mod:`core.plotting` throughout, so these match the figures the
pipeline itself produced. Two conventions worth stating because they are easy to get
wrong and invisible when you do:

* ``ax.set_axisbelow(True)`` plus ``zorder=3`` on every bar. Matplotlib draws gridlines
  over bars by default, which reads as white banding across the data.
* Tokens never appear in a figure. The readouts are largely CJK and matplotlib's default
  font renders those as tofu boxes; token content belongs in markdown tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core import plotting

__all__ = [
    "gate_b_identity", "quadrant_coverage", "split_half", "pc2_pc3_scatter",
    "auroc_per_pc_per_end", "gate_a_tiers", "probe_rank_burial", "dictionary_validity",
    "readout_concentration", "remainder_bars",
    "phase7_baselines", "phase7_report_probe", "phase8_dose_response",
]


def gate_b_identity(gate_b: dict, run_name: str):
    """``||J - I||_F / ||I||_F`` against block, with the endpoints and the peak marked."""
    distances = {int(k): float(v) for k, v in gate_b["identity_distances"].items()}
    blocks = sorted(distances)
    values = np.array([distances[b] for b in blocks])
    low, high = distances[blocks[0]], distances[blocks[-1]]
    peak = int(np.argmax(values))

    fig, ax = plt.subplots()
    ax.plot(blocks, values, color=plotting.SERIES[0], linewidth=1.8)
    for x, y, text, (dx, dy), ha in [
        (blocks[0], low, f"block {blocks[0]}: {low:.3f}", (7, 8), "left"),
        (blocks[peak], values[peak],
         f"peak, block {blocks[peak]}: {values[peak]:.3f}", (-8, 4), "right"),
        (blocks[-1], high, f"block {blocks[-1]}: {high:.3f}", (-6, -14), "right"),
    ]:
        ax.plot([x], [y], marker="o", color=plotting.SERIES[0], markersize=5,
                markeredgecolor=plotting.SURFACE, markeredgewidth=1.2, zorder=5)
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=plotting.INK_SECONDARY, ha=ha)
    ax.axhline(0.0, color=plotting.BASELINE, linewidth=0.8, zorder=0)
    plotting.finish(
        fig, ax,
        "GATE B: the transport approaches the identity at the top of the stack",
        "block", r"$\|J - I\|_F\ /\ \|I\|_F$",
        subtitle=(f"{run_name}; falls {low:.3f} -> {high:.3f} across blocks "
                  f"{blocks[0]}-{blocks[-1]}, non-monotonically"),
        integer_x=True,
    )
    return fig


def quadrant_coverage(counts_by_run: dict[str, dict], labels: dict[str, str]):
    """One panel per run: stimulus count per circumplex quadrant."""
    order = ["HA-P", "HA-N", "LA-P", "LA-N", "neutral", "unlabelled"]
    keys = list(counts_by_run)
    fig, axes = plt.subplots(1, len(keys), figsize=(9.6, 3.9))
    for ax, key in zip(np.atleast_1d(axes), keys):
        counts = counts_by_run[key]
        names = [q for q in order if q in counts]
        values = [counts[q] for q in names]
        colours = [plotting.SERIES[0] if q not in ("neutral", "unlabelled")
                   else plotting.INK_MUTED for q in names]
        ax.set_axisbelow(True)
        bars = ax.bar(names, values, color=colours, width=0.66, zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:,}", xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=plotting.INK_SECONDARY)
        ax.set_title(labels[key], loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        ax.set_ylabel("stimuli")
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.16)
    fig.suptitle("Phase 1: stimulus coverage by circumplex quadrant", x=0.02, ha="left",
                 fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def split_half(frames_by_run: dict[str, pd.DataFrame], labels: dict[str, str],
               threshold: float = 0.9):
    """Per-emotion split-half cosine, sorted ascending, one panel per run.

    Below ~40 emotions each bar is tick-labelled. Above that the bars are a pixel apart,
    so the sub-threshold emotions go into a text box instead of unreadable rotated
    labels.
    """
    keys = list(frames_by_run)
    fig, axes = plt.subplots(len(keys), 1, figsize=(9.6, 3.6 * len(keys)),
                             gridspec_kw={"hspace": 0.55})
    for ax, key in zip(np.atleast_1d(axes), keys):
        frame = frames_by_run[key].sort_values("cosine_centered").reset_index(drop=True)
        values = frame["cosine_centered"].to_numpy()
        names = frame["emotion"].tolist()
        below = values < threshold
        xs = np.arange(len(values))

        ax.set_axisbelow(True)
        ax.bar(xs, values,
               color=[plotting.DIVERGING_HIGH if b else plotting.SERIES[0] for b in below],
               width=0.85 if len(values) < 40 else 1.0, linewidth=0, zorder=3)
        ax.axhline(threshold, color=plotting.INK_SECONDARY, linewidth=1.0,
                   linestyle="--", zorder=4)
        ax.set_ylim(min(0.78, float(values.min()) - 0.03), 1.005)
        ax.annotate(f"{threshold} reliability threshold",
                    xy=(len(values) * 0.995, threshold), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8.5,
                    color=plotting.INK_SECONDARY, zorder=5)

        if len(values) <= 40:
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        else:
            ax.set_xticks([])
            listed = ", ".join(f"{n} {v:.3f}"
                               for n, v in zip(names, values) if v < threshold)
            ax.text(0.30, 0.06, f"below {threshold}: " + (listed or "none"),
                    transform=ax.transAxes, fontsize=7.5,
                    color=plotting.INK_SECONDARY, va="bottom", ha="left", wrap=True,
                    bbox=dict(facecolor=plotting.SURFACE, edgecolor=plotting.GRIDLINE,
                              linewidth=0.7, boxstyle="round,pad=0.4"))
        ax.set_title(f"{labels[key]}   ({int(below.sum())} of {len(values)} below {threshold})",
                     loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        ax.set_ylabel("split-half cosine (centred)")
        ax.set_xlabel("emotion, sorted ascending")
    fig.suptitle("Phase 2: split-half reliability per emotion", x=0.02, ha="left",
                 fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def pc2_pc3_scatter(scores: pd.DataFrame, alignment: pd.DataFrame, run_name: str):
    """Emotion vectors in the PC2-PC3 plane, anchors highlighted, axes projected in."""
    axis_rows = alignment.set_index("pc")
    anchors = scores[scores["is_anchor"] == True]  # noqa: E712 -- pandas mask
    others = scores[(scores["is_anchor"] != True) & (scores["in_pca_fit"] == True)]  # noqa: E712
    neutral = scores[scores["emotion"] == "neutral"]

    # Anchors cluster; nudge a colliding label down rather than overprinting it.
    x_span = float(scores["pc2"].max() - scores["pc2"].min())
    y_span = float(scores["pc3"].max() - scores["pc3"].min())
    placed: list[tuple[float, float]] = []

    def label_offset(x, y):
        for px, py in placed:
            if abs(x - px) < 0.10 * x_span and abs(y - py) < 0.035 * y_span:
                placed.append((x, y))
                return (6, -11)
        placed.append((x, y))
        return (6, 3)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.grid(True, axis="both", color=plotting.GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color=plotting.BASELINE, linewidth=0.8)
    ax.axvline(0, color=plotting.BASELINE, linewidth=0.8)

    ax.scatter(others["pc2"], others["pc3"], s=14, color=plotting.INK_MUTED, alpha=0.45,
               linewidths=0,
               label=f"{len(others)} unlabelled emotions (in fit, no circumplex coords)")

    for (valence, arousal), (colour, filled, name) in {
        (1.0, 1.0): (plotting.SERIES[0], True, "pleasant, activated"),
        (1.0, -1.0): (plotting.SERIES[0], False, "pleasant, deactivated"),
        (-1.0, 1.0): (plotting.SERIES[1], True, "unpleasant, activated"),
        (-1.0, -1.0): (plotting.SERIES[1], False, "unpleasant, deactivated"),
    }.items():
        sub = anchors[(anchors["valence"] == valence) & (anchors["arousal"] == arousal)]
        ax.scatter(sub["pc2"], sub["pc3"], s=58,
                   facecolors=colour if filled else "none", edgecolors=colour,
                   linewidths=1.6, label=name, zorder=4)
        for row in sub.itertuples():
            ax.annotate(row.emotion, xy=(row.pc2, row.pc3),
                        xytext=label_offset(row.pc2, row.pc3),
                        textcoords="offset points", fontsize=8.5,
                        color=plotting.INK_PRIMARY, zorder=5)

    if len(neutral):
        point = neutral.iloc[0]
        ax.scatter([point["pc2"]], [point["pc3"]], marker="D", s=54,
                   color=plotting.INK_SECONDARY, zorder=5,
                   label="neutral (projected, not fitted)")
        ax.annotate("neutral", xy=(point["pc2"], point["pc3"]), xytext=(7, 3),
                    textcoords="offset points", fontsize=8.5,
                    color=plotting.INK_SECONDARY)

    span = float(max(abs(scores["pc2"]).max(), abs(scores["pc3"]).max()))
    for axis, label in [("valence", "a priori valence"), ("arousal", "a priori arousal")]:
        dx = float(axis_rows.loc[2, f"cos_{axis}"])
        dy = float(axis_rows.loc[3, f"cos_{axis}"])
        ax.annotate("", xy=(dx * span, dy * span), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=plotting.INK_SECONDARY,
                                    linewidth=1.3))
        ax.annotate(f"{label}  ({np.hypot(dx, dy):.2f} in plane)",
                    xy=(dx * span, dy * span), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=plotting.INK_SECONDARY)

    ax.set_xlabel(f"PC2  ({float(axis_rows.loc[2, 'explained_variance_ratio']):.0%} of variance)")
    ax.set_ylabel(f"PC3  ({float(axis_rows.loc[3, 'explained_variance_ratio']):.0%} of variance)")
    ax.set_title("The 171-emotion circumplex lives in PC2-PC3, not PC1-PC2",
                 loc="left", pad=22)
    ax.text(0.0, 1.015,
            f"{run_name}; PC2 r_arousal {float(axis_rows.loc[2, 'r_arousal']):+.2f}, "
            f"PC3 r_valence {float(axis_rows.loc[3, 'r_valence']):+.2f}",
            transform=ax.transAxes, fontsize=8.5, color=plotting.INK_MUTED,
            ha="left", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def auroc_per_pc_per_end(per_pc_by_run: dict[str, list], labels: dict[str, str],
                         threshold: float = 0.75):
    """Grid of runs x {valence, arousal}: AUROC of each PC's + and - end."""
    keys = list(per_pc_by_run)
    fig, axes = plt.subplots(len(keys), 2, figsize=(9.8, 3.5 * len(keys)),
                             sharey=True, squeeze=False)
    for row, key in enumerate(keys):
        per_pc = per_pc_by_run[key]
        pcs = [int(p["pc"]) for p in per_pc]
        xs = np.arange(len(pcs))
        width = 0.36
        for col, axis in enumerate(["valence", "arousal"]):
            ax = axes[row][col]
            ax.set_axisbelow(True)
            ax.bar(xs - width / 2, [p[f"auroc_{axis}"] for p in per_pc], width,
                   color=plotting.SERIES[0], label="+ end", zorder=3)
            ax.bar(xs + width / 2, [p[f"auroc_{axis}_minus_end"] for p in per_pc], width,
                   color=plotting.SERIES[1], label="- end", zorder=3)
            ax.axhline(threshold, color=plotting.INK_PRIMARY, linewidth=1.1,
                       linestyle="--", zorder=4, label=f"{threshold} pass threshold")
            ax.axhline(0.5, color=plotting.INK_MUTED, linewidth=0.9, zorder=4,
                       label="0.5 chance")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"PC{p}" for p in pcs])
            ax.set_ylim(0, 1.16)
            ax.set_title(f"{labels[key]} -- {axis}", loc="left", fontsize=9.5,
                         color=plotting.INK_PRIMARY)
            if col == 0:
                ax.set_ylabel("AUROC over 14 anchors")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase 4: how well each end of each PC orders the anchors",
                 x=0.02, ha="left", fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return fig


def gate_a_tiers(summary: pd.DataFrame, labels: dict[str, str], pass_bar: float = 0.5):
    """Hit rate per naming tier, with the permutation null p95 drawn over each bar."""
    keys = [k for k in labels if (summary["run"] == k).any()]
    fig, axes = plt.subplots(1, len(keys), figsize=(10.2, 4.2), sharey=True, squeeze=False)
    tier_labels = ["T1\nEnglish\nlemma", "T2\nEnglish\nsynonym", "T3\nChinese\ntranslation"]
    for col, key in enumerate(keys):
        ax = axes[0][col]
        ax.set_axisbelow(True)
        sub = summary[summary["run"] == key]
        xs = np.arange(len(sub))
        ax.bar(xs, sub["rate (all)"], width=0.55, color=plotting.SERIES[0], zorder=3,
               label="observed hit rate")
        for x, (_, row) in zip(xs, sub.iterrows()):
            if not np.isnan(row["permutation null p95"]):
                ax.plot([x - 0.32, x + 0.32], [row["permutation null p95"]] * 2,
                        color=plotting.SERIES[1], linewidth=2.0, zorder=5,
                        solid_capstyle="butt",
                        label="permutation null, p95" if x == xs[1] else None)
            ax.annotate(f"{row['rate (all)']:.0%}", xy=(x, row["rate (all)"]),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9, color=plotting.INK_SECONDARY)
        ax.axhline(pass_bar, color=plotting.INK_PRIMARY, linewidth=1.1, linestyle="--",
                   zorder=4, label=f"GATE A pass bar ({pass_bar:.0%})")
        ax.set_xticks(xs)
        ax.set_xticklabels(tier_labels[:len(sub)], fontsize=8.5)
        ax.set_ylim(0, 0.82)
        ax.set_title(labels[key], loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        if col == 0:
            ax.set_ylabel("fraction of emotions whose top-12 names them")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("GATE A depends on what language you ask in", x=0.02, ha="left",
                 fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


def probe_rank_burial(axes_blobs: dict, probe_labels: dict[str, dict], vocab_size: int):
    """Rank of every English probe word in each a-priori axis readout, log scale.

    ``axes_blobs`` maps an axis name to its Phase 4 entry; ``probe_labels`` maps the same
    name to ``{word: +-1}`` for the dimension that axis is supposed to order.
    """
    names = list(axes_blobs)
    fig, axes = plt.subplots(1, len(names), figsize=(10.4, 4.4), squeeze=False)
    for col, name in enumerate(names):
        ax = axes[0][col]
        blob = axes_blobs[name]
        labels = probe_labels[name]["labels"]
        positive, negative = probe_labels[name]["poles"]
        ranks = {k: v for k, v in (blob.get("probe_ranks") or {}).items() if v is not None}
        order = sorted(ranks, key=lambda w: ranks[w])
        ys = np.arange(len(order))

        ax.set_axisbelow(True)
        ax.scatter([ranks[w] for w in order], ys,
                   c=[plotting.SERIES[0] if labels.get(w, 0) > 0 else plotting.SERIES[1]
                      for w in order], s=46, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1, vocab_size * 1.6)
        ax.axvline(12, color=plotting.INK_PRIMARY, linewidth=1.2, linestyle="--", zorder=4)
        ax.annotate("GATE A's top-12 window", xy=(12, -0.45), xytext=(6, 0),
                    textcoords="offset points", fontsize=8,
                    color=plotting.INK_SECONDARY, va="bottom")
        ax.axvline(vocab_size, color=plotting.BASELINE, linewidth=1.0, zorder=2)
        ax.annotate(f"vocab\n{vocab_size:,}", xy=(vocab_size, 0.2), xytext=(-4, 0),
                    textcoords="offset points", fontsize=8, ha="right",
                    color=plotting.INK_MUTED)
        ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
        ax.grid(False, axis="y")
        ax.invert_yaxis()
        ax.set_title(f"{name} axis  --  AUROC {probe_labels[name]['auroc']:.2f}\n"
                     f"blue = {positive}, orange = {negative}",
                     loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        ax.set_xlabel("rank of the English probe word in this direction's readout")
    fig.suptitle("The directions order the English words perfectly and never say them",
                 x=0.02, ha="left", fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def readout_concentration(frame: pd.DataFrame, labels: dict[str, str],
                          diffuse_at: float = 200.0):
    """How peaked each readout distribution is, as effective tokens on a log axis.

    ``effective_tokens`` is ``exp(entropy)`` of the full-vocabulary softmax, so 1 means
    the direction says one token and 151,936 means it says nothing. The shaded band marks
    the region where a top-12 list is an arbitrary slice of a flat distribution.
    """
    keys = [k for k in labels if (frame["run"] == k).any()]
    fig, axes = plt.subplots(1, len(keys), figsize=(10.6, 5.2), squeeze=False, sharex=True)
    for col, key in enumerate(keys):
        ax = axes[0][col]
        sub = frame[frame["run"] == key].sort_values("effective_tokens")
        ys = np.arange(len(sub))
        ax.set_axisbelow(True)
        ax.axvspan(diffuse_at, 2e5, color=plotting.DIVERGING_HIGH, alpha=0.07, zorder=0)
        for y, (_, row) in zip(ys, sub.iterrows()):
            colour = (plotting.INK_MUTED if row["whitespace end"]
                      else (plotting.SERIES[1] if row["effective_tokens"] >= diffuse_at
                            else plotting.SERIES[0]))
            ax.plot([1, row["effective_tokens"]], [y, y], color=colour,
                    linewidth=1.1, alpha=0.55, zorder=2)
            ax.plot([row["effective_tokens"]], [y], marker="o", color=colour,
                    markersize=7, zorder=3)
            ax.annotate(f"{row['top-12 mass']:.0%}",
                        xy=(row["effective_tokens"], y), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=plotting.INK_MUTED)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["direction"], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(1, 2e5)
        ax.axvline(diffuse_at, color=plotting.DIVERGING_HIGH, linewidth=1.0,
                   linestyle="--", zorder=4)
        ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
        ax.grid(False, axis="y")
        ax.set_title(labels[key], loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        ax.set_xlabel("effective tokens  =  exp(entropy of the full-vocab softmax)")
    axes[0][0].annotate("top-12 is an\narbitrary slice", xy=(diffuse_at * 1.5, 0.4),
                        fontsize=8, color=plotting.DIVERGING_HIGH, va="bottom")
    fig.suptitle("How much does each readout actually commit to?  "
                 "(grey = whitespace-dominated; label = share of mass in the top 12)",
                 x=0.02, ha="left", fontsize=10.5, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def remainder_bars(frames: dict[str, pd.DataFrame], controls: dict[str, dict],
                   labels: dict[str, str], k: int = 16):
    """``frac_remainder`` per emotion, with the random-direction control marked.

    The remainder is what the pursuit could *not* rebuild, so a **shorter** bar is a more
    reportable emotion. The control lines sit at ``1 - control`` for the same reason: an
    emotion only beats chance by reaching further left than the null does.
    """
    keys = list(frames)
    heights = [max(2.6, 0.19 * len(frames[k_])) if len(frames[k_]) <= 25 else 6.4
               for k_ in keys]
    fig, axes = plt.subplots(len(keys), 1, figsize=(9.4, sum(heights) + 1.2),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.28},
                             squeeze=False)
    for row, key in enumerate(keys):
        ax = axes[row][0]
        frame = frames[key].sort_values("frac_remainder")
        control = controls[key]
        ys = np.arange(len(frame))
        ax.set_axisbelow(True)
        ax.barh(ys, frame["frac_remainder"], color=plotting.SERIES[0], height=0.72
                if len(frame) <= 25 else 1.0, linewidth=0, zorder=3)
        for level, style, name in [
            (1 - control["mean"], "--", "control mean"),
            (1 - control["p95"], ":", "control p95"),
        ]:
            ax.axvline(level, color=plotting.DIVERGING_HIGH, linewidth=1.1,
                       linestyle=style, zorder=5, label=f"1 - {name}")
        left = min(frame["frac_remainder"].min(), 1 - control["p95"]) - 0.006
        ax.set_xlim(left, 1.0005)
        if len(frame) <= 25:
            ax.set_yticks(ys)
            ax.set_yticklabels(frame["emotion"], fontsize=8)
        else:
            # 172 bars are a pixel apart; only the extremes can carry a legible label,
            # and it goes in the empty space to the right of the bar tip.
            # Rows are one pixel apart, so only the two extremes can be labelled at
            # all. y is inverted below, so index 0 (lowest remainder = MOST
            # reportable) ends up at the top.
            ax.set_yticks([])
            for idx, role in ((0, "most reportable"), (len(frame) - 1, "least reportable")):
                ax.annotate(f"{frame['emotion'].iloc[idx]}  ({role})",
                            xy=(frame["frac_remainder"].iloc[idx], ys[idx]),
                            xytext=(8, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=8, color=plotting.INK_SECONDARY,
                            zorder=6)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
        ax.grid(False, axis="y")
        ax.set_title(f"{labels[key]}   (k = {k}; shorter bar = more reportable)",
                     loc="left", fontsize=9.5, color=plotting.INK_PRIMARY)
        ax.set_xlabel("frac_remainder  --  share of the vector the pursuit could not rebuild")
        if row == 0:
            ax.legend(loc="lower right", ncol=1, fontsize=8)
    fig.suptitle("Phase 6: the remainder dominates every emotion vector, "
                 "but reliably less than chance would give",
                 x=0.02, ha="left", fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def dictionary_validity(dictionary: dict, run_name: str, valid: bool):
    """Atom-validity check: what fraction of atoms return their own token."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    names = ["own token at rank 0\n(needs ~1.00)", "own token in top 10"]
    values = [dictionary.get("frac_rank_zero") or 0.0,
              dictionary.get("frac_in_top10") or 0.0]
    ax.set_axisbelow(True)
    bars = ax.barh(names, values,
                   color=plotting.SERIES[0] if valid else plotting.DIVERGING_HIGH,
                   height=0.5, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.1%}", xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9, color=plotting.INK_SECONDARY)
    ax.axvline(1.0, color=plotting.INK_PRIMARY, linewidth=1.1, linestyle="--")
    ax.annotate("a valid dictionary\nsits here", xy=(1.0, 0.5),
                xycoords=("data", "axes fraction"), xytext=(-6, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8.5,
                color=plotting.INK_SECONDARY)
    ax.set_xlim(0, 1.14)
    ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
    ax.grid(False, axis="y")
    plotting.finish(
        fig, ax,
        "Phase 6 precondition: does lensing an atom return its own token?",
        "fraction of probed atoms", "",
        subtitle=(f"{run_name}; {dictionary.get('checked')} atoms probed, "
                  f"median self-rank {dictionary.get('median_self_rank')}, "
                  f"max {dictionary.get('max_self_rank')}"),
    )
    return fig


def phase7_baselines(report: pd.DataFrame, families: dict, run_name: str):
    """Both Phase 7 baselines side by side: report score per prompt, behaviour per family.

    Left panel is scored 0-4 (the report rubric's range), right 0-3 (the behaviour
    rubric's). Bars pinned at a scale endpoint are drawn in red because a directional
    prediction cannot be tested against them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))

    ax = axes[0]
    ax.set_axisbelow(True)
    labels = [p if len(p) < 34 else p[:31] + "..." for p in report["prompt"]]
    ys = np.arange(len(report))
    ax.barh(ys, report["score"], color=plotting.DIVERGING_HIGH, height=0.6, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.2)
    ax.set_xticks(range(5))
    ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
    ax.grid(False, axis="y")
    ax.annotate("every prompt scores 0 -- the model denies having inner states",
                xy=(0.06, len(report) - 0.6), fontsize=8.5,
                color=plotting.DIVERGING_HIGH, va="center")
    ax.set_title("Report channel, unsteered  (judge rubric, 0-4)", loc="left",
                 fontsize=9.5, color=plotting.INK_PRIMARY)
    ax.set_xlabel("emotion-expression score")

    ax = axes[1]
    ax.set_axisbelow(True)
    names = list(families)
    means = [families[f]["mean"] for f in names]
    pinned = [not families[f]["has_range"] for f in names]
    ys = np.arange(len(names))
    ax.barh(ys, means, height=0.6, zorder=3,
            color=[plotting.DIVERGING_HIGH if p else plotting.SERIES[0] for p in pinned])
    for y, f in zip(ys, names):
        ax.annotate(f"n={families[f]['n']}, spread={families[f]['spread']:.2f}"
                    + ("  ← pinned" if not families[f]["has_range"] else ""),
                    xy=(max(means[y], 0.02), y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=plotting.DIVERGING_HIGH if pinned[y] else plotting.INK_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 5.6)
    ax.grid(True, axis="x", color=plotting.GRIDLINE, linewidth=0.7)
    ax.grid(False, axis="y")
    ax.set_title("Behaviour channel, unsteered  (mean per family)", loc="left",
                 fontsize=9.5, color=plotting.INK_PRIMARY)
    ax.set_xlabel("mean score")

    fig.suptitle(f"Phase 7: both channels at baseline, before any steering  ({run_name})",
                 x=0.02, ha="left", fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def phase7_report_probe(availability: dict, run_name: str):
    """Activation-level report availability: mean cosine to each emotion's lens tokens."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    names = list(availability)
    means = [availability[e]["mean_cosine"] for e in names]
    maxes = [availability[e]["max_cosine"] for e in names]
    xs = np.arange(len(names))
    ax.set_axisbelow(True)
    ax.bar(xs, means, width=0.5, color=plotting.SERIES[0], zorder=3, label="mean cosine")
    ax.plot(xs, maxes, linestyle="none", marker="D", markersize=6,
            color=plotting.SERIES[1], zorder=4, label="max over the 5 texts")
    ax.axhline(0, color=plotting.INK_PRIMARY, linewidth=1.0, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(-0.06, 0.12)
    ax.legend(loc="upper left", fontsize=8.5)
    plotting.finish(
        fig, ax,
        "Report availability at the activation level is also at floor",
        "emotion", "cosine(residual, emotion's lens tokens)",
        subtitle=f"{run_name}; unsteered baseline, 5 report texts per emotion",
    )
    return fig


def phase8_dose_response(summary: pd.DataFrame, nice: dict, run_name: str):
    """Signed effect against signed steering strength, per channel and emotion.

    A real effect is a monotone line through the origin that reverses sign with alpha and
    separates from the random control. The control is drawn dashed and grey precisely so
    that "is anything above the noise?" is answerable at a glance.
    """
    concepts = sorted(summary["concept"].unique())
    channels = ["behaviour", "report"]
    fig, axes = plt.subplots(len(channels), len(concepts),
                             figsize=(3.5 * len(concepts), 3.3 * len(channels)),
                             sharex=True, squeeze=False)
    colours = {"v": plotting.INK_PRIMARY, "v_reportable": plotting.SERIES[0],
               "v_remainder": plotting.SERIES[1], "v_random": plotting.INK_MUTED}
    for row, channel in enumerate(channels):
        for col, concept in enumerate(concepts):
            ax = axes[row][col]
            ax.set_axisbelow(True)
            sub = summary[(summary.concept == concept) & (summary.channel == channel)]
            for condition, group in sub.groupby("condition"):
                group = group.sort_values("alpha")
                is_random = condition == "v_random"
                ax.plot(group["alpha"], group["abs_z"],
                        color=colours.get(condition, plotting.SERIES[0]),
                        linestyle="--" if is_random else "-",
                        linewidth=2.0 if is_random else 1.6,
                        marker="o", markersize=4.5, zorder=4 if is_random else 3,
                        label=nice.get(condition, condition) if row == 0 and col == 0 else None)
            ax.axhline(0, color=plotting.BASELINE, linewidth=0.9)
            ax.axvline(0, color=plotting.BASELINE, linewidth=0.9)
            ax.set_ylim(-0.15, max(2.3, sub["abs_z"].max() * 1.15))
            if row == 0:
                ax.set_title(concept, loc="left", fontsize=10,
                             color=plotting.INK_PRIMARY)
            if col == 0:
                ax.set_ylabel(f"{channel}\n|z| (grid SD)")
            if row == len(channels) - 1:
                ax.set_xlabel("steering strength alpha")
            if channel == "report" and sub["abs_z"].max() == 0:
                ax.annotate("flat zero at every strength", xy=(0.5, 0.5),
                            xycoords="axes fraction", ha="center", fontsize=9,
                            color=plotting.DIVERGING_HIGH)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Phase 8: no dose-response in either channel, and random is not below "
                 f"the rest  ({run_name})",
                 x=0.02, ha="left", fontsize=11, color=plotting.INK_PRIMARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig
