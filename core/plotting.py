"""Shared matplotlib styling for pipeline diagnostics and analysis notebooks.

Colour choices come from a validated reference palette, used unchanged:

* Categorical series use **slots 1-3 only** (blue / orange / aqua). Those three
  clear the all-pairs colour-vision-deficiency and normal-vision separation
  floors; slot 4 onwards does not, so anything needing more than three series
  gets faceted, aggregated, or drawn as unlabelled hairlines instead of a fourth
  hue. Hues are assigned in fixed order and never cycled.
* Magnitude uses the single-hue blue ramp; signed quantities (cosine similarity)
  use the blue<->red diverging pair with a **gray** midpoint, so "no relationship"
  reads as nothing rather than as a colour.
* Every figure this module produces is accompanied by a CSV of the same numbers,
  which is what satisfies the "table view" requirement for the two series colours
  that sit below 3:1 contrast on a light surface.
"""

from __future__ import annotations

from pathlib import Path

# Categorical slots, in fixed assignment order (light surface).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

# Single-hue sequential ramp (blue, light -> dark).
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b")

# Diverging poles + neutral midpoint.
DIVERGING_LOW = "#2a78d6"   # blue
DIVERGING_MID = "#f0efec"   # neutral gray
DIVERGING_HIGH = "#e34948"  # red

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FIGSIZE = (7.2, 4.2)
DPI = 150


def apply_style() -> None:
    """Set rcParams: recessive chrome, thin marks, no top/right spines."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": 500,  # numeric: "medium" is not a matplotlib weight name
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
    })


def diverging_cmap(name: str = "emotion_diverging"):
    """Blue -> neutral gray -> red colormap for signed quantities."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        name, [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH], N=256
    )


def sequential_cmap(name: str = "emotion_sequential"):
    """Single-hue blue ramp for unsigned magnitude."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL), N=256)


def label_line_end(ax, x, y, text: str, color: str, dx: float = 0.0) -> None:
    """Direct-label a series at its last point, in ink rather than series colour."""
    if len(x) == 0:
        return
    ax.annotate(
        text,
        xy=(x[-1], y[-1]),
        xytext=(6 + dx, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.plot([x[-1]], [y[-1]], marker="o", color=color, markersize=5, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)


def integer_xaxis(ax) -> None:
    """Layer indices are integers; never label them 0.25, 0.50, ..."""
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))


def finish(
    fig,
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    subtitle: str | None = None,
    integer_x: bool = False,
    right_margin: float = 0.0,
) -> None:
    """Apply titles and tighten layout.

    ``right_margin`` reserves room on the right for end-of-line direct labels.
    """
    if integer_x:
        integer_xaxis(ax)
    if right_margin:
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + (right - left) * right_margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if subtitle:
        ax.set_title(title, loc="left", pad=16)
        ax.text(
            0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=INK_MUTED, ha="left", va="bottom",
        )
    else:
        ax.set_title(title, loc="left", pad=8)
    fig.tight_layout()


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
