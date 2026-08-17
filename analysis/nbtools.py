"""Machinery for `results_notebook.ipynb`: artefact loading, report accumulation, export.

The notebook is a *document*. Anything that is not prose, a number, or a decision about
how to present one belongs here instead, so that reading the notebook top to bottom
shows the argument rather than the plumbing.

Two rules this module exists to enforce
---------------------------------------
**Nothing raises.** Every artefact read goes through :class:`Runs`, which records what
was missing and returns ``None``. Every section body runs inside :meth:`Report.guard`,
which converts an exception into a visible note and continues. A notebook assembling
results from six pipeline phases across two runs will meet missing files; it should say
so and keep going, not die on cell 14.

**Every figure is explained.** :meth:`Report.fig` takes ``how_to_read`` and
``what_it_shows`` and refuses to let them default silently -- it prints a `TODO` marker
into the output if either is absent. A plot with no stated reading is a plot the reader
has to reverse-engineer.

The companion module :mod:`nbfigures` holds the plot builders, which return bare
matplotlib figures and know nothing about the report.
"""

from __future__ import annotations

import json
import os
import struct
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "find_root", "Runs", "Report",
    "fmt_cell", "df_to_md", "show_token",
    "char_script", "token_script", "is_cjk_token", "script_mix",
    "tiered_gate_a",
    "provenance_row", "phase1_summary", "phase2_summary", "phase3_variance_summary",
    "phase4_verdicts",
    "render_tokens", "pc_pole_table", "emotion_readout_table",
]

# --------------------------------------------------------------------------- #
# Locating things
# --------------------------------------------------------------------------- #

def find_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the repo root (the directory holding outputs/ and core/)."""
    start = Path(start or Path.cwd())
    for candidate in [start, *start.parents]:
        if (candidate / "outputs").is_dir() and (candidate / "core").is_dir():
            return candidate
    raise FileNotFoundError(
        f"no repo root above {start}: expected a directory containing both outputs/ and core/"
    )


# --------------------------------------------------------------------------- #
# Markdown rendering -- `tabulate` is not in the conda base env, so this is by hand
# --------------------------------------------------------------------------- #

def fmt_cell(value) -> str:
    """One dataframe cell as markdown text.

    Floats get fixed 3-decimal formatting rather than ``rstrip('0')``: an AUROC of
    exactly 1.0 must read ``1.000``, not ``1``, or a reader cannot tell a rounded value
    from an integer count. Pipes are escaped so a token containing ``|`` cannot break
    the table.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        magnitude = abs(float(value))
        if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e6):
            return f"{value:.2e}"
        return f"{value:,.1f}" if magnitude >= 1000 else f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def df_to_md(frame: pd.DataFrame, index: bool = False) -> str:
    data = frame.reset_index() if index else frame
    cols = [str(c).replace("|", "\\|") for c in data.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(fmt_cell(v) for v in row) + " |")
    return "\n".join(lines)


def show_token(tok) -> str:
    """A vocabulary token as visible markdown.

    Readouts are full of whitespace tokens. Printing ``' \\n'`` raw would put a line
    break inside a table cell and silently corrupt the table, so tokens are shown as
    their Python repr minus the quotes, inside backticks.
    """
    return "`" + repr(str(tok))[1:-1].replace("|", "\\|") + "`"


# --------------------------------------------------------------------------- #
# Script classification
# --------------------------------------------------------------------------- #

def char_script(ch: str) -> str | None:
    """``'CJK'`` / ``'Latin'`` / ``'other script'`` / ``None`` for a single character.

    ``None`` means the character carries no script information (punctuation, digits,
    whitespace) and should not vote. Kana and hangul are folded into ``CJK``: neither
    occurs in this data, but the bucket name is then about the writing system rather
    than about Chinese specifically.
    """
    code = ord(ch)
    if ((0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)
            or (0xF900 <= code <= 0xFAFF) or (0x20000 <= code <= 0x2FA1F)
            or (0x3040 <= code <= 0x30FF) or (0xAC00 <= code <= 0xD7AF)):
        return "CJK"
    if not ch.isalpha():
        return None
    if ch.isascii():
        return "Latin"
    try:
        return "Latin" if "LATIN" in unicodedata.name(ch) else "other script"
    except ValueError:
        return "other script"


def token_script(tok) -> str:
    """Script of a whole token. Mixed-script tokens count as CJK (biases CJK upward)."""
    scripts = {s for s in (char_script(c) for c in str(tok)) if s}
    for label in ("CJK", "Latin", "other script"):
        if label in scripts:
            return label
    return "punct / whitespace"


def is_cjk_token(tok) -> bool:
    return token_script(tok) == "CJK"


def script_mix(readouts: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Per-token script labels for the top-``top_k`` rows of a readout table."""
    sub = readouts[readouts["rank"] < top_k].copy()
    sub["script"] = sub["token"].map(token_script)
    return sub


# --------------------------------------------------------------------------- #
# Artefact access
# --------------------------------------------------------------------------- #

_ST_DTYPES = {"F32": np.float32, "F64": np.float64, "F16": np.float16,
              "I64": np.int64, "I32": np.int32}


class Runs:
    """Read-only access to the phase artefacts of several named runs.

    Every accessor returns ``None`` and records a note rather than raising, so a caller
    can check for ``None`` and a whole section can degrade to "this artefact is absent".
    """

    def __init__(self, root: Path, mapping: dict[str, str], designs: dict[str, str],
                 report: "Report | None" = None):
        self.root = Path(root)
        self.mapping = dict(mapping)
        self.designs = dict(designs)
        self.report = report

    @property
    def keys(self) -> list[str]:
        return list(self.mapping)

    def name(self, key: str) -> str:
        return self.mapping[key]

    def design(self, key: str) -> str:
        return self.designs.get(key, "")

    def phases(self, key: str) -> Path:
        return self.root / "outputs" / self.mapping[key] / "results" / "phases"

    def has(self, key: str, filename: str) -> bool:
        return (self.phases(key) / filename).exists()

    def _miss(self, what: str, detail: str = "") -> None:
        if self.report is not None:
            self.report.note_missing(what, detail)
        else:
            print("MISSING:", what, detail)

    def json(self, key: str, filename: str):
        path = self.phases(key) / filename
        if not path.exists():
            self._miss(f"{self.mapping[key]}/{filename}")
            return None
        with path.open() as handle:
            return json.load(handle)

    def csv(self, key: str, filename: str):
        path = self.phases(key) / filename
        if not path.exists():
            self._miss(f"{self.mapping[key]}/{filename}")
            return None
        return pd.read_csv(path)

    def safetensors(self, key: str, filename: str):
        """Minimal safetensors reader -- avoids a dependency for four tensors."""
        path = self.phases(key) / filename
        if not path.exists():
            self._miss(f"{self.mapping[key]}/{filename}")
            return None
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
            base = 8 + header_len
            out = {}
            for tensor_name, spec in header.items():
                if tensor_name == "__metadata__":
                    out["__metadata__"] = spec
                    continue
                start, end = spec["data_offsets"]
                handle.seek(base + start)
                out[tensor_name] = np.frombuffer(
                    handle.read(end - start), dtype=_ST_DTYPES[spec["dtype"]]
                ).reshape(spec["shape"])
        return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

class Report:
    """Accumulates markdown blocks in order, then writes ``RESULTS.md``.

    Anything shown to a notebook reader should go through here, because anything that
    does *not* is invisible to the export -- a plain markdown cell renders in Jupyter
    and then silently vanishes from the pasteable summary.
    """

    def __init__(self, root: Path, subdir: str = "analysis", figures: str = "figures"):
        self.root = Path(root)
        self.analysis = self.root / subdir
        self.fig_dir = self.analysis / figures
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.blocks: list[str] = []
        self.missing: list[str] = []
        self.figures: list[tuple[str, str]] = []
        self._fig_n = 0

    # -- display helpers -----------------------------------------------------
    @staticmethod
    def _show(markdown_text: str) -> None:
        try:
            from IPython.display import Markdown, display
            display(Markdown(markdown_text))
        except ImportError:  # pragma: no cover -- running outside IPython
            print(markdown_text)

    def md(self, text: str, show: bool = True) -> None:
        """Record a markdown block (and render it) so it reaches RESULTS.md."""
        self.blocks.append(text.strip("\n"))
        if show:
            self._show(text)

    def heading(self, text: str, level: int = 2) -> None:
        self.md(f"{'#' * level} {text}")

    def table(self, frame: pd.DataFrame, caption: str, index: bool = False,
              show: bool = True) -> None:
        self.blocks.append(f"**{caption}**\n\n" + df_to_md(frame, index=index))
        if show:
            self._show(f"**{caption}**")
            try:
                from IPython.display import display
                display(frame)
            except ImportError:  # pragma: no cover
                print(frame.to_string())

    # -- figures -------------------------------------------------------------
    def _fig_block(self, rel_path: str, caption: str, how_to_read: str | None,
                   what_it_shows: str | None, plain: str | None = None) -> None:
        self._fig_n += 1
        label = f"Figure {self._fig_n} -- {caption}"
        self.figures.append((caption, rel_path))

        # Three notes in increasing technicality: what question the picture answers,
        # how its marks encode the data, then what it turned out to say. The plain line
        # comes first so a reader who bounces off the axes still learns what was asked.
        head = [f"**{label}**"]
        head.append(f"*In plain terms.* {plain.strip()}" if plain
                    else "*In plain terms.* **TODO -- unexplained figure.**")
        head.append(f"*How to read it.* {how_to_read.strip()}" if how_to_read
                    else "*How to read it.* **TODO -- unexplained figure.**")
        self.md("\n\n".join(head))

        self.blocks.append(f"![{caption}]({rel_path})")
        try:
            from IPython.display import Image, display
            display(Image(filename=str(self.analysis / rel_path)))
        except ImportError:  # pragma: no cover
            pass

        if what_it_shows:
            self.md(f"*What it shows.* {what_it_shows.strip()}")
        else:
            self.md("*What it shows.* **TODO -- unexplained figure.**")

    def fig(self, figure, filename: str, caption: str,
            how_to_read: str | None = None, what_it_shows: str | None = None,
            plain: str | None = None) -> None:
        """Save a new figure, embed it, and require all three explanations around it."""
        from core import plotting
        path = plotting.save(figure, self.fig_dir / filename)
        self._fig_block(os.path.relpath(path, self.analysis), caption,
                        how_to_read, what_it_shows, plain)

    def existing_fig(self, path, caption: str, how_to_read: str | None = None,
                     what_it_shows: str | None = None, plain: str | None = None) -> None:
        """Embed a figure the pipeline already wrote, without regenerating it."""
        path = Path(path)
        if not path.exists():
            self.note_missing(f"figure {path.name}", str(path))
            self.md(f"> **Figure missing:** `{path}` -- {caption}")
            return
        self._fig_block(os.path.relpath(path, self.analysis), caption,
                        how_to_read, what_it_shows, plain)

    # -- degradation ---------------------------------------------------------
    def note_missing(self, what: str, detail: str = "") -> None:
        line = what + (f" -- {detail}" if detail else "")
        self.missing.append(line)
        print("MISSING:", line)

    @contextmanager
    def guard(self, what: str):
        """Run a section body; on any exception, say so in the output and continue."""
        try:
            yield
        except Exception as exc:  # noqa: BLE001 -- deliberate: degrade, never raise
            self.note_missing(what, f"{type(exc).__name__}: {exc}")
            self.md(f"> **Skipped -- {what}.** `{type(exc).__name__}: {exc}`")

    # -- export --------------------------------------------------------------
    def write(self, path: Path, header: str = "") -> Path:
        path = Path(path)
        index = "\n".join(f"* [{cap}]({rel})" for cap, rel in self.figures) or "* none"
        body = "\n\n".join(self.blocks)
        footer = "\n\n---\n\n## Figure index\n\n" + index + "\n"
        path.write_text(header + body + footer, encoding="utf-8")
        return path

    def unexplained_figures(self) -> int:
        """How many figures were emitted without both explanations. Should be 0."""
        return sum(b.count("TODO -- unexplained figure.") for b in self.blocks)


# --------------------------------------------------------------------------- #
# Per-phase field extraction
#
# These pull a flat summary row out of a phase's gate JSON. They are here rather than
# in the notebook because choosing *which* field to read is mechanical, while deciding
# what the number means is not -- the latter stays in the notebook prose.
# --------------------------------------------------------------------------- #

def provenance_row(runs: "Runs", key: str) -> dict:
    """Model / lens / target / dataset provenance for one run, as one flat column."""
    p0, p1 = runs.json(key, "phase0_gate.json"), runs.json(key, "phase1_gate.json")
    p2, p4 = runs.json(key, "phase2_gate.json"), runs.json(key, "phase4_gate.json")
    lens = (p0 or {}).get("lens") or (p4 or {}).get("lens") or {}
    resolved, fingerprint = (p2 or {}).get("resolved", {}), (p2 or {}).get("fingerprint", {})
    dataset, stimuli = (p1 or {}).get("dataset", {}), (p2 or {}).get("stimuli", {})
    gates = sorted(f.name.split("_")[0] for f in runs.phases(key).glob("phase*_gate.json"))
    return {
        "design": runs.design(key),
        "model": fingerprint.get("model_name"),
        "model_sha": fingerprint.get("model_sha") or resolved.get("model_sha"),
        "dtype / activation dtype":
            f"{fingerprint.get('dtype')} / {fingerprint.get('activation_dtype')}",
        "lens repo": lens.get("repo") or lens.get("lens_repo"),
        "lens subfolder": lens.get("subfolder"),
        "lens file": Path(lens.get("lens_file") or lens.get("path") or "").name or None,
        "checkpoint format": (p0 or {}).get("lens", {}).get(
            "checkpoint_format", "fit_checkpoint (from the 16 run)"),
        "d_model": lens.get("d_model") or fingerprint.get("hidden_size"),
        "lens prompts (checkpoint n_done)": lens.get("n_prompts"),
        "lens prompts (config.yaml claim)": lens.get("recorded_prompts_fitted"),
        "fitted blocks": "..".join(str(b) for b in lens.get("fitted_block_range", [])),
        "target block (block convention)": resolved.get("target_block"),
        "target block (hidden_state index)": resolved.get("target_hidden_state"),
        "target resolved from": resolved.get("target_resolved_from"),
        "pooling": f"{fingerprint.get('pooling')} "
                   f"(offset {fingerprint.get('token_offset')}, "
                   f"max_len {fingerprint.get('max_length')})",
        "R2 prefix": resolved.get("r2_prefix"),
        "dataset": dataset.get("dataset_id"),
        "dataset sha": dataset.get("dataset_sha"),
        "stimuli rows": stimuli.get("n_stimuli"),
        "stimuli sha256": (stimuli.get("sha256") or "")[:16] + "...",
        "emotion vectors (incl. neutral)": stimuli.get("n_groups"),
        "gate files present": ", ".join(gates) or "none",
    }


def phase1_summary(p1: dict) -> dict:
    """Stimulus design, topic matching and length-risk facts for one run."""
    dataset, topics = p1["dataset"], p1["topic_matching"]
    length, emotion_set = p1["length_fit"], p1["emotion_set"]
    per_emotion = {k: v for k, v in dataset["per_emotion_counts"].items() if k != "neutral"}
    return {
        "emotions in run": emotion_set["n_emotions"],
        "labelled anchors": emotion_set["n_anchors"],
        "stimulus rows": dataset["n_rows"],
        "emotional / neutral": f"{dataset['n_emotional']} / {dataset['n_neutral']}",
        "stories per emotion": f"{min(per_emotion.values())}..{max(per_emotion.values())}",
        "topics total": topics["n_topics_total"],
        "all emotions share topics": topics["all_emotions_same_topics"],
        "topics exactly matched": topics["exactly_matched"],
        "stories per (emotion, topic) cell": ", ".join(
            str(c) for c in topics["distinct_per_cell_counts"]),
        "split train/val/test": "/".join(
            str(dataset["per_split_counts"].get(s))
            for s in ("train", "validation", "test")),
        "words min / median / max":
            f"{length['words_min']} / {length['words_median']:.0f} / {length['words_max']}",
        "approx pooled tokens, min": round(length["approx_pooled_min"], 1),
        "tokens needed": length["tokens_needed"],
        "stimuli at length risk": length["n_at_risk"],
    }


def phase2_summary(frame: pd.DataFrame, p2: dict | None, threshold: float = 0.9) -> dict:
    """Split-half reliability summary. Neutral is excluded -- it is not an emotion."""
    summary = ((p2 or {}).get("split_half", {}) or {}).get("summary", {})
    emotional = frame[frame["emotion"] != "neutral"]
    below = emotional[emotional["cosine_centered"] < threshold]
    return {
        "emotions scored": int(summary.get("n_scored", len(frame))),
        "threshold": threshold,
        "mean centred cosine": float(emotional["cosine_centered"].mean()),
        "median centred cosine": float(emotional["cosine_centered"].median()),
        "min centred cosine": float(emotional["cosine_centered"].min()),
        "mean RAW cosine (inflated)": summary.get("mean_cosine_raw"),
        "n below threshold": len(below),
        "frac below threshold": len(below) / max(len(emotional), 1),
        "weakest 5": ", ".join(emotional.nsmallest(5, "cosine_centered")["emotion"]),
    }


def phase3_variance_summary(gate: dict) -> dict:
    """Variance spectrum, effective dimensionality and PC stability for one run."""
    pca, stability = gate["pca"], gate["pc_stability"]
    null = gate.get("null_band", {})
    ratios = pca["explained_variance_ratio"]
    return {
        "emotions in fit": pca["rank"] + 1,
        "PCA rank": pca["rank"],
        "PC1 variance": ratios[0], "PC2 variance": ratios[1], "PC3 variance": ratios[2],
        "top-2 cumulative": sum(ratios[:2]), "top-3 cumulative": sum(ratios[:3]),
        "participation ratio (effective dim)": pca["participation_ratio"],
        "isotropic null, per-PC p50": null.get("pc1", {}).get("p50"),
        "PCs stable of 10 (cos > 0.8)": stability["n_stable"],
        "PC stability: min cosine": stability["min_cosine"],
        "axes identified": stability["axes_identified"],
        "top-2 plane stable": stability["plane_stable"],
    }


def phase4_verdicts(p4: dict) -> pd.DataFrame:
    """One decision row per PC, with the pipeline's own verdict wording preserved.

    The verdict string is derived from the gate's flags rather than re-judged: a PC that
    the pipeline called MURKY is called MURKY here, and one it flagged exploratory keeps
    that qualifier.
    """
    rows = []
    for pc in p4["gate_b"]["per_pc"]:
        note = (pc.get("notes") or [""])[0]
        if pc.get("lexicalised"):
            verdict = f"reads as {pc['best_axis'].upper()}"
            if pc.get("exploratory"):
                verdict += " (exploratory)"
        elif note.startswith("MURKY"):
            verdict = "MURKY"
        elif pc.get("significant_uncorrected"):
            verdict = "not interpreted (fails family correction)"
        else:
            verdict = "MURKY"
        rows.append({
            "PC": int(pc["pc"]),
            "variance": pc["explained_variance_ratio"],
            "verdict": verdict,
            "best axis": pc["best_axis"],
            "AUROC (best end, best axis)": max(
                pc["auroc_valence"], pc["auroc_valence_minus_end"],
                pc["auroc_arousal"], pc["auroc_arousal_minus_end"]),
            "p (best axis)": pc["p_best_axis"],
            "alpha": pc["alpha"],
            "sig. at alpha": pc["p_best_axis"] < pc["alpha"],
            "PC stability (Phase 3)": pc["phase3_split_half_cosine"],
            "sign agrees with Phase 3": pc.get("sign_agrees_with_phase3"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Translated readout tables
#
# `phase4_readouts_translated.csv` is `phase4_readouts.csv` row-for-row, plus three
# columns: `script`, `token_english`, `gloss_confidence`. Coverage of non-Latin tokens is
# complete, which is why these tables use it in preference to the hand-written
# TOKEN_GLOSS in zh_en_glossary. That module is still the source for *matching* in
# `tiered_gate_a` -- the matching lists are pre-committed and must not be swapped for a
# different translation source after the fact.
# --------------------------------------------------------------------------- #

def render_tokens(sub: pd.DataFrame, top_k: int = 12, show_source: bool = False) -> str:
    """One direction's top-``top_k`` tokens as markdown, in English.

    Glossed tokens are rendered as their English gloss alone; Latin tokens are already
    English and pass through; whitespace and punctuation have no English form and are
    shown escaped. The source tokens are **not** printed -- they remain in
    ``phase4_readouts_translated.csv`` next to their glosses, which is where an audit of
    the translation itself belongs. Pass ``show_source=True`` to get them back inline.
    """
    parts = []
    for _, row in sub.sort_values("rank").head(top_k).iterrows():
        gloss = row.get("token_english")
        gloss = gloss.strip() if isinstance(gloss, str) and gloss.strip() else ""
        if not gloss:
            parts.append(show_token(row["token"]))    # Latin / whitespace / punctuation
        elif show_source:
            parts.append(f"**{gloss}** ⟨{row['token']}⟩")
        else:
            parts.append(f"**{gloss}**")
    return " · ".join(parts)


def gloss_map(runs: "Runs", filename: str = "phase4_readouts_translated.csv") -> dict:
    """``token -> English gloss`` pooled across every run's translated readouts.

    Phase 6's atom tokens are not covered by their own translation file, so they are
    glossed by lookup against this. Coverage is partial; callers should report it rather
    than let an unglossed token pass as if it had been checked.
    """
    mapping: dict[str, str] = {}
    for key in runs.keys:
        frame = runs.csv(key, filename)
        if frame is None or "token_english" not in frame.columns:
            continue
        for token, english in zip(frame["token"], frame["token_english"]):
            if isinstance(english, str) and english.strip():
                mapping.setdefault(str(token), english.strip())
    return mapping


def render_atom_tokens(tokens, mapping: dict) -> tuple[str, int]:
    """Phase 6 atom tokens in English where a gloss exists. Returns (markdown, n_missing).

    A token with no gloss and no Latin letters is printed raw and flagged, because
    silently dropping it would overstate how much of the atom list has been translated.
    """
    parts, missing = [], 0
    for token in tokens:
        text = str(token)
        gloss = mapping.get(text)
        if gloss:
            parts.append(f"**{gloss}**")
        elif any(c.isalpha() and c.isascii() for c in text):
            parts.append(show_token(text))            # already English
        elif not any(c.isalpha() for c in text):
            parts.append(show_token(text))            # punctuation / whitespace / symbols
        else:                                         # a word, in a script we cannot render
            parts.append(f"{show_token(text)} ⚠")
            missing += 1
    return " · ".join(parts), missing


def pc_pole_table(p4: dict, translated: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Every lensed principal component, both poles, with translated readouts.

    Includes the concentration statistics beside the tokens on purpose: a top-``top_k``
    list from a readout spread over 3,000 effective tokens is not the same kind of
    evidence as one from a readout spread over 7, and putting them in separate tables
    invites reading the first as if it were the second.
    """
    rows = []
    for pc in p4["gate_b"]["per_pc"]:
        note = (pc.get("notes") or [""])[0]
        if pc.get("lexicalised"):
            verdict = f"reads as {pc['best_axis']}"
            if pc.get("exploratory"):
                verdict += " (exploratory)"
        else:
            verdict = "MURKY" if note.startswith("MURKY") else "not interpreted"
        for end in ("plus", "minus"):
            blob = pc.get(end) or {}
            name = blob.get("name", f"{end}PC{pc['pc']}")
            sub = translated[translated["direction"] == name]
            effective = blob.get("effective_tokens", float("nan"))
            rows.append({
                "PC": int(pc["pc"]),
                "pole": "+" if end == "plus" else "-",
                "variance": pc["explained_variance_ratio"],
                "PC verdict": verdict,
                "AUROC valence": blob.get("auroc_valence"),
                "AUROC arousal": blob.get("auroc_arousal"),
                "top-1 prob": blob.get("top1_prob"),
                "effective tokens": effective,
                "top-12 mass": (float(sub["prob"].sum())
                                if sub["prob"].notna().any() else float("nan")),
                "interpretable?": ("top-12 is an arbitrary slice" if effective >= 200
                                   else "yes"),
                f"top-{top_k} tokens (translated)": render_tokens(sub, top_k),
            })
    return pd.DataFrame(rows)


def emotion_readout_table(p4: dict, translated: pd.DataFrame,
                          top_k: int = 12) -> pd.DataFrame:
    """Every lensed emotion vector with its translated readout.

    Emotion vectors were lensed at the positive pole only and had no per-token
    probabilities persisted, so there is no concentration column here -- the absence is
    the point, and section 5k records it.
    """
    gate_a = {row["emotion"]: row for row in p4["gate_a"]["rows"]}
    untokenizable = set(p4["gate_a"]["untokenizable"])
    sub_all = translated[translated["group"] == "gate_a_emotion_vector"]
    rows = []
    for emotion, sub in sub_all.groupby("direction"):
        record = gate_a.get(emotion, {})
        scripts = sub.head(top_k)["script"].value_counts(normalize=True)
        rank = record.get("own_word_rank")
        rows.append({
            "emotion": emotion,
            # Ranks and percentages are integers; letting them through as floats prints
            # an own-word rank of 335 as "335.000".
            "own-word rank": "--" if rank is None else f"{int(rank):,}",
            "GATE A": record.get("verdict") or (
                "unscorable (multi-token in English)" if emotion in untokenizable else ""),
            "% CJK": f"{float(scripts.get('cjk', 0.0)) * 100:.0f}%",
            "% Latin": f"{float(scripts.get('latin', 0.0)) * 100:.0f}%",
            f"top-{top_k} tokens (translated)": render_tokens(sub, top_k),
        })
    return pd.DataFrame(rows).sort_values("emotion").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The cross-lingual GATE A re-scoring
# --------------------------------------------------------------------------- #

def tiered_gate_a(readouts: pd.DataFrame, gate: dict, glossary,
                  n_permutations: int = 2000, seed: int = 0, top_k: int = 12):
    """Re-score GATE A under three definitions of "the emotion's name".

    All three tiers ask GATE A's own question -- *is it inside the top-``top_k``?* -- and
    vary only what counts as a name: the exact English lemma (tier 1, taken from the
    pipeline's recorded verdict rather than recomputed), an English near-synonym, or a
    Chinese translation.

    The permutation null is the load-bearing control. Tiers 2 and 3 give each emotion
    several candidate strings where tier 1 gives one lemma, so their hit rate must rise
    for uninteresting reasons. Shuffling *which list is scored against which emotion's
    readout*, while leaving every list's contents and length untouched, isolates exactly
    that: if the lists were merely generous, the shuffled rate would match the observed
    one.

    Returns ``(summary_frame, per_emotion_frame)``.
    """
    rng = np.random.default_rng(seed)
    ga = readouts[(readouts["rank"] < top_k)
                  & (readouts["group"] == "gate_a_emotion_vector")]
    tops = {emotion: [str(t) for t in sub.sort_values("rank")["token"]]
            for emotion, sub in ga.groupby("direction")}
    emotions = sorted(tops)
    pipeline = {row["emotion"]: row for row in gate["gate_a"]["rows"]}
    untokenizable = set(gate["gate_a"]["untokenizable"])

    per = pd.DataFrame([{
        "emotion": e,
        "single-token in English": e not in untokenizable,
        "T1 exact English lemma": pipeline.get(e, {}).get("verdict") == "HIT",
        "T2 English near-synonym": any(
            glossary.matches_en_synonym(t, e) for t in tops[e]),
        "T3 Chinese translation": any(
            glossary.matches_chinese(t, e) for t in tops[e]),
        "matched Chinese tokens": " ".join(
            t for t in tops[e] if glossary.matches_chinese(t, e)),
        "matched English tokens": " ".join(
            t.strip() for t in tops[e] if glossary.matches_en_synonym(t, e)),
    } for e in emotions])
    scorable = per[per["single-token in English"]]

    def hit(read_emotion: str, list_emotion: str, table: str) -> bool:
        match = (glossary.matches_chinese if table == "zh" else glossary.matches_en_synonym)
        return any(match(t, list_emotion) for t in tops[read_emotion])

    rows = []
    for tier, column, table in [
        ("T1  exact English lemma (= GATE A as specified)", "T1 exact English lemma", None),
        ("T2  English near-synonym", "T2 English near-synonym", "en"),
        ("T3  Chinese translation", "T3 Chinese translation", "zh"),
    ]:
        observed = float(per[column].mean())
        if table is None:
            mismatch = null_mean = null_p95 = p_value = float("nan")
        else:
            mismatch = float(np.mean([hit(a, b, table)
                                      for a in emotions for b in emotions if a != b]))
            draws = np.array([
                np.mean([hit(a, b, table)
                         for a, b in zip(emotions, rng.permutation(emotions))])
                for _ in range(n_permutations)
            ])
            null_mean = float(draws.mean())
            null_p95 = float(np.percentile(draws, 95))
            p_value = float((np.sum(draws >= observed) + 1) / (len(draws) + 1))
        rows.append({
            "tier": tier,
            "hits / all emotions": f"{int(per[column].sum())}/{len(per)}",
            "rate (all)": observed,
            "rate (English-single-token subset)": (
                float(scorable[column].mean()) if len(scorable) else float("nan")),
            "wrong-emotion list rate": mismatch,
            "permutation null mean": null_mean,
            "permutation null p95": null_p95,
            "p": p_value,
        })
    return pd.DataFrame(rows), per
