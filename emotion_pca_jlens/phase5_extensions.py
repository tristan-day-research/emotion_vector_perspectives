"""Phase 5 (GATES): the three structural extensions, each its own subcommand.

Only worth running if Phases 0-4 came out clean. Each subcommand ends at its own
gate; none of them chains into another.

``layer-sweep`` -- where in the stack does the circumplex live?
    Repeats Phase 2's averaging and Phase 3's PCA at many blocks, from the same
    stored activations. Free in compute because Phase 2 stored every hidden state,
    which is the whole reason it did. Reports variance explained, split-half PC
    stability and cross-fit circumplex alignment against depth, so "the structure
    emerges and dissolves" becomes a curve rather than an impression. Optionally
    lenses PC1/PC2 at each block (``--lens``, needs the GPU).

    The payoff is retroactive: if the alignment peaks somewhere other than the
    block Phases 2-4 used, Phases 3 and 4 can be re-run there for nothing. Choosing
    the target block by *this* curve and then reporting the circumplex at that block
    would be selection on the outcome, so the gate says which block it is and leaves
    the decision to a human who knows that.

``perspective`` -- is "who is feeling it" a separate axis?
    Re-frames a subset of stimuli as happening to *you* versus to *them*, extracts
    both framings, and tests whether the self-minus-other difference is roughly
    orthogonal to the emotion axes -- and what it lenses to. This is the only
    subcommand that collects activations, into its own run directory and R2 prefix,
    because different input text means different activations.

``within-emotion`` -- the contrast that justifies the cross-emotion design.
    PCA *inside* one emotion, over individual stimuli rather than centroids. If the
    top axes there are topic and scenario rather than affect, that is the argument
    for why cross-emotion PCA was the right design. Quantified by how much of each
    within-emotion PC's variance is between-topic, against a label-shuffle null,
    plus how far those axes sit from Phase 3's affect plane.

Two things all three share
--------------------------
**They read the activation chunks, unlike Phase 3.** Phase 3 deliberately needs
only two small files; these need the pooled activations themselves, which live in
R2 by default. ``ActivationStore`` raises with the exact ``r2 pull`` command when
they are absent, and that is the expected first failure on a fresh machine.

**"Roughly orthogonal" needs a baseline.** In 5,120 dimensions two random
directions have |cos| about ``1/sqrt(d)`` = 0.014, so a cosine of 0.05 *looks*
orthogonal while being nearly four times chance. Every orthogonality claim here is
reported against that baseline rather than against zero, because high-dimensional
intuition runs the wrong way.

``remove_neutral_pcs`` lives here
---------------------------------
Phase 3 refuses the flag and points at this phase, because the neutral subspace is
an SVD of the neutral *stories* and Phase 3 only has their centroid. The sweep and
the within-emotion contrast have the chunks, so ``--set remove_neutral_pcs=true``
works in both, reusing ``core.directions.fit_neutral_subspace`` so it means exactly
what it means in the mean-difference pipeline. It is a robustness check, never a
default: projecting off neutral-story structure could remove the very cross-emotion
axes this experiment is looking for.

Usage::

    python run.py phase5 layer-sweep                     # CPU, needs the chunks
    python run.py phase5 layer-sweep --lens              # + lens PC1/PC2 per block
    python run.py phase5 layer-sweep --set remove_neutral_pcs=true
    python run.py phase5 within-emotion                  # CPU, needs the chunks
    python run.py phase5 perspective                     # GPU, extracts
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from core import activation_store, env_file, jlens_lens, model_utils, paths, provenance
from core.activation_store import ActivationStore, human_bytes, init_or_check_manifest
from core.directions import fit_neutral_subspace, project_out
from core.seeds import rng_for, set_global_seeds

# Phase 5 is re-analysis, so it reuses the earlier phases' machinery rather than
# restating it: the PCA and subspace comparison from Phase 3, the extraction loop and
# topic halving from Phase 2, the lens readout from Phase 4. A second copy of any of
# them would be the thing that drifts, and a layer sweep whose PCA differed from
# Phase 3's would not be a sweep of Phase 3.
from emotion_pca_jlens.pca_jlens_config import (
    PCAJLensConfig,
    load_config,
    resolve_block_spec,
)
from emotion_pca_jlens.phase1_stimuli import (
    DEFAULT_CIRCUMPLEX_SET,
    NEUTRAL_QUADRANT,
    QUADRANT_ORDER,
)
from emotion_pca_jlens.phase2_vectors import (
    VectorAccumulator,
    decide_r2,
    extract,
    read_stimuli,
    stimuli_fingerprint,
    topic_halves,
)
from emotion_pca_jlens.phase3_pca import (
    _contrast_axes,
    fit_pca,
    principal_angle_cosines,
)
from emotion_pca_jlens.phase4_lens_pcs import (
    build_probes,
    read_direction,
    read_pcs,
    resolve_lens,
)

RULE = "=" * 78
THIN = "-" * 78

#: Shuffles behind each between-topic variance p-value in the within-emotion gate.
TOPIC_PERMUTATIONS = 1000


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--config-json", type=Path, default=None, help="JSON file of config overrides"
    )
    shared.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field; repeatable",
    )

    p = argparse.ArgumentParser(
        description="Phase 5: optional structural extensions, one subcommand each.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subs = p.add_subparsers(dest="extension", required=True)

    sweep = subs.add_parser(
        "layer-sweep", parents=[shared],
        help="repeat the Phase 2 averaging + Phase 3 PCA at many blocks (CPU)",
    )
    sweep.add_argument(
        "--lens", action="store_true",
        help="also read PC1/PC2 out through the lens at every swept block "
             "(loads the model and the lens)",
    )
    sweep.add_argument(
        "--dry-run", action="store_true",
        help="report which blocks and how much activation data would be read; "
             "reads nothing",
    )

    persp = subs.add_parser(
        "perspective", parents=[shared],
        help="extract a self/other re-framing and test for a perspective axis (GPU)",
    )
    persp.add_argument(
        "--dry-run", action="store_true",
        help="verify the framings tokenize to equal length and estimate storage; "
             "never loads model weights",
    )
    persp.add_argument(
        "--overwrite", action="store_true",
        help="delete an existing incompatible perspective activations directory",
    )

    within = subs.add_parser(
        "within-emotion", parents=[shared],
        help="PCA inside single emotions, as the contrast to the cross-emotion "
             "design (CPU)",
    )
    within.add_argument(
        "--dry-run", action="store_true", help="report the plan; reads nothing"
    )
    return p


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def open_store(config: PCAJLensConfig) -> ActivationStore:
    """Open Phase 2's activation store, or say how to get the chunks.

    Unlike Phase 3, these extensions need the pooled activations themselves, and
    ``delete_local_after_sync=True`` is the default -- so on a fresh machine the
    index parquets are present and the tensors are not. That is the expected first
    failure here, not a bug, and it has a one-line fix.
    """
    try:
        return ActivationStore(config.activations_dir)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Phase 5 re-reads Phase 2's pooled activations, which Phase 3 and 4 never\n"
            "needed. If this run was extracted with delete_local_after_sync=True (the\n"
            "default) the tensors are in R2 only:\n\n"
            f"  python run.py r2 pull {config.activations_dir} "
            f"--prefix {config.resolved_r2_prefix()}\n"
        ) from exc


def chance_cosine(d_model: int, subspace_dim: int = 1) -> float:
    """Expected |cos| between a random direction and a random ``subspace_dim``-space.

    ``sqrt(k/d)``: for a random unit vector, ``E[cos^2]`` against a k-dimensional
    subspace is ``k/d``, so the familiar ``1/sqrt(d)`` is only the k=1 case. Getting
    this wrong understates the baseline by ``sqrt(k)`` -- at d=5120 a plane's chance
    cosine is 0.0198, not 0.0140, which is the difference between reporting a cosine
    as "1.4x chance" and "2x chance". Printed beside every orthogonality claim
    because in 5,120 dimensions a cosine of 0.05 reads as orthogonal to human
    intuition while being several times chance.
    """
    return float(np.sqrt(max(subspace_dim, 1) / max(d_model, 1)))


@dataclass(frozen=True)
class BlockMeans:
    """Per-emotion means at one hidden state, plus the two topic halves."""

    emotions: list[str]
    full: np.ndarray            # (n_emotions, d_model)
    halves: np.ndarray          # (n_emotions, 2, d_model)
    counts: np.ndarray          # (n_emotions, 2)


def emotion_means_at(
    store: ActivationStore,
    rows: pd.DataFrame,
    hidden_state: int,
    emotions: list[str],
    half_of_topic: dict[int, int],
) -> BlockMeans:
    """Average one stored hidden state per (emotion, topic half).

    The same quantity Phase 2 accumulated during extraction, recomputed from the
    chunks. Recomputing rather than reusing Phase 2's partials is what lets this
    happen at *any* block, which is the point of the sweep -- and the arithmetic is
    identical, so the target block reproduces Phase 2's saved vectors.
    """
    index = {emotion: i for i, emotion in enumerate(emotions)}
    activations = store.load_layer(hidden_state, rows)
    sums = np.zeros((len(emotions), 2, activations.shape[1]), dtype=np.float64)
    counts = np.zeros((len(emotions), 2), dtype=np.int64)
    row_ids = np.asarray([index[e] for e in rows["emotion"]], dtype=np.int64)
    halves = np.asarray([half_of_topic[int(t)] for t in rows["topic_id"]], dtype=np.int64)
    np.add.at(sums, (row_ids, halves), activations.astype(np.float64))
    np.add.at(counts, (row_ids, halves), 1)

    totals = counts.sum(axis=1)
    full = np.zeros((len(emotions), activations.shape[1]), dtype=np.float64)
    have = totals > 0
    full[have] = sums.sum(axis=1)[have] / totals[have][:, None]
    per_half = np.zeros_like(sums)
    for h in (0, 1):
        ok = counts[:, h] > 0
        per_half[ok, h] = sums[ok, h] / counts[ok, h][:, None]
    return BlockMeans(emotions=emotions, full=full, halves=per_half, counts=counts)


def neutral_basis(
    store: ActivationStore, hidden_state: int, threshold: float
) -> tuple[np.ndarray | None, dict]:
    """Nuisance subspace from the neutral *stories* at one hidden state.

    What Phase 3 could not do: it has the neutral centroid, not the ~400 neutral
    stories an SVD needs. Delegated to ``core.directions.fit_neutral_subspace`` so
    the projection means exactly what it means in the mean-difference pipeline.
    """
    neutral = store.subset(source="neutral")
    if len(neutral) < 2:
        return None, {"available": False, "reason": f"only {len(neutral)} neutral rows"}
    activations = store.load_layer(hidden_state, neutral)
    subspace = fit_neutral_subspace(activations, threshold)
    return subspace.components, {
        "available": True,
        "n_neutral_stories": int(len(neutral)),
        "n_pcs_removed": int(subspace.components.shape[0]),
        "variance_threshold": threshold,
    }


def crossfit_plane_cosine(
    means: BlockMeans, labels: dict[str, dict], fit_rows: np.ndarray, mean_center: bool
) -> float | None:
    """Cross-fit alignment of the PC1-PC2 plane with the a priori plane.

    Axes from one topic half, PCs from the other, exactly as Phase 3 does it: the
    within-sample version is optimistically biased because both sides are group means
    of the same matrix, and at a block where the structure is weak that bias is
    largest -- which would put the sweep's peak in the wrong place.
    """
    names = [e for i, e in enumerate(means.emotions) if fit_rows[i]]
    valence = np.asarray([float(labels.get(e, {}).get("valence", 0) or 0) for e in names])
    arousal = np.asarray([float(labels.get(e, {}).get("arousal", 0) or 0) for e in names])
    anchors = np.asarray([
        labels.get(e, {}).get("quadrant") in QUADRANT_ORDER for e in names
    ])
    scores: list[float] = []
    for axis_half, pc_half in ((0, 1), (1, 0)):
        pcs = fit_pca(means.halves[fit_rows, pc_half], names, mean_center=mean_center)
        if pcs.rank < 2:
            return None
        source = means.halves[fit_rows, axis_half]
        centred = source - source.mean(axis=0) if mean_center else source
        axes, _ = _contrast_axes(centred, valence, arousal, anchors)
        if axes is None:
            return None
        plane = principal_angle_cosines(
            pcs.components[:2], np.vstack([axes["valence"], axes["arousal"]])
        )
        scores.append(float(np.mean(plane)))
    return float(np.mean(scores))


def half_plane_stability(
    means: BlockMeans, fit_rows: np.ndarray, mean_center: bool
) -> float | None:
    """Minimum principal-angle cosine between the top-2 planes of the two halves."""
    names = [e for i, e in enumerate(means.emotions) if fit_rows[i]]
    planes = []
    for h in (0, 1):
        pcs = fit_pca(means.halves[fit_rows, h], names, mean_center=mean_center)
        if pcs.rank < 2:
            return None
        planes.append(pcs.components[:2])
    return float(principal_angle_cosines(*planes).min())


# --------------------------------------------------------------------------- #
# (a) Layer sweep
# --------------------------------------------------------------------------- #

def sweep(
    config: PCAJLensConfig,
    store: ActivationStore,
    blocks: list[int],
    emotions: list[str],
    labels: dict[str, dict],
    fit_rows: np.ndarray,
    half_of_topic: dict[int, int],
    rows: pd.DataFrame,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """One PCA per block. Returns ``(rows, {block: components})``."""
    results: list[dict] = []
    components: dict[int, np.ndarray] = {}
    names = [e for i, e in enumerate(emotions) if fit_rows[i]]
    neutral_report: dict = {}

    for i, block in enumerate(blocks):
        hidden_state = jlens_lens.hidden_state_index(block)
        means = emotion_means_at(store, rows, hidden_state, emotions, half_of_topic)

        if config.remove_neutral_pcs:
            basis, neutral_report = neutral_basis(
                store, hidden_state, config.neutral_variance_threshold
            )
            if basis is not None:
                means = BlockMeans(
                    emotions=means.emotions,
                    full=project_out(means.full, basis),
                    halves=np.stack(
                        [project_out(means.halves[:, h], basis) for h in (0, 1)], axis=1
                    ),
                    counts=means.counts,
                )

        pca = fit_pca(means.full[fit_rows], names, mean_center=config.mean_center)
        components[block] = pca.components
        norms = np.linalg.norm(means.full[fit_rows], axis=1)
        results.append({
            "block": block,
            "hidden_state": hidden_state,
            "depth_fraction": (block + 1) / max(store_n_layers(store), 1),
            "rank": pca.rank,
            "pc1": float(pca.explained_variance_ratio[0]),
            "top2": float(pca.cumulative_ratio[min(1, pca.rank - 1)]),
            "participation_ratio": pca.participation_ratio,
            "mean_norm": float(norms.mean()),
            "plane_stability": half_plane_stability(means, fit_rows, config.mean_center),
            "crossfit_alignment": crossfit_plane_cosine(
                means, labels, fit_rows, config.mean_center
            ),
            "n_neutral_pcs_removed": neutral_report.get("n_pcs_removed"),
        })
        row = results[-1]
        print(f"  block {block:>3} (hs {hidden_state:>3})  |v| {row['mean_norm']:>8.1f}  "
              f"PC1 {row['pc1']:>6.1%}  top2 {row['top2']:>6.1%}  "
              f"plane {_fmt(row['plane_stability'])}  "
              f"circumplex {_fmt(row['crossfit_alignment'])}"
              f"   [{i + 1}/{len(blocks)}]", flush=True)
    return results, components


def store_n_layers(store: ActivationStore) -> int:
    """Block count implied by the stored hidden states (``n_hidden_states - 1``)."""
    return max(store.layers) if store.layers else 0


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def plane_drift(components: dict[int, np.ndarray], reference: int) -> list[dict]:
    """How far each block's top-2 plane sits from the reference block's.

    A circumplex that is the *same* plane throughout the middle of the stack is a
    different claim from one that rotates block by block, and the variance table
    cannot tell them apart.
    """
    if reference not in components or components[reference].shape[0] < 2:
        return []
    anchor = components[reference][:2]
    out = []
    for block, comps in sorted(components.items()):
        if comps.shape[0] < 2:
            continue
        cosines = principal_angle_cosines(comps[:2], anchor)
        out.append({
            "block": block,
            "plane_cosine_vs_reference": float(np.mean(cosines)),
            "worst_angle_deg": float(np.degrees(np.arccos(cosines.min()))),
        })
    return out


def plot_sweep(out_dir: Path, rows: list[dict], target_block: int | None) -> Path:
    """Depth curves: the shape of "emerges and dissolves", which no table shows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core import plotting

    plotting.apply_style()
    frame = pd.DataFrame(rows).sort_values("block")
    fig, ax = plt.subplots()
    series = [
        ("top2", "PC1+PC2 variance", plotting.SERIES[0]),
        ("plane_stability", "plane stability (split-half)", plotting.SERIES[1]),
        ("crossfit_alignment", "circumplex alignment (cross-fit)", plotting.SERIES[2]),
    ]
    labels = []
    for column, label, colour in series:
        values = frame[column].astype(float)
        if values.isna().all():
            continue
        ax.plot(frame["block"], values, color=colour, marker="o", markersize=3.5,
                label=label)
        labels.append((frame["block"].to_numpy(), values.to_numpy(), label, colour))
    plotting.label_line_ends(ax, labels)
    if target_block is not None:
        ax.axvline(target_block, color=plotting.INK_MUTED, linestyle=(0, (4, 3)),
                   linewidth=1)
        ax.annotate("Phase 2-4 target", xy=(target_block, 0.02),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8,
                    color=plotting.INK_MUTED)
    ax.set_ylim(0, 1.02)
    plotting.finish(
        fig, ax,
        title="Where in the stack the circumplex lives",
        subtitle="one PCA per residual block, from the activations Phase 2 already stored",
        xlabel="residual block (jlens convention)", ylabel="fraction / cosine",
        integer_x=True, right_margin=0.30,
    )
    return plotting.save(fig, out_dir / "phase5_layer_sweep.png")


def cmd_layer_sweep(config: PCAJLensConfig, args) -> int:
    out_dir = config.phase_dir / "phase5_layer_sweep"
    print(RULE)
    print(f"PHASE 5a GATE -- layer sweep   run '{config.run_name}'")
    print(RULE)
    print("Repeats Phase 2's averaging and Phase 3's PCA at many blocks, from the")
    print("activations Phase 2 already stored. Free in compute, which is why Phase 2")
    print("kept every hidden state rather than only the target.")
    print()

    axes = read_pcs(config.pcs_path, config.pcs_meta_path)
    arch_layers = axes.n_layers or 0
    fitted = list(range(jlens_lens.max_lens_block(arch_layers) + 1)) if arch_layers else []
    if not fitted:
        raise SystemExit(
            "Phase 3's sidecar does not record the model's layer count, so the "
            "lens-covered\nblock range is unknown. Re-run Phase 3."
        )
    blocks = resolve_block_spec(config.sweep_blocks, fitted)
    emotions = axes.emotions
    labels = axes.labels
    fit_rows = np.asarray([e in axes.fit_emotions for e in emotions])

    print(f"reference PCs   : {axes.pcs_path.name} at "
          f"{jlens_lens.describe_block(axes.target_block, arch_layers)}")
    print(f"blocks to sweep : {len(blocks)} of {len(fitted)} lens-covered "
          f"({blocks[0]}..{blocks[-1]})")
    print(f"  spec          : sweep_blocks={config.sweep_blocks!r}")
    print(f"emotions        : {int(fit_rows.sum())} fitted, "
          f"{len(emotions) - int(fit_rows.sum())} projected only")
    print(f"neutral-PC removal: {config.remove_neutral_pcs}"
          + ("  (an SVD of the neutral stories per block -- the check Phase 3 "
             "could not run)" if config.remove_neutral_pcs else ""))

    if args.dry_run:
        print()
        print(RULE)
        print("--dry-run: nothing read. This sweep would load one hidden state per block")
        print(f"for every stored stimulus, {len(blocks)} passes over the chunks.")
        print(RULE)
        return 0

    store = open_store(config)
    rows = store.index[store.index["emotion"].isin(emotions)].reset_index(drop=True)
    missing = [b for b in blocks if jlens_lens.hidden_state_index(b) not in store.layers]
    if missing:
        raise SystemExit(
            f"blocks {missing} map to hidden states not in the store "
            f"(stored: {store.layers[:4]}..{store.layers[-4:]}).\n"
            "Narrow sweep_blocks, or re-extract with layer_spec=all."
        )
    half_of_topic, half_report = topic_halves(rows, config.seed)
    print(f"stimuli         : {len(rows):,} across {half_report['n_topics']} topics")
    print()
    print(THIN)
    t0 = time.time()
    results, components = sweep(
        config, store, blocks, emotions, labels, fit_rows, half_of_topic, rows
    )
    print(THIN)
    print(f"  {len(blocks)} blocks in {time.time() - t0:.0f}s")

    drift = plane_drift(components, axes.target_block)
    lens_rows: list[dict] = []
    if args.lens:
        lens_rows = sweep_lens_readouts(config, axes, components, blocks)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "phase5_layer_sweep.csv", index=False)
    if drift:
        pd.DataFrame(drift).to_csv(out_dir / "phase5_plane_drift.csv", index=False)
    if lens_rows:
        pd.DataFrame(lens_rows).to_csv(out_dir / "phase5_sweep_readouts.csv", index=False)
    figure = plot_sweep(out_dir, results, axes.target_block) if config.make_plots else None

    print_sweep_gate(results, drift, axes, lens_rows)

    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 5a LAYER SWEEP -- {config.run_name}",
        sections={
            "run": {"stage": "phase5_extensions", "extension": "layer-sweep",
                    "run_name": config.run_name, "output_dir": str(out_dir)},
            "config": config.to_dict(),
            "source": {"pcs": str(axes.pcs_path),
                       "reference_block": axes.target_block,
                       "activations": str(config.activations_dir)},
            "blocks": blocks,
            "sweep": results,
            "plane_drift": drift,
            "lens_readouts": lens_rows,
            "artifacts": {"csv": str(out_dir / "phase5_layer_sweep.csv"),
                          "figure": str(figure) if figure else None},
        },
        txt_name="phase5_layer_sweep.txt", json_name="phase5_layer_sweep.json",
    )
    print(f"\n  records : {txt_path}\n            {json_path}")
    if figure:
        print(f"  figure  : {figure}")
    print()
    print("STOPPING at the Phase 5a gate. Nothing was re-run at another block.")
    print(RULE)
    return 0


def sweep_lens_readouts(
    config: PCAJLensConfig, axes, components: dict[int, np.ndarray], blocks: list[int]
) -> list[dict]:
    """Lens PC1/PC2 at every swept block, so depth shows up lexically too."""
    print()
    print(RULE)
    print("Lensing PC1/PC2 at every swept block (--lens)")
    print(RULE)
    cache_dir = paths.hf_cache_dir()
    lens_path, lens_report = resolve_lens(config, cache_dir)
    tokenizer = model_utils.load_tokenizer(
        config.model_name, config.model_revision, cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    hf_model = model_utils.load_model(
        config.model_name, revision=config.model_revision, cache_dir=cache_dir,
        dtype=config.dtype, device_map=config.device_map,
        quantization=config.quantization,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    readout = jlens_lens.LensReadout.build(hf_model, tokenizer, lens_path)
    probes = build_probes(axes)
    rng = rng_for(config.seed, "phase5_sweep_lens")

    rows: list[dict] = []
    for block in blocks:
        comps = components.get(block)
        if comps is None or comps.shape[0] < 2:
            continue
        print(f"\n  block {block}")
        for pc in range(2):
            result = read_direction(
                readout, comps[pc], block, f"+PC{pc + 1}",
                probes["words"], probes["valence"], probes["arousal"], config.topk, rng,
            )
            rows.append({
                "block": block, "pc": pc + 1, "lens": lens_report["source"],
                "tokens": " ".join(result.token_strings(8)),
                "auroc_valence": result.auroc_valence,
                "auroc_arousal": result.auroc_arousal,
                "p_valence": result.p_valence, "p_arousal": result.p_arousal,
            })
            print(f"    +PC{pc + 1}  " + " ".join(repr(t) for t in result.token_strings(8)))
            print(f"          AUROC valence {result.auroc_valence:5.2f} "
                  f"arousal {result.auroc_arousal:5.2f}")
    return rows


def print_sweep_gate(
    results: list[dict], drift: list[dict], axes, lens_rows: list[dict],
) -> None:
    print()
    print(RULE)
    print("GATE  Where does the circumplex live?")
    print(RULE)
    scored = [r for r in results if r["crossfit_alignment"] is not None]
    if not scored:
        print("  no block could be scored for circumplex alignment (no labelled anchors).")
        return
    best = max(scored, key=lambda r: r["crossfit_alignment"])
    target = next((r for r in results if r["block"] == axes.target_block), None)

    print(f"{'block':>6}{'depth':>8}{'|v|':>10}{'PC1':>8}{'top2':>8}"
          f"{'plane':>9}{'circumplex':>12}")
    print(THIN)
    for row in results:
        mark = ""
        if row["block"] == axes.target_block:
            mark += "  <- Phase 2-4 target"
        if row is best:
            mark += "  <- peak alignment"
        print(f"{row['block']:>6}{row['depth_fraction']:>7.0%}{row['mean_norm']:>10.1f}"
              f"{row['pc1']:>8.1%}{row['top2']:>8.1%}"
              f"{_fmt(row['plane_stability']):>9}"
              f"{_fmt(row['crossfit_alignment']):>12}{mark}")
    print(THIN)
    print(f"  peak circumplex alignment: {best['crossfit_alignment']:.3f} at block "
          f"{best['block']} ({best['depth_fraction']:.0%} of depth)")
    if target is not None and target["crossfit_alignment"] is not None:
        gap = best["crossfit_alignment"] - target["crossfit_alignment"]
        print(f"  Phase 2-4 target block {axes.target_block}: "
              f"{target['crossfit_alignment']:.3f}  (peak is {gap:+.3f})")
        if gap > 0.05:
            print()
            print("  The target block is not the best one. Phases 3 and 4 can be re-run at")
            print(f"  block {best['block']} for nothing, because Phase 2 stored it:")
            print(f"    python run.py phase2 --set target_block={best['block']}   "
                  "# re-derives vectors, no forward passes")
            print("    python run.py phase3 && python run.py phase4")
            print()
            print("  But read this first. Picking the block by this curve and then")
            print("  reporting the circumplex there is selection on the outcome -- the")
            print("  alignment at the chosen block is no longer an unbiased estimate.")
            print("  Either report both blocks, or treat the peak as the pre-registered")
            print("  choice for a fresh stimulus set.")
    if drift:
        worst = max(drift, key=lambda r: r["worst_angle_deg"])
        near = [r for r in drift if r["plane_cosine_vs_reference"] > 0.9]
        print()
        print(f"  plane drift vs block {axes.target_block}: {len(near)}/{len(drift)} blocks "
              f"share the plane at cos > 0.9;")
        print(f"    furthest is block {worst['block']} at {worst['worst_angle_deg']:.0f} deg.")
        print("    A single plane held across depth is a different claim from one that")
        print("    rotates block by block, and the variance column cannot tell them apart.")
    if lens_rows:
        print()
        print("  lens readouts per block are in phase5_sweep_readouts.csv; the tokens")
        print("  should sharpen where the alignment curve peaks and blur at both ends.")


# --------------------------------------------------------------------------- #
# (b) Perspective axis
# --------------------------------------------------------------------------- #

def verify_frames(tokenizer, config: PCAJLensConfig) -> dict:
    """Check the two framings tokenize to the same length.

    The confound this exists to stop: pooling excludes the first ``token_offset``
    real tokens, counted from the start of the *input*. A framing prefix pushes that
    window deeper into the story, so if the self prefix is 7 tokens and the other is
    9, the two conditions pool different parts of the narrative and the difference
    between them is partly "which sentences were averaged". Equal length makes the
    shift identical in both arms, where it cancels out of the contrast.
    """
    lengths = {}
    for name, frame in (("self", config.perspective_self_frame),
                        ("other", config.perspective_other_frame)):
        lengths[name] = len(tokenizer.encode(frame, add_special_tokens=False))
    matched = lengths["self"] == lengths["other"]
    return {
        "self_frame": config.perspective_self_frame,
        "other_frame": config.perspective_other_frame,
        "tokens": lengths,
        "matched": matched,
        "offset_after_frame": config.token_offset - lengths["self"],
    }


def build_framed_stimuli(
    config: PCAJLensConfig, stimuli: pd.DataFrame, emotions: list[str]
) -> pd.DataFrame:
    """Both framings over the *same* stories, so the contrast is paired.

    Paired rather than two independent samples: the same story appears under both
    framings, so topic, wording and length are identical on the two sides and the
    difference of means isolates the framing. An unpaired design would confound
    perspective with whatever stories happened to land in each arm.
    """
    subset = stimuli[stimuli["emotion"].isin(emotions)]
    per_emotion = max(config.perspective_stories_per_emotion, 1)
    chosen = (
        subset.sort_values(["emotion", "topic_id", "story_idx"])
        .groupby("emotion", sort=True)
        .head(per_emotion)
    )
    frames = {
        "self": config.perspective_self_frame,
        "other": config.perspective_other_frame,
    }
    parts = []
    for frame_name, prefix in frames.items():
        framed = chosen.copy()
        framed["text"] = prefix + framed["text"].astype(str)
        framed["frame"] = frame_name
        framed["example_id"] = f"persp-{frame_name}-" + framed["example_id"].astype(str)
        parts.append(framed)
    return pd.concat(parts, ignore_index=True)


def perspective_axis(
    store: ActivationStore, hidden_state: int, rows: pd.DataFrame, seed: int
) -> tuple[np.ndarray, dict]:
    """Paired self-minus-other difference, plus its split-half reliability.

    Reliability by topic half, matching every other split in this repo: twelve
    stories share a scenario, so halving by story would leak it and inflate the
    cosine.
    """
    self_rows = rows[rows["frame"] == "self"].reset_index(drop=True)
    other_rows = rows[rows["frame"] == "other"].reset_index(drop=True)
    # Pair by the underlying stimulus id, so a skipped row on one side cannot shift
    # the alignment of the two arms.
    def strip(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return frame.assign(
            _key=frame["example_id"].str.replace(f"persp-{name}-", "", regex=False)
        )

    self_rows, other_rows = strip(self_rows, "self"), strip(other_rows, "other")
    shared = sorted(set(self_rows["_key"]) & set(other_rows["_key"]))
    self_rows = self_rows[self_rows["_key"].isin(shared)].sort_values("_key")
    other_rows = other_rows[other_rows["_key"].isin(shared)].sort_values("_key")

    self_acts = store.load_layer(hidden_state, self_rows)
    other_acts = store.load_layer(hidden_state, other_rows)
    difference = self_acts.astype(np.float64) - other_acts.astype(np.float64)
    axis = difference.mean(axis=0)

    topics = self_rows["topic_id"].to_numpy()
    halves = topic_halves(self_rows, seed)[0]
    mask = np.asarray([halves[int(t)] for t in topics]) == 0
    reliability = None
    if mask.any() and (~mask).any():
        first, second = difference[mask].mean(axis=0), difference[~mask].mean(axis=0)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 0:
            reliability = float(first @ second / denominator)
    return axis / max(float(np.linalg.norm(axis)), 1e-12), {
        "n_pairs": int(len(shared)),
        "n_unpaired_self": int(len(set(self_rows["_key"]) - set(shared))),
        "raw_norm": float(np.linalg.norm(axis)),
        "split_half_cosine": reliability,
    }


def cmd_perspective(config: PCAJLensConfig, args) -> int:
    run_config = replace(config, run_name=f"{config.run_name}_perspective")
    out_dir = config.phase_dir / "phase5_perspective"
    cache_dir = paths.hf_cache_dir()

    print(RULE)
    print(f"PHASE 5b GATE -- perspective axis   run '{run_config.run_name}'")
    print(RULE)
    print("The only Phase 5 extension that collects activations: different input text")
    print("means different activations, so they go to their own run directory and R2")
    print(f"prefix ({run_config.resolved_r2_prefix()}).")
    print()

    axes = read_pcs(config.pcs_path, config.pcs_meta_path)
    entries = list(DEFAULT_CIRCUMPLEX_SET)
    if config.perspective_emotions is None:
        # One per quadrant: spans affect while keeping the extraction to minutes.
        chosen = [
            next(e.emotion for e in entries if e.quadrant == q) for q in QUADRANT_ORDER
        ]
    elif config.perspective_emotions == "all":
        chosen = [e for e in axes.fit_emotions]
    else:
        chosen = list(config.perspective_emotions)
    stimuli = read_stimuli(config)
    unknown = sorted(set(chosen) - set(stimuli["emotion"]))
    if unknown:
        raise SystemExit(
            f"perspective_emotions {unknown} are not in the Phase 1 stimulus table "
            f"(it has {sorted(set(stimuli['emotion']))})."
        )

    framed = build_framed_stimuli(config, stimuli, chosen)
    print(f"emotions        : {chosen}")
    print(f"stories/emotion : {config.perspective_stories_per_emotion} "
          "(the SAME stories under both framings -- a paired contrast)")
    print(f"stimuli         : {len(framed):,} = "
          f"{len(framed) // 2:,} stories x 2 framings")

    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )
    layers = model_utils.resolve_layers(config.layer_spec, arch.n_hidden_states)
    hidden_state = jlens_lens.hidden_state_index(axes.target_block)
    if hidden_state not in layers:
        raise SystemExit(
            f"the reference block {axes.target_block} maps to hidden state "
            f"{hidden_state}, which layer_spec={config.layer_spec!r} does not store."
        )

    tokenizer = None
    frames = None
    try:
        tokenizer = model_utils.load_tokenizer(
            config.model_name, config.model_revision, cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        frames = verify_frames(tokenizer, config)
    except Exception as exc:
        if not args.dry_run:
            raise
        frames = {"error": str(exc)}

    print()
    print(RULE)
    print("STEP 1  Do the two framings tokenize to the same length?")
    print(RULE)
    if frames is not None and "error" in frames:
        print(f"  tokenizer unavailable: {frames['error']}")
        print()
        print("ABORTED: the framing-length check is the whole point of this dry run, and",
              file=sys.stderr)
        print("it needs the real tokenizer. Reporting 'no problems found' without having",
              file=sys.stderr)
        print("run it would be worse than failing, because the confound it catches is",
              file=sys.stderr)
        print("invisible in the results. Fix the tokenizer download and re-run.",
              file=sys.stderr)
        return 3
    if frames is not None:
        print(f"  self  {frames['self_frame']!r} -> {frames['tokens']['self']} tokens")
        print(f"  other {frames['other_frame']!r} -> {frames['tokens']['other']} tokens")
        print(f"  matched: {frames['matched']}")
        print()
        print("  Why this is a gate and not a detail. Pooling drops the first")
        print(f"  token_offset={config.token_offset} real tokens of the *input*, so a")
        print(f"  {frames['tokens']['self']}-token prefix starts the average "
              f"{frames['tokens']['self']} tokens later into the story")
        print(f"  (effectively story token {frames['offset_after_frame']} onward). Equal-"
              "length prefixes shift both")
        print("  arms identically and it cancels; unequal ones would make the contrast")
        print("  partly 'which sentences were averaged' rather than who it happened to.")
        if not frames["matched"]:
            print()
            print("ABORTED: the framings differ in token length, so the contrast would be",
                  file=sys.stderr)
            print("confounded with the pooling window. Edit perspective_self_frame /",
                  file=sys.stderr)
            print("perspective_other_frame so they differ in one equal-length word.",
                  file=sys.stderr)
            return 3

    nbytes = model_utils.dtype_nbytes(config.activation_dtype)
    estimated = len(framed) * len(layers) * arch.hidden_size * nbytes
    use_r2, r2_reason = decide_r2(run_config, estimated)
    print()
    print(f"storage         : {human_bytes(estimated)} to "
          f"{run_config.activations_dir}")
    print(f"R2 mirror       : {use_r2} ({r2_reason})")

    if args.dry_run:
        print()
        print(RULE)
        print("--dry-run: framings verified, storage estimated, no weights loaded.")
        print(RULE)
        return 0

    fingerprint = run_config.fingerprint(
        layers, arch.hidden_size, arch.resolved_sha, stimuli_fingerprint(framed)
    )
    # The framings are what makes these activations mean something different from
    # Phase 2's, and PCAJLensConfig has no prefix field, so they go in explicitly.
    # Without this, editing a frame would resume into the old run and average two
    # different contrasts together.
    fingerprint["perspective_self_frame"] = config.perspective_self_frame
    fingerprint["perspective_other_frame"] = config.perspective_other_frame
    try:
        init_or_check_manifest(
            run_config.activations_dir, fingerprint=fingerprint,
            extra={"created": provenance.utc_timestamp(), "stage": "phase5_perspective",
                   "layers": layers, "hidden_size": arch.hidden_size,
                   "activation_dtype": config.activation_dtype},
            allow_overwrite=args.overwrite,
        )
    except activation_store.IncompatibleRunError as exc:
        print(f"\nABORTED -- incompatible existing run\n\n{exc}\n", file=sys.stderr)
        return 3

    already = activation_store.completed_example_ids(run_config.activations_dir)
    todo = framed[~framed["example_id"].isin(already)].reset_index(drop=True)
    print(f"already stored  : {len(already):,};  to extract: {len(todo):,}")

    hf_model = None
    if not todo.empty:
        print()
        print(f"Loading {config.model_name} ...")
        t0 = time.time()
        hf_model = model_utils.load_model(
            config.model_name, revision=config.model_revision, cache_dir=cache_dir,
            dtype=config.dtype, device_map=config.device_map,
            quantization=config.quantization,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        )
        print(f"  loaded in {time.time() - t0:.0f}s")
        on_chunk = None
        if use_r2:
            from core.r2 import make_chunk_uploader

            on_chunk = make_chunk_uploader(
                run_config.resolved_r2_prefix(), run_config.activations_dir,
                delete_local=False,   # the analysis below re-reads them immediately
            )
        target = _target_layer(axes, arch)
        stats = extract(
            config=run_config, todo=todo, model=hf_model, tokenizer=tokenizer,
            layers=layers, target=target,
            accumulator=VectorAccumulator.empty(sorted(set(framed["emotion"])),
                                                arch.hidden_size),
            half_of_topic=topic_halves(framed, config.seed)[0],
            shard_index=0, on_chunk_written=on_chunk,
        )
        print(f"  written {stats['n_written']:,}, skipped {stats['n_skipped']:,}, "
              f"{stats['elapsed_s'] / 60:.1f} min")

    store = ActivationStore(run_config.activations_dir)
    rows = store.index.copy()
    rows["frame"] = np.where(
        rows["example_id"].str.startswith("persp-self-"), "self", "other"
    )
    axis, axis_report = perspective_axis(store, hidden_state, rows, config.seed)
    report = analyse_perspective(config, axes, store, rows, axis, axis_report,
                                 hidden_state, chosen)

    lens_report = lens_perspective(config, axes, axis, hf_model, tokenizer)
    print_perspective_gate(report, axis_report, lens_report, axes)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report["cosines"]]).to_csv(
        out_dir / "phase5_perspective_cosines.csv", index=False
    )
    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 5b PERSPECTIVE -- {run_config.run_name}",
        sections={
            "run": {"stage": "phase5_extensions", "extension": "perspective",
                    "run_name": run_config.run_name, "output_dir": str(out_dir)},
            "config": config.to_dict(),
            "frames": frames,
            "emotions": chosen,
            "axis": axis_report,
            "analysis": report,
            "lens": lens_report,
            "fingerprint": fingerprint,
        },
        txt_name="phase5_perspective.txt", json_name="phase5_perspective.json",
    )
    print(f"\n  records : {txt_path}\n            {json_path}")
    print()
    print("STOPPING at the Phase 5b gate.")
    print(RULE)
    return 0


def _target_layer(axes, arch):
    """A Phase 2 ``TargetLayer`` for the reference block, so ``extract`` can be reused."""
    from emotion_pca_jlens.phase2_vectors import TargetLayer

    return TargetLayer(
        block=axes.target_block,
        hidden_state=jlens_lens.hidden_state_index(axes.target_block),
        n_layers=arch.n_layers,
        max_lens_block=jlens_lens.max_lens_block(arch.n_layers),
        resolved_from="Phase 3 artefact",
    )


def analyse_perspective(
    config: PCAJLensConfig, axes, store: ActivationStore, rows: pd.DataFrame,
    axis: np.ndarray, axis_report: dict, hidden_state: int, emotions: list[str],
) -> dict:
    """Is the perspective axis orthogonal to the emotion axes -- beyond chance?"""
    cosines = {
        "chance_baseline": chance_cosine(len(axis)),
        "chance_baseline_plane": chance_cosine(len(axis), 2),
        "split_half_cosine": axis_report["split_half_cosine"],
    }
    for i in range(min(2, axes.rank)):
        cosines[f"cos_phase3_pc{i + 1}"] = float(abs(axis @ axes.components[i]))
    if axes.rank >= 2:
        plane = principal_angle_cosines(
            axis[None, :] / np.linalg.norm(axis), axes.components[:2]
        )
        cosines["cos_into_phase3_plane"] = float(plane.max())

    # The same test against emotion PCs refitted *within* the framed condition, so
    # the comparison is not across a change of input format.
    framed_emotions = sorted(set(rows["emotion"]))
    half_of_topic = topic_halves(rows, config.seed)[0]
    means = emotion_means_at(store, rows, hidden_state, framed_emotions, half_of_topic)
    keep = np.asarray([e != NEUTRAL_QUADRANT for e in framed_emotions])
    refit = None
    if int(keep.sum()) >= 3:
        refit = fit_pca(
            means.full[keep], [e for i, e in enumerate(framed_emotions) if keep[i]],
            mean_center=config.mean_center,
        )
        for i in range(min(2, refit.rank)):
            cosines[f"cos_framed_pc{i + 1}"] = float(abs(axis @ refit.components[i]))
        if refit.rank >= 2:
            cosines["cos_into_framed_plane"] = float(
                principal_angle_cosines(axis[None, :], refit.components[:2]).max()
            )
    return {
        "cosines": cosines,
        "n_framed_emotions": len(framed_emotions),
        "framed_pca_rank": None if refit is None else refit.rank,
        "emotions": emotions,
    }


def lens_perspective(
    config: PCAJLensConfig, axes, axis: np.ndarray, hf_model, tokenizer
) -> dict:
    """What does the perspective axis read out as, at both ends?

    Takes the model extraction already loaded rather than loading its own. Two
    ``from_pretrained`` calls would put two copies of a 65 GiB checkpoint in memory
    and OOM the card that just finished the forward passes -- and the weights are
    identical, so there is nothing to gain by it.
    """
    if hf_model is None or tokenizer is None:
        return {"available": False,
                "reason": "nothing was extracted this run, so no model is loaded; "
                          "re-run with --overwrite, or lens the axis from the CSV"}
    try:
        lens_path, lens_report = resolve_lens(config, paths.hf_cache_dir())
        readout = jlens_lens.LensReadout.build(hf_model, tokenizer, lens_path)
    except Exception as exc:  # pragma: no cover - lens unavailable
        return {"available": False, "reason": str(exc)}

    probes = build_probes(axes)
    rng = rng_for(config.seed, "phase5_perspective_lens")
    out: dict = {"available": True, "lens": lens_report["source"], "ends": {}}
    for sign, label in ((+1.0, "+perspective (self)"), (-1.0, "-perspective (other)")):
        result = read_direction(
            readout, axis * sign, axes.target_block, label,
            probes["words"], probes["valence"], probes["arousal"], config.topk, rng,
        )
        out["ends"][label] = {
            "tokens": result.token_strings(config.topk),
            "auroc_valence": result.auroc_valence,
            "auroc_arousal": result.auroc_arousal,
        }
    return out


def print_perspective_gate(
    report: dict, axis_report: dict, lens_report: dict, axes,
) -> None:
    cosines = report["cosines"]
    chance = cosines["chance_baseline"]
    plane_chance = cosines["chance_baseline_plane"]
    print()
    print(RULE)
    print("GATE  Is perspective a separate axis?")
    print(RULE)
    print(f"paired stories        : {axis_report['n_pairs']:,}")
    print(f"axis split-half cosine: {_fmt(axis_report['split_half_cosine'])}  "
          "(by topic; the axis itself has to be reliable first)")
    print()
    print(f"chance |cos| in {len(axes.components[0]):,} dimensions: "
          f"{chance:.4f} vs a single axis, {plane_chance:.4f} vs a plane")
    print("  Every number below is against those, not against zero. A plane's baseline")
    print("  is sqrt(2/d), not 1/sqrt(d) -- using the wrong one understates chance by")
    print("  sqrt(2). In this many dimensions a cosine of 0.05 looks orthogonal while")
    print("  being several times chance; the intuition runs the wrong way here.")
    print()
    print(f"{'comparison':<34}{'|cos|':>9}{'x chance':>11}")
    print(THIN)
    for key, value in cosines.items():
        if key.startswith("chance_baseline") or key == "split_half_cosine" or value is None:
            continue
        baseline = plane_chance if "plane" in key else chance
        print(f"{key:<34}{value:>9.4f}{value / baseline:>10.1f}x")
    print(THIN)
    emotion_keys = [
        k for k in cosines
        if k.startswith(("cos_phase3_pc", "cos_framed_pc")) and cosines[k] is not None
    ]
    worst_key = max(emotion_keys, key=lambda k: cosines[k] / (
        plane_chance if "plane" in k else chance), default=None)
    worst = None if worst_key is None else cosines[worst_key]
    if worst is not None:
        ratio = worst / (plane_chance if "plane" in worst_key else chance)
        print(f"  largest alignment with an emotion axis: {worst:.4f} = {ratio:.1f}x chance")
        if ratio < 3:
            print("    Roughly orthogonal in the only sense that means anything here:")
            print("    no more aligned than two unrelated directions would be.")
        else:
            print("    NOT orthogonal. Perspective and affect share a component, so a")
            print("    steering result on one cannot be attributed cleanly to it alone.")
    if lens_report.get("available"):
        print()
        print("  lens readout of the axis (both ends):")
        for label, end in lens_report["ends"].items():
            print(f"    {label:<22} " + " ".join(repr(t) for t in end["tokens"][:8]))
        print("    The ends are exact complements by construction (see Phase 4), so read")
        print("    the token lists, not the symmetry.")
    elif lens_report.get("reason"):
        print(f"\n  lens readout unavailable: {lens_report['reason']}")


# --------------------------------------------------------------------------- #
# (c) Within-emotion PCA
# --------------------------------------------------------------------------- #

def topic_eta_squared(scores: np.ndarray, topic_ids: np.ndarray) -> float:
    """Fraction of a PC's score variance that is between-topic.

    A one-way ANOVA effect size. 0 means the axis ignores which scenario a story is
    about; 1 means it separates scenarios perfectly. This is the number that turns
    "within one emotion the axes are topic axes" from an impression into a claim.
    """
    grand = float(scores.mean())
    total = float(((scores - grand) ** 2).sum())
    if total <= 0:
        return 0.0
    # bincount rather than a loop over topics: this runs inside a 1,000-shuffle
    # permutation test, so a per-topic Python loop would dominate the whole stage.
    _, inverse = np.unique(topic_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    means = np.bincount(inverse, weights=scores) / counts
    return float((counts * (means - grand) ** 2).sum() / total)


def topic_eta_pvalue(
    scores: np.ndarray, topic_ids: np.ndarray, rng, n_permutations: int
) -> float:
    """Permutation p-value for :func:`topic_eta_squared`.

    Needed because eta-squared is bounded below by the group count: with 100 topics
    over 400 stories, shuffled labels already explain a quarter of the variance. The
    raw number alone would look like strong topic structure where there is none.
    """
    observed = topic_eta_squared(scores, topic_ids)
    labels = np.asarray(topic_ids)
    hits = sum(
        1 for _ in range(n_permutations)
        if topic_eta_squared(scores, rng.permutation(labels)) >= observed
    )
    return (hits + 1) / (n_permutations + 1)


def cmd_within_emotion(config: PCAJLensConfig, args) -> int:
    out_dir = config.phase_dir / "phase5_within_emotion"
    print(RULE)
    print(f"PHASE 5c GATE -- within-emotion PCA   run '{config.run_name}'")
    print(RULE)
    print("The contrast that justifies the cross-emotion design. PCA inside a single")
    print("emotion, over individual stimuli rather than centroids: if the top axes")
    print("there are topic and scenario rather than affect, that is the argument for")
    print("why PCA had to run *across* emotions to find the circumplex.")
    print()

    axes = read_pcs(config.pcs_path, config.pcs_meta_path)
    if config.within_emotion_targets is None:
        # Two from opposite quadrants: whatever the within-emotion axes turn out to
        # be, seeing the same answer twice from unrelated affect is the check.
        wanted = [
            next((e for e in axes.fit_emotions
                  if axes.labels.get(e, {}).get("quadrant") == q), None)
            for q in ("HA-P", "LA-N")
        ]
        targets = [e for e in wanted if e]
    elif config.within_emotion_targets == "all":
        targets = list(axes.fit_emotions)
    else:
        targets = list(config.within_emotion_targets)
    unknown = sorted(set(targets) - set(axes.emotions))
    if unknown:
        raise SystemExit(f"within_emotion_targets {unknown} are not in {axes.emotions}")

    print(f"emotions        : {targets}")
    print(f"PCs per emotion : {config.within_emotion_n_pcs}")
    print(f"affect plane    : PC1-PC2 from {config.pcs_path.name}")
    if args.dry_run:
        print("\n--dry-run: nothing read.")
        return 0

    store = open_store(config)
    hidden_state = jlens_lens.hidden_state_index(axes.target_block)
    if hidden_state not in store.layers:
        raise SystemExit(
            f"hidden state {hidden_state} (block {axes.target_block}) is not stored; "
            f"the store has {store.layers[:4]}..{store.layers[-4:]}"
        )
    basis, neutral_report = (None, {})
    if config.remove_neutral_pcs:
        basis, neutral_report = neutral_basis(
            store, hidden_state, config.neutral_variance_threshold
        )
        print(f"neutral PCs removed: {neutral_report.get('n_pcs_removed')} "
              f"(from {neutral_report.get('n_neutral_stories')} neutral stories)")

    rng = rng_for(config.seed, "phase5_within_emotion")
    rows_out: list[dict] = []
    print()
    for emotion in targets:
        subset = store.subset(emotions=[emotion])
        if len(subset) < 10:
            print(f"  {emotion}: only {len(subset)} stimuli stored; skipping")
            continue
        activations = store.load_layer(hidden_state, subset).astype(np.float64)
        if basis is not None:
            activations = project_out(activations, basis)
        pca = fit_pca(activations, [emotion] * len(subset), mean_center=True)
        topics = subset["topic_id"].to_numpy()
        n_pcs = min(config.within_emotion_n_pcs, pca.rank)

        print(THIN)
        print(f"{emotion}  ({len(subset):,} stimuli, {len(np.unique(topics))} topics, "
              f"rank {pca.rank})")
        print(THIN)
        print(f"{'PC':>4}{'share':>9}{'topic eta^2':>13}{'p':>8}"
              f"{'|cos| affect plane':>21}")
        for i in range(n_pcs):
            scores = pca.scores[:, i]
            eta = topic_eta_squared(scores, topics)
            p_value = topic_eta_pvalue(scores, topics, rng, TOPIC_PERMUTATIONS)
            into_plane = (
                float(principal_angle_cosines(
                    pca.components[i][None, :], axes.components[:2]
                ).max()) if axes.rank >= 2 else None
            )
            rows_out.append({
                "emotion": emotion, "pc": i + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio[i]),
                "topic_eta_squared": eta, "topic_p_value": p_value,
                "cos_into_affect_plane": into_plane,
                "n_stimuli": int(len(subset)), "n_topics": int(len(np.unique(topics))),
            })
            print(f"{i + 1:>4}{pca.explained_variance_ratio[i]:>9.1%}{eta:>13.3f}"
                  f"{p_value:>8.3f}{_fmt(into_plane):>21}")

    if not rows_out:
        print("\nNo emotion had enough stored stimuli; nothing to report.")
        return 3

    frame = pd.DataFrame(rows_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "phase5_within_emotion.csv", index=False)
    print_within_gate(frame, axes)

    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 5c WITHIN-EMOTION -- {config.run_name}",
        sections={
            "run": {"stage": "phase5_extensions", "extension": "within-emotion",
                    "run_name": config.run_name, "output_dir": str(out_dir)},
            "config": config.to_dict(),
            "source": {"pcs": str(config.pcs_path),
                       "block": axes.target_block,
                       "activations": str(config.activations_dir)},
            "neutral_subspace": neutral_report,
            "per_pc": rows_out,
        },
        txt_name="phase5_within_emotion.txt", json_name="phase5_within_emotion.json",
    )
    print(f"\n  table   : {out_dir / 'phase5_within_emotion.csv'}")
    print(f"  records : {txt_path}\n            {json_path}")
    print()
    print("STOPPING at the Phase 5c gate.")
    print(RULE)
    return 0


def print_within_gate(frame: pd.DataFrame, axes) -> None:
    # A within-emotion PC is compared against the *plane* PC1-PC2 spans, so the
    # baseline is sqrt(2/d) rather than 1/sqrt(d).
    chance = chance_cosine(int(axes.components.shape[1]), 2)
    top = frame[frame["pc"] <= 2]
    print()
    print(RULE)
    print("GATE  Do the within-emotion axes recover topic rather than affect?")
    print(RULE)
    topicy = top[(top["topic_eta_squared"] > 0.3) & (top["topic_p_value"] <= 0.05)]
    print(f"  top-2 within-emotion PCs that separate topics beyond chance: "
          f"{len(topicy)}/{len(top)}")
    print(f"  median topic eta^2 over those PCs: {top['topic_eta_squared'].median():.3f}")
    into = top["cos_into_affect_plane"].dropna()
    if not into.empty:
        print(f"  |cos| into the cross-emotion affect plane: max {into.max():.4f} "
              f"= {into.max() / chance:.1f}x chance ({chance:.4f})")
        if into.max() / chance < 3:
            print("    So the within-emotion axes are not the affect axes. That is the")
            print("    argument for the cross-emotion design: the circumplex is a fact")
            print("    about the relation *between* emotions, and it is invisible from")
            print("    inside one of them however many stimuli you have.")
        else:
            print("    These axes DO share a component with the affect plane, which")
            print("    weakens the cross-emotion argument -- some of what looked like a")
            print("    between-emotion structure is present within a single emotion.")
    print()
    print("  eta^2 is read against its p-value, not on its own: with many topics and")
    print("  few stories each, shuffled labels already explain a large share, so a big")
    print("  raw eta^2 is not by itself topic structure.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    handlers = {
        "layer-sweep": cmd_layer_sweep,
        "perspective": cmd_perspective,
        "within-emotion": cmd_within_emotion,
    }
    try:
        return handlers[args.extension](config, args)
    except activation_store.MissingChunkError as exc:
        # Caught here rather than at store construction, because the index parquets
        # are local and only the tensors are missing -- so the store opens fine and
        # the failure lands on the first load_layer, deep inside a sweep. The store's
        # own message has to guess the prefix; this one knows it.
        print(f"\nABORTED -- the pooled activations are not on this machine\n\n{exc}\n",
              file=sys.stderr)
        print("For this run specifically:\n\n"
              f"  python run.py r2 pull {config.activations_dir} "
              f"--prefix {config.resolved_r2_prefix()}\n", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
