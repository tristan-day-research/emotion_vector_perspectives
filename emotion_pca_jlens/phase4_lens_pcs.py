"""Phase 4 (GATE): read the principal axes out through the Jacobian lens.

What this stage does
--------------------
Each PC from Phase 3 is a unit direction in residual space at a block the lens
covers, so it goes straight to the lens. Both ends of every axis are read out --
``+PC`` and ``-PC`` -- because an axis has two, and valence should read pleasant
one way and unpleasant the other. Output is a table of PC index, variance
explained, and top-k tokens per end.

No activations are collected and no ``J`` is fitted. The lens is the pre-fitted
one Phase 0 verified; fitting one costs a backward pass per prompt over hundreds
of prompts, and reusing it is the entire reason this experiment fits in a day.

The problem this gate has to solve first
----------------------------------------
A murky readout has two completely different causes and the tokens alone cannot
tell them apart:

* the PC is real but **not lexicalised** -- the model has no single vocabulary
  token for it, which is a finding about the vocabulary (README caveat 1); or
* the **lens is too weak** to verbalise anything at this block, in which case the
  PC readout is uninterpretable and says nothing either way.

That second branch is live here, not hypothetical. The published ``qwen3-32b``
lens is an interrupted fit -- 80 prompts, stopping at ``mean_rel_change`` 0.026
against its own 0.002 threshold -- which is why ``run.py refit_lens`` exists. So
this phase runs the lens against known-answer directions from *this* experiment
before it reads a single PC:

**GATE A -- can the lens verbalise these emotion vectors at all?** Every fitted
emotion's own mean-centred vector is read out, and the rank of *its own word*
recorded. Ground truth we did not choose: the words come from the dataset and the
Phase 1 design. Sixteen independent trials, and if the lens cannot find "sad" in
the sad vector then nothing in GATE B is evidence about anything. The a priori
valence and arousal axes Phase 3 saved are read out in the same block, as
hand-built directions whose readout is predictable.

**GATE B -- the PC readouts.** Top-k tokens per end, plus three things that make
the table interpretable rather than suggestive:

1. **An ordering test against a permutation null.** Instead of eyeballing whether
   the tokens "look pleasant", the anchor emotion words are ranked in the readout
   and scored by AUROC -- pleasant against unpleasant, activated against
   deactivated -- with a p-value from shuffling the labels. The null is not
   optional decoration: the balanced design gives 8 words a side, where a chance
   AUROC has a standard error near 0.15, so 0.75 is under two sigma from nothing
   and a bare threshold would mint lexicalised axes out of noise.

   One thing that looks like a second check and is not. ``unembed`` is odd --
   ``final_norm`` is an RMSNorm, and the transport and LM head are linear -- so
   ``logits(-v) = -logits(v)`` exactly, every probe word's ordering reverses, and
   ``AUROC(-PC) = 1 - AUROC(+PC)`` is arithmetic. "One end reads pleasant and the
   other unpleasant" is therefore guaranteed and cannot be evidence. Only the
   ``+`` end is scored. What the ``-`` end genuinely contributes is its *token
   list*: whether those words read as antonyms of the other end's is not fixed by
   the algebra, and it is what the gate asks a human to judge.
2. **The plain logit lens on the same direction.** If the J-lens and logit-lens
   readouts agree exactly, the transport is doing nothing and the "J-lens result"
   is a logit-lens result.
3. **Phase 3's geometry, quoted per PC.** A PC's split-half stability and its
   correlation with the a priori labels come from the artefact, so a murky readout
   can be attributed: PC3's tokens are noise *and* PC3's component did not survive
   an independent refit. Those two facts belong on the same line.

The sign cross-check that neither phase can do alone
----------------------------------------------------
Phase 3 measures *where the emotions sit* on an axis; Phase 4 measures *what the
axis wants to say*. They are independent measurements of the same object, so
their signs have to agree: if pleasant emotions have negative PC1 scores, then the
``-PC1`` end is the one that should read pleasant. Agreement is real evidence.
Disagreement means one of the two is wrong, and the gate says so rather than
reporting both and letting the reader assume they cohere.

What a readout is and is not
----------------------------
A lens reading of "anxious" is a **disposition to say "anxious"** -- not evidence
the model is anxious (README caveat 2). And the lens decodes one vocabulary token
at a time, so a word that is not a single token cannot surface: those are reported
as excluded, never scored as misses (caveat 1). Expect PC3 onward to get murky.
The gate reports that rather than straining to interpret noise.

Usage::

    python run.py phase4 --dry-run     # lens facts + which words are single tokens
    python run.py phase4               # the gate
    python run.py phase4 --set topk=20 --set n_pcs_to_lens=8
    python run.py phase4 --set lens_local_path=outputs/<run>/results/phases/lens_merged_n580.pt
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
from core.seeds import rng_for, set_global_seeds
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig, load_config
from emotion_pca_jlens.phase1_stimuli import (
    DEFAULT_CIRCUMPLEX_SET,
    NEUTRAL_QUADRANT,
    QUADRANT_ORDER,
)

RULE = "=" * 78
THIN = "-" * 78

#: Rank below which a word counts as "near" rather than a miss. Phase 0's
#: convention, reused so the two gates' verdicts mean the same thing.
NEAR_RANK = 100

#: ``n_prompts`` of the published qwen3-32b lens. Its fit was interrupted, so a
#: weak readout on it is ambiguous between lens noise and non-lexicalisation --
#: exactly the confound GATE A exists to resolve. Detected by value so the warning
#: fires on the published artefact and not on a merged refit.
PUBLISHED_INTERRUPTED_N_PROMPTS = 80

#: Label shuffles behind each ordering p-value. Enough resolution for the balanced
#: design's C(16,8) = 12,870 distinct arrangements, and vectorised, so the cost is
#: microseconds per readout.
ORDERING_PERMUTATIONS = 2000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 4 gate: read the principal components out through the "
                    "Jacobian lens, both ends of each axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="describe the lens, cross-check it against the PCs, and report which "
             "probe words are single tokens; never loads model weights",
    )
    p.add_argument(
        "--pcs",
        type=Path,
        default=None,
        help="Phase 3 principal-component safetensors to read (default: this run's). "
             "Its JSON sidecar must sit alongside it, and the block it was fitted at "
             "is taken from that sidecar rather than from the config",
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
# Reading Phase 3's artefact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PrincipalAxes:
    """Phase 3's principal components plus what is needed to interpret them."""

    components: np.ndarray            # (rank, d_model) unit rows
    explained_variance_ratio: np.ndarray
    mean: np.ndarray                  # (d_model,) the centring mean
    scores: np.ndarray                # (n_emotions, rank)
    emotions: list[str]
    fit_emotions: list[str]
    labels: dict[str, dict]
    valence_axis: np.ndarray | None
    arousal_axis: np.ndarray | None
    target_block: int
    target_hidden_state: int
    n_layers: int
    metadata: dict
    pcs_path: Path

    @property
    def d_model(self) -> int:
        return int(self.components.shape[1])

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def centred_vector(self, emotion: str) -> np.ndarray | None:
        """The emotion's mean-centred vector, reconstructed from scores.

        Exact rather than approximate for any *fitted* emotion: the components span
        the centred row space (their rank is ``n_fitted - 1``), so projecting onto
        them and back is the identity on those rows. That is why Phase 4 needs only
        Phase 3's artefact and never re-opens Phase 2's -- one less path to go stale.

        ``None`` for a row that was not fitted (neutral by default), where the
        reconstruction would be a projection rather than the vector itself.
        """
        if emotion not in self.fit_emotions or emotion not in self.emotions:
            return None
        return self.scores[self.emotions.index(emotion)] @ self.components

    def label(self, emotion: str, key: str) -> int:
        return int(self.labels.get(emotion, {}).get(key, 0) or 0)

    def anchors(self) -> list[str]:
        """Fitted emotions carrying an a priori circumplex position."""
        return [
            e for e in self.fit_emotions
            if self.labels.get(e, {}).get("quadrant") in QUADRANT_ORDER
        ]

    def phase3_row(self, pc_index: int, section: str, key: str):
        """One value from Phase 3's per-PC tables, or ``None`` if absent."""
        rows = self.metadata.get(section, {}).get("per_pc", [])
        for row in rows:
            if int(row.get("pc", -1)) == pc_index + 1:
                return row.get(key)
        return None


def _fallback_labels() -> dict[str, dict]:
    """Circumplex labels from the Phase 1 table, for an older Phase 3 sidecar."""
    labels = {
        e.emotion: {"quadrant": e.quadrant, "valence": e.valence,
                    "arousal": e.arousal, "family": e.family, "source": "emotion"}
        for e in DEFAULT_CIRCUMPLEX_SET
    }
    labels[NEUTRAL_QUADRANT] = {"quadrant": NEUTRAL_QUADRANT, "valence": 0,
                                "arousal": 0, "family": "neutral", "source": "neutral"}
    return labels


def read_pcs(pcs_path: Path, meta_path: Path) -> PrincipalAxes:
    """Load Phase 3's components, or explain how to produce them."""
    from safetensors.numpy import load_file

    if not pcs_path.exists():
        raise SystemExit(
            f"no principal components at\n  {pcs_path}\n\n"
            "Phase 4 reads Phase 3's output. Run it first:\n\n"
            "  python run.py phase3\n"
        )
    if not meta_path.exists():
        raise SystemExit(
            f"{pcs_path} has no sidecar at\n  {meta_path}\n\n"
            "The sidecar records which block the PCs were fitted at, and lensing a "
            "direction at\nthe wrong block produces plausible nonsense. Phase 4 will "
            "not guess it.\nRe-run `python run.py phase3` (it writes both together)."
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = load_file(str(pcs_path))
    missing = [k for k in ("components", "mean", "scores_all_emotions") if k not in tensors]
    if missing:
        raise SystemExit(
            f"{pcs_path} is missing {missing}; it was written by an older Phase 3. "
            "Re-run it."
        )
    target = metadata.get("target", {})
    if "block" not in target:
        raise SystemExit(
            f"{meta_path} does not record target.block. Phase 4 needs the block the "
            "PCs were\nfitted at, because that selects which J_l transports them. "
            "Re-run Phase 3."
        )

    labels = metadata.get("labels")
    if not labels:
        print("  NOTE this Phase 3 sidecar carries no labels; falling back to the "
              "Phase 1 circumplex")
        print("       table. Re-run Phase 3 to record them alongside the components.")
        labels = _fallback_labels()

    return PrincipalAxes(
        components=np.asarray(tensors["components"], dtype=np.float64),
        explained_variance_ratio=np.asarray(
            tensors.get("explained_variance_ratio", []), dtype=np.float64
        ),
        mean=np.asarray(tensors["mean"], dtype=np.float64),
        scores=np.asarray(tensors["scores_all_emotions"], dtype=np.float64),
        emotions=list(metadata["emotions"]),
        fit_emotions=list(metadata.get("fit_emotions", metadata["emotions"])),
        labels=labels,
        valence_axis=(
            np.asarray(tensors["valence_axis"], dtype=np.float64)
            if "valence_axis" in tensors else None
        ),
        arousal_axis=(
            np.asarray(tensors["arousal_axis"], dtype=np.float64)
            if "arousal_axis" in tensors else None
        ),
        target_block=int(target["block"]),
        target_hidden_state=int(target.get("hidden_state", int(target["block"]) + 1)),
        n_layers=int(target.get("n_layers", 0)),
        metadata=metadata,
        pcs_path=pcs_path,
    )


# --------------------------------------------------------------------------- #
# Locating the lens
# --------------------------------------------------------------------------- #

def resolve_lens(config: PCAJLensConfig, cache_dir: Path) -> tuple[Path, dict]:
    """Path to the lens this run should use, plus what it is.

    Honours ``lens_local_path`` for the output of ``run.py refit_lens``. That path
    skips Hub resolution, and with it the ``hf_model_name`` guard that normally
    refuses a lens fitted on a different checkpoint -- so it is recorded loudly
    rather than silently, because a Jacobian is a function of the weights and a
    mismatched lens produces confident nonsense rather than an error.
    """
    if config.lens_local_path:
        path = Path(config.lens_local_path)
        if not path.exists():
            raise SystemExit(f"lens_local_path does not exist: {path}")
        return path, {
            "source": "lens_local_path",
            "path": str(path),
            "hf_model_name_verified": False,
            "note": "local lens: the hf_model_name guard does not apply, so it is on "
                    "you that this was fitted on config.model_name",
        }

    artifact = jlens_lens.resolve_lens_artifact(
        config.model_name,
        lens_repo=config.lens_repo,
        revision=config.lens_revision,
        subfolder=config.lens_subfolder,
    )
    path = jlens_lens.download_lens(
        artifact, config.lens_repo, config.lens_revision, cache_dir
    )
    fit = artifact.fit_config
    results = fit.get("results", {}) if isinstance(fit.get("results"), dict) else {}
    return path, {
        "source": "published",
        "repo": config.lens_repo,
        "subfolder": artifact.subfolder,
        "path": str(path),
        "fitted_on": fit.get("hf_model_name"),
        "hf_model_name_verified": fit.get("hf_model_name") == config.model_name,
        "recorded_prompts_fitted": results.get("prompts_fitted"),
        "recorded_final_mean_rel_change": results.get("final_mean_rel_change"),
    }


def crosscheck_lens(description, axes: PrincipalAxes, arch) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` from comparing the lens, the model and the PCs."""
    problems = description.problems(n_layers=arch.n_layers, d_model=arch.hidden_size)
    warnings: list[str] = []

    if description.d_model != axes.d_model:
        problems.append(
            f"lens d_model={description.d_model} but the PCs are {axes.d_model}-"
            "dimensional; these do not belong to the same model"
        )
    if axes.target_block not in description.source_layers:
        covered = f"{description.source_layers[0]}..{description.source_layers[-1]}"
        problems.append(
            f"the PCs were fitted at block {axes.target_block}, which has no fitted "
            f"J_l (the lens covers {covered}). Nothing can be read out there"
        )
    if axes.n_layers and axes.n_layers != arch.n_layers:
        problems.append(
            f"the PCs were built on a {axes.n_layers}-block model but "
            f"{arch.model_name} has {arch.n_layers}"
        )
    if description.n_prompts == PUBLISHED_INTERRUPTED_N_PROMPTS:
        warnings.append(
            f"this lens was fitted on {description.n_prompts} prompts -- the published "
            "qwen3-32b artefact, whose fit was interrupted well short of its own "
            "convergence threshold. A weak PC readout on it cannot be distinguished "
            "from lens noise. GATE A is what decides whether the readouts below mean "
            "anything; `run.py refit_lens` produces a converged alternative."
        )
    return problems, warnings


# --------------------------------------------------------------------------- #
# Readout primitives
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Readout:
    """One direction read out through the lens at one block."""

    name: str
    tokens: list                      # list[jlens_lens.TokenReadout]
    logit_lens_tokens: list
    probe_ranks: dict[str, int | None]
    auroc_valence: float | None
    auroc_arousal: float | None
    p_valence: float | None
    p_arousal: float | None
    n_probes_scored: int
    top1_prob: float
    effective_tokens: float
    jaccard_with_logit_lens: float

    def token_strings(self, k: int) -> list[str]:
        return [t.token for t in self.tokens[:k]]


def _ordering_auroc(scores: list[float], labels: list[int]) -> float | None:
    """AUROC of ``scores`` against binary ``labels``; ``None`` if a class is empty.

    Used with ``-rank`` as the score, so a word the readout ranks highly counts as a
    high score. Ties are impossible (ranks are distinct), so no tie correction is
    needed. 0.5 is chance.

    **The two ends of an axis are not two measurements.** ``unembed`` is odd --
    ``final_norm`` is an RMSNorm, so it commutes with negation, and the transport
    and the LM head are linear -- so ``logits(-v) = -logits(v)`` exactly. Every
    probe word's ordering therefore reverses exactly, and
    ``AUROC(-v) = 1 - AUROC(+v)`` is arithmetic, not evidence. Only one of the two
    numbers carries information, which is why the strength of an axis is scored
    against a permutation null rather than against "does the other end disagree".
    """
    positive = [s for s, y in zip(scores, labels) if y > 0]
    negative = [s for s, y in zip(scores, labels) if y < 0]
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if a > b else 0.5 if a == b else 0.0
        for a in positive for b in negative
    )
    return wins / (len(positive) * len(negative))


def _ordering_pvalue(
    scores: list[float], labels: list[int], rng, n_permutations: int
) -> float | None:
    """Two-sided permutation p-value for ``_ordering_auroc``.

    Needed because the balanced design gives only 8 words per side, where the
    standard error of a chance AUROC is about 0.15 -- so 0.75 sits barely 1.7 sigma
    from nothing, and with two axes and two ends per PC, a threshold alone would
    manufacture "lexicalised" axes out of noise. Shuffling the labels answers the
    only question that matters: could this ordering have arisen by chance at this
    sample size?

    Computed by rank-sum, so all permutations are evaluated in one vectorised pass:
    the scores are fixed, only the label assignment moves, so the ranks are computed
    once.
    """
    observed = _ordering_auroc(scores, labels)
    if observed is None:
        return None
    label_array = np.asarray(labels)
    keep = label_array != 0
    values = np.asarray(scores, dtype=np.float64)[keep]
    signs = label_array[keep]
    n_pos = int((signs > 0).sum())
    n_neg = int((signs < 0).sum())
    if n_pos == 0 or n_neg == 0:  # pragma: no cover - observed would have been None
        return None

    # Mann-Whitney U from ranks: AUROC = (sum of positive ranks - n_pos(n_pos+1)/2)
    # / (n_pos * n_neg). Ranks depend only on the scores, so they are reused.
    order = np.argsort(values)
    ranks = np.empty_like(values)
    ranks[order] = np.arange(1, values.size + 1)

    draws = rng.permuted(
        np.tile(signs > 0, (n_permutations, 1)), axis=1
    )
    rank_sums = (draws * ranks).sum(axis=1)
    null = (rank_sums - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    deviation = abs(observed - 0.5)
    # +1 in both terms: the observed value is one of the possible arrangements, so a
    # p of exactly 0 is not a claim the sample size can support.
    return float((np.sum(np.abs(null - 0.5) >= deviation - 1e-12) + 1) / (n_permutations + 1))


def read_direction(
    readout,
    direction: np.ndarray,
    block: int,
    name: str,
    probe_words: list[str],
    valence: list[int],
    arousal: list[int],
    topk: int,
    rng,
) -> Readout:
    """Read one direction out both ways and score the probe-word ordering.

    The plain logit-lens readout of the same direction is taken alongside the
    J-lens one, because identical results would mean the transport contributed
    nothing and the finding is a logit-lens finding.
    """
    import torch

    logits = readout.direction_logits(direction, block, use_jacobian=True)
    plain = readout.direction_logits(direction, block, use_jacobian=False)
    top = readout.decode_top(logits, topk)
    plain_top = readout.decode_top(plain, topk)

    ranks: dict[str, int | None] = {}
    scores: list[float] = []
    kept_valence: list[int] = []
    kept_arousal: list[int] = []
    for word, val, aro in zip(probe_words, valence, arousal):
        rank, _, _ = readout.rank_of_word(logits, word)
        ranks[word] = rank
        if rank is None:  # not a single token: excluded, never a miss (caveat 1)
            continue
        scores.append(-float(rank))
        kept_valence.append(val)
        kept_arousal.append(aro)

    probs = torch.softmax(logits.float(), dim=-1)
    # exp(entropy): how many tokens this direction is effectively spread over. A
    # lexicalised direction is peaked; noise is close to the vocabulary size.
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum())
    j_ids = {t.token_id for t in top}
    l_ids = {t.token_id for t in plain_top}
    return Readout(
        name=name,
        tokens=top,
        logit_lens_tokens=plain_top,
        probe_ranks=ranks,
        auroc_valence=_ordering_auroc(scores, kept_valence),
        auroc_arousal=_ordering_auroc(scores, kept_arousal),
        p_valence=_ordering_pvalue(scores, kept_valence, rng, ORDERING_PERMUTATIONS),
        p_arousal=_ordering_pvalue(scores, kept_arousal, rng, ORDERING_PERMUTATIONS),
        n_probes_scored=len(scores),
        top1_prob=float(top[0].prob) if top else 0.0,
        effective_tokens=float(np.exp(entropy)),
        jaccard_with_logit_lens=(
            len(j_ids & l_ids) / len(j_ids | l_ids) if (j_ids | l_ids) else 0.0
        ),
    )


def verify_scale_invariance(readout, direction: np.ndarray, block: int, k: int = 5) -> dict:
    """Confirm the readout of these PCs does not depend on their magnitude.

    Phase 0 checks this on a random direction; repeated here on a real PC because it
    is what makes ``+PC`` and ``-PC`` well defined without choosing a step size. The
    algebra says it holds -- the transport is linear and the final norm is an
    RMSNorm -- but eps and dtype effects would not show up in the algebra.
    """
    small = readout.top_tokens(direction, block, k=k)
    large = readout.top_tokens(direction * 250.0, block, k=k)
    same = [t.token_id for t in small] == [t.token_id for t in large]
    return {
        "identical_topk": same,
        "unit": [t.token for t in small],
        "scaled": [t.token for t in large],
    }


# --------------------------------------------------------------------------- #
# GATE A: can the lens verbalise these vectors at all?
# --------------------------------------------------------------------------- #

def gate_a_self_readout(
    readout, axes: PrincipalAxes, config: PCAJLensConfig, block: int
) -> dict:
    """Read each emotion's own vector and look for its own word.

    The calibration the rest of the phase depends on. Ground truth is not something
    we picked to suit the lens: the words are the emotion set, and the vectors are
    the ones Phase 2 measured. Sixteen independent trials at the same block, in the
    same units, as the PC readouts.

    Mean-centred vectors rather than raw centroids, because a raw centroid is
    dominated by whatever every story shares at this layer and would read out as
    that. Centred, the vector is what makes this emotion different from the
    average emotion -- the same quantity Phase 3 ran PCA on.
    """
    print()
    print(RULE)
    print("GATE A  Can the lens verbalise THESE emotion vectors?")
    print(RULE)
    print("Each fitted emotion's mean-centred vector, read out at the same block as")
    print("the PCs, scored on the rank of its own word. If the lens cannot find 'sad'")
    print("in the sad vector, a murky PC readout in GATE B is uninterpretable -- it")
    print("would be indistinguishable from lens noise. Ground truth here is the")
    print("emotion set itself, not a probe chosen to suit the lens.")
    print()

    rows: list[dict] = []
    untokenizable: list[str] = []
    for emotion in axes.fit_emotions:
        vector = axes.centred_vector(emotion)
        if vector is None:  # pragma: no cover - fit_emotions are always reconstructable
            continue
        if not readout.single_token_variants(emotion):
            untokenizable.append(emotion)
        logits = readout.direction_logits(vector, block, use_jacobian=True)
        rank, variant, prob = readout.rank_of_word(logits, emotion)
        top = readout.decode_top(logits, config.topk)
        verdict = (
            "not-single-token" if rank is None
            else "HIT" if rank < config.topk
            else "near" if rank < NEAR_RANK
            else "MISS"
        )
        rows.append({
            "emotion": emotion,
            "own_word_rank": rank,
            "own_word_variant": variant,
            "own_word_prob": prob,
            "verdict": verdict,
            "top_tokens": [t.token for t in top],
        })
        shown = " ".join(repr(t.token) for t in top[:8])
        rank_text = "n/a" if rank is None else f"{rank}"
        print(f"  {emotion:<16} own word @ {rank_text:>7}  [{verdict:<15}] {shown}")

    scorable = [r for r in rows if r["verdict"] != "not-single-token"]
    hits = [r for r in scorable if r["verdict"] == "HIT"]
    near = [r for r in scorable if r["verdict"] == "near"]
    hit_rate = len(hits) / len(scorable) if scorable else 0.0
    # "rank < topk" means nothing without the size of the vocabulary it is a rank
    # within: a random readout hits at topk/vocab_size. Negligible for a real
    # tokenizer (12 of ~151k) and worth printing so the gate is self-calibrating
    # rather than asking the reader to assume it.
    vocab_size = int(readout.model._lm_head.weight.shape[0])
    chance = config.topk / vocab_size if vocab_size else float("nan")

    print()
    if untokenizable:
        print(f"  excluded as multi-token: {untokenizable}")
        print("    Not misses. The lens decodes one vocabulary token at a time, so a")
        print("    word it cannot spell is a fact about the vocabulary (caveat 1).")
    print(f"  {len(hits)}/{len(scorable)} emotions surfaced their own word in the "
          f"top-{config.topk}; {len(near)} more inside rank {NEAR_RANK}.")
    print(f"  chance hit rate for a random readout: {chance:.2%} "
          f"({config.topk} of {vocab_size:,} tokens)")
    if chance > 0.05:
        print("    NOTE that baseline is not negligible, so this gate is weak here. It")
        print("    assumes a full vocabulary; check what tokenizer is loaded.")
    return {
        "vocab_size": vocab_size,
        "chance_hit_rate": chance,
        "rows": rows,
        "n_scorable": len(scorable),
        "n_hits": len(hits),
        "n_near": len(near),
        "hit_rate": hit_rate,
        "untokenizable": untokenizable,
        "threshold": config.readout_min_self_hit_rate,
        "passed": hit_rate >= config.readout_min_self_hit_rate,
    }


def gate_a_apriori_axes(
    readout, axes: PrincipalAxes, config: PCAJLensConfig, block: int,
    probes: dict, rng,
) -> dict:
    """Read out the hand-built valence and arousal axes as a second calibration.

    Phase 3 saved these precisely so they could be lensed here. They are not a
    result -- they were constructed from the labels -- which is what makes them a
    control: a lens that cannot verbalise a hand-built valence contrast is not
    evidence about PC1 in either direction.
    """
    available = {
        "valence": axes.valence_axis,
        "arousal": axes.arousal_axis,
    }
    out: dict = {}
    if not any(v is not None for v in available.values()):
        print("\n  a priori axes were not saved by Phase 3; skipping the second control.")
        return out

    print()
    print(THIN)
    print("  Control: the hand-built a priori axes, read out at the same block")
    print(THIN)
    for name, axis in available.items():
        if axis is None:
            continue
        for sign, label in ((+1.0, f"+{name}"), (-1.0, f"-{name}")):
            result = read_direction(
                readout, axis * sign, block, label,
                probes["words"], probes["valence"], probes["arousal"], config.topk, rng,
            )
            out[label] = _readout_record(result)
            print(f"    {label:<10} "
                  + " ".join(repr(t) for t in result.token_strings(8)))
            print(f"    {'':<10} ordering AUROC  valence "
                  f"{_fmt_auroc(result.auroc_valence)}  arousal "
                  f"{_fmt_auroc(result.auroc_arousal)}")
    return out


# --------------------------------------------------------------------------- #
# GATE B: the principal components
# --------------------------------------------------------------------------- #

def gate_b_pc_readouts(
    readout, axes: PrincipalAxes, config: PCAJLensConfig, block: int, probes: dict,
    rng,
) -> dict:
    """Read out both ends of each of the top PCs."""
    n_pcs = min(config.n_pcs_to_lens, axes.rank)
    # Two axes scored per exploratory component (PC3 onward). PC1/PC2 are the
    # README's pre-registered predictions and are not part of this family.
    n_exploratory_tests = max(n_pcs - 2, 0) * 2

    print()
    print(RULE)
    print("GATE B  The principal axes, both ends of each")
    print(RULE)
    print("Both ends of each axis are printed, because an axis has two and the token")
    print("lists are what a human reads. The ordering AUROC scores the numbers instead")
    print("-- anchor emotion words ranked in the readout, pleasant against unpleasant,")
    print("0.5 chance, p from shuffling the labels.")
    print()
    print("Note what the second end does NOT add. logits(-v) = -logits(v) exactly, so")
    print("its AUROC is 1 minus the first end's: a 'reversal' is arithmetic, not")
    print("evidence, and only the + end is scored. Its tokens are the real contribution.")
    print()
    print("Phase 3's stability and label correlation are quoted per PC, because a murky")
    print("readout means something different for an axis that survived an independent")
    print("refit than for one that did not.")
    print()
    print("PC1/PC2 are pre-registered -- the README predicts valence and arousal -- and")
    print(f"are read at p<=0.05. PC3+ are exploratory: {n_exploratory_tests} tests in that "
          f"family, so their")
    print(f"threshold is {0.05 / max(n_exploratory_tests, 1):.4f}. About one hit at "
          "p<0.05 is expected among them by chance.")

    results: list[dict] = []
    for i in range(n_pcs):
        ratio = (
            float(axes.explained_variance_ratio[i])
            if i < axes.explained_variance_ratio.size else float("nan")
        )
        stability = axes.phase3_row(i, "pc_stability", "cos_half_a_vs_half_b")
        r_valence = axes.phase3_row(i, "alignment", "r_valence")
        r_arousal = axes.phase3_row(i, "alignment", "r_arousal")

        print()
        print(THIN)
        header = f"PC{i + 1}   {ratio:.1%} of variance"
        if stability is not None:
            header += f"   |   Phase 3: split-half |cos| {stability:.3f}"
        if r_valence is not None and r_arousal is not None:
            header += f", r_valence {r_valence:+.2f}, r_arousal {r_arousal:+.2f}"
        print(header)
        print(THIN)

        ends: dict[str, Readout] = {}
        for sign, label in ((+1.0, f"+PC{i + 1}"), (-1.0, f"-PC{i + 1}")):
            result = read_direction(
                readout, axes.components[i] * sign, block, label,
                probes["words"], probes["valence"], probes["arousal"], config.topk, rng,
            )
            ends[label] = result
            print(f"  {label:<7} "
                  + " ".join(repr(t) for t in result.token_strings(config.topk)))
            print(f"  {'':<7} ordering AUROC  valence {_fmt_auroc(result.auroc_valence)}"
                  f" (p {_fmt_p(result.p_valence)})"
                  f"  arousal {_fmt_auroc(result.auroc_arousal)}"
                  f" (p {_fmt_p(result.p_arousal)})")
            print(f"  {'':<7}                top-1 prob {result.top1_prob:.3f}, spread "
                  f"over ~{result.effective_tokens:,.0f} tokens")
            print(f"  {'':<7} logit lens   "
                  + " ".join(repr(t.token) for t in result.logit_lens_tokens[:8]))
            if result.jaccard_with_logit_lens > 0.9:
                print(f"  {'':<7}   NOTE this is {result.jaccard_with_logit_lens:.0%} the "
                      "same as the plain logit lens: the")
                print(f"  {'':<7}        transport contributed almost nothing at this block.")

        plus, minus = ends[f"+PC{i + 1}"], ends[f"-PC{i + 1}"]
        record = {
            "pc": i + 1,
            "explained_variance_ratio": ratio,
            "phase3_split_half_cosine": stability,
            "phase3_r_valence": r_valence,
            "phase3_r_arousal": r_arousal,
            "plus": _readout_record(plus),
            "minus": _readout_record(minus),
        }
        record.update(_interpret_pc(
            plus, r_valence, r_arousal, config, i, n_exploratory_tests
        ))
        results.append(record)
        for note in record["notes"]:
            for i, line in enumerate(_wrap(note, 72)):
                print(f"  {'->' if i == 0 else '  '} {line}")

    return {"per_pc": results, "n_pcs": n_pcs}


def _interpret_pc(
    plus: Readout,
    r_valence: float | None,
    r_arousal: float | None,
    config: PCAJLensConfig,
    pc_index: int,
    n_exploratory_tests: int,
) -> dict:
    """Strength, significance and the sign cross-check against Phase 3's geometry.

    PC1 and PC2 are **pre-registered**: the README predicts they are valence and
    arousal, so their p-values are read as they are. PC3 onward are exploratory, and
    two axes times however many further components is a family of tests where a
    p <= 0.05 hit is *expected* -- with five PCs it is ten tests and about one false
    positive. Those get a Bonferroni threshold within the exploratory family, which
    is what stops the gate from reporting a chance ordering on PC4 as a discovery.
    Both the raw and the corrected verdicts are recorded either way.
    """
    notes: list[str] = []
    out: dict = {"notes": notes}

    # Only the ``+`` end is scored. ``logits(-v) = -logits(v)`` exactly, so the two
    # ends' AUROCs are arithmetic complements and scoring both would double-count one
    # measurement. What the ``-`` end contributes is its *token list*, which is not
    # determined by the algebra and is what a human reads.
    best_axis, best_strength, best = None, 0.0, (None, None)
    for axis, auroc, p_value in (
        ("valence", plus.auroc_valence, plus.p_valence),
        ("arousal", plus.auroc_arousal, plus.p_arousal),
    ):
        if auroc is None:
            continue
        strength = abs(auroc - 0.5) * 2.0
        out[f"auroc_{axis}"] = auroc
        out[f"auroc_{axis}_minus_end"] = 1.0 - auroc  # recorded, not independent
        out[f"p_{axis}"] = p_value
        if strength > best_strength:
            best_axis, best_strength, best = axis, strength, (auroc, p_value)
    out["best_axis"] = best_axis
    out["best_axis_strength"] = best_strength
    out["n_probes_scored"] = plus.n_probes_scored

    if best_axis is None:
        out["lexicalised"] = False
        notes.append("no probe word was a single token; this PC cannot be scored by "
                     "ordering (caveat 1)")
        return out

    auroc, p_value = best
    strong = best_strength >= (config.readout_min_ordering_auroc - 0.5) * 2
    exploratory = pc_index >= 2
    alpha = 0.05 / max(n_exploratory_tests, 1) if exploratory else 0.05
    significant = p_value is not None and p_value <= alpha
    out["lexicalised"] = bool(strong and significant)
    out["p_best_axis"] = p_value
    out["alpha"] = alpha
    out["exploratory"] = exploratory
    out["significant_uncorrected"] = bool(p_value is not None and p_value <= 0.05)

    if not strong:
        notes.append(
            f"MURKY: the strongest ordering is AUROC {auroc:.2f} on {best_axis} "
            f"(p={p_value:.3f}), short of {config.readout_min_ordering_auroc:.2f}. "
            "Reported as noise rather than interpreted."
        )
    elif not significant and out["significant_uncorrected"]:
        notes.append(
            f"AUROC {auroc:.2f} on {best_axis} at p={p_value:.3f} -- significant on its "
            f"own, but PC{pc_index + 1} is exploratory and this is one of "
            f"{n_exploratory_tests} such tests, where the threshold is "
            f"{alpha:.4f}. About one hit at p<0.05 is expected across them. Not "
            "interpreted; the README's expectation that PC3 onward goes murky is the "
            "honest reading."
        )
    elif not significant:
        notes.append(
            f"AUROC {auroc:.2f} on {best_axis} looks strong but p={p_value:.3f} on "
            f"{plus.n_probes_scored} probe words -- at this sample size that ordering "
            "is within chance. Not interpreted."
        )
    else:
        high = "+" if auroc > 0.5 else "-"
        side = "pleasant" if best_axis == "valence" else "activated"
        notes.append(
            f"reads as {best_axis}: the {high} end ranks the {side} anchors above the "
            f"others, AUROC {max(auroc, 1 - auroc):.2f}, p={p_value:.3f}. The opposite "
            "end is its exact complement by construction, so read its tokens for "
            "whether they are antonyms -- that part is not arithmetic."
        )
        if exploratory:
            # Passing a corrected threshold inside a family is weaker evidence than a
            # pre-registered hit, and saying so is the difference between an honest
            # exploratory finding and one quietly promoted to a result.
            notes.append(
                f"EXPLORATORY: PC{pc_index + 1} was not predicted in advance. It clears "
                f"the family-corrected threshold ({alpha:.4f} over "
                f"{n_exploratory_tests} tests), which is worth following up, but it is "
                "not the same standing as PC1/PC2 and should not be reported as if it "
                "were."
            )

    # The cross-check neither phase can do alone: Phase 3 says where the emotions
    # sit, Phase 4 says what the axis wants to say, and the signs must agree.
    correlation = {"valence": r_valence, "arousal": r_arousal}.get(best_axis)
    if correlation is not None and abs(correlation) > 0.3 and out["lexicalised"]:
        # Phase 3's r says which end of the axis the pleasant (or activated) emotions
        # sit on; Phase 4's AUROC says which end wants to *say* those words. Neither
        # determines the other -- an axis could order the emotions correctly in
        # residual space while its readout pointed the wrong way -- so agreement here
        # is a real cross-check rather than the arithmetic identity between the ends.
        predicted_plus_high = correlation > 0
        observed_plus_high = auroc > 0.5
        agrees = predicted_plus_high == observed_plus_high
        out["sign_agrees_with_phase3"] = bool(agrees)
        if agrees:
            notes.append(
                f"sign check OK: Phase 3 has r_{best_axis} {correlation:+.2f}, so the "
                f"{'+' if predicted_plus_high else '-'} end should read high on "
                f"{best_axis} -- and it does. Two independent measurements agreeing."
            )
        else:
            notes.append(
                f"SIGN MISMATCH: Phase 3 has r_{best_axis} {correlation:+.2f}, which "
                f"predicts the {'+' if predicted_plus_high else '-'} end reads high on "
                f"{best_axis}, but the readout says the opposite. The geometry and the "
                "lexical readout disagree; one of them is wrong. Do not interpret this "
                "axis until it is resolved."
            )
    return out


def _readout_record(result: Readout) -> dict:
    return {
        "name": result.name,
        "tokens": [
            {"rank": t.rank, "token": t.token, "token_id": t.token_id,
             "logit": t.logit, "prob": t.prob}
            for t in result.tokens
        ],
        "logit_lens_tokens": [t.token for t in result.logit_lens_tokens],
        "jaccard_with_logit_lens": result.jaccard_with_logit_lens,
        "auroc_valence": result.auroc_valence,
        "auroc_arousal": result.auroc_arousal,
        "p_valence": result.p_valence,
        "p_arousal": result.p_arousal,
        "n_probes_scored": result.n_probes_scored,
        "top1_prob": result.top1_prob,
        "effective_tokens": result.effective_tokens,
        "probe_ranks": result.probe_ranks,
    }


def _fmt_auroc(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def _fmt_p(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.3f}"


def _single_token(tokenizer, word: str) -> bool:
    """Whether any casing/leading-space variant of ``word`` is a single token.

    Mirrors ``LensReadout.single_token_variants``, which needs a loaded model. This
    needs only the tokenizer, so ``--dry-run`` can report which probe words the lens
    is even able to surface before any GPU time is spent.
    """
    return any(
        len(tokenizer.encode(variant, add_special_tokens=False)) == 1
        for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
    )


# --------------------------------------------------------------------------- #
# Probe words
# --------------------------------------------------------------------------- #

def build_probes(axes: PrincipalAxes) -> dict:
    """Anchor emotion words plus their a priori signs, for the ordering test.

    Derived from the experiment's own design rather than invented here. That
    distinction is the same one Phase 0 makes when it uses Anthropic's published
    prompts: a probe set chosen after seeing the readout would let any token list
    pass as a valence axis.
    """
    words = axes.anchors()
    return {
        "words": words,
        "valence": [axes.label(w, "valence") for w in words],
        "arousal": [axes.label(w, "arousal") for w in words],
        "n_pleasant": sum(1 for w in words if axes.label(w, "valence") > 0),
        "n_unpleasant": sum(1 for w in words if axes.label(w, "valence") < 0),
        "n_activated": sum(1 for w in words if axes.label(w, "arousal") > 0),
        "n_deactivated": sum(1 for w in words if axes.label(w, "arousal") < 0),
    }


# --------------------------------------------------------------------------- #
# Artefacts and gate output
# --------------------------------------------------------------------------- #

def write_readout_table(out_dir: Path, gate_a: dict, gate_b: dict, controls: dict) -> Path:
    """One row per (direction, rank) so every printed token is in a CSV too."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for row in gate_a["rows"]:
        for rank, token in enumerate(row["top_tokens"]):
            rows.append({
                "group": "gate_a_emotion_vector", "direction": row["emotion"],
                "rank": rank, "token": token,
                "own_word_rank": row["own_word_rank"], "verdict": row["verdict"],
            })
    for name, record in controls.items():
        for token in record["tokens"]:
            rows.append({
                "group": "control_apriori_axis", "direction": name,
                "rank": token["rank"], "token": token["token"],
                "logit": token["logit"], "prob": token["prob"],
                "auroc_valence": record["auroc_valence"],
                "auroc_arousal": record["auroc_arousal"],
            })
    for pc in gate_b["per_pc"]:
        for end in ("plus", "minus"):
            record = pc[end]
            for token in record["tokens"]:
                rows.append({
                    "group": "gate_b_pc", "direction": record["name"],
                    "pc": pc["pc"], "rank": token["rank"], "token": token["token"],
                    "logit": token["logit"], "prob": token["prob"],
                    "explained_variance_ratio": pc["explained_variance_ratio"],
                    "auroc_valence": record["auroc_valence"],
                    "auroc_arousal": record["auroc_arousal"],
                    "phase3_split_half_cosine": pc["phase3_split_half_cosine"],
                })
    path = out_dir / "phase4_readouts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def print_header(config: PCAJLensConfig, axes: PrincipalAxes, out_dir: Path) -> None:
    print(RULE)
    print(f"PHASE 4 GATE -- J-lens the principal components   run '{config.run_name}'")
    print(RULE)
    print(f"model   : {config.model_name} ({config.dtype})")
    print(f"pcs     : {axes.pcs_path}")
    print(f"outputs : {out_dir}")
    print()
    print("Collects no activations and fits no J. The lens is the pre-fitted one Phase")
    print("0 verified; a readout is a disposition to say a token, not evidence about")
    print("what the model feels (caveat 2).")
    print()


def print_pc_source(axes: PrincipalAxes, config: PCAJLensConfig) -> None:
    print(RULE)
    print("STEP 1  What Phase 3 handed over")
    print(RULE)
    described = (
        jlens_lens.describe_block(axes.target_block, axes.n_layers)
        if axes.n_layers else f"block {axes.target_block}"
    )
    print(f"components     : {axes.rank} x {axes.d_model}")
    print(f"fitted at      : {described}")
    print("                 taken from the artefact, not from config.target_block: the")
    print("                 PCs live at the block whose vectors they were fitted on, and")
    print("                 lensing them anywhere else transports the wrong thing.")
    if config.target_block is not None and config.target_block != axes.target_block:
        print(f"  MISMATCH     config.target_block={config.target_block} differs from the "
              f"artefact's {axes.target_block}.")
        print("               The artefact wins. Re-run Phase 2 and 3 if you meant to move.")
    print(f"emotions       : {len(axes.fit_emotions)} fitted"
          + (f", {len(axes.emotions) - len(axes.fit_emotions)} projected only"
             if len(axes.emotions) > len(axes.fit_emotions) else ""))
    ratios = axes.explained_variance_ratio
    if ratios.size >= 2:
        print(f"variance       : PC1 {ratios[0]:.1%}, PC2 {ratios[1]:.1%}, "
              f"top-2 {ratios[:2].sum():.1%}")
    stability = axes.metadata.get("pc_stability", {})
    crossfit = axes.metadata.get("alignment", {}).get("crossfit", {})
    if stability:
        print(f"Phase 3 gate   : top-2 plane stable "
              f"{stability.get('plane_stable')}, axes identified "
              f"{stability.get('axes_identified')}")
    if crossfit.get("available"):
        print(f"                 cross-fit circumplex alignment "
              f"{crossfit['plane_mean_cosine']:.3f}")


def print_lens_summary(lens_report: dict, description, problems: list[str],
                       warnings: list[str], axes: PrincipalAxes) -> None:
    print()
    print(RULE)
    print("STEP 2  The lens, and whether it is the right one")
    print(RULE)
    print(f"source          : {lens_report['source']}")
    print(f"path            : {lens_report['path']}")
    if lens_report["source"] == "published":
        print(f"repo/subfolder  : {lens_report['repo']}/{lens_report['subfolder']}")
        print(f"fitted on       : {lens_report.get('fitted_on')}  "
              f"(verified: {lens_report['hf_model_name_verified']})")
    else:
        print(f"                  {lens_report['note']}")
    print(f"d_model         : {description.d_model}   (PCs: {axes.d_model})")
    print(f"fitted blocks   : {description.source_layers[0]}.."
          f"{description.source_layers[-1]}   (PCs at {axes.target_block})")
    print(f"prompts fitted  : {description.n_prompts}")
    if lens_report.get("recorded_final_mean_rel_change") is not None:
        print(f"recorded change : {lens_report['recorded_final_mean_rel_change']} "
              "(from the fit's own config.yaml)")
    print()
    if problems:
        print("  MISMATCH:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  OK  lens d_model, J shape and block coverage all fit these PCs.")
    for warning in warnings:
        print()
        print("  WARNING:")
        for line in _wrap(warning, 72):
            print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


def print_probe_summary(probes: dict, single_token: dict | None) -> None:
    print()
    print(RULE)
    print("STEP 3  Probe words for the ordering test")
    print(RULE)
    print(f"anchors: {len(probes['words'])} words -- "
          f"{probes['n_pleasant']} pleasant / {probes['n_unpleasant']} unpleasant, "
          f"{probes['n_activated']} activated / {probes['n_deactivated']} deactivated")
    print("These come from the Phase 1 circumplex design, not from looking at a")
    print("readout first. A probe set chosen afterwards would let any token list pass.")
    if single_token is not None:
        usable = [w for w, ok in single_token.items() if ok]
        missing = [w for w, ok in single_token.items() if not ok]
        print(f"single-token   : {len(usable)}/{len(single_token)} usable")
        if missing:
            print(f"  not a single token: {missing}")
            print("  Excluded from the ordering test, not scored as misses (caveat 1).")


def print_verdict(
    gate_a: dict, gate_b: dict, scale: dict, warnings: list[str],
    config: PCAJLensConfig, artifacts: dict[str, object],
) -> bool:
    lexicalised = [pc for pc in gate_b["per_pc"] if pc.get("lexicalised")]
    chance_hits = [
        pc for pc in gate_b["per_pc"]
        if pc.get("exploratory") and pc.get("significant_uncorrected")
        and not pc.get("lexicalised")
    ]
    mismatches = [pc for pc in gate_b["per_pc"] if pc.get("sign_agrees_with_phase3") is False]
    print()
    print(RULE)
    print("PHASE 4 VERDICT")
    print(RULE)
    print(f"  GATE A lens usable   : {'PASS' if gate_a['passed'] else 'REVIEW'}  "
          f"({gate_a['n_hits']}/{gate_a['n_scorable']} emotions surfaced their own word "
          f"in the top-{config.topk}, threshold {config.readout_min_self_hit_rate:.0%})")
    confirmatory = [pc for pc in lexicalised if not pc.get("exploratory")]
    exploratory = [pc for pc in lexicalised if pc.get("exploratory")]
    n_pre = min(2, gate_b["n_pcs"])
    print(f"  GATE B PCs read out  : {len(confirmatory)}/{n_pre} pre-registered "
          f"(PC1-PC{n_pre}) order the anchor words beyond")
    print(f"                         chance (AUROC >= "
          f"{config.readout_min_ordering_auroc:.2f}, p <= 0.05)"
          + (f"; {len(exploratory)} exploratory also clear their"
             if exploratory else ""))
    if exploratory:
        print("                         corrected threshold -- follow-up, not a result.")
    for pc in gate_b["per_pc"]:
        axis = pc.get("best_axis") or "unscored"
        mark = "lexicalised" if pc.get("lexicalised") else "murky"
        if pc.get("lexicalised") and pc.get("exploratory"):
            mark = "exploratory"
        detail = ""
        if pc.get("best_axis") and pc.get("p_best_axis") is not None:
            plus_auroc = pc[f"auroc_{pc['best_axis']}"]
            # Report the end that achieves the ordering, not always the + end: the two
            # are complements, so a bare "AUROC 0.00" next to "lexicalised" reads as a
            # failure when it is the - end scoring 1.00.
            end = "+" if plus_auroc >= 0.5 else "-"
            detail = (f", AUROC {max(plus_auroc, 1 - plus_auroc):.2f} at the {end} end, "
                      f"p={pc['p_best_axis']:.3f}")
        print(f"      PC{pc['pc']}  {pc['explained_variance_ratio']:>6.1%}  "
              f"{mark:<12} best axis {axis}{detail}")
    if chance_hits:
        print("      "
              + ", ".join(f"PC{p['pc']}" for p in chance_hits)
              + " reached p<=0.05 uncorrected but are exploratory; that is")
        print("      the expected chance yield across those tests, not a readout.")
    print(f"  sign cross-check     : "
          f"{'MISMATCH on ' + ', '.join('PC' + str(p['pc']) for p in mismatches) if mismatches else 'no disagreement with Phase 3'}")
    print(f"  scale invariance     : "
          f"{'OK' if scale.get('identical_topk') else 'FAILED -- +/-PC is not well defined'}")
    print()
    for label, path in artifacts.items():
        print(f"  {label:<9}: {path}" if label else f"  {'':<9}  {path}")
    print()
    if not gate_a["passed"]:
        print("  GATE A did not pass, so GATE B is not interpretable. A weak readout")
        print("  here is ambiguous between 'the PC is not lexicalised' and 'this lens")
        print("  cannot verbalise anything at this block', and the tokens above cannot")
        print("  distinguish them. Before reading GATE B as a result:")
        print("    python run.py refit_lens --dry-run     # cost of a converged lens")
        print("    python run.py phase0                   # re-verify with it")
        print()
    if warnings:
        print("  The lens warnings in STEP 2 bound every claim below them. Read them")
        print("  again before quoting a token list.")
        print()
    print("  What a readout licenses. 'PC1's minus end reads pleasant' means the model")
    print("  is disposed to say pleasant words when that direction is added -- not that")
    print("  it feels pleasant (caveat 2). And a PC scoring murky may be real but not")
    print("  lexicalised: absence of a readout is not absence of structure (caveat 1).")
    print()
    print("STOPPING at the Phase 4 gate, as agreed. Nothing downstream has run.")
    print(RULE)
    return bool(gate_a["passed"] and lexicalised and not mismatches)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    cache_dir = paths.hf_cache_dir()
    pcs_path = args.pcs or config.pcs_path
    meta_path = (
        pcs_path.with_name(config.pcs_meta_path.name) if args.pcs else config.pcs_meta_path
    )
    out_dir = pcs_path.parent

    axes = read_pcs(pcs_path, meta_path)
    print_header(config, axes, out_dir)
    print_pc_source(axes, config)

    # --- the lens ---------------------------------------------------------- #
    lens_path, lens_report = resolve_lens(config, cache_dir)
    print()
    print("reading the lens checkpoint (loads it into host RAM) ...")
    description = jlens_lens.describe_lens_checkpoint(lens_path)
    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )
    problems, warnings = crosscheck_lens(description, axes, arch)
    print_lens_summary(lens_report, description, problems, warnings, axes)

    probes = build_probes(axes)
    block = axes.target_block

    sections: dict = {
        "run": {"stage": "phase4_lens_pcs", "run_name": config.run_name,
                "dry_run": args.dry_run, "output_dir": str(out_dir)},
        "config": config.to_dict(),
        "pcs": {
            "path": str(pcs_path),
            "metadata_path": str(meta_path),
            "target_block": block,
            "target_hidden_state": axes.target_hidden_state,
            "rank": axes.rank,
            "d_model": axes.d_model,
            "n_fitted_emotions": len(axes.fit_emotions),
            "phase3_source": axes.metadata.get("source", {}),
        },
        "lens": {
            **lens_report,
            "d_model": description.d_model,
            "n_prompts": description.n_prompts,
            "fitted_block_range": [description.source_layers[0],
                                   description.source_layers[-1]],
            "problems": problems,
            "warnings": warnings,
        },
        "probes": {k: v for k, v in probes.items()},
    }

    # --- dry run ----------------------------------------------------------- #
    if args.dry_run:
        single_token = None
        try:
            tokenizer = model_utils.load_tokenizer(
                config.model_name, config.model_revision, cache_dir,
                trust_remote_code=config.trust_remote_code,
            )
            # Which probe words the lens could surface at all, before any GPU time.
            single_token = {w: _single_token(tokenizer, w) for w in probes["words"]}
        except Exception as exc:
            print(f"\nWARNING tokenizer unavailable ({exc}); "
                  "cannot report single-token coverage")
        print_probe_summary(probes, single_token)
        sections["probes"]["single_token"] = single_token
        txt_path, json_path = provenance.write_run_record(
            out_dir / "dry_run",
            title=f"PHASE 4 DRY RUN -- {config.run_name}",
            sections=sections,
            txt_name="phase4_dry_run.txt", json_name="phase4_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: lens and PCs cross-checked; no weights loaded.")
        print(RULE)
        print(f"  lens        : {lens_path}")
        print(f"  PCs at      : block {block} (hidden state {axes.target_hidden_state})")
        print(f"  records     : {txt_path}")
        print(f"                {json_path}")
        if problems:
            print("\n  MISMATCH above must be fixed first; the gate would read out nonsense.")
        print(RULE)
        return 0 if not problems else 3

    if problems:
        print("\nABORTED: the lens does not fit these PCs (see MISMATCH above).",
              file=sys.stderr)
        return 3

    # --- load the model ---------------------------------------------------- #
    print()
    print(RULE)
    print(f"Loading {config.model_name} ({config.dtype}) ...")
    print(RULE)
    t0 = time.time()
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
    print(f"  weights loaded in {time.time() - t0:.0f}s")
    readout = jlens_lens.LensReadout.build(hf_model, tokenizer, lens_path)
    print(f"  lens loaded: {len(readout.source_layers)} fitted blocks")
    print(f"  readout formula: {readout.unembed_description()['formula']}")

    single_token = {
        word: bool(readout.single_token_variants(word)) for word in probes["words"]
    }
    print_probe_summary(probes, single_token)
    sections["probes"]["single_token"] = single_token

    # ``+PC`` and ``-PC`` are only well defined if magnitude does not matter.
    scale = verify_scale_invariance(readout, axes.components[0], block)
    print()
    print(f"  scale invariance on PC1: identical top-5 at |v|=1 and |v|=250: "
          f"{scale['identical_topk']}")
    if not scale["identical_topk"]:
        print(f"    |v|=1   {scale['unit']}")
        print(f"    |v|=250 {scale['scaled']}")
        print("    WARNING the readout depends on magnitude, so +/-PC is not well")
        print("    defined without choosing a step size. Investigate before reading on.")

    # --- gates ------------------------------------------------------------- #
    rng = rng_for(config.seed, "phase4_ordering")
    gate_a = gate_a_self_readout(readout, axes, config, block)
    controls = gate_a_apriori_axes(readout, axes, config, block, probes, rng)
    gate_b = gate_b_pc_readouts(readout, axes, config, block, probes, rng)

    sections["scale_invariance"] = scale
    sections["gate_a"] = gate_a
    sections["controls_apriori_axes"] = controls
    sections["gate_b"] = gate_b

    table_path = write_readout_table(out_dir, gate_a, gate_b, controls)
    txt_path, json_path = provenance.write_run_record(
        out_dir,
        title=f"PHASE 4 GATE -- {config.run_name}",
        sections=sections,
        txt_name="phase4_gate.txt", json_name="phase4_gate.json",
    )

    print_verdict(
        gate_a, gate_b, scale, warnings, config,
        artifacts={
            "readouts": table_path,
            "records": txt_path,
            "": json_path,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
