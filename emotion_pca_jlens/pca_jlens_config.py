"""Configuration for the emotion-space PCA + J-lens experiment.

Separate from :mod:`extract_emotion_vectors.vector_extraction_config` on purpose.
That config drives the mean-difference-direction pipeline; this one drives a
different analysis (PCA across emotion centroids, then J-lens readout of the
principal axes) on a different model, and conflating them would make either one
harder to change. The two share :mod:`core` and the ``--set field=value``
override convention.

Override on the command line::

    python run.py phase0 --set target_block=40 --set topk=20
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from core import paths
from core.judge import DEFAULT_JUDGE_MODEL

# The ``--set`` coercion rules are mostly identical for both configs, so the
# implementation is imported rather than duplicated. Deliberately the private
# name: it is one repo, and a second copy of 40 lines of type coercion would be
# the thing that drifts.
from extract_emotion_vectors.vector_extraction_config import _coerce


def _coerce_field(key: str, value: object, current: object) -> object:
    """Coerce one ``--set`` value, patching the fields ``_coerce`` cannot infer.

    ``_coerce`` reads the target type off the field's *current* value, which gets
    five fields here wrong. Every one of them fails *silently* rather than
    raising, which is why they are handled up front instead of being discovered
    mid-run:

    * ``target_block`` defaults to ``None``, and ``_coerce`` only parses an int out
      of a ``None`` field for two hardcoded names from the other config. Without
      this, ``--set target_block=40`` yields the string ``"40"`` and ``validate()``
      then raises on ``"40" < 0``.
    * ``gate_blocks`` is polymorphic (a spec string *or* a list), the same shape as
      ``layer_spec``, which ``_coerce`` special-cases by name.
    * ``layer_spec`` is polymorphic in exactly the same way. ``_coerce`` happens to
      special-case this *name* -- but that list (``layer_spec``,
      ``eval_layer_spec``) was written for the other config's fields, so relying on
      it here is an accident waiting to be tidied away. Without a branch,
      ``--set layer_spec=[1,32,63]`` would reach ``resolve_layers`` as a string.
    * ``emotions`` accepts the sentinel ``"all"``, which the shared helper's
      comma split would turn into ``["all"]`` -- failing much later with
      "emotion not present in the dataset". ``perspective_emotions`` and
      ``within_emotion_targets`` are the same shape and default to ``None``, where
      ``_coerce`` would hand back the raw string.
    * ``sweep_blocks`` and ``clamp_blocks`` are polymorphic like ``gate_blocks``.
    * ``dict_atom_counts`` is a tuple of ints that also accepts a comma list, so
      ``--set dict_atom_counts=16,25`` has to parse rather than becoming one string.
    * ``steer_strengths`` and ``clamp_strengths`` are tuples of floats that also accept
      a JSON list, which the shared helper's tuple branch does not handle.
    * ``specificity_topic`` defaults to ``None``, so "none" has to mean ``None``.
    * ``r2_sync`` must reject typos rather than defaulting them to "off"; see below.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if key == "target_block":
        return None if text.lower() in ("none", "null", "") else int(text)
    if key in ("gate_blocks", "layer_spec", "sweep_blocks", "clamp_blocks"):
        return json.loads(text) if text.startswith("[") else text
    if key == "dict_atom_counts":
        parsed = json.loads(text) if text.startswith("[") else text.split(",")
        return tuple(int(part) for part in parsed if str(part).strip() != "")
    if key in ("steer_strengths", "clamp_strengths"):
        # A tuple of floats. _coerce would infer the element type from the current
        # value and cope, but only for the comma form -- `--set steer_strengths=[0,1]`
        # would reach validate() as the string "[0,1]" and compare > 0 against a str.
        parsed = json.loads(text) if text.startswith("[") else text.split(",")
        return tuple(float(part) for part in parsed if str(part).strip() != "")
    if key == "specificity_topic":
        # Defaults to None, where _coerce hands back the raw string -- fine for a str
        # field, but "none" must mean None rather than the literal topic "none".
        return None if text.lower() in ("none", "null", "") else text
    if key in ("emotions", "perspective_emotions", "within_emotion_targets",
               "channel_emotions"):
        # Polymorphic: the sentinel "all", None, or a list. Handled here because
        # the shared helper special-cases this field name into a comma split, which
        # would silently turn "all" into the one-element list ["all"] and then fail
        # much later with "emotion not present in the dataset".
        if text.lower() in ("none", "null", ""):
            return None
        if text.lower() == "all":
            return "all"
        if text.startswith("["):
            return json.loads(text)
        return [part.strip() for part in text.split(",") if part.strip()]
    if key == "r2_sync":
        # Strict, unlike the shared helper, which maps *any* unrecognised string to
        # False. That is the wrong default here: `--set r2_sync=ture` would then
        # silently disable the activation mirror and a pod teardown would take the
        # activations with it. A typo must be an error, not a quiet "off".
        lowered = text.lower()
        if lowered == "auto":
            return "auto"
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"r2_sync={value!r} is not recognised. Use true, false, or auto. "
            "Refusing to guess: an unrecognised value would silently stop "
            "activations being mirrored to R2."
        )
    return _coerce(key, value, current)


@dataclass
class PCAJLensConfig:
    """Parameters for one run of the PCA + J-lens experiment.

    Only the fields Phase 0 needs are load-bearing today; the extraction and PCA
    fields are here so later phases do not force a config reshuffle. Fields whose
    value is a real research choice rather than plumbing say so.
    """

    # ---------------------------------------------------------------- run --- #
    run_name: str = "qwen3-32b_pca-jlens"
    """Output directory under ``outputs/``."""

    seed: int = 0

    # -------------------------------------------------------------- model --- #
    model_name: str = "Qwen/Qwen3-32B"
    """Must be a checkpoint that ``neuronpedia/jacobian-lens`` has a lens for.

    ``Qwen/Qwen3-32B`` is Qwen3's *post-trained* (instruct) release; the base
    model is the separate ``Qwen/Qwen3-32B-Base``, which has no published lens.
    The lens is a function of the weights, so it cannot be moved between the two
    -- :func:`core.jlens_lens.resolve_lens_artifact` refuses the mismatch rather
    than producing plausible nonsense.
    """

    model_revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    quantization: str | None = None
    """Leave off. Quantisation perturbs the residual stream the lens was fitted
    against, so the transport would no longer match the model."""

    # --------------------------------------------------------------- lens --- #
    lens_repo: str = "neuronpedia/jacobian-lens"
    lens_revision: str | None = None

    lens_subfolder: str | None = None
    """Force a subfolder of ``lens_repo``. ``None`` resolves it from
    ``model_name`` and verifies against the fit's recorded ``hf_model_name``."""

    lens_local_path: str | None = None
    """Use a lens file on disk instead of the published one.

    For the output of ``run.py refit_lens``, which extends the published
    (interrupted, 80-prompt) fit with more prompts. Skips resolution and download
    entirely, so the ``hf_model_name`` guard does not apply -- it is on you to
    point this at a lens fitted on ``model_name``.
    """

    # -------------------------------------------------------------- gates --- #
    gate_blocks: object = "evenly_spaced:10"
    """Blocks to read out at in the Phase 0 concept gate.

    Same spec grammar as :func:`core.model_utils.resolve_layers`, but resolved
    over the lens's fitted blocks (``0 .. n_layers-2``), not over hidden states.
    A sweep rather than one layer: the documented signature of a working lens is
    that the implied concept *peaks in the middle* and gives way to the answer
    token late, so a single layer cannot show it.
    """

    topk: int = 12
    """Vocabulary tokens to print per readout."""

    gate_max_seq_len: int = 512

    identity_check_blocks: int = 6
    """How many of the highest fitted blocks to report identity distance for in
    full. All blocks are still summarised."""

    # ------------------------------------------------------------ stimuli --- #
    emotions: object = None
    """Which emotions to extract.

    ``None``   the balanced 16-emotion :data:`DEFAULT_CIRCUMPLEX_SET`
    ``"all"``  all 171 words in ``data/emotions_171.txt``
    a list     an explicit set, e.g. ``--set emotions=joyful,sad,calm``

    ``"all"`` exists because of a real statistical limit of the 16-emotion design:
    after mean-centring, ``n`` emotion centroids span a space of rank ``n-1``, so
    16 emotions in 5120 dimensions make PC1/PC2 explain a large variance fraction
    almost by construction. That number only becomes interpretable with many more
    emotions. The intended use is therefore to fit PCA on all 171 and validate
    against the 16 a priori-labelled anchors, which stay balanced and orthogonal;
    words outside the circumplex table are carried as ``unlabelled`` and excluded
    from the alignment score rather than being hand-labelled here.
    """

    stories_per_emotion: int = 400
    """Stories per emotion, stratified across topics.

    A multiple of the dataset's 100 topics on purpose: 400 gives *exactly* 4
    stories per topic for every emotion, so the stimulus set is topic-matched with
    no remainder. A non-multiple leaves a remainder that
    ``dataset.subsample_per_emotion`` scatters over a random subset of topics,
    which breaks exact matching for no benefit. The paper's own check found ~300
    already enough for stable centroids; Phase 2's split-half gate is what
    actually decides.
    """

    neutral_stories: int = 400
    include_neutral: bool = True
    """Include the 1,200 emotionally neutral stories (same 100 topics) as a
    ``neutral`` row group. It is the circumplex origin, and the brief's "plus
    neutral" point."""

    dataset_revision: str | None = None

    split_seed: int = 12345
    split_proportions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    """Topic-level split. Recorded on every row, but Phase 3's PCA uses *all*
    splits: nothing is fitted and then evaluated, so there is nothing to hold out.
    The split matters to Phase 2, which halves by *topic* for its reliability
    check -- splitting by story would leak a scenario across halves and inflate
    the cosine."""

    gate_examples_per_emotion: int = 3
    """Stimuli printed per emotion at the Phase 1 gate."""

    # --------------------------------------------------------- extraction --- #
    target_block: int | None = None
    """Residual block whose pooled activation becomes the emotion vector (Phase 2).

    ``None`` means "middle of the stack", resolved once the model's depth is
    known. Block indexing throughout this experiment is the J-lens convention --
    the output of residual block ``i`` -- because the vectors must be readable by
    the lens. See :mod:`core.jlens_lens` for the conversion to this repo's
    hidden-state indexing.

    Phase 2 rejects a block outside the lens's fitted range (``0 .. n_layers-2``),
    which cannot be checked here because it needs the model's depth. A vector the
    lens cannot read is useless to Phase 4 onward.

    Deliberately *not* part of the activation fingerprint: every hidden state is
    stored, so changing this re-derives the vectors from existing chunks instead of
    forcing a re-extraction. That is what makes the Phase 5 layer sweep free.
    """

    layer_spec: object = "all"
    """Which hidden states to store (Phase 2). Same grammar as
    :func:`core.model_utils.resolve_layers`, and indexed over **hidden states**
    (``0`` = embedding output), not blocks -- it is handed straight to
    ``core.pooling``.

    ``"all"`` because ``output_hidden_states=True`` materialises every layer
    anyway: keeping them costs disk (~4.4 GiB for the 16-emotion run, ~22 GiB for
    171) but no extra compute, and it turns the Phase 5 layer sweep into a re-read
    rather than another pass of a 32B model. Narrowing this is a false economy
    unless disk is the binding constraint -- and it must still include the target
    block's hidden state, which Phase 2 checks.
    """

    token_offset: int = 50
    min_pooled_tokens: int = 10
    max_length: int = 512
    """Stories run 93-170 words (~125-230 tokens), so this truncates nothing; it
    only bounds worst-case activation memory."""

    add_special_tokens: bool = True
    """Tokenizer default. Note a BOS token counts as a real token and therefore
    against ``token_offset``."""

    batch_size: int = 8
    use_chat_template: bool = False
    """``False`` = activations over raw story text, matching how the lens itself
    was fitted (raw WikiText). An instruct checkpoint does not require its
    template for a forward pass.

    Phases 7-9 will need chat-formatted prompts for the behaviour channel; that
    boundary is an assumption about steering-vector transfer, not a verified
    property, and belongs in the writeup rather than being smoothed over here."""

    activation_dtype: str = "bfloat16"
    """Storage dtype for pooled activations. Pooling always reduces in fp32 (see
    :mod:`core.pooling`); only the saved value is cast. bf16 halves the footprint
    and is far inside the noise of a 400-stimulus mean.

    The Phase 2 emotion vectors are accumulated from these *stored* values rather
    than from the pre-cast fp32 ones, so re-deriving a vector from the chunks gives
    the same answer as the run that wrote them."""

    chunk_size: int = 512
    """Stimuli per activation chunk file. At 65 hidden states x 5120 dims x bf16
    that is ~341 MiB/chunk; also the granularity of resume and R2 upload."""

    log_every_batches: int = 20

    split_half_min_cosine: float = 0.90
    """Phase 2 gate threshold on the *mean-centred* split-half cosine.

    0.9 is the brief's "trustworthy" line; ~0.6 means more stimuli are needed
    before PCA can mean anything. Compared on mean-centred vectors because that is
    what Phase 3 consumes -- raw pooled residuals share a large layer-wide common
    component, so their split-half cosine sits near 0.999 for every emotion and
    cannot fail. A summary, not the judgement: read the per-emotion table.
    """

    # ------------------------------------------------------------ storage --- #
    r2_sync: object = True
    """Mirror activation chunks to Cloudflare R2 as they are written.

    ``True``, not the other pipeline's ``"auto"``: this experiment's activations
    always belong in R2. ``"auto"`` would silently keep them on an ephemeral pod's
    disk whenever the estimated size fell under a threshold, and "silently local"
    is the failure mode that loses a run. With ``True``,
    ``extract_activations``-style staging aborts up front when credentials are
    missing rather than discovering it after the GPU time is spent.
    """

    r2_threshold_gib: float = 5.0
    """Size threshold for ``r2_sync="auto"`` only. Unused at the ``True`` default,
    and kept so that opting into ``"auto"`` does not silently mean "never"."""

    r2_root: str = "pca-jlens-activations"
    """Top-level folder in the bucket for this experiment family.

    Deliberately its own folder, separate from the mean-difference pipeline's
    ``story-activations/``. Different model (Qwen3-32B, not Qwen2.5-32B),
    different layer-index convention (residual blocks, not hidden states) and a
    different analysis, so the two must not share a prefix -- a resumed run keys
    off the prefix and would otherwise try to reuse incompatible vectors.
    """

    r2_prefix: str | None = None
    """Full bucket key prefix, overriding ``r2_root``/``run_name``.
    ``None`` -> ``<r2_root>/<run_name>``."""

    delete_local_after_sync: bool = True
    """Delete each local ``.safetensors`` chunk once its upload is *verified*
    (object present, size matches). Index parquets stay local so a resumed run
    knows what is done without touching R2. See ``core.r2.make_chunk_uploader``:
    nothing is ever deleted without a confirmed remote copy."""

    # ---------------------------------------------------------------- pca --- #
    n_pcs_to_lens: int = 5
    """Principal components Phase 4 reads out. Phase 3 saves every component it
    fits, so raising this needs no re-fit."""

    readout_min_self_hit_rate: float = 0.50
    """Phase 4 GATE A: fraction of emotions that must surface their *own* word in
    the top-``topk`` of their own vector's readout.

    The calibration the PC readouts depend on. A murky PC has two causes that its
    tokens cannot distinguish -- the axis is real but not lexicalised, or the lens
    cannot verbalise anything at this block -- and the published qwen3-32b lens is
    an interrupted fit, so the second is live. If the lens cannot find "sad" in the
    sad vector, nothing in GATE B is evidence either way.
    """

    readout_min_ordering_auroc: float = 0.75
    """Phase 4 GATE B: AUROC one end of a PC must reach, ranking the anchor emotion
    words pleasant-against-unpleasant (or activated-against-deactivated), for the
    axis to count as lexicalised rather than murky.

    An ordering test rather than reading the token list, because "these tokens look
    positive" is not a measurement. 0.5 is chance; the opposite end of a real axis
    should land symmetrically below it.
    """

    n_pcs_to_report: int = 10
    """Rows in the Phase 3 variance-explained table. Capped at the rank, which is
    ``n_emotions - 1`` after centring."""

    mean_center: bool = True
    """Centre across emotion centroids before PCA. The load-bearing choice in
    Phase 3: without it, PC1 is overall affect magnitude and the circumplex
    cannot appear. Settable only so the gate can *show* what goes wrong."""

    include_neutral_in_pca: bool = False
    """Fit the PCA on the neutral centroid as well as the emotions.

    Off by default, and the reason is the same one that makes centring
    load-bearing. Neutral is the circumplex *origin*: including it adds a point
    that differs from the emotions along "how much affect is present at all", and
    PC1 would partly become that contrast -- the very thing centring exists to
    remove. Left off, neutral is instead *projected* into the fitted space as a
    reference point, where landing near the origin is a check that the centring
    did what it claims.
    """

    pca_null_samples: int = 200
    """Random-direction resamples for the variance-explained null band.

    Needed because ``n`` centroids span rank ``n-1``, so PC1/PC2 explain a large
    share almost by construction at n=16. The null answers "large compared to
    what" for this n and d; it does not make the circumplex claim.
    """

    pc_stability_min_cosine: float = 0.80
    """Gate threshold on |cosine| between a PC fitted on each half of the topics.

    Phase 2 saves both half-vectors, so the top PCs can be refitted independently
    per half at no cost. This is the check that separates real covariance
    structure from noise, and it is stricter evidence than variance explained --
    which cannot fail at small n.
    """

    # ------------------------------------------------- phase 5 extensions --- #
    sweep_blocks: object = "evenly_spaced:13"
    """Blocks for the Phase 5 layer sweep. Same grammar as ``gate_blocks``, resolved
    over the lens's fitted blocks so a swept block can also be lensed.

    A spec rather than every block: the PCA itself is milliseconds, but each block
    re-reads every stored activation, so 63 blocks means 63 passes over the chunks.
    13 is enough to see a curve.
    """

    perspective_emotions: object = None
    """Emotions for the Phase 5 perspective contrast. ``None`` uses the first four
    of the circumplex set (one per quadrant), which keeps the extraction to minutes
    while spanning affect."""

    perspective_stories_per_emotion: int = 200
    """Stories per emotion per framing. Both framings reuse the *same* stories, so
    the contrast is paired and topic drops out of the difference."""

    perspective_self_frame: str = "What follows happened to you.\n\n"
    perspective_other_frame: str = "What follows happened to them.\n\n"
    """Framing prefixes for the self/other contrast.

    They differ in exactly one word and **must tokenize to the same length**. A
    longer prefix shifts the ``token_offset`` window deeper into the story, so an
    unmatched pair would make the contrast partly "which part of the story was
    pooled" rather than who it happened to. Phase 5 checks this against the real
    tokenizer and refuses a mismatch instead of reporting the confound as a result.
    """

    within_emotion_targets: object = None
    """Emotions for the Phase 5 within-emotion contrast. ``None`` uses two from
    opposite quadrants; ``"all"`` runs every one."""

    within_emotion_n_pcs: int = 5

    neutral_variance_threshold: float = 0.50
    """Neutral variance the projected-out PCs must explain when
    ``remove_neutral_pcs`` is on. The mean-difference pipeline's value, so the two
    experiments' robustness checks mean the same thing."""

    # ------------------------------------------------ phase 6 decomposition - #
    dict_atom_counts: object = (16, 25)
    """Atom counts ``k`` the gate reports, both of them. The paper's settings.

    Reported as a pair rather than picked, because under sparse nonnegative coding the
    reconstructable set is a union of cones, not a subspace -- so the "reportable
    fraction" is a function of ``k`` and quoting one value hides that. Two values make
    the dependence visible in the table."""

    n_dict_atoms: int = 16
    """The ``k`` the *saved* ``v_J`` / ``v_perp`` tensors use, and the one Phase 8 steers
    with. Must be one of :attr:`dict_atom_counts`; recorded in the sidecar so a reader
    cannot mistake which ``k`` the artefact is at."""

    dict_pool_size: int = 512
    """Candidate atoms considered, taken as the top-N tokens of the vector's own lens
    readout.

    That restriction is the definition of the reportable part, not a shortcut: ``v_J``
    is meant to be what the lens can express *as this vector's tokens*. A larger pool
    admits atoms that reconstruct ``v`` better while being tokens ``v`` is not
    disposed to say, which would inflate the reportable fraction with words the model
    would never produce.
    """

    pursuit_steps: int = 300
    """Projected-gradient iterations per coefficient update in the pursuit. The step
    size is 1/L from the Gram matrix's spectral norm, so this is not a tuning knob --
    it only has to be enough to converge."""

    write_space: bool = False
    """Ablation: build atoms from ``J^+`` (write directions) instead of ``J^T`` (read).

    Off by default, because reportability is a **reading** question. The lens score for
    token ``t`` is ``u_t' J h = (J^T u_t)' h``, so ``J^T u_t`` *is* the measurement weight
    vector the lens reads ``t`` with, and anything orthogonal to every ``J^T u_t`` moves
    no logit at all. ``J^+ u_t`` is a different direction: the one that most efficiently
    *writes* token ``t``. Both are real questions; only the first is "how much of this
    vector can the lens see".

    On, the whole gate re-runs against write directions and every line of output is
    labelled ``write_space``. It is kept because the write question is interesting on its
    own -- which part of an emotion vector would most efficiently *produce* its words --
    and because keeping it runnable is what stops the two from being conflated again.
    """

    dict_pinv_rcond: float = 1e-2
    """Relative singular-value cutoff for the pseudo-inverse of ``J``. **Used only when
    :attr:`write_space` is set**; the default read-direction path needs no inverse.

    ``J^+`` amplifies each right-singular direction by ``1/s``, so the directions ``J``
    nearly annihilates dominate every atom, and because atoms are unit-normalised
    afterwards the amplification does not wash out -- it becomes the atom. Truncating at
    ``dict_pinv_rcond * s_max`` (numpy's ``pinv`` convention) bounds that amplification at
    ``1 / dict_pinv_rcond`` relative to the leading direction; the default bounds it at
    100x, which is the only claim being made for the value. The right cutoff depends on
    the fitted lens's spectrum, so ``write_space`` runs print ``cond(J)`` and the retained
    rank.
    """

    n_random_controls: int = 500
    """Matched-norm random directions put through the identical decomposition.

    **The gate itself**, not a footnote. "``v_J`` holds 8% of the variance" means nothing
    without knowing what ``k`` atoms capture from a direction with no structure at all; if
    random also lands at 8%, the decomposition has measured its own degrees of freedom.
    The comparison against this null -- ratio and p-value per emotion -- is the only claim
    the method supports.

    500 rather than a handful because it sets the p-value's resolution, and the resolution
    has to clear the *corrected* threshold. The Monte-Carlo estimator floors at
    ``1 / (n + 1)``; the gate's alpha is Bonferroni-corrected across the emotions, so with
    16 of them it is ``0.05 / 16 = 0.0031``. 16 controls could never report below 0.059
    and 200 could never report below 0.005 -- both are *incapable* of a significant
    result, whatever the data. 500 floors at 0.002, which clears it. The gate prints the
    floor beside the threshold and says TOO COARSE when it does not.
    """

    frac_j_expected_max: float = 0.30
    """Upper end of the expected reportable fraction. The workspace paper reports
    ~6-10% of a concept vector's variance in J-space and the brief expects 5-15%, so
    a value far above this says the decomposition or the lens is wrong rather than
    that the theory is."""

    # ---------------------------------------------- phase 7 channels --- #
    channel_emotions: object = None
    """Emotions for the two measurement channels. ``None`` auto-selects the cleanest
    Phase 6 decompositions, preferring an arousal-heavy negative one -- the brief's
    best behavioural probe, because risk aversion, refusal, persistence and hedging
    are things anxiety plausibly moves and a low-arousal positive emotion is not."""

    n_channel_emotions: int = 2
    """How many to auto-select. The brief's 2-3: Phases 7-9 need generation rather
    than forward passes, so each extra emotion multiplies hours, not minutes."""

    judge_model: str = DEFAULT_JUDGE_MODEL
    """Model that scores generated text.

    Must not be the model being steered. Using Qwen3-32B to score its own steered
    outputs would let the steering perturb the judge as well as the subject, so a
    score shift could be the judge moving rather than the behaviour. This is a real
    API dependency and a real cost line."""

    generation_max_new_tokens: int = 256
    """Decode steps per sample. The brief's ~150, and the reason Phases 7-9 are the
    expensive part: a forward pass is one step, this is 150 sequential ones."""

    phase8_grid_calls: int = 640
    """Judge calls Phase 8's grid implies -- 4 conditions x 4 strengths x 2 channels x
    N prompts. Used only to price it at the Phase 7 gate, so the decision to batch is
    made against a number rather than discovered on an invoice."""

    # ------------------------------------------------ phase 8 steering --- #
    steer_strengths: object = (0.0, 0.5, 1.0, 2.0)
    """Steering strengths, as multiples of the emotion vector's own norm.

    Phase 6 norm-matched all four conditions to ``||v||``, so one alpha means the
    same perturbation size in every condition and the grid is comparable by
    construction rather than by correction. 0.0 is included deliberately: it is the
    shared baseline row, and at 0.0 all four conditions are the identical unsteered
    model -- so it is measured once and reused, which is a quarter of the grid saved.
    """

    steer_positions: str = "all"
    """Where the steering vector is added: ``all`` token positions, or ``generated``
    only.

    ``all`` is the standard steering-vector protocol and the default. The choice is
    load-bearing enough to be a field rather than a constant: adding at the prompt
    positions too means the model reads its instructions through the perturbation,
    which is what makes the behaviour channel move at all -- but it also means a
    fluency change could come from a corrupted prompt rather than a steered
    disposition, which is why the perplexity check runs at every strength.
    """

    enable_thinking: bool = False
    """Forwarded to ``apply_chat_template`` by Phases 7-9. **Off**, unlike the template's
    own default.

    Qwen3 is a hybrid reasoning model and its chat template opens a ``<think>`` block
    unless told not to, so the generation budget goes on reasoning before any answer
    appears. Left on, at any plausible ``generation_max_new_tokens``, most completions are
    *truncated* reasoning traces -- and a truncated trace is indistinguishable from an
    answer to a regex scorer, which reads "let me consider option A" as choosing A. This
    already invalidated a Phase 7 and a Phase 8 run: ~7.5% of responses reached
    ``</think>`` and none of the behavioural ones did.

    Every gate that generates prints the *resolved* state -- off, on, or unsupported by
    this template -- because ``apply_chat_template`` ignores keyword arguments its
    template does not use, so passing the flag is not evidence that it took effect."""

    generation_batch_size: int = 8
    """Prompts generated concurrently. Generation is ~150 sequential decode steps per
    sample and the dominant cost of Phases 7-9, so batching is the difference between
    hours and most of a day. Requires left padding -- see phase8's generate()."""

    specificity_topic: object = None
    """Topic for the specificity control. ``None`` picks the first topic in the Phase 1
    table.

    The control the brief asks for: re-run the grid with a topic vector in place of
    the emotion, so a dissociation can be shown not to be generic to any concept at
    all. Built as a topic centroid from the stored activations and decomposed by the
    same pursuit, so it differs from the emotion vector in *what it is about* and in
    nothing else."""

    judge_use_batches: bool = True
    """Score Phase 8's grid through the Batches API at half price. It runs after hours
    of generation and nobody waits on an individual score, so there is no reason to
    pay interactive rates. Phase 7 scores interactively because there the point is to
    see a rubric's output now."""

    max_invalid_rate: float = 0.10
    """Share of a task family's responses that may be unscoreable before Phase 7 aborts.

    A hard gate, not a report. Truncated, empty and format-non-conforming responses used
    to be *scored* -- the first standalone letter in an unfinished reasoning trace became a
    risk choice -- so the invalid rate was 0% by construction and the failure was
    invisible. Now every scorer can return INVALID, and a family that cannot be scored
    reliably has to stop the pipeline rather than hand Phase 8 a number.
    """

    perplexity_max_ratio: float = 1.5
    """Perplexity ratio (steered / unsteered) above which a cell counts as degraded.

    The gate that stops "behaviour changed" from being "the output fell apart". A
    behavioural score means nothing at a strength where the text stopped being
    fluent, and the grid marks those cells rather than averaging them in."""

    # ------------------------------------------- phase 9 the re-entry clamp - #
    clamp_blocks: object = "all"
    """Blocks whose J-space coordinates are held at clean-pass values. Same grammar as
    ``gate_blocks``, resolved over the lens's fitted blocks.

    ``all`` is the default and the only setting that answers the question. Re-entry is
    downstream layers re-deriving the concept, so a clamp that skips layers leaves it
    exactly the room it needs -- a partial clamp cannot distinguish "the effect bypassed
    the workspace" from "the effect re-entered at block 41". Narrower specs exist for
    the layer-attribution follow-up (*where* does re-entry happen), not for the decisive
    cell, and Phase 9 marks any run that is not ``all`` as not decisive.
    """

    clamp_token_count: int = 24
    """Atom tokens from Phase 6's ``v_J`` that define the emotion's J-subspace, on top of
    the emotion word's own single-token variants.

    The subspace is ``span{J_l^T (g * w_t)}`` over that token set, per block: those are
    the directions the lens reads those tokens with, so holding the residual's component
    in them fixed is exactly "the lens sees no change in this concept". Too few tokens
    and the concept re-enters through a synonym the clamp never covered; too many and the
    clamp starts holding a large fraction of the residual fixed, which is collateral
    damage rather than a control. The gate reports the subspace dimension and the
    collateral it costs, so this is a number to read rather than to trust.
    """

    clamp_strengths: object = (0.0, 1.0)
    """Steering strengths Phase 9 runs, as multiples of ``||v||``. Deliberately shorter
    than ``steer_strengths``: Phase 9 costs two forward passes per decode step, and its
    job is one decisive cell rather than a dose-response curve. ``0.0`` is the shared
    baseline, as in Phase 8."""

    clamp_min_report_suppression: float = 0.7
    """Fraction of the report-channel lift under ``v`` that the clamp must remove before
    any behavioural number from Phase 9 is trusted.

    The verification the brief puts *first*. If steering with ``v`` raises the report
    score and the clamp does not bring it back down, the clamp is not clamping, and the
    decisive cell below it is meaningless -- so the gate prints this before anything
    else and refuses to interpret the behaviour channel when it fails.
    """

    clamp_max_collateral: float = 0.15
    """Largest tolerated disturbance to *unrelated* J-space content, as 1 - Spearman
    correlation between the clamped and unclamped lens readouts over control tokens.

    The other half of the verification. A clamp that suppresses the report channel by
    flattening the whole residual would pass the suppression check and prove nothing;
    this is what separates "the concept was held fixed" from "the model was broken".
    """

    remove_neutral_pcs: bool = False
    """Project emotion vectors off the top neutral-*story* PCs before PCA, as the
    mean-difference pipeline does. Off by default here: that step is meant to
    strip nuisance structure from a *single* emotion direction, and it could
    remove the very cross-emotion axes this experiment is looking for. Worth
    running as a robustness check, not as the default.

    Not implemented in Phase 3, which refuses it rather than ignoring it: the
    neutral subspace is an SVD of the ~400 neutral *stories*, and Phase 3 only has
    their centroid. Fitting it needs the activation chunks, which is why it belongs
    in Phase 5 alongside the layer sweep.
    """

    make_plots: bool = True

    # ------------------------------------------------------------ helpers --- #

    def __post_init__(self) -> None:
        self.split_proportions = tuple(float(p) for p in self.split_proportions)
        if isinstance(self.emotions, (list, tuple)):
            self.emotions = list(self.emotions)

    def validate(self) -> list[str]:
        problems: list[str] = []
        n_topics = 100  # the dataset's topic count; the loader re-checks it for real
        if self.stories_per_emotion < 1:
            problems.append("stories_per_emotion must be >= 1")
        elif self.stories_per_emotion % n_topics:
            # Not fatal, but it silently breaks exact topic matching, which is the
            # property that makes a direction encode emotion rather than subject.
            problems.append(
                f"stories_per_emotion={self.stories_per_emotion} is not a multiple of "
                f"{n_topics} topics, so per-topic counts differ between emotions and "
                "the set is no longer exactly topic-matched. Use a multiple of 100."
            )
        if self.include_neutral and self.neutral_stories < 2:
            problems.append("neutral_stories must be >= 2 when include_neutral is set")
        if abs(sum(self.split_proportions) - 1.0) > 1e-9:
            problems.append(
                f"split_proportions must sum to 1, got {sum(self.split_proportions)}"
            )
        if len(self.split_proportions) != 3:
            problems.append("split_proportions must have three entries")
        if self.emotions is not None and not isinstance(self.emotions, str):
            if len(self.emotions) != len(set(self.emotions)):
                problems.append("emotions contains duplicates")
            if len(self.emotions) < 3:
                problems.append(
                    "at least three emotions are needed: PCA across emotion centroids "
                    "after mean-centring has rank n_emotions-1, so two emotions yield a "
                    "single axis and no circumplex is possible"
                )
        if isinstance(self.emotions, str) and self.emotions != "all":
            problems.append(
                f"emotions={self.emotions!r}: the only string form is 'all'. "
                "Pass a list for an explicit set."
            )
        if self.gate_examples_per_emotion < 1:
            problems.append("gate_examples_per_emotion must be >= 1")
        if self.n_pcs_to_lens < 1:
            problems.append("n_pcs_to_lens must be >= 1")
        if not 0 <= self.readout_min_self_hit_rate <= 1:
            problems.append("readout_min_self_hit_rate must be in [0, 1]")
        if not 0.5 <= self.readout_min_ordering_auroc <= 1:
            problems.append(
                "readout_min_ordering_auroc must be in [0.5, 1]: 0.5 is chance, so a "
                "lower threshold would pass every direction"
            )
        if not self.steer_strengths:
            problems.append("steer_strengths must contain at least one value")
        elif all(float(a) == 0.0 for a in self.steer_strengths):
            problems.append("steer_strengths needs at least one non-zero value")
        # Negative strengths are deliberately allowed: -alpha subtracts the
        # direction, and a *predicted sign reversal* -- terrified+ raises hedging,
        # terrified- lowers it, random does neither -- is much stronger evidence
        # than "bigger perturbation, bigger change". It also removes the failure
        # mode where any absolute change counts as a hit. Nothing in
        # phase8_steer.py assumes alpha > 0; the hook simply adds alpha * vector.
        if self.steer_positions not in ("all", "generated"):
            problems.append(
                f"steer_positions must be 'all' or 'generated', got "
                f"{self.steer_positions!r}"
            )
        if self.generation_max_new_tokens < 256:
            problems.append(
                f"generation_max_new_tokens ({self.generation_max_new_tokens}) must be "
                ">= 256: below that, constrained answers were being cut off before they "
                "were reached, and a truncated response scored as if it were an answer"
            )
        if not 0 <= self.max_invalid_rate < 1:
            problems.append("max_invalid_rate must be in [0, 1)")
        if self.generation_batch_size < 1:
            problems.append("generation_batch_size must be >= 1")
        if not self.clamp_strengths:
            problems.append("clamp_strengths must contain at least one value")
        elif any(float(a) < 0 for a in self.clamp_strengths):
            problems.append("clamp_strengths must be >= 0")
        elif 0.0 not in tuple(float(a) for a in self.clamp_strengths):
            problems.append(
                "clamp_strengths must include 0.0: every Phase 9 number is a shift from "
                "the unsteered baseline, and the clamp's no-op check needs it too"
            )
        if self.clamp_token_count < 1:
            problems.append("clamp_token_count must be >= 1")
        if not 0 < self.clamp_min_report_suppression <= 1:
            problems.append("clamp_min_report_suppression must be in (0, 1]")
        if not 0 <= self.clamp_max_collateral < 1:
            problems.append("clamp_max_collateral must be in [0, 1)")
        if self.perplexity_max_ratio <= 1.0:
            problems.append(
                "perplexity_max_ratio must be > 1: it is a steered/unsteered ratio, so "
                "1.0 would mark every cell degraded"
            )
        if self.n_channel_emotions < 1:
            problems.append("n_channel_emotions must be >= 1")
        if self.generation_max_new_tokens < 1:
            problems.append("generation_max_new_tokens must be >= 1")
        if self.phase8_grid_calls < 1:
            problems.append("phase8_grid_calls must be >= 1")
        if not self.judge_model:
            problems.append("judge_model must be set")
        counts = [int(k) for k in (self.dict_atom_counts or ())]
        if not counts or any(k < 1 for k in counts):
            problems.append("dict_atom_counts must be one or more positive integers")
        elif self.n_dict_atoms not in counts:
            problems.append(
                f"n_dict_atoms ({self.n_dict_atoms}) must be one of dict_atom_counts "
                f"({counts}): the saved v_J is at one k, and it has to be a k the gate "
                "actually reports"
            )
        if counts and self.dict_pool_size < max(counts):
            problems.append(
                f"dict_pool_size ({self.dict_pool_size}) must be >= max(dict_atom_counts) "
                f"({max(counts)}): the pursuit selects from the pool"
            )
        if self.pursuit_steps < 1:
            problems.append("pursuit_steps must be >= 1")
        if not 0 < self.dict_pinv_rcond < 1:
            problems.append(
                "dict_pinv_rcond must be in (0, 1): it is a cutoff relative to J's "
                "largest singular value, so 0 keeps every numerically dead direction "
                "and 1 keeps none"
            )
        if self.n_random_controls < 0:
            problems.append("n_random_controls must be >= 0")
        if not 0 < self.frac_j_expected_max <= 1:
            problems.append("frac_j_expected_max must be in (0, 1]")
        if self.perspective_stories_per_emotion < 1:
            problems.append("perspective_stories_per_emotion must be >= 1")
        if self.within_emotion_n_pcs < 1:
            problems.append("within_emotion_n_pcs must be >= 1")
        if not 0 < self.neutral_variance_threshold < 1:
            problems.append("neutral_variance_threshold must be in (0, 1)")
        if self.perspective_self_frame == self.perspective_other_frame:
            problems.append(
                "perspective_self_frame and perspective_other_frame are identical; "
                "the contrast would be zero"
            )
        if self.n_pcs_to_report < 1:
            problems.append("n_pcs_to_report must be >= 1")
        if self.pca_null_samples < 0:
            problems.append("pca_null_samples must be >= 0 (0 disables the null band)")
        if not 0 < self.pc_stability_min_cosine <= 1:
            problems.append("pc_stability_min_cosine must be in (0, 1]")
        if self.quantization not in (None, "none"):
            problems.append(
                f"quantization={self.quantization!r}: the lens was fitted on the "
                "unquantised weights and is not valid against quantised activations"
            )
        if self.topk < 1:
            problems.append("topk must be >= 1")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            problems.append(f"unsupported dtype {self.dtype!r}")
        if self.activation_dtype not in ("bfloat16", "float16", "float32"):
            problems.append(f"unsupported activation_dtype {self.activation_dtype!r}")
        if self.target_block is not None and self.target_block < 0:
            # The upper bound needs the model's depth, so Phase 2 checks it against
            # the lens's fitted range once the config is known.
            problems.append("target_block must be >= 0 or None")
        if self.identity_check_blocks < 1:
            problems.append("identity_check_blocks must be >= 1")
        if self.token_offset < 0:
            problems.append("token_offset must be >= 0")
        if self.min_pooled_tokens < 1:
            problems.append("min_pooled_tokens must be >= 1")
        if self.max_length - self.token_offset < self.min_pooled_tokens:
            problems.append(
                f"max_length - token_offset = {self.max_length - self.token_offset} < "
                f"min_pooled_tokens = {self.min_pooled_tokens}: every stimulus would "
                "be skipped"
            )
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.chunk_size < 1:
            problems.append("chunk_size must be >= 1")
        if self.log_every_batches < 1:
            problems.append("log_every_batches must be >= 1")
        if not 0 < self.split_half_min_cosine <= 1:
            problems.append("split_half_min_cosine must be in (0, 1]")
        if self.r2_sync not in (True, False, "auto"):
            problems.append(f"r2_sync must be True, False or 'auto', got {self.r2_sync!r}")
        if self.r2_threshold_gib < 0:
            problems.append("r2_threshold_gib must be >= 0")
        if not self.r2_root and not self.r2_prefix:
            problems.append("set r2_root or r2_prefix; activations need a bucket folder")
        return problems

    def fingerprint(
        self,
        resolved_layers: list[int],
        hidden_size: int,
        model_sha: str,
        stimuli: dict,
    ) -> dict:
        """Settings that change the *meaning* of a stored activation vector.

        Two runs sharing a fingerprint may share an activations directory; a
        mismatch makes Phase 2 abort with a field-by-field diff rather than mixing
        incomparable vectors. ``--overwrite`` is the explicit opt-in.

        ``stimuli`` is :func:`emotion_pca_jlens.phase2_vectors.stimuli_fingerprint`
        -- a content hash of the Phase 1 table plus its row/group counts. Including
        it is **stricter than the mean-difference pipeline**, which deliberately
        leaves ``emotions``/``stories_per_emotion`` out so a run can be extended.
        The difference is real: there, a stored activation means the same thing
        whatever else is extracted alongside it. Here the artefact is an *average
        over exactly this stimulus set*, accumulated once as extraction streams
        past, so a changed set does not extend the run -- it silently changes what
        every emotion vector is a mean of. That must abort; use a new ``run_name``.

        Deliberately excluded: ``target_block``. Every hidden state is stored, so
        moving the target re-derives the vectors from the same chunks. Putting it in
        the fingerprint would make the Phase 5 layer sweep a re-extraction, which is
        the one cost this design exists to avoid.
        """
        return {
            "model_name": self.model_name,
            "model_sha": model_sha,
            "dtype": self.dtype,
            "quantization": self.quantization or "none",
            "layers": [int(l) for l in resolved_layers],
            "hidden_size": int(hidden_size),
            "activation_dtype": self.activation_dtype,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens,
            "use_chat_template": self.use_chat_template,
            "token_offset": self.token_offset,
            "min_pooled_tokens": self.min_pooled_tokens,
            # One implementation, recorded so a future pooling change is visible in
            # the diff rather than inferred from the code version.
            "pooling": "mean_after_offset",
            "layer_index_convention": "hidden_state",
            **{f"stimuli_{k}": v for k, v in stimuli.items()},
        }

    def resolved_r2_prefix(self) -> str:
        """Bucket key prefix for this run's activations.

        Keyed by ``run_name`` so two runs never overwrite each other inside
        ``r2_root``.
        """
        return self.r2_prefix or f"{self.r2_root}/{self.run_name}"

    @property
    def output_dir(self) -> Path:
        return paths.run_dir(self.run_name)

    @property
    def results_dir(self) -> Path:
        """Small artefacts. The ``results/`` level is what makes the shared
        RunPod ``pull-results`` wrapper find them -- see ``README.md``."""
        return self.output_dir / "results"

    @property
    def activations_dir(self) -> Path:
        return self.output_dir / "activations"

    @property
    def phase_dir(self) -> Path:
        return self.results_dir / "phases"

    @property
    def stimuli_path(self) -> Path:
        """Phase 1's output: one row per stimulus, read by Phase 2."""
        return self.phase_dir / "phase1_stimuli.parquet"

    @property
    def emotion_vectors_path(self) -> Path:
        """Phase 2's output: one pooled residual vector per emotion.

        Under ``results/`` and a few MB, so ``pull-results`` fetches it and Phase 3
        can run on a laptop CPU without touching a single activation chunk -- which
        matters because the chunks live in R2 only by default.
        """
        return self.phase_dir / "phase2_emotion_vectors.safetensors"

    @property
    def emotion_vectors_meta_path(self) -> Path:
        """Sidecar for :attr:`emotion_vectors_path`: row order, counts, provenance.

        The vectors are uninterpretable without it -- row *i* is only an emotion
        because this file says so."""
        return self.phase_dir / "phase2_emotion_vectors.json"

    @property
    def decomposition_path(self) -> Path:
        """Phase 6's output: norm-matched ``v``, ``v_J``, ``v_perp`` and a random
        control per emotion, which is what Phase 8 steers with."""
        return self.phase_dir / "phase6_decomposition.safetensors"

    @property
    def decomposition_meta_path(self) -> Path:
        return self.phase_dir / "phase6_decomposition.json"

    @property
    def pcs_path(self) -> Path:
        """Phase 3's output: the principal axes of emotion space, read by Phase 4.

        Unit directions in residual space at the same target block as the emotion
        vectors, so they can be handed straight to the lens."""
        return self.phase_dir / "phase3_pcs.safetensors"

    @property
    def pcs_meta_path(self) -> Path:
        """Sidecar for :attr:`pcs_path`: variance table, alignment, provenance."""
        return self.phase_dir / "phase3_pcs.json"

    def to_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, object]) -> "PCAJLensConfig":
        valid = {f.name: f for f in fields(self)}
        coerced: dict[str, object] = {}
        for key, value in overrides.items():
            if key not in valid:
                raise KeyError(f"unknown config field {key!r}. Valid: {sorted(valid)}")
            coerced[key] = _coerce_field(key, value, getattr(self, key))
        return replace(self, **coerced)


def resolve_block_spec(spec: object, fitted_blocks: list[int]) -> list[int]:
    """Resolve a block spec against the blocks the lens actually has.

    Reuses :func:`core.model_utils.resolve_layers`' grammar but over the fitted
    block list, so ``evenly_spaced:10`` spreads across ``0..n_layers-2`` and
    never selects a block with no ``J_l``.
    """
    if spec is None or spec == "all":
        return list(fitted_blocks)
    if isinstance(spec, str) and spec.startswith("evenly_spaced:"):
        import numpy as np

        k = int(spec.split(":", 1)[1])
        if not 1 <= k <= len(fitted_blocks):
            raise ValueError(
                f"evenly_spaced:{k} outside 1..{len(fitted_blocks)} fitted blocks"
            )
        picks = np.linspace(0, len(fitted_blocks) - 1, k).round().astype(int)
        return sorted({int(fitted_blocks[i]) for i in picks})

    # Explicit indices / ranges: resolve then verify each is fitted.
    from core import model_utils

    resolved = model_utils.resolve_layers(spec, len(fitted_blocks) + 1)
    unknown = sorted(set(resolved) - set(fitted_blocks))
    if unknown:
        raise ValueError(
            f"blocks {unknown} have no fitted J; the lens covers "
            f"{fitted_blocks[0]}..{fitted_blocks[-1]}"
        )
    return resolved


def load_config(args) -> PCAJLensConfig:
    """Build a config from ``--config-json`` and repeated ``--set KEY=VALUE``."""
    import sys

    overrides: dict[str, object] = {}
    if getattr(args, "config_json", None):
        overrides.update(json.loads(Path(args.config_json).read_text(encoding="utf-8")))
    for item in getattr(args, "set", []) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value

    config = CONFIG.with_overrides(overrides) if overrides else CONFIG
    problems = config.validate()
    if problems:
        print("Configuration problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(2)
    return config


#: Used when no overrides are supplied.
CONFIG = PCAJLensConfig()
