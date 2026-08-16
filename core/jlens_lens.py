"""Jacobian-lens (J-lens) loading and readout.

What the J-lens is
------------------
The Jacobian lens reads out what a residual-stream vector is *disposed to make
the model say*. It linearly transports a residual at layer ``l`` into the
final-layer basis and decodes it with the model's own unembedding::

    lens_l(h) = unembed(J_l @ h),   J_l = E[d h_final / d h_l]

Reference implementation: ``anthropics/jacobian-lens`` (Apache-2.0), companion
code for "Verbalizable Representations Form a Global Workspace in Language
Models". Pre-fitted lenses: ``neuronpedia/jacobian-lens`` on the Hub (MIT).

Conventions verified against the reference implementation
---------------------------------------------------------
These were read out of the library's source and tests at commit
``581d398613e5602a5af361e1c34d3a92ea82ba8e``, not assumed. They are the three
things easiest to get silently wrong:

1. **What ``unembed`` is.** ``jlens.hf.HFLensModel.unembed`` is
   ``lm_head(final_norm(h))`` -- the model's *own* final norm (with its learned
   weight), then the LM head, then ``final_logit_softcapping`` if the config has
   one (Gemma-2 does; Qwen3 does not). So the "norm step" is the model's real
   final norm, not an ad-hoc normalisation. Softcapping is monotonic, so it
   never changes top-k *order*, only the probabilities.

2. **Layer indexing is by residual block, not by hidden-state index.**
   ``jlens`` hooks ``model.layers[l]`` with ``register_forward_hook``, so index
   ``l`` means *the output of block* ``l``, for ``l`` in ``0 .. n_layers-1``.
   This repo's :mod:`core.model_utils` / :mod:`core.pooling` instead index the
   ``output_hidden_states`` tuple, where ``0`` is the embedding output. The two
   differ by one:

       hidden_state_index = block_index + 1

   Use :func:`hidden_state_index` / :func:`block_index` rather than writing
   ``+1`` inline. See also the caveat in :func:`hidden_state_index` about the
   *last* hidden state, which ``transformers`` returns already normed.

3. **The final block has no fitted J.** ``fit`` uses
   ``source_layers = 0 .. n_layers-2``: ``J_l`` maps *output of block l* to
   *output of the last block* (pre-final-norm), so the last block itself is not
   a source. ``tests/test_fitting.py`` pins this exactly -- for a 4-layer model
   it asserts ``source_layers == [0, 1, 2]`` and
   ``J_2 == I + W_3`` (one block of transport). Consequently the "J ~= identity
   at very late layers" sanity check must be run at ``n_layers - 2``, which is
   the *highest layer the lens has*, not at ``n_layers - 1``.

A fourth property, used by the PC readout in this project: the transport is
linear and ``final_norm`` is an RMSNorm, which is scale-invariant up to
``eps``. So the readout of a *bare direction* does not depend on its magnitude
-- ``+PC`` and ``-PC`` are well defined without choosing a step size. Phase 0
verifies this numerically rather than trusting the algebra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Hub repo of pre-fitted lenses (MIT).
LENS_REPO = "neuronpedia/jacobian-lens"

#: Reference implementation, pinned for provenance.
JLENS_REPO_URL = "https://github.com/anthropics/jacobian-lens"
JLENS_VERIFIED_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

_LENS_FILE_RE = re.compile(r"_jacobian_lens.*\.pt$")


# --------------------------------------------------------------------------- #
# Layer index conventions
# --------------------------------------------------------------------------- #

def hidden_state_index(block: int) -> int:
    """``output_hidden_states`` index for the output of residual block ``block``.

    ``hidden_states[i]`` is the *input* to block ``i``, so the output of block
    ``i`` is ``hidden_states[i + 1]``.

    Caveat at the top of the stack: ``transformers`` applies the model's final
    norm to the *last* entry of the tuple, so ``hidden_states[n_layers]`` is
    ``norm(output of block n_layers-1)`` while ``jlens`` captures that block's
    raw output. This never bites us here, because the lens only has layers
    ``0 .. n_layers-2``, which map to hidden states ``1 .. n_layers-1`` -- all
    pre-final-norm.
    """
    return block + 1


def block_index(hidden_state: int) -> int:
    """Residual-block index whose output is ``hidden_states[hidden_state]``."""
    if hidden_state < 1:
        raise ValueError(
            f"hidden state {hidden_state} is the embedding output; it is not the "
            "output of any residual block, so no J_l is fitted for it"
        )
    return hidden_state - 1


def max_lens_block(n_layers: int) -> int:
    """Highest block index the lens is fitted for (``n_layers - 2``)."""
    return n_layers - 2


def describe_block(block: int, n_layers: int) -> str:
    """One-line description of a block index in both conventions."""
    frac = (block + 1) / n_layers
    return (
        f"block {block} (= hidden_states[{hidden_state_index(block)}], "
        f"{frac:.0%} of depth, of {n_layers} blocks)"
    )


# --------------------------------------------------------------------------- #
# Locating a pre-fitted lens on the Hub
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LensArtifact:
    """A resolved pre-fitted lens in the Hub repo."""

    model_name: str
    """HF model the lens was fitted on, as recorded in the lens's own config."""

    subfolder: str
    """Top-level folder in :data:`LENS_REPO`, e.g. ``qwen3-32b``."""

    lens_file: str
    """Repo-relative path of the ``*_jacobian_lens*.pt`` file."""

    config_file: str | None
    """Repo-relative path of the fit's ``config.yaml``, if present."""

    extra_files: tuple[str, ...]
    """Other files in the subfolder (convergence CSV etc.), excluding junk."""

    fit_config: dict
    """Parsed ``config.yaml``. Empty if absent/unparseable."""

    def allow_pattern(self) -> str:
        """``allow_patterns`` entry that fetches this lens and nothing else.

        The repo holds lenses for ~40 models; a bare ``snapshot_download`` would
        pull tens of gigabytes.
        """
        return f"{self.subfolder}/**"


def model_slug(model_name: str) -> str:
    """Hub-repo folder name convention: the model id's last path segment, lowercased.

    ``"Qwen/Qwen3-32B"`` -> ``"qwen3-32b"``.
    """
    return model_name.rsplit("/", 1)[-1].lower()


def _is_junk(path: str) -> bool:
    return Path(path).name in (".DS_Store", ".gitattributes")


def _parse_fit_config(text: str) -> dict:
    """Parse a lens ``config.yaml``.

    Uses PyYAML when available (it is, transitively, via ``transformers``) and
    falls back to a flat scalar scan so a missing dependency degrades to partial
    information rather than an exception.
    """
    try:
        import yaml

        parsed = yaml.safe_load(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        flat: dict = {}
        for line in text.splitlines():
            match = re.match(r'^(\w+):\s*"?([^"#]+?)"?\s*$', line)
            if match:
                flat[match.group(1)] = match.group(2)
        return flat


def resolve_lens_artifact(
    model_name: str,
    lens_repo: str = LENS_REPO,
    revision: str | None = None,
    subfolder: str | None = None,
) -> LensArtifact:
    """Find the pre-fitted lens for ``model_name`` and verify it really is that model.

    The folder-name convention (:func:`model_slug`) is only a hint: the
    authoritative check is the ``hf_model_name`` recorded in the fit's own
    ``config.yaml``. If the slug guess is absent or disagrees, every subfolder's
    config is scanned before giving up, so a renamed folder is found rather than
    silently mismatched.

    Args:
        model_name: HF model id, e.g. ``"Qwen/Qwen3-32B"``.
        lens_repo: Hub repo of pre-fitted lenses.
        revision: Hub revision of ``lens_repo``.
        subfolder: Force a specific subfolder, skipping resolution. The
            ``hf_model_name`` check still runs and still raises on a mismatch.

    Raises:
        LookupError: If no subfolder's config claims ``model_name``.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(lens_repo, revision=revision) if not _is_junk(f)]
    by_folder: dict[str, list[str]] = {}
    for path in files:
        if "/" in path:
            by_folder.setdefault(path.split("/", 1)[0], []).append(path)

    def read_config(folder: str) -> tuple[str | None, dict]:
        config_path = next(
            (f for f in by_folder[folder] if Path(f).name == "config.yaml"), None
        )
        if config_path is None:
            return None, {}
        local = hf_hub_download(lens_repo, config_path, revision=revision)
        return config_path, _parse_fit_config(Path(local).read_text(encoding="utf-8"))

    def build(folder: str) -> LensArtifact:
        lens_files = [f for f in by_folder[folder] if _LENS_FILE_RE.search(f)]
        if not lens_files:
            raise LookupError(
                f"{lens_repo}/{folder} has no *_jacobian_lens*.pt "
                f"(found {[Path(f).name for f in by_folder[folder]]})"
            )
        if len(lens_files) > 1:
            raise LookupError(
                f"{lens_repo}/{folder} has {len(lens_files)} lens files "
                f"({[Path(f).name for f in lens_files]}); pass subfolder= and pick one"
            )
        config_file, fit_config = read_config(folder)
        claimed = fit_config.get("hf_model_name")
        if claimed and claimed != model_name:
            raise LookupError(
                f"{lens_repo}/{folder} was fitted on {claimed!r}, not {model_name!r}. "
                "Applying a lens across checkpoints is not valid -- the Jacobian is "
                "specific to the weights."
            )
        lens_file = lens_files[0]
        return LensArtifact(
            model_name=model_name,
            subfolder=folder,
            lens_file=lens_file,
            config_file=config_file,
            extra_files=tuple(
                f for f in sorted(by_folder[folder])
                if f not in (lens_file, config_file)
            ),
            fit_config=fit_config,
        )

    if subfolder is not None:
        if subfolder not in by_folder:
            raise LookupError(
                f"{lens_repo} has no subfolder {subfolder!r}; "
                f"available: {sorted(by_folder)}"
            )
        return build(subfolder)

    slug = model_slug(model_name)
    if slug in by_folder:
        try:
            return build(slug)
        except LookupError:
            pass  # fall through to the exhaustive scan

    for folder in sorted(by_folder):
        if folder == slug:
            continue
        _, fit_config = read_config(folder)
        if fit_config.get("hf_model_name") == model_name:
            return build(folder)

    raise LookupError(
        f"no lens in {lens_repo} claims hf_model_name={model_name!r}. "
        f"Subfolders present: {sorted(by_folder)}"
    )


def download_lens(
    artifact: LensArtifact,
    lens_repo: str = LENS_REPO,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download only ``artifact``'s subfolder and return the local lens file path."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        lens_repo,
        allow_patterns=[artifact.allow_pattern()],
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    path = Path(root) / artifact.lens_file
    if not path.exists():
        raise FileNotFoundError(
            f"expected {artifact.lens_file} under {root} after download; "
            f"got {[str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file()]}"
        )
    return path


# --------------------------------------------------------------------------- #
# Inspecting a lens checkpoint
# --------------------------------------------------------------------------- #

#: The two on-disk shapes a published lens comes in. ``neuronpedia/jacobian-lens``
#: contains both, so a loader that assumes either one is wrong half the time.
#:
#: ``lens``            written by ``JacobianLens.save()``: key ``J`` already holds
#:                     the *mean* Jacobian, stored fp16 by default.
#: ``fit_checkpoint``  the resumable state written by ``fitting.fit()``: key
#:                     ``jacobian_sum`` holds the running **sum** over prompts,
#:                     fp32, and the mean is ``jacobian_sum / n_done`` (see
#:                     ``fitting.fit``, which only divides on the final line).
#:
#: Getting this wrong is a silent failure of a specific and nasty kind: the model's
#: final norm is an RMSNorm, so a globally mis-scaled ``J`` produces *identical*
#: top-k tokens. A lens left un-normalised by a factor of ``n_done`` would sail
#: through any readout-based sanity check while making every magnitude-sensitive
#: quantity -- ``||J - I||``, variance decompositions -- wrong by that factor.
LENS_FORMATS = ("lens", "fit_checkpoint")


def _jacobians_from_checkpoint(checkpoint: dict) -> tuple[dict, int, str]:
    """Normalise either on-disk format to ``(mean jacobians, n_prompts, format)``.

    Division is done **in place**: these tensors are ~6.6 GiB for Qwen3-32B and a
    non-mutating divide would transiently double that.
    """
    if "J" in checkpoint:
        jacobians = {int(k): v for k, v in checkpoint["J"].items()}
        return jacobians, int(checkpoint["n_prompts"]), "lens"

    if "jacobian_sum" in checkpoint:
        n_done = int(checkpoint["n_done"])
        if n_done < 1:
            raise ValueError(
                f"fit checkpoint reports n_done={n_done}; cannot form a mean Jacobian"
            )
        jacobians = {int(k): v for k, v in checkpoint["jacobian_sum"].items()}
        for tensor in jacobians.values():
            tensor.div_(n_done)
        declared = checkpoint.get("source_layers")
        if declared is not None and sorted(jacobians) != sorted(int(l) for l in declared):
            raise ValueError(
                "fit checkpoint's source_layers disagree with the jacobian_sum keys: "
                f"{sorted(int(l) for l in declared)[:6]}... vs {sorted(jacobians)[:6]}..."
            )
        return jacobians, n_done, "fit_checkpoint"

    raise ValueError(
        f"not a Jacobian lens file: expected key 'J' (a saved lens) or "
        f"'jacobian_sum' (a fit checkpoint), found {sorted(checkpoint)}"
    )


@dataclass(frozen=True)
class LensDescription:
    """What is actually inside a lens ``.pt``, read without reinterpretation."""

    path: Path
    file_bytes: int
    checkpoint_keys: tuple[str, ...]
    checkpoint_format: str
    d_model: int
    n_prompts: int
    source_layers: tuple[int, ...]
    j_shapes: tuple[tuple[int, ...], ...]
    j_dtypes: tuple[str, ...]
    stored_bytes: int

    @property
    def uniform_shape(self) -> tuple[int, ...] | None:
        return self.j_shapes[0] if len(set(self.j_shapes)) == 1 else None

    @property
    def uniform_dtype(self) -> str | None:
        return self.j_dtypes[0] if len(set(self.j_dtypes)) == 1 else None

    def problems(self, n_layers: int | None = None, d_model: int | None = None) -> list[str]:
        """Structural complaints; empty means the lens looks as documented."""
        issues: list[str] = []
        if self.uniform_shape is None:
            issues.append(f"J shapes are not uniform: {sorted(set(self.j_shapes))}")
        elif self.uniform_shape != (self.d_model, self.d_model):
            issues.append(
                f"J shape {self.uniform_shape} is not (d_model, d_model) = "
                f"({self.d_model}, {self.d_model})"
            )
        if self.uniform_dtype is None:
            issues.append(f"J dtypes are not uniform: {sorted(set(self.j_dtypes))}")
        expected = list(range(len(self.source_layers)))
        if list(self.source_layers) != expected:
            issues.append(
                f"source_layers are not contiguous from 0: "
                f"{self.source_layers[:4]}..{self.source_layers[-4:]}"
            )
        if d_model is not None and self.d_model != d_model:
            issues.append(f"lens d_model={self.d_model} but the model has {d_model}")
        if n_layers is not None:
            want = max_lens_block(n_layers)
            if self.source_layers and self.source_layers[-1] != want:
                issues.append(
                    f"highest fitted layer is {self.source_layers[-1]}, expected "
                    f"{want} (= n_layers - 2) for a {n_layers}-block model"
                )
        return issues


def describe_lens_checkpoint(path: str | Path) -> LensDescription:
    """Read a lens ``.pt`` and report its real contents, in either format.

    Note this loads the whole checkpoint into host RAM (the Qwen3-32B file is
    ~6.6 GiB, stored fp32). Deliberately does *not* go through
    ``JacobianLens.load`` -- that method rejects a fit checkpoint outright, and it
    would also upcast to fp32, hiding the on-disk dtype this is meant to report.
    """
    import torch

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    jacobians, n_prompts, fmt = _jacobians_from_checkpoint(checkpoint)
    layers = sorted(jacobians)
    d_model = int(jacobians[layers[0]].shape[0])
    return LensDescription(
        path=path,
        file_bytes=path.stat().st_size,
        checkpoint_keys=tuple(sorted(str(k) for k in checkpoint)),
        checkpoint_format=fmt,
        # Neither format is guaranteed to store d_model as a field -- the fit
        # checkpoint does not -- so take it from the tensors themselves.
        d_model=int(checkpoint.get("d_model", d_model)),
        n_prompts=n_prompts,
        source_layers=tuple(layers),
        j_shapes=tuple(tuple(jacobians[l].shape) for l in layers),
        j_dtypes=tuple(str(jacobians[l].dtype).removeprefix("torch.") for l in layers),
        stored_bytes=sum(
            jacobians[l].numel() * jacobians[l].element_size() for l in layers
        ),
    )


def load_lens(path: str | Path):
    """Load a ``JacobianLens`` from either on-disk format.

    ``JacobianLens.load`` handles only files written by ``JacobianLens.save``; it
    raises on a fit checkpoint (its own error message even guesses as much). The
    published Qwen3-32B lens *is* a fit checkpoint, so this wrapper exists to
    normalise the running sum to a mean before handing the tensors over. See
    :data:`LENS_FORMATS` for why guessing wrong here fails silently.
    """
    import torch

    from jlens import JacobianLens

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    jacobians, n_prompts, _ = _jacobians_from_checkpoint(checkpoint)
    d_model = int(jacobians[sorted(jacobians)[0]].shape[0])
    return JacobianLens(jacobians=jacobians, n_prompts=n_prompts, d_model=d_model)


def identity_distances(lens) -> dict[int, float]:
    """Per-layer ``||J_l - I||_F / ||I||_F``.

    0 means "J is the identity", i.e. the lens is a plain logit lens at that
    layer. This should fall towards 0 as ``l`` approaches ``n_layers - 2``,
    where the transport spans a single block. Reported rather than asserted: the
    published fits record only a single aggregate ``final_identity_distance``,
    and the script that computes it is not public, so the *trend* is the check,
    not the absolute value.
    """
    import torch

    out: dict[int, float] = {}
    for layer in lens.source_layers:
        J = lens.jacobians[layer]
        d = J.shape[0]
        # ||J - I||_F^2 = ||J||_F^2 - 2 tr(J) + d, and ||I||_F = sqrt(d).
        # Computed this way to avoid materialising a d x d identity (105 MiB at
        # d_model=5120) and a d x d difference for every one of ~63 layers.
        squared = (
            float(torch.linalg.norm(J.float()) ** 2)
            - 2.0 * float(torch.diagonal(J).float().sum())
            + d
        )
        out[int(layer)] = (max(squared, 0.0) / d) ** 0.5
    return out


# --------------------------------------------------------------------------- #
# Readout
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TokenReadout:
    """One decoded vocabulary token from a lens readout."""

    rank: int
    token_id: int
    token: str
    logit: float
    prob: float


class LensReadout:
    """Applies a fitted lens to residual vectors and to whole prompts.

    Wraps ``jlens.JacobianLens`` plus a ``jlens.hf.HFLensModel``. The one thing
    it adds over calling the library directly is deliberate *device discipline*:
    the Jacobians stay in host RAM and transport runs there as a matrix-vector
    product (microseconds at d_model=5120), so no 100 MiB ``J_l`` is copied to
    the GPU per readout and the VRAM budget is left for the weights.
    """

    def __init__(self, lens, model) -> None:
        self.lens = lens
        self.model = model
        self.tokenizer = model.tokenizer

    @classmethod
    def build(cls, hf_model, tokenizer, lens_path: str | Path) -> "LensReadout":
        """Wrap a loaded HF model and a lens file (either on-disk format)."""
        import jlens

        lens = load_lens(lens_path)
        model = jlens.from_hf(hf_model, tokenizer)
        if lens.d_model != model.d_model:
            raise ValueError(
                f"lens d_model={lens.d_model} but model d_model={model.d_model}: "
                "wrong lens for this checkpoint"
            )
        return cls(lens, model)

    # -- properties -------------------------------------------------------- #

    @property
    def n_layers(self) -> int:
        return self.model.n_layers

    @property
    def source_layers(self) -> list[int]:
        return self.lens.source_layers

    def unembed_description(self) -> dict:
        """What ``unembed`` will actually do, read off the wrapped model.

        Reported at the gate so the readout formula is confirmed from the loaded
        objects rather than from this module's docstring.

        Reaches into ``HFLensModel``'s private attributes deliberately: the point
        is to report what the library will *really* do, and there is no public
        accessor. Safe because ``requirements.txt`` pins ``jlens`` to the commit
        these names were read from; if that pin moves, this method is the first
        thing to check.
        """
        return {
            "final_norm": type(self.model._final_norm).__name__,
            "lm_head": type(self.model._lm_head).__name__,
            "lm_head_dtype": str(self.model._lm_head.weight.dtype).removeprefix("torch."),
            "lm_head_device": str(self.model._lm_head.weight.device),
            "logit_softcapping": self.model._logit_softcap,
            "vocab_size": int(self.model._lm_head.weight.shape[0]),
            "formula": "logits = lm_head(final_norm(J_l @ h))"
            + (" then softcap*tanh(logits/softcap)" if self.model._logit_softcap else ""),
        }

    # -- direction readout -------------------------------------------------- #

    def direction_logits(self, direction, block: int, use_jacobian: bool = True):
        """Lens logits for a bare direction in residual space.

        Args:
            direction: ``(d_model,)`` array/tensor. Magnitude is irrelevant --
                the final RMSNorm is scale-invariant -- so a unit PC is fine.
            block: Residual-block index (jlens convention; see
                :func:`hidden_state_index`).
            use_jacobian: ``False`` skips the transport, giving the plain
                logit-lens readout of the same direction.

        Returns:
            ``(vocab_size,)`` float32 CPU tensor of logits.
        """
        import torch

        if use_jacobian and block not in self.lens.source_layers:
            raise ValueError(
                f"block {block} has no fitted J; the lens covers "
                f"{self.lens.source_layers[0]}..{self.lens.source_layers[-1]} "
                f"(= 0..n_layers-2 for this {self.n_layers}-block model)"
            )
        h = torch.as_tensor(direction, dtype=torch.float32).reshape(-1)
        if h.numel() != self.model.d_model:
            raise ValueError(
                f"direction has {h.numel()} entries, expected d_model={self.model.d_model}"
            )
        # Transport on CPU: keeps J_l off the GPU (see the class docstring).
        if use_jacobian:
            h = self.lens.transport(h, block)
        with torch.no_grad():
            return self.model.unembed(h).float().cpu().reshape(-1)

    def top_tokens(
        self,
        direction,
        block: int,
        k: int = 15,
        use_jacobian: bool = True,
    ) -> list[TokenReadout]:
        """Top-``k`` vocabulary tokens for a bare direction."""
        return self.decode_top(self.direction_logits(direction, block, use_jacobian), k)

    def decode_top(self, logits, k: int = 15) -> list[TokenReadout]:
        """Decode the top-``k`` entries of a logit vector."""
        import torch

        probs = torch.softmax(logits.float(), dim=-1)
        top = torch.topk(logits.float(), k)
        return [
            TokenReadout(
                rank=i,
                token_id=int(token_id),
                token=self.tokenizer.decode([int(token_id)]),
                logit=float(logits[int(token_id)]),
                prob=float(probs[int(token_id)]),
            )
            for i, token_id in enumerate(top.indices.tolist())
        ]

    # -- word-level rank lookup -------------------------------------------- #

    def single_token_variants(self, word: str) -> dict[str, int]:
        """Casing/leading-space variants of ``word`` that are *one* token.

        The lens decodes one vocabulary token at a time, so a target concept
        that is not a single token cannot be read out directly. Returning an
        empty dict is therefore a real finding about the vocabulary, not a
        failure -- it is exactly the "J-lens is single-token" caveat, made
        concrete.
        """
        out: dict[str, int] = {}
        for variant in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[variant] = int(ids[0])
        return out

    def rank_of_word(self, logits, word: str) -> tuple[int | None, str | None, float]:
        """Best rank of any single-token variant of ``word`` in ``logits``.

        Returns ``(rank, variant, prob)``; rank 0 is the top of the vocabulary.
        ``(None, None, 0.0)`` means no variant is a single token.
        """
        import torch

        variants = self.single_token_variants(word)
        if not variants:
            return None, None, 0.0
        order = torch.argsort(logits.float(), descending=True)
        position = torch.empty_like(order)
        position[order] = torch.arange(order.numel())
        probs = torch.softmax(logits.float(), dim=-1)
        best = min(variants.items(), key=lambda kv: int(position[kv[1]]))
        return int(position[best[1]]), best[0], float(probs[best[1]])

    # -- prompt readout ----------------------------------------------------- #

    def apply_to_prompt(
        self,
        prompt: str,
        blocks: list[int],
        position: int = -1,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ):
        """Lens logits at one token position of ``prompt``, per requested block.

        Thin pass-through to ``JacobianLens.apply`` so the gate exercises the
        library's own code path (hooks, position selection, unembedding) rather
        than a reimplementation.

        Returns:
            ``(lens_logits, model_logits, input_ids)`` where ``lens_logits`` maps
            block index to a ``(vocab_size,)`` tensor at ``position``, and
            ``model_logits`` is the model's real logits at the same position.
        """
        lens_logits, model_logits, input_ids = self.lens.apply(
            self.model,
            prompt,
            layers=blocks,
            positions=[position],
            max_seq_len=max_seq_len,
            use_jacobian=use_jacobian,
        )
        return (
            {int(l): v.reshape(-1) for l, v in lens_logits.items()},
            model_logits.reshape(-1),
            input_ids,
        )
