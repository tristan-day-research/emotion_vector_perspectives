"""Model-agnostic loading, layer selection, and text preparation.

Nothing here is specific to Qwen: we only rely on ``AutoConfig`` /
``AutoTokenizer`` / ``AutoModel`` and on ``output_hidden_states=True``, which
every decoder-only ``transformers`` model supports.

Layer indexing convention
-------------------------
We index the ``hidden_states`` tuple returned with ``output_hidden_states=True``:

* ``0``  = embedding output (before any transformer block)
* ``i``  = residual stream after block ``i`` (1-indexed blocks)
* ``n_layers`` = final residual stream (after the last block, before ``lm_head``;
  note ``transformers`` applies the model's final norm to this entry)

So a model with ``num_hidden_layers = 64`` exposes 65 hidden states, ``0..64``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DTYPE_ALIASES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}


def torch_dtype(name: str):
    import torch

    canonical = DTYPE_ALIASES.get(name.lower())
    if canonical is None:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(DTYPE_ALIASES)}")
    return getattr(torch, canonical)


def dtype_nbytes(name: str) -> int:
    canonical = DTYPE_ALIASES.get(name.lower())
    return {"bfloat16": 2, "float16": 2, "float32": 4}[canonical]


# --------------------------------------------------------------------------- #
# Architecture introspection (no weights loaded)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArchitectureInfo:
    model_name: str
    revision: str
    resolved_sha: str
    n_layers: int
    n_hidden_states: int
    hidden_size: int
    architectures: tuple[str, ...]
    config_dtype: str | None
    max_position_embeddings: int | None


def resolve_model_revision(model_name: str, revision: str | None) -> str:
    """Resolve a model revision to a commit sha (best effort; ``unknown`` offline)."""
    try:
        from huggingface_hub import model_info

        return model_info(model_name, revision=revision).sha
    except Exception as exc:
        warnings.warn(f"Could not resolve model revision sha for {model_name}: {exc}")
        return revision or "unknown"


def load_architecture_info(
    model_name: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = False,
) -> ArchitectureInfo:
    """Read the model's config only. Cheap enough for ``--dry-run``."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
    )
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if n_layers is None or hidden_size is None:
        raise ValueError(
            f"Could not determine num_hidden_layers/hidden_size from {model_name}'s config."
        )
    raw_dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
    return ArchitectureInfo(
        model_name=model_name,
        revision=revision or "main",
        resolved_sha=resolve_model_revision(model_name, revision),
        n_layers=int(n_layers),
        n_hidden_states=int(n_layers) + 1,
        hidden_size=int(hidden_size),
        architectures=tuple(getattr(cfg, "architectures", None) or ()),
        config_dtype=str(raw_dtype) if raw_dtype else None,
        max_position_embeddings=getattr(cfg, "max_position_embeddings", None),
    )


# --------------------------------------------------------------------------- #
# Layer selection
# --------------------------------------------------------------------------- #

def resolve_layers(spec: object, n_hidden_states: int) -> list[int]:
    """Turn a layer spec into a sorted list of hidden-state indices.

    Accepted specs:
        ``None`` / ``"all"``        every hidden state, ``0 .. n_layers``
        ``"blocks"``               block outputs only, ``1 .. n_layers``
        ``"evenly_spaced:K"``      ``K`` indices spread over ``0 .. n_layers``
        ``"range:a:b"`` / ``"range:a:b:step"``   half-open range
        ``Sequence[int]``          explicit indices; negatives count from the end
    """
    import numpy as np

    max_idx = n_hidden_states - 1

    def normalise(indices: Sequence[int]) -> list[int]:
        out = []
        for i in indices:
            i = int(i)
            j = i if i >= 0 else n_hidden_states + i
            if not 0 <= j <= max_idx:
                raise ValueError(
                    f"layer index {i} out of range for a model with {n_hidden_states} "
                    f"hidden states (valid: 0..{max_idx}, or -1..-{n_hidden_states})"
                )
            out.append(j)
        if not out:
            raise ValueError("layer spec selected zero layers")
        return sorted(set(out))

    if spec is None or spec == "all":
        return list(range(n_hidden_states))
    if isinstance(spec, str):
        if spec == "blocks":
            return list(range(1, n_hidden_states))
        if spec.startswith("evenly_spaced:"):
            k = int(spec.split(":", 1)[1])
            if k < 1:
                raise ValueError("evenly_spaced:K requires K >= 1")
            if k > n_hidden_states:
                raise ValueError(f"evenly_spaced:{k} exceeds {n_hidden_states} hidden states")
            return normalise(np.linspace(0, max_idx, k).round().astype(int).tolist())
        if spec.startswith("range:"):
            parts = spec.split(":")[1:]
            if len(parts) not in (2, 3):
                raise ValueError("use range:a:b or range:a:b:step")
            a, b = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            return normalise(list(range(a, b, step)))
        raise ValueError(f"unrecognised layer spec {spec!r}")
    if isinstance(spec, (list, tuple, set, range)):
        return normalise(list(spec))
    raise TypeError(f"unsupported layer spec type {type(spec).__name__}")


# --------------------------------------------------------------------------- #
# Tokenizer / model loading
# --------------------------------------------------------------------------- #

def load_tokenizer(
    model_name: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    padding_side: str = "right",
    trust_remote_code: bool = False,
):
    """Load a tokenizer with a usable pad token.

    Pooling is padding-agnostic (see :mod:`core.pooling`), so ``padding_side``
    only affects performance/kernels, not results.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
    )
    tok.padding_side = padding_side
    if tok.pad_token is None:
        # Base models often lack a pad token. Padding positions are excluded by
        # the attention mask, so reusing EOS is safe here.
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def load_model(
    model_name: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    dtype: str = "bfloat16",
    device_map: str | dict | None = "auto",
    quantization: str | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a causal LM for inference-only activation extraction.

    ``quantization`` may be ``None``, ``"4bit"`` or ``"8bit"`` (bitsandbytes).
    Quantisation perturbs the very activations we are measuring, so it is off by
    default; use it only to fit a model you otherwise cannot run, and never mix
    quantised and unquantised activations in one analysis.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict = {
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        warnings.warn(
            f"Loading with {quantization} quantisation: residual-stream activations will "
            "differ from a full-precision run. Do not compare directions across "
            "quantisation settings."
        )
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype(dtype),
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization not in (None, "none"):
        raise ValueError(f"unsupported quantization {quantization!r}; use None, '4bit' or '8bit'")
    else:
        kwargs["dtype"] = torch_dtype(dtype)

    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model.config.use_cache = False
    if device_map is None:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def model_input_device(model):
    """Device that batched inputs should be moved to."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        import torch

        return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Text preparation
# --------------------------------------------------------------------------- #

def thinking_flag_effect(tokenizer, chat_role: str = "user") -> dict:
    """Does ``enable_thinking`` actually change what this tokenizer renders?

    Determined by **rendering the same message both ways and comparing**, not by
    inspecting the template string. ``apply_chat_template`` forwards unknown keyword
    arguments into the Jinja context and silently ignores them, so a template that has
    never heard of ``enable_thinking`` accepts the flag without complaint -- and a gate
    that printed ``enable_thinking=False`` on the strength of having passed it would be
    reporting a setting that did nothing. This is the difference between "thinking is off"
    and "we asked".

    Returns ``supported`` plus whether either rendering opens a reasoning block, so a gate
    can say which of the three states it is in.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return {"supported": False, "reason": "no chat template"}
    probe = [{"role": chat_role, "content": "probe"}]

    def render(flag: bool | None) -> str | None:
        kwargs = {} if flag is None else {"enable_thinking": flag}
        try:
            return tokenizer.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=True, **kwargs
            )
        except Exception:  # pragma: no cover - a template that rejects the kwarg outright
            return None

    default, enabled, disabled = render(None), render(True), render(False)
    supported = (
        enabled is not None and disabled is not None and enabled != disabled
    )
    def has_unclosed_think_block(rendered: str | None) -> bool:
        """Whether the generation prompt ends inside a reasoning block.

        Qwen's non-thinking rendering contains an *already closed*
        ``<think>\n\n</think>`` pair.  Checking only for the opening marker therefore
        reports thinking as on when it is actually disabled.  What matters for
        generation is whether the final opening marker occurs after the final close.
        """
        if not rendered:
            return False
        return rendered.rfind("<think>") > rendered.rfind("</think>")

    return {
        "supported": supported,
        "default_opens_think_block": has_unclosed_think_block(default),
        "enabled_opens_think_block": has_unclosed_think_block(enabled),
        "disabled_opens_think_block": has_unclosed_think_block(disabled),
        "default_matches_enabled": default == enabled,
        "reason": "" if supported else "the template renders identically either way, so "
                                       "the flag is a no-op here",
    }


def prepare_texts(
    stories: Sequence[str],
    tokenizer,
    use_chat_template: bool = False,
    chat_role: str = "user",
    chat_add_generation_prompt: bool = False,
    prefix: str = "",
    suffix: str = "",
    enable_thinking: bool = True,
) -> list[str]:
    """Turn raw stories into the exact strings that get tokenized.

    By default stories are passed through verbatim: the paper extracts
    activations over story text, not over a chat transcript. ``use_chat_template``
    exists for the later experimenter-binding work (assistant / first-person
    persona / third-person character conditions), where the story must sit inside
    a conversation.

    ``enable_thinking`` defaults to ``True`` -- the template's own default, so existing
    callers are unaffected -- and is forwarded only when
    :func:`thinking_flag_effect` says the template responds to it.

    **Why the flag exists.** Qwen3 is a hybrid reasoning model: its chat template opens a
    ``<think>`` block unless told not to, so a generation budget is spent on reasoning
    before any answer appears. At 150 new tokens with greedy decoding, nearly every
    completion is a *truncated* reasoning trace -- and a truncated reasoning trace is
    indistinguishable from an answer to a regex scorer. "Let me consider option A first"
    is scored as choosing A. That is not a scoring bug that can be patched downstream: the
    answer was never generated, so the fix has to be here, and the resolved value has to
    be printed at every gate that generates.
    """
    texts = [f"{prefix}{s}{suffix}" for s in stories]
    if not use_chat_template:
        return texts

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"use_chat_template=True but {tokenizer.name_or_path} has no chat template "
            "(base models usually do not). Use an -Instruct checkpoint or set it to False."
        )
    kwargs = (
        {"enable_thinking": enable_thinking}
        if thinking_flag_effect(tokenizer, chat_role)["supported"] else {}
    )
    return [
        tokenizer.apply_chat_template(
            [{"role": chat_role, "content": t}],
            tokenize=False,
            add_generation_prompt=chat_add_generation_prompt,
            **kwargs,
        )
        for t in texts
    ]
