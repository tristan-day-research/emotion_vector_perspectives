"""Offset-aware, padding-agnostic mean pooling of residual-stream activations.

Anthropic (Sofroniew et al., 2026) pool residual-stream activations by
"averaging across all token positions within each story, beginning with the 50th
token (at which point the emotional content should be apparent)".

The subtlety this module exists to remove: ``hidden[:, offset:seq_len]`` is only
correct for right-padded batches. With left padding, the first ``offset``
positions are pad tokens, so that slice would average pad activations and keep
the story's opening tokens. Instead we rank tokens by their position *among real
tokens* using the cumulative sum of the attention mask, which is correct for
left, right, or any other padding layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def build_pool_mask(attention_mask: torch.Tensor, offset: int) -> torch.Tensor:
    """Mask selecting real tokens strictly after the first ``offset`` real tokens.

    Args:
        attention_mask: ``(batch, seq)`` tensor of 0/1 (1 = real token).
        offset: number of leading real tokens to exclude (50 in the paper).

    Returns:
        Bool tensor ``(batch, seq)``; ``True`` at positions to pool.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be (batch, seq), got {tuple(attention_mask.shape)}")
    if offset < 0:
        raise ValueError("offset must be >= 0")

    real = attention_mask.bool()
    # rank[i, t] = how many real tokens occur at or before position t (1-indexed
    # on real positions). Independent of where padding sits.
    rank = real.long().cumsum(dim=1)
    return real & (rank > offset)


def real_token_counts(attention_mask: torch.Tensor) -> torch.Tensor:
    """Number of real (non-pad) tokens per sequence."""
    return attention_mask.bool().long().sum(dim=1)


@dataclass
class PooledBatch:
    """Result of pooling one batch.

    Attributes:
        pooled: ``{layer_index: (batch, hidden)}`` pooled activations.
        n_real_tokens: ``(batch,)`` real token count per example.
        n_pooled_tokens: ``(batch,)`` tokens actually averaged (0 => too short).
        keep: ``(batch,)`` bool, ``True`` where enough tokens remained.
    """

    pooled: dict[int, torch.Tensor]
    n_real_tokens: torch.Tensor
    n_pooled_tokens: torch.Tensor
    keep: torch.Tensor


def pool_hidden_states(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
    attention_mask: torch.Tensor,
    layers: list[int],
    offset: int,
    min_pooled_tokens: int = 1,
    compute_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> PooledBatch:
    """Mean-pool selected layers over real tokens after ``offset``.

    The pooling reduction runs in ``compute_dtype`` (fp32 by default) because a
    bf16 sum over ~130 tokens loses meaningful precision; the result is cast to
    ``out_dtype`` only for storage.

    Args:
        hidden_states: as returned by ``output_hidden_states=True``; index 0 is
            the embedding output and index ``i`` the output of block ``i``.
        attention_mask: ``(batch, seq)`` 0/1 mask.
        layers: hidden-state indices to pool.
        offset: leading real tokens to exclude.
        min_pooled_tokens: minimum tokens required after the offset; examples
            below this are flagged ``keep=False`` and their pooled values are
            meaningless (callers must drop them).
        compute_dtype: dtype for the sum/divide.
        out_dtype: dtype of the returned tensors.
    """
    pool_mask = build_pool_mask(attention_mask, offset)
    n_real = real_token_counts(attention_mask)
    n_pooled = pool_mask.long().sum(dim=1)
    keep = n_pooled >= max(1, min_pooled_tokens)

    weights = pool_mask.to(compute_dtype).unsqueeze(-1)  # (B, T, 1)
    denom = weights.sum(dim=1).clamp(min=1.0)  # (B, 1); clamp avoids nan for skipped rows

    pooled: dict[int, torch.Tensor] = {}
    for layer in layers:
        h = hidden_states[layer]
        if h.shape[:2] != attention_mask.shape:
            raise ValueError(
                f"layer {layer} hidden state {tuple(h.shape)} incompatible with "
                f"attention mask {tuple(attention_mask.shape)}"
            )
        summed = (h.to(compute_dtype) * weights).sum(dim=1)  # (B, D)
        pooled[layer] = (summed / denom).to(out_dtype)

    return PooledBatch(
        pooled=pooled,
        n_real_tokens=n_real.cpu(),
        n_pooled_tokens=n_pooled.cpu(),
        keep=keep.cpu(),
    )


def _selftest() -> None:
    """Verify left/right padding equivalence, offset semantics, and skip logic."""
    torch.manual_seed(0)
    d = 4
    # 7 and 12 survive the offset; 3 leaves exactly one token; 2 leaves none.
    lengths = [7, 12, 3, 2]
    offset = 2
    seqs = [torch.randn(n, d) for n in lengths]
    max_len = max(lengths)

    def pad(side: str):
        hs, am = [], []
        for s in seqs:
            n_pad = max_len - s.shape[0]
            pad_block = torch.full((n_pad, d), 999.0)  # poison: must never be averaged
            if side == "right":
                hs.append(torch.cat([s, pad_block]))
                am.append(torch.cat([torch.ones(s.shape[0]), torch.zeros(n_pad)]))
            else:
                hs.append(torch.cat([pad_block, s]))
                am.append(torch.cat([torch.zeros(n_pad), torch.ones(s.shape[0])]))
        return torch.stack(hs), torch.stack(am).long()

    results = {}
    for side in ("left", "right"):
        h, am = pad(side)
        out = pool_hidden_states([h], am, layers=[0], offset=offset, out_dtype=torch.float32)
        results[side] = out

    torch.testing.assert_close(results["left"].pooled[0], results["right"].pooled[0])

    # Match an explicit per-sequence reference.
    for i, s in enumerate(seqs):
        if s.shape[0] <= offset:
            assert not results["right"].keep[i], f"seq {i} should be skipped"
            continue
        expected = s[offset:].mean(dim=0)
        torch.testing.assert_close(results["right"].pooled[0][i], expected)
        assert int(results["right"].n_pooled_tokens[i]) == s.shape[0] - offset
        assert int(results["right"].n_real_tokens[i]) == s.shape[0]

    # min_pooled_tokens=1 (default): only the sequence with nothing left is dropped.
    assert results["right"].keep.tolist() == [True, True, True, False]
    assert results["right"].n_pooled_tokens.tolist() == [5, 10, 1, 0]

    # A stricter threshold drops the 1-token case too, and pooled values stay finite
    # (no division by zero) for the rows it rejects.
    h, am = pad("left")
    strict = pool_hidden_states(
        [h], am, layers=[0], offset=offset, min_pooled_tokens=3, out_dtype=torch.float32
    )
    assert strict.keep.tolist() == [True, True, False, False]
    assert torch.isfinite(strict.pooled[0]).all(), "skipped rows must not produce nan/inf"

    print("core.pooling selftest OK (left/right equivalent, offset exact, skips correct)")


if __name__ == "__main__":
    _selftest()
