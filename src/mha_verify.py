"""Static contract verifier for ``nn.MultiheadAttention``.

PyTorch's MultiheadAttention accepts both packed q/k/v projections and
separate k/v projection dimensions, two input layouts, optional score masks,
optional key-padding masks, and two different attention-weight output layouts.
This module models those shape rules without executing kernels.  Consistent
with TensorGuard's soundness posture, only concrete, provable mismatches are
refuted; symbolic dimensions are carried through without false positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]

__all__ = ["MultiheadAttentionVerdict", "verify_multihead_attention"]


@dataclass
class MultiheadAttentionVerdict:
    ok: bool
    output_shape: Optional[Tuple[Dim, ...]] = None
    weights_shape: Optional[Tuple[Dim, ...]] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover
        return self.ok


def _as_shape(shape: Sequence[Dim]) -> Tuple[Dim, ...]:
    return tuple(shape)


def _is_int_dim(d: Dim) -> bool:
    return isinstance(d, int) and not isinstance(d, bool)


def _dim_mismatch(a: Dim, b: Dim) -> bool:
    return _is_int_dim(a) and _is_int_dim(b) and a != b


def _known_product(a: Dim, b: Dim) -> Optional[int]:
    if _is_int_dim(a) and _is_int_dim(b):
        return int(a) * int(b)
    return None


def _fail(kind: str, message: str) -> MultiheadAttentionVerdict:
    return MultiheadAttentionVerdict(False, error=message, error_kind=kind)


def _layout(
    shape: Tuple[Dim, ...],
    *,
    batch_first: bool,
) -> Tuple[bool, Dim, Optional[Dim], Dim]:
    """Return ``(batched, target_len, batch, embed)`` for q/k/v shapes."""
    if len(shape) == 2:
        return False, shape[0], None, shape[1]
    if batch_first:
        return True, shape[1], shape[0], shape[2]
    return True, shape[0], shape[1], shape[2]


def _output_like_query(query: Tuple[Dim, ...], embed_dim: Dim) -> Tuple[Dim, ...]:
    return tuple(query[:-1]) + (embed_dim,)


def _weights_shape(
    *,
    batched: bool,
    batch: Optional[Dim],
    target_len: Dim,
    source_len: Dim,
    num_heads: Dim,
    need_weights: bool,
    average_attn_weights: bool,
) -> Optional[Tuple[Dim, ...]]:
    if not need_weights:
        return None
    if batched:
        n = batch if batch is not None else "N"
        if average_attn_weights:
            return (n, target_len, source_len)
        return (n, num_heads, target_len, source_len)
    if average_attn_weights:
        return (target_len, source_len)
    return (num_heads, target_len, source_len)


def verify_multihead_attention(
    query: Sequence[Dim],
    key: Sequence[Dim],
    value: Sequence[Dim],
    embed_dim: Dim,
    num_heads: Dim,
    *,
    kdim: Optional[Dim] = None,
    vdim: Optional[Dim] = None,
    batch_first: bool = False,
    attn_mask: Optional[Sequence[Dim]] = None,
    key_padding_mask: Optional[Sequence[Dim]] = None,
    need_weights: bool = True,
    average_attn_weights: bool = True,
    use_separate_proj_weight: Optional[bool] = None,
    nested_tensor: bool = False,
) -> MultiheadAttentionVerdict:
    """Verify one MultiheadAttention call from operand shapes.

    ``batch_first`` models the ``nn.MultiheadAttention`` module convention.  The
    lower-level functional API is always sequence-first, so callers checking
    ``torch.nn.functional.multi_head_attention_forward`` should leave it false.
    """
    if nested_tensor:
        return MultiheadAttentionVerdict(
            True,
            abstained=True,
            abstain_reason=(
                "nested tensor inputs have ragged, data-dependent sizes; "
                "TensorGuard abstains instead of fabricating a rectangular shape"
            ),
        )

    q = _as_shape(query)
    k = _as_shape(key)
    v = _as_shape(value)

    if len(q) not in (2, 3):
        return _fail(
            "rank",
            f"query must be 2-D or 3-D for MultiheadAttention, got rank {len(q)}",
        )
    if len(k) != len(q) or len(v) != len(q):
        return _fail(
            "rank",
            "key and value must have the same rank as query "
            f"(query={len(q)}, key={len(k)}, value={len(v)})",
        )

    batched, tgt_len, batch, q_embed = _layout(q, batch_first=batch_first)
    _kb, src_len, key_batch, k_embed = _layout(k, batch_first=batch_first)
    _vb, value_src_len, value_batch, v_embed = _layout(v, batch_first=batch_first)

    if _is_int_dim(num_heads) and int(num_heads) <= 0:
        return _fail("num_heads", f"num_heads must be positive, got {num_heads}")
    if (_is_int_dim(embed_dim) and _is_int_dim(num_heads)
            and int(embed_dim) % int(num_heads) != 0):
        return _fail(
            "head_divisibility",
            f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}",
        )

    if _dim_mismatch(q_embed, embed_dim):
        return _fail(
            "query_embed_dim",
            f"query embed dim {q_embed} must match module embed_dim {embed_dim}",
        )

    separate = use_separate_proj_weight
    if separate is None:
        separate = (
            (kdim is not None and kdim != embed_dim)
            or (vdim is not None and vdim != embed_dim)
        )
    expected_k = kdim if separate and kdim is not None else embed_dim
    expected_v = vdim if separate and vdim is not None else embed_dim
    if _dim_mismatch(k_embed, expected_k):
        family = "unpacked kdim" if separate else "packed embed_dim"
        return _fail(
            "key_embed_dim",
            f"key embed dim {k_embed} must match {family} {expected_k}",
        )
    if _dim_mismatch(v_embed, expected_v):
        family = "unpacked vdim" if separate else "packed embed_dim"
        return _fail(
            "value_embed_dim",
            f"value embed dim {v_embed} must match {family} {expected_v}",
        )

    if batched:
        if key_batch is None or value_batch is None:
            return _fail("rank", "batched query requires batched key and value")
        if batch is not None and _dim_mismatch(batch, key_batch):
            return _fail(
                "batch",
                f"query batch {batch} must match key batch {key_batch}",
            )
        if _dim_mismatch(key_batch, value_batch):
            return _fail(
                "batch",
                f"key batch {key_batch} must match value batch {value_batch}",
            )

    if _dim_mismatch(src_len, value_src_len):
        return _fail(
            "source_length",
            f"key source length {src_len} must match value source length {value_src_len}",
        )

    if attn_mask is not None:
        mask = _as_shape(attn_mask)
        if len(mask) not in (2, 3):
            return _fail(
                "attn_mask_rank",
                f"attn_mask must be 2-D or 3-D, got rank {len(mask)}",
            )
        if len(mask) == 2:
            if _dim_mismatch(mask[0], tgt_len) or _dim_mismatch(mask[1], src_len):
                return _fail(
                    "attn_mask",
                    f"2-D attn_mask {mask} must have shape ({tgt_len}, {src_len})",
                )
        else:
            expected_lead = (
                _known_product(batch if batch is not None else "N", num_heads)
                if batched else num_heads
            )
            if expected_lead is not None and _dim_mismatch(mask[0], expected_lead):
                return _fail(
                    "attn_mask",
                    f"3-D attn_mask leading dim {mask[0]} must be {expected_lead}",
                )
            if _dim_mismatch(mask[1], tgt_len) or _dim_mismatch(mask[2], src_len):
                return _fail(
                    "attn_mask",
                    "3-D attn_mask "
                    f"{mask} must have trailing shape ({tgt_len}, {src_len})",
                )

    if key_padding_mask is not None:
        kpm = _as_shape(key_padding_mask)
        expected = (batch, src_len) if batched else (src_len,)
        if len(kpm) != len(expected):
            return _fail(
                "key_padding_mask",
                f"key_padding_mask rank {len(kpm)} must be {len(expected)}",
            )
        for got, want in zip(kpm, expected):
            if want is not None and _dim_mismatch(got, want):
                return _fail(
                    "key_padding_mask",
                    f"key_padding_mask {kpm} must have shape {expected}",
                )

    out = _output_like_query(q, embed_dim)
    weights = _weights_shape(
        batched=batched,
        batch=batch,
        target_len=tgt_len,
        source_len=src_len,
        num_heads=num_heads,
        need_weights=bool(need_weights),
        average_attn_weights=bool(average_attn_weights),
    )
    return MultiheadAttentionVerdict(True, output_shape=out, weights_shape=weights)
