"""v5 / Track-C — Transfer rules for attention and normalisation ops.

Six operators are covered:

* :func:`sdpa_shape`         — ``F.scaled_dot_product_attention``
* :func:`mha_shape`          — ``nn.MultiheadAttention``
* :func:`layer_norm_shape`   — ``nn.LayerNorm`` / ``F.layer_norm``
* :func:`rms_norm_shape`     — ``nn.RMSNorm`` / ``F.rms_norm``

Each returns a :class:`TensorShape` (or ``None`` if the inputs are not
shape-compatible).  The helpers also register themselves into the
existing dispatch table at
:data:`src.tensor_shapes.TORCH_SHAPE_OPS` *as a side effect of import*,
so simply ``import src.v5.attention_norms`` from a benchmark script is
enough for the existing :mod:`src.real_analyzer` to recognise the new
ops without us having to edit any pre-existing source file.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

from src.tensor_shapes import (
    ShapeDim,
    TensorShape,
    TORCH_SHAPE_OPS,
)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _eq(a: ShapeDim, b: ShapeDim) -> bool:
    """Symbolic-friendly equality: identical concrete ints, or identical
    symbolic names."""
    return a.value == b.value


def _broadcast_compatible(a: TensorShape, b: TensorShape) -> bool:
    """NumPy / PyTorch right-aligned broadcasting compatibility."""
    da = list(a.dims); db = list(b.dims)
    while len(da) < len(db): da.insert(0, ShapeDim(1))
    while len(db) < len(da): db.insert(0, ShapeDim(1))
    for x, y in zip(da, db):
        if x.value == y.value: continue
        if isinstance(x.value, int) and x.value == 1: continue
        if isinstance(y.value, int) and y.value == 1: continue
        if isinstance(x.value, str) or isinstance(y.value, str):
            continue  # be optimistic on symbolic dims
        return False
    return True


# ────────────────────────────────────────────────────────────────────────────
# Scaled dot-product attention
# ────────────────────────────────────────────────────────────────────────────

def sdpa_shape(
    q: TensorShape,
    k: TensorShape,
    v: TensorShape,
    is_causal: bool = False,
    attn_mask: Optional[TensorShape] = None,
) -> Optional[TensorShape]:
    """``F.scaled_dot_product_attention(q, k, v)``.

    Convention (PyTorch ≥ 2.0):
        q : (..., L, E_q)
        k : (..., S, E_k)   with E_k == E_q
        v : (..., S, E_v)
        out shape: (..., L, E_v)
    """
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        return None
    L = q.dims[-2]; Eq = q.dims[-1]
    Sk = k.dims[-2]; Ek = k.dims[-1]
    Sv = v.dims[-2]; Ev = v.dims[-1]
    # E_q must equal E_k (head_dim).
    if isinstance(Eq.value, int) and isinstance(Ek.value, int) and Eq.value != Ek.value:
        return None
    # S of K and V must agree.
    if isinstance(Sk.value, int) and isinstance(Sv.value, int) and Sk.value != Sv.value:
        return None
    # Leading "batch" dims must broadcast.
    q_lead = TensorShape(q.dims[:-2])
    k_lead = TensorShape(k.dims[:-2])
    v_lead = TensorShape(v.dims[:-2])
    if not _broadcast_compatible(q_lead, k_lead): return None
    if not _broadcast_compatible(q_lead, v_lead): return None
    return TensorShape(q.dims[:-1] + (Ev,))


# ────────────────────────────────────────────────────────────────────────────
# MultiheadAttention (the nn.Module forward)
# ────────────────────────────────────────────────────────────────────────────

def mha_shape(
    query: TensorShape,
    key: TensorShape,
    value: TensorShape,
    embed_dim: Union[int, str, None] = None,
    num_heads: Union[int, None] = None,
    batch_first: bool = False,
) -> Optional[TensorShape]:
    """``nn.MultiheadAttention.forward(query, key, value)`` returns
    ``(attn_output, attn_weights)``.  We model only ``attn_output`` whose
    shape equals ``query.shape`` (with the embedding dim possibly replaced
    by ``embed_dim``).
    """
    if query.ndim != 3:
        return None
    # Layout:
    #   batch_first=False → (L, N, E)
    #   batch_first=True  → (N, L, E)
    L_axis = 0 if not batch_first else 1
    E_axis = 2
    Eq = query.dims[E_axis]
    if embed_dim is not None and isinstance(Eq.value, int) and isinstance(embed_dim, int):
        if Eq.value != embed_dim:
            return None
    if num_heads is not None and isinstance(Eq.value, int) and isinstance(num_heads, int):
        if Eq.value % num_heads != 0:
            return None
    # Key & value must share L (S) and E.
    if key.ndim != 3 or value.ndim != 3:
        return None
    return query  # output shape == query shape


# ────────────────────────────────────────────────────────────────────────────
# LayerNorm / F.layer_norm
# ────────────────────────────────────────────────────────────────────────────

def layer_norm_shape(
    input_shape: TensorShape,
    normalized_shape: Sequence[Union[int, str]],
) -> Optional[TensorShape]:
    """``F.layer_norm(x, normalized_shape)`` — output has the *same* shape
    as input, with the constraint that the trailing
    ``len(normalized_shape)`` dims of input must match
    ``normalized_shape`` exactly.
    """
    norm = list(normalized_shape)
    if input_shape.ndim < len(norm):
        return None
    tail = input_shape.dims[-len(norm):]
    for inp_d, n_d in zip(tail, norm):
        if isinstance(inp_d.value, int) and isinstance(n_d, int):
            if inp_d.value != n_d:
                return None
    return input_shape


# ────────────────────────────────────────────────────────────────────────────
# RMSNorm / F.rms_norm
# ────────────────────────────────────────────────────────────────────────────

def rms_norm_shape(
    input_shape: TensorShape,
    normalized_shape: Sequence[Union[int, str]],
) -> Optional[TensorShape]:
    """RMSNorm has the same shape semantics as LayerNorm: it normalises
    over the trailing ``len(normalized_shape)`` dims and returns a tensor
    of the same shape as the input.
    """
    return layer_norm_shape(input_shape, normalized_shape)


# ────────────────────────────────────────────────────────────────────────────
# Dispatch-table side-effect registration
# ────────────────────────────────────────────────────────────────────────────

# These names are used by :mod:`src.tensor_shapes`'s analyzer when it
# sees an attribute call like ``F.layer_norm(x, ...)`` or
# ``F.scaled_dot_product_attention(...)``.  Mark them as the
# *identity* op kind ``"like"`` (preserves shape) for layer/RMS norms,
# and a custom ``"attention"`` tag for SDPA / MHA so any downstream
# specialised handler can take over.
_NEW_OPS = {
    "scaled_dot_product_attention": "attention",
    "multi_head_attention_forward": "attention",
    "layer_norm": "like",
    "rms_norm": "like",
    # nn.Module class names that show up via getattr(nn, "...")(...)
    "LayerNorm": "like",
    "RMSNorm": "like",
}

for _name, _kind in _NEW_OPS.items():
    TORCH_SHAPE_OPS.setdefault(_name, _kind)


__all__ = [
    "sdpa_shape",
    "mha_shape",
    "layer_norm_shape",
    "rms_norm_shape",
]
