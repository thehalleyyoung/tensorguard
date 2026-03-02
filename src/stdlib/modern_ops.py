"""
Modern PyTorch operator shape transfer functions.

Adds shape inference rules for operators missing from TensorGuard's
original coverage (~145 ops), targeting patterns used in contemporary
architectures: Transformers with RoPE, MoE models, vision transformers,
diffusion models, and einops-style manipulation.

Categories:
  - Attention operators (scaled_dot_product_attention, multi_head_attention)
  - Normalization (group_norm, instance_norm, layer_norm variants)
  - Activation (GELU, SiLU, Mish, etc.)
  - Sparse / MoE (sparse_softmax, top-k gating)
  - Positional encoding (rotary embedding shape semantics)
  - Memory-efficient (checkpoint, gradient_checkpoint)
  - Einops-style (rearrange, repeat, reduce)
  - Additional standard ops (adaptive pooling, pixel shuffle, etc.)

Each function follows the project convention: accept TensorShape(s) and
keyword config, return Optional[TensorShape].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.tensor_shapes import TensorShape, ShapeDim


# ═══════════════════════════════════════════════════════════════════════════
# Attention operators
# ═══════════════════════════════════════════════════════════════════════════

def transfer_scaled_dot_product_attention(
    query: TensorShape,
    key: TensorShape,
    value: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for F.scaled_dot_product_attention.

    Inputs:  query  (B, ..., L, E)
             key    (B, ..., S, E)
             value  (B, ..., S, Ev)
    Output:  (B, ..., L, Ev)

    The last dim of the output equals the last dim of value.
    """
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        return None
    batch_dims = query.dims[:-2]
    seq_len = query.dims[-2]   # L
    ev = value.dims[-1]        # Ev
    return TensorShape(batch_dims + (seq_len, ev))


def transfer_multi_head_attention(
    query: TensorShape,
    key: TensorShape,
    value: TensorShape,
    num_heads: int,
    embed_dim: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.MultiheadAttention forward.

    Inputs:  query (L, B, E) or (B, L, E) depending on batch_first
             key   (S, B, E) or (B, S, E)
             value (S, B, E) or (B, S, E)
    Output:  attn_output same shape as query, attn_weights (B, L, S)
    Returns: the shape of attn_output (same as query).
    """
    if query.ndim < 2:
        return None
    return TensorShape(query.dims)


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

def transfer_group_norm(
    input_shape: TensorShape,
    num_groups: int,
    num_channels: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.GroupNorm: output shape == input shape.

    Requires input_shape[1] == num_channels and
    num_channels % num_groups == 0.
    """
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims)


def transfer_instance_norm(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for nn.InstanceNorm{1,2,3}d: output shape == input shape."""
    if input_shape.ndim < 3:
        return None
    return TensorShape(input_shape.dims)


def transfer_layer_norm(
    input_shape: TensorShape,
    normalized_shape: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for nn.LayerNorm: output shape == input shape.

    The trailing dims of input must match normalized_shape.
    """
    if input_shape.ndim < len(normalized_shape):
        return None
    return TensorShape(input_shape.dims)


def transfer_rms_norm(
    input_shape: TensorShape,
    normalized_shape: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for RMSNorm (LLaMA-style): output shape == input shape."""
    if input_shape.ndim < len(normalized_shape):
        return None
    return TensorShape(input_shape.dims)


# ═══════════════════════════════════════════════════════════════════════════
# Activation functions (element-wise → shape preserved)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_elementwise(input_shape: TensorShape) -> Optional[TensorShape]:
    """Shape rule for any element-wise activation (GELU, SiLU, Mish, etc.).

    Output shape is always identical to input shape.
    """
    return TensorShape(input_shape.dims)


# Aliases for documentation / explicit registration
transfer_gelu = transfer_elementwise
transfer_silu = transfer_elementwise
transfer_mish = transfer_elementwise
transfer_swiglu_act = transfer_elementwise  # activation part only


def transfer_glu(
    input_shape: TensorShape,
    dim: int = -1,
) -> Optional[TensorShape]:
    """Shape rule for F.glu / nn.GLU.

    Splits input along *dim* into two halves and applies sigmoid gate.
    Output shape has dim halved on that axis.
    """
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    d = input_shape.dims[dim]
    if d.is_symbolic:
        result_dims = list(input_shape.dims)
        result_dims[dim] = ShapeDim(f"{d.value}_half")
        return TensorShape(tuple(result_dims))
    if d.value % 2 != 0:
        return None
    result_dims = list(input_shape.dims)
    result_dims[dim] = ShapeDim(d.value // 2)
    return TensorShape(tuple(result_dims))


# ═══════════════════════════════════════════════════════════════════════════
# Sparse / MoE operators
# ═══════════════════════════════════════════════════════════════════════════

def transfer_sparse_softmax(
    input_shape: TensorShape,
    dim: int = -1,
) -> Optional[TensorShape]:
    """Shape rule for sparse_softmax (e.g., entmax / α-entmax).

    Output shape == input shape; sparsity is in the values, not the shape.
    """
    return TensorShape(input_shape.dims)


def transfer_topk(
    input_shape: TensorShape,
    k: int,
    dim: int = -1,
) -> Optional[TensorShape]:
    """Shape rule for torch.topk.

    Returns (values, indices) each of shape (..., k, ...) where the
    *dim* axis is replaced by k.
    """
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    result_dims[dim] = ShapeDim(k)
    return TensorShape(tuple(result_dims))


def transfer_moe_routing(
    input_shape: TensorShape,
    num_experts: int,
    top_k: int,
) -> Optional[TensorShape]:
    """Shape rule for MoE top-k gating / routing.

    Input:  (batch, seq_len, hidden) or (tokens, hidden)
    Output: dispatched tensor has shape (..., top_k, hidden).
    The last-but-one dim is the number of selected experts.
    """
    if input_shape.ndim < 2:
        return None
    prefix = input_shape.dims[:-1]
    hidden = input_shape.dims[-1]
    return TensorShape(prefix + (ShapeDim(top_k), hidden))


def transfer_moe_gate_scores(
    input_shape: TensorShape,
    num_experts: int,
) -> Optional[TensorShape]:
    """Shape rule for MoE gating scores.

    Input:  (..., hidden)
    Output: (..., num_experts)  — logits over experts.
    """
    if input_shape.ndim < 1:
        return None
    prefix = input_shape.dims[:-1]
    return TensorShape(prefix + (ShapeDim(num_experts),))


# ═══════════════════════════════════════════════════════════════════════════
# Positional encoding
# ═══════════════════════════════════════════════════════════════════════════

def transfer_rotary_embedding(
    input_shape: TensorShape,
    dim: int,
) -> Optional[TensorShape]:
    """Shape rule for rotary positional embedding (RoPE).

    RoPE applies in-place rotation to the last *dim* dimensions
    (must be even).  Output shape == input shape.
    """
    if input_shape.ndim < 1:
        return None
    return TensorShape(input_shape.dims)


def transfer_sinusoidal_pos_encoding(
    seq_len: int,
    d_model: int,
) -> Optional[TensorShape]:
    """Shape rule for classic sinusoidal positional encoding.

    Output: (seq_len, d_model)
    """
    return TensorShape((ShapeDim(seq_len), ShapeDim(d_model)))


def transfer_alibi_bias(
    num_heads: int,
    seq_len: int,
) -> Optional[TensorShape]:
    """Shape rule for ALiBi position bias.

    Output: (num_heads, seq_len, seq_len)
    """
    return TensorShape((
        ShapeDim(num_heads),
        ShapeDim(seq_len),
        ShapeDim(seq_len),
    ))


# ═══════════════════════════════════════════════════════════════════════════
# Memory-efficient / checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def transfer_checkpoint(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.utils.checkpoint.checkpoint.

    Checkpointing is purely a memory optimisation; the output shape is
    identical to whatever the wrapped function produces.  We conservatively
    return the same shape as the input.
    """
    return TensorShape(input_shape.dims)


def transfer_gradient_checkpoint(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Alias for torch.utils.checkpoint — same semantics."""
    return transfer_checkpoint(input_shape)


# ═══════════════════════════════════════════════════════════════════════════
# Einops-style operators
# ═══════════════════════════════════════════════════════════════════════════

def _parse_einops_pattern(pattern: str) -> Tuple[List[str], List[str]]:
    """Parse an einops rearrange pattern 'a b c -> a (b c)' into axes."""
    if '->' not in pattern:
        return [], []
    lhs, rhs = pattern.split('->')
    lhs_tokens = re.findall(r'[a-zA-Z_]\w*|\(.*?\)', lhs)
    rhs_tokens = re.findall(r'[a-zA-Z_]\w*|\(.*?\)', rhs)
    return lhs_tokens, rhs_tokens


def transfer_rearrange(
    input_shape: TensorShape,
    pattern: str,
    **axes_lengths: int,
) -> Optional[TensorShape]:
    """Shape rule for einops.rearrange.

    Parses the pattern string and maps symbolic dimensions.
    E.g. 'b c h w -> b (c h) w' with input (B, C, H, W)
    produces (B, C*H, W).
    """
    lhs_tokens, rhs_tokens = _parse_einops_pattern(pattern)
    if not lhs_tokens or not rhs_tokens:
        return None

    # Build axis name → ShapeDim from LHS + input_shape
    flat_lhs: List[str] = []
    for tok in lhs_tokens:
        if tok.startswith('(') and tok.endswith(')'):
            inner = re.findall(r'[a-zA-Z_]\w*', tok)
            flat_lhs.extend(inner)
        else:
            flat_lhs.append(tok)

    if len(flat_lhs) != input_shape.ndim:
        return None

    axis_map: Dict[str, ShapeDim] = {}
    for name, dim in zip(flat_lhs, input_shape.dims):
        axis_map[name] = dim
    for name, val in axes_lengths.items():
        axis_map[name] = ShapeDim(val)

    # Build output shape from RHS
    out_dims: List[ShapeDim] = []
    for tok in rhs_tokens:
        if tok.startswith('(') and tok.endswith(')'):
            inner = re.findall(r'[a-zA-Z_]\w*', tok)
            product = 1
            all_concrete = True
            for ax in inner:
                d = axis_map.get(ax)
                if d is None:
                    return None
                if d.is_symbolic:
                    all_concrete = False
                else:
                    product *= d.value
            if all_concrete:
                out_dims.append(ShapeDim(product))
            else:
                out_dims.append(ShapeDim("_rearranged"))
        else:
            d = axis_map.get(tok)
            if d is None:
                return None
            out_dims.append(d)
    return TensorShape(tuple(out_dims))


def transfer_einops_repeat(
    input_shape: TensorShape,
    pattern: str,
    **axes_lengths: int,
) -> Optional[TensorShape]:
    """Shape rule for einops.repeat.

    E.g. 'b c -> b c n' with n=3 produces (..., 3).
    """
    lhs_tokens, rhs_tokens = _parse_einops_pattern(pattern)
    if not lhs_tokens or not rhs_tokens:
        return None

    flat_lhs = [t for t in lhs_tokens
                if not (t.startswith('(') and t.endswith(')'))]
    if len(flat_lhs) != input_shape.ndim:
        return None

    axis_map: Dict[str, ShapeDim] = {}
    for name, dim in zip(flat_lhs, input_shape.dims):
        axis_map[name] = dim
    for name, val in axes_lengths.items():
        axis_map[name] = ShapeDim(val)

    out_dims: List[ShapeDim] = []
    for tok in rhs_tokens:
        if tok.startswith('(') and tok.endswith(')'):
            inner = re.findall(r'[a-zA-Z_]\w*', tok)
            product = 1
            all_concrete = True
            for ax in inner:
                d = axis_map.get(ax)
                if d is None:
                    return None
                if d.is_symbolic:
                    all_concrete = False
                else:
                    product *= d.value
            out_dims.append(ShapeDim(product) if all_concrete
                           else ShapeDim("_repeated"))
        else:
            d = axis_map.get(tok)
            if d is None:
                return None
            out_dims.append(d)
    return TensorShape(tuple(out_dims))


def transfer_einops_reduce(
    input_shape: TensorShape,
    pattern: str,
    reduction: str = "mean",
) -> Optional[TensorShape]:
    """Shape rule for einops.reduce.

    E.g. 'b c h w -> b c' with reduction='mean' removes h, w dims.
    """
    lhs_tokens, rhs_tokens = _parse_einops_pattern(pattern)
    if not lhs_tokens or not rhs_tokens:
        return None

    flat_lhs = [t for t in lhs_tokens
                if not (t.startswith('(') and t.endswith(')'))]
    if len(flat_lhs) != input_shape.ndim:
        return None

    axis_map: Dict[str, ShapeDim] = {}
    for name, dim in zip(flat_lhs, input_shape.dims):
        axis_map[name] = dim

    out_dims: List[ShapeDim] = []
    for tok in rhs_tokens:
        if tok.startswith('(') and tok.endswith(')'):
            inner = re.findall(r'[a-zA-Z_]\w*', tok)
            product = 1
            all_concrete = True
            for ax in inner:
                d = axis_map.get(ax)
                if d is None:
                    return None
                if d.is_symbolic:
                    all_concrete = False
                else:
                    product *= d.value
            out_dims.append(ShapeDim(product) if all_concrete
                           else ShapeDim("_reduced"))
        else:
            d = axis_map.get(tok)
            if d is None:
                return None
            out_dims.append(d)
    return TensorShape(tuple(out_dims))


# ═══════════════════════════════════════════════════════════════════════════
# Additional common modern ops
# ═══════════════════════════════════════════════════════════════════════════

def transfer_adaptive_avg_pool(
    input_shape: TensorShape,
    output_size: Union[int, Tuple[int, ...]],
) -> Optional[TensorShape]:
    """Shape rule for nn.AdaptiveAvgPool{1,2,3}d.

    Keeps batch + channel dims, replaces spatial dims with output_size.
    """
    if input_shape.ndim < 3:
        return None
    prefix = input_shape.dims[:2]
    if isinstance(output_size, int):
        spatial_ndim = input_shape.ndim - 2
        suffix = tuple(ShapeDim(output_size) for _ in range(spatial_ndim))
    else:
        suffix = tuple(ShapeDim(s) for s in output_size)
    return TensorShape(prefix + suffix)


def transfer_pixel_shuffle(
    input_shape: TensorShape,
    upscale_factor: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.PixelShuffle.

    Input:  (B, C * r^2, H, W)
    Output: (B, C, H * r, W * r)
    """
    if input_shape.ndim != 4:
        return None
    b = input_shape.dims[0]
    c_in = input_shape.dims[1]
    h = input_shape.dims[2]
    w = input_shape.dims[3]
    r2 = upscale_factor * upscale_factor
    if c_in.is_symbolic or h.is_symbolic or w.is_symbolic:
        return TensorShape((b, ShapeDim("_ps_c"),
                            ShapeDim("_ps_h"), ShapeDim("_ps_w")))
    if c_in.value % r2 != 0:
        return None
    return TensorShape((
        b,
        ShapeDim(c_in.value // r2),
        ShapeDim(h.value * upscale_factor),
        ShapeDim(w.value * upscale_factor),
    ))


def transfer_pixel_unshuffle(
    input_shape: TensorShape,
    downscale_factor: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.PixelUnshuffle (inverse of PixelShuffle).

    Input:  (B, C, H * r, W * r)
    Output: (B, C * r^2, H, W)
    """
    if input_shape.ndim != 4:
        return None
    b = input_shape.dims[0]
    c = input_shape.dims[1]
    h = input_shape.dims[2]
    w = input_shape.dims[3]
    r = downscale_factor
    if h.is_symbolic or w.is_symbolic or c.is_symbolic:
        return TensorShape((b, ShapeDim("_pus_c"),
                            ShapeDim("_pus_h"), ShapeDim("_pus_w")))
    if h.value % r != 0 or w.value % r != 0:
        return None
    return TensorShape((
        b,
        ShapeDim(c.value * r * r),
        ShapeDim(h.value // r),
        ShapeDim(w.value // r),
    ))


def transfer_unfold(
    input_shape: TensorShape,
    dim: int,
    size: int,
    step: int,
) -> Optional[TensorShape]:
    """Shape rule for Tensor.unfold.

    Adds a new dimension at the end with given *size*.  The *dim* axis
    length becomes (L - size) // step + 1.
    """
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    d = input_shape.dims[dim]
    result_dims = list(input_shape.dims)
    if d.is_symbolic:
        result_dims[dim] = ShapeDim("_unfold")
    else:
        new_len = (d.value - size) // step + 1
        if new_len <= 0:
            return None
        result_dims[dim] = ShapeDim(new_len)
    result_dims.append(ShapeDim(size))
    return TensorShape(tuple(result_dims))


def transfer_fold(
    input_shape: TensorShape,
    output_size: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for nn.Fold.

    Input:  (B, C * prod(kernel_size), L)
    Output: (B, C, *output_size)
    """
    if input_shape.ndim != 3:
        return None
    b = input_shape.dims[0]
    ck = input_shape.dims[1]
    kprod = 1
    for k in kernel_size:
        kprod *= k
    if ck.is_symbolic:
        c = ShapeDim("_fold_c")
    elif ck.value % kprod != 0:
        return None
    else:
        c = ShapeDim(ck.value // kprod)
    return TensorShape((b, c) + tuple(ShapeDim(s) for s in output_size))


def transfer_cross_attention(
    query: TensorShape,
    key: TensorShape,
    value: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for cross-attention (same as scaled_dot_product_attention)."""
    return transfer_scaled_dot_product_attention(query, key, value)


def transfer_embedding(
    input_shape: TensorShape,
    embedding_dim: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.Embedding.

    Input:  (*) integer indices
    Output: (*, embedding_dim)
    """
    return TensorShape(input_shape.dims + (ShapeDim(embedding_dim),))


def transfer_dropout(input_shape: TensorShape) -> Optional[TensorShape]:
    """Shape rule for nn.Dropout — identity on shape."""
    return TensorShape(input_shape.dims)


def transfer_stochastic_depth(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torchvision StochasticDepth — identity on shape."""
    return TensorShape(input_shape.dims)


def transfer_chunk(
    input_shape: TensorShape,
    chunks: int,
    dim: int = 0,
) -> Optional[TensorShape]:
    """Shape rule for torch.chunk — returns shape of each chunk.

    Divides dim into *chunks* pieces (last may be smaller).
    """
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    d = input_shape.dims[dim]
    result_dims = list(input_shape.dims)
    if d.is_symbolic:
        result_dims[dim] = ShapeDim(f"{d.value}_chunk")
    else:
        chunk_size = (d.value + chunks - 1) // chunks
        result_dims[dim] = ShapeDim(chunk_size)
    return TensorShape(tuple(result_dims))


def transfer_split(
    input_shape: TensorShape,
    split_size: int,
    dim: int = 0,
) -> Optional[TensorShape]:
    """Shape rule for torch.split — returns shape of each split."""
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    result_dims[dim] = ShapeDim(split_size)
    return TensorShape(tuple(result_dims))


def transfer_repeat_interleave(
    input_shape: TensorShape,
    repeats: int,
    dim: Optional[int] = None,
) -> Optional[TensorShape]:
    """Shape rule for torch.repeat_interleave.

    If dim is None, flattens then repeats → 1-D tensor of size numel * repeats.
    If dim is given, the dim axis is scaled by repeats.
    """
    if dim is None:
        total = 1
        all_concrete = True
        for d in input_shape.dims:
            if d.is_symbolic:
                all_concrete = False
                break
            total *= d.value
        if all_concrete:
            return TensorShape((ShapeDim(total * repeats),))
        return TensorShape((ShapeDim("_repeat_flat"),))
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    d = input_shape.dims[dim]
    if d.is_symbolic:
        result_dims[dim] = ShapeDim(f"{d.value}_rep")
    else:
        result_dims[dim] = ShapeDim(d.value * repeats)
    return TensorShape(tuple(result_dims))


def transfer_conv1d(
    input_shape: TensorShape,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Optional[TensorShape]:
    """Shape rule for nn.Conv1d.

    Input:  (B, C_in, L)
    Output: (B, C_out, L_out)
    """
    if input_shape.ndim != 3:
        return None
    b = input_shape.dims[0]
    l_in = input_shape.dims[2]
    if l_in.is_symbolic:
        l_out = ShapeDim("_conv1d_l")
    else:
        l_out_val = (l_in.value + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        if l_out_val <= 0:
            return None
        l_out = ShapeDim(l_out_val)
    return TensorShape((b, ShapeDim(out_channels), l_out))


def transfer_conv_transpose1d(
    input_shape: TensorShape,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> Optional[TensorShape]:
    """Shape rule for nn.ConvTranspose1d.

    Input:  (B, C_in, L)
    Output: (B, C_out, L_out)
    """
    if input_shape.ndim != 3:
        return None
    b = input_shape.dims[0]
    l_in = input_shape.dims[2]
    if l_in.is_symbolic:
        l_out = ShapeDim("_convt1d_l")
    else:
        l_out_val = (l_in.value - 1) * stride - 2 * padding + kernel_size + output_padding
        l_out = ShapeDim(l_out_val)
    return TensorShape((b, ShapeDim(out_channels), l_out))


def transfer_conv3d(
    input_shape: TensorShape,
    out_channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
) -> Optional[TensorShape]:
    """Shape rule for nn.Conv3d.

    Input:  (B, C_in, D, H, W)
    Output: (B, C_out, D_out, H_out, W_out)
    """
    if input_shape.ndim != 5:
        return None
    b = input_shape.dims[0]
    spatial = []
    for i in range(3):
        d = input_shape.dims[2 + i]
        k, s, p = kernel_size[i], stride[i], padding[i]
        if d.is_symbolic:
            spatial.append(ShapeDim(f"_conv3d_{i}"))
        else:
            out = (d.value + 2 * p - k) // s + 1
            if out <= 0:
                return None
            spatial.append(ShapeDim(out))
    return TensorShape((b, ShapeDim(out_channels)) + tuple(spatial))


# ═══════════════════════════════════════════════════════════════════════════
# Linear algebra operators (torch.linalg)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_linalg_square(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for square-matrix-preserving linalg ops (inv, pinv, cholesky, matrix_power).

    Input:  (..., M, M)
    Output: (..., M, M)
    """
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims)


def transfer_linalg_det(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.det / matrix_rank.

    Input:  (..., M, N)
    Output: (...)  — removes last two dims.
    """
    if input_shape.ndim < 2:
        return None
    if input_shape.ndim == 2:
        return TensorShape((ShapeDim(1),))
    return TensorShape(input_shape.dims[:-2])


def transfer_linalg_slogdet(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.slogdet.

    Input:  (..., M, M)
    Output: sign (...), logabsdet (...) — returns batch shape.
    """
    if input_shape.ndim < 2:
        return None
    if input_shape.ndim == 2:
        return TensorShape((ShapeDim(1),))
    return TensorShape(input_shape.dims[:-2])


def transfer_linalg_svd(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.svd.

    Input:  (..., M, N)
    Output: U (..., M, K), S (..., K), Vh (..., K, N) where K = min(M, N).
    Returns S shape (most common use case).
    """
    if input_shape.ndim < 2:
        return None
    m = input_shape.dims[-2]
    n = input_shape.dims[-1]
    batch = input_shape.dims[:-2]
    if m.is_symbolic or n.is_symbolic:
        k = ShapeDim("_svd_k")
    else:
        k = ShapeDim(min(m.value, n.value))
    return TensorShape(batch + (k,))


def transfer_linalg_qr(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.qr.

    Input:  (..., M, N)
    Output: Q (..., M, K), R (..., K, N) where K = min(M, N).
    Returns Q shape.
    """
    if input_shape.ndim < 2:
        return None
    m = input_shape.dims[-2]
    n = input_shape.dims[-1]
    batch = input_shape.dims[:-2]
    if m.is_symbolic or n.is_symbolic:
        k = ShapeDim("_qr_k")
    else:
        k = ShapeDim(min(m.value, n.value))
    return TensorShape(batch + (m, k))


def transfer_linalg_solve(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.solve.

    Input A: (..., M, M), B: (..., M, K) or (..., M)
    Output: same shape as B.  Returns input_shape (representing B).
    """
    if input_shape.ndim < 1:
        return None
    return TensorShape(input_shape.dims)


def transfer_linalg_eig(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.eig / eigvalsh.

    Input:  (..., M, M)
    Output eigenvalues: (..., M)
    """
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims[:-1])


def transfer_linalg_lstsq(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.lstsq.

    Input A: (..., M, N), B: (..., M, K)
    Output solution: (..., N, K).  Returns input shape (B) as approximation.
    """
    if input_shape.ndim < 1:
        return None
    return TensorShape(input_shape.dims)


def transfer_linalg_norm(
    input_shape: TensorShape,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[TensorShape]:
    """Shape rule for torch.linalg.norm.

    If dim is None, reduces to scalar.  Otherwise reduces along dim.
    """
    if dim is None:
        return TensorShape((ShapeDim(1),))
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    if keepdim:
        result_dims[dim] = ShapeDim(1)
    else:
        result_dims.pop(dim)
    if not result_dims:
        return TensorShape((ShapeDim(1),))
    return TensorShape(tuple(result_dims))


# ═══════════════════════════════════════════════════════════════════════════
# FFT operators (torch.fft)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_fft_c2c(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for complex-to-complex FFT (fft, ifft, fft2, ifft2, fftn, ifftn).

    Output shape == input shape.
    """
    if input_shape.ndim < 1:
        return None
    return TensorShape(input_shape.dims)


def transfer_rfft(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.fft.rfft.

    Input:  (..., N)
    Output: (..., N//2 + 1)
    """
    if input_shape.ndim < 1:
        return None
    last = input_shape.dims[-1]
    prefix = input_shape.dims[:-1]
    if last.is_symbolic:
        return TensorShape(prefix + (ShapeDim("_rfft_n"),))
    return TensorShape(prefix + (ShapeDim(last.value // 2 + 1),))


def transfer_irfft(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.fft.irfft.

    Input:  (..., N//2+1)
    Output: (..., 2*(N//2+1)-2)
    """
    if input_shape.ndim < 1:
        return None
    last = input_shape.dims[-1]
    prefix = input_shape.dims[:-1]
    if last.is_symbolic:
        return TensorShape(prefix + (ShapeDim("_irfft_n"),))
    return TensorShape(prefix + (ShapeDim(2 * (last.value - 1)),))


def transfer_rfft2(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.fft.rfft2.

    Input:  (..., H, W)
    Output: (..., H, W//2+1)
    """
    if input_shape.ndim < 2:
        return None
    last = input_shape.dims[-1]
    prefix = input_shape.dims[:-1]
    if last.is_symbolic:
        return TensorShape(prefix + (ShapeDim("_rfft2_w"),))
    return TensorShape(prefix + (ShapeDim(last.value // 2 + 1),))


def transfer_irfft2(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.fft.irfft2.

    Input:  (..., H, W//2+1)
    Output: (..., H, W)
    """
    if input_shape.ndim < 2:
        return None
    last = input_shape.dims[-1]
    prefix = input_shape.dims[:-1]
    if last.is_symbolic:
        return TensorShape(prefix + (ShapeDim("_irfft2_w"),))
    return TensorShape(prefix + (ShapeDim(2 * (last.value - 1)),))


# ═══════════════════════════════════════════════════════════════════════════
# Tensor manipulation ops
# ═══════════════════════════════════════════════════════════════════════════

def transfer_narrow(
    input_shape: TensorShape,
    dim: int,
    length: int,
) -> Optional[TensorShape]:
    """Shape rule for Tensor.narrow — narrows dimension *dim* to *length*."""
    if input_shape.ndim == 0:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    result_dims[dim] = ShapeDim(length)
    return TensorShape(tuple(result_dims))


def transfer_select(
    input_shape: TensorShape,
    dim: int,
) -> Optional[TensorShape]:
    """Shape rule for Tensor.select — removes the selected dimension."""
    if input_shape.ndim <= 1:
        return None
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    result_dims.pop(dim)
    return TensorShape(tuple(result_dims))


def transfer_movedim(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for movedim/moveaxis/swapaxes — preserves set of dims."""
    return TensorShape(input_shape.dims) if input_shape.ndim >= 1 else None


def transfer_tile(
    input_shape: TensorShape,
    reps: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for torch.tile — each dim multiplied by repeat factor."""
    if input_shape.ndim == 0:
        return None
    ndim = max(input_shape.ndim, len(reps))
    dims = list(input_shape.dims)
    while len(dims) < ndim:
        dims.insert(0, ShapeDim(1))
    padded_reps = list(reps)
    while len(padded_reps) < ndim:
        padded_reps.insert(0, 1)
    result = []
    for d, r in zip(dims, padded_reps):
        if d.is_symbolic:
            result.append(ShapeDim(f"{d.value}_x{r}"))
        else:
            result.append(ShapeDim(d.value * r))
    return TensorShape(tuple(result))


def transfer_expand(
    input_shape: TensorShape,
    sizes: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for Tensor.expand / expand_as.

    Each -1 in sizes means keep the original size.
    """
    result = []
    for s in sizes:
        if s == -1:
            result.append(ShapeDim("_expand"))
        else:
            result.append(ShapeDim(s))
    return TensorShape(tuple(result))


def transfer_view_as_real(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.view_as_real: adds trailing dim of 2."""
    if input_shape.ndim < 1:
        return None
    return TensorShape(input_shape.dims + (ShapeDim(2),))


def transfer_view_as_complex(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.view_as_complex: removes trailing dim of 2."""
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims[:-1])


def transfer_reduction(
    input_shape: TensorShape,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> Optional[TensorShape]:
    """Generic reduction transfer for argmax, argmin, all, any, etc."""
    if dim is None:
        return TensorShape((ShapeDim(1),))
    if dim < 0:
        dim = input_shape.ndim + dim
    if dim >= input_shape.ndim:
        return None
    result_dims = list(input_shape.dims)
    if keepdim:
        result_dims[dim] = ShapeDim(1)
    else:
        result_dims.pop(dim)
    if not result_dims:
        return TensorShape((ShapeDim(1),))
    return TensorShape(tuple(result_dims))


# ═══════════════════════════════════════════════════════════════════════════
# Creation / like ops
# ═══════════════════════════════════════════════════════════════════════════

def transfer_like_create(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for *_like ops (zeros_like, ones_like, etc.) — same shape."""
    return TensorShape(input_shape.dims)


def transfer_new_tensor(
    input_shape: TensorShape,
    size: Tuple[int, ...],
) -> Optional[TensorShape]:
    """Shape rule for Tensor.new_zeros, new_ones, new_empty, new_full."""
    return TensorShape(tuple(ShapeDim(s) for s in size))


def transfer_linspace_create(
    steps: int,
) -> Optional[TensorShape]:
    """Shape rule for torch.linspace / logspace."""
    return TensorShape((ShapeDim(steps),))


# ═══════════════════════════════════════════════════════════════════════════
# Loss functions (scalar output)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_scalar_loss(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for loss functions that reduce to scalar."""
    return TensorShape((ShapeDim(1),))


# ═══════════════════════════════════════════════════════════════════════════
# Spectral ops (STFT / ISTFT)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_stft(
    input_shape: TensorShape,
    n_fft: int,
    hop_length: Optional[int] = None,
) -> Optional[TensorShape]:
    """Shape rule for torch.stft.

    Input:  (B, L) or (L,)
    Output: (B, n_fft//2+1, n_frames) or (n_fft//2+1, n_frames)
    """
    if input_shape.ndim < 1 or input_shape.ndim > 2:
        return None
    freq_bins = ShapeDim(n_fft // 2 + 1)
    l_dim = input_shape.dims[-1]
    hl = hop_length if hop_length else n_fft // 4
    if l_dim.is_symbolic:
        n_frames = ShapeDim("_stft_frames")
    else:
        n_frames = ShapeDim(l_dim.value // hl + 1)
    if input_shape.ndim == 2:
        return TensorShape((input_shape.dims[0], freq_bins, n_frames))
    return TensorShape((freq_bins, n_frames))


def transfer_istft(
    input_shape: TensorShape,
    n_fft: int,
    hop_length: Optional[int] = None,
) -> Optional[TensorShape]:
    """Shape rule for torch.istft.

    Input:  (B, n_fft//2+1, n_frames) or (n_fft//2+1, n_frames)
    Output: (B, signal_length) or (signal_length,)
    """
    if input_shape.ndim < 2 or input_shape.ndim > 3:
        return None
    n_frames = input_shape.dims[-1]
    hl = hop_length if hop_length else n_fft // 4
    if n_frames.is_symbolic:
        sig_len = ShapeDim("_istft_len")
    else:
        sig_len = ShapeDim((n_frames.value - 1) * hl)
    if input_shape.ndim == 3:
        return TensorShape((input_shape.dims[0], sig_len))
    return TensorShape((sig_len,))


# ═══════════════════════════════════════════════════════════════════════════
# nn.Module layer ops
# ═══════════════════════════════════════════════════════════════════════════

def transfer_rnn_cell(
    input_shape: TensorShape,
    hidden_size: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.RNNCell / nn.GRUCell.

    Input:  (B, input_size)
    Output: (B, hidden_size)
    """
    if input_shape.ndim != 2:
        return None
    return TensorShape((input_shape.dims[0], ShapeDim(hidden_size)))


def transfer_lstm_cell(
    input_shape: TensorShape,
    hidden_size: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.LSTMCell.

    Input:  (B, input_size)
    Output: (h, c) each (B, hidden_size).  Returns h shape.
    """
    if input_shape.ndim != 2:
        return None
    return TensorShape((input_shape.dims[0], ShapeDim(hidden_size)))


def transfer_transformer_layer(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for TransformerEncoderLayer / TransformerDecoderLayer.

    Output shape == input shape.
    """
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims)


def transfer_channel_shuffle(
    input_shape: TensorShape,
    groups: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.ChannelShuffle — shape preserving."""
    if input_shape.ndim < 3:
        return None
    return TensorShape(input_shape.dims)


def transfer_adaptive_log_softmax(
    input_shape: TensorShape,
    n_classes: int,
) -> Optional[TensorShape]:
    """Shape rule for nn.AdaptiveLogSoftmaxWithLoss.

    Input:  (B, in_features)
    Output: (B, n_classes) log-probabilities.
    """
    if input_shape.ndim != 2:
        return None
    return TensorShape((input_shape.dims[0], ShapeDim(n_classes)))


# ═══════════════════════════════════════════════════════════════════════════
# Broadcast binary ops / BLAS
# ═══════════════════════════════════════════════════════════════════════════

def transfer_addmm(
    input_shape: TensorShape,
    mat1_rows: int,
    mat2_cols: int,
) -> Optional[TensorShape]:
    """Shape rule for torch.addmm: alpha * (mat1 @ mat2) + beta * input."""
    return TensorShape((ShapeDim(mat1_rows), ShapeDim(mat2_cols)))


def transfer_addmv(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.addmv: mat (N, M) @ vec (M,) + input (N,) -> (N,)."""
    return TensorShape(input_shape.dims) if input_shape.ndim >= 1 else None


def transfer_addr(
    vec1_shape: TensorShape,
    vec2_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.addr: outer product vec1 (N,) x vec2 (M,) -> (N, M)."""
    if vec1_shape.ndim != 1 or vec2_shape.ndim != 1:
        return None
    return TensorShape((vec1_shape.dims[0], vec2_shape.dims[0]))


def transfer_baddbmm(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.baddbmm: batch matmul + bias, shape = input shape."""
    if input_shape.ndim < 2:
        return None
    return TensorShape(input_shape.dims)


def transfer_addbmm(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.addbmm: reduced batch matmul, returns (N, P)."""
    if input_shape.ndim != 2:
        return None
    return TensorShape(input_shape.dims)


# ═══════════════════════════════════════════════════════════════════════════
# Misc: einsum, tensordot, outer, inner, kron, etc.
# ═══════════════════════════════════════════════════════════════════════════

def transfer_outer(
    vec1_shape: TensorShape,
    vec2_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.outer: (N,) x (M,) -> (N, M)."""
    if vec1_shape.ndim != 1 or vec2_shape.ndim != 1:
        return None
    return TensorShape((vec1_shape.dims[0], vec2_shape.dims[0]))


def transfer_inner(
    input_shape: TensorShape,
    other_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.inner: contracts last dim."""
    if input_shape.ndim < 1 or other_shape.ndim < 1:
        return None
    out_dims = input_shape.dims[:-1] + other_shape.dims[:-1]
    return TensorShape(out_dims) if out_dims else TensorShape((ShapeDim(1),))


def transfer_vdot(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.vdot — dot product of flattened tensors → scalar."""
    return TensorShape((ShapeDim(1),))


def transfer_kron(
    a_shape: TensorShape,
    b_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.kron — Kronecker product."""
    if a_shape.ndim == 0 or b_shape.ndim == 0:
        return None
    ndim = max(a_shape.ndim, b_shape.ndim)
    a_dims = list(a_shape.dims)
    b_dims = list(b_shape.dims)
    while len(a_dims) < ndim:
        a_dims.insert(0, ShapeDim(1))
    while len(b_dims) < ndim:
        b_dims.insert(0, ShapeDim(1))
    result = []
    for ad, bd in zip(a_dims, b_dims):
        if ad.is_symbolic or bd.is_symbolic:
            result.append(ShapeDim("_kron"))
        else:
            result.append(ShapeDim(ad.value * bd.value))
    return TensorShape(tuple(result))


def transfer_block_diag(
    *shapes: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.block_diag — diagonal stacking of 2D matrices."""
    total_rows = 0
    total_cols = 0
    any_symbolic = False
    for s in shapes:
        if s.ndim != 2:
            return None
        r, c = s.dims
        if r.is_symbolic or c.is_symbolic:
            any_symbolic = True
        else:
            total_rows += r.value
            total_cols += c.value
    if any_symbolic:
        return TensorShape((ShapeDim("_blkdiag_r"), ShapeDim("_blkdiag_c")))
    return TensorShape((ShapeDim(total_rows), ShapeDim(total_cols)))


def transfer_tensordot(
    a_shape: TensorShape,
    b_shape: TensorShape,
    dims: int = 2,
) -> Optional[TensorShape]:
    """Shape rule for torch.tensordot — contracts last *dims* of a with first *dims* of b."""
    if a_shape.ndim < dims or b_shape.ndim < dims:
        return None
    if dims > 0:
        out_dims = a_shape.dims[:-dims] + b_shape.dims[dims:]
    else:
        out_dims = a_shape.dims + b_shape.dims
    return TensorShape(out_dims) if out_dims else TensorShape((ShapeDim(1),))


def transfer_einsum(
    equation: str,
    *operands: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.einsum — simplified subscript parser."""
    if '->' not in equation:
        return None
    _, rhs = equation.split('->')
    rhs = rhs.strip()
    if not rhs:
        return TensorShape((ShapeDim(1),))
    return TensorShape(tuple(ShapeDim(f"_ein_{c}") for c in rhs if c.isalpha()))


def transfer_cartesian_prod(
    *shapes: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for torch.cartesian_prod — N 1-D tensors → (product, N)."""
    total = 1
    any_symbolic = False
    for s in shapes:
        if s.ndim != 1:
            return None
        if s.dims[0].is_symbolic:
            any_symbolic = True
        else:
            total *= s.dims[0].value
    n = len(shapes)
    if any_symbolic:
        return TensorShape((ShapeDim("_cartprod"), ShapeDim(n)))
    return TensorShape((ShapeDim(total), ShapeDim(n)))


def transfer_size_tensor(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for Tensor.size() returning as IntTensor — (ndim,)."""
    return TensorShape((ShapeDim(input_shape.ndim),))


def transfer_numel_tensor(
    input_shape: TensorShape,
) -> Optional[TensorShape]:
    """Shape rule for Tensor.numel() — returns scalar."""
    return TensorShape((ShapeDim(1),))


# ═══════════════════════════════════════════════════════════════════════════
# Registry of all modern ops with their shape-op categories
# ═══════════════════════════════════════════════════════════════════════════

MODERN_TORCH_SHAPE_OPS = {
    # Attention
    "scaled_dot_product_attention": "sdpa",
    "multi_head_attention_forward": "mha",
    # Normalization
    "group_norm": "norm_preserve",
    "instance_norm": "norm_preserve",
    "layer_norm": "norm_preserve",
    "rms_norm": "norm_preserve",
    # Activations (element-wise)
    "gelu": "elementwise",
    "silu": "elementwise",
    "mish": "elementwise",
    "swish": "elementwise",
    "hardswish": "elementwise",
    "hardtanh": "elementwise",
    "hardsigmoid": "elementwise",
    "leaky_relu": "elementwise",
    "elu": "elementwise",
    "celu": "elementwise",
    "selu": "elementwise",
    "prelu": "elementwise",
    "rrelu": "elementwise",
    "softplus": "elementwise",
    "softsign": "elementwise",
    "softshrink": "elementwise",
    "hardshrink": "elementwise",
    "tanhshrink": "elementwise",
    "threshold": "elementwise",
    "logsigmoid": "elementwise",
    "softmax": "elementwise",
    "log_softmax": "elementwise",
    "sigmoid": "elementwise",
    "tanh": "elementwise",
    "relu": "elementwise",
    "relu6": "elementwise",
    "clamp": "elementwise",
    "abs": "elementwise",
    "neg": "elementwise",
    "sign": "elementwise",
    "floor": "elementwise",
    "ceil": "elementwise",
    "round": "elementwise",
    "exp": "elementwise",
    "log": "elementwise",
    "log2": "elementwise",
    "log10": "elementwise",
    "sqrt": "elementwise",
    "rsqrt": "elementwise",
    "reciprocal": "elementwise",
    "sin": "elementwise",
    "cos": "elementwise",
    "tan": "elementwise",
    "asin": "elementwise",
    "acos": "elementwise",
    "atan": "elementwise",
    "sinh": "elementwise",
    "cosh": "elementwise",
    "erf": "elementwise",
    "erfc": "elementwise",
    "erfinv": "elementwise",
    "logical_not": "elementwise",
    "bitwise_not": "elementwise",
    "isnan": "elementwise",
    "isinf": "elementwise",
    "isfinite": "elementwise",
    # GLU family
    "glu": "glu",
    # Sparse / MoE
    "sparse_softmax": "elementwise",
    "topk": "topk",
    # Positional encoding
    "rotary_embedding": "rope",
    # Memory-efficient
    "checkpoint": "checkpoint",
    "gradient_checkpoint": "checkpoint",
    # Pooling
    "adaptive_avg_pool1d": "adaptive_pool",
    "adaptive_avg_pool2d": "adaptive_pool",
    "adaptive_avg_pool3d": "adaptive_pool",
    "adaptive_max_pool1d": "adaptive_pool",
    "adaptive_max_pool2d": "adaptive_pool",
    "adaptive_max_pool3d": "adaptive_pool",
    # Reshape / manipulation
    "pixel_shuffle": "pixel_shuffle",
    "pixel_unshuffle": "pixel_unshuffle",
    "unfold": "unfold",
    "fold": "fold",
    "chunk": "chunk",
    "split": "split",
    "repeat_interleave": "repeat_interleave",
    # Convolutions
    "conv1d": "conv1d",
    "conv_transpose1d": "conv_transpose1d",
    "conv3d": "conv3d",
    # Embedding
    "embedding": "embedding",
    # Dropout / stochastic depth (shape-preserving)
    "dropout": "elementwise",
    "dropout2d": "elementwise",
    "dropout3d": "elementwise",
    "alpha_dropout": "elementwise",
    "feature_alpha_dropout": "elementwise",
    "stochastic_depth": "elementwise",
    # Cross attention
    "cross_attention": "sdpa",
    # Einops
    "rearrange": "einops_rearrange",
    "repeat": "einops_repeat",
    "reduce": "einops_reduce",
    # Tensor creation extras
    "eye": "create",
    "tensor": "create",
    "as_tensor": "create",
    "from_numpy": "create",
    "complex": "create",
    # Comparison (element-wise bool output)
    "eq": "elementwise",
    "ne": "elementwise",
    "gt": "elementwise",
    "ge": "elementwise",
    "lt": "elementwise",
    "le": "elementwise",
    # Scatter / gather
    "gather": "gather",
    "scatter": "scatter",
    "scatter_add": "scatter",
    "index_select": "index_select",
    # Cumulative
    "cumsum": "elementwise",
    "cumprod": "elementwise",
    "cummax": "elementwise",
    "cummin": "elementwise",
    # Sorting
    "sort": "elementwise",
    "argsort": "elementwise",
    # Misc
    "contiguous": "elementwise",
    "detach": "elementwise",
    "clone": "elementwise",
    "to": "elementwise",
    "float": "elementwise",
    "half": "elementwise",
    "bfloat16": "elementwise",
    "int": "elementwise",
    "long": "elementwise",
    "bool": "elementwise",
    "type": "elementwise",
    "requires_grad_": "elementwise",
    "triu": "elementwise",
    "tril": "elementwise",
    "diag": "diag",
    "diagonal": "diagonal",
    "trace": "trace",
    "norm": "reduce",
    "cross": "broadcast",
    "masked_fill": "elementwise",
    "masked_select": "masked_select",
    "nonzero": "nonzero",
    "unique": "unique",
    "bincount": "bincount",
    "meshgrid": "meshgrid",
    "cdist": "cdist",
    "pdist": "pdist",
    "kl_div": "broadcast",
    "cross_entropy": "reduce",
    "binary_cross_entropy": "broadcast",
    "mse_loss": "broadcast",
    "l1_loss": "broadcast",
    "smooth_l1_loss": "broadcast",
    "nll_loss": "reduce",
    "cosine_similarity": "cosine_sim",
    "pairwise_distance": "pairwise_dist",
    "triplet_margin_loss": "reduce",
    "multi_head_attention_forward": "mha",
    "batch_norm": "elementwise",
    "pad": "pad",
    "interpolate": "interpolate",
    "grid_sample": "grid_sample",
    "affine_grid": "affine_grid",
    # ── torch.linalg ops ──
    "linalg_svd": "linalg_svd",
    "linalg_qr": "linalg_qr",
    "linalg_cholesky": "linalg_square",
    "linalg_solve": "linalg_solve",
    "linalg_inv": "linalg_square",
    "linalg_eig": "linalg_eig",
    "linalg_eigvalsh": "linalg_eig",
    "linalg_det": "linalg_det",
    "linalg_slogdet": "linalg_slogdet",
    "linalg_matrix_rank": "linalg_det",
    "linalg_pinv": "linalg_square",
    "linalg_lstsq": "linalg_lstsq",
    "linalg_norm": "linalg_norm",
    "linalg_cross": "elementwise",
    "linalg_matrix_power": "linalg_square",
    # ── torch.fft ops ──
    "fft_fft": "fft_c2c",
    "fft_ifft": "fft_c2c",
    "fft_rfft": "rfft",
    "fft_irfft": "irfft",
    "fft_fft2": "fft_c2c",
    "fft_ifft2": "fft_c2c",
    "fft_rfft2": "rfft2",
    "fft_irfft2": "irfft2",
    "fft_fftn": "fft_c2c",
    "fft_ifftn": "fft_c2c",
    # ── Advanced indexing ──
    "index_put": "elementwise",
    "index_copy": "elementwise",
    "scatter_reduce": "elementwise",
    "index_add": "elementwise",
    "index_fill": "elementwise",
    # ── Tensor manipulation ──
    "narrow": "narrow",
    "select": "select",
    "unbind": "select",
    "movedim": "movedim",
    "moveaxis": "movedim",
    "swapaxes": "movedim",
    "roll": "elementwise",
    "rot90": "elementwise",
    "flip": "elementwise",
    "fliplr": "elementwise",
    "flipud": "elementwise",
    "tile": "tile",
    "tensor_repeat": "tile",
    "expand": "expand",
    "expand_as": "expand",
    "where_3arg": "broadcast",
    "masked_scatter": "elementwise",
    "take": "reduce",
    "take_along_dim": "elementwise",
    # ── Reduction variants ──
    "argmax": "reduce",
    "argmin": "reduce",
    "all": "reduce",
    "any": "reduce",
    "count_nonzero": "reduce",
    "nanmean": "reduce",
    "nansum": "reduce",
    "logsumexp": "reduce",
    "aminmax": "reduction",
    "amax": "reduce",
    "amin": "reduce",
    "median": "reduce",
    "mode": "reduce",
    "quantile": "reduce",
    "std": "reduce",
    "var": "reduce",
    "var_mean": "reduction",
    "std_mean": "reduction",
    # ── Type / creation ops ──
    "full_like": "like_create",
    "empty_like": "like_create",
    "rand_like": "like_create",
    "new_zeros": "new_tensor",
    "new_ones": "new_tensor",
    "new_empty": "new_tensor",
    "new_full": "new_tensor",
    "scalar_tensor": "create",
    "tensor_from_sequence": "create",
    "logspace": "linspace",
    # ── Loss functions ──
    "focal_loss": "scalar_loss",
    "dice_loss": "scalar_loss",
    "ctc_loss": "scalar_loss",
    "margin_ranking_loss": "scalar_loss",
    "hinge_embedding_loss": "scalar_loss",
    "cosine_embedding_loss": "scalar_loss",
    "multi_margin_loss": "scalar_loss",
    "multilabel_margin_loss": "scalar_loss",
    "multilabel_soft_margin_loss": "scalar_loss",
    "poisson_nll_loss": "scalar_loss",
    # ── Spectral ops ──
    "stft": "stft",
    "istft": "istft",
    # ── Tensor properties ──
    "size_as_tensor": "size_tensor",
    "numel_as_tensor": "numel_tensor",
    # ── nn.Module layers ──
    "transformer_encoder_layer": "transformer_layer",
    "transformer_decoder_layer": "transformer_layer",
    "rnn_cell": "rnn_cell",
    "lstm_cell": "lstm_cell",
    "gru_cell": "rnn_cell",
    "channel_shuffle": "channel_shuffle",
    "softmin": "elementwise",
    "softmax2d": "elementwise",
    "adaptive_log_softmax_with_loss": "adaptive_log_softmax",
    # ── Comparison / broadcast binary ops ──
    "atan2": "broadcast",
    "fmod": "broadcast",
    "remainder": "broadcast",
    "pow": "broadcast",
    "maximum": "broadcast",
    "minimum": "broadcast",
    "addmm": "addmm",
    "addmv": "addmv",
    "addr": "addr",
    "baddbmm": "baddbmm",
    "addbmm": "addbmm",
    "lerp": "broadcast",
    "xlogy": "broadcast",
    "igamma": "broadcast",
    "igammac": "broadcast",
    "nextafter": "broadcast",
    "copysign": "broadcast",
    "heaviside": "broadcast",
    "lcm": "broadcast",
    "gcd": "broadcast",
    "logaddexp": "broadcast",
    "logaddexp2": "broadcast",
    "hypot": "broadcast",
    "bitwise_and": "broadcast",
    "bitwise_or": "broadcast",
    "bitwise_xor": "broadcast",
    "bitwise_left_shift": "broadcast",
    "bitwise_right_shift": "broadcast",
    # ── Complex ops ──
    "view_as_real": "view_as_real",
    "view_as_complex": "view_as_complex",
    # ── Misc ──
    "einsum": "einsum",
    "tensordot": "tensordot",
    "outer": "outer",
    "inner": "inner",
    "vdot": "vdot",
    "kron": "kron",
    "block_diag": "block_diag",
    "cartesian_prod": "cartesian_prod",
}


# Shape transfer dispatch table
MODERN_SHAPE_TRANSFERS = {
    "sdpa": transfer_scaled_dot_product_attention,
    "mha": transfer_multi_head_attention,
    "norm_preserve": transfer_layer_norm,
    "elementwise": transfer_elementwise,
    "glu": transfer_glu,
    "topk": transfer_topk,
    "rope": transfer_rotary_embedding,
    "checkpoint": transfer_checkpoint,
    "adaptive_pool": transfer_adaptive_avg_pool,
    "pixel_shuffle": transfer_pixel_shuffle,
    "pixel_unshuffle": transfer_pixel_unshuffle,
    "unfold": transfer_unfold,
    "fold": transfer_fold,
    "chunk": transfer_chunk,
    "split": transfer_split,
    "repeat_interleave": transfer_repeat_interleave,
    "conv1d": transfer_conv1d,
    "conv_transpose1d": transfer_conv_transpose1d,
    "conv3d": transfer_conv3d,
    "embedding": transfer_embedding,
    "einops_rearrange": transfer_rearrange,
    "einops_repeat": transfer_einops_repeat,
    "einops_reduce": transfer_einops_reduce,
    "moe_routing": transfer_moe_routing,
    "moe_gate": transfer_moe_gate_scores,
    "sinusoidal_pe": transfer_sinusoidal_pos_encoding,
    "alibi": transfer_alibi_bias,
    "cross_attention": transfer_cross_attention,
    "dropout": transfer_dropout,
    "stochastic_depth": transfer_stochastic_depth,
    "rms_norm": transfer_rms_norm,
    # ── linalg ──
    "linalg_square": transfer_linalg_square,
    "linalg_det": transfer_linalg_det,
    "linalg_slogdet": transfer_linalg_slogdet,
    "linalg_svd": transfer_linalg_svd,
    "linalg_qr": transfer_linalg_qr,
    "linalg_solve": transfer_linalg_solve,
    "linalg_eig": transfer_linalg_eig,
    "linalg_lstsq": transfer_linalg_lstsq,
    "linalg_norm": transfer_linalg_norm,
    # ── fft ──
    "fft_c2c": transfer_fft_c2c,
    "rfft": transfer_rfft,
    "irfft": transfer_irfft,
    "rfft2": transfer_rfft2,
    "irfft2": transfer_irfft2,
    # ── manipulation ──
    "narrow": transfer_narrow,
    "select": transfer_select,
    "movedim": transfer_movedim,
    "tile": transfer_tile,
    "expand": transfer_expand,
    # ── reductions ──
    "reduction": transfer_reduction,
    # ── creation ──
    "like_create": transfer_like_create,
    "new_tensor": transfer_new_tensor,
    "linspace": transfer_linspace_create,
    # ── losses ──
    "scalar_loss": transfer_scalar_loss,
    # ── spectral ──
    "stft": transfer_stft,
    "istft": transfer_istft,
    # ── modules ──
    "rnn_cell": transfer_rnn_cell,
    "lstm_cell": transfer_lstm_cell,
    "transformer_layer": transfer_transformer_layer,
    "channel_shuffle": transfer_channel_shuffle,
    "adaptive_log_softmax": transfer_adaptive_log_softmax,
    # ── blas ──
    "addmm": transfer_addmm,
    "addmv": transfer_addmv,
    "addr": transfer_addr,
    "baddbmm": transfer_baddbmm,
    "addbmm": transfer_addbmm,
    # ── complex ──
    "view_as_real": transfer_view_as_real,
    "view_as_complex": transfer_view_as_complex,
    # ── misc ──
    "outer": transfer_outer,
    "inner": transfer_inner,
    "vdot": transfer_vdot,
    "kron": transfer_kron,
    "block_diag": transfer_block_diag,
    "tensordot": transfer_tensordot,
    "einsum": transfer_einsum,
    "cartesian_prod": transfer_cartesian_prod,
    "size_tensor": transfer_size_tensor,
    "numel_tensor": transfer_numel_tensor,
}


def register_modern_ops(registry) -> None:
    """Register modern operator shape signatures in a ModelRegistry.

    Uses FunctionSignature-based registration matching the pattern in
    python_models.py.  Imports are deferred to avoid circular deps.
    """
    try:
        from src.stdlib.python_models import (
            FunctionSignature, ParamSpec, ReturnSpec, Sort,
            _SignatureModel, RefinementConstraint,
        )
    except ImportError:
        return

    ops_to_register = [
        # Attention
        ("torch.nn.functional.scaled_dot_product_attention",
         "Fused scaled dot-product attention (FlashAttention / memory-efficient)."),
        ("torch.nn.MultiheadAttention.forward",
         "Multi-head attention forward pass."),
        # Normalization
        ("torch.nn.functional.group_norm", "Group normalization."),
        ("torch.nn.functional.instance_norm", "Instance normalization."),
        ("torch.nn.functional.layer_norm", "Layer normalization."),
        ("torch.nn.functional.rms_norm", "RMS normalization (LLaMA-style)."),
        # Activations
        ("torch.nn.functional.gelu", "Gaussian Error Linear Unit."),
        ("torch.nn.functional.silu", "Sigmoid Linear Unit (SiLU / Swish)."),
        ("torch.nn.functional.mish", "Mish activation."),
        ("torch.nn.functional.glu", "Gated Linear Unit."),
        # MoE
        ("torch.topk", "Top-k values and indices."),
        # Positional
        ("torch.nn.functional.rotary_embedding",
         "Rotary positional embedding (RoPE)."),
        # Checkpoint
        ("torch.utils.checkpoint.checkpoint",
         "Activation checkpointing for memory-efficient training."),
        # Pooling
        ("torch.nn.functional.adaptive_avg_pool2d",
         "2D adaptive average pooling."),
        # Reshape
        ("torch.nn.PixelShuffle.forward", "Pixel shuffle upsampling."),
        ("torch.chunk", "Split tensor into chunks."),
        ("torch.split", "Split tensor into parts."),
        # Einops
        ("einops.rearrange", "Einops rearrange."),
        ("einops.repeat", "Einops repeat."),
        ("einops.reduce", "Einops reduce."),
        # Conv
        ("torch.nn.functional.conv1d", "1D convolution."),
        ("torch.nn.functional.conv3d", "3D convolution."),
        # Embedding
        ("torch.nn.Embedding.forward", "Embedding lookup."),
        # Dropout
        ("torch.nn.functional.dropout", "Dropout regularization."),
        # Type/device
        ("torch.Tensor.contiguous", "Return contiguous tensor."),
        ("torch.Tensor.detach", "Detach from computation graph."),
        ("torch.Tensor.clone", "Clone tensor."),
        ("torch.Tensor.to", "Cast tensor dtype/device."),
    ]

    for qname, desc in ops_to_register:
        sig = FunctionSignature(
            qualified_name=qname,
            params=(ParamSpec("input", Sort.ANY, description="Input tensor"),),
            returns=ReturnSpec(Sort.ANY),
            description=desc,
        )
        registry.register(_SignatureModel(sig))


UNSUPPORTED_OP_POLICY = "unknown"
"""Policy tag for operators not in any shape registry.

Callers should treat ``UNSUPPORTED_OP_POLICY`` as *shape unknown* (``None``)
rather than silently assuming identity, which could mask dimension mismatches.
"""


def get_unsupported_op_shape(*_args: Any, **_kwargs: Any) -> None:
    """Return ``None`` (UNKNOWN) for any op not in the registry."""
    return None


def get_all_covered_ops() -> Dict[str, str]:
    """Return the merged dict of original + modern ops for counting.

    Also exposes ``get_unsupported_op_shape`` as an attribute so callers can
    retrieve it from the same entry-point::

        ops = get_all_covered_ops()
        unknown = get_all_covered_ops.get_unsupported_op_shape
    """
    from src.tensor_shapes import TORCH_SHAPE_OPS, NUMPY_SHAPE_OPS
    merged: Dict[str, str] = {}
    merged.update(TORCH_SHAPE_OPS)
    merged.update(NUMPY_SHAPE_OPS)
    merged.update(MODERN_TORCH_SHAPE_OPS)
    return merged


get_all_covered_ops.get_unsupported_op_shape = get_unsupported_op_shape  # type: ignore[attr-defined]
