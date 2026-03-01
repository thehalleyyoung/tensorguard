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


def get_all_covered_ops() -> Dict[str, str]:
    """Return the merged dict of original + modern ops for counting."""
    from src.tensor_shapes import TORCH_SHAPE_OPS, NUMPY_SHAPE_OPS
    merged: Dict[str, str] = {}
    merged.update(TORCH_SHAPE_OPS)
    merged.update(NUMPY_SHAPE_OPS)
    merged.update(MODERN_TORCH_SHAPE_OPS)
    return merged
