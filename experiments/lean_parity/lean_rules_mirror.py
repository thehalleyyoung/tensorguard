"""
Python mirrors of Lean shape transfer functions for parity testing.

Each function EXACTLY mirrors the Lean implementation's control flow,
conditionals, and case analysis. These are *not* optimized Python code - they
are line-by-line translations to ensure behavioral equivalence.

NOTE: These mirror the Lean definitions in lean/TensorGuard/*.lean files.
"""

from typing import List, Optional, Tuple

#=============================================================================
# From TensorGuard.Soundness
#=============================================================================

def list_prod(xs: List[int]) -> int:
    """Mirror of Lean listProd."""
    result = 1
    for x in xs:
        result *= x
    return result

def apply_op_linear(n: int, i: int, o: int) -> Optional[List[int]]:
    """Mirror of applyOp for linear: rank-1 input, last dim = i."""
    if n == i:
        return [o]
    else:
        return None

def apply_op_view(shape: List[int], out: List[int]) -> Optional[List[int]]:
    """Mirror of applyOp for view: element count must match."""
    shape_prod = 1
    for d in shape:
        shape_prod *= d
    out_prod = list_prod(out)
    if shape_prod == out_prod:
        return out
    else:
        return None

def apply_op_broadcast_add(shape: List[int]) -> List[int]:
    """Mirror of applyOp for broadcast_add: identity."""
    return shape

#=============================================================================
# From TensorGuard.Extended
#=============================================================================

def matmul2(s1: List[int], s2: List[int]) -> Optional[List[int]]:
    """Mirror of matmul2: (m, k) @ (k, n) -> (m, n)."""
    if len(s1) != 2 or len(s2) != 2:
        return None
    m, k1 = s1
    k2, n = s2
    if k1 == k2:
        return [m, n]
    else:
        return None

def bmm(s1: List[int], s2: List[int]) -> Optional[List[int]]:
    """Mirror of bmm: (b, m, k) @ (b, k, n) -> (b, m, n)."""
    if len(s1) != 3 or len(s2) != 3:
        return None
    b1, m, k1 = s1
    b2, k2, n = s2
    if b1 == b2 and k1 == k2:
        return [b1, m, n]
    else:
        return None

def transpose2(s: List[int]) -> Optional[List[int]]:
    """Mirror of transpose2: (m, n) -> (n, m)."""
    if len(s) != 2:
        return None
    m, n = s
    return [n, m]

def perm_list(perm: List[int], dims: List[int]) -> List[int]:
    """Mirror of permList: apply permutation to dims."""
    result = []
    for i in perm:
        if i < len(dims):
            result.append(dims[i])
        else:
            result.append(0)  # fallback
    return result

def conv1d_out(h_in: int, pad: int, dilation: int, k: int, stride: int) -> Optional[int]:
    """Mirror of conv1dOut formula."""
    if stride == 0:
        return None
    num = h_in + 2 * pad
    denom_part = dilation * (k - 1) + 1
    if denom_part > num:
        return None
    return (num - denom_part) // stride + 1

def relu_identity(shape: List[int]) -> List[int]:
    """Mirror of relu: identity on shapes."""
    return shape

def bcast_dim(a: int, b: int) -> Optional[int]:
    """Mirror of bcastDim: dimension-wise broadcast."""
    if a == b:
        return a
    elif a == 1:
        return b
    elif b == 1:
        return a
    else:
        return None

def bcast(s1: List[int], s2: List[int]) -> Optional[List[int]]:
    """Mirror of bcast: recursive shape broadcast."""
    if len(s1) == 0 and len(s2) == 0:
        return []
    elif len(s1) > 0 and len(s2) > 0:
        d1 = bcast_dim(s1[0], s2[0])
        rest = bcast(s1[1:], s2[1:])
        if d1 is not None and rest is not None:
            return [d1] + rest
        else:
            return None
    else:
        return None

#=============================================================================
# From TensorGuard.Parity
#=============================================================================

def conv2d_out_h(h_in: int, pad: int, dilation: int, k: int, stride: int) -> Optional[int]:
    """Mirror of conv2dOutH."""
    return conv1d_out(h_in, pad, dilation, k, stride)

def conv2d_out_w(w_in: int, pad: int, dilation: int, k: int, stride: int) -> Optional[int]:
    """Mirror of conv2dOutW."""
    return conv1d_out(w_in, pad, dilation, k, stride)

def conv3d_out_d(d_in: int, pad: int, dilation: int, k: int, stride: int) -> Optional[int]:
    """Mirror of conv3dOutD."""
    return conv1d_out(d_in, pad, dilation, k, stride)

def maxpool2d_out_h(h_in: int, pad: int, k: int, stride: int) -> Optional[int]:
    """Mirror of maxpool2dOutH."""
    return conv1d_out(h_in, pad, 1, k, stride)

def maxpool2d_out_w(w_in: int, pad: int, k: int, stride: int) -> Optional[int]:
    """Mirror of maxpool2dOutW."""
    return conv1d_out(w_in, pad, 1, k, stride)

def avgpool2d_out_h(h_in: int, pad: int, k: int, stride: int) -> Optional[int]:
    """Mirror of avgpool2dOutH (same as maxpool)."""
    return maxpool2d_out_h(h_in, pad, k, stride)

def avgpool2d_out_w(w_in: int, pad: int, k: int, stride: int) -> Optional[int]:
    """Mirror of avgpool2dOutW (same as maxpool)."""
    return maxpool2d_out_w(w_in, pad, k, stride)

def cat_along(shapes: List[List[int]], axis: int) -> Optional[List[int]]:
    """Mirror of catAlong: concatenate along axis."""
    if len(shapes) == 0:
        return None
    s = shapes[0]
    if axis >= len(s):
        return None
    # Check all shapes have same length and matching non-axis dims
    for t in shapes[1:]:
        if len(s) != len(t):
            return None
        for k in range(len(s)):
            if k != axis and s[k] != t[k]:
                return None
    # Sum along axis
    out_axis_dim = sum(sh[axis] for sh in shapes if axis < len(sh))
    # Build output shape
    return [out_axis_dim if i == axis else s[i] for i in range(len(s))]

def stack(shapes: List[List[int]], axis: int) -> Optional[List[int]]:
    """Mirror of stack: insert new axis."""
    if len(shapes) == 0:
        return None
    s = shapes[0]
    # Check all shapes match
    for t in shapes[1:]:
        if s != t:
            return None
    n = len(shapes)
    return s[:axis] + [n] + s[axis:]

def squeeze(shape: List[int]) -> List[int]:
    """Mirror of squeeze: remove all dims of size 1."""
    return [d for d in shape if d != 1]

def unsqueeze(shape: List[int], axis: int) -> List[int]:
    """Mirror of unsqueeze: insert dim of size 1 at axis."""
    return shape[:axis] + [1] + shape[axis:]

def flatten(shape: List[int], start: int, end: int) -> Optional[List[int]]:
    """Mirror of flatten: merge dimensions start (incl) to end (excl)."""
    if end <= start or end > len(shape):
        return None
    middle = shape[start:end]
    flat_dim = 1
    for d in middle:
        flat_dim *= d
    return shape[:start] + [flat_dim] + shape[end:]

def split(shape: List[int], axis: int, chunks: int) -> Optional[List[int]]:
    """Mirror of split: divide axis into equal chunks."""
    if axis >= len(shape):
        return None
    d = shape[axis]
    if chunks == 0 or d % chunks != 0:
        return None
    new_d = d // chunks
    return [new_d if i == axis else shape[i] for i in range(len(shape))]

def chunk(shape: List[int], axis: int, chunk_size: int) -> Optional[List[int]]:
    """Mirror of chunk: model as identity."""
    if axis >= len(shape):
        return None
    if chunk_size == 0:
        return None
    return shape  # Simplified as identity

def layer_norm_shape(shape: List[int], normalized_dims: int) -> Optional[List[int]]:
    """Mirror of layerNormShape: identity if valid."""
    if normalized_dims <= len(shape):
        return shape
    else:
        return None

def linear_shape(shape: List[int], in_features: int, out_features: int) -> Optional[List[int]]:
    """Mirror of linearShape: contract last dim."""
    if len(shape) == 0:
        return None
    if shape[-1] == in_features:
        return shape[:-1] + [out_features]
    else:
        return None

def embedding_shape(input_shape: List[int], embed_dim: int) -> List[int]:
    """Mirror of embeddingShape: append embedding dim."""
    return input_shape + [embed_dim]
