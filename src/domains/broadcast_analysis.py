"""
Explicit Broadcasting Constraint Generation for Real-World Models.

Extends the existing broadcast_theory.py with higher-level constraint
generation for common broadcasting patterns found in production code:

  - Element-wise binary ops with shape mismatch (``a + b`` where shapes differ)
  - Reduction followed by broadcast expansion
  - Attention score masking (``scores + mask`` where mask has fewer dims)
  - Loss computation broadcasting

Generates Z3 constraints that encode NumPy-style broadcasting rules:
  dimensions are compatible if they're equal or one of them is 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.tensor_shapes import ShapeDim, TensorShape, compute_broadcast_shape


@dataclass
class BroadcastResult:
    """Result of a broadcast compatibility check with diagnostics."""

    compatible: bool
    output_shape: Optional[TensorShape] = None
    error_message: Optional[str] = None
    # Which dimension pairs required broadcasting (for warnings)
    broadcast_dims: List[Tuple[int, int, int]] = None  # type: ignore[assignment]
    # Whether shapes were padded with leading 1s
    left_padded: int = 0
    right_padded: int = 0

    def __post_init__(self):
        if self.broadcast_dims is None:
            self.broadcast_dims = []


def check_broadcast_compatible(
    a: TensorShape, b: TensorShape
) -> BroadcastResult:
    """Check if two shapes are broadcast-compatible with detailed diagnostics.

    NumPy/PyTorch rules:
      1. Align shapes from the right
      2. For each pair: dims are compatible if equal, or one is 1
      3. Missing dims (shorter shape) are treated as 1
    """
    ndim = max(a.ndim, b.ndim)
    result_dims: List[ShapeDim] = []
    broadcast_dims: List[Tuple[int, int, int]] = []
    left_padded = max(0, b.ndim - a.ndim)
    right_padded = max(0, a.ndim - b.ndim)

    for i in range(1, ndim + 1):
        d_a = a.dims[-i] if i <= a.ndim else ShapeDim(1)
        d_b = b.dims[-i] if i <= b.ndim else ShapeDim(1)

        # Symbolic dimensions: assume compatible but flag
        if d_a.is_symbolic or d_b.is_symbolic:
            if not d_a.is_symbolic and d_a.value == 1:
                result_dims.append(d_b)
            elif not d_b.is_symbolic and d_b.value == 1:
                result_dims.append(d_a)
            else:
                result_dims.append(d_a)  # Prefer first symbolic
            continue

        va, vb = d_a.value, d_b.value

        if va == vb:
            result_dims.append(d_a)
        elif va == 1:
            result_dims.append(d_b)
            broadcast_dims.append((ndim - i, va, vb))
        elif vb == 1:
            result_dims.append(d_a)
            broadcast_dims.append((ndim - i, va, vb))
        else:
            return BroadcastResult(
                compatible=False,
                error_message=(
                    f"Broadcast failure at dimension {ndim - i}: "
                    f"{va} vs {vb} (neither is 1)"
                ),
            )

    result_dims.reverse()
    return BroadcastResult(
        compatible=True,
        output_shape=TensorShape(tuple(result_dims)),
        broadcast_dims=broadcast_dims,
        left_padded=left_padded,
        right_padded=right_padded,
    )


def encode_broadcast_constraints_z3(
    a_vars: List[Any],
    b_vars: List[Any],
    out_vars: List[Any],
) -> Optional[Any]:
    """Encode full broadcasting constraints as Z3 formulas.

    For each aligned dimension pair (a_i, b_i, o_i):
      - (a_i == b_i) => (o_i == a_i)
      - (a_i == 1) => (o_i == b_i)
      - (b_i == 1) => (o_i == a_i)
      - (a_i != b_i AND a_i != 1 AND b_i != 1) => UNSAT (broadcast failure)

    Also handles shape padding (shorter shape gets leading 1s).
    """
    if not HAS_Z3:
        return None

    ndim = max(len(a_vars), len(b_vars))
    if len(out_vars) != ndim:
        return None

    constraints: List[Any] = []

    for i in range(ndim):
        a_idx = i - (ndim - len(a_vars))
        b_idx = i - (ndim - len(b_vars))

        a_v = a_vars[a_idx] if a_idx >= 0 else z3.IntVal(1)
        b_v = b_vars[b_idx] if b_idx >= 0 else z3.IntVal(1)
        o_v = out_vars[i]

        # Broadcasting rule as Z3
        constraints.append(
            z3.And(
                z3.Or(a_v == b_v, a_v == 1, b_v == 1),
                z3.If(a_v == 1, o_v == b_v,
                       z3.If(b_v == 1, o_v == a_v, o_v == a_v)),
            )
        )

    # All dims positive
    for v in a_vars + b_vars + out_vars:
        constraints.append(v > 0)

    return z3.And(*constraints) if constraints else z3.BoolVal(True)


def check_matmul_broadcast_compatible(
    a: TensorShape, b: TensorShape
) -> BroadcastResult:
    """Check matmul compatibility including batch dimension broadcasting.

    For ``a @ b``:
      - Last two dims follow matmul rules: a[-1] == b[-2]
      - Leading batch dims follow broadcasting rules
    """
    if a.ndim < 1 or b.ndim < 1:
        return BroadcastResult(
            compatible=False,
            error_message="matmul requires at least 1D tensors",
        )

    # Check inner dimension compatibility
    if a.ndim >= 2 and b.ndim >= 2:
        k_a = a.dims[-1]
        k_b = b.dims[-2]

        if not k_a.is_symbolic and not k_b.is_symbolic:
            if k_a.value != k_b.value:
                return BroadcastResult(
                    compatible=False,
                    error_message=(
                        f"matmul inner dimension mismatch: "
                        f"a has {k_a.value}, b has {k_b.value}"
                    ),
                )

    # Check batch dimensions via broadcasting
    if a.ndim > 2 and b.ndim > 2:
        batch_a = TensorShape(a.dims[:-2])
        batch_b = TensorShape(b.dims[:-2])
        batch_result = check_broadcast_compatible(batch_a, batch_b)
        if not batch_result.compatible:
            return BroadcastResult(
                compatible=False,
                error_message=f"matmul batch dimension {batch_result.error_message}",
            )

        out_dims = batch_result.output_shape.dims + (a.dims[-2], b.dims[-1])
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape(out_dims),
            broadcast_dims=batch_result.broadcast_dims,
        )

    # Simple matmul cases
    if a.ndim == 2 and b.ndim == 2:
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape((a.dims[0], b.dims[1])),
        )

    if a.ndim == 2 and b.ndim == 1:
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape((a.dims[0],)),
        )

    if a.ndim == 1 and b.ndim == 2:
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape((b.dims[1],)),
        )

    if a.ndim == 1 and b.ndim == 1:
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape(()),
        )

    # Batched matmul with 2D
    if a.ndim >= 3 and b.ndim == 2:
        out_dims = a.dims[:-2] + (a.dims[-2], b.dims[-1])
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape(out_dims),
        )

    if a.ndim == 2 and b.ndim >= 3:
        out_dims = b.dims[:-2] + (a.dims[-2], b.dims[-1])
        return BroadcastResult(
            compatible=True,
            output_shape=TensorShape(out_dims),
        )

    return BroadcastResult(
        compatible=True,
        output_shape=TensorShape(()),
    )


def generate_attention_broadcast_constraints(
    query_shape: TensorShape,
    key_shape: TensorShape,
    mask_shape: Optional[TensorShape] = None,
) -> List[str]:
    """Generate human-readable broadcast constraints for attention patterns.

    Common pattern: scores = q @ k.T, then scores + mask where mask
    may have fewer dimensions.
    """
    issues: List[str] = []

    if query_shape.ndim < 2 or key_shape.ndim < 2:
        issues.append("Query and key must be at least 2D for attention")
        return issues

    # Check q @ k^T compatibility
    q_last = query_shape.dims[-1]
    k_last = key_shape.dims[-1]  # Will be transposed, so this is k's head dim
    if not q_last.is_symbolic and not k_last.is_symbolic:
        if q_last.value != k_last.value:
            issues.append(
                f"Query head_dim ({q_last.value}) != Key head_dim ({k_last.value})"
            )

    if mask_shape is not None:
        # Scores shape is (..., seq_q, seq_k) after q @ k^T
        seq_q = query_shape.dims[-2]
        seq_k = key_shape.dims[-2]
        scores_shape = TensorShape((seq_q, seq_k))

        if mask_shape.ndim > 0:
            bcast = check_broadcast_compatible(scores_shape, mask_shape)
            if not bcast.compatible:
                issues.append(
                    f"Attention mask shape {mask_shape.pretty()} "
                    f"not broadcast-compatible with scores "
                    f"shape (..., {seq_q}, {seq_k}): {bcast.error_message}"
                )

    return issues
