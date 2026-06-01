"""
Tensor Shape Verification via Liquid Types.

Statically verifies tensor shape compatibility in PyTorch and NumPy code
by encoding shapes as refinement type predicates and using Z3 to check
compatibility at operation sites (matmul, add, reshape, view, cat, etc.).

Key insight: tensor shapes are naturally expressible as refinement types:
  {v: Tensor | shape(v) == (batch, channels, height, width)}
and shape errors are the #1 runtime error in ML codebases.

This module extends TensorGuard's predicate harvesting to:
  1. Harvest shape predicates from assertions, constructors, and reshape calls
  2. Propagate shapes through operations (matmul, add, cat, view, etc.)
  3. Generate Z3 constraints at every shape-sensitive operation
  4. Report shape mismatches with concrete counterexamples

Unlike TorchScript (which checks shapes at trace time) or Pyright (which
only checks types, not shapes), TensorGuard checks shapes *statically* with
*zero annotations* on *untyped Python code*.
"""

from __future__ import annotations

import ast
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)

logger = logging.getLogger(__name__)

try:
    import z3 as _z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src._experimental.refinement_lattice import (
    Pred, PredOp, RefType, BaseTypeR, BaseTypeKind,
    INT_TYPE, FLOAT_TYPE, ANY_TYPE, TENSOR_TYPE,
    Z3Encoder, RefinementLattice, RefEnvironment,
)


# ═══════════════════════════════════════════════════════════════════════════
# Shape representation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ShapeDim:
    """A single dimension in a tensor shape — concrete int or symbolic name."""
    value: Union[int, str]

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.value, str)

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class TensorShape:
    """Statically known (possibly symbolic) tensor shape."""
    dims: Tuple[ShapeDim, ...]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def dim(self, axis: int) -> ShapeDim:
        if axis < 0:
            axis = len(self.dims) + axis
        return self.dims[axis]

    def pretty(self) -> str:
        return "(" + ", ".join(str(d.value) for d in self.dims) + ")"

    @staticmethod
    def from_tuple(t: tuple) -> TensorShape:
        return TensorShape(tuple(ShapeDim(v) for v in t))

    @staticmethod
    def unknown(ndim: int) -> TensorShape:
        return TensorShape(tuple(ShapeDim(f"d{i}") for i in range(ndim)))

    def to_pred(self, var: str) -> Pred:
        """Convert to a SHAPE_EQ predicate."""
        dim_values = tuple(d.value for d in self.dims)
        return Pred.shape_eq(var, dim_values)


# ═══════════════════════════════════════════════════════════════════════════
# Symbolic dimension support
# ═══════════════════════════════════════════════════════════════════════════

class SymbolicDimension:
    """Represents a symbolic tensor dimension (e.g., seq_len, batch_size).

    Used for tracking dynamic shapes through transformer attention.
    Supports arithmetic: seq_len * num_heads, batch_size // world_size, etc.
    """

    def __init__(self, name: str, constraints: Optional[List] = None):
        self.name = name
        self.constraints: List = constraints or []

    def __mul__(self, other: Union[int, "SymbolicDimension"]) -> "SymbolicDimension":
        if isinstance(other, int):
            return SymbolicDimension(f"({self.name}*{other})", self.constraints)
        return SymbolicDimension(f"({self.name}*{other.name})",
                                 self.constraints + other.constraints)

    def __rmul__(self, other: int) -> "SymbolicDimension":
        return SymbolicDimension(f"({other}*{self.name})", self.constraints)

    def __floordiv__(self, other: Union[int, "SymbolicDimension"]) -> "SymbolicDimension":
        if isinstance(other, int):
            return SymbolicDimension(f"({self.name}//{other})", self.constraints)
        return SymbolicDimension(f"({self.name}//{other.name})",
                                 self.constraints + other.constraints)

    def __add__(self, other: Union[int, "SymbolicDimension"]) -> "SymbolicDimension":
        if isinstance(other, int):
            return SymbolicDimension(f"({self.name}+{other})", self.constraints)
        return SymbolicDimension(f"({self.name}+{other.name})",
                                 self.constraints + other.constraints)

    def __sub__(self, other: Union[int, "SymbolicDimension"]) -> "SymbolicDimension":
        if isinstance(other, int):
            return SymbolicDimension(f"({self.name}-{other})", self.constraints)
        return SymbolicDimension(f"({self.name}-{other.name})",
                                 self.constraints + other.constraints)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SymbolicDimension):
            return self.name == other.name
        if isinstance(other, int):
            return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"SymDim({self.name})"

    def to_shape_dim(self) -> ShapeDim:
        return ShapeDim(self.name)


# ═══════════════════════════════════════════════════════════════════════════
# Shape error types
# ═══════════════════════════════════════════════════════════════════════════

class ShapeErrorKind(Enum):
    DIM_MISMATCH = auto()       # incompatible dimensions for operation
    NDIM_MISMATCH = auto()      # wrong number of dimensions
    RESHAPE_INVALID = auto()    # reshape to incompatible total size
    BROADCAST_FAIL = auto()     # cannot broadcast shapes
    MATMUL_INCOMPAT = auto()    # inner dimensions don't match for matmul
    CAT_INCOMPAT = auto()       # non-matching dims for concatenation
    CONV_INCOMPAT = auto()      # wrong input shape for conv layer


@dataclass
class ShapeError:
    """A tensor shape error found by static analysis."""
    kind: ShapeErrorKind
    line: int
    col: int
    message: str
    function: str
    variable: str
    actual_shape: Optional[TensorShape] = None
    expected_shape: Optional[TensorShape] = None
    severity: str = "error"
    z3_counterexample: Optional[Dict[str, str]] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.name,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "function": self.function,
            "variable": self.variable,
            "actual_shape": self.actual_shape.pretty() if self.actual_shape else None,
            "expected_shape": self.expected_shape.pretty() if self.expected_shape else None,
            "severity": self.severity,
        }

class DetailedShapeError:
    """Enhanced shape error with full reshape chain context.

    When a shape mismatch is found, shows:
    1. The full chain of reshape operations that led to the error
    2. The expected vs actual shape at the point of mismatch
    3. Which dimension specifically failed
    4. Suggested fix (if deterministic)
    """

    def __init__(self, expected: Optional[TensorShape], actual: Optional[TensorShape],
                 operation: str, chain: Optional[List] = None):
        self.expected = expected
        self.actual = actual
        self.operation = operation
        self.chain = chain or []

    def format_message(self) -> str:
        parts = [f"Shape mismatch at {self.operation}:"]
        if self.expected:
            parts.append(f"  expected: {self.expected.pretty()}")
        if self.actual:
            parts.append(f"  actual:   {self.actual.pretty()}")
        mismatched = self._find_mismatched_dims()
        if mismatched:
            parts.append(f"  mismatched dims: {mismatched}")
        if self.chain:
            parts.append("  reshape chain:")
            for step in self.chain:
                parts.append(f"    {step.get('op', '?')}: "
                             f"{step.get('input_shape', '?')} → "
                             f"{step.get('output_shape', '?')}")
        fix = self.suggest_fix()
        if fix:
            parts.append(f"  suggested fix: {fix}")
        return "\n".join(parts)

    def suggest_fix(self) -> Optional[str]:
        if not self.expected or not self.actual:
            return None
        if self.expected.ndim != self.actual.ndim:
            diff = self.expected.ndim - self.actual.ndim
            if diff > 0:
                return f"Add {diff} dimension(s) with unsqueeze"
            return f"Remove {-diff} dimension(s) with squeeze or index"
        for i in range(min(self.expected.ndim, self.actual.ndim)):
            ed = self.expected.dims[i]
            ad = self.actual.dims[i]
            if (not ed.is_symbolic and not ad.is_symbolic
                    and ed.value != ad.value):
                return (f"Dimension {i} is {ad.value} but should be "
                        f"{ed.value}")
        return None

    def _find_mismatched_dims(self) -> List[int]:
        if not self.expected or not self.actual:
            return []
        result = []
        for i in range(min(self.expected.ndim, self.actual.ndim)):
            ed = self.expected.dims[i]
            ad = self.actual.dims[i]
            if (not ed.is_symbolic and not ad.is_symbolic
                    and ed.value != ad.value):
                result.append(i)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Shape environment: maps variables to their known shapes
# ═══════════════════════════════════════════════════════════════════════════

class ShapeEnv:
    """Maps tensor variables to their statically-known shapes."""

    def __init__(self, bindings: Optional[Dict[str, TensorShape]] = None):
        self._bindings: Dict[str, TensorShape] = dict(bindings or {})

    def get(self, var: str) -> Optional[TensorShape]:
        return self._bindings.get(var)

    def set(self, var: str, shape: TensorShape) -> ShapeEnv:
        new_bindings = dict(self._bindings)
        new_bindings[var] = shape
        return ShapeEnv(new_bindings)

    def copy(self) -> ShapeEnv:
        return ShapeEnv(dict(self._bindings))

    def join(self, other: ShapeEnv) -> ShapeEnv:
        """Join two shape environments (intersection of known shapes)."""
        result: Dict[str, TensorShape] = {}
        for var in self._bindings:
            if var in other._bindings:
                if self._bindings[var] == other._bindings[var]:
                    result[var] = self._bindings[var]
        return ShapeEnv(result)


class ReshapeChainTracker:
    """Track sequences of view/reshape/permute/transpose operations.

    The #1 source of real shape bugs is long chains like:
    x.view(B, S, H, D).permute(0, 2, 1, 3).contiguous().view(B*H, S, D)

    This tracker maintains the full chain so errors can show WHERE in the
    chain the mismatch occurred, not just that the final shape is wrong.
    """

    def __init__(self):
        self._chain: List[Dict[str, Any]] = []

    def record_op(self, op_name: str, input_shape: Optional[TensorShape],
                  output_shape: Optional[TensorShape], args: Any = None) -> None:
        self._chain.append({
            "op": op_name,
            "input_shape": input_shape.pretty() if input_shape else "unknown",
            "output_shape": output_shape.pretty() if output_shape else "unknown",
            "args": str(args) if args else None,
        })

    def get_chain(self) -> List[Dict]:
        return list(self._chain)

    def format_error_chain(self) -> str:
        if not self._chain:
            return "(no reshape chain recorded)"
        lines = ["Reshape chain:"]
        for i, step in enumerate(self._chain):
            lines.append(
                f"  [{i}] {step['op']}: {step['input_shape']} "
                f"→ {step['output_shape']}"
                + (f"  args={step['args']}" if step.get("args") else "")
            )
        return "\n".join(lines)

    def clear(self) -> None:
        self._chain.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Tensor operation shape rules
# ═══════════════════════════════════════════════════════════════════════════

# PyTorch/NumPy operations and their shape semantics
TORCH_SHAPE_OPS = {
    # Creation ops
    "zeros": "create", "ones": "create", "randn": "create",
    "rand": "create", "empty": "create", "full": "create",
    "zeros_like": "like", "ones_like": "like", "randn_like": "like",
    "arange": "arange", "linspace": "linspace",
    # Reshape ops
    "reshape": "reshape", "view": "reshape",
    "flatten": "flatten", "squeeze": "squeeze", "unsqueeze": "unsqueeze",
    "permute": "permute", "transpose": "transpose",
    # Reduction ops
    "sum": "reduce", "mean": "reduce", "max": "reduce", "min": "reduce",
    "prod": "reduce", "norm": "reduce",
    # Combination ops
    "cat": "cat", "stack": "stack", "concatenate": "cat",
    # Matmul
    "matmul": "matmul", "mm": "matmul", "bmm": "bmm",
    "linear": "linear",
    # Element-wise (broadcasting)
    "add": "broadcast", "mul": "broadcast", "sub": "broadcast",
    "div": "broadcast",
}

NUMPY_SHAPE_OPS = {
    "zeros": "create", "ones": "create", "empty": "create",
    "reshape": "reshape", "flatten": "flatten",
    "concatenate": "cat", "stack": "stack",
    "dot": "matmul", "matmul": "matmul",
    "sum": "reduce", "mean": "reduce",
    "transpose": "transpose",
}

# Import modern ops and merge them into the dispatch table
try:
    from src.stdlib.modern_ops import MODERN_TORCH_SHAPE_OPS
    TORCH_SHAPE_OPS.update(MODERN_TORCH_SHAPE_OPS)
except ImportError:
    pass


def compute_matmul_shape(
    a: TensorShape, b: TensorShape
) -> Optional[TensorShape]:
    """Compute the result shape of a @ b (matmul).

    Rules:
      - (m, k) @ (k, n) → (m, n)
      - (b, m, k) @ (b, k, n) → (b, m, n)  (batched)
      - (m, k) @ (k,) → (m,)  (matrix-vector)
    """
    if a.ndim < 1 or b.ndim < 1:
        return None

    if a.ndim == 1 and b.ndim == 1:
        # dot product
        return TensorShape(())

    if a.ndim == 2 and b.ndim == 2:
        # (m, k) @ (k, n) → (m, n)
        return TensorShape((a.dims[0], b.dims[1]))

    if a.ndim == 2 and b.ndim == 1:
        # (m, k) @ (k,) → (m,)
        return TensorShape((a.dims[0],))

    if a.ndim == 1 and b.ndim == 2:
        # (k,) @ (k, n) → (n,)
        return TensorShape((b.dims[1],))

    # Batched matmul
    if a.ndim >= 3 and b.ndim >= 3:
        batch = a.dims[:-2]
        return TensorShape(batch + (a.dims[-2], b.dims[-1]))

    return None


def check_matmul_compatible(a: TensorShape, b: TensorShape) -> Optional[str]:
    """Check if matmul(a, b) is valid. Returns error message or None."""
    if a.ndim < 1 or b.ndim < 1:
        return f"matmul requires at least 1D tensors, got {a.ndim}D and {b.ndim}D"

    # Get the contracting dimensions
    k_a = a.dims[-1] if a.ndim >= 1 else None
    k_b = b.dims[-2] if b.ndim >= 2 else (b.dims[0] if b.ndim == 1 else None)

    if k_a is None or k_b is None:
        return None  # Can't determine

    # Both concrete: direct comparison
    if not k_a.is_symbolic and not k_b.is_symbolic:
        if k_a.value != k_b.value:
            return (f"matmul dimension mismatch: "
                    f"a has inner dim {k_a.value}, b has inner dim {k_b.value}")
    return None


def compute_reshape_shape(
    original: TensorShape, new_dims: Tuple
) -> Optional[TensorShape]:
    """Compute result shape of reshape(original, new_dims).

    Sentinel values:
      0         — copy this dimension from the corresponding input dim (same index).
      ≤ -2      — copy from input dim (-d - 2).  Encoding: -k-2 means dim k.
                  Used for "B, C, H, W = x.shape; x.view(B, ...)" patterns.
      -1        — infer this dimension (standard PyTorch -1 in view/reshape).
    """
    # Resolve sentinel 0 and ≤ -2 values by copying from input
    resolved = list(new_dims)
    copied_symbolic = {}
    for i, d in enumerate(resolved):
        if d == 0 and i < original.ndim:
            inp_d = original.dims[i]
            if not inp_d.is_symbolic:
                resolved[i] = inp_d.value
            else:
                copied_symbolic[i] = inp_d
        elif isinstance(d, int) and d <= -2:
            # Extended sentinel: copy from source dim k = -d - 2
            src_k = -d - 2
            if src_k < original.ndim:
                inp_d = original.dims[src_k]
                if not inp_d.is_symbolic:
                    resolved[i] = inp_d.value
                else:
                    copied_symbolic[i] = inp_d
            else:
                resolved[i] = -1  # out-of-bounds → treat as unknown

    # Count -1's (exclude sentinel-resolved positions)
    neg_ones = sum(1 for i, d in enumerate(resolved) if d == -1 and i not in copied_symbolic)
    if neg_ones > 1:
        # Multiple -1s arise when view() args are runtime shape vars that
        # could not be resolved to copy-from-dim sentinels (e.g. variables
        # from a different tensor's shape).  We cannot infer each dim
        # independently, but we CAN check concrete product compatibility.
        concrete_input_product = 1
        for d in original.dims:
            if not d.is_symbolic:
                concrete_input_product *= d.value
        specified_product = 1
        all_specified = True
        for i, d in enumerate(resolved):
            if i in copied_symbolic:
                if not copied_symbolic[i].is_symbolic:
                    specified_product *= copied_symbolic[i].value
                else:
                    all_specified = False
            elif isinstance(d, int) and d > 0:
                specified_product *= d
            elif d == -1:
                pass  # unknown — skip
            else:
                all_specified = False
        if all_specified and specified_product > 0 and concrete_input_product > 0:
            if concrete_input_product % specified_product != 0:
                return None  # Concrete dims incompatible
        # Return shape with symbolic dims for all unresolved positions
        result_dims = []
        for i, d in enumerate(resolved):
            if i in copied_symbolic:
                result_dims.append(copied_symbolic[i])
            elif isinstance(d, int) and d > 0:
                result_dims.append(ShapeDim(d))
            elif d == -1:
                result_dims.append(ShapeDim("_unknown"))
            elif isinstance(d, str):
                result_dims.append(ShapeDim(d))
            else:
                result_dims.append(ShapeDim("_unknown"))
        return TensorShape(tuple(result_dims))

    # Validate element count compatibility when all dims are concrete
    all_input_concrete = all(not d.is_symbolic for d in original.dims)
    if all_input_concrete and original.ndim > 0:
        input_total = 1
        for d in original.dims:
            input_total *= d.value
        # Compute product of specified (non-negative, non -1) output dims
        specified_product = 1
        all_output_specified = True
        has_neg_one = False
        for i, d in enumerate(resolved):
            if i in copied_symbolic:
                if not copied_symbolic[i].is_symbolic:
                    specified_product *= copied_symbolic[i].value
                else:
                    all_output_specified = False
            elif isinstance(d, int) and d > 0:
                specified_product *= d
            elif d == -1:
                has_neg_one = True
            elif isinstance(d, int) and d == 0:
                pass  # sentinel already resolved
            else:
                all_output_specified = False

        if all_output_specified and specified_product > 0:
            if has_neg_one:
                if input_total % specified_product != 0:
                    return None  # Reshape incompatible: element count mismatch
            elif not has_neg_one:
                if input_total != specified_product:
                    return None  # Reshape incompatible: element count mismatch

    # Handle partially symbolic inputs: when some dims are symbolic and
    # the reshape has a -1, check concrete dim products are compatible.
    # Common pattern: (batch, C, H, W) -> (-1, C*H*W) where batch is symbolic
    if not all_input_concrete and original.ndim > 0 and neg_ones == 1:
        concrete_input_product = 1
        num_symbolic = 0
        for d in original.dims:
            if d.is_symbolic:
                num_symbolic += 1
            else:
                concrete_input_product *= d.value
        specified_product = 1
        all_output_specified = True
        for i, d in enumerate(resolved):
            if i in copied_symbolic:
                if not copied_symbolic[i].is_symbolic:
                    specified_product *= copied_symbolic[i].value
                else:
                    all_output_specified = False
            elif isinstance(d, int) and d > 0:
                specified_product *= d
            elif d == -1:
                pass  # skip the inferred dim
            elif isinstance(d, int) and d == 0:
                pass
            else:
                all_output_specified = False
        # If only symbolic dims feed into -1, the concrete parts must match.
        # For the reshape to work for arbitrary symbolic dim values,
        # concrete_input_product must be divisible by specified_product.
        if (all_output_specified and specified_product > 0
                and concrete_input_product > 0 and num_symbolic >= 1):
            if concrete_input_product % specified_product != 0:
                return None  # Reshape incompatible with symbolic batch

    result_dims = []
    neg_one_idx = -1
    for i, d in enumerate(resolved):
        if i in copied_symbolic:
            result_dims.append(copied_symbolic[i])
        elif isinstance(d, int) and d >= 0:
            result_dims.append(ShapeDim(d))
        elif d == -1:
            neg_one_idx = i
            result_dims.append(ShapeDim("_inferred"))
        elif isinstance(d, str):
            result_dims.append(ShapeDim(d))
        else:
            result_dims.append(ShapeDim("_unknown"))

    # Compute inferred dimension when possible
    if neg_one_idx >= 0 and all_input_concrete and original.ndim > 0:
        input_total = 1
        for d in original.dims:
            input_total *= d.value
        specified_product = 1
        can_infer = True
        for i, rd in enumerate(result_dims):
            if i == neg_one_idx:
                continue
            if rd.is_symbolic:
                can_infer = False
                break
            specified_product *= rd.value
        if can_infer and specified_product > 0 and input_total % specified_product == 0:
            inferred = input_total // specified_product
            result_dims[neg_one_idx] = ShapeDim(inferred)

    # Compute inferred dim when symbolic dims are accounted for via sentinel copies
    if neg_one_idx >= 0 and not all_input_concrete and original.ndim > 0:
        concrete_input = 1
        all_sym_copied = True
        for j, d in enumerate(original.dims):
            if d.is_symbolic:
                if j not in copied_symbolic:
                    all_sym_copied = False
                    break
            else:
                concrete_input *= d.value
        if all_sym_copied and concrete_input > 0:
            concrete_output = 1
            can_infer = True
            for j, rd in enumerate(result_dims):
                if j == neg_one_idx or j in copied_symbolic:
                    continue
                if rd.is_symbolic:
                    can_infer = False
                    break
                concrete_output *= rd.value
            if (can_infer and concrete_output > 0
                    and concrete_input % concrete_output == 0):
                result_dims[neg_one_idx] = ShapeDim(
                    concrete_input // concrete_output)

    return TensorShape(tuple(result_dims))


def compute_broadcast_shape(
    a: TensorShape, b: TensorShape
) -> Optional[TensorShape]:
    """Compute the broadcast result of shapes a and b.

    NumPy/PyTorch broadcasting rules:
      - Align shapes from the right
      - Each dim pair must be (d, d), (d, 1), or (1, d)
    """
    ndim = max(a.ndim, b.ndim)
    result_dims: List[ShapeDim] = []

    for i in range(1, ndim + 1):
        d_a = a.dims[-i] if i <= a.ndim else ShapeDim(1)
        d_b = b.dims[-i] if i <= b.ndim else ShapeDim(1)

        if d_a.is_symbolic or d_b.is_symbolic:
            # Symbolic: can't determine statically, assume OK
            if not d_a.is_symbolic:
                result_dims.append(d_a if d_a.value != 1 else d_b)
            else:
                result_dims.append(d_b if (not d_b.is_symbolic and d_b.value != 1) else d_a)
        elif d_a.value == d_b.value:
            result_dims.append(d_a)
        elif d_a.value == 1:
            result_dims.append(d_b)
        elif d_b.value == 1:
            result_dims.append(d_a)
        else:
            return None  # Broadcast failure

    result_dims.reverse()
    return TensorShape(tuple(result_dims))


# ═══════════════════════════════════════════════════════════════════════════
# Tensor Shape Analyzer: the main analysis engine
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ShapeAnalysisResult:
    """Results from tensor shape analysis."""
    errors: List[ShapeError] = field(default_factory=list)
    shapes: Dict[str, TensorShape] = field(default_factory=dict)
    constraints_generated: int = 0
    constraints_checked: int = 0
    functions_analyzed: int = 0
    analysis_time_ms: float = 0.0

    def summary(self) -> str:
        n_err = len(self.errors)
        return (
            f"Shape Analysis: {self.functions_analyzed} functions, "
            f"{len(self.shapes)} tensor shapes inferred, "
            f"{self.constraints_generated} constraints, "
            f"{n_err} shape errors found, "
            f"{self.analysis_time_ms:.1f}ms"
        )


class TensorShapeAnalyzer(ast.NodeVisitor):
    """Static tensor shape verifier using liquid types and Z3.

    Walks the AST, tracks tensor shapes through assignments and operations,
    and generates Z3 constraints at every shape-sensitive operation site.

    Supports:
      - torch.zeros/ones/randn/rand/empty/full (shape from args)
      - torch.matmul, @, mm, bmm (inner dimension matching)
      - torch.cat/stack (compatible dimensions)
      - reshape/view (total element preservation)
      - Broadcasting for element-wise ops
      - Shape assertions (assert x.shape == ...)
      - nn.Linear, nn.Conv2d (parameter shape matching)
    """

    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self.encoder = Z3Encoder()
        self.shape_env = ShapeEnv()
        self.errors: List[ShapeError] = []
        self.constraints_generated = 0
        self.constraints_checked = 0
        self.func_name = "<module>"
        # Track nn.Module layer definitions
        self._layer_shapes: Dict[str, Dict[str, Any]] = {}
        # Track shape predicates for liquid integration
        self._shape_preds: List[Pred] = []
        # Track class self.attr = value assignments from __init__
        self._class_attrs: Dict[str, Any] = {}
        # Track __init__ parameter defaults for attribute resolution
        self._init_param_defaults: Dict[str, Any] = {}
        # Track the RHS AST expression each variable was assigned from,
        # so we can determine the "origin" of a variable even when its
        # shape cannot be statically determined.
        self._var_origins: Dict[str, ast.expr] = {}

    def analyze_source(self, source: str) -> ShapeAnalysisResult:
        """Analyze Python source for tensor shape errors."""
        t0 = time.monotonic()
        tree = ast.parse(source)
        result = ShapeAnalysisResult()

        # First pass: analyze classes (to extract layer shapes and check forward methods)
        funcs_analyzed = 0
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self.errors = []
                self._analyze_class(node)
                result.errors.extend(self.errors)
                result.shapes.update(self.shape_env._bindings)
                funcs_analyzed += 1

        # Second pass: analyze top-level functions
        funcs_analyzed = 0
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.func_name = node.name
                self.shape_env = ShapeEnv()
                self.errors = []
                self._shape_preds = []
                self._analyze_function(node)
                result.errors.extend(self.errors)
                result.shapes.update(self.shape_env._bindings)
                funcs_analyzed += 1

        # Third pass: analyze module-level code
        self.func_name = "<module>"
        self.shape_env = ShapeEnv()
        self.errors = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._analyze_stmt(node)
        result.errors.extend(self.errors)
        result.shapes.update(self.shape_env._bindings)

        result.functions_analyzed = funcs_analyzed
        result.constraints_generated = self.constraints_generated
        result.constraints_checked = self.constraints_checked
        result.analysis_time_ms = (time.monotonic() - t0) * 1000

        # Deduplicate errors by (line, kind)
        seen = set()
        deduped = []
        for e in result.errors:
            key = (e.line, e.kind, e.message)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        result.errors = deduped

        return result

    def _analyze_function(self, func: ast.FunctionDef):
        """Analyze a single function for shape errors."""
        # Reset per-function variable origin tracking
        self._var_origins = {}
        # Initialize parameter shapes from annotations or conventions
        for arg in func.args.args:
            name = arg.arg
            if arg.annotation:
                shape = self._shape_from_annotation(arg.annotation)
                if shape:
                    self.shape_env = self.shape_env.set(name, shape)

        for stmt in func.body:
            self._analyze_stmt(stmt)

    def _analyze_stmt(self, node: ast.stmt):
        """Analyze a statement for shape operations."""
        if isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value:
                shape = self._infer_shape(node.value)
                if shape:
                    self.shape_env = self.shape_env.set(node.target.id, shape)
        elif isinstance(node, ast.Return):
            if node.value:
                self._check_expr_shapes(node.value)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.For):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.While):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.With):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.Try):
            for s in node.body:
                self._analyze_stmt(s)
        elif isinstance(node, ast.Expr):
            self._check_expr_shapes(node.value)
        elif isinstance(node, ast.Assert):
            self._harvest_shape_assert(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass  # handled separately
        elif isinstance(node, ast.ClassDef):
            self._analyze_class(node)

    def _analyze_assign(self, node: ast.Assign):
        """Track shape through assignment."""
        self._check_expr_shapes(node.value)
        shape = self._infer_shape(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Always record origin even when shape is unknown, so that
                # downstream checks can inspect the RHS expression.
                self._var_origins[target.id] = node.value
                if shape:
                    self.shape_env = self.shape_env.set(target.id, shape)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Call):
                # Handle unpacking: a, b, c = x.unbind(dim) or a, b = torch.chunk(x, 2)
                self._handle_tuple_unpacking(target, node.value)

    def _handle_tuple_unpacking(self, target: ast.Tuple, value: ast.Call):
        """Handle tuple unpacking from operations like unbind, chunk, split."""
        func_name = self._get_call_name(value)
        if not func_name:
            return
        base_name = func_name.split(".")[-1] if "." in func_name else func_name
        
        # Handle unbind: q, k, v = qkv.unbind(0)
        if base_name == "unbind":
            if isinstance(value.func, ast.Attribute):
                obj_shape = self._infer_shape(value.func.value)
                if obj_shape and value.args:
                    dim = self._const_val(value.args[0])
                    if dim is not None:
                        if dim < 0:
                            dim = obj_shape.ndim + dim
                        # Get the size of the dimension being unbound
                        if dim < obj_shape.ndim:
                            dim_size = obj_shape.dims[dim]
                            # Create the shape for each unbound tensor (remove the dim)
                            result_dims = list(obj_shape.dims)
                            result_dims.pop(dim)
                            elem_shape = TensorShape(tuple(result_dims))
                            # Assign the shape to each unpacked variable
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    self.shape_env = self.shape_env.set(elt.id, elem_shape)
        
        # Handle chunk/split: a, b = torch.chunk(x, 2)
        elif base_name in ("chunk", "split"):
            elem_shape = self._infer_shape(value)
            if elem_shape:
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.shape_env = self.shape_env.set(elt.id, elem_shape)

    def _analyze_if(self, node: ast.If):
        """Handle if-else with shape environment joining."""
        old_env = self.shape_env.copy()

        for s in node.body:
            self._analyze_stmt(s)
        true_env = self.shape_env

        self.shape_env = old_env.copy()
        for s in node.orelse:
            self._analyze_stmt(s)
        false_env = self.shape_env

        self.shape_env = true_env.join(false_env)

    def _analyze_class(self, node: ast.ClassDef):
        """Analyze nn.Module subclass for layer definitions."""
        is_module = any(
            (isinstance(b, ast.Name) and b.id in ("Module", "nn.Module"))
            or (isinstance(b, ast.Attribute) and b.attr == "Module")
            for b in node.bases
        )
        if not is_module:
            return

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    self._analyze_module_init(item, node.name)
                elif item.name == "forward":
                    self.func_name = f"{node.name}.forward"
                    self._analyze_function(item)

    def _analyze_module_init(self, func: ast.FunctionDef, class_name: str):
        """Extract layer shapes from __init__ (nn.Linear, nn.Conv2d, etc.)."""
        # Collect __init__ parameter defaults for attribute resolution
        self._init_param_defaults = {}
        defaults = func.args.defaults
        args = func.args.args
        n_defaults = len(defaults)
        n_args = len(args)
        for i, default in enumerate(defaults):
            arg_idx = n_args - n_defaults + i
            if arg_idx >= 0 and arg_idx < n_args:
                val = self._eval_const_expr(default)
                if val is not None:
                    self._init_param_defaults[args[arg_idx].arg] = val

        for stmt in func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        layer_name = target.attr
                        layer_info = self._extract_layer_info(stmt.value)
                        if layer_info:
                            key = f"{class_name}.{layer_name}"
                            self._layer_shapes[key] = layer_info
                        # Track self.attr = value for attribute consistency checks
                        val = self._eval_const_expr(stmt.value)
                        if val is not None:
                            self._class_attrs[layer_name] = val
                        elif isinstance(stmt.value, ast.Name):
                            pname = stmt.value.id
                            if pname in self._init_param_defaults:
                                self._class_attrs[layer_name] = (
                                    self._init_param_defaults[pname])

    def _extract_layer_info(self, node: ast.expr) -> Optional[Dict[str, Any]]:
        """Extract layer parameters from nn.Linear(in, out) etc."""
        if not isinstance(node, ast.Call):
            return None

        func_name = self._get_call_name(node)
        if not func_name:
            return None

        # nn.Linear(in_features, out_features)
        if func_name in ("Linear", "nn.Linear"):
            if len(node.args) >= 2:
                in_f = self._const_or_name(node.args[0])
                out_f = self._const_or_name(node.args[1])
                if in_f is not None and out_f is not None:
                    return {"type": "Linear", "in_features": in_f, "out_features": out_f}

        # nn.Conv2d(in_channels, out_channels, kernel_size) or keyword args
        if func_name in ("Conv2d", "nn.Conv2d"):
            in_c = None
            out_c = None
            ks = None
            stride = 1
            padding = 0
            if len(node.args) >= 2:
                in_c = self._const_or_name(node.args[0])
                out_c = self._const_or_name(node.args[1])
            if len(node.args) >= 3:
                ks = self._const_or_name(node.args[2])
            for kw in node.keywords:
                if kw.arg == "kernel_size":
                    ks = self._const_or_name(kw.value)
                elif kw.arg == "stride":
                    v = self._const_val(kw.value)
                    if v is not None:
                        stride = v
                elif kw.arg == "padding":
                    v = self._const_val(kw.value)
                    if v is not None:
                        padding = v
            if in_c is not None and out_c is not None:
                return {"type": "Conv2d", "in_channels": in_c,
                        "out_channels": out_c, "kernel_size": ks,
                        "stride": stride, "padding": padding}

        # nn.Linear with constant expression args (e.g., 64 * 14 * 14)
        if func_name in ("Linear", "nn.Linear"):
            if len(node.args) >= 2:
                in_f = self._eval_const_expr(node.args[0])
                out_f = self._eval_const_expr(node.args[1])
                if in_f is not None and out_f is not None:
                    return {"type": "Linear", "in_features": in_f,
                            "out_features": out_f}

        # nn.BatchNorm2d(num_features)
        if func_name in ("BatchNorm2d", "nn.BatchNorm2d"):
            if len(node.args) >= 1:
                n = self._const_or_name(node.args[0])
                if n is not None:
                    return {"type": "BatchNorm2d", "num_features": n}

        # nn.AdaptiveAvgPool2d(output_size)
        if func_name in ("AdaptiveAvgPool2d", "nn.AdaptiveAvgPool2d"):
            out_size = None
            if node.args:
                arg = node.args[0]
                if isinstance(arg, (ast.Tuple, ast.List)):
                    vals = [self._const_val(e) for e in arg.elts]
                    if all(v is not None for v in vals):
                        out_size = tuple(vals)
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    out_size = (arg.value, arg.value)
            if out_size:
                return {"type": "AdaptiveAvgPool2d", "output_size": out_size}

        # nn.MultiheadAttention(embed_dim, num_heads, ..., batch_first=...)
        if func_name in ("MultiheadAttention", "nn.MultiheadAttention"):
            embed_dim = self._const_or_name(node.args[0]) if node.args else None
            num_heads = self._const_or_name(node.args[1]) if len(node.args) >= 2 else None
            batch_first = False
            for kw in node.keywords:
                if kw.arg == "batch_first":
                    if isinstance(kw.value, ast.Constant):
                        batch_first = bool(kw.value.value)
            if embed_dim is not None:
                return {"type": "MultiheadAttention", "embed_dim": embed_dim,
                        "num_heads": num_heads, "batch_first": batch_first}

        # nn.LayerNorm(normalized_shape)
        if func_name in ("LayerNorm", "nn.LayerNorm"):
            if node.args:
                n = self._const_or_name(node.args[0])
                if n is not None:
                    return {"type": "LayerNorm", "normalized_shape": n}

        # nn.Dropout(p)
        if func_name in ("Dropout", "nn.Dropout"):
            return {"type": "Dropout"}

        # nn.MaxPool2d(kernel_size, stride)
        if func_name in ("MaxPool2d", "nn.MaxPool2d"):
            ks = self._const_or_name(node.args[0]) if node.args else None
            stride = self._const_or_name(node.args[1]) if len(node.args) >= 2 else ks
            return {"type": "MaxPool2d", "kernel_size": ks, "stride": stride}

        # nn.ReLU, nn.GELU, nn.SiLU — shape-preserving activations
        if func_name in ("ReLU", "nn.ReLU", "GELU", "nn.GELU",
                         "SiLU", "nn.SiLU", "Sigmoid", "nn.Sigmoid",
                         "Tanh", "nn.Tanh"):
            return {"type": "Activation"}

        # nn.Conv1d(in_channels, out_channels, kernel_size, ...)
        if func_name in ("Conv1d", "nn.Conv1d"):
            in_c = self._const_or_name(node.args[0]) if node.args else None
            out_c = self._const_or_name(node.args[1]) if len(node.args) >= 2 else None
            ks = self._const_or_name(node.args[2]) if len(node.args) >= 3 else None
            stride = 1
            padding = 0
            for kw in node.keywords:
                if kw.arg == "kernel_size":
                    ks = self._const_or_name(kw.value)
                elif kw.arg == "stride":
                    v = self._const_val(kw.value)
                    if v is not None:
                        stride = v
                elif kw.arg == "padding":
                    v = self._const_val(kw.value)
                    if v is not None:
                        padding = v
            if in_c is not None and out_c is not None:
                return {"type": "Conv1d", "in_channels": in_c,
                        "out_channels": out_c, "kernel_size": ks,
                        "stride": stride, "padding": padding}

        # nn.Conv3d(in_channels, out_channels, kernel_size, ...)
        if func_name in ("Conv3d", "nn.Conv3d"):
            in_c = self._const_or_name(node.args[0]) if node.args else None
            out_c = self._const_or_name(node.args[1]) if len(node.args) >= 2 else None
            ks = self._const_or_name(node.args[2]) if len(node.args) >= 3 else None
            if in_c is not None and out_c is not None:
                return {"type": "Conv3d", "in_channels": in_c,
                        "out_channels": out_c, "kernel_size": ks}

        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        if func_name in ("ConvTranspose2d", "nn.ConvTranspose2d"):
            in_c = self._const_or_name(node.args[0]) if node.args else None
            out_c = self._const_or_name(node.args[1]) if len(node.args) >= 2 else None
            ks = self._const_val(node.args[2]) if len(node.args) >= 3 else None
            stride = self._const_val(node.args[3]) if len(node.args) >= 4 else 1
            padding = self._const_val(node.args[4]) if len(node.args) >= 5 else 0
            output_padding = self._const_val(node.args[5]) if len(node.args) >= 6 else 0
            dilation = 1
            for kw in node.keywords:
                if kw.arg == "kernel_size":
                    v = self._const_val(kw.value)
                    if v is not None:
                        ks = v
                elif kw.arg == "stride":
                    v = self._const_val(kw.value)
                    if v is not None:
                        stride = v
                elif kw.arg == "padding":
                    v = self._const_val(kw.value)
                    if v is not None:
                        padding = v
                elif kw.arg == "output_padding":
                    v = self._const_val(kw.value)
                    if v is not None:
                        output_padding = v
                elif kw.arg == "dilation":
                    v = self._const_val(kw.value)
                    if v is not None:
                        dilation = v
            if in_c is not None and out_c is not None:
                return {"type": "ConvTranspose2d", "in_channels": in_c,
                        "out_channels": out_c, "kernel_size": ks,
                        "stride": stride if stride is not None else 1,
                        "padding": padding if padding is not None else 0,
                        "output_padding": output_padding if output_padding is not None else 0,
                        "dilation": dilation}

        # nn.AvgPool2d(kernel_size, stride)
        if func_name in ("AvgPool2d", "nn.AvgPool2d"):
            ks = self._const_val(node.args[0]) if node.args else None
            stride = self._const_val(node.args[1]) if len(node.args) >= 2 else ks
            for kw in node.keywords:
                if kw.arg == "kernel_size":
                    v = self._const_val(kw.value)
                    if v is not None:
                        ks = v
                elif kw.arg == "stride":
                    v = self._const_val(kw.value)
                    if v is not None:
                        stride = v
            return {"type": "AvgPool2d", "kernel_size": ks, "stride": stride}

        # nn.AdaptiveMaxPool2d(output_size)
        if func_name in ("AdaptiveMaxPool2d", "nn.AdaptiveMaxPool2d"):
            out_size = None
            if node.args:
                arg = node.args[0]
                if isinstance(arg, (ast.Tuple, ast.List)):
                    vals = [self._const_val(e) for e in arg.elts]
                    if all(v is not None for v in vals):
                        out_size = tuple(vals)
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    out_size = (arg.value, arg.value)
            if out_size:
                return {"type": "AdaptiveMaxPool2d", "output_size": out_size}

        # nn.BatchNorm1d / BatchNorm3d (shape-preserving)
        if func_name in ("BatchNorm1d", "nn.BatchNorm1d"):
            if node.args:
                n = self._const_or_name(node.args[0])
                if n is not None:
                    return {"type": "BatchNorm1d", "num_features": n}
        if func_name in ("BatchNorm3d", "nn.BatchNorm3d"):
            if node.args:
                n = self._const_or_name(node.args[0])
                if n is not None:
                    return {"type": "BatchNorm3d", "num_features": n}

        # nn.GroupNorm(num_groups, num_channels)
        if func_name in ("GroupNorm", "nn.GroupNorm"):
            ng = self._const_or_name(node.args[0]) if node.args else None
            nc = self._const_or_name(node.args[1]) if len(node.args) >= 2 else None
            if ng is not None and nc is not None:
                return {"type": "GroupNorm", "num_groups": ng, "num_channels": nc}

        return None

    # ── Shape inference for expressions ────────────────────────────────

    def _infer_shape(self, node: ast.expr) -> Optional[TensorShape]:
        """Infer the tensor shape of an expression."""
        # Variable lookup
        if isinstance(node, ast.Name):
            return self.shape_env.get(node.id)

        # Function/method call
        if isinstance(node, ast.Call):
            return self._infer_call_shape(node)

        # Binary op: a @ b (matmul), a + b (broadcast)
        if isinstance(node, ast.BinOp):
            return self._infer_binop_shape(node)

        # Attribute: self.layer(x) or x.T (transpose)
        if isinstance(node, ast.Attribute):
            # Handle .T attribute (transpose last two dimensions)
            if node.attr == "T":
                obj_shape = self._infer_shape(node.value)
                if obj_shape and obj_shape.ndim >= 2:
                    dims = list(obj_shape.dims)
                    dims[-1], dims[-2] = dims[-2], dims[-1]
                    return TensorShape(tuple(dims))
                return obj_shape
            # Regular attribute access
            if isinstance(node.value, ast.Name):
                return self.shape_env.get(node.value.id)

        # Subscript: x[0], x[:, 1:3]
        if isinstance(node, ast.Subscript):
            return self._infer_subscript_shape(node)

        return None

    def _infer_call_shape(self, node: ast.Call) -> Optional[TensorShape]:
        """Infer shape from a function call."""
        func_name = self._get_call_name(node)
        if not func_name:
            return None

        # torch.zeros/ones/randn(d1, d2, ...) or torch.zeros((d1, d2, ...))
        base_name = func_name.split(".")[-1] if "." in func_name else func_name
        if base_name in ("zeros", "ones", "randn", "rand", "empty", "full",
                         "np.zeros", "np.ones", "np.empty"):
            return self._shape_from_creation_args(node)

        # torch.zeros_like(x) / ones_like(x)
        if base_name in ("zeros_like", "ones_like", "randn_like", "empty_like"):
            if node.args:
                return self._infer_shape(node.args[0])

        # reshape / view
        if base_name in ("reshape", "view"):
            if isinstance(node.func, ast.Attribute) and node.args:
                obj_shape = self._infer_shape(node.func.value)
                new_dims = self._extract_shape_args_enhanced(node, obj_shape)
                if new_dims is None:
                    new_dims = self._extract_shape_args(node)
                if obj_shape and new_dims:
                    return compute_reshape_shape(obj_shape, new_dims)
            elif node.args:
                obj_shape = self._infer_shape(node.args[0])
                if len(node.args) >= 2:
                    new_dims = self._args_to_dims(node.args[1:])
                    if obj_shape and new_dims:
                        return compute_reshape_shape(obj_shape, new_dims)

        # transpose
        if base_name == "transpose":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape and len(node.args) >= 2:
                    d0 = self._const_val(node.args[0])
                    d1 = self._const_val(node.args[1])
                    if d0 is not None and d1 is not None and obj_shape.ndim > max(d0, d1):
                        dims = list(obj_shape.dims)
                        dims[d0], dims[d1] = dims[d1], dims[d0]
                        return TensorShape(tuple(dims))

        # squeeze
        if base_name == "squeeze":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    if node.args:
                        dim = self._const_val(node.args[0])
                        if dim is not None and dim < obj_shape.ndim:
                            dims = list(obj_shape.dims)
                            if not dims[dim].is_symbolic and dims[dim].value == 1:
                                dims.pop(dim)
                            return TensorShape(tuple(dims))
                    else:
                        dims = [d for d in obj_shape.dims
                                if d.is_symbolic or d.value != 1]
                        return TensorShape(tuple(dims))

        # unsqueeze
        if base_name == "unsqueeze":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape and node.args:
                    dim = self._const_val(node.args[0])
                    if dim is not None:
                        if dim < 0:
                            dim = obj_shape.ndim + 1 + dim
                        dims = list(obj_shape.dims)
                        dims.insert(dim, ShapeDim(1))
                        return TensorShape(tuple(dims))

        # flatten
        if base_name == "flatten":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    start_dim = 0
                    end_dim = -1
                    if node.args:
                        s = self._const_val(node.args[0])
                        if s is not None:
                            start_dim = s
                    if len(node.args) >= 2:
                        e = self._const_val(node.args[1])
                        if e is not None:
                            end_dim = e
                    if end_dim < 0:
                        end_dim = obj_shape.ndim + end_dim
                    # Flatten dims[start_dim:end_dim+1]
                    prefix = obj_shape.dims[:start_dim]
                    suffix = obj_shape.dims[end_dim + 1:]
                    flat_dims = obj_shape.dims[start_dim:end_dim + 1]
                    total = 1
                    all_concrete = True
                    for d in flat_dims:
                        if d.is_symbolic:
                            all_concrete = False
                            break
                        total *= d.value
                    if all_concrete:
                        return TensorShape(prefix + (ShapeDim(total),) + suffix)
                    return TensorShape(prefix + (ShapeDim("_flat"),) + suffix)

        # matmul / mm / bmm / dot
        if base_name in ("matmul", "mm", "dot"):
            if len(node.args) >= 2:
                a = self._infer_shape(node.args[0])
                b = self._infer_shape(node.args[1])
                if a and b:
                    return compute_matmul_shape(a, b)
            elif isinstance(node.func, ast.Attribute) and node.args:
                a = self._infer_shape(node.func.value)
                b = self._infer_shape(node.args[0])
                if a and b:
                    return compute_matmul_shape(a, b)

        # cat / concatenate
        if base_name in ("cat", "concatenate"):
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                if all(s is not None for s in shapes) and shapes:
                    dim = 0
                    if len(node.args) >= 2:
                        d = self._const_val(node.args[1])
                        if d is not None:
                            dim = d
                    # Check for keyword 'dim'
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    return self._compute_cat_shape(shapes, dim)

        # stack
        if base_name == "stack":
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                if all(s is not None for s in shapes) and shapes:
                    dim = 0
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    base = shapes[0]
                    dims = list(base.dims)
                    dims.insert(dim, ShapeDim(len(shapes)))
                    return TensorShape(tuple(dims))

        # sum/mean/max/min/var/std/argmax/argmin/amax/amin with dim
        if base_name in ("sum", "mean", "max", "min", "prod",
                         "var", "std", "argmax", "argmin",
                         "amax", "amin", "logsumexp", "any", "all",
                         "nansum", "nanmean", "count_nonzero"):
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    if node.args:
                        dim = self._const_val(node.args[0])
                        if dim is not None:
                            if dim < 0:
                                dim = obj_shape.ndim + dim
                            keepdim = False
                            for kw in node.keywords:
                                if kw.arg == "keepdim":
                                    if isinstance(kw.value, ast.Constant):
                                        keepdim = bool(kw.value.value)
                            dims = list(obj_shape.dims)
                            if keepdim:
                                dims[dim] = ShapeDim(1)
                            else:
                                dims.pop(dim)
                            return TensorShape(tuple(dims))
                    else:
                        return TensorShape(())  # scalar reduction

        # torch.where(cond, x, y) → broadcast(cond, x, y)
        if base_name == "where":
            if len(node.args) >= 3:
                cond_shape = self._infer_shape(node.args[0])
                x_shape = self._infer_shape(node.args[1])
                y_shape = self._infer_shape(node.args[2])
                result = cond_shape
                if result and x_shape:
                    result = compute_broadcast_shape(result, x_shape)
                elif x_shape:
                    result = x_shape
                if result and y_shape:
                    result = compute_broadcast_shape(result, y_shape)
                elif y_shape:
                    result = y_shape
                return result

        # torch.einsum(equation, *tensors) → shape from equation string
        if base_name == "einsum":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                eq = node.args[0].value
                operand_shapes = [self._infer_shape(a) for a in node.args[1:]]
                return self._infer_einsum_shape(eq, operand_shapes)

        # F.interpolate(input, size=..., scale_factor=...)
        if base_name == "interpolate":
            if node.args:
                inp_shape = self._infer_shape(node.args[0])
                if inp_shape and inp_shape.ndim >= 3:
                    # Determine target spatial dims from size or scale_factor
                    target_size = None
                    for kw in node.keywords:
                        if kw.arg == "size":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                                n_spatial = inp_shape.ndim - 2
                                target_size = [kw.value.value] * n_spatial
                            elif isinstance(kw.value, (ast.Tuple, ast.List)):
                                target_size = [self._const_val(e) for e in kw.value.elts]
                    if len(node.args) >= 2 and target_size is None:
                        arg1 = node.args[1]
                        if isinstance(arg1, ast.Constant) and isinstance(arg1.value, int):
                            n_spatial = inp_shape.ndim - 2
                            target_size = [arg1.value] * n_spatial
                        elif isinstance(arg1, (ast.Tuple, ast.List)):
                            target_size = [self._const_val(e) for e in arg1.elts]
                    if target_size and all(v is not None for v in target_size):
                        # Keep batch + channel dims, replace spatial dims
                        new_dims = list(inp_shape.dims[:2])
                        new_dims.extend(ShapeDim(v) for v in target_size)
                        return TensorShape(tuple(new_dims))
                    # scale_factor: mark spatial dims as symbolic
                    new_dims = list(inp_shape.dims[:2])
                    new_dims.extend(ShapeDim("_interp") for _ in range(inp_shape.ndim - 2))
                    return TensorShape(tuple(new_dims))

         # permute: x.permute(0, 2, 1, 3)
        if base_name == "permute":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    perm = [self._const_val(a) for a in node.args]
                    if all(p is not None for p in perm) and len(perm) == obj_shape.ndim:
                        new_dims = tuple(obj_shape.dims[p] for p in perm)
                        return TensorShape(new_dims)

        # contiguous: shape-preserving
        if base_name == "contiguous":
            if isinstance(node.func, ast.Attribute):
                return self._infer_shape(node.func.value)

        # expand: x.expand(batch_size, -1, -1) — -1 means keep that dim
        if base_name == "expand":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape:
                    new_dims = []
                    for i, arg in enumerate(node.args):
                        v = self._const_val(arg)
                        if v is not None and v == -1 and i < obj_shape.ndim:
                            new_dims.append(obj_shape.dims[i])
                        elif v is not None:
                            new_dims.append(ShapeDim(v))
                        else:
                            cn = self._const_or_name(arg)
                            new_dims.append(ShapeDim(cn if cn else "_expand"))
                    return TensorShape(tuple(new_dims))

        # split: x.split(size, dim) → returns tuple, take first
        if base_name == "split":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape and node.args:
                    return obj_shape

        # size / shape access: x.size() returns shape, x.size(dim) returns int
        if base_name == "size":
            if isinstance(node.func, ast.Attribute):
                return None  # Returns int, not tensor

        # Functional ops: F.softmax, F.relu, F.gelu, F.dropout, F.layer_norm
        if base_name in ("softmax", "log_softmax", "relu", "gelu", "silu",
                         "leaky_relu", "tanh", "sigmoid", "dropout",
                         "layer_norm", "batch_norm", "group_norm",
                         "instance_norm"):
            return self._analyze_functional_call(node, base_name)

        # F.linear(input, weight, bias) — skip self.linear (handled by layer forward)
        if base_name == "linear":
            is_self_call = (isinstance(node.func, ast.Attribute)
                            and isinstance(getattr(node.func, 'value', None), ast.Name)
                            and node.func.value.id == "self")
            if not is_self_call and node.args:
                inp_shape = self._infer_shape(node.args[0])
                if inp_shape and inp_shape.ndim >= 1 and len(node.args) >= 2:
                    w_shape = self._infer_shape(node.args[1])
                    if w_shape and w_shape.ndim == 2:
                        new_dims = list(inp_shape.dims[:-1]) + [w_shape.dims[0]]
                        return TensorShape(tuple(new_dims))
                if inp_shape:
                    return inp_shape

        # F.cross_entropy(input, target) → scalar
        if base_name == "cross_entropy":
            return TensorShape(())

        # F.embedding(input, weight) → (*input_shape, embedding_dim)
        if base_name == "embedding":
            if len(node.args) >= 2:
                inp_shape = self._infer_shape(node.args[0])
                w_shape = self._infer_shape(node.args[1])
                if inp_shape and w_shape and w_shape.ndim == 2:
                    return TensorShape(inp_shape.dims + (w_shape.dims[1],))

        # nn.Module layers: self.layer(x) — Conv2d, Pool, BN, MHA, etc.
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                layer_attr = node.func.attr
                # Look up layer info from any class
                for key, info in self._layer_shapes.items():
                    if key.endswith(f".{layer_attr}"):
                        if info["type"] == "Linear" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim >= 1:
                                out_f = info["out_features"]
                                new_dims = list(x_shape.dims[:-1]) + [ShapeDim(out_f)]
                                return TensorShape(tuple(new_dims))

                        if info["type"] == "Conv2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            out_c = info["out_channels"]
                            if x_shape and x_shape.ndim == 4:
                                new_dims = [x_shape.dims[0], ShapeDim(out_c),
                                            x_shape.dims[2], x_shape.dims[3]]
                                return TensorShape(tuple(new_dims))
                            elif isinstance(out_c, int):
                                return TensorShape((
                                    ShapeDim("_batch"), ShapeDim(out_c),
                                    ShapeDim("_h"), ShapeDim("_w")))

                        if info["type"] == "AdaptiveAvgPool2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            oh, ow = info["output_size"]
                            if x_shape and x_shape.ndim == 4:
                                new_dims = [x_shape.dims[0], x_shape.dims[1],
                                            ShapeDim(oh), ShapeDim(ow)]
                                return TensorShape(tuple(new_dims))
                            elif x_shape and x_shape.ndim >= 2:
                                return TensorShape((
                                    x_shape.dims[0], x_shape.dims[1],
                                    ShapeDim(oh), ShapeDim(ow)))
                            else:
                                return TensorShape((
                                    ShapeDim("_batch"), ShapeDim("_ch"),
                                    ShapeDim(oh), ShapeDim(ow)))

                        if info["type"] == "BatchNorm2d" and node.args:
                            return self._infer_shape(node.args[0])

                        if info["type"] in ("Dropout", "Activation",
                                            "LayerNorm") and node.args:
                            return self._infer_shape(node.args[0])

                        if info["type"] == "MaxPool2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim == 4:
                                ks = info.get("kernel_size", 2)
                                stride = info.get("stride", ks)
                                if isinstance(ks, int) and isinstance(stride, int):
                                    h = x_shape.dims[2]
                                    w = x_shape.dims[3]
                                    if not h.is_symbolic and not w.is_symbolic:
                                        nh = h.value // stride
                                        nw = w.value // stride
                                        return TensorShape((
                                            x_shape.dims[0], x_shape.dims[1],
                                            ShapeDim(nh), ShapeDim(nw)))
                                return TensorShape((
                                    x_shape.dims[0], x_shape.dims[1],
                                    ShapeDim("_pool_h"), ShapeDim("_pool_w")))

                        if info["type"] == "MultiheadAttention" and node.args:
                            return self._infer_shape(node.args[0])

                        if info["type"] == "Conv1d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            out_c = info["out_channels"]
                            if x_shape and x_shape.ndim == 3:
                                ks = info.get("kernel_size", 1)
                                stride = info.get("stride", 1)
                                padding = info.get("padding", 0)
                                l = x_shape.dims[2]
                                if (isinstance(ks, int) and isinstance(stride, int)
                                        and isinstance(padding, int)
                                        and not l.is_symbolic):
                                    new_l = (l.value + 2 * padding - ks) // stride + 1
                                    return TensorShape((x_shape.dims[0], ShapeDim(out_c), ShapeDim(new_l)))
                                return TensorShape((x_shape.dims[0], ShapeDim(out_c), ShapeDim("_l")))
                            elif isinstance(out_c, int):
                                return TensorShape((ShapeDim("_batch"), ShapeDim(out_c), ShapeDim("_l")))

                        if info["type"] == "Conv3d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            out_c = info["out_channels"]
                            if x_shape and x_shape.ndim == 5:
                                return TensorShape((x_shape.dims[0], ShapeDim(out_c),
                                                    x_shape.dims[2], x_shape.dims[3], x_shape.dims[4]))
                            elif isinstance(out_c, int):
                                return TensorShape((ShapeDim("_batch"), ShapeDim(out_c),
                                                    ShapeDim("_d"), ShapeDim("_h"), ShapeDim("_w")))

                        if info["type"] == "ConvTranspose2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            out_c = info["out_channels"]
                            if x_shape and x_shape.ndim == 4:
                                ks = info.get("kernel_size")
                                stride = info.get("stride", 1)
                                padding = info.get("padding", 0)
                                output_padding = info.get("output_padding", 0)
                                dilation = info.get("dilation", 1)
                                h = x_shape.dims[2]
                                w = x_shape.dims[3]
                                if (isinstance(ks, int) and isinstance(stride, int)
                                        and isinstance(padding, int)
                                        and isinstance(output_padding, int)
                                        and isinstance(dilation, int)
                                        and not h.is_symbolic and not w.is_symbolic):
                                    nh = (h.value - 1) * stride - 2 * padding + dilation * (ks - 1) + output_padding + 1
                                    nw = (w.value - 1) * stride - 2 * padding + dilation * (ks - 1) + output_padding + 1
                                    return TensorShape((x_shape.dims[0], ShapeDim(out_c),
                                                        ShapeDim(nh), ShapeDim(nw)))
                                return TensorShape((x_shape.dims[0], ShapeDim(out_c),
                                                    ShapeDim("_ct_h"), ShapeDim("_ct_w")))

                        if info["type"] == "AvgPool2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim == 4:
                                ks = info.get("kernel_size", 2)
                                stride = info.get("stride", ks)
                                if isinstance(ks, int) and isinstance(stride, int) and stride:
                                    h = x_shape.dims[2]
                                    w = x_shape.dims[3]
                                    if not h.is_symbolic and not w.is_symbolic:
                                        nh = (h.value - ks) // stride + 1
                                        nw = (w.value - ks) // stride + 1
                                        return TensorShape((x_shape.dims[0], x_shape.dims[1],
                                                            ShapeDim(nh), ShapeDim(nw)))
                                return TensorShape((x_shape.dims[0], x_shape.dims[1],
                                                    ShapeDim("_avg_h"), ShapeDim("_avg_w")))

                        if info["type"] == "AdaptiveMaxPool2d" and node.args:
                            x_shape = self._infer_shape(node.args[0])
                            oh, ow = info["output_size"]
                            if x_shape and x_shape.ndim == 4:
                                return TensorShape((x_shape.dims[0], x_shape.dims[1],
                                                    ShapeDim(oh), ShapeDim(ow)))
                            return TensorShape((ShapeDim("_batch"), ShapeDim("_ch"),
                                                ShapeDim(oh), ShapeDim(ow)))

                        if info["type"] in ("BatchNorm1d", "BatchNorm3d", "GroupNorm") and node.args:
                            return self._infer_shape(node.args[0])

        # nn.Sequential: self.block(x) — just pass through
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                if node.args:
                    return self._infer_shape(node.args[0])

        # ── Indexing / gather family ──────────────────────────────
        # Helper: detect free-function style (torch.foo(...) or F.foo(...))
        def _is_namespace_call(call_node):
            if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
                return call_node.func.value.id in ("torch", "F", "np", "numpy",
                                                    "nn", "torch.nn",
                                                    "functional")
            return False

        # gather(dim, index): output shape == index shape
        if base_name == "gather":
            if isinstance(node.func, ast.Attribute) and not _is_namespace_call(node):
                obj_shape = self._infer_shape(node.func.value)
                idx_shape = None
                if len(node.args) >= 2:
                    idx_shape = self._infer_shape(node.args[1])
                if idx_shape is not None:
                    return TensorShape(idx_shape.dims)
                if obj_shape is not None:
                    return TensorShape(obj_shape.dims)
            elif len(node.args) >= 3:  # torch.gather(input, dim, index)
                obj_shape = self._infer_shape(node.args[0])
                idx_shape = self._infer_shape(node.args[2])
                if idx_shape is not None:
                    return TensorShape(idx_shape.dims)
                if obj_shape is not None:
                    return TensorShape(obj_shape.dims)

        # scatter / scatter_add: shape-preserving (returns self shape)
        if base_name in ("scatter", "scatter_add", "scatter_", "scatter_add_"):
            if isinstance(node.func, ast.Attribute) and not _is_namespace_call(node):
                return self._infer_shape(node.func.value)
            if node.args:
                return self._infer_shape(node.args[0])

        # index_select(dim, index): output dims = obj.dims with dim replaced by len(index)
        if base_name == "index_select":
            if isinstance(node.func, ast.Attribute) and not _is_namespace_call(node):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape is None:
                    return None
                dim = self._const_val(node.args[0]) if node.args else None
                idx_shape = self._infer_shape(node.args[1]) if len(node.args) >= 2 else None
                if dim is None:
                    return None
                if dim < 0:
                    dim = obj_shape.ndim + dim
                if dim < 0 or dim >= obj_shape.ndim:
                    return None
                dims = list(obj_shape.dims)
                if idx_shape is not None and idx_shape.ndim == 1:
                    dims[dim] = idx_shape.dims[0]
                else:
                    dims[dim] = ShapeDim("_index_select")
                return TensorShape(tuple(dims))
            elif len(node.args) >= 3:  # torch.index_select(input, dim, index)
                obj_shape = self._infer_shape(node.args[0])
                if obj_shape is None:
                    return None
                dim = self._const_val(node.args[1])
                idx_shape = self._infer_shape(node.args[2])
                if dim is None:
                    return None
                if dim < 0:
                    dim = obj_shape.ndim + dim
                if dim < 0 or dim >= obj_shape.ndim:
                    return None
                dims = list(obj_shape.dims)
                if idx_shape is not None and idx_shape.ndim == 1:
                    dims[dim] = idx_shape.dims[0]
                else:
                    dims[dim] = ShapeDim("_index_select")
                return TensorShape(tuple(dims))

        # masked_select(mask): returns 1-D with symbolic length
        if base_name == "masked_select":
            return TensorShape((ShapeDim("_masked_select"),))

        # take_along_dim(input, indices, dim): returns shape of indices
        if base_name == "take_along_dim":
            if isinstance(node.func, ast.Attribute) and not _is_namespace_call(node):
                # x.take_along_dim(indices, dim)
                idx_shape = self._infer_shape(node.args[0]) if node.args else None
                if idx_shape is not None:
                    return TensorShape(idx_shape.dims)
                return self._infer_shape(node.func.value)
            elif len(node.args) >= 2:
                # torch.take_along_dim(input, indices, dim)
                idx_shape = self._infer_shape(node.args[1])
                if idx_shape is not None:
                    return TensorShape(idx_shape.dims)
                return self._infer_shape(node.args[0])

        # narrow(dim, start, length): replace dim with length
        if base_name == "narrow":
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if obj_shape is None or len(node.args) < 3:
                    return None
                dim = self._const_val(node.args[0])
                length = self._const_val(node.args[2])
                if dim is None or length is None:
                    return None
                if dim < 0:
                    dim = obj_shape.ndim + dim
                if dim < 0 or dim >= obj_shape.ndim:
                    return None
                dims = list(obj_shape.dims)
                dims[dim] = ShapeDim(length)
                return TensorShape(tuple(dims))

        # roll(shifts, dims) and flip(dims): shape-preserving
        if base_name in ("roll", "flip", "fliplr", "flipud", "rot90"):
            if isinstance(node.func, ast.Attribute):
                return self._infer_shape(node.func.value)
            if node.args:
                return self._infer_shape(node.args[0])

        # repeat_interleave(repeats, dim)
        if base_name == "repeat_interleave":
            obj_shape = None
            repeats = None
            dim = None
            if isinstance(node.func, ast.Attribute):
                obj_shape = self._infer_shape(node.func.value)
                if node.args:
                    repeats = self._const_val(node.args[0])
                if len(node.args) >= 2:
                    dim = self._const_val(node.args[1])
            else:
                if node.args:
                    obj_shape = self._infer_shape(node.args[0])
                if len(node.args) >= 2:
                    repeats = self._const_val(node.args[1])
                if len(node.args) >= 3:
                    dim = self._const_val(node.args[2])
            for kw in node.keywords:
                if kw.arg == "dim":
                    dim = self._const_val(kw.value)
                elif kw.arg == "repeats":
                    repeats = self._const_val(kw.value)
            if obj_shape is None:
                return None
            if dim is None:
                return TensorShape((ShapeDim("_repeat_flat"),))
            if dim < 0:
                dim = obj_shape.ndim + dim
            if dim < 0 or dim >= obj_shape.ndim:
                return None
            dims = list(obj_shape.dims)
            d = dims[dim]
            if repeats is not None and not d.is_symbolic:
                dims[dim] = ShapeDim(d.value * repeats)
            else:
                dims[dim] = ShapeDim("_repeat")
            return TensorShape(tuple(dims))

        # broadcast_to(shape) / x.broadcast_to(shape)
        if base_name == "broadcast_to":
            if isinstance(node.func, ast.Attribute) and node.args:
                shape_arg = node.args[0]
            elif len(node.args) >= 2:
                shape_arg = node.args[1]
            else:
                shape_arg = None
            if shape_arg is not None:
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    dims = []
                    for e in shape_arg.elts:
                        v = self._const_val(e)
                        if v is not None:
                            dims.append(ShapeDim(v))
                        else:
                            cn = self._const_or_name(e)
                            dims.append(ShapeDim(cn if cn is not None else "_bcast"))
                    return TensorShape(tuple(dims))

        # F.scaled_dot_product_attention(q, k, v) → (*q.dims[:-1], v.dims[-1])
        if base_name == "scaled_dot_product_attention":
            if len(node.args) >= 3:
                q = self._infer_shape(node.args[0])
                v = self._infer_shape(node.args[2])
                if q is not None and v is not None and q.ndim >= 1 and v.ndim >= 1:
                    return TensorShape(q.dims[:-1] + (v.dims[-1],))

        # Modern ops: element-wise activations and shape-preserving ops
        if base_name in TORCH_SHAPE_OPS:
            category = TORCH_SHAPE_OPS[base_name]
            if category == "elementwise":
                if isinstance(node.func, ast.Attribute):
                    obj_shape = self._infer_shape(node.func.value)
                    if obj_shape:
                        return TensorShape(obj_shape.dims)
                if node.args:
                    arg_shape = self._infer_shape(node.args[0])
                    if arg_shape:
                        return TensorShape(arg_shape.dims)

        return None

    def _infer_binop_shape(self, node: ast.BinOp) -> Optional[TensorShape]:
        """Infer shape from binary operations."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)

        # @ operator (matmul)
        if isinstance(node.op, ast.MatMult):
            if left and right:
                return compute_matmul_shape(left, right)

        # Element-wise ops: broadcasting
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                                ast.FloorDiv, ast.Mod, ast.Pow)):
            if left and right:
                return compute_broadcast_shape(left, right)
            return left or right

        return left or right

    def _infer_subscript_shape(self, node: ast.Subscript) -> Optional[TensorShape]:
        """Infer shape from subscript operations (indexing/slicing)."""
        obj_shape = self._infer_shape(node.value)
        if not obj_shape:
            return None

        # Simple integer index: x[0] removes one dimension
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            if obj_shape.ndim > 1:
                dims = list(obj_shape.dims)
                dims.pop(0)
                return TensorShape(tuple(dims))
            return TensorShape(())

        return None

    # ── Shape constraint checking ──────────────────────────────────────

    def _check_expr_shapes(self, node: ast.expr):
        """Check shape constraints at operation sites."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            self._check_matmul(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._check_broadcast(node)
        elif isinstance(node, ast.Call):
            self._check_call_shapes(node)

        # Recurse
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._check_expr_shapes(child)

    def _check_matmul(self, node: ast.BinOp):
        """Check matmul shape compatibility."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)
        if not left or not right:
            return

        self.constraints_generated += 1
        err = check_matmul_compatible(left, right)
        if err:
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.MATMUL_INCOMPAT,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=err,
                function=self.func_name,
                variable="",
                actual_shape=left,
                expected_shape=right,
            ))
        else:
            # Use Z3 for symbolic dimension checking
            self._check_matmul_z3(node, left, right)
        self.constraints_checked += 1

    def _check_matmul_z3(self, node: ast.BinOp,
                          a: TensorShape, b: TensorShape):
        """Use Z3 to verify symbolic matmul compatibility.

        Check validity of k_a == k_b: if Not(k_a == k_b) is UNSAT,
        dimensions always match (no error). If SAT, a counterexample
        exists (report error). If unknown, don't report.
        """
        if not HAS_Z3:
            return

        k_a = a.dims[-1] if a.ndim >= 1 else None
        k_b = b.dims[-2] if b.ndim >= 2 else (b.dims[0] if b.ndim == 1 else None)
        if k_a is None or k_b is None:
            return

        # If both concrete, already checked
        if not k_a.is_symbolic and not k_b.is_symbolic:
            return

        if k_a.is_symbolic:
            z_ka = _z3.Int(str(k_a.value))
        else:
            z_ka = _z3.IntVal(k_a.value)

        if k_b.is_symbolic:
            z_kb = _z3.Int(str(k_b.value))
        else:
            z_kb = _z3.IntVal(k_b.value)

        self.constraints_generated += 1
        self.constraints_checked += 1

        # Check if k_a == k_b is VALID by checking if Not(k_a == k_b) is UNSAT
        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        # Dimensions are positive
        if k_a.is_symbolic:
            s.add(z_ka > 0)
        if k_b.is_symbolic:
            s.add(z_kb > 0)
        s.add(_z3.Not(z_ka == z_kb))

        result = s.check()
        if result == _z3.unsat:
            # k_a == k_b is valid (always true) — no error
            return
        elif result == _z3.sat:
            # Counterexample exists — dimensions can mismatch
            model = s.model()
            cex = {d.name(): str(model[d]) for d in model.decls()}
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.MATMUL_INCOMPAT,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=f"Possible matmul dimension mismatch (Z3 counterexample: {cex})",
                function=self.func_name,
                variable="",
                actual_shape=a,
                expected_shape=b,
                z3_counterexample=cex,
            ))
        # If unknown, don't report an error

    def _check_broadcast(self, node: ast.BinOp):
        """Check broadcasting compatibility for element-wise ops."""
        left = self._infer_shape(node.left)
        right = self._infer_shape(node.right)
        if not left or not right:
            # Even with one unknown shape, flag the suspicious pattern where
            # a 2D tensor with a concrete leading dim=1 is being combined with
            # the output of a layer call (self.xxx(...)).  This indicates the
            # programmer constructed a (1, hidden) bias and forgot that the
            # activation it is added to is 3D+ (missing unsqueeze).
            known = left if left is not None else right
            unknown_node = node.right if left is not None else node.left
            if known is not None:
                self._check_suspicious_one_sided_broadcast(node, known,
                                                           unknown_node)
            return

        self.constraints_generated += 1
        result = compute_broadcast_shape(left, right)
        if result is None:
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.BROADCAST_FAIL,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=(f"Cannot broadcast shapes {left.pretty()} and "
                         f"{right.pretty()} for element-wise operation"),
                function=self.func_name,
                variable="",
                actual_shape=left,
                expected_shape=right,
            ))
        else:
            # Use Z3 for symbolic broadcast checking
            self._check_broadcast_z3(node, left, right)
            # Check for suspicious ndim mismatch (potential missing unsqueeze)
            self._check_broadcast_ndim_mismatch(node, left, right)
        self.constraints_checked += 1

    def _check_suspicious_one_sided_broadcast(
            self,
            node: ast.BinOp,
            known_shape: "TensorShape",
            unknown_node: ast.expr):
        """Flag a potential missing-unsqueeze when only one operand shape is known.

        Specifically: when the known operand is 2D with a concrete leading
        dim=1 (the classic ``torch.zeros(1, hidden)`` bias pattern) and the
        unknown operand is the result of a ``self.<layer>(...)`` call (i.e.
        a module layer whose input ndim we cannot determine statically), we
        emit a warning.  The combination strongly suggests the programmer
        intended the bias to broadcast over a batch+sequence axis but
        omitted one or more unsqueeze calls.
        """
        if not (known_shape.ndim == 2
                and not known_shape.dims[0].is_symbolic
                and known_shape.dims[0].value == 1):
            return
        # Resolve variable names through the origin map so that
        # ``out + bias`` (where ``out = self.linear(...)``) is handled
        # identically to ``self.linear(...) + bias``.
        resolved = self._resolve_origin(unknown_node)
        if not self._is_layer_call(resolved):
            return
        self.errors.append(ShapeError(
            kind=ShapeErrorKind.BROADCAST_FAIL,
            line=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0),
            message=(
                f"Suspicious broadcast: 2D tensor {known_shape.pretty()} with "
                f"leading dim=1 is being added to the output of a layer whose "
                f"rank is unknown. If the layer output is 3D+ (e.g. batch × seq "
                f"× hidden), this tensor needs unsqueeze(0) or unsqueeze(1) "
                f"before the addition — consider torch.zeros(1, 1, ...) instead."
            ),
            function=self.func_name,
            variable="",
            actual_shape=known_shape,
            expected_shape=known_shape,
            severity="warning",
        ))

    def _resolve_origin(self, node: ast.expr) -> ast.expr:
        """If *node* is a Name that was assigned from another expression,
        return that expression; otherwise return *node* unchanged.

        Follows at most one level of assignment to avoid chasing long
        chains where the connection to a layer call is too tenuous.
        """
        if isinstance(node, ast.Name):
            origin = self._var_origins.get(node.id)
            if origin is not None:
                return origin
        return node

    @staticmethod
    def _is_layer_call(node: ast.expr) -> bool:
        """Return True if *node* looks like ``self.<attr>(...)``."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        )

    def _check_broadcast_z3(self, node: ast.BinOp,
                             a: TensorShape, b: TensorShape):
        """Use Z3 to verify symbolic broadcasting compatibility.

        NumPy/PyTorch broadcasting: align from right, each dim pair
        must be (d,d), (d,1), or (1,d).
        """
        if not HAS_Z3:
            return

        # Check if any dims are symbolic
        has_symbolic = any(
            d.is_symbolic
            for shape in (a, b)
            for d in shape.dims
        )
        if not has_symbolic:
            return

        ndim = max(a.ndim, b.ndim)
        constraints = []
        sym_vars = []

        for i in range(1, ndim + 1):
            d_a = a.dims[-i] if i <= a.ndim else ShapeDim(1)
            d_b = b.dims[-i] if i <= b.ndim else ShapeDim(1)

            if d_a.is_symbolic:
                z_a = _z3.Int(str(d_a.value))
                sym_vars.append((z_a, d_a))
            else:
                z_a = _z3.IntVal(d_a.value)

            if d_b.is_symbolic:
                z_b = _z3.Int(str(d_b.value))
                sym_vars.append((z_b, d_b))
            else:
                z_b = _z3.IntVal(d_b.value)

            # Broadcasting rule: d_a == d_b OR d_a == 1 OR d_b == 1
            constraints.append(_z3.Or(z_a == z_b, z_a == 1, z_b == 1))

        if not constraints:
            return

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        # Dimensions are positive
        for z_var, dim in sym_vars:
            s.add(z_var > 0)
        # Check if broadcast constraints can be violated
        s.add(_z3.Not(_z3.And(*constraints)))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            # Broadcasting always holds — OK
            return
        # If sat or unknown, don't report (might be valid for some assignments)

    def _check_broadcast_ndim_mismatch(self, node: ast.BinOp,
                                        a: TensorShape, b: TensorShape):
        """Check for suspicious ndim mismatch in broadcast operations.

        When a 2D tensor with leading dim=1 (e.g., torch.zeros(1, hidden))
        is added to a 3D+ tensor, this often indicates a missing unsqueeze
        or incorrect tensor construction.
        """
        if a.ndim == b.ndim:
            return
        smaller, larger = (a, b) if a.ndim < b.ndim else (b, a)
        ndim_diff = larger.ndim - smaller.ndim
        if ndim_diff < 1:
            return
        # Flag if the smaller tensor has a leading dimension of 1
        # This suggests the programmer was thinking about batch dim
        # but forgot intermediate dimensions
        if (smaller.ndim >= 2
                and not smaller.dims[0].is_symbolic
                and smaller.dims[0].value == 1):
            self.errors.append(ShapeError(
                kind=ShapeErrorKind.BROADCAST_FAIL,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                message=(
                    f"Suspicious broadcast: {smaller.pretty()} ({smaller.ndim}D) "
                    f"with {larger.pretty()} ({larger.ndim}D). "
                    f"Leading dim=1 in the smaller tensor suggests a missing "
                    f"dimension — consider unsqueeze or reshape"),
                function=self.func_name,
                variable="",
                actual_shape=smaller,
                expected_shape=larger,
                severity="warning",
            ))

    def _check_call_shapes(self, node: ast.Call):
        """Check shape constraints at function call sites."""
        func_name = self._get_call_name(node)
        if not func_name:
            return
        base_name = func_name.split(".")[-1] if "." in func_name else func_name

        # matmul/mm/dot calls
        if base_name in ("matmul", "mm", "dot"):
            args = node.args
            if isinstance(node.func, ast.Attribute) and args:
                a_shape = self._infer_shape(node.func.value)
                b_shape = self._infer_shape(args[0])
            elif len(args) >= 2:
                a_shape = self._infer_shape(args[0])
                b_shape = self._infer_shape(args[1])
            else:
                return
            if a_shape and b_shape:
                self.constraints_generated += 1
                err = check_matmul_compatible(a_shape, b_shape)
                if err:
                    self.errors.append(ShapeError(
                        kind=ShapeErrorKind.MATMUL_INCOMPAT,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=err,
                        function=self.func_name,
                        variable="",
                        actual_shape=a_shape,
                        expected_shape=b_shape,
                    ))
                self.constraints_checked += 1

        # cat/concatenate
        if base_name in ("cat", "concatenate"):
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                shapes = [self._infer_shape(elt) for elt in node.args[0].elts]
                valid_shapes = [s for s in shapes if s is not None]
                if len(valid_shapes) >= 2:
                    dim = 0
                    if len(node.args) >= 2:
                        d = self._const_val(node.args[1])
                        if d is not None:
                            dim = d
                    for kw in node.keywords:
                        if kw.arg == "dim":
                            d = self._const_val(kw.value)
                            if d is not None:
                                dim = d
                    self._check_cat_shapes(node, valid_shapes, dim)

        # nn.Linear: check input dimension matches in_features
        if isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self" and node.args):
                layer_attr = node.func.attr
                for key, info in self._layer_shapes.items():
                    if key.endswith(f".{layer_attr}"):
                        if info["type"] == "Linear":
                            x_shape = self._infer_shape(node.args[0])
                            if x_shape and x_shape.ndim >= 1:
                                last_dim = x_shape.dims[-1]
                                in_f = info["in_features"]
                                if (not last_dim.is_symbolic
                                        and isinstance(in_f, int)
                                        and last_dim.value != in_f):
                                    self.constraints_generated += 1
                                    self.errors.append(ShapeError(
                                        kind=ShapeErrorKind.DIM_MISMATCH,
                                        line=getattr(node, "lineno", 0),
                                        col=getattr(node, "col_offset", 0),
                                        message=(f"Linear layer expects input dim "
                                                 f"{in_f}, got {last_dim.value}"),
                                        function=self.func_name,
                                        variable=layer_attr,
                                        actual_shape=x_shape,
                                    ))
                                    self.constraints_checked += 1
                                elif last_dim.is_symbolic and isinstance(in_f, int):
                                    self._check_linear_z3(
                                        node, x_shape, last_dim, in_f, layer_attr)

                        # MultiheadAttention: check batch_first format
                        if info["type"] == "MultiheadAttention":
                            if not info.get("batch_first", False):
                                self.constraints_generated += 1
                                self.errors.append(ShapeError(
                                    kind=ShapeErrorKind.DIM_MISMATCH,
                                    line=getattr(node, "lineno", 0),
                                    col=getattr(node, "col_offset", 0),
                                    message=(
                                        "MultiheadAttention has batch_first=False "
                                        "but input may be batch-first (batch, seq, dim). "
                                        "Pass batch_first=True or transpose input to "
                                        "(seq, batch, dim)"),
                                    function=self.func_name,
                                    variable=layer_attr,
                                ))
                                self.constraints_checked += 1

        # Identity permute detection: x.permute(0, 1, 2, ...) is a no-op
        if base_name == "permute":
            if isinstance(node.func, ast.Attribute):
                perm_args = [self._const_val(a) for a in node.args]
                if all(p is not None for p in perm_args):
                    identity = list(range(len(perm_args)))
                    if perm_args == identity:
                        self.constraints_generated += 1
                        self.errors.append(ShapeError(
                            kind=ShapeErrorKind.DIM_MISMATCH,
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            message=(
                                f"No-op permute{tuple(perm_args)}: this permutation "
                                f"does not rearrange any dimensions. Did you mean "
                                f"permute(0, 2, 1, 3) to move heads before sequence?"),
                            function=self.func_name,
                            variable="",
                        ))
                        self.constraints_checked += 1

        # Reshape inconsistency with tracked class attributes
        if base_name in ("reshape", "view"):
            self._check_reshape_attribute_consistency(node)

    def _check_reshape_attribute_consistency(self, node: ast.Call):
        """Check if reshape literals are inconsistent with class attributes.

        Detects patterns like: self.num_heads=12 but reshape uses 8 for
        the heads dimension. Common bug in multi-head attention implementations.
        """
        if not self._class_attrs:
            return

        reshape_literals = set()
        for arg in node.args:
            v = self._const_val(arg)
            if v is not None and v > 1:
                reshape_literals.add(v)

        head_attrs = {k: v for k, v in self._class_attrs.items()
                      if any(tok in k.lower() for tok in
                             ("head", "nhead", "n_head", "num_head"))}
        for attr_name, attr_val in head_attrs.items():
            if not isinstance(attr_val, int) or attr_val <= 1:
                continue
            for lit in reshape_literals:
                if lit != attr_val and lit > 1:
                    if lit < attr_val * 3 and attr_val < lit * 3:
                        self.constraints_generated += 1
                        self.errors.append(ShapeError(
                            kind=ShapeErrorKind.RESHAPE_INVALID,
                            line=getattr(node, "lineno", 0),
                            col=getattr(node, "col_offset", 0),
                            message=(
                                f"Reshape uses {lit} but self.{attr_name}="
                                f"{attr_val}. The reshape dimension may be "
                                f"inconsistent with the number of attention heads"),
                            function=self.func_name,
                            variable=attr_name,
                        ))
                        self.constraints_checked += 1
                        return

    def _check_linear_z3(self, node: ast.Call, x_shape: TensorShape,
                          last_dim: ShapeDim, in_f: int, layer_attr: str):
        """Use Z3 to verify symbolic nn.Linear input dimension."""
        if not HAS_Z3:
            return

        z_dim = _z3.Int(str(last_dim.value))
        z_inf = _z3.IntVal(in_f)

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(z_dim > 0)
        s.add(_z3.Not(z_dim == z_inf))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            return  # Dimension always matches

    def _check_cat_shapes(self, node: ast.Call,
                           shapes: List[TensorShape], dim: int):
        """Check that all tensors in cat have matching shapes on non-cat dims."""
        self.constraints_generated += 1
        ref = shapes[0]
        has_symbolic_mismatch = False
        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != ref.ndim:
                self.errors.append(ShapeError(
                    kind=ShapeErrorKind.CAT_INCOMPAT,
                    line=getattr(node, "lineno", 0),
                    col=getattr(node, "col_offset", 0),
                    message=(f"cat: tensor {i} has {s.ndim} dims, "
                             f"expected {ref.ndim} to match tensor 0"),
                    function=self.func_name,
                    variable="",
                    actual_shape=s,
                    expected_shape=ref,
                ))
                continue
            for j in range(ref.ndim):
                if j == dim:
                    continue
                d_ref = ref.dims[j]
                d_s = s.dims[j]
                if (not d_ref.is_symbolic and not d_s.is_symbolic
                        and d_ref.value != d_s.value):
                    self.errors.append(ShapeError(
                        kind=ShapeErrorKind.CAT_INCOMPAT,
                        line=getattr(node, "lineno", 0),
                        col=getattr(node, "col_offset", 0),
                        message=(f"cat dim {dim}: tensor {i} has dim[{j}]="
                                 f"{d_s.value}, expected {d_ref.value}"),
                        function=self.func_name,
                        variable="",
                        actual_shape=s,
                        expected_shape=ref,
                    ))
                elif d_ref.is_symbolic or d_s.is_symbolic:
                    has_symbolic_mismatch = True
        self.constraints_checked += 1
        # Use Z3 for symbolic cat dim checking
        if has_symbolic_mismatch:
            self._check_cat_z3(node, shapes, dim)

    def _check_cat_z3(self, node: ast.Call,
                       shapes: List[TensorShape], dim: int):
        """Use Z3 to verify symbolic cat dimension compatibility."""
        if not HAS_Z3:
            return

        ref = shapes[0]
        constraints = []
        sym_vars = []

        for i, s in enumerate(shapes[1:], 1):
            if s.ndim != ref.ndim:
                continue
            for j in range(ref.ndim):
                if j == dim:
                    continue
                d_ref = ref.dims[j]
                d_s = s.dims[j]
                if d_ref.is_symbolic:
                    z_ref = _z3.Int(str(d_ref.value))
                    sym_vars.append(z_ref)
                else:
                    z_ref = _z3.IntVal(d_ref.value)
                if d_s.is_symbolic:
                    z_s = _z3.Int(str(d_s.value))
                    sym_vars.append(z_s)
                else:
                    z_s = _z3.IntVal(d_s.value)
                constraints.append(z_ref == z_s)

        if not constraints:
            return

        s = _z3.Solver()
        s.set("timeout", self.timeout_ms)
        for v in sym_vars:
            s.add(v > 0)
        # Check if constraints can be violated
        s.add(_z3.Not(_z3.And(*constraints)))
        self.constraints_generated += 1
        self.constraints_checked += 1

        result = s.check()
        if result == _z3.unsat:
            return  # All constraints always hold
        # If sat or unknown, dims might not match but don't report without certainty

    # ── Shape assertion harvesting ─────────────────────────────────────

    def _harvest_shape_assert(self, node: ast.Assert):
        """Harvest shape information from assert statements.

        Patterns:
          assert x.shape == (3, 4)
          assert x.shape[0] == batch_size
          assert len(x.shape) == 3
          assert x.ndim == 2
        """
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            if isinstance(op, ast.Eq):
                left = test.left
                comp = test.comparators[0]

                # x.shape == (d1, d2, ...)
                if (isinstance(left, ast.Attribute) and left.attr == "shape"
                        and isinstance(left.value, ast.Name)):
                    var = left.value.id
                    shape = self._extract_shape_literal(comp)
                    if shape:
                        self.shape_env = self.shape_env.set(var, shape)
                        self._shape_preds.append(shape.to_pred(var))
                        return

                # x.shape[i] == d
                if (isinstance(left, ast.Subscript)
                        and isinstance(left.value, ast.Attribute)
                        and left.value.attr == "shape"
                        and isinstance(left.value.value, ast.Name)):
                    var = left.value.value.id
                    axis = self._const_val(left.slice)
                    dim_val = self._const_or_name(comp)
                    if axis is not None and dim_val is not None:
                        existing = self.shape_env.get(var)
                        if existing and axis < existing.ndim:
                            dims = list(existing.dims)
                            dims[axis] = ShapeDim(dim_val)
                            self.shape_env = self.shape_env.set(
                                var, TensorShape(tuple(dims)))

                # x.ndim == n
                if (isinstance(left, ast.Attribute) and left.attr == "ndim"
                        and isinstance(left.value, ast.Name)):
                    var = left.value.id
                    ndim = self._const_val(comp)
                    if ndim is not None:
                        existing = self.shape_env.get(var)
                        if not existing:
                            self.shape_env = self.shape_env.set(
                                var, TensorShape.unknown(ndim))

    # ── Utility methods ────────────────────────────────────────────────

    def _shape_from_creation_args(self, node: ast.Call) -> Optional[TensorShape]:
        """Extract shape from torch.zeros(d1, d2, ...) or torch.zeros((d1, d2)).

        Falls back to a permissive parse that uses symbolic dims for any
        argument that is not a literal constant or variable name (e.g.
        ``x.size(-1)``, ``hidden * 2``).  This preserves rank information
        and concrete values (such as leading ``1``s) so that the
        missing-unsqueeze broadcast check can still fire.
        """
        if not node.args:
            return None

        first = node.args[0]
        # torch.zeros((3, 4)) — tuple argument
        if isinstance(first, ast.Tuple):
            dims = self._args_to_dims([first])
            if dims:
                return TensorShape.from_tuple(dims)
            return None

        # torch.zeros(3, 4) — individual arguments (strict path first)
        dims = self._args_to_dims(node.args)
        if dims:
            return TensorShape.from_tuple(dims)

        # Permissive fallback: keep concrete ints where available, use a
        # unique symbolic name for expressions we cannot evaluate (e.g.
        # ``x.size(-1)``).  This lets us at least track rank and leading-1
        # dims for the missing-unsqueeze heuristic.
        sym_counter = getattr(self, "_sym_counter", 0)
        result_dims = []
        for arg in node.args:
            v = self._const_or_name(arg)
            if v is not None:
                result_dims.append(v)
            else:
                # Use a fresh symbolic name so different unknown dims are
                # treated as independent (not aliased).
                sym_counter += 1
                result_dims.append(f"_sym{sym_counter}")
        self._sym_counter = sym_counter
        if result_dims:
            return TensorShape.from_tuple(tuple(result_dims))
        return None

    def _extract_shape_literal(self, node: ast.expr) -> Optional[TensorShape]:
        """Extract a shape tuple literal from AST node."""
        if isinstance(node, ast.Tuple):
            dims = []
            for elt in node.elts:
                v = self._const_or_name(elt)
                if v is None:
                    return None
                dims.append(ShapeDim(v))
            return TensorShape(tuple(dims))
        return None

    def _extract_shape_args(self, node: ast.Call) -> Optional[Tuple]:
        """Extract reshape/view args."""
        if not node.args:
            return None
        dims = []
        for arg in node.args:
            v = self._const_or_name(arg)
            if v is not None:
                dims.append(v)
            else:
                dims.append("_unknown")
        return tuple(dims)

    def _args_to_dims(self, args: list) -> Optional[Tuple]:
        """Convert AST arguments to a tuple of dimension values."""
        dims = []
        for arg in args:
            if isinstance(arg, ast.Tuple):
                for elt in arg.elts:
                    v = self._const_or_name(elt)
                    if v is not None:
                        dims.append(v)
                    else:
                        return None
            else:
                v = self._const_or_name(arg)
                if v is not None:
                    dims.append(v)
                else:
                    return None
        return tuple(dims)

    def _compute_cat_shape(self, shapes: List[Optional[TensorShape]],
                            dim: int) -> Optional[TensorShape]:
        """Compute result shape of torch.cat."""
        valid = [s for s in shapes if s is not None]
        if not valid:
            return None
        base = valid[0]
        total_dim = ShapeDim(0)
        all_concrete = True
        cat_total = 0
        for s in valid:
            d = s.dims[dim] if dim < s.ndim else ShapeDim(0)
            if d.is_symbolic:
                all_concrete = False
            else:
                cat_total += d.value
        result_dims = list(base.dims)
        if all_concrete:
            result_dims[dim] = ShapeDim(cat_total)
        else:
            result_dims[dim] = ShapeDim("_cat_dim")
        return TensorShape(tuple(result_dims))

    def _shape_from_annotation(self, ann: ast.expr) -> Optional[TensorShape]:
        """Extract shape from type annotation if present."""
        # TODO: support Annotated[Tensor, Shape(3, 4)]
        return None

    def _infer_einsum_shape(
        self, equation: str, operand_shapes: List[Optional[TensorShape]]
    ) -> Optional[TensorShape]:
        """Infer output shape from an einsum equation string.

        Delegates to the canonical, torch-equivalent parser in
        ``src.smt.einsum_theory`` (handles explicit/implicit output, diagonals,
        and ellipsis broadcasting). Imported locally to avoid a circular import
        (``einsum_theory`` depends on this module).
        """
        if any(s is None for s in operand_shapes):
            return None
        try:
            from src.smt.einsum_theory import infer_einsum_shape
        except Exception:
            return None
        return infer_einsum_shape(equation, list(operand_shapes))

    @staticmethod
    def _get_call_name(node: ast.Call) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            if isinstance(node.func.value, ast.Attribute):
                return node.func.attr
        return None

    @staticmethod
    def _const_val(node: ast.expr) -> Optional[int]:
        """Extract an integer constant from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        return None

    @staticmethod
    def _const_or_name(node: ast.expr) -> Optional[Union[int, str]]:
        """Extract int constant or variable name."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return -node.operand.value
        return None

    @staticmethod
    def _eval_const_expr(node: ast.expr) -> Optional[int]:
        """Evaluate a constant arithmetic expression (e.g., 64 * 14 * 14).

        Handles int constants, unary minus, and binary +, -, *, //, **, %.
        Returns None if the expression contains non-constant parts.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = TensorShapeAnalyzer._eval_const_expr(node.operand)
            return -inner if inner is not None else None
        if isinstance(node, ast.BinOp):
            left = TensorShapeAnalyzer._eval_const_expr(node.left)
            right = TensorShapeAnalyzer._eval_const_expr(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv) and right != 0:
                return left // right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod) and right != 0:
                return left % right
        if isinstance(node, ast.Name):
            return None
        return None

    def _analyze_functional_call(self, node: ast.Call,
                                 func_name: str) -> Optional[TensorShape]:
        """Handle torch.nn.functional calls: F.linear, F.conv2d, F.softmax,
        F.layer_norm, F.relu, F.gelu, F.dropout, F.cross_entropy, etc.

        These are NOT tracked by nn.Module-based analysis but are widely used.
        """
        # Shape-preserving ops: F.relu, F.gelu, F.softmax, F.dropout, etc.
        shape_preserving = {
            "softmax", "log_softmax", "relu", "gelu", "silu",
            "leaky_relu", "tanh", "sigmoid", "dropout",
            "layer_norm", "batch_norm", "group_norm", "instance_norm",
        }
        if func_name in shape_preserving:
            if node.args:
                return self._infer_shape(node.args[0])
            if isinstance(node.func, ast.Attribute):
                return self._infer_shape(node.func.value)
        return None

    def _extract_shape_args_enhanced(self, node: ast.Call,
                                     obj_shape: Optional[TensorShape]
                                     ) -> Optional[Tuple]:
        """Enhanced reshape/view arg extraction.

        Handles x.size(i) and x.shape[i] by mapping them to sentinel 0
        (copy from input dim).
        """
        if not node.args:
            return None
        dims: list = []
        for arg in node.args:
            v = self._const_or_name(arg)
            if v is not None:
                dims.append(v)
            elif self._is_size_call(arg, obj_shape):
                dims.append(0)  # sentinel: copy from input
            else:
                dims.append("_unknown")
        return tuple(dims)

    @staticmethod
    def _is_size_call(node: ast.expr,
                      obj_shape: Optional[TensorShape] = None) -> bool:
        """Check if node is x.size(i) or x.shape[i]."""
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "size"):
                return True
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_shapes(source: str, **kwargs) -> ShapeAnalysisResult:
    """Analyze Python source for tensor shape errors.

    Usage::

        from src.tensor_shapes import analyze_shapes

        result = analyze_shapes('''
        import torch
        x = torch.randn(3, 4)
        y = torch.randn(5, 6)
        z = x @ y  # Shape error: inner dims 4 != 5
        ''')
        for err in result.errors:
            print(f"L{err.line}: {err.message}")
    """
    analyzer = TensorShapeAnalyzer(**kwargs)
    return analyzer.analyze_source(source)
