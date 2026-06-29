"""Bug records produced by the symbolic executor, and their mapping to the
public :class:`src.api.Bug` type.

The engine is *self-contained*: it can run and be tested without importing the
rest of TensorGuard.  :meth:`SymBug.to_api_bug` is only used when integrating
with ``src.api`` and imports it lazily so the package has no hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["SymBugKind", "SymBug"]


class SymBugKind(Enum):
    UNPACK_ARITY_MISMATCH = "unpack_arity_mismatch"
    RANK_INDEX_ERROR = "rank_index_error"
    LAYER_DIM_MISMATCH = "layer_dim_mismatch"
    AXIS_NAME_CONSTRUCTION = "axis_name_construction"
    CHANNEL_AXIS_MISMATCH = "channel_axis_mismatch"
    NONE_PROPAGATION = "none_propagation"
    RETURN_ARITY_CONTRACT = "return_arity_contract"
    DIVISION_BY_ZERO = "division_by_zero"
    RESHAPE_SIZE_MISMATCH = "reshape_size_mismatch"
    NEGATIVE_DIMENSION = "negative_dimension"
    BROADCAST_MISMATCH = "broadcast_mismatch"
    MATMUL_DIM_MISMATCH = "matmul_dim_mismatch"
    CAT_SHAPE_MISMATCH = "cat_shape_mismatch"
    AXIS_OUT_OF_RANGE = "axis_out_of_range"
    TENSOR_INDEX_OOB = "tensor_index_oob"
    EINSUM_DIM_MISMATCH = "einsum_dim_mismatch"
    EINOPS_PATTERN_MISMATCH = "einops_pattern_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    ITEM_ON_NONSCALAR = "item_on_nonscalar"
    BOOL_ON_NONSCALAR = "bool_on_nonscalar"
    INPLACE_ON_LEAF = "inplace_on_leaf"
    DISCARDED_TENSOR_RESULT = "discarded_tensor_result"
    DIRECT_FORWARD_CALL = "direct_forward_call"
    TENSOR_DATA_ACCESS = "tensor_data_access"
    MISSING_SUPER_INIT = "missing_super_init"
    TENSOR_COPY_CONSTRUCT = "tensor_copy_construct"
    BACKWARD_ON_NONSCALAR = "backward_on_nonscalar"
    NUMPY_ON_GRAD = "numpy_on_grad"
    REQUIRES_GRAD_NON_FLOAT = "requires_grad_non_float"
    REPEAT_DIMS_TOO_FEW = "repeat_dims_too_few"
    EXPAND_SHAPE_MISMATCH = "expand_shape_mismatch"
    MISSING_ZERO_GRAD = "missing_zero_grad"
    STEP_WITHOUT_BACKWARD = "step_without_backward"
    BACKWARD_WITHOUT_STEP = "backward_without_step"
    BACKWARD_NO_GRAD = "backward_no_grad"


# Map symexec kinds onto the existing public BugCategory names.  Anything
# without a dedicated public category falls back to TYPE_ERROR, which is the
# honest umbrella for "this would raise at runtime".
_API_CATEGORY = {
    SymBugKind.UNPACK_ARITY_MISMATCH: "TYPE_ERROR",
    SymBugKind.RANK_INDEX_ERROR: "INDEX_OUT_OF_BOUNDS",
    SymBugKind.LAYER_DIM_MISMATCH: "TYPE_ERROR",
    SymBugKind.AXIS_NAME_CONSTRUCTION: "TYPE_ERROR",
    SymBugKind.CHANNEL_AXIS_MISMATCH: "TYPE_ERROR",
    SymBugKind.NONE_PROPAGATION: "NULL_DEREFERENCE",
    SymBugKind.RETURN_ARITY_CONTRACT: "TYPE_ERROR",
    SymBugKind.DIVISION_BY_ZERO: "DIVISION_BY_ZERO",
    SymBugKind.RESHAPE_SIZE_MISMATCH: "TYPE_ERROR",
    SymBugKind.NEGATIVE_DIMENSION: "TYPE_ERROR",
    SymBugKind.BROADCAST_MISMATCH: "TYPE_ERROR",
    SymBugKind.MATMUL_DIM_MISMATCH: "TYPE_ERROR",
    SymBugKind.CAT_SHAPE_MISMATCH: "TYPE_ERROR",
    SymBugKind.AXIS_OUT_OF_RANGE: "TYPE_ERROR",
    SymBugKind.TENSOR_INDEX_OOB: "INDEX_OUT_OF_BOUNDS",
    SymBugKind.EINSUM_DIM_MISMATCH: "TYPE_ERROR",
    SymBugKind.EINOPS_PATTERN_MISMATCH: "TYPE_ERROR",
    SymBugKind.DEVICE_MISMATCH: "TYPE_ERROR",
    SymBugKind.ITEM_ON_NONSCALAR: "TYPE_ERROR",
    SymBugKind.BOOL_ON_NONSCALAR: "TYPE_ERROR",
    SymBugKind.INPLACE_ON_LEAF: "TYPE_ERROR",
    SymBugKind.DISCARDED_TENSOR_RESULT: "TYPE_ERROR",
    SymBugKind.DIRECT_FORWARD_CALL: "TYPE_ERROR",
    SymBugKind.TENSOR_DATA_ACCESS: "TYPE_ERROR",
    SymBugKind.MISSING_SUPER_INIT: "TYPE_ERROR",
    SymBugKind.TENSOR_COPY_CONSTRUCT: "TYPE_ERROR",
    SymBugKind.BACKWARD_ON_NONSCALAR: "TYPE_ERROR",
    SymBugKind.NUMPY_ON_GRAD: "TYPE_ERROR",
    SymBugKind.REQUIRES_GRAD_NON_FLOAT: "TYPE_ERROR",
    SymBugKind.REPEAT_DIMS_TOO_FEW: "TYPE_ERROR",
    SymBugKind.EXPAND_SHAPE_MISMATCH: "TYPE_ERROR",
    SymBugKind.MISSING_ZERO_GRAD: "TYPE_ERROR",
    SymBugKind.STEP_WITHOUT_BACKWARD: "TYPE_ERROR",
    SymBugKind.BACKWARD_WITHOUT_STEP: "TYPE_ERROR",
    SymBugKind.BACKWARD_NO_GRAD: "TYPE_ERROR",
}


@dataclass(frozen=True)
class SymBug:
    kind: SymBugKind
    message: str
    line: int
    col: int
    function: str = ""
    severity: str = "error"
    confidence: float = 0.9
    fix_suggestion: Optional[str] = None
    evidence: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "line": self.line,
            "col": self.col,
            "function": self.function,
            "severity": self.severity,
            "confidence": self.confidence,
            "fix_suggestion": self.fix_suggestion,
            "evidence": self.evidence,
        }

    def to_api_bug(self, filename: str = "<unknown>"):
        """Convert to ``src.api.Bug`` (lazy import to avoid a hard dependency)."""
        from ..api import Bug, BugCategory, SourceLocation

        cat = getattr(BugCategory, _API_CATEGORY[self.kind], BugCategory.TYPE_ERROR)
        return Bug(
            category=cat,
            message=self.message,
            location=SourceLocation(file=filename, line=self.line, column=self.col),
            severity=self.severity,
            confidence=self.confidence,
            fix_suggestion=self.fix_suggestion,
            guard_evidence=self.evidence,
        )
