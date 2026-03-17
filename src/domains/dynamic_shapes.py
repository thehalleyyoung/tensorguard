"""
Dynamic Shape Inference for Real-World Tensor Operations.

Handles patterns common in production PyTorch models:
  - ``tensor.view(-1, dim)`` / ``tensor.reshape(batch, -1)`` where -1 means
    "infer this dimension from the total element count"
  - ``tensor.expand(batch, -1, -1)`` where -1 means "keep original size"
  - Symbolic dimension arithmetic with Z3 constraint generation

The key insight: reshape(tensor, (a, b, -1)) on a tensor of total size N
requires -1 = N / (a * b), which is encoded as a Z3 divisibility constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.tensor_shapes import ShapeDim, TensorShape


@dataclass(frozen=True)
class SymbolicProduct:
    """Represents a product of symbolic and concrete dimensions."""

    concrete_factor: int = 1
    symbolic_names: Tuple[str, ...] = ()

    @property
    def is_fully_concrete(self) -> bool:
        return len(self.symbolic_names) == 0

    def __mul__(self, other: "SymbolicProduct") -> "SymbolicProduct":
        return SymbolicProduct(
            concrete_factor=self.concrete_factor * other.concrete_factor,
            symbolic_names=self.symbolic_names + other.symbolic_names,
        )


def _shape_total_elements(shape: TensorShape) -> SymbolicProduct:
    """Compute the total number of elements as a symbolic product."""
    result = SymbolicProduct(concrete_factor=1)
    for dim in shape.dims:
        if dim.is_symbolic:
            result = result * SymbolicProduct(symbolic_names=(dim.value,))
        else:
            result = result * SymbolicProduct(concrete_factor=dim.value)
    return result


def infer_neg_one_dim(
    input_shape: TensorShape,
    new_dims: Tuple[Union[int, str], ...],
) -> Optional[TensorShape]:
    """Infer the -1 dimension in a reshape/view operation.

    Given an input tensor shape and a target shape with exactly one -1,
    compute the concrete value for -1 and return the fully-resolved shape.
    Returns None if inference fails (incompatible shapes).
    """
    neg_one_count = sum(1 for d in new_dims if d == -1)
    if neg_one_count == 0:
        return TensorShape(tuple(ShapeDim(d) for d in new_dims))
    if neg_one_count > 1:
        return None

    neg_one_idx = next(i for i, d in enumerate(new_dims) if d == -1)

    total = _shape_total_elements(input_shape)
    specified = SymbolicProduct(concrete_factor=1)
    for i, d in enumerate(new_dims):
        if i == neg_one_idx:
            continue
        if isinstance(d, str):
            specified = specified * SymbolicProduct(symbolic_names=(d,))
        elif isinstance(d, int) and d > 0:
            specified = specified * SymbolicProduct(concrete_factor=d)
        elif isinstance(d, int) and d == 0 and i < input_shape.ndim:
            # Sentinel 0 = copy from input
            dim = input_shape.dims[i]
            if dim.is_symbolic:
                specified = specified * SymbolicProduct(symbolic_names=(dim.value,))
            else:
                specified = specified * SymbolicProduct(concrete_factor=dim.value)

    # Cancel common symbolic factors
    remaining_total_concrete = total.concrete_factor
    remaining_total_sym = list(total.symbolic_names)
    remaining_spec_sym = list(specified.symbolic_names)

    for sym in list(remaining_spec_sym):
        if sym in remaining_total_sym:
            remaining_total_sym.remove(sym)
            remaining_spec_sym.remove(sym)

    if remaining_spec_sym:
        # Can't fully resolve: remaining symbolic divisor
        inferred_name = f"_inferred_{neg_one_idx}"
        result_dims = []
        for i, d in enumerate(new_dims):
            if i == neg_one_idx:
                result_dims.append(ShapeDim(inferred_name))
            elif isinstance(d, int) and d == 0 and i < input_shape.ndim:
                result_dims.append(input_shape.dims[i])
            else:
                result_dims.append(ShapeDim(d))
        return TensorShape(tuple(result_dims))

    if specified.concrete_factor == 0:
        return None

    if remaining_total_concrete % specified.concrete_factor != 0:
        return None  # Not evenly divisible

    inferred_concrete = remaining_total_concrete // specified.concrete_factor

    if remaining_total_sym:
        # Result is symbolic * concrete
        inferred_name = "*".join(remaining_total_sym)
        if inferred_concrete != 1:
            inferred_name = f"{inferred_concrete}*{inferred_name}"
        inferred_dim = ShapeDim(inferred_name)
    else:
        inferred_dim = ShapeDim(inferred_concrete)

    result_dims = []
    for i, d in enumerate(new_dims):
        if i == neg_one_idx:
            result_dims.append(inferred_dim)
        elif isinstance(d, int) and d == 0 and i < input_shape.ndim:
            result_dims.append(input_shape.dims[i])
        else:
            result_dims.append(ShapeDim(d))

    return TensorShape(tuple(result_dims))


def encode_reshape_constraint_z3(
    input_shape_vars: List[Any],
    output_shape_vars: List[Any],
    neg_one_idx: int,
) -> Optional[Any]:
    """Encode reshape with -1 as a Z3 constraint.

    The constraint is: product(input_dims) == product(output_dims)
    where output_dims[neg_one_idx] is the free variable to solve for.

    Returns a Z3 BoolRef or None if Z3 is unavailable.
    """
    if not HAS_Z3:
        return None

    input_product = z3.IntVal(1)
    for v in input_shape_vars:
        input_product = input_product * v

    output_product = z3.IntVal(1)
    for v in output_shape_vars:
        output_product = output_product * v

    # Core constraint: total elements must be preserved
    preservation = input_product == output_product

    # All dimensions must be positive
    positivity = z3.And(*[v > 0 for v in input_shape_vars + output_shape_vars])

    return z3.And(preservation, positivity)


def encode_expand_constraint_z3(
    input_shape_vars: List[Any],
    expand_args: List[Union[int, Any]],
    output_shape_vars: List[Any],
) -> Optional[Any]:
    """Encode tensor.expand() constraints.

    In expand(), -1 means "keep the original dimension size".
    A dimension can only be expanded if the input dimension is 1.
    """
    if not HAS_Z3:
        return None

    constraints = []
    for i, arg in enumerate(expand_args):
        if i < len(input_shape_vars) and i < len(output_shape_vars):
            if isinstance(arg, int) and arg == -1:
                # -1 means keep original
                constraints.append(output_shape_vars[i] == input_shape_vars[i])
            else:
                out_v = output_shape_vars[i]
                in_v = input_shape_vars[i]
                if isinstance(arg, int):
                    constraints.append(out_v == arg)
                    # Can only expand from dim=1
                    constraints.append(z3.Or(in_v == 1, in_v == arg))
                else:
                    constraints.append(out_v == arg)
                    constraints.append(z3.Or(in_v == 1, in_v == arg))

    return z3.And(*constraints) if constraints else z3.BoolVal(True)


def infer_view_with_size_calls(
    input_shape: TensorShape,
    view_args: Tuple,
) -> Optional[TensorShape]:
    """Handle common pattern: x.view(x.size(0), -1).

    When view args reference tensor.size(dim), resolve them against
    the known input shape before applying -1 inference.
    """
    resolved = []
    for arg in view_args:
        if isinstance(arg, tuple) and len(arg) == 2 and arg[0] == "size_call":
            dim_idx = arg[1]
            if dim_idx < input_shape.ndim:
                dim = input_shape.dims[dim_idx]
                resolved.append(dim.value)
            else:
                resolved.append(arg)
        else:
            resolved.append(arg)

    return infer_neg_one_dim(input_shape, tuple(resolved))
