"""
Einsum Shape Inference and Z3 Constraint Encoding.

Parses Einstein summation notation (e.g. ``"bij,bjk->bik"``) used in
``torch.einsum``, extracts dimension constraints (shared subscripts must
match), and encodes them as Z3 formulas for static verification.

Supports:
  - Explicit output notation: ``"ij,jk->ik"``
  - Implicit output (sorted unique non-repeated subscripts)
  - Batch dimensions (repeated across all inputs)
  - Ellipsis broadcasting: ``"...ij,...jk->...ik"``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.tensor_shapes import ShapeDim, TensorShape


@dataclass
class EinsumParsed:
    """Parsed representation of an einsum expression."""

    input_subscripts: List[str]  # e.g. ["bij", "bjk"]
    output_subscripts: str  # e.g. "bik"
    has_ellipsis: bool = False

    # Derived analysis
    all_subscripts: Set[str] = field(default_factory=set)
    # Maps subscript char -> list of (input_idx, dim_idx) occurrences
    subscript_locations: Dict[str, List[Tuple[int, int]]] = field(
        default_factory=dict
    )
    # Subscripts that appear in output
    output_chars: List[str] = field(default_factory=list)
    # Subscripts that are summed over (in inputs but not output)
    contraction_chars: Set[str] = field(default_factory=set)


def parse_einsum(equation: str) -> EinsumParsed:
    """Parse an einsum equation string into structured form.

    Examples:
        >>> parse_einsum("bij,bjk->bik")
        >>> parse_einsum("ij,jk->ik")
        >>> parse_einsum("...ij,...jk->...ik")
    """
    equation = equation.replace(" ", "")
    has_ellipsis = "..." in equation

    if "->" in equation:
        inputs_str, output_str = equation.split("->", 1)
    else:
        inputs_str = equation
        output_str = None

    # Handle ellipsis by replacing with placeholder
    if has_ellipsis:
        inputs_str = inputs_str.replace("...", "")
        if output_str:
            output_str = output_str.replace("...", "")

    input_subscripts = inputs_str.split(",")

    # Determine output subscripts if not explicit
    if output_str is None:
        # Implicit: sorted unique subscripts that appear exactly once across all inputs
        all_chars: List[str] = []
        for inp in input_subscripts:
            all_chars.extend(inp)
        char_count: Dict[str, int] = {}
        for c in all_chars:
            char_count[c] = char_count.get(c, 0) + 1
        output_str = "".join(sorted(c for c, n in char_count.items() if n == 1))

    # Build subscript location map
    subscript_locations: Dict[str, List[Tuple[int, int]]] = {}
    all_subscripts: Set[str] = set()
    for inp_idx, inp in enumerate(input_subscripts):
        for dim_idx, c in enumerate(inp):
            all_subscripts.add(c)
            subscript_locations.setdefault(c, []).append((inp_idx, dim_idx))

    output_chars = list(output_str)
    contraction_chars = all_subscripts - set(output_chars)

    return EinsumParsed(
        input_subscripts=input_subscripts,
        output_subscripts=output_str,
        has_ellipsis=has_ellipsis,
        all_subscripts=all_subscripts,
        subscript_locations=subscript_locations,
        output_chars=output_chars,
        contraction_chars=contraction_chars,
    )


def infer_einsum_shape(
    equation: str,
    input_shapes: List[TensorShape],
) -> Optional[TensorShape]:
    """Infer the output shape of an einsum operation.

    Returns None if the input shapes are incompatible with the equation.
    """
    parsed = parse_einsum(equation)

    if len(input_shapes) != len(parsed.input_subscripts):
        return None

    # Build dimension map: subscript char -> resolved ShapeDim
    dim_map: Dict[str, ShapeDim] = {}

    for inp_idx, (subscript, shape) in enumerate(
        zip(parsed.input_subscripts, input_shapes)
    ):
        # Handle ellipsis: extra leading dims are batch dims
        effective_subscript = subscript
        ellipsis_dims = 0
        if parsed.has_ellipsis:
            ellipsis_dims = shape.ndim - len(subscript)
            if ellipsis_dims < 0:
                return None

        offset = ellipsis_dims
        for dim_idx, c in enumerate(effective_subscript):
            actual_dim_idx = dim_idx + offset
            if actual_dim_idx >= shape.ndim:
                return None

            shape_dim = shape.dims[actual_dim_idx]

            if c in dim_map:
                existing = dim_map[c]
                # Check compatibility
                if not existing.is_symbolic and not shape_dim.is_symbolic:
                    if existing.value != shape_dim.value:
                        return None  # Dimension mismatch
            else:
                dim_map[c] = shape_dim

    # Build output shape
    output_dims: List[ShapeDim] = []

    if parsed.has_ellipsis:
        # Add ellipsis (batch) dimensions from first input
        first_shape = input_shapes[0]
        n_explicit = len(parsed.input_subscripts[0])
        batch_ndim = first_shape.ndim - n_explicit
        for i in range(batch_ndim):
            output_dims.append(first_shape.dims[i])

    for c in parsed.output_chars:
        if c in dim_map:
            output_dims.append(dim_map[c])
        else:
            return None  # Output references unknown subscript

    return TensorShape(tuple(output_dims))


def check_einsum_compatible(
    equation: str,
    input_shapes: List[TensorShape],
) -> Optional[str]:
    """Check if an einsum operation is valid. Returns error message or None."""
    parsed = parse_einsum(equation)

    if len(input_shapes) != len(parsed.input_subscripts):
        return (
            f"einsum expects {len(parsed.input_subscripts)} inputs, "
            f"got {len(input_shapes)}"
        )

    for inp_idx, (subscript, shape) in enumerate(
        zip(parsed.input_subscripts, input_shapes)
    ):
        expected_ndim = len(subscript)
        if parsed.has_ellipsis:
            if shape.ndim < expected_ndim:
                return (
                    f"einsum input {inp_idx}: expected at least {expected_ndim} dims "
                    f"(plus ellipsis), got {shape.ndim}"
                )
        else:
            if shape.ndim != expected_ndim:
                return (
                    f"einsum input {inp_idx}: expected {expected_ndim} dims "
                    f"for subscript '{subscript}', got {shape.ndim}"
                )

    # Check shared subscript dimension matching
    dim_values: Dict[str, List[Tuple[int, ShapeDim]]] = {}
    for inp_idx, (subscript, shape) in enumerate(
        zip(parsed.input_subscripts, input_shapes)
    ):
        offset = shape.ndim - len(subscript) if parsed.has_ellipsis else 0
        for dim_idx, c in enumerate(subscript):
            actual_idx = dim_idx + offset
            if actual_idx < shape.ndim:
                dim_values.setdefault(c, []).append((inp_idx, shape.dims[actual_idx]))

    for c, occurrences in dim_values.items():
        concrete_vals = set()
        for inp_idx, dim in occurrences:
            if not dim.is_symbolic:
                concrete_vals.add(dim.value)
        if len(concrete_vals) > 1:
            return (
                f"einsum subscript '{c}' has mismatched dimensions: "
                f"{concrete_vals}"
            )

    return None


def encode_einsum_constraints_z3(
    equation: str,
    input_shape_vars: List[List[Any]],
    output_shape_vars: List[Any],
) -> Optional[Any]:
    """Encode einsum dimension constraints as Z3 formulas.

    For each shared subscript character, all corresponding dimensions
    must be equal. Output dimensions map to their subscript's value.

    Returns a Z3 BoolRef conjunction of all constraints, or None if Z3
    is unavailable.
    """
    if not HAS_Z3:
        return None

    parsed = parse_einsum(equation)
    constraints: List[Any] = []

    # Build subscript -> Z3 variable mapping
    subscript_vars: Dict[str, List[Any]] = {}
    for inp_idx, subscript in enumerate(parsed.input_subscripts):
        if inp_idx >= len(input_shape_vars):
            continue
        shape_vars = input_shape_vars[inp_idx]
        offset = len(shape_vars) - len(subscript) if parsed.has_ellipsis else 0
        for dim_idx, c in enumerate(subscript):
            actual_idx = dim_idx + offset
            if actual_idx < len(shape_vars):
                subscript_vars.setdefault(c, []).append(shape_vars[actual_idx])

    # Shared subscripts must have equal dimensions
    for c, var_list in subscript_vars.items():
        for i in range(1, len(var_list)):
            constraints.append(var_list[0] == var_list[i])

    # Output dimensions must match their subscript's resolved value
    out_offset = 0
    if parsed.has_ellipsis and input_shape_vars:
        first_vars = input_shape_vars[0]
        n_explicit = len(parsed.input_subscripts[0])
        batch_ndim = len(first_vars) - n_explicit
        for i in range(batch_ndim):
            if i < len(output_shape_vars):
                constraints.append(output_shape_vars[i] == first_vars[i])
        out_offset = batch_ndim

    for i, c in enumerate(parsed.output_chars):
        out_idx = i + out_offset
        if out_idx < len(output_shape_vars) and c in subscript_vars:
            constraints.append(output_shape_vars[out_idx] == subscript_vars[c][0])

    # All dimensions positive
    for var_list in input_shape_vars:
        for v in var_list:
            constraints.append(v > 0)
    for v in output_shape_vars:
        constraints.append(v > 0)

    return z3.And(*constraints) if constraints else z3.BoolVal(True)
