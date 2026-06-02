"""
Einsum shape inference and Z3 constraint encoding (torch-equivalent).

Parses Einstein summation notation (e.g. ``"bij,bjk->bik"``) used in
``torch.einsum``, resolves dimension labels, and infers the output shape with
the *exact* PyTorch/NumPy semantics — including:

  - Explicit output notation: ``"ij,jk->ik"``
  - Implicit output (ellipsis first, then sorted labels appearing exactly once)
  - Diagonals / repeated input labels: ``"ii->i"``
  - Ellipsis broadcasting across operands: ``"...ij,...jk->...ik"`` where the
    ellipsis block is broadcast (NumPy rules: right-aligned, size-1 expands) and
    placed wherever ``...`` appears in the (possibly implicit) output, and is
    *reduced away* when the explicit output omits ``...``.

It also rejects malformed/invalid equations (multiple ``->``, non-letter
labels, repeated output labels, output labels absent from the inputs, output
ellipsis without any input ellipsis), so the verifier can flag them as bugs
rather than silently accepting them.

`TensorShape`/`ShapeDim` are the engine's own classes (`src.tensor_shapes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import ascii_letters
from typing import Any, Dict, List, Optional, Tuple

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.tensor_shapes import ShapeDim, TensorShape

_ELLIPSIS = "..."
_VALID_LABELS = set(ascii_letters)


@dataclass
class EinsumOperand:
    """One input/output operand: labels before/after the ellipsis block."""

    pre: str
    post: str
    has_ellipsis: bool

    @property
    def labels(self) -> str:
        return self.pre + self.post


@dataclass
class EinsumParsed:
    """Parsed representation of an einsum expression."""

    operands: List[EinsumOperand]
    out: EinsumOperand
    implicit_output: bool

    # Backward-compatible derived views (used by the Z3 encoder).
    input_subscripts: List[str] = field(default_factory=list)
    output_subscripts: str = ""
    has_ellipsis: bool = False
    output_chars: List[str] = field(default_factory=list)


def _split_ellipsis(spec: str) -> EinsumOperand:
    """Split a single operand spec around its (optional) single ellipsis."""
    if spec.count(_ELLIPSIS) > 1 or spec.replace(_ELLIPSIS, "").count(".") > 0:
        raise ValueError("einsum: malformed ellipsis in %r" % spec)
    if _ELLIPSIS in spec:
        pre, post = spec.split(_ELLIPSIS, 1)
        has_ell = True
    else:
        pre, post, has_ell = spec, "", False
    for ch in pre + post:
        if ch not in _VALID_LABELS:
            raise ValueError("einsum: invalid label %r in %r" % (ch, spec))
    return EinsumOperand(pre=pre, post=post, has_ellipsis=has_ell)


def parse_einsum(equation: str) -> EinsumParsed:
    """Parse an einsum equation string into structured form.

    Raises ``ValueError`` for malformed equations.
    """
    equation = equation.replace(" ", "")
    if equation.count("->") > 1:
        raise ValueError("einsum: more than one '->' in %r" % equation)

    if "->" in equation:
        inputs_str, output_str = equation.split("->", 1)
        implicit = False
    else:
        inputs_str, output_str, implicit = equation, None, True

    operands = [_split_ellipsis(s) for s in inputs_str.split(",")]
    any_ell = any(op.has_ellipsis for op in operands)

    if implicit:
        # Implicit output: ellipsis (if any) first, then labels appearing
        # exactly once across all inputs, sorted alphabetically.
        counts: Dict[str, int] = {}
        for op in operands:
            for ch in op.labels:
                counts[ch] = counts.get(ch, 0) + 1
        sorted_once = "".join(sorted(c for c, n in counts.items() if n == 1))
        out = EinsumOperand(pre="", post=sorted_once, has_ellipsis=any_ell)
    else:
        out = _split_ellipsis(output_str)
        # Validation: no repeated output labels.
        if len(set(out.labels)) != len(out.labels):
            raise ValueError("einsum: repeated label in output %r" % output_str)
        # Validation: every output label must appear in some input.
        input_labels = set().union(*(set(op.labels) for op in operands)) \
            if operands else set()
        for ch in out.labels:
            if ch not in input_labels:
                raise ValueError(
                    "einsum: output label %r not present in inputs" % ch)
        # Validation: output ellipsis requires some input ellipsis.
        if out.has_ellipsis and not any_ell:
            raise ValueError("einsum: output '...' but no input has ellipsis")

    return EinsumParsed(
        operands=operands,
        out=out,
        implicit_output=implicit,
        input_subscripts=[op.labels for op in operands],
        output_subscripts=out.labels,
        has_ellipsis=any_ell,
        output_chars=list(out.labels),
    )


def _broadcast_dim(a: ShapeDim, b: ShapeDim) -> Optional[ShapeDim]:
    """NumPy-broadcast two ellipsis dims. None ⇒ concrete incompatibility."""
    if not a.is_symbolic and not b.is_symbolic:
        if a.value == b.value:
            return a
        if a.value == 1:
            return b
        if b.value == 1:
            return a
        return None
    # At least one symbolic.
    if not a.is_symbolic and a.value == 1:
        return b
    if not b.is_symbolic and b.value == 1:
        return a
    if not a.is_symbolic:
        return a
    if not b.is_symbolic:
        return b
    return a if a.value == b.value else ShapeDim("_bcast")


def _broadcast_ellipsis(
    slices: List[List[ShapeDim]],
) -> Tuple[Optional[List[ShapeDim]], bool]:
    """Right-aligned broadcast of per-operand ellipsis blocks.

    Returns ``(dims, ok)``; ``ok`` is False on a concrete incompatibility.
    """
    maxlen = max((len(s) for s in slices), default=0)
    result: List[Optional[ShapeDim]] = [None] * maxlen
    for s in slices:
        offset = maxlen - len(s)
        for i, d in enumerate(s):
            pos = offset + i
            cur = result[pos]
            if cur is None:
                result[pos] = d
            else:
                merged = _broadcast_dim(cur, d)
                if merged is None:
                    return None, False
                result[pos] = merged
    return [d for d in result if d is not None], True


def _resolve_operands(
    parsed: EinsumParsed,
    input_shapes: List[TensorShape],
) -> Tuple[Optional[Dict[str, ShapeDim]], Optional[List[ShapeDim]], Optional[str]]:
    """Resolve named labels + broadcasted ellipsis.

    Returns ``(label_map, ellipsis_dims, error)``.
    """
    if len(input_shapes) != len(parsed.operands):
        return None, None, (
            "einsum expects %d inputs, got %d"
            % (len(parsed.operands), len(input_shapes)))

    label_map: Dict[str, ShapeDim] = {}
    ellipsis_slices: List[List[ShapeDim]] = []

    for idx, (op, shape) in enumerate(zip(parsed.operands, input_shapes)):
        n_named = len(op.pre) + len(op.post)
        if op.has_ellipsis:
            if shape.ndim < n_named:
                return None, None, (
                    "einsum input %d: expected at least %d dims, got %d"
                    % (idx, n_named, shape.ndim))
            ell_count = shape.ndim - n_named
            named_dims = (list(shape.dims[:len(op.pre)])
                          + list(shape.dims[shape.ndim - len(op.post):]
                                 if op.post else []))
            ellipsis_slices.append(
                list(shape.dims[len(op.pre):len(op.pre) + ell_count]))
        else:
            if shape.ndim != n_named:
                return None, None, (
                    "einsum input %d: expected %d dims for subscript %r, got %d"
                    % (idx, n_named, op.labels, shape.ndim))
            named_dims = list(shape.dims)

        for ch, dim in zip(op.labels, named_dims):
            if ch in label_map:
                prev = label_map[ch]
                if (not prev.is_symbolic and not dim.is_symbolic
                        and prev.value != dim.value):
                    return None, None, (
                        "einsum subscript %r has mismatched dimensions: "
                        "%s vs %s" % (ch, prev.value, dim.value))
                if prev.is_symbolic and not dim.is_symbolic:
                    label_map[ch] = dim  # prefer the concrete witness
            else:
                label_map[ch] = dim

    ellipsis_dims: List[ShapeDim] = []
    if ellipsis_slices:
        bc, ok = _broadcast_ellipsis(ellipsis_slices)
        if not ok:
            return None, None, "einsum: ellipsis dimensions are not broadcastable"
        ellipsis_dims = bc or []
    return label_map, ellipsis_dims, None


def infer_einsum_shape(
    equation: str,
    input_shapes: List[TensorShape],
) -> Optional[TensorShape]:
    """Infer the output shape of an einsum operation, or None if invalid."""
    try:
        parsed = parse_einsum(equation)
    except ValueError:
        return None

    label_map, ellipsis_dims, err = _resolve_operands(parsed, input_shapes)
    if err is not None or label_map is None:
        return None

    out_dims: List[ShapeDim] = []
    for ch in parsed.out.pre:
        if ch not in label_map:
            return None
        out_dims.append(label_map[ch])
    if parsed.out.has_ellipsis:
        out_dims.extend(ellipsis_dims or [])
    for ch in parsed.out.post:
        if ch not in label_map:
            return None
        out_dims.append(label_map[ch])
    return TensorShape(tuple(out_dims))


def check_einsum_compatible(
    equation: str,
    input_shapes: List[TensorShape],
) -> Optional[str]:
    """Validate an einsum operation. Returns an error message, or None."""
    try:
        parsed = parse_einsum(equation)
    except ValueError as exc:
        return str(exc)
    _, _, err = _resolve_operands(parsed, input_shapes)
    return err


def encode_einsum_constraints_z3(
    equation: str,
    input_shape_vars: List[List[Any]],
    output_shape_vars: List[Any],
) -> Optional[Any]:
    """Encode einsum dimension constraints as Z3 formulas.

    For each shared subscript character (including repeated labels within one
    operand), all corresponding dimensions must be equal. Ellipsis blocks are
    right-aligned and broadcast; output dimensions map to their label or to the
    broadcasted ellipsis position wherever ``...`` appears in the output.
    Returns a Z3 BoolRef, or None if Z3 is unavailable.
    """
    if not HAS_Z3:
        return None
    try:
        parsed = parse_einsum(equation)
    except ValueError:
        return None

    if len(input_shape_vars) != len(parsed.operands):
        return z3.BoolVal(False)

    constraints: List[Any] = []
    subscript_vars: Dict[str, List[Any]] = {}
    ellipsis_slices: List[List[Any]] = []
    for inp_idx, op in enumerate(parsed.operands):
        shape_vars = input_shape_vars[inp_idx]
        n_pre, n_post = len(op.pre), len(op.post)
        n_named = n_pre + n_post
        if op.has_ellipsis:
            if len(shape_vars) < n_named:
                return z3.BoolVal(False)
            ell_end = len(shape_vars) - n_post if n_post else len(shape_vars)
            named_vars = (
                list(shape_vars[:n_pre])
                + (list(shape_vars[ell_end:]) if n_post else [])
            )
            ellipsis_slices.append(list(shape_vars[n_pre:ell_end]))
        else:
            if len(shape_vars) != n_named:
                return z3.BoolVal(False)
            named_vars = list(shape_vars)
        for ch, v in zip(op.labels, named_vars):
            subscript_vars.setdefault(ch, []).append(v)

    for var_list in subscript_vars.values():
        for i in range(1, len(var_list)):
            constraints.append(var_list[0] == var_list[i])

    ellipsis_len = max((len(s) for s in ellipsis_slices), default=0)
    expected_output_rank = (
        len(parsed.out.pre)
        + (ellipsis_len if parsed.out.has_ellipsis else 0)
        + len(parsed.out.post)
    )
    if len(output_shape_vars) != expected_output_rank:
        return z3.BoolVal(False)

    out_idx = 0
    for ch in parsed.out.pre:
        if ch in subscript_vars:
            constraints.append(output_shape_vars[out_idx] == subscript_vars[ch][0])
        out_idx += 1

    if parsed.out.has_ellipsis:
        for ell_pos in range(ellipsis_len):
            out_var = output_shape_vars[out_idx + ell_pos]
            aligned_inputs = []
            for ell in ellipsis_slices:
                offset = ellipsis_len - len(ell)
                local = ell_pos - offset
                if 0 <= local < len(ell):
                    aligned_inputs.append(ell[local])
            if aligned_inputs:
                constraints.append(z3.Or(*[out_var == v for v in aligned_inputs]))
                for v in aligned_inputs:
                    constraints.append(z3.Or(v == out_var, v == 1))
        out_idx += ellipsis_len

    for ch in parsed.out.post:
        if ch in subscript_vars:
            constraints.append(output_shape_vars[out_idx] == subscript_vars[ch][0])
        out_idx += 1

    for var_list in input_shape_vars:
        for v in var_list:
            constraints.append(v > 0)
    for v in output_shape_vars:
        constraints.append(v > 0)

    return z3.And(*constraints) if constraints else z3.BoolVal(True)
