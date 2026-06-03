"""Static shape contracts for ``torch.func`` automatic-differentiation wrappers.

The functions here model the tensor-shape part of ``torch.func.grad``,
``jacrev``, ``jacfwd``, ``jvp``, and ``vjp`` without importing or executing
PyTorch.  They are intentionally shape-only and sound-by-abstention: if a
closure or branch can make the body output shape depend on runtime values, the
checker returns ``ok=True`` with ``unknown_reason`` rather than claiming an exact
contract.

Scope: each differentiated positional argument is a single tensor shape.  Pytree
outputs are supported because they are common for Jacobians and pullbacks; pytree
input arguments abstain until the main verifier carries argument-tree metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "Dim",
    "Shape",
    "FuncAutodiffVerdict",
    "verify_func_autodiff",
    "verify_func_grad",
    "verify_func_jacfwd",
    "verify_func_jacrev",
    "verify_func_jvp",
    "verify_func_vjp",
]


@dataclass(frozen=True)
class FuncAutodiffVerdict:
    """Result of one ``torch.func`` autodiff shape-contract check."""

    ok: bool
    output_shapes: Any = None
    body_output_shapes: Any = None
    pullback_input_shapes: Any = None
    pullback_output_shapes: Optional[Tuple[Shape, ...]] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    unknown_reason: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _fail(kind: str, message: str) -> FuncAutodiffVerdict:
    return FuncAutodiffVerdict(False, error=message, error_kind=kind)


def _ok(
    output_shapes: Any,
    *,
    body_output_shapes: Any = None,
    pullback_input_shapes: Any = None,
    pullback_output_shapes: Optional[Tuple[Shape, ...]] = None,
    unknown_reason: Optional[str] = None,
) -> FuncAutodiffVerdict:
    return FuncAutodiffVerdict(
        True,
        output_shapes=output_shapes,
        body_output_shapes=body_output_shapes,
        pullback_input_shapes=pullback_input_shapes,
        pullback_output_shapes=pullback_output_shapes,
        unknown_reason=unknown_reason,
    )


def _unknown(reason: str) -> FuncAutodiffVerdict:
    return FuncAutodiffVerdict(True, unknown_reason=reason)


def _is_dim(value: object) -> bool:
    return type(value) is int or isinstance(value, str)


def _is_shape(value: object) -> bool:
    return isinstance(value, (tuple, list)) and all(_is_dim(v) for v in value)


def _shape(value: Sequence[Dim]) -> Shape:
    return tuple(value)


def _has_negative_dim(shape: Shape) -> Optional[int]:
    for dim in shape:
        if type(dim) is int and dim < 0:
            return dim
    return None


def _normalise_input_shapes(
    input_shapes: Sequence[object],
) -> Tuple[Optional[Tuple[Shape, ...]], Optional[FuncAutodiffVerdict]]:
    shapes: List[Shape] = []
    for index, raw_shape in enumerate(input_shapes):
        if not _is_shape(raw_shape):
            return None, _unknown(
                f"input {index} is not a single tensor shape; pytree input "
                "arguments are outside this shape-only contract"
            )
        shape = _shape(raw_shape)  # type: ignore[arg-type]
        negative = _has_negative_dim(shape)
        if negative is not None:
            return None, _fail("negative_dim", f"input {index} contains negative dimension {negative}")
        shapes.append(shape)
    return tuple(shapes), None


def _normalise_output_tree(value: Any, label: str) -> Tuple[Any, Optional[FuncAutodiffVerdict]]:
    if _is_shape(value):
        shape = _shape(value)
        negative = _has_negative_dim(shape)
        if negative is not None:
            return None, _fail("negative_dim", f"{label} contains negative dimension {negative}")
        return shape, None
    if isinstance(value, tuple):
        children = []
        for child in value:
            normalised, err = _normalise_output_tree(child, label)
            if err is not None:
                return None, err
            children.append(normalised)
        return tuple(children), None
    if isinstance(value, list):
        children = []
        for child in value:
            normalised, err = _normalise_output_tree(child, label)
            if err is not None:
                return None, err
            children.append(normalised)
        return children, None
    if isinstance(value, dict):
        children = {}
        for key in sorted(value):
            normalised, err = _normalise_output_tree(value[key], label)
            if err is not None:
                return None, err
            children[key] = normalised
        return children, None
    return None, _fail("output_shape", f"unsupported {label} shape tree {value!r}")


def _normalise_argnums(
    argnums: object,
    n_inputs: int,
) -> Tuple[Optional[Tuple[int, ...]], bool, Optional[FuncAutodiffVerdict]]:
    argnums_is_int = type(argnums) is int
    if argnums_is_int:
        raw = (argnums,)
    elif isinstance(argnums, (tuple, list)):
        raw = tuple(argnums)
        if not raw:
            return None, False, _fail("argnums", "argnums must contain at least one argument index")
    else:
        return None, False, _fail("argnums", f"argnums must be an int or tuple/list of ints, got {argnums!r}")

    normalised: List[int] = []
    for value in raw:
        if type(value) is not int:
            return None, argnums_is_int, _fail("argnums", f"argnums entries must be ints, got {value!r}")
        index = value + n_inputs if value < 0 else value
        if index < 0 or index >= n_inputs:
            return None, argnums_is_int, _fail(
                "argnum_range",
                f"argnum {value} is out of range for {n_inputs} positional tensor arguments",
            )
        normalised.append(index)

    if len(set(normalised)) != len(normalised):
        return None, argnums_is_int, _fail(
            "argnums_duplicate",
            f"argnums elements must be unique after normalization, got {tuple(normalised)}",
        )
    return tuple(normalised), argnums_is_int, None


def _join_unknown(reasons: Sequence[str]) -> Optional[str]:
    unique = [reason for reason in dict.fromkeys(reasons) if reason]
    if not unique:
        return None
    return "; ".join(unique)


def _shape_match_unknown(
    expected: Shape,
    actual: Shape,
    *,
    label: str,
    kind: str,
) -> Tuple[Optional[FuncAutodiffVerdict], Optional[str]]:
    if len(expected) != len(actual):
        return _fail(
            f"{kind}_rank",
            f"{label} rank mismatch: expected rank {len(expected)}, got rank {len(actual)}",
        ), None
    unknowns = []
    for axis, (want, got) in enumerate(zip(expected, actual)):
        if type(want) is int and type(got) is int and want != got:
            return _fail(
                kind,
                f"{label} dimension {axis} mismatch: expected {want}, got {got}",
            ), None
        if want != got and not (type(want) is int and type(got) is int):
            unknowns.append(f"{label} dimension {axis} equality depends on symbolic dims {want!r} and {got!r}")
    return None, _join_unknown(unknowns)


def _tree_match_unknown(
    expected: Any,
    actual: Any,
    *,
    label: str,
    kind: str,
) -> Tuple[Optional[FuncAutodiffVerdict], Optional[str]]:
    if _is_shape(expected) and _is_shape(actual):
        return _shape_match_unknown(_shape(expected), _shape(actual), label=label, kind=kind)
    if _is_shape(expected) or _is_shape(actual):
        return _fail(f"{kind}_structure", f"{label} tree structure mismatch"), None

    unknowns = []
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(expected) != len(actual):
            return _fail(f"{kind}_structure", f"{label} tuple structure mismatch"), None
        for i, (want, got) in enumerate(zip(expected, actual)):
            err, unknown = _tree_match_unknown(want, got, label=f"{label}[{i}]", kind=kind)
            if err is not None:
                return err, None
            if unknown:
                unknowns.append(unknown)
        return None, _join_unknown(unknowns)

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return _fail(f"{kind}_structure", f"{label} list structure mismatch"), None
        for i, (want, got) in enumerate(zip(expected, actual)):
            err, unknown = _tree_match_unknown(want, got, label=f"{label}[{i}]", kind=kind)
            if err is not None:
                return err, None
            if unknown:
                unknowns.append(unknown)
        return None, _join_unknown(unknowns)

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return _fail(f"{kind}_structure", f"{label} dict keys mismatch"), None
        for key in sorted(expected):
            err, unknown = _tree_match_unknown(
                expected[key],
                actual[key],
                label=f"{label}[{key!r}]",
                kind=kind,
            )
            if err is not None:
                return err, None
            if unknown:
                unknowns.append(unknown)
        return None, _join_unknown(unknowns)

    return _fail(f"{kind}_structure", f"{label} tree structure mismatch"), None


def _map_shape_tree(value: Any, leaf_fn: Any) -> Any:
    if _is_shape(value):
        return leaf_fn(_shape(value))
    if isinstance(value, tuple):
        return tuple(_map_shape_tree(child, leaf_fn) for child in value)
    if isinstance(value, list):
        return [_map_shape_tree(child, leaf_fn) for child in value]
    if isinstance(value, dict):
        return {key: _map_shape_tree(value[key], leaf_fn) for key in sorted(value)}
    raise TypeError(f"unsupported shape tree {value!r}")  # guarded by normalisation


def _prepare_common(
    input_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    value_dependent: bool,
) -> Tuple[Optional[Tuple[Shape, ...]], Any, Optional[FuncAutodiffVerdict]]:
    if value_dependent:
        return None, None, _unknown(
            "body output shape may depend on runtime values captured by the closure; "
            "TensorGuard abstains instead of claiming an autodiff contract"
        )

    inputs, err = _normalise_input_shapes(input_shapes)
    if err is not None:
        return None, None, err
    assert inputs is not None

    body_outputs, err = _normalise_output_tree(body_output_shapes, "body output")
    if err is not None:
        return None, None, err
    return inputs, body_outputs, None


def verify_func_grad(
    input_shapes: Sequence[object],
    body_output_shape: Sequence[Dim],
    *,
    argnums: object = 0,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    """Verify the tensor-shape contract of ``torch.func.grad``.

    ``body_output_shape`` is the differentiable function's primary output shape;
    PyTorch requires it to be a rank-0 scalar tensor, not merely a size-one
    tensor.  ``argnums=(0,)`` deliberately returns a one-tuple of gradient shapes,
    matching PyTorch's sequence-argnums convention.
    """

    inputs, body_output, err = _prepare_common(
        input_shapes,
        body_output_shape,
        value_dependent=value_dependent,
    )
    if err is not None:
        return err
    assert inputs is not None

    selected, argnums_is_int, err = _normalise_argnums(argnums, len(inputs))
    if err is not None:
        return err
    assert selected is not None

    if not _is_shape(body_output) or _shape(body_output) != ():
        return _fail(
            "scalar_output",
            "torch.func.grad expects the function to return a rank-0 scalar tensor",
        )

    grad_shapes: Any
    if argnums_is_int:
        grad_shapes = inputs[selected[0]]
    else:
        grad_shapes = tuple(inputs[index] for index in selected)
    if has_aux:
        return _ok((grad_shapes, aux_shapes), body_output_shapes=body_output)
    return _ok(grad_shapes, body_output_shapes=body_output)


def _verify_func_jacobian(
    input_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    argnums: object = 0,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    inputs, body_outputs, err = _prepare_common(
        input_shapes,
        body_output_shapes,
        value_dependent=value_dependent,
    )
    if err is not None:
        return err
    assert inputs is not None

    selected, argnums_is_int, err = _normalise_argnums(argnums, len(inputs))
    if err is not None:
        return err
    assert selected is not None
    selected_shapes = tuple(inputs[index] for index in selected)

    def jac_leaf(output_shape: Shape) -> Any:
        if argnums_is_int:
            return output_shape + selected_shapes[0]
        return tuple(output_shape + input_shape for input_shape in selected_shapes)

    jac_shapes = _map_shape_tree(body_outputs, jac_leaf)
    if has_aux:
        return _ok((jac_shapes, aux_shapes), body_output_shapes=body_outputs)
    return _ok(jac_shapes, body_output_shapes=body_outputs)


def verify_func_jacrev(
    input_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    argnums: object = 0,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    """Verify the output-shape contract of ``torch.func.jacrev``."""

    return _verify_func_jacobian(
        input_shapes,
        body_output_shapes,
        argnums=argnums,
        has_aux=has_aux,
        aux_shapes=aux_shapes,
        value_dependent=value_dependent,
    )


def verify_func_jacfwd(
    input_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    argnums: object = 0,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    """Verify the output-shape contract of ``torch.func.jacfwd``."""

    return _verify_func_jacobian(
        input_shapes,
        body_output_shapes,
        argnums=argnums,
        has_aux=has_aux,
        aux_shapes=aux_shapes,
        value_dependent=value_dependent,
    )


def verify_func_jvp(
    primal_shapes: Sequence[object],
    tangent_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    """Verify ``torch.func.jvp`` tangent and output-shape contracts."""

    primals, body_outputs, err = _prepare_common(
        primal_shapes,
        body_output_shapes,
        value_dependent=value_dependent,
    )
    if err is not None:
        return err
    assert primals is not None

    tangents, err = _normalise_input_shapes(tangent_shapes)
    if err is not None:
        return err
    assert tangents is not None
    if len(primals) != len(tangents):
        return _fail(
            "tangent_structure",
            f"jvp expects {len(primals)} tangent shapes, got {len(tangents)}",
        )

    unknowns = []
    for index, (primal, tangent) in enumerate(zip(primals, tangents)):
        shape_err, unknown = _shape_match_unknown(
            primal,
            tangent,
            label=f"tangent {index}",
            kind="tangent_shape",
        )
        if shape_err is not None:
            return shape_err
        if unknown:
            unknowns.append(unknown)

    unknown_reason = _join_unknown(unknowns)
    if has_aux:
        return _ok(
            (body_outputs, body_outputs, aux_shapes),
            body_output_shapes=body_outputs,
            unknown_reason=unknown_reason,
        )
    return _ok(
        (body_outputs, body_outputs),
        body_output_shapes=body_outputs,
        unknown_reason=unknown_reason,
    )


def verify_func_vjp(
    primal_shapes: Sequence[object],
    body_output_shapes: Any,
    *,
    cotangent_shapes: Any = None,
    has_aux: bool = False,
    aux_shapes: Any = None,
    value_dependent: bool = False,
) -> FuncAutodiffVerdict:
    """Verify ``torch.func.vjp`` output and pullback shape contracts.

    The pullback returned by PyTorch always returns a tuple of gradients, one per
    primal positional argument, even when there is only one primal.
    """

    primals, body_outputs, err = _prepare_common(
        primal_shapes,
        body_output_shapes,
        value_dependent=value_dependent,
    )
    if err is not None:
        return err
    assert primals is not None

    expected_cotangents = body_outputs
    unknown_reason = None
    if cotangent_shapes is not None:
        actual_cotangents, err = _normalise_output_tree(cotangent_shapes, "cotangent")
        if err is not None:
            return err
        tree_err, unknown_reason = _tree_match_unknown(
            expected_cotangents,
            actual_cotangents,
            label="cotangent",
            kind="cotangent_shape",
        )
        if tree_err is not None:
            return tree_err

    tensor_output_shapes = (body_outputs, aux_shapes) if has_aux else body_outputs
    return _ok(
        tensor_output_shapes,
        body_output_shapes=body_outputs,
        pullback_input_shapes=expected_cotangents,
        pullback_output_shapes=tuple(primals),
        unknown_reason=unknown_reason,
    )


def verify_func_autodiff(
    op: str,
    input_shapes: Sequence[object],
    body_output_shapes: Any,
    **kwargs: object,
) -> FuncAutodiffVerdict:
    """Dispatch to a specific ``torch.func`` autodiff shape-contract checker."""

    if op == "grad":
        return verify_func_grad(input_shapes, body_output_shapes, **kwargs)
    if op == "jacrev":
        return verify_func_jacrev(input_shapes, body_output_shapes, **kwargs)
    if op == "jacfwd":
        return verify_func_jacfwd(input_shapes, body_output_shapes, **kwargs)
    if op == "jvp":
        if "tangent_shapes" not in kwargs:
            return _fail("argument", "verify_func_autodiff('jvp', ...) requires tangent_shapes=")
        tangent_shapes = kwargs.pop("tangent_shapes")
        return verify_func_jvp(input_shapes, tangent_shapes, body_output_shapes, **kwargs)
    if op == "vjp":
        return verify_func_vjp(input_shapes, body_output_shapes, **kwargs)
    return _fail("op", f"unsupported torch.func autodiff op {op!r}")
