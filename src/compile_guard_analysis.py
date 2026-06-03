"""Compare TensorGuard constraints with ``torch.compile`` / Dynamo guards.

This module is intentionally diagnostic: it does not decide whether Dynamo's
guards are sound.  Instead it extracts the TensorGuard constraints that should be
visible at a compile boundary, normalizes Dynamo guard predicates, and reports
which static constraints have an equivalent or stronger runtime guard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.model_checker import ComputationGraph, LayerKind, OpKind, verify_model


@dataclass(frozen=True)
class TensorGuardSymbolicConstraint:
    """One TensorGuard fact that can be compared with a Dynamo guard."""

    kind: str
    expression: str
    source: str
    guard_keys: Tuple[str, ...] = ()
    guard_required: bool = True


@dataclass(frozen=True)
class DynamoGuard:
    """A normalized view of a Dynamo guard object or guard code string."""

    name: str = ""
    source: str = ""
    create_fn: str = ""
    code: Tuple[str, ...] = ()
    raw: str = ""


@dataclass(frozen=True)
class GuardConstraintMatch:
    """A TensorGuard constraint matched to one Dynamo predicate."""

    constraint: TensorGuardSymbolicConstraint
    guard: DynamoGuard
    key: str


@dataclass(frozen=True)
class CompileGuardIssue:
    """A mismatch or extraction issue in compile-guard comparison."""

    category: str
    message: str
    severity: str = "warning"
    constraint: Optional[TensorGuardSymbolicConstraint] = None


@dataclass(frozen=True)
class CompileGuardComparisonResult:
    """Result of comparing static TensorGuard facts to Dynamo guard code."""

    ok: bool
    input_shapes: Dict[str, Tuple[Any, ...]]
    tensorguard_constraints: Tuple[TensorGuardSymbolicConstraint, ...]
    dynamo_guards: Tuple[DynamoGuard, ...]
    matched_constraints: Tuple[GuardConstraintMatch, ...]
    issues: Tuple[CompileGuardIssue, ...] = ()
    dynamo_error: Optional[str] = None
    tensorguard_error: Optional[str] = None
    graph_count: int = 0
    graph_break_count: int = 0
    verification_safe: Optional[bool] = None

    @property
    def missing_constraints(self) -> Tuple[TensorGuardSymbolicConstraint, ...]:
        return tuple(
            issue.constraint
            for issue in self.issues
            if issue.category == "missing_dynamo_guard" and issue.constraint is not None
        )


_DIM_REF_RE = re.compile(
    r"(?P<expr>[LG]\[['\"](?P<name>[^'\"]+)['\"]\]\.(?:size\(\)|shape)\[(?P<axis>\d+)\])"
)
_RANK_METHOD_RE = re.compile(
    r"[LG]\[['\"](?P<name>[^'\"]+)['\"]\]\.(?:dim|ndimension)\(\)\s*==\s*(?P<rank>\d+)"
)
_RANK_LEN_RE = re.compile(
    r"len\([LG]\[['\"](?P<name>[^'\"]+)['\"]\]\.size\(\)\)\s*==\s*(?P<rank>\d+)"
)
_INT_PREFIX_RE = re.compile(r"(?P<value>-?\d+)\s*(?P<op>==|<=|>=|<|>)\s*$")
_OP_INT_RE = re.compile(r"^\s*(?P<op>==|<=|>=|<|>)\s*(?P<value>-?\d+)")
_OP_DIM_RE = re.compile(
    r"^\s*(?P<op>==)\s*(?P<expr>[LG]\[['\"][^'\"]+['\"]\]\.(?:size\(\)|shape)\[(?P<axis>\d+)\])"
)


def verify_compile_guard_interactions(
    model: Any,
    example_args: Sequence[Any],
    *,
    input_shapes: Optional[Dict[str, Tuple[Any, ...]]] = None,
    dynamo_guards: Optional[Iterable[Any]] = None,
    require_dynamo: bool = False,
) -> CompileGuardComparisonResult:
    """Compare TensorGuard symbolic facts with Dynamo guard predicates.

    ``example_args`` are the real tensors passed to Dynamo.  If ``input_shapes``
    is omitted, TensorGuard infers a shape contract from those examples using
    the same helper as the export/compile gates.  When ``dynamo_guards`` is not
    supplied, the function attempts to collect real guard objects from
    ``torch._dynamo.explain(model)(*example_args)``; unsupported interpreters are
    reported in ``dynamo_error`` unless ``require_dynamo=True``.
    """

    args = tuple(example_args)
    if input_shapes is None:
        from src.torch_integration import _infer_shapes_from_args

        input_shapes = _infer_shapes_from_args(model, args)
    normalized_shapes = {
        str(name): tuple(shape) for name, shape in (input_shapes or {}).items()
    }

    graph: Optional[ComputationGraph] = None
    verification_safe: Optional[bool] = None
    tensorguard_error: Optional[str] = None
    try:
        from src.torch_integration import module_source

        source = module_source(model)
        if source is None:
            tensorguard_error = (
                "could not recover nn.Module source for TensorGuard verification"
            )
        else:
            verification = verify_model(source, input_shapes=normalized_shapes)
            verification_safe = bool(getattr(verification, "safe", False))
            graph = getattr(verification, "graph", None)
            if graph is None:
                tensorguard_error = "TensorGuard verification produced no computation graph"
    except Exception as exc:
        tensorguard_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    constraints = _extract_tensorguard_constraints(graph, normalized_shapes)
    if tensorguard_error:
        constraints = tuple(constraints)

    if dynamo_guards is None:
        guards, dynamo_error, graph_count, graph_break_count = _collect_dynamo_guards(
            model, args
        )
        if dynamo_error and require_dynamo:
            raise RuntimeError(dynamo_error)
    else:
        guards = tuple(_coerce_dynamo_guard(guard) for guard in dynamo_guards)
        dynamo_error = None
        graph_count = 0
        graph_break_count = 0

    matches, issues = _compare_constraints_to_guards(constraints, guards)
    if dynamo_error:
        issues.append(
            CompileGuardIssue(
                category="dynamo_unavailable",
                severity="warning",
                message=dynamo_error,
            )
        )
    if tensorguard_error:
        issues.append(
            CompileGuardIssue(
                category="tensorguard_unavailable",
                severity="warning",
                message=tensorguard_error,
            )
        )

    ok = not any(issue.category == "missing_dynamo_guard" for issue in issues)
    ok = ok and dynamo_error is None and tensorguard_error is None
    return CompileGuardComparisonResult(
        ok=ok,
        input_shapes=normalized_shapes,
        tensorguard_constraints=tuple(constraints),
        dynamo_guards=tuple(guards),
        matched_constraints=tuple(matches),
        issues=tuple(issues),
        dynamo_error=dynamo_error,
        tensorguard_error=tensorguard_error,
        graph_count=graph_count,
        graph_break_count=graph_break_count,
        verification_safe=verification_safe,
    )


def _extract_tensorguard_constraints(
    graph: Optional[ComputationGraph],
    input_shapes: Dict[str, Tuple[Any, ...]],
) -> Tuple[TensorGuardSymbolicConstraint, ...]:
    constraints: List[TensorGuardSymbolicConstraint] = []
    symbol_axes: Dict[str, List[Tuple[str, int]]] = {}

    for name, shape in input_shapes.items():
        constraints.append(
            TensorGuardSymbolicConstraint(
                kind="input_rank",
                expression=f"rank({name}) == {len(shape)}",
                source=f"input:{name}",
                guard_keys=(f"rank:{name}:eq:{len(shape)}",),
            )
        )
        for axis, dim in enumerate(shape):
            if isinstance(dim, int):
                constraints.append(
                    TensorGuardSymbolicConstraint(
                        kind="input_dim_eq",
                        expression=f"{name}[{axis}] == {dim}",
                        source=f"input:{name}",
                        guard_keys=(f"dim:{name}:{axis}:eq:{dim}",),
                    )
                )
            elif isinstance(dim, str):
                constraints.append(
                    TensorGuardSymbolicConstraint(
                        kind="symbolic_dim_positive",
                        expression=f"{name}[{axis}] > 0  # {dim}",
                        source=f"input:{name}",
                        guard_keys=(f"dim:{name}:{axis}:ge:1",),
                    )
                )
                symbol_axes.setdefault(dim, []).append((name, axis))

    for symbol, axes in symbol_axes.items():
        for index, left in enumerate(axes):
            for right in axes[index + 1 :]:
                constraints.append(
                    TensorGuardSymbolicConstraint(
                        kind="symbolic_dim_equality",
                        expression=(
                            f"{left[0]}[{left[1]}] == {right[0]}[{right[1]}]  # {symbol}"
                        ),
                        source=f"symbol:{symbol}",
                        guard_keys=(_dim_eq_key(left[0], left[1], right[0], right[1]),),
                    )
                )

    if graph is not None:
        constraints.extend(_extract_layer_boundary_constraints(graph, input_shapes))

    return _dedupe_constraints(constraints)


def _extract_layer_boundary_constraints(
    graph: ComputationGraph,
    input_shapes: Dict[str, Tuple[Any, ...]],
) -> List[TensorGuardSymbolicConstraint]:
    constraints: List[TensorGuardSymbolicConstraint] = []
    input_names = set(input_shapes)
    for step in graph.steps:
        if step.op != OpKind.LAYER_CALL or not step.layer_ref or not step.inputs:
            continue
        layer = graph.layers.get(step.layer_ref)
        if layer is None:
            continue
        tensor_name = step.inputs[0]
        shape = input_shapes.get(tensor_name)
        direct_input = tensor_name in input_names and shape is not None
        source = f"layer:{layer.attr_name}"

        if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
            axis = len(shape) - 1 if direct_input else -1
            key_axis = axis if axis >= 0 else -1
            constraints.append(
                TensorGuardSymbolicConstraint(
                    kind="linear_in_features",
                    expression=f"{tensor_name}[{key_axis}] == {layer.in_features}",
                    source=source,
                    guard_keys=(
                        (f"dim:{tensor_name}:{axis}:eq:{layer.in_features}",)
                        if direct_input
                        else ()
                    ),
                    guard_required=direct_input,
                )
            )
        elif (
            layer.kind in {LayerKind.CONV1D, LayerKind.CONV2D, LayerKind.CONV3D}
            and layer.in_channels is not None
        ):
            constraints.append(
                TensorGuardSymbolicConstraint(
                    kind="conv_in_channels",
                    expression=f"{tensor_name}[1] == {layer.in_channels}",
                    source=source,
                    guard_keys=(
                        (f"dim:{tensor_name}:1:eq:{layer.in_channels}",)
                        if direct_input and len(shape) > 1
                        else ()
                    ),
                    guard_required=direct_input and len(shape) > 1,
                )
            )
        elif layer.kind in {
            LayerKind.BATCHNORM1D,
            LayerKind.BATCHNORM2D,
            LayerKind.BATCHNORM3D,
        } and layer.num_features is not None:
            constraints.append(
                TensorGuardSymbolicConstraint(
                    kind="batchnorm_num_features",
                    expression=f"{tensor_name}[1] == {layer.num_features}",
                    source=source,
                    guard_keys=(
                        (f"dim:{tensor_name}:1:eq:{layer.num_features}",)
                        if direct_input and len(shape) > 1
                        else ()
                    ),
                    guard_required=direct_input and len(shape) > 1,
                )
            )
        elif layer.kind == LayerKind.LAYERNORM:
            normalized_shape = layer.params.get("normalized_shape")
            if isinstance(normalized_shape, int):
                normalized = (normalized_shape,)
            elif isinstance(normalized_shape, (tuple, list)):
                normalized = tuple(v for v in normalized_shape if isinstance(v, int))
            else:
                normalized = ()
            if direct_input and normalized:
                for offset, expected in enumerate(reversed(normalized), start=1):
                    axis = len(shape) - offset
                    constraints.append(
                        TensorGuardSymbolicConstraint(
                            kind="layernorm_normalized_shape",
                            expression=f"{tensor_name}[{axis}] == {expected}",
                            source=source,
                            guard_keys=(f"dim:{tensor_name}:{axis}:eq:{expected}",),
                        )
                    )
    return constraints


def _dedupe_constraints(
    constraints: Sequence[TensorGuardSymbolicConstraint],
) -> Tuple[TensorGuardSymbolicConstraint, ...]:
    seen: Set[Tuple[str, str, Tuple[str, ...], bool]] = set()
    deduped: List[TensorGuardSymbolicConstraint] = []
    for constraint in constraints:
        key = (
            constraint.kind,
            constraint.expression,
            constraint.guard_keys,
            constraint.guard_required,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(constraint)
    return tuple(deduped)


def _collect_dynamo_guards(
    model: Any,
    args: Tuple[Any, ...],
) -> Tuple[Tuple[DynamoGuard, ...], Optional[str], int, int]:
    try:
        import torch._dynamo as dynamo

        dynamo.eval_frame.check_if_dynamo_supported()
        try:
            explain_out = dynamo.explain(model)(*args)
        except TypeError:
            explain_out = dynamo.explain(model, *args)
    except Exception as exc:
        return (
            (),
            f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            0,
            0,
        )

    raw_guards = getattr(explain_out, "out_guards", None)
    if raw_guards is None:
        raw_guards = getattr(explain_out, "guards", None) or ()
    guards = tuple(_coerce_dynamo_guard(guard) for guard in raw_guards)
    graph_count = int(getattr(explain_out, "graph_count", 0) or 0)
    graph_break_count = int(getattr(explain_out, "graph_break_count", 0) or 0)
    return guards, None, graph_count, graph_break_count


def _coerce_dynamo_guard(value: Any) -> DynamoGuard:
    if isinstance(value, DynamoGuard):
        return value
    if isinstance(value, str):
        return DynamoGuard(code=(value,), raw=value)
    if isinstance(value, dict):
        code = _normalize_code_list(value.get("code") or value.get("code_list"))
        return DynamoGuard(
            name=str(value.get("name") or ""),
            source=str(value.get("source") or ""),
            create_fn=str(value.get("create_fn") or ""),
            code=code,
            raw=str(value),
        )
    return DynamoGuard(
        name=str(getattr(value, "name", "") or ""),
        source=str(getattr(value, "source", "") or ""),
        create_fn=str(getattr(value, "create_fn", "") or ""),
        code=_normalize_code_list(getattr(value, "code_list", None)),
        raw=repr(value),
    )


def _normalize_code_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _compare_constraints_to_guards(
    constraints: Sequence[TensorGuardSymbolicConstraint],
    guards: Sequence[DynamoGuard],
) -> Tuple[List[GuardConstraintMatch], List[CompileGuardIssue]]:
    guard_index: Dict[str, List[DynamoGuard]] = {}
    for guard in guards:
        for key in _guard_keys(guard):
            guard_index.setdefault(key, []).append(guard)

    matches: List[GuardConstraintMatch] = []
    issues: List[CompileGuardIssue] = []
    for constraint in constraints:
        if not constraint.guard_keys:
            continue
        found = _find_supporting_guard(constraint.guard_keys, guard_index)
        if found is not None:
            key, guard = found
            matches.append(GuardConstraintMatch(constraint, guard, key))
        elif constraint.guard_required:
            issues.append(
                CompileGuardIssue(
                    category="missing_dynamo_guard",
                    message=(
                        "Dynamo guard set does not cover TensorGuard constraint "
                        f"{constraint.expression!r} from {constraint.source}"
                    ),
                    constraint=constraint,
                )
            )
    return matches, issues


def _guard_keys(guard: DynamoGuard) -> Set[str]:
    keys: Set[str] = set()
    for code in guard.code:
        keys.update(_guard_code_keys(code))
    return keys


def _guard_code_keys(code: str) -> Set[str]:
    keys: Set[str] = set()
    keys.update(_rank_keys(code))
    keys.update(_dim_keys(code))
    return keys


def _rank_keys(code: str) -> Set[str]:
    keys: Set[str] = set()
    for match in _RANK_METHOD_RE.finditer(code):
        keys.add(f"rank:{match.group('name')}:eq:{int(match.group('rank'))}")
    for match in _RANK_LEN_RE.finditer(code):
        keys.add(f"rank:{match.group('name')}:eq:{int(match.group('rank'))}")
    return keys


def _dim_keys(code: str) -> Set[str]:
    keys: Set[str] = set()
    dim_matches = list(_DIM_REF_RE.finditer(code))
    expr_to_ref = {
        match.group("expr"): (match.group("name"), int(match.group("axis")))
        for match in dim_matches
    }
    for match in dim_matches:
        name = match.group("name")
        axis = int(match.group("axis"))
        right = code[match.end() :]
        left = code[: match.start()]

        op_int = _OP_INT_RE.match(right)
        if op_int:
            keys.add(
                _dim_bound_key(name, axis, op_int.group("op"), int(op_int.group("value")))
            )

        op_dim = _OP_DIM_RE.match(right)
        if op_dim:
            rhs = expr_to_ref.get(op_dim.group("expr"))
            if rhs is not None:
                keys.add(_dim_eq_key(name, axis, rhs[0], rhs[1]))

        int_prefix = _INT_PREFIX_RE.search(left)
        if int_prefix:
            keys.add(
                _dim_bound_key(
                    name,
                    axis,
                    _reverse_op(int_prefix.group("op")),
                    int(int_prefix.group("value")),
                )
            )
    return keys


def _dim_bound_key(name: str, axis: int, op: str, value: int) -> str:
    if op == "==":
        return f"dim:{name}:{axis}:eq:{value}"
    if op == ">=":
        return f"dim:{name}:{axis}:ge:{value}"
    if op == ">":
        return f"dim:{name}:{axis}:ge:{value + 1}"
    if op == "<=":
        return f"dim:{name}:{axis}:le:{value}"
    if op == "<":
        return f"dim:{name}:{axis}:le:{value - 1}"
    return f"dim:{name}:{axis}:op:{op}:{value}"


def _reverse_op(op: str) -> str:
    return {
        "<=": ">=",
        "<": ">",
        ">=": "<=",
        ">": "<",
        "==": "==",
    }[op]


def _dim_eq_key(left_name: str, left_axis: int, right_name: str, right_axis: int) -> str:
    left = (left_name, int(left_axis))
    right = (right_name, int(right_axis))
    first, second = sorted((left, right))
    return f"dim_eq:{first[0]}:{first[1]}:{second[0]}:{second[1]}"


def _find_supporting_guard(
    required_keys: Sequence[str],
    guard_index: Dict[str, List[DynamoGuard]],
) -> Optional[Tuple[str, DynamoGuard]]:
    for key in required_keys:
        guards = guard_index.get(key)
        if guards:
            return key, guards[0]
        entailed = _find_entailed_dim_range(key, guard_index)
        if entailed is not None:
            return entailed
    return None


def _find_entailed_dim_range(
    key: str,
    guard_index: Dict[str, List[DynamoGuard]],
) -> Optional[Tuple[str, DynamoGuard]]:
    parsed = _parse_dim_range_key(key)
    if parsed is None:
        return None
    name, axis, op, value = parsed
    for guard_key, guards in guard_index.items():
        guard_parsed = _parse_dim_range_key(guard_key)
        if guard_parsed is None:
            continue
        guard_name, guard_axis, guard_op, guard_value = guard_parsed
        if guard_name != name or guard_axis != axis:
            continue
        if op == "ge":
            if guard_op == "ge" and guard_value >= value:
                return guard_key, guards[0]
            if guard_op == "eq" and guard_value >= value:
                return guard_key, guards[0]
        if op == "le":
            if guard_op == "le" and guard_value <= value:
                return guard_key, guards[0]
            if guard_op == "eq" and guard_value <= value:
                return guard_key, guards[0]
    return None


def _parse_dim_range_key(key: str) -> Optional[Tuple[str, int, str, int]]:
    parts = key.split(":")
    if len(parts) != 5 or parts[0] != "dim":
        return None
    name, axis_text, op, value_text = parts[1], parts[2], parts[3], parts[4]
    if op not in {"eq", "ge", "le"}:
        return None
    try:
        return name, int(axis_text), op, int(value_text)
    except ValueError:
        return None


__all__ = [
    "CompileGuardComparisonResult",
    "CompileGuardIssue",
    "DynamoGuard",
    "GuardConstraintMatch",
    "TensorGuardSymbolicConstraint",
    "verify_compile_guard_interactions",
]
