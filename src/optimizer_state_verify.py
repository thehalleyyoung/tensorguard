"""Optimizer-state compatibility checks for training resume and sharding gates.

TensorGuard's forward verifier proves architectural tensor contracts. Optimizer
state is another high-value resume boundary: stale Adam moments or factored
Adafactor buffers with the wrong shape can fail only after a checkpoint is
loaded into a long-running training job. This module validates real
``torch.optim`` state dictionaries without stepping the optimizer.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class OptimizerStateIssue:
    """One actionable optimizer-state compatibility finding."""

    category: str
    message: str
    param_name: Optional[str] = None
    state_key: Optional[str] = None
    expected_shape: Optional[Shape] = None
    actual_shape: Optional[Shape] = None
    expected_dtype: Optional[str] = None
    actual_dtype: Optional[str] = None
    shard_index: Optional[int] = None
    severity: str = "error"


@dataclass(frozen=True)
class OptimizerStateShard:
    """A declared local shard of a per-parameter optimizer-state tensor.

    ``start`` and ``length`` describe a contiguous slice along ``dim`` of the
    logical full state tensor for ``param_name`` and ``state_key``.
    """

    param_name: str
    state_key: str
    shape: Shape
    dtype: Any
    start: int
    length: int
    dim: int = 0
    shard_index: Optional[int] = None


@dataclass(frozen=True)
class OptimizerStateVerificationResult:
    """Result of TensorGuard's optimizer-state gate."""

    ok: bool
    issues: Tuple[OptimizerStateIssue, ...]
    warnings: Tuple[OptimizerStateIssue, ...] = ()
    checked_states: Tuple[str, ...] = ()
    optimizer_name: str = "optimizer"


class TensorGuardOptimizerStateError(ValueError):
    """Raised when optimizer state is incompatible with the target model."""

    def __init__(self, issues: Sequence[OptimizerStateIssue]):
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected optimizer state with "
            f"{len(self.issues)} issue(s): {details}{more}"
        )


@dataclass(frozen=True)
class _StateEntry:
    param_name: str
    param: Any
    state: Mapping[str, Any]
    group_index: int
    param_index: int
    saved_param_id: Any = None


_ADAM_SHAPED_KEYS = {
    "exp_avg",
    "exp_avg_sq",
    "max_exp_avg_sq",
    "momentum_buffer",
    "fp32_from_fp16_params",
    "master_param",
}
_ADFACTOR_ROW_KEYS = {"row_var", "exp_avg_sq_row"}
_ADFACTOR_COL_KEYS = {"col_var", "exp_avg_sq_col"}
_ADFACTOR_FULL_KEYS = {"variance", "exp_avg_sq", "exp_avg"}
_SCALAR_KEYS = {"step"}
_FLOAT32_DTYPES = {"torch.float32", "float32", "single"}
_LOW_PRECISION_DTYPES = {
    "torch.float16",
    "float16",
    "half",
    "torch.bfloat16",
    "bfloat16",
}


def verify_optimizer_state(
    model: Any,
    optimizer_or_state: Any,
    *,
    optimizer_name: Optional[str] = None,
    allow_lazy: bool = True,
    sharded_state: Sequence[OptimizerStateShard] = (),
    check_dtype: bool = True,
) -> OptimizerStateVerificationResult:
    """Validate optimizer state against a target model's parameters.

    ``optimizer_or_state`` may be a live ``torch.optim.Optimizer`` or a raw
    ``optimizer.state_dict()``-style mapping. Live optimizers are mapped by
    ``Parameter`` identity through their own param groups, so frozen or
    intentionally unoptimized model parameters are not considered missing state.
    Raw mappings fall back to PyTorch's positional state-dict convention.
    """

    named_params = _named_parameters(model)
    issues: List[OptimizerStateIssue] = []
    warn: List[OptimizerStateIssue] = []
    checked: List[str] = []

    if _is_live_optimizer(optimizer_or_state):
        opt_name = optimizer_name or type(optimizer_or_state).__name__
        entries = _entries_from_live_optimizer(model, optimizer_or_state)
    elif isinstance(optimizer_or_state, Mapping):
        opt_name = optimizer_name or str(optimizer_or_state.get("optimizer_name", "optimizer"))
        entries = _entries_from_state_dict(named_params, optimizer_or_state, issues, warn)
    else:
        raise TypeError(
            "optimizer_or_state must be a torch.optim.Optimizer-like object "
            "or an optimizer.state_dict() mapping"
        )

    for entry in entries:
        if not entry.state:
            target = warn if allow_lazy else issues
            target.append(
                OptimizerStateIssue(
                    category="lazy_uninitialized",
                    param_name=entry.param_name,
                    message=(
                        f"{entry.param_name}: optimizer state is not initialized yet; "
                        "this is valid before the first optimizer.step()"
                    ),
                    severity="warning" if allow_lazy else "error",
                )
            )
            continue
        _check_state_entry(
            entry,
            optimizer_name=opt_name,
            issues=issues,
            warnings_out=warn,
            checked=checked,
            check_dtype=check_dtype,
        )

    _check_sharded_state(named_params, sharded_state, issues, warn, checked, check_dtype)

    return OptimizerStateVerificationResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_states=tuple(dict.fromkeys(checked)),
        optimizer_name=opt_name,
    )


def guarded_optimizer_load_state_dict(
    model: Any,
    optimizer: Any,
    state_dict: Mapping[str, Any],
    *,
    on_violation: str = "raise",
    allow_lazy: bool = True,
) -> OptimizerStateVerificationResult:
    """Verify a checkpointed optimizer state before and after loading it.

    The pre-load gate blocks shape and param-group incompatibilities. Dtype
    mismatches are intentionally not blocking at this boundary because PyTorch
    casts floating optimizer state to the target parameter dtype during
    ``load_state_dict``.
    """

    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")

    pre = _verify_state_dict_for_load(
        model,
        optimizer,
        state_dict,
        allow_lazy=allow_lazy,
    )
    _handle_result(pre, on_violation)
    if pre.issues and on_violation != "ignore":
        return pre

    optimizer.load_state_dict(state_dict)
    post = verify_optimizer_state(
        model,
        optimizer,
        allow_lazy=allow_lazy,
        check_dtype=True,
    )
    _handle_result(post, on_violation)
    return post


def _handle_result(result: OptimizerStateVerificationResult, on_violation: str) -> None:
    if result.ok or on_violation == "ignore":
        return
    if on_violation == "warn":
        warnings.warn(
            str(TensorGuardOptimizerStateError(result.issues)),
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise TensorGuardOptimizerStateError(result.issues)


def _verify_state_dict_for_load(
    model: Any,
    optimizer: Any,
    state_dict: Mapping[str, Any],
    *,
    allow_lazy: bool,
) -> OptimizerStateVerificationResult:
    named_params = _named_parameters(model)
    name_by_id = {id(param): name for name, param in named_params}
    issues: List[OptimizerStateIssue] = []
    warn: List[OptimizerStateIssue] = []
    checked: List[str] = []
    entries: List[_StateEntry] = []

    saved_groups = list(state_dict.get("param_groups", ()))
    target_groups = list(getattr(optimizer, "param_groups", ()))
    if len(saved_groups) != len(target_groups):
        issues.append(
            OptimizerStateIssue(
                category="param_group_count_mismatch",
                message=(
                    f"checkpoint has {len(saved_groups)} optimizer param group(s), "
                    f"target optimizer has {len(target_groups)}"
                ),
            )
        )

    state = state_dict.get("state", {})
    for group_index, (saved_group, target_group) in enumerate(zip(saved_groups, target_groups)):
        saved_ids = list(saved_group.get("params", ()))
        target_params = list(target_group.get("params", ()))
        if len(saved_ids) != len(target_params):
            issues.append(
                OptimizerStateIssue(
                    category="param_count_mismatch",
                    message=(
                        f"param group {group_index}: checkpoint has {len(saved_ids)} "
                        f"parameter state slot(s), target optimizer has {len(target_params)}"
                    ),
                )
            )
        for param_index, (saved_id, param) in enumerate(zip(saved_ids, target_params)):
            entries.append(
                _StateEntry(
                    param_name=name_by_id.get(id(param), f"group{group_index}.param{param_index}"),
                    param=param,
                    state=state.get(saved_id, {}),
                    group_index=group_index,
                    param_index=param_index,
                    saved_param_id=saved_id,
                )
            )

    opt_name = type(optimizer).__name__
    for entry in entries:
        if not entry.state:
            target = warn if allow_lazy else issues
            target.append(
                OptimizerStateIssue(
                    category="lazy_uninitialized",
                    param_name=entry.param_name,
                    message=f"{entry.param_name}: checkpoint has no optimizer state for this parameter",
                    severity="warning" if allow_lazy else "error",
                )
            )
            continue
        _check_state_entry(
            entry,
            optimizer_name=opt_name,
            issues=issues,
            warnings_out=warn,
            checked=checked,
            check_dtype=False,
        )

    return OptimizerStateVerificationResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_states=tuple(dict.fromkeys(checked)),
        optimizer_name=opt_name,
    )


def _named_parameters(model: Any) -> List[Tuple[str, Any]]:
    if hasattr(model, "named_parameters"):
        return list(model.named_parameters())
    if isinstance(model, Mapping):
        return list(model.items())
    raise TypeError("model must expose named_parameters() or be a mapping of name -> parameter")


def _is_live_optimizer(obj: Any) -> bool:
    return hasattr(obj, "param_groups") and hasattr(obj, "state")


def _entries_from_live_optimizer(model: Any, optimizer: Any) -> List[_StateEntry]:
    name_by_id = {id(param): name for name, param in _named_parameters(model)}
    entries: List[_StateEntry] = []
    for group_index, group in enumerate(getattr(optimizer, "param_groups", ())):
        for param_index, param in enumerate(group.get("params", ())):
            state = optimizer.state.get(param, {})
            entries.append(
                _StateEntry(
                    param_name=name_by_id.get(id(param), f"group{group_index}.param{param_index}"),
                    param=param,
                    state=state,
                    group_index=group_index,
                    param_index=param_index,
                )
            )
    return entries


def _entries_from_state_dict(
    named_params: Sequence[Tuple[str, Any]],
    state_dict: Mapping[str, Any],
    issues: List[OptimizerStateIssue],
    warn: List[OptimizerStateIssue],
) -> List[_StateEntry]:
    entries: List[_StateEntry] = []
    state = state_dict.get("state", {})
    referenced = set()
    param_by_name = dict(named_params)

    for group_index, group in enumerate(state_dict.get("param_groups", ())):
        param_names = list(group.get("param_names", ()))
        for param_index, saved_id in enumerate(group.get("params", ())):
            referenced.add(saved_id)
            if param_index < len(param_names):
                param_name = param_names[param_index]
                if param_name in param_by_name:
                    entries.append(
                        _StateEntry(
                            param_name=param_name,
                            param=param_by_name[param_name],
                            state=state.get(saved_id, {}),
                            group_index=group_index,
                            param_index=param_index,
                            saved_param_id=saved_id,
                        )
                    )
                    continue
                issues.append(
                    OptimizerStateIssue(
                        category="unknown_param_reference",
                        message=(
                            f"state_dict param group {group_index} names parameter "
                            f"{param_name!r}, but the target model has no matching parameter"
                        ),
                    )
                )
                continue
            if not isinstance(saved_id, int) or saved_id < 0 or saved_id >= len(named_params):
                issues.append(
                    OptimizerStateIssue(
                        category="unknown_param_reference",
                        message=(
                            f"state_dict param group {group_index} references parameter id "
                            f"{saved_id!r}, but only {len(named_params)} model parameter(s) "
                            "are available for positional matching"
                        ),
                    )
                )
                continue
            name, param = named_params[saved_id]
            entries.append(
                _StateEntry(
                    param_name=name,
                    param=param,
                    state=state.get(saved_id, {}),
                    group_index=group_index,
                    param_index=param_index,
                    saved_param_id=saved_id,
                )
            )

    for saved_id in set(state) - referenced:
        warn.append(
            OptimizerStateIssue(
                category="orphan_state",
                message=f"state_dict contains state for unreferenced parameter id {saved_id!r}",
                severity="warning",
            )
        )
    return entries


def _check_state_entry(
    entry: _StateEntry,
    *,
    optimizer_name: str,
    issues: List[OptimizerStateIssue],
    warnings_out: List[OptimizerStateIssue],
    checked: List[str],
    check_dtype: bool,
) -> None:
    param_shape = _shape(entry.param)
    if param_shape is None:
        return
    is_adafactor = _is_adafactor_state(optimizer_name, entry.state)

    for key, value in entry.state.items():
        shape = _shape(value)
        if shape is None:
            continue
        checked.append(f"{entry.param_name}:{key}")

        if key in _SCALAR_KEYS:
            if not _is_scalar_value(value, shape):
                issues.append(
                    OptimizerStateIssue(
                        category="state_scalar_mismatch",
                        param_name=entry.param_name,
                        state_key=key,
                        expected_shape=(),
                        actual_shape=shape,
                        message=f"{entry.param_name}.{key}: expected scalar step state, got shape {shape}",
                    )
                )
            continue

        expected = _expected_state_shape(key, param_shape, is_adafactor)
        if expected is not None and shape != expected:
            issues.append(
                OptimizerStateIssue(
                    category="state_shape_mismatch",
                    param_name=entry.param_name,
                    state_key=key,
                    expected_shape=expected,
                    actual_shape=shape,
                    message=(
                        f"{entry.param_name}.{key}: optimizer state shape {shape} "
                        f"does not match expected {expected}"
                    ),
                )
            )
        if check_dtype and expected is not None:
            _check_dtype(
                entry.param_name,
                key,
                expected_dtype=_dtype(entry.param),
                actual_dtype=_dtype(value),
                issues=issues,
                warnings_out=warnings_out,
            )


def _is_adafactor_state(optimizer_name: str, state: Mapping[str, Any]) -> bool:
    low = optimizer_name.lower()
    return "adafactor" in low or bool((_ADFACTOR_ROW_KEYS | _ADFACTOR_COL_KEYS | {"variance"}) & set(state))


def _expected_state_shape(key: str, param_shape: Shape, is_adafactor: bool) -> Optional[Shape]:
    if is_adafactor:
        if key in _ADFACTOR_ROW_KEYS:
            return param_shape[:-1] + (1,) if len(param_shape) >= 2 else param_shape
        if key in _ADFACTOR_COL_KEYS:
            return param_shape[:-2] + (1, param_shape[-1]) if len(param_shape) >= 2 else param_shape
        if key in _ADFACTOR_FULL_KEYS:
            return param_shape
        return param_shape
    if key in _ADAM_SHAPED_KEYS:
        return param_shape
    if key not in _SCALAR_KEYS:
        return param_shape
    return None


def _check_sharded_state(
    named_params: Sequence[Tuple[str, Any]],
    shards: Sequence[OptimizerStateShard],
    issues: List[OptimizerStateIssue],
    warnings_out: List[OptimizerStateIssue],
    checked: List[str],
    check_dtype: bool,
) -> None:
    if not shards:
        return
    params = {name: param for name, param in named_params}
    grouped: Dict[Tuple[str, str, int], List[OptimizerStateShard]] = {}
    for shard in shards:
        grouped.setdefault((shard.param_name, shard.state_key, shard.dim), []).append(shard)

    for (param_name, state_key, dim), group in grouped.items():
        param = params.get(param_name)
        if param is None:
            issues.append(
                OptimizerStateIssue(
                    category="unknown_sharded_param",
                    param_name=param_name,
                    state_key=state_key,
                    message=f"sharded state references unknown parameter {param_name!r}",
                )
            )
            continue
        param_shape = _shape(param)
        if param_shape is None or dim < 0 or dim >= len(param_shape):
            issues.append(
                OptimizerStateIssue(
                    category="invalid_shard_dimension",
                    param_name=param_name,
                    state_key=state_key,
                    message=f"{param_name}.{state_key}: cannot shard dimension {dim} of shape {param_shape}",
                )
            )
            continue

        full = param_shape[dim]
        intervals: List[Tuple[int, int, OptimizerStateShard]] = []
        for shard in group:
            checked.append(f"{param_name}:{state_key}:shard{shard.shard_index if shard.shard_index is not None else len(intervals)}")
            expected_shape = param_shape[:dim] + (shard.length,) + param_shape[dim + 1 :]
            end = shard.start + shard.length
            if shard.start < 0 or shard.length <= 0 or end > full:
                issues.append(
                    OptimizerStateIssue(
                        category="shard_bounds",
                        param_name=param_name,
                        state_key=state_key,
                        expected_shape=expected_shape,
                        actual_shape=shard.shape,
                        shard_index=shard.shard_index,
                        message=(
                            f"{param_name}.{state_key} shard {shard.shard_index}: "
                            f"slice [{shard.start}, {end}) is outside dimension {dim} of length {full}"
                        ),
                    )
                )
            if shard.shape != expected_shape:
                issues.append(
                    OptimizerStateIssue(
                        category="shard_shape_mismatch",
                        param_name=param_name,
                        state_key=state_key,
                        expected_shape=expected_shape,
                        actual_shape=shard.shape,
                        shard_index=shard.shard_index,
                        message=(
                            f"{param_name}.{state_key} shard {shard.shard_index}: "
                            f"local state shape {shard.shape} does not match expected {expected_shape}"
                        ),
                    )
                )
            if check_dtype:
                _check_dtype(
                    param_name,
                    state_key,
                    expected_dtype=_dtype(param),
                    actual_dtype=_dtype(shard),
                    issues=issues,
                    warnings_out=warnings_out,
                    shard_index=shard.shard_index,
                )
            intervals.append((shard.start, end, shard))

        intervals.sort(key=lambda item: (item[0], item[1]))
        cursor = 0
        for start, end, shard in intervals:
            if start > cursor:
                issues.append(
                    OptimizerStateIssue(
                        category="shard_coverage_gap",
                        param_name=param_name,
                        state_key=state_key,
                        shard_index=shard.shard_index,
                        message=(
                            f"{param_name}.{state_key}: missing shard coverage "
                            f"for dimension {dim} slice [{cursor}, {start})"
                        ),
                    )
                )
            if start < cursor:
                issues.append(
                    OptimizerStateIssue(
                        category="shard_overlap",
                        param_name=param_name,
                        state_key=state_key,
                        shard_index=shard.shard_index,
                        message=(
                            f"{param_name}.{state_key}: shard starting at {start} "
                            f"overlaps prior coverage ending at {cursor}"
                        ),
                    )
                )
            cursor = max(cursor, end)
        if cursor < full:
            issues.append(
                OptimizerStateIssue(
                    category="shard_coverage_gap",
                    param_name=param_name,
                    state_key=state_key,
                    message=(
                        f"{param_name}.{state_key}: missing shard coverage "
                        f"for dimension {dim} slice [{cursor}, {full})"
                    ),
                )
            )


def _shape(value: Any) -> Optional[Shape]:
    if isinstance(value, OptimizerStateShard):
        return value.shape
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype(value: Any) -> Optional[str]:
    if isinstance(value, OptimizerStateShard):
        return _normalize_dtype(value.dtype)
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return _normalize_dtype(dtype)


def _normalize_dtype(dtype: Any) -> str:
    text = str(dtype)
    return text if text.startswith("torch.") else text.lower()


def _is_scalar_value(value: Any, shape: Shape) -> bool:
    if isinstance(value, Number):
        return True
    if not shape:
        return True
    return math.prod(shape) == 1


def _check_dtype(
    param_name: str,
    state_key: str,
    *,
    expected_dtype: Optional[str],
    actual_dtype: Optional[str],
    issues: List[OptimizerStateIssue],
    warnings_out: List[OptimizerStateIssue],
    shard_index: Optional[int] = None,
) -> None:
    if expected_dtype is None or actual_dtype is None or expected_dtype == actual_dtype:
        return
    target = warnings_out if _is_fp32_master_state(expected_dtype, actual_dtype) else issues
    severity = "warning" if target is warnings_out else "error"
    category = "master_state_dtype" if target is warnings_out else "state_dtype_mismatch"
    target.append(
        OptimizerStateIssue(
            category=category,
            param_name=param_name,
            state_key=state_key,
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
            shard_index=shard_index,
            severity=severity,
            message=(
                f"{param_name}.{state_key}: optimizer state dtype {actual_dtype} "
                f"does not match parameter dtype {expected_dtype}"
            ),
        )
    )


def _is_fp32_master_state(expected_dtype: str, actual_dtype: str) -> bool:
    return expected_dtype in _LOW_PRECISION_DTYPES and actual_dtype in _FLOAT32_DTYPES


__all__ = [
    "OptimizerStateIssue",
    "OptimizerStateShard",
    "OptimizerStateVerificationResult",
    "TensorGuardOptimizerStateError",
    "guarded_optimizer_load_state_dict",
    "verify_optimizer_state",
]
