"""Checkpoint ``state_dict`` schema gates for safe model resume.

PyTorch already rejects shape-incompatible tensors during
``load_state_dict``.  It is less strict at several high-value boundaries:
dtype-mismatched tensors are silently cast, tied parameters can be overwritten
by the last alias loaded, adapter-only LoRA checkpoints are easy to misroute,
and tensor-parallel shards are ordinary unexpected keys unless a caller knows
their sharding schema.  This module makes those cases explicit before a model
is mutated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class CheckpointIssue:
    """One actionable model-checkpoint compatibility finding."""

    category: str
    message: str
    key: Optional[str] = None
    expected_shape: Optional[Shape] = None
    actual_shape: Optional[Shape] = None
    expected_dtype: Optional[str] = None
    actual_dtype: Optional[str] = None
    severity: str = "error"
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class TensorParallelCheckpointShard:
    """Declared shard of one logical model parameter in a checkpoint.

    ``start`` and ``length`` describe the slice along ``dim`` of the full
    parameter named by ``param_name``.  The tensor is read from ``shard_key`` in
    the checkpoint state dict.
    """

    param_name: str
    shard_key: str
    dim: int
    start: int
    length: int
    rank: Optional[int] = None


@dataclass(frozen=True)
class CheckpointVerificationResult:
    """Result of TensorGuard's model-checkpoint schema gate."""

    ok: bool
    issues: Tuple[CheckpointIssue, ...]
    warnings: Tuple[CheckpointIssue, ...] = ()
    checked_keys: Tuple[str, ...] = ()
    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()


class TensorGuardCheckpointError(ValueError):
    """Raised when a model checkpoint is incompatible with a target model."""

    def __init__(self, issues: Sequence[CheckpointIssue]):
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected checkpoint with {len(self.issues)} issue(s): "
            f"{details}{more}"
        )


def verify_checkpoint_state_dict(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
    check_dtype: bool = True,
    allow_dtype_cast: bool = False,
    tensor_parallel_shards: Sequence[TensorParallelCheckpointShard] = (),
    allow_adapter_only: bool = True,
) -> CheckpointVerificationResult:
    """Validate a model checkpoint against a target model schema.

    ``checkpoint`` may be either a raw ``state_dict`` mapping or a common
    envelope containing ``state_dict`` / ``model_state_dict`` / ``model``.
    Tensor-parallel shard specs are verification-only: complete shard coverage
    suppresses the missing full parameter, but this function does not merge
    shards into a loadable tensor.
    """

    state = _unwrap_state_dict(checkpoint)
    target = _target_state_dict(model)
    target_keys = set(target)
    checkpoint_keys = set(state)
    issues: List[CheckpointIssue] = []
    warn: List[CheckpointIssue] = []
    checked: List[str] = []

    lora_groups = _collect_lora_groups(state)
    lora_keys = {key for group in lora_groups.values() for key in group.values()}
    shard_keys = {shard.shard_key for shard in tensor_parallel_shards}
    covered_sharded_params = _check_tensor_parallel_shards(
        target,
        state,
        tensor_parallel_shards,
        issues,
        warn,
        checked,
        check_dtype=check_dtype,
        allow_dtype_cast=allow_dtype_cast,
    )

    _check_exact_state_entries(
        target,
        state,
        issues,
        warn,
        checked,
        check_dtype=check_dtype,
        allow_dtype_cast=allow_dtype_cast,
    )
    _check_tied_weights(model, state, issues, checked)
    _check_lora_adapters(
        target,
        state,
        lora_groups,
        issues,
        warn,
        checked,
        check_dtype=check_dtype,
        allow_dtype_cast=allow_dtype_cast,
    )

    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()
    if strict:
        adapter_only = (
            allow_adapter_only
            and bool(lora_keys)
            and not any(key in target_keys for key in checkpoint_keys - lora_keys)
        )
        missing = target_keys - checkpoint_keys - covered_sharded_params
        if adapter_only:
            missing = {key for key in missing if _parse_lora_key(key) is not None}
        unexpected = checkpoint_keys - target_keys - shard_keys - lora_keys
        missing_keys = tuple(sorted(missing))
        unexpected_keys = tuple(sorted(unexpected))
        for key in missing_keys:
            issues.append(
                CheckpointIssue(
                    category="missing_key",
                    key=key,
                    message=f"checkpoint is missing target state key {key!r}",
                    suggestion="Save a checkpoint from the same model schema or load with strict=False.",
                )
            )
        for key in unexpected_keys:
            issues.append(
                CheckpointIssue(
                    category="unexpected_key",
                    key=key,
                    message=f"checkpoint contains unexpected state key {key!r}",
                    suggestion="Remove stale keys or verify against the model version that produced them.",
                )
            )

    return CheckpointVerificationResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_keys=tuple(dict.fromkeys(checked)),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def guarded_load_state_dict(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    strict: bool = True,
    on_violation: str = "raise",
    check_dtype: bool = True,
    allow_dtype_cast: bool = False,
    tensor_parallel_shards: Sequence[TensorParallelCheckpointShard] = (),
    allow_adapter_only: bool = False,
) -> CheckpointVerificationResult:
    """Verify a checkpoint before mutating ``model`` with ``load_state_dict``."""

    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")

    result = verify_checkpoint_state_dict(
        model,
        checkpoint,
        strict=strict,
        check_dtype=check_dtype,
        allow_dtype_cast=allow_dtype_cast,
        tensor_parallel_shards=tensor_parallel_shards,
        allow_adapter_only=allow_adapter_only,
    )
    _handle_result(result, on_violation)
    if result.issues and on_violation != "ignore":
        return result

    unwrapped = _unwrap_state_dict(checkpoint)
    if not hasattr(model, "load_state_dict"):
        raise TypeError("model must expose load_state_dict() to use guarded_load_state_dict")
    model.load_state_dict(unwrapped, strict=strict)
    return result


def _handle_result(result: CheckpointVerificationResult, on_violation: str) -> None:
    if result.ok or on_violation == "ignore":
        return
    if on_violation == "warn":
        warnings.warn(
            str(TensorGuardCheckpointError(result.issues)),
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise TensorGuardCheckpointError(result.issues)


def _unwrap_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def _target_state_dict(model: Any) -> Mapping[str, Any]:
    if isinstance(model, Mapping):
        return model
    if hasattr(model, "state_dict"):
        return model.state_dict()
    raise TypeError("model must expose state_dict() or be a mapping")


def _check_exact_state_entries(
    target: Mapping[str, Any],
    state: Mapping[str, Any],
    issues: List[CheckpointIssue],
    warn: List[CheckpointIssue],
    checked: List[str],
    *,
    check_dtype: bool,
    allow_dtype_cast: bool,
) -> None:
    for key in sorted(set(target) & set(state)):
        expected_shape = _shape(target[key])
        actual_shape = _shape(state[key])
        if expected_shape is not None and actual_shape is not None:
            checked.append(key)
            if actual_shape != expected_shape:
                issues.append(
                    CheckpointIssue(
                        category="shape_mismatch",
                        key=key,
                        expected_shape=expected_shape,
                        actual_shape=actual_shape,
                        message=(
                            f"{key}: checkpoint tensor shape {actual_shape} "
                            f"does not match target shape {expected_shape}"
                        ),
                        suggestion="Regenerate the checkpoint after the model architecture change.",
                    )
                )
        if check_dtype:
            _check_dtype(
                key,
                expected_dtype=_dtype(target[key]),
                actual_dtype=_dtype(state[key]),
                category="dtype_mismatch",
                issues=issues,
                warn=warn,
                allow_dtype_cast=allow_dtype_cast,
                suggestion=(
                    "PyTorch would silently cast this tensor during load; save "
                    "the checkpoint in the target dtype or pass allow_dtype_cast=True."
                ),
            )


def _check_tied_weights(
    model: Any,
    state: Mapping[str, Any],
    issues: List[CheckpointIssue],
    checked: List[str],
) -> None:
    for group in _tied_parameter_groups(model):
        present = [key for key in group if key in state]
        if len(present) < 2:
            continue
        anchor_key = present[0]
        anchor = state[anchor_key]
        checked.extend(present)
        for key in present[1:]:
            value = state[key]
            anchor_shape = _shape(anchor)
            value_shape = _shape(value)
            if anchor_shape is not None and value_shape is not None and anchor_shape != value_shape:
                issues.append(
                    CheckpointIssue(
                        category="tied_weight_shape_mismatch",
                        key=key,
                        expected_shape=anchor_shape,
                        actual_shape=value_shape,
                        message=(
                            f"tied weights {anchor_key!r} and {key!r} have "
                            f"different checkpoint shapes {anchor_shape} vs {value_shape}"
                        ),
                        suggestion="Store tied aliases with identical tensor metadata.",
                    )
                )
                continue
            if _dtype(anchor) != _dtype(value):
                issues.append(
                    CheckpointIssue(
                        category="tied_weight_dtype_mismatch",
                        key=key,
                        expected_dtype=_dtype(anchor),
                        actual_dtype=_dtype(value),
                        message=(
                            f"tied weights {anchor_key!r} and {key!r} have "
                            f"different checkpoint dtypes {_dtype(anchor)} vs {_dtype(value)}"
                        ),
                        suggestion="Store tied aliases with identical tensor metadata.",
                    )
                )
                continue
            if _device(anchor) != _device(value):
                issues.append(
                    CheckpointIssue(
                        category="tied_weight_device_mismatch",
                        key=key,
                        message=(
                            f"tied weights {anchor_key!r} and {key!r} are stored on "
                            f"different devices {_device(anchor)} vs {_device(value)}"
                        ),
                        suggestion="Move tied aliases to the same device before saving.",
                    )
                )
                continue
            equal = getattr(anchor, "equal", None)
            if callable(equal):
                try:
                    if not bool(equal(value)):
                        issues.append(
                            CheckpointIssue(
                                category="tied_weight_value_mismatch",
                                key=key,
                                message=(
                                    f"target model ties {', '.join(group)}, but checkpoint "
                                    f"stores different values for {anchor_key!r} and {key!r}; "
                                    "PyTorch load_state_dict would silently let the last alias win"
                                ),
                                suggestion="Save tied aliases with identical values or untie the target model.",
                            )
                        )
                except RuntimeError as exc:
                    issues.append(
                        CheckpointIssue(
                            category="tied_weight_compare_failed",
                            key=key,
                            message=f"could not compare tied weights {anchor_key!r} and {key!r}: {exc}",
                        )
                    )


def _tied_parameter_groups(model: Any) -> List[Tuple[str, ...]]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return []
    try:
        params = list(named_parameters(remove_duplicate=False))
    except TypeError:
        return []
    grouped: Dict[int, List[str]] = {}
    for name, param in params:
        grouped.setdefault(id(param), []).append(name)
    return [tuple(names) for names in grouped.values() if len(names) > 1]


def _collect_lora_groups(state: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for key in state:
        parsed = _parse_lora_key(key)
        if parsed is None:
            continue
        base, role = parsed
        groups.setdefault(base, {})[role] = key
    return groups


def _parse_lora_key(key: str) -> Optional[Tuple[str, str]]:
    suffixes = (
        (".lora_A.weight", "A"),
        (".lora_B.weight", "B"),
        (".lora_A", "A"),
        (".lora_B", "B"),
    )
    for suffix, role in suffixes:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], role
    return None


def _check_lora_adapters(
    target: Mapping[str, Any],
    state: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, str]],
    issues: List[CheckpointIssue],
    warn: List[CheckpointIssue],
    checked: List[str],
    *,
    check_dtype: bool,
    allow_dtype_cast: bool,
) -> None:
    for base, group in sorted(groups.items()):
        a_key = group.get("A")
        b_key = group.get("B")
        if not a_key or not b_key:
            existing = a_key or b_key or f"{base}.lora_<missing>"
            missing_role = "lora_B" if a_key else "lora_A"
            issues.append(
                CheckpointIssue(
                    category="lora_pair_incomplete",
                    key=existing,
                    message=f"{base}: checkpoint has only one LoRA matrix; missing {missing_role}",
                    suggestion="Save both LoRA A and B matrices for each adapted module.",
                )
            )
            continue
        checked.extend([a_key, b_key])
        weight_key = f"{base}.weight"
        base_weight = target.get(weight_key)
        if base_weight is None:
            issues.append(
                CheckpointIssue(
                    category="lora_target_missing",
                    key=a_key,
                    message=(
                        f"{base}: LoRA adapter targets {weight_key!r}, "
                        "but the target model has no such weight"
                    ),
                    suggestion="Load this adapter into the base model architecture it was trained for.",
                )
            )
            continue
        base_shape = _shape(base_weight)
        a_shape = _shape(state[a_key])
        b_shape = _shape(state[b_key])
        if base_shape is None or len(base_shape) != 2:
            issues.append(
                CheckpointIssue(
                    category="lora_target_not_linear",
                    key=weight_key,
                    actual_shape=base_shape,
                    message=f"{base}: LoRA target weight is not a rank-2 Linear-style matrix",
                    suggestion="Only Linear-style LoRA adapter checkpoints are supported by this gate.",
                )
            )
            continue
        if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
            issues.append(
                CheckpointIssue(
                    category="lora_matrix_rank",
                    key=a_key,
                    actual_shape=a_shape,
                    message=f"{base}: LoRA A/B tensors must both be rank-2 matrices",
                )
            )
            continue
        out_features, in_features = base_shape
        rank_a, in_a = a_shape
        out_b, rank_b = b_shape
        if rank_a != rank_b:
            issues.append(
                CheckpointIssue(
                    category="lora_rank_mismatch",
                    key=b_key,
                    expected_shape=(out_b, rank_a),
                    actual_shape=b_shape,
                    message=(
                        f"{base}: lora_A rank {rank_a} does not match "
                        f"lora_B rank {rank_b}"
                    ),
                    suggestion="Use matching low-rank dimensions for LoRA A and B.",
                )
            )
        if in_a != in_features:
            issues.append(
                CheckpointIssue(
                    category="lora_input_mismatch",
                    key=a_key,
                    expected_shape=(rank_a, in_features),
                    actual_shape=a_shape,
                    message=(
                        f"{base}: lora_A input dimension {in_a} does not match "
                        f"base in_features {in_features}"
                    ),
                )
            )
        if out_b != out_features:
            issues.append(
                CheckpointIssue(
                    category="lora_output_mismatch",
                    key=b_key,
                    expected_shape=(out_features, rank_b),
                    actual_shape=b_shape,
                    message=(
                        f"{base}: lora_B output dimension {out_b} does not match "
                        f"base out_features {out_features}"
                    ),
                )
            )
        if rank_a <= 0 or rank_a > min(in_features, out_features):
            issues.append(
                CheckpointIssue(
                    category="lora_rank_invalid",
                    key=a_key,
                    message=(
                        f"{base}: LoRA rank {rank_a} must be in "
                        f"[1, min({in_features}, {out_features})]"
                    ),
                )
            )
        if check_dtype:
            for key in (a_key, b_key):
                _check_dtype(
                    key,
                    expected_dtype=_dtype(base_weight),
                    actual_dtype=_dtype(state[key]),
                    category="lora_dtype_mismatch",
                    issues=issues,
                    warn=warn,
                    allow_dtype_cast=allow_dtype_cast,
                    suggestion="Save adapter matrices in the same dtype as the target base weight.",
                )


def _check_tensor_parallel_shards(
    target: Mapping[str, Any],
    state: Mapping[str, Any],
    shards: Sequence[TensorParallelCheckpointShard],
    issues: List[CheckpointIssue],
    warn: List[CheckpointIssue],
    checked: List[str],
    *,
    check_dtype: bool,
    allow_dtype_cast: bool,
) -> Set[str]:
    if not shards:
        return set()
    grouped: Dict[Tuple[str, int], List[TensorParallelCheckpointShard]] = {}
    for shard in shards:
        grouped.setdefault((shard.param_name, shard.dim), []).append(shard)

    complete: Set[str] = set()
    for (param_name, dim), group in grouped.items():
        full = target.get(param_name)
        full_shape = _shape(full)
        if full is None or full_shape is None:
            issues.append(
                CheckpointIssue(
                    category="tp_unknown_param",
                    key=param_name,
                    message=f"tensor-parallel shard spec references unknown parameter {param_name!r}",
                )
            )
            continue
        normalized_dim = dim + len(full_shape) if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= len(full_shape):
            issues.append(
                CheckpointIssue(
                    category="tp_shard_dim_invalid",
                    key=param_name,
                    expected_shape=full_shape,
                    message=f"{param_name}: cannot shard dimension {dim} of shape {full_shape}",
                )
            )
            continue
        intervals: List[Tuple[int, int, TensorParallelCheckpointShard]] = []
        full_extent = full_shape[normalized_dim]
        for shard in group:
            checked.append(shard.shard_key)
            tensor = state.get(shard.shard_key)
            if tensor is None:
                issues.append(
                    CheckpointIssue(
                        category="tp_shard_missing",
                        key=shard.shard_key,
                        message=f"{param_name}: checkpoint is missing tensor-parallel shard {shard.shard_key!r}",
                    )
                )
                continue
            expected_shape = (
                full_shape[:normalized_dim]
                + (shard.length,)
                + full_shape[normalized_dim + 1 :]
            )
            actual_shape = _shape(tensor)
            end = shard.start + shard.length
            if shard.start < 0 or shard.length <= 0 or end > full_extent:
                issues.append(
                    CheckpointIssue(
                        category="tp_shard_bounds",
                        key=shard.shard_key,
                        expected_shape=full_shape,
                        actual_shape=actual_shape,
                        message=(
                            f"{param_name}: shard {shard.shard_key!r} slice "
                            f"[{shard.start}, {end}) is outside dim {normalized_dim} "
                            f"of length {full_extent}"
                        ),
                    )
                )
            if actual_shape is not None and actual_shape != expected_shape:
                issues.append(
                    CheckpointIssue(
                        category="tp_shard_shape_mismatch",
                        key=shard.shard_key,
                        expected_shape=expected_shape,
                        actual_shape=actual_shape,
                        message=(
                            f"{param_name}: shard {shard.shard_key!r} shape {actual_shape} "
                            f"does not match expected local shape {expected_shape}"
                        ),
                    )
                )
            if check_dtype:
                _check_dtype(
                    shard.shard_key,
                    expected_dtype=_dtype(full),
                    actual_dtype=_dtype(tensor),
                    category="tp_shard_dtype_mismatch",
                    issues=issues,
                    warn=warn,
                    allow_dtype_cast=allow_dtype_cast,
                    suggestion="Store tensor-parallel shards in the target parameter dtype.",
                )
            intervals.append((shard.start, end, shard))
        intervals.sort(key=lambda item: (item[0], item[1]))
        cursor = 0
        coverage_ok = True
        for start, end, shard in intervals:
            if start > cursor:
                coverage_ok = False
                issues.append(
                    CheckpointIssue(
                        category="tp_shard_gap",
                        key=shard.shard_key,
                        message=(
                            f"{param_name}: missing tensor-parallel shard coverage "
                            f"for dim {normalized_dim} slice [{cursor}, {start})"
                        ),
                    )
                )
            if start < cursor:
                coverage_ok = False
                issues.append(
                    CheckpointIssue(
                        category="tp_shard_overlap",
                        key=shard.shard_key,
                        message=(
                            f"{param_name}: shard {shard.shard_key!r} overlaps "
                            f"previous coverage ending at {cursor}"
                        ),
                    )
                )
            cursor = max(cursor, end)
        if cursor < full_extent:
            coverage_ok = False
            issues.append(
                CheckpointIssue(
                    category="tp_shard_gap",
                    key=param_name,
                    message=(
                        f"{param_name}: missing tensor-parallel shard coverage "
                        f"for dim {normalized_dim} slice [{cursor}, {full_extent})"
                    ),
                )
            )
        if coverage_ok and all(shard.shard_key in state for shard in group):
            complete.add(param_name)
    return complete


def _check_dtype(
    key: str,
    *,
    expected_dtype: Optional[str],
    actual_dtype: Optional[str],
    category: str,
    issues: List[CheckpointIssue],
    warn: List[CheckpointIssue],
    allow_dtype_cast: bool,
    suggestion: Optional[str],
) -> None:
    if expected_dtype is None or actual_dtype is None or expected_dtype == actual_dtype:
        return
    target = warn if allow_dtype_cast else issues
    target.append(
        CheckpointIssue(
            category=category,
            key=key,
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
            severity="warning" if allow_dtype_cast else "error",
            message=f"{key}: checkpoint dtype {actual_dtype} does not match target dtype {expected_dtype}",
            suggestion=suggestion,
        )
    )


def _shape(value: Any) -> Optional[Shape]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype(value: Any) -> Optional[str]:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    text = str(dtype)
    return text if text.startswith("torch.") else text.lower()


def _device(value: Any) -> Optional[str]:
    device = getattr(value, "device", None)
    return None if device is None else str(device)


__all__ = [
    "CheckpointIssue",
    "CheckpointVerificationResult",
    "TensorGuardCheckpointError",
    "TensorParallelCheckpointShard",
    "guarded_load_state_dict",
    "verify_checkpoint_state_dict",
]
