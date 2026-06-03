"""Model-serving request/response schema gates.

The pure ``verify_serving_schema`` function checks already-materialized request,
preprocessing, model-output, and response payloads.  The guarded helpers run
FastAPI- or TorchServe-style pipelines only up to the next unsafe boundary:
request and preprocessing violations are reported before model invocation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


DimSpec = Union[int, str, None]
ShapeSpec = Tuple[DimSpec, ...]
Shape = Tuple[int, ...]


@dataclass(frozen=True)
class ServingTensorSpec:
    """Tensor-like schema for a serving boundary value.

    ``name`` is a dot path for mappings/sequences.  Use ``"$"`` or ``""`` to
    refer to the whole payload.  Shape dimensions may be concrete integers,
    ``None``/``-1``/``"*"`` wildcards, or symbolic strings such as ``"B"``;
    symbolic strings are scoped to one validation group unless
    ``bind_shared_symbols=True`` is requested.
    """

    name: str
    shape: Optional[ShapeSpec] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    required: bool = True
    aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "aliases", tuple(self.aliases))


@dataclass(frozen=True)
class ServingSchemaIssue:
    """One actionable model-serving schema finding."""

    category: str
    message: str
    path: Optional[str] = None
    expected_shape: Optional[ShapeSpec] = None
    actual_shape: Optional[Shape] = None
    expected_dtype: Optional[str] = None
    actual_dtype: Optional[str] = None
    expected_device: Optional[str] = None
    actual_device: Optional[str] = None
    severity: str = "error"
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class ServingSchemaGateResult:
    """Result of TensorGuard's serving-boundary schema gate."""

    ok: bool
    issues: Tuple[ServingSchemaIssue, ...]
    warnings: Tuple[ServingSchemaIssue, ...] = ()
    checked_paths: Tuple[str, ...] = ()
    framework: str = "generic"
    model_invoked: bool = False
    postprocess_invoked: bool = False


class TensorGuardServingSchemaError(ValueError):
    """Raised when a serving request, model input, output, or response is unsafe."""

    def __init__(self, result: ServingSchemaGateResult):
        self.result = result
        self.issues = tuple(result.issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected {result.framework} serving schema with "
            f"{len(self.issues)} issue(s): {details}{more}"
        )


def verify_serving_schema(
    *,
    request: Any = None,
    request_specs: Sequence[ServingTensorSpec] = (),
    inputs: Any = None,
    input_specs: Sequence[ServingTensorSpec] = (),
    outputs: Any = None,
    output_specs: Sequence[ServingTensorSpec] = (),
    response: Any = None,
    response_specs: Sequence[ServingTensorSpec] = (),
    framework: str = "generic",
    bind_shared_symbols: bool = False,
) -> ServingSchemaGateResult:
    """Purely validate materialized serving-boundary values against specs.

    This function never calls preprocessing, model, or postprocessing code.  It
    is useful for checking captured FastAPI/TorchServe payloads and for the
    guarded helpers after each pipeline stage has produced a value.
    """

    shared_bindings: Dict[str, int] = {}
    results: List[ServingSchemaGateResult] = []

    for stage, value, specs in (
        ("request", request, request_specs),
        ("input", inputs, input_specs),
        ("output", outputs, output_specs),
        ("response", response, response_specs),
    ):
        if not specs:
            continue
        bindings = shared_bindings if bind_shared_symbols else {}
        results.append(_verify_one_group(value, specs, stage, framework, bindings))

    return _merge_results(
        framework,
        results,
        model_invoked=False,
        postprocess_invoked=False,
    )


def guarded_model_serving_call(
    model: Callable[..., Any],
    request: Any,
    *,
    input_specs: Sequence[ServingTensorSpec],
    output_specs: Sequence[ServingTensorSpec] = (),
    request_specs: Sequence[ServingTensorSpec] = (),
    response_specs: Sequence[ServingTensorSpec] = (),
    preprocess: Optional[Callable[[Any], Any]] = None,
    postprocess: Optional[Callable[[Any], Any]] = None,
    framework: str = "generic",
    on_violation: str = "raise",
    bind_shared_symbols: bool = False,
) -> ServingSchemaGateResult:
    """Run a serving pipeline with schema gates before and after invocation.

    Request and preprocessing-output violations always stop before model
    invocation; ``on_violation`` controls whether the result is raised, warned,
    or returned.  Output/response violations are reported after the model has
    produced the incompatible value.
    """

    return _guarded_serving_pipeline(
        lambda prepared: _call_model(model, prepared),
        request,
        input_specs=input_specs,
        output_specs=output_specs,
        request_specs=request_specs,
        response_specs=response_specs,
        preprocess=preprocess,
        postprocess=postprocess,
        framework=framework,
        on_violation=on_violation,
        bind_shared_symbols=bind_shared_symbols,
    )


def guarded_fastapi_endpoint(
    preprocess: Callable[[Any], Any],
    model: Callable[..., Any],
    request: Any,
    *,
    input_specs: Sequence[ServingTensorSpec],
    output_specs: Sequence[ServingTensorSpec] = (),
    request_specs: Sequence[ServingTensorSpec] = (),
    response_specs: Sequence[ServingTensorSpec] = (),
    postprocess: Optional[Callable[[Any], Any]] = None,
    on_violation: str = "raise",
    bind_shared_symbols: bool = False,
) -> ServingSchemaGateResult:
    """Guard a FastAPI-style request -> preprocess -> model -> response path."""

    return guarded_model_serving_call(
        model,
        request,
        input_specs=input_specs,
        output_specs=output_specs,
        request_specs=request_specs,
        response_specs=response_specs,
        preprocess=preprocess,
        postprocess=postprocess,
        framework="fastapi",
        on_violation=on_violation,
        bind_shared_symbols=bind_shared_symbols,
    )


def guarded_torchserve_handler(
    handler: Any,
    requests: Any,
    *,
    input_specs: Sequence[ServingTensorSpec],
    output_specs: Sequence[ServingTensorSpec] = (),
    request_specs: Sequence[ServingTensorSpec] = (),
    response_specs: Sequence[ServingTensorSpec] = (),
    on_violation: str = "raise",
    bind_shared_symbols: bool = False,
) -> ServingSchemaGateResult:
    """Guard a TorchServe-style handler's preprocess/inference/postprocess path."""

    preprocess = getattr(handler, "preprocess", None)
    postprocess = getattr(handler, "postprocess", None)
    inference = getattr(handler, "inference", None)

    if inference is None:
        model = getattr(handler, "model", None)
        if model is None:
            raise TypeError("handler must expose inference() or model")
        inference = lambda prepared: _call_model(model, prepared)

    return _guarded_serving_pipeline(
        inference,
        requests,
        input_specs=input_specs,
        output_specs=output_specs,
        request_specs=request_specs,
        response_specs=response_specs,
        preprocess=preprocess if callable(preprocess) else None,
        postprocess=postprocess if callable(postprocess) else None,
        framework="torchserve",
        on_violation=on_violation,
        bind_shared_symbols=bind_shared_symbols,
    )


def _guarded_serving_pipeline(
    inference: Callable[[Any], Any],
    request: Any,
    *,
    input_specs: Sequence[ServingTensorSpec],
    output_specs: Sequence[ServingTensorSpec],
    request_specs: Sequence[ServingTensorSpec],
    response_specs: Sequence[ServingTensorSpec],
    preprocess: Optional[Callable[[Any], Any]],
    postprocess: Optional[Callable[[Any], Any]],
    framework: str,
    on_violation: str,
    bind_shared_symbols: bool,
) -> ServingSchemaGateResult:
    _validate_policy(on_violation)
    shared_bindings: Dict[str, int] = {}

    def bindings_for_stage() -> Dict[str, int]:
        return shared_bindings if bind_shared_symbols else {}

    result = _verify_one_group(
        request,
        request_specs,
        "request",
        framework,
        bindings_for_stage(),
    )
    if not result.ok:
        return _handle_and_return(
            _runtime_result(result, model_invoked=False, postprocess_invoked=False),
            on_violation,
        )

    prepared = preprocess(request) if preprocess is not None else request
    input_result = _verify_one_group(
        prepared,
        input_specs,
        "input",
        framework,
        bindings_for_stage(),
    )
    if not input_result.ok:
        return _handle_and_return(
            _merge_results(
                framework,
                (result, input_result),
                model_invoked=False,
                postprocess_invoked=False,
            ),
            on_violation,
        )

    model_output = inference(prepared)
    output_result = _verify_one_group(
        model_output,
        output_specs,
        "output",
        framework,
        bindings_for_stage(),
    )
    if not output_result.ok:
        return _handle_and_return(
            _merge_results(
                framework,
                (result, input_result, output_result),
                model_invoked=True,
                postprocess_invoked=False,
            ),
            on_violation,
        )

    response = postprocess(model_output) if postprocess is not None else model_output
    response_result = _verify_one_group(
        response,
        response_specs,
        "response",
        framework,
        bindings_for_stage(),
    )
    return _handle_and_return(
        _merge_results(
            framework,
            (result, input_result, output_result, response_result),
            model_invoked=True,
            postprocess_invoked=postprocess is not None,
        ),
        on_violation,
    )


def _verify_one_group(
    value: Any,
    specs: Sequence[ServingTensorSpec],
    stage: str,
    framework: str,
    bindings: Dict[str, int],
) -> ServingSchemaGateResult:
    issues: List[ServingSchemaIssue] = []
    warn: List[ServingSchemaIssue] = []
    checked: List[str] = []
    if specs:
        _validate_spec_group(
            value,
            specs,
            stage=stage,
            bindings=bindings,
            issues=issues,
            warnings_out=warn,
            checked=checked,
        )
    return ServingSchemaGateResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_paths=tuple(dict.fromkeys(checked)),
        framework=framework,
    )


def _validate_spec_group(
    value: Any,
    specs: Sequence[ServingTensorSpec],
    *,
    stage: str,
    bindings: Dict[str, int],
    issues: List[ServingSchemaIssue],
    warnings_out: List[ServingSchemaIssue],
    checked: List[str],
) -> None:
    for index, spec in enumerate(specs):
        path, extracted = _extract_for_spec(value, spec, index, len(specs))
        full_path = f"{stage}.{path}"
        if extracted is _MISSING:
            if spec.required:
                issues.append(
                    ServingSchemaIssue(
                        category="missing_field",
                        path=full_path,
                        message=f"{full_path}: required serving schema field is missing",
                        expected_shape=spec.shape,
                        expected_dtype=spec.dtype,
                        suggestion="Update the request/preprocessor/postprocessor schema or mark the field optional.",
                    )
                )
            continue

        checked.append(full_path)
        if spec.shape is not None:
            actual_shape, ragged = _shape_of(extracted)
            if ragged:
                issues.append(
                    ServingSchemaIssue(
                        category="ragged_payload",
                        path=full_path,
                        message=f"{full_path}: payload is ragged and cannot satisfy shape {spec.shape}",
                        expected_shape=spec.shape,
                        suggestion="Pad or stack the payload into a rectangular tensor before this boundary.",
                    )
                )
            elif actual_shape is None:
                issues.append(
                    ServingSchemaIssue(
                        category="shape_unavailable",
                        path=full_path,
                        message=f"{full_path}: value exposes no static shape for expected shape {spec.shape}",
                        expected_shape=spec.shape,
                        suggestion="Return a tensor, ndarray, or rectangular nested sequence at this boundary.",
                    )
                )
            else:
                _check_shape(full_path, spec.shape, actual_shape, bindings, issues)

        if spec.dtype is not None:
            actual_dtype = _dtype_of(extracted)
            if actual_dtype is None:
                warnings_out.append(
                    ServingSchemaIssue(
                        category="dtype_unavailable",
                        path=full_path,
                        message=f"{full_path}: dtype cannot be proven for non-tensor payload",
                        expected_dtype=spec.dtype,
                        severity="warning",
                    )
                )
            elif not _dtype_matches(actual_dtype, spec.dtype):
                issues.append(
                    ServingSchemaIssue(
                        category="dtype_mismatch",
                        path=full_path,
                        message=(
                            f"{full_path}: dtype {actual_dtype!r} does not match "
                            f"expected dtype {spec.dtype!r}"
                        ),
                        expected_dtype=spec.dtype,
                        actual_dtype=actual_dtype,
                        suggestion="Cast in preprocessing or update the serving schema.",
                    )
                )

        if spec.device is not None:
            actual_device = _device_of(extracted)
            if actual_device is None:
                warnings_out.append(
                    ServingSchemaIssue(
                        category="device_unavailable",
                        path=full_path,
                        message=f"{full_path}: device cannot be proven for non-tensor payload",
                        expected_device=spec.device,
                        severity="warning",
                    )
                )
            elif str(actual_device) != str(spec.device):
                issues.append(
                    ServingSchemaIssue(
                        category="device_mismatch",
                        path=full_path,
                        message=(
                            f"{full_path}: device {actual_device!r} does not match "
                            f"expected device {spec.device!r}"
                        ),
                        expected_device=spec.device,
                        actual_device=str(actual_device),
                        suggestion="Move tensors to the serving device before model invocation.",
                    )
                )


def _check_shape(
    path: str,
    expected: ShapeSpec,
    actual: Shape,
    bindings: Dict[str, int],
    issues: List[ServingSchemaIssue],
) -> None:
    if len(expected) != len(actual):
        issues.append(
            ServingSchemaIssue(
                category="rank_mismatch",
                path=path,
                message=f"{path}: rank {len(actual)} does not match expected rank {len(expected)}",
                expected_shape=expected,
                actual_shape=actual,
                suggestion="Fix preprocessing layout/batching before calling the model.",
            )
        )
        return

    for axis, (expected_dim, actual_dim) in enumerate(zip(expected, actual)):
        if _is_wildcard_dim(expected_dim):
            continue
        if isinstance(expected_dim, str) and not expected_dim.isdigit():
            bound = bindings.get(expected_dim)
            if bound is None:
                bindings[expected_dim] = actual_dim
            elif bound != actual_dim:
                issues.append(
                    ServingSchemaIssue(
                        category="symbolic_dim_mismatch",
                        path=path,
                        message=(
                            f"{path}: axis {axis} has dimension {actual_dim}, "
                            f"but symbol {expected_dim!r} was already bound to {bound}"
                        ),
                        expected_shape=expected,
                        actual_shape=actual,
                        suggestion="Use distinct symbols or set bind_shared_symbols=False for shape-changing stages.",
                    )
                )
            continue

        if int(expected_dim) != actual_dim:
            issues.append(
                ServingSchemaIssue(
                    category="shape_mismatch",
                    path=path,
                    message=(
                        f"{path}: axis {axis} has dimension {actual_dim}, "
                        f"expected {expected_dim}"
                    ),
                    expected_shape=expected,
                    actual_shape=actual,
                    suggestion="Fix preprocessing layout/batching before calling the model.",
                )
            )


def _extract_for_spec(
    value: Any,
    spec: ServingTensorSpec,
    index: int,
    spec_count: int,
) -> Tuple[str, Any]:
    paths = (spec.name, *spec.aliases)
    for path in paths:
        extracted = _get_path(value, path)
        if extracted is not _MISSING:
            return path or "$", extracted

    if spec_count == 1 and _is_whole_payload_spec(spec.name):
        return spec.name or "$", value
    if spec_count == 1 and not isinstance(value, ABCMapping) and not _is_sequence_container(value):
        return spec.name or "$", value
    if _is_sequence_container(value) and index < len(value):
        return str(index), value[index]
    return spec.name or "$", _MISSING


def _get_path(value: Any, path: str) -> Any:
    if _is_whole_payload_spec(path):
        return value
    current = value
    for part in str(path).split("."):
        if isinstance(current, ABCMapping):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if _is_sequence_container(current) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
            continue
        return _MISSING
    return current


def _shape_of(value: Any) -> Tuple[Optional[Shape], bool]:
    shape_attr = getattr(value, "shape", None)
    if shape_attr is not None:
        try:
            return tuple(int(dim) for dim in tuple(shape_attr)), False
        except (TypeError, ValueError):
            return None, False
    if _is_sequence_container(value):
        return _sequence_shape(value)
    if _is_scalar(value):
        return (), False
    return None, False


def _sequence_shape(value: Sequence[Any]) -> Tuple[Optional[Shape], bool]:
    length = len(value)
    if length == 0:
        return (0,), False
    first_shape, first_ragged = _shape_of(value[0])
    if first_shape is None:
        for item in value[1:]:
            item_shape, item_ragged = _shape_of(item)
            if item_ragged or item_shape is not None:
                return None, True
        return (length,), first_ragged
    for item in value[1:]:
        item_shape, item_ragged = _shape_of(item)
        if item_ragged or item_shape != first_shape:
            return None, True
    return (length, *first_shape), False


def _dtype_of(value: Any) -> Optional[str]:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _device_of(value: Any) -> Optional[str]:
    device = getattr(value, "device", None)
    if device is None:
        return None
    return str(device)


def _dtype_matches(actual: str, expected: str) -> bool:
    return _normalize_dtype(actual) == _normalize_dtype(expected)


def _normalize_dtype(dtype: str) -> str:
    normalized = str(dtype).lower()
    for prefix in ("torch.", "numpy.", "np."):
        normalized = normalized.replace(prefix, "")
    aliases = {
        "float": "float32",
        "single": "float32",
        "double": "float64",
        "half": "float16",
        "long": "int64",
        "int": "int32",
        "bool_": "bool",
    }
    return aliases.get(normalized, normalized)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, complex, str, bytes))


def _is_sequence_container(value: Any) -> bool:
    return isinstance(value, ABCSequence) and not isinstance(value, (str, bytes, bytearray))


def _is_whole_payload_spec(path: str) -> bool:
    return path in ("", "$")


def _is_wildcard_dim(dim: DimSpec) -> bool:
    return dim is None or dim == -1 or dim in ("*", "?")


def _call_model(model: Callable[..., Any], prepared: Any) -> Any:
    if not callable(model):
        raise TypeError("model must be callable")
    with _no_grad_context():
        if isinstance(prepared, ABCMapping):
            return model(**prepared)
        if isinstance(prepared, tuple):
            return model(*prepared)
        return model(prepared)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _no_grad_context() -> Any:
    try:
        import torch
    except ImportError:
        return _NullContext()
    return torch.no_grad()


def _handle_and_return(
    result: ServingSchemaGateResult,
    on_violation: str,
) -> ServingSchemaGateResult:
    _handle_result(result, on_violation)
    return result


def _handle_result(result: ServingSchemaGateResult, on_violation: str) -> None:
    _validate_policy(on_violation)
    if result.ok or on_violation == "ignore":
        return
    if on_violation == "warn":
        warnings.warn(
            str(TensorGuardServingSchemaError(result)),
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise TensorGuardServingSchemaError(result)


def _validate_policy(on_violation: str) -> None:
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")


def _runtime_result(
    result: ServingSchemaGateResult,
    *,
    model_invoked: bool,
    postprocess_invoked: bool,
) -> ServingSchemaGateResult:
    return ServingSchemaGateResult(
        ok=result.ok,
        issues=result.issues,
        warnings=result.warnings,
        checked_paths=result.checked_paths,
        framework=result.framework,
        model_invoked=model_invoked,
        postprocess_invoked=postprocess_invoked,
    )


def _merge_results(
    framework: str,
    results: Iterable[ServingSchemaGateResult],
    *,
    model_invoked: bool,
    postprocess_invoked: bool,
) -> ServingSchemaGateResult:
    issues: List[ServingSchemaIssue] = []
    warn: List[ServingSchemaIssue] = []
    checked: List[str] = []
    for result in results:
        issues.extend(result.issues)
        warn.extend(result.warnings)
        checked.extend(result.checked_paths)
    return ServingSchemaGateResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_paths=tuple(dict.fromkeys(checked)),
        framework=framework,
        model_invoked=model_invoked,
        postprocess_invoked=postprocess_invoked,
    )


_MISSING = object()
