"""Mixed-precision and autocast placement gates for PyTorch modules.

``torch.autocast`` is intentionally dynamic: FX graphs do not retain context
manager boundaries, and backend policies differ across CPU, CUDA, and MPS.  This
module therefore checks an explicit autocast configuration against a live module
or FX graph: backend dtype support, fp16 GradScaler requirements, trainable
parameter dtypes, visible explicit casts, and the dtype flow induced by the AMP
allowlist under the declared context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "AutocastTraceEntry",
    "MixedPrecisionIssue",
    "MixedPrecisionVerdict",
    "verify_mixed_precision",
    "verify_mixed_precision_fx",
]

_UNKNOWN = "unknown"
_FLOAT16 = "float16"
_BFLOAT16 = "bfloat16"
_FLOAT32 = "float32"
_FLOAT64 = "float64"
_REDUCED_FLOATS = {_FLOAT16, _BFLOAT16}
_FLOAT_ORDER = {_FLOAT16: 1, _BFLOAT16: 1, _FLOAT32: 2, _FLOAT64: 3}

_BACKEND_RULES = {
    "cpu": {"default": _BFLOAT16, "supported": {_BFLOAT16, _FLOAT16}},
    "cuda": {"default": _FLOAT16, "supported": {_FLOAT16, _BFLOAT16}},
    "mps": {"default": _FLOAT16, "supported": {_FLOAT16, _BFLOAT16}},
}

_LOWER_PRECISION_MODULES = {
    "Linear",
    "Bilinear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
}
_CUDA_FP32_MODULES = {
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "LayerNorm",
    "LocalResponseNorm",
    "Softmax",
    "LogSoftmax",
}
_LOWER_PRECISION_FUNCTIONS = {
    "addmm",
    "baddbmm",
    "bmm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "linear",
    "matmul",
    "mm",
    "scaled_dot_product_attention",
}
_FP32_FUNCTIONS = {
    "binary_cross_entropy_with_logits",
    "cholesky",
    "cholesky_ex",
    "cross_entropy",
    "eig",
    "eigvals",
    "eigh",
    "eigvalsh",
    "inv",
    "inverse",
    "layer_norm",
    "log_softmax",
    "mse_loss",
    "nll_loss",
    "norm",
    "qr",
    "softmax",
    "solve",
    "svd",
    "svdvals",
}
_PROMOTE_FUNCTIONS = {"add", "mul", "sub", "truediv"}
_LOWER_PRECISION_METHODS = {"bmm", "matmul", "mm"}
_CUDA_FP32_METHODS = {"log_softmax", "softmax"}
_CAST_METHOD_TO_DTYPE = {
    "bfloat16": _BFLOAT16,
    "float": _FLOAT32,
    "half": _FLOAT16,
}


@dataclass(frozen=True)
class MixedPrecisionIssue:
    """One statically visible mixed-precision contract violation."""

    kind: str
    message: str
    location: str = "module"


@dataclass(frozen=True)
class AutocastTraceEntry:
    """Predicted dtype transfer for one FX node under an explicit AMP context."""

    location: str
    op: str
    policy: str
    input_dtype: str
    parameter_dtype: str
    predicted_dtype: str


@dataclass(frozen=True)
class MixedPrecisionVerdict:
    """Result of checking an eager module or FX graph under autocast."""

    ok: bool
    issues: Tuple[MixedPrecisionIssue, ...] = ()
    warnings: Tuple[str, ...] = ()
    mode: str = "eager"
    backend: str = "cpu"
    autocast_dtype: str = _BFLOAT16
    trace: Tuple[AutocastTraceEntry, ...] = ()

    def has_issue(self, kind: str) -> bool:
        return any(issue.kind == kind for issue in self.issues)

    def dtype_at(self, location: str) -> Optional[str]:
        for entry in self.trace:
            if entry.location == location:
                return entry.predicted_dtype
        return None


def verify_mixed_precision(
    model: Any,
    *,
    backend: str = "cpu",
    autocast_dtype: Any = None,
    input_dtypes: Optional[Mapping[str, Any]] = None,
    training: bool = False,
    uses_grad_scaler: Optional[bool] = None,
    require_grad_scaler: bool = True,
    require_fp32_parameters: bool = True,
    require_float_output: bool = False,
    mode: str = "auto",
) -> MixedPrecisionVerdict:
    """Check a module or FX graph under an explicit autocast configuration.

    FX does not preserve ``with torch.autocast(...)`` regions, so ``backend`` and
    ``autocast_dtype`` are caller-supplied assumptions.  ``mode="auto"`` analyzes
    an existing FX graph when present and otherwise attempts ``symbolic_trace``.
    """

    if mode not in {"auto", "eager", "fx"}:
        raise ValueError("mode must be one of 'auto', 'eager', or 'fx'")

    backend_name = _canonical_backend(backend)
    dtype_name = _resolve_autocast_dtype(backend_name, autocast_dtype)
    issues = list(_configuration_issues(backend_name, dtype_name, training, uses_grad_scaler, require_grad_scaler))
    issues.extend(_parameter_issues(model, training=training, require_fp32_parameters=require_fp32_parameters))
    warnings = list(_backend_warnings(backend_name))

    should_trace = mode in {"auto", "fx"}
    trace: Tuple[AutocastTraceEntry, ...] = ()
    if should_trace:
        try:
            graph_module = model if hasattr(model, "graph") else _symbolic_trace(model)
            fx_verdict = _verify_fx_graph(
                graph_module,
                backend=backend_name,
                autocast_dtype=dtype_name,
                input_dtypes=input_dtypes,
                require_float_output=require_float_output,
            )
            issues.extend(fx_verdict.issues)
            warnings.extend(fx_verdict.warnings)
            trace = fx_verdict.trace
        except Exception as exc:  # pragma: no cover - model/version dependent
            if mode == "fx":
                raise
            warnings.append(f"fx mixed-precision refinement skipped: {type(exc).__name__}: {exc}")

    final_issues = _dedupe_issues(issues)
    return MixedPrecisionVerdict(
        not final_issues,
        final_issues,
        _dedupe_strings(warnings),
        "fx" if hasattr(model, "graph") or mode == "fx" else "eager",
        backend_name,
        dtype_name,
        trace,
    )


def verify_mixed_precision_fx(
    graph_module: Any,
    *,
    backend: str = "cpu",
    autocast_dtype: Any = None,
    input_dtypes: Optional[Mapping[str, Any]] = None,
    training: bool = False,
    uses_grad_scaler: Optional[bool] = None,
    require_grad_scaler: bool = True,
    require_fp32_parameters: bool = True,
    require_float_output: bool = False,
) -> MixedPrecisionVerdict:
    """Check an FX graph as if it were run under the supplied autocast context."""

    backend_name = _canonical_backend(backend)
    dtype_name = _resolve_autocast_dtype(backend_name, autocast_dtype)
    issues = list(_configuration_issues(backend_name, dtype_name, training, uses_grad_scaler, require_grad_scaler))
    issues.extend(_parameter_issues(graph_module, training=training, require_fp32_parameters=require_fp32_parameters))
    warnings = list(_backend_warnings(backend_name))
    fx_verdict = _verify_fx_graph(
        graph_module,
        backend=backend_name,
        autocast_dtype=dtype_name,
        input_dtypes=input_dtypes,
        require_float_output=require_float_output,
    )
    issues.extend(fx_verdict.issues)
    warnings.extend(fx_verdict.warnings)
    final_issues = _dedupe_issues(issues)
    return MixedPrecisionVerdict(
        not final_issues,
        final_issues,
        _dedupe_strings(warnings + list(fx_verdict.warnings)),
        "fx",
        backend_name,
        dtype_name,
        fx_verdict.trace,
    )


def _verify_fx_graph(
    graph_module: Any,
    *,
    backend: str,
    autocast_dtype: str,
    input_dtypes: Optional[Mapping[str, Any]],
    require_float_output: bool,
) -> MixedPrecisionVerdict:
    states: Dict[Any, str] = {}
    issues: List[MixedPrecisionIssue] = []
    trace: List[AutocastTraceEntry] = []

    for node in graph_module.graph.nodes:
        location = f"fx:{node.name}"
        if node.op == "placeholder":
            states[node] = _input_dtype(node, input_dtypes)
            trace.append(AutocastTraceEntry(location, str(node.target), "input", _UNKNOWN, _UNKNOWN, states[node]))
            continue

        if node.op == "get_attr":
            states[node] = _dtype_name(_resolve_attr(graph_module, str(node.target)))
            trace.append(AutocastTraceEntry(location, str(node.target), "attribute", _UNKNOWN, _UNKNOWN, states[node]))
            continue

        if node.op == "call_module":
            submod = graph_module.get_submodule(str(node.target))
            input_dtype = _first_dtype(node.args, states)
            param_dtype = _module_parameter_dtype(submod)
            policy = _module_policy(submod, backend)
            predicted = _apply_policy(policy, input_dtype, param_dtype, autocast_dtype)
            states[node] = predicted
            trace.append(
                AutocastTraceEntry(location, _type_name(submod), policy, input_dtype, param_dtype, predicted)
            )
            issues.extend(_node_issues(policy, predicted, None, location, _type_name(submod)))
            continue

        if node.op == "call_function":
            input_dtype = _first_dtype(node.args, states)
            param_dtype = _UNKNOWN
            policy = _function_policy(node.target, backend)
            predicted = _apply_policy(policy, input_dtype, param_dtype, autocast_dtype)
            states[node] = predicted
            op = _target_name(node.target)
            trace.append(AutocastTraceEntry(location, op, policy, input_dtype, param_dtype, predicted))
            issues.extend(_node_issues(policy, predicted, None, location, op))
            continue

        if node.op == "call_method":
            input_dtype = _first_dtype(node.args, states)
            explicit_dtype = _method_cast_dtype(str(node.target), node.args, node.kwargs)
            policy = "explicit_cast" if explicit_dtype is not None else _method_policy(str(node.target), backend)
            predicted = explicit_dtype or _apply_policy(policy, input_dtype, _UNKNOWN, autocast_dtype)
            states[node] = predicted
            op = str(node.target)
            trace.append(AutocastTraceEntry(location, op, policy, input_dtype, _UNKNOWN, predicted))
            issues.extend(_node_issues(policy, predicted, explicit_dtype, location, op))
            continue

        if node.op == "output":
            output_dtype = _state_of(node.args[0] if node.args else None, states)
            states[node] = output_dtype
            trace.append(AutocastTraceEntry(location, "output", "output", output_dtype, _UNKNOWN, output_dtype))
            if require_float_output and output_dtype in _REDUCED_FLOATS:
                issues.append(
                    MixedPrecisionIssue(
                        "reduced_precision_output",
                        f"public output remains {output_dtype}; add an explicit .float() boundary if callers expect fp32",
                        location,
                    )
                )
            continue

        states[node] = _UNKNOWN

    final_issues = _dedupe_issues(issues)
    return MixedPrecisionVerdict(not final_issues, final_issues, (), "fx", backend, autocast_dtype, tuple(trace))


def _configuration_issues(
    backend: str,
    autocast_dtype: str,
    training: bool,
    uses_grad_scaler: Optional[bool],
    require_grad_scaler: bool,
) -> Tuple[MixedPrecisionIssue, ...]:
    issues: List[MixedPrecisionIssue] = []
    supported = _BACKEND_RULES[backend]["supported"]
    if autocast_dtype not in supported:
        issues.append(
            MixedPrecisionIssue(
                "unsupported_autocast_dtype",
                f"{backend} autocast supports {sorted(supported)}, not {autocast_dtype}; PyTorch disables or rejects this context",
            )
        )
    if training and require_grad_scaler and autocast_dtype == _FLOAT16 and uses_grad_scaler is not True:
        issues.append(
            MixedPrecisionIssue(
                "amp_missing_grad_scaler",
                "float16 autocast training should use torch.amp.GradScaler to avoid gradient underflow",
                "training_loop",
            )
        )
    return tuple(issues)


def _parameter_issues(model: Any, *, training: bool, require_fp32_parameters: bool) -> Tuple[MixedPrecisionIssue, ...]:
    params = list(_floating_parameters(model))
    if not params:
        return ()
    issues: List[MixedPrecisionIssue] = []
    trainable = [(name, dtype) for name, dtype, requires_grad in params if requires_grad]
    trainable_dtypes = {dtype for _, dtype in trainable if dtype != _UNKNOWN}
    if len(trainable_dtypes) > 1:
        detail = ", ".join(f"{name}:{dtype}" for name, dtype in trainable)
        issues.append(
            MixedPrecisionIssue(
                "parameter_dtype_mismatch",
                f"trainable parameters use mixed floating dtypes ({detail}); autocast does not fix parameter-to-parameter dtype drift",
            )
        )
    if training and require_fp32_parameters:
        for name, dtype in trainable:
            if dtype not in {_UNKNOWN, _FLOAT32}:
                issues.append(
                    MixedPrecisionIssue(
                        "parameter_not_fp32",
                        f"trainable parameter {name!r} is {dtype}; AMP training expects fp32 master parameters",
                        name,
                    )
                )
    return tuple(issues)


def _floating_parameters(model: Any) -> Iterable[Tuple[str, str, bool]]:
    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is None:
        return ()
    out: List[Tuple[str, str, bool]] = []
    for name, param in named_parameters():
        if _is_floating_tensor(param):
            out.append((name, _dtype_name(param), bool(getattr(param, "requires_grad", False))))
    return tuple(out)


def _symbolic_trace(model: Any) -> Any:
    from torch.fx import symbolic_trace  # type: ignore

    return symbolic_trace(model)


def _canonical_backend(backend: str) -> str:
    lowered = str(backend).lower()
    aliases = {"gpu": "cuda", "cuda:0": "cuda", "mps:0": "mps"}
    lowered = aliases.get(lowered, lowered)
    if lowered not in _BACKEND_RULES:
        raise ValueError("backend must be one of 'cpu', 'cuda', or 'mps'")
    return lowered


def _resolve_autocast_dtype(backend: str, dtype: Any) -> str:
    if dtype is None:
        return str(_BACKEND_RULES[backend]["default"])
    return _dtype_name(dtype)


def _backend_warnings(backend: str) -> Tuple[str, ...]:
    try:
        import torch  # type: ignore
    except Exception:
        return ()
    if backend == "cuda" and hasattr(torch, "cuda") and not torch.cuda.is_available():
        return ("cuda backend requested but torch.cuda.is_available() is false; only static policy checks ran",)
    if backend == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and hasattr(mps, "is_available") and not mps.is_available():
            return ("mps backend requested but torch.backends.mps.is_available() is false; only static policy checks ran",)
    return ()


def _module_policy(module: Any, backend: str) -> str:
    name = _type_name(module).split(".")[-1]
    if name in _LOWER_PRECISION_MODULES:
        return "lower_precision"
    if backend == "cuda" and name in _CUDA_FP32_MODULES:
        return "fp32"
    return "transparent"


def _function_policy(target: Any, backend: str) -> str:
    base = _target_basename(target)
    if base in _LOWER_PRECISION_FUNCTIONS:
        return "lower_precision"
    if base in _FP32_FUNCTIONS:
        if backend == "cuda" or base not in {"softmax", "log_softmax", "layer_norm"}:
            return "fp32"
    if base in _PROMOTE_FUNCTIONS:
        return "promote"
    return "transparent"


def _method_policy(target: str, backend: str) -> str:
    if target in _LOWER_PRECISION_METHODS:
        return "lower_precision"
    if backend == "cuda" and target in _CUDA_FP32_METHODS:
        return "fp32"
    return "transparent"


def _apply_policy(policy: str, input_dtype: str, parameter_dtype: str, autocast_dtype: str) -> str:
    if policy == "lower_precision":
        return autocast_dtype
    if policy == "fp32":
        return _FLOAT32
    if policy == "promote":
        return _promote_dtypes((input_dtype, parameter_dtype))
    return input_dtype if input_dtype != _UNKNOWN else parameter_dtype


def _node_issues(
    policy: str,
    predicted_dtype: str,
    explicit_dtype: Optional[str],
    location: str,
    op: str,
) -> Tuple[MixedPrecisionIssue, ...]:
    if policy == "explicit_cast" and explicit_dtype in _REDUCED_FLOATS:
        return (
            MixedPrecisionIssue(
                "explicit_reduced_precision_cast",
                f"{op} explicitly casts to {explicit_dtype}; this bypasses autocast's op-specific promotion rules",
                location,
            ),
        )
    return ()


def _method_cast_dtype(method: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Optional[str]:
    if method in _CAST_METHOD_TO_DTYPE:
        return _CAST_METHOD_TO_DTYPE[method]
    if method != "to":
        return None
    dtype = kwargs.get("dtype")
    if dtype is None:
        for arg in args[1:]:
            dtype_name = _dtype_name(arg)
            if dtype_name in _FLOAT_ORDER:
                dtype = arg
                break
    dtype_name = _dtype_name(dtype)
    return dtype_name if dtype_name in _FLOAT_ORDER else None


def _input_dtype(node: Any, input_dtypes: Optional[Mapping[str, Any]]) -> str:
    if not input_dtypes:
        return _FLOAT32
    for key in (str(node.target), str(node.name)):
        if key in input_dtypes:
            return _dtype_name(input_dtypes[key])
    return _FLOAT32


def _state_of(obj: Any, states: Mapping[Any, str]) -> str:
    if isinstance(obj, (tuple, list)):
        child = [_state_of(item, states) for item in obj]
        if not child:
            return _UNKNOWN
        if all(dtype == child[0] for dtype in child):
            return child[0]
        return _promote_dtypes(child)
    if isinstance(obj, dict):
        return _state_of(tuple(obj.values()), states)
    try:
        return states.get(obj, _UNKNOWN)
    except TypeError:
        return _UNKNOWN


def _first_dtype(args: Sequence[Any], states: Mapping[Any, str]) -> str:
    for arg in args:
        dtype = _state_of(arg, states)
        if dtype != _UNKNOWN:
            return dtype
    return _UNKNOWN


def _promote_dtypes(dtypes: Iterable[str]) -> str:
    known = [dtype for dtype in dtypes if dtype in _FLOAT_ORDER]
    if not known:
        return _UNKNOWN
    return max(known, key=lambda dtype: _FLOAT_ORDER[dtype])


def _module_parameter_dtype(module: Any) -> str:
    for _, dtype, _ in _floating_parameters(module):
        return dtype
    return _UNKNOWN


def _is_floating_tensor(value: Any) -> bool:
    try:
        is_floating_point = getattr(value, "is_floating_point", None)
        if callable(is_floating_point):
            return bool(is_floating_point())
        return _dtype_name(value) in _FLOAT_ORDER
    except Exception:
        return False


def _dtype_name(value: Any) -> str:
    if value is None:
        return _UNKNOWN
    dtype = getattr(value, "dtype", value)
    text = str(dtype)
    if text.startswith("torch."):
        text = text[len("torch.") :]
    text = text.lower()
    aliases = {
        "float": _FLOAT32,
        "float32": _FLOAT32,
        "torch.float": _FLOAT32,
        "torch.float32": _FLOAT32,
        "half": _FLOAT16,
        "float16": _FLOAT16,
        "torch.float16": _FLOAT16,
        "bfloat16": _BFLOAT16,
        "torch.bfloat16": _BFLOAT16,
        "double": _FLOAT64,
        "float64": _FLOAT64,
        "torch.float64": _FLOAT64,
    }
    return aliases.get(text, text if text in _FLOAT_ORDER else _UNKNOWN)


def _resolve_attr(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _target_name(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", repr(target))
    return f"{module}.{name}" if module else name


def _target_basename(target: Any) -> str:
    return _target_name(target).split(".")[-1]


def _type_name(obj: Any) -> str:
    return type(obj).__qualname__


def _dedupe_issues(issues: Sequence[MixedPrecisionIssue]) -> Tuple[MixedPrecisionIssue, ...]:
    seen = set()
    out: List[MixedPrecisionIssue] = []
    for issue in issues:
        key = (issue.kind, issue.location, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return tuple(out)


def _dedupe_strings(items: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
