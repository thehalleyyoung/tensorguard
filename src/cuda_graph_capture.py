"""CUDA graph capture eligibility diagnostics.

The checker is deliberately conservative: it proves common static-input and
forward-body preconditions before a caller enters ``torch.cuda.graph``.  It does
not perform capture itself, so it remains useful on CPU-only CI while still
surfacing CUDA-only requirements as opt-in diagnostics.
"""

from __future__ import annotations

import ast
import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


Shape = Tuple[int, ...]
Stride = Tuple[int, ...]


@dataclass(frozen=True)
class CudaGraphCaptureIssue:
    """One actionable CUDA graph capture eligibility finding."""

    category: str
    message: str
    op_name: Optional[str] = None
    input_name: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    severity: str = "error"
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class CudaGraphInputSignature:
    """Static replay signature for one tensor argument."""

    name: str
    shape: Shape
    dtype: str
    device: str
    stride: Stride
    requires_grad: bool = False
    data_ptr: Optional[int] = None


@dataclass(frozen=True)
class CudaGraphCaptureEligibilityResult:
    """Result of TensorGuard's CUDA graph capture eligibility gate."""

    ok: bool
    issues: Tuple[CudaGraphCaptureIssue, ...]
    warnings: Tuple[CudaGraphCaptureIssue, ...] = ()
    checked_ops: Tuple[str, ...] = ()
    input_signatures: Tuple[CudaGraphInputSignature, ...] = ()
    source_available: bool = False
    fx_trace_available: bool = False
    verification_scope: str = "inputs"


class TensorGuardCudaGraphCaptureError(ValueError):
    """Raised when CUDA graph capture preconditions are rejected."""

    def __init__(self, issues: Sequence[CudaGraphCaptureIssue]):
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected CUDA graph capture with "
            f"{len(self.issues)} issue(s): {details}{more}"
        )


_DYNAMIC_FACTORY_NAMES = frozenset({
    "torch.arange",
    "torch.as_tensor",
    "torch.empty",
    "torch.empty_like",
    "torch.eye",
    "torch.full",
    "torch.full_like",
    "torch.linspace",
    "torch.ones",
    "torch.ones_like",
    "torch.rand",
    "torch.randn",
    "torch.randn_like",
    "torch.rand_like",
    "torch.randint",
    "torch.tensor",
    "torch.zeros",
    "torch.zeros_like",
})
_DYNAMIC_FACTORY_SUFFIXES = (
    ".new_empty",
    ".new_full",
    ".new_ones",
    ".new_tensor",
    ".new_zeros",
)
_DATA_DEPENDENT_NAMES = frozenset({
    "torch.argwhere",
    "torch.bincount",
    "torch.histc",
    "torch.masked_select",
    "torch.nonzero",
    "torch.unique",
    "torch.unique_consecutive",
})
_DATA_DEPENDENT_SUFFIXES = (
    ".argwhere",
    ".masked_select",
    ".nonzero",
    ".unique",
    ".unique_consecutive",
)
_UNSUPPORTED_NAMES = frozenset({
    "print",
    "torch.cuda.Event",
    "torch.cuda.Stream",
    "torch.cuda.current_stream",
    "torch.cuda.memory_allocated",
    "torch.cuda.synchronize",
})
_UNSUPPORTED_SUFFIXES = (
    ".copy_",
    ".data_ptr",
    ".item",
    ".numpy",
    ".random_",
    ".resize_",
    ".tolist",
)


def verify_cuda_graph_capture_eligibility(
    model: Any,
    example_args: Sequence[Any] = (),
    *,
    example_kwargs: Optional[Mapping[str, Any]] = None,
    replay_args: Optional[Sequence[Any]] = None,
    replay_kwargs: Optional[Mapping[str, Any]] = None,
    source: Optional[str] = None,
    require_cuda_inputs: bool = False,
    require_contiguous_inputs: bool = False,
    require_static_input_addresses: bool = False,
    check_fx: bool = True,
) -> CudaGraphCaptureEligibilityResult:
    """Diagnose whether a module is eligible for CUDA graph capture.

    ``example_args`` describe the tensors used during capture.  When
    ``replay_args`` are provided, every replay tensor must match the capture
    tensor's shape, dtype, device, and stride; optionally it must reuse the same
    storage address.  ``require_cuda_inputs`` is opt-in so CPU-only CI can still
    prove all source/shape diagnostics without a GPU.
    """

    args = _as_tuple(example_args)
    kwargs = dict(example_kwargs or {})
    issues: List[CudaGraphCaptureIssue] = []
    warn: List[CudaGraphCaptureIssue] = []

    capture_tensors = _flatten_tensor_args(args, kwargs)
    signatures = tuple(_signature(name, tensor) for name, tensor in capture_tensors)
    _check_capture_inputs(
        capture_tensors,
        issues,
        require_cuda_inputs=require_cuda_inputs,
        require_contiguous_inputs=require_contiguous_inputs,
    )

    if replay_args is not None or replay_kwargs is not None:
        replay_tensors = _flatten_tensor_args(
            _as_tuple(replay_args or ()),
            dict(replay_kwargs or {}),
        )
        _check_replay_inputs(
            capture_tensors,
            replay_tensors,
            issues,
            require_static_input_addresses=require_static_input_addresses,
        )

    src, source_available = _resolve_source(model, source)
    checked_ops: List[str] = []
    if src is None:
        issues.append(
            CudaGraphCaptureIssue(
                category="source_unavailable",
                message=(
                    "model source could not be recovered, so TensorGuard cannot "
                    "prove CUDA graph capture eligibility"
                ),
                suggestion="Define the module in a file or pass source=... to this checker.",
            )
        )
    else:
        _scan_source(src, issues, checked_ops)

    fx_trace_available = False
    if check_fx:
        fx_trace_available = _scan_fx_graph(model, issues, checked_ops)

    scope_parts = ["inputs"]
    if source_available:
        scope_parts.append("source")
    if fx_trace_available:
        scope_parts.append("fx")

    return CudaGraphCaptureEligibilityResult(
        ok=not issues,
        issues=tuple(issues),
        warnings=tuple(warn),
        checked_ops=tuple(dict.fromkeys(checked_ops)),
        input_signatures=signatures,
        source_available=source_available,
        fx_trace_available=fx_trace_available,
        verification_scope="+".join(scope_parts),
    )


def guarded_cuda_graph_capture(
    model: Any,
    example_args: Sequence[Any] = (),
    *,
    example_kwargs: Optional[Mapping[str, Any]] = None,
    capture: Optional[Callable[..., Any]] = None,
    on_violation: str = "raise",
    dry_run: bool = False,
    **verify_kwargs: Any,
) -> Any:
    """Run the eligibility gate before invoking a user-supplied capture callable.

    ``capture`` is a callable representing the caller's capture body.  It is not
    invoked when the gate finds issues and ``on_violation="raise"``.  With
    ``dry_run=True`` the eligibility result is returned without invoking capture.
    """

    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")
    result = verify_cuda_graph_capture_eligibility(
        model,
        example_args,
        example_kwargs=example_kwargs,
        **verify_kwargs,
    )
    if result.issues and on_violation != "ignore":
        error = TensorGuardCudaGraphCaptureError(result.issues)
        if on_violation == "raise":
            raise error
        warnings.warn(str(error), stacklevel=2)
    if dry_run or capture is None:
        return result
    return capture(*_as_tuple(example_args), **dict(example_kwargs or {}))


def _as_tuple(value: Optional[Sequence[Any]]) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _flatten_tensor_args(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for index, value in enumerate(args):
        _flatten_tensors(value, f"arg{index}", items)
    for key in sorted(kwargs, key=str):
        _flatten_tensors(kwargs[key], f"kw.{key}", items)
    return items


def _flatten_tensors(value: Any, path: str, out: List[Tuple[str, Any]]) -> None:
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    if isinstance(value, torch.Tensor):
        out.append((path, value))
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _flatten_tensors(child, f"{path}.{index}", out)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _flatten_tensors(value[key], f"{path}.{key}", out)


def _signature(name: str, tensor: Any) -> CudaGraphInputSignature:
    data_ptr: Optional[int]
    try:
        data_ptr = int(tensor.data_ptr())
    except Exception:
        data_ptr = None
    return CudaGraphInputSignature(
        name=name,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        stride=tuple(int(dim) for dim in tensor.stride()),
        requires_grad=bool(getattr(tensor, "requires_grad", False)),
        data_ptr=data_ptr,
    )


def _check_capture_inputs(
    tensors: Sequence[Tuple[str, Any]],
    issues: List[CudaGraphCaptureIssue],
    *,
    require_cuda_inputs: bool,
    require_contiguous_inputs: bool,
) -> None:
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    for name, tensor in tensors:
        layout = getattr(tensor, "layout", None)
        if layout is not torch.strided:
            issues.append(
                CudaGraphCaptureIssue(
                    category="input_layout",
                    input_name=name,
                    message=(
                        f"{name} has layout {layout}; CUDA graph capture "
                        "requires dense strided tensor inputs in this gate"
                    ),
                    suggestion="Materialize a dense strided capture buffer before capture.",
                )
            )
        if require_contiguous_inputs and not tensor.is_contiguous():
            issues.append(
                CudaGraphCaptureIssue(
                    category="input_layout",
                    input_name=name,
                    message=(
                        f"{name} is non-contiguous with stride {tuple(tensor.stride())}; "
                        "the configured capture policy requires contiguous inputs"
                    ),
                    suggestion="Copy into a persistent contiguous capture buffer.",
                )
            )
        if require_cuda_inputs and getattr(tensor.device, "type", None) != "cuda":
            issues.append(
                CudaGraphCaptureIssue(
                    category="input_device",
                    input_name=name,
                    message=f"{name} is on {tensor.device}; CUDA graph capture requires CUDA tensors",
                    suggestion="Move the static capture input buffer to CUDA before capture.",
                )
            )


def _check_replay_inputs(
    capture_tensors: Sequence[Tuple[str, Any]],
    replay_tensors: Sequence[Tuple[str, Any]],
    issues: List[CudaGraphCaptureIssue],
    *,
    require_static_input_addresses: bool,
) -> None:
    if len(capture_tensors) != len(replay_tensors):
        issues.append(
            CudaGraphCaptureIssue(
                category="static_input_mismatch",
                message=(
                    f"capture saw {len(capture_tensors)} tensor input(s), but replay "
                    f"provided {len(replay_tensors)}"
                ),
                suggestion="Replay must feed the same tensor argument structure used during capture.",
            )
        )
        return

    for (cap_name, cap), (rep_name, rep) in zip(capture_tensors, replay_tensors):
        cap_sig = _signature(cap_name, cap)
        rep_sig = _signature(rep_name, rep)
        checks = (
            ("shape", cap_sig.shape, rep_sig.shape),
            ("dtype", cap_sig.dtype, rep_sig.dtype),
            ("device", cap_sig.device, rep_sig.device),
            ("stride", cap_sig.stride, rep_sig.stride),
        )
        for field, expected, actual in checks:
            if expected != actual:
                issues.append(
                    CudaGraphCaptureIssue(
                        category="static_input_mismatch",
                        input_name=cap_name,
                        message=(
                            f"replay tensor {rep_name} has {field} {actual}, "
                            f"but capture tensor {cap_name} used {expected}"
                        ),
                        suggestion=(
                            "Copy replay data into the persistent capture buffer "
                            "without changing shape, dtype, device, or stride."
                        ),
                    )
                )
        if (
            require_static_input_addresses
            and cap_sig.data_ptr is not None
            and rep_sig.data_ptr is not None
            and cap_sig.data_ptr != rep_sig.data_ptr
        ):
            issues.append(
                CudaGraphCaptureIssue(
                    category="static_input_address",
                    input_name=cap_name,
                    message=(
                        f"replay tensor {rep_name} uses storage address {rep_sig.data_ptr}, "
                        f"but capture tensor {cap_name} used {cap_sig.data_ptr}"
                    ),
                    suggestion="Replay into the same long-lived input tensor captured by the graph.",
                )
            )


def _resolve_source(model: Any, source: Optional[str]) -> Tuple[Optional[str], bool]:
    if source is not None:
        return textwrap.dedent(source), True
    try:
        from src.torch_integration import module_source

        recovered = module_source(model)
    except Exception:
        recovered = None
    if recovered is None:
        return None, False
    return recovered, True


def _scan_source(
    source: str,
    issues: List[CudaGraphCaptureIssue],
    checked_ops: List[str],
) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        issues.append(
            CudaGraphCaptureIssue(
                category="source_parse",
                message=f"model source could not be parsed: {exc.msg}",
                line=exc.lineno,
                column=exc.offset,
                suggestion="Pass parseable module source or use a file-defined nn.Module.",
            )
        )
        return

    methods = _reachable_methods_from_forward(tree)
    if not methods:
        issues.append(
            CudaGraphCaptureIssue(
                category="source_unavailable",
                message="no forward method was found in the recovered model source",
                suggestion="Pass source=... containing the nn.Module forward method.",
            )
        )
        return

    for method in methods:
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    checked_ops.append(name)
                    _classify_source_call(name, node, issues)
            elif isinstance(node, ast.Subscript) and _looks_like_data_dependent_subscript(node):
                issues.append(
                    CudaGraphCaptureIssue(
                        category="data_dependent_shape",
                        op_name="boolean_indexing",
                        line=getattr(node, "lineno", None),
                        column=getattr(node, "col_offset", None),
                        message=(
                            "boolean/tensor-mask indexing can produce a data-dependent "
                            "output shape, which is not replay-stable for CUDA graphs"
                        ),
                        suggestion="Replace mask indexing with a fixed-shape masked operation or preallocate outputs.",
                    )
                )


def _reachable_methods_from_forward(tree: ast.AST) -> Tuple[ast.FunctionDef, ...]:
    class_methods: Dict[str, ast.FunctionDef] = {}
    for class_def in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        methods = {
            node.name: node
            for node in class_def.body
            if isinstance(node, ast.FunctionDef) and node.name != "__init__"
        }
        if "forward" in methods:
            class_methods = methods
            break
    if "forward" not in class_methods:
        return ()

    visited: Set[str] = set()
    ordered: List[ast.FunctionDef] = []

    def visit(name: str) -> None:
        if name in visited or name not in class_methods:
            return
        visited.add(name)
        method = class_methods[name]
        ordered.append(method)
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                called = _self_method_name(node.func)
                if called is not None:
                    visit(called)

    visit("forward")
    return tuple(ordered)


def _self_method_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "self":
            return func.attr
    return None


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        if base is None:
            return node.attr
        return f"{base}.{node.attr}"
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return None


def _classify_source_call(
    name: str,
    node: ast.Call,
    issues: List[CudaGraphCaptureIssue],
) -> None:
    if _is_dynamic_factory(name):
        issues.append(
            CudaGraphCaptureIssue(
                category="dynamic_allocation",
                op_name=name,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
                message=(
                    f"{name} allocates a fresh tensor inside the captured region; "
                    "CUDA graph replay expects allocation-free steady-state inputs"
                ),
                suggestion="Hoist the allocation to a persistent buffer and reuse it during replay.",
            )
        )
        return
    if _is_data_dependent(name) or (name == "torch.where" and len(node.args) == 1):
        issues.append(
            CudaGraphCaptureIssue(
                category="data_dependent_shape",
                op_name=name,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
                message=(
                    f"{name} can produce an output shape that depends on tensor values, "
                    "so captured replay may not be shape-stable"
                ),
                suggestion="Use fixed-shape alternatives or preallocate a maximum-size output with an explicit length.",
            )
        )
        return
    if _is_unsupported(name):
        issues.append(
            CudaGraphCaptureIssue(
                category="unsupported_op",
                op_name=name,
                line=getattr(node, "lineno", None),
                column=getattr(node, "col_offset", None),
                message=f"{name} is not admitted by TensorGuard's CUDA graph capture gate",
                suggestion="Remove host synchronization or mutation from the captured forward path.",
            )
        )


def _looks_like_data_dependent_subscript(node: ast.Subscript) -> bool:
    slc = node.slice
    if isinstance(slc, ast.Compare):
        return True
    if isinstance(slc, ast.BoolOp):
        return True
    if isinstance(slc, ast.Tuple):
        return any(isinstance(elt, (ast.Compare, ast.BoolOp)) for elt in slc.elts)
    return False


def _scan_fx_graph(
    model: Any,
    issues: List[CudaGraphCaptureIssue],
    checked_ops: List[str],
) -> bool:
    try:
        import torch.fx as fx

        graph_module = fx.symbolic_trace(model)
    except Exception as exc:
        issues.append(
            CudaGraphCaptureIssue(
                category="unsupported_graph",
                message=(
                    f"torch.fx could not trace the model for capture diagnostics "
                    f"({type(exc).__name__}: {exc})"
                ),
                suggestion="Remove data-dependent Python control flow or pass a traceable wrapper.",
            )
        )
        return False

    for node in graph_module.graph.nodes:
        if node.op in {"call_function", "call_method", "call_module"}:
            name = _fx_name(node.target)
            checked_ops.append(name)
            _classify_fx_op(name, node.name, issues)
    return True


def _fx_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if name:
        module = getattr(target, "__module__", "")
        return f"{module}.{name}" if module and module != "builtins" else name
    return str(target)


def _classify_fx_op(
    name: str,
    node_name: str,
    issues: List[CudaGraphCaptureIssue],
) -> None:
    if _is_dynamic_factory(name):
        issues.append(
            CudaGraphCaptureIssue(
                category="dynamic_allocation",
                op_name=name,
                message=(
                    f"FX node {node_name} calls {name}, a fresh allocation in the captured region"
                ),
                suggestion="Hoist allocation out of the captured forward path.",
            )
        )
    elif _is_data_dependent(name):
        issues.append(
            CudaGraphCaptureIssue(
                category="data_dependent_shape",
                op_name=name,
                message=f"FX node {node_name} calls {name}, whose output shape can depend on values",
                suggestion="Use fixed-shape alternatives before CUDA graph capture.",
            )
        )
    elif _is_unsupported(name):
        issues.append(
            CudaGraphCaptureIssue(
                category="unsupported_op",
                op_name=name,
                message=f"FX node {node_name} calls unsupported capture op {name}",
                suggestion="Remove host sync, pointer inspection, or in-place resizing from capture.",
            )
        )


def _is_dynamic_factory(name: str) -> bool:
    short = _short_name(name)
    qualified = name if name.startswith("torch.") else f"torch.{short}"
    return qualified in _DYNAMIC_FACTORY_NAMES or any(name.endswith(s) for s in _DYNAMIC_FACTORY_SUFFIXES)


def _is_data_dependent(name: str) -> bool:
    short = _short_name(name)
    qualified = name if name.startswith("torch.") else f"torch.{short}"
    return qualified in _DATA_DEPENDENT_NAMES or any(name.endswith(s) for s in _DATA_DEPENDENT_SUFFIXES)


def _is_unsupported(name: str) -> bool:
    short = _short_name(name)
    qualified = name if name.startswith("torch.") else f"torch.{short}"
    return name in _UNSUPPORTED_NAMES or qualified in _UNSUPPORTED_NAMES or any(
        name.endswith(s) for s in _UNSUPPORTED_SUFFIXES
    )


def _short_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]

