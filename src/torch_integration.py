"""Step 74 — torch.compile / torch.export integration.

TensorGuard's static verification can run as an *optional pre-pass* in the
compile pipeline, so a shape/device/phase bug is reported before a model is
handed to ``torch.compile`` (where the same bug surfaces as an opaque guard
failure or a deep inductor traceback).

Three entry points:

* :func:`verify_module` — verify a *live* ``nn.Module`` instance (source is
  recovered with ``inspect.getsource``), returning the usual ``AnalysisResult``.
* :func:`guarded_compile` — verify first (raise/warn on a real bug), then return
  ``torch.compile(model, **kwargs)``.  If ``torch.compile`` is unavailable on the
  running interpreter, the verified model is returned unchanged so the pre-pass
  value is delivered regardless.
* :func:`make_tensorguard_backend` — a ``torch.compile`` backend that verifies
  the captured module and then delegates to an inner backend, i.e. verification
  literally inside the compile pipeline.

On a violation the pre-pass raises :class:`TensorGuardViolation`, whose ``bugs``
attribute carries the structured findings.
"""

from __future__ import annotations

import inspect
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_IMPORT_PRELUDE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"

_CLASS_HEADER_RE = re.compile(r"^(\s*class\s+\w+\s*)\([^)]*\)(\s*:)", re.MULTILINE)


def _rewrite_bases_to_nn_module(src: str) -> str:
    """Rewrite the first class header's bases to ``nn.Module``.

    A live instance may be an ``nn.Module`` *subclass* whose declared base is a
    framework class (e.g. ``pl.LightningModule``) the static analyzer doesn't
    recognise.  Since we already know ``isinstance(model, nn.Module)``, retarget
    the base to ``nn.Module`` so verification analyses the model's ``forward``.
    """
    return _CLASS_HEADER_RE.sub(r"\1(nn.Module)\2", src, count=1)


class TensorGuardViolation(RuntimeError):
    """Raised by the compile pre-pass when verification finds a real bug."""

    def __init__(self, bugs: List[Any], message: Optional[str] = None):
        self.bugs = bugs
        if message is None:
            head = "; ".join(
                (getattr(b, "message", "") or "").splitlines()[0] for b in bugs[:3]
            )
            more = "" if len(bugs) <= 3 else f" (+{len(bugs) - 3} more)"
            message = (
                f"TensorGuard found {len(bugs)} verification issue(s) before "
                f"compiling: {head}{more}"
            )
        super().__init__(message)


class TensorGuardDynamicShapeError(ValueError):
    """Raised when a ``torch.export`` dynamic-shape contract contradicts TG."""


class TensorGuardAOTPackageError(ValueError):
    """Raised when an AOTInductor package contract is rejected before packaging."""

    def __init__(self, issues: Sequence["AOTPackageIssue"]):
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected AOTInductor packaging with "
            f"{len(self.issues)} issue(s): {details}{more}"
        )


@dataclass(frozen=True)
class AOTPackageIssue:
    """A precise pre-package contract violation for AOTInductor artifacts."""

    category: str
    message: str
    input_name: Optional[str] = None
    op_name: Optional[str] = None


@dataclass(frozen=True)
class AOTPackageGateResult:
    """Result of TensorGuard's AOTInductor package gate."""

    ok: bool
    issues: Tuple[AOTPackageIssue, ...]
    checked_ops: Tuple[str, ...] = ()
    dynamic_guard_count: int = 0


_DimRange = Tuple[Optional[int], Optional[int]]
_DynamicKey = Tuple[str, _DimRange]
_AxisDynamic = Tuple[str, _DimRange, str, int, bool]

_AOT_UNSUPPORTED_LOWERING_BASES = frozenset({
    # Tuple-valued/data-dependent ATen ops that are not admitted by the stable
    # AOT package gate.  This is intentionally a denylist: TensorGuard's own
    # analysis allowlist is not an Inductor lowering oracle.
    "linalg_svd",
    "linalg_eig",
    "linalg_eigh",
    "svd",
    "eig",
    "unique",
    "unique_dim",
    "nonzero",
    "bincount",
    "histc",
})


def module_source(model: Any) -> Optional[str]:
    """Recover importable source for a live ``nn.Module`` instance, or None."""
    try:
        cls_src = inspect.getsource(type(model))
    except (OSError, TypeError):
        return None
    try:
        import torch.nn as _nn

        if isinstance(model, _nn.Module):
            cls_src = _rewrite_bases_to_nn_module(cls_src)
    except Exception:
        pass
    return _IMPORT_PRELUDE + cls_src


def verify_module(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    soundness_mode: str = "balanced",
):
    """Statically verify a live module instance; returns ``AnalysisResult`` or None.

    Returns ``None`` (abstain) when the source cannot be recovered — e.g. a model
    defined in a REPL or built dynamically — mirroring the decorator's behaviour.
    """
    from src.api import verify_architecture

    source = module_source(model)
    if source is None:
        return None
    return verify_architecture(
        source, input_shapes=input_shapes, soundness_mode=soundness_mode
    )


def _real_bugs(result: Any) -> List[Any]:
    if result is None:
        return []
    if str(getattr(result, "verdict", "")).upper().endswith("UNSAFE"):
        return list(getattr(result, "bugs", None) or [])
    # Some result shapes carry bugs without an UNSAFE verdict string; be lenient.
    verdict = str(getattr(result, "verdict", "")).upper()
    if verdict in ("UNSAFE", "BUG", "FAIL"):
        return list(getattr(result, "bugs", None) or [])
    return []


def _check(
    model: Any,
    input_shapes: Optional[Dict[str, Tuple]],
    on_violation: str,
    soundness_mode: str,
):
    """Run the pre-pass; raise/warn per ``on_violation``. Returns the result."""
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(
            f"on_violation must be raise/warn/ignore, got {on_violation!r}"
        )
    result = verify_module(
        model, input_shapes=input_shapes, soundness_mode=soundness_mode
    )
    bugs = _real_bugs(result)
    if bugs:
        if on_violation == "raise":
            raise TensorGuardViolation(bugs)
        if on_violation == "warn":
            warnings.warn(
                TensorGuardViolation(bugs).args[0],
                stacklevel=2,
            )
        # "ignore" → drop through
    return result


def guarded_compile(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **compile_kwargs: Any,
):
    """Verify *model* as a pre-pass, then return ``torch.compile(model, …)``.

    ``on_violation`` is ``"raise"`` (default), ``"warn"`` or ``"ignore"``.  If
    ``torch.compile`` is unavailable (e.g. an unsupported interpreter) the
    verified model is returned unchanged with a warning, so the verification
    pre-pass always runs.
    """
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")
    _check(model, input_shapes, on_violation, soundness_mode)

    import torch

    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile unavailable; returning the verified model.")
        return model
    try:
        return torch.compile(model, **compile_kwargs)
    except (RuntimeError, NotImplementedError) as exc:
        warnings.warn(
            f"torch.compile failed ({exc}); returning the verified model."
        )
        return model


def make_tensorguard_backend(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    inner: Any = None,
):
    """A ``torch.compile`` backend that verifies *model* then delegates.

    Usage::

        backend = make_tensorguard_backend(model, input_shapes={"x": ("b", 10)})
        compiled = torch.compile(model, backend=backend)

    The verification runs once, on the first compiled invocation; on a real bug
    it raises :class:`TensorGuardViolation` from inside the compile pipeline.
    ``inner`` is an optional inner backend ``(gm, example_inputs) -> callable``;
    when omitted the eager ``gm.forward`` is used.
    """
    state = {"checked": False}

    def backend(gm: Any, example_inputs: Any):
        if not state["checked"]:
            _check(model, input_shapes, on_violation, soundness_mode)
            state["checked"] = True
        if inner is not None:
            return inner(gm, example_inputs)
        return getattr(gm, "forward", gm)

    return backend


def verify_aot_package_contract(
    model: Any,
    example_args: Tuple,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    dynamic_shapes: Any = None,
    exported_program: Any = None,
    require_contiguous_inputs: bool = True,
    allowed_dtypes: Optional[Iterable[Any]] = None,
    allowed_devices: Optional[Iterable[Any]] = None,
    allow_unsupported_ops: bool = False,
    require_dynamic_shape_guards: bool = True,
    check_inputs: bool = True,
    check_exported_program: bool = True,
) -> AOTPackageGateResult:
    """Validate AOTInductor package preconditions without invoking the packager.

    The gate is deliberately conservative and concrete: it checks the example
    tensors that specialise the AOT artifact (layout/dtype/device), then checks
    the already-exported program for preserved dynamic-shape guards and a small
    denylist of ATen ops that this packaging path does not admit.  It does not
    conflate TensorGuard's analysis coverage with Inductor lowering coverage.
    """
    args = example_args if isinstance(example_args, tuple) else (example_args,)
    issues: List[AOTPackageIssue] = []
    if check_inputs:
        issues.extend(
            _aot_input_contract_issues(
                args,
                require_contiguous_inputs=require_contiguous_inputs,
                allowed_dtypes=allowed_dtypes,
                allowed_devices=allowed_devices,
            )
        )

    checked_ops: Tuple[str, ...] = ()
    dynamic_guard_count = 0
    if check_exported_program and exported_program is not None:
        if require_dynamic_shape_guards and _has_recognized_dynamic_dims(
            model, args, dynamic_shapes
        ):
            dynamic_guard_count = len(
                getattr(exported_program, "range_constraints", {}) or {}
            )
            if dynamic_guard_count == 0:
                issues.append(
                    AOTPackageIssue(
                        category="dynamic_shape_guard",
                        message=(
                            "dynamic_shapes declares torch.export.Dim axes, but "
                            "the exported program carries no range constraints"
                        ),
                    )
                )
        if not allow_unsupported_ops:
            op_issues, checked_ops = _aot_unsupported_lowering_issues(exported_program)
            issues.extend(op_issues)

    return AOTPackageGateResult(
        ok=not issues,
        issues=tuple(issues),
        checked_ops=checked_ops,
        dynamic_guard_count=dynamic_guard_count,
    )


def _handle_aot_gate_result(
    result: AOTPackageGateResult,
    on_violation: str,
) -> None:
    if result.ok or on_violation == "ignore":
        return
    error = TensorGuardAOTPackageError(result.issues)
    if on_violation == "raise":
        raise error
    if on_violation == "warn":
        warnings.warn(str(error), stacklevel=2)


def _flatten_tensor_args(value: Any, prefix: str = "arg0") -> List[Tuple[str, Any]]:
    try:
        import torch
    except Exception:  # pragma: no cover
        return []
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (tuple, list)):
        items: List[Tuple[str, Any]] = []
        for index, child in enumerate(value):
            items.extend(_flatten_tensor_args(child, f"{prefix}.{index}"))
        return items
    if isinstance(value, dict):
        items = []
        for key in sorted(value, key=str):
            items.extend(_flatten_tensor_args(value[key], f"{prefix}.{key}"))
        return items
    return []


def _normalize_dtype_set(values: Optional[Iterable[Any]]) -> Optional[Set[Any]]:
    if values is None:
        return None
    try:
        import torch
    except Exception:  # pragma: no cover
        return set(values)
    normalized: Set[Any] = set()
    for value in values:
        if isinstance(value, torch.dtype):
            normalized.add(value)
            continue
        text = str(value)
        normalized.add(getattr(torch, text.removeprefix("torch."), value))
    return normalized


def _normalize_device_set(values: Optional[Iterable[Any]]) -> Optional[Set[str]]:
    if values is None:
        return None
    normalized: Set[str] = set()
    for value in values:
        device = getattr(value, "type", None)
        normalized.add(str(device or value))
    return normalized


def _aot_input_contract_issues(
    args: Tuple[Any, ...],
    *,
    require_contiguous_inputs: bool,
    allowed_dtypes: Optional[Iterable[Any]],
    allowed_devices: Optional[Iterable[Any]],
) -> List[AOTPackageIssue]:
    try:
        import torch
    except Exception:  # pragma: no cover
        return []

    dtype_policy = _normalize_dtype_set(allowed_dtypes)
    device_policy = _normalize_device_set(allowed_devices)
    issues: List[AOTPackageIssue] = []
    seen_device_types: Set[str] = set()
    tensor_items: List[Tuple[str, Any]] = []
    for index, arg in enumerate(args):
        tensor_items.extend(_flatten_tensor_args(arg, f"arg{index}"))

    for name, tensor in tensor_items:
        layout = getattr(tensor, "layout", None)
        if layout is not torch.strided:
            issues.append(
                AOTPackageIssue(
                    category="input_layout",
                    input_name=name,
                    message=(
                        f"{name} has layout {layout}; AOTInductor packages in "
                        "this gate require dense strided example inputs"
                    ),
                )
            )
        elif require_contiguous_inputs and not tensor.is_contiguous():
            issues.append(
                AOTPackageIssue(
                    category="input_layout",
                    input_name=name,
                    message=(
                        f"{name} is non-contiguous with stride {tuple(tensor.stride())}; "
                        "package with a contiguous example input or opt out of "
                        "the contiguous-input gate"
                    ),
                )
            )

        if dtype_policy is not None and tensor.dtype not in dtype_policy:
            allowed = ", ".join(sorted(str(dtype) for dtype in dtype_policy))
            issues.append(
                AOTPackageIssue(
                    category="input_dtype",
                    input_name=name,
                    message=f"{name} has dtype {tensor.dtype}; allowed dtypes: {allowed}",
                )
            )
        elif tensor.is_complex() or getattr(tensor, "is_quantized", False):
            issues.append(
                AOTPackageIssue(
                    category="input_dtype",
                    input_name=name,
                    message=(
                        f"{name} has dtype {tensor.dtype}; complex and quantized "
                        "example inputs are rejected before AOT packaging"
                    ),
                )
            )

        device = tensor.device
        seen_device_types.add(device.type)
        if device.type == "meta":
            issues.append(
                AOTPackageIssue(
                    category="input_device",
                    input_name=name,
                    message=f"{name} is on the meta device, which cannot be packaged",
                )
            )
        if device_policy is not None and str(device) not in device_policy and device.type not in device_policy:
            allowed = ", ".join(sorted(device_policy))
            issues.append(
                AOTPackageIssue(
                    category="input_device",
                    input_name=name,
                    message=f"{name} is on {device}; allowed devices: {allowed}",
                )
            )

    if len(seen_device_types) > 1:
        issues.append(
            AOTPackageIssue(
                category="input_device",
                message=(
                    "AOTInductor package examples span multiple device types "
                    f"{sorted(seen_device_types)}; package one target device at a time"
                ),
            )
        )
    return issues


def _has_recognized_dynamic_dims(model: Any, args: Tuple[Any, ...], dynamic_shapes: Any) -> bool:
    if dynamic_shapes is None:
        return False
    names = _forward_param_names(model, args)
    for index, name in enumerate(names):
        spec = _dynamic_spec_for_input(dynamic_shapes, name, index)
        for _axis, dim in _iter_axis_specs(spec):
            if _is_export_dim(dim):
                return True
    return False


def _aot_op_display_name(target: Any) -> str:
    module = getattr(target, "__module__", "") or ""
    name = getattr(target, "__name__", None) or str(target)
    if module and module not in {"builtins", "operator"}:
        return f"{module}.{name}"
    return name


def _aot_op_base_name(target: Any) -> str:
    name = getattr(target, "__name__", None) or getattr(target, "_opname", None) or str(target)
    name = name.split(".")[0]
    if "::" in name:
        name = name.split("::")[-1]
    return name


def _aot_unsupported_lowering_issues(
    exported_program: Any,
) -> Tuple[List[AOTPackageIssue], Tuple[str, ...]]:
    graph_module = getattr(exported_program, "graph_module", None)
    graph = getattr(graph_module, "graph", None)
    if graph is None:
        return [], ()
    issues: List[AOTPackageIssue] = []
    checked_ops: List[str] = []
    seen_unsupported: Set[str] = set()
    for node in graph.nodes:
        if getattr(node, "op", None) != "call_function":
            continue
        target = getattr(node, "target", None)
        display = _aot_op_display_name(target)
        checked_ops.append(display)
        base = _aot_op_base_name(target)
        if base in _AOT_UNSUPPORTED_LOWERING_BASES and display not in seen_unsupported:
            seen_unsupported.add(display)
            issues.append(
                AOTPackageIssue(
                    category="unsupported_lowering",
                    op_name=display,
                    message=(
                        f"{display} is not admitted by TensorGuard's stable "
                        "AOTInductor package lowering gate"
                    ),
                )
            )
    return issues, tuple(checked_ops)


def verify_exported_program(
    model: Any,
    example_args: Tuple,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    dynamic_shapes: Any = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
):
    """Verify a module as a pre-pass, then ``torch.export.export`` it.

    Parity with :func:`guarded_onnx_export`: verification is the **first** side
    effect and ``on_violation`` defaults to ``"raise"``, so a real bug becomes
    one :class:`TensorGuardViolation` *before* the tracer runs (where the same
    bug would surface as an opaque export error or a silently wrong graph).
    When ``input_shapes`` is omitted it is inferred from the example tensor
    ``example_args`` against the ``forward`` signature, so the shape that is
    verified is the shape that is exported.  If ``dynamic_shapes`` is supplied,
    TensorGuard validates the common ``torch.export.Dim`` forms against the same
    input-shape contract before tracing: inconsistent ranges, repeated-symbol
    equality mismatches, and integer-multiple derived dimensions fail as
    :class:`TensorGuardDynamicShapeError`.  Returns the ``ExportedProgram``.
    """
    inferred_input_shapes = input_shapes is None
    args = example_args if isinstance(example_args, tuple) else (example_args,)
    if input_shapes is None:
        input_shapes = _infer_shapes_from_args(model, args)
    input_shapes = _validate_export_dynamic_shapes(
        model,
        args,
        input_shapes,
        dynamic_shapes,
        inferred_input_shapes=inferred_input_shapes,
    )
    _check(model, input_shapes, on_violation, soundness_mode)
    import torch

    if dynamic_shapes is None:
        return torch.export.export(model, args)
    return torch.export.export(model, args, dynamic_shapes=dynamic_shapes)


def guarded_aot_package(
    model: Any,
    example_args: Tuple,
    *,
    package_path: Optional[str] = None,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    dynamic_shapes: Any = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    inductor_configs: Optional[Dict[str, Any]] = None,
    aot_require_contiguous_inputs: bool = True,
    aot_allowed_dtypes: Optional[Iterable[Any]] = None,
    aot_allowed_devices: Optional[Iterable[Any]] = None,
    aot_allow_unsupported_ops: bool = False,
    aot_require_dynamic_shape_guards: bool = True,
):
    """Verify *model*, then AOTInductor-compile and package it.

    The packaging analogue of :func:`guarded_onnx_export`: TensorGuard's static
    verification runs **before** ``torch.export.export`` /
    ``torch._inductor.aoti_compile_and_package``, so a real shape/device/phase
    bug is reported as one :class:`TensorGuardViolation` *before* any artifact is
    written to ``package_path`` — instead of a deep Inductor compile error or a
    packaged-but-wrong ``.pt2``.  Shapes are inferred from ``example_args`` when
    ``input_shapes`` is omitted (parity with the ONNX/export gates).

    Returns the path to the compiled ``.pt2`` package (the string
    ``aoti_compile_and_package`` returns).
    """
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")
    args = example_args if isinstance(example_args, tuple) else (example_args,)
    _handle_aot_gate_result(
        verify_aot_package_contract(
            model,
            args,
            input_shapes=input_shapes,
            dynamic_shapes=dynamic_shapes,
            require_contiguous_inputs=aot_require_contiguous_inputs,
            allowed_dtypes=aot_allowed_dtypes,
            allowed_devices=aot_allowed_devices,
            allow_unsupported_ops=aot_allow_unsupported_ops,
            require_dynamic_shape_guards=aot_require_dynamic_shape_guards,
            check_exported_program=False,
        ),
        on_violation,
    )

    ep = verify_exported_program(
        model,
        args,
        input_shapes=input_shapes,
        dynamic_shapes=dynamic_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    _handle_aot_gate_result(
        verify_aot_package_contract(
            model,
            args,
            input_shapes=input_shapes,
            dynamic_shapes=dynamic_shapes,
            exported_program=ep,
            require_contiguous_inputs=aot_require_contiguous_inputs,
            allowed_dtypes=aot_allowed_dtypes,
            allowed_devices=aot_allowed_devices,
            allow_unsupported_ops=aot_allow_unsupported_ops,
            require_dynamic_shape_guards=aot_require_dynamic_shape_guards,
            check_inputs=False,
        ),
        on_violation,
    )
    import torch

    return torch._inductor.aoti_compile_and_package(
        ep, package_path=package_path, inductor_configs=inductor_configs
    )


def _forward_param_names(model: Any, args: Tuple[Any, ...]) -> List[str]:
    try:
        params = list(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return [f"arg{i}" for i in range(len(args))]
    if len(params) < len(args):
        params.extend(f"arg{i}" for i in range(len(params), len(args)))
    return params


def _is_export_dim(value: Any) -> bool:
    return (
        hasattr(value, "min")
        and hasattr(value, "max")
        and hasattr(value, "__name__")
    )


def _finite_bound(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    if text in {"int_oo", "oo", "inf", "Infinity"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _format_range(lo: Optional[int], hi: Optional[int]) -> str:
    return f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"


def _dim_bounds(dim: Any) -> _DimRange:
    return (
        _finite_bound(getattr(dim, "min", None)),
        _finite_bound(getattr(dim, "max", None)),
    )


def _dim_relation(dim: Any) -> Tuple[str, str, int, bool]:
    """Return ``(display_name, root_name, integer_factor, is_derived)``."""
    name = str(getattr(dim, "__name__", dim)).replace(" ", "")
    root = getattr(dim, "root", None)
    if root is None:
        return name, name, 1, False
    root_name = str(getattr(root, "__name__", root)).replace(" ", "")
    escaped = re.escape(root_name)
    for pat in (rf"^(\d+)\*{escaped}$", rf"^{escaped}\*(\d+)$"):
        m = re.match(pat, name)
        if m:
            return name, root_name, int(m.group(1)), True
    return name, root_name, 1, True


def _dynamic_spec_for_input(dynamic_shapes: Any, name: str, index: int) -> Any:
    if dynamic_shapes is None:
        return None
    if isinstance(dynamic_shapes, dict):
        return dynamic_shapes.get(name)
    if isinstance(dynamic_shapes, (tuple, list)) and index < len(dynamic_shapes):
        return dynamic_shapes[index]
    return None


def _iter_axis_specs(spec: Any):
    if isinstance(spec, dict):
        for axis, dim in spec.items():
            if isinstance(axis, int):
                yield axis, dim
        return
    if isinstance(spec, (tuple, list)):
        for axis, dim in enumerate(spec):
            yield axis, dim


def _parse_tg_dim(value: Any) -> Tuple[str, Any, Optional[str], int]:
    """Parse TensorGuard's input-shape atom.

    TensorGuard currently stores shape constraints as tuple atoms.  This parser
    therefore treats ``2*b`` as a named relation that the export contract must
    also name; it does not claim the core verifier has an affine arithmetic
    language for input specs.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return "concrete", value, None, 1
    text = str(value).replace(" ", "")
    if re.fullmatch(r"-?\d+", text):
        return "concrete", int(text), None, 1
    m = re.fullmatch(r"(\d+)\*([A-Za-z_]\w*)", text)
    if m:
        return "derived", text, m.group(2), int(m.group(1))
    m = re.fullmatch(r"([A-Za-z_]\w*)\*(\d+)", text)
    if m:
        return "derived", f"{m.group(2)}*{m.group(1)}", m.group(1), int(m.group(2))
    if re.fullmatch(r"[A-Za-z_]\w*", text):
        return "symbol", text, text, 1
    return "opaque", text, text, 1


def _validate_export_dynamic_shapes(
    model: Any,
    args: Tuple[Any, ...],
    input_shapes: Optional[Dict[str, Tuple]],
    dynamic_shapes: Any,
    *,
    inferred_input_shapes: bool,
) -> Optional[Dict[str, Tuple]]:
    """Check common ``torch.export.Dim`` contracts before export tracing.

    Unknown/nested ``dynamic_shapes`` forms are left to PyTorch's own validator;
    recognized forms are checked early so invalid contracts cannot reach the
    tracer or AOT packager.
    """
    if dynamic_shapes is None or input_shapes is None:
        return input_shapes

    names = _forward_param_names(model, args)
    refined_shapes: Dict[str, List[Any]] = {
        name: list(shape) for name, shape in input_shapes.items()
    }
    errors: List[str] = []
    axis_dynamic: Dict[Tuple[str, int], _AxisDynamic] = {}
    axis_examples: Dict[Tuple[str, int], int] = {}
    root_examples: Dict[str, Tuple[int, str, int]] = {}

    for index, (name, value) in enumerate(zip(names, args)):
        shape = refined_shapes.get(name)
        actual_shape = getattr(value, "shape", None)
        spec = _dynamic_spec_for_input(dynamic_shapes, name, index)
        if shape is None or actual_shape is None or spec is None:
            continue
        rank = len(shape)
        actual_rank = len(actual_shape)
        for raw_axis, dim in _iter_axis_specs(spec):
            if not _is_export_dim(dim):
                continue
            axis = raw_axis if raw_axis >= 0 else rank + raw_axis
            if axis < 0 or axis >= rank or axis >= actual_rank:
                errors.append(
                    f"equality: dynamic_shapes for {name}[{raw_axis}] has no "
                    f"matching TensorGuard axis in rank-{rank} input"
                )
                continue
            example_size = int(actual_shape[axis])
            lo, hi = _dim_bounds(dim)
            display, root, factor, is_derived = _dim_relation(dim)
            if lo is not None and hi is not None and lo > hi:
                errors.append(
                    f"min/max: Dim {display!r} has invalid range {_format_range(lo, hi)}"
                )
            if (
                (lo is not None and example_size < lo)
                or (hi is not None and example_size > hi)
            ):
                errors.append(
                    f"min/max: example {name}[{axis}]={example_size} is outside "
                    f"Dim {display!r} range {_format_range(lo, hi)}"
                )

            if inferred_input_shapes:
                refined_shapes[name][axis] = (
                    f"{factor}*{root}" if is_derived and factor != 1 else root
                )

            kind, tg_value, tg_root, tg_factor = _parse_tg_dim(
                refined_shapes[name][axis]
            )
            if kind == "concrete":
                concrete = int(tg_value)
                if lo != concrete or hi != concrete:
                    errors.append(
                        f"min/max: TensorGuard fixes {name}[{axis}]={concrete}, "
                        f"but export Dim {display!r} allows {_format_range(lo, hi)}"
                    )
            elif is_derived:
                if kind != "derived" or tg_root != root or tg_factor != factor:
                    errors.append(
                        f"divisibility: export declares {name}[{axis}] as "
                        f"{display!r}, but TensorGuard input_shapes uses "
                        f"{refined_shapes[name][axis]!r}"
                    )
                if factor > 1 and example_size % factor != 0:
                    errors.append(
                        f"divisibility: example {name}[{axis}]={example_size} is "
                        f"not divisible by derived Dim factor {factor}"
                    )
            elif kind == "derived":
                errors.append(
                    f"divisibility: TensorGuard input_shapes uses "
                    f"{refined_shapes[name][axis]!r}, but export Dim {display!r} "
                    "does not encode that integer-multiple relation"
                )

            axis_examples[(name, axis)] = example_size
            axis_dynamic[(name, axis)] = (display, (lo, hi), root, factor, is_derived)
            if not is_derived:
                previous = root_examples.get(root)
                if previous is not None and previous[0] != example_size:
                    prev_size, prev_name, prev_axis = previous
                    errors.append(
                        f"equality: export Dim {root!r} appears on "
                        f"{prev_name}[{prev_axis}]={prev_size} and "
                        f"{name}[{axis}]={example_size}"
                    )
                else:
                    root_examples[root] = (example_size, name, axis)

    for (name, axis), (display, _bounds, root, factor, is_derived) in axis_dynamic.items():
        if is_derived and factor > 1 and root in root_examples:
            expected = factor * root_examples[root][0]
            actual = axis_examples[(name, axis)]
            if actual != expected:
                errors.append(
                    f"divisibility: {name}[{axis}]={actual} should equal "
                    f"{factor}*{root}={expected} for export Dim {display!r}"
                )

    tg_to_dyn: Dict[str, List[Optional[_DynamicKey]]] = {}
    dyn_to_tg: Dict[_DynamicKey, List[str]] = {}
    for index, (name, value) in enumerate(zip(names, args)):
        shape = refined_shapes.get(name)
        actual_shape = getattr(value, "shape", None)
        if shape is None or actual_shape is None:
            continue
        for axis, tg_dim in enumerate(shape):
            kind, tg_value, _root, _factor = _parse_tg_dim(tg_dim)
            if kind not in {"symbol", "derived", "opaque"}:
                continue
            tg_key = str(tg_value)
            dyn = axis_dynamic.get((name, axis))
            dyn_key = None if dyn is None else (dyn[0], dyn[1])
            tg_to_dyn.setdefault(tg_key, []).append(dyn_key)
            if dyn_key is not None:
                dyn_to_tg.setdefault(dyn_key, []).append(tg_key)

    for tg_key, dyn_keys in tg_to_dyn.items():
        concrete_dyns = {d for d in dyn_keys if d is not None}
        if not concrete_dyns:
            continue
        if any(d is None for d in dyn_keys):
            errors.append(
                f"equality: TensorGuard symbol {tg_key!r} appears on multiple "
                "axes, but export dynamic_shapes does not attach the same Dim "
                "to every occurrence"
            )
        if len(concrete_dyns) > 1:
            formatted = ", ".join(
                f"{name}{_format_range(*bounds)}"
                for name, bounds in sorted(concrete_dyns)
            )
            errors.append(
                f"equality: TensorGuard symbol {tg_key!r} maps to inconsistent "
                f"export Dims ({formatted})"
            )

    for dyn_key, tg_keys in dyn_to_tg.items():
        unique_tg = set(tg_keys)
        if len(unique_tg) > 1:
            name, bounds = dyn_key
            errors.append(
                f"equality: export Dim {name!r}{_format_range(*bounds)} is shared "
                f"by distinct TensorGuard symbols {sorted(unique_tg)}"
            )

    if errors:
        raise TensorGuardDynamicShapeError(
            "Invalid torch.export dynamic_shapes contract:\n- "
            + "\n- ".join(errors)
        )
    return {name: tuple(shape) for name, shape in refined_shapes.items()}


def _infer_shapes_from_args(
    model: Any, args: Any
) -> Optional[Dict[str, Tuple]]:
    """Map example positional tensor ``args`` to ``forward`` parameter names.

    Returns ``{param_name: tuple(shape)}`` for the tensor arguments, with the
    batch (leading) dim symbolised as ``"b"`` so the verifier reasons over a
    symbolic batch rather than a single concrete value.  Returns ``None`` if the
    signature cannot be read.
    """
    if not isinstance(args, (tuple, list)):
        args = (args,)
    try:
        params = list(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return None
    shapes: Dict[str, Tuple] = {}
    for name, value in zip(params, args):
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        dims = list(shape)
        if dims:
            dims[0] = "b"
        shapes[name] = tuple(dims)
    return shapes or None


def guarded_onnx_export(
    model: Any,
    args: Any,
    f: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    check_model: bool = True,
    **export_kwargs: Any,
):
    """Verify *model* as a pre-pass, then ``torch.onnx.export`` it.

    A bad shape/device/phase bug becomes a single :class:`TensorGuardViolation`
    *before* anything is written to ``f`` — instead of a confusing tracer error
    or, worse, a silently malformed ONNX graph.  Verification is the **first**
    side effect, so on a violation with ``on_violation="raise"`` the export sink
    ``f`` (path or file-like) is never touched.

    ``args`` is the usual ``torch.onnx.export`` example input (a tensor or a
    tuple of them).  When ``input_shapes`` is omitted it is inferred from the
    tensor ``args`` against the ``forward`` signature, so the shape that is
    *verified* is the shape that is *exported*.

    The legacy (TorchScript) exporter is selected by default
    (``dynamo=False``) for broad interpreter compatibility — the Dynamo-based
    exporter is unavailable on some interpreters (e.g. Python 3.14).  Pass
    ``dynamo=True`` explicitly to opt into the Dynamo/``onnxscript`` exporter
    where it is available.

    When ``check_model=True`` (default) the exported proto is parsed back and
    validated with ``onnx.checker.check_model`` as a post-export assertion, so a
    structurally invalid graph fails loudly at export time rather than at load
    time in a downstream runtime.  The check runs for both ``BytesIO``/file-like
    and path sinks; it is skipped only when ``onnx`` is not importable.
    """
    if input_shapes is None:
        input_shapes = _infer_shapes_from_args(model, args)
    _check(model, input_shapes, on_violation, soundness_mode)

    import torch

    export_kwargs.setdefault("dynamo", False)
    result = torch.onnx.export(model, args, f, **export_kwargs)
    if check_model:
        _post_export_check(f)
    return result


def _post_export_check(f: Any) -> None:
    """Parse the just-written ONNX sink and run ``onnx.checker.check_model``.

    Silently no-ops if ``onnx`` is unavailable.  Raises whatever
    ``onnx.checker.check_model`` raises (``onnx.checker.ValidationError``) on a
    structurally invalid graph.
    """
    try:
        import onnx  # type: ignore
    except Exception:
        return
    # File-like sink (e.g. io.BytesIO): parse the written bytes and check.
    getvalue = getattr(f, "getvalue", None)
    if callable(getvalue):
        data = getvalue()
        if data:
            onnx.checker.check_model(onnx.load_from_string(bytes(data)))
        return
    # Filesystem sink: check by path so large/external-data models validate the
    # way onnx.checker intends (without loading the whole proto into memory).
    if isinstance(f, (str, bytes)) or hasattr(f, "__fspath__"):
        onnx.checker.check_model(os.fspath(f))
