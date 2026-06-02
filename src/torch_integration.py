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
from typing import Any, Dict, List, Optional, Tuple

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


def verify_exported_program(
    model: Any,
    example_args: Tuple,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
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
    verified is the shape that is exported.  Returns the ``ExportedProgram``.
    """
    if input_shapes is None:
        input_shapes = _infer_shapes_from_args(model, example_args)
    _check(model, input_shapes, on_violation, soundness_mode)
    import torch

    args = example_args if isinstance(example_args, tuple) else (example_args,)
    return torch.export.export(model, args)


def guarded_aot_package(
    model: Any,
    example_args: Tuple,
    *,
    package_path: Optional[str] = None,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    inductor_configs: Optional[Dict[str, Any]] = None,
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
    ep = verify_exported_program(
        model,
        example_args,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    import torch

    return torch._inductor.aoti_compile_and_package(
        ep, package_path=package_path, inductor_configs=inductor_configs
    )


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
