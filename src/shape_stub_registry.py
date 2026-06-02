"""Pluggable shape-stub registry for third-party layers (Step 41).

TensorGuard's core analysis only understands ``torch.nn`` layers and locally
defined ``nn.Module`` subclasses.  Real models frequently embed *third-party*
blocks — HuggingFace ``Conv1D``, ``timm`` ``Mlp``/``DropPath``, custom attention
modules — whose source is not in the file under analysis.  Without help, such a
block becomes an opaque ``UNKNOWN`` layer that abstains with a fully symbolic
output, which (soundly) prevents proving the surrounding model safe.

This module lets users (and TensorGuard's own built-ins) **register a shape
stub** for a third-party class *by name* — no import of the third-party package
is required, because matching is purely textual on the constructor call.  A stub
declares:

* the *positional constructor argument names* (so ``Conv1D(nf, nx)`` binds
  ``nf``/``nx``), with optional defaults, and
* a **transfer function** ``(input_shape, params) -> (output_shape, error)`` in
  exactly the same contract as the built-in ``nn`` propagators.

The transfer function is responsible for *soundness*: it should return an error
when the input is incompatible, and otherwise a precise (possibly symbolic)
output shape.  Returning ``(None, None)`` means "I cannot say" and the engine
falls back to a sound symbolic abstention.

Public API
----------
``register_shape_stub(class_name, transfer, arg_names=(), defaults=None)``
``register_shape_preserving(class_name)``
``register_last_dim_linear(class_name, in_arg, out_arg, ...)``
``get_shape_stub(class_name) -> ShapeStub | None``
``clear_user_stubs()``   — remove user registrations, keep built-ins
``registered_stub_names() -> list[str]``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from src.tensor_shapes import ShapeDim, TensorShape

# A transfer function: (input_shape, resolved_params) -> (output_shape, error).
# Mirrors the built-in nn propagator contract used by model_checker.
TransferFn = Callable[
    [TensorShape, Dict[str, Any]],
    Tuple[Optional[TensorShape], Optional[str]],
]


@dataclass(frozen=True)
class ShapeStub:
    """A registered shape transfer for a third-party layer class."""
    name: str
    transfer: TransferFn
    arg_names: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    is_builtin: bool = False

    def bind_params(
        self,
        positional: Tuple[Any, ...],
        keywords: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bind constructor call arguments to declared parameter names.

        *positional* and *keywords* hold already-resolved values (concrete ints,
        symbolic strings, or ``None`` when unresolved).  Defaults fill any gaps.
        """
        params: Dict[str, Any] = dict(self.defaults)
        for i, val in enumerate(positional):
            if i < len(self.arg_names):
                params[self.arg_names[i]] = val
        for k, val in keywords.items():
            params[k] = val
        return params


_STUB_REGISTRY: Dict[str, ShapeStub] = {}


def register_shape_stub(
    class_name: str,
    transfer: TransferFn,
    arg_names: Tuple[str, ...] = (),
    defaults: Optional[Dict[str, Any]] = None,
    *,
    _is_builtin: bool = False,
) -> None:
    """Register *transfer* as the shape stub for the class named *class_name*.

    ``class_name`` is matched against the constructor used in ``__init__`` — both
    the bare name (``Conv1D``) and dotted forms (``transformers.Conv1D``) are
    accepted at lookup time, so register under whichever name appears in source.
    """
    _STUB_REGISTRY[class_name] = ShapeStub(
        name=class_name,
        transfer=transfer,
        arg_names=tuple(arg_names),
        defaults=dict(defaults or {}),
        is_builtin=_is_builtin,
    )


def get_shape_stub(class_name: Optional[str]) -> Optional[ShapeStub]:
    """Look up a stub by full or trailing class name."""
    if not class_name:
        return None
    if class_name in _STUB_REGISTRY:
        return _STUB_REGISTRY[class_name]
    # Allow dotted constructors (``a.b.Conv1D``) to match a bare registration.
    tail = class_name.rsplit(".", 1)[-1]
    return _STUB_REGISTRY.get(tail)


def registered_stub_names() -> list:
    return sorted(_STUB_REGISTRY.keys())


def clear_user_stubs() -> None:
    """Drop user-registered stubs, retaining built-ins."""
    for name in list(_STUB_REGISTRY.keys()):
        if not _STUB_REGISTRY[name].is_builtin:
            del _STUB_REGISTRY[name]


# ──────────────────────────────────────────────────────────────────────────
# Convenience registrars for the two most common third-party shape contracts.
# ──────────────────────────────────────────────────────────────────────────
def register_shape_preserving(class_name: str, *, _is_builtin: bool = False) -> None:
    """Register a block that returns its input shape unchanged (e.g. DropPath,
    StochasticDepth, residual wrappers, most normalization variants)."""
    def _transfer(inp: TensorShape, params: Dict[str, Any]):
        return inp, None
    register_shape_stub(class_name, _transfer, _is_builtin=_is_builtin)


def register_last_dim_linear(
    class_name: str,
    in_arg: str,
    out_arg: str,
    arg_names: Tuple[str, ...],
    defaults: Optional[Dict[str, Any]] = None,
    *,
    out_defaults_to_in: bool = False,
    _is_builtin: bool = False,
) -> None:
    """Register a "linear-like" block mapping ``(*, in) -> (*, out)``.

    *in_arg*/*out_arg* name the constructor parameters carrying the input and
    output feature sizes.  When *out_defaults_to_in* is set and the output size
    is unspecified, the output feature size falls back to the input size (this
    matches ``timm``'s ``Mlp(in_features, hidden_features=None, out_features=None)``
    contract, which returns ``out_features or in_features``).
    """
    def _transfer(inp: TensorShape, params: Dict[str, Any]):
        if inp.ndim < 1:
            return None, f"{class_name} requires at least 1D input"
        last = inp.dims[-1]
        in_feat = params.get(in_arg)
        if (isinstance(in_feat, int) and not last.is_symbolic
                and isinstance(last.value, int) and last.value != in_feat):
            return None, (
                f"{class_name} expects last dim={in_feat}, got {last.value}"
            )
        out_feat = params.get(out_arg)
        if out_feat is None and out_defaults_to_in:
            out_feat = in_feat
        if out_feat is None:
            # Unresolved output size: sound symbolic abstention on last dim.
            new = inp.dims[:-1] + (ShapeDim(f"_stub_{class_name}_out"),)
            return TensorShape(new), None
        new = inp.dims[:-1] + (ShapeDim(out_feat),)
        return TensorShape(new), None

    register_shape_stub(class_name, _transfer, arg_names=arg_names,
                        defaults=dict(defaults or {}), _is_builtin=_is_builtin)


# ──────────────────────────────────────────────────────────────────────────
# Built-in stubs for well-known third-party blocks with stable contracts.
# These are textual-name matches; no third-party import occurs.
# ──────────────────────────────────────────────────────────────────────────
def _install_builtins() -> None:
    # HuggingFace transformers GPT-2 ``Conv1D(nf, nx)``: a linear layer that maps
    # the last dim ``nx`` -> ``nf`` (note: argument order is OUTPUT first).
    register_last_dim_linear(
        "Conv1D", in_arg="nx", out_arg="nf",
        arg_names=("nf", "nx"), _is_builtin=True,
    )

    # timm ``Mlp(in_features, hidden_features=None, out_features=None, ...)``:
    # returns ``out_features or in_features`` on the last dim.
    register_last_dim_linear(
        "Mlp", in_arg="in_features", out_arg="out_features",
        arg_names=("in_features", "hidden_features", "out_features"),
        out_defaults_to_in=True, _is_builtin=True,
    )

    # Shape-preserving regularizers / wrappers commonly seen in timm & friends.
    for nm in ("DropPath", "StochasticDepth", "LayerScale", "Identity"):
        register_shape_preserving(nm, _is_builtin=True)


_install_builtins()
