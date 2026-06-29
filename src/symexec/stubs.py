"""Third-party stub library: shape/arity summaries for common libraries
(roadmap Step 83).

The interpreter models a hand-picked set of torch constructors, ``nn`` layers and
tensor methods precisely; *every other* library call — ``F.relu(x)``,
``torch.softmax(x, -1)``, ``np.zeros((3, 4))``, ``torch.flatten(x, 1)`` — currently
falls through to ``Top``.  Once a value becomes ``Top`` the analysis can no longer
reason about its rank/shape, so a downstream ``matmul``/``view``/layer mismatch
that depends on that value goes unfound.

This module closes that gap with a **declarative, side-effect-free** registry of
*return-shape summaries* for the most common pure library functions.  A summary
is a transfer function ``(pos, kw) -> AbstractValue | None`` over already-evaluated
abstract argument values; it produces the *result* abstraction and **never emits a
bug**.  Keeping stubs report-free is the soundness contract: a stub can only ever
*add* shape knowledge that flows forward — it cannot itself fabricate a diagnostic,
and if it is unsure of the result it returns ``None`` (the caller keeps ``Top``,
i.e. abstains).  Summaries are intentionally conservative: only shape transforms
that hold for *every* runtime input are encoded; broadcasting binary ops and
shape-reordering ops with non-constant arguments degrade to "a tensor of unknown
rank" rather than guess.

Matching is alias-aware and sound: the interpreter resolves the call's leading
name through the module's actual ``import`` bindings (see
``Interpreter._import_aliases``) to a canonical dotted path before lookup, so a
stub only ever engages when the callee genuinely *is* the library function — never
because an unrelated local happens to share its name.  The module is torch-free.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .values import (
    AbstractValue,
    IntVal,
    ListVal,
    TensorVal,
    TupleVal,
)

# A summary maps already-evaluated (positional, keyword) abstract arguments to a
# result abstraction, or ``None`` to abstain (caller keeps ``Top``).
Summary = Callable[[List[AbstractValue], Dict[str, AbstractValue]], Optional[AbstractValue]]


# --------------------------------------------------------------------------- #
# Small helpers over abstract argument values                                 #
# --------------------------------------------------------------------------- #

def _first_tensor(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[TensorVal]:
    """The tensor the op operates on: first positional tensor, else ``input=``."""
    if pos and isinstance(pos[0], TensorVal):
        return pos[0]
    t = kw.get("input")
    return t if isinstance(t, TensorVal) else None


def _const_int(v: Optional[AbstractValue]) -> Optional[int]:
    if isinstance(v, IntVal):
        if v.sym is not None and v.sym.value is not None:
            return v.sym.value
        return v.const
    return None


def _shape_from_value(v: AbstractValue) -> Optional[List[Optional[object]]]:
    """Read a numpy/torch ``shape`` argument (an int, or a tuple/list of ints)
    into a list of per-dim ``SymDim`` (or ``None`` for an unknown dim), or
    ``None`` if it is not a statically-shaped int/tuple/list."""
    if isinstance(v, IntVal):
        return [v.sym]
    if isinstance(v, TupleVal) and v.exact_len:
        return [e.sym if isinstance(e, IntVal) else None for e in v.elems]
    if isinstance(v, ListVal) and v.exact_elems is not None:
        return [e.sym if isinstance(e, IntVal) else None for e in v.exact_elems]
    return None


# --------------------------------------------------------------------------- #
# Transfer functions                                                          #
# --------------------------------------------------------------------------- #

def _elementwise(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[AbstractValue]:
    """Unary, shape-preserving op (activations, unary math, ``*_like``): the
    result is a tensor with the *same* rank/shape/dtype/device as its input."""
    t = _first_tensor(pos, kw)
    if t is None:
        return None
    return TensorVal(rank=t.rank, shape=t.shape, dtype=t.dtype, device=t.device)


def _factory(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[AbstractValue]:
    """numpy ``zeros``/``ones``/``empty``/``full``: the first positional (or
    ``shape=``) gives the result shape."""
    arg = pos[0] if pos else kw.get("shape")
    if arg is None:
        return None
    dims = _shape_from_value(arg)
    if dims is None:
        return None
    return TensorVal(rank=len(dims), shape=tuple(dims))


def _flatten(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[AbstractValue]:
    """``torch.flatten(input, start_dim=0, end_dim=-1)``: collapse dims
    ``[start_dim, end_dim]`` into one.  Computes the resulting *rank* when the
    bounds are constant (or defaulted); abstains on rank when a bound is a
    non-constant runtime value."""
    t = _first_tensor(pos, kw)
    if t is None:
        return None
    if t.rank is None:
        return TensorVal(rank=None, dtype=t.dtype, device=t.device)
    rank = t.rank
    start_arg = pos[1] if len(pos) >= 2 else kw.get("start_dim")
    end_arg = pos[2] if len(pos) >= 3 else kw.get("end_dim")
    start = 0 if start_arg is None else _const_int(start_arg)
    end = -1 if end_arg is None else _const_int(end_arg)
    if start is None or end is None:  # non-constant bound: unknown result rank
        return TensorVal(rank=None, dtype=t.dtype, device=t.device)
    s = start if start >= 0 else start + rank
    e = end if end >= 0 else end + rank
    if not (0 <= s <= e < rank):  # out-of-range bounds: stay sound, drop rank
        return TensorVal(rank=None, dtype=t.dtype, device=t.device)
    new_rank = rank - (e - s)
    return TensorVal(rank=max(new_rank, 1), dtype=t.dtype, device=t.device)


def _unsqueeze(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[AbstractValue]:
    """``torch.unsqueeze(input, dim)``: inserts a size-1 dim → rank + 1."""
    t = _first_tensor(pos, kw)
    if t is None:
        return None
    if t.rank is None:
        return TensorVal(rank=None, dtype=t.dtype, device=t.device)
    return TensorVal(rank=t.rank + 1, dtype=t.dtype, device=t.device)


def _squeeze(
    pos: List[AbstractValue], kw: Dict[str, AbstractValue]
) -> Optional[AbstractValue]:
    """``torch.squeeze``: removes size-1 dims.  The exact resulting rank depends
    on runtime dim *sizes*, so we keep only "it is a tensor" (rank unknown) —
    sound, never a false shape."""
    t = _first_tensor(pos, kw)
    if t is None:
        return None
    return TensorVal(rank=None, dtype=t.dtype, device=t.device)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

# Unary, shape-preserving ops.  Registered under both ``torch.<name>`` and
# ``torch.nn.functional.<name>`` (the ``F.`` alias) where applicable, plus the
# numpy elementwise ufuncs.  Broadcasting *binary* ops (add/mul/pow/...) are
# deliberately excluded: their output shape depends on both operands.
_ELEMENTWISE_BASE = [
    # activations
    "relu", "relu6", "leaky_relu", "elu", "selu", "celu", "gelu", "silu",
    "mish", "hardtanh", "hardswish", "hardsigmoid", "hardshrink", "softshrink",
    "tanhshrink", "softsign", "softplus", "sigmoid", "logsigmoid", "tanh",
    "softmax", "log_softmax", "softmin", "glu",
    # dropout (identity on shape under inference)
    "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout",
    "feature_alpha_dropout",
    # unary math
    "exp", "expm1", "log", "log2", "log10", "log1p", "sqrt", "rsqrt",
    "reciprocal", "neg", "negative", "sign", "sgn", "floor", "ceil", "round",
    "trunc", "frac", "clamp", "clip", "clamp_min", "clamp_max",
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "asinh",
    "acosh", "atanh", "erf", "erfc", "square", "sigmoid", "isnan", "isinf",
    "isfinite", "nan_to_num",
]

# torch ops that take a tensor and preserve its shape (``*_like`` factories and
# the unary math/activations above).
_TORCH_LIKE = [
    "zeros_like", "ones_like", "empty_like", "full_like", "rand_like",
    "randn_like", "randint_like",
]

# numpy elementwise ufuncs (shape-preserving).
_NUMPY_ELEMENTWISE = [
    "exp", "expm1", "log", "log2", "log10", "log1p", "sqrt", "cbrt", "abs",
    "absolute", "fabs", "sign", "negative", "reciprocal", "square", "sin",
    "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "floor", "ceil", "round", "rint", "trunc", "clip", "isnan", "isinf",
    "isfinite", "nan_to_num",
]

_NUMPY_LIKE = ["zeros_like", "ones_like", "empty_like", "full_like"]


def _build_registry() -> Dict[str, Summary]:
    reg: Dict[str, Summary] = {}
    # torch elementwise + abs (abs lives under torch.abs / functional has no abs)
    for name in _ELEMENTWISE_BASE + ["abs", "absolute"]:
        reg[f"torch.{name}"] = _elementwise
        reg[f"torch.nn.functional.{name}"] = _elementwise
    for name in _TORCH_LIKE:
        reg[f"torch.{name}"] = _elementwise
    # rank transforms
    reg["torch.flatten"] = _flatten
    reg["torch.unsqueeze"] = _unsqueeze
    reg["torch.squeeze"] = _squeeze
    # numpy elementwise + factories
    for name in _NUMPY_ELEMENTWISE:
        reg[f"numpy.{name}"] = _elementwise
    for name in _NUMPY_LIKE:
        reg[f"numpy.{name}"] = _elementwise
    for name in ("zeros", "ones", "empty", "full"):
        reg[f"numpy.{name}"] = _factory
    return reg


#: canonical dotted function name -> shape summary.  Built once at import.
STUB_REGISTRY: Dict[str, Summary] = _build_registry()


def lookup(canonical_name: Optional[str]) -> Optional[Summary]:
    """Return the shape summary for a canonical dotted callee name, or ``None``
    if the function is not stubbed."""
    if canonical_name is None:
        return None
    return STUB_REGISTRY.get(canonical_name)
