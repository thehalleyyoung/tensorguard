"""Transfer functions for tensor methods and nn-layer calls.

Each function maps an abstract receiver/arguments to an abstract result, used by
the interpreter when it sees ``recv.method(...)``.  They are deliberately
conservative: when an effect cannot be modeled precisely the result degrades to
``TOP`` rather than guessing.  Rank tracking is the priority, because rank is
what the rank-index safety check depends on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .symdim import SymDim
from .values import (
    AbstractValue,
    BoolVal,
    IntVal,
    ListVal,
    StrVal,
    TensorVal,
    TupleVal,
    TOP,
)

__all__ = ["tensor_method", "RANK_PRESERVING", "is_tensor"]


# dtype-casting methods → the canonical torch dtype string they produce.
_DTYPE_METHOD = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "bool": "bool",
}


def _device_type(spec: str) -> str:
    """Normalise a device spec to its *type* (the part before ``:``), so that
    ``cuda`` and ``cuda:0`` compare equal but ``cpu`` and ``cuda`` differ."""
    return spec.split(":", 1)[0].strip()


def _parse_device(v: AbstractValue) -> Optional[str]:
    if isinstance(v, StrVal) and v.const:
        s = v.const.strip()
        if _device_type(s) in ("cpu", "cuda", "mps", "xpu"):
            return _device_type(s)
    return None



def is_tensor(v: AbstractValue) -> bool:
    return isinstance(v, TensorVal)


# Methods that return a tensor of the same rank as the receiver.
RANK_PRESERVING = {
    "to", "cuda", "cpu", "float", "double", "half", "long", "int", "bool",
    "detach", "clone", "contiguous", "abs", "neg", "relu", "sigmoid", "tanh",
    "softmax", "log_softmax", "exp", "log", "sqrt", "type_as", "masked_fill",
    "clamp", "clamp_min", "clamp_max", "dropout", "requires_grad_", "t_",
}


def _rank_of(v: AbstractValue) -> Optional[int]:
    return v.rank if isinstance(v, TensorVal) else None


def tensor_method(
    recv: TensorVal,
    method: str,
    args: List[AbstractValue],
    kwargs: Optional[Dict[str, AbstractValue]] = None,
) -> AbstractValue:
    """Return the abstract result of ``recv.method(*args, **kwargs)``.

    Returns ``TOP`` whenever the effect on rank/shape is not modeled.
    """
    kwargs = kwargs or {}
    r = recv.rank

    # -- device / dtype movers (rank preserving, but they UPDATE metadata) ----
    if method == "cuda":
        return TensorVal(rank=r, shape=recv.shape, dtype=recv.dtype, device="cuda")
    if method == "cpu":
        return TensorVal(rank=r, shape=recv.shape, dtype=recv.dtype, device="cpu")
    if method in _DTYPE_METHOD:
        return TensorVal(rank=r, shape=recv.shape, dtype=_DTYPE_METHOD[method],
                         device=recv.device)
    # ``detach()`` returns a new tensor sharing storage but cut from the autograd
    # graph: it never requires grad and is a leaf.  Modeling this positively lets
    # the heuristic ``backward_no_grad`` check fire on ``loss.detach().backward()``.
    if method == "detach":
        return TensorVal(rank=r, shape=recv.shape, dtype=recv.dtype,
                         device=recv.device, requires_grad=False, is_leaf=True)
    if method == "to":
        new_device, new_dtype = recv.device, recv.dtype
        dev_kw = _parse_device(kwargs.get("device")) if "device" in kwargs else None
        if dev_kw is not None:
            new_device = dev_kw
        for a in args:  # positional: .to('cuda') / .to(other_tensor) / .to(dtype)
            dev = _parse_device(a)
            if dev is not None:
                new_device = dev
            elif isinstance(a, TensorVal):  # .to(other) copies device+dtype
                if a.device is not None:
                    new_device = a.device
                if a.dtype is not None:
                    new_dtype = a.dtype
        return TensorVal(rank=r, shape=recv.shape, dtype=new_dtype, device=new_device)

    if method in RANK_PRESERVING:
        return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)

    # -- shape queries ---------------------------------------------------
    if method in ("dim", "ndimension"):
        return IntVal(sym=SymDim.const_dim(r) if r is not None else None)
    if method == "size":
        if args:  # size(d) -> int
            d = _as_int(args[0])
            if recv.shape is not None and d is not None and r is not None:
                return IntVal(sym=recv.dim(d))
            return IntVal()
        # size() with no args -> a tuple (torch.Size) of length rank
        if r is not None:
            elems = tuple(IntVal(sym=(recv.dim(i) if recv.shape is not None else None)) for i in range(r))
            return TupleVal(elems=elems, exact_len=True)
        return TupleVal(elems=(), exact_len=False)

    # -- rank changers ---------------------------------------------------
    if method == "unsqueeze":
        return TensorVal(rank=(r + 1) if r is not None else None, dtype=recv.dtype, device=recv.device)
    if method == "squeeze":
        if args:  # squeeze(dim) removes at most one dim; rank may stay or drop
            return TensorVal(rank=None, dtype=recv.dtype, device=recv.device)
        return TensorVal(rank=None, dtype=recv.dtype, device=recv.device)
    if method in ("flatten",):
        # flatten() -> 1-D; flatten(start, end) -> reduced rank (unknown precise)
        if not args:
            return TensorVal(rank=1, dtype=recv.dtype, device=recv.device)
        return TensorVal(rank=None, dtype=recv.dtype, device=recv.device)

    # -- axis movers (rank preserving) -----------------------------------
    if method in ("transpose", "swapaxes", "permute", "movedim", "moveaxis", "t", "mT"):
        return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)

    # -- reshape family --------------------------------------------------
    if method in ("view", "reshape"):
        # new rank = number of args (ignoring a single tuple arg we can't size)
        if args and all(isinstance(a, IntVal) for a in args):
            return TensorVal(rank=len(args), dtype=recv.dtype, device=recv.device)
        return TensorVal(rank=None, dtype=recv.dtype, device=recv.device)
    if method == "expand":
        if args:
            return TensorVal(rank=len(args), dtype=recv.dtype, device=recv.device)
        return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)

    # -- matmul / bmm ----------------------------------------------------
    if method in ("matmul", "bmm", "mm"):
        return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)

    # -- reductions ------------------------------------------------------
    if method in ("sum", "mean", "max", "min", "prod", "std", "var", "argmax", "argmin", "any", "all", "median", "norm", "logsumexp", "amax", "amin", "nanmean", "nansum"):
        return _reduce_rank(recv, args, kwargs, r)

    # Unknown method: be conservative.
    return TOP


def _reduce_rank(
    recv: TensorVal,
    args: List[AbstractValue],
    kwargs: Dict[str, AbstractValue],
    r: Optional[int],
) -> TensorVal:
    """Rank effect of a reduction.

    * no ``dim`` → full reduction to a 0-d scalar;
    * ``keepdim=True`` → rank preserved;
    * ``dim`` is a single axis (default ``keepdim=False``) → rank drops by 1;
    * ``dim`` is a tuple/list of ``k`` axes → rank drops by ``k``.

    Anything not pinned down keeps ``rank=None`` (sound).
    """
    dim = kwargs.get("dim")
    if dim is None and args:
        dim = args[0]  # positional dim, e.g. ``x.sum(1)``
    keepdim = _kw_bool(kwargs.get("keepdim"))
    if keepdim is None and len(args) >= 2:
        keepdim = _kw_bool(args[1])  # positional keepdim, e.g. ``x.sum(1, True)``

    if dim is None:
        # Whole-tensor reduction.  ``keepdim=True`` yields all-ones rank r.
        if keepdim is True:
            return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)
        return TensorVal(rank=0, dtype=recv.dtype, device=recv.device)

    if keepdim is True:
        return TensorVal(rank=r, dtype=recv.dtype, device=recv.device)

    dropped = _num_axes(dim)
    if dropped is not None and r is not None:
        return TensorVal(rank=max(r - dropped, 0), dtype=recv.dtype, device=recv.device)
    return TensorVal(rank=None, dtype=recv.dtype, device=recv.device)


def _num_axes(v: AbstractValue) -> Optional[int]:
    """How many axes a ``dim=`` argument removes: 1 for a scalar int, ``k`` for a
    fixed-length tuple/list of ints."""
    if isinstance(v, IntVal):
        return 1
    if isinstance(v, TupleVal) and v.exact_len:
        return len(v.elems)
    if isinstance(v, ListVal) and v.length is not None and v.exact_elems is not None:
        return v.length
    return None


def _as_int(v: AbstractValue) -> Optional[int]:
    if isinstance(v, IntVal):
        if v.sym is not None and v.sym.value is not None:
            return v.sym.value
        return v.const
    return None


def _kw_bool(v: Optional[AbstractValue]) -> Optional[bool]:
    if isinstance(v, BoolVal):
        return v.const
    return None

