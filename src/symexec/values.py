"""Abstract value lattice for the TensorGuard symbolic executor.

Every Python value flowing through the interpreter is abstracted to one of the
:class:`AbstractValue` subclasses below.  The lattice is intentionally simple
and *sound*: when two values disagree we widen toward ``Top`` (unknown) rather
than risk an unsound narrowing.  ``Top`` means "could be anything" and never
triggers a bug report on its own; ``Bottom`` marks unreachable state.

The values most load-bearing for the wild bugs we target are:

* :class:`TensorVal` — carries ``rank`` (and optionally a symbolic ``shape``),
  enough to check rank-dependent indexing (``x[-1, :, :]`` on a 2-D tensor).
* :class:`TupleVal` — carries an *exact length* when known, enough to check
  tuple-unpacking arity (``a, b = f()`` when ``f`` returns a single value).
* :class:`ModuleVal` — carries layer metadata (``in_features`` …) so dims built
  in ``__init__`` are visible in ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .symdim import SymDim

try:  # interval domain is part of the wider package; keep symexec importable
    from ..domains.intervals import Interval as IntervalT
except Exception:  # pragma: no cover - fallback when domains is unavailable
    IntervalT = None  # type: ignore


def int_const(n: int) -> "IntVal":
    """Construct an :class:`IntVal` for a known constant ``n`` (both facets)."""
    iv = IntervalT.singleton(n) if IntervalT is not None else None
    return IntVal(sym=SymDim.const_dim(n), interval=iv)


def int_range(lo: Optional[int], hi: Optional[int]) -> "IntVal":
    """Construct an :class:`IntVal` from an inclusive numeric range."""
    if IntervalT is None:
        return IntVal()
    from ..domains.intervals import Bound

    lob = Bound.finite(lo) if lo is not None else Bound.neg_inf()
    hib = Bound.finite(hi) if hi is not None else Bound.pos_inf()
    return IntVal(interval=IntervalT(lo=lob, hi=hib))


__all__ = [
    "AbstractValue",
    "Top",
    "Bottom",
    "TensorVal",
    "IntVal",
    "FloatVal",
    "BoolVal",
    "StrVal",
    "NoneVal",
    "TupleVal",
    "ListVal",
    "DictVal",
    "SetVal",
    "ModuleVal",
    "CallableVal",
    "TOP",
    "BOTTOM",
    "NONE",
    "join",
    "join_many",
    "meet",
    "leq",
    "int_const",
    "int_range",
]


@dataclass(frozen=True)
class AbstractValue:
    """Base class.  ``provenance`` is a short human-readable derivation."""

    provenance: Tuple[str, ...] = field(default=(), compare=False)

    # Lattice interface -------------------------------------------------
    def is_top(self) -> bool:
        return isinstance(self, Top)

    def is_bottom(self) -> bool:
        return isinstance(self, Bottom)

    def join(self, other: "AbstractValue") -> "AbstractValue":
        return _join2(self, other)

    def meet(self, other: "AbstractValue") -> "AbstractValue":
        return _meet2(self, other)

    def leq(self, other: "AbstractValue") -> bool:
        """Lattice order: ``self ⊑ other`` iff ``join(self, other) == other``
        (equivalently ``self`` is at least as precise as ``other``)."""
        return _leq2(self, other)

    def widen(self, other: "AbstractValue") -> "AbstractValue":
        """Widening operator ``self ▽ other``.

        Numeric ranges widen unstable bounds to ±∞ (and containers widen
        component-wise) so loop fixpoints over the interval domain terminate;
        the finite-height non-numeric facets fall back to join.  Always a sound
        over-approximation of :meth:`join`."""
        return _widen2(self, other)

    def narrow(self, other: "AbstractValue") -> "AbstractValue":
        """Narrowing operator ``self ▵ other`` — recovers precision lost to
        widening by replacing ±∞ numeric bounds with finite ones from ``other``
        (which the caller guarantees is ⊑ ``self``).  Sound and tightening
        only, so post-fixpoint narrowing terminates."""
        return _narrow2(self, other)

    def with_prov(self, *msg: str) -> "AbstractValue":
        return replace(self, provenance=tuple(self.provenance) + tuple(msg))


@dataclass(frozen=True)
class Top(AbstractValue):
    """Unknown / could-be-anything.  Never the basis of a bug report."""


@dataclass(frozen=True)
class Bottom(AbstractValue):
    """Unreachable."""


@dataclass(frozen=True)
class NoneVal(AbstractValue):
    """The literal ``None``."""


@dataclass(frozen=True)
class IntVal(AbstractValue):
    """An integer abstract value.

    Carries two complementary facets:

    * ``sym`` — an affine *symbolic* expression over dimension variables, so a
      value read from ``x.size(0)`` flows by name into later shape arithmetic.
    * ``interval`` — a numeric *range* ``[lo, hi]`` (an
      :class:`src.domains.intervals.Interval`), so range-dependent bugs
      (division by a possibly-zero value, negative dimensions, off-by-one index
      bounds) become decidable.  ``None`` means "unknown range" (⊤).

    The two facets are kept consistent: a constant ``sym`` implies a singleton
    ``interval`` and vice-versa via :meth:`normalized`.
    """

    sym: Optional[SymDim] = None
    interval: Optional["IntervalT"] = None

    @property
    def const(self) -> Optional[int]:
        if self.sym is not None and self.sym.value is not None:
            return self.sym.value
        if self.interval is not None and self.interval.is_singleton:
            return self.interval.singleton_value()
        return None

    def lo(self) -> Optional[int]:
        if self.interval is not None and not self.interval.is_bottom:
            b = self.interval.lo
            return b.value if getattr(b, "is_finite", False) else None
        c = self.const
        return c

    def hi(self) -> Optional[int]:
        if self.interval is not None and not self.interval.is_bottom:
            b = self.interval.hi
            return b.value if getattr(b, "is_finite", False) else None
        c = self.const
        return c

    def contains_only_zero(self) -> bool:
        return self.const == 0

    def may_be_zero(self) -> bool:
        if self.interval is not None and not self.interval.is_bottom:
            return self.interval.contains_zero()
        c = self.const
        return c is None or c == 0


@dataclass(frozen=True)
class FloatVal(AbstractValue):
    const: Optional[float] = None


@dataclass(frozen=True)
class BoolVal(AbstractValue):
    const: Optional[bool] = None


@dataclass(frozen=True)
class StrVal(AbstractValue):
    const: Optional[str] = None


@dataclass(frozen=True)
class TensorVal(AbstractValue):
    """Tensor metadata.  ``rank`` may be known while ``shape`` is not."""

    rank: Optional[int] = None
    shape: Optional[Tuple[Optional[SymDim], ...]] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    requires_grad: Optional[bool] = None
    is_leaf: Optional[bool] = None

    def __post_init__(self):
        # keep rank and shape consistent
        if self.shape is not None and self.rank is None:
            object.__setattr__(self, "rank", len(self.shape))

    def dim(self, i: int) -> Optional[SymDim]:
        if self.shape is None or self.rank is None:
            return None
        return self.shape[i % self.rank]


@dataclass(frozen=True)
class TupleVal(AbstractValue):
    elems: Tuple[AbstractValue, ...] = ()
    exact_len: bool = True  # False ⇒ length is only a lower bound / unknown

    @property
    def length(self) -> Optional[int]:
        return len(self.elems) if self.exact_len else None


@dataclass(frozen=True)
class ListVal(AbstractValue):
    elem: AbstractValue = field(default_factory=lambda: TOP)
    length: Optional[int] = None
    exact_elems: Optional[Tuple[AbstractValue, ...]] = None


@dataclass(frozen=True)
class DictVal(AbstractValue):
    """Abstract dict.  ``value`` summarises all values; ``known`` carries the
    precise value for statically-known string keys (small-map precision), which
    lets ``d['key']`` resolve and missing-key accesses be flagged.  ``exact_keys``
    is True when ``known`` enumerates *every* key the dict can have."""

    value: AbstractValue = field(default_factory=lambda: TOP)
    known: Tuple[Tuple[str, AbstractValue], ...] = ()
    exact_keys: bool = False

    def __post_init__(self):
        # Canonicalise ``known``: dedupe (last write wins) and sort by key so that
        # structural equality — and hence ``leq`` (defined as ``join==``) — is
        # insensitive to insertion order.  Without this, ``join(d, d) != d`` for
        # any dict whose keys were not already sorted, breaking reflexivity.
        if self.known:
            collapsed: Dict[str, AbstractValue] = {}
            for k, v in self.known:
                collapsed[k] = v
            canon = tuple(sorted(collapsed.items(), key=lambda kv: kv[0]))
            if canon != self.known:
                object.__setattr__(self, "known", canon)

    def get_key(self, key: str) -> Optional[AbstractValue]:
        for k, v in self.known:
            if k == key:
                return v
        return None


@dataclass(frozen=True)
class SetVal(AbstractValue):
    """Abstract set: an element summary and an optional known length."""

    elem: AbstractValue = field(default_factory=lambda: TOP)
    length: Optional[int] = None


@dataclass(frozen=True)
class ModuleVal(AbstractValue):
    class_name: str = "?"
    attrs: Tuple[Tuple[str, AbstractValue], ...] = ()  # frozen attr map
    # convenience layer metadata for nn.* layers
    meta: Tuple[Tuple[str, int], ...] = ()

    def get_attr(self, name: str) -> Optional[AbstractValue]:
        for k, v in self.attrs:
            if k == name:
                return v
        return None

    def get_meta(self, name: str) -> Optional[int]:
        for k, v in self.meta:
            if k == name:
                return v
        return None


@dataclass(frozen=True)
class CallableVal(AbstractValue):
    """A resolvable callable: a function/method def node id + a label."""

    qualname: str = "?"
    func_id: Optional[int] = None  # id() of the ast.FunctionDef
    bound_self: Optional[AbstractValue] = None


# Singletons
TOP = Top()
BOTTOM = Bottom()
NONE = NoneVal()


# --------------------------------------------------------------------------
# join (least upper bound, sound over-approximation)
# --------------------------------------------------------------------------
def _join2(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    if a.is_bottom():
        return b
    if b.is_bottom():
        return a
    if a.is_top() or b.is_top():
        return TOP
    if type(a) is not type(b):
        return TOP

    if isinstance(a, NoneVal):
        return a
    if isinstance(a, IntVal) and isinstance(b, IntVal):
        iv = None
        if a.interval is not None and b.interval is not None:
            iv = a.interval.join(b.interval)
        sym = a.sym if (a.sym is not None and b.sym is not None and a.sym.definitely_eq(b.sym)) else None
        return IntVal(sym=sym, interval=iv)
    if isinstance(a, (FloatVal, BoolVal, StrVal)):
        return type(a)(const=getattr(a, "const") if getattr(a, "const") == getattr(b, "const") else None)
    if isinstance(a, TensorVal) and isinstance(b, TensorVal):
        rank = a.rank if a.rank == b.rank else None
        shape = None
        if rank is not None and a.shape is not None and b.shape is not None:
            shape = tuple(
                (da if (da is not None and db is not None and da.definitely_eq(db)) else None)
                for da, db in zip(a.shape, b.shape)
            )
        return TensorVal(
            rank=rank,
            shape=shape,
            dtype=a.dtype if a.dtype == b.dtype else None,
            device=a.device if a.device == b.device else None,
            requires_grad=a.requires_grad if a.requires_grad == b.requires_grad else None,
            is_leaf=a.is_leaf if a.is_leaf == b.is_leaf else None,
        )
    if isinstance(a, TupleVal) and isinstance(b, TupleVal):
        if a.exact_len and b.exact_len and len(a.elems) == len(b.elems):
            return TupleVal(
                elems=tuple(_join2(x, y) for x, y in zip(a.elems, b.elems)),
                exact_len=True,
            )
        # Otherwise the precise length is unknown; keep the element-wise join
        # over the common prefix so the result still over-approximates both and
        # ``join(t, t) == t`` (lattice reflexivity) holds for inexact tuples.
        n = min(len(a.elems), len(b.elems))
        return TupleVal(
            elems=tuple(_join2(a.elems[i], b.elems[i]) for i in range(n)),
            exact_len=False,
        )
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        exact = None
        if a.exact_elems is not None and b.exact_elems is not None and len(a.exact_elems) == len(b.exact_elems):
            exact = tuple(_join2(x, y) for x, y in zip(a.exact_elems, b.exact_elems))
        return ListVal(
            elem=_join2(a.elem, b.elem),
            length=a.length if a.length == b.length else None,
            exact_elems=exact,
        )
    if isinstance(a, DictVal) and isinstance(b, DictVal):
        ak, bk = dict(a.known), dict(b.known)
        common = sorted(set(ak) & set(bk))
        known = tuple((k, _join2(ak[k], bk[k])) for k in common)
        # keys present in only one side fold into the value summary
        only = [ak[k] for k in ak if k not in bk] + [bk[k] for k in bk if k not in ak]
        value = _join2(a.value, b.value)
        for v in only:
            value = _join2(value, v)
        exact_keys = a.exact_keys and b.exact_keys and set(ak) == set(bk)
        return DictVal(value=value, known=known, exact_keys=exact_keys)
    if isinstance(a, SetVal) and isinstance(b, SetVal):
        return SetVal(elem=_join2(a.elem, b.elem), length=a.length if a.length == b.length else None)
    if isinstance(a, ModuleVal) and isinstance(b, ModuleVal):
        if a.class_name == b.class_name and a.attrs == b.attrs and a.meta == b.meta:
            return a
        return TOP
    if isinstance(a, CallableVal) and isinstance(b, CallableVal):
        if a.func_id == b.func_id:
            return a
        return TOP
    return TOP


def join(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    return _join2(a, b)


def _widen2(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    """Value-level widening ``a ▽ b``.

    Defined recursively so an ascending Kleene chain over the *whole* value
    lattice stabilises: numeric ranges use interval widening (unstable bounds
    jump to ±∞), and structurally recursive containers widen their components.
    Everything else falls back to :func:`_join2` — sound because the non-numeric
    facets already have finite height (concrete → unknown → Top).
    """
    if a.is_bottom():
        return b
    if b.is_bottom():
        return a
    if a.is_top() or b.is_top() or type(a) is not type(b):
        return _join2(a, b)
    if isinstance(a, IntVal) and isinstance(b, IntVal):
        iv = None
        if a.interval is not None and b.interval is not None:
            iv = a.interval.widen(b.interval)
        sym = a.sym if (a.sym is not None and b.sym is not None and a.sym.definitely_eq(b.sym)) else None
        return IntVal(sym=sym, interval=iv)
    if isinstance(a, TupleVal) and isinstance(b, TupleVal):
        if a.exact_len and b.exact_len and len(a.elems) == len(b.elems):
            return TupleVal(elems=tuple(_widen2(x, y) for x, y in zip(a.elems, b.elems)), exact_len=True)
        n = min(len(a.elems), len(b.elems))
        return TupleVal(elems=tuple(_widen2(a.elems[i], b.elems[i]) for i in range(n)), exact_len=False)
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        exact = None
        if (
            a.exact_elems is not None
            and b.exact_elems is not None
            and len(a.exact_elems) == len(b.exact_elems)
        ):
            exact = tuple(_widen2(x, y) for x, y in zip(a.exact_elems, b.exact_elems))
        return ListVal(
            elem=_widen2(a.elem, b.elem),
            length=a.length if a.length == b.length else None,
            exact_elems=exact,
        )
    if isinstance(a, SetVal) and isinstance(b, SetVal):
        return SetVal(elem=_widen2(a.elem, b.elem), length=a.length if a.length == b.length else None)
    if isinstance(a, DictVal) and isinstance(b, DictVal):
        ak, bk = dict(a.known), dict(b.known)
        common = sorted(set(ak) & set(bk))
        known = tuple((k, _widen2(ak[k], bk[k])) for k in common)
        only = [ak[k] for k in ak if k not in bk] + [bk[k] for k in bk if k not in ak]
        value = _widen2(a.value, b.value)
        for v in only:
            value = _widen2(value, v)
        exact_keys = a.exact_keys and b.exact_keys and set(ak) == set(bk)
        return DictVal(value=value, known=known, exact_keys=exact_keys)
    return _join2(a, b)


def widen(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    return _widen2(a, b)


def _narrow2(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    """Value-level narrowing ``a ▵ b`` — the post-widening precision-recovery
    dual of :func:`_widen2`.

    Callers guarantee ``b ⊑ a`` (``b`` is one more transfer step from the
    widened post-fixpoint ``a``).  Numeric ranges replace their ±∞ bounds with
    ``b``'s finite ones; containers narrow component-wise.  Everything else keeps
    ``a`` unchanged — those facets have already converged and have no infinite
    chains to recover, so the identity is the correct (and sound) narrowing.
    """
    if a.is_bottom() or b.is_bottom():
        return a
    if a.is_top():
        # A widened ⊤ carries no recoverable structure; only descend if ``b``
        # refines it to the same shape (rare); otherwise stay ⊤ (sound).
        return a
    if type(a) is not type(b):
        return a
    if isinstance(a, IntVal) and isinstance(b, IntVal):
        if a.interval is None:
            return a
        iv = a.interval.narrow(b.interval) if b.interval is not None else a.interval
        sym = a.sym if a.sym is not None else b.sym
        return IntVal(sym=sym, interval=iv)
    if isinstance(a, TupleVal) and isinstance(b, TupleVal):
        if a.exact_len and b.exact_len and len(a.elems) == len(b.elems):
            return TupleVal(elems=tuple(_narrow2(x, y) for x, y in zip(a.elems, b.elems)), exact_len=True)
        n = min(len(a.elems), len(b.elems))
        return TupleVal(elems=tuple(_narrow2(a.elems[i], b.elems[i]) for i in range(n)), exact_len=a.exact_len)
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        exact = a.exact_elems
        if (
            a.exact_elems is not None
            and b.exact_elems is not None
            and len(a.exact_elems) == len(b.exact_elems)
        ):
            exact = tuple(_narrow2(x, y) for x, y in zip(a.exact_elems, b.exact_elems))
        return ListVal(elem=_narrow2(a.elem, b.elem), length=a.length, exact_elems=exact)
    if isinstance(a, SetVal) and isinstance(b, SetVal):
        return SetVal(elem=_narrow2(a.elem, b.elem), length=a.length)
    if isinstance(a, DictVal) and isinstance(b, DictVal):
        bk = dict(b.known)
        known = tuple((k, _narrow2(v, bk[k]) if k in bk else v) for k, v in a.known)
        value = _narrow2(a.value, b.value)
        return DictVal(value=value, known=known, exact_keys=a.exact_keys)
    return a


def narrow(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    return _narrow2(a, b)


def join_many(vals: List[AbstractValue]) -> AbstractValue:
    acc: AbstractValue = BOTTOM
    for v in vals:
        acc = _join2(acc, v)
    return acc


# Above this size a precise element-wise container is collapsed to its summary
# form during loop widening, so loop-grown collections terminate.
_CONTAINER_SUMMARY_THRESHOLD = 16


def summarize_container(v: AbstractValue) -> AbstractValue:
    """Collapse a large precise container to summary form (drop per-element
    precision, keep the element join and lose the exact length).  Used when a
    collection grows inside a loop so the abstract value reaches a fixpoint."""
    if isinstance(v, ListVal) and v.exact_elems is not None and len(v.exact_elems) > _CONTAINER_SUMMARY_THRESHOLD:
        return ListVal(elem=join_many(list(v.exact_elems)), length=None, exact_elems=None)
    if isinstance(v, TupleVal) and v.exact_len and len(v.elems) > _CONTAINER_SUMMARY_THRESHOLD:
        return TupleVal(elems=(), exact_len=False)
    return v



# --------------------------------------------------------------------------
# meet (greatest lower bound) and the lattice order
# --------------------------------------------------------------------------
def _meet2(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    """Greatest lower bound.  Used to *refine* a value with a guard (e.g. an
    ``isinstance``/``is not None`` branch).  When two values are incompatible
    the meet is ``Bottom`` (the path is infeasible)."""
    if a.is_top():
        return b
    if b.is_top():
        return a
    if a.is_bottom() or b.is_bottom():
        return BOTTOM
    if type(a) is not type(b):
        return BOTTOM  # disjoint concrete types ⇒ infeasible

    if isinstance(a, NoneVal):
        return a
    if isinstance(a, IntVal) and isinstance(b, IntVal):
        # meet narrows both facets; an empty interval ⇒ infeasible path.
        iv = a.interval
        if a.interval is not None and b.interval is not None:
            iv = a.interval.meet(b.interval)
            if iv.is_bottom:
                return BOTTOM
        elif b.interval is not None:
            iv = b.interval
        sym = a.sym if a.sym is not None else b.sym
        if a.sym is not None and b.sym is not None:
            if a.sym.definitely_eq(b.sym):
                sym = a.sym
            elif a.sym.value is not None and b.sym.value is not None:
                return BOTTOM  # distinct constants
        return IntVal(sym=sym, interval=iv)
    if isinstance(a, (FloatVal, BoolVal, StrVal)):
        ca, cb = getattr(a, "const"), getattr(b, "const")
        if ca is None:
            return b
        if cb is None:
            return a
        return a if ca == cb else BOTTOM
    if isinstance(a, TensorVal) and isinstance(b, TensorVal):
        if a.rank is not None and b.rank is not None and a.rank != b.rank:
            return BOTTOM
        rank = a.rank if a.rank is not None else b.rank
        # dtype/device conflicts make the meet infeasible
        if a.dtype is not None and b.dtype is not None and a.dtype != b.dtype:
            return BOTTOM
        if a.device is not None and b.device is not None and a.device != b.device:
            return BOTTOM
        shape = None
        if a.shape is not None and b.shape is not None:
            dims = []
            for da, db in zip(a.shape, b.shape):
                if da is None:
                    dims.append(db)
                elif db is None:
                    dims.append(da)
                elif da.definitely_eq(db):
                    dims.append(da)
                else:
                    va, vb = da.value, db.value
                    if va is not None and vb is not None:
                        return BOTTOM  # distinct constant sizes ⇒ infeasible
                    dims.append(da if va is not None else db)  # keep the more precise dim
            shape = tuple(dims)
        elif a.shape is not None:
            shape = a.shape
        elif b.shape is not None:
            shape = b.shape
        return TensorVal(
            rank=rank,
            shape=shape,
            dtype=a.dtype if a.dtype is not None else b.dtype,
            device=a.device if a.device is not None else b.device,
            requires_grad=a.requires_grad if a.requires_grad is not None else b.requires_grad,
            is_leaf=a.is_leaf if a.is_leaf is not None else b.is_leaf,
        )
    if isinstance(a, TupleVal) and isinstance(b, TupleVal):
        if a.exact_len and b.exact_len:
            if len(a.elems) != len(b.elems):
                return BOTTOM
            elems = tuple(_meet2(x, y) for x, y in zip(a.elems, b.elems))
            if any(e.is_bottom() for e in elems):
                return BOTTOM
            return TupleVal(elems=elems, exact_len=True)
        # one side has unknown length: keep the precise side
        return a if a.exact_len else b
    if isinstance(a, ListVal) and isinstance(b, ListVal):
        if a.length is not None and b.length is not None and a.length != b.length:
            return BOTTOM  # disjoint lengths ⇒ infeasible
        length = a.length if a.length is not None else b.length
        exact = None
        if a.exact_elems is not None and b.exact_elems is not None:
            if len(a.exact_elems) != len(b.exact_elems):
                return BOTTOM
            exact = tuple(_meet2(x, y) for x, y in zip(a.exact_elems, b.exact_elems))
            if any(e.is_bottom() for e in exact):
                return BOTTOM
        elif a.exact_elems is not None:
            exact = a.exact_elems  # refine the unknown side with the precise one
        elif b.exact_elems is not None:
            exact = b.exact_elems
        return ListVal(elem=_meet2(a.elem, b.elem), length=length, exact_elems=exact)
    if isinstance(a, ModuleVal) and isinstance(b, ModuleVal):
        return a if a == b else BOTTOM
    return a


def meet(a: AbstractValue, b: AbstractValue) -> AbstractValue:
    return _meet2(a, b)


def _leq2(a: AbstractValue, b: AbstractValue) -> bool:
    """``a ⊑ b`` defined via join: ``a ⊑ b ⇔ join(a, b) == b``."""
    if a.is_bottom():
        return True
    if b.is_top():
        return True
    return _eq_ignoring_prov(_join2(a, b), b)


def leq(a: AbstractValue, b: AbstractValue) -> bool:
    return _leq2(a, b)


def _eq_ignoring_prov(a: AbstractValue, b: AbstractValue) -> bool:
    # ``provenance`` is compare=False already, but fresh symbolic dims created
    # during join would otherwise spoil equality; compare structurally.
    return a == b

