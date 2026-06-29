"""Step 8 — concretization oracle.

The abstract domain maps a concrete program value to an over-approximating
abstract value via an (implicit) abstraction ``α``.  This module supplies the
adjoint direction:

* :func:`gamma` / :func:`gamma_samples` — the *concretization* ``γ``: concrete
  representatives drawn from the set an abstract value stands for.
* :func:`alpha` — a minimal abstraction of a concrete representative, used to
  state and check the soundness law ``α(γ(v)) ⊑ v``.
* :func:`force_counterexample` — search ``γ(v)`` for a concrete witness that
  *forces* a predicate (e.g. a detector's failing condition) to hold, thereby
  **certifying** that a reported bug is real rather than spurious.

Everything here is conservative: when a value's concretization cannot be
sampled finitely the oracle returns the empty witness set (it never invents a
representative it cannot justify), so a certified counterexample is always a
genuine member of ``γ(v)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .values import (
    AbstractValue,
    BoolVal,
    Bottom,
    DictVal,
    FloatVal,
    IntVal,
    ListVal,
    NoneVal,
    SetVal,
    StrVal,
    TensorVal,
    Top,
    TupleVal,
    int_const,
)
from .values import _leq2 as leq  # lattice order

__all__ = [
    "ConcreteTensor",
    "ANY",
    "NO_WITNESS",
    "gamma",
    "gamma_samples",
    "alpha",
    "force_counterexample",
]


@dataclass(frozen=True)
class ConcreteTensor:
    """A concrete tensor representative: just its shape (a tuple of sizes)."""

    shape: Tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.shape)


class _Any:
    """Sentinel standing for "any concrete value" — the concretization of ⊤.

    It is intentionally *not* a usable representative: callers that need a
    sample must treat ``ANY`` as "no finite witness available"."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ANY"


ANY = _Any()

# Returned by ``force_counterexample`` when no certifying witness exists.
NO_WITNESS = None

# A representative size used when a tensor dimension is symbolic/unknown but the
# rank is fixed.  Small and positive so it never spuriously triggers size-0 or
# negative-dim corner cases.
_FREE_DIM = 1
# Bound on how many concrete candidates an interval is expanded to when
# searching for a counterexample, keeping the oracle decidable and fast.
_MAX_SAMPLES = 64


def _dim_value(d) -> Optional[int]:
    """The concrete size of a symbolic dim, or ``None`` if it is not a constant."""
    if d is None:
        return None
    v = getattr(d, "value", None)
    return v


def gamma(value: AbstractValue):
    """A single concrete representative drawn from ``γ(value)``.

    Returns ``ANY`` for ⊤ (no finite representative) and raises for ⊥ (empty
    concretization).  Otherwise the returned object is a genuine member of the
    value's concretization set."""
    samples = gamma_samples(value, 1)
    if samples:
        return samples[0]
    if isinstance(value, Top):
        return ANY
    if isinstance(value, Bottom):
        raise ValueError("Bottom has an empty concretization (γ(⊥) = ∅)")
    return ANY


def gamma_samples(value: AbstractValue, k: int = _MAX_SAMPLES) -> List:
    """Up to ``k`` concrete representatives from ``γ(value)``.

    Soundness: every element returned is a member of ``γ(value)`` (verified by
    the ``α(γ(v)) ⊑ v`` law in the tests).  The list is empty when the
    concretization cannot be finitely sampled (⊤) or is empty (⊥)."""
    if k <= 0:
        return []
    if isinstance(value, Bottom):
        return []
    if isinstance(value, Top):
        return []
    if isinstance(value, NoneVal):
        return [None]
    if isinstance(value, BoolVal):
        return [value.const] if value.const is not None else [False, True][:k]
    if isinstance(value, FloatVal):
        return [value.const] if value.const is not None else [0.0]
    if isinstance(value, StrVal):
        return [value.const] if value.const is not None else [""]
    if isinstance(value, IntVal):
        return _int_samples(value, k)
    if isinstance(value, TensorVal):
        return _tensor_samples(value, k)
    if isinstance(value, TupleVal):
        return _tuple_samples(value, k)
    if isinstance(value, ListVal):
        return _list_samples(value, k)
    if isinstance(value, SetVal):
        elem = gamma_samples(value.elem, 1)
        return [set(elem)] if elem else [set()]
    if isinstance(value, DictVal):
        return _dict_samples(value, k)
    return []


def _int_samples(value: IntVal, k: int) -> List[int]:
    c = value.const
    if c is not None:
        return [c]
    lo, hi = value.lo(), value.hi()
    if lo is not None and hi is not None and hi >= lo:
        span = hi - lo + 1
        if span <= k:
            return list(range(lo, hi + 1))
        # sample the endpoints and an interior point
        return [lo, hi, lo + (hi - lo) // 2][:k]
    if lo is not None:
        return [lo, lo + 1][:k]
    if hi is not None:
        return [hi - 1, hi][:k]
    # a "top int" still denotes *some* integer (unlike the universal ⊤); 0/±1 are
    # genuine members, which keeps containers holding it concretizable.
    return [0, 1, -1][:k]


def _tensor_samples(value: TensorVal, k: int) -> List[ConcreteTensor]:
    if value.rank is None:
        return []
    if value.shape is None:
        return [ConcreteTensor(tuple(_FREE_DIM for _ in range(value.rank)))]
    dims = []
    for i in range(value.rank):
        v = _dim_value(value.dim(i))
        dims.append(v if v is not None else _FREE_DIM)
    return [ConcreteTensor(tuple(dims))]


def _tuple_samples(value: TupleVal, k: int) -> List[tuple]:
    if not value.exact_len:
        return []
    parts = [gamma_samples(e, 1) for e in value.elems]
    if any(not p for p in parts):
        return []
    return [tuple(p[0] for p in parts)]


def _list_samples(value: ListVal, k: int) -> List[list]:
    if value.exact_elems is not None:
        parts = [gamma_samples(e, 1) for e in value.exact_elems]
        if any(not p for p in parts):
            return []
        return [[p[0] for p in parts]]
    if value.length is not None:
        elem = gamma_samples(value.elem, 1)
        if not elem:
            return [[]] if value.length == 0 else []
        return [[elem[0]] * value.length]
    elem = gamma_samples(value.elem, 1)
    return [elem[:1]]  # an arbitrary-length list; a singleton is a member


def _dict_samples(value: DictVal, k: int) -> List[dict]:
    out = {}
    for key, val in value.known:
        s = gamma_samples(val, 1)
        if not s:
            return []  # a key's value (universal ⊤/⊥) is not soundly representable
        out[key] = s[0]
    return [out]


def alpha(concrete) -> AbstractValue:
    """A minimal abstraction of a concrete representative.

    Used to state the soundness law ``α(γ(v)) ⊑ v``.  Only the concrete forms
    produced by :func:`gamma_samples` need be handled."""
    if concrete is None:
        return NoneVal()
    if isinstance(concrete, bool):
        return BoolVal(const=concrete)
    if isinstance(concrete, int):
        return int_const(concrete)
    if isinstance(concrete, float):
        return FloatVal(const=concrete)
    if isinstance(concrete, str):
        return StrVal(const=concrete)
    if isinstance(concrete, ConcreteTensor):
        from .symdim import SymDim

        shape = tuple(SymDim.const_dim(s) for s in concrete.shape)
        return TensorVal(rank=concrete.rank, shape=shape)
    if isinstance(concrete, tuple):
        return TupleVal(elems=tuple(alpha(e) for e in concrete), exact_len=True)
    if isinstance(concrete, list):
        elems = tuple(alpha(e) for e in concrete)
        from .values import join_many

        return ListVal(
            elem=join_many(list(elems)) if elems else Top(),
            length=len(concrete),
            exact_elems=elems,
        )
    if isinstance(concrete, set):
        from .values import join_many

        elems = [alpha(e) for e in concrete]
        return SetVal(elem=join_many(elems) if elems else Top(), length=len(concrete))
    if isinstance(concrete, dict):
        known = tuple(
            (key, alpha(val)) for key, val in concrete.items() if isinstance(key, str)
        )
        from .values import join_many

        return DictVal(
            value=join_many([v for _, v in known]) if known else Top(),
            known=known,
            exact_keys=all(isinstance(key, str) for key in concrete),
        )
    return Top()


def is_sound_sample(value: AbstractValue, concrete) -> bool:
    """True iff ``concrete`` is a genuine member of ``γ(value)`` — i.e.
    ``α(concrete) ⊑ value``.  The certification predicate behind the oracle."""
    if concrete is ANY:
        return isinstance(value, Top)
    return leq(alpha(concrete), value)


def force_counterexample(
    value: AbstractValue, predicate: Callable[[object], bool], k: int = _MAX_SAMPLES
):
    """Return a concrete witness ``w ∈ γ(value)`` with ``predicate(w)`` true, or
    ``NO_WITNESS`` (``None``) if no sampled member satisfies it.

    A returned witness **certifies** a report: it is a concrete input the
    analysis admits on which the failing condition genuinely holds."""
    for w in gamma_samples(value, k):
        try:
            if predicate(w):
                return w
        except Exception:
            continue
    return NO_WITNESS
