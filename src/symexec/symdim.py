"""Symbolic dimension expressions for the TensorGuard symbolic executor.

A :class:`SymDim` is an affine expression ``c0 + Σ cᵢ·vᵢ`` over named dimension
variables (``batch``, ``seq``, ``heads``, or fresh/unknown symbols).  It supports
the small amount of arithmetic that flows between tensor shapes and ordinary
Python integers (``x.size(0)`` participating in later computations), plus a
cheap three-valued comparison oracle.  Hard questions (non-linear equality,
divisibility under constraints) are deferred to Z3 in a later phase; the oracle
here only ever answers ``True``/``False`` when it is *certain*, otherwise
``None`` ("maybe"), which keeps the analysis sound.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

__all__ = ["SymDim", "Three", "fresh_dim"]

# Three-valued logic: True / False / None(=unknown)
Three = Optional[bool]

_counter = itertools.count()


def fresh_dim(prefix: str = "_d") -> "SymDim":
    """A brand-new, independent symbolic dimension variable."""
    return SymDim.var(f"{prefix}{next(_counter)}")


@dataclass(frozen=True)
class SymDim:
    """Affine expression ``const + Σ coeff·var`` with integer coefficients."""

    const: int = 0
    # variable name -> integer coefficient (never zero)
    terms: Tuple[Tuple[str, int], ...] = ()

    # -- constructors ----------------------------------------------------
    @staticmethod
    def const_dim(c: int) -> "SymDim":
        return SymDim(const=int(c), terms=())

    @staticmethod
    def var(name: str) -> "SymDim":
        return SymDim(const=0, terms=((name, 1),))

    # -- helpers ---------------------------------------------------------
    def _terms_map(self) -> Dict[str, int]:
        return dict(self.terms)

    @staticmethod
    def _normalize(const: int, terms: Dict[str, int]) -> "SymDim":
        clean = tuple(sorted((k, v) for k, v in terms.items() if v != 0))
        return SymDim(const=int(const), terms=clean)

    @property
    def is_const(self) -> bool:
        return not self.terms

    @property
    def value(self) -> Optional[int]:
        """Concrete integer value if this is a constant, else ``None``."""
        return self.const if self.is_const else None

    # -- arithmetic ------------------------------------------------------
    def __add__(self, other: Union["SymDim", int]) -> "SymDim":
        other = _coerce(other)
        terms = self._terms_map()
        for k, v in other.terms:
            terms[k] = terms.get(k, 0) + v
        return SymDim._normalize(self.const + other.const, terms)

    __radd__ = __add__

    def __sub__(self, other: Union["SymDim", int]) -> "SymDim":
        return self + (_coerce(other) * -1)

    def __rsub__(self, other: Union["SymDim", int]) -> "SymDim":
        return _coerce(other) - self

    def __mul__(self, other: Union["SymDim", int]) -> "SymDim":
        other = _coerce(other)
        # Only linear * constant stays affine; otherwise give up (fresh dim).
        if self.is_const:
            k = self.const
            return SymDim._normalize(other.const * k, {n: c * k for n, c in other.terms})
        if other.is_const:
            k = other.const
            return SymDim._normalize(self.const * k, {n: c * k for n, c in self.terms})
        return fresh_dim("_mul")

    __rmul__ = __mul__

    def floordiv(self, other: Union["SymDim", int]) -> "SymDim":
        other = _coerce(other)
        if other.is_const and other.const != 0 and self.is_const:
            return SymDim.const_dim(self.const // other.const)
        # exact division of every coefficient keeps it affine
        if other.is_const and other.const != 0:
            k = other.const
            if self.const % k == 0 and all(c % k == 0 for _, c in self.terms):
                return SymDim._normalize(self.const // k, {n: c // k for n, c in self.terms})
        return fresh_dim("_div")

    def mod(self, other: Union["SymDim", int]) -> "SymDim":
        other = _coerce(other)
        if self.is_const and other.is_const and other.const != 0:
            return SymDim.const_dim(self.const % other.const)
        return fresh_dim("_mod")

    # -- comparison oracle (three-valued) --------------------------------
    def definitely_eq(self, other: "SymDim") -> bool:
        return (self - _coerce(other)).is_const and (self - _coerce(other)).const == 0

    def maybe_eq(self, other: "SymDim") -> Three:
        diff = self - _coerce(other)
        if diff.is_const:
            return diff.const == 0
        return None  # symbolic difference: unknown

    def definitely_divisible_by(self, k: int) -> Three:
        if k == 0:
            return False
        if self.is_const:
            return self.const % k == 0
        if self.const % k == 0 and all(c % k == 0 for _, c in self.terms):
            return True
        return None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.is_const:
            return str(self.const)
        parts = []
        for n, c in self.terms:
            parts.append(n if c == 1 else f"{c}*{n}")
        s = " + ".join(parts)
        if self.const:
            s += f" + {self.const}"
        return s


def _coerce(x: Union[SymDim, int]) -> SymDim:
    if isinstance(x, SymDim):
        return x
    if isinstance(x, bool):  # avoid bool-as-int surprises
        return SymDim.const_dim(int(x))
    if isinstance(x, int):
        return SymDim.const_dim(x)
    raise TypeError(f"cannot coerce {type(x)!r} to SymDim")
