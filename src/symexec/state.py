"""Abstract program state for the symbolic executor.

A :class:`State` bundles the variable environment, an object/attribute store
(for ``self.<attr>``), and a reachability flag.  States form a lattice via
:meth:`join` so that branches can be merged at control-flow joins and loops can
reach a fixpoint with :meth:`widen`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Optional

from .values import AbstractValue, TOP, BOTTOM, join as vjoin, widen as vwiden, narrow as vnarrow

__all__ = ["State"]


@dataclass
class State:
    env: Dict[str, AbstractValue] = field(default_factory=dict)
    # store: object-key -> {attr -> value}.  "self" is the canonical key for the
    # module instance under analysis.
    store: Dict[str, Dict[str, AbstractValue]] = field(default_factory=dict)
    reachable: bool = True
    # Path constraints over symbolic dimensions accumulated by guard refinement
    # along the current path (Step 52).  Each entry is a ``smt_bridge.DimConstraint``
    # known to hold here; a feasibility gate may suppress a report whose failing
    # condition is provably unsatisfiable under these facts.  Stored as an opaque
    # tuple to avoid importing the bridge into the state lattice.
    dim_facts: tuple = ()

    # -- env access ------------------------------------------------------
    def get(self, name: str) -> AbstractValue:
        return self.env.get(name, TOP)

    def set(self, name: str, value: AbstractValue) -> None:
        self.env[name] = value

    def has(self, name: str) -> bool:
        return name in self.env

    # -- store access ----------------------------------------------------
    def get_attr(self, obj: str, attr: str) -> AbstractValue:
        return self.store.get(obj, {}).get(attr, TOP)

    def set_attr(self, obj: str, attr: str, value: AbstractValue) -> None:
        self.store.setdefault(obj, {})[attr] = value

    # -- lattice ---------------------------------------------------------
    def copy(self) -> "State":
        return State(
            env=dict(self.env),
            store={k: dict(v) for k, v in self.store.items()},
            reachable=self.reachable,
            dim_facts=self.dim_facts,
        )

    def join(self, other: "State") -> "State":
        return self._merge(other, vjoin)

    def widen(self, other: "State") -> "State":
        # Per-variable widening so an ascending chain over the (infinite-height)
        # interval domain stabilises and loop fixpoints terminate.  ``self`` is
        # the previous iterate and ``other`` the newest — widen in that order so
        # unstable numeric bounds jump to ±∞.
        return self._merge(other, vwiden)

    def narrow(self, other: "State") -> "State":
        # Post-fixpoint narrowing: ``self`` is the widened invariant and
        # ``other = F(self) ⊑ self`` one transfer step further; per-variable
        # narrowing recovers ±∞ bounds back to finite ones the loop guard
        # implies.  Only tightens, so the narrowing chain terminates.
        return self._merge(other, vnarrow)

    def _merge(self, other: "State", op) -> "State":
        if not self.reachable:
            return other.copy()
        if not other.reachable:
            return self.copy()
        out = State(reachable=True)
        for k in set(self.env) | set(other.env):
            out.env[k] = op(self.env.get(k, BOTTOM), other.env.get(k, BOTTOM))
        for obj in set(self.store) | set(other.store):
            a = self.store.get(obj, {})
            b = other.store.get(obj, {})
            merged: Dict[str, AbstractValue] = {}
            for attr in set(a) | set(b):
                merged[attr] = op(a.get(attr, BOTTOM), b.get(attr, BOTTOM))
            out.store[obj] = merged
        # A dim fact holds on the merged path only if it holds on *both* incoming
        # paths.  The relational domain (Step 56) keeps any constraint **entailed
        # by both** branches — strictly more precise than a syntactic
        # intersection, which only retains literally-shared constraints — while
        # remaining sound (a kept fact provably holds on each path).
        if self.dim_facts and other.dim_facts:
            if self.dim_facts == other.dim_facts:
                out.dim_facts = self.dim_facts
            else:
                from .relational import join_facts

                out.dim_facts = join_facts(self.dim_facts, other.dim_facts)
        return out

    def equals(self, other: "State") -> bool:
        return (
            self.reachable == other.reachable
            and self.env == other.env
            and self.store == other.store
        )

    @staticmethod
    def unreachable() -> "State":
        return State(reachable=False)
