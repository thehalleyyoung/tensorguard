"""Step 56 — relational domains (optional octagon / polyhedra for coupled dims).

The symbolic executor tracks linear *relations* between dimension variables as a
conjunction of :class:`smt_bridge.DimConstraint`s (``a == b``, ``a <= b + 1`` …)
accumulated from guards along a path.  Affine equalities are already captured
exactly by shared ``SymDim`` variables; this module adds the missing
**relational lattice** so those constraints can be merged precisely at
control-flow joins instead of being dropped by a syntactic intersection.

A :class:`RelationalDomain` is a polyhedron-style abstract value: a finite set
of dimension constraints denoting their conjunction.  Its distinguishing
operation is a *semantic* join — a constraint survives the merge of two
branches iff **both** branches entail it (proved via Z3), drawing candidates
from the union of the two branches' stated constraints.  This keeps facts that
hold on both paths even when they are written differently (e.g. ``{a == b}`` on
one branch and ``{a <= b, a >= b}`` on the other), which a syntactic
intersection would lose.

Soundness contract (same floor as the rest of the engine): a merged constraint
is kept only when Z3 *proves* it entailed by both operands; when z3 is
unavailable, ``entails`` degrades to syntactic membership, so the join reduces
to the previous (sound) syntactic intersection — never to something stronger
than is justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from . import smt_bridge
from .smt_bridge import DimConstraint

__all__ = ["RelationalDomain", "join_facts", "meet_facts"]


def _dedup(constraints: Iterable[DimConstraint]) -> Tuple[DimConstraint, ...]:
    """Order-preserving de-duplication."""
    seen = set()
    out: List[DimConstraint] = []
    for c in constraints:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


@dataclass(frozen=True)
class RelationalDomain:
    """A conjunction of dimension constraints (a polyhedron over dim vars)."""

    constraints: Tuple[DimConstraint, ...] = ()

    # -- constructors ----------------------------------------------------
    @staticmethod
    def top() -> "RelationalDomain":
        """The no-information element (empty conjunction)."""
        return RelationalDomain(())

    @staticmethod
    def of(constraints: Iterable[DimConstraint]) -> "RelationalDomain":
        return RelationalDomain(_dedup(constraints))

    # -- predicates ------------------------------------------------------
    def is_top(self) -> bool:
        return not self.constraints

    def is_bottom(self) -> bool:
        """``True`` only when the conjunction is **proved** unsatisfiable; an
        ``unknown`` / no-z3 verdict is *not* bottom (sound)."""
        if not self.constraints:
            return False
        return not smt_bridge.feasible(list(self.constraints))

    def entails(self, c: DimConstraint) -> bool:
        """Whether the conjunction logically implies ``c``.

        Cheap syntactic membership first; otherwise ``facts ∧ ¬c`` is checked
        for unsatisfiability.  When z3 is unavailable only membership is used,
        so a join built on ``entails`` degrades to a syntactic intersection."""
        if c in self.constraints:
            return True
        if not smt_bridge.Z3_AVAILABLE:
            return False
        return not smt_bridge.feasible([*self.constraints, smt_bridge.negate(c)])

    # -- lattice ops -----------------------------------------------------
    def meet(self, other: "RelationalDomain") -> "RelationalDomain":
        """Greatest lower bound: the conjunction of both constraint sets."""
        return RelationalDomain.of((*self.constraints, *other.constraints))

    def join(self, other: "RelationalDomain") -> "RelationalDomain":
        """Least upper bound (over the stated-constraint basis): keep every
        candidate constraint entailed by **both** operands.

        Sound by construction — a kept constraint holds on each incoming path,
        hence on the merged one — and strictly more precise than a syntactic
        intersection, which only keeps literally-shared constraints."""
        if self.is_top() or other.is_top():
            return RelationalDomain.top()
        if self.constraints == other.constraints:
            return self
        # Candidate basis: constraints stated on either side (self first for a
        # deterministic, self-biased ordering).
        candidates = _dedup((*self.constraints, *other.constraints))
        kept = [c for c in candidates if self.entails(c) and other.entails(c)]
        return RelationalDomain(tuple(kept))

    def widen(self, other: "RelationalDomain") -> "RelationalDomain":
        """Widening for fixpoints: keep only the constraints of ``self`` that
        remain entailed by ``other``.  The constraint set can only shrink, so an
        ascending chain stabilises in finitely many steps."""
        if self.is_top():
            return self
        kept = [c for c in self.constraints if other.entails(c)]
        return RelationalDomain(tuple(kept))


# ---------------------------------------------------------------------------
# Thin tuple-level helpers, used by ``State._merge`` so the lattice never has to
# import :class:`State` (and vice versa).
# ---------------------------------------------------------------------------

def join_facts(
    a: Tuple[DimConstraint, ...], b: Tuple[DimConstraint, ...]
) -> Tuple[DimConstraint, ...]:
    """The relational join of two path-fact tuples (see
    :meth:`RelationalDomain.join`)."""
    return RelationalDomain(tuple(a)).join(RelationalDomain(tuple(b))).constraints


def meet_facts(
    a: Tuple[DimConstraint, ...], b: Tuple[DimConstraint, ...]
) -> Tuple[DimConstraint, ...]:
    """The relational meet (conjunction) of two path-fact tuples."""
    return RelationalDomain(tuple(a)).meet(RelationalDomain(tuple(b))).constraints
