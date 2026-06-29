"""Step 57 — disjunctive states (bounded powerset for path precision).

At a control-flow merge the executor normally *joins* the two branch states into
one.  Joining is sound but lossy: a dimension that is ``a`` on the then-branch
and ``b`` on the else-branch becomes ``⊤`` (unknown) afterwards, masking a fault
that is real on one of the paths.  A **disjunctive** (powerset) domain keeps the
branches *apart* — a finite set of alternative states — so subsequent
straight-line code is analysed on each path precisely, and only collapses back
to a single state at the next control-flow boundary (or when a cardinality bound
is hit, which keeps the analysis finite and cheap).

:class:`DisjunctiveState` is that bounded powerset of :class:`State`.  It is a
sound over-approximation of its disjuncts' union: :meth:`collapse` (the join of
all disjuncts) is always a valid single-state summary, so the executor can drop
back to single-state mode at any point without losing soundness.

Determinism & termination: disjuncts are de-duplicated (via ``State.equals``)
and capped at ``bound``; on overflow the whole set collapses to a single joined
state, so the domain has bounded width and every operation terminates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .state import State

__all__ = ["DisjunctiveState", "DEFAULT_BOUND"]

# The maximum number of alternative states kept apart before they are collapsed.
# Small enough to bound cost (a block can at most multiply work by this), large
# enough to keep a couple of independent two-way branches distinct.
DEFAULT_BOUND = 8


def _dedup(states: List[State]) -> List[State]:
    """Order-preserving de-duplication by structural state equality."""
    out: List[State] = []
    for s in states:
        if not any(s.equals(o) for o in out):
            out.append(s)
    return out


@dataclass(frozen=True)
class DisjunctiveState:
    """A bounded set of alternative program states (their disjunction)."""

    disjuncts: Tuple[State, ...] = ()
    bound: int = DEFAULT_BOUND

    # -- constructors ----------------------------------------------------
    @staticmethod
    def singleton(state: State, bound: int = DEFAULT_BOUND) -> "DisjunctiveState":
        return DisjunctiveState((state,), bound)

    @staticmethod
    def of(states, bound: int = DEFAULT_BOUND) -> "DisjunctiveState":
        return DisjunctiveState(tuple(states), bound)._capped()

    # -- queries ---------------------------------------------------------
    def live(self) -> "DisjunctiveState":
        """Drop unreachable disjuncts (terminated paths carry no continuation)."""
        return DisjunctiveState(
            tuple(d for d in self.disjuncts if d.reachable), self.bound
        )

    def is_empty(self) -> bool:
        """``True`` when no reachable disjunct remains."""
        return not any(d.reachable for d in self.disjuncts)

    def width(self) -> int:
        return len(self.disjuncts)

    # -- internal bound enforcement -------------------------------------
    def _capped(self) -> "DisjunctiveState":
        ds = _dedup(list(self.disjuncts))
        if len(ds) > self.bound:
            # Overflow: collapse the whole set to a single sound summary so the
            # powerset width never exceeds the bound (keeps analysis finite).
            ds = [_join_all(ds)]
        return DisjunctiveState(tuple(ds), self.bound)

    # -- transfers -------------------------------------------------------
    def map(self, fn: Callable[[State], State]) -> "DisjunctiveState":
        """Apply a state transfer to every reachable disjunct (terminated paths
        are dropped), returning the de-duplicated, capped result."""
        out = [fn(d) for d in self.disjuncts if d.reachable]
        return DisjunctiveState(tuple(out), self.bound)._capped()

    def flat_map(self, fn: Callable[[State], List[State]]) -> "DisjunctiveState":
        """Like :meth:`map` but each disjunct may expand into several states
        (e.g. an ``if`` splitting into then/else paths)."""
        out: List[State] = []
        for d in self.disjuncts:
            if d.reachable:
                out.extend(fn(d))
        return DisjunctiveState(tuple(out), self.bound)._capped()

    def extend(self, states) -> "DisjunctiveState":
        return DisjunctiveState((*self.disjuncts, *states), self.bound)._capped()

    def join(self, other: "DisjunctiveState") -> "DisjunctiveState":
        """Powerset join: the (deduped, capped) union of the two disjunct sets."""
        return DisjunctiveState(
            (*self.disjuncts, *other.disjuncts), max(self.bound, other.bound)
        )._capped()

    # -- collapse back to single-state mode -----------------------------
    def collapse(self) -> State:
        """The join of all disjuncts — a sound single-state summary.

        A single disjunct is returned unchanged (so single-state execution is
        byte-identical).  When every path has terminated, an unreachable state is
        returned so the caller's reachability handling is preserved."""
        if not self.disjuncts:
            return State.unreachable()
        reach = [d for d in self.disjuncts if d.reachable]
        if not reach:
            return self.disjuncts[0]  # carry a terminated state (env irrelevant)
        out = reach[0]
        for d in reach[1:]:
            out = out.join(d)
        return out


def _join_all(states: List[State]) -> State:
    out = states[0]
    for s in states[1:]:
        out = out.join(s)
    return out
