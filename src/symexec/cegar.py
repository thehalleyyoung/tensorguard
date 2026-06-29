"""Step 55 — interpolant-based refinement (CEGAR).

The abstract executor explores branches optimistically: a guard such as
``if a == b:`` records the symbolic path fact ``a == b`` but cannot, by
itself, notice that an *enclosing* guard already asserted ``a != b``.  Such a
path is **spurious** — infeasible for every concrete shape — yet the abstract
state would keep exploring it and any fault discovered along it is a potential
false positive (the feasibility gate of Step 52 suppresses the *report*, but
the wasted exploration and its derived facts remain).

This module closes the loop in the spirit of counterexample-guided abstraction
refinement: it asks the Z3 bridge whether the accumulated path facts are
*jointly* unsatisfiable and, when they are, extracts a **Craig-style
interpolant** — the minimal subset of facts responsible for the contradiction
(an :func:`smt_bridge.unsat_core`).  The interpreter uses that verdict to prune
the dead path (mark it unreachable) and records the interpolant as the reason.

Soundness contract (identical floor to the rest of the engine): a path is
refined away **only** when Z3 *proves* the facts unsat under dimensions
``>= 1``.  Missing z3, a solver timeout, or an ``unknown`` answer all leave the
path reachable — refinement never removes a feasible path, so it can never hide
a real bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from . import smt_bridge

__all__ = ["Refinement", "refine", "interpolant"]


@dataclass(frozen=True)
class Refinement:
    """The verdict for a candidate path.

    ``spurious`` is ``True`` only when the facts are a Z3-proved contradiction;
    ``interpolant`` then holds the minimal responsible subset (possibly empty if
    z3 could not minimise but still proved unsat).  ``reason`` is a short human
    string for diagnostics / abstain accounting (Step 59).
    """

    spurious: bool
    interpolant: Tuple[smt_bridge.DimConstraint, ...] = field(default_factory=tuple)
    reason: str = ""


# A non-spurious (feasible / unknown) verdict, shared to avoid re-allocation.
_FEASIBLE = Refinement(spurious=False, interpolant=(), reason="feasible-or-unknown")


def refine(facts: Sequence["smt_bridge.DimConstraint"]) -> Refinement:
    """Classify ``facts`` as a spurious (infeasible) path or not.

    Returns a :class:`Refinement` whose ``spurious`` flag is ``True`` **iff** the
    conjunction of ``facts`` is provably unsatisfiable under the well-formedness
    floor; in that case ``interpolant`` is the minimal unsat core.  Anything
    else (SAT, ``unknown``, no z3, no facts) yields the shared feasible verdict.
    """
    facts = list(facts)
    if not facts:
        return _FEASIBLE
    # Cheap proved-unsat gate first; only pay for core extraction on a real
    # contradiction.  ``feasible`` is True unless Z3 *proves* unsat.
    if smt_bridge.feasible(facts):
        return _FEASIBLE
    core = smt_bridge.unsat_core(facts)
    interp: Tuple[smt_bridge.DimConstraint, ...] = tuple(core) if core else ()
    return Refinement(
        spurious=True,
        interpolant=interp,
        reason=_describe(interp) if interp else "path constraints are contradictory",
    )


def interpolant(facts: Sequence["smt_bridge.DimConstraint"]) -> List["smt_bridge.DimConstraint"]:
    """The minimal contradiction core of ``facts`` (an interpolant), or ``[]``
    when the facts are satisfiable / undecided."""
    r = refine(facts)
    return list(r.interpolant)


def _describe(core: Sequence["smt_bridge.DimConstraint"]) -> str:
    parts = []
    for c in core:
        if c.op in ("%==0", "%!=0"):
            sign = "==0" if c.op == "%==0" else "!=0"
            parts.append(f"{c.lhs} % {c.rhs} {sign}")
        else:
            parts.append(f"{c.lhs} {c.op} {c.rhs}")
    return "unsatisfiable under " + " ∧ ".join(parts)
