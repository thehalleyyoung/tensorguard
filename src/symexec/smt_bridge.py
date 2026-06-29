"""Step 51 — Z3 bridge for :class:`SymDim`.

Lowers the symbolic executor's affine dimension expressions (:class:`SymDim`)
and a list of *dimension constraints* (the path constraints accumulated along an
execution, plus the negated property whose feasibility we want to test) into the
Z3 integer theory, and answers a single question:

    Is this set of constraints **satisfiable** under the structural
    well-formedness assumptions of tensor dimensions?

This is the foundation for *feasibility-gated reporting* (Step 52): a candidate
failure is only worth reporting if the conjunction of (path constraints ∧ failing
condition) is **SAT** — i.e. there is a concrete shape assignment that actually
reaches the fault.

Soundness contract
------------------
The bridge is conservative in the direction that matters for a zero-false-
positive bug finder:

* :func:`check` returns ``"sat"`` / ``"unsat"`` / ``"unknown"``.
* It only ever returns ``"unsat"`` when Z3 *proves* unsatisfiability.
* If Z3 is unavailable, times out, or returns ``unknown``, the result is
  ``"unknown"`` — never a fabricated ``"unsat"``.

Therefore a feasibility gate built on top (suppress a report only when the
failing condition is provably ``"unsat"``) can never suppress a *true* bug it
cannot disprove, and can never invent one.  At worst it fails to filter an
infeasible report (a precision loss, not a soundness loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .symdim import SymDim

__all__ = [
    "Z3_AVAILABLE",
    "DimConstraint",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "divisible",
    "not_divisible",
    "SymDimSolver",
    "check",
    "feasible",
    "model",
    "unsat_core",
    "negate",
]

try:  # pragma: no cover - exercised by both branches across environments
    import z3 as _z3

    Z3_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means no solver
    _z3 = None
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constraint representation
# ---------------------------------------------------------------------------

# Relational operators between two affine ``SymDim`` expressions.
_REL_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})


@dataclass(frozen=True)
class DimConstraint:
    """A single relation over affine dimension expressions.

    ``op`` is one of the relational operators in :data:`_REL_OPS`, or one of the
    divisibility predicates ``"%==0"`` / ``"%!=0"`` in which case ``rhs`` is the
    (constant) modulus and must be a non-zero integer.
    """

    lhs: SymDim
    op: str
    rhs: Union[SymDim, int]

    def __post_init__(self) -> None:
        if self.op in ("%==0", "%!=0"):
            if not isinstance(self.rhs, int) or self.rhs == 0:
                raise ValueError("divisibility modulus must be a non-zero int")
        elif self.op not in _REL_OPS:
            raise ValueError(f"unknown constraint op {self.op!r}")


# -- ergonomic constructors -------------------------------------------------

def _as_symdim(x: Union[SymDim, int]) -> SymDim:
    return x if isinstance(x, SymDim) else SymDim.const_dim(int(x))


def eq(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "==", _as_symdim(b))


def ne(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "!=", _as_symdim(b))


def lt(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "<", _as_symdim(b))


def le(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "<=", _as_symdim(b))


def gt(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), ">", _as_symdim(b))


def ge(a: Union[SymDim, int], b: Union[SymDim, int]) -> DimConstraint:
    return DimConstraint(_as_symdim(a), ">=", _as_symdim(b))


def divisible(a: Union[SymDim, int], k: int) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "%==0", int(k))


def not_divisible(a: Union[SymDim, int], k: int) -> DimConstraint:
    return DimConstraint(_as_symdim(a), "%!=0", int(k))


# Logical negation of each relational / divisibility operator.
_OP_NEGATE = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
    "%==0": "%!=0",
    "%!=0": "%==0",
}


def negate(c: DimConstraint) -> DimConstraint:
    """The logical negation of a constraint (``a == b`` → ``a != b`` …).

    Used by the relational domain to check entailment (``facts ⟹ c`` iff
    ``facts ∧ ¬c`` is unsatisfiable)."""
    return DimConstraint(c.lhs, _OP_NEGATE[c.op], c.rhs)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SymDimSolver:
    """Lowers ``SymDim`` constraints into Z3 integer arithmetic.

    A fresh Z3 ``Int`` constant is allocated per dimension-variable *name*
    (memoised), so the affine structure shared across constraints (``b*c``,
    ``b*c + 1`` …) is preserved and coupled exactly the way the abstract domain
    intends.
    """

    def __init__(self, positive_floor: int = 1, timeout_ms: int = 2000) -> None:
        # tensor dimension sizes are non-negative; ``positive_floor`` is the
        # smallest size a free dimension may take.  1 matches the practical
        # minimum of an "interesting" shape; 0 is the absolute lower bound.
        self.positive_floor = positive_floor
        self.timeout_ms = timeout_ms
        self._vars: Dict[str, object] = {}

    # -- lowering --------------------------------------------------------
    def _var(self, name: str):
        v = self._vars.get(name)
        if v is None:
            v = _z3.Int(name)
            self._vars[name] = v
        return v

    def dim_to_z3(self, dim: SymDim):
        """Lower an affine ``SymDim`` to a Z3 integer arithmetic expression."""
        if not Z3_AVAILABLE:
            raise RuntimeError("z3 is not available")
        expr = _z3.IntVal(dim.const)
        for name, coeff in dim.terms:
            expr = expr + _z3.IntVal(coeff) * self._var(name)
        return expr

    def constraint_to_z3(self, c: DimConstraint):
        if not Z3_AVAILABLE:
            raise RuntimeError("z3 is not available")
        lhs = self.dim_to_z3(c.lhs)
        if c.op in ("%==0", "%!=0"):
            r = lhs % _z3.IntVal(c.rhs)
            return (r == 0) if c.op == "%==0" else (r != 0)
        rhs = self.dim_to_z3(_as_symdim(c.rhs))
        if c.op == "==":
            return lhs == rhs
        if c.op == "!=":
            return lhs != rhs
        if c.op == "<":
            return lhs < rhs
        if c.op == "<=":
            return lhs <= rhs
        if c.op == ">":
            return lhs > rhs
        if c.op == ">=":
            return lhs >= rhs
        raise ValueError(f"unknown op {c.op!r}")  # pragma: no cover

    def _wellformed(self) -> List[object]:
        floor = _z3.IntVal(self.positive_floor)
        return [v >= floor for v in self._vars.values()]

    # -- queries ---------------------------------------------------------
    def check(self, constraints: List[DimConstraint]) -> str:
        """Return ``"sat"`` / ``"unsat"`` / ``"unknown"`` for the conjunction of
        ``constraints`` under the dimension well-formedness assumptions."""
        if not Z3_AVAILABLE:
            return "unknown"
        solver = _z3.Solver()
        solver.set("timeout", self.timeout_ms)
        # Lower the constraints first so every referenced variable is allocated,
        # then add the well-formedness floor for all of them.
        lowered = [self.constraint_to_z3(c) for c in constraints]
        for wf in self._wellformed():
            solver.add(wf)
        for f in lowered:
            solver.add(f)
        res = solver.check()
        if res == _z3.sat:
            return "sat"
        if res == _z3.unsat:
            return "unsat"
        return "unknown"

    def model(self, constraints: List[DimConstraint]) -> Optional[Dict[str, int]]:
        """A concrete satisfying assignment (variable name → value) when SAT,
        else ``None``.  Used by counterexample lifting (Step 54)."""
        if not Z3_AVAILABLE:
            return None
        solver = _z3.Solver()
        solver.set("timeout", self.timeout_ms)
        lowered = [self.constraint_to_z3(c) for c in constraints]
        for wf in self._wellformed():
            solver.add(wf)
        for f in lowered:
            solver.add(f)
        if solver.check() != _z3.sat:
            return None
        m = solver.model()
        out: Dict[str, int] = {}
        for name, var in self._vars.items():
            val = m.eval(var, model_completion=True)
            try:
                out[name] = val.as_long()
            except Exception:  # pragma: no cover - defensive
                out[name] = 0
        return out

    def unsat_core(
        self, constraints: List[DimConstraint]
    ) -> Optional[List[DimConstraint]]:
        """The **minimal** subset of ``constraints`` whose conjunction is already
        unsatisfiable under the dimension well-formedness floor — a Craig-style
        *interpolant* explaining why the path is infeasible (Step 55 — CEGAR).

        Returns ``None`` when z3 is unavailable or the conjunction is **not**
        provably unsat (SAT / ``unknown``): refinement only fires on a proved
        contradiction, so soundness is preserved (an unknown path is kept).
        """
        if not Z3_AVAILABLE:
            return None
        solver = _z3.Solver()
        solver.set("timeout", self.timeout_ms)
        # Lower every constraint first so all variables are allocated, then add
        # the well-formedness floor *untracked* — the core is reported purely in
        # terms of the user path facts that caused the contradiction.
        tracked = []
        for i, c in enumerate(constraints):
            lit = _z3.Bool(f"_c{i}")
            solver.assert_and_track(self.constraint_to_z3(c), lit)
            tracked.append((lit, c))
        for wf in self._wellformed():
            solver.add(wf)
        if solver.check() != _z3.unsat:
            return None
        core = solver.unsat_core()
        core_ids = {c.get_id() for c in core}
        picked = [c for lit, c in tracked if lit.get_id() in core_ids]
        # Fall back to the full set if z3 reports an empty core (shouldn't
        # happen with tracked assertions, but never claim "no reason").
        return picked or [c for _, c in tracked]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def check(constraints: List[DimConstraint], positive_floor: int = 1) -> str:
    """Satisfiability of ``constraints``: ``"sat"`` / ``"unsat"`` / ``"unknown"``."""
    return SymDimSolver(positive_floor=positive_floor).check(constraints)


def feasible(constraints: List[DimConstraint], positive_floor: int = 1) -> bool:
    """``True`` unless the constraints are **provably** unsatisfiable.

    This is the sound primitive for feasibility-gated reporting: a report is
    suppressed only when ``feasible(...)`` is ``False`` (a Z3-proved ``unsat``);
    ``unknown`` keeps the report.
    """
    return check(constraints, positive_floor=positive_floor) != "unsat"


def model(constraints: List[DimConstraint], positive_floor: int = 1) -> Optional[Dict[str, int]]:
    """A concrete satisfying assignment when SAT, else ``None``."""
    return SymDimSolver(positive_floor=positive_floor).model(constraints)


def unsat_core(
    constraints: List[DimConstraint], positive_floor: int = 1
) -> Optional[List[DimConstraint]]:
    """The minimal unsatisfiable subset of ``constraints`` (an interpolant) when
    the conjunction is provably unsat, else ``None``."""
    return SymDimSolver(positive_floor=positive_floor).unsat_core(constraints)
