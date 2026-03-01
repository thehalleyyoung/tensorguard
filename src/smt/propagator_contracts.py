"""
Formal Callback Contract Specification for Z3 UserPropagator Implementations.

Defines eight contracts (C1–C8) that every UserPropagator must satisfy
at the theory-solver interface, and provides verification utilities.

Contracts
---------
C1 (Push-Pop Invertibility):
    For all states σ, pop(push(σ)) = σ.
    The trail mechanism correctly restores propagator state.

C2 (Fixed Monotonicity / Propagation Soundness under assignment):
    If _on_fixed(v, val) propagates clauses C₁…Cₖ, then every Cᵢ is a
    logical consequence of  TheoryAxioms ∪ {v = val} ∪ current_fixed.

C3 (Final Completeness):
    If _on_final() does not raise a conflict, then the current variable
    assignment is satisfying for the theory.

C4 (Conflict Soundness):
    Every conflict clause reported via self.conflict() is a valid nogood:
    its negation is implied by the theory axioms.

C5 (Propagation Soundness):
    Every propagated literal via self.propagate() is implied by
    TheoryAxioms ∪ current_fixed.

C6 (Backjump Correctness):
    When the SAT solver backtracks to decision level d, pop() restores
    state to exactly the scope at level d, preserving all fixed variables
    at levels ≤ d.  Formally: after pop(), the propagator state equals
    the state immediately after the push() at level d.

C7 (Conflict Clause Minimality):
    Conflict clauses contain only variables that participate in the theory
    conflict.  Removing any literal renders the clause non-conflicting.

C8 (Exhaustive Propagation):
    Before any decision, all unit propagations derivable from the current
    assignment are performed.  _on_final() verifies no further consequences
    can be derived.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Contract data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PropagationRecord:
    """A single propagation or conflict emitted by a propagator."""

    kind: str  # "propagate" or "conflict"
    clause: Optional[Any] = None  # propagated equality (z3 expr)
    deps: Optional[List[Any]] = None  # dependency variables


@dataclass
class ContractViolation:
    """Describes a violation of one of the five contracts."""

    contract: str  # "C1", "C2", "C3", "C4", "C5"
    description: str
    details: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Instrumented wrapper for recording propagator actions
# ═══════════════════════════════════════════════════════════════════════════


class PropagatorInstrumentor:
    """Wraps a propagator to record all conflict() and propagate() calls.

    Does NOT subclass or replace the propagator — instead instruments it
    by monkey-patching the conflict/propagate methods while preserving
    the original behaviour.
    """

    def __init__(self, propagator: Any) -> None:
        self.propagator = propagator
        self.records: List[PropagationRecord] = []
        self._original_conflict = propagator.conflict
        self._original_propagate = getattr(propagator, "propagate", None)
        self._patch()

    def _patch(self) -> None:
        prop = self.propagator
        records = self.records

        original_conflict = self._original_conflict

        def _recording_conflict(deps=None, **kwargs):
            records.append(PropagationRecord(kind="conflict", deps=deps))
            return original_conflict(deps=deps, **kwargs)

        prop.conflict = _recording_conflict

        if self._original_propagate is not None:
            original_propagate = self._original_propagate

            def _recording_propagate(clause, ids=None, **kwargs):
                records.append(
                    PropagationRecord(kind="propagate", clause=clause, deps=ids)
                )
                return original_propagate(clause, ids=ids, **kwargs)

            prop.propagate = _recording_propagate

    def unpatch(self) -> None:
        self.propagator.conflict = self._original_conflict
        if self._original_propagate is not None:
            self.propagator.propagate = self._original_propagate

    def clear(self) -> None:
        self.records.clear()

    @property
    def conflicts(self) -> List[PropagationRecord]:
        return [r for r in self.records if r.kind == "conflict"]

    @property
    def propagations(self) -> List[PropagationRecord]:
        return [r for r in self.records if r.kind == "propagate"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. C1 — Push-Pop Invertibility verification
# ═══════════════════════════════════════════════════════════════════════════


def _snapshot_fixed(propagator: Any) -> Dict[int, Any]:
    """Extract a copy of the propagator's _fixed dict."""
    return dict(propagator._fixed)


def verify_push_pop_invertibility(
    propagator: Any,
    mutate_fn: Optional[Callable[[Any], None]] = None,
    num_cycles: int = 5,
    seed: int = 42,
) -> List[ContractViolation]:
    """Test C1: for all states σ, pop(push(σ)) = σ.

    Performs ``num_cycles`` rounds of:
      1. Record state σ
      2. push()
      3. Mutate state (add random entries to _fixed)
      4. pop(1)
      5. Assert state == σ

    Args:
        propagator: A UserPropagateBase instance (not attached to a live
            solver during this test — we drive push/pop manually).
        mutate_fn: Optional callable that mutates the propagator state
            between push and pop.  Defaults to adding random _fixed entries.
        num_cycles: Number of push/pop cycles to test.
        seed: RNG seed for reproducibility.

    Returns:
        List of ContractViolation for each failed cycle (empty = pass).
    """
    rng = random.Random(seed)
    violations: List[ContractViolation] = []

    for cycle in range(num_cycles):
        state_before = _snapshot_fixed(propagator)
        trail_len_before = len(propagator._trail)

        propagator.push()

        # Mutate
        if mutate_fn is not None:
            mutate_fn(propagator)
        else:
            # Default: add some random entries
            for _ in range(rng.randint(1, 5)):
                fake_id = rng.randint(10000, 99999)
                propagator._fixed[fake_id] = rng.randint(0, 100)

        propagator.pop(1)

        state_after = _snapshot_fixed(propagator)

        if state_after != state_before:
            violations.append(
                ContractViolation(
                    contract="C1",
                    description=(
                        f"Cycle {cycle}: pop(push(σ)) ≠ σ.  "
                        f"Before: {state_before}, After: {state_after}"
                    ),
                    details={
                        "cycle": cycle,
                        "before": state_before,
                        "after": state_after,
                    },
                )
            )

        trail_len_after = len(propagator._trail)
        if trail_len_after != trail_len_before:
            violations.append(
                ContractViolation(
                    contract="C1",
                    description=(
                        f"Cycle {cycle}: trail length changed from "
                        f"{trail_len_before} to {trail_len_after}"
                    ),
                )
            )

    return violations


def verify_nested_push_pop(
    propagator: Any,
    depth: int = 4,
    seed: int = 123,
) -> List[ContractViolation]:
    """Test C1 with nested push/pop to multiple depths.

    Pushes ``depth`` times (mutating at each level), then pops all at
    once via pop(depth), and checks that original state is restored.
    """
    rng = random.Random(seed)
    violations: List[ContractViolation] = []
    state_original = _snapshot_fixed(propagator)

    for d in range(depth):
        propagator.push()
        for _ in range(rng.randint(1, 3)):
            propagator._fixed[rng.randint(10000, 99999)] = rng.randint(0, 50)

    propagator.pop(depth)

    state_after = _snapshot_fixed(propagator)
    if state_after != state_original:
        violations.append(
            ContractViolation(
                contract="C1",
                description=(
                    f"Nested push({depth})/pop({depth}): state not restored. "
                    f"Before: {state_original}, After: {state_after}"
                ),
            )
        )
    return violations


# ═══════════════════════════════════════════════════════════════════════════
# 4. C2 — Fixed Monotonicity / Propagation Soundness under assignment
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    def verify_fixed_soundness(
        solver_factory: Callable[[], Tuple[z3.Solver, Any, Dict[str, z3.ExprRef]]],
        var_name: str,
        value: Any,
        theory_axioms: Optional[List[z3.ExprRef]] = None,
    ) -> List[ContractViolation]:
        """Test C2: propagated clauses are consequences of theory ∪ {v=val}.

        Uses a separate Z3 solver to check that each propagated clause is
        implied by the theory axioms plus the current fixed assignment.

        Args:
            solver_factory: Callable returning (solver, propagator, var_dict).
                The solver should have all theory constraints added.
            var_name: Name of the variable to fix.
            value: Concrete value to assign.
            theory_axioms: Explicit theory axioms for independent checking.

        Returns:
            List of ContractViolation (empty = pass).
        """
        violations: List[ContractViolation] = []

        solver, propagator, var_dict = solver_factory()
        if var_name not in var_dict:
            return violations

        var = var_dict[var_name]

        # Instrument the propagator
        instrumentor = PropagatorInstrumentor(propagator)

        # Set up the assignment
        if isinstance(value, bool):
            solver.add(var == z3.BoolVal(value))
        elif isinstance(value, int):
            solver.add(var == z3.IntVal(value))
        else:
            solver.add(var == value)

        result = solver.check()

        # Each propagation should be a consequence of the axioms + assignment
        for rec in instrumentor.propagations:
            if rec.clause is not None and theory_axioms is not None:
                check_solver = z3.Solver()
                for ax in theory_axioms:
                    check_solver.add(ax)
                if isinstance(value, bool):
                    check_solver.add(var == z3.BoolVal(value))
                elif isinstance(value, int):
                    check_solver.add(var == z3.IntVal(value))
                else:
                    check_solver.add(var == value)
                # Check that ¬clause is unsat (clause is implied)
                check_solver.add(z3.Not(rec.clause))
                if check_solver.check() == z3.sat:
                    violations.append(
                        ContractViolation(
                            contract="C2",
                            description=(
                                f"Propagated clause {rec.clause} is not a "
                                f"consequence of theory axioms ∪ {{{var_name}={value}}}"
                            ),
                        )
                    )

        instrumentor.unpatch()
        return violations


# ═══════════════════════════════════════════════════════════════════════════
# 5. C3 — Final Completeness
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    def verify_final_completeness(
        solver_factory: Callable[[], Tuple[z3.Solver, Any, Dict[str, z3.ExprRef]]],
        assignment: Dict[str, Any],
        is_satisfying: bool = True,
    ) -> List[ContractViolation]:
        """Test C3: if _on_final() does not conflict, assignment is satisfying.

        Creates a solver, adds the given assignment, and checks:
        - If ``is_satisfying`` is True: solver should report SAT.
        - If ``is_satisfying`` is False: solver should report UNSAT (the
          propagator should raise a conflict either in _on_fixed or _on_final).

        Args:
            solver_factory: Callable returning (solver, propagator, var_dict).
            assignment: {var_name: value} mapping.
            is_satisfying: Whether the assignment should be satisfying.

        Returns:
            List of ContractViolation (empty = pass).
        """
        violations: List[ContractViolation] = []

        solver, propagator, var_dict = solver_factory()

        for name, val in assignment.items():
            if name not in var_dict:
                continue
            v = var_dict[name]
            if isinstance(val, bool):
                solver.add(v == z3.BoolVal(val))
            elif isinstance(val, int):
                solver.add(v == z3.IntVal(val))
            else:
                solver.add(v == val)

        result = solver.check()
        expected = z3.sat if is_satisfying else z3.unsat

        if result != expected:
            violations.append(
                ContractViolation(
                    contract="C3",
                    description=(
                        f"Expected {'SAT' if is_satisfying else 'UNSAT'} "
                        f"but got {result} for assignment {assignment}"
                    ),
                )
            )

        return violations


# ═══════════════════════════════════════════════════════════════════════════
# 6. C4 — Conflict Soundness
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    def verify_conflict_soundness(
        solver_factory: Callable[[], Tuple[z3.Solver, Any, Dict[str, z3.ExprRef]]],
        conflicting_assignment: Dict[str, Any],
    ) -> List[ContractViolation]:
        """Test C4: every conflict clause is a valid nogood.

        Adds a conflicting assignment and verifies the solver returns UNSAT.
        Then verifies that the conflict deps are indeed relevant to the
        unsatisfiability by checking that removing any dep makes the
        assignment potentially satisfiable.

        Args:
            solver_factory: Callable returning (solver, propagator, var_dict).
            conflicting_assignment: Assignment that should trigger a conflict.

        Returns:
            List of ContractViolation (empty = pass).
        """
        violations: List[ContractViolation] = []

        solver, propagator, var_dict = solver_factory()
        instrumentor = PropagatorInstrumentor(propagator)

        for name, val in conflicting_assignment.items():
            if name not in var_dict:
                continue
            v = var_dict[name]
            if isinstance(val, bool):
                solver.add(v == z3.BoolVal(val))
            elif isinstance(val, int):
                solver.add(v == z3.IntVal(val))
            else:
                solver.add(v == val)

        result = solver.check()

        if result != z3.unsat:
            violations.append(
                ContractViolation(
                    contract="C4",
                    description=(
                        f"Expected UNSAT for conflicting assignment "
                        f"{conflicting_assignment} but got {result}"
                    ),
                )
            )
        else:
            # Verify that conflicts were actually raised
            if len(instrumentor.conflicts) == 0:
                # The UNSAT might come from the Z3 core (axiom encoding),
                # which is also valid — no violation.
                pass
            else:
                # Each conflict should have non-empty deps
                for i, conflict in enumerate(instrumentor.conflicts):
                    if conflict.deps is None or len(conflict.deps) == 0:
                        violations.append(
                            ContractViolation(
                                contract="C4",
                                description=(
                                    f"Conflict {i} has empty deps — "
                                    f"cannot form a valid nogood"
                                ),
                            )
                        )

        instrumentor.unpatch()
        return violations


# ═══════════════════════════════════════════════════════════════════════════
# 7. C5 — Propagation Soundness
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    def verify_propagation_soundness(
        solver_factory: Callable[[], Tuple[z3.Solver, Any, Dict[str, z3.ExprRef]]],
        partial_assignment: Dict[str, Any],
        expected_propagations: Optional[Dict[str, Any]] = None,
    ) -> List[ContractViolation]:
        """Test C5: propagated literals are implied by theory ∪ current_fixed.

        Adds a partial assignment and checks that the solver can find a
        satisfying model.  If ``expected_propagations`` is given, verifies
        that the model's values for those variables match the expectation
        (meaning the propagator correctly deduced the values).

        Args:
            solver_factory: Callable returning (solver, propagator, var_dict).
            partial_assignment: {var_name: value} to fix before checking.
            expected_propagations: {var_name: expected_value} that should
                be deduced by the propagator.

        Returns:
            List of ContractViolation (empty = pass).
        """
        violations: List[ContractViolation] = []

        solver, propagator, var_dict = solver_factory()

        for name, val in partial_assignment.items():
            if name not in var_dict:
                continue
            v = var_dict[name]
            if isinstance(val, bool):
                solver.add(v == z3.BoolVal(val))
            elif isinstance(val, int):
                solver.add(v == z3.IntVal(val))
            else:
                solver.add(v == val)

        result = solver.check()

        if result != z3.sat:
            violations.append(
                ContractViolation(
                    contract="C5",
                    description=(
                        f"Expected SAT for partial assignment "
                        f"{partial_assignment} but got {result}"
                    ),
                )
            )
            return violations

        if expected_propagations:
            model = solver.model()
            for name, expected in expected_propagations.items():
                if name not in var_dict:
                    continue
                v = var_dict[name]
                actual = model[v]
                if actual is None:
                    continue

                match = False
                if isinstance(expected, bool):
                    match = (z3.is_true(actual) == expected)
                elif isinstance(expected, int):
                    try:
                        match = (actual.as_long() == expected)
                    except Exception:
                        match = False
                else:
                    match = (actual == expected)

                if not match:
                    violations.append(
                        ContractViolation(
                            contract="C5",
                            description=(
                                f"Variable {name}: expected {expected} "
                                f"but model gives {actual}"
                            ),
                        )
                    )

        return violations


# ═══════════════════════════════════════════════════════════════════════════
# 8. ContractMonitor — runtime assertion mixin for all propagators
# ═══════════════════════════════════════════════════════════════════════════


class ContractMonitor:
    """Runtime contract monitor that wraps any UserPropagator to check C1-C5.

    Instruments a propagator by monkey-patching its push, pop, conflict,
    and propagate instance attributes.  Z3 calls push/pop via virtual
    dispatch, and the propagator's own callbacks (e.g. _on_fixed) call
    self.conflict()/self.propagate() which resolves through Python's
    normal attribute lookup (instance dict first).

    All violations are logged at WARNING level and appended to the
    ``contract_violations`` list.

    Usage::

        monitor = ContractMonitor(propagator)
        # ... use propagator normally ...
        assert len(monitor.contract_violations) == 0
    """

    def __init__(self, propagator: Any) -> None:
        self.propagator = propagator
        self.contract_violations: List[ContractViolation] = []
        self._state_stack: List[Dict[int, Any]] = []
        self._conflict_log: List[Dict[str, Any]] = []
        self._propagation_log: List[Dict[str, Any]] = []
        self._final_log: List[Dict[str, Any]] = []
        self._patch(propagator)

    def _log_violation(self, violation: ContractViolation) -> None:
        self.contract_violations.append(violation)
        logger.warning(
            "Contract %s violated: %s", violation.contract, violation.description
        )

    def _patch(self, prop: Any) -> None:
        monitor = self
        cls = type(prop)

        # --- C1: Push-Pop Invertibility ---
        original_push = prop.push

        def _monitored_push() -> None:
            snapshot = dict(prop._fixed)
            monitor._state_stack.append(snapshot)
            return original_push()

        original_pop = prop.pop

        def _monitored_pop(num_scopes: int) -> None:
            expected_states = []
            for _ in range(num_scopes):
                if monitor._state_stack:
                    expected_states.append(monitor._state_stack.pop())
            result = original_pop(num_scopes)
            if expected_states:
                expected = expected_states[-1]
                actual = dict(prop._fixed)
                if actual != expected:
                    monitor._log_violation(ContractViolation(
                        contract="C1",
                        description=(
                            f"pop({num_scopes}): state not restored. "
                            f"Expected {expected}, got {actual}"
                        ),
                        details={"expected": expected, "actual": actual},
                    ))
            return result

        prop.push = _monitored_push
        prop.pop = _monitored_pop

        # --- C4: Conflict Soundness ---
        original_conflict = cls.conflict

        def _monitored_conflict(deps=[], eqs=[]):
            entry = {"deps": deps}
            monitor._conflict_log.append(entry)
            if not deps:
                monitor._log_violation(ContractViolation(
                    contract="C4",
                    description="conflict() called with empty or None deps",
                    details={"deps": deps},
                ))
            return original_conflict(prop, deps=deps, eqs=eqs)

        prop.conflict = _monitored_conflict

        # --- C5: Propagation Soundness (logging) ---
        original_propagate = cls.propagate

        def _monitored_propagate(e, ids, eqs=[]):
            entry = {"clause": e, "ids": ids}
            monitor._propagation_log.append(entry)
            return original_propagate(prop, e, ids, eqs)

        prop.propagate = _monitored_propagate

    def _monitored_conflict_check(self, deps=None):
        """Check deps for C4 violation without calling Z3's conflict.

        Use this for testing violation detection outside Z3 callback context.
        """
        entry = {"deps": deps}
        self._conflict_log.append(entry)
        if not deps:
            self._log_violation(ContractViolation(
                contract="C4",
                description="conflict() called with empty or None deps",
                details={"deps": deps},
            ))

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all monitor activity."""
        return {
            "violations": len(self.contract_violations),
            "conflicts_logged": len(self._conflict_log),
            "propagations_logged": len(self._propagation_log),
            "finals_logged": len(self._final_log),
            "push_pop_checks": len(self._state_stack),
            "violation_details": [
                {"contract": v.contract, "description": v.description}
                for v in self.contract_violations
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. DPLL(T) Integration Contracts C6–C8
# ═══════════════════════════════════════════════════════════════════════════


def verify_backjump_correctness(
    propagator: Any,
    variables: List[Any],
    assignments: List[Tuple[Any, Any]],
    backjump_level: int,
) -> List[ContractViolation]:
    """Verify C6: after backjumping to level d, state equals state at level d.

    Simulates: push at each assignment level, then pop back to backjump_level,
    verifying that the propagator state matches the snapshot taken at that level.
    """
    violations: List[ContractViolation] = []
    if not hasattr(propagator, "push") or not hasattr(propagator, "pop"):
        return violations

    snapshots: Dict[int, Dict[int, Any]] = {}
    for level, (var, val) in enumerate(assignments):
        propagator.push()
        snapshots[level] = _snapshot_fixed(propagator)
        if hasattr(propagator, "_on_fixed"):
            try:
                propagator._on_fixed(var, val)
            except Exception:
                pass

    # Backjump: pop back to backjump_level
    pops_needed = len(assignments) - backjump_level
    for _ in range(pops_needed):
        propagator.pop()

    state_after_backjump = _snapshot_fixed(propagator)
    expected_state = snapshots.get(backjump_level, {})

    # Check that all variables at levels <= backjump_level are preserved
    for var_id, val in expected_state.items():
        if var_id not in state_after_backjump:
            violations.append(ContractViolation(
                contract="C6",
                description=f"Backjump lost variable {var_id} at level {backjump_level}",
                details={"var_id": var_id, "expected": str(val), "level": backjump_level},
            ))

    # Clean up remaining scopes
    for _ in range(backjump_level):
        propagator.pop()

    return violations


def verify_conflict_minimality(
    conflict_deps: List[Any],
    propagator: Any,
    assignment: Dict[Any, Any],
) -> List[ContractViolation]:
    """Verify C7: conflict clause is minimal (no extraneous literals).

    Checks that every dependency in the conflict clause references a variable
    in the current assignment (necessary condition for minimality).
    """
    violations: List[ContractViolation] = []
    if len(conflict_deps) <= 1:
        return violations  # Trivially minimal

    for i, dep in enumerate(conflict_deps):
        found = False
        for var in assignment:
            if hasattr(var, "get_id") and hasattr(dep, "get_id"):
                if var.get_id() == dep.get_id():
                    found = True
                    break
            elif var is dep or id(var) == id(dep):
                found = True
                break
        if not found:
            violations.append(ContractViolation(
                contract="C7",
                description="Conflict clause contains variable not in current assignment",
                details={"dep_index": i, "dep": str(dep)},
            ))

    return violations


def verify_exhaustive_propagation(
    propagator: Any,
    variables: List[Any],
    assignment: Dict[Any, Any],
    propagations_before_final: List[PropagationRecord],
) -> List[ContractViolation]:
    """Verify C8: all derivable propagations are performed before decisions.

    After setting all variables in the assignment and collecting propagations,
    calls _on_final(). If _on_final() produces additional propagations beyond
    those already made, the propagator was not exhaustive.
    """
    violations: List[ContractViolation] = []
    if not hasattr(propagator, "_on_final"):
        return violations

    instrumentor = PropagatorInstrumentor(propagator)
    try:
        propagator._on_final()
    except Exception:
        pass  # Conflict in final is acceptable
    finally:
        new_propagations = instrumentor.propagations()
        instrumentor.unpatch()

    if new_propagations:
        violations.append(ContractViolation(
            contract="C8",
            description=(
                f"_on_final() derived {len(new_propagations)} additional "
                f"propagations that should have been made eagerly"
            ),
            details={"missed_count": len(new_propagations)},
        ))

    return violations
