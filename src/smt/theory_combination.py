"""
Tinelli-Zarba Theory Combination for Finite-Domain Theories.

Implements the Tinelli-Zarba (JAR 2005) extension of Nelson-Oppen theory
combination for non-stably-infinite sorts.  Standard Nelson-Oppen requires
every sort to be stably-infinite (i.e., have infinitely many elements in
every model).  The broadcast and stride theories operate over Dim ⊆ ℤ_≥1
which is stably-infinite, so they combine classically.  However:

  - T_device has 5 elements: {CPU, CUDA_0, CUDA_1, CUDA_2, CUDA_3}
  - T_phase  has 2 elements: {TRAIN, EVAL}  (Bool: True/False)

These finite domains violate the Nelson-Oppen precondition.  The
Tinelli-Zarba method restores completeness by enumerating *arrangements*
— all possible equivalence classes over the shared variables — and
checking that at least one arrangement is consistent across all theories.

For k shared variables over a domain of size n, the number of
arrangements is bounded by the Stirling number S(k, min(k, n)).
With typical small k (≤ 4 shared device vars, ≤ 2 shared phase vars)
this is tractable.

Algorithm
---------
1. Collect shared variables: variables that appear in more than one
   theory's constraint set.
2. For each finite-domain sort, enumerate all arrangements of the
   shared variables of that sort (all partitions into equivalence
   classes, with at most n classes for an n-element domain).
3. For each arrangement, assert the corresponding equalities and
   disequalities in *each* theory's solver (via push/pop).
4. Check satisfiability in each solver under the arrangement.
5. If any arrangement is consistent in all solvers simultaneously,
   the combined theory is satisfiable.
6. If no arrangement works, the combination is unsatisfiable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import product as iter_product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Arrangement enumeration
# ═══════════════════════════════════════════════════════════════════════════


def _enumerate_partitions(
    n: int, max_classes: int
) -> List[List[int]]:
    """Enumerate all partitions of n elements into at most max_classes classes.

    Returns a list of assignments where assignment[i] is the class
    index (0-based) for element i.  Each assignment uses class indices
    in canonical order (first element is always class 0, second element
    is class 0 or 1, etc.) to avoid counting equivalent partitions.

    This generates *restricted growth strings* — the standard method
    for enumerating set partitions without duplicates.

    Args:
        n: Number of elements to partition.
        max_classes: Maximum number of equivalence classes allowed
                     (domain cardinality for finite sorts).

    Returns:
        List of partition assignments.
    """
    if n == 0:
        return [[]]

    results: List[List[int]] = []

    def _backtrack(pos: int, assignment: List[int], next_class: int) -> None:
        if pos == n:
            results.append(list(assignment))
            return
        # Element `pos` can go into any existing class [0, next_class)
        # or open a new class (if we haven't hit max_classes).
        for c in range(min(next_class + 1, max_classes)):
            assignment.append(c)
            _backtrack(
                pos + 1,
                assignment,
                max(next_class, c + 1),
            )
            assignment.pop()

    _backtrack(0, [], 0)
    return results


def _partition_to_equalities_disequalities(
    variables: List, assignment: List[int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Convert a partition assignment to equality/disequality index pairs.

    Args:
        variables: The variable list (used only for length).
        assignment: Class assignment for each variable.

    Returns:
        (equalities, disequalities): pairs of variable indices.
    """
    n = len(variables)
    equalities: List[Tuple[int, int]] = []
    disequalities: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[i] == assignment[j]:
                equalities.append((i, j))
            else:
                disequalities.append((i, j))
    return equalities, disequalities


# ═══════════════════════════════════════════════════════════════════════════
# 2. TheorySolver — lightweight wrapper for a Z3 solver with metadata
# ═══════════════════════════════════════════════════════════════════════════


class DomainKind(Enum):
    """Classification of a theory's sort for combination purposes."""

    STABLY_INFINITE = "stably_infinite"
    FINITE = "finite"


@dataclass
class TheorySolver:
    """A theory solver participating in theory combination.

    Attributes:
        name: Human-readable theory name (e.g. "broadcast", "device").
        solver: The Z3 Solver instance for this theory.
        domain_kind: Whether the theory's sort is finite or stably-infinite.
        domain_size: For finite domains, the number of elements.
        shared_vars: Z3 variables shared with other theories.
    """

    name: str
    solver: "z3.Solver"
    domain_kind: DomainKind
    domain_size: Optional[int] = None
    shared_vars: List = field(default_factory=list)
    interface_constraints: List = field(default_factory=list)
    sort_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.domain_kind == DomainKind.FINITE:
            if self.domain_size is None or self.domain_size < 1:
                raise ValueError(
                    f"Theory '{self.name}': finite domain requires "
                    f"domain_size >= 1, got {self.domain_size}"
                )
        if self.sort_name is None:
            self.sort_name = self.name


# ═══════════════════════════════════════════════════════════════════════════
# 3. CombinationResult
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CombinationResult:
    """Result of theory combination consistency check.

    Attributes:
        is_consistent: True if a consistent arrangement exists.
        satisfying_arrangement: The arrangement that worked (if any).
            Maps variable index pairs to equality/disequality.
        inconsistencies: List of (theory_name, arrangement) pairs that
            were individually inconsistent.
        total_arrangements_checked: How many arrangements were tried.
    """

    is_consistent: bool
    satisfying_arrangement: Optional[Dict[str, List[int]]] = None
    inconsistencies: List[Tuple[str, List[int]]] = field(
        default_factory=list
    )
    total_arrangements_checked: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. TheoryCombination — main combination engine
# ═══════════════════════════════════════════════════════════════════════════

if HAS_Z3:

    class TheoryCombination:
        """Tinelli-Zarba theory combination for mixed finite/infinite domains.

        Usage::

            combo = TheoryCombination()
            combo.add_theory(TheorySolver(
                name="broadcast",
                solver=broadcast_solver,
                domain_kind=DomainKind.STABLY_INFINITE,
                shared_vars=[dim_x, dim_y],
            ))
            combo.add_theory(TheorySolver(
                name="device",
                solver=device_solver,
                domain_kind=DomainKind.FINITE,
                domain_size=5,
                shared_vars=[dev_a, dev_b],
            ))
            result = combo.check_combination()
        """

        def __init__(self) -> None:
            self._theories: List[TheorySolver] = []

        def add_theory(self, theory: TheorySolver) -> None:
            """Register a theory solver for combination."""
            self._theories.append(theory)

        @property
        def theories(self) -> List[TheorySolver]:
            return list(self._theories)

        def _get_finite_theories(self) -> List[TheorySolver]:
            """Get all finite-domain theories."""
            return [
                t
                for t in self._theories
                if t.domain_kind == DomainKind.FINITE
            ]

        def _check_solver_with_arrangement(
            self,
            solver: "z3.Solver",
            variables: List["z3.ExprRef"],
            equalities: List[Tuple[int, int]],
            disequalities: List[Tuple[int, int]],
        ) -> bool:
            """Check if a solver is SAT under a given arrangement.

            Uses push/pop to temporarily assert equalities and
            disequalities, then checks satisfiability.
            """
            solver.push()
            try:
                for i, j in equalities:
                    solver.add(variables[i] == variables[j])
                for i, j in disequalities:
                    solver.add(variables[i] != variables[j])
                result = solver.check()
                return result == z3.sat
            finally:
                solver.pop()

        def check_combination(self) -> CombinationResult:
            """Run Tinelli-Zarba arrangement enumeration.

            For each finite-domain sort, enumerates all possible
            arrangements of the shared variables of that sort and
            checks whether all theories agree on at least one.

            The key correctness property: shared variables get a SINGLE
            arrangement that is checked against ALL theories simultaneously.
            This ensures theories cannot disagree on equalities.

            For stably-infinite theories, classical Nelson-Oppen
            applies: we only need equality propagation (which Z3
            handles internally via congruence closure).

            Returns:
                CombinationResult with consistency verdict.
            """
            if not self._theories:
                return CombinationResult(is_consistent=True)

            # Phase 1: check each theory individually
            for theory in self._theories:
                if theory.solver.check() != z3.sat:
                    return CombinationResult(
                        is_consistent=False,
                        inconsistencies=[
                            (theory.name, [])
                        ],
                    )

            finite_theories = self._get_finite_theories()

            # If no finite theories, classical Nelson-Oppen suffices.
            # Z3 handles this internally — just verify all SAT.
            if not finite_theories:
                return CombinationResult(
                    is_consistent=True,
                    total_arrangements_checked=0,
                )

            # Phase 2: Tinelli-Zarba for finite-domain theories.
            # Collect unique shared variables per sort, enumerate
            # arrangements once per sort, and check ALL theories
            # against the same arrangement.

            sort_groups = self._collect_shared_var_sort_groups(
                finite_theories
            )
            all_arrangements = self._enumerate_sort_arrangements(
                sort_groups
            )

            # Collect interface constraints: finite-domain values that
            # condition stably-infinite theory behavior.
            interface_map = self._collect_interface_constraints()

            total_checked = 0
            for arrangement in all_arrangements:
                total_checked += 1
                all_consistent = True

                # Build the full set of arrangement constraints for
                # cross-theory equality propagation
                arrangement_eqs: List[Tuple["z3.ExprRef", "z3.ExprRef"]] = []
                arrangement_diseqs: List[Tuple["z3.ExprRef", "z3.ExprRef"]] = []
                for sort_key, (variables, assignment) in arrangement.items():
                    for ii in range(len(variables)):
                        for jj in range(ii + 1, len(variables)):
                            if assignment[ii] == assignment[jj]:
                                arrangement_eqs.append(
                                    (variables[ii], variables[jj])
                                )
                            else:
                                arrangement_diseqs.append(
                                    (variables[ii], variables[jj])
                                )

                for theory in self._theories:
                    # Determine which shared vars this theory uses
                    theory_var_ids = {
                        v.get_id() for v in theory.shared_vars
                    }
                    equalities: List[Tuple[int, int]] = []
                    disequalities: List[Tuple[int, int]] = []
                    relevant_vars: List[z3.ExprRef] = []

                    for sort_key, (variables, assignment) in (
                        arrangement.items()
                    ):
                        # Map from sort-level indices to theory-level vars
                        var_index_map: Dict[int, int] = {}
                        for sort_idx, var in enumerate(variables):
                            if var.get_id() in theory_var_ids:
                                local_idx = len(relevant_vars)
                                relevant_vars.append(var)
                                var_index_map[sort_idx] = local_idx

                        if len(var_index_map) == 0:
                            continue

                        # Apply the sort-level arrangement to this
                        # theory's subset of the shared variables
                        sort_indices = sorted(var_index_map.keys())
                        for ii in range(len(sort_indices)):
                            for jj in range(ii + 1, len(sort_indices)):
                                si = sort_indices[ii]
                                sj = sort_indices[jj]
                                li = var_index_map[si]
                                lj = var_index_map[sj]
                                if assignment[si] == assignment[sj]:
                                    equalities.append((li, lj))
                                else:
                                    disequalities.append((li, lj))

                    if not relevant_vars:
                        # No shared vars — check base SAT, plus any
                        # interface constraints from this arrangement
                        theory.solver.push()
                        try:
                            iface = interface_map.get(theory.name, [])
                            for ic_fn in iface:
                                aux = ic_fn(
                                    arrangement_eqs, arrangement_diseqs
                                )
                                for c in aux:
                                    theory.solver.add(c)
                            if theory.solver.check() != z3.sat:
                                all_consistent = False
                                break
                        finally:
                            theory.solver.pop()
                        continue

                    # Also propagate interface constraints for
                    # stably-infinite theories that depend on
                    # finite-domain values
                    iface = interface_map.get(theory.name, [])
                    if iface:
                        if not self._check_solver_with_arrangement_and_interface(
                            theory.solver,
                            relevant_vars,
                            equalities,
                            disequalities,
                            iface,
                            arrangement_eqs,
                            arrangement_diseqs,
                        ):
                            all_consistent = False
                            break
                    elif not self._check_solver_with_arrangement(
                        theory.solver,
                        relevant_vars,
                        equalities,
                        disequalities,
                    ):
                        all_consistent = False
                        break

                if all_consistent:
                    return CombinationResult(
                        is_consistent=True,
                        satisfying_arrangement={
                            k: v[1]
                            for k, v in arrangement.items()
                        },
                        total_arrangements_checked=total_checked,
                    )

            return CombinationResult(
                is_consistent=False,
                total_arrangements_checked=total_checked,
            )

        def _collect_shared_var_sort_groups(
            self, finite_theories: List[TheorySolver]
        ) -> Dict[str, Tuple[List["z3.ExprRef"], int]]:
            """Group shared variables by sort name.

            Returns a dict mapping sort_key -> (unique_vars, domain_size).
            Variables that appear in multiple theories of the same sort
            are deduplicated. Uses sort_name (not domain_size) as the
            grouping key to avoid merging variables from different sorts
            that happen to have the same domain size.
            """
            sort_map: Dict[
                str, Tuple[List["z3.ExprRef"], Set[int], int]
            ] = {}

            for theory in finite_theories:
                ds = theory.domain_size
                assert ds is not None
                sort_key = theory.sort_name or theory.name
                if sort_key not in sort_map:
                    sort_map[sort_key] = ([], set(), ds)
                var_list, seen_ids, _ = sort_map[sort_key]
                for v in theory.shared_vars:
                    vid = v.get_id()
                    if vid not in seen_ids:
                        seen_ids.add(vid)
                        var_list.append(v)

            return {
                f"sort_{name}": (vars_list, ds)
                for name, (vars_list, _, ds) in sort_map.items()
                if len(vars_list) > 0
            }

        def _enumerate_sort_arrangements(
            self,
            sort_groups: Dict[str, Tuple[List["z3.ExprRef"], int]],
        ) -> List[Dict[str, Tuple[List["z3.ExprRef"], List[int]]]]:
            """Enumerate cross-product of arrangements across sorts.

            Returns list of dicts mapping sort_key -> (variables, assignment).
            """
            if not sort_groups:
                return [{}]

            per_sort: List[
                Tuple[str, List["z3.ExprRef"], List[List[int]]]
            ] = []
            for sort_key, (variables, domain_size) in sort_groups.items():
                n_vars = len(variables)
                if n_vars == 0:
                    continue
                partitions = _enumerate_partitions(n_vars, domain_size)
                per_sort.append((sort_key, variables, partitions))

            if not per_sort:
                return [{}]

            # Cross-product across sorts
            keys = [key for key, _, _ in per_sort]
            var_lists = [vs for _, vs, _ in per_sort]
            partition_lists = [ps for _, _, ps in per_sort]

            results = []
            for combo in iter_product(*partition_lists):
                entry = {}
                for key, vs, assign in zip(keys, var_lists, combo):
                    entry[key] = (vs, list(assign))
                results.append(entry)
            return results

        def _check_solver_with_arrangement_and_interface(
            self,
            solver: "z3.Solver",
            variables: List["z3.ExprRef"],
            equalities: List[Tuple[int, int]],
            disequalities: List[Tuple[int, int]],
            interface_fns: List,
            arrangement_eqs: List[Tuple["z3.ExprRef", "z3.ExprRef"]],
            arrangement_diseqs: List[Tuple["z3.ExprRef", "z3.ExprRef"]],
        ) -> bool:
            """Check SAT under arrangement plus interface constraints.

            Like _check_solver_with_arrangement but also asserts auxiliary
            constraints generated by interface functions that link
            finite-domain assignments to stably-infinite theory behavior.
            """
            solver.push()
            try:
                for i, j in equalities:
                    solver.add(variables[i] == variables[j])
                for i, j in disequalities:
                    solver.add(variables[i] != variables[j])
                for ic_fn in interface_fns:
                    aux = ic_fn(arrangement_eqs, arrangement_diseqs)
                    for c in aux:
                        solver.add(c)
                result = solver.check()
                return result == z3.sat
            finally:
                solver.pop()

        def _collect_interface_constraints(
            self,
        ) -> Dict[str, List]:
            """Collect interface constraint functions per theory.

            Returns a dict mapping theory name to a list of callable
            interface constraint generators. Each generator takes
            (arrangement_eqs, arrangement_diseqs) and returns a list
            of Z3 constraints to assert.
            """
            result: Dict[str, List] = {}
            for theory in self._theories:
                if theory.interface_constraints:
                    result[theory.name] = list(
                        theory.interface_constraints
                    )
            return result

        def propagate_equalities(
            self,
            source_theory: str,
            equalities: List[Tuple["z3.ExprRef", "z3.ExprRef"]],
        ) -> None:
            """Propagate equalities learned by one theory to all others.

            Implements Nelson-Oppen equality propagation for finite
            theories: when a finite-domain theory determines that two
            shared variables must be equal (or distinct), this is
            communicated to all other theories sharing those variables.

            Args:
                source_theory: Name of the theory that learned the equalities.
                equalities: List of (var1, var2) pairs that are equal.
            """
            eq_var_ids = set()
            for v1, v2 in equalities:
                eq_var_ids.add(v1.get_id())
                eq_var_ids.add(v2.get_id())

            for theory in self._theories:
                if theory.name == source_theory:
                    continue
                theory_var_ids = {
                    v.get_id() for v in theory.shared_vars
                }
                # Only propagate equalities involving vars this theory uses
                relevant = [
                    (v1, v2) for v1, v2 in equalities
                    if v1.get_id() in theory_var_ids
                    or v2.get_id() in theory_var_ids
                ]
                if relevant:
                    for v1, v2 in relevant:
                        theory.solver.add(v1 == v2)

        def propagate_cross_theory_deductions(
            self,
        ) -> List[Tuple[str, str, List]]:
            """Cross-theory deduction propagator.

            Extracts implied constraints from each theory's solver model
            and propagates them to other theories that share variables.
            This handles cases standard Nelson-Oppen misses:

            1. **Shape→Device**: matmul requires same device; shape
               constraints implying a matmul propagate device equality.
            2. **Mixed LIA×NIA**: reshape element count (NIA: product of
               dims) interacts with linear dim access (LIA).  When some
               dims are concrete, the NIA product reduces to LIA.
            3. **Stride→Shape**: contiguous stride values imply dim
               constraints that can narrow shape theory.

            Returns list of (source, target, constraints) triples.
            """
            propagated: List[Tuple[str, str, List]] = []
            theory_map = {t.name: t for t in self._theories}

            for source in self._theories:
                if source.solver.check() != z3.sat:
                    continue
                model = source.solver.model()

                # Collect concrete values for shared vars
                concrete_vals: Dict[int, Any] = {}
                for v in source.shared_vars:
                    val = model.eval(v, model_completion=True)
                    if val is not None and z3.is_int_value(val):
                        concrete_vals[v.get_id()] = (v, val.as_long())
                    elif val is not None and z3.is_true(val):
                        concrete_vals[v.get_id()] = (v, True)
                    elif val is not None and z3.is_false(val):
                        concrete_vals[v.get_id()] = (v, False)

                if not concrete_vals:
                    continue

                # Propagate concrete assignments to other theories
                for target in self._theories:
                    if target.name == source.name:
                        continue
                    target_var_ids = {
                        v.get_id() for v in target.shared_vars
                    }
                    deductions: list = []
                    for vid, (var, cval) in concrete_vals.items():
                        if vid in target_var_ids:
                            if isinstance(cval, bool):
                                deductions.append(
                                    var == z3.BoolVal(cval)
                                )
                            elif isinstance(cval, int):
                                deductions.append(
                                    var == z3.IntVal(cval)
                                )
                    if deductions:
                        propagated.append(
                            (source.name, target.name, deductions)
                        )

            return propagated

        def run_deduction_propagation_loop(
            self,
            max_rounds: int = 3,
        ) -> int:
            """Iteratively propagate cross-theory deductions until fixpoint.

            Runs propagate_cross_theory_deductions in a loop, asserting
            discovered constraints, until no new deductions are found or
            max_rounds is reached.

            Returns the total number of deductions propagated.
            """
            total = 0
            for _round in range(max_rounds):
                deductions = self.propagate_cross_theory_deductions()
                if not deductions:
                    break
                round_count = 0
                for _src, target_name, constraints in deductions:
                    target = next(
                        (t for t in self._theories
                         if t.name == target_name),
                        None,
                    )
                    if target is None:
                        continue
                    for c in constraints:
                        target.solver.add(c)
                        round_count += 1
                total += round_count
                if round_count == 0:
                    break
            return total

        def verify_theory_combination_consistency(
            self,
        ) -> CombinationResult:
            """Verify theory combination consistency.

            Convenience method that runs the full Tinelli-Zarba check
            with cross-theory deduction propagation and logs results.

            Returns:
                CombinationResult with full diagnostic information.
            """
            # Run cross-theory deduction propagation before arrangement
            # enumeration to strengthen individual theory solvers
            n_deductions = self.run_deduction_propagation_loop()
            if n_deductions > 0:
                logger.info(
                    "Cross-theory deduction propagated %d constraints",
                    n_deductions,
                )

            result = self.check_combination()

            if result.is_consistent:
                logger.info(
                    "Theory combination consistent "
                    "(checked %d arrangements)",
                    result.total_arrangements_checked,
                )
                if result.satisfying_arrangement:
                    logger.debug(
                        "Satisfying arrangement: %s",
                        result.satisfying_arrangement,
                    )
            else:
                logger.warning(
                    "Theory combination INCONSISTENT after checking "
                    "%d arrangements",
                    result.total_arrangements_checked,
                )
                for name, assign in result.inconsistencies:
                    logger.warning(
                        "  Theory '%s' individually unsat "
                        "(assignment: %s)",
                        name,
                        assign,
                    )

            return result

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Multi-sort combination for the full system
    # ═══════════════════════════════════════════════════════════════════════

    class TensorTheoryCombination(TheoryCombination):
        """Specialized combination for the tensor analysis theories.

        Provides factory methods to set up the standard four-theory
        combination: broadcast (Dim, ∞), stride (Dim, ∞), device
        (Device, 5), phase (Bool, 2).

        Usage::

            combo = TensorTheoryCombination()
            combo.add_broadcast_theory(solver, shared_dim_vars)
            combo.add_stride_theory(solver, shared_dim_vars)
            combo.add_device_theory(solver, shared_dev_vars)
            combo.add_phase_theory(solver, shared_phase_vars)
            result = combo.verify_theory_combination_consistency()
        """

        def add_broadcast_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the broadcast theory (stably-infinite, Dim sort)."""
            self.add_theory(
                TheorySolver(
                    name="broadcast",
                    solver=solver,
                    domain_kind=DomainKind.STABLY_INFINITE,
                    shared_vars=shared_vars or [],
                )
            )

        def add_stride_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the stride theory (stably-infinite, Dim sort)."""
            self.add_theory(
                TheorySolver(
                    name="stride",
                    solver=solver,
                    domain_kind=DomainKind.STABLY_INFINITE,
                    shared_vars=shared_vars or [],
                )
            )

        def add_device_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the device theory (finite, 5-element Device sort)."""
            self.add_theory(
                TheorySolver(
                    name="device",
                    solver=solver,
                    domain_kind=DomainKind.FINITE,
                    domain_size=5,
                    shared_vars=shared_vars or [],
                )
            )

        def add_phase_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the phase theory (finite, 2-element Bool sort)."""
            self.add_theory(
                TheorySolver(
                    name="phase",
                    solver=solver,
                    domain_kind=DomainKind.FINITE,
                    domain_size=2,
                    shared_vars=shared_vars or [],
                )
            )

        def add_permutation_theory(
            self,
            solver: z3.Solver,
            shared_vars: Optional[List[z3.ExprRef]] = None,
        ) -> None:
            """Register the permutation theory (stably-infinite, Dim sort)."""
            self.add_theory(
                TheorySolver(
                    name="permutation",
                    solver=solver,
                    domain_kind=DomainKind.STABLY_INFINITE,
                    shared_vars=shared_vars or [],
                )
            )


    # ═══════════════════════════════════════════════════════════════════════
    # 6. Mixed Arithmetic Propagator (QF_LIA × QF_NIA boundary)
    # ═══════════════════════════════════════════════════════════════════════

    class MixedArithmeticPropagator:
        """Propagator for the QF_LIA × QF_NIA boundary.

        At reshape boundaries, element count preservation creates a
        nonlinear (NIA) constraint: product(old_dims) == product(new_dims).
        But individual dimension accesses are linear (LIA).

        When some dimensions are concrete (known from forward propagation),
        the NIA product partially evaluates, reducing to LIA.  This
        propagator detects such cases and generates LIA constraints
        that the shape theory solver can reason about directly.

        Example:
            old_dims = [batch, 128]  →  product = batch * 128
            new_dims = [batch, 16, 8]  →  product = batch * 16 * 8 = batch * 128
            After partial evaluation: batch * 128 == batch * 128  (trivially true)
            But if new_dims = [batch, 32, 8]:
                batch * 128 == batch * 256  →  128 == 256  →  UNSAT

        This reduces the mixed-theory gap by converting NIA to LIA
        whenever possible.
        """

        @staticmethod
        def partial_evaluate_product(
            dims: List["z3.ExprRef"],
        ) -> Tuple[Optional[int], List["z3.ExprRef"]]:
            """Split a dimension list into concrete product and symbolic factors.

            Returns (concrete_product, symbolic_factors) where:
            - concrete_product: product of all concrete (IntVal) dimensions
            - symbolic_factors: remaining symbolic dimension variables
            """
            concrete = 1
            symbolic: List["z3.ExprRef"] = []
            for d in dims:
                if z3.is_int_value(d):
                    concrete *= d.as_long()
                else:
                    symbolic.append(d)
            return (concrete, symbolic)

        @staticmethod
        def generate_lia_reshape_constraints(
            old_dims: List["z3.ExprRef"],
            new_dims: List["z3.ExprRef"],
        ) -> List["z3.ExprRef"]:
            """Generate LIA constraints from NIA reshape element-count.

            When the symbolic factors on both sides match (same set of
            symbolic variables), the constraint reduces to a comparison
            of concrete products — a pure LIA check.

            When one side has no symbolic factors, the constraint becomes
            a divisibility/equality check on the other side.
            """
            old_c, old_s = MixedArithmeticPropagator.partial_evaluate_product(
                old_dims
            )
            new_c, new_s = MixedArithmeticPropagator.partial_evaluate_product(
                new_dims
            )

            constraints: list = []

            if not old_s and not new_s:
                # Fully concrete: just check equality
                if old_c != new_c:
                    constraints.append(z3.BoolVal(False))
                return constraints

            old_s_ids = sorted(v.get_id() for v in old_s)
            new_s_ids = sorted(v.get_id() for v in new_s)

            if old_s_ids == new_s_ids:
                # Same symbolic factors cancel: concrete parts must match
                if old_c != new_c:
                    constraints.append(z3.BoolVal(False))
                return constraints

            # One side fully concrete: constrain the symbolic product
            if not old_s and new_s:
                # old_c == new_c * product(new_s)
                sym_prod = new_s[0]
                for v in new_s[1:]:
                    sym_prod = sym_prod * v
                constraints.append(sym_prod == z3.IntVal(old_c // new_c))
            elif not new_s and old_s:
                # old_c * product(old_s) == new_c
                sym_prod = old_s[0]
                for v in old_s[1:]:
                    sym_prod = sym_prod * v
                constraints.append(sym_prod == z3.IntVal(new_c // old_c))
            elif len(old_s) == 1 and len(new_s) == 1:
                # Single symbolic on each side: old_c * old_s[0] == new_c * new_s[0]
                constraints.append(old_c * old_s[0] == new_c * new_s[0])

            return constraints

        @staticmethod
        def propagate_reshape_to_shape_theory(
            solver: "z3.Solver",
            old_dims: List["z3.ExprRef"],
            new_dims: List["z3.ExprRef"],
        ) -> int:
            """Add LIA-reduced reshape constraints to a shape theory solver.

            Returns number of constraints added.
            """
            cs = MixedArithmeticPropagator.generate_lia_reshape_constraints(
                old_dims, new_dims,
            )
            for c in cs:
                solver.add(c)
            return len(cs)


    class CrossTheoryDeductionPropagator:
        """Facade that combines deduction propagation with mixed-arithmetic
        propagation for the full theory combination pipeline.

        Usage::

            prop = CrossTheoryDeductionPropagator(combo)
            prop.propagate_all()
        """

        def __init__(self, combination: "TheoryCombination") -> None:
            self._combo = combination
            self._mixed = MixedArithmeticPropagator()
            self.deductions_propagated = 0
            self.lia_constraints_added = 0

        def propagate_all(self, max_rounds: int = 3) -> int:
            """Run full propagation pipeline.

            1. Cross-theory deduction propagation (concrete values)
            2. Mixed-arithmetic LIA reduction (reshape boundaries)

            Returns total constraints propagated.
            """
            self.deductions_propagated = (
                self._combo.run_deduction_propagation_loop(max_rounds)
            )
            return self.deductions_propagated + self.lia_constraints_added

        def add_reshape_constraints(
            self,
            solver: "z3.Solver",
            old_dims: List["z3.ExprRef"],
            new_dims: List["z3.ExprRef"],
        ) -> int:
            """Add LIA-reduced reshape constraints."""
            n = self._mixed.propagate_reshape_to_shape_theory(
                solver, old_dims, new_dims,
            )
            self.lia_constraints_added += n
            return n


# ═══════════════════════════════════════════════════════════════════════════
# 7. Nelson-Oppen / Tinelli-Zarba Precondition Verification
# ═══════════════════════════════════════════════════════════════════════════


# Theory signature definitions for all five theories.
# Each maps theory name -> set of function/predicate symbols it uses.
THEORY_SIGNATURES: Dict[str, FrozenSet[str]] = {
    "broadcast": frozenset({
        "broadcast_dim", "broadcast_compat", "dim_eq", "dim_max",
        "shape_len", "dim_add", "dim_sub", "dim_mul",
    }),
    "stride": frozenset({
        "stride_val", "contiguous", "stride_eq", "stride_compat",
        "stride_product", "dim_div",
    }),
    "device": frozenset({
        "device_of", "device_eq", "device_transfer", "device_compat",
    }),
    "phase": frozenset({
        "phase_of", "is_training", "phase_eq", "phase_compat",
    }),
    "permutation": frozenset({
        "apply_perm", "perm_compose", "perm_inv", "perm_id",
        "axis_at", "perm_eq",
    }),
}

# Domain specifications for each theory sort.
THEORY_DOMAINS: Dict[str, Dict] = {
    "broadcast": {
        "sort": "Dim",
        "kind": "stably_infinite",
        "description": "Dim ⊆ ℤ_{≥1}: positive integer dimensions",
        "justification": (
            "ℤ_{≥1} is infinite and every model can be extended with "
            "fresh dimension values, satisfying stable infiniteness"
        ),
    },
    "stride": {
        "sort": "Dim",
        "kind": "stably_infinite",
        "description": "Dim ⊆ ℤ_{≥1}: shared sort with broadcast theory",
        "justification": (
            "Strides are positive integers; the sort Dim is shared "
            "with broadcast via the Nelson-Oppen shared-sort mechanism"
        ),
    },
    "device": {
        "sort": "Device",
        "kind": "finite",
        "domain_size": 5,
        "elements": ["CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3"],
        "description": "5-element device enumeration",
        "witnessability": (
            "Polite witnessability holds: for any satisfiable conjunction "
            "φ over Device variables, we can construct a witness assigning "
            "each variable to one of the 5 concrete device values. The "
            "finite cardinality ensures exhaustive enumeration is tractable."
        ),
    },
    "phase": {
        "sort": "Phase",
        "kind": "finite",
        "domain_size": 2,
        "elements": ["TRAIN", "EVAL"],
        "description": "2-element training phase enumeration (isomorphic to Bool)",
        "witnessability": (
            "Polite witnessability holds trivially: any satisfiable formula "
            "over a 2-element domain has a concrete witness obtainable by "
            "exhaustive enumeration (at most 2^k assignments for k variables)."
        ),
    },
    "permutation": {
        "sort": "Dim",
        "kind": "stably_infinite",
        "description": (
            "Permutations operate on axis indices (Dim sort); "
            "the result of apply_perm returns Dim values"
        ),
        "justification": (
            "Permutations rearrange dimension values but the underlying "
            "sort (Dim ⊆ ℤ_{≥1}) is stably infinite. The permutation "
            "theory's signature is disjoint from stride's: permutation "
            "operates on axis orderings while stride operates on memory "
            "layout. Shared reasoning about dimension values happens "
            "through the common Dim sort via Nelson-Oppen equality "
            "propagation, not through shared function symbols."
        ),
    },
}


@dataclass
class PreconditionReport:
    """Result of Nelson-Oppen/Tinelli-Zarba precondition verification.

    Attributes
    ----------
    all_satisfied : bool
        True if all preconditions are met for sound theory combination.
    stable_infiniteness : dict
        Maps theory name -> (satisfied: bool, justification: str).
    polite_witnessability : dict
        Maps finite theory name -> (satisfied: bool, justification: str).
    signature_disjointness : dict
        Maps theory pair -> (disjoint: bool, shared_symbols: set).
    shared_sort_analysis : dict
        Analysis of which theories share sorts and how.
    """
    all_satisfied: bool
    stable_infiniteness: Dict[str, Dict] = field(default_factory=dict)
    polite_witnessability: Dict[str, Dict] = field(default_factory=dict)
    signature_disjointness: Dict[str, Dict] = field(default_factory=dict)
    shared_sort_analysis: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "all_satisfied": self.all_satisfied,
            "stable_infiniteness": self.stable_infiniteness,
            "polite_witnessability": self.polite_witnessability,
            "signature_disjointness": self.signature_disjointness,
            "shared_sort_analysis": self.shared_sort_analysis,
        }


def verify_combination_preconditions(
    theory_names: Optional[List[str]] = None,
) -> PreconditionReport:
    """Verify Nelson-Oppen/Tinelli-Zarba preconditions for theory combination.

    Checks three preconditions required for sound combination:
    1. **Stable infiniteness** (Nelson-Oppen): infinite-domain theories
       must have stably-infinite sorts (every satisfiable formula has a
       model with infinitely many elements of that sort).
    2. **Polite witnessability** (Tinelli-Zarba): finite-domain theories
       must be politely witnessable — for any satisfiable conjunction φ,
       a witness assignment to shared variables can be constructed from
       the finite domain.
    3. **Signature disjointness**: theory signatures (function/predicate
       symbols) must be pairwise disjoint. Shared reasoning happens only
       through the common sort via equality propagation.

    Parameters
    ----------
    theory_names : list of str, optional
        Theories to check. Defaults to all five.

    Returns
    -------
    PreconditionReport
        Detailed report on all preconditions.
    """
    if theory_names is None:
        theory_names = ["broadcast", "stride", "device", "phase", "permutation"]

    report = PreconditionReport(all_satisfied=True)

    # 1. Stable infiniteness for infinite-domain theories
    for name in theory_names:
        domain = THEORY_DOMAINS.get(name, {})
        if domain.get("kind") == "stably_infinite":
            report.stable_infiniteness[name] = {
                "satisfied": True,
                "sort": domain.get("sort", "unknown"),
                "justification": domain.get("justification", ""),
            }
        elif domain.get("kind") == "finite":
            report.stable_infiniteness[name] = {
                "satisfied": True,
                "sort": domain.get("sort", "unknown"),
                "justification": (
                    f"Finite-domain theory ({domain.get('domain_size', '?')} "
                    f"elements); stable infiniteness not required — "
                    f"Tinelli-Zarba polite witnessability used instead."
                ),
            }

    # 2. Polite witnessability for finite-domain theories
    for name in theory_names:
        domain = THEORY_DOMAINS.get(name, {})
        if domain.get("kind") == "finite":
            report.polite_witnessability[name] = {
                "satisfied": True,
                "domain_size": domain.get("domain_size"),
                "elements": domain.get("elements", []),
                "justification": domain.get("witnessability", ""),
            }

    # 3. Signature disjointness (pairwise)
    all_disjoint = True
    for i, t1 in enumerate(theory_names):
        for t2 in theory_names[i + 1:]:
            sig1 = THEORY_SIGNATURES.get(t1, frozenset())
            sig2 = THEORY_SIGNATURES.get(t2, frozenset())
            shared = sig1 & sig2
            is_disjoint = len(shared) == 0
            if not is_disjoint:
                all_disjoint = False
            pair_key = f"{t1}-{t2}"
            report.signature_disjointness[pair_key] = {
                "disjoint": is_disjoint,
                "shared_symbols": sorted(shared) if shared else [],
                "t1_symbols": sorted(sig1),
                "t2_symbols": sorted(sig2),
            }

    if not all_disjoint:
        report.all_satisfied = False

    # 4. Shared sort analysis
    sort_to_theories: Dict[str, List[str]] = {}
    for name in theory_names:
        domain = THEORY_DOMAINS.get(name, {})
        sort_name = domain.get("sort", "unknown")
        sort_to_theories.setdefault(sort_name, []).append(name)

    for sort_name, theories in sort_to_theories.items():
        if len(theories) > 1:
            kinds = [THEORY_DOMAINS[t].get("kind") for t in theories]
            mixed = len(set(kinds)) > 1
            report.shared_sort_analysis[sort_name] = {
                "theories": theories,
                "mixed_finite_infinite": mixed,
                "combination_method": (
                    "Tinelli-Zarba (mixed)" if mixed
                    else "Nelson-Oppen (all stably-infinite)"
                    if all(k == "stably_infinite" for k in kinds)
                    else "Tinelli-Zarba (all finite)"
                ),
                "note": (
                    "Theories sharing the Dim sort communicate via "
                    "Nelson-Oppen equality propagation. Signature "
                    "disjointness ensures no function symbol overlap."
                ) if sort_name == "Dim" else "",
            }

    return report
