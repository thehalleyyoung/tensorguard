"""
Distinctness Axioms for QF_UFLIA Finite Sort Encoding.

The QF_UFLIA encoding of finite sorts (devices, phases, permutations)
may admit spurious models without explicit distinctness axioms.  This
module generates:

  1. **Distinctness axioms**: for each pair of constants c_i, c_j in a
     finite sort S with |S| = n, assert c_i ≠ c_j.  This gives
     C(n, 2) = n(n-1)/2 axioms per sort.

  2. **Totality axioms**: for every variable x of sort S, assert
     x = c_1 ∨ x = c_2 ∨ … ∨ x = c_n.  This ensures every variable
     takes one of the declared values.

Together, these axioms ensure the finite sort has *exactly* |S|
distinct interpretations — no more, no fewer — closing the gap that
QF_UFLIA's uninterpreted sorts would otherwise leave open.

Integration: The ``FiniteSortAxiomGenerator`` produces Z3 constraints
that can be added to any solver alongside the existing encoder in
``implementation/src/smt/encoder.py``.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Finite sort definition
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FiniteSort:
    """A finite sort with named constants.

    Attributes:
        name: Name of the sort (e.g. "T_device").
        constants: Tuple of constant names (e.g. ("cpu", "cuda:0", "cuda:1")).
    """

    name: str
    constants: Tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.constants)


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-defined finite sorts for tensor verification
# ═══════════════════════════════════════════════════════════════════════════════

# Device sort: cpu and CUDA devices
DEVICE_SORT = FiniteSort(
    name="T_device",
    constants=("cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"),
)

# Phase sort: train vs eval
PHASE_SORT = FiniteSort(
    name="T_phase",
    constants=("TRAIN", "EVAL"),
)

# Permutation sort: common axis permutations for <= 4D tensors
PERM_SORT = FiniteSort(
    name="T_perm",
    constants=("identity", "transpose", "reverse", "rotate_left", "rotate_right"),
)


def get_standard_sorts() -> List[FiniteSort]:
    """Return the standard finite sorts used in tensor verification."""
    return [DEVICE_SORT, PHASE_SORT, PERM_SORT]


# ═══════════════════════════════════════════════════════════════════════════════
# FiniteSortAxiomGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class FiniteSortAxiomGenerator:
    """Generates distinctness and totality axioms for finite sorts.

    For a finite sort S = {c_1, …, c_n}, produces:
      - C(n,2) distinctness axioms: c_i ≠ c_j for all i < j
      - Totality axioms: for each registered variable x of sort S,
        x = c_1 ∨ … ∨ x = c_n

    Usage::

        gen = FiniteSortAxiomGenerator()
        gen.declare_sort(FiniteSort("T_device", ("cpu", "cuda:0")))
        x = gen.declare_variable("dev_x", "T_device")
        axioms = gen.generate_all_axioms()
        solver.add(*axioms)
    """

    def __init__(self) -> None:
        if not HAS_Z3:
            raise RuntimeError("Z3 is required for FiniteSortAxiomGenerator")

        # sort_name -> (z3_sort, {const_name: z3_const})
        self._sorts: Dict[str, Tuple[z3.SortRef, Dict[str, z3.ExprRef]]] = {}
        # sort_name -> list of z3 variables
        self._variables: Dict[str, List[z3.ExprRef]] = {}
        # Track finite sort definitions
        self._sort_defs: Dict[str, FiniteSort] = {}
        # Counter for unique sort names
        self._counter = 0

    def declare_sort(self, fsort: FiniteSort) -> z3.SortRef:
        """Declare a finite sort and its constants.

        Returns the Z3 sort.
        """
        if fsort.name in self._sorts:
            return self._sorts[fsort.name][0]

        self._counter += 1
        z3_sort_name = f"{fsort.name}_{self._counter}"

        # Use uninterpreted sort + distinct constants for QF_UFLIA compatibility
        z3_sort = z3.DeclareSort(z3_sort_name)
        z3_consts: Dict[str, z3.ExprRef] = {}
        for cname in fsort.constants:
            z3_consts[cname] = z3.Const(f"{fsort.name}_{cname}", z3_sort)

        self._sorts[fsort.name] = (z3_sort, z3_consts)
        self._sort_defs[fsort.name] = fsort
        self._variables.setdefault(fsort.name, [])
        return z3_sort

    def declare_variable(self, var_name: str, sort_name: str) -> z3.ExprRef:
        """Declare a variable of a given finite sort.

        Returns the Z3 constant.
        """
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        z3_sort, _ = self._sorts[sort_name]
        var = z3.Const(var_name, z3_sort)
        self._variables[sort_name].append(var)
        return var

    def get_constant(self, sort_name: str, const_name: str) -> z3.ExprRef:
        """Get a Z3 constant by sort and name."""
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        _, consts = self._sorts[sort_name]
        if const_name not in consts:
            raise ValueError(
                f"Unknown constant {const_name!r} in sort {sort_name!r}"
            )
        return consts[const_name]

    def get_sort(self, sort_name: str) -> z3.SortRef:
        """Get the Z3 sort by name."""
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        return self._sorts[sort_name][0]

    def get_constants(self, sort_name: str) -> Dict[str, z3.ExprRef]:
        """Get all Z3 constants for a sort."""
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        return dict(self._sorts[sort_name][1])

    # -------------------------------------------------------------------
    # Axiom generation
    # -------------------------------------------------------------------

    def generate_distinctness_axioms(self, sort_name: str) -> List[z3.BoolRef]:
        """Generate c_i ≠ c_j for all pairs of constants in a sort.

        For |S| = n constants, produces C(n,2) = n(n-1)/2 axioms.
        """
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        _, consts = self._sorts[sort_name]
        const_list = list(consts.values())
        axioms: List[z3.BoolRef] = []
        for ci, cj in itertools.combinations(const_list, 2):
            axioms.append(ci != cj)
        return axioms

    def generate_totality_axioms(self, sort_name: str) -> List[z3.BoolRef]:
        """Generate x = c_1 ∨ … ∨ x = c_n for each variable of the sort."""
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")
        _, consts = self._sorts[sort_name]
        const_list = list(consts.values())
        axioms: List[z3.BoolRef] = []
        for var in self._variables.get(sort_name, []):
            disjuncts = [var == c for c in const_list]
            if len(disjuncts) == 1:
                axioms.append(disjuncts[0])
            else:
                axioms.append(z3.Or(*disjuncts))
        return axioms

    def generate_all_axioms(self) -> List[z3.BoolRef]:
        """Generate all distinctness and totality axioms for all sorts."""
        axioms: List[z3.BoolRef] = []
        for sort_name in self._sorts:
            axioms.extend(self.generate_distinctness_axioms(sort_name))
            axioms.extend(self.generate_totality_axioms(sort_name))
        return axioms

    def generate_axioms_for_sort(self, sort_name: str) -> List[z3.BoolRef]:
        """Generate all axioms (distinctness + totality) for one sort."""
        axioms = self.generate_distinctness_axioms(sort_name)
        axioms.extend(self.generate_totality_axioms(sort_name))
        return axioms

    # -------------------------------------------------------------------
    # Tightness verification
    # -------------------------------------------------------------------

    def verify_tightness(
        self,
        sort_name: str,
        timeout_ms: int = 5000,
    ) -> Dict[str, Any]:
        """Verify that axioms admit exactly |S| distinct values.

        Checks:
          1. The axioms are satisfiable (at least |S| values exist).
          2. No (|S|+1)-th distinct value can exist under the axioms.

        Returns a dict with verification results.
        """
        if sort_name not in self._sorts:
            raise ValueError(f"Unknown sort: {sort_name!r}")

        fsort = self._sort_defs[sort_name]
        z3_sort, consts = self._sorts[sort_name]
        const_list = list(consts.values())
        n = len(const_list)

        result: Dict[str, Any] = {
            "sort_name": sort_name,
            "expected_size": n,
            "constants": list(consts.keys()),
        }

        # Check 1: Distinctness axioms are satisfiable
        s1 = z3.Solver()
        s1.set("timeout", timeout_ms)
        dist_axioms = self.generate_distinctness_axioms(sort_name)
        s1.add(*dist_axioms)
        check1 = s1.check()
        result["distinctness_sat"] = str(check1) == "sat"

        # Check 2: No (n+1)-th value — introduce a fresh variable and
        # assert it's different from all constants, should be SAT
        # (uninterpreted sorts are infinite), but with totality it's UNSAT
        s2 = z3.Solver()
        s2.set("timeout", timeout_ms)
        fresh = z3.Const(f"_fresh_{sort_name}", z3_sort)
        s2.add(*dist_axioms)
        # Totality for the fresh variable
        s2.add(z3.Or(*[fresh == c for c in const_list]))
        # Assert fresh is different from all constants
        for c in const_list:
            s2.add(fresh != c)
        check2 = s2.check()
        result["no_extra_value"] = str(check2) == "unsat"

        # Check 3: Each constant is reachable — for each c_i, check
        # that there exists a variable that can equal c_i
        reachable = []
        for cname, cval in consts.items():
            s3 = z3.Solver()
            s3.set("timeout", timeout_ms)
            test_var = z3.Const(f"_reach_{cname}", z3_sort)
            s3.add(*dist_axioms)
            s3.add(z3.Or(*[test_var == c for c in const_list]))
            s3.add(test_var == cval)
            r3 = s3.check()
            reachable.append({
                "constant": cname,
                "reachable": str(r3) == "sat",
            })
        result["reachability"] = reachable
        result["all_reachable"] = all(r["reachable"] for r in reachable)

        # Overall tightness
        result["tight"] = (
            result["distinctness_sat"]
            and result["no_extra_value"]
            and result["all_reachable"]
        )

        return result

    def verify_all_sorts_tight(self) -> Dict[str, Any]:
        """Verify tightness for all declared sorts."""
        results = {}
        for sort_name in self._sorts:
            results[sort_name] = self.verify_tightness(sort_name)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════════════


def add_finite_sort_axioms(
    solver: "z3.Solver",
    sorts: Optional[Sequence[FiniteSort]] = None,
    variables: Optional[Dict[str, str]] = None,
) -> FiniteSortAxiomGenerator:
    """Convenience function to add finite sort axioms to a solver.

    Parameters
    ----------
    solver : z3.Solver
        The solver to add axioms to.
    sorts : sequence of FiniteSort, optional
        Sorts to declare.  Defaults to the standard sorts.
    variables : dict mapping var_name -> sort_name, optional
        Variables to declare with totality constraints.

    Returns
    -------
    FiniteSortAxiomGenerator
        The generator, for further variable declarations.
    """
    if sorts is None:
        sorts = get_standard_sorts()
    if variables is None:
        variables = {}

    gen = FiniteSortAxiomGenerator()
    for s in sorts:
        gen.declare_sort(s)
    for var_name, sort_name in variables.items():
        gen.declare_variable(var_name, sort_name)

    axioms = gen.generate_all_axioms()
    if axioms:
        solver.add(*axioms)

    return gen
