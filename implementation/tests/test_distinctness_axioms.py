"""
Tests for QF_UFLIA distinctness axioms.

Verifies:
  - Distinctness axiom generation for each finite sort
  - Totality axiom generation
  - Tightness verification (exactly |S| values)
  - Integration with SMT solver
  - Spurious model prevention
"""

from __future__ import annotations

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")

from src.smt.distinctness_axioms import (
    DEVICE_SORT,
    PHASE_SORT,
    PERM_SORT,
    FiniteSort,
    FiniteSortAxiomGenerator,
    add_finite_sort_axioms,
    get_standard_sorts,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def gen():
    """A fresh FiniteSortAxiomGenerator."""
    return FiniteSortAxiomGenerator()


@pytest.fixture
def device_gen(gen):
    """Generator with device sort declared."""
    gen.declare_sort(DEVICE_SORT)
    return gen


@pytest.fixture
def phase_gen(gen):
    """Generator with phase sort declared."""
    gen.declare_sort(PHASE_SORT)
    return gen


# ======================================================================
# Test: FiniteSort dataclass
# ======================================================================


class TestFiniteSort:
    def test_device_sort_size(self):
        assert DEVICE_SORT.size == 5

    def test_phase_sort_size(self):
        assert PHASE_SORT.size == 2

    def test_perm_sort_size(self):
        assert PERM_SORT.size == 5

    def test_custom_sort(self):
        s = FiniteSort("Test", ("a", "b", "c"))
        assert s.size == 3
        assert s.name == "Test"


# ======================================================================
# Test: Distinctness axiom count
# ======================================================================


class TestDistinctnessAxiomCount:
    def test_device_distinctness_count(self, device_gen):
        axioms = device_gen.generate_distinctness_axioms("T_device")
        # C(5, 2) = 10
        assert len(axioms) == 10

    def test_phase_distinctness_count(self, phase_gen):
        axioms = phase_gen.generate_distinctness_axioms("T_phase")
        # C(2, 2) = 1
        assert len(axioms) == 1

    def test_perm_distinctness_count(self, gen):
        gen.declare_sort(PERM_SORT)
        axioms = gen.generate_distinctness_axioms("T_perm")
        # C(5, 2) = 10
        assert len(axioms) == 10

    def test_singleton_sort(self, gen):
        s = FiniteSort("Singleton", ("only",))
        gen.declare_sort(s)
        axioms = gen.generate_distinctness_axioms("Singleton")
        assert len(axioms) == 0  # C(1,2) = 0


# ======================================================================
# Test: Totality axioms
# ======================================================================


class TestTotalityAxioms:
    def test_no_variables_no_axioms(self, device_gen):
        axioms = device_gen.generate_totality_axioms("T_device")
        assert len(axioms) == 0

    def test_one_variable(self, device_gen):
        device_gen.declare_variable("x", "T_device")
        axioms = device_gen.generate_totality_axioms("T_device")
        assert len(axioms) == 1

    def test_multiple_variables(self, device_gen):
        device_gen.declare_variable("x", "T_device")
        device_gen.declare_variable("y", "T_device")
        device_gen.declare_variable("z", "T_device")
        axioms = device_gen.generate_totality_axioms("T_device")
        assert len(axioms) == 3

    def test_totality_is_satisfiable(self, device_gen):
        x = device_gen.declare_variable("x", "T_device")
        axioms = device_gen.generate_all_axioms()
        s = z3.Solver()
        s.add(*axioms)
        assert s.check() == z3.sat


# ======================================================================
# Test: Distinctness axioms prevent merging
# ======================================================================


class TestDistinctnessPreventsMerging:
    def test_two_vars_can_differ(self, device_gen):
        """With distinctness, two vars assigned to different constants is SAT."""
        x = device_gen.declare_variable("x", "T_device")
        y = device_gen.declare_variable("y", "T_device")
        cpu = device_gen.get_constant("T_device", "cpu")
        cuda0 = device_gen.get_constant("T_device", "cuda:0")
        axioms = device_gen.generate_all_axioms()
        s = z3.Solver()
        s.add(*axioms)
        s.add(x == cpu)
        s.add(y == cuda0)
        assert s.check() == z3.sat

    def test_distinctness_blocks_equality(self, device_gen):
        """Distinctness ensures cpu != cuda:0."""
        cpu = device_gen.get_constant("T_device", "cpu")
        cuda0 = device_gen.get_constant("T_device", "cuda:0")
        axioms = device_gen.generate_distinctness_axioms("T_device")
        s = z3.Solver()
        s.add(*axioms)
        s.add(cpu == cuda0)
        assert s.check() == z3.unsat


# ======================================================================
# Test: Tightness verification
# ======================================================================


class TestTightness:
    def test_device_sort_tight(self, device_gen):
        device_gen.declare_variable("test_v", "T_device")
        result = device_gen.verify_tightness("T_device")
        assert result["distinctness_sat"] is True
        assert result["no_extra_value"] is True
        assert result["all_reachable"] is True
        assert result["tight"] is True

    def test_phase_sort_tight(self, phase_gen):
        phase_gen.declare_variable("test_v", "T_phase")
        result = phase_gen.verify_tightness("T_phase")
        assert result["tight"] is True
        assert result["expected_size"] == 2

    def test_all_standard_sorts_tight(self, gen):
        for sort in get_standard_sorts():
            gen.declare_sort(sort)
            gen.declare_variable(f"v_{sort.name}", sort.name)
        results = gen.verify_all_sorts_tight()
        for sort_name, r in results.items():
            assert r["tight"] is True, f"Sort {sort_name} is not tight"


# ======================================================================
# Test: Spurious model prevention
# ======================================================================


class TestSpuriousModelPrevention:
    def test_without_distinctness_spurious_model_possible(self):
        """Without distinctness, an uninterpreted sort admits collapsed models."""
        s = z3.DeclareSort("NaiveDevice")
        cpu = z3.Const("cpu_n", s)
        cuda0 = z3.Const("cuda0_n", s)
        solver = z3.Solver()
        # Without distinctness, cpu == cuda0 is possible
        solver.add(cpu == cuda0)
        assert solver.check() == z3.sat

    def test_with_distinctness_no_spurious(self, device_gen):
        """With distinctness, collapsing constants is blocked."""
        cpu = device_gen.get_constant("T_device", "cpu")
        cuda0 = device_gen.get_constant("T_device", "cuda:0")
        axioms = device_gen.generate_distinctness_axioms("T_device")
        solver = z3.Solver()
        solver.add(*axioms)
        solver.add(cpu == cuda0)
        assert solver.check() == z3.unsat

    def test_totality_forces_known_value(self, device_gen):
        """With totality, a variable must equal one of the constants."""
        x = device_gen.declare_variable("x", "T_device")
        consts = device_gen.get_constants("T_device")
        axioms = device_gen.generate_all_axioms()
        solver = z3.Solver()
        solver.add(*axioms)
        # Assert x is different from ALL constants -> UNSAT
        for c in consts.values():
            solver.add(x != c)
        assert solver.check() == z3.unsat


# ======================================================================
# Test: Integration with solver
# ======================================================================


class TestIntegration:
    def test_add_finite_sort_axioms(self):
        solver = z3.Solver()
        gen = add_finite_sort_axioms(
            solver,
            sorts=[DEVICE_SORT, PHASE_SORT],
            variables={"dev": "T_device", "phase": "T_phase"},
        )
        assert solver.check() == z3.sat

    def test_convenience_function_default_sorts(self):
        solver = z3.Solver()
        gen = add_finite_sort_axioms(solver)
        assert solver.check() == z3.sat

    def test_get_constant_after_declare(self):
        gen = FiniteSortAxiomGenerator()
        gen.declare_sort(PHASE_SORT)
        train = gen.get_constant("T_phase", "TRAIN")
        ev = gen.get_constant("T_phase", "EVAL")
        assert train is not None
        assert ev is not None

    def test_unknown_sort_raises(self):
        gen = FiniteSortAxiomGenerator()
        with pytest.raises(ValueError, match="Unknown sort"):
            gen.generate_distinctness_axioms("nonexistent")

    def test_unknown_constant_raises(self):
        gen = FiniteSortAxiomGenerator()
        gen.declare_sort(PHASE_SORT)
        with pytest.raises(ValueError, match="Unknown constant"):
            gen.get_constant("T_phase", "INVALID")

    def test_bidirectional_simulation(self):
        """Two device variables with same-device constraint + axioms."""
        gen = FiniteSortAxiomGenerator()
        gen.declare_sort(DEVICE_SORT)
        x = gen.declare_variable("dev_a", "T_device")
        y = gen.declare_variable("dev_b", "T_device")
        axioms = gen.generate_all_axioms()

        solver = z3.Solver()
        solver.add(*axioms)
        solver.add(x == y)  # same device
        cpu = gen.get_constant("T_device", "cpu")
        solver.add(x == cpu)
        assert solver.check() == z3.sat
