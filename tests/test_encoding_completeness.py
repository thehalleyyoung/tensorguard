"""
Tests for model-theoretic completeness and Craig interpolation projection.

Deliverable 3: comprehensive tests for encoding_completeness.py and
the projection completeness additions to craig_interpolation.py.
"""

from __future__ import annotations

import json
import os
import sys
import pytest

# Ensure the implementation package is importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir,
    ),
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")


# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

from src.smt.distinctness_axioms import (
    FiniteSort,
    FiniteSortAxiomGenerator,
    DEVICE_SORT,
    PHASE_SORT,
    PERM_SORT,
    get_standard_sorts,
)
from src.smt.encoding_completeness import (
    CategoricityResult,
    PermutationGroupResult,
    verify_categoricity,
    verify_permutation_group,
    verify_device_theory_completeness,
    verify_phase_theory_completeness,
    verify_perm_theory_completeness,
    verify_all_encoding_completeness,
    assert_no_spurious_models,
    assert_no_missing_constraints,
    assert_permutation_group_correct,
)
from src.craig_interpolation import (
    DimMapping,
    InterpolationPredicateDiscovery,
    InterpolationMethod,
    LinearComboPredicate,
    ProductEqualityPredicate,
    parse_interpolant,
    verify_projection_completeness,
    demonstrate_lossy_projection,
    ProjectionResult,
    TEMPLATE_KINDS,
)
from src.shape_cegar import ShapePredicate, PredicateKind


# ═══════════════════════════════════════════════════════════════════════════════
# Part A: Categoricity Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCategoricityDeviceSort:
    """Tests that T_device encoding is categorical."""

    def test_device_sort_satisfiable(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.satisfiable, "T_device axioms must be satisfiable"

    def test_device_sort_no_extra_elements(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.no_extra_elements, (
            "No model of T_device should have more than 5 elements"
        )

    def test_device_sort_all_reachable(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.all_reachable, "All device constants must be reachable"

    def test_device_sort_no_missing_constraints(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.no_missing_constraints, (
            "All distinctness constraints must hold"
        )

    def test_device_sort_is_categorical(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.is_categorical, "T_device must be categorical"

    def test_device_sort_expected_cardinality(self):
        cat = verify_categoricity(DEVICE_SORT)
        assert cat.expected_cardinality == 5


class TestCategoricityPhaseSort:
    """Tests that T_phase encoding is categorical."""

    def test_phase_sort_is_categorical(self):
        cat = verify_categoricity(PHASE_SORT)
        assert cat.is_categorical, "T_phase must be categorical"

    def test_phase_sort_cardinality(self):
        cat = verify_categoricity(PHASE_SORT)
        assert cat.expected_cardinality == 2


class TestCategoricityPermSort:
    """Tests that T_perm encoding is categorical."""

    def test_perm_sort_is_categorical(self):
        cat = verify_categoricity(PERM_SORT)
        assert cat.is_categorical, "T_perm must be categorical"

    def test_perm_sort_cardinality(self):
        cat = verify_categoricity(PERM_SORT)
        assert cat.expected_cardinality == 5


class TestCategoricityCustomSorts:
    """Tests categoricity for arbitrary finite sorts."""

    def test_singleton_sort(self):
        s = FiniteSort("singleton", ("only",))
        cat = verify_categoricity(s)
        assert cat.is_categorical

    def test_pair_sort(self):
        s = FiniteSort("pair", ("a", "b"))
        cat = verify_categoricity(s)
        assert cat.is_categorical

    def test_large_sort(self):
        names = tuple(f"c{i}" for i in range(10))
        s = FiniteSort("large", names)
        cat = verify_categoricity(s)
        assert cat.is_categorical


# ═══════════════════════════════════════════════════════════════════════════════
# Part B: Permutation Group Axiom Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPermutationGroup:
    """Tests S_n group axioms for n=1..4."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_identity_axiom(self, n):
        gr = verify_permutation_group(n)
        assert gr.identity_axiom, f"S_{n} identity axiom failed"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_closure_axiom(self, n):
        gr = verify_permutation_group(n)
        assert gr.closure_axiom, f"S_{n} closure axiom failed"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_associativity_axiom(self, n):
        gr = verify_permutation_group(n)
        assert gr.associativity_axiom, f"S_{n} associativity axiom failed"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_inverse_axiom(self, n):
        gr = verify_permutation_group(n)
        assert gr.inverse_axiom, f"S_{n} inverse axiom failed"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_composition_correctness(self, n):
        gr = verify_permutation_group(n)
        assert gr.composition_correctness, f"S_{n} composition failed"

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_all_axioms_hold(self, n):
        gr = verify_permutation_group(n)
        assert gr.all_axioms_hold, f"S_{n}: not all axioms hold"

    def test_permutation_counts(self):
        """Verify |S_n| = n! for n=1..4."""
        import math
        for n in range(1, 5):
            gr = verify_permutation_group(n)
            assert gr.num_permutations == math.factorial(n)


# ═══════════════════════════════════════════════════════════════════════════════
# Part C: Theory-Specific Completeness Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeviceTheoryCompleteness:
    """Tests device theory completeness."""

    def test_device_theory_complete(self):
        result = verify_device_theory_completeness()
        assert result["device_theory_complete"]

    def test_same_device_expressible(self):
        result = verify_device_theory_completeness()
        assert result["same_device_expressible"]

    def test_transfer_expressible(self):
        result = verify_device_theory_completeness()
        assert result["transfer_expressible"]


class TestPhaseTheoryCompleteness:
    """Tests phase theory completeness."""

    def test_phase_theory_complete(self):
        result = verify_phase_theory_completeness()
        assert result["phase_theory_complete"]

    def test_no_third_phase_value(self):
        result = verify_phase_theory_completeness()
        assert result["no_third_value"]


class TestPermTheoryCompleteness:
    """Tests permutation theory completeness."""

    def test_perm_theory_complete(self):
        result = verify_perm_theory_completeness(max_n=3)
        assert result["perm_theory_complete"]


# ═══════════════════════════════════════════════════════════════════════════════
# Part D: Assertion-Based Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssertions:
    """Tests the assertion-based verification functions."""

    def test_assert_no_spurious_models_device(self):
        assert_no_spurious_models(DEVICE_SORT)

    def test_assert_no_spurious_models_phase(self):
        assert_no_spurious_models(PHASE_SORT)

    def test_assert_no_spurious_models_perm(self):
        assert_no_spurious_models(PERM_SORT)

    def test_assert_no_missing_constraints_all(self):
        for fsort in get_standard_sorts():
            assert_no_missing_constraints(fsort)

    def test_assert_permutation_group_correct(self):
        assert_permutation_group_correct(max_n=3)


# ═══════════════════════════════════════════════════════════════════════════════
# Part E: Craig Interpolation Projection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestProjectionCompleteness:
    """Tests projection from QF_UFLIA interpolants to template language."""

    def test_template_kinds_count(self):
        """Verify the template language has the expected number of kinds."""
        assert len(TEMPLATE_KINDS) == 9

    def test_lossless_projection_dim_eq(self):
        """A simple d0 == 3 interpolant should project losslessly."""
        d0 = z3.Int("d0")
        interp = d0 == 3

        dm = DimMapping()
        dm.register("d0", "x", 0)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless
        assert result.total_conjuncts == 1
        assert result.projected_conjuncts == 1
        assert len(result.projected_predicates) == 1

    def test_lossless_projection_dim_ge(self):
        """d0 >= 1 should project losslessly to DIM_GE."""
        d0 = z3.Int("d0")
        interp = d0 >= 1

        dm = DimMapping()
        dm.register("d0", "x", 0)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless

    def test_lossless_projection_dim_match(self):
        """d0 == d1 should project losslessly to DIM_MATCH."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        interp = d0 == d1

        dm = DimMapping()
        dm.register("d0", "x", 0)
        dm.register("d1", "w", 0)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless

    def test_lossless_projection_conjunction(self):
        """Conjunction of projectable atoms should be lossless."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        interp = z3.And(d0 >= 1, d1 >= 1, d0 == d1)

        dm = DimMapping()
        dm.register("d0", "x", 0)
        dm.register("d1", "w", 0)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless
        assert result.total_conjuncts == 3
        assert result.projected_conjuncts == 3

    def test_lossless_projection_linear_combo(self):
        """2*d0 + d1 >= 5 should project as DIM_LINEAR_COMBO."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        interp = 2 * d0 + d1 >= 5

        dm = DimMapping()
        dm.register("d0", "x", 0)
        dm.register("d1", "x", 1)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless

    def test_lossy_projection_disjunction(self):
        """Disjunctive interpolant d0==3 ∨ d0==7 is lossy."""
        result = demonstrate_lossy_projection()
        assert not result.is_lossless
        assert result.total_conjuncts == 1
        assert result.projected_conjuncts == 0
        assert len(result.unprojected_conjuncts) == 1

    def test_lossy_preserves_soundness(self):
        """Even lossy projections preserve soundness."""
        result = demonstrate_lossy_projection()
        assert result.soundness_preserved

    def test_true_interpolant(self):
        """True interpolant projects losslessly (trivially)."""
        interp = z3.BoolVal(True)
        dm = DimMapping()
        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless

    def test_projection_with_product(self):
        """Product d0 * d1 == 64 should project as DIM_PRODUCT_EQ."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        interp = d0 * d1 == 64

        dm = DimMapping()
        dm.register("d0", "x", 2)
        dm.register("d1", "x", 3)

        result = verify_projection_completeness(interp, dm)
        assert result.is_lossless
        assert any(
            isinstance(p, ProductEqualityPredicate)
            for p in result.projected_predicates
        )

    def test_projection_soundness_with_path_constraints(self):
        """Verify A ⊨ projected_I when path constraints are provided."""
        d0 = z3.Int("d0")
        path = [d0 >= 5, d0 <= 10]
        interp = d0 >= 5

        dm = DimMapping()
        dm.register("d0", "x", 0)

        result = verify_projection_completeness(
            interp, dm, path_constraints=path
        )
        assert result.soundness_preserved
        assert result.is_lossless


# ═══════════════════════════════════════════════════════════════════════════════
# Part F: Integration — run all and save results
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullVerificationAndSave:
    """Integration test that runs all verification and saves results."""

    def test_full_verification(self):
        results = verify_all_encoding_completeness()
        assert results["all_complete"], "Not all encodings verified complete"

    def test_save_results(self):
        """Run all checks and save to encoding_completeness_results.json."""
        results = verify_all_encoding_completeness()

        # Add projection completeness examples
        projection_examples = {}

        # Lossless example
        d0, d1 = z3.Int("d0"), z3.Int("d1")
        dm = DimMapping()
        dm.register("d0", "x", 0)
        dm.register("d1", "x", 1)
        lossless_result = verify_projection_completeness(
            z3.And(d0 >= 1, d1 == d0), dm
        )
        projection_examples["lossless"] = lossless_result.summary()

        # Lossy example
        lossy_result = demonstrate_lossy_projection()
        projection_examples["lossy_disjunction"] = lossy_result.summary()

        # Product example
        dm2 = DimMapping()
        dm2.register("d0", "x", 2)
        dm2.register("d1", "x", 3)
        product_result = verify_projection_completeness(
            d0 * d1 == 64, dm2
        )
        projection_examples["product_equality"] = product_result.summary()

        results["projection_completeness"] = projection_examples
        results["template_language_kinds"] = TEMPLATE_KINDS

        output_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, "experiments"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, "encoding_completeness_results.json"
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        assert os.path.exists(output_path)
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["all_complete"]
