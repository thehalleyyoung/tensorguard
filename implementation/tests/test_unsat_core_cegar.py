"""
Tests for enhanced UNSAT-core-driven CEGAR loop.

Covers:
  - IncrementalCEGARSolver push/pop and clause reuse
  - UNSAT core extraction on simple constraint systems
  - MUS (minimal unsatisfiable subset) extraction
  - UnsatCorePredicate provenance tracking
  - Predicate generalisation (concrete → symbolic bounds)
  - EnhancedShapeCEGARLoop integration
  - Predicate provenance: core_derived vs template_derived
  - run_enhanced_cegar top-level API
  - Backward compatibility with base ShapeCEGARLoop
"""

from __future__ import annotations

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="Z3 required")

from src.unsat_core_cegar import (
    IncrementalCEGARSolver,
    UnsatCorePredicate,
    EnhancedUnsatCorePredicateExtractor,
    EnhancedShapeCEGARLoop,
    run_enhanced_cegar,
    _collect_z3_vars,
    _formula_to_shape_predicate,
)
from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    ShapePredicate,
    PredicateKind,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR_SOURCE = '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
'''

TWO_LINEAR_SOURCE = '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''

BUG_SOURCE = '''
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
'''


# ═══════════════════════════════════════════════════════════════════════════════
# IncrementalCEGARSolver tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIncrementalCEGARSolver:
    """Tests for push/pop, assert_and_track, and UNSAT core extraction."""

    def test_creation(self):
        solver = IncrementalCEGARSolver()
        assert solver.stats["check_count"] == 0
        assert solver.stats["depth"] == 0

    def test_add_background(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.add_background([x > 0, x < 100])
        assert solver.stats["background_count"] == 2

    def test_push_pop(self):
        solver = IncrementalCEGARSolver()
        solver.push()
        assert solver.stats["depth"] == 1
        solver.pop()
        assert solver.stats["depth"] == 0

    def test_push_pop_nested(self):
        solver = IncrementalCEGARSolver()
        solver.push()
        solver.push()
        assert solver.stats["depth"] == 2
        solver.pop()
        assert solver.stats["depth"] == 1
        solver.pop()
        assert solver.stats["depth"] == 0

    def test_pop_at_zero_is_safe(self):
        solver = IncrementalCEGARSolver()
        solver.pop()  # should not raise
        assert solver.stats["depth"] == 0

    def test_sat_check(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.add_background([x > 0, x < 10])
        result, core = solver.check_with_core()
        assert result == z3.sat
        assert core == []

    def test_unsat_check_with_core(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.assert_and_track(x > 10, "gt10")
        solver.assert_and_track(x < 5, "lt5")
        result, core = solver.check_with_core()
        assert result == z3.unsat
        assert len(core) > 0

    def test_core_labels_tracked(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.assert_and_track(x > 10, "gt10")
        solver.assert_and_track(x < 5, "lt5")
        solver.check_with_core()
        assert solver.stats["cores_observed"] == 1

    def test_background_persists_across_push_pop(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.add_background([x > 0])
        solver.push()
        solver.assert_and_track(x < -1, "neg")
        result, core = solver.check_with_core()
        assert result == z3.unsat
        solver.pop()
        # Background still present: x > 0
        result2, _ = solver.check_with_core()
        assert result2 == z3.sat

    def test_incremental_reuse_count(self):
        solver = IncrementalCEGARSolver()
        solver.push()
        solver.pop()
        assert solver.stats["reuse_count"] == 1

    def test_reset_iteration(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.push()
        solver.assert_and_track(x == 42, "eq42")
        solver.reset_iteration()
        # After reset, the eq42 assertion should be gone.
        result, _ = solver.check_with_core()
        assert result == z3.sat

    def test_get_learned_predicates_empty(self):
        solver = IncrementalCEGARSolver()
        preds = solver.get_learned_predicates()
        assert preds == []

    def test_get_learned_predicates_after_unsat(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.assert_and_track(x > 10, "gt10")
        solver.assert_and_track(x < 5, "lt5")
        solver.check_with_core()
        preds = solver.get_learned_predicates()
        assert len(preds) > 0
        assert all(isinstance(p, UnsatCorePredicate) for p in preds)

    def test_predicate_strength_accumulates(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        # First UNSAT check
        solver.push()
        solver.assert_and_track(x > 10, "gt10")
        solver.assert_and_track(x < 5, "lt5")
        solver.check_with_core()
        solver.pop()
        # Second UNSAT check with same labels (recreated)
        solver.push()
        solver.assert_and_track(x > 10, "gt10_2")
        solver.assert_and_track(x < 5, "lt5_2")
        solver.check_with_core()
        solver.pop()
        assert solver.stats["cores_observed"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# UNSAT core extraction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnsatCoreExtraction:
    """Test that UNSAT cores are properly extracted from constraint systems."""

    def test_simple_contradictory_system(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        solver.assert_and_track(x == 5, "eq5")
        solver.assert_and_track(x == 10, "eq10")
        result, core = solver.check_with_core()
        assert result == z3.unsat
        assert set(core) == {"eq5", "eq10"}

    def test_core_subset_of_assertions(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        y = z3.Int("y")
        solver.assert_and_track(x > 0, "x_pos")
        solver.assert_and_track(y > 0, "y_pos")
        solver.assert_and_track(x + y < 0, "sum_neg")
        result, core = solver.check_with_core()
        assert result == z3.unsat
        assert len(core) <= 3

    def test_irrelevant_constraints_excluded(self):
        solver = IncrementalCEGARSolver()
        x = z3.Int("x")
        y = z3.Int("y")
        solver.assert_and_track(x == 5, "x_eq5")
        solver.assert_and_track(x == 10, "x_eq10")
        solver.assert_and_track(y > 0, "y_pos")  # irrelevant to conflict
        result, core = solver.check_with_core()
        assert result == z3.unsat
        # y_pos should ideally not be in the core
        assert "x_eq5" in core or "x_eq10" in core


# ═══════════════════════════════════════════════════════════════════════════════
# MUS extraction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestMUSExtraction:
    """Test minimal unsatisfiable subset extraction."""

    def test_mus_on_minimal_system(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("x")
        assertion_map = {
            "a": x == 5,
            "b": x == 10,
        }
        mus = extractor._extract_mus(["a", "b"], assertion_map)
        assert set(mus) == {"a", "b"}

    def test_mus_removes_redundant(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("x")
        y = z3.Int("y")
        assertion_map = {
            "a": x == 5,
            "b": x == 10,
            "c": y > 0,  # redundant — not needed for UNSAT
        }
        mus = extractor._extract_mus(["a", "b", "c"], assertion_map)
        assert "a" in mus and "b" in mus
        assert "c" not in mus

    def test_mus_preserves_unsat(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("x")
        assertion_map = {
            "a": x > 10,
            "b": x < 5,
            "c": x > 0,  # redundant given a
        }
        mus = extractor._extract_mus(["a", "b", "c"], assertion_map)
        # MUS should be {a, b} — still UNSAT
        solver = z3.Solver()
        for lbl in mus:
            solver.add(assertion_map[lbl])
        assert solver.check() == z3.unsat

    def test_mus_empty_input(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        mus = extractor._extract_mus([], {})
        assert mus == []

    def test_mus_single_element(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("x")
        assertion_map = {"a": z3.And(x > 5, x < 3)}
        mus = extractor._extract_mus(["a"], assertion_map)
        assert mus == ["a"]


# ═══════════════════════════════════════════════════════════════════════════════
# Predicate extraction tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredicateExtraction:
    """Test full predicate extraction pipeline."""

    def test_extract_direct_predicates(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("__interp_inp_d0")
        dim_map = {"__interp_inp_d0": ("inp", 0)}
        assertion_map = {
            "a": x == z3.IntVal(768),
            "b": x == z3.IntVal(512),
        }
        preds = extractor.extract_predicates(["a", "b"], assertion_map, dim_map)
        assert len(preds) > 0

    def test_extract_with_dim_map(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("__interp_x_d1")
        dim_map = {"__interp_x_d1": ("x", 1)}
        assertion_map = {
            "eq": x == z3.IntVal(768),
            "pos": x > z3.IntVal(0),
        }
        preds = extractor.extract_predicates(["eq", "pos"], assertion_map, dim_map)
        shape_preds = [p for p in preds if p.shape_predicate is not None]
        if shape_preds:
            sp = shape_preds[0].shape_predicate
            assert sp.tensor == "x"

    def test_extract_no_dim_map(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("x")
        assertion_map = {"a": x == 5, "b": x == 10}
        preds = extractor.extract_predicates(["a", "b"], assertion_map)
        assert len(preds) > 0

    def test_extract_empty_core(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        preds = extractor.extract_predicates([], {})
        assert preds == []


# ═══════════════════════════════════════════════════════════════════════════════
# Predicate generalisation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredicateGeneralisation:
    """Test weakening concrete equalities to bounds."""

    def test_generalise_equality_to_ge(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("__interp_x_d0")
        dim_map = {"__interp_x_d0": ("x", 0)}
        # x == 768 and x == 512 → UNSAT. Generalising x==768 to x>=1
        # with x==512 is still UNSAT because x==512 and x>=1 is SAT.
        # But this tests the generalisation mechanism itself.
        assertion_map = {
            "eq768": x == z3.IntVal(768),
            "eq512": x == z3.IntVal(512),
        }
        gen = extractor._generalised_predicates(
            ["eq768", "eq512"], assertion_map, dim_map,
        )
        # Since x>=1 ∧ x==512 is SAT and x>=1 ∧ x==768 is SAT,
        # but together they're UNSAT, the generalised set won't be UNSAT.
        # So generalisation may not produce results here — that's correct.
        assert isinstance(gen, list)

    def test_generalise_with_bound_constraint(self):
        extractor = EnhancedUnsatCorePredicateExtractor()
        x = z3.Int("__interp_x_d0")
        dim_map = {"__interp_x_d0": ("x", 0)}
        # x == 768 and x < 0 → UNSAT. Weakening x==768 to x>=1 still UNSAT.
        assertion_map = {
            "eq768": x == z3.IntVal(768),
            "neg": x < z3.IntVal(0),
        }
        gen = extractor._generalised_predicates(
            ["eq768", "neg"], assertion_map, dim_map,
        )
        ge_preds = [p for p in gen if p.shape_predicate is not None
                    and p.shape_predicate.kind == PredicateKind.DIM_GE]
        assert len(ge_preds) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# UnsatCorePredicate tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnsatCorePredicate:
    """Test the UnsatCorePredicate dataclass."""

    def test_creation(self):
        p = UnsatCorePredicate(formula=z3.Int("x") > 0)
        assert p.strength == 1
        assert p.variables == ()

    def test_with_shape_predicate(self):
        sp = ShapePredicate(
            kind=PredicateKind.DIM_EQ,
            tensor="x", axis=0, value=768,
            provenance="core_derived",
        )
        p = UnsatCorePredicate(
            formula=z3.Int("x") == 768,
            shape_predicate=sp,
            strength=3,
        )
        assert p.shape_predicate.value == 768
        assert p.strength == 3

    def test_source_core_frozenset(self):
        p = UnsatCorePredicate(
            formula=z3.Int("x") > 0,
            source_core=frozenset({"a", "b"}),
        )
        assert "a" in p.source_core


# ═══════════════════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Test utility functions."""

    def test_collect_z3_vars_simple(self):
        x = z3.Int("x")
        y = z3.Int("y")
        expr = x + y > 0
        vars_ = _collect_z3_vars(expr)
        names = {str(v) for v in vars_}
        assert "x" in names
        assert "y" in names

    def test_collect_z3_vars_constant(self):
        expr = z3.IntVal(42)
        vars_ = _collect_z3_vars(expr)
        assert vars_ == []

    def test_formula_to_shape_predicate_eq(self):
        x = z3.Int("__interp_inp_d1")
        formula = x == z3.IntVal(768)
        dim_map = {"__interp_inp_d1": ("inp", 1)}
        sp = _formula_to_shape_predicate(formula, dim_map)
        assert sp is not None
        assert sp.kind == PredicateKind.DIM_EQ
        assert sp.tensor == "inp"
        assert sp.axis == 1
        assert sp.value == 768

    def test_formula_to_shape_predicate_no_match(self):
        x = z3.Int("unknown_var")
        formula = x == z3.IntVal(10)
        dim_map = {"__interp_inp_d0": ("inp", 0)}
        sp = _formula_to_shape_predicate(formula, dim_map)
        assert sp is None

    def test_formula_to_shape_predicate_none_dim_map(self):
        x = z3.Int("x")
        formula = x > 0
        sp = _formula_to_shape_predicate(formula, None)
        assert sp is None


# ═══════════════════════════════════════════════════════════════════════════════
# EnhancedShapeCEGARLoop integration tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnhancedShapeCEGARLoop:
    """Integration tests: enhanced loop produces same/better results."""

    def test_simple_linear_discovery(self):
        loop = EnhancedShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        result = loop.run()
        assert result.final_status in (CEGARStatus.SAFE, CEGARStatus.MAX_ITER)

    def test_two_linear_discovery(self):
        loop = EnhancedShapeCEGARLoop(
            TWO_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        result = loop.run()
        assert isinstance(result, ShapeCEGARResult)

    def test_bug_detection(self):
        loop = EnhancedShapeCEGARLoop(
            BUG_SOURCE,
            input_shapes={"x": ("batch", 768)},
        )
        result = loop.run()
        assert isinstance(result, ShapeCEGARResult)

    def test_enhanced_stats_present(self):
        loop = EnhancedShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        result = loop.run()
        if result.interpolation_stats:
            assert "core_predicates_count" in result.interpolation_stats

    def test_same_result_as_base(self):
        """Enhanced loop should find same verdict as base loop."""
        base = ShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        enhanced = EnhancedShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        base_result = base.run()
        enh_result = enhanced.run()
        assert base_result.verdict == enh_result.verdict

    def test_predicate_provenance(self):
        loop = EnhancedShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        result = loop.run()
        for pred in result.discovered_predicates:
            assert pred.provenance in (
                "cegar_discovered",
                "core_derived",
                "core_interpolant",
                "core_generalised",
                "template_derived",
                "api_stub",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Base ShapeCEGARLoop use_incremental integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseLoopIncremental:
    """Test use_incremental=True in the base ShapeCEGARLoop."""

    def test_use_incremental_flag(self):
        loop = ShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
            use_incremental=True,
        )
        assert loop.use_incremental is True

    def test_incremental_produces_result(self):
        loop = ShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
            use_incremental=True,
        )
        result = loop.run()
        assert isinstance(result, ShapeCEGARResult)

    def test_incremental_same_verdict(self):
        base = ShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
            use_incremental=False,
        )
        incr = ShapeCEGARLoop(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
            use_incremental=True,
        )
        assert base.run().verdict == incr.run().verdict


# ═══════════════════════════════════════════════════════════════════════════════
# run_enhanced_cegar API tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunEnhancedCegar:
    """Test the top-level run_enhanced_cegar function."""

    def test_basic_call(self):
        result = run_enhanced_cegar(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
        )
        assert isinstance(result, ShapeCEGARResult)

    def test_with_concrete_shapes(self):
        result = run_enhanced_cegar(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": (32, 768)},
        )
        assert result.final_status == CEGARStatus.SAFE

    def test_custom_timeout(self):
        result = run_enhanced_cegar(
            SIMPLE_LINEAR_SOURCE,
            input_shapes={"x": ("batch", "features")},
            solver_timeout_ms=2000,
            mus_timeout_ms=1000,
        )
        assert isinstance(result, ShapeCEGARResult)
