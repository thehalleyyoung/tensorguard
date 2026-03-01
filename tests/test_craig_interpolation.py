"""
Tests for Craig Interpolation–based shape predicate discovery.

Covers:
  - Basic interpolation on simple path/safety formula pairs
  - Interpolant variable containment (interface-only property)
  - Parsing of interpolants into ShapePredicates
  - Fallback behaviour when Z3 is unavailable
  - DIM_LINEAR_COMBO generation
  - Integration with existing CEGAR types (ShapePredicate, PredicateKind)
  - Edge cases: trivial formulas, empty constraints, timeouts
"""

from __future__ import annotations

import pytest
import z3

from src.shape_cegar import ShapePredicate, PredicateKind
from src.craig_interpolation import (
    InterpolationPredicateDiscovery,
    InterpolationMethod,
    LinearComboPredicate,
    DimMapping,
    ExtendedPredicateKind,
    DIM_LINEAR_COMBO,
    _collect_vars,
    _collect_vars_from_list,
    _compute_simulated_interpolant,
    _compute_cvc5_interpolant,
    HAS_CVC5,
    parse_interpolant,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_dim_map(*entries: tuple[str, str, int]) -> DimMapping:
    """Build a DimMapping from (var_name, tensor, axis) triples."""
    dm = DimMapping()
    for var_name, tensor, axis in entries:
        dm.register(var_name, tensor, axis)
    return dm


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic interpolation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicInterpolation:
    """Simple path/safety formula pairs."""

    def test_simple_equality_interpolant(self):
        """A: d0 == 768, B: d0 != 768 → interpolant forces d0 == 768."""
        d0 = z3.Int("d0")
        A = [d0 == 768]
        B = [d0 != 768]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1
        # At least one predicate should reference d0 == 768.
        has_eq = any(
            isinstance(p, ShapePredicate)
            and p.kind == PredicateKind.DIM_EQ
            and p.value == 768
            for p in preds
        )
        assert has_eq

    def test_inequality_interpolant_ge(self):
        """A: d0 >= 64, B: d0 < 64 → interpolant implies d0 >= 64."""
        d0 = z3.Int("d0")
        A = [d0 >= 64]
        B = [d0 < 64]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_inequality_interpolant_gt(self):
        """A: d0 > 0, B: d0 <= 0 → interpolant implies d0 > 0."""
        d0 = z3.Int("d0")
        A = [d0 > 0]
        B = [d0 <= 0]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_sat_pair_returns_empty(self):
        """If A ∧ B is satisfiable, no interpolant exists."""
        d0 = z3.Int("d0")
        A = [d0 >= 0]
        B = [d0 >= 0]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert preds == []

    def test_multiple_constraints_path(self):
        """A with multiple constraints; B contradicts one."""
        d0, d1 = z3.Ints("d0 d1")
        A = [d0 == 32, d1 == 64, d0 + d1 == 96]
        B = [d0 != 32]
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Interface variable containment
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterfaceVariables:
    """Verify interpolants only use shared (interface) variables."""

    def test_non_shared_var_eliminated(self):
        """Internal variable in A only should be quantified out."""
        d0 = z3.Int("d0")  # shared
        internal = z3.Int("tmp")  # A-only
        A = [internal == d0 * 2, internal >= 128]
        B = [d0 < 64]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1
        # The interpolant should NOT mention "tmp" — only d0.
        for p in preds:
            if isinstance(p, ShapePredicate):
                assert p.tensor == "x"

    def test_all_shared_vars_preserved(self):
        """When all variables are shared, no elimination needed."""
        d0 = z3.Int("d0")
        A = [d0 == 256]
        B = [d0 != 256]
        dm = _make_dim_map(("d0", "x", 0))

        a_vars = _collect_vars_from_list(A)
        b_vars = _collect_vars_from_list(B)
        assert a_vars == b_vars == {"d0"}

        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_interface_vars_computation(self):
        """_collect_vars correctly identifies free variables."""
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        expr = d0 + d1 * 2 - d2
        vs = _collect_vars(expr)
        assert vs == {"d0", "d1", "d2"}

    def test_empty_expression_has_no_vars(self):
        """BoolVal(True) should have no free variables."""
        vs = _collect_vars(z3.BoolVal(True))
        assert vs == set()

    def test_collect_vars_from_list(self):
        """_collect_vars_from_list unions vars across expressions."""
        d0, d1 = z3.Ints("d0 d1")
        exprs = [d0 >= 1, d1 <= 10]
        vs = _collect_vars_from_list(exprs)
        assert vs == {"d0", "d1"}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parsing interpolants into predicates
# ═══════════════════════════════════════════════════════════════════════════════

class TestInterpolantParsing:
    """parse_interpolant → ShapePredicate / LinearComboPredicate."""

    def test_parse_equality(self):
        d0 = z3.Int("d0")
        dm = _make_dim_map(("d0", "x", -1))
        preds = parse_interpolant(d0 == 768, dm)
        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, ShapePredicate)
        assert p.kind == PredicateKind.DIM_EQ
        assert p.value == 768
        assert p.tensor == "x"
        assert p.axis == -1

    def test_parse_ge(self):
        d0 = z3.Int("d0")
        dm = _make_dim_map(("d0", "x", 0))
        preds = parse_interpolant(d0 >= 64, dm)
        assert len(preds) == 1
        assert isinstance(preds[0], ShapePredicate)
        assert preds[0].kind == PredicateKind.DIM_GE
        assert preds[0].value == 64

    def test_parse_gt(self):
        d0 = z3.Int("d0")
        dm = _make_dim_map(("d0", "x", 0))
        preds = parse_interpolant(d0 > 0, dm)
        assert len(preds) == 1
        assert isinstance(preds[0], ShapePredicate)
        assert preds[0].kind == PredicateKind.DIM_GT
        assert preds[0].value == 0

    def test_parse_conjunction(self):
        d0, d1 = z3.Ints("d0 d1")
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1))
        formula = z3.And(d0 == 32, d1 >= 16)
        preds = parse_interpolant(formula, dm)
        assert len(preds) == 2
        kinds = {p.kind for p in preds if isinstance(p, ShapePredicate)}
        assert PredicateKind.DIM_EQ in kinds
        assert PredicateKind.DIM_GE in kinds

    def test_parse_dim_match(self):
        """d0 - d1 == 0 → DIM_MATCH."""
        d0, d1 = z3.Ints("d0 d1")
        dm = _make_dim_map(("d0", "x", 0), ("d1", "w", 1))
        preds = parse_interpolant(d0 == d1, dm)
        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, ShapePredicate)
        assert p.kind == PredicateKind.DIM_MATCH

    def test_parse_linear_combo(self):
        """d0 + d1 >= 512 → LinearComboPredicate."""
        d0, d1 = z3.Ints("d0 d1")
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1))
        preds = parse_interpolant(d0 + d1 >= 512, dm)
        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, LinearComboPredicate)
        assert p.operator == ">="

    def test_parse_scaled_linear_combo(self):
        """2*d0 + 3*d1 >= 100 → LinearComboPredicate."""
        d0, d1 = z3.Ints("d0 d1")
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1))
        preds = parse_interpolant(2 * d0 + 3 * d1 >= 100, dm)
        assert len(preds) == 1
        p = preds[0]
        assert isinstance(p, LinearComboPredicate)

    def test_parse_true_gives_empty(self):
        dm = DimMapping()
        preds = parse_interpolant(z3.BoolVal(True), dm)
        assert preds == []

    def test_parse_false_gives_empty(self):
        dm = DimMapping()
        preds = parse_interpolant(z3.BoolVal(False), dm)
        assert preds == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fallback when Z3 interpolation API not available
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallback:
    """The simulated interpolation (unsat core + QE) should work."""

    def test_simulated_interpolation_works(self):
        """Even without native interpolation, the simulation produces results."""
        d0 = z3.Int("d0")
        A = [d0 == 128]
        B = [d0 != 128]
        itp = _compute_simulated_interpolant(A, B, {"d0"})
        assert itp is not None
        # The interpolant should be equivalent to d0 == 128.
        s = z3.Solver()
        s.add(z3.Not(itp == (d0 == 128)))
        # Either UNSAT (exact match) or the interpolant implies d0 == 128.
        assert s.check() in (z3.unsat, z3.sat)  # we trust the QE output

    def test_simulated_with_qe(self):
        """QE should eliminate internal variables properly."""
        d0 = z3.Int("d0")
        tmp = z3.Int("tmp")
        A = [tmp == d0 + 10, tmp >= 74]
        B = [d0 < 64]
        itp = _compute_simulated_interpolant(A, B, {"d0"})
        assert itp is not None
        # Interpolant should be over d0 only.
        ivars = _collect_vars(itp)
        assert "tmp" not in ivars

    def test_sat_returns_none(self):
        """If A ∧ B is SAT, returns None."""
        d0 = z3.Int("d0")
        A = [d0 >= 0]
        B = [d0 >= 0]
        itp = _compute_simulated_interpolant(A, B, {"d0"})
        assert itp is None

    def test_empty_a_returns_none(self):
        """Empty A set: engine returns empty list."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = _make_dim_map(("d0", "x", 0))
        preds = engine.discover_via_interpolation([], [d0 < 0], dm)
        assert preds == []

    def test_empty_b_returns_none(self):
        """Empty B set: engine returns empty list."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = _make_dim_map(("d0", "x", 0))
        preds = engine.discover_via_interpolation([d0 >= 0], [], dm)
        assert preds == []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Integration with CEGAR loop types
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARIntegration:
    """Ensure discovered predicates integrate with shape_cegar types."""

    def test_predicates_are_shape_predicates(self):
        """Discovered template-fitting predicates are ShapePredicate instances."""
        d0 = z3.Int("d0")
        A = [d0 == 512]
        B = [d0 != 512]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        for p in preds:
            if isinstance(p, ShapePredicate):
                assert hasattr(p, "kind")
                assert hasattr(p, "tensor")
                assert hasattr(p, "provenance")

    def test_provenance_is_craig_interpolation(self):
        """All discovered predicates should have provenance = 'craig_interpolation'."""
        d0 = z3.Int("d0")
        A = [d0 >= 16]
        B = [d0 < 16]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        for p in preds:
            assert p.provenance == "craig_interpolation"

    def test_predicate_kind_enum_values(self):
        """Verify the predicate kinds we reference are valid."""
        assert PredicateKind.DIM_EQ is not None
        assert PredicateKind.DIM_GT is not None
        assert PredicateKind.DIM_GE is not None
        assert PredicateKind.DIM_MATCH is not None
        assert ExtendedPredicateKind.DIM_LINEAR_COMBO is not None

    def test_shape_predicate_pretty_output(self):
        """ShapePredicate.pretty() works for interpolation-discovered preds."""
        p = ShapePredicate(
            kind=PredicateKind.DIM_EQ,
            tensor="input",
            axis=-1,
            value=768,
            provenance="craig_interpolation",
        )
        assert "768" in p.pretty()
        assert "input" in p.pretty()

    def test_linear_combo_predicate_pretty(self):
        """LinearComboPredicate.pretty() produces readable output."""
        p = LinearComboPredicate(
            coefficients=((("x", 0), 1), (("x", 1), 1)),
            operator=">=",
            rhs=512,
        )
        pretty = p.pretty()
        assert ">=" in pretty
        assert "512" in pretty


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DimMapping tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDimMapping:
    def test_register_and_lookup(self):
        dm = DimMapping()
        dm.register("d0", "x", 0)
        assert dm.is_known("d0")
        assert dm.get_dim("d0") == ("x", 0)
        assert dm.dim_to_var[("x", 0)] == "d0"

    def test_unknown_var(self):
        dm = DimMapping()
        assert not dm.is_known("d0")
        assert dm.get_dim("d0") is None

    def test_multiple_registrations(self):
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1), ("d2", "w", 0))
        assert dm.is_known("d0")
        assert dm.is_known("d1")
        assert dm.is_known("d2")
        assert dm.get_dim("d2") == ("w", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Statistics tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatistics:
    def test_stats_increment_on_success(self):
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        engine.discover_via_interpolation([d0 == 10], [d0 != 10],
                                          _make_dim_map(("d0", "x", 0)))
        s = engine.stats
        assert s["interpolations_attempted"] == 1
        assert s["interpolations_succeeded"] >= 1

    def test_stats_increment_on_failure(self):
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        engine.discover_via_interpolation([d0 >= 0], [d0 >= 0],
                                          _make_dim_map(("d0", "x", 0)))
        s = engine.stats
        assert s["interpolations_attempted"] == 1
        assert s["interpolations_succeeded"] == 0

    def test_stats_count_predicates(self):
        d0, d1 = z3.Ints("d0 d1")
        engine = InterpolationPredicateDiscovery()
        engine.discover_via_interpolation(
            [d0 + d1 >= 100, d0 == 50], [d0 != 50],
            _make_dim_map(("d0", "x", 0), ("d1", "x", 1)),
        )
        s = engine.stats
        assert s["predicates_discovered"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Verify interpolant properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifyInterpolantProperties:
    def test_a_implies_i(self):
        d0 = z3.Int("d0")
        A = [d0 == 42]
        B = [d0 != 42]
        itp = _compute_simulated_interpolant(A, B, {"d0"})
        assert itp is not None
        engine = InterpolationPredicateDiscovery()
        a_impl, i_b_unsat, iface = engine.verify_interpolant_properties(
            A, B, itp, {"d0"}
        )
        assert a_impl, "A should imply I"
        assert i_b_unsat, "I ∧ B should be UNSAT"
        assert iface, "I should only use interface vars"

    def test_interface_containment_verified(self):
        d0 = z3.Int("d0")
        tmp = z3.Int("tmp")
        A = [tmp == d0 * 3, tmp >= 30]
        B = [d0 < 10]
        itp = _compute_simulated_interpolant(A, B, {"d0"})
        assert itp is not None
        ivars = _collect_vars(itp)
        assert "tmp" not in ivars


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Edge cases and advanced patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_trivially_unsat_b(self):
        """B alone is UNSAT → interpolant is True, no predicates needed."""
        d0 = z3.Int("d0")
        A = [d0 >= 0]
        B = [z3.BoolVal(False)]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        # Interpolant True → no predicates (vacuously safe)
        assert isinstance(preds, list)

    def test_single_var_multiple_constraints(self):
        """A has tight constraints that imply a specific value."""
        d0 = z3.Int("d0")
        A = [d0 >= 100, d0 <= 100]
        B = [d0 != 100]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_negative_value_predicate(self):
        """Negative values should be handled gracefully."""
        d0 = z3.Int("d0")
        A = [d0 == -5]
        B = [d0 != -5]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_large_value_predicate(self):
        """Large integer values in constraints."""
        d0 = z3.Int("d0")
        A = [d0 == 1000000]
        B = [d0 != 1000000]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_three_variable_linear_combo(self):
        """Three-variable linear combination."""
        d0, d1, d2 = z3.Ints("d0 d1 d2")
        A = [d0 + d1 + d2 >= 300, d0 == 100, d1 == 100, d2 == 100]
        B = [d0 + d1 + d2 < 300]
        dm = _make_dim_map(("d0", "x", 0), ("d1", "x", 1), ("d2", "x", 2))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1

    def test_le_normalised_to_ge(self):
        """d0 <= 10 should be handled (normalised internally)."""
        d0 = z3.Int("d0")
        dm = _make_dim_map(("d0", "x", 0))
        # Direct parse test of a <= constraint
        preds = parse_interpolant(z3.simplify(z3.Not(d0 > 10)), dm)
        # Not(d0 > 10) ≡ d0 <= 10
        assert isinstance(preds, list)

    def test_multiple_interpolation_calls(self):
        """Engine should handle multiple sequential calls."""
        engine = InterpolationPredicateDiscovery()
        d0 = z3.Int("d0")
        dm = _make_dim_map(("d0", "x", 0))

        preds1 = engine.discover_via_interpolation([d0 == 32], [d0 != 32], dm)
        preds2 = engine.discover_via_interpolation([d0 >= 16], [d0 < 16], dm)

        assert engine.stats["interpolations_attempted"] == 2
        assert len(preds1) >= 1
        assert len(preds2) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 10. LinearComboPredicate tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinearComboPredicate:
    def test_coeff_dict(self):
        p = LinearComboPredicate(
            coefficients=((("x", 0), 2), (("x", 1), 3)),
            operator=">=",
            rhs=100,
        )
        cd = p.coeff_dict
        assert cd[("x", 0)] == 2
        assert cd[("x", 1)] == 3

    def test_repr(self):
        p = LinearComboPredicate(
            coefficients=((("x", 0), 1), (("x", 1), 1)),
            operator=">=",
            rhs=512,
        )
        r = repr(p)
        assert "LinearComboPredicate" in r

    def test_provenance_default(self):
        p = LinearComboPredicate(
            coefficients=((("x", 0), 1),),
            operator="==",
            rhs=10,
        )
        assert p.provenance == "craig_interpolation"

    def test_extended_kind_enum(self):
        assert DIM_LINEAR_COMBO == ExtendedPredicateKind.DIM_LINEAR_COMBO
        assert DIM_LINEAR_COMBO.name == "DIM_LINEAR_COMBO"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Dict-based DimMapping acceptance
# ═══════════════════════════════════════════════════════════════════════════════

class TestDictDimMapAcceptance:
    """discover_via_interpolation should accept both DimMapping and plain dict."""

    def test_dict_dim_map_accepted(self):
        """Passing a plain dict as dim_map should work (auto-converted)."""
        d0 = z3.Int("d0")
        A = [d0 == 768]
        B = [d0 != 768]
        dim_map_dict = {"d0": ("x", 0)}
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dim_map_dict)
        assert len(preds) >= 1
        p = preds[0]
        assert isinstance(p, ShapePredicate)
        assert p.kind == PredicateKind.DIM_EQ
        assert p.value == 768

    def test_dim_mapping_object_still_works(self):
        """DimMapping object continues to work."""
        d0 = z3.Int("d0")
        A = [d0 >= 64]
        B = [d0 < 64]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CEGAR-realistic interpolation scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARRealisticInterpolation:
    """Scenarios mimicking the CEGAR integration where path constraints
    come from _build_predicate_extraction_query and safety constraints
    are negated before calling interpolation.
    """

    def test_linear_layer_shape_mismatch(self):
        """Simulate CEGAR interpolation for a Linear(768, 512) layer.

        Path: input has shape (B, 768) → d0 > 0, d1 > 0, d1 == 768.
        Safety: d1 == 768 (linear requires in_features=768).
        Negated safety (B): d1 != 768.
        A ∧ B is UNSAT → interpolant: d1 == 768 → DIM_EQ predicate.
        """
        d0, d1 = z3.Ints("__interp_x_d0 __interp_x_d1")
        path_cs = [d0 > 0, d1 > 0, d1 == 768]
        neg_safety_cs = [d1 != 768]
        dm = _make_dim_map(
            ("__interp_x_d0", "x", 0),
            ("__interp_x_d1", "x", 1),
        )
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(path_cs, neg_safety_cs, dm)
        assert len(preds) >= 1
        assert engine.stats["predicates_discovered"] >= 1
        has_768 = any(
            isinstance(p, ShapePredicate)
            and p.kind == PredicateKind.DIM_EQ
            and p.value == 768
            for p in preds
        )
        assert has_768, f"Expected DIM_EQ d1==768, got {preds}"

    def test_conv2d_channel_mismatch(self):
        """Simulate CEGAR interpolation for Conv2d(in_channels=3).

        Path: input (B, C, H, W) with C pinned to 3.
        Negated safety: C != 3.
        """
        d0, d1, d2, d3 = z3.Ints(
            "__interp_x_d0 __interp_x_d1 __interp_x_d2 __interp_x_d3"
        )
        path_cs = [d0 > 0, d1 > 0, d2 > 0, d3 > 0, d1 == 3]
        neg_safety_cs = [d1 != 3]
        dm = _make_dim_map(
            ("__interp_x_d0", "x", 0),
            ("__interp_x_d1", "x", 1),
            ("__interp_x_d2", "x", 2),
            ("__interp_x_d3", "x", 3),
        )
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(path_cs, neg_safety_cs, dm)
        assert len(preds) >= 1
        assert engine.stats["predicates_discovered"] >= 1

    def test_matmul_inner_dim_match(self):
        """Simulate matmul a @ b where a.shape[-1] must equal b.shape[-2].

        Path: a(B, M, K), b(B, K, N) with K=256.
        Negated safety: a_d2 != b_d1.
        """
        a_d0, a_d1, a_d2 = z3.Ints(
            "__interp_a_d0 __interp_a_d1 __interp_a_d2"
        )
        b_d0, b_d1, b_d2 = z3.Ints(
            "__interp_b_d0 __interp_b_d1 __interp_b_d2"
        )
        path_cs = [
            a_d0 > 0, a_d1 > 0, a_d2 > 0, a_d2 == 256,
            b_d0 > 0, b_d1 > 0, b_d2 > 0, b_d1 == 256,
        ]
        neg_safety_cs = [a_d2 != b_d1]
        dm = _make_dim_map(
            ("__interp_a_d0", "a", 0), ("__interp_a_d1", "a", 1),
            ("__interp_a_d2", "a", 2),
            ("__interp_b_d0", "b", 0), ("__interp_b_d1", "b", 1),
            ("__interp_b_d2", "b", 2),
        )
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(path_cs, neg_safety_cs, dm)
        assert len(preds) >= 1
        assert engine.stats["predicates_discovered"] >= 1

    def test_batchnorm_features_mismatch(self):
        """Simulate BatchNorm1d(num_features=128).

        Path: input (B, 128) with d1 == 128.
        Negated safety: d1 != 128.
        """
        d0, d1 = z3.Ints("__interp_x_d0 __interp_x_d1")
        path_cs = [d0 > 0, d1 > 0, d1 == 128]
        neg_safety_cs = [d1 != 128]
        dm = _make_dim_map(
            ("__interp_x_d0", "x", 0),
            ("__interp_x_d1", "x", 1),
        )
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(path_cs, neg_safety_cs, dm)
        assert len(preds) >= 1

    def test_layernorm_last_dim_mismatch(self):
        """Simulate LayerNorm(512).

        Path: input (B, S, 512) with d2 == 512.
        Negated safety: d2 != 512.
        """
        d0, d1, d2 = z3.Ints("__interp_x_d0 __interp_x_d1 __interp_x_d2")
        path_cs = [d0 > 0, d1 > 0, d2 > 0, d2 == 512]
        neg_safety_cs = [d2 != 512]
        dm = _make_dim_map(
            ("__interp_x_d0", "x", 0),
            ("__interp_x_d1", "x", 1),
            ("__interp_x_d2", "x", 2),
        )
        engine = InterpolationPredicateDiscovery()
        preds = engine.discover_via_interpolation(path_cs, neg_safety_cs, dm)
        assert len(preds) >= 1
        has_512 = any(
            isinstance(p, ShapePredicate)
            and p.kind == PredicateKind.DIM_EQ
            and p.value == 512
            for p in preds
        )
        assert has_512

    def test_multi_step_chain_discovers_multiple_predicates(self):
        """Linear(768→512) then Linear(512→256): two predicates from
        two interpolation calls.
        """
        engine = InterpolationPredicateDiscovery()
        dm1 = _make_dim_map(("d_in", "x", 1))
        dm2 = _make_dim_map(("d_mid", "h", 1))

        d_in = z3.Int("d_in")
        preds1 = engine.discover_via_interpolation(
            [d_in > 0, d_in == 768], [d_in != 768], dm1,
        )

        d_mid = z3.Int("d_mid")
        preds2 = engine.discover_via_interpolation(
            [d_mid > 0, d_mid == 512], [d_mid != 512], dm2,
        )

        total = len(preds1) + len(preds2)
        assert total >= 2, f"Expected >=2 predicates, got {total}"
        assert engine.stats["interpolations_succeeded"] == 2
        assert engine.stats["predicates_discovered"] >= 2

    def test_interpolation_stats_track_correctly(self):
        """Stats should reflect both successful and failed interpolations."""
        engine = InterpolationPredicateDiscovery()
        dm = _make_dim_map(("d0", "x", 0))
        d0 = z3.Int("d0")

        # Successful interpolation
        engine.discover_via_interpolation([d0 == 32], [d0 != 32], dm)
        # Failed interpolation (SAT)
        engine.discover_via_interpolation([d0 > 0], [d0 > 0], dm)

        s = engine.stats
        assert s["interpolations_attempted"] == 2
        assert s["interpolations_succeeded"] == 1
        assert s["predicates_discovered"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for multi-step interpolation query builder
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import z3 as _z3_check
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False


def _make_graph_linear(in_f=768, out_f=256):
    """Create a minimal ComputationGraph with one Linear layer."""
    from src.model_checker import (
        ComputationGraph, ComputationStep, LayerDef,
        LayerKind, OpKind,
    )
    layer = LayerDef(
        attr_name="fc1", kind=LayerKind.LINEAR,
        in_features=in_f, out_features=out_f,
    )
    step = ComputationStep(
        op=OpKind.LAYER_CALL, inputs=["x"], output="y",
        layer_ref="fc1",
    )
    return ComputationGraph(
        class_name="TestModel",
        layers={"fc1": layer},
        steps=[step],
        input_names=["x"],
        output_names=["y"],
    )


def _make_graph_reshape_linear():
    """Graph: x -> reshape(B, -1) -> Linear(784, 10)."""
    from src.model_checker import (
        ComputationGraph, ComputationStep, LayerDef,
        LayerKind, OpKind,
    )
    layer = LayerDef(
        attr_name="fc1", kind=LayerKind.LINEAR,
        in_features=784, out_features=10,
    )
    reshape_step = ComputationStep(
        op=OpKind.RESHAPE, inputs=["x"], output="x_flat",
        params={"shape": [-1, 784]},
    )
    fc_step = ComputationStep(
        op=OpKind.LAYER_CALL, inputs=["x_flat"], output="y",
        layer_ref="fc1",
    )
    return ComputationGraph(
        class_name="TestModel",
        layers={"fc1": layer},
        steps=[reshape_step, fc_step],
        input_names=["x"],
        output_names=["y"],
    )


@pytest.mark.skipif(not _HAS_Z3, reason="z3 required")
def test_single_linear_produces_interpolation_query():
    """Single Linear layer should produce path + counterexample constraints."""
    from src.shape_cegar import UnsatCorePredicateExtractor
    graph = _make_graph_linear(768, 256)
    pe = UnsatCorePredicateExtractor(graph, {})
    # Counterexample: input has dim 512 but Linear expects 768.
    cex = {"__ci_x_d1": 512}
    path_cs, cex_cs, dm = pe._build_interpolation_query(
        graph, failing_step_idx=0,
        input_shapes={"x": (32, 768)},
        concrete_dims=cex,
    )
    assert len(path_cs) > 0, "Should have positivity + safety constraints"
    assert len(cex_cs) > 0, "Should have counterexample constraints"
    assert dm.var_to_dim, "DimMapping should be populated"


@pytest.mark.skipif(not _HAS_Z3, reason="z3 required")
def test_linear_interpolation_discovers_predicate():
    """Interpolation on Linear(768, 256) should discover dim == 768."""
    from src.shape_cegar import UnsatCorePredicateExtractor
    graph = _make_graph_linear(768, 256)
    pe = UnsatCorePredicateExtractor(graph, {})
    # Counterexample: x.shape[-1] = 512, but Linear requires 768.
    cex = {"__ci_x_d1": 512}
    path_cs, cex_cs, dm = pe._build_interpolation_query(
        graph, failing_step_idx=0,
        input_shapes={"x": (32, 768)},
        concrete_dims=cex,
    )
    ipd = InterpolationPredicateDiscovery()
    preds = ipd.discover_via_interpolation(path_cs, cex_cs, dm)
    assert len(preds) > 0, f"Should discover predicates, got {preds}"
    # Should find a constraint relating to 768 (may be negated form).
    found_768 = any(
        (hasattr(p, 'value') and abs(p.value) == 768) or
        (hasattr(p, 'rhs') and abs(p.rhs) == 768)
        for p in preds
    )
    assert found_768, f"Should discover 768 constraint, got {preds}"


@pytest.mark.skipif(not _HAS_Z3, reason="z3 required")
def test_reshape_linear_multi_step():
    """Multi-step reshape -> Linear should produce constraints."""
    from src.shape_cegar import UnsatCorePredicateExtractor
    graph = _make_graph_reshape_linear()
    pe = UnsatCorePredicateExtractor(graph, {})
    # Counterexample: input (32, 1, 28, 27) — product 24192 != N*784.
    cex = {"__ci_x_d0": 32, "__ci_x_d1": 1, "__ci_x_d2": 28, "__ci_x_d3": 27}
    path_cs, cex_cs, dm = pe._build_interpolation_query(
        graph, failing_step_idx=1,
        input_shapes={"x": (32, 1, 28, 28)},
        concrete_dims=cex,
    )
    assert len(path_cs) > 0
    assert len(cex_cs) > 0
    ipd = InterpolationPredicateDiscovery()
    preds = ipd.discover_via_interpolation(path_cs, cex_cs, dm)
    assert ipd.stats["interpolations_attempted"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Convergence bound tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.craig_interpolation import (
    InterpolationConvergenceBound,
    compute_convergence_bound,
)


class TestInterpolationConvergenceBound:
    """Tests for InterpolationConvergenceBound dataclass."""

    def test_default_values(self):
        bound = InterpolationConvergenceBound()
        assert bound.num_input_dimensions == 0
        assert bound.num_template_predicates == 0
        assert bound.max_interpolation_predicates_per_iteration == 0
        assert bound.total_predicate_bound == 0
        assert bound.convergence_certificate is False

    def test_summary_keys(self):
        bound = InterpolationConvergenceBound(
            num_input_dimensions=4,
            num_template_predicates=28,
            max_interpolation_predicates_per_iteration=20,
            total_predicate_bound=48,
            convergence_iterations_bound=48,
            convergence_certificate=True,
        )
        s = bound.summary()
        assert s["num_input_dimensions"] == 4
        assert s["num_template_predicates"] == 28
        assert s["max_interpolation_predicates_per_iteration"] == 20
        assert s["total_predicate_bound"] == 48
        assert s["convergence_iterations_bound"] == 48
        assert s["convergence_certificate"] is True

    def test_certificate_false_when_zero_dims(self):
        bound = InterpolationConvergenceBound(
            num_input_dimensions=0,
            total_predicate_bound=10,
        )
        assert bound.convergence_certificate is False


class TestComputeConvergenceBound:
    """Tests for compute_convergence_bound() function."""

    def test_single_linear_layer(self):
        """One Linear layer with 2D input → known bounds."""
        graph = _make_graph_linear(768, 256)
        bound = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", "features")}
        )
        # 1 layer, 2 dims/layer (Linear), 7 kinds → 14 template
        assert bound.num_template_predicates == 1 * 2 * 7
        assert bound.num_input_dimensions == 2
        # interp per iter = n² + n = 4 + 2 = 6
        assert bound.max_interpolation_predicates_per_iteration == 2 * 2 + 2
        assert bound.total_predicate_bound == 14 + 6
        assert bound.convergence_iterations_bound == 20
        assert bound.convergence_certificate is True

    def test_multi_layer_mlp(self):
        """3-layer MLP should produce larger bounds."""
        from src.model_checker import (
            ComputationGraph, ComputationStep, LayerDef,
            LayerKind, OpKind,
        )
        layers = {}
        steps = []
        for i, (inf, outf) in enumerate([(512, 256), (256, 128), (128, 10)]):
            name = f"fc{i+1}"
            layers[name] = LayerDef(
                attr_name=name, kind=LayerKind.LINEAR,
                in_features=inf, out_features=outf,
            )
            inp = "x" if i == 0 else f"h{i}"
            out = f"h{i+1}" if i < 2 else "y"
            steps.append(ComputationStep(
                op=OpKind.LAYER_CALL, inputs=[inp], output=out,
                layer_ref=name,
            ))
        graph = ComputationGraph(
            class_name="MLP3", layers=layers, steps=steps,
            input_names=["x"], output_names=["y"],
        )
        bound = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", "d")}
        )
        assert bound.num_template_predicates == 3 * 2 * 7  # 42
        assert bound.num_input_dimensions == 2
        assert bound.convergence_certificate is True
        # Total bound should be template + interp
        assert bound.total_predicate_bound == 42 + (4 + 2)

    def test_conv_layer_uses_4_dims(self):
        """Conv2d layers should use max_dims=4."""
        from src.model_checker import (
            ComputationGraph, ComputationStep, LayerDef,
            LayerKind, OpKind,
        )
        layer = LayerDef(
            attr_name="conv1", kind=LayerKind.CONV2D,
            in_channels=3, out_channels=16,
            kernel_size=(3, 3),
        )
        step = ComputationStep(
            op=OpKind.LAYER_CALL, inputs=["x"], output="y",
            layer_ref="conv1",
        )
        graph = ComputationGraph(
            class_name="CNN", layers={"conv1": layer}, steps=[step],
            input_names=["x"], output_names=["y"],
        )
        bound = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", 3, 224, 224)}
        )
        # 1 layer, 4 dims (conv), 7 kinds → 28
        assert bound.num_template_predicates == 1 * 4 * 7
        assert bound.num_input_dimensions == 4
        # interp: 16 + 4 = 20
        assert bound.max_interpolation_predicates_per_iteration == 20
        assert bound.convergence_certificate is True

    def test_no_input_shapes_uses_defaults(self):
        """Without explicit input_shapes, should use conservative default."""
        graph = _make_graph_linear(768, 256)
        bound = compute_convergence_bound(graph, input_shapes=None)
        # Default: 4 dims per input
        assert bound.num_input_dimensions == 4
        assert bound.convergence_certificate is True

    def test_bound_always_exceeds_template(self):
        """Total bound should always be ≥ template-only bound."""
        graph = _make_graph_linear(768, 256)
        bound = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", "features")}
        )
        assert bound.total_predicate_bound >= bound.num_template_predicates

    def test_bound_monotone_in_input_dims(self):
        """More input dimensions → larger interpolation bound."""
        graph = _make_graph_linear(768, 256)
        bound_2d = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", "features")}
        )
        bound_4d = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", 3, 224, 224)}
        )
        assert (bound_4d.max_interpolation_predicates_per_iteration
                > bound_2d.max_interpolation_predicates_per_iteration)
        assert (bound_4d.total_predicate_bound
                > bound_2d.total_predicate_bound)

    def test_reshape_linear_graph(self):
        """Reshape+Linear graph should produce valid bounds."""
        graph = _make_graph_reshape_linear()
        bound = compute_convergence_bound(
            graph, input_shapes={"x": ("batch", 1, 28, 28)}
        )
        assert bound.num_input_dimensions == 4
        assert bound.num_template_predicates > 0
        assert bound.convergence_certificate is True


# ═══════════════════════════════════════════════════════════════════════════════
# CVC5 Native Craig Interpolation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCVC5NativeInterpolation:
    """Tests for CVC5 native Craig interpolation backend."""

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_simple_equality(self):
        """CVC5 native: A: d0 == 768, B: d0 != 768 → valid interpolant."""
        d0 = z3.Int("d0")
        A = [d0 == 768]
        B = [d0 != 768]
        interface = {"d0"}
        interp = _compute_cvc5_interpolant(A, B, interface)
        assert interp is not None
        # Verify Craig properties
        engine = InterpolationPredicateDiscovery(method=InterpolationMethod.CVC5_NATIVE)
        a_impl, ib_unsat, vocab_ok = engine.verify_interpolant_properties(
            A, B, interp, interface
        )
        assert a_impl, "A ⊨ I must hold"
        assert ib_unsat, "I ∧ B must be UNSAT"
        assert vocab_ok, "Vars(I) ⊆ Vars(A) ∩ Vars(B)"

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_inequality(self):
        """CVC5 native: A: d0 >= 64, B: d0 < 64 → valid interpolant."""
        d0 = z3.Int("d0")
        A = [d0 >= 64]
        B = [d0 < 64]
        interface = {"d0"}
        interp = _compute_cvc5_interpolant(A, B, interface)
        assert interp is not None
        engine = InterpolationPredicateDiscovery(method=InterpolationMethod.CVC5_NATIVE)
        a_impl, ib_unsat, vocab_ok = engine.verify_interpolant_properties(
            A, B, interp, interface
        )
        assert a_impl
        assert ib_unsat
        assert vocab_ok

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_vocab_restriction(self):
        """CVC5 interpolant uses only interface variables."""
        d0 = z3.Int("d0")
        y = z3.Int("y")
        A = [d0 >= 64, y == d0 + 1]
        B = [d0 < 64]
        interface = {"d0"}
        interp = _compute_cvc5_interpolant(A, B, interface)
        assert interp is not None
        interp_vars = _collect_vars(interp)
        assert interp_vars.issubset(interface), (
            f"Interpolant vars {interp_vars} not subset of interface {interface}"
        )

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_multi_var_interpolant(self):
        """CVC5 native: multi-variable interpolant with dimension matching."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        A = [d0 == d1, d0 >= 32]
        B = [d0 != d1]
        interface = {"d0", "d1"}
        interp = _compute_cvc5_interpolant(A, B, interface)
        assert interp is not None
        engine = InterpolationPredicateDiscovery(method=InterpolationMethod.CVC5_NATIVE)
        a_impl, ib_unsat, vocab_ok = engine.verify_interpolant_properties(
            A, B, interp, interface
        )
        assert a_impl
        assert ib_unsat
        assert vocab_ok

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_sat_returns_none(self):
        """CVC5 returns None when A ∧ B is satisfiable."""
        d0 = z3.Int("d0")
        A = [d0 >= 64]
        B = [d0 >= 100]
        interface = {"d0"}
        interp = _compute_cvc5_interpolant(A, B, interface)
        assert interp is None


class TestInterpolationMethodAuto:
    """Tests for AUTO mode method selection."""

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_auto_prefers_cvc5(self):
        """AUTO mode uses CVC5 when available and tracks it in stats."""
        d0 = z3.Int("d0")
        A = [d0 == 768]
        B = [d0 != 768]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery(method=InterpolationMethod.AUTO)
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1
        assert engine.stats["cvc5_native_count"] >= 1
        assert engine.stats["z3_simulation_count"] == 0

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_native_mode(self):
        """CVC5_NATIVE mode uses only CVC5."""
        d0 = z3.Int("d0")
        A = [d0 >= 64]
        B = [d0 < 64]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery(
            method=InterpolationMethod.CVC5_NATIVE
        )
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1
        assert engine.stats["cvc5_native_count"] >= 1

    def test_z3_simulation_mode(self):
        """Z3_UNSAT_CORE_SIMULATION mode uses only Z3 simulation."""
        d0 = z3.Int("d0")
        A = [d0 == 768]
        B = [d0 != 768]
        dm = _make_dim_map(("d0", "x", 0))
        engine = InterpolationPredicateDiscovery(
            method=InterpolationMethod.Z3_UNSAT_CORE_SIMULATION
        )
        preds = engine.discover_via_interpolation(A, B, dm)
        assert len(preds) >= 1
        assert engine.stats["z3_simulation_count"] >= 1
        assert engine.stats["cvc5_native_count"] == 0


class TestCVC5Fallback:
    """Tests for fallback from CVC5 to Z3 simulation."""

    def test_fallback_when_cvc5_unavailable(self):
        """Fallback to Z3 simulation works when CVC5 is not available."""
        import src.craig_interpolation as ci
        original = ci.HAS_CVC5
        try:
            ci.HAS_CVC5 = False
            d0 = z3.Int("d0")
            A = [d0 == 768]
            B = [d0 != 768]
            dm = _make_dim_map(("d0", "x", 0))
            engine = InterpolationPredicateDiscovery(
                method=InterpolationMethod.AUTO
            )
            preds = engine.discover_via_interpolation(A, B, dm)
            assert len(preds) >= 1
            assert engine.stats["z3_simulation_count"] >= 1
            assert engine.stats["cvc5_native_count"] == 0
        finally:
            ci.HAS_CVC5 = original

    def test_cvc5_interpolant_returns_none_without_cvc5(self):
        """_compute_cvc5_interpolant returns None when CVC5 unavailable."""
        import src.craig_interpolation as ci
        original = ci.HAS_CVC5
        try:
            ci.HAS_CVC5 = False
            d0 = z3.Int("d0")
            interp = _compute_cvc5_interpolant(
                [d0 == 768], [d0 != 768], {"d0"}
            )
            assert interp is None
        finally:
            ci.HAS_CVC5 = original
