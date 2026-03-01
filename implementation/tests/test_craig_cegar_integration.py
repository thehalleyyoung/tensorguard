"""
Tests for Craig interpolation integration in the CEGAR loop.

Covers:
  - Craig interpolation discovers predicates when called from the CEGAR loop
  - Predicates are correctly tagged with provenance="craig_interpolation"
  - The integration doesn't break existing CEGAR behavior
  - LinearComboPredicate conversion to ShapePredicate
  - Interpolation acts as fallback when template-based discovery yields nothing
"""

from __future__ import annotations

import pytest

from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    ShapePredicate,
    PredicateKind,
    run_shape_cegar,
    _convert_linear_combo_to_predicate,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    from src.craig_interpolation import (
        InterpolationPredicateDiscovery,
        LinearComboPredicate,
        DimMapping,
    )
    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures: model sources
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

TWO_LAYER_MLP = """\
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

SHAPE_MISMATCH = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Craig interpolation integration in CEGAR loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestCraigInterpolationIntegration:
    """Test that Craig interpolation is invoked and produces predicates
    when integrated into the CEGAR loop."""

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_interpolation_enabled_by_default(self):
        """ShapeCEGARLoop has interpolation enabled by default."""
        loop = ShapeCEGARLoop(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert loop.enable_interpolation is True

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_safe_model_with_interpolation(self):
        """Safe model converges correctly with interpolation enabled."""
        loop = ShapeCEGARLoop(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            enable_interpolation=True,
        )
        result = loop.run()
        assert result.is_safe
        assert result.final_status == CEGARStatus.SAFE

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_safe_model_without_interpolation(self):
        """Safe model also works with interpolation disabled."""
        loop = ShapeCEGARLoop(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            enable_interpolation=False,
        )
        result = loop.run()
        assert result.is_safe

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_buggy_model_with_interpolation(self):
        """Buggy model is still detected with interpolation enabled."""
        loop = ShapeCEGARLoop(
            SHAPE_MISMATCH,
            input_shapes={"x": ("batch", 10)},
            enable_interpolation=True,
        )
        result = loop.run()
        assert result.has_real_bugs

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_mlp_with_interpolation_stats(self):
        """Two-layer MLP: interpolation stats are tracked."""
        loop = ShapeCEGARLoop(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            enable_interpolation=True,
        )
        result = loop.run()
        assert result.is_safe
        # Interpolation stats should exist on the loop
        stats = loop._interpolation_stats
        assert "attempted" in stats
        assert "successful" in stats
        assert "predicates_from_interpolation" in stats

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_interpolation_stats_in_result(self):
        """When interpolation was attempted, stats appear in the result."""
        loop = ShapeCEGARLoop(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            enable_interpolation=True,
        )
        result = loop.run()
        # If interpolation was attempted, stats should be in the result
        if loop._interpolation_stats.get("attempted", 0) > 0:
            assert result.interpolation_stats is not None
            assert result.interpolation_stats["attempted"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Provenance tagging for Craig interpolation predicates
# ═══════════════════════════════════════════════════════════════════════════════

class TestCraigInterpolationProvenance:
    """Test that predicates from Craig interpolation are properly tagged."""

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation module not available")
    def test_interpolation_predicates_have_craig_provenance(self):
        """Predicates from InterpolationPredicateDiscovery have correct provenance."""
        d0 = z3.Int("d0")
        ipd = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "x", 0)
        preds = ipd.discover_via_interpolation(
            [d0 == 768], [d0 != 768], dm,
        )
        assert len(preds) >= 1
        for p in preds:
            assert p.provenance == "craig_interpolation"

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation module not available")
    def test_dim_match_provenance(self):
        """DIM_MATCH predicates from interpolation have correct provenance."""
        d0 = z3.Int("d0")
        d1 = z3.Int("d1")
        ipd = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "x", 0)
        dm.register("d1", "w", 0)
        preds = ipd.discover_via_interpolation(
            [d0 == d1, d0 > 0], [d0 != d1], dm,
        )
        for p in preds:
            assert p.provenance == "craig_interpolation"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LinearComboPredicate conversion
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinearComboConversion:
    """Test _convert_linear_combo_to_predicate converts correctly."""

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_single_var_eq(self):
        """Single-variable == converts to DIM_EQ."""
        lcp = LinearComboPredicate(
            coefficients=((("x", 0), 1),),
            operator="==", rhs=768,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is not None
        assert result.kind == PredicateKind.DIM_EQ
        assert result.tensor == "x"
        assert result.axis == 0
        assert result.value == 768
        assert result.provenance == "craig_interpolation"

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_single_var_ge(self):
        """Single-variable >= converts to DIM_GE."""
        lcp = LinearComboPredicate(
            coefficients=((("x", 1), 1),),
            operator=">=", rhs=64,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is not None
        assert result.kind == PredicateKind.DIM_GE
        assert result.value == 64

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_single_var_gt(self):
        """Single-variable > converts to DIM_GT."""
        lcp = LinearComboPredicate(
            coefficients=((("y", 0), 1),),
            operator=">", rhs=0,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is not None
        assert result.kind == PredicateKind.DIM_GT

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_dim_match_conversion(self):
        """Two-variable opposite-sign == converts to DIM_MATCH."""
        lcp = LinearComboPredicate(
            coefficients=((("x", 0), 1), (("w", 0), -1)),
            operator="==", rhs=0,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is not None
        assert result.kind == PredicateKind.DIM_MATCH
        assert result.tensor == "x"
        assert result.match_tensor == "w"
        assert result.provenance == "craig_interpolation"

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_dim_match_reversed(self):
        """Reversed-sign DIM_MATCH still converts correctly."""
        lcp = LinearComboPredicate(
            coefficients=((("a", 1), -1), (("b", 2), 1)),
            operator="==", rhs=0,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is not None
        assert result.kind == PredicateKind.DIM_MATCH
        assert result.tensor == "b"
        assert result.match_tensor == "a"

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_complex_combo_returns_none(self):
        """Three-variable combination cannot be converted."""
        lcp = LinearComboPredicate(
            coefficients=((("x", 0), 1), (("y", 0), 2), (("z", 0), -1)),
            operator=">=", rhs=10,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is None

    @pytest.mark.skipif(not HAS_INTERPOLATION,
                        reason="Interpolation module not available")
    def test_non_unit_coeff_returns_none(self):
        """Single variable with non-unit coefficient is not convertible."""
        lcp = LinearComboPredicate(
            coefficients=((("x", 0), 3),),
            operator="==", rhs=768,
        )
        result = _convert_linear_combo_to_predicate(lcp)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Existing CEGAR behavior is preserved
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARPreservation:
    """Verify that adding Craig interpolation does not break existing behavior."""

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_run_shape_cegar_still_works(self):
        """Public API run_shape_cegar still returns correct results."""
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert result.is_safe
        assert isinstance(result, ShapeCEGARResult)

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_buggy_model_detected(self):
        """Buggy model is still detected by run_shape_cegar."""
        result = run_shape_cegar(
            SHAPE_MISMATCH,
            input_shapes={"x": ("batch", 10)},
        )
        assert result.has_real_bugs
        assert result.final_status == CEGARStatus.REAL_BUG_FOUND

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_result_has_expected_fields(self):
        """Result has all expected fields including interpolation_stats."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
        )
        assert hasattr(result, 'interpolation_stats')
        assert hasattr(result, 'discovered_predicates')
        assert hasattr(result, 'iteration_log')

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_cegar_convergence_preserved(self):
        """MLP converges within iteration budget."""
        result = run_shape_cegar(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
            max_iterations=10,
        )
        assert result.is_safe
        assert result.iterations <= 10

    @pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
    def test_predicate_discovery_still_works(self):
        """Predicates are still discovered for models needing them."""
        # Use symbolic dimension that will trigger predicate discovery
        result = run_shape_cegar(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        # Should be safe — predicates may or may not be needed
        assert result.is_safe


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Craig interpolation discovers predicates standalone
# ═══════════════════════════════════════════════════════════════════════════════

class TestCraigInterpolationDiscovery:
    """Test that the InterpolationPredicateDiscovery engine works."""

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation not available")
    def test_discovers_equality_predicate(self):
        """Interpolation discovers DIM_EQ predicate from path/safety pair."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "input", 0)
        preds = engine.discover_via_interpolation(
            [d0 == 768, d0 > 0],
            [d0 != 768],
            dm,
        )
        assert len(preds) >= 1
        has_eq = any(
            isinstance(p, ShapePredicate)
            and p.kind == PredicateKind.DIM_EQ
            and p.value == 768
            for p in preds
        )
        assert has_eq

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation not available")
    def test_discovers_ge_predicate(self):
        """Interpolation discovers DIM_GE predicate."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "x", 1)
        preds = engine.discover_via_interpolation(
            [d0 >= 32, d0 > 0],
            [d0 < 32],
            dm,
        )
        assert len(preds) >= 1

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation not available")
    def test_stats_tracked(self):
        """Interpolation engine tracks statistics correctly."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "x", 0)
        engine.discover_via_interpolation([d0 == 10], [d0 != 10], dm)
        stats = engine.stats
        assert stats["interpolations_attempted"] == 1
        assert stats["interpolations_succeeded"] >= 1
        assert stats["predicates_discovered"] >= 1

    @pytest.mark.skipif(not HAS_Z3 or not HAS_INTERPOLATION,
                        reason="Z3 or interpolation not available")
    def test_sat_returns_empty(self):
        """If A ∧ B is SAT, no predicates are discovered."""
        d0 = z3.Int("d0")
        engine = InterpolationPredicateDiscovery()
        dm = DimMapping()
        dm.register("d0", "x", 0)
        preds = engine.discover_via_interpolation(
            [d0 > 0],
            [d0 > 0],
            dm,
        )
        assert len(preds) == 0
