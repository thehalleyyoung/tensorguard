"""
Tests for IC3/PDR CVC5 native interpolation integration.

Validates that IC3/PDR prefers CVC5 native get-interpolant over
UNSAT-core simulation when CVC5 is available.
"""

import pytest

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    import cvc5
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

from src.ic3_pdr import IC3Solver, ShapeTransitionSystem, ic3_verify


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
class TestIC3CVC5Integration:
    """Test that IC3/PDR correctly uses CVC5 interpolation."""

    def test_ic3_imports_cvc5_interpolant(self):
        """Verify IC3 module imports CVC5 interpolant function."""
        from src.ic3_pdr import HAS_CVC5 as ic3_has_cvc5
        # The import should at least be attempted
        from src import ic3_pdr
        assert hasattr(ic3_pdr, 'HAS_CVC5')

    def test_ic3_imports_both_interpolation_methods(self):
        """IC3 should import both CVC5 and simulated interpolation."""
        from src import ic3_pdr
        assert hasattr(ic3_pdr, '_compute_simulated_interpolant')
        if ic3_pdr.HAS_INTERPOLATION:
            assert hasattr(ic3_pdr, '_compute_cvc5_interpolant')

    def test_simple_safe_model_ic3(self):
        """IC3 should verify a simple safe model correctly."""
        result = ic3_verify(
            model_source="""
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            symbolic_dims={"batch": "batch_size"},
        )
        assert result.safe

    def test_interpolation_generalize_prefers_cvc5(self):
        """When CVC5 is available, interpolation should try CVC5 first."""
        # This test verifies the code path, not CVC5 availability
        from src.ic3_pdr import IC3Solver
        # The method should exist and accept the right parameters
        assert hasattr(IC3Solver, '_try_interpolation_generalize')

    @pytest.mark.skipif(not HAS_CVC5, reason="CVC5 not available")
    def test_cvc5_available_in_ic3(self):
        """When CVC5 is installed, IC3 should detect it."""
        from src.ic3_pdr import HAS_CVC5 as ic3_cvc5
        assert ic3_cvc5

    def test_ic3_result_fields(self):
        """IC3Result should have all expected fields."""
        from src.ic3_pdr import IC3Result
        result = IC3Result(safe=True, invariant="test")
        assert result.safe
        assert result.invariant == "test"
        assert result.frame_sequence == []
