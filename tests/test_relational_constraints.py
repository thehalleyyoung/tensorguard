"""Tests for relational shape constraints (e.g. heads * head_dim == embed_dim)."""

import pytest
from src.model_checker import (
    verify_model,
    ConstraintVerifier,
    extract_computation_graph,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ─── Model sources ───────────────────────────────────────────────────────────

CORRECT_MHA = """\
import torch.nn as nn

class MultiHeadAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
"""

BUGGY_MHA = """\
import torch.nn as nn

class BuggyMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        return self.out_proj(q)
"""

SIMPLE_LINEAR_MODEL = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_Z3, reason="Z3 not installed")
class TestRelationalConstraints:
    """Test relational constraints between symbolic dimensions."""

    def test_correct_mha_with_constraints_is_safe(self):
        """A correct MHA model with heads * head_dim == embed_dim should be SAFE."""
        result = verify_model(
            CORRECT_MHA,
            input_shapes={"x": ("batch", "seq_len", "embed_dim")},
            constraints={
                "embed_dim": "heads * head_dim",
                "heads": 8,
                "head_dim": 64,
            },
        )
        assert result.safe, f"Expected safe, got errors: {result.errors}"

    def test_buggy_mha_detected(self):
        """A buggy MHA with mismatched projection should be detected as unsafe."""
        result = verify_model(
            BUGGY_MHA,
            input_shapes={"x": ("batch", "seq_len", "embed_dim")},
            constraints={"embed_dim": "heads * head_dim", "heads": 8},
        )
        # The out_proj expects 512-dim input but q_proj outputs 256 → shape mismatch
        assert not result.safe

    def test_constraints_without_model(self):
        """Constraints are correctly parsed and added even for a simple model."""
        result = verify_model(
            SIMPLE_LINEAR_MODEL,
            input_shapes={"x": ("batch", 10)},
            constraints={"batch": 32},
        )
        assert result.safe

    def test_constraint_parser_multiplication(self):
        """Test that multiplicative constraints are parsed correctly."""
        graph = extract_computation_graph(CORRECT_MHA)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", "seq_len", "embed_dim")},
            constraints={"embed_dim": "heads * head_dim", "heads": 8},
        )
        # The constraint should create Z3 variables for heads, head_dim, embed_dim
        assert "heads" in checker.ctx._sym_dims
        assert "head_dim" in checker.ctx._sym_dims
        assert "embed_dim" in checker.ctx._sym_dims

    def test_constraint_parser_addition(self):
        """Test that additive constraints are parsed correctly."""
        graph = extract_computation_graph(SIMPLE_LINEAR_MODEL)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", 10)},
            constraints={"total": "a + b"},
        )
        # Verify the Z3 variables were created
        assert "total" in checker.ctx._sym_dims
        assert "a" in checker.ctx._sym_dims
        assert "b" in checker.ctx._sym_dims

    def test_constraint_parser_division(self):
        """Test that division constraints are parsed correctly."""
        graph = extract_computation_graph(SIMPLE_LINEAR_MODEL)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", 10)},
            constraints={"head_dim": "embed_dim / heads"},
        )
        assert "head_dim" in checker.ctx._sym_dims
        assert "embed_dim" in checker.ctx._sym_dims
        assert "heads" in checker.ctx._sym_dims

    def test_constraint_parser_subtraction(self):
        """Test that subtraction constraints are parsed correctly."""
        graph = extract_computation_graph(SIMPLE_LINEAR_MODEL)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", 10)},
            constraints={"remainder": "total - used"},
        )
        assert "remainder" in checker.ctx._sym_dims
        assert "total" in checker.ctx._sym_dims
        assert "used" in checker.ctx._sym_dims

    def test_integer_constraint(self):
        """Test that integer constraints fix a dimension to a concrete value."""
        graph = extract_computation_graph(SIMPLE_LINEAR_MODEL)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", 10)},
            constraints={"batch": 32},
        )
        assert "batch" in checker.ctx._sym_dims

    def test_verify_model_passes_constraints_through(self):
        """verify_model correctly forwards constraints to ConstraintVerifier."""
        result = verify_model(
            CORRECT_MHA,
            input_shapes={"x": ("batch", "seq_len", "embed_dim")},
            constraints={"embed_dim": "heads * head_dim", "heads": 8, "head_dim": 64},
        )
        # Should still be safe (512 == 8 * 64)
        assert result.safe

    def test_complex_expression(self):
        """Test a more complex arithmetic expression."""
        graph = extract_computation_graph(SIMPLE_LINEAR_MODEL)
        checker = ConstraintVerifier(
            graph,
            input_shapes={"x": ("batch", 10)},
            constraints={"total": "a * b + c"},
        )
        assert "total" in checker.ctx._sym_dims
        assert "a" in checker.ctx._sym_dims
        assert "b" in checker.ctx._sym_dims
        assert "c" in checker.ctx._sym_dims
