"""
Tests for IC3/PDR unbounded tensor shape verification.

Tests cover:
  - Basic IC3/PDR on simple models (Linear chain, Conv+Linear)
  - Parametric verification (verify for all batch sizes)
  - Counterexample generation when model is unsafe
  - Fixed-point detection (inductive invariant found)
  - Integration with verify_model(verification_mode="unbounded")
  - Edge cases and error handling
"""

import pytest

from src.ic3_pdr import (
    IC3Result,
    IC3Solver,
    IC3Status,
    ProofObligation,
    ShapeClause,
    ShapeCube,
    ShapeTransitionSystem,
    ic3_verify,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
    VerificationResult,
    extract_computation_graph,
    verify_model,
)

# ---------------------------------------------------------------------------
# Model source snippets
# ---------------------------------------------------------------------------

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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""

THREE_LAYER_MLP = """\
import torch.nn as nn

class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""

DEEP_LINEAR_CHAIN = """\
import torch.nn as nn

class DeepLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
"""

UNSAFE_DIMENSION_MISMATCH = """\
import torch.nn as nn

class UnsafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

CONV_LINEAR_MODEL = """\
import torch.nn as nn

class ConvLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
"""

SIMPLE_CONV = """\
import torch.nn as nn

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""

WITH_RELU = """\
import torch.nn as nn

class ReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""

WITH_DROPOUT = """\
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
"""

BATCHNORM_MODEL = """\
import torch.nn as nn

class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
"""

IDENTITY_MODEL = """\
import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)
"""

WIDE_MLP = """\
import torch.nn as nn

class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Test IC3Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestIC3Result:
    def test_safe_result(self):
        result = IC3Result(safe=True, frames_computed=5, invariant="True")
        assert result.safe is True
        assert result.frames_computed == 5
        assert result.invariant == "True"
        assert result.counterexample_depth is None

    def test_unsafe_result(self):
        result = IC3Result(safe=False, counterexample_depth=3, frames_computed=4)
        assert result.safe is False
        assert result.counterexample_depth == 3
        assert result.invariant is None

    def test_default_values(self):
        result = IC3Result(safe=True)
        assert result.frames_computed == 0
        assert result.verification_time_ms == 0.0
        assert result.symbolic_dims == {}
        assert result.num_blocked_cubes == 0
        assert result.z3_queries == 0
        assert result.invariant_clauses == []
        assert result.counterexample_trace is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test ShapeCube / ShapeClause
# ═══════════════════════════════════════════════════════════════════════════════


class TestCubeClause:
    def test_cube_creation(self):
        cube = ShapeCube(literals=frozenset({"x == 10", "y == 20"}))
        assert "x == 10" in cube.literals
        assert "y == 20" in cube.literals

    def test_cube_repr(self):
        cube = ShapeCube(literals=frozenset({"a == 1"}))
        assert "ShapeCube" in repr(cube)

    def test_cube_negate(self):
        cube = ShapeCube(literals=frozenset({"x == 10"}))
        clause = cube.negate()
        assert isinstance(clause, ShapeClause)

    def test_clause_creation(self):
        clause = ShapeClause(literals=frozenset({"x != 10"}))
        assert "x != 10" in clause.literals

    def test_clause_repr(self):
        clause = ShapeClause(literals=frozenset({"b != 2"}))
        assert "ShapeClause" in repr(clause)


# ═══════════════════════════════════════════════════════════════════════════════
# Test ProofObligation
# ═══════════════════════════════════════════════════════════════════════════════


class TestProofObligation:
    def test_creation(self):
        cube = ShapeCube(literals=frozenset({"x == 1"}))
        ob = ProofObligation(cube=cube, frame_level=2, depth=1)
        assert ob.frame_level == 2
        assert ob.depth == 1

    def test_ordering(self):
        c1 = ShapeCube(literals=frozenset({"a == 1"}))
        c2 = ShapeCube(literals=frozenset({"b == 2"}))
        ob1 = ProofObligation(cube=c1, frame_level=1, depth=0)
        ob2 = ProofObligation(cube=c2, frame_level=2, depth=0)
        assert ob1 < ob2


# ═══════════════════════════════════════════════════════════════════════════════
# Test ShapeTransitionSystem
# ═══════════════════════════════════════════════════════════════════════════════


class TestShapeTransitionSystem:
    def test_build_simple_linear(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {"batch": "batch_size"}
        )
        assert ts.num_steps() > 0
        assert len(ts.get_init_constraints()) > 0

    def test_build_two_layer(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 784)}, {"batch": "batch_size"}
        )
        assert ts.num_steps() > 0

    def test_dim_vars(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {"batch": "batch_size"}
        )
        dim_vars = ts.get_dim_vars()
        assert len(dim_vars) > 0

    def test_safety_constraints_exist(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 784)}, {}
        )
        all_safety = ts.get_all_safety_constraints()
        assert len(all_safety) > 0

    def test_transition_constraints_exist(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 784)}, {}
        )
        all_trans = ts.get_all_transition_constraints()
        assert len(all_trans) > 0

    def test_conv_model(self):
        graph = extract_computation_graph(SIMPLE_CONV)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 3, 32, 32)}, {"batch": "batch_size"}
        )
        assert ts.num_steps() > 0

    def test_bad_constraints(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {}
        )
        bad = ts.get_bad_constraints()
        assert len(bad) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test IC3Solver directly
# ═══════════════════════════════════════════════════════════════════════════════


class TestIC3Solver:
    def test_safe_simple_linear(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {"batch": "batch_size"}
        )
        solver = IC3Solver(ts)
        status = solver.solve()
        assert status == IC3Status.SAFE

    def test_safe_two_layer(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 784)}, {"batch": "batch_size"}
        )
        solver = IC3Solver(ts)
        status = solver.solve()
        assert status == IC3Status.SAFE

    def test_unsafe_mismatch(self):
        graph = extract_computation_graph(UNSAFE_DIMENSION_MISMATCH)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {"batch": "batch_size"}
        )
        solver = IC3Solver(ts)
        status = solver.solve()
        assert status == IC3Status.UNSAFE

    def test_frames_computed(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {}
        )
        solver = IC3Solver(ts)
        solver.solve()
        assert solver.frames_computed >= 1

    def test_z3_queries_tracked(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {}
        )
        solver = IC3Solver(ts)
        solver.solve()
        assert solver.z3_queries >= 1

    def test_invariant_on_safe(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {}
        )
        solver = IC3Solver(ts)
        status = solver.solve()
        if status == IC3Status.SAFE:
            inv = solver.get_invariant_str()
            assert inv is not None

    def test_no_invariant_on_unsafe(self):
        graph = extract_computation_graph(UNSAFE_DIMENSION_MISMATCH)
        ts = ShapeTransitionSystem(
            graph, {"x": ("batch", 10)}, {}
        )
        solver = IC3Solver(ts)
        solver.solve()
        inv = solver.get_invariant_str()
        # If unsafe, invariant should be None
        assert solver._status == IC3Status.UNSAFE
        assert inv is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test ic3_verify API
# ═══════════════════════════════════════════════════════════════════════════════


class TestIC3Verify:
    def test_safe_simple(self):
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert isinstance(result, IC3Result)
        assert result.safe is True

    def test_safe_two_layer(self):
        result = ic3_verify(TWO_LAYER_MLP, input_shapes={"x": ("batch", 784)})
        assert result.safe is True

    def test_safe_three_layer(self):
        result = ic3_verify(THREE_LAYER_MLP, input_shapes={"x": ("batch", 100)})
        assert result.safe is True

    def test_safe_deep_chain(self):
        result = ic3_verify(DEEP_LINEAR_CHAIN, input_shapes={"x": ("batch", 128)})
        assert result.safe is True

    def test_unsafe_mismatch(self):
        result = ic3_verify(
            UNSAFE_DIMENSION_MISMATCH, input_shapes={"x": ("batch", 10)}
        )
        assert result.safe is False
        assert result.counterexample_depth is not None

    def test_with_symbolic_dims(self):
        result = ic3_verify(
            SIMPLE_LINEAR,
            symbolic_dims={"batch": "batch_size"},
            input_shapes={"x": ("batch", 10)},
        )
        assert result.safe is True
        assert "batch" in result.symbolic_dims or "batch_size" in str(result.symbolic_dims)

    def test_parametric_all_batch_sizes(self):
        result = ic3_verify(
            TWO_LAYER_MLP,
            symbolic_dims={"batch": "N"},
            input_shapes={"x": ("batch", 784)},
        )
        assert result.safe is True

    def test_verification_time_recorded(self):
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert result.verification_time_ms > 0

    def test_frames_computed_positive(self):
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert result.frames_computed >= 1

    def test_z3_queries_positive(self):
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert result.z3_queries >= 1

    def test_invalid_source(self):
        result = ic3_verify("not valid python {{{{", input_shapes={"x": (1, 10)})
        assert result.safe is False

    def test_no_input_shapes(self):
        result = ic3_verify(SIMPLE_LINEAR)
        assert isinstance(result, IC3Result)

    def test_with_relu(self):
        result = ic3_verify(WITH_RELU, input_shapes={"x": ("batch", 10)})
        assert result.safe is True

    def test_with_dropout(self):
        result = ic3_verify(WITH_DROPOUT, input_shapes={"x": ("batch", 10)})
        assert result.safe is True

    def test_with_batchnorm(self):
        result = ic3_verify(BATCHNORM_MODEL, input_shapes={"x": ("batch", 10)})
        assert result.safe is True

    def test_identity_dims(self):
        result = ic3_verify(IDENTITY_MODEL, input_shapes={"x": ("batch", 10)})
        assert result.safe is True

    def test_wide_mlp(self):
        result = ic3_verify(WIDE_MLP, input_shapes={"x": ("batch", 512)})
        assert result.safe is True

    def test_max_frames_limit(self):
        result = ic3_verify(
            SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)}, max_frames=5
        )
        assert isinstance(result, IC3Result)

    def test_solver_timeout(self):
        result = ic3_verify(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            solver_timeout_ms=100,
        )
        assert isinstance(result, IC3Result)

    def test_no_interpolation(self):
        result = ic3_verify(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            use_interpolation=False,
        )
        assert result.safe is True

    def test_invariant_clauses_on_safe(self):
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        if result.safe:
            assert isinstance(result.invariant_clauses, list)

    def test_counterexample_trace_on_unsafe(self):
        result = ic3_verify(
            UNSAFE_DIMENSION_MISMATCH, input_shapes={"x": ("batch", 10)}
        )
        assert result.safe is False
        # Trace may or may not be populated
        if result.counterexample_trace is not None:
            assert isinstance(result.counterexample_trace, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixed-point detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestFixedPoint:
    def test_trivially_safe_fixed_point(self):
        """A model with matching dims should reach fixed point quickly."""
        result = ic3_verify(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert result.safe is True

    def test_deep_chain_fixed_point(self):
        result = ic3_verify(DEEP_LINEAR_CHAIN, input_shapes={"x": ("batch", 128)})
        assert result.safe is True
        assert result.frames_computed >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test integration with verify_model
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerifyModelIntegration:
    def test_bounded_mode_default(self):
        result = verify_model(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert isinstance(result, VerificationResult)
        assert result.safe is True

    def test_unbounded_mode(self):
        result = verify_model(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            verification_mode="unbounded",
            symbolic_dims={"batch": "batch_size"},
        )
        assert isinstance(result, VerificationResult)
        assert result.safe is True

    def test_unbounded_unsafe(self):
        result = verify_model(
            UNSAFE_DIMENSION_MISMATCH,
            input_shapes={"x": ("batch", 10)},
            verification_mode="unbounded",
        )
        assert isinstance(result, VerificationResult)
        assert result.safe is False

    def test_bounded_backward_compatible(self):
        """Ensure adding verification_mode doesn't break existing behavior."""
        result_old = verify_model(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        result_new = verify_model(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            verification_mode="bounded",
        )
        assert result_old.safe == result_new.safe


# ═══════════════════════════════════════════════════════════════════════════════
# Test parametric verification (for all symbolic values)
# ═══════════════════════════════════════════════════════════════════════════════


class TestParametricVerification:
    def test_all_batch_sizes(self):
        """Verify model is safe for ALL batch sizes (symbolic N > 0)."""
        result = ic3_verify(
            SIMPLE_LINEAR,
            symbolic_dims={"batch": "N"},
            input_shapes={"x": ("batch", 10)},
        )
        assert result.safe is True

    def test_all_batch_sizes_deep(self):
        result = ic3_verify(
            DEEP_LINEAR_CHAIN,
            symbolic_dims={"batch": "B"},
            input_shapes={"x": ("batch", 128)},
        )
        assert result.safe is True

    def test_symbolic_batch_unsafe(self):
        result = ic3_verify(
            UNSAFE_DIMENSION_MISMATCH,
            symbolic_dims={"batch": "B"},
            input_shapes={"x": ("batch", 10)},
        )
        assert result.safe is False
