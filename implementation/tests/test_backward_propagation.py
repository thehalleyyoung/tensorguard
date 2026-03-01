"""Tests for backward constraint propagation in ConstraintVerifier.

These tests verify that backward propagation catches wrong_out_features
mutations that forward-only propagation misses.
"""

import pytest
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    ConstraintVerifier,
    ComputationGraph,
    ComputationStep,
    SafetyViolation,
    VerificationResult,
    ModelState,
    Phase,
    Device,
    LayerKind,
    OpKind,
    LayerDef,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test models
# ═══════════════════════════════════════════════════════════════════════════════

# Correct two-layer MLP: fc1 outputs 256, fc2 expects 256
TWO_LAYER_CORRECT = """\
import torch.nn as nn

class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

# Wrong out_features: fc1 outputs 128 but fc2 expects 256
WRONG_OUT_FEATURES_MLP = """\
import torch.nn as nn

class WrongOutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

# Wrong out_features with ReLU in between
WRONG_OUT_WITH_RELU = """\
import torch.nn as nn

class WrongOutReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
"""

# Three layers: wrong out_features in middle
THREE_LAYER_WRONG_MIDDLE = """\
import torch.nn as nn

class WrongMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""

# Conv layers with wrong out_channels
WRONG_OUT_CHANNELS_CONV = """\
import torch.nn as nn

class WrongConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""

# Correct conv chain for comparison
CORRECT_CONV = """\
import torch.nn as nn

class CorrectConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardPropagation:
    """Test backward constraint propagation catches wrong_out_features."""

    def test_correct_mlp_is_safe(self):
        """A correctly wired MLP should verify as safe."""
        result = verify_model(TWO_LAYER_CORRECT, {"x": ("batch", 784)})
        assert result.safe, f"Expected safe, got violations: {result.counterexample}"

    def test_wrong_out_features_detected(self):
        """Wrong out_features (128 vs expected 256) must be caught."""
        result = verify_model(WRONG_OUT_FEATURES_MLP, {"x": ("batch", 784)})
        assert not result.safe, "Expected violation for wrong_out_features"

    def test_wrong_out_features_with_relu(self):
        """Wrong out_features should be caught even with ReLU between layers."""
        result = verify_model(WRONG_OUT_WITH_RELU, {"x": ("batch", 784)})
        assert not result.safe, "Expected violation for wrong_out_features through ReLU"

    def test_three_layer_wrong_middle(self):
        """Wrong out_features in the middle layer should be caught."""
        result = verify_model(THREE_LAYER_WRONG_MIDDLE, {"x": ("batch", 784)})
        assert not result.safe, "Expected violation for wrong middle layer out_features"

    def test_wrong_out_channels_conv(self):
        """Wrong out_channels on conv layer should be caught."""
        result = verify_model(
            WRONG_OUT_CHANNELS_CONV,
            {"x": ("batch", 3, 32, 32)},
        )
        assert not result.safe, "Expected violation for wrong conv out_channels"

    def test_correct_conv_is_safe(self):
        """A correctly wired conv chain should verify as safe."""
        result = verify_model(
            CORRECT_CONV,
            {"x": ("batch", 3, 32, 32)},
        )
        assert result.safe, f"Expected safe, got violations: {result.counterexample}"


class TestBackwardConstraintPassDirect:
    """Direct tests for _backward_constraint_pass and related methods."""

    @pytest.mark.skipif(not HAS_Z3, reason="z3 required")
    def test_backward_consumer_constraints_linear(self):
        """_backward_consumer_constraints produces correct constraints for LINEAR."""
        graph = extract_computation_graph(TWO_LAYER_CORRECT)
        checker = ConstraintVerifier(graph, {"x": ("batch", 784)})

        # Find the step that calls fc2
        fc2_step = None
        for step in graph.steps:
            if step.op == OpKind.LAYER_CALL and step.layer_ref == "fc2":
                fc2_step = step
                break
        if fc2_step is None:
            pytest.skip("fc2 step not found in graph")

        # Build a kripke state that has the input tensor for fc2
        init_state = checker._init_state.copy()
        # Simulate: after fc1, the output tensor exists
        from src.tensor_shapes import TensorShape, ShapeDim
        init_state.shape_env[fc2_step.inputs[0]] = TensorShape(
            (ShapeDim("batch"), ShapeDim(256))
        )
        k = checker._build_kripke_state(1, init_state)

        cs = checker._backward_consumer_constraints(fc2_step, k, 1)
        assert len(cs) >= 1, "Expected backward constraints for LINEAR consumer"

    @pytest.mark.skipif(not HAS_Z3, reason="z3 required")
    def test_backward_pass_adds_constraints(self):
        """The backward pass should add constraints to the solver."""
        graph = extract_computation_graph(WRONG_OUT_FEATURES_MLP)
        checker = ConstraintVerifier(graph, {"x": ("batch", 784)})
        result = checker.verify()
        # The backward pass should catch the mismatch
        assert not result.safe, "Backward pass should detect wrong_out_features"

    @pytest.mark.skipif(not HAS_Z3, reason="z3 required")
    def test_backward_pass_no_false_positives(self):
        """Backward pass should not flag correctly wired models."""
        graph = extract_computation_graph(TWO_LAYER_CORRECT)
        checker = ConstraintVerifier(graph, {"x": ("batch", 784)})
        result = checker.verify()
        assert result.safe, "Backward pass should not flag correct models"


class TestBackwardOutputSpecConstraints:
    """Tests for _backward_output_spec_constraints."""

    @pytest.mark.skipif(not HAS_Z3, reason="z3 required")
    def test_reshape_backward_propagation(self):
        """Backward propagation through reshape should preserve element count."""
        src = """\
import torch
import torch.nn as nn

class ReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 784)})
        assert result.safe, f"Expected safe, got: {result.counterexample}"

    @pytest.mark.skipif(not HAS_Z3, reason="z3 required")
    def test_reshape_with_wrong_dims(self):
        """Reshape with wrong dims after mutated out_features should fail."""
        src = """\
import torch
import torch.nn as nn

class BadReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
"""
        result = verify_model(src, {"x": ("batch", 784)})
        assert not result.safe, "Expected violation for wrong reshape dims"
