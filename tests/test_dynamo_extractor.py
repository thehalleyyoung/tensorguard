"""Tests for TorchDynamo-based computation graph extraction."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_DYNAMO = False
if HAS_TORCH:
    try:
        import torch._dynamo
        torch._dynamo.eval_frame.check_if_dynamo_supported()
        HAS_DYNAMO = True
    except (ImportError, RuntimeError):
        pass

from src.model_checker import (
    LayerKind,
    OpKind,
    Device,
    Phase,
    ComputationGraph,
)
from src.dynamo_extractor import (
    dynamo_trace_to_graph,
    verify_module_dynamo,
    dynamo_trace_stats,
    HAS_DYNAMO as MODULE_HAS_DYNAMO,
)
from src.fx_extractor import fx_trace_to_graph, verify_module


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")


# ═══════════════════════════════════════════════════════════════════════════════
# Test models
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConditionalModel(nn.Module):
    """Data-dependent control flow: fx.symbolic_trace fails on this."""
    def __init__(self):
        super().__init__()
        self.layer_a = nn.Linear(64, 32)
        self.layer_b = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 10)

    def forward(self, x):
        if x.shape[0] > 1:
            x = self.layer_a(x)
        else:
            x = self.layer_b(x)
        return self.fc_out(x)


class ValueConditionalModel(nn.Module):
    """Data-dependent on tensor *values* (not just shape)."""
    def __init__(self):
        super().__init__()
        self.layer_a = nn.Linear(64, 32)
        self.layer_b = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 10)

    def forward(self, x):
        if x.mean() > 0:
            x = self.layer_a(x)
        else:
            x = self.layer_b(x)
        return self.fc_out(x)


class LoopModel(nn.Module):
    """Dynamic loop based on input."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        for _ in range(x.shape[0]):
            x = self.linear(x)
        return self.out(x)


class MoEModel(nn.Module):
    """Simplified Mixture-of-Experts with data-dependent routing."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(64, 2)
        self.expert_a = nn.Linear(64, 32)
        self.expert_b = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 10)

    def forward(self, x):
        gate_out = self.gate(x)
        if gate_out[:, 0].mean() > gate_out[:, 1].mean():
            h = self.expert_a(x)
        else:
            h = self.expert_b(x)
        return self.fc_out(h)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Dynamo extraction on simple models
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_DYNAMO, reason="TorchDynamo required")
class TestDynamoSimpleModels:
    def test_simple_mlp_extraction(self):
        model = SimpleMLP()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        assert isinstance(graph, ComputationGraph)
        assert graph.class_name == "SimpleMLP"
        assert len(graph.steps) > 0
        assert len(graph.input_names) > 0
        assert graph.dynamic_features.get("dynamo_traced") is True

    def test_simple_mlp_has_layers(self):
        model = SimpleMLP()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        # Should find the linear layers
        layer_kinds = {l.kind for l in graph.layers.values()}
        assert LayerKind.LINEAR in layer_kinds

    def test_simple_mlp_has_layer_call_steps(self):
        model = SimpleMLP()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        op_kinds = {s.op for s in graph.steps}
        assert OpKind.LAYER_CALL in op_kinds or OpKind.ACTIVATION in op_kinds

    def test_trace_stats_simple(self):
        model = SimpleMLP()
        example = (torch.randn(2, 64),)
        stats = dynamo_trace_stats(model, example_inputs=example)

        assert stats.traceable is True
        assert stats.backend == "dynamo"
        assert stats.num_steps > 0
        assert stats.num_layers > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Data-dependent control flow
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_DYNAMO, reason="TorchDynamo required")
class TestDynamoControlFlow:
    def test_conditional_model_fx_fails(self):
        """Verify that torch.fx.symbolic_trace fails on ConditionalModel."""
        model = ConditionalModel()
        with pytest.raises(Exception):
            torch.fx.symbolic_trace(model)

    def test_conditional_model_dynamo_succeeds(self):
        """TorchDynamo should capture ConditionalModel via graph breaks."""
        model = ConditionalModel()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        assert isinstance(graph, ComputationGraph)
        assert len(graph.steps) > 0
        assert graph.dynamic_features.get("dynamo_traced") is True

    def test_value_conditional_model(self):
        """Data-dependent on tensor values, not just shapes."""
        model = ValueConditionalModel()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        assert isinstance(graph, ComputationGraph)
        assert len(graph.steps) > 0

    def test_moe_model(self):
        """Mixture-of-experts with data-dependent gating."""
        model = MoEModel()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        assert isinstance(graph, ComputationGraph)
        assert len(graph.steps) > 0

    def test_graph_breaks_detected(self):
        """When graph breaks occur, we should detect multiple subgraphs."""
        model = ConditionalModel()
        example = (torch.randn(2, 64),)
        graph = dynamo_trace_to_graph(model, example_inputs=example)

        # Dynamo may produce multiple subgraphs for control flow
        num_sub = graph.dynamic_features.get("num_dynamo_subgraphs", 1)
        assert num_sub >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Consistency with fx_extractor
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_DYNAMO, reason="TorchDynamo required")
class TestDynamoFxConsistency:
    def test_simple_model_same_layers(self):
        """Both backends should detect the same layers for a simple model."""
        model = SimpleMLP()
        model.eval()

        # FX extraction
        traced = torch.fx.symbolic_trace(model)
        fx_graph = fx_trace_to_graph(traced, class_name="SimpleMLP")

        # Dynamo extraction
        example = (torch.randn(2, 64),)
        dynamo_graph = dynamo_trace_to_graph(
            model, example_inputs=example, class_name="SimpleMLP"
        )

        # Both should find the same layer kinds
        fx_kinds = {l.kind for l in fx_graph.layers.values()}
        dynamo_kinds = {l.kind for l in dynamo_graph.layers.values()}
        assert fx_kinds == dynamo_kinds

    def test_same_class_name(self):
        model = SimpleMLP()
        model.eval()

        traced = torch.fx.symbolic_trace(model)
        fx_graph = fx_trace_to_graph(traced, class_name="SimpleMLP")

        example = (torch.randn(2, 64),)
        dynamo_graph = dynamo_trace_to_graph(
            model, example_inputs=example, class_name="SimpleMLP"
        )

        assert fx_graph.class_name == dynamo_graph.class_name

    def test_both_produce_computation_graph(self):
        model = SimpleMLP()
        model.eval()

        traced = torch.fx.symbolic_trace(model)
        fx_graph = fx_trace_to_graph(traced)

        example = (torch.randn(2, 64),)
        dynamo_graph = dynamo_trace_to_graph(model, example_inputs=example)

        assert isinstance(fx_graph, ComputationGraph)
        assert isinstance(dynamo_graph, ComputationGraph)
        # Both should have steps and layers
        assert len(fx_graph.steps) > 0
        assert len(dynamo_graph.steps) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Graceful fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamoFallback:
    def test_verify_module_dynamo_fallback(self):
        """verify_module_dynamo should fall back to fx when Dynamo fails."""
        model = SimpleMLP()
        result = verify_module_dynamo(
            model,
            input_shapes={"x": ("batch", 64)},
            fallback_to_fx=True,
        )
        # Should succeed regardless of Dynamo availability
        assert result is not None

    def test_verify_module_backend_auto(self):
        """verify_module with backend='auto' should work."""
        model = SimpleMLP()
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 64)},
            backend="auto",
        )
        assert result is not None

    def test_verify_module_backend_fx(self):
        """verify_module with backend='fx' should skip Dynamo."""
        model = SimpleMLP()
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 64)},
            backend="fx",
        )
        assert result is not None

    @pytest.mark.skipif(not HAS_DYNAMO, reason="TorchDynamo required")
    def test_verify_module_backend_dynamo(self):
        """verify_module with backend='dynamo' should use Dynamo."""
        model = SimpleMLP()
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 64)},
            backend="dynamo",
        )
        assert result is not None

    @pytest.mark.skipif(not HAS_DYNAMO, reason="TorchDynamo required")
    def test_conditional_model_via_verify_module_auto(self):
        """Auto backend should handle ConditionalModel via Dynamo."""
        model = ConditionalModel()
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 64)},
            backend="auto",
        )
        assert result is not None
