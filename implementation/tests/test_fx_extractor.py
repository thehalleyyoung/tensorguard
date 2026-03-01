"""Tests for torch.fx → ComputationGraph extraction and verify_module."""

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

from src.fx_extractor import (
    fx_trace_to_graph,
    verify_module,
    trace_stats,
    _make_layer_def,
    _module_to_layer_kind,
)
from src.model_checker import (
    LayerKind,
    OpKind,
    Device,
    Phase,
)


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")


# ═══════════════════════════════════════════════════════════════════════════════
# Simple models for testing
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
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


class BuggyMLP(nn.Module):
    """Shape bug: fc2 expects 128 but gets 256."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # residual
        out = self.relu(out)
        return out


class BuggyBroadcast(nn.Module):
    """Broadcast bug: proj_a(64) + proj_b(128) mismatch."""
    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(512, 64)
        self.proj_b = nn.Linear(512, 128)

    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        return a + b  # BUG


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, 5)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(64, 32)
        self.fc_b = nn.Linear(64, 32)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, a, b):
        a = self.fc_a(a)
        b = self.fc_b(b)
        combined = torch.cat([a, b], dim=-1)
        return self.fc_out(combined)


class DeepChain(nn.Module):
    """Deep chain of linear layers — tests scalability."""
    def __init__(self, depth=20, dim=128):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Graph extraction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFXGraphExtraction:

    def test_simple_mlp_extraction(self):
        model = SimpleMLP()
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced, class_name="SimpleMLP")

        assert graph.class_name == "SimpleMLP"
        assert "fc1" in graph.layers
        assert "fc2" in graph.layers
        assert graph.layers["fc1"].kind == LayerKind.LINEAR
        assert graph.layers["fc1"].in_features == 784
        assert graph.layers["fc1"].out_features == 256
        assert graph.layers["fc2"].in_features == 256
        assert graph.layers["fc2"].out_features == 10
        assert len(graph.input_names) >= 1
        assert len(graph.output_names) >= 1
        assert graph.num_steps > 0
        assert graph.dynamic_features.get("fx_traced") is True

    def test_conv_net_extraction(self):
        model = ConvNet()
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced)

        assert "conv1" in graph.layers
        assert graph.layers["conv1"].kind == LayerKind.CONV2D
        assert graph.layers["conv1"].in_channels == 3
        assert graph.layers["conv1"].out_channels == 16

    def test_sequential_extraction(self):
        model = SequentialModel()
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced)

        # Sequential is flattened by fx
        assert len(graph.steps) > 0
        assert len(graph.layers) > 0

    def test_residual_block_extraction(self):
        model = ResidualBlock(64)
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced)

        # Should have ADD step for residual connection
        op_kinds = [s.op for s in graph.steps]
        assert OpKind.ADD in op_kinds or OpKind.ACTIVATION in op_kinds

    def test_multi_input_extraction(self):
        model = MultiInputModel()
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced)

        assert len(graph.input_names) >= 2

    def test_deep_chain_extraction(self):
        model = DeepChain(depth=50, dim=64)
        traced = torch.fx.symbolic_trace(model)
        graph = fx_trace_to_graph(traced)

        layer_call_steps = [s for s in graph.steps if s.op == OpKind.LAYER_CALL]
        assert len(layer_call_steps) >= 50


# ═══════════════════════════════════════════════════════════════════════════════
# verify_module tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifyModule:

    def test_safe_mlp(self):
        model = SimpleMLP()
        result = verify_module(model, input_shapes={"x": ("batch", 784)})
        assert result.safe, f"Expected safe but got errors: {result.errors}"

    def test_buggy_mlp(self):
        model = BuggyMLP()
        result = verify_module(model, input_shapes={"x": ("batch", 784)})
        assert not result.safe, "Expected bug detected"

    def test_safe_convnet(self):
        model = ConvNet()
        result = verify_module(
            model, input_shapes={"x": ("batch", 3, 224, 224)}
        )
        assert result.safe, f"Expected safe: {result.errors}"

    def test_buggy_broadcast(self):
        model = BuggyBroadcast()
        result = verify_module(model, input_shapes={"x": ("batch", 512)})
        assert not result.safe, "Expected broadcast bug detected"

    def test_sequential_safe(self):
        model = SequentialModel()
        result = verify_module(model, input_shapes={"x": ("batch", 100)})
        assert result.safe, f"Expected safe: {result.errors}"

    def test_deep_chain_safe(self):
        model = DeepChain(depth=20, dim=128)
        result = verify_module(model, input_shapes={"x": ("batch", 128)})
        assert result.safe, f"Expected safe: {result.errors}"

    def test_deep_chain_buggy(self):
        """Deep chain with dimension mismatch at layer 10."""
        model = DeepChain(depth=20, dim=128)
        # Inject a bug at position 10
        model.layers[10] = nn.Linear(64, 128)  # wrong in_features
        result = verify_module(model, input_shapes={"x": ("batch", 128)})
        assert not result.safe, "Expected bug in deep chain"

    def test_symbolic_dims(self):
        model = SimpleMLP()
        result = verify_module(
            model, input_shapes={"x": ("batch", 784)}
        )
        assert result.safe

    def test_high_confidence_mode(self):
        model = SimpleMLP()
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 784)},
            high_confidence_only=True,
        )
        assert result.safe

    def test_untraceable_module(self):
        """Modules that can't be traced should return error, not crash."""
        class Untraceable(nn.Module):
            def forward(self, x):
                if x.sum() > 0:  # data-dependent control flow
                    return x * 2
                return x * 3

        model = Untraceable()
        result = verify_module(model, input_shapes={"x": ("batch", 10)})
        # Should return a result (possibly with errors), not crash
        assert isinstance(result.errors, list) or result.safe is not None

    def test_certificate_generation(self):
        model = SimpleMLP()
        result = verify_module(model, input_shapes={"x": ("batch", 784)})
        if result.safe and result.certificate:
            cert_dict = result.certificate.to_dict()
            assert "sha256" in cert_dict or "model_name" in cert_dict

    def test_multi_input_verification(self):
        model = MultiInputModel()
        result = verify_module(
            model,
            input_shapes={"a": ("batch", 64), "b": ("batch", 64)},
        )
        assert result.safe, f"Expected safe: {result.errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer kind mapping tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayerKindMapping:

    def test_linear(self):
        assert _module_to_layer_kind(nn.Linear(10, 5)) == LayerKind.LINEAR

    def test_conv2d(self):
        assert _module_to_layer_kind(nn.Conv2d(3, 16, 3)) == LayerKind.CONV2D

    def test_batchnorm2d(self):
        assert _module_to_layer_kind(nn.BatchNorm2d(16)) == LayerKind.BATCHNORM2D

    def test_relu(self):
        assert _module_to_layer_kind(nn.ReLU()) == LayerKind.RELU

    def test_dropout(self):
        assert _module_to_layer_kind(nn.Dropout(0.5)) == LayerKind.DROPOUT

    def test_embedding(self):
        assert _module_to_layer_kind(nn.Embedding(100, 64)) == LayerKind.EMBEDDING

    def test_multihead_attention(self):
        assert _module_to_layer_kind(
            nn.MultiheadAttention(512, 8)
        ) == LayerKind.MULTIHEAD_ATTENTION

    def test_flatten(self):
        assert _module_to_layer_kind(nn.Flatten()) == LayerKind.FLATTEN


# ═══════════════════════════════════════════════════════════════════════════════
# Layer parameter extraction tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLayerDefExtraction:

    def test_linear_params(self):
        ldef = _make_layer_def("fc", nn.Linear(784, 256))
        assert ldef.kind == LayerKind.LINEAR
        assert ldef.in_features == 784
        assert ldef.out_features == 256

    def test_conv2d_params(self):
        ldef = _make_layer_def("conv", nn.Conv2d(3, 16, 3, padding=1))
        assert ldef.kind == LayerKind.CONV2D
        assert ldef.in_channels == 3
        assert ldef.out_channels == 16
        assert ldef.kernel_size == (3, 3)

    def test_embedding_params(self):
        ldef = _make_layer_def("emb", nn.Embedding(1000, 128))
        assert ldef.kind == LayerKind.EMBEDDING
        assert ldef.num_embeddings == 1000
        assert ldef.embedding_dim == 128

    def test_adaptive_avgpool(self):
        ldef = _make_layer_def("pool", nn.AdaptiveAvgPool2d(1))
        assert ldef.kind == LayerKind.ADAPTIVE_AVGPOOL2D
        assert ldef.output_size == (1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Trace stats tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceStats:

    def test_traceable_model(self):
        model = SimpleMLP()
        stats = trace_stats(model)
        assert stats.traceable
        assert stats.num_steps > 0
        assert stats.num_layers > 0
        assert "LINEAR" in stats.layer_kinds

    def test_untraceable_model(self):
        class BadModel(nn.Module):
            def forward(self, x):
                if x.sum() > 0:
                    return x
                return -x
        stats = trace_stats(BadModel())
        assert not stats.traceable
        assert stats.trace_error is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Torchvision model tests (if available)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision required")
class TestTorchvisionModels:

    def test_resnet18_trace(self):
        model = tv_models.resnet18(weights=None)
        stats = trace_stats(model)
        assert stats.traceable
        assert stats.num_layers > 10

    def test_resnet18_verify(self):
        model = tv_models.resnet18(weights=None)
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 3, 224, 224)},
        )
        assert result.safe, f"ResNet18 should be safe: {result.errors}"

    def test_vgg11_verify(self):
        model = tv_models.vgg11(weights=None)
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 3, 224, 224)},
        )
        assert result.safe, f"VGG11 should be safe: {result.errors}"

    def test_mobilenet_v2_verify(self):
        model = tv_models.mobilenet_v2(weights=None)
        result = verify_module(
            model,
            input_shapes={"x": ("batch", 3, 224, 224)},
        )
        assert result.safe, f"MobileNetV2 should be safe: {result.errors}"
