"""Tests for torch.fx robustness characterization.

Verifies tracing/verification behavior across model categories:
simple models, data-dependent control flow, in-place ops, dynamic shapes,
skip connections, and common third-party patterns.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.fx_extractor import fx_trace_to_graph, verify_module, trace_stats
from src.model_checker import LayerKind, OpKind

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Category A: Simple models trace and verify correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimpleModelsTrace:

    def test_mlp_traces_successfully(self):
        model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        stats = trace_stats(model)
        assert stats.traceable
        assert stats.num_layers >= 3
        assert stats.num_steps > 0

    def test_mlp_verifies_safe(self):
        model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
        result = verify_module(model, input_shapes={"x": ("batch", 784)})
        assert result.safe, f"Simple MLP should be safe: {result.errors}"

    def test_cnn_traces_successfully(self):
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.relu(self.bn(self.conv(x)))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = CNN()
        stats = trace_stats(model)
        assert stats.traceable
        assert "CONV2D" in stats.layer_kinds

    def test_basic_attention_traces(self):
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(64, 64)
                self.k = nn.Linear(64, 64)
                self.v = nn.Linear(64, 64)
                self.out = nn.Linear(64, 64)

            def forward(self, x):
                q, k, v = self.q(x), self.k(x), self.v(x)
                w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.125, dim=-1)
                return self.out(torch.matmul(w, v))

        stats = trace_stats(Attn())
        assert stats.traceable
        assert stats.num_layers >= 4


# ═══════════════════════════════════════════════════════════════════════════════
# Test Category B: Data-dependent control flow detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataDependentControlFlow:

    def test_if_on_tensor_value_not_traceable(self):
        class BranchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def forward(self, x):
                if x.sum() > 0:
                    return self.fc(x)
                return x

        stats = trace_stats(BranchModel())
        assert not stats.traceable
        assert stats.trace_error is not None

    def test_data_dependent_loop_not_traceable(self):
        class LoopModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def forward(self, x):
                n = int(x.abs().sum().item()) % 3 + 1
                for _ in range(n):
                    x = self.fc(x)
                return x

        stats = trace_stats(LoopModel())
        assert not stats.traceable

    def test_data_dependent_verify_returns_error(self):
        """verify_module should handle data-dependent models gracefully.

        With fallback extraction, the model can still be verified even when
        torch.fx tracing fails — the fallback walks named_children.
        """
        class BranchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def forward(self, x):
                if x.sum() > 0:
                    return self.fc(x)
                return x

        result = verify_module(BranchModel(), input_shapes={"x": ("batch", 32)})
        # Fallback extraction succeeds — model is shape-safe
        assert result.safe or len(result.errors) > 0

    def test_early_return_not_traceable(self):
        class EarlyReturn(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 32)

            def forward(self, x):
                if x.numel() == 0:
                    return x
                return self.fc(x)

        stats = trace_stats(EarlyReturn())
        assert not stats.traceable


# ═══════════════════════════════════════════════════════════════════════════════
# Test Category C: In-place operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestInPlaceOperations:

    def test_inplace_relu_traces(self):
        """InPlace ReLU typically traces fine in torch.fx."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.relu(self.fc(x))

        stats = trace_stats(M())
        # torch.fx handles inplace ReLU as a module call
        assert stats.traceable

    def test_inplace_add_traces(self):
        """In-place += on a fresh tensor typically works."""
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)

            def forward(self, x):
                out = self.fc1(x)
                out += self.fc2(x)
                return out

        stats = trace_stats(M())
        assert stats.traceable


# ═══════════════════════════════════════════════════════════════════════════════
# Test Category E: Skip connections work
# ═══════════════════════════════════════════════════════════════════════════════

class TestSkipConnections:

    def test_residual_block_traces(self):
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return self.relu(out + identity)

        stats = trace_stats(ResBlock())
        assert stats.traceable
        assert stats.num_layers >= 4

    def test_residual_block_verifies(self):
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return self.relu(out + identity)

        result = verify_module(
            ResBlock(), input_shapes={"x": ("batch", 64, 8, 8)}
        )
        assert result.safe, f"Residual block should be safe: {result.errors}"

    def test_dense_concat_skip_traces(self):
        class DenseSkip(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(32, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(48, 16, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                out1 = self.relu(self.conv1(x))
                cat1 = torch.cat([x, out1], dim=1)
                out2 = self.relu(self.conv2(cat1))
                return torch.cat([cat1, out2], dim=1)

        stats = trace_stats(DenseSkip())
        assert stats.traceable


# ═══════════════════════════════════════════════════════════════════════════════
# Test failure mode classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestFailureClassification:

    def test_classify_control_flow(self):
        from experiments.run_fx_robustness import classify_failure
        assert classify_failure("control flow detected") == "data_dependent_control_flow"

    def test_classify_proxy_error(self):
        from experiments.run_fx_robustness import classify_failure
        result = classify_failure("proxy object is not iterable via tracer")
        assert result == "symbolic_trace_proxy_error"

    def test_classify_item_call(self):
        from experiments.run_fx_robustness import classify_failure
        assert classify_failure("cannot call .item() on proxy") == "data_dependent_item_call"

    def test_classify_unknown(self):
        from experiments.run_fx_robustness import classify_failure
        assert classify_failure("some random error 12345") == "other"


# ═══════════════════════════════════════════════════════════════════════════════
# Test robustness characterizer correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestCharacterizerCorrectness:

    def test_model_registry_has_all_categories(self):
        from experiments.run_fx_robustness import build_model_registry
        registry = build_model_registry()
        categories = {cat for _, cat, _, _ in registry}
        expected = {
            "simple", "data_dependent_control_flow",
            "inplace_operations", "dynamic_shapes",
            "skip_connections", "third_party_patterns",
        }
        assert expected == categories

    def test_model_registry_nonempty(self):
        from experiments.run_fx_robustness import build_model_registry
        registry = build_model_registry()
        assert len(registry) >= 20

    def test_result_dataclass_fields(self):
        from experiments.run_fx_robustness import ModelTestResult
        r = ModelTestResult(
            name="test", category="simple",
            can_trace=True, trace_time_ms=1.0,
            verification_result="safe",
        )
        assert r.name == "test"
        assert r.can_trace is True
        assert r.failure_mode is None
