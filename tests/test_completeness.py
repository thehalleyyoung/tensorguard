"""Tests for relative completeness of the linear fragment."""

from __future__ import annotations

import textwrap

import pytest

from src.completeness import (
    LINEAR_FRAGMENT_OPS,
    NON_LINEAR_OPS,
    CompletenessResult,
    FragmentClassification,
    FragmentStatus,
    classify_fragment,
    check_relative_completeness,
    is_in_linear_fragment,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
    extract_computation_graph,
)
from src.decidability import ComplexityClass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _src(body: str) -> str:
    """Wrap model body in proper nn.Module boilerplate."""
    return textwrap.dedent(body).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# A.  Fragment classification — is_in_linear_fragment / classify_fragment
# ═══════════════════════════════════════════════════════════════════════════════

class TestLinearFragmentDefinition:
    """Verify that the linear fragment sets are correctly defined."""

    def test_linear_and_nonlinear_disjoint(self):
        assert LINEAR_FRAGMENT_OPS & NON_LINEAR_OPS == frozenset()

    def test_reshape_is_nonlinear(self):
        assert OpKind.RESHAPE in NON_LINEAR_OPS

    def test_flatten_is_nonlinear(self):
        assert OpKind.FLATTEN in NON_LINEAR_OPS

    def test_layer_call_is_linear(self):
        assert OpKind.LAYER_CALL in LINEAR_FRAGMENT_OPS

    def test_matmul_is_linear(self):
        assert OpKind.MATMUL in LINEAR_FRAGMENT_OPS

    def test_add_is_linear(self):
        assert OpKind.ADD in LINEAR_FRAGMENT_OPS


class TestIsInLinearFragment:
    """Test is_in_linear_fragment on various computation graphs."""

    def test_single_linear_layer(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        assert is_in_linear_fragment(graph)

    def test_mlp_in_fragment(self):
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        """)
        graph = extract_computation_graph(src)
        assert is_in_linear_fragment(graph)

    def test_model_with_reshape_outside_fragment(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        assert not is_in_linear_fragment(graph)

    def test_model_with_flatten_outside_fragment(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.flat = nn.Flatten()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = self.flat(x)
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        assert not is_in_linear_fragment(graph)

    def test_conv_batchnorm_in_fragment(self):
        src = _src("""
        import torch.nn as nn
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x
        """)
        graph = extract_computation_graph(src)
        assert is_in_linear_fragment(graph)

    def test_empty_graph_in_fragment(self):
        graph = ComputationGraph(class_name="Empty")
        assert is_in_linear_fragment(graph)


class TestClassifyFragment:
    """Test detailed fragment classification."""

    def test_linear_model_complexity_p(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        frag = classify_fragment(graph)
        assert frag.in_fragment
        assert frag.complexity == ComplexityClass.P
        assert len(frag.non_linear_ops) == 0

    def test_reshape_model_np_hard(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = x.reshape(-1, 784)
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        frag = classify_fragment(graph)
        assert not frag.in_fragment
        assert frag.complexity == ComplexityClass.NP_HARD
        assert OpKind.RESHAPE in frag.non_linear_ops

    def test_non_linear_steps_recorded(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        """)
        graph = extract_computation_graph(src)
        frag = classify_fragment(graph)
        assert len(frag.non_linear_steps) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# B.  Completeness on SAFE linear models
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletenessSafeModels:
    """Verify completeness holds for safe models in the linear fragment."""

    def test_single_linear_safe(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 10)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified

    def test_two_layer_mlp_safe(self):
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 10)
                self.fc2 = nn.Linear(10, 5)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return self.fc2(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 20)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified

    def test_conv2d_safe(self):
        src = _src("""
        import torch.nn as nn
        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
            def forward(self, x):
                return self.conv(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 3, 32, 32)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified

    def test_batchnorm_chain_safe(self):
        src = _src("""
        import torch.nn as nn
        class BNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.bn = nn.BatchNorm2d(16)
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x
        """)
        r = check_relative_completeness(src, {"x": ("batch", 3, 32, 32)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified

    def test_deep_mlp_safe(self):
        """10+ layer deep MLP should still be in fragment and complete."""
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class DeepMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 64)
                self.fc4 = nn.Linear(64, 64)
                self.fc5 = nn.Linear(64, 64)
                self.fc6 = nn.Linear(64, 64)
                self.fc7 = nn.Linear(64, 64)
                self.fc8 = nn.Linear(64, 64)
                self.fc9 = nn.Linear(64, 64)
                self.fc10 = nn.Linear(64, 32)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = F.relu(self.fc6(x))
                x = F.relu(self.fc7(x))
                x = F.relu(self.fc8(x))
                x = F.relu(self.fc9(x))
                return self.fc10(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 64)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified


# ═══════════════════════════════════════════════════════════════════════════════
# C.  Completeness on UNSAFE linear models
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletenessUnsafeModels:
    """Verify completeness holds for unsafe models in the linear fragment."""

    def test_dimension_mismatch_detected(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(50, 5)
            def forward(self, x):
                x = self.fc1(x)
                return self.fc2(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 10)})
        assert r.in_fragment
        assert r.tg_verdict == "UNSAFE"
        assert r.completeness_verified

    def test_wrong_input_shape(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 999)})
        assert r.in_fragment
        assert r.tg_verdict == "UNSAFE"
        assert r.completeness_verified

    def test_conv_channel_mismatch(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
            def forward(self, x):
                x = self.conv1(x)
                return self.conv2(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 3, 32, 32)})
        assert r.in_fragment
        assert r.tg_verdict == "UNSAFE"
        assert r.completeness_verified


# ═══════════════════════════════════════════════════════════════════════════════
# D.  Fragment boundary — models with reshape are NOT in fragment
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentBoundary:
    """Verify models with reshape/flatten are classified outside the fragment."""

    def test_reshape_excludes_from_fragment(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = x.view(-1, 784)
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 1, 28, 28)})
        assert not r.in_fragment
        assert r.tg_verdict == "N/A"
        assert not r.completeness_verified
        assert "outside the linear fragment" in r.explanation.lower() or "outside" in r.explanation.lower()

    def test_flatten_excludes_from_fragment(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.flat = nn.Flatten()
                self.fc = nn.Linear(784, 10)
            def forward(self, x):
                x = self.flat(x)
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 1, 28, 28)})
        assert not r.in_fragment
        assert not r.completeness_verified

    def test_mixed_linear_reshape_outside(self):
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.fc = nn.Linear(400, 10)
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 3, 8, 8)})
        assert not r.in_fragment


# ═══════════════════════════════════════════════════════════════════════════════
# E.  Architecture patterns
# ═══════════════════════════════════════════════════════════════════════════════

class TestArchitecturePatterns:
    """Test completeness on various architecture patterns."""

    def test_residual_connection_safe(self):
        """Skip/residual connections should be in the linear fragment."""
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)
            def forward(self, x):
                residual = x
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = x + residual
                return F.relu(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 64)})
        assert r.in_fragment
        assert r.tg_verdict == "SAFE"
        assert r.completeness_verified

    def test_layernorm_in_fragment(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(64)
                self.fc = nn.Linear(64, 32)
            def forward(self, x):
                x = self.ln(x)
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 64)})
        assert r.in_fragment

    def test_dropout_in_fragment(self):
        src = _src("""
        import torch.nn as nn
        import torch.nn.functional as F
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 16)
                self.drop = nn.Dropout(0.5)
                self.fc2 = nn.Linear(16, 8)
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.drop(x)
                return self.fc2(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 32)})
        assert r.in_fragment
        assert r.completeness_verified


# ═══════════════════════════════════════════════════════════════════════════════
# F.  Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for completeness checking."""

    def test_invalid_source(self):
        r = check_relative_completeness("not valid python {{{{", {"x": ("batch", 10)})
        assert not r.completeness_verified
        assert r.tg_verdict == "ERROR"

    def test_no_input_shapes(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        # Should still work (TG uses symbolic shapes)
        r = check_relative_completeness(src)
        assert r.in_fragment

    def test_completeness_result_fields(self):
        src = _src("""
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)
        """)
        r = check_relative_completeness(src, {"x": ("batch", 10)})
        assert isinstance(r, CompletenessResult)
        assert isinstance(r.in_fragment, bool)
        assert isinstance(r.tg_verdict, str)
        assert isinstance(r.completeness_verified, bool)
        assert isinstance(r.explanation, str)
        assert r.fragment_classification is not None
        assert r.verification_result is not None
