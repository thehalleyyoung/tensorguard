"""Tests for parametric architecture verification."""

import pytest
from src.parametric import (
    verify_parametric,
    ParametricResult,
    ParametricConstraint,
    _extract_parametric_graph,
    _collect_symbolic_params,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Safe architecture families
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeArchitectures:
    """Test architectures that are universally safe."""

    def test_simple_mlp_safe(self):
        """MLP where fc1 output feeds fc2 input — same symbolic param."""
        result = verify_parametric(
            """
import torch.nn as nn
class SafeMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            arch_params={
                "d_in": {"min": 1},
                "d_hidden": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is True

    def test_single_layer(self):
        """Single linear layer — trivially safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class SingleLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={"d_in": {"min": 1}, "d_out": {"min": 1}},
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is True

    def test_three_layer_chain(self):
        """Three layers in chain — safe because each output feeds next input."""
        result = verify_parametric(
            """
import torch.nn as nn
class ThreeLayer(nn.Module):
    def __init__(self, d1, d2, d3, d4):
        super().__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, d3)
        self.fc3 = nn.Linear(d3, d4)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
            arch_params={
                "d1": {"min": 1},
                "d2": {"min": 1},
                "d3": {"min": 1},
                "d4": {"min": 1},
            },
            input_shapes={"x": ("batch", "d1")},
        )
        assert result.universally_safe is True

    def test_mlp_with_activation(self):
        """MLP with relu — activation preserves shape."""
        result = verify_parametric(
            """
import torch.nn as nn
class MLPBN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
            arch_params={
                "d_in": {"min": 1},
                "d_hidden": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unsafe architecture families
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnsafeArchitectures:
    """Test architectures that have shape mismatches regardless of params."""

    def test_dimension_mismatch(self):
        """fc2 expects d_out but receives d_hidden (different params)."""
        result = verify_parametric(
            """
import torch.nn as nn
class Broken(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_out, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            arch_params={
                "d_in": {"min": 1},
                "d_hidden": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is False
        assert result.safety_constraints  # should discover needed constraints

    def test_concrete_mismatch_in_parametric(self):
        """One concrete value doesn't match — unsafe no matter what."""
        result = verify_parametric(
            """
import torch.nn as nn
class FixedBroken(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 100)
        self.fc2 = nn.Linear(200, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            arch_params={"d_in": {"min": 1}},
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Conditionally safe architectures
# ═══════════════════════════════════════════════════════════════════════════════


class TestConditionallySafe:
    """Architectures safe only under certain parameter constraints."""

    def test_safe_when_d_ff_equals_d_model(self):
        """Architecture safe only when d_ff == d_model (skip connection)."""
        # This model uses x + fc2(fc1(x)) so fc2 output must match x's
        # last dim (d_in). fc2 outputs d_out, so d_out must equal d_in.
        result = verify_parametric(
            """
import torch.nn as nn
class SkipMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return x + self.fc2(self.fc1(x))
""",
            arch_params={
                "d_in": {"min": 1},
                "d_hidden": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        # Not universally safe because d_out may differ from d_in
        assert result.universally_safe is False
        # Should discover constraint linking d_out and d_in
        constraint_exprs = [c.expression for c in result.safety_constraints]
        has_relevant = any(
            ("d_out" in e and "d_in" in e) or ("d_in" in e and "d_out" in e)
            for e in constraint_exprs
        )
        assert has_relevant

    def test_autoencoder_bottleneck(self):
        """Autoencoder: encoder and decoder must have compatible dims."""
        result = verify_parametric(
            """
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, d_in, d_bottleneck, d_out):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_bottleneck)
        self.decoder = nn.Linear(d_bottleneck, d_out)
    def forward(self, x):
        return self.decoder(self.encoder(x))
""",
            arch_params={
                "d_in": {"min": 1},
                "d_bottleneck": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        # Safe because encoder output feeds decoder input (same param)
        assert result.universally_safe is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multi-parameter families
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiParameter:
    """Test with multiple symbolic parameters."""

    def test_four_params(self):
        """Four independent parameters in a chain — safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class FourParam(nn.Module):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.l1 = nn.Linear(a, b)
        self.l2 = nn.Linear(b, c)
        self.l3 = nn.Linear(c, d)
    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))
""",
            arch_params={
                "a": {"min": 1},
                "b": {"min": 1},
                "c": {"min": 1},
                "d": {"min": 1},
            },
            input_shapes={"x": ("batch", "a")},
        )
        assert result.universally_safe is True

    def test_bounded_params(self):
        """Parameters with both min and max bounds — still safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class Bounded(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={
                "d_in": {"min": 1, "max": 1024},
                "d_out": {"min": 1, "max": 512},
            },
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Conv architectures with symbolic channels
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvArchitectures:
    """Test conv layers with symbolic channel counts."""

    def test_conv_chain_safe(self):
        """Conv2d chain with matching channels — safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_mid, 3, padding=1)
        self.conv2 = nn.Conv2d(c_mid, c_out, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
            arch_params={
                "c_in": {"min": 1},
                "c_mid": {"min": 1},
                "c_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "c_in", 32, 32)},
        )
        assert result.universally_safe is True

    def test_conv_channel_mismatch(self):
        """Conv2d with mismatched channels — unsafe."""
        result = verify_parametric(
            """
import torch.nn as nn
class BrokenConv(nn.Module):
    def __init__(self, c_in, c1, c2):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c2, 64, 3, padding=1)
    def forward(self, x):
        return self.conv2(self.conv1(x))
""",
            arch_params={
                "c_in": {"min": 1},
                "c1": {"min": 1},
                "c2": {"min": 1},
            },
            input_shapes={"x": ("batch", "c_in", 32, 32)},
        )
        assert result.universally_safe is False


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and special scenarios."""

    def test_no_arch_params(self):
        """No symbolic params — degenerates to standard verification."""
        result = verify_parametric(
            """
import torch.nn as nn
class Fixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={},
            input_shapes={"x": ("batch", 10)},
        )
        assert result.universally_safe is True

    def test_result_has_arch_params_used(self):
        """Result contains the arch_params that were used."""
        params = {"d_in": {"min": 1}, "d_out": {"min": 1}}
        result = verify_parametric(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params=params,
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.arch_params_used == params

    def test_no_module_raises(self):
        """Source without nn.Module → result with errors."""
        result = verify_parametric(
            "x = 1",
            arch_params={"d": {"min": 1}},
        )
        assert result.universally_safe is False
        assert result.verification_result is not None
        assert result.verification_result.errors

    def test_pretty_output(self):
        """ParametricResult.pretty() returns a string."""
        result = verify_parametric(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={"d": {"min": 1}},
            input_shapes={"x": ("batch", "d")},
        )
        assert isinstance(result.pretty(), str)
        assert "UNIVERSALLY SAFE" in result.pretty()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Graph extraction and symbolic param collection
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternals:
    """Test internal helper functions."""

    def test_extract_parametric_graph(self):
        """Symbolic params appear in layer definitions."""
        graph = _extract_parametric_graph(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={"d_in": {"min": 1}, "d_out": {"min": 1}},
        )
        assert "fc" in graph.layers
        fc = graph.layers["fc"]
        assert fc.in_features == "d_in"
        assert fc.out_features == "d_out"

    def test_collect_symbolic_params(self):
        """Collect symbolic param names from graph."""
        graph = _extract_parametric_graph(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.fc1 = nn.Linear(a, b)
        self.fc2 = nn.Linear(b, c)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            arch_params={"a": {}, "b": {}, "c": {}},
        )
        syms = _collect_symbolic_params(graph)
        assert "a" in syms
        assert "b" in syms
        assert "c" in syms

    def test_identity_param_safe(self):
        """Same param for both in and out — universally safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
            arch_params={"d": {"min": 1}},
            input_shapes={"x": ("batch", "d")},
        )
        assert result.universally_safe is True

    def test_max_bound_respected(self):
        """Max bound on params doesn't break verification."""
        result = verify_parametric(
            """
import torch.nn as nn
class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={"d": {"min": 1, "max": 256}},
            input_shapes={"x": ("batch", "d")},
        )
        assert result.universally_safe is True


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Constraint discovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstraintDiscovery:
    """Test safety constraint discovery."""

    def test_discovers_skip_constraint(self):
        """Discover that d_out must equal d_in for skip connection."""
        result = verify_parametric(
            """
import torch.nn as nn
class Skip(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        return x + self.fc2(self.fc1(x))
""",
            arch_params={
                "d_in": {"min": 1},
                "d_hidden": {"min": 1},
                "d_out": {"min": 1},
            },
            input_shapes={"x": ("batch", "d_in")},
            discover_constraints=True,
        )
        assert result.universally_safe is False
        assert len(result.safety_constraints) > 0

    def test_no_discovery_when_safe(self):
        """No constraints needed when already safe."""
        result = verify_parametric(
            """
import torch.nn as nn
class OK(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.fc(x)
""",
            arch_params={"d_in": {"min": 1}, "d_out": {"min": 1}},
            input_shapes={"x": ("batch", "d_in")},
        )
        assert result.universally_safe is True
        assert result.safety_constraints == []
