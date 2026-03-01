"""Tests for the Bounded Model Checking (BMC) baseline verifier."""

from __future__ import annotations

import pytest

from src.bmc_baseline import (
    BMCResult,
    BMCVerdict,
    verify_model_bmc,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures: source-code snippets
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_SAFE = """\
import torch.nn as nn

class SafeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

SIMPLE_BUGGY = """\
import torch.nn as nn

class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

TWO_LAYER_SAFE = """\
import torch.nn as nn

class TwoLayerNet(nn.Module):
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

MULTI_LAYER_SAFE = """\
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

MULTI_LAYER_BUGGY = """\
import torch.nn as nn

class DeepBuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(99, 16)
        self.fc4 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
"""

SYMBOLIC_DIM_SAFE = """\
import torch.nn as nn

class SymbolicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        return self.fc(x)
"""

CONV_MODEL_SAFE = """\
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
"""

DROPOUT_SAFE = """\
import torch.nn as nn

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""

EMBEDDING_SAFE = """\
import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x
"""

INVALID_SOURCE = "this is not valid python at all {"

EMPTY_MODULE = """\
import torch.nn as nn

class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBMCBaseline:
    """Tests for verify_model_bmc."""

    def test_simple_safe_model(self):
        """BMC should verify a simple safe model as SAFE."""
        result = verify_model_bmc(SIMPLE_SAFE, input_shapes={"x": ("batch", 10)})
        assert isinstance(result, BMCResult)
        assert result.verdict == BMCVerdict.SAFE
        assert result.safe is True
        assert result.time_ms > 0

    def test_simple_buggy_model(self):
        """BMC should detect a shape mismatch as UNSAFE."""
        result = verify_model_bmc(SIMPLE_BUGGY, input_shapes={"x": ("batch", 10)})
        assert isinstance(result, BMCResult)
        assert result.verdict == BMCVerdict.UNSAFE
        assert result.safe is False
        assert result.counterexample is not None

    def test_two_layer_safe(self):
        """BMC should verify a two-layer MLP as SAFE."""
        result = verify_model_bmc(TWO_LAYER_SAFE, input_shapes={"x": ("batch", 784)})
        assert result.verdict == BMCVerdict.SAFE

    def test_multi_layer_safe(self):
        """BMC should verify a 4-layer deep network as SAFE."""
        result = verify_model_bmc(MULTI_LAYER_SAFE, input_shapes={"x": ("batch", 100)})
        assert result.verdict == BMCVerdict.SAFE
        assert result.num_steps > 0

    def test_multi_layer_buggy(self):
        """BMC should detect a bug deep in a multi-layer network."""
        result = verify_model_bmc(MULTI_LAYER_BUGGY, input_shapes={"x": ("batch", 100)})
        assert result.verdict == BMCVerdict.UNSAFE

    def test_symbolic_dimensions(self):
        """BMC should handle symbolic (string) dimensions."""
        result = verify_model_bmc(
            SYMBOLIC_DIM_SAFE,
            input_shapes={"x": ("batch", "seq", 768)},
        )
        assert result.verdict == BMCVerdict.SAFE

    def test_conv_model(self):
        """BMC should handle Conv2d models."""
        result = verify_model_bmc(
            CONV_MODEL_SAFE,
            input_shapes={"x": ("batch", 3, 32, 32)},
        )
        assert isinstance(result, BMCResult)
        assert result.verdict in (BMCVerdict.SAFE, BMCVerdict.UNSAFE, BMCVerdict.UNKNOWN)
        assert result.time_ms > 0

    def test_dropout_model(self):
        """BMC should handle models with dropout layers."""
        result = verify_model_bmc(DROPOUT_SAFE, input_shapes={"x": ("batch", 10)})
        assert result.verdict == BMCVerdict.SAFE

    def test_timeout_handling(self):
        """BMC should respect the timeout parameter without crashing."""
        result = verify_model_bmc(
            SIMPLE_SAFE,
            input_shapes={"x": ("batch", 10)},
            timeout=1,
        )
        assert isinstance(result, BMCResult)
        assert result.verdict in (BMCVerdict.SAFE, BMCVerdict.UNKNOWN)

    def test_invalid_source(self):
        """BMC should return UNKNOWN for unparseable source."""
        result = verify_model_bmc(INVALID_SOURCE)
        assert result.verdict == BMCVerdict.UNKNOWN

    def test_empty_module(self):
        """BMC should handle an empty nn.Module (no layers)."""
        result = verify_model_bmc(EMPTY_MODULE, input_shapes={"x": ("batch", 10)})
        assert result.verdict == BMCVerdict.SAFE

    def test_result_summary(self):
        """BMCResult.summary() should return a readable string."""
        result = verify_model_bmc(SIMPLE_SAFE, input_shapes={"x": ("batch", 10)})
        s = result.summary()
        assert "BMC" in s
        assert result.verdict.name in s

    def test_no_input_shapes(self):
        """BMC should work even without explicit input shapes."""
        result = verify_model_bmc(SIMPLE_SAFE)
        assert isinstance(result, BMCResult)

    def test_embedding_model(self):
        """BMC should handle models with Embedding layers."""
        result = verify_model_bmc(
            EMBEDDING_SAFE,
            input_shapes={"x": ("batch", "seq")},
        )
        assert isinstance(result, BMCResult)
        assert result.verdict in (BMCVerdict.SAFE, BMCVerdict.UNSAFE, BMCVerdict.UNKNOWN)
