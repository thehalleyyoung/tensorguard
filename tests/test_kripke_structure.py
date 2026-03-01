"""Tests for the formal Kripke structure semantics in the model checker."""

import pytest
from src.model_checker import (
    extract_computation_graph,
    extract_kripke_structure,
    verify_model,
    KripkeState,
    KripkeTransition,
    KripkeStructure,
    Device,
    Phase,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Test source snippets
# ═══════════════════════════════════════════════════════════════════════════════

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

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
"""

CONV_MODEL = """\
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)

    def forward(self, x):
        return self.conv(x)
"""

SHAPE_MISMATCH = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(999, 5)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
"""

DEEP_MODEL_TEMPLATE = """\
import torch.nn as nn

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
{layers}

    def forward(self, x):
{forward}
"""


def _make_deep_model(depth: int) -> str:
    layers = "\n".join(
        f"        self.fc{i} = nn.Linear(64, 64)" for i in range(depth)
    )
    forward_lines = []
    for i in range(depth):
        forward_lines.append(f"        x = self.fc{i}(x)")
    forward_lines.append("        return x")
    forward = "\n".join(forward_lines)
    return DEEP_MODEL_TEMPLATE.format(layers=layers, forward=forward)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Kripke structure extraction from simple models
# ═══════════════════════════════════════════════════════════════════════════════

class TestKripkeExtraction:
    """Test Kripke structure extraction from simple models."""

    def test_linear_model_extraction(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert isinstance(ks, KripkeStructure)
        assert ks.num_states > 0
        assert ks.initial_state_idx == 0
        assert ks.initial_state.layer_name == "input"

    def test_conv2d_model_extraction(self):
        graph = extract_computation_graph(CONV_MODEL)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 3, 32, 32)}
        )
        assert isinstance(ks, KripkeStructure)
        assert ks.num_states > 0

    def test_two_layer_mlp_extraction(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert isinstance(ks, KripkeStructure)
        # 2 layers → at least 3 states (initial + 2 steps)
        assert ks.num_states >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Safe models have shape_safe at all states
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeLabeling:
    """Test that safe models have shape_safe at all states."""

    def test_safe_linear_all_states_labeled(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert ks.is_safe(), "Safe model should have shape_safe at all states"

    def test_safe_two_layer_mlp_all_states_labeled(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert ks.is_safe(), "Safe MLP should have shape_safe at all states"

    def test_safe_model_no_violation_trace(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        trace = ks.get_violation_trace()
        assert trace is None, "Safe model should have no violation trace"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Buggy models have violation traces
# ═══════════════════════════════════════════════════════════════════════════════

class TestViolationTraces:
    """Test that buggy models produce violation traces."""

    def test_shape_mismatch_not_safe(self):
        graph = extract_computation_graph(SHAPE_MISMATCH)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert not ks.is_safe(), "Model with shape mismatch should be unsafe"

    def test_shape_mismatch_has_violation_trace(self):
        graph = extract_computation_graph(SHAPE_MISMATCH)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        trace = ks.get_violation_trace()
        assert trace is not None, "Buggy model should have a violation trace"
        assert len(trace) > 0, "Violation trace should be non-empty"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: State count = layers + 1
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateCounts:
    """Test that state count equals computation steps + 1."""

    def test_single_layer_state_count(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        # states = initial + num_steps
        assert ks.num_states == graph.num_steps + 1

    def test_two_layer_state_count(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert ks.num_states == graph.num_steps + 1

    def test_deep_model_state_count(self):
        source = _make_deep_model(10)
        graph = extract_computation_graph(source)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 64)}
        )
        assert ks.num_states == graph.num_steps + 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Transition count matches DAG edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransitionCounts:
    """Test that transition count matches computation step count."""

    def test_single_layer_transitions(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert ks.num_transitions == graph.num_steps

    def test_two_layer_transitions(self):
        graph = extract_computation_graph(TWO_LAYER_MLP)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert ks.num_transitions == graph.num_steps

    def test_deep_model_transitions(self):
        source = _make_deep_model(10)
        graph = extract_computation_graph(source)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 64)}
        )
        assert ks.num_transitions == graph.num_steps


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Serialization and integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestKripkeSerialization:
    """Test Kripke structure serialization and verify_model integration."""

    def test_to_dict_structure(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        d = ks.to_dict()
        assert "num_states" in d
        assert "num_transitions" in d
        assert "atomic_propositions" in d
        assert "states" in d
        assert "transitions" in d
        assert d["state_space_finite"] is True
        assert d["initial_state_idx"] == 0

    def test_verify_model_return_kripke(self):
        result = verify_model(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
            return_kripke=True,
        )
        assert result.kripke_structure is not None
        assert isinstance(result.kripke_structure, KripkeStructure)

    def test_verify_model_no_kripke_by_default(self):
        result = verify_model(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert result.kripke_structure is None

    def test_atomic_propositions(self):
        graph = extract_computation_graph(SIMPLE_LINEAR)
        ks = extract_kripke_structure(
            graph, input_shapes={"x": ("batch", 10)}
        )
        assert "shape_safe" in ks.atomic_propositions
        assert "device_consistent" in ks.atomic_propositions
        assert "gradient_valid" in ks.atomic_propositions
        assert "phase_correct" in ks.atomic_propositions

    def test_kripke_state_as_tuple(self):
        state = KripkeState(
            step_index=0,
            shape_vars={"x": ["batch", "10"]},
            device_vars={"x": "cpu"},
        )
        t = state.as_tuple()
        assert isinstance(t, tuple)
