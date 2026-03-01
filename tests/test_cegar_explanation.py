"""
Tests for the CEGAR explanation generation module.

Covers:
  - Safe model explanation rendering
  - Unsafe model explanation with counterexample path
  - Multi-layer model explanations
  - Explanation from pre-built ShapeCEGARResult (unit-level)
  - explain_verification end-to-end API
"""

from __future__ import annotations

import pytest

from src.cegar_explanation import (
    VerificationExplanation,
    generate_explanation,
    explain_verification,
    _layer_description,
    _build_refinement_trace,
)
from src.shape_cegar import (
    ShapeCEGARResult,
    CEGARStatus,
    CEGARVerdict,
    ShapePredicate,
    PredicateKind,
    IterationRecord,
    InferredContract,
    run_shape_cegar,
)
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    VerificationResult,
    SafetyViolation,
    CounterexampleTrace,
    LayerDef,
    LayerKind,
    OpKind,
    extract_computation_graph,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test model source snippets
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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

SHAPE_MISMATCH_MODEL = """\
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
"""

THREE_LAYER_NET = """\
import torch.nn as nn

class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""

CONV_MODEL = """\
import torch.nn as nn

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Safe model explanation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeModelExplanation:
    """Test explanation generation for safe models."""

    def test_safe_simple_linear(self):
        """A safe single-layer model produces a SAFE explanation."""
        explanation = explain_verification(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        assert explanation.verdict == "SAFE"
        assert "SimpleModel" in explanation.model_name
        rendered = explanation.render()
        assert "Safety Explanation" in rendered or "SAFE" in rendered

    def test_safe_explanation_has_layer_info(self):
        """Safe explanation includes layer descriptions."""
        explanation = explain_verification(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", 10)},
        )
        rendered = explanation.render()
        # Should mention the fc layer
        assert "fc" in rendered or "Linear" in rendered


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unsafe model explanation
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnsafeModelExplanation:
    """Test explanation generation for unsafe models."""

    def test_unsafe_shape_mismatch(self):
        """A model with mismatched layers is flagged UNSAFE."""
        explanation = explain_verification(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
        )
        assert explanation.verdict == "UNSAFE"
        rendered = explanation.render()
        assert "Violation" in rendered or "UNSAFE" in rendered or "✗" in rendered

    def test_unsafe_has_counterexample_path(self):
        """Unsafe explanation includes a counterexample path."""
        explanation = explain_verification(
            SHAPE_MISMATCH_MODEL,
            input_shapes={"x": ("batch", 10)},
        )
        # Should have counterexample info (either in path or rendered output)
        rendered = explanation.render()
        assert explanation.verdict == "UNSAFE"
        # The explanation should contain step info or violation message
        has_step_info = (
            len(explanation.counterexample_path) > 0
            or "Step" in rendered
            or "✗" in rendered
        )
        assert has_step_info


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-layer model explanation
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiLayerExplanation:
    """Test explanations for models with multiple layers."""

    def test_three_layer_safe(self):
        """A three-layer model with matching shapes is explained as SAFE."""
        explanation = explain_verification(
            THREE_LAYER_NET,
            input_shapes={"x": ("batch", 768)},
        )
        assert explanation.verdict == "SAFE"
        rendered = explanation.render()
        # Should mention multiple layers
        assert "fc1" in rendered or "fc2" in rendered or "fc3" in rendered

    def test_two_layer_mlp_safe(self):
        """A two-layer MLP produces a safe explanation with layer info."""
        explanation = explain_verification(
            TWO_LAYER_MLP,
            input_shapes={"x": ("batch", 784)},
        )
        assert explanation.verdict == "SAFE"
        assert explanation.model_name == "MLP"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Unit-level: generate_explanation from pre-built result
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateExplanationUnit:
    """Test generate_explanation with manually constructed results."""

    def test_explanation_from_minimal_result(self):
        """generate_explanation works with a minimal ShapeCEGARResult."""
        result = ShapeCEGARResult(
            final_status=CEGARStatus.SAFE,
            iterations=1,
            discovered_predicates=[
                ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
            ],
            iteration_log=[
                IterationRecord(
                    iteration=0,
                    num_violations=0,
                    num_spurious=0,
                    num_real=0,
                    predicates_added=[
                        ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10),
                    ],
                ),
            ],
        )
        explanation = generate_explanation(result, model_name="TestModel")
        assert explanation.verdict == "SAFE"
        assert explanation.model_name == "TestModel"
        assert explanation.num_predicates == 1
        rendered = explanation.render()
        assert "TestModel" in rendered

    def test_explanation_to_dict_serializable(self):
        """Explanation can be serialized to a dictionary."""
        result = ShapeCEGARResult(
            final_status=CEGARStatus.SAFE,
            iterations=1,
        )
        explanation = generate_explanation(result, model_name="DictTest")
        d = explanation.to_dict()
        assert d["model_name"] == "DictTest"
        assert d["verdict"] == "SAFE"
        assert isinstance(d["layer_explanations"], list)
        assert isinstance(d["refinement_trace"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Refinement trace rendering
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefinementTrace:
    """Test the CEGAR refinement trace portion of explanations."""

    def test_trace_with_predicates(self):
        """Refinement trace shows discovered predicates."""
        result = ShapeCEGARResult(
            final_status=CEGARStatus.SAFE,
            iterations=2,
            discovered_predicates=[
                ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768),
                ShapePredicate(PredicateKind.DIM_EQ, "fc1_out", axis=-1, value=256),
            ],
            iteration_log=[
                IterationRecord(
                    iteration=0,
                    num_violations=1,
                    num_spurious=1,
                    num_real=0,
                    predicates_added=[
                        ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768),
                    ],
                ),
                IterationRecord(
                    iteration=1,
                    num_violations=0,
                    num_spurious=0,
                    num_real=0,
                    predicates_added=[],
                ),
            ],
        )
        explanation = generate_explanation(result, model_name="TraceTest")
        rendered = explanation.render()
        assert "Iteration 1" in rendered
        assert "768" in rendered
        assert "CEGAR Refinement Trace" in rendered

    def test_symbolic_input_explanation(self):
        """Symbolic inputs produce meaningful explanation with predicates."""
        explanation = explain_verification(
            SIMPLE_LINEAR,
            input_shapes={"x": ("batch", "features")},
        )
        rendered = explanation.render()
        # Should show refinement trace
        assert "Refinement Trace" in rendered or "Iteration" in rendered
