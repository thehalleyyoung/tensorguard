"""Tests for the high_confidence_only mode in verify_model.

Validates that:
  - Default behavior (high_confidence_only=False) is unchanged.
  - high_confidence_only=True filters out LOW/MEDIUM confidence violations.
  - HIGH confidence violations still surface in high-confidence mode.
  - Safe models remain safe regardless of the flag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import (
    Confidence,
    CounterexampleTrace,
    SafetyViolation,
    VerificationResult,
    verify_model,
)


# ── Helper: a known-safe model ──────────────────────────────────────────

SAFE_MODEL = """
import torch.nn as nn

class SafeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""

# ── Helper: a known-buggy model (shape mismatch) ────────────────────────

BUGGY_MODEL = """
import torch.nn as nn

class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(999, 5)   # mismatch: expects 999, gets 20

    def forward(self, x):
        return self.fc2(self.fc1(x))
"""


# ── 1. Default behaviour unchanged ──────────────────────────────────────

def test_default_returns_all_violations():
    """high_confidence_only=False (default) must not filter anything."""
    result = verify_model(BUGGY_MODEL, input_shapes={"x": ("batch", 10)})
    # Should detect the shape mismatch
    assert not result.safe, "Buggy model should be detected as unsafe"
    assert result.counterexample is not None


def test_default_flag_is_false():
    """Calling without the flag should behave identically to False."""
    r1 = verify_model(SAFE_MODEL, input_shapes={"x": ("batch", 10)})
    r2 = verify_model(
        SAFE_MODEL, input_shapes={"x": ("batch", 10)},
        high_confidence_only=False,
    )
    assert r1.safe == r2.safe


# ── 2. high_confidence_only=True filters LOW confidence ─────────────────

def test_high_confidence_filters_low():
    """Synthetically inject LOW violations and verify they are removed."""
    from src.model_checker import ComputationStep, TensorShape, OpKind

    low_violation = SafetyViolation(
        kind="shape_incompatible",
        step_index=0,
        step=ComputationStep(
            op=OpKind.LAYER_CALL, inputs=["a"], output="b",
        ),
        message="heuristic mismatch",
        confidence=Confidence.LOW,
        fp_category="abstract_imprecision",
    )
    cex = CounterexampleTrace(
        model_name="test",
        violations=[low_violation],
        failing_step=0,
    )
    result = VerificationResult(safe=False, counterexample=cex)

    filtered = result.filter_by_confidence(Confidence.HIGH)
    assert filtered.safe, "LOW-confidence violation should be filtered out"


# ── 3. HIGH violations still surface ────────────────────────────────────

def test_high_confidence_keeps_high():
    """HIGH-confidence violations must survive the filter."""
    from src.model_checker import ComputationStep, TensorShape, OpKind

    high_violation = SafetyViolation(
        kind="shape_incompatible",
        step_index=0,
        step=ComputationStep(
            op=OpKind.LAYER_CALL, inputs=["a"], output="b",
        ),
        message="proven mismatch",
        confidence=Confidence.HIGH,
    )
    cex = CounterexampleTrace(
        model_name="test",
        violations=[high_violation],
        failing_step=0,
    )
    result = VerificationResult(safe=False, counterexample=cex)

    filtered = result.filter_by_confidence(Confidence.HIGH)
    assert not filtered.safe, "HIGH-confidence violation should NOT be filtered"
    assert len(filtered.counterexample.violations) == 1


# ── 4. Safe model stays safe with the flag ──────────────────────────────

def test_safe_model_stays_safe():
    """A safe model must remain safe regardless of the flag."""
    result = verify_model(
        SAFE_MODEL, input_shapes={"x": ("batch", 10)},
        high_confidence_only=True,
    )
    assert result.safe, "Safe model must stay safe with high_confidence_only=True"


# ── 5. verify_model integration with high_confidence_only ───────────────

def test_verify_model_high_confidence_buggy():
    """Buggy model with high_confidence_only should still detect Z3-proven bugs."""
    result = verify_model(
        BUGGY_MODEL, input_shapes={"x": ("batch", 10)},
        high_confidence_only=True,
    )
    # The shape mismatch is Z3-proven (HIGH confidence), so it should still be caught
    assert not result.safe, "Z3-proven bug should survive high_confidence_only filter"
