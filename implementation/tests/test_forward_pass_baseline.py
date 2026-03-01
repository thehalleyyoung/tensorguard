"""Tests for the forward-pass baseline experiment methodology."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_forward_pass_baseline import (
    ALL_BENCHMARKS,
    ApproachResult,
    BenchmarkModel,
    classify,
    compute_metrics,
    run_tensorguard,
    run_forward_pass,
    run_forward_pass_multi,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Test 1: Benchmark suite has at least 20 models with correct balance
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    def test_minimum_model_count(self):
        assert len(ALL_BENCHMARKS) >= 20, (
            f"Expected >=20 benchmark models, got {len(ALL_BENCHMARKS)}"
        )

    def test_has_buggy_and_safe_models(self):
        buggy = [m for m in ALL_BENCHMARKS if m.has_bug]
        safe = [m for m in ALL_BENCHMARKS if not m.has_bug]
        assert len(buggy) >= 8, f"Expected >=8 buggy models, got {len(buggy)}"
        assert len(safe) >= 6, f"Expected >=6 safe models, got {len(safe)}"


# ---------------------------------------------------------------------------
# Test 2: Classification logic (TP/FP/TN/FN)
# ---------------------------------------------------------------------------

class TestClassification:
    def test_true_positive(self):
        assert classify(has_bug=True, detected=True) == "TP"

    def test_false_negative(self):
        assert classify(has_bug=True, detected=False) == "FN"

    def test_false_positive(self):
        assert classify(has_bug=False, detected=True) == "FP"

    def test_true_negative(self):
        assert classify(has_bug=False, detected=False) == "TN"


# ---------------------------------------------------------------------------
# Test 3: Metrics computation
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect_classifier(self):
        classes = ["TP", "TP", "TN", "TN"]
        m = compute_metrics(classes)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_false_negatives(self):
        classes = ["FN", "FN", "TN", "TN"]
        m = compute_metrics(classes)
        assert m["recall"] == 0.0
        assert m["FN"] == 2

    def test_mixed(self):
        classes = ["TP", "FP", "FN", "TN"]
        m = compute_metrics(classes)
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5

    def test_empty(self):
        m = compute_metrics([])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0


# ---------------------------------------------------------------------------
# Test 4: TensorGuard produces parametric results
# ---------------------------------------------------------------------------

class TestTensorGuardParametric:
    def test_tensorguard_result_is_parametric(self):
        """TensorGuard results should always be marked as parametric."""
        model = ALL_BENCHMARKS[0]
        result = run_tensorguard(model)
        assert isinstance(result, ApproachResult)
        assert result.parametric is True, "TensorGuard should provide parametric guarantees"

    def test_tensorguard_detects_known_bug(self):
        """TensorGuard should detect the linear dim mismatch bug."""
        buggy = [m for m in ALL_BENCHMARKS if m.name == "linear_dim_mismatch"][0]
        result = run_tensorguard(buggy)
        assert result.detected_bug is True, "TensorGuard should detect linear dim mismatch"


# ---------------------------------------------------------------------------
# Test 5: Forward pass correctly runs models
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestForwardPassExecution:
    def test_forward_pass_detects_obvious_bug(self):
        """Forward pass should catch a basic dimension mismatch."""
        buggy = [m for m in ALL_BENCHMARKS if m.name == "linear_dim_mismatch"][0]
        result = run_forward_pass(buggy)
        assert isinstance(result, ApproachResult)
        assert result.parametric is False, "Forward pass is not parametric"
        assert result.detected_bug is True, "Forward pass should catch linear dim mismatch"

    def test_forward_pass_safe_model_no_false_positive(self):
        """Forward pass should not flag a correct model."""
        safe = [m for m in ALL_BENCHMARKS if m.name == "simple_mlp_safe"][0]
        result = run_forward_pass(safe)
        assert result.detected_bug is False, "Forward pass should not flag safe model"

    def test_multi_input_runs_multiple_shapes(self):
        """Multi-input forward pass should test all 5 shapes."""
        model = ALL_BENCHMARKS[0]
        result = run_forward_pass_multi(model)
        assert isinstance(result, ApproachResult)
        assert result.time_ms > 0
