"""Tests for confidence distribution analysis."""

import json
import math
import os
import sys

import pytest

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from experiments.run_confidence_distribution import (
    BENCHMARK_CONFIDENCES,
    CONFIDENCE_LEVELS,
    compute_confidence_distribution,
    compute_confidence_fractions,
    entropy_conditioned_calibration,
    explain_resolution_zero,
    max_entropy,
    normalized_entropy,
    shannon_entropy,
)


class TestConfidenceDistribution:
    """Tests for confidence distribution computation."""

    def test_distribution_counts_all_benchmarks(self):
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        total = sum(dist.values())
        assert total == len(BENCHMARK_CONFIDENCES)

    def test_distribution_has_all_levels(self):
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        for level in CONFIDENCE_LEVELS:
            assert level in dist

    def test_distribution_formal_dominates(self):
        """FORMAL should be the most common confidence level."""
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        assert dist["FORMAL"] >= dist.get("HIGH", 0)
        assert dist["FORMAL"] >= dist.get("MEDIUM", 0)
        assert dist["FORMAL"] >= dist.get("LOW", 0)
        assert dist["FORMAL"] >= dist.get("NONE", 0)

    def test_fractions_sum_to_one(self):
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        fracs = compute_confidence_fractions(dist)
        total = sum(fracs.values())
        assert abs(total - 1.0) < 1e-10


class TestShannonEntropy:
    """Tests for Shannon entropy computation."""

    def test_uniform_distribution_max_entropy(self):
        fracs = {level: 1.0 / len(CONFIDENCE_LEVELS) for level in CONFIDENCE_LEVELS}
        h = shannon_entropy(fracs)
        h_max = max_entropy(len(CONFIDENCE_LEVELS))
        assert abs(h - h_max) < 1e-10

    def test_degenerate_distribution_zero_entropy(self):
        fracs = {"FORMAL": 1.0, "HIGH": 0.0, "MEDIUM": 0.0, "LOW": 0.0, "NONE": 0.0}
        h = shannon_entropy(fracs)
        assert abs(h) < 1e-10

    def test_entropy_nonnegative(self):
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        fracs = compute_confidence_fractions(dist)
        h = shannon_entropy(fracs)
        assert h >= 0.0

    def test_normalized_entropy_in_range(self):
        dist = compute_confidence_distribution(BENCHMARK_CONFIDENCES)
        fracs = compute_confidence_fractions(dist)
        h = shannon_entropy(fracs)
        h_norm = normalized_entropy(h, len(CONFIDENCE_LEVELS))
        assert 0.0 <= h_norm <= 1.0


class TestEntropyConditionedCalibration:
    """Tests for entropy-conditioned calibration."""

    def test_nontrivial_subset_not_empty(self):
        result = entropy_conditioned_calibration(
            BENCHMARK_CONFIDENCES, ["MEDIUM", "LOW", "NONE"], n_bins=3
        )
        assert result["subset_size"] > 0

    def test_empty_subset_returns_zero(self):
        result = entropy_conditioned_calibration(
            BENCHMARK_CONFIDENCES, ["NONE"], n_bins=3
        )
        # If no benchmarks have NONE confidence, should be empty
        if result["subset_size"] == 0:
            assert result["brier_score"] == 0.0

    def test_full_calibration_includes_all(self):
        result = entropy_conditioned_calibration(
            BENCHMARK_CONFIDENCES, CONFIDENCE_LEVELS, n_bins=5
        )
        assert result["subset_size"] == len(BENCHMARK_CONFIDENCES)

    def test_brier_score_in_range(self):
        result = entropy_conditioned_calibration(
            BENCHMARK_CONFIDENCES, CONFIDENCE_LEVELS, n_bins=5
        )
        assert 0.0 <= result["brier_score"] <= 1.0


class TestResolutionExplanation:
    """Tests for the RES=0 explanation."""

    def test_explanation_has_required_fields(self):
        explanation = explain_resolution_zero()
        assert "phenomenon" in explanation
        assert "root_cause" in explanation
        assert "explanation" in explanation
        assert "mathematical_detail" in explanation
        assert "implications" in explanation

    def test_root_cause_is_deterministic_smt(self):
        explanation = explain_resolution_zero()
        assert "deterministic" in explanation["root_cause"]
        assert "smt" in explanation["root_cause"].lower()
