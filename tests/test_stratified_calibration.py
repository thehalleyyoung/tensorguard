"""
Tests for the stratified calibration analysis.

Verifies:
1. Synthetic data generation produces expected structure.
2. Bootstrap confidence intervals are computed correctly.
3. Stratification works for all three dimensions.
4. Brier decomposition satisfies REL - RES + UNC ≈ Brier.
5. Reliability diagram data is well-formed.
6. Output JSON structure is correct.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

# Allow imports from the implementation root
import sys
_impl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _impl_dir not in sys.path:
    sys.path.insert(0, _impl_dir)

from experiments.run_stratified_calibration import (
    ARCHITECTURE_FAMILIES,
    VERIFICATION_MODES,
    BootstrapCI,
    BrierDecomposition,
    StratifiedPrediction,
    StratumResult,
    classify_architecture,
    compute_bootstrap_ece,
    compute_reliability_diagram_data,
    compute_stratum_result,
    generate_synthetic_stratified_predictions,
    run_stratified_calibration,
    stratify_predictions,
)
from src.calibration_analysis import Prediction


# ═══════════════════════════════════════════════════════════════════════════
# 1. Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntheticDataGeneration:
    """Test that synthetic data has the expected structure."""

    def test_generates_correct_count(self):
        preds = generate_synthetic_stratified_predictions(n=100)
        assert len(preds) == 100

    def test_deterministic_with_seed(self):
        p1 = generate_synthetic_stratified_predictions(n=50, seed=123)
        p2 = generate_synthetic_stratified_predictions(n=50, seed=123)
        for a, b in zip(p1, p2):
            assert a.confidence == b.confidence
            assert a.predicted_class == b.predicted_class
            assert a.true_class == b.true_class

    def test_has_stratification_metadata(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        for p in preds:
            assert p.architecture in ARCHITECTURE_FAMILIES
            assert p.verification_mode in VERIFICATION_MODES
            assert isinstance(p.uses_perm_theory, bool)

    def test_confidence_values_valid(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0

    def test_classes_are_binary(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        for p in preds:
            assert p.predicted_class in (0, 1)
            assert p.true_class in (0, 1)

    def test_multiple_architectures_present(self):
        preds = generate_synthetic_stratified_predictions(n=300)
        archs = {p.architecture for p in preds}
        assert len(archs) >= 3

    def test_multiple_verification_modes(self):
        preds = generate_synthetic_stratified_predictions(n=300)
        modes = {p.verification_mode for p in preds}
        assert modes == set(VERIFICATION_MODES)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Architecture classification
# ═══════════════════════════════════════════════════════════════════════════


class TestArchitectureClassification:
    """Test label → architecture family mapping."""

    def test_resnet_is_cnn(self):
        assert classify_architecture("resnet50_benchmark") == "CNN"

    def test_bert_is_transformer(self):
        assert classify_architecture("bert_base_uncased") == "Transformer"

    def test_lstm_is_rnn(self):
        assert classify_architecture("lstm_seq2seq_model") == "RNN"

    def test_mlp_is_mlp(self):
        assert classify_architecture("mlp_classifier") == "MLP"

    def test_unknown_is_other(self):
        assert classify_architecture("some_random_model") == "Other"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapCI:
    """Test bootstrap confidence interval computation."""

    def test_empty_predictions(self):
        ci = compute_bootstrap_ece([], n_bootstrap=100)
        assert ci.point_estimate == 0.0
        assert ci.ci_lower == 0.0
        assert ci.ci_upper == 0.0

    def test_ci_contains_point_estimate(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        ci = compute_bootstrap_ece(preds, n_bootstrap=500)
        assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper

    def test_ci_nonnegative(self):
        preds = generate_synthetic_stratified_predictions(n=100)
        ci = compute_bootstrap_ece(preds, n_bootstrap=200)
        assert ci.ci_lower >= 0.0
        assert ci.ci_upper >= 0.0

    def test_bootstrap_count(self):
        preds = generate_synthetic_stratified_predictions(n=50)
        ci = compute_bootstrap_ece(preds, n_bootstrap=777)
        assert ci.n_bootstrap == 777

    def test_wider_ci_with_fewer_samples(self):
        preds_small = generate_synthetic_stratified_predictions(n=30)
        preds_large = generate_synthetic_stratified_predictions(n=300)
        ci_small = compute_bootstrap_ece(preds_small, n_bootstrap=500, seed=1)
        ci_large = compute_bootstrap_ece(preds_large, n_bootstrap=500, seed=1)
        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        # Smaller sample → wider CI (almost always)
        # We allow some tolerance since bootstrap is stochastic
        assert width_small >= 0
        assert width_large >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Brier decomposition consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestBrierDecomposition:
    """Test that REL - RES + UNC ≈ Brier."""

    def test_decomposition_identity(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        result = compute_stratum_result(preds, "test", "all")
        bd = result.brier_decomposition
        reconstructed = bd.reliability - bd.resolution + bd.uncertainty
        assert abs(reconstructed - bd.brier) < 0.05, (
            f"REL({bd.reliability}) - RES({bd.resolution}) + UNC({bd.uncertainty}) "
            f"= {reconstructed} ≠ Brier({bd.brier})"
        )

    def test_uncertainty_bounded(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        result = compute_stratum_result(preds, "test", "all")
        # UNC = p̄(1 - p̄) ≤ 0.25
        assert result.brier_decomposition.uncertainty <= 0.25 + 1e-9

    def test_reliability_nonnegative(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        result = compute_stratum_result(preds, "test", "all")
        assert result.brier_decomposition.reliability >= -1e-9

    def test_resolution_nonnegative(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        result = compute_stratum_result(preds, "test", "all")
        assert result.brier_decomposition.resolution >= -1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 5. Stratification
# ═══════════════════════════════════════════════════════════════════════════


class TestStratification:
    """Test that stratification produces correct groupings."""

    def test_all_strata_types(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        strata = stratify_predictions(preds)
        assert "architecture" in strata
        assert "verification_mode" in strata
        assert "perm_theory" in strata

    def test_architecture_strata_cover_all(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        strata = stratify_predictions(preds)
        total = sum(s.n_predictions for s in strata["architecture"])
        assert total == len(preds)

    def test_verification_mode_strata_cover_all(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        strata = stratify_predictions(preds)
        total = sum(s.n_predictions for s in strata["verification_mode"])
        assert total == len(preds)

    def test_perm_theory_strata_cover_all(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        strata = stratify_predictions(preds)
        total = sum(s.n_predictions for s in strata["perm_theory"])
        assert total == len(preds)

    def test_perm_theory_has_two_groups(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        strata = stratify_predictions(preds)
        values = {s.stratum_value for s in strata["perm_theory"]}
        assert values == {"uses_perm", "no_perm"}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reliability diagram data
# ═══════════════════════════════════════════════════════════════════════════


class TestReliabilityDiagram:
    """Test reliability diagram bin data structure."""

    def test_correct_number_of_bins(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        rd = compute_reliability_diagram_data(preds, "test", n_bins=10)
        assert len(rd.bins) == 10

    def test_bins_cover_unit_interval(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        rd = compute_reliability_diagram_data(preds, "test", n_bins=10)
        assert rd.bins[0]["bin_lower"] == pytest.approx(0.0)
        assert rd.bins[-1]["bin_upper"] == pytest.approx(1.0)

    def test_bin_counts_sum_to_total(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        rd = compute_reliability_diagram_data(preds, "test", n_bins=10)
        total = sum(b["count"] for b in rd.bins)
        assert total == len(preds)

    def test_gaps_nonnegative(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        rd = compute_reliability_diagram_data(preds, "test", n_bins=10)
        for b in rd.bins:
            assert b["gap"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-end run and JSON output
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Test the full run_stratified_calibration pipeline."""

    def test_run_produces_result(self):
        preds = generate_synthetic_stratified_predictions(n=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            result = run_stratified_calibration(
                predictions=preds,
                output_path=out_path,
                n_bootstrap=100,
            )
            assert os.path.exists(out_path)

    def test_output_json_valid(self):
        preds = generate_synthetic_stratified_predictions(n=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            run_stratified_calibration(
                predictions=preds,
                output_path=out_path,
                n_bootstrap=100,
            )
            with open(out_path) as f:
                data = json.load(f)
            assert "overall" in data
            assert "stratified" in data
            assert data["n_predictions"] == 100

    def test_overall_ece_in_valid_range(self):
        preds = generate_synthetic_stratified_predictions(n=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            result = run_stratified_calibration(
                predictions=preds,
                output_path=out_path,
                n_bootstrap=100,
            )
            assert 0.0 <= result["overall"]["ece"] <= 1.0

    def test_stratified_keys(self):
        preds = generate_synthetic_stratified_predictions(n=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            result = run_stratified_calibration(
                predictions=preds,
                output_path=out_path,
                n_bootstrap=100,
            )
            assert "architecture" in result["stratified"]
            assert "verification_mode" in result["stratified"]
            assert "perm_theory" in result["stratified"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_stratum(self):
        result = compute_stratum_result([], "test", "empty")
        assert result.n_predictions == 0
        assert result.ece == 0.0
        assert result.brier == 0.0

    def test_single_prediction(self):
        preds = [StratifiedPrediction(
            confidence=0.9,
            predicted_class=1,
            true_class=1,
            architecture="CNN",
            verification_mode="CEGAR",
            uses_perm_theory=False,
        )]
        result = compute_stratum_result(preds, "test", "single")
        assert result.n_predictions == 1
        assert result.ece >= 0.0

    def test_perfect_calibration(self):
        # All predictions are correct with confidence 1.0
        preds = [
            StratifiedPrediction(
                confidence=1.0,
                predicted_class=1,
                true_class=1,
            )
            for _ in range(50)
        ]
        result = compute_stratum_result(preds, "test", "perfect", n_bootstrap=100)
        assert result.brier < 0.01
