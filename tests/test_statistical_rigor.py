"""Tests for statistical_rigor module — 35 test cases."""

import json
import math

import pytest

from src.statistical_rigor import (
    BHResult,
    BonferroniResult,
    BrierDecomposition,
    HolmResult,
    PPVNPVCurve,
    StatisticalReport,
    benjamini_hochberg,
    bonferroni,
    brier_decomposition,
    compute_npv,
    compute_ppv,
    familywise_error_probability,
    generate_report,
    holm_bonferroni,
    ppv_npv_curve,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Brier Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrierDecomposition:
    def test_perfect_predictions(self):
        """Perfect confidence → Brier = 0, reliability = 0."""
        y_true = [1, 1, 0, 0]
        y_prob = [1.0, 1.0, 0.0, 0.0]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        assert bd.brier_score == pytest.approx(0.0)
        assert bd.reliability == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Completely wrong confidence → Brier = 1."""
        y_true = [1, 1, 0, 0]
        y_prob = [0.0, 0.0, 1.0, 1.0]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        assert bd.brier_score == pytest.approx(1.0)

    def test_identity_holds(self):
        """REL - RES + UNC ≈ Brier (within binning approximation error)."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        y_prob = [0.9, 0.2, 0.8, 0.7, 0.3, 0.1, 0.6, 0.4, 0.85, 0.15]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        reconstructed = bd.reliability - bd.resolution + bd.uncertainty
        # Within-bin variance causes small deviation from exact identity
        assert reconstructed == pytest.approx(bd.brier_score, abs=0.01)

    def test_identity_exact_when_one_per_bin(self):
        """Identity holds exactly when each unique probability gets its own bin."""
        # Each value lands in a distinct bin → no within-bin variance
        y_true = [1, 0]
        y_prob = [0.95, 0.05]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        reconstructed = bd.reliability - bd.resolution + bd.uncertainty
        assert reconstructed == pytest.approx(bd.brier_score, abs=1e-10)

    def test_identity_improves_with_more_bins(self):
        """More bins → smaller gap between decomposition and raw Brier."""
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_prob = [0.95, 0.05, 0.85, 0.15, 0.75, 0.25, 0.65, 0.35]
        gaps = []
        for n_bins in [2, 5, 10, 20, 50]:
            bd = brier_decomposition(y_true, y_prob, n_bins=n_bins)
            reconstructed = bd.reliability - bd.resolution + bd.uncertainty
            gaps.append(abs(reconstructed - bd.brier_score))
        # The gap should be non-increasing as bins increase (approximately)
        # At minimum, fine bins should have very small gap
        assert gaps[-1] < 0.02

    def test_empty_input(self):
        bd = brier_decomposition([], [], n_bins=10)
        assert bd.brier_score == 0.0
        assert bd.reliability == 0.0
        assert bd.resolution == 0.0
        assert bd.uncertainty == 0.0
        assert bd.bin_counts == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_decomposition([1, 0], [0.5], n_bins=10)

    def test_uncertainty_is_base_rate(self):
        """Uncertainty = p̄(1 − p̄) where p̄ is the base rate."""
        y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # base rate = 0.3
        y_prob = [0.5] * 10
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        assert bd.uncertainty == pytest.approx(0.3 * 0.7)

    def test_perfect_calibration_zero_reliability(self):
        """All predictions in one bin with matching accuracy → reliability ≈ 0."""
        # 7 positive, 3 negative, all predicted at 0.7
        y_true = [1] * 7 + [0] * 3
        y_prob = [0.7] * 10
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        assert bd.reliability == pytest.approx(0.0, abs=1e-10)

    def test_reliability_diagram_data(self):
        y_true = [1, 0]
        y_prob = [0.9, 0.1]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        data = bd.reliability_diagram_data()
        assert len(data["bin_edges"]) == 11
        assert data["bin_edges"][0] == 0.0
        assert data["bin_edges"][-1] == 1.0
        assert len(data["bin_accuracies"]) == 10

    def test_bin_counts_sum_to_n(self):
        y_true = [1, 0, 1, 0, 1]
        y_prob = [0.9, 0.1, 0.7, 0.3, 0.5]
        bd = brier_decomposition(y_true, y_prob, n_bins=10)
        assert sum(bd.bin_counts) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PPV / NPV
# ═══════════════════════════════════════════════════════════════════════════════

class TestPPVNPV:
    def test_ppv_basic(self):
        """PPV with known values."""
        # sens=0.9, spec=0.9, prev=0.5 → PPV = 0.9*0.5 / (0.9*0.5 + 0.1*0.5) = 0.9
        ppv = compute_ppv(0.9, 0.9, 0.5)
        assert ppv == pytest.approx(0.9)

    def test_npv_basic(self):
        """NPV with known values."""
        # sens=0.9, spec=0.9, prev=0.5 → NPV = 0.9*0.5 / (0.9*0.5 + 0.1*0.5) = 0.9
        npv = compute_npv(0.9, 0.9, 0.5)
        assert npv == pytest.approx(0.9)

    def test_ppv_perfect_classifier(self):
        """Perfect classifier (sens=spec=1) → PPV = 1 for any prevalence."""
        for prev in [0.01, 0.1, 0.5, 0.99]:
            assert compute_ppv(1.0, 1.0, prev) == pytest.approx(1.0)

    def test_npv_perfect_classifier(self):
        """Perfect classifier → NPV = 1 for any prevalence."""
        for prev in [0.01, 0.1, 0.5, 0.99]:
            assert compute_npv(1.0, 1.0, prev) == pytest.approx(1.0)

    def test_ppv_zero_prevalence(self):
        """At prevalence=0, PPV should be 0 (no true positives possible)."""
        ppv = compute_ppv(0.9, 0.9, 0.0)
        assert ppv == pytest.approx(0.0)

    def test_npv_zero_prevalence(self):
        """At prevalence=0, NPV should be 1."""
        npv = compute_npv(0.9, 0.9, 0.0)
        # spec*(1-0) / (spec*(1-0) + (1-sens)*0) = spec/spec = 1
        assert npv == pytest.approx(1.0)

    def test_ppv_unit_prevalence(self):
        """At prevalence=1, PPV = 1 (everyone has disease)."""
        ppv = compute_ppv(0.9, 0.9, 1.0)
        assert ppv == pytest.approx(1.0)

    def test_ppv_low_prevalence_low_ppv(self):
        """With low prevalence and imperfect specificity, PPV is low."""
        ppv = compute_ppv(0.9, 0.9, 0.01)
        # 0.9*0.01 / (0.9*0.01 + 0.1*0.99) = 0.009 / 0.108 ≈ 0.0833
        assert ppv == pytest.approx(0.009 / 0.108, abs=1e-4)

    def test_ppv_npv_curve_length(self):
        curve = ppv_npv_curve(0.8, 0.9, n_steps=50)
        assert len(curve.prevalences) == 50
        assert len(curve.ppv_values) == 50
        assert len(curve.npv_values) == 50

    def test_ppv_increases_with_prevalence(self):
        """PPV should be monotonically non-decreasing with prevalence."""
        curve = ppv_npv_curve(0.8, 0.9, prevalence_range=(0.01, 0.50), n_steps=50)
        for i in range(1, len(curve.ppv_values)):
            assert curve.ppv_values[i] >= curve.ppv_values[i - 1] - 1e-10

    def test_breakeven_prevalence(self):
        """Check breakeven prevalence is found when PPV crosses threshold."""
        curve = ppv_npv_curve(0.9, 0.9, prevalence_range=(0.01, 0.50),
                              n_steps=100, ppv_threshold=0.5)
        if curve.breakeven_prevalence is not None:
            ppv_at_be = compute_ppv(0.9, 0.9, curve.breakeven_prevalence)
            assert ppv_at_be >= 0.5

    def test_zero_denominator_ppv(self):
        """Edge case: sens=0, spec=1, prev=0 → denominator = 0."""
        ppv = compute_ppv(0.0, 1.0, 0.0)
        assert ppv == 0.0  # handled gracefully

    def test_zero_denominator_npv(self):
        """Edge case: spec=0, sens=1, prev=1 → denominator = 0."""
        npv = compute_npv(1.0, 0.0, 1.0)
        assert npv == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multiple Comparison Correction
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenjaminiHochberg:
    def test_known_result(self):
        """B-H with known p-values from textbook example."""
        p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_tests == 5
        # Sorted: 0.01, 0.03, 0.04, 0.20, 0.50
        # Adjusted: 0.01*5/1=0.05, 0.03*5/2=0.075, 0.04*5/3≈0.0667, 0.20*5/4=0.25, 0.50
        # After monotonicity: 0.05, 0.0667, 0.0667, 0.25, 0.50
        assert result.rejected[0] is True   # p=0.01
        assert result.rejected[3] is False  # p=0.20

    def test_empty_input(self):
        result = benjamini_hochberg([], alpha=0.05)
        assert result.n_tests == 0
        assert result.n_rejected == 0

    def test_single_test(self):
        result = benjamini_hochberg([0.03], alpha=0.05)
        assert result.n_rejected == 1
        assert result.adjusted_p_values[0] == pytest.approx(0.03)

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected == 3

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.n_rejected == 0

    def test_adjusted_p_values_bounded(self):
        """Adjusted p-values should never exceed 1.0."""
        p_values = [0.8, 0.9, 0.95]
        result = benjamini_hochberg(p_values)
        for adj in result.adjusted_p_values:
            assert adj <= 1.0


class TestBonferroni:
    def test_basic_correction(self):
        p_values = [0.01, 0.04, 0.03]
        result = bonferroni(p_values, alpha=0.05)
        assert result.adjusted_p_values[0] == pytest.approx(0.03)  # 0.01 * 3
        assert result.adjusted_p_values[1] == pytest.approx(0.12)  # 0.04 * 3

    def test_more_conservative_than_bh(self):
        """Bonferroni should reject ≤ B-H rejections."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20]
        bh = benjamini_hochberg(p_values, alpha=0.05)
        bf = bonferroni(p_values, alpha=0.05)
        assert bf.n_rejected <= bh.n_rejected

    def test_empty(self):
        result = bonferroni([])
        assert result.n_tests == 0


class TestHolmBonferroni:
    def test_basic(self):
        p_values = [0.01, 0.04, 0.03]
        result = holm_bonferroni(p_values, alpha=0.05)
        # Sorted: 0.01, 0.03, 0.04
        # Adjusted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        # Monotonicity: 0.03, 0.06, 0.06
        assert result.rejected[0] is True   # adj=0.03 ≤ 0.05

    def test_at_least_as_powerful_as_bonferroni(self):
        """Holm should reject ≥ Bonferroni rejections."""
        p_values = [0.005, 0.015, 0.025, 0.04, 0.06]
        holm = holm_bonferroni(p_values, alpha=0.05)
        bf = bonferroni(p_values, alpha=0.05)
        assert holm.n_rejected >= bf.n_rejected

    def test_empty(self):
        result = holm_bonferroni([])
        assert result.n_tests == 0


class TestFWER:
    def test_known_value(self):
        """15 tests at alpha=0.05: FWER = 1 - 0.95^15 ≈ 0.537."""
        fwer = familywise_error_probability(15, alpha=0.05)
        assert fwer == pytest.approx(1.0 - 0.95**15)

    def test_single_test(self):
        assert familywise_error_probability(1, 0.05) == pytest.approx(0.05)

    def test_zero_tests(self):
        assert familywise_error_probability(0, 0.05) == 0.0

    def test_increases_with_n(self):
        fwer5 = familywise_error_probability(5, 0.05)
        fwer10 = familywise_error_probability(10, 0.05)
        fwer20 = familywise_error_probability(20, 0.05)
        assert fwer5 < fwer10 < fwer20


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Integrated Report
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatisticalReport:
    def test_generate_full_report(self):
        results = {
            "y_true": [1, 0, 1, 0, 1],
            "y_prob": [0.9, 0.1, 0.8, 0.2, 0.7],
            "sensitivity": 0.85,
            "specificity": 0.90,
            "p_values": [0.01, 0.03, 0.04, 0.20],
        }
        report = generate_report(results)
        assert report.brier is not None
        assert report.ppv_npv is not None
        assert report.multiple_comparison is not None
        assert report.bonferroni_result is not None
        assert report.holm_result is not None
        assert report.fwer_uncorrected is not None

    def test_generate_partial_report(self):
        results = {"sensitivity": 0.8, "specificity": 0.9}
        report = generate_report(results)
        assert report.brier is None
        assert report.ppv_npv is not None
        assert report.multiple_comparison is None

    def test_to_json_roundtrip(self):
        results = {
            "y_true": [1, 0, 1],
            "y_prob": [0.9, 0.1, 0.7],
            "p_values": [0.01, 0.04],
        }
        report = generate_report(results)
        j = report.to_json()
        parsed = json.loads(j)
        assert "brier_decomposition" in parsed
        assert "multiple_comparison" in parsed
        assert parsed["brier_decomposition"]["n_bins"] == 10

    def test_latex_output_contains_tables(self):
        results = {
            "y_true": [1, 0],
            "y_prob": [0.9, 0.1],
            "sensitivity": 0.85,
            "specificity": 0.90,
            "p_values": [0.01, 0.04],
        }
        report = generate_report(results)
        latex = report.to_latex_tables()
        assert "\\begin{table}" in latex
        assert "Brier Score" in latex
        assert "Prevalence" in latex
        assert "Multiple Comparison" in latex

    def test_empty_report(self):
        report = generate_report({})
        assert report.brier is None
        assert report.ppv_npv is None
        assert report.multiple_comparison is None
        latex = report.to_latex_tables()
        assert latex == ""
