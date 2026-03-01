"""Tests for corrected DerSimonian-Laird meta-analysis in statistical_rigor."""

from __future__ import annotations

import math

import pytest

from src.statistical_rigor import (
    CorrectedMetaAnalysis,
    HonestRangeReport,
    LogitTransformResult,
    MetaRegressionResult,
    SuiteResult,
    corrected_meta_analysis,
    honest_range_report,
    logit_transform_meta_analysis,
    meta_regression_difficulty,
    _logit,
    _expit,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

FOUR_SUITES = [
    SuiteResult(name="easy", f1=0.972, n=50, difficulty=0.0),
    SuiteResult(name="medium", f1=0.941, n=40, difficulty=0.33),
    SuiteResult(name="hard", f1=0.903, n=30, difficulty=0.67),
    SuiteResult(name="very_hard", f1=0.875, n=20, difficulty=1.0),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Logit / expit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogitExpit:
    def test_logit_expit_inverse(self):
        for p in [0.1, 0.5, 0.9, 0.01, 0.99]:
            assert _expit(_logit(p)) == pytest.approx(p, abs=1e-8)

    def test_logit_half(self):
        assert _logit(0.5) == pytest.approx(0.0)

    def test_expit_zero(self):
        assert _expit(0.0) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Honest range report tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHonestRangeReport:
    def test_four_suites(self):
        report = honest_range_report(FOUR_SUITES)
        assert report.f1_min == pytest.approx(0.875)
        assert report.f1_max == pytest.approx(0.972)
        assert len(report.f1_values) == 4

    def test_sorted_by_difficulty(self):
        shuffled = [FOUR_SUITES[2], FOUR_SUITES[0], FOUR_SUITES[3], FOUR_SUITES[1]]
        report = honest_range_report(shuffled)
        assert report.suite_names[0] == "easy"
        assert report.suite_names[-1] == "very_hard"

    def test_performance_trend(self):
        report = honest_range_report(FOUR_SUITES)
        assert "degrading" in report.performance_trend

    def test_caveat_present(self):
        report = honest_range_report(FOUR_SUITES)
        assert "caveat" in report.caveat.lower()

    def test_weighted_mean(self):
        report = honest_range_report(FOUR_SUITES)
        total_n = sum(s.n for s in FOUR_SUITES)
        expected = sum(s.f1 * s.n for s in FOUR_SUITES) / total_n
        assert report.weighted_mean == pytest.approx(expected)

    def test_range_format_string(self):
        report = honest_range_report(FOUR_SUITES)
        assert "0.875" in report.performance_trend
        assert "0.972" in report.performance_trend

    def test_single_suite(self):
        report = honest_range_report([FOUR_SUITES[0]])
        assert report.f1_min == report.f1_max


# ═══════════════════════════════════════════════════════════════════════════════
# Logit transformation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogitTransform:
    def test_basic(self):
        result = logit_transform_meta_analysis(FOUR_SUITES)
        assert result is not None
        assert len(result.logit_values) == 4
        assert 0 < result.pooled_f1 < 1
        assert result.ci_lower_f1 < result.pooled_f1 < result.ci_upper_f1

    def test_boundary_f1(self):
        suites_with_zero = [SuiteResult("test", 0.0, 10, 0.0)]
        result = logit_transform_meta_analysis(suites_with_zero)
        assert result is None

    def test_ci_within_01(self):
        result = logit_transform_meta_analysis(FOUR_SUITES)
        assert result is not None
        assert 0 <= result.ci_lower_f1 <= 1
        assert 0 <= result.ci_upper_f1 <= 1

    def test_empty_suites(self):
        result = logit_transform_meta_analysis([])
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Meta-regression tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetaRegression:
    def test_identifiable(self):
        result = meta_regression_difficulty(FOUR_SUITES)
        assert result.identifiable

    def test_negative_slope(self):
        result = meta_regression_difficulty(FOUR_SUITES)
        assert result.slope < 0  # harder → lower F1

    def test_r_squared_bounds(self):
        result = meta_regression_difficulty(FOUR_SUITES)
        assert 0 <= result.r_squared <= 1

    def test_predicted_f1_count(self):
        result = meta_regression_difficulty(FOUR_SUITES)
        assert len(result.predicted_f1) == 4

    def test_single_suite_not_identifiable(self):
        result = meta_regression_difficulty([FOUR_SUITES[0]])
        assert not result.identifiable

    def test_same_difficulty_not_identifiable(self):
        same_diff = [
            SuiteResult("a", 0.9, 10, 0.5),
            SuiteResult("b", 0.8, 10, 0.5),
        ]
        result = meta_regression_difficulty(same_diff)
        assert not result.identifiable


# ═══════════════════════════════════════════════════════════════════════════════
# Corrected meta-analysis integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectedMetaAnalysis:
    def test_full_analysis(self):
        result = corrected_meta_analysis(FOUR_SUITES)
        assert result.honest_range is not None
        assert result.logit_transform is not None
        assert result.meta_regression is not None

    def test_dl_caveat_present(self):
        result = corrected_meta_analysis(FOUR_SUITES)
        assert "misleading" in result.dl_caveat.lower()

    def test_dl_estimate_computed(self):
        result = corrected_meta_analysis(FOUR_SUITES)
        assert 0.8 < result.original_dl_estimate < 1.0

    def test_honest_range_is_primary(self):
        result = corrected_meta_analysis(FOUR_SUITES)
        assert result.honest_range is not None
        assert result.honest_range.f1_min == pytest.approx(0.875)
        assert result.honest_range.f1_max == pytest.approx(0.972)
