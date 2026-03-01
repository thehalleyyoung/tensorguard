"""Tests for deployment cost analysis."""

import pytest

from src.deployment_cost_analysis import (
    VerificationOutcome,
    CostBreakdown,
    DeploymentCostReport,
    CompositionSemantics,
    compute_deployment_cost,
    compute_full_deployment_analysis,
    compare_semantics_cost,
    compute_cost_sensitivity,
    DEFAULT_FP_COST,
    SEMANTICS_FN_MULTIPLIER,
    SEMANTICS_FP_MULTIPLIER,
    DEFAULT_LOSS_RATIOS,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_outcomes(tp=0, tn=0, fp=0, fn=0):
    """Create verification outcomes with known confusion matrix."""
    results = []
    for _ in range(tp):
        results.append(VerificationOutcome(
            predicted_safe=False, actual_safe=False, confidence=0.9,
        ))
    for _ in range(tn):
        results.append(VerificationOutcome(
            predicted_safe=True, actual_safe=True, confidence=0.9,
        ))
    for _ in range(fp):
        results.append(VerificationOutcome(
            predicted_safe=False, actual_safe=True, confidence=0.9,
        ))
    for _ in range(fn):
        results.append(VerificationOutcome(
            predicted_safe=True, actual_safe=False, confidence=0.9,
        ))
    return results


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestBasicCostComputation:

    def test_empty_results(self):
        bd = compute_deployment_cost([], 10.0)
        assert bd.total_expected_cost == 0.0
        assert bd.num_fp == 0
        assert bd.num_fn == 0

    def test_all_true_negatives(self):
        outcomes = _make_outcomes(tn=10)
        bd = compute_deployment_cost(outcomes, 10.0)
        assert bd.total_expected_cost == 0.0
        assert bd.num_tn == 10
        assert bd.num_fn == 0

    def test_false_negative_cost_scales_with_ratio(self):
        outcomes = _make_outcomes(fn=1)
        bd10 = compute_deployment_cost(outcomes, 10.0)
        bd100 = compute_deployment_cost(outcomes, 100.0)
        bd1000 = compute_deployment_cost(outcomes, 1000.0)
        assert bd10.fn_cost < bd100.fn_cost < bd1000.fn_cost
        assert bd1000.fn_cost == pytest.approx(1000.0)

    def test_false_positive_cost_constant(self):
        outcomes = _make_outcomes(fp=5)
        bd10 = compute_deployment_cost(outcomes, 10.0)
        bd1000 = compute_deployment_cost(outcomes, 1000.0)
        assert bd10.fp_cost == bd1000.fp_cost  # FP cost doesn't scale with ratio


class TestCompositionSemanticsCost:

    def test_monolithic_lowest_cost(self):
        outcomes = _make_outcomes(fn=1)
        mono = compute_deployment_cost(outcomes, 100.0, "MONOLITHIC_SAFE")
        per_sg = compute_deployment_cost(outcomes, 100.0, "PER_SUBGRAPH_SAFE")
        unknown = compute_deployment_cost(outcomes, 100.0, "UNKNOWN")
        assert mono.fn_cost <= per_sg.fn_cost <= unknown.fn_cost

    def test_semantics_multipliers_applied(self):
        outcomes = _make_outcomes(fn=1)
        mono = compute_deployment_cost(outcomes, 100.0, "MONOLITHIC_SAFE")
        per_sg = compute_deployment_cost(outcomes, 100.0, "PER_SUBGRAPH_SAFE")
        # PER_SUBGRAPH_SAFE has 1.15x FN multiplier
        assert per_sg.fn_cost == pytest.approx(
            mono.fn_cost * SEMANTICS_FN_MULTIPLIER["PER_SUBGRAPH_SAFE"], rel=1e-6
        )

    def test_unknown_highest_cost(self):
        outcomes = _make_outcomes(fn=2, fp=1)
        unknown = compute_deployment_cost(outcomes, 100.0, "UNKNOWN")
        mono = compute_deployment_cost(outcomes, 100.0, "MONOLITHIC_SAFE")
        assert unknown.total_expected_cost > mono.total_expected_cost


class TestStrategyRecommendation:

    def test_high_ratio_recommends_monolithic(self):
        outcomes = _make_outcomes(fn=1)
        bd = compute_deployment_cost(outcomes, 1000.0, "PER_SUBGRAPH_SAFE")
        assert "MONOLITHIC_SAFE" in bd.strategy_recommendation

    def test_low_ratio_accepts_per_subgraph(self):
        outcomes = _make_outcomes(fp=5)
        bd = compute_deployment_cost(outcomes, 10.0, "MONOLITHIC_SAFE")
        assert "PER_SUBGRAPH_SAFE" in bd.strategy_recommendation

    def test_unknown_at_high_ratio_warns(self):
        outcomes = _make_outcomes(fn=1)
        bd = compute_deployment_cost(outcomes, 1000.0, "UNKNOWN")
        assert "CRITICAL" in bd.strategy_recommendation


class TestProbabilisticEstimation:
    """Test cost computation when ground truth is unknown."""

    def test_high_confidence_safe_low_fn(self):
        outcomes = [VerificationOutcome(
            predicted_safe=True, actual_safe=None, confidence=0.95,
        )]
        bd = compute_deployment_cost(outcomes, 100.0)
        # 5% chance of FN
        assert bd.fn_cost > 0
        assert bd.fn_cost == pytest.approx(0.05 * 100.0, rel=1e-6)

    def test_low_confidence_unsafe_high_fp(self):
        outcomes = [VerificationOutcome(
            predicted_safe=False, actual_safe=None, confidence=0.6,
        )]
        bd = compute_deployment_cost(outcomes, 100.0)
        # 40% chance of FP
        assert bd.fp_cost > 0
        assert bd.fp_cost == pytest.approx(0.4, rel=1e-6)


class TestFullDeploymentAnalysis:

    def test_covers_default_ratios(self):
        outcomes = _make_outcomes(tp=2, tn=8, fp=1, fn=1)
        report = compute_full_deployment_analysis(outcomes)
        assert set(report.cost_by_ratio.keys()) == {10.0, 100.0, 1000.0}
        assert report.optimal_strategy != ""

    def test_custom_ratios(self):
        outcomes = _make_outcomes(tp=1, tn=5)
        report = compute_full_deployment_analysis(
            outcomes, loss_ratios=[5.0, 50.0, 500.0]
        )
        assert set(report.cost_by_ratio.keys()) == {5.0, 50.0, 500.0}

    def test_report_serialization(self):
        outcomes = _make_outcomes(tp=1, tn=5, fp=1, fn=1)
        report = compute_full_deployment_analysis(outcomes)
        d = report.to_dict()
        assert "cost_by_ratio" in d
        assert "optimal_strategy" in d
        assert "composition_semantics" in d


class TestCrossSemanticsComparison:

    def test_compare_all_three(self):
        outcomes = _make_outcomes(fn=2, fp=1, tn=10)
        comparison = compare_semantics_cost(outcomes, 100.0)
        assert "MONOLITHIC_SAFE" in comparison
        assert "PER_SUBGRAPH_SAFE" in comparison
        assert "UNKNOWN" in comparison
        assert (comparison["MONOLITHIC_SAFE"].total_expected_cost
                <= comparison["UNKNOWN"].total_expected_cost)


class TestCostSensitivity:

    def test_sensitivity_returns_elasticities(self):
        outcomes = _make_outcomes(fn=1, tn=10)
        sens = compute_cost_sensitivity(outcomes)
        assert len(sens["elasticities"]) == len(DEFAULT_LOSS_RATIOS) - 1
        assert len(sens["expected_costs"]) == len(DEFAULT_LOSS_RATIOS)

    def test_fn_dominated(self):
        outcomes = _make_outcomes(fn=5, fp=0)
        sens = compute_cost_sensitivity(outcomes)
        assert sens["cost_dominated_by"] == "FN"


class TestCostBreakdownSerialization:

    def test_to_dict_all_fields(self):
        bd = CostBreakdown(
            total_expected_cost=150.0,
            fp_cost=5.0, fn_cost=100.0,
            tp_cost=45.0, tn_cost=0.0,
            num_fp=5, num_fn=1, num_tp=3, num_tn=10,
            loss_ratio=100.0,
            composition_semantics="PER_SUBGRAPH_SAFE",
            strategy_recommendation="test",
        )
        d = bd.to_dict()
        assert d["loss_ratio"] == 100.0
        assert d["total_expected_cost"] == 150.0
        assert d["strategy_recommendation"] == "test"
