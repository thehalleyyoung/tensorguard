from __future__ import annotations

import pytest

import reproducibility.statistical_meta_analysis as sma


@pytest.fixture(scope="module")
def data():
    return sma.measure()


def test_collects_heterogeneous_real_synthetic_and_stress_suites(data):
    distributions = set(data["distributions"])
    assert {"real_minimized", "synthetic_fuzz", "synthetic_mutation", "clean_stress"} <= distributions
    assert data["n_suites"] >= 12


def test_no_naive_global_pooling_contract(data):
    method = data["method"]
    assert method["naive_pooling_across_distributions_allowed"] is False
    assert method["case_weighted_rates_are_diagnostic_only"] is True
    assert "global" not in data
    assert "raw pooled" in method["why_no_global_pool"]


def test_bootstrap_intervals_are_per_distribution_and_suite_level(data):
    for summary in data["distributions"].values():
        boot = summary["robust_suite_bootstrap"]
        assert boot["n_resamples"] == sma.BOOTSTRAP_RESAMPLES
        assert 0.0 <= boot["ci_low"] <= boot["point"] <= boot["ci_high"] <= 1.0
        assert summary["suite_rate_min"] <= boot["point"] <= summary["suite_rate_max"]


def test_real_distribution_retains_hard_recall_miss(data):
    real = data["distributions"]["real_minimized"]
    assert real["suite_rate_min"] == pytest.approx(0.75)
    assert real["suite_rate_max"] == pytest.approx(1.0)
    assert real["robust_suite_bootstrap"]["ci_low"] < 1.0


def test_random_effects_is_diagnostic_only_when_present(data):
    multi_suite = [s for s in data["distributions"].values() if s["n_suites"] >= 2]
    assert multi_suite
    for summary in multi_suite:
        re = summary["random_effects_logit_diagnostic"]
        assert re is not None
        assert re["diagnostic_only"] is True
        assert 0.0 <= re["pooled_rate"] <= 1.0


def test_metric_summaries_do_not_merge_distributions_into_headline(data):
    bug = data["metrics"]["bug_detection"]
    assert bug["n_suites"] >= 5
    assert bug["case_weighted_rate_diagnostic_only"] >= bug["suite_mean_rate"]
    assert "robust_suite_bootstrap" in bug


def test_artifact_is_deterministic_and_byte_identical():
    assert sma.measure() == sma.measure()
    assert sma.run(check=True) == 0
