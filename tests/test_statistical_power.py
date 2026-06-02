"""Tests for the Step 120 statistical-power / sample-size harness."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import reproducibility.statistical_power as sp

REPO = Path(__file__).resolve().parent.parent

VOLATILE = ("time", "elapsed", "seconds", "date", "timestamp", "duration", "wall")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture(scope="module")
def data():
    return sp.measure()


def test_exact_rule_of_three_upper_bound():
    # 0 failures in n: upper bound 1 - alpha^(1/n); for large n ~ 3/n.
    for n in (50, 100, 500, 1235):
        b = sp.one_sided_upper_zero_fail(n)
        assert b == pytest.approx(1.0 - 0.05 ** (1.0 / n))
        # rule of three is a good approximation for large n
        assert abs(b - 3.0 / n) < 0.5 / n + 1e-9 or n < 100


def test_lower_bound_complements_upper():
    # perfect success lower bound = alpha^(1/n) = 1 - upper(zero-fail).
    for n in (29, 138, 376):
        assert sp.one_sided_lower_all_success(n) == pytest.approx(
            1.0 - sp.one_sided_upper_zero_fail(n))


def test_power_monotone_in_n_and_effect():
    # More trials -> more power; larger true effect -> more power.
    assert sp.power_to_detect_failure(200, 0.05) > sp.power_to_detect_failure(50, 0.05)
    assert sp.power_to_detect_failure(100, 0.10) > sp.power_to_detect_failure(100, 0.01)
    assert sp.power_to_detect_recall_below(300, 0.90) > sp.power_to_detect_recall_below(
        300, 0.99)


def test_min_n_matches_power_threshold():
    # min_n_zero_fail(t) is the smallest n whose zero-fail bound <= t.
    for t in (0.01, 0.05):
        n = sp.min_n_zero_fail(t)
        assert sp.one_sided_upper_zero_fail(n) <= t + 1e-12
        assert sp.one_sided_upper_zero_fail(n - 1) > t


def test_min_discordant_for_significance_exact():
    need = sp.min_discordant_for_significance(0.05)
    assert need == 6  # 2*2^-6 = 0.03125 <= 0.05, 2*2^-5 = 0.0625 > 0.05
    assert 2 * 0.5 ** need <= 0.05
    assert 2 * 0.5 ** (need - 1) > 0.05


def test_claims_grounded_in_real_artifacts(data):
    # The n's must equal the counts in the committed artifacts (no invented n).
    fp = json.loads((REPO / "reproducibility/fp_stress_eval.json").read_text())
    mut = json.loads((REPO / "reproducibility/mutation_clean_models.json").read_text())
    by_id = {c["id"]: c for c in data["claims"]}
    assert by_id["fp_stress_sound_zero_fa"]["n"] == fp["per_mode"]["sound"][
        "false_alarm_rate"]["n"]
    assert by_id["mutation_kill_rate"]["n"] == mut["per_mode"]["sound"][
        "kill_rate"]["n"]


def test_perfect_recall_and_zero_fail_flags(data):
    by_id = {c["id"]: c for c in data["claims"]}
    # The huge differential corpus is powered against a 1% rate.
    assert by_id["differential_zero_fa"]["adequately_powered_1pct"]
    # Every perfect-recall claim certifies the 95% floor.
    for c in data["claims"]:
        if c["kind"] == "perfect_recall":
            assert c["adequately_powered_95pct"]


def test_noop_mcnemar_significant_and_powered(data):
    by_id = {c["id"]: c for c in data["claims"]}
    noop = by_id["mcnemar_vs_noop"]
    assert noop["all_discordant_favour_tg"]
    assert noop["adequately_powered"]
    assert noop["exact_two_sided_p"] <= 0.05


def test_pooled_bound_is_tightest(data):
    # Pooling all clean trials gives a bound at least as tight as any single one.
    singles = [c["one_sided_upper_bound_95"] for c in data["claims"]
               if c["kind"] == "zero_failure"]
    assert data["pooled_clean_zero_fail_upper_bound_95"] <= min(singles) + 1e-12


def test_no_volatile_keys_and_deterministic(data):
    for k in _walk_keys(data):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert sp.measure() == data  # deterministic


def test_check_mode_byte_identical():
    assert sp.run(check=True) == 0
