"""Tests for the Step 121 effect-size + dual multiple-comparison harness."""

from __future__ import annotations

import pytest

import reproducibility.effect_sizes as es

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
    return es.measure()


def test_cohen_g_bands():
    assert es.cohen_g_magnitude(0.0) == "negligible"
    assert es.cohen_g_magnitude(0.10) == "small"
    assert es.cohen_g_magnitude(0.20) == "medium"
    assert es.cohen_g_magnitude(0.40) == "large"
    # symmetric in sign
    assert es.cohen_g_magnitude(-0.40) == "large"


def test_every_comparison_has_effect_size_and_corrections(data):
    assert data["every_comparison_has_effect_size"]
    for c in data["comparisons"]:
        es_block = c["effect_sizes"]
        for key in ("cohen_g", "cohen_g_magnitude",
                    "haldane_anscombe_odds_ratio", "risk_difference"):
            assert key in es_block
        for key in ("holm_bonferroni_adjusted_p", "holm_bonferroni_reject",
                    "benjamini_hochberg_adjusted_p", "benjamini_hochberg_reject"):
            assert key in c["corrections"]


def test_haldane_or_finite_when_c_zero(data):
    # raw OR is None (divergent) when c==0 but the continuity-corrected OR is finite
    for c in data["comparisons"]:
        if c["c_tg_wrong_base_right"] == 0 and c["b_tg_right_base_wrong"] > 0:
            assert c["effect_sizes"]["raw_odds_ratio"] is None
            assert c["effect_sizes"]["haldane_anscombe_odds_ratio"] is not None
            assert c["effect_sizes"]["haldane_anscombe_odds_ratio"] > 1.0


def test_noop_is_significant_under_both_corrections(data):
    noop = next(c for c in data["comparisons"] if c["baseline"] == "noop")
    assert noop["corrections"]["holm_bonferroni_reject"]
    assert noop["corrections"]["benjamini_hochberg_reject"]
    assert noop["effect_sizes"]["cohen_g_magnitude"] == "large"


def test_risk_difference_matches_counts(data):
    n = data["n_items"]
    for c in data["comparisons"]:
        expect = (c["b_tg_right_base_wrong"] - c["c_tg_wrong_base_right"]) / n
        assert c["effect_sizes"]["risk_difference"] == pytest.approx(expect)


def test_adjusted_p_never_below_raw(data):
    # Multiple-comparison correction can only inflate (or hold) a p-value.
    for c in data["comparisons"]:
        raw = c["mcnemar_p_value"]
        for key in ("holm_bonferroni_adjusted_p", "benjamini_hochberg_adjusted_p"):
            adj = c["corrections"][key]
            if adj is not None:
                assert adj >= raw - 1e-9


def test_corrections_agree_count(data):
    assert data["corrections_agree"]
    assert data["n_holm_significant"] == data["n_bh_significant"]


def test_no_volatile_keys_and_deterministic(data):
    for k in _walk_keys(data):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert es.measure() == data


def test_check_mode_byte_identical():
    assert es.run(check=True) == 0
