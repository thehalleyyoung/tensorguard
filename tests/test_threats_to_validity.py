"""Tests for the Step 124 threats-to-validity generator."""

from __future__ import annotations

import pytest

import reproducibility.threats_to_validity as ttv

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
    return ttv.measure()


def test_all_four_validity_categories(data):
    assert data["all_four_categories_covered"]
    assert data["categories_covered"] == [
        "conclusion", "construct", "external", "internal"]


def test_threats_are_grounded_in_real_artifacts(data):
    # The summary must equal a fresh read of the artifacts (no hardcoding).
    assert data["summary"] == ttv.summarise()
    s = data["summary"]
    # Sanity: rates are real probabilities, counts are non-negative.
    for m in s["modes"]:
        assert 0.0 <= s["fp_false_alarm_rate"][m] <= 1.0
        assert 0.0 <= s["natural_coverage"][m] <= 1.0


def test_residual_levels_are_computed_not_asserted(data):
    by_id = {t["id"]: t for t in data["threats"]}
    s = data["summary"]
    # Construct: low iff zero buggy items were abstained.
    construct = by_id["construct_abstention_masking"]
    if s["extended_abstained_buggy"] == 0:
        assert construct["residual_risk"] == "low"
    # External: low only if the smallest clean sample is individually powered.
    external = by_id["external_synthetic_generalisation"]
    if not s["smallest_clean_sample_individually_powered"]:
        assert external["residual_risk"] == "medium"


def test_zero_false_positive_threat_low(data):
    by_id = {t["id"]: t for t in data["threats"]}
    conclusion = by_id["conclusion_false_alarm_undercount"]
    assert data["summary"]["total_false_positives_observed"] == 0
    assert conclusion["residual_risk"] == "low"


def test_every_threat_has_required_fields(data):
    for t in data["threats"]:
        for f in ("id", "category", "threat", "evidence", "mitigation",
                  "residual_risk"):
            assert f in t and t[f] not in (None, "", {})
        assert t["residual_risk"] in ("low", "medium", "high")


def test_level_helper_ordering():
    assert ttv._level(True, True) == "low"
    assert ttv._level(False, True) == "medium"
    assert ttv._level(False, False) == "high"


def test_markdown_renders_all_threats(data):
    md = ttv.render_markdown(data)
    for t in data["threats"]:
        assert t["id"] in md
    assert "Residual risk" in md


def test_no_volatile_keys_and_deterministic(data):
    for k in _walk_keys(data):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert ttv.measure() == data


def test_check_mode_byte_identical():
    assert ttv.run(check=True) == 0
