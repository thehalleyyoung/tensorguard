"""Tests for Step 251: pre-specified evaluation protocol registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reproducibility.evaluation_protocol as ep

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
    return ep.measure()


def test_registered_splits_and_disjoint_blind_split(data):
    by_id = {s["id"]: s for s in data["splits"]}
    assert {"development_corpus", "primary_heldout_blind_split",
            "external_real_benchmark"} <= set(by_id)
    assert by_id["development_corpus"]["held_out"] is False
    assert by_id["primary_heldout_blind_split"]["held_out"] is True
    assert by_id["primary_heldout_blind_split"]["tuning_allowed"] == "forbidden"
    dis = data["split_disjointness"]
    assert dis["development_vs_blind_disjoint"] is True
    assert dis["intersection_n"] == 0


def test_blind_preregistration_cross_checks_manifest(data):
    blind = data["blind_preregistration"]
    assert blind["hash_matches_document"] is True
    assert blind["registered_manifest_sha256"] == ep._sha256(ep.BLIND_MANIFEST)
    counts = blind["registered_counts"]
    split = next(s for s in data["splits"]
                 if s["id"] == "primary_heldout_blind_split")
    assert counts["n_cases"] == split["n_cases"]
    assert counts["buggy"] == split["label_counts"]["buggy"]
    assert counts["clean"] == split["label_counts"]["clean"]
    assert counts["n_families"] == split["n_families"]


def test_tuning_freeze_has_nonnegotiable_rules(data):
    rules = {r["id"]: r for r in data["tuning_freeze"]}
    assert "no_blind_threshold_tuning" in rules
    assert "baseline_case_policy" in rules
    assert "negative_result_policy" in rules
    assert "forbidden" in next(
        s for s in data["splits"] if s["id"] == "primary_heldout_blind_split"
    )["tuning_allowed"]


def test_metric_registry_covers_headline_statistics(data):
    metric_ids = {m["id"] for m in data["metric_definitions"]}
    required = {
        "recall_on_decided",
        "recall_on_all_buggy",
        "specificity_on_decided",
        "false_positive_rate",
        "precision",
        "abstention_rate",
        "wilson_95_ci",
        "paired_mcnemar",
        "paired_effect_size",
        "power_and_sample_size",
        "overfitting_gap",
    }
    assert required <= metric_ids
    for metric in data["metric_definitions"]:
        assert metric["formula"]
        assert metric["scripts"]


def test_analysis_scripts_are_present_and_hashed(data):
    assert data["all_analysis_scripts_present"] is True
    for script in data["analysis_scripts"]:
        path = REPO / script["path"]
        assert path.exists()
        assert script["sha256"] == ep._sha256(script["path"])
    governed = [s for s in data["analysis_scripts"] if s["governed_by_protocol"]]
    assert len(governed) >= 10


def test_protocol_precedes_governed_scoring_in_reproduce_all(data):
    order = data["reproduction_order"]
    assert order["protocol_step_index"] is not None
    assert order["protocol_precedes_governed_scoring"] is True
    for row in order["governed_scoring_scripts"]:
        assert row["step_index"] is not None, row["path"]
        assert row["after_protocol"] is True, row["path"]


def test_dashboard_includes_protocol_card():
    from reproducibility import build_dashboard

    cards = {c["id"]: c for c in build_dashboard.measure()["cards"]}
    assert "evaluation_protocol" in cards
    assert cards["evaluation_protocol"]["source_artifact"] == "evaluation_protocol.json"


def test_no_volatile_keys_and_deterministic(data):
    for key in _walk_keys(data):
        assert not any(tok in str(key).lower() for tok in VOLATILE), key
    assert ep.measure() == data


def test_check_mode_byte_identical():
    assert ep.run(check=True) == 0


def test_artifact_json_matches_measure():
    artifact = json.loads(ep.OUT_JSON.read_text())
    assert artifact == ep.measure()
