from __future__ import annotations

import json

import pytest

from evaluation import negative_controls as nc


@pytest.fixture(scope="module")
def artifact():
    return nc.run(check=True)


def test_committed_summary_reports_honest_loss(artifact):
    summary = artifact["summary"]
    assert summary["n_cases"] == 6
    assert summary["tensorguard_caught"] == 0
    assert summary["tensorguard_recall"] == 0.0
    assert summary["runtime_finite_output_check_caught"] == summary["n_cases"]
    assert summary["runtime_finite_output_check_recall"] == 1.0
    assert "loss" in summary["honest_outcome"]


def test_plain_smoke_test_is_not_overcredited(artifact):
    summary = artifact["summary"]
    assert summary["runtime_smoke_caught"] == 2
    nonfinite = artifact["by_family"]["nonfinite_value"]
    assert nonfinite["runtime_smoke_caught"] == 0
    assert nonfinite["runtime_finite_output_check_caught"] == nonfinite["total"]


@pytest.mark.parametrize("case", nc.build_corpus(), ids=lambda c: c["id"])
def test_each_case_is_a_genuine_runtime_value_failure(case):
    genuine, detail = nc.is_genuine_negative_control(case)
    assert genuine, detail
    if case["family"] == "nonfinite_value":
        assert detail == "nonfinite_output"
    else:
        assert detail.startswith("exception:")


@pytest.mark.parametrize("case", nc.build_corpus(), ids=lambda c: c["id"])
def test_tensorguard_does_not_flag_structurally_clean_value_controls(case):
    result = nc.tensorguard_result(case)
    assert result["bug_count"] == 0
    assert result["caught"] is False
    assert result["verdict"] in {"SAFE", "UNKNOWN"}


def test_artifact_scope_declares_value_semantics_out_of_contract(artifact):
    scope = artifact["meta"]["tensorguard_scope"]
    assert "shape/device/dtype/phase" in scope
    assert "not arbitrary value-domain assertions" in scope
    assert {row["out_of_contract_dimension"] for row in artifact["per_case"]} == {
        "value_semantics"
    }


def test_json_artifact_matches_check_result(artifact):
    with open(nc.OUT_JSON, encoding="utf-8") as fh:
        committed = json.load(fh)
    assert committed == artifact


def test_artifact_regenerates_byte_identically():
    assert nc.run(check=True)["summary"]["n_cases"] == 6
