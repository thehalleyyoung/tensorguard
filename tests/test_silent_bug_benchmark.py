from __future__ import annotations

import os

from reproducibility import silent_bug_benchmark


def test_corpus_has_all_required_families():
    data = silent_bug_benchmark.measure()
    assert set(data["families"]) == {
        "gradient_freeze",
        "stale_buffer",
        "optimizer_state_drift",
        "train_eval_mode_leakage",
        "quantization_wrong_output",
    }
    assert all(row["total"] >= 3 for row in data["by_family"].values())


def test_every_case_runs_without_exception_and_oracle_fires():
    data = silent_bug_benchmark.measure()
    for case in data["per_case"]:
        assert case["runtime_nonraising"], case["case_id"]
        assert case["oracle_positive"], case["case_id"]


def test_tensor_guard_gates_catch_every_curated_silent_bug():
    data = silent_bug_benchmark.measure()
    summary = data["summary"]
    assert summary["gate_caught"] == summary["total_cases"]
    assert summary["gate_recall"] == 1.0
    for case in data["per_case"]:
        assert case["issue_kinds"], case["case_id"]


def test_optimizer_and_quantization_cases_use_real_semantic_deltas():
    data = silent_bug_benchmark.measure()
    by_id = {case["case_id"]: case for case in data["per_case"]}
    assert by_id["optimizer_drift_zero_exp_avg"]["semantic_delta"] > 0
    assert by_id["optimizer_drift_scaled_exp_avg_sq"]["semantic_delta"] > 0
    assert by_id["optimizer_drift_stale_step"]["semantic_delta"] > 0
    assert by_id["quant_wrong_scale_coarse"]["semantic_delta"] > 0
    assert by_id["quant_wrong_zero_point"]["semantic_delta"] > 0


def test_oracle_independence_is_documented():
    data = silent_bug_benchmark.measure()
    note = data["oracle_independence"]
    assert "reference-output" in note
    assert "gates inspect" in note


def test_measurement_is_deterministic_in_process():
    assert silent_bug_benchmark.measure() == silent_bug_benchmark.measure()


def test_artifact_on_disk_is_up_to_date():
    assert os.path.exists(silent_bug_benchmark.OUT_JSON)
    assert os.path.exists(silent_bug_benchmark.OUT_MD)
    assert silent_bug_benchmark.run(check=True) == 0
