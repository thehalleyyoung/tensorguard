"""Tests for the Step 254 PR-history survival study."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reproducibility import pr_history_survival as surv  # noqa: E402


_VOLATILE = (
    "elapsed",
    "wall",
    "clock",
    "_ms",
    "seconds",
    "duration",
    "timestamp",
    "created_at",
)


def _artifact() -> dict:
    return json.loads(surv.OUT_JSON.read_text())


def _walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_keys(value)


def _corpus_link_counts():
    records = [
        json.loads(line)
        for line in surv.CORPUS.read_text().splitlines()
        if line.strip()
    ]
    linked = [r for r in records if r["commit_links"]]
    return records, linked, Counter(r["commit_link_status"] for r in linked)


def test_artifact_check_mode_passes():
    assert surv.run(check=True) == 0


def test_reproduction_pipeline_and_makefile_are_wired():
    import reproducibility.reproduce_all as ra

    assert "reproducibility/pr_history_survival.json" in ra.GENERATED_DETERMINISTIC
    assert "reproducibility/pr_history_survival.md" in ra.GENERATED_DETERMINISTIC
    assert any(
        argv[:2] == [sys.executable, "reproducibility/pr_history_survival.py"]
        for _, argv, _ in ra.STEPS
    )
    makefile = (REPO / "Makefile").read_text()
    assert "pr-history-survival:" in makefile
    assert "$(PYTHON) reproducibility/pr_history_survival.py" in makefile


def test_fix_linked_counts_are_recomputed_from_the_frozen_corpus():
    data = _artifact()
    records, linked, by_status = _corpus_link_counts()
    src = data["source_corpus"]
    assert src["records"] == len(records) == 2704
    assert src["fix_linked_records"] == len(linked) == 332
    assert src["direct_pr_records"] == by_status[surv.DIRECT_PR] == 205
    assert src["candidate_issue_records"] == by_status[surv.CANDIDATE_PR] == 127
    assert src["records_without_fix_link"] == 2372


def test_scope_is_explicitly_category_level_not_historical_checkout_replay():
    scope = _artifact()["evidence_scope"]
    assert scope["granularity"] == "runtime_signature_category_replay"
    assert scope["historical_pre_fix_checkouts_replayed"] is False
    assert scope["stored_source_blob"] is False
    assert scope["stored_patch"] is False
    assert scope["per_row_counts_are_category_multiplicities"] is True
    assert "not on the historical repository checkout" in scope["claim"]


def test_all_fix_linked_categories_have_static_replay_and_are_caught():
    data = _artifact()
    rows = data["category_replay"]
    assert sorted(rows) == sorted(surv.CASES)
    assert data["survival_estimate"]["all_categories_represented"] is True
    for category, row in rows.items():
        assert row["fix_linked_records"] > 0, category
        assert row["static_replay"]["verdict"] == "UNSAFE", category
        assert row["static_replay"]["caught_by_tensorguard"] is True, category
        assert row["category_level_missed_records"] == 0, category
        assert row["static_replay"]["graph_steps"] >= 1, category
    assert data["survival_estimate"]["direct_pr_only"]["category_level_caught"] == 205


def test_cpu_minimized_reproducers_raise_live_pytorch_and_device_is_qualified():
    torch = pytest.importorskip("torch")
    data = _artifact()
    for category, row in data["category_replay"].items():
        repro = row["eager_reproducer"]
        if category == "device_mismatch":
            assert repro["status"] == "cuda_qualified_not_executed_in_cpu_ci"
            if torch.cuda.is_available():
                proc = subprocess.run(
                    [sys.executable, str(REPO / repro["path"])],
                    cwd=str(REPO),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                assert proc.returncode != 0
                assert "Expected all tensors to be on the same device" in (
                    proc.stdout + proc.stderr
                )
            continue

        assert repro["status"] == "proven_live", category
        assert repro["proven_live_on_this_host"] is True, category
        assert repro["observed_nonzero_exit"] is True, category
        assert (REPO / repro["path"]).exists(), category


def test_detection_depth_and_ci_cost_proxy_is_structural_and_consistent():
    data = _artifact()
    cost = data["detection_depth_and_ci_cost_proxy"]
    rows = data["category_replay"]
    total = data["source_corpus"]["fix_linked_records"]
    assert cost["unit"] == "structural_proxy_not_wall_clock"
    assert cost["static"]["model_executions"] == 0
    assert cost["static"]["analyzer_passes"] == total
    assert cost["dynamic_forward_baseline"]["forward_invocations"] == total
    assert cost["dynamic_forward_baseline"]["requires_concrete_inputs"] is True
    expected_prefix = sum(
        row["fix_linked_records"] * row["dynamic_prefix_ops_before_failure"]
        for row in rows.values()
    )
    expected_graph_steps = sum(
        row["fix_linked_records"] * row["static_replay"]["graph_steps"]
        for row in rows.values()
    )
    assert (
        cost["dynamic_forward_baseline"]["successful_prefix_ops_before_failure"]
        == expected_prefix
    )
    assert cost["static"]["graph_steps_analyzed"] == expected_graph_steps
    assert cost["comparison"]["static_detects_before_runtime_execution"] is True


def test_artifact_has_no_volatile_keys_and_is_byte_deterministic():
    data = surv.measure()
    assert json.dumps(data, sort_keys=True) == json.dumps(surv.measure(), sort_keys=True)
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(token in low for token in _VOLATILE), key
