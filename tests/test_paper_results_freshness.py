from __future__ import annotations

import json

from reproducibility import paper_results_freshness as freshness
from scripts.benchmark_delta import compute_delta


def test_quality_snapshot_matches_committed_dashboard_baseline():
    baseline = freshness.quality_baseline_snapshot()
    current = freshness.quality_current_snapshot()
    assert baseline.keys() == current.keys()
    report = compute_delta(baseline, current)
    assert not report.has_regression()
    assert "tg_recall" in current


def test_deployment_snapshot_matches_committed_release_baseline():
    baseline = freshness.deployment_baseline_snapshot()
    current = freshness.deployment_current_snapshot()
    assert baseline.keys() == current.keys()
    report = compute_delta(baseline, current)
    assert not report.has_regression()
    assert current["supported_passed"] == baseline["supported_passed"]


def test_snapshot_delta_detects_paper_quality_regression():
    baseline = {"tg_recall": 1.0, "tg_false_positives": 0.0}
    current = {"tg_recall": 0.5, "tg_false_positives": 2.0}
    report = compute_delta(baseline, current)
    assert report.has_regression()
    assert {delta.path for delta in report.regressions} == {
        "tg_false_positives",
        "tg_recall",
    }


def test_rendered_report_and_json_summary_are_bot_readable():
    checks = freshness.build_checks()
    md = freshness.render_report(checks)
    summary = freshness.result_json(checks)
    assert "# Paper results freshness" in md
    assert "benchmark_delta.py" in md
    assert summary["ok"] is True
    assert {entry["name"] for entry in summary["checks"]} == {
        "paper_quality_dashboard",
        "deployment_release_dashboard",
    }


def test_cli_json_skip_rerun_writes_machine_summary(capsys):
    assert freshness.main(["--skip-rerun", "--json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["ok"] is True
    assert all(item["metric_count"] > 0 for item in data["checks"])


def test_refresh_registry_uses_real_generators_before_check_mode():
    refresh_scripts = [tuple(command[1:]) for command in freshness.REGENERATION_COMMANDS]
    check_scripts = [tuple(command[1:]) for command in freshness.FRESHNESS_COMMANDS]
    assert ("evaluation/precision_recall.py",) in refresh_scripts
    assert ("evaluation/precision_recall.py", "--check") in check_scripts
    assert all("--check" not in command for command in refresh_scripts)
