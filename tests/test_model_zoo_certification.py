"""Step 282 -- public model-zoo certification queue is real and reproducible."""

from __future__ import annotations

import json
from pathlib import Path

from reproducibility import model_zoo_certification as mz


def test_committed_model_zoo_certification_artifacts_are_fresh():
    assert mz.run(check=True) == 0
    payload = json.loads(mz.OUT_JSON.read_text(encoding="utf-8"))
    assert payload["schema"] == mz.SCHEMA
    assert payload["queue"]["job_count"] >= 25
    assert payload["queue"]["failed_count"] == 0
    assert payload["queue"]["certified_count"] == payload["queue"]["job_count"]


def test_queue_rows_are_verifier_backed_and_badged():
    rows = mz.queue_rows()
    assert len(rows) >= 25
    assert len({row["job_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["job_id"].startswith("tgzoo-")
        assert row["status"] == "certified"
        assert row["observed_clean_verdict"] == "SAFE"
        assert row["observed_buggy_verdict"] == "UNSAFE"
        assert row["buggy_bug_count"] > 0
        assert "img.shields.io/badge/model--zoo-certified-brightgreen" in row["badge_markdown"]
        assert row["failure_explanation"] == ""


def test_failure_explanation_is_deterministic_for_verdict_drift():
    row = {
        "expected_clean_verdict": "SAFE",
        "observed_clean_verdict": "UNKNOWN",
        "expected_buggy_verdict": "UNSAFE",
        "observed_buggy_verdict": "SAFE",
    }
    assert mz._explain_failure(row) == (
        "clean variant expected SAFE, observed UNKNOWN; "
        "buggy variant expected UNSAFE, observed SAFE"
    )


def test_markdown_exposes_status_badges_and_failure_contract():
    data = mz.manifest()
    md = mz.render_markdown(data)
    assert "model-zoo certification queue" in md
    assert "img.shields.io/badge/model--zoo-certified-brightgreen" in md
    assert "Failure explanation" in md
    assert "clean model verifies `SAFE`" in md


def test_monthly_freshness_workflow_is_wired_to_check_command():
    workflow = Path(mz.REPO / mz.WORKFLOW).read_text(encoding="utf-8")
    assert f'- cron: "{mz.CRON_UTC}"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "python reproducibility/model_zoo_certification.py --check" in workflow
    assert "reproducibility/model_zoo_certification.py" in workflow
