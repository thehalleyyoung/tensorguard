"""Tests for Step 259: registered controlled developer-study packet."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reproducibility.developer_study as ds

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def artifact():
    return ds.measure()


def test_study_is_registered_but_does_not_claim_human_results(artifact):
    meta = artifact["meta"]
    assert meta["no_human_subjects_results"] is True
    assert "not measured participant outcomes" in meta["proxy_disclaimer"]
    assert "no human-subjects outcomes executed" in meta["status"]
    assert set(artifact["registration"]["registered_constructs"]) == {
        "localization_time_proxy",
        "fix_quality",
        "trust_calibration",
    }
    assert all(artifact["registration"]["contains_research_questions"].values())


def test_task_battery_uses_real_marker_bearing_repros(artifact):
    assert artifact["summary"]["n_tasks"] == len(artifact["tasks"]) >= 10
    for task in artifact["tasks"]:
        path = REPO / task["repro_path"]
        assert path.exists()
        assert task["source_sha256"] == ds._sha256(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert "# BUG" in lines[task["marker_line"] - 1]
        assert 1 <= task["gt_line"] <= len(lines)
        assert task["upstream_references"]
        assert task["preferred_fix_reference"].startswith("https://github.com/")


def test_localization_time_is_labeled_linear_proxy(artifact):
    assert artifact["summary"]["localization_effect_is_linear_rescale"] is True
    for task in artifact["tasks"]:
        proxy = task["localization_time_proxy"]
        assert proxy["seconds_per_line"] == ds.SECONDS_PER_LINE
        assert proxy["assisted_seconds"] == pytest.approx(
            proxy["assisted_lines"] * ds.SECONDS_PER_LINE
        )
        assert proxy["unaided_seconds"] == pytest.approx(
            proxy["unaided_lines"] * ds.SECONDS_PER_LINE
        )
        assert "not observed human timing" in proxy["note"]


def test_fix_quality_rubric_covers_real_patch_acceptance(artifact):
    for task in artifact["tasks"]:
        rubric = task["fix_quality_rubric"]
        assert rubric["max_score"] == 3
        assert len(rubric["criteria"]) == 3
        assert any("INPUT_SHAPES" in criterion for criterion in rubric["criteria"])
        assert "runtime execution belongs to the human-study environment" in (
            rubric["gold_evidence"]
        )


def test_trust_calibration_includes_helpful_and_misleading_items(artifact):
    instruments = artifact["instruments"]["trust_calibration"]
    assert instruments["items_include_helpful_and_misleading_tg_advice"] is True
    misleading = {
        task["id"]
        for task in artifact["tasks"]
        if not task["trust_calibration"]["tg_advice_expected_helpful"]
    }
    assert misleading == ds.TRUST_MISLEADING_IDS
    for task in artifact["tasks"]:
        expected = task["trust_calibration"]["expected_response"]
        if task["id"] in ds.TRUST_MISLEADING_IDS:
            assert expected == "verify_and_do_not_follow_blindly"
            assert task["trust_calibration"]["scoring_key"] == "calibrated_skepticism"
        else:
            assert expected == "trust_after_local_confirmation"


def test_participant_packet_omits_scoring_keys(artifact):
    packet = ds.render_task_packet_json(artifact)
    text = json.dumps(packet, sort_keys=True)
    assert "scoring_key" not in text
    assert "expected_response" not in text
    assert len(packet["tasks"]) == artifact["summary"]["n_tasks"]
    assert {"time_to_localize_seconds", "fix_quality_score_0_to_3",
            "trust_calibration_choice"} == set(packet["outcomes_collected"])


def test_committed_artifacts_match_generator():
    assert ds.run(check=True)["summary"]["n_tasks"] >= 10
    committed = json.loads(ds.OUT_JSON.read_text(encoding="utf-8"))
    assert committed == ds.measure()
