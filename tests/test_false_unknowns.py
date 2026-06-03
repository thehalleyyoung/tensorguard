"""Step 15 regression tests -- sound mode should not over-abstain.

The false-UNKNOWN benchmark complements the zero-false-positive benchmark:
strict sound mode is allowed to abstain on unsupported programs, but it should
decide executable, ground-truthed cases that fall inside TensorGuard's supported
fragment.
"""

from __future__ import annotations

import os

import pytest

from evaluation import false_unknowns


@pytest.fixture(scope="module")
def artifact():
    return false_unknowns.run(check=False)


def test_corpus_has_clean_and_buggy_ground_truth():
    corpus = false_unknowns.build_corpus()
    assert len(corpus) >= 80
    assert {c["kind"] for c in corpus} == {"clean", "buggy"}
    assert sum(1 for c in corpus if c["expected_verdict"] == "UNSAFE") >= 6
    assert sum(1 for c in corpus if c["expected_verdict"] == "SAFE") >= 70


def test_buggy_cases_are_real_latent_faults():
    corpus = false_unknowns.build_corpus()
    buggy = [c for c in corpus if c["kind"] == "buggy"]
    assert buggy
    for case in buggy:
        assert case["source_kind"] == "latent_bug"
        assert case["ground_truth"].startswith(("eval:", "branch:"))


def test_sound_mode_has_zero_false_unknowns(artifact):
    summary = artifact["summary"]
    assert summary["false_unknowns"] == 0
    assert summary["false_unknown_rate"] == 0.0
    assert summary["decision_rate"] == 1.0
    assert artifact["false_unknown_ids"] == []


def test_sound_mode_decides_clean_and_buggy_cases(artifact):
    by_kind = artifact["by_kind"]
    assert by_kind["clean"]["UNKNOWN"] == 0
    assert by_kind["buggy"]["UNKNOWN"] == 0
    assert by_kind["clean"]["SAFE"] == by_kind["clean"]["total"]
    assert by_kind["buggy"]["UNSAFE"] == by_kind["buggy"]["total"]


def test_no_misclassifications_when_decided(artifact):
    assert artifact["summary"]["misclassified"] == 0
    assert artifact["misclassified_ids"] == []


def test_artifact_on_disk_is_up_to_date():
    assert os.path.exists(false_unknowns.OUT_JSON)
    false_unknowns.run(check=True)


def test_generator_is_deterministic():
    a = false_unknowns.build_corpus()
    b = false_unknowns.build_corpus()
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert [c["source"] for c in a] == [c["source"] for c in b]
