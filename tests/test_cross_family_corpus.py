"""Step 168 — cross-family, runtime-grounded regression corpus.

These tests prove TensorGuard's verdicts generalise across six architecture
families (MLP, CNN, attention, normalization, residual, matmul) *and* that the
corpus is honest evidence rather than a rigged demo: every case is re-validated
against eager PyTorch on each run (clean executes; buggy raises the declared
``RuntimeError`` substring), and only then is the static verdict scored.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmarks.cross_family_corpus import CASES, evaluate  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return evaluate()


def test_spans_at_least_six_families(report):
    assert report["meta"]["n_families"] >= 6
    # both polarities present in the cross-family set
    labels = {c["label"] for c in CASES.values()}
    assert labels == {"clean", "buggy"}


def test_runtime_ground_truth_holds_for_every_case(report):
    """Clean cases run; buggy cases raise the declared error substring."""
    bad = [
        c
        for c in report["cases"]
        if not c["runtime_ground_truth_ok"]
    ]
    assert not bad, "runtime ground truth mismatch: " + ", ".join(
        f"{c['id']}({c['runtime_error']})" for c in bad
    )
    assert report["meta"]["all_runtime_ground_truth_ok"] is True


def test_perfect_recall_and_zero_false_positives(report):
    sc = report["scorecard"]
    assert sc["recall"] == 1.0, [c for c in report["cases"] if not c["tg_correct"]]
    assert sc["fp"] == 0, [
        c for c in report["cases"] if c["label"] == "clean" and c["tg_verdict"] == "UNSAFE"
    ]
    assert sc["precision"] == 1.0


def test_every_family_has_a_caught_bug(report):
    """No family is silently uncovered: each contributes >=1 true positive."""
    for family, counts in report["by_family"].items():
        if counts["tp"] + counts["fn"] > 0:  # family contains a buggy case
            assert counts["tp"] >= 1, f"{family}: a bug went uncaught"
            assert counts["fn"] == 0, f"{family}: false negative"
        assert counts["fp"] == 0, f"{family}: false positive on a clean case"


def test_corpus_is_content_addressed_and_deterministic(report):
    """Hashes are stable across runs (frozen corpus)."""
    again = evaluate()
    h1 = {c["id"]: c["sha256"] for c in report["cases"]}
    h2 = {c["id"]: c["sha256"] for c in again["cases"]}
    assert h1 == h2
    assert all(len(h) == 64 for h in h1.values())
