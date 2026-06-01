"""Regression tests for the precision/recall evaluation harness.

These lock in the committed confusion matrices (`evaluation/confusion_matrices.json`)
and re-derive the deterministic, dependency-light methods (TensorGuard, the two
runtime baselines, and the no-op floor) live to prove the artifact is faithful.
The PyTea rows are re-checked only when a `node` toolchain is available.
"""

from __future__ import annotations

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation import precision_recall as pr  # noqa: E402
from real_benchmarks import load  # noqa: E402


@pytest.fixture(scope="module")
def committed():
    with open(pr.OUT_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def items():
    return load.load_items()


# --------------------------------------------------------------------------
# Committed artifact: exact confusion matrices (the headline result)
# --------------------------------------------------------------------------
EXPECTED_ALL = {
    # method: (TP, FP, TN, FN)
    "tensorguard": (8, 0, 8, 0),
    "runtime_forward": (7, 0, 8, 1),
    "runtime_backward": (8, 0, 8, 0),
    "pytea": (6, 0, 8, 2),
    "noop": (0, 0, 8, 8),
}


def test_committed_full_corpus_matrices(committed):
    for method, (tp, fp, tn, fn) in EXPECTED_ALL.items():
        c = committed["confusion"][method]["all"]
        assert (c["TP"], c["FP"], c["TN"], c["FN"]) == (tp, fp, tn, fn), method


def test_committed_meta_corpus_is_balanced(committed):
    meta = committed["meta"]
    assert meta["n_models"] == 16
    assert meta["n_clean"] == 8
    assert meta["n_buggy"] == 8
    assert meta["methods"] == pr.METHODS


def test_tensorguard_dominates_every_baseline(committed):
    """TensorGuard has the best (or tied-best) recall and zero false positives."""
    tg = committed["confusion"]["tensorguard"]["all"]
    assert tg["FP"] == 0
    assert tg["recall"] == 1.0
    assert tg["precision"] == 1.0
    for method in pr.METHODS:
        if method == "tensorguard":
            continue
        other = committed["confusion"][method]["all"]
        assert tg["recall"] >= (other["recall"] or 0.0), method


def test_runtime_forward_misses_only_the_silent_gradient_bug(committed):
    """The forward-only runtime baseline's single miss is the gradient domain."""
    misses = [
        m["id"] for m in committed["per_model"]
        if m["label"] == "buggy"
        and m["predictions"]["runtime_forward"]["pred"] != "buggy"
    ]
    assert len(misses) == 1
    miss = next(m for m in committed["per_model"] if m["id"] == misses[0])
    assert miss["domain"] == "gradient"


def test_pytea_misses_device_and_gradient_only(committed):
    misses = sorted(
        m["domain"] for m in committed["per_model"]
        if m["label"] == "buggy"
        and m["predictions"]["pytea"]["pred"] != "buggy"
    )
    assert misses == ["device", "gradient"]


def test_shape_only_subcorpus_is_a_tie(committed):
    """Every real analyser ties on pure shape bugs (fair apples-to-apples)."""
    for method in ("tensorguard", "runtime_forward", "runtime_backward", "pytea"):
        c = committed["confusion"][method]["shape_only"]
        assert c["recall"] == 1.0, method
        assert c["FP"] == 0, method


def test_capabilities_only_tensorguard_is_static_and_full_domain(committed):
    caps = committed["capabilities"]
    assert caps["tensorguard"]["static"] is True
    assert caps["tensorguard"]["sound_mode_available"] is True
    assert set(caps["tensorguard"]["domains"]) == {
        "shape", "device", "phase", "gradient"
    }
    # No baseline is simultaneously static AND covers all four domains.
    for method in ("runtime_forward", "runtime_backward", "pytea", "noop"):
        cap = caps[method]
        full = set(cap["domains"]) >= {"shape", "device", "phase", "gradient"}
        assert not (cap["static"] and full), method


# --------------------------------------------------------------------------
# Live re-derivation of the deterministic, node-free methods
# --------------------------------------------------------------------------
DETERMINISTIC_METHODS = [
    "tensorguard", "runtime_forward", "runtime_backward", "noop",
]


@pytest.mark.parametrize("method", DETERMINISTIC_METHODS)
def test_live_predictions_match_committed(method, committed, items):
    committed_by_id = {m["id"]: m for m in committed["per_model"]}
    for item in items:
        pred, _detail = pr.PREDICTORS[method](item)
        expected = committed_by_id[item["id"]]["predictions"][method]["pred"]
        assert pred == expected, (method, item["id"])


def test_live_predictions_are_repeatable(items):
    """Two runs of a runtime baseline give identical verdicts (seeded)."""
    first = [pr.predict_runtime_backward(it)[0] for it in items]
    second = [pr.predict_runtime_backward(it)[0] for it in items]
    assert first == second


# --------------------------------------------------------------------------
# PyTea (guarded on node availability)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not pr.pytea_available(),
                    reason="node/pytea toolchain not available")
def test_live_pytea_matches_committed(committed, items):
    committed_by_id = {m["id"]: m for m in committed["per_model"]}
    for item in items:
        pred, _detail = pr.predict_pytea(item)
        expected = committed_by_id[item["id"]]["predictions"]["pytea"]["pred"]
        assert pred == expected, item["id"]


# --------------------------------------------------------------------------
# Artifact is regenerable byte-for-byte (guarded on node availability)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not pr.pytea_available(),
                    reason="node/pytea toolchain not available")
def test_artifact_regenerates_identically():
    with open(pr.OUT_JSON, "r", encoding="utf-8") as fh:
        before = fh.read()
    pr.run(check=True)  # raises SystemExit if stale
    with open(pr.OUT_JSON, "r", encoding="utf-8") as fh:
        after = fh.read()
    assert before == after


def test_confusion_math_handles_na_two_ways():
    """NA is a miss in the headline view but excluded from covered-only."""
    rows = [("buggy", "na"), ("clean", "na"), ("buggy", "buggy"), ("clean", "clean")]
    c = pr.confusion(rows)
    assert c["NA"] == 2
    assert c["FN"] == 1  # na on buggy -> miss
    assert c["FP"] == 1  # na on clean -> spurious
    assert c["coverage"] == 0.5
    assert c["covered_only"]["TP"] == 1
    assert c["covered_only"]["TN"] == 1
    assert c["covered_only"]["FP"] == 0
    assert c["covered_only"]["FN"] == 0
