"""Step 54 -- Criterion-style latency regression benchmark.

Tests the pure regression-detection logic (deterministic, no timing), the
committed baseline's well-formedness, and that the live gate passes on HEAD.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evaluation import regression_bench as rb  # noqa: E402


def _doc(ratios):
    return {
        "meta": {"tolerance": 0.10, "samples": rb.SAMPLES},
        "cases": {name: {"steps": 1, "baseline_ratio": r}
                  for name, r in ratios.items()},
    }


def test_evaluate_no_regression_when_within_tolerance():
    doc = _doc({"a": 10.0, "b": 20.0})
    # 9 percent slower: under the 10 percent budget -> no regression.
    fresh = {"a": 10.9, "b": 21.8}
    assert rb.evaluate(doc, fresh, 0.10) == []


def test_evaluate_flags_regression_over_tolerance():
    doc = _doc({"a": 10.0})
    fresh = {"a": 11.5}  # 15 percent slower
    regs = rb.evaluate(doc, fresh, 0.10)
    assert len(regs) == 1
    assert "a:" in regs[0]


def test_evaluate_exactly_at_limit_is_ok():
    doc = _doc({"a": 10.0})
    fresh = {"a": 11.0}  # exactly +10 percent
    assert rb.evaluate(doc, fresh, 0.10) == []


def test_evaluate_speedup_never_flags():
    doc = _doc({"a": 10.0, "b": 5.0})
    fresh = {"a": 6.0, "b": 1.0}  # much faster
    assert rb.evaluate(doc, fresh, 0.10) == []


def test_evaluate_missing_case_is_a_regression():
    doc = _doc({"a": 10.0})
    assert len(rb.evaluate(doc, {}, 0.10)) == 1


def test_baseline_document_is_deterministic_structure():
    ratios = {"stack_6": 3.0, "stack_12": 11.0, "stack_24": 40.0}
    d1 = rb.baseline_document(ratios)
    d2 = rb.baseline_document(ratios)
    assert rb._dumps(d1) == rb._dumps(d2)
    assert set(d1["cases"]) == {name for name, _s, _sh in rb.cases()}
    assert d1["meta"]["tolerance"] == rb.TOLERANCE


def test_committed_baseline_check_passes():
    assert rb.check() == 0


def test_markdown_in_sync_with_committed_json():
    doc = rb._load_baseline()
    md = rb.render_markdown(doc)
    with open(rb.MD_PATH) as fh:
        assert fh.read() == md


def test_live_gate_passes_on_head():
    # Re-measures (slow); on HEAD nothing should regress past the baseline.
    assert rb.gate() == 0
