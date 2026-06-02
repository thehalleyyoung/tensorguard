"""Tests for the per-release benchmark-delta tool (``scripts/benchmark_delta.py``).

Includes a real-artifact smoke test: the committed headline 60-bug figure is
compared against a perturbed copy and the regression direction is asserted.
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.benchmark_delta import (
    compute_delta,
    flatten,
    main,
    render_markdown,
)

REPRO = os.path.join(os.path.dirname(__file__), "..", "reproducibility")


def test_flatten_numeric_leaves():
    flat = flatten({"a": {"b": 3, "c": [1, 2]}, "d": True, "s": "x"})
    assert flat["a.b"] == 3.0
    assert flat["a.c[0]"] == 1.0
    assert flat["d"] == 1.0
    assert "s" not in flat  # strings dropped


def test_recall_regression_detected():
    old = {"recall": 0.90, "false_positive": 2}
    new = {"recall": 0.80, "false_positive": 2}
    r = compute_delta(old, new)
    assert r.has_regression()
    assert any(d.path == "recall" for d in r.regressions)


def test_recall_improvement_not_regression():
    old = {"recall": 0.80}
    new = {"recall": 0.95}
    r = compute_delta(old, new)
    assert not r.has_regression()
    assert any(d.path == "recall" for d in r.improvements)


def test_lower_is_better_silent_miss():
    old = {"headline.silent_miss": 7}
    new = {"headline.silent_miss": 12}
    r = compute_delta(old, new)
    assert r.has_regression()


def test_silent_miss_decrease_is_improvement():
    r = compute_delta({"headline.silent_miss": 12}, {"headline.silent_miss": 7})
    assert not r.has_regression()
    assert r.improvements


def test_unknown_metric_never_gates():
    r = compute_delta({"weird_counter": 1}, {"weird_counter": 999})
    assert not r.has_regression()


def test_walltime_within_tolerance_not_regression():
    # elapsed_s has an enormous tolerance: machine-dependent, never gates.
    r = compute_delta({"meta.elapsed_s": 2.0}, {"meta.elapsed_s": 50.0})
    assert not r.has_regression()


def test_render_markdown_lists_regression():
    r = compute_delta({"recall": 0.9}, {"recall": 0.5})
    md = render_markdown(r, "old.json", "new.json")
    assert "regressed" in md
    assert "`recall`" in md


def test_main_exit_codes(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text(json.dumps({"recall": 0.9}))
    new.write_text(json.dumps({"recall": 0.9}))
    assert main([str(old), str(new), "--json"]) == 0
    new.write_text(json.dumps({"recall": 0.4}))
    assert main([str(old), str(new), "--json"]) == 1


def test_real_headline_artifact_perturbation(tmp_path):
    path = os.path.join(REPRO, "reproduce_headline_60bug.json")
    if not os.path.exists(path):
        pytest.skip("headline artifact not present")
    with open(path) as fh:
        data = json.load(fh)
    # identical comparison: no regression
    assert not compute_delta(data, data).has_regression()
    # perturb the headline catch count downward → must regress
    worse = json.loads(json.dumps(data))
    hr = worse.get("headline_regime", {})
    if "refuted_proof_high_confidence" in hr:
        hr["refuted_proof_high_confidence"] -= 5
        hr["silent_miss"] = hr.get("silent_miss", 0) + 5
        r = compute_delta(data, worse)
        assert r.has_regression()
