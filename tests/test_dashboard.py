"""Step 20 -- tests for the precision/recall regression dashboard gate.

Covers: metrics extracted from the committed artifacts match those artifacts;
the committed baseline passes `--check`; the gate logic detects synthetic
regressions (drop on higher_better, rise on lower_better, non-finite/missing
values, integer-exact comparison) and allows improvements; metric parity is
enforced in both directions (orphans + unregistered); and the markdown/baseline
regenerate byte-identically.
"""

from __future__ import annotations

import json
import os

import pytest

from evaluation import dashboard as D

EVAL = os.path.dirname(D.__file__)


# ---- extraction matches the underlying artifacts --------------------------
def test_compute_metrics_matches_artifacts():
    metrics = D.compute_metrics()
    # Every registered metric is produced.
    assert set(metrics) == {m.key for m in D.METRICS}
    # Spot-check a few against the raw artifacts.
    conf = json.load(open(os.path.join(EVAL, "confusion_matrices.json")))
    assert metrics["tg_precision"] == \
        conf["confusion"]["tensorguard"]["all"]["precision"]
    assert metrics["tg_false_negatives"] == \
        conf["confusion"]["tensorguard"]["all"]["FN"]
    neg = json.load(open(os.path.join(EVAL, "neg_fuzz.json")))
    assert metrics["neg_fuzz_recall"] == neg["summary"]["recall"]
    assert metrics["neg_fuzz_genuine_faults"] == neg["summary"]["genuine_faults"]


def test_every_metric_has_direction_and_kind():
    for m in D.METRICS:
        assert m.direction in ("higher_better", "lower_better")
        assert m.kind in ("quality", "integrity")
    # keys are unique
    keys = [m.key for m in D.METRICS]
    assert len(keys) == len(set(keys))


# ---- committed baseline passes --------------------------------------------
def test_committed_baseline_passes_gate():
    current = D.compute_metrics()
    baseline = D.load_baseline()
    result = D.gate(current, baseline)
    assert result.ok, (result.regressions, result.orphans, result.unregistered)


def test_check_mode_returns_zero():
    assert D.run(check=True) == 0


def test_baseline_and_md_are_byte_identical_on_regen():
    current = D.compute_metrics()
    baseline = D.build_baseline(current)
    on_disk = open(D.BASELINE_PATH).read()
    assert D._dumps(baseline) == on_disk
    md = D.render_markdown(current, D.load_baseline())
    assert md == open(D.MD_PATH).read()


# ---- gate logic: synthetic regressions ------------------------------------
def _baseline_from(metrics):
    return D.build_baseline(metrics)


def test_gate_detects_higher_better_drop():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    worse = dict(base_vals)
    worse["tg_recall"] = base_vals["tg_recall"] - 0.1
    result = D.gate(worse, baseline)
    assert not result.ok
    assert any("tg_recall" in r for r in result.regressions)


def test_gate_detects_lower_better_rise():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    worse = dict(base_vals)
    worse["tg_false_positives"] = base_vals["tg_false_positives"] + 1
    result = D.gate(worse, baseline)
    assert not result.ok
    assert any("tg_false_positives" in r for r in result.regressions)


def test_gate_detects_integrity_shrink():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    worse = dict(base_vals)
    worse["triage_regression_suite"] = base_vals["triage_regression_suite"] - 1
    result = D.gate(worse, baseline)
    assert not result.ok
    assert any("triage_regression_suite" in r for r in result.regressions)


def test_gate_allows_improvement():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    better = dict(base_vals)
    better["hard_recall_advantage"] = base_vals["hard_recall_advantage"] + 0.1
    better["tg_false_positives"] = max(0, base_vals["tg_false_positives"] - 1) \
        if base_vals["tg_false_positives"] > 0 else base_vals["tg_false_positives"]
    result = D.gate(better, baseline)
    assert result.ok, result.regressions


def test_gate_treats_non_finite_as_regression():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    for bad in (float("nan"), float("inf"), None):
        broken = dict(base_vals)
        broken["tg_precision"] = bad
        result = D.gate(broken, baseline)
        assert not result.ok


def test_gate_detects_orphans_and_unregistered():
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    # Orphan: baseline has a key the current run no longer produces.
    dropped = dict(base_vals)
    dropped.pop("tg_f1")
    result = D.gate(dropped, baseline)
    assert "tg_f1" in result.orphans
    assert not result.ok
    # Unregistered: current produces a key the baseline lacks.
    extra = dict(base_vals)
    extra["brand_new_metric"] = 1.0
    result2 = D.gate(extra, baseline)
    assert "brand_new_metric" in result2.unregistered
    assert not result2.ok


def test_integer_comparison_is_exact():
    # A lower_better integer count rising by exactly 1 must regress even though
    # the float epsilon is non-zero.
    base_vals = D.compute_metrics()
    baseline = _baseline_from(base_vals)
    worse = dict(base_vals)
    worse["hard_recall_tensorguard_misses"] = \
        base_vals["hard_recall_tensorguard_misses"] + 1
    result = D.gate(worse, baseline)
    assert any("hard_recall_tensorguard_misses" in r for r in result.regressions)


def test_is_regression_direction_validation():
    with pytest.raises(ValueError):
        D._is_regression("sideways", 1.0, 1.0)
