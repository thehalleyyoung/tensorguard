"""Tests for stratified per-class metrics with Wilson CIs (Step 104)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases  # noqa: E402

STRAT_JSON = REPO / "reproducibility" / "corpus_stratified.json"

_VOLATILE = ("time", "elapsed", "timestamp", "wall", "clock",
             "_ms", "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_artifact_no_volatile_fields():
    data = json.loads(STRAT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import corpus_stratified as cs

    assert cs.run(check=True) == 0


def test_every_buggy_class_present():
    data = json.loads(STRAT_JSON.read_text())
    buggy_families = {c.family for c in all_cases() if c.label == "buggy"}
    for mode in ("balanced", "sound"):
        got = set(data[mode]["per_class_recall"].keys())
        assert got == buggy_families


def test_every_clean_class_present():
    data = json.loads(STRAT_JSON.read_text())
    clean_families = {c.family for c in all_cases() if c.label == "clean"}
    for mode in ("balanced", "sound"):
        got = set(data[mode]["per_class_specificity"].keys())
        assert got == clean_families


def test_wilson_ci_bounds_are_ordered():
    data = json.loads(STRAT_JSON.read_text())
    for mode in ("balanced", "sound"):
        for v in data[mode]["per_class_recall"].values():
            ci = v["recall"]
            if ci["point"] is not None:
                assert 0.0 <= ci["low"] <= ci["point"] <= ci["high"] <= 1.0
        for v in data[mode]["per_class_specificity"].values():
            ci = v["specificity"]
            if ci["point"] is not None:
                assert 0.0 <= ci["low"] <= ci["point"] <= ci["high"] <= 1.0


def test_macro_averages_present_and_bounded():
    data = json.loads(STRAT_JSON.read_text())
    for mode in ("balanced", "sound"):
        mr = data[mode]["macro_recall"]
        ms = data[mode]["macro_specificity"]
        assert mr is None or 0.0 <= mr <= 1.0
        assert ms is None or 0.0 <= ms <= 1.0


def test_no_false_positive_per_clean_class_in_sound_mode():
    data = json.loads(STRAT_JSON.read_text())
    assert data["sound"]["every_clean_class_no_false_positive"] is True


def test_worst_class_recall_reported():
    data = json.loads(STRAT_JSON.read_text())
    for mode in ("balanced", "sound"):
        w = data[mode]["worst_class_recall"]
        assert w["family"] is not None
        assert w["recall_low"] is not None


def test_per_class_decided_counts_consistent():
    data = json.loads(STRAT_JSON.read_text())
    for mode in ("balanced", "sound"):
        for v in data[mode]["per_class_recall"].values():
            assert v["caught"] <= v["decided"] <= v["total"]
        for v in data[mode]["per_class_specificity"].values():
            assert v["true_negative"] <= v["decided"] <= v["total"]
