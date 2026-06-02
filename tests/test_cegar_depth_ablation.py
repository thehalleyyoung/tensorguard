"""Tests for the Step 119 CEGAR refinement-depth ablation."""

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module("reproducibility.cegar_depth_ablation")


@pytest.fixture(scope="module")
def data(mod):
    return mod.measure()


def test_no_volatile_keys_in_deterministic_artifact(data):
    for k in _walk_keys(data):
        assert not any(s in k.lower() for s in _VOLATILE), f"volatile key: {k}"


def test_corpus_is_real_torch_validated():
    from corpus_extended.cegar_depth_corpus import build_corpus, cegar_depth_validate

    cases = build_corpus()
    cegar_depth_validate(cases)  # raises if any label disagrees with torch
    assert sum(c.family == "conflict" for c in cases) >= 12
    assert sum(c.family == "clean" for c in cases) >= 6


def test_detection_is_depth_invariant_and_no_false_alarms(data):
    # CEGAR depth is a diagnosis knob, not a soundness knob.
    assert data["recall_is_depth_invariant_full"] is True
    assert data["zero_false_alarms_all_depths"] is True
    for r in data["per_depth"]:
        assert r["bugs_detected"] == r["n_conflict"], r["depth"]
        assert r["clean_false_alarms"] == 0, r["depth"]


def test_diagnostic_precision_rises_then_plateaus(data):
    assert data["refined_diagnoses_at_depth_0"] == 0
    assert data["precision_knee_depth"] is not None
    assert data["refined_diagnoses_at_knee"] == data["n_conflict_cases"]
    assert data["precision_rises_then_plateaus"] is True
    # Below the knee no contract diagnosis; at/after the knee, full.
    knee = data["precision_knee_depth"]
    for r in data["per_depth"]:
        if r["depth"] < knee:
            assert r["refined_contract_diagnoses"] < data["n_conflict_cases"]
        else:
            assert r["refined_contract_diagnoses"] == data["n_conflict_cases"]


def test_work_saturates_at_convergence(data):
    assert data["work_saturates_at_convergence"] is True
    iters = [r["total_cegar_iterations"] for r in data["per_depth"]]
    # Non-decreasing then flat (monotone Houdini accumulation, self-terminating).
    assert iters == sorted(iters)
    assert iters[-1] == data["max_total_iterations"]
    sat = data["work_saturation_depth"]
    # Everything at/after saturation is identical work.
    plateau = [r["total_cegar_iterations"] for r in data["per_depth"] if r["depth"] >= sat]
    assert len(set(plateau)) == 1


def test_artifact_byte_deterministic(mod):
    assert mod.run(check=True) == 0
