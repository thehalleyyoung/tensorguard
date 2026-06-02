"""Step 36 -- tests for the torch.fx frontend trace-success harness."""

from __future__ import annotations

import json
import os

import pytest

from evaluation import fx_trace_success as TS

torchvision = pytest.importorskip("torchvision")


def test_small_model_traces_and_lowers():
    rec = TS._eval_torchvision("resnet18")
    assert rec["traced"] and rec["lowered"]
    assert rec["steps"] > 0
    assert rec["error"] is None


def test_eval_never_raises_on_bad_name():
    # A non-existent attribute is reported as a construct error, not an exception.
    rec = TS._eval_torchvision("definitely_not_a_model_xyz")
    assert rec["traced"] is False and rec["lowered"] is False
    assert rec["error"] is not None


def test_summary_consistency():
    records = [
        {"model": "a", "traced": True, "lowered": True, "steps": 10,
         "unsupported_ops": 2, "error": None},
        {"model": "b", "traced": True, "lowered": False, "steps": 0,
         "unsupported_ops": 0, "error": "lower: X"},
    ]
    s = TS._summarise(records)
    assert s["n_models"] == 2
    assert s["traced"] == 2 and s["lowered"] == 1 and s["succeeded"] == 1
    assert s["trace_success_rate"] == 0.5
    assert s["trace_success_percent"] == 50.0
    assert s["total_steps"] == 10 and s["total_unsupported_ops"] == 2


def test_committed_artifact_is_fresh_or_qualified():
    # Mirrors `--check`: version-gated byte-identical check, else QUALIFIED skip.
    assert os.path.exists(TS.JSON_PATH)
    assert TS.run(check=True) == 0


def test_gate_passes_or_qualified():
    assert TS.gate() == 0


def test_committed_corpus_has_full_success():
    rep = json.load(open(TS.JSON_PATH))
    assert rep["summary"]["trace_success_rate"] == 1.0
    assert rep["summary"]["succeeded"] == rep["summary"]["n_models"]
