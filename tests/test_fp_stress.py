"""Tests for the false-positive stress test (Step 111)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FP_JSON = REPO / "reproducibility" / "fp_stress_eval.json"

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
    data = json.loads(FP_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import fp_stress_eval as fse

    assert fse.run(check=True) == 0


def test_corpus_is_large_and_diverse():
    from corpus_extended.fp_stress import all_models

    models = all_models()
    assert len(models) >= 100
    assert len({m.id for m in models}) == len(models)  # unique ids
    assert len({m.family for m in models}) >= 10


def test_all_stress_models_execute_clean():
    # Every stress model must actually run under eager PyTorch, otherwise the
    # false-alarm claim is meaningless.
    import torch

    from corpus_extended.fp_stress import all_models

    for m in all_models():
        ns = {}
        exec(compile(m.source, f"<{m.id}>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
        inputs = [torch.randn(*s) for s in m.input_shapes.values()]
        with torch.no_grad():
            net(*inputs)  # must not raise


def test_zero_false_alarms_all_modes():
    data = json.loads(FP_JSON.read_text())
    assert data["zero_false_alarms_sound_mode"] is True
    assert data["zero_false_alarms_all_modes"] is True
    for mode, d in data["per_mode"].items():
        assert d["n_false_alarms"] == 0, f"{mode} false-alarmed"
        assert d["false_alarm_ids"] == []


def test_corpus_size_at_least_100():
    data = json.loads(FP_JSON.read_text())
    assert data["corpus_size_at_least_100"] is True
    assert data["n_models"] >= 100


def test_false_alarm_wilson_interval_present():
    data = json.loads(FP_JSON.read_text())
    for mode, d in data["per_mode"].items():
        fr = d["false_alarm_rate"]
        assert fr["point"] == 0.0
        assert fr["low"] == 0.0
        assert 0.0 <= fr["high"] <= 1.0
        assert fr["n"] == data["n_models"]
