"""Tests for the natural-distribution clean-model study (Step 258)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STUDY_JSON = REPO / "reproducibility" / "natural_distribution_study.json"

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
    data = json.loads(STUDY_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import natural_distribution_study as nds

    assert nds.run(check=True) == 0


def test_sample_is_diverse():
    from corpus_extended.natural_models import all_models

    models = all_models()
    assert len(models) >= 150
    assert len({m.family for m in models}) >= 10
    assert len({m.repo_slug for m in models}) >= 8
    assert len({m.variant for m in models}) >= 5
    assert len({m.id for m in models}) == len(models)  # unique ids


def test_all_natural_models_execute_clean():
    # Every model in the sample must actually run under real eager PyTorch,
    # otherwise it is not a clean natural model and the coverage claim is void.
    import torch

    from corpus_extended.natural_models import all_models

    for m in all_models():
        ns = {}
        exec(compile(m.source, f"<{m.id}>", "exec"), ns)
        net = ns["Net"]()
        net.eval()
        inputs = []
        for _, shape in m.input_shapes.items():
            if m.family == "embedding":
                inputs.append(torch.randint(0, 100, tuple(shape)))
            else:
                inputs.append(torch.randn(*shape))
        with torch.no_grad():
            net(*inputs)  # must not raise


def test_full_coverage_and_zero_false_alarms():
    data = json.loads(STUDY_JSON.read_text())
    assert data["step"] == 258
    assert data["full_coverage_all_modes"] is True
    assert data["zero_false_alarms_all_modes"] is True
    for mode, d in data["per_mode"].items():
        assert d["n_abstained"] == 0, f"{mode} abstained"
        assert d["n_false_alarms"] == 0, f"{mode} false-alarmed"
        assert d["false_alarm_ids"] == []
        assert d["abstention_causes"] == {}
        assert d["abstention_examples"] == {}


def test_coverage_wilson_interval_present():
    data = json.loads(STUDY_JSON.read_text())
    for mode, d in data["per_mode"].items():
        cov = d["coverage"]
        assert cov["point"] == 1.0
        assert 0.0 <= cov["low"] <= cov["high"] <= 1.0
        assert cov["n"] == data["n_models"]
        assert d["false_alarm_upper_bound_95"] == d["false_alarm_rate"]["high"]
        assert 0.0 < d["false_alarm_upper_bound_95"] < 0.03


def test_modes_cover_all_three():
    data = json.loads(STUDY_JSON.read_text())
    assert set(data["modes"]) == {"sound", "balanced", "heuristic"}


def test_public_repo_strata_and_source_policy_are_explicit():
    data = json.loads(STUDY_JSON.read_text())
    assert data["n_repo_strata"] >= 8
    assert "pytorch/vision" in data["repo_strata"]
    assert "huggingface/transformers" in data["repo_strata"]
    assert "not vendored third-party source files" in data["redistribution_policy"]
