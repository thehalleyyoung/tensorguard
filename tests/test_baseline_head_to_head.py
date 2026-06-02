"""Tests for the head-to-head baseline comparison (Step 110)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

H2H_JSON = REPO / "reproducibility" / "baseline_head_to_head.json"

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
    data = json.loads(H2H_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import baseline_head_to_head as h2h

    assert h2h.run(check=True) == 0


def test_subset_covers_all_families():
    data = json.loads(H2H_JSON.read_text())
    assert len(data["families_covered"]) >= 9
    assert data["subset_buggy"] >= 12
    assert data["subset_clean"] >= 6


def test_tensorguard_catches_all_subset_no_fp():
    data = json.loads(H2H_JSON.read_text())
    tg = data["tools"]["tensorguard"]
    assert tg["buggy_caught"] == tg["buggy_total"]
    assert tg["clean_false_alarms"] == 0
    assert tg["static_no_execution"] is True
    assert tg["needs_concrete_inputs"] is False


def test_torch_export_is_dynamic_baseline():
    data = json.loads(H2H_JSON.read_text())
    te = data["tools"]["torch_export_trace"]
    # It catches the bugs too, but it is NOT static and needs inputs.
    assert te["buggy_caught"] == te["buggy_total"]
    assert te["static_no_execution"] is False
    assert te["needs_concrete_inputs"] is True


def test_mypy_catches_zero_shape_bugs():
    data = json.loads(H2H_JSON.read_text())
    assert data["mypy_catches_zero_shape_bugs"] is True
    assert data["tools"]["mypy"]["buggy_caught"] == 0


def test_tensorguard_unique_static_input_free_complete():
    data = json.loads(H2H_JSON.read_text())
    assert data["tensorguard_is_unique_static_input_free_complete"] is True
    assert data["static_input_free_complete_tools"] == ["tensorguard"]


def test_full_corpus_tensorguard_complete_no_fp():
    data = json.loads(H2H_JSON.read_text())
    f = data["tensorguard_full_corpus"]
    assert f["buggy_caught"] == f["buggy_total"]
    assert f["clean_false_alarms"] == 0
