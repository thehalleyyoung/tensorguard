"""Tests for the held-out blind split + pre-registered evaluation (Step 105)."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from corpus_extended.blind_split import all_blind_cases  # noqa: E402
from corpus_extended.generators import all_cases  # noqa: E402

EVAL_JSON = REPO / "reproducibility" / "blind_split_eval.json"
BLIND_MANIFEST = REPO / "corpus_extended" / "blind_manifest.json"
BLIND_CASES = REPO / "corpus_extended" / "blind_cases"
PREREG = REPO / "corpus_extended" / "PRE_REGISTRATION.md"

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


def test_blind_split_disjoint_from_dev():
    blind = {c.id for c in all_blind_cases()}
    dev = {c.id for c in all_cases()}
    assert blind.isdisjoint(dev)


def test_blind_split_is_substantial():
    cases = all_blind_cases()
    assert len(cases) >= 150
    assert sum(1 for c in cases if c.label == "buggy") >= 100
    assert sum(1 for c in cases if c.label == "clean") >= 30


def test_blind_sources_parse():
    for c in all_blind_cases():
        ast.parse(c.source)


def test_blind_manifest_matches_generation():
    man = json.loads(BLIND_MANIFEST.read_text())
    by_id = {e["id"]: e for e in man["items"]}
    for c in all_blind_cases():
        assert c.id in by_id
        digest = hashlib.sha256(c.source.encode("utf-8")).hexdigest()
        assert by_id[c.id]["sha256"] == digest


def test_blind_case_files_match_hashes():
    man = json.loads(BLIND_MANIFEST.read_text())
    for e in man["items"]:
        path = BLIND_CASES / f"{e['id']}.py"
        assert path.exists()
        disk = path.read_text()
        assert hashlib.sha256(disk.encode("utf-8")).hexdigest() == e["sha256"]


def test_preregistered_sha_matches_manifest():
    # The committed pre-registration must name the exact frozen manifest hash.
    sha = hashlib.sha256(BLIND_MANIFEST.read_bytes()).hexdigest()
    text = PREREG.read_text()
    assert sha in text, "PRE_REGISTRATION.md does not match frozen manifest hash"


def test_blind_buggy_sample_raises_on_real_torch():
    pytest.importorskip("torch")
    import torch

    buggy = [c for c in all_blind_cases() if c.label == "buggy"][::13]
    for case in buggy:
        ns: dict = {}
        exec(compile(case.source, f"<{case.id}>", "exec"), ns)
        m = ns["M"]()
        m.eval()
        args = [torch.randn(*s) for s in case.input_shapes.values()]
        with pytest.raises(Exception) as ei:
            with torch.no_grad():
                m(*args)
        assert case.expected_error_substring in f"{type(ei.value).__name__}: {ei.value}"


def test_eval_artifact_no_volatile_fields():
    data = json.loads(EVAL_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_eval_artifact_is_byte_deterministic():
    from reproducibility import blind_split_eval as bse

    assert bse.run(check=True) == 0


def test_preregistered_hypotheses_confirmed():
    data = json.loads(EVAL_JSON.read_text())
    assert data["manifest_matches_registration"] is True
    assert data["all_modes_confirm_preregistration"] is True
    for mode in ("balanced", "sound"):
        m = data[mode]
        assert m["H1_no_false_positive"] is True
        assert m["H2_recall_floor_met"] is True
        assert m["H3_no_overfitting_gap"] is True
