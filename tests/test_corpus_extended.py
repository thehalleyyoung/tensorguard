"""Tests for the extended benchmark corpus (Step 101).

Covers:
* generator integrity (count, unique ids, valid Python, label/family sanity);
* ground-truth validation -- a sample of cases is executed against real torch
  and must behave as labeled (buggy raises matching substring; clean runs);
* manifest/content-addressing integrity (on-disk corpus matches a fresh build);
* the TensorGuard scoring artifact: no volatile fields, byte-deterministic,
  and the soundness invariant (zero false positives on clean code) holds.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases, family_counts  # noqa: E402

SCORE_JSON = REPO / "reproducibility" / "corpus_extended_score.json"
MANIFEST = REPO / "corpus_extended" / "manifest.json"
CASES_DIR = REPO / "corpus_extended" / "cases"

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


# --------------------------------------------------------------------------- #
# Generator integrity
# --------------------------------------------------------------------------- #
def test_corpus_is_large_and_balanced():
    cases = all_cases()
    assert len(cases) >= 200, f"expected >=200 cases, got {len(cases)}"
    n_buggy = sum(1 for c in cases if c.label == "buggy")
    n_clean = sum(1 for c in cases if c.label == "clean")
    assert n_buggy >= 100
    assert n_clean >= 50


def test_case_ids_are_unique():
    ids = [c.id for c in all_cases()]
    assert len(ids) == len(set(ids))


def test_every_source_is_valid_python():
    for c in all_cases():
        ast.parse(c.source)  # raises SyntaxError on bad indentation


def test_labels_and_families_are_sane():
    for c in all_cases():
        assert c.label in {"clean", "buggy"}
        assert c.family
        if c.label == "buggy":
            assert c.expected_error_substring
            assert c.expected_error_substring.strip()
        else:
            assert c.expected_error_substring is None


def test_multiple_families_present():
    fc = family_counts()
    assert len(fc) >= 8
    assert sum(fc.values()) == len(all_cases())


# --------------------------------------------------------------------------- #
# Ground-truth validation against real torch (a representative sample)
# --------------------------------------------------------------------------- #
def _run_forward(case):
    import torch

    ns: dict = {}
    exec(compile(case.source, f"<{case.id}>", "exec"), ns)
    module = ns["M"]()
    module.eval()
    args = [torch.randn(*shape) for shape in case.input_shapes.values()]
    with torch.no_grad():
        module(*args)


@pytest.mark.parametrize("label", ["buggy", "clean"])
def test_sample_cases_behave_as_labeled(label):
    torch = pytest.importorskip("torch")  # noqa: F841
    cases = [c for c in all_cases() if c.label == label]
    # Sample deterministically (every 17th) to keep the test fast but broad.
    sample = cases[::17] or cases[:1]
    for case in sample:
        if label == "clean":
            _run_forward(case)  # must not raise
        else:
            with pytest.raises(Exception) as ei:
                _run_forward(case)
            assert case.expected_error_substring in f"{type(ei.value).__name__}: {ei.value}"


# --------------------------------------------------------------------------- #
# Materialized corpus / content-addressing integrity
# --------------------------------------------------------------------------- #
def test_manifest_matches_generation():
    assert MANIFEST.exists(), "run `python -m corpus_extended.build` first"
    man = json.loads(MANIFEST.read_text())
    by_id = {e["id"]: e for e in man["items"]}
    cases = all_cases()
    assert man["meta"]["total"] == len(cases)
    import hashlib

    for c in cases:
        assert c.id in by_id, f"{c.id} missing from manifest"
        e = by_id[c.id]
        assert e["label"] == c.label
        assert e["expected_verdict"] == ("SAFE" if c.label == "clean" else "UNSAFE")
        digest = hashlib.sha256(c.source.encode("utf-8")).hexdigest()
        assert e["sha256"] == digest, f"hash drift in {c.id}"


def test_case_files_exist_and_match_hashes():
    man = json.loads(MANIFEST.read_text())
    import hashlib

    for e in man["items"]:
        path = CASES_DIR / f"{e['id']}.py"
        assert path.exists(), f"missing {path.name}"
        disk = path.read_text()
        assert hashlib.sha256(disk.encode("utf-8")).hexdigest() == e["sha256"]


# --------------------------------------------------------------------------- #
# Scoring artifact
# --------------------------------------------------------------------------- #
def test_score_artifact_has_no_volatile_fields():
    data = json.loads(SCORE_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_score_artifact_is_byte_deterministic():
    from reproducibility import corpus_extended_score as ces

    rc = ces.run(check=True)
    assert rc == 0


def test_sound_mode_has_no_false_positive():
    data = json.loads(SCORE_JSON.read_text())
    assert data["sound_mode_has_no_false_positive"] is True
    assert data["sound"]["confusion"]["fp"] == 0


def test_recall_is_reported_with_confidence_intervals():
    data = json.loads(SCORE_JSON.read_text())
    for mode in ("balanced", "sound"):
        ci = data[mode]["recall_on_all_buggy"]
        assert ci["n"] == data[mode]["n_buggy"]
        if ci["point"] is not None:
            assert 0.0 <= ci["low"] <= ci["point"] <= ci["high"] <= 1.0


def test_corpus_score_counts_every_case():
    data = json.loads(SCORE_JSON.read_text())
    for mode in ("balanced", "sound"):
        m = data[mode]
        conf = m["confusion"]
        total_accounted = (
            conf["tp"] + conf["fp"] + conf["tn"] + conf["fn"]
            + m["abstained_buggy"] + m["abstained_clean"]
        )
        assert total_accounted == m["n_total"] == len(all_cases())
