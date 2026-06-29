"""Steps 71 & 72 — the curated symexec corpus, run end to end.

This is the data-driven counterpart to the per-detector unit tests: it runs the
symbolic-execution engine over the trimmed-repro corpus under
``tests/symexec_corpus/`` and checks each file against its ``manifest.json``
expectation.

* ``wild/`` files (Step 71 — regression corpus) must fire **exactly one** report
  of the named kind, at the pinned line — proving the engine keeps catching each
  real-world / representative defect class.
* ``correct/`` files (Step 72 — soundness corpus) must produce **zero** reports —
  the no-false-positive guarantee, including cases the engine must stay silent on
  by abstaining.

The corpus is additive: a new ``.py`` plus a ``manifest.json`` entry is picked up
automatically, with no edits here.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from src.symexec import analyze_source

_CORPUS = pathlib.Path(__file__).resolve().parent / "symexec_corpus"
_MANIFEST = json.loads((_CORPUS / "manifest.json").read_text(encoding="utf-8"))


def _kinds(result):
    return [b.kind.value for b in result.bugs]


# -- the manifest and corpus directory agree ----------------------------


def test_every_wild_file_has_a_manifest_entry():
    on_disk = {p.name for p in (_CORPUS / "wild").glob("*.py")}
    in_manifest = set(_MANIFEST["wild"])
    assert on_disk == in_manifest


def test_every_correct_file_has_a_manifest_entry():
    on_disk = {p.name for p in (_CORPUS / "correct").glob("*.py")}
    in_manifest = set(_MANIFEST["correct"])
    assert on_disk == in_manifest


def test_corpus_is_non_trivial():
    # Guard against an empty/half-deleted corpus silently passing.
    assert len(_MANIFEST["wild"]) >= 9
    assert len(_MANIFEST["correct"]) >= 10


# -- Step 71: wild repros each fire their expected bug ------------------


@pytest.mark.parametrize("name", sorted(_MANIFEST["wild"]))
def test_wild_repro_fires_expected_bug(name):
    spec = _MANIFEST["wild"][name]
    path = _CORPUS / "wild" / name
    result = analyze_source(path.read_text(encoding="utf-8"), str(path))

    if spec["expect"] == "parse_error":
        assert len(result.bugs) == 1
        bug = result.bugs[0]
        assert "does not parse" in bug.message
        assert bug.confidence == 1.0
        return

    assert spec["expect"] == "kind"
    kinds = _kinds(result)
    assert spec["kind"] in kinds, f"{name}: expected {spec['kind']}, got {kinds}"
    matching = [b for b in result.bugs if b.kind.value == spec["kind"]]
    # exactly one report of the expected kind (a faithful, single-defect repro)
    assert len(matching) == 1
    if "line" in spec:
        assert matching[0].line == spec["line"], (
            f"{name}: {spec['kind']} expected at line {spec['line']}, "
            f"got {matching[0].line}"
        )


# -- Step 72: correct models stay silent (no false positives) ----------


@pytest.mark.parametrize("name", sorted(_MANIFEST["correct"]))
def test_correct_model_is_silent(name):
    path = _CORPUS / "correct" / name
    result = analyze_source(path.read_text(encoding="utf-8"), str(path))
    assert result.bugs == [], f"{name}: expected zero reports, got {_kinds(result)}"


# -- determinism over the whole corpus ----------------------------------


def test_corpus_reports_are_deterministic():
    for name in _MANIFEST["wild"]:
        path = _CORPUS / "wild" / name
        src = path.read_text(encoding="utf-8")
        a = analyze_source(src, str(path)).fingerprint()
        b = analyze_source(src, str(path)).fingerprint()
        assert a == b
