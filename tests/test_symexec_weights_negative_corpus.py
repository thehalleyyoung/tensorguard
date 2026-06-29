"""Roadmap step 4 — the **negative corpus** of known-bad checkpoints locks the
certifier's detector behavior.

``tests/data/weights_bad/`` holds one hand-built malformed ``*.safetensors`` file
per ``WeightsFinding`` kind plus a ``manifest.json`` mapping each file to the
*exact* set of findings it must produce.  These tests assert:

* every bad file is rejected (never ``proven_safe``) with **exactly** its manifest
  finding kinds, and the file's *named* kind is among them;
* the corpus covers **all 15** finding kinds emitted anywhere in
  ``src/symexec/weights.py`` (one file per kind) — no kind is left untested;
* the on-disk corpus is **byte-identical** to what the deterministic generator
  produces (so the committed artifacts are reproducible and locked);
* certificates are deterministic (re-certifying gives the same structural
  fingerprint), and the directory contains no stray files.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _weights_bad_gen as gen  # noqa: E402

from src.symexec.weights import certify_weights_file  # noqa: E402

_DATA_DIR = _HERE / "data" / "weights_bad"
_WEIGHTS_PY = _HERE.parent / "src" / "symexec" / "weights.py"


def _load_manifest() -> dict:
    with open(_DATA_DIR / "manifest.json") as fh:
        return json.load(fh)


def _code_finding_kinds() -> set[str]:
    """Every WeightsFinding kind constructed in weights.py (AST source of truth)."""
    tree = ast.parse(_WEIGHTS_PY.read_text())
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "WeightsFinding"):
            if node.args and isinstance(node.args[0], ast.Constant):
                kinds.add(node.args[0].value)
            for kw in node.keywords:
                if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                    kinds.add(kw.value.value)
    return kinds


def _certify(entry: dict):
    path = str(_DATA_DIR / entry["file"])
    exp = gen.expected_contract(entry["expected"])
    return certify_weights_file(
        path, check_finite=entry["check_finite"],
        expected=exp, contract_partial=entry["partial"],
    )


# --------------------------------------------------------------------------- #
# Corpus presence + structure.                                                  #
# --------------------------------------------------------------------------- #
def test_corpus_dir_and_manifest_exist():
    assert _DATA_DIR.is_dir(), f"missing negative corpus dir {_DATA_DIR}"
    assert (_DATA_DIR / "manifest.json").is_file()
    entries = _load_manifest()["entries"]
    assert len(entries) == 15
    for e in entries:
        assert (_DATA_DIR / e["file"]).is_file(), f"missing {e['file']}"


def test_one_file_per_finding_kind_covers_all_15():
    entries = _load_manifest()["entries"]
    named = {e["named_kind"] for e in entries}
    code = _code_finding_kinds()
    assert len(code) == 15
    assert named == code, (
        f"corpus named-kind coverage mismatch:\n"
        f"  missing files for: {sorted(code - named)}\n"
        f"  stale corpus kinds: {sorted(named - code)}"
    )
    # Filenames are exactly <kind>.safetensors.
    for e in entries:
        assert e["file"] == f"{e['named_kind']}.safetensors"


def test_no_stray_files_in_corpus():
    on_disk = {p.name for p in _DATA_DIR.iterdir()}
    expected = {e["file"] for e in _load_manifest()["entries"]} | {"manifest.json"}
    assert on_disk == expected, f"unexpected files: {on_disk ^ expected}"


# --------------------------------------------------------------------------- #
# The core lock: each bad file yields exactly its expected findings.            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", _load_manifest()["entries"],
                         ids=lambda e: e["named_kind"])
def test_bad_file_yields_exactly_expected_findings(entry):
    cert = _certify(entry)
    # Never safe.
    assert not cert.proven_safe, f"{entry['file']} was wrongly certified safe"
    # Exactly the manifest's finding kinds.
    assert sorted(cert.finding_kinds) == entry["kinds"], (
        f"{entry['file']}: expected {entry['kinds']} got {sorted(cert.finding_kinds)}"
    )
    # The file's named kind is among them.
    assert entry["named_kind"] in cert.finding_kinds
    # At least one finding carries detail + the offending tensor name where applicable.
    assert cert.findings
    for f in cert.findings:
        assert isinstance(f.detail, str) and f.detail


def test_every_bad_file_has_at_least_one_finding():
    for e in _load_manifest()["entries"]:
        cert = _certify(e)
        assert len(cert.findings) >= 1


# --------------------------------------------------------------------------- #
# Determinism + reproducibility of the corpus.                                  #
# --------------------------------------------------------------------------- #
def test_certificates_are_deterministic():
    for e in _load_manifest()["entries"]:
        c1 = _certify(e)
        c2 = _certify(e)
        assert c1.structural_fingerprint == c2.structural_fingerprint
        assert c1.file_sha256 == c2.file_sha256
        assert sorted(c1.finding_kinds) == sorted(c2.finding_kinds)


def test_corpus_is_byte_identical_to_generator(tmp_path):
    """The committed files must match the deterministic generator exactly, so the
    corpus is reproducible and any silent edit is caught."""
    regen = tmp_path / "weights_bad"
    gen.write_corpus(str(regen))
    for e in gen.corpus():
        committed = (_DATA_DIR / e["file"]).read_bytes()
        fresh = (regen / e["file"]).read_bytes()
        assert committed == fresh, f"{e['file']} drifted from the generator"
    # Manifest content matches too (ignoring file ordering / formatting).
    committed_m = _load_manifest()["entries"]
    fresh_m = json.loads((regen / "manifest.json").read_text())["entries"]
    key = lambda lst: sorted(lst, key=lambda e: e["file"])
    assert key(committed_m) == key(fresh_m)


def test_manifest_records_match_runtime():
    """The manifest's recorded kinds equal what the certifier actually emits —
    the manifest is not allowed to lie about the corpus."""
    for e in _load_manifest()["entries"]:
        cert = _certify(e)
        assert sorted(cert.finding_kinds) == e["kinds"], e["file"]


# --------------------------------------------------------------------------- #
# Cross-check: a contract file is well-formed *without* its contract.           #
# --------------------------------------------------------------------------- #
def test_contract_files_are_well_formed_without_contract():
    """The contract_* bad files are only bad *relative to a contract*; the file
    bytes themselves are valid safetensors (isolates the contract failure)."""
    for e in _load_manifest()["entries"]:
        if not e["named_kind"].startswith("contract_"):
            continue
        path = str(_DATA_DIR / e["file"])
        # No expected contract => the data layer alone is safe.
        cert = certify_weights_file(path, check_finite=True)
        assert cert.proven_safe, f"{e['file']} should be data-layer-safe on its own"
