"""Tests for corpus provenance + license-compatibility audit (Step 102)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from corpus_extended.generators import all_cases  # noqa: E402
from corpus_extended.provenance import all_provenance, provenance_for  # noqa: E402

AUDIT_JSON = REPO / "reproducibility" / "corpus_provenance_audit.json"

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


def test_provenance_record_for_every_case():
    cases = all_cases()
    prov = all_provenance()
    assert len(prov) == len(cases)
    required = {"id", "origin", "generator", "authors", "license", "spdx",
                "redistributable", "copied_third_party_code"}
    for p in prov:
        assert required.issubset(p.keys())


def test_every_case_is_synthetic_and_redistributable():
    for p in all_provenance():
        assert p["origin"] == "synthetic_generated"
        assert p["redistributable"] is True
        assert p["copied_third_party_code"] is False
        assert p["license"] == "MIT"


def test_seed_reference_is_marked_reference_only():
    for p in all_provenance():
        assert p["seed_is_reference_only"] is True


def test_provenance_ids_match_cases():
    ids_cases = {c.id for c in all_cases()}
    ids_prov = {p["id"] for p in all_provenance()}
    assert ids_cases == ids_prov


def test_provenance_for_single_case():
    c = all_cases()[0]
    p = provenance_for(c)
    assert p["id"] == c.id
    assert p["generator"] == c.family


def test_audit_says_corpus_redistributable():
    data = json.loads(AUDIT_JSON.read_text())
    assert data["corpus_is_redistributable"] is True
    assert data["none_copied_third_party"] is True
    assert data["no_copy_markers_in_sources"] is True
    assert data["license_compatible_with_redistribution"] is True


def test_audit_has_no_volatile_fields():
    data = json.loads(AUDIT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_audit_is_byte_deterministic():
    from reproducibility import corpus_provenance_audit as cpa

    assert cpa.run(check=True) == 0


def test_audit_counts_all_cases():
    data = json.loads(AUDIT_JSON.read_text())
    assert data["n_cases"] == len(all_cases())
    assert data["n_case_files_scanned"] == len(all_cases())
