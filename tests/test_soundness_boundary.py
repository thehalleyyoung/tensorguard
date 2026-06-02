"""Tests for the soundness/incompleteness boundary harness (Step 94).

These validate that the documented unsound/incomplete boundary
(``src/soundness_contract.py`` / ``SOUNDNESS_CONTRACT.md``) reproduces on the
LIVE verifier, that the emitted artifact is byte-deterministic, and that it
carries no volatile (timing) fields.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import reproducibility.soundness_boundary as sb  # noqa: E402

VOLATILE_TOKENS = ("time", "elapsed", "timestamp", "wall", "clock", "_ms",
                   "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_measure_all_probes_match_contract():
    data = sb.measure()
    assert data["all_match"] is True
    assert data["n_probes"] == 4
    for p in data["probes"]:
        assert p["match"] is True, f"{p['name']} diverged: {p['observed']}"


def test_refutation_probe_is_unsafe_in_every_mode():
    data = sb.measure()
    bug = next(p for p in data["probes"]
               if p["name"] == "in_fragment_shape_bug")
    for mode in sb.MODES:
        assert bug["observed"][mode] == "UNSAFE"


def test_clean_probe_is_safe_in_every_mode():
    data = sb.measure()
    clean = next(p for p in data["probes"] if p["name"] == "in_fragment_clean")
    for mode in sb.MODES:
        assert clean["observed"][mode] == "SAFE"


def test_out_of_fragment_is_mode_dependent():
    """The U1 boundary: sound abstains (UNKNOWN), permissive modes pass."""
    data = sb.measure()
    oof = [p for p in data["probes"] if p["region"].startswith("fragment")]
    assert len(oof) == 2
    for p in oof:
        assert p["observed"]["sound"] == "UNKNOWN"
        assert p["observed"]["balanced"] == "SAFE"
        assert p["observed"]["heuristic"] == "SAFE"


def test_contract_coverage_reported():
    data = sb.measure()
    c = data["contract"]
    assert c["out_of_fragment_clauses"] >= 1
    assert c["domain_clauses"] >= 1
    ids = {g["id"] for g in c["known_unsoundness_gaps"]}
    assert {"U1", "U2"}.issubset(ids)


def test_artifact_is_byte_deterministic():
    assert sb.run(check=True) == 0


def test_artifact_has_no_volatile_fields():
    data = json.loads(sb.OUT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        for tok in VOLATILE_TOKENS:
            assert tok not in low, f"volatile key token {tok!r} in {key!r}"
