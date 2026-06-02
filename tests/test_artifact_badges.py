"""Tests for the Step 123 artifact-evaluation badge-evidence harness."""

from __future__ import annotations

from pathlib import Path

import pytest

import reproducibility.artifact_badges as ab

REPO = Path(__file__).resolve().parent.parent

VOLATILE = ("time", "elapsed", "seconds", "date", "timestamp", "duration", "wall")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


@pytest.fixture(scope="module")
def data():
    return ab.measure()


def test_four_standard_badges_present(data):
    ids = {b["id"] for b in data["badges"]}
    assert ids == {"available", "functional", "reusable", "reproduced"}


def test_all_evidence_paths_actually_exist(data):
    # Every path the appendix cites must really be in the tree (no dangling claim).
    for b in data["badges"]:
        for e in b["evidence"]:
            assert e["present"] == (REPO / e["path"]).exists()
            assert e["present"], f"{b['id']}: missing {e['path']}"


def test_all_badges_evidence_complete(data):
    assert data["all_badges_evidence_complete"]
    assert data["n_badges_evidence_complete"] == data["n_badges"] == 4


def test_reproduced_badge_points_at_capsule_and_audit(data):
    repro = next(b for b in data["badges"] if b["id"] == "reproduced")
    paths = {e["path"] for e in repro["evidence"]}
    assert "capsule/reproduce.sh" in paths
    assert "reproducibility/audit_numeric_claims.py" in paths
    assert "reproducibility/reproduce_all.py" in paths


def test_builds_on_step89_ae_docs(data):
    # Step 123 extends the Step 89 docs/artifact package rather than replacing it.
    all_paths = {e["path"] for b in data["badges"] for e in b["evidence"]}
    assert any(p.startswith("docs/artifact/") for p in all_paths)


def test_available_badge_has_license_and_citation(data):
    avail = next(b for b in data["badges"] if b["id"] == "available")
    paths = {e["path"] for e in avail["evidence"]}
    assert "LICENSE" in paths
    assert "CITATION.cff" in paths


def test_markdown_lists_one_command_repro(data):
    md = ab.render_markdown(data)
    assert "bash capsule/reproduce.sh" in md
    assert "docker run --rm tensorguard-capsule" in md


def test_no_volatile_keys_and_deterministic(data):
    for k in _walk_keys(data):
        assert not any(tok in k.lower() for tok in VOLATILE), k
    assert ab.measure() == data


def test_check_mode_byte_identical():
    assert ab.run(check=True) == 0
