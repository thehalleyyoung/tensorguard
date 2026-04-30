"""Tests for the pen-and-paper handler classification artifact.

Verifies that:
1. The JSON artifact exists and contains exactly 13 records.
2. No record has class == "unknown".
3. All records have non-empty evidence_lines.
4. All classes are one of {"T-Identity", "T-Broadcast"}.
5. Spot-checks for a representative set of handlers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "reproducibility" / "pen_and_paper_classification.json"

EXPECTED_HANDLERS = {
    "elementwise_binary",
    "reduce",
    "einsum",
    "flatten",
    "softmax",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "where",
    "detach",
    "pad",
}

VALID_CLASSES = {"T-Identity", "T-Broadcast"}


@pytest.fixture(scope="module")
def records():
    assert ARTIFACT.exists(), f"Artifact not found: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def test_artifact_has_13_records(records):
    assert len(records) == 13, f"Expected 13 records, got {len(records)}"


def test_all_handlers_present(records):
    found = {r["handler"] for r in records}
    assert found == EXPECTED_HANDLERS, (
        f"Missing: {EXPECTED_HANDLERS - found}; Extra: {found - EXPECTED_HANDLERS}"
    )


def test_no_unknown_class(records):
    unknown = [r for r in records if r["class"] not in VALID_CLASSES]
    assert not unknown, f"Records with unknown class: {unknown}"


def test_all_evidence_lines_nonempty(records):
    empty = [r["handler"] for r in records if not r.get("evidence_lines")]
    assert not empty, f"Handlers with empty evidence_lines: {empty}"


def test_all_sha_present(records):
    missing_sha = [r["handler"] for r in records if not r.get("sha")]
    assert not missing_sha, f"Handlers missing sha: {missing_sha}"


@pytest.mark.parametrize("handler,expected_class", [
    ("elementwise_binary", "T-Broadcast"),
    ("where", "T-Broadcast"),
    ("einsum", "T-Broadcast"),
    ("relu", "T-Identity"),
    ("gelu", "T-Identity"),
    ("silu", "T-Identity"),
    ("tanh", "T-Identity"),
    ("sigmoid", "T-Identity"),
    ("softmax", "T-Identity"),
    ("detach", "T-Identity"),
    ("flatten", "T-Identity"),
    ("pad", "T-Identity"),
    ("reduce", "T-Identity"),
])
def test_handler_class(records, handler, expected_class):
    record = next((r for r in records if r["handler"] == handler), None)
    assert record is not None, f"Handler {handler!r} not found in records"
    assert record["class"] == expected_class, (
        f"{handler}: expected {expected_class!r}, got {record['class']!r}"
    )
