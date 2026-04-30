"""Tests for scripts/build_audited_footprint_table.py.

Validates that:
  1. The script exits 0.
  2. experiments_v5/audited_footprint_unconditional_rp.json exists and has
     exactly 5 catches.
  3. Every catch satisfies non_audited_handlers == [] and lean_rule is
     non-null/non-empty.
  4. experiments_v5/handler_classification.json exists with the expected
     schema (lean_audited bool on every handler entry).
  5. Schema invariants: required fields are present on every catch row.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_audited_footprint_table.py"
FOOTPRINT_JSON = REPO / "experiments_v5" / "audited_footprint_unconditional_rp.json"
HANDLER_CLASS_JSON = REPO / "experiments_v5" / "handler_classification.json"

EXPECTED_CATCH_FIELDS = {"block_id", "module_path", "handler_chain", "lean_rule",
                          "non_audited_handlers", "verdict"}


def _run_script() -> None:
    """Re-run the build script; assert it exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script exited {result.returncode}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _load_footprint() -> dict:
    assert FOOTPRINT_JSON.exists(), f"Missing: {FOOTPRINT_JSON}"
    return json.loads(FOOTPRINT_JSON.read_text())


def test_script_runs_clean() -> None:
    _run_script()


def test_output_file_exists() -> None:
    _run_script()
    assert FOOTPRINT_JSON.exists()


def test_exactly_five_catches() -> None:
    _run_script()
    d = _load_footprint()
    catches = d["catches"]
    assert len(catches) == 5, f"Expected 5 catches, got {len(catches)}"


def test_non_audited_handlers_empty() -> None:
    _run_script()
    d = _load_footprint()
    for r in d["catches"]:
        assert r["non_audited_handlers"] == [], (
            f"Block {r['block_id']} has non_audited_handlers: {r['non_audited_handlers']}"
        )


def test_lean_rule_non_null() -> None:
    _run_script()
    d = _load_footprint()
    for r in d["catches"]:
        assert r["lean_rule"], (
            f"Block {r['block_id']} has null/empty lean_rule"
        )


def test_catch_schema() -> None:
    _run_script()
    d = _load_footprint()
    for r in d["catches"]:
        missing = EXPECTED_CATCH_FIELDS - set(r.keys())
        assert not missing, (
            f"Block {r['block_id']} missing fields: {missing}"
        )
        assert isinstance(r["block_id"], str) and r["block_id"]
        assert isinstance(r["module_path"], str)
        assert isinstance(r["handler_chain"], list)
        assert isinstance(r["lean_rule"], dict)
        assert isinstance(r["non_audited_handlers"], list)
        assert isinstance(r["verdict"], str)


def test_handler_classification_exists_and_has_lean_audited() -> None:
    _run_script()
    assert HANDLER_CLASS_JSON.exists(), f"Missing: {HANDLER_CLASS_JSON}"
    d = json.loads(HANDLER_CLASS_JSON.read_text())
    assert "handlers" in d
    for h in d["handlers"]:
        assert "name" in h
        assert "lean_audited" in h, f"Handler {h['name']} missing lean_audited field"
        assert isinstance(h["lean_audited"], bool)


def test_handler_classification_counts_derivable() -> None:
    """Counts in handler_classification.json must match actual handler list."""
    _run_script()
    d = json.loads(HANDLER_CLASS_JSON.read_text())
    handlers = d["handlers"]
    counts = d["counts"]
    assert counts["lean_audited_total"] == sum(1 for h in handlers if h["lean_audited"])
    assert counts["lean_verified"] == sum(1 for h in handlers if h["scope"] == "lean_verified")
    assert counts["pen_and_paper"] == sum(1 for h in handlers if h["scope"] == "pen_and_paper")
