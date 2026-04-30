"""
tests/test_footprint_strict_488.py
===================================

Success criterion for the footprint-strict real-source evaluation:

  1. run_footprint_strict_488.py produces experiments_v5/footprint_strict_488.csv
     with exactly ≥488 data rows.
  2. experiments_v5/footprint_strict_488_summary.json is valid JSON with an
     'audited' key.
  3. audited.V + audited.CV + audited.RP + audited.A equals the audited-class
     row count from the CSV.
  4. audited.V + audited.CV + audited.RP > 0 (non-trivial audited verdicts).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments_v5"
CSV_PATH = EXP / "footprint_strict_488.csv"
JSON_PATH = EXP / "footprint_strict_488_summary.json"
SCRIPT = EXP / "run_footprint_strict_488.py"


def _run_script() -> None:
    """Run the footprint-strict script if outputs are missing."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"run_footprint_strict_488.py failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _ensure_outputs() -> None:
    if not CSV_PATH.exists() or not JSON_PATH.exists():
        _run_script()


def test_csv_exists_with_488_rows() -> None:
    _ensure_outputs()
    assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
    rows = list(csv.DictReader(CSV_PATH.open()))
    assert len(rows) >= 488, f"Expected ≥488 rows, got {len(rows)}"


def test_csv_columns() -> None:
    _ensure_outputs()
    rows = list(csv.DictReader(CSV_PATH.open()))
    assert rows, "CSV is empty"
    expected_cols = {"block_id", "verdict", "footprint_class", "ops_touched"}
    assert expected_cols.issubset(set(rows[0].keys())), (
        f"Missing columns. Found: {set(rows[0].keys())}"
    )


def test_summary_json_parses_and_has_audited_key() -> None:
    _ensure_outputs()
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"
    data = json.loads(JSON_PATH.read_text())
    assert "audited" in data, f"'audited' key missing from JSON. Keys: {list(data.keys())}"


def test_audited_counts_sum_to_audited_class_row_count() -> None:
    _ensure_outputs()
    rows = list(csv.DictReader(CSV_PATH.open()))
    audited_row_count = sum(1 for r in rows if r["footprint_class"] == "audited")

    data = json.loads(JSON_PATH.read_text())
    audited = data["audited"]
    total = audited["V"] + audited["CV"] + audited["RP"] + audited["A"]
    assert total == audited_row_count, (
        f"audited V+CV+RP+A={total} does not match audited CSV rows={audited_row_count}"
    )


def test_audited_has_nonzero_verdicts() -> None:
    _ensure_outputs()
    data = json.loads(JSON_PATH.read_text())
    audited = data["audited"]
    assert audited["V"] + audited["CV"] + audited["RP"] > 0, (
        "No non-Abstain verdicts in audited footprint class — something went wrong"
    )


def test_total_blocks_is_488() -> None:
    _ensure_outputs()
    data = json.loads(JSON_PATH.read_text())
    assert data["total_blocks"] == 488, (
        f"Expected total_blocks=488, got {data['total_blocks']}"
    )
