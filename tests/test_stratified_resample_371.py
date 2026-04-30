"""Tests for the stratified random resample of the 371 Verified tied-weight population.

Validates:
1. The script runs successfully.
2. The CSV has ≥80 rows and ≥3 distinct strata.
3. The Wilson JSON has n ≥ 80 and wilson_hi < 0.1332 (tighter than the
   original shortest-LoC-first interval of 0.13319649...).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "experiments_v5" / "stratified_resample_371.py"
OUT_CSV = REPO / "experiments_v5" / "stratified_resample_371.csv"
OUT_JSON = REPO / "experiments_v5" / "stratified_resample_371_wilson.json"


def _run_script():
    """Run the stratified resample script and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO),
    )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture(scope="module")
def script_output():
    """Run the script once for the whole module; cache results."""
    rc, stdout, stderr = _run_script()
    return rc, stdout, stderr


@pytest.fixture(scope="module")
def wilson_json(script_output):
    rc, stdout, stderr = script_output
    assert rc == 0, (
        f"Script exited with code {rc}\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}"
    )
    assert OUT_JSON.exists(), f"Wilson JSON not found: {OUT_JSON}"
    return json.loads(OUT_JSON.read_text())


@pytest.fixture(scope="module")
def csv_rows(script_output):
    import csv
    rc, stdout, stderr = script_output
    assert rc == 0, (
        f"Script exited with code {rc}\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}"
    )
    assert OUT_CSV.exists(), f"CSV not found: {OUT_CSV}"
    with open(OUT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def test_csv_has_at_least_80_rows(csv_rows):
    assert len(csv_rows) >= 80, (
        f"CSV has only {len(csv_rows)} rows; expected ≥80"
    )


def test_csv_has_at_least_3_strata(csv_rows):
    strata = {r["family"] for r in csv_rows if r.get("family")}
    assert len(strata) >= 3, (
        f"CSV has only {len(strata)} distinct strata: {strata}; expected ≥3"
    )


def test_wilson_json_n_at_least_80(wilson_json):
    n = wilson_json["n"]
    assert n >= 80, f"JSON n={n}; expected ≥80"


def test_wilson_json_hi_tighter_than_original(wilson_json):
    wilson_hi = wilson_json["wilson_hi"]
    assert wilson_hi < 0.1332, (
        f"wilson_hi={wilson_hi:.6f} is not < 0.1332 "
        f"(original interval upper bound)"
    )
