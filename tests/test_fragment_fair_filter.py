"""Tests for the deterministic 60→34 fragment-fair filter.

Verifies:
  1. The CSV has exactly 60 data rows (+ 1 header = 61 lines).
  2. Exactly 34 rows have included_in_34 == "True".
  3. Every excluded row has a non-empty exclusion_reason from the closed enumeration.
  4. The recomputed McNemar table matches the published 32/34 vs 25/34 counts.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "reproducibility", "build_fragment_fair_filter.py")
_CSV = os.path.join(_REPO, "reproducibility", "fragment_fair_audit.csv")


def _run_script() -> None:
    result = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stdout}\n{result.stderr}"


def _load_csv() -> list[dict]:
    with open(_CSV, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def audit_rows() -> list[dict]:
    _run_script()
    return _load_csv()


def test_total_row_count(audit_rows: list[dict]) -> None:
    assert len(audit_rows) == 60, f"Expected 60 data rows, got {len(audit_rows)}"


def test_csv_line_count() -> None:
    with open(_CSV) as f:
        lines = f.readlines()
    assert len(lines) == 61, f"Expected 61 lines (header + 60 data), got {len(lines)}"


def test_included_count(audit_rows: list[dict]) -> None:
    included = [r for r in audit_rows if r["included_in_34"] == "True"]
    assert len(included) == 34, f"Expected 34 included rows, got {len(included)}"


def test_excluded_count(audit_rows: list[dict]) -> None:
    excluded = [r for r in audit_rows if r["included_in_34"] == "False"]
    assert len(excluded) == 26, f"Expected 26 excluded rows, got {len(excluded)}"


def test_excluded_rows_have_nonempty_exclusion_reason(audit_rows: list[dict]) -> None:
    from reproducibility.build_fragment_fair_filter import VALID_EXCLUSION_REASONS
    for row in audit_rows:
        if row["included_in_34"] == "False":
            reason = row["exclusion_reason"]
            assert reason, f"Bug {row['bug_id']} is excluded but has no exclusion_reason"
            assert reason in VALID_EXCLUSION_REASONS, (
                f"Bug {row['bug_id']} has exclusion_reason {reason!r} not in closed enumeration "
                f"{VALID_EXCLUSION_REASONS}"
            )


def test_included_rows_have_empty_exclusion_reason(audit_rows: list[dict]) -> None:
    for row in audit_rows:
        if row["included_in_34"] == "True":
            assert row["exclusion_reason"] == "", (
                f"Included bug {row['bug_id']} should have empty exclusion_reason, "
                f"got {row['exclusion_reason']!r}"
            )


def test_mcnemar_table_matches_published_counts(audit_rows: list[dict]) -> None:
    """Published headline: TG 32/34 vs Pytea 25/34; both=25, tg_only=7, pytea_only=0, neither=2."""
    included = [r for r in audit_rows if r["included_in_34"] == "True"]
    assert len(included) == 34

    tg_refuted = sum(1 for r in included if r["tg_verdict"] == "Refuted")
    pytea_refuted = sum(1 for r in included if r["pytea_verdict"] == "Refuted")
    both = sum(
        1 for r in included
        if r["tg_verdict"] == "Refuted" and r["pytea_verdict"] == "Refuted"
    )
    tg_only = sum(
        1 for r in included
        if r["tg_verdict"] == "Refuted" and r["pytea_verdict"] != "Refuted"
    )
    pytea_only = sum(
        1 for r in included
        if r["tg_verdict"] != "Refuted" and r["pytea_verdict"] == "Refuted"
    )
    neither = sum(
        1 for r in included
        if r["tg_verdict"] != "Refuted" and r["pytea_verdict"] != "Refuted"
    )

    assert tg_refuted == 32, f"TG catches: expected 32, got {tg_refuted}"
    assert pytea_refuted == 25, f"Pytea catches: expected 25, got {pytea_refuted}"
    assert both == 25, f"Both refute: expected 25, got {both}"
    assert tg_only == 7, f"TG only: expected 7, got {tg_only}"
    assert pytea_only == 0, f"Pytea only: expected 0, got {pytea_only}"
    assert neither == 2, f"Neither: expected 2, got {neither}"


def test_all_bug_ids_present(audit_rows: list[dict]) -> None:
    """All 60 entries from BUG_MODERN_MAP appear in the CSV."""
    import importlib.util
    import os as _os

    spec = importlib.util.spec_from_file_location(
        "_bff",
        os.path.join(_REPO, "reproducibility", "build_fragment_fair_filter.py"),
    )
    bff = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(bff)  # type: ignore[union-attr]

    bug_modern_map = bff._load_bug_modern_map()
    expected_ids = set(bug_modern_map.keys())
    csv_ids = {r["bug_id"] for r in audit_rows}
    assert csv_ids == expected_ids, f"Missing: {expected_ids - csv_ids}; extra: {csv_ids - expected_ids}"


def test_required_csv_columns(audit_rows: list[dict]) -> None:
    required = {"bug_id", "included_in_34", "exclusion_reason", "tg_verdict", "pytea_verdict"}
    if audit_rows:
        assert required <= set(audit_rows[0].keys()), (
            f"Missing columns: {required - set(audit_rows[0].keys())}"
        )
