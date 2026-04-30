"""
tests/test_hf_mechanical_corpus.py

Validates the mechanically-extracted HuggingFace bug corpus produced by
scripts/mine_hf_shape_bugs.py.

Assertions:
  (i)   ≥15 entries
  (ii)  ≥5 distinct `family` values
  (iii) every entry has a non-null `tg_verdict` and `provenance.regex_hit`
  (iv)  the script is deterministic (two consecutive runs produce byte-identical
        `entries` lists, modulo the top-level `generated_at` timestamp)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_LIST = REPO_ROOT / "experiments_v5" / "hf_pr_seed_list.txt"
OUTPUT_JSON = REPO_ROOT / "experiments_v5" / "hf_natural_bugs_mechanical.json"
SCRIPT = REPO_ROOT / "scripts" / "mine_hf_shape_bugs.py"


def _run_script() -> None:
    """Run the mining script and assert it exits 0."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--offline-fixture", str(SEED_LIST)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mine_hf_shape_bugs.py exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _load() -> dict:
    assert OUTPUT_JSON.exists(), f"Output JSON not found: {OUTPUT_JSON}"
    with open(OUTPUT_JSON) as fh:
        return json.load(fh)


# ── Ensure JSON exists before running structural tests ────────────────────────

@pytest.fixture(scope="module", autouse=True)
def ensure_output_exists():
    """Run the script once (if output is absent) so structural tests have data."""
    if not OUTPUT_JSON.exists():
        _run_script()


# ── Structural tests ──────────────────────────────────────────────────────────

def test_min_15_entries():
    data = _load()
    entries = data["entries"]
    assert len(entries) >= 15, f"Expected ≥15 entries, got {len(entries)}"


def test_min_5_families():
    data = _load()
    families = {e["family"] for e in data["entries"]}
    assert len(families) >= 5, (
        f"Expected ≥5 distinct family values, got {len(families)}: {families}"
    )


def test_all_entries_have_verdict_and_provenance_regex_hit():
    data = _load()
    for entry in data["entries"]:
        assert entry.get("tg_verdict") is not None, (
            f"Missing tg_verdict in entry {entry.get('pr')}"
        )
        regex_hit = entry.get("provenance", {}).get("regex_hit")
        assert regex_hit, (
            f"Missing or empty provenance.regex_hit in entry {entry.get('pr')}"
        )


# ── Determinism test ──────────────────────────────────────────────────────────

def test_determinism():
    """Two consecutive runs produce identical `entries` lists."""
    _run_script()
    with open(OUTPUT_JSON) as fh:
        run1 = json.load(fh)

    _run_script()
    with open(OUTPUT_JSON) as fh:
        run2 = json.load(fh)

    assert run1["entries"] == run2["entries"], (
        "Script is not deterministic: entries differ between consecutive runs"
    )


# ── Summary print (informational, never fails) ────────────────────────────────

def test_print_summary():
    data = _load()
    entries = data["entries"]
    families = sorted({e["family"] for e in entries})
    tg_buggy = sum(1 for e in entries if e.get("tg_verdict") == "BUGGY")
    print(
        f"\nMECHANICAL_HF: {tg_buggy}/{len(entries)} detected "
        f"across {len(families)} families: {families}"
    )
