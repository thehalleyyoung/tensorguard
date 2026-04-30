"""Tests for the head-to-head N=15 benchmark output.

Asserts:
  - CSV has exactly 15 rows
  - All three verdict columns are populated for every row
  - Stats JSON parses with the three required keys
"""
from __future__ import annotations

import csv
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "benchmarks", "results", "headtohead_n15.csv")
STATS_PATH = os.path.join(ROOT, "benchmarks", "results", "headtohead_n15_stats.json")


def test_csv_exists():
    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"


def test_csv_has_15_rows():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 15, f"Expected 15 rows, got {len(rows)}"


def test_all_verdict_columns_populated():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["tg_verdict"], f"tg_verdict empty for {row['bug_id']}"
        assert row["compile_verdict"], f"compile_verdict empty for {row['bug_id']}"
        assert row["pytea_verdict"], f"pytea_verdict empty for {row['bug_id']}"


def test_stats_json_exists():
    assert os.path.exists(STATS_PATH), f"Stats JSON not found: {STATS_PATH}"


def test_stats_json_has_required_keys():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    required = {"pairwise_mcnemar", "fisher_vs_groundtruth", "bh_adjusted_fisher"}
    missing = required - set(stats.keys())
    assert not missing, f"Stats JSON missing keys: {missing}"


def test_stats_json_pairwise_has_tool_pairs():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    pairs = set(stats["pairwise_mcnemar"].keys())
    expected = {"tg_vs_compile", "tg_vs_pytea", "compile_vs_pytea"}
    missing = expected - pairs
    assert not missing, f"pairwise_mcnemar missing pairs: {missing}"


def test_stats_json_bh_has_all_tools():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    bh = stats["bh_adjusted_fisher"]
    for tool in ("tg", "compile", "pytea"):
        assert tool in bh, f"bh_adjusted_fisher missing tool: {tool}"
