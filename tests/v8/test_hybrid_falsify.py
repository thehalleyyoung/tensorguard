"""Tests for hybrid falsification experiment (TG vs FT)."""
import json
from pathlib import Path

import pytest

RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments_v5" / "v8" / "hybrid_falsify" / "results.json"
)


@pytest.fixture(scope="module")
def results():
    assert RESULTS_PATH.exists(), (
        f"results.json missing — run experiments_v5/v8/hybrid_falsify/run_falsify.py "
        f"first ({RESULTS_PATH})"
    )
    return json.loads(RESULTS_PATH.read_text())


def _by_id(results):
    return {b["id"]: b for b in results["blocks"]}


def test_25_blocks_present(results):
    assert len(results["blocks"]) == 25


def test_categories_present(results):
    cats = {b.get("category") for b in results["blocks"]}
    assert "A_symbolic_shape" in cats
    assert "B_grad_flag" in cats
    assert "C_ft_only" in cats


def test_tg_only_total_at_least_12(results):
    n = sum(1 for b in results["blocks"] if b["cell"] == "TG-only")
    assert n >= 12, f"expected >=12 TG-only blocks, got {n}"


def test_grad_blocks_tg_refuted(results):
    """At least 8 grad-category blocks should be TG-Refuted (TG-only or Both)."""
    grad_ids = [f"blk_{i:02d}" for i in range(13, 21)]
    by = _by_id(results)
    refuted = 0
    for prefix in grad_ids:
        match = [b for bid, b in by.items() if bid.startswith(prefix)]
        assert match, f"missing grad block {prefix}"
        if match[0]["tg_verdict"] == "Refuted":
            refuted += 1
    assert refuted >= 8, f"expected all 8 grad blocks TG-Refuted, got {refuted}"


def test_ft_only_blocks_at_least_4(results):
    """FT catches at least 4 bugs TG misses (cat C reality + bonus)."""
    n = sum(1 for b in results["blocks"] if b["cell"] == "FT-only")
    assert n >= 4, f"expected >=4 FT-only blocks, got {n}"


def test_specific_tg_only_blocks(results):
    by = _by_id(results)
    expected_tg_only = [
        "blk_01_hardcoded_batch8",
        "blk_03_else_branch_linear32",
        "blk_05_hardcoded_spatial_flatten",
        "blk_07_hardcoded_batch32_fc",
        "blk_13_no_grad_trunk_b1",
        "blk_15_frozen_param_b2",
        "blk_17_no_rg_leaf_b4",
        "blk_18_nested_no_grad_b1",
    ]
    for bid in expected_tg_only:
        assert bid in by, f"missing block {bid}"
        assert by[bid]["cell"] == "TG-only", (
            f"{bid} expected TG-only, got {by[bid]['cell']}"
        )


def test_specific_ft_only_blocks(results):
    by = _by_id(results)
    expected_ft_only = [
        "blk_21_view_size0_wrong_fc",
        "blk_22_shape_product_wrong_fc",
        "blk_23_halved_slice_wrong_fc",
    ]
    for bid in expected_ft_only:
        assert bid in by, f"missing block {bid}"
        assert by[bid]["cell"] == "FT-only", (
            f"{bid} expected FT-only, got {by[bid]['cell']}"
        )


def test_contingency_keys(results):
    c = results["contingency"]
    for k in ("TG-only", "Both", "FT-only", "Neither"):
        assert k in c
    assert sum(c.values()) == 25


def test_at_least_one_per_category_in_expected_cell(results):
    by_cat = results.get("by_category", {})
    a = by_cat.get("A_symbolic_shape", {})
    b = by_cat.get("B_grad_flag", {})
    c = by_cat.get("C_ft_only", {})
    assert a.get("TG-only", 0) >= 1
    assert b.get("TG-only", 0) >= 1
    assert c.get("FT-only", 0) >= 1
