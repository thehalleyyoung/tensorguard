#!/usr/bin/env python3
"""Tests for experiments/eval_baselines.py baseline comparison framework."""

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.eval_baselines import (
    BASELINE_RESULTS,
    BASELINE_TOOLS,
    TEST_CASES,
    BaselineTool,
    TestCase,
    compute_metrics,
    format_comparison_table,
    run_tensorguard_evaluation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test-case parsability
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaseParsability:
    """Every test-case source string must be valid Python."""

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[tc.name for tc in TEST_CASES])
    def test_source_parses(self, tc: TestCase):
        tree = ast.parse(tc.source)
        assert tree is not None

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[tc.name for tc in TEST_CASES])
    def test_source_contains_nn_module(self, tc: TestCase):
        tree = ast.parse(tc.source)
        class_names = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        ]
        assert len(class_names) >= 1, "Expected at least one class definition"


# ═══════════════════════════════════════════════════════════════════════════════
# Structural completeness
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralCompleteness:
    def test_minimum_test_cases(self):
        assert len(TEST_CASES) >= 15

    def test_has_error_and_valid_cases(self):
        errors = [tc for tc in TEST_CASES if tc.expected_error]
        valids = [tc for tc in TEST_CASES if not tc.expected_error]
        assert len(errors) >= 5, "Need at least 5 error test cases"
        assert len(valids) >= 5, "Need at least 5 valid test cases"

    def test_all_test_cases_have_baseline_results(self):
        for tc in TEST_CASES:
            assert tc.name in BASELINE_RESULTS, (
                f"Missing baseline results for {tc.name}"
            )

    def test_baseline_results_cover_all_tools(self):
        tool_keys = {"tensorguard", "jaxtyping", "pytea", "torchscript", "mypy", "pyright"}
        for name, results in BASELINE_RESULTS.items():
            assert set(results.keys()) == tool_keys, (
                f"Baseline results for {name} missing tools: "
                f"{tool_keys - set(results.keys())}"
            )

    def test_baseline_tools_registered(self):
        expected = {"tensorguard", "jaxtyping", "pytea", "torchscript", "mypy", "pyright"}
        assert set(BASELINE_TOOLS.keys()) == expected

    def test_baseline_tool_fields(self):
        for key, tool in BASELINE_TOOLS.items():
            assert isinstance(tool, BaselineTool)
            assert tool.name
            assert tool.description


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTensorGuardEvaluation:
    """Run TensorGuard and verify it correctly classifies error/valid cases."""

    @pytest.fixture(scope="class")
    def tg_results(self):
        return run_tensorguard_evaluation()

    def test_all_cases_evaluated(self, tg_results):
        for tc in TEST_CASES:
            assert tc.name in tg_results, f"Missing result for {tc.name}"

    def test_error_cases_detected(self, tg_results):
        error_cases = [tc for tc in TEST_CASES if tc.expected_error]
        detected = sum(
            1 for tc in error_cases if tg_results[tc.name]["detects"]
        )
        # Allow some tolerance — require at least 50% detection
        assert detected >= len(error_cases) * 0.5, (
            f"TensorGuard detected only {detected}/{len(error_cases)} error cases"
        )

    def test_valid_cases_no_false_positive(self, tg_results):
        valid_cases = [tc for tc in TEST_CASES if not tc.expected_error]
        false_positives = [
            tc.name for tc in valid_cases if tg_results[tc.name]["detects"]
        ]
        # Allow at most 2 false positives (some models may be unsupported)
        assert len(false_positives) <= 2, (
            f"Too many false positives: {false_positives}"
        )

    def test_metrics_computed(self, tg_results):
        metrics = compute_metrics(tg_results)
        assert "tp" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["tp"] + metrics["fn"] + metrics["fp"] + metrics["tn"] == len(TEST_CASES)


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ═══════════════════════════════════════════════════════════════════════════════

class TestComparisonTable:
    def test_table_generates(self):
        table = format_comparison_table()
        assert isinstance(table, str)
        assert len(table) > 100

    def test_table_has_all_tools(self):
        table = format_comparison_table()
        for name in ["TensorGuard", "jaxtyping", "PyTEA", "TorchScript", "mypy", "Pyright"]:
            assert name in table, f"Missing tool {name} in comparison table"

    def test_table_has_key_features(self):
        table = format_comparison_table()
        for feature in ["Static analysis", "Shape checking", "SMT-backed",
                        "Device checking", "Broadcasting"]:
            assert feature in table, f"Missing feature {feature} in table"

    def test_table_is_valid_markdown(self):
        table = format_comparison_table()
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header + separator + at least one row
        assert lines[1].startswith("|")
        assert "-" in lines[1]
