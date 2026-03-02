"""Tests for the real-world model evaluation harness.

Verifies that each model source can be parsed, verify_model returns a result,
no crashes occur, and coverage fraction is reasonable.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_real_models import (
    REAL_MODELS,
    MODEL_SOURCES,
    ModelSpec,
    EvalResult,
    evaluate_model,
    run_evaluation,
    format_results_table,
)
from src.model_checker import verify_model, VerificationResult


# ── Parse tests ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", REAL_MODELS, ids=[m.name for m in REAL_MODELS])
def test_model_source_parses(spec: ModelSpec):
    """Each model source string must be valid Python."""
    source = MODEL_SOURCES[spec.name]
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module)
    # Should contain at least one class definition
    class_defs = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert len(class_defs) >= 1, f"{spec.name}: no class definition found"


# ── Verification tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", REAL_MODELS, ids=[m.name for m in REAL_MODELS])
def test_verify_model_returns_result(spec: ModelSpec):
    """verify_model must return a VerificationResult for each model."""
    source = MODEL_SOURCES[spec.name]
    result = verify_model(source, input_shapes=spec.input_shapes)
    assert isinstance(result, VerificationResult)


@pytest.mark.parametrize("spec", REAL_MODELS, ids=[m.name for m in REAL_MODELS])
def test_no_crash(spec: ModelSpec):
    """Verification must not raise for any model."""
    source = MODEL_SOURCES[spec.name]
    # Should not raise
    result = verify_model(source, input_shapes=spec.input_shapes)
    assert result is not None


@pytest.mark.parametrize("spec", REAL_MODELS, ids=[m.name for m in REAL_MODELS])
def test_coverage_reasonable(spec: ModelSpec):
    """Operator coverage fraction should be > 0.5 for realistic models."""
    eval_result = evaluate_model(spec)
    assert eval_result.coverage_fraction > 0.5, (
        f"{spec.name}: coverage {eval_result.coverage_fraction:.2f} <= 0.5"
    )


# ── Integration tests ────────────────────────────────────────────────────────

def test_run_evaluation():
    """run_evaluation should return a well-formed results dict."""
    results = run_evaluation()
    assert "num_models" in results
    assert results["num_models"] == len(REAL_MODELS)
    assert "results" in results
    assert len(results["results"]) >= len(REAL_MODELS)


def test_format_results_table():
    """format_results_table should produce valid markdown."""
    results = run_evaluation()
    table = format_results_table(results)
    assert "| Model |" in table
    assert "|----" in table
    for spec in REAL_MODELS:
        assert spec.name in table


# ── Registry consistency ─────────────────────────────────────────────────────

def test_all_models_have_source():
    """Every model in REAL_MODELS must have a source in MODEL_SOURCES."""
    for spec in REAL_MODELS:
        assert spec.name in MODEL_SOURCES, f"Missing source for {spec.name}"


def test_model_spec_fields():
    """Each ModelSpec should have valid fields."""
    for spec in REAL_MODELS:
        assert spec.category in ("vision", "nlp", "multimodal")
        assert spec.expected_ops > 0
        assert len(spec.input_shapes) > 0
        assert spec.source in ("torchvision", "huggingface", "manual")
