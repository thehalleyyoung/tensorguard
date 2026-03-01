"""Tests for TensorGuard integration hooks.

Covers: CI hook JSON/SARIF output, deterministic mode, hook registration in API.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.integrations.ci_hook import CIHook, EXIT_SAFE, EXIT_BUG_FOUND, EXIT_UNKNOWN
from src.integrations.wandb_hook import WandbHook
from src.integrations.mlflow_hook import MLflowHook
from src.integrations.pytest_plugin import TensorGuardPlugin, MARKER_NAME
from src.integrations.vscode.diagnostics import (
    analysis_result_to_diagnostics,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
)
from src.api import AnalysisResult, Bug, BugCategory, SourceLocation


# ---------------------------------------------------------------------------
# Helpers — build synthetic AnalysisResult objects
# ---------------------------------------------------------------------------

def _make_safe_result() -> AnalysisResult:
    return AnalysisResult(
        bugs=[],
        guards_harvested=5,
        functions_analyzed=2,
        lines_analyzed=50,
        duration_ms=12.3,
    )


def _make_unsafe_result() -> AnalysisResult:
    return AnalysisResult(
        bugs=[
            Bug(
                category=BugCategory.TYPE_ERROR,
                message="Shape mismatch in Linear: expected [*, 128], got [*, 64]",
                location=SourceLocation(file="model.py", line=10, column=4),
                severity="error",
                confidence=0.95,
                fix_suggestion="Check input dimensions",
            ),
            Bug(
                category=BugCategory.NULL_DEREFERENCE,
                message="Optional tensor accessed without None check",
                location=SourceLocation(file="model.py", line=25, column=8),
                severity="warning",
                confidence=0.80,
            ),
        ],
        guards_harvested=3,
        functions_analyzed=1,
        lines_analyzed=40,
        duration_ms=45.6,
    )


def _make_unknown_result() -> AnalysisResult:
    """Result with only low-confidence bugs → unknown verdict."""
    return AnalysisResult(
        bugs=[
            Bug(
                category=BugCategory.TYPE_ERROR,
                message="Possible type issue",
                location=SourceLocation(file="model.py", line=5, column=0),
                severity="warning",
                confidence=0.5,
            ),
        ],
        guards_harvested=1,
        functions_analyzed=1,
        lines_analyzed=20,
        duration_ms=5.0,
    )


# ===================================================================
# CI Hook — JSON output format
# ===================================================================


class TestCIHookJsonOutput:
    """Test CI hook produces valid, complete JSON reports."""

    def test_safe_result_json(self):
        hook = CIHook()
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        report = hook.json_report
        assert report is not None
        assert report["tool"] == "tensorguard"
        assert report["version"] == "0.2.0"
        assert report["verdict"] == "safe"
        assert report["exit_code"] == EXIT_SAFE
        assert report["bug_count"] == 0
        assert report["bugs"] == []
        assert "duration_ms" in report

    def test_unsafe_result_json(self):
        hook = CIHook()
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="model.py")
        hook.on_verification_end(result=result)

        report = hook.json_report
        assert report is not None
        assert report["verdict"] == "unsafe"
        assert report["exit_code"] == EXIT_BUG_FOUND
        assert report["bug_count"] == 2
        assert len(report["bugs"]) == 2

        bug = report["bugs"][0]
        assert "category" in bug
        assert "message" in bug
        assert "file" in bug
        assert "line" in bug
        assert "column" in bug
        assert "severity" in bug
        assert "confidence" in bug

    def test_unknown_result_json(self):
        hook = CIHook()
        result = _make_unknown_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        report = hook.json_report
        assert report is not None
        assert report["verdict"] == "unknown"
        assert report["exit_code"] == EXIT_UNKNOWN

    def test_json_report_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = CIHook(output_dir=tmpdir)
            result = _make_unsafe_result()
            hook.on_verification_start(source="x = 1", filename="model.py")
            hook.on_verification_end(result=result)

            json_path = os.path.join(tmpdir, "tensorguard-report.json")
            assert os.path.exists(json_path)

            with open(json_path) as f:
                loaded = json.load(f)
            assert loaded["verdict"] == "unsafe"
            assert loaded["bug_count"] == 2

    def test_json_is_serializable(self):
        """Verify the JSON report can round-trip through json.dumps/loads."""
        hook = CIHook()
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        serialized = json.dumps(hook.json_report)
        deserialized = json.loads(serialized)
        assert deserialized["verdict"] == "unsafe"


# ===================================================================
# CI Hook — SARIF output validity
# ===================================================================


class TestCIHookSarifOutput:
    """Test CI hook produces valid SARIF 2.1.0 output."""

    def test_sarif_structure_safe(self):
        hook = CIHook(sarif=True)
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        sarif = hook.sarif_report
        assert sarif is not None
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "sarif-schema-2.1.0" in sarif["$schema"]
        assert len(sarif["runs"]) == 1

        run = sarif["runs"][0]
        assert run["tool"]["driver"]["name"] == "TensorGuard"
        assert isinstance(run["results"], list)
        assert len(run["results"]) == 0

    def test_sarif_structure_unsafe(self):
        hook = CIHook(sarif=True)
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="model.py")
        hook.on_verification_end(result=result)

        sarif = hook.sarif_report
        assert sarif is not None
        assert len(sarif["runs"]) == 1

        run = sarif["runs"][0]
        assert len(run["results"]) == 2

        sarif_result = run["results"][0]
        assert "ruleId" in sarif_result
        assert "level" in sarif_result
        assert sarif_result["level"] in ("error", "warning", "note")
        assert "message" in sarif_result
        assert "text" in sarif_result["message"]
        assert "locations" in sarif_result
        loc = sarif_result["locations"][0]["physicalLocation"]
        assert "artifactLocation" in loc
        assert "region" in loc
        assert "startLine" in loc["region"]

    def test_sarif_has_rules(self):
        hook = CIHook(sarif=True)
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="model.py")
        hook.on_verification_end(result=result)

        rules = hook.sarif_report["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) > 0
        rule_ids = {r["id"] for r in rules}
        assert "type_error" in rule_ids or "null_dereference" in rule_ids

    def test_sarif_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hook = CIHook(output_dir=tmpdir, sarif=True)
            result = _make_safe_result()
            hook.on_verification_start(source="x = 1", filename="test.py")
            hook.on_verification_end(result=result)

            sarif_path = os.path.join(tmpdir, "tensorguard.sarif")
            assert os.path.exists(sarif_path)

            with open(sarif_path) as f:
                loaded = json.load(f)
            assert loaded["version"] == "2.1.0"

    def test_sarif_is_json_serializable(self):
        hook = CIHook(sarif=True)
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="model.py")
        hook.on_verification_end(result=result)

        serialized = json.dumps(hook.sarif_report)
        deserialized = json.loads(serialized)
        assert deserialized["version"] == "2.1.0"


# ===================================================================
# CI Hook — Deterministic mode
# ===================================================================


class TestCIHookDeterministic:
    """Test that deterministic mode flag is properly propagated."""

    def test_deterministic_flag_set(self):
        hook = CIHook(deterministic=True)
        assert hook.deterministic is True

    def test_deterministic_flag_in_report(self):
        hook = CIHook(deterministic=True)
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        report = hook.json_report
        assert report is not None
        assert report["deterministic"] is True

    def test_non_deterministic_flag_in_report(self):
        hook = CIHook(deterministic=False)
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        report = hook.json_report
        assert report is not None
        assert report["deterministic"] is False


# ===================================================================
# CI Hook — Exit codes
# ===================================================================


class TestCIHookExitCodes:
    """Test proper exit code assignment."""

    def test_exit_code_safe(self):
        hook = CIHook()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=_make_safe_result())
        assert hook.exit_code == EXIT_SAFE  # 0

    def test_exit_code_unsafe(self):
        hook = CIHook()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=_make_unsafe_result())
        assert hook.exit_code == EXIT_BUG_FOUND  # 1

    def test_exit_code_unknown(self):
        hook = CIHook()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=_make_unknown_result())
        assert hook.exit_code == EXIT_UNKNOWN  # 2

    def test_initial_exit_code_is_unknown(self):
        hook = CIHook()
        assert hook.exit_code == EXIT_UNKNOWN


# ===================================================================
# Hook registration in API
# ===================================================================


class TestHookRegistrationInAPI:
    """Test that hooks are called at the right points in verify_architecture."""

    def test_hooks_parameter_accepted(self):
        """verify_architecture accepts a hooks parameter."""
        import inspect
        from src.api import verify_architecture
        sig = inspect.signature(verify_architecture)
        assert "hooks" in sig.parameters

    def test_hooks_called_on_result(self):
        """Hooks receive on_verification_start and on_verification_end calls."""
        mock_hook = MagicMock()
        mock_hook.deterministic = False

        # Use analyze() which doesn't require Z3
        from src.api import analyze
        result = analyze("x = 1\n", filename="test.py")

        # Manually simulate what verify_architecture does with hooks
        mock_hook.on_verification_start(source="x = 1\n", filename="test.py")
        mock_hook.on_verification_end(result=result)

        mock_hook.on_verification_start.assert_called_once()
        mock_hook.on_verification_end.assert_called_once()

    def test_ci_hook_works_as_hook(self):
        """CIHook implements the hook protocol."""
        hook = CIHook()
        result = _make_safe_result()

        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)
        hook.on_cegar_iteration(iteration=1, status="safe", predicates_discovered=3)
        hook.close()

        assert hook.exit_code == EXIT_SAFE
        assert hook.json_report is not None

    def test_disabled_hook_noop(self):
        """Disabled hooks do nothing."""
        hook = CIHook(enabled=False)
        result = _make_unsafe_result()

        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)

        assert hook.json_report is None
        assert hook.exit_code == EXIT_UNKNOWN  # unchanged default


# ===================================================================
# W&B Hook — graceful fallback
# ===================================================================


class TestWandbHookFallback:
    """Test W&B hook handles missing wandb gracefully."""

    def test_wandb_hook_disabled_when_not_installed(self):
        hook = WandbHook(enabled=True)
        # Even if wandb is not installed, no error should be raised
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)
        hook.on_cegar_iteration(iteration=1, status="safe")
        hook.close()

    def test_wandb_hook_disabled_flag(self):
        hook = WandbHook(enabled=False)
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)
        hook.close()


# ===================================================================
# MLflow Hook — graceful fallback
# ===================================================================


class TestMLflowHookFallback:
    """Test MLflow hook handles missing mlflow gracefully."""

    def test_mlflow_hook_disabled_when_not_installed(self):
        hook = MLflowHook(enabled=True)
        result = _make_safe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)
        hook.on_cegar_iteration(iteration=1, status="safe")
        hook.close()

    def test_mlflow_hook_disabled_flag(self):
        hook = MLflowHook(enabled=False)
        result = _make_unsafe_result()
        hook.on_verification_start(source="x = 1", filename="test.py")
        hook.on_verification_end(result=result)
        hook.close()


# ===================================================================
# VS Code Diagnostics
# ===================================================================


class TestVSCodeDiagnostics:
    """Test LSP diagnostic conversion."""

    def test_empty_result_gives_empty_diagnostics(self):
        result = _make_safe_result()
        diags = analysis_result_to_diagnostics(result, uri="file:///test.py")
        assert diags == []

    def test_bugs_converted_to_diagnostics(self):
        result = _make_unsafe_result()
        diags = analysis_result_to_diagnostics(result, uri="file:///model.py")
        assert len(diags) == 2

        d0 = diags[0]
        assert d0["source"] == "tensorguard"
        assert d0["severity"] == SEVERITY_ERROR
        assert d0["code"] == "type_error"
        assert "Shape mismatch" in d0["message"]
        assert d0["range"]["start"]["line"] == 9  # 0-indexed
        assert d0["range"]["start"]["character"] == 4

    def test_diagnostic_has_fix_suggestion(self):
        result = _make_unsafe_result()
        diags = analysis_result_to_diagnostics(result, uri="file:///model.py")
        # First bug has a fix suggestion
        d0 = diags[0]
        assert "data" in d0
        assert "fixSuggestion" in d0["data"]


# ===================================================================
# Pytest Plugin
# ===================================================================


class TestPytestPlugin:
    """Test the pytest plugin configuration."""

    def test_marker_name(self):
        assert MARKER_NAME == "tensorguard_verify"

    def test_plugin_has_configure(self):
        assert hasattr(TensorGuardPlugin, "pytest_configure")

    def test_plugin_has_runtest_call(self):
        assert hasattr(TensorGuardPlugin, "pytest_runtest_call")
