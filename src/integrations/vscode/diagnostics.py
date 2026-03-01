"""Convert TensorGuard verification results to LSP diagnostic format.

The Language Server Protocol defines diagnostics as structured objects
sent from the server to the client. This module converts ``AnalysisResult``
objects into LSP-compatible diagnostic dictionaries.

Reference: https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnostic
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# LSP DiagnosticSeverity enum values
SEVERITY_ERROR = 1
SEVERITY_WARNING = 2
SEVERITY_INFORMATION = 3
SEVERITY_HINT = 4

_SEVERITY_MAP = {
    "error": SEVERITY_ERROR,
    "warning": SEVERITY_WARNING,
    "info": SEVERITY_INFORMATION,
}

# Diagnostic source identifier
SOURCE = "tensorguard"


def analysis_result_to_diagnostics(result: Any, uri: str = "") -> List[Dict[str, Any]]:
    """Convert an ``AnalysisResult`` to a list of LSP diagnostic dicts.

    Args:
        result: An ``AnalysisResult`` from ``src.api``.
        uri: The document URI for the diagnostics.

    Returns:
        List of LSP Diagnostic objects (as dicts).
    """
    diagnostics: List[Dict[str, Any]] = []

    for bug in result.bugs:
        line = max(bug.location.line - 1, 0)  # LSP lines are 0-indexed
        col = max(bug.location.column, 0)
        end_line = (bug.location.end_line - 1) if bug.location.end_line else line
        end_col = bug.location.end_column if bug.location.end_column else col + 1

        diag: Dict[str, Any] = {
            "range": {
                "start": {"line": line, "character": col},
                "end": {"line": end_line, "character": end_col},
            },
            "severity": _SEVERITY_MAP.get(bug.severity, SEVERITY_WARNING),
            "source": SOURCE,
            "message": bug.message,
            "code": bug.category.value,
        }

        # Add code description if fix suggestion available
        if bug.fix_suggestion:
            diag["codeDescription"] = {"href": ""}
            diag["data"] = {"fixSuggestion": bug.fix_suggestion}

        # Add related information for guard evidence
        if bug.guard_evidence:
            diag["relatedInformation"] = [
                {
                    "location": {
                        "uri": uri,
                        "range": {
                            "start": {"line": line, "character": 0},
                            "end": {"line": line, "character": 0},
                        },
                    },
                    "message": f"Guard context: {bug.guard_evidence}",
                }
            ]

        diagnostics.append(diag)

    return diagnostics


def bug_to_diagnostic(bug: Any, uri: str = "") -> Dict[str, Any]:
    """Convert a single ``Bug`` to an LSP diagnostic dict."""
    line = max(bug.location.line - 1, 0)
    col = max(bug.location.column, 0)
    end_line = (bug.location.end_line - 1) if bug.location.end_line else line
    end_col = bug.location.end_column if bug.location.end_column else col + 1

    return {
        "range": {
            "start": {"line": line, "character": col},
            "end": {"line": end_line, "character": end_col},
        },
        "severity": _SEVERITY_MAP.get(bug.severity, SEVERITY_WARNING),
        "source": SOURCE,
        "message": bug.message,
        "code": bug.category.value,
    }
