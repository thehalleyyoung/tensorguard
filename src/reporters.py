"""Step 73 — JSON, JUnit-XML and GitHub-annotation reporters.

The GitHub Action, pre-commit hook and pytest plugin all produce the same
``results_by_file`` payload (a list of ``(path, AnalysisResult)`` pairs) and the
same per-file annotations via :func:`src.github_action.annotations_for_result`.
This module turns that one canonical finding stream into the output formats CI
systems expect, so a team can wire TensorGuard into whatever they already run:

* **JSON** — a stable, machine-readable schema for dashboards and bots.
* **JUnit-XML** — one ``<testcase>`` per analysed file (failing files carry a
  ``<failure>`` per finding) so any JUnit consumer renders TensorGuard results.
* **GitHub annotations** — the workflow-command lines (delegates to Step 66).

SARIF 2.1.0 remains available via :mod:`src.sarif_codescan`; these reporters sit
alongside it and share the exact same finding extraction, so no format can drift.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

from src.github_action import annotations_for_result
from src.tg_config import rule_tag

SCHEMA_VERSION = "1.0"
ResultsByFile = List[Tuple[str, Any]]


def _findings(results_by_file: ResultsByFile) -> List[Dict[str, Any]]:
    """Flatten results into a canonical finding list (one per annotation)."""
    findings: List[Dict[str, Any]] = []
    for file, result in results_by_file:
        for ann in annotations_for_result(file, result):
            msg = getattr(ann, "message", "") or ""
            findings.append(
                {
                    "file": file,
                    "line": getattr(ann, "line", 0),
                    "column": getattr(ann, "col", None),
                    "level": getattr(ann, "level", "error"),
                    "rule": rule_tag(msg) or "verification-issue",
                    "message": msg,
                }
            )
    return findings


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def build_json(results_by_file: ResultsByFile) -> Dict[str, Any]:
    """Structured JSON report payload."""
    findings = _findings(results_by_file)
    files_checked = len(results_by_file)
    files_with_issues = len({f["file"] for f in findings})
    return {
        "schema": "tensorguard-report",
        "version": SCHEMA_VERSION,
        "summary": {
            "files_checked": files_checked,
            "files_with_issues": files_with_issues,
            "total_findings": len(findings),
        },
        "findings": findings,
    }


def to_json(results_by_file: ResultsByFile, *, indent: int = 2) -> str:
    return json.dumps(build_json(results_by_file), indent=indent, sort_keys=True)


# --------------------------------------------------------------------------- #
# JUnit-XML
# --------------------------------------------------------------------------- #
def build_junit(results_by_file: ResultsByFile) -> ET.Element:
    """A JUnit ``<testsuites>`` element: one ``<testcase>`` per analysed file."""
    findings = _findings(results_by_file)
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)

    total_failures = len(findings)
    suite = ET.Element(
        "testsuite",
        {
            "name": "tensorguard",
            "tests": str(len(results_by_file)),
            "failures": str(total_failures),
            "errors": "0",
            "skipped": "0",
        },
    )
    for file, _result in results_by_file:
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": "tensorguard", "name": file},
        )
        for finding in by_file.get(file, []):
            line = finding["line"]
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": finding["rule"],
                    "message": f"{file}:{line}: {finding['message']}",
                },
            )
            failure.text = (
                f"{finding['rule']}: {finding['message']}\n"
                f"  at {file}:{line}"
            )
    suites = ET.Element(
        "testsuites",
        {"tests": str(len(results_by_file)), "failures": str(total_failures)},
    )
    suites.append(suite)
    return suites


def to_junit_xml(results_by_file: ResultsByFile) -> str:
    root = build_junit(results_by_file)
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


# --------------------------------------------------------------------------- #
# GitHub annotations (delegates to Step 66)
# --------------------------------------------------------------------------- #
def to_github_annotations(results_by_file: ResultsByFile) -> str:
    lines: List[str] = []
    for file, result in results_by_file:
        for ann in annotations_for_result(file, result):
            lines.append(ann.render())
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dispatch + file writing
# --------------------------------------------------------------------------- #
_FORMATS = {"json", "junit", "github"}


def render(results_by_file: ResultsByFile, fmt: str) -> str:
    fmt = (fmt or "").lower()
    if fmt == "json":
        return to_json(results_by_file)
    if fmt in ("junit", "junit-xml", "junitxml"):
        return to_junit_xml(results_by_file)
    if fmt in ("github", "annotations"):
        return to_github_annotations(results_by_file)
    raise ValueError(f"unknown report format: {fmt!r} (choose from {_FORMATS})")


def write_report(path: str, results_by_file: ResultsByFile, fmt: str) -> str:
    text = render(results_by_file, fmt)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return text
