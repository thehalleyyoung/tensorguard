"""Structured export of symbolic-execution results (Step 68).

Symexec findings already flow into the public ``src.api.Bug`` type (via
:meth:`SymBug.to_api_bug`) and therefore through the repository's existing
reporters.  That path, however, *flattens* every symexec-specific signal — the
calibrated confidence (Step 63), the provenance derivation / counterexample /
certificate / minimal-conditions evidence (Steps 7/54/58/65), the deterministic
proof fingerprint (Step 60) and the abstain-coverage profile (Step 59) — down to
a single ``guard_evidence`` string.

This module surfaces those richer fields directly, in two machine-readable
shapes that an owner or CI can consume without re-running analysis:

* :func:`result_to_dict` — a stable JSON object for one analyzed file.
* :func:`to_sarif` — a valid `SARIF 2.1.0 <https://sarifweb.azurewebsites.net>`_
  log (one ``run`` per file) whose ``result`` objects carry the symexec fields
  in their ``properties`` bag, and whose ``run`` ``properties`` carry the proof
  fingerprint and abstain coverage.  GitHub code-scanning, editors, and other
  SARIF consumers can ingest it directly.

The module is torch-free and pure: it only reads already-computed result data.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .explain import explain_bug

__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "SARIF_VERSION",
    "bug_to_dict",
    "result_to_dict",
    "result_to_sarif_run",
    "to_sarif",
    "sarif_replacement",
]

#: Bump when the *shape* of :func:`result_to_dict` / SARIF properties changes.
EXPORT_SCHEMA_VERSION = 1
SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_TOOL_NAME = "TensorGuard-Symexec"
_TOOL_URI = "https://github.com/thehalleyyoung/halley-labs"


def _tool_version() -> str:
    try:  # the package version is the honest tool version when available
        from .. import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "0"


def _sarif_level(severity: str) -> str:
    """Map a symexec severity onto a SARIF result ``level``."""
    s = (severity or "").lower()
    if s in ("error", "fatal"):
        return "error"
    if s in ("warning", "warn"):
        return "warning"
    if s in ("note", "info", "information"):
        return "note"
    return "warning"


def _provenance(bug) -> Dict[str, Any]:
    """Structured provenance (derivation/counterexample/…) for one bug.

    Reuses the Step-65 explainer so the JSON/SARIF view and the ``--explain``
    text view never diverge."""
    exp = explain_bug(bug)
    out: Dict[str, Any] = {}
    if exp.derivation:
        out["derivation"] = list(exp.derivation)
    if exp.counterexample:
        out["counterexample"] = exp.counterexample
    if exp.certificate:
        out["certificate"] = exp.certificate
    if exp.minimal_conditions:
        out["minimal_conditions"] = exp.minimal_conditions
    if exp.notes:
        out["notes"] = list(exp.notes)
    return out


def bug_to_dict(bug) -> Dict[str, Any]:
    """One symexec bug as a stable JSON object with all symexec-specific fields.

    A superset of :meth:`SymBug.to_dict` that also carries the structured
    provenance parsed from the report's evidence."""
    d = bug.to_dict()
    prov = _provenance(bug)
    if prov:
        d["provenance"] = prov
    return d


def result_to_dict(result, filename: str = "<unknown>") -> Dict[str, Any]:
    """A stable JSON object for one analyzed file's :class:`SymResult`."""
    coverage = {c.value: n for c, n in result.abstentions.coverage().items()}
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "file": filename,
        "functions_analyzed": result.functions_analyzed,
        "ran_main": result.ran_main,
        "fingerprint": result.fingerprint(),
        "bugs": [bug_to_dict(b) for b in result.bugs],
        "abstain_total": result.abstentions.total,
        "abstain_coverage": coverage,
    }


def _rules_for(result) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build the deduplicated SARIF ``rules`` list (one per kind seen) and a
    kind→index map for ``ruleIndex`` cross-references."""
    rules: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    for b in result.bugs:
        rid = b.kind.value
        if rid in index:
            continue
        index[rid] = len(rules)
        name = "".join(part.capitalize() for part in rid.split("_"))
        rules.append(
            {
                "id": rid,
                "name": name,
                "shortDescription": {"text": name},
                "defaultConfiguration": {"level": "error"},
            }
        )
    return rules, index


def sarif_replacement(
    original_source: str, patched_source: str
) -> Optional[Dict[str, Any]]:
    """Compute the minimal SARIF ``artifactChanges`` *replacement* turning
    ``original_source`` into ``patched_source``, or ``None`` if they are equal.

    We diff at line granularity and emit one replacement covering the single
    contiguous bounding region of all changed lines.  A pure insertion (no
    original line deleted) becomes a zero-width ``deletedRegion`` at the start of
    the insertion line; a line-block change becomes a full-line
    ``startLine..endLine`` deletion with the patched lines as ``insertedContent``.
    The result is a precise, GitHub-renderable "Apply suggested fix" edit.
    """
    a = original_source.splitlines()
    b = patched_source.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    i1 = min(o[1] for o in ops)
    i2 = max(o[2] for o in ops)
    j1 = min(o[3] for o in ops)
    j2 = max(o[4] for o in ops)
    inserted = "\n".join(b[j1:j2])
    if i2 > i1:  # full-line block replacement (covers replace / delete)
        deleted_region: Dict[str, Any] = {"startLine": i1 + 1, "endLine": i2}
        inserted_text = inserted
    else:  # pure insertion before line i1+1
        deleted_region = {"startLine": i1 + 1, "startColumn": 1, "endColumn": 1}
        inserted_text = inserted + "\n"
    return {
        "deletedRegion": deleted_region,
        "insertedContent": {"text": inserted_text},
    }


def _result_object(
    bug,
    filename: str,
    rule_index: Dict[str, int],
    fix: Optional[Tuple[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rid = bug.kind.value
    region = {"startLine": max(int(bug.line), 1), "startColumn": int(bug.col) + 1}
    props: Dict[str, Any] = {
        "kind": rid,
        "confidence": bug.confidence,
        "function": bug.function,
    }
    if bug.fix_suggestion:
        props["fix_suggestion"] = bug.fix_suggestion
    if bug.evidence:
        props["evidence"] = bug.evidence
    prov = _provenance(bug)
    if prov:
        props["provenance"] = prov
    obj: Dict[str, Any] = {
        "ruleId": rid,
        "level": _sarif_level(bug.severity),
        "message": {"text": bug.message},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": filename},
                    "region": region,
                }
            }
        ],
        "properties": props,
    }
    if rid in rule_index:
        obj["ruleIndex"] = rule_index[rid]
    if fix is not None:
        description, replacement = fix
        obj["fixes"] = [
            {
                "description": {"text": description},
                "artifactChanges": [
                    {
                        "artifactLocation": {"uri": filename},
                        "replacements": [replacement],
                    }
                ],
            }
        ]
    return obj


def result_to_sarif_run(
    result,
    filename: str = "<unknown>",
    fixes_by_loc: Optional[Dict[Tuple[str, int], Tuple[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """A single SARIF ``run`` object for one analyzed file.

    The proof fingerprint and abstain-coverage profile live in the run's
    ``properties`` so a consumer gets the reproducibility receipt alongside the
    findings.  When ``fixes_by_loc`` maps a ``(kind, line)`` to a
    ``(description, replacement)`` (see :func:`sarif_replacement`), the matching
    result object gains a SARIF ``fixes[]`` entry — a GitHub-renderable
    "Apply suggested fix"."""
    rules, rule_index = _rules_for(result)
    fixes_by_loc = fixes_by_loc or {}
    results = [
        _result_object(
            b, filename, rule_index,
            fix=fixes_by_loc.get((b.kind.value, int(b.line))),
        )
        for b in result.bugs
    ]
    coverage = {c.value: n for c, n in result.abstentions.coverage().items()}
    return {
        "tool": {
            "driver": {
                "name": _TOOL_NAME,
                "informationUri": _TOOL_URI,
                "version": _tool_version(),
                "rules": rules,
            }
        },
        "results": results,
        "properties": {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "fingerprint": result.fingerprint(),
            "functions_analyzed": result.functions_analyzed,
            "ran_main": result.ran_main,
            "abstain_total": result.abstentions.total,
            "abstain_coverage": coverage,
        },
    }


def to_sarif(items: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """A complete SARIF 2.1.0 log for many ``(filename, SymResult)`` pairs.

    Emits one ``run`` per file so each file keeps its own fingerprint and
    abstain profile.  The output is deterministic for deterministic inputs."""
    runs = [result_to_sarif_run(result, filename) for filename, result in items]
    return {
        "version": SARIF_VERSION,
        "$schema": SARIF_SCHEMA,
        "runs": runs,
    }
