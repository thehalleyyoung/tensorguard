"""Step 67 — finalize GitHub Code Scanning-ready SARIF 2.1.0 output.

GitHub Advanced Security / Code Scanning ingests a SARIF 2.1.0 log and renders
each result as a code-scanning alert on the offending line.  For that ingestion
to succeed and for alerts to be *tracked* across commits, a log must:

* declare ``version == "2.1.0"`` and a ``$schema``;
* name the driver and list every rule it reports under ``tool.driver.rules``;
* have every ``result.ruleId`` resolve to one of those rules;
* give each result a ``level`` in {error, warning, note}, a message, and a
  ``physicalLocation`` with an ``artifactLocation.uri`` and a 1-based region;
* carry ``partialFingerprints`` so GitHub can de-duplicate and track alerts even
  when line numbers shift.

This module builds exactly such a log from the public :class:`AnalysisResult`s
that ``verify_architecture`` already produces, so the action and the CLI emit
one canonical, validated artifact.  It is pure and offline-testable.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Schemata/sarif-schema-2.1.0.json"
)
TOOL_NAME = "TensorGuard"
INFORMATION_URI = "https://github.com/thehalleyyoung/tensorguard"
_VALID_LEVELS = {"error", "warning", "note"}
_TAG_RE = re.compile(r"^\s*\[([A-Z0-9_\-]+)\]")

# Curated metadata for the rule kinds TensorGuard emits. Unknown kinds get a
# generic-but-valid rule synthesized on the fly.
_RULE_METADATA: Dict[str, Dict[str, str]] = {
    "shape-incompatible": {
        "name": "ShapeIncompatible",
        "short": "Incompatible tensor shape",
        "full": "A layer receives a tensor whose shape is incompatible with the "
        "shape it requires; the forward pass would raise at runtime.",
    },
    "cegar-real-bug": {
        "name": "ShapeIncompatibleProven",
        "short": "Z3-proven shape bug",
        "full": "A counterexample-guided refinement proof found a concrete input "
        "for which the module's shapes are inconsistent.",
    },
    "device-mismatch": {
        "name": "DeviceMismatch",
        "short": "Tensor device mismatch",
        "full": "An operation combines tensors on different devices; the forward "
        "pass would raise a device-mismatch error.",
    },
    "dtype-mismatch": {
        "name": "DtypeMismatch",
        "short": "Tensor dtype mismatch",
        "full": "An operation combines tensors of incompatible dtypes.",
    },
    "phase-error": {
        "name": "PhaseError",
        "short": "Train/eval phase error",
        "full": "A layer is used in a train/eval phase it does not support.",
    },
    "gradient-error": {
        "name": "GradientError",
        "short": "Gradient-flow error",
        "full": "Gradient flow is inconsistent with the module's requirements.",
    },
}
_GENERIC_RULE = {
    "name": "VerificationIssue",
    "short": "TensorGuard verification issue",
    "full": "TensorGuard found a static verification issue in this module.",
}


def _rule_tag(message: str) -> str:
    m = _TAG_RE.match(message or "")
    return m.group(1).lower() if m else ""


def _bug_rule_id(bug: Any) -> str:
    """Stable ruleId for a bug: its ``[KIND]`` tag, else its category."""
    tag = _rule_tag(getattr(bug, "message", "") or "")
    if tag:
        return tag
    cat = getattr(bug, "category", None)
    if cat is not None:
        return str(getattr(cat, "value", cat)).lower().replace("_", "-")
    return "verification-issue"


def _bug_level(bug: Any) -> str:
    sev = getattr(bug, "severity", "error")
    sev = str(getattr(sev, "value", sev)).lower()
    if sev in _VALID_LEVELS:
        return sev
    if sev in ("warn", "warning"):
        return "warning"
    if sev in ("info", "note", "notice"):
        return "note"
    return "error"


def _first_line(text: str) -> str:
    return (text or "").splitlines()[0] if text else ""


def _diag_for_line(result: Any, line: int):
    for d in getattr(result, "diagnostics", None) or []:
        if getattr(d, "source_line", None) == line:
            return d
    return None


def _rule_for_id(rule_id: str) -> Dict[str, Any]:
    meta = _RULE_METADATA.get(rule_id, _GENERIC_RULE)
    return {
        "id": rule_id,
        "name": meta["name"],
        "shortDescription": {"text": meta["short"]},
        "fullDescription": {"text": meta["full"]},
        "helpUri": INFORMATION_URI,
        "help": {"text": meta["full"]},
        "defaultConfiguration": {"level": "error"},
    }


def _partial_fingerprint(rule_id: str, uri: str, message: str) -> str:
    """Deterministic fingerprint for alert tracking across line shifts.

    Intentionally excludes the line number so a moved-but-identical bug keeps
    the same fingerprint, which is how GitHub keeps an alert stable.
    """
    norm = re.sub(r"\d+", "N", _first_line(message))
    h = hashlib.sha256(f"{rule_id}|{uri}|{norm}".encode("utf-8")).hexdigest()
    return h[:16]


def _results_for_file(uri: str, result: Any) -> Tuple[List[dict], set]:
    sarif_results: List[dict] = []
    rule_ids: set = set()
    seen: set = set()
    for bug in getattr(result, "bugs", None) or []:
        loc = getattr(bug, "location", None)
        line = getattr(loc, "line", 0) if loc else 0
        if not line or line <= 0:
            continue
        rule_id = _bug_rule_id(bug)
        diag = _diag_for_line(result, line)
        message = (
            getattr(diag, "message", None)
            if diag is not None
            else None
        ) or _first_line(getattr(bug, "message", ""))
        col = getattr(loc, "column", None)
        if diag is not None and getattr(diag, "source_col", None):
            col = getattr(diag, "source_col")
        key = (rule_id, line, message)
        if key in seen:
            continue
        seen.add(key)
        rule_ids.add(rule_id)
        region: Dict[str, Any] = {"startLine": int(line)}
        if col and col > 0:
            region["startColumn"] = int(col)
        sarif_results.append(
            {
                "ruleId": rule_id,
                "level": _bug_level(bug),
                "message": {"text": message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": uri},
                            "region": region,
                        }
                    }
                ],
                "partialFingerprints": {
                    "tensorguard/v1": _partial_fingerprint(rule_id, uri, message)
                },
            }
        )
    return sarif_results, rule_ids


def build_sarif(
    results_by_file: Iterable[Tuple[str, Any]],
    *,
    tool_version: str = "0.1.0",
    automation_id: str = "tensorguard/verify",
) -> Dict[str, Any]:
    """Build a GitHub Code Scanning-ready SARIF 2.1.0 log.

    ``results_by_file`` is an iterable of ``(uri, AnalysisResult)`` pairs.
    """
    all_results: List[dict] = []
    rule_ids: set = set()
    for uri, result in results_by_file:
        file_results, ids = _results_for_file(uri, result)
        all_results.extend(file_results)
        rule_ids.update(ids)

    rules = [_rule_for_id(rid) for rid in sorted(rule_ids)]
    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": TOOL_NAME,
                        "informationUri": INFORMATION_URI,
                        "version": tool_version,
                        "semanticVersion": tool_version,
                        "rules": rules,
                    }
                },
                "automationDetails": {"id": automation_id},
                "columnKind": "utf16CodeUnits",
                "results": all_results,
            }
        ],
    }


def to_json(sarif: Dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(sarif, indent=indent)


def write_sarif(
    path: str,
    results_by_file: Iterable[Tuple[str, Any]],
    *,
    tool_version: str = "0.1.0",
) -> Dict[str, Any]:
    import pathlib

    sarif = build_sarif(results_by_file, tool_version=tool_version)
    p = pathlib.Path(path)
    if p.parent and str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_json(sarif), encoding="utf-8")
    return sarif


def check_code_scanning_requirements(sarif: Dict[str, Any]) -> List[str]:
    """Return a list of GitHub Code Scanning violations (empty == OK).

    This encodes the documented ingestion requirements so the contract can be
    asserted offline, independently of a network schema fetch.
    """
    problems: List[str] = []
    if sarif.get("version") != SARIF_VERSION:
        problems.append("version must be 2.1.0")
    if not sarif.get("$schema"):
        problems.append("$schema is required")
    runs = sarif.get("runs")
    if not isinstance(runs, list) or not runs:
        problems.append("runs must be a non-empty array")
        return problems
    for ri, run in enumerate(runs):
        driver = run.get("tool", {}).get("driver", {})
        if not driver.get("name"):
            problems.append(f"runs[{ri}].tool.driver.name is required")
        rule_ids = {r.get("id") for r in driver.get("rules", [])}
        for r in driver.get("rules", []):
            if not r.get("id"):
                problems.append(f"runs[{ri}] has a rule without an id")
            if not r.get("shortDescription", {}).get("text"):
                problems.append(f"rule {r.get('id')} missing shortDescription")
        for qi, res in enumerate(run.get("results", [])):
            rp = f"runs[{ri}].results[{qi}]"
            rid = res.get("ruleId")
            if rid not in rule_ids:
                problems.append(f"{rp}.ruleId '{rid}' not in driver.rules")
            if res.get("level") not in _VALID_LEVELS:
                problems.append(f"{rp}.level invalid: {res.get('level')}")
            if not res.get("message", {}).get("text"):
                problems.append(f"{rp}.message.text is required")
            locs = res.get("locations") or []
            if not locs:
                problems.append(f"{rp} has no locations")
                continue
            phys = locs[0].get("physicalLocation", {})
            uri = phys.get("artifactLocation", {}).get("uri")
            if not uri:
                problems.append(f"{rp} missing artifactLocation.uri")
            start = phys.get("region", {}).get("startLine")
            if not isinstance(start, int) or start < 1:
                problems.append(f"{rp} region.startLine must be >= 1")
            if not res.get("partialFingerprints"):
                problems.append(f"{rp} missing partialFingerprints")
    return problems
