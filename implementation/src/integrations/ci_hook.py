"""CI / GitHub Actions integration hook for TensorGuard.

Generates machine-readable JSON reports, SARIF output for GitHub Code
Scanning, and returns proper exit codes.

Exit codes:
    0 — model is safe (no bugs found)
    1 — bug found (at least one high-confidence violation)
    2 — unknown (analysis inconclusive or timed out)

Usage::

    from src.integrations.ci_hook import CIHook
    hook = CIHook(output_dir="reports/", deterministic=True)
    result = verify_architecture(source, hooks=[hook])
    sys.exit(hook.exit_code)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Exit codes
EXIT_SAFE = 0
EXIT_BUG_FOUND = 1
EXIT_UNKNOWN = 2


@dataclass
class CIHook:
    """TensorGuard hook for CI/CD pipelines.

    Attributes:
        output_dir: Directory to write report files. None = don't write.
        deterministic: If True, skip the neuro-symbolic LLM pipeline
            and only use the deterministic SMT-based verifier.
        sarif: If True, generate SARIF-format output alongside JSON.
        enabled: If False, all operations are silently skipped.
    """

    output_dir: Optional[str] = None
    deterministic: bool = False
    sarif: bool = True
    enabled: bool = True

    exit_code: int = field(default=EXIT_UNKNOWN, init=False)
    _report: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _sarif_report: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)

    # ── Hook protocol ─────────────────────────────────────────────

    def on_verification_start(self, *, source: str, filename: str, **kwargs: Any) -> None:
        """Called when verification begins."""
        if not self.enabled:
            return
        self._start_time = time.monotonic()

    def on_verification_end(self, *, result: Any, **kwargs: Any) -> None:
        """Called when verification finishes."""
        if not self.enabled:
            return

        verdict = self._compute_verdict(result)
        self.exit_code = {
            "safe": EXIT_SAFE,
            "unsafe": EXIT_BUG_FOUND,
            "unknown": EXIT_UNKNOWN,
        }.get(verdict, EXIT_UNKNOWN)

        self._report = self._build_json_report(result, verdict)

        # SARIF generation
        if self.sarif and hasattr(result, "to_sarif"):
            contracts = getattr(result, "_liquid_contracts", None)
            self._sarif_report = result.to_sarif(contracts=contracts)
        elif self.sarif:
            self._sarif_report = self._build_sarif_report(result)

        # Write files
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            json_path = os.path.join(self.output_dir, "tensorguard-report.json")
            with open(json_path, "w") as f:
                json.dump(self._report, f, indent=2)

            if self._sarif_report:
                sarif_path = os.path.join(self.output_dir, "tensorguard.sarif")
                with open(sarif_path, "w") as f:
                    json.dump(self._sarif_report, f, indent=2)

    def on_cegar_iteration(self, *, iteration: int, status: str,
                           predicates_discovered: int = 0, **kwargs: Any) -> None:
        """Called after each CEGAR iteration (no-op for CI)."""
        pass

    def close(self) -> None:
        """No-op for CI hook."""
        pass

    # ── Public accessors ──────────────────────────────────────────

    @property
    def json_report(self) -> Optional[Dict[str, Any]]:
        """The JSON report dict, available after on_verification_end."""
        return self._report

    @property
    def sarif_report(self) -> Optional[Dict[str, Any]]:
        """The SARIF report dict, available after on_verification_end."""
        return self._sarif_report

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _compute_verdict(result: Any) -> str:
        if result.bug_count == 0:
            return "safe"
        high_conf = any(
            getattr(b, "confidence", 0) >= 0.9
            for b in result.bugs
        )
        return "unsafe" if high_conf else "unknown"

    def _build_json_report(self, result: Any, verdict: str) -> Dict[str, Any]:
        elapsed = (time.monotonic() - self._start_time) * 1000
        report: Dict[str, Any] = {
            "tool": "tensorguard",
            "version": "0.2.0",
            "verdict": verdict,
            "exit_code": self.exit_code,
            "deterministic": self.deterministic,
            "bug_count": result.bug_count,
            "guards_harvested": result.guards_harvested,
            "functions_analyzed": result.functions_analyzed,
            "lines_analyzed": result.lines_analyzed,
            "duration_ms": round(elapsed, 2),
            "bugs": [],
        }

        for b in result.bugs:
            report["bugs"].append({
                "category": b.category.value,
                "message": b.message,
                "file": b.location.file,
                "line": b.location.line,
                "column": b.location.column,
                "severity": b.severity,
                "confidence": b.confidence,
            })

        cegar_iters = getattr(result, "_cegar_iterations", None)
        if cegar_iters is not None:
            report["cegar_iterations"] = cegar_iters

        shape_contracts = getattr(result, "_shape_contracts", None)
        if shape_contracts is not None:
            report["predicate_count"] = len(shape_contracts)

        return report

    def _build_sarif_report(self, result: Any) -> Dict[str, Any]:
        """Build a SARIF 2.1.0 report from an AnalysisResult."""
        from src.api import BugCategory

        rules = [
            {
                "id": c.value,
                "shortDescription": {"text": c.value.replace("_", " ").title()},
            }
            for c in BugCategory
        ]

        results_list: List[Dict[str, Any]] = []
        for b in result.bugs:
            sarif_result: Dict[str, Any] = {
                "ruleId": b.category.value,
                "level": "error" if b.severity == "error" else "warning",
                "message": {"text": b.message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": b.location.file},
                            "region": {
                                "startLine": b.location.line,
                                "startColumn": b.location.column,
                            },
                        }
                    }
                ],
            }
            results_list.append(sarif_result)

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "TensorGuard",
                            "version": "0.2.0",
                            "informationUri": "https://github.com/tensorguard/tensorguard",
                            "rules": rules,
                        }
                    },
                    "results": results_list,
                }
            ],
        }
