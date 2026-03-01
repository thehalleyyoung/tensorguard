"""MLflow integration hook for TensorGuard.

Logs verification metrics and certificates as MLflow metrics/artifacts.

Usage::

    from src.integrations.mlflow_hook import MLflowHook
    from src.api import verify_architecture

    hook = MLflowHook()
    result = verify_architecture(source, hooks=[hook])
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False


@dataclass
class MLflowHook:
    """TensorGuard hook that logs verification results to MLflow.

    Attributes:
        experiment_name: MLflow experiment name.
        run_name: Optional run name.
        enabled: If False, all logging is silently skipped.
    """

    experiment_name: str = "tensorguard"
    run_name: Optional[str] = None
    enabled: bool = True

    _run: Any = field(default=None, repr=False)

    # ── Hook protocol ─────────────────────────────────────────────

    def on_verification_start(self, *, source: str, filename: str, **kwargs: Any) -> None:
        """Called when verification begins."""
        if not self._is_active():
            return
        self._ensure_run()
        mlflow.log_param("filename", filename)  # type: ignore[union-attr]
        mlflow.log_param("source_lines", len(source.splitlines()))  # type: ignore[union-attr]

    def on_verification_end(self, *, result: Any, **kwargs: Any) -> None:
        """Called when verification finishes."""
        if not self._is_active():
            return
        self._ensure_run()

        verdict = self._compute_verdict(result)

        mlflow.log_metric("bug_count", result.bug_count)  # type: ignore[union-attr]
        mlflow.log_metric("guards_harvested", result.guards_harvested)  # type: ignore[union-attr]
        mlflow.log_metric("functions_analyzed", result.functions_analyzed)  # type: ignore[union-attr]
        mlflow.log_metric("lines_analyzed", result.lines_analyzed)  # type: ignore[union-attr]
        mlflow.log_metric("duration_ms", result.duration_ms)  # type: ignore[union-attr]

        # Encode verdict as numeric: safe=0, unsafe=1, unknown=2
        verdict_map = {"safe": 0, "unsafe": 1, "unknown": 2}
        mlflow.log_metric("verdict_code", verdict_map.get(verdict, 2))  # type: ignore[union-attr]

        cegar_iters = getattr(result, "_cegar_iterations", None)
        if cegar_iters is not None:
            mlflow.log_metric("cegar_iterations", cegar_iters)  # type: ignore[union-attr]

        shape_contracts = getattr(result, "_shape_contracts", None)
        if shape_contracts is not None:
            mlflow.log_metric("predicate_count", len(shape_contracts))  # type: ignore[union-attr]

        # Log certificate/counterexample as artifact
        self._log_artifact(result, verdict)

    def on_cegar_iteration(self, *, iteration: int, status: str,
                           predicates_discovered: int = 0, **kwargs: Any) -> None:
        """Called after each CEGAR iteration."""
        if not self._is_active():
            return
        self._ensure_run()
        mlflow.log_metric("cegar_predicates", predicates_discovered, step=iteration)  # type: ignore[union-attr]

    def close(self) -> None:
        """End the MLflow run."""
        if self._run is not None:
            mlflow.end_run()  # type: ignore[union-attr]
            self._run = None

    # ── Internal helpers ──────────────────────────────────────────

    def _is_active(self) -> bool:
        return self.enabled and _HAS_MLFLOW

    def _ensure_run(self) -> None:
        if self._run is None and _HAS_MLFLOW:
            mlflow.set_experiment(self.experiment_name)  # type: ignore[union-attr]
            self._run = mlflow.start_run(run_name=self.run_name)  # type: ignore[union-attr]

    @staticmethod
    def _compute_verdict(result: Any) -> str:
        if result.bug_count == 0:
            return "safe"
        high_conf = any(
            getattr(b, "confidence", 0) >= 0.9
            for b in result.bugs
        )
        return "unsafe" if high_conf else "unknown"

    def _log_artifact(self, result: Any, verdict: str) -> None:
        """Log a JSON report as an MLflow artifact."""
        if not _HAS_MLFLOW:
            return

        summary: Dict[str, Any] = {
            "verdict": verdict,
            "bug_count": result.bug_count,
            "duration_ms": result.duration_ms,
        }

        if result.bugs:
            summary["bugs"] = [
                {
                    "category": b.category.value,
                    "message": b.message,
                    "line": b.location.line,
                    "severity": b.severity,
                    "confidence": b.confidence,
                }
                for b in result.bugs
            ]

        cert = getattr(result, "_liquid_contracts", None)
        if cert:
            summary["contracts"] = {name: str(c) for name, c in cert.items()}

        shape_contracts = getattr(result, "_shape_contracts", None)
        if shape_contracts:
            summary["shape_predicates"] = [str(p) for p in shape_contracts]

        tmpdir = tempfile.mkdtemp()
        report_path = os.path.join(tmpdir, "tensorguard_report.json")
        try:
            with open(report_path, "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact(report_path)  # type: ignore[union-attr]
        finally:
            os.unlink(report_path)
            os.rmdir(tmpdir)
