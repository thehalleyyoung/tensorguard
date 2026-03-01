"""Weights & Biases integration hook for TensorGuard.

Logs verification verdicts, timing, predicate counts, CEGAR iteration
metrics, and certificate/counterexample summaries as W&B artifacts.

Usage::

    from src.integrations.wandb_hook import WandbHook
    from src.api import verify_architecture

    hook = WandbHook(project="my-project")
    result = verify_architecture(source, hooks=[hook])
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False


@dataclass
class WandbHook:
    """TensorGuard hook that logs verification results to W&B.

    Attributes:
        project: W&B project name.
        entity: W&B entity (team/user). None uses default.
        run_name: Optional run name. Auto-generated if None.
        tags: Tags to attach to the W&B run.
        enabled: If False, all logging is silently skipped.
    """

    project: str = "tensorguard"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["tensorguard"])
    enabled: bool = True

    _run: Any = field(default=None, repr=False)

    # ── Hook protocol ─────────────────────────────────────────────

    def on_verification_start(self, *, source: str, filename: str, **kwargs: Any) -> None:
        """Called when verification begins."""
        if not self._is_active():
            return
        self._ensure_run()
        self._run.config.update({  # type: ignore[union-attr]
            "filename": filename,
            "source_lines": len(source.splitlines()),
        })

    def on_verification_end(self, *, result: Any, **kwargs: Any) -> None:
        """Called when verification finishes.

        ``result`` is an ``AnalysisResult`` from ``src.api``.
        """
        if not self._is_active():
            return
        self._ensure_run()

        verdict = self._compute_verdict(result)
        metrics: Dict[str, Any] = {
            "verdict": verdict,
            "bug_count": result.bug_count,
            "guards_harvested": result.guards_harvested,
            "functions_analyzed": result.functions_analyzed,
            "lines_analyzed": result.lines_analyzed,
            "duration_ms": result.duration_ms,
        }

        # CEGAR metrics if available
        cegar_iters = getattr(result, "_cegar_iterations", None)
        if cegar_iters is not None:
            metrics["cegar_iterations"] = cegar_iters

        shape_contracts = getattr(result, "_shape_contracts", None)
        if shape_contracts is not None:
            metrics["predicate_count"] = len(shape_contracts)

        self._run.log(metrics)  # type: ignore[union-attr]

        # Log certificate or counterexample as artifact
        self._log_artifact(result)

    def on_cegar_iteration(self, *, iteration: int, status: str,
                           predicates_discovered: int = 0, **kwargs: Any) -> None:
        """Called after each CEGAR iteration."""
        if not self._is_active():
            return
        self._ensure_run()
        self._run.log({  # type: ignore[union-attr]
            "cegar/iteration": iteration,
            "cegar/status": status,
            "cegar/predicates_discovered": predicates_discovered,
        })

    def close(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None

    # ── Internal helpers ──────────────────────────────────────────

    def _is_active(self) -> bool:
        return self.enabled and _HAS_WANDB

    def _ensure_run(self) -> None:
        if self._run is None and _HAS_WANDB:
            self._run = wandb.init(  # type: ignore[union-attr]
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                tags=self.tags,
                reinit=True,
            )

    @staticmethod
    def _compute_verdict(result: Any) -> str:
        if result.bug_count == 0:
            return "safe"
        # Check if any bug has high confidence
        high_conf = any(
            getattr(b, "confidence", 0) >= 0.9
            for b in result.bugs
        )
        return "unsafe" if high_conf else "unknown"

    def _log_artifact(self, result: Any) -> None:
        """Log certificate or counterexample summary as W&B artifact."""
        if not _HAS_WANDB or self._run is None:
            return

        cert = getattr(result, "_liquid_contracts", None)
        shape_contracts = getattr(result, "_shape_contracts", None)

        summary: Dict[str, Any] = {
            "verdict": self._compute_verdict(result),
            "bug_count": result.bug_count,
            "duration_ms": result.duration_ms,
        }

        if cert:
            summary["contracts"] = {
                name: str(c) for name, c in cert.items()
            }
        if shape_contracts:
            summary["shape_predicates"] = [str(p) for p in shape_contracts]

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

        artifact = wandb.Artifact(  # type: ignore[union-attr]
            name="tensorguard-report",
            type="verification-report",
        )
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(summary, f, indent=2)
            tmp_path = f.name
        try:
            artifact.add_file(tmp_path, name="report.json")
            self._run.log_artifact(artifact)
        finally:
            os.unlink(tmp_path)
