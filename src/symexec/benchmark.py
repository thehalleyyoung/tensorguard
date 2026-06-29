"""Performance benchmarking for the symbolic-execution engine (Step 78).

The engine ships with deterministic worst-case iteration caps — bounded
recursion depth, loop unroll/fixpoint/narrow passes and disjunction width (see
:data:`src.symexec.engine.ITERATION_CAPS`) — so the cost of analysing any single
construct is bounded *independently of wall-clock time*.  That is what turns a
per-file latency target into an enforceable contract.

This module is the measurement side of that contract.  It is torch-free and
deterministic: :func:`benchmark_source` times :func:`~src.symexec.engine.analyze_source`
over a source string (taking the *minimum* of a few repeats to damp scheduler
noise) and records the latency alongside the analysis facts (statements, bugs,
coverage, abstentions, whether a budget tripped).  :func:`benchmark_paths`
aggregates over a tree of files and :func:`summarise` rolls the records up into a
latency profile (mean / p95 / max / slowest file) suitable for a CI regression
gate.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .abstain import AbstainCategory
from .engine import ITERATION_CAPS, analyze_source

__all__ = [
    "BenchmarkRecord",
    "benchmark_source",
    "benchmark_paths",
    "summarise",
    "ITERATION_CAPS",
]


@dataclass(frozen=True)
class BenchmarkRecord:
    """The latency and analysis facts for one benchmarked source."""

    filename: str
    wall_ms: float
    statements: int
    bugs: int
    coverage: float
    abstentions: int
    budget_exceeded: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "wall_ms": round(self.wall_ms, 4),
            "statements": self.statements,
            "bugs": self.bugs,
            "coverage": round(self.coverage, 6),
            "abstentions": self.abstentions,
            "budget_exceeded": self.budget_exceeded,
            "error": self.error,
        }


def benchmark_source(
    source: str,
    filename: str = "<bench>",
    repeats: int = 3,
    budget_ms: Optional[float] = None,
) -> BenchmarkRecord:
    """Time :func:`analyze_source` over ``source``.

    The reported ``wall_ms`` is the *minimum* over ``repeats`` runs — the run
    least perturbed by GC / scheduler noise, the standard practice for a stable
    latency floor.  A crash in the engine is captured (never raised) so a single
    pathological file can't abort a whole benchmark sweep; the record's
    ``error`` field carries the exception text and its latency the time to fail.
    """
    repeats = max(1, repeats)
    best = float("inf")
    result = None
    error: Optional[str] = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            result = analyze_source(source, filename=filename, budget_ms=budget_ms)
        except Exception as exc:  # pragma: no cover - defensive; engine never should
            error = f"{type(exc).__name__}: {exc}"
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    if result is None:
        return BenchmarkRecord(
            filename=filename,
            wall_ms=best,
            statements=0,
            bugs=0,
            coverage=0.0,
            abstentions=0,
            budget_exceeded=False,
            error=error,
        )
    budget_hit = any(
        r.category is AbstainCategory.RESOURCE_BUDGET
        for r in result.abstentions.reasons
    )
    return BenchmarkRecord(
        filename=filename,
        wall_ms=best,
        statements=result.coverage.total,
        bugs=len(result.bugs),
        coverage=result.coverage.coverage,
        abstentions=result.abstentions.total,
        budget_exceeded=budget_hit,
        error=error,
    )


def benchmark_paths(
    paths: Sequence[str],
    repeats: int = 3,
    budget_ms: Optional[float] = None,
) -> List[BenchmarkRecord]:
    """Benchmark every ``*.py`` file under ``paths`` (files or directories).

    Files are visited in sorted order so the sweep is deterministic.  An
    unreadable file becomes a record with an ``error`` rather than aborting the
    sweep."""
    records: List[BenchmarkRecord] = []
    seen: set = set()
    for raw in paths:
        p = pathlib.Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("*.py"))
        elif p.is_file():
            found = [p]
        else:
            found = []
        for f in found:
            key = str(f.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                source = f.read_text(encoding="utf-8")
            except OSError as exc:
                records.append(
                    BenchmarkRecord(
                        filename=str(f),
                        wall_ms=0.0,
                        statements=0,
                        bugs=0,
                        coverage=0.0,
                        abstentions=0,
                        budget_exceeded=False,
                        error=f"unreadable: {exc}",
                    )
                )
                continue
            records.append(
                benchmark_source(
                    source, filename=str(f), repeats=repeats, budget_ms=budget_ms
                )
            )
    return records


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list (``q`` in ``[0, 1]``)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[rank]


def summarise(records: Sequence[BenchmarkRecord]) -> dict:
    """Roll a set of records up into a latency profile for a CI regression gate.

    Reports counts, total/mean/p95/max wall-time, the slowest file, and the
    engine's iteration caps (so a profile is self-describing about the bounds
    that produced it)."""
    lat = sorted(r.wall_ms for r in records)
    slowest = max(records, key=lambda r: r.wall_ms, default=None)
    total = sum(lat)
    return {
        "files": len(records),
        "errors": sum(1 for r in records if r.error),
        "budget_exceeded": sum(1 for r in records if r.budget_exceeded),
        "total_ms": round(total, 4),
        "mean_ms": round(total / len(records), 4) if records else 0.0,
        "p95_ms": round(_percentile(lat, 0.95), 4),
        "max_ms": round(lat[-1], 4) if lat else 0.0,
        "slowest_file": slowest.filename if slowest else None,
        "iteration_caps": dict(ITERATION_CAPS),
    }
