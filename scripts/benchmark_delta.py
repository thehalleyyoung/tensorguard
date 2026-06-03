#!/usr/bin/env python3
"""Per-release benchmark-delta report (100_STEPS Phase 11).

Compares two JSON metric snapshots (e.g. the headline 60-bug figure, the
precision/recall confusion matrices, the scaling-walltime fit) from two
releases and emits a Markdown regression report plus a CI-friendly exit code.

The tool is metric-direction aware: a small registry declares whether each
metric path is ``higher_is_better`` or ``lower_is_better``; a change that
worsens a declared metric by more than its tolerance is a *regression* and
flips the exit code to non-zero, so a release that silently degrades recall or
inflates false positives cannot merge.  Metrics with no declared direction are
reported as informational (never gate).

Usage::

    python scripts/benchmark_delta.py OLD.json NEW.json [--report out.md]
    python scripts/benchmark_delta.py OLD.json NEW.json --json   # machine read

Both files are flattened to dot-paths (``headline_regime.silent_miss`` …) so
arbitrarily nested artifacts compare cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── metric-direction registry ────────────────────────────────────────────────
# (substring matched against the flattened dot-path, longest match wins)
#   direction:  +1 → higher is better,  -1 → lower is better
#   tol:        absolute tolerance below which a change is "noise"
_DIRECTIONS: List[Tuple[str, int, float]] = [
    ("refuted_proof_high_confidence", +1, 0.0),
    ("recall", +1, 0.0),
    ("precision", +1, 0.0),
    ("f1", +1, 0.0),
    ("coverage", +1, 0.0),
    ("verified_safe", +1, 0.0),
    ("genuine_faults", +1, 0.0),
    ("population", +1, 0.0),
    ("total", +1, 0.0),
    ("supported_rows", +1, 0.0),
    ("supported_passed", +1, 0.0),
    ("total_rows", +1, 0.0),
    ("true_positive", +1, 0.0),
    ("silent_miss", -1, 0.0),
    ("misses", -1, 0.0),
    ("false_positive", -1, 0.0),
    ("false_positive_rate", -1, 0.0),
    ("false_negative", -1, 0.0),
    ("abstain", -1, 0.0),
    ("supported_failed", -1, 0.0),
    ("supported_skipped", -1, 0.0),
    ("failed", -1, 0.0),
    ("skipped", -1, 0.0),
    ("loglog_scaling_exponent", -1, 0.05),
    ("elapsed_s", -1, 1e9),   # informational: wall-clock is machine dependent
    ("median_ms", -1, 1e9),
]


def _direction_for(path: str) -> Optional[Tuple[int, float]]:
    best: Optional[Tuple[int, int, float]] = None  # (match_len, dir, tol)
    for needle, direction, tol in _DIRECTIONS:
        if needle in path:
            if best is None or len(needle) > best[0]:
                best = (len(needle), direction, tol)
    if best is None:
        return None
    return best[1], best[2]


# ── flattening ───────────────────────────────────────────────────────────────


def flatten(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Flatten nested JSON to {dot.path: number} for numeric leaves only."""
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        out[prefix] = 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


# ── delta computation ────────────────────────────────────────────────────────


@dataclass
class MetricDelta:
    path: str
    old: Optional[float]
    new: Optional[float]
    direction: Optional[int]  # +1 / -1 / None
    tol: float = 0.0

    @property
    def change(self) -> Optional[float]:
        if self.old is None or self.new is None:
            return None
        return self.new - self.old

    @property
    def status(self) -> str:
        if self.old is None:
            return "added"
        if self.new is None:
            return "removed"
        delta = self.change or 0.0
        if abs(delta) <= self.tol or self.direction is None:
            return "info" if self.direction is None else "same"
        improved = (delta > 0) == (self.direction > 0)
        return "improved" if improved else "regressed"


@dataclass
class DeltaReport:
    deltas: List[MetricDelta] = field(default_factory=list)

    @property
    def regressions(self) -> List[MetricDelta]:
        return [d for d in self.deltas if d.status == "regressed"]

    @property
    def improvements(self) -> List[MetricDelta]:
        return [d for d in self.deltas if d.status == "improved"]

    def has_regression(self) -> bool:
        return bool(self.regressions)


def compute_delta(old: Dict[str, Any], new: Dict[str, Any]) -> DeltaReport:
    fo, fn = flatten(old), flatten(new)
    report = DeltaReport()
    for path in sorted(set(fo) | set(fn)):
        dir_tol = _direction_for(path)
        direction = dir_tol[0] if dir_tol else None
        tol = dir_tol[1] if dir_tol else 0.0
        report.deltas.append(
            MetricDelta(
                path=path,
                old=fo.get(path),
                new=fn.get(path),
                direction=direction,
                tol=tol,
            )
        )
    return report


# ── rendering ────────────────────────────────────────────────────────────────


def render_markdown(report: DeltaReport, old_name: str, new_name: str) -> str:
    lines = [
        f"# Benchmark delta: `{old_name}` → `{new_name}`",
        "",
        f"- regressions: **{len(report.regressions)}**",
        f"- improvements: **{len(report.improvements)}**",
        "",
        "| metric | old | new | Δ | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    icon = {
        "regressed": "🔴 regressed",
        "improved": "🟢 improved",
        "same": "⚪ same",
        "info": "ℹ️ info",
        "added": "➕ added",
        "removed": "➖ removed",
    }
    interesting = [
        d for d in report.deltas
        if d.status in ("regressed", "improved", "added", "removed")
    ]
    for d in interesting:
        old = "—" if d.old is None else f"{d.old:g}"
        new = "—" if d.new is None else f"{d.new:g}"
        ch = "—" if d.change is None else f"{d.change:+g}"
        lines.append(f"| `{d.path}` | {old} | {new} | {ch} | {icon[d.status]} |")
    if not interesting:
        lines.append("| _(no changes)_ | | | | |")
    return "\n".join(lines) + "\n"


def _load(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--report", help="write the Markdown report to this path")
    ap.add_argument("--json", action="store_true",
                    help="emit a machine-readable JSON summary on stdout")
    args = ap.parse_args(argv)

    report = compute_delta(_load(args.old), _load(args.new))
    md = render_markdown(report, args.old, args.new)

    if args.report:
        with open(args.report, "w") as fh:
            fh.write(md)
    if args.json:
        print(json.dumps({
            "regressions": [d.path for d in report.regressions],
            "improvements": [d.path for d in report.improvements],
            "has_regression": report.has_regression(),
        }, indent=2))
    else:
        sys.stdout.write(md)
        if report.has_regression():
            sys.stderr.write(
                f"\nREGRESSION: {len(report.regressions)} metric(s) worsened\n"
            )
    return 1 if report.has_regression() else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
