#!/usr/bin/env python3
"""Paper-results freshness bot.

This scheduled/PR gate closes the gap between "the paper results look good" and
"the paper results were just regenerated from source." It first runs the
deterministic freshness checks for the committed evaluation artifacts, then
normalizes the paper-facing dashboards into JSON snapshots and feeds them
through ``scripts.benchmark_delta``. Any stale artifact, missing dashboard row,
or quality regression fails the command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation import dashboard, deployment_dashboard  # noqa: E402
from scripts.benchmark_delta import DeltaReport, compute_delta, render_markdown  # noqa: E402

OUT_JSON = os.path.join(ROOT, "reproducibility", "paper_results_freshness.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "paper_results_freshness.md")

FRESHNESS_COMMANDS: Sequence[Sequence[str]] = (
    (sys.executable, "evaluation/precision_recall.py", "--check"),
    (sys.executable, "evaluation/sound_mode_fp.py", "--check"),
    (sys.executable, "evaluation/diff_fuzz.py", "--check"),
    (sys.executable, "evaluation/neg_fuzz.py", "--check"),
    (sys.executable, "evaluation/hard_recall.py", "--check"),
    (sys.executable, "evaluation/triage.py", "--check"),
    (sys.executable, "evaluation/dashboard.py", "--check"),
    (sys.executable, "evaluation/deployment_dashboard.py", "--check"),
)

REGENERATION_COMMANDS: Sequence[Sequence[str]] = (
    (sys.executable, "evaluation/precision_recall.py"),
    (sys.executable, "evaluation/sound_mode_fp.py"),
    (sys.executable, "evaluation/diff_fuzz.py"),
    (sys.executable, "evaluation/neg_fuzz.py"),
    (sys.executable, "evaluation/hard_recall.py"),
    (sys.executable, "evaluation/triage.py"),
    (sys.executable, "evaluation/dashboard.py"),
    (sys.executable, "evaluation/deployment_dashboard.py"),
)


@dataclass(frozen=True)
class SnapshotCheck:
    name: str
    baseline: Dict[str, Any]
    current: Dict[str, Any]
    report: DeltaReport

    @property
    def ok(self) -> bool:
        return not self.report.has_regression()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def quality_baseline_snapshot() -> Dict[str, float]:
    """Return the reviewed paper-quality dashboard baseline as flat metrics."""

    baseline = dashboard.load_baseline()
    metrics = baseline.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("dashboard baseline must contain a metric map")
    return {
        key: float(entry["value"])
        for key, entry in sorted(metrics.items())
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float))
    }


def quality_current_snapshot() -> Dict[str, float]:
    """Recompute current paper-quality metrics from committed artifacts."""

    return {key: float(value) for key, value in sorted(dashboard.compute_metrics().items())}


def deployment_baseline_snapshot() -> Dict[str, float]:
    """Summarize the reviewed deployment baseline into benchmark-delta metrics."""

    baseline = deployment_dashboard.load_baseline()
    rows = baseline.get("rows", {})
    if not isinstance(rows, dict):
        raise ValueError("deployment baseline must contain a row map")
    supported = [row for row in rows.values() if isinstance(row, dict) and row.get("supported")]
    return {
        "supported_rows": float(len(supported)),
        "supported_passed": float(sum(1 for row in supported if row.get("status") == "passed")),
        "supported_failed": float(sum(1 for row in supported if row.get("status") == "failed")),
        "supported_skipped": float(sum(1 for row in supported if row.get("status") == "skipped")),
        "total_rows": float(len(rows)),
    }


def deployment_current_snapshot() -> Dict[str, float]:
    """Summarize the current deployment manifest into benchmark-delta metrics."""

    manifest = _read_json(deployment_dashboard.JSON_PATH)
    releases = manifest.get("releases", [])
    summaries = [
        release.get("summary", {})
        for release in releases
        if isinstance(release, dict)
    ]
    if not summaries:
        raise ValueError("deployment dashboard must contain at least one release summary")
    summary = summaries[-1]
    return {
        key: float(summary[key])
        for key in (
            "supported_rows",
            "supported_passed",
            "supported_failed",
            "supported_skipped",
            "total_rows",
        )
    }


def build_checks() -> List[SnapshotCheck]:
    pairs = [
        ("paper_quality_dashboard", quality_baseline_snapshot(), quality_current_snapshot()),
        ("deployment_release_dashboard", deployment_baseline_snapshot(), deployment_current_snapshot()),
    ]
    return [
        SnapshotCheck(name=name, baseline=baseline, current=current,
                      report=compute_delta(baseline, current))
        for name, baseline, current in pairs
    ]


def _run_commands(commands: Iterable[Sequence[str]]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_regeneration_commands(commands: Iterable[Sequence[str]] = REGENERATION_COMMANDS) -> None:
    _run_commands(commands)


def run_freshness_commands(commands: Iterable[Sequence[str]] = FRESHNESS_COMMANDS) -> None:
    _run_commands(commands)


def render_report(checks: Sequence[SnapshotCheck]) -> str:
    lines = [
        "# Paper results freshness",
        "",
        (
            "Scheduled/PR bot output: committed benchmark artifacts are regenerated "
            "by their `--check` commands, then paper-facing snapshots are compared "
            "with `scripts/benchmark_delta.py` semantics."
        ),
        "",
    ]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        lines.extend([
            f"## {check.name} — {status}",
            "",
            render_markdown(check.report, f"{check.name}:baseline", f"{check.name}:current").strip(),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def result_json(checks: Sequence[SnapshotCheck]) -> Dict[str, Any]:
    return {
        "ok": all(check.ok for check in checks),
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "regressions": [delta.path for delta in check.report.regressions],
                "improvements": [delta.path for delta in check.report.improvements],
                "metric_count": len(check.report.deltas),
            }
            for check in checks
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-rerun", action="store_true",
                        help="Compare committed snapshots without running artifact freshness commands.")
    parser.add_argument("--refresh", action="store_true",
                        help="Regenerate paper-facing artifacts before running freshness checks.")
    parser.add_argument("--json", action="store_true",
                        help="Print the machine-readable summary to stdout.")
    parser.add_argument("--write", action="store_true",
                        help="Write reproducibility/paper_results_freshness.{json,md}.")
    args = parser.parse_args(argv)

    if args.refresh:
        run_regeneration_commands()
    if not args.skip_rerun:
        run_freshness_commands()
    checks = build_checks()
    summary = result_json(checks)
    markdown = render_report(checks)

    if args.write:
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        with open(OUT_MD, "w", encoding="utf-8") as fh:
            fh.write(markdown)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        sys.stdout.write(markdown)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
