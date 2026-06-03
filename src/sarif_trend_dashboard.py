"""SARIF-to-Code-Scanning trend dashboard.

TensorGuard's SARIF fingerprints are intentionally stable across line shifts,
which makes them good Code Scanning alert identities but not unique row IDs.
This module therefore treats each release snapshot as a multiset of alert
fingerprints: repeated same-rule findings are counted, opened/closed deltas are
computed from counts, and recurrence means "absent or reduced in the previous
release, but seen in an earlier release."
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "tensorguard-sarif-trends"
VERSION = "1.0"
FINGERPRINT_KEYS = (
    "tensorguard/v1",
    "primaryLocationLineHash",
    "primaryLocationStartColumnFingerprint",
)


@dataclass(frozen=True)
class SarifSnapshot:
    """A named SARIF log in release order."""

    release: str
    sarif: Mapping[str, Any]


def _location(result: Mapping[str, Any]) -> Tuple[str, int]:
    locs = result.get("locations") or []
    if not locs:
        return "", 0
    phys = (locs[0] or {}).get("physicalLocation") or {}
    uri = (phys.get("artifactLocation") or {}).get("uri") or ""
    line = (phys.get("region") or {}).get("startLine") or 0
    return str(uri), int(line) if isinstance(line, int) else 0


def _identity(result: Mapping[str, Any]) -> str:
    fps = result.get("partialFingerprints") or {}
    for key in FINGERPRINT_KEYS:
        if fps.get(key):
            return f"fp:{key}:{fps[key]}"
    uri, line = _location(result)
    rule = str(result.get("ruleId") or "verification-issue")
    message = str((result.get("message") or {}).get("text") or "")
    # Fallback is intentionally line-sensitive: without partialFingerprints we
    # cannot promise Code Scanning-style continuity across moved alerts.
    return f"loc:{rule}:{uri}:{line}:{message}"


def extract_alerts(sarif: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten all SARIF runs into alert records with stable identities."""

    alerts: List[Dict[str, Any]] = []
    for run_index, run in enumerate(sarif.get("runs") or []):
        for result_index, result in enumerate(run.get("results") or []):
            if not isinstance(result, Mapping):
                continue
            uri, line = _location(result)
            message = str((result.get("message") or {}).get("text") or "")
            alerts.append(
                {
                    "identity": _identity(result),
                    "ruleId": str(result.get("ruleId") or "verification-issue"),
                    "level": str(result.get("level") or "warning"),
                    "uri": uri,
                    "line": line,
                    "message": message,
                    "run_index": run_index,
                    "result_index": result_index,
                }
            )
    alerts.sort(
        key=lambda a: (
            a["identity"],
            a["ruleId"],
            a["uri"],
            a["line"],
            a["message"],
            a["run_index"],
            a["result_index"],
        )
    )
    return alerts


def _exemplars(alerts: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for alert in alerts:
        ident = str(alert["identity"])
        out.setdefault(
            ident,
            {
                "identity": ident,
                "ruleId": alert["ruleId"],
                "level": alert["level"],
                "uri": alert["uri"],
                "line": alert["line"],
                "message": alert["message"],
            },
        )
    return out


def _release_summary(
    release: str,
    current: Counter,
    previous: Counter,
    ever_seen_before: Counter,
    exemplars: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    keys = set(current) | set(previous) | set(ever_seen_before)
    opened = Counter({k: max(current[k] - previous[k], 0) for k in keys})
    closed = Counter({k: max(previous[k] - current[k], 0) for k in keys})
    carried = Counter({k: min(previous[k], current[k]) for k in keys})
    recurred = Counter(
        {
            k: min(opened[k], max(ever_seen_before[k] - previous[k], 0))
            for k in keys
        }
    )
    per_rule: Counter = Counter()
    for ident, count in current.items():
        rule = str(exemplars.get(ident, {}).get("ruleId", "verification-issue"))
        per_rule[rule] += count
    return {
        "release": release,
        "open_total": int(sum(current.values())),
        "opened": int(sum(opened.values())),
        "closed": int(sum(closed.values())),
        "carried": int(sum(carried.values())),
        "recurred": int(sum(recurred.values())),
        "net_open_delta": int(sum(current.values()) - sum(previous.values())),
        "per_rule_open": dict(sorted(per_rule.items())),
        "opened_identities": sorted(k for k, v in opened.items() if v),
        "closed_identities": sorted(k for k, v in closed.items() if v),
        "recurred_identities": sorted(k for k, v in recurred.items() if v),
    }


def build_trend_dashboard(snapshots: Sequence[SarifSnapshot]) -> Dict[str, Any]:
    """Compute open/closed/recurred alert trends for ordered SARIF snapshots."""

    if not snapshots:
        raise ValueError("at least one SARIF snapshot is required")
    releases = [s.release for s in snapshots]
    if len(set(releases)) != len(releases):
        raise ValueError("release names must be unique")

    previous: Counter = Counter()
    ever_seen: Counter = Counter()
    all_exemplars: Dict[str, Dict[str, Any]] = {}
    release_rows: List[Dict[str, Any]] = []

    for snapshot in snapshots:
        alerts = extract_alerts(snapshot.sarif)
        current = Counter(str(a["identity"]) for a in alerts)
        all_exemplars.update(_exemplars(alerts))
        row = _release_summary(
            snapshot.release,
            current,
            previous,
            ever_seen.copy(),
            all_exemplars,
        )
        release_rows.append(row)
        ever_seen |= current
        previous = current

    current_open = []
    final_counts = previous
    for ident in sorted(final_counts):
        exemplar = dict(all_exemplars[ident])
        exemplar["count"] = int(final_counts[ident])
        current_open.append(exemplar)

    recurring = []
    for row in release_rows:
        for ident in row["recurred_identities"]:
            exemplar = dict(all_exemplars[ident])
            exemplar["release"] = row["release"]
            recurring.append(exemplar)

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "releases": release_rows,
        "summary": {
            "release_count": len(release_rows),
            "current_open": int(sum(final_counts.values())),
            "resolved_total": int(sum(r["closed"] for r in release_rows)),
            "recurrence_total": int(sum(r["recurred"] for r in release_rows)),
            "max_open": int(max(r["open_total"] for r in release_rows)),
            "net_open_delta": int(
                release_rows[-1]["open_total"] - release_rows[0]["open_total"]
            ),
        },
        "current_open_alerts": current_open,
        "recurring_alerts": recurring,
    }


def render_markdown(dashboard: Mapping[str, Any]) -> str:
    """Render a compact Markdown dashboard for release notes or PR summaries."""

    summary = dashboard["summary"]
    lines = [
        "# TensorGuard Code Scanning Trend Dashboard",
        "",
        "| releases | current open | resolved events | recurrence events | net open delta |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['release_count']} | {summary['current_open']} | "
            f"{summary['resolved_total']} | {summary['recurrence_total']} | "
            f"{summary['net_open_delta']} |"
        ),
        "",
        "## Per-release deltas",
        "",
        "| release | open | opened | closed | carried | recurred | net delta | top rules |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in dashboard["releases"]:
        rules = ", ".join(
            f"{rule}:{count}" for rule, count in row["per_rule_open"].items()
        ) or "-"
        lines.append(
            f"| {row['release']} | {row['open_total']} | {row['opened']} | "
            f"{row['closed']} | {row['carried']} | {row['recurred']} | "
            f"{row['net_open_delta']} | {rules} |"
        )
    lines.extend(["", "## Currently open alerts", ""])
    current = dashboard.get("current_open_alerts") or []
    if not current:
        lines.append("No TensorGuard alerts are open in the latest SARIF snapshot.")
    else:
        lines.extend(
            [
                "| count | rule | location | message |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for alert in current:
            location = f"{alert['uri']}:{alert['line']}"
            message = str(alert["message"]).replace("|", "\\|")
            lines.append(
                f"| {alert['count']} | {alert['ruleId']} | {location} | {message} |"
            )
    lines.append("")
    return "\n".join(lines)


def load_snapshot(spec: str) -> SarifSnapshot:
    if "=" not in spec:
        raise ValueError("snapshot must be RELEASE=path/to/file.sarif")
    release, path = spec.split("=", 1)
    if not release:
        raise ValueError("snapshot release name cannot be empty")
    with open(path, encoding="utf-8") as fh:
        return SarifSnapshot(release=release, sarif=json.load(fh))


def write_dashboard(
    snapshots: Sequence[SarifSnapshot],
    *,
    json_path: Optional[str] = None,
    markdown_path: Optional[str] = None,
) -> Dict[str, Any]:
    dashboard = build_trend_dashboard(snapshots)
    if json_path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n")
    if markdown_path:
        p = Path(markdown_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_markdown(dashboard), encoding="utf-8")
    return dashboard


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshots",
        nargs="+",
        help="Ordered release snapshots as RELEASE=path/to/tensorguard.sarif",
    )
    parser.add_argument("--output", "-o", help="Write JSON dashboard here")
    parser.add_argument("--markdown", "-m", help="Write Markdown dashboard here")
    args = parser.parse_args(argv)
    snapshots = [load_snapshot(s) for s in args.snapshots]
    dashboard = write_dashboard(
        snapshots,
        json_path=args.output,
        markdown_path=args.markdown,
    )
    if not args.output:
        print(json.dumps(dashboard, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
