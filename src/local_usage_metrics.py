from __future__ import annotations

"""Telemetry-free local usage summaries for TensorGuard JSON outputs.

The module intentionally never reads model source and never sends data anywhere:
it aggregates already-produced TensorGuard JSON reports into a compact summary
that teams can inspect locally or choose to publish themselves.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class UsageMetricsSummary:
    reports: int
    analyzed_files: int
    verdicts: Mapping[str, int]
    abstentions: int
    top_unsupported_ops: Sequence[tuple[str, int]]
    bug_categories: Mapping[str, int]
    redacted_files: Sequence[str] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "tensorguard.local_usage_metrics.v1",
            "privacy": {
                "telemetry_free": True,
                "network": "none",
                "source_code_included": False,
                "file_paths": "sha256:16",
            },
            "reports": self.reports,
            "analyzed_files": self.analyzed_files,
            "verdicts": dict(sorted(self.verdicts.items())),
            "abstentions": self.abstentions,
            "top_unsupported_ops": [
                {"op": op, "count": count} for op, count in self.top_unsupported_ops
            ],
            "bug_categories": dict(sorted(self.bug_categories.items())),
            "redacted_files": list(self.redacted_files),
        }

    def to_markdown(self) -> str:
        verdicts = ", ".join(
            f"{name}: {count}" for name, count in sorted(self.verdicts.items())
        ) or "none"
        lines = [
            "# TensorGuard Local Usage Metrics",
            "",
            "Telemetry: **off**. Source code: **not included**. File paths: **hashed**.",
            "",
            f"- Reports summarized: {self.reports}",
            f"- Files analyzed: {self.analyzed_files}",
            f"- Verdicts: {verdicts}",
            f"- Abstentions: {self.abstentions}",
        ]
        if self.top_unsupported_ops:
            lines.extend(["", "## Top unsupported ops", ""])
            lines.extend(f"- `{op}`: {count}" for op, count in self.top_unsupported_ops)
        if self.bug_categories:
            lines.extend(["", "## Bug categories", ""])
            lines.extend(
                f"- `{cat}`: {count}" for cat, count in sorted(self.bug_categories.items())
            )
        return "\n".join(lines) + "\n"


def summarize_files(paths: Iterable[str | Path], *, limit: int = 10) -> UsageMetricsSummary:
    records: list[Mapping[str, Any]] = []
    for path in paths:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        records.extend(_iter_report_records(loaded))
    return summarize_records(records, limit=limit)


def summarize_records(
    records: Iterable[Mapping[str, Any]], *, limit: int = 10
) -> UsageMetricsSummary:
    reports = 0
    files: set[str] = set()
    verdicts: Counter[str] = Counter()
    bug_categories: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    abstentions = 0

    for record in records:
        reports += 1
        file_id = _redact_path(str(record.get("file", f"report:{reports}")))
        files.add(file_id)
        verdict = str(record.get("verdict") or record.get("status") or "UNKNOWN").upper()
        verdicts[verdict] += 1
        if bool(record.get("abstained")) or verdict == "UNKNOWN":
            abstentions += 1

        for bug in _as_list(record.get("bugs")):
            category = _bug_category(bug)
            if category:
                bug_categories[category] += 1

        for op, count in _unsupported_ops(record).items():
            unsupported[op] += count

    return UsageMetricsSummary(
        reports=reports,
        analyzed_files=len(files),
        verdicts=dict(verdicts),
        abstentions=abstentions,
        top_unsupported_ops=tuple(unsupported.most_common(max(0, limit))),
        bug_categories=dict(bug_categories),
        redacted_files=tuple(sorted(files)),
    )


def _iter_report_records(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, dict):
        if _looks_like_report(obj):
            yield obj
        for key in ("results", "files", "reports"):
            value = obj.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield from _iter_report_records(item)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield from _iter_report_records(item)


def _looks_like_report(obj: Mapping[str, Any]) -> bool:
    return any(k in obj for k in ("verdict", "status", "bugs", "unknown_reasons"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bug_category(bug: Any) -> str | None:
    if not isinstance(bug, Mapping):
        return None
    category = bug.get("category") or bug.get("ruleId")
    if category:
        return str(category)
    message = str(bug.get("message", ""))
    if message.startswith("[") and "]" in message:
        return message[1 : message.index("]")].lower()
    return None


def _unsupported_ops(record: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    raw = record.get("unsupported_ops") or record.get("top_unsupported_ops")
    if isinstance(raw, Mapping):
        for op, count in raw.items():
            counts[str(op)] += int(count)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                counts[item] += 1
            elif isinstance(item, Mapping):
                op = item.get("op") or item.get("name")
                if op:
                    counts[str(op)] += int(item.get("count", 1))
    for reason in _as_list(record.get("unknown_reasons")):
        text = str(reason)
        if "unsupported" in text.lower() or "heuristic-tagged operator" in text:
            for op in text.split(":", 1)[-1].split(","):
                op = op.strip(" `.")
                if op:
                    counts[op] += 1
    return counts


def _redact_path(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
