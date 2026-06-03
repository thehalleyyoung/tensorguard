#!/usr/bin/env python3
"""Registered controlled-developer-study task packet (Step 259).

This generator completes the study material around the existing localization
proxy without pretending that a human-subjects RCT has already been run.  It is
torch-free and deterministic: it consumes committed real-bug localization
artifacts, validates that each task maps to a real repro with a ``# BUG``
marker and an upstream reference, and emits:

* a study/scoring artifact covering localization-time proxy, fix-quality rubric,
  and trust-calibration scoring keys;
* a participant-facing task packet under ``docs/user_study/``.

``--check`` regenerates the artifacts in memory and byte-diffs the committed
copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOCALIZATION_EFFORT = REPO / "evaluation" / "localization_effort.json"
MARKER_AUDIT = REPO / "reproducibility" / "localization_marker_only_n30.json"
PROTOCOL = REPO / "docs" / "user_study" / "protocol.md"
TASK_PACKET_JSON = REPO / "docs" / "user_study" / "task_packet.json"
TASK_PACKET_MD = REPO / "docs" / "user_study" / "task_packet.md"
OUT_JSON = REPO / "reproducibility" / "developer_study.json"
OUT_MD = REPO / "reproducibility" / "developer_study.md"

CORPORA = [
    REPO / "experiments_v5" / "v8" / "real_bugs_upstream",
    REPO / "experiments_v5" / "v8" / "real_bugs_postfreeze",
    REPO / "experiments_v5" / "v8" / "real_bugs_unfiltered",
]

SECONDS_PER_LINE = 6.0
PLANNED_PARTICIPANTS = 24
FIX_QUALITY_MAX_SCORE = 3
TRUST_MISLEADING_IDS = {
    "rb_pf_001_diffusers_longcat_ffmult",
    "rb_pf_003_peft_lora_moe_swap",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(x: float, ndigits: int = 6) -> float:
    return round(float(x), ndigits)


def _find_repro(stem: str) -> Path:
    for cdir in CORPORA:
        candidate = cdir / f"{stem}.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing real-bug repro for {stem}")


def _extract_urls(text: str) -> List[str]:
    urls = []
    for raw in re.findall(r"https?://[^\s)`]+", text):
        url = raw.rstrip(".,;:")
        if url not in urls:
            urls.append(url)
    return urls


def _prefer_fix_reference(urls: Iterable[str]) -> str:
    ordered = list(urls)
    for url in ordered:
        if "/pull/" in url or "/commit/" in url:
            return url
    if not ordered:
        raise ValueError("expected at least one upstream reference")
    return ordered[0]


def _protocol_rqs() -> Dict[str, bool]:
    text = PROTOCOL.read_text(encoding="utf-8")
    return {rq: (rq in text) for rq in ("RQ1", "RQ2", "RQ3")}


def _source_line(lines: List[str], line_no: int) -> str:
    if line_no < 1 or line_no > len(lines):
        raise ValueError(f"line {line_no} outside source with {len(lines)} lines")
    return lines[line_no - 1]


def _build_task(row: Dict[str, Any], marker: Dict[str, Any]) -> Dict[str, Any]:
    repro = _find_repro(row["id"])
    source = repro.read_text(encoding="utf-8")
    lines = source.splitlines()
    marker_line = int(marker["marker_line"])
    marker_text = _source_line(lines, marker_line)
    if "# BUG" not in marker_text:
        raise ValueError(f"{repro}: marker_line {marker_line} lacks # BUG marker")
    urls = _extract_urls(source)
    if not urls:
        raise ValueError(f"{repro}: missing upstream issue/PR reference")

    assisted_lines = float(row["assisted_effort_lines"])
    unaided_lines = float(row["unaided_effort_lines"])
    tg_helped = bool(row["tg_helped"])
    expected_trust = (
        "trust_after_local_confirmation"
        if tg_helped
        else "verify_and_do_not_follow_blindly"
    )

    return {
        "id": row["id"],
        "corpus": row["corpus"],
        "repro_path": str(repro.relative_to(REPO)),
        "source_sha256": _sha256(repro),
        "gt_line": int(row["gt_line"]),
        "marker_line": marker_line,
        "marker_text": marker_text.strip(),
        "tg_reported_line": int(row["tg_line_v5"]),
        "tg_distance_lines": int(row["dist_v5"]),
        "upstream_references": urls,
        "preferred_fix_reference": _prefer_fix_reference(urls),
        "localization_time_proxy": {
            "assisted_lines": _round(assisted_lines),
            "unaided_lines": _round(unaided_lines),
            "assisted_seconds": _round(assisted_lines * SECONDS_PER_LINE),
            "unaided_seconds": _round(unaided_lines * SECONDS_PER_LINE),
            "seconds_per_line": _round(SECONDS_PER_LINE),
            "note": (
                "Seconds are a preregistered linear rescaling of the "
                "lines-inspected proxy, not observed human timing."
            ),
        },
        "fix_quality_rubric": {
            "max_score": FIX_QUALITY_MAX_SCORE,
            "criteria": [
                "the submitted patch addresses the marked root-cause line",
                "the module executes for its declared INPUT_SHAPES",
                "the patch preserves the module's public forward signature",
            ],
            "gold_evidence": (
                "source # BUG marker plus upstream issue/PR reference; runtime "
                "execution belongs to the human-study environment, not this "
                "deterministic CI artifact"
            ),
        },
        "trust_calibration": {
            "tg_advice_expected_helpful": tg_helped,
            "expected_response": expected_trust,
            "scoring_key": (
                "calibrated_trust"
                if tg_helped
                else "calibrated_skepticism"
            ),
        },
    }


def collect() -> Dict[str, Any]:
    effort = _load_json(LOCALIZATION_EFFORT)
    marker_audit = _load_json(MARKER_AUDIT)
    markers = {row["id"]: row for row in marker_audit["per_item"]}

    tasks = []
    for row in effort["per_bug"]:
        marker = markers.get(row["id"])
        if marker is None:
            raise KeyError(f"missing marker audit row for {row['id']}")
        if not marker.get("refuted") or marker.get("dist_v5") is None:
            raise ValueError(f"marker row for {row['id']} is not eligible")
        tasks.append(_build_task(row, marker))
    tasks.sort(key=lambda item: item["id"])
    return {"tasks": tasks, "localization_summary": effort["summary"]}


def measure() -> Dict[str, Any]:
    data = collect()
    tasks = data["tasks"]
    assisted_seconds = [
        task["localization_time_proxy"]["assisted_seconds"] for task in tasks
    ]
    unaided_seconds = [
        task["localization_time_proxy"]["unaided_seconds"] for task in tasks
    ]
    helped = [task for task in tasks
              if task["trust_calibration"]["tg_advice_expected_helpful"]]
    misleading = [task for task in tasks
                  if not task["trust_calibration"]["tg_advice_expected_helpful"]]
    protocol_rqs = _protocol_rqs()

    artifact = {
        "meta": {
            "generated_by": "reproducibility/developer_study.py",
            "command": "python3 reproducibility/developer_study.py",
            "status": (
                "registered controlled-study task packet plus deterministic "
                "proxy/scoring artifact; no human-subjects outcomes executed"
            ),
            "no_human_subjects_results": True,
            "proxy_disclaimer": (
                "Localization time is the existing lines-inspected proxy "
                "expressed as a fixed seconds-per-line calibration. Fix quality "
                "and trust calibration are preregistered scoring instruments and "
                "task keys, not measured participant outcomes."
            ),
            "inputs": [
                str(LOCALIZATION_EFFORT.relative_to(REPO)),
                str(MARKER_AUDIT.relative_to(REPO)),
                str(PROTOCOL.relative_to(REPO)),
            ],
            "seconds_per_line": _round(SECONDS_PER_LINE),
            "planned_participants": PLANNED_PARTICIPANTS,
            "fix_quality_max_score": FIX_QUALITY_MAX_SCORE,
        },
        "registration": {
            "protocol_document": str(PROTOCOL.relative_to(REPO)),
            "protocol_sha256": _sha256(PROTOCOL),
            "contains_research_questions": protocol_rqs,
            "registered_constructs": [
                "localization_time_proxy",
                "fix_quality",
                "trust_calibration",
            ],
            "participant_task_packet_json": str(TASK_PACKET_JSON.relative_to(REPO)),
            "participant_task_packet_md": str(TASK_PACKET_MD.relative_to(REPO)),
        },
        "summary": {
            "n_tasks": len(tasks),
            "n_tasks_with_real_repro": len(tasks),
            "n_tasks_with_bug_marker": len(tasks),
            "n_tasks_with_upstream_reference": len(tasks),
            "n_expected_helpful_tg_advice": len(helped),
            "n_expected_misleading_tg_advice": len(misleading),
            "misleading_task_ids": [task["id"] for task in misleading],
            "median_assisted_seconds_proxy": _round(
                statistics.median(assisted_seconds)
            ),
            "median_unaided_seconds_proxy": _round(
                statistics.median(unaided_seconds)
            ),
            "median_assisted_lines_proxy": data["localization_summary"][
                "median_assisted_lines"
            ],
            "median_unaided_lines_proxy": data["localization_summary"][
                "median_unaided_lines"
            ],
            "localization_effect_is_linear_rescale": True,
        },
        "instruments": {
            "localization_time": {
                "control_material": "stock runtime traceback plus repro file",
                "treatment_material": (
                    "TensorGuard verdict, counterexample, and reported source "
                    "line plus the same repro file"
                ),
                "primary_measure": (
                    "observed time-to-first-correct-root-cause in the human "
                    "study; deterministic artifact reports only the proxy"
                ),
            },
            "fix_quality": {
                "score_range": [0, FIX_QUALITY_MAX_SCORE],
                "passing_score": FIX_QUALITY_MAX_SCORE,
                "criteria": [
                    "root cause removed",
                    "declared INPUT_SHAPES execute",
                    "public forward signature preserved",
                ],
            },
            "trust_calibration": {
                "items_include_helpful_and_misleading_tg_advice": bool(
                    helped and misleading
                ),
                "expected_misleading_ids": sorted(TRUST_MISLEADING_IDS),
                "question": (
                    "After seeing the diagnostic, should the participant trust "
                    "the suggested location directly, verify it first, or reject it?"
                ),
            },
        },
        "tasks": tasks,
    }
    return artifact


def render_markdown(artifact: Dict[str, Any]) -> str:
    meta = artifact["meta"]
    summary = artifact["summary"]
    lines = [
        "# Controlled developer-study packet (Step 259)",
        "",
        "_Generated by `reproducibility/developer_study.py`; do not edit by hand._",
        "",
        "> **No human outcomes are claimed.** " + meta["proxy_disclaimer"],
        "",
        "## What is registered",
        "",
        f"- Protocol: `{artifact['registration']['protocol_document']}`",
        "- Constructs: localization time, fix quality, trust calibration",
        f"- Planned human RCT size: {meta['planned_participants']} participants",
        f"- Task battery: {summary['n_tasks']} real refuted bugs with repros, "
        "markers, and upstream references",
        f"- Trust calibration includes {summary['n_expected_misleading_tg_advice']} "
        "known misleading localizations so the study can measure over-trust",
        "",
        "## Deterministic proxy/readiness checks",
        "",
        f"- Median localization proxy: {summary['median_assisted_lines_proxy']} "
        f"lines assisted vs {summary['median_unaided_lines_proxy']} unaided",
        f"- Same proxy as time units: {summary['median_assisted_seconds_proxy']} "
        f"seconds assisted vs {summary['median_unaided_seconds_proxy']} unaided "
        f"at {meta['seconds_per_line']} seconds/line",
        f"- All {summary['n_tasks_with_bug_marker']} tasks have a committed `# BUG` "
        "marker at the registered line",
        f"- All {summary['n_tasks_with_upstream_reference']} tasks carry upstream "
        "issue/PR evidence for the gold fix-quality rubric",
        "",
        "## Task list",
        "",
        "| Task | Repro | TG line | marker line | expected TG advice | reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in artifact["tasks"]:
        expected = (
            "helpful"
            if task["trust_calibration"]["tg_advice_expected_helpful"]
            else "misleading"
        )
        lines.append(
            f"| {task['id']} | `{task['repro_path']}` | "
            f"{task['tg_reported_line']} | {task['marker_line']} | "
            f"{expected} | {task['preferred_fix_reference']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_task_packet_json(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "TensorGuard controlled developer-study task packet",
        "status": (
            "participant-facing materials; scoring keys live in "
            "reproducibility/developer_study.json"
        ),
        "protocol": artifact["registration"]["protocol_document"],
        "conditions": {
            "control": "runtime traceback + repro source",
            "treatment": "TensorGuard diagnostic + same repro source",
        },
        "outcomes_collected": [
            "time_to_localize_seconds",
            "fix_quality_score_0_to_3",
            "trust_calibration_choice",
        ],
        "tasks": [
            {
                "id": task["id"],
                "repro_path": task["repro_path"],
                "upstream_references": task["upstream_references"],
                "participant_prompt": (
                    "Find the root cause, patch the module, and record whether "
                    "you trusted, verified, or rejected the diagnostic."
                ),
            }
            for task in artifact["tasks"]
        ],
    }


def render_task_packet_markdown(packet: Dict[str, Any]) -> str:
    lines = [
        "# TensorGuard controlled developer-study task packet",
        "",
        "_Generated by `reproducibility/developer_study.py`; do not edit by hand._",
        "",
        "This is the participant-facing packet. It intentionally omits scoring keys; "
        "the registered scoring artifact is `reproducibility/developer_study.json`.",
        "",
        "## Conditions",
        "",
        f"- Control: {packet['conditions']['control']}",
        f"- Treatment: {packet['conditions']['treatment']}",
        "",
        "## Outcomes collected",
        "",
    ]
    lines.extend(f"- `{name}`" for name in packet["outcomes_collected"])
    lines.extend([
        "",
        "## Tasks",
        "",
        "| Task | Repro | Prompt |",
        "| --- | --- | --- |",
    ])
    for task in packet["tasks"]:
        lines.append(
            f"| {task['id']} | `{task['repro_path']}` | "
            f"{task['participant_prompt']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_or_check(path: Path, text: str, check: bool) -> None:
    if check:
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(REPO)}; regenerate first")
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path.relative_to(REPO)} is stale; regenerate it")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def run(check: bool = False) -> Dict[str, Any]:
    artifact = measure()
    packet = render_task_packet_json(artifact)

    artifact_json = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    artifact_md = render_markdown(artifact) + "\n"
    packet_json = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    packet_md = render_task_packet_markdown(packet) + "\n"

    for path, text in [
        (OUT_JSON, artifact_json),
        (OUT_MD, artifact_md),
        (TASK_PACKET_JSON, packet_json),
        (TASK_PACKET_MD, packet_md),
    ]:
        _write_or_check(path, text, check)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if committed artifacts are stale")
    args = parser.parse_args()
    artifact = run(check=args.check)
    summary = artifact["summary"]
    if args.check:
        print("developer_study artifacts OK (byte-identical)")
    else:
        print("Wrote", OUT_JSON.relative_to(REPO))
        print("Wrote", OUT_MD.relative_to(REPO))
        print("Wrote", TASK_PACKET_JSON.relative_to(REPO))
        print("Wrote", TASK_PACKET_MD.relative_to(REPO))
    print(
        f"  tasks={summary['n_tasks']} "
        f"helpful={summary['n_expected_helpful_tg_advice']} "
        f"misleading={summary['n_expected_misleading_tg_advice']}"
    )


if __name__ == "__main__":
    main()
