#!/usr/bin/env python3
"""Public model-zoo certification queue over the copyable TensorGuard gallery.

The queue is deliberately deterministic: it records model sources, expected
verdicts, observed verdicts, status badges, and failure explanations, but no
wall-clock fields.  Regeneration is therefore safe for monthly freshness checks
and for the global reproducibility harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples import model_gallery as gallery  # noqa: E402
from src.api import verify_architecture  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "model_zoo_certification.json"
OUT_MD = REPO / "reproducibility" / "model_zoo_certification.md"
SCHEMA = "tensorguard.model_zoo_certification.v1"
SOUNDNESS_MODE = "balanced"
WORKFLOW = ".github/workflows/model-zoo-certification.yml"
CRON_UTC = "31 8 3 * *"


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _job_id(case: gallery.GalleryCase) -> str:
    h = hashlib.sha256()
    h.update(case.slug.encode("utf-8"))
    h.update(b"\0")
    h.update(case.clean_source.encode("utf-8"))
    h.update(b"\0")
    h.update(case.buggy_source.encode("utf-8"))
    h.update(b"\0")
    h.update(SOUNDNESS_MODE.encode("utf-8"))
    return f"tgzoo-{h.hexdigest()[:16]}"


def _source_hash(case: gallery.GalleryCase) -> str:
    h = hashlib.sha256()
    h.update(case.clean_source.encode("utf-8"))
    h.update(b"\0")
    h.update(case.buggy_source.encode("utf-8"))
    return h.hexdigest()


def _result_for(source: str, input_shapes: Dict[str, tuple], filename: str) -> Dict[str, object]:
    result = verify_architecture(
        source,
        input_shapes=input_shapes,
        filename=filename,
        soundness_mode=SOUNDNESS_MODE,
    )
    categories = sorted(
        {
            getattr(getattr(bug, "category", ""), "value", str(getattr(bug, "category", "")))
            for bug in result.bugs
        }
    )
    return {
        "verdict": result.verdict,
        "bug_count": result.bug_count,
        "bug_categories": categories,
        "unknown_reasons": sorted(result.unknown_reasons),
    }


def _explain_failure(row: Dict[str, object]) -> str:
    failures: List[str] = []
    for variant in ("clean", "buggy"):
        expected = row[f"expected_{variant}_verdict"]
        observed = row[f"observed_{variant}_verdict"]
        if expected != observed:
            failures.append(
                f"{variant} variant expected {expected}, observed {observed}"
            )
    if not failures:
        return ""
    return "; ".join(failures)


def _badge(status: str) -> str:
    color = "brightgreen" if status == "certified" else "red"
    label = "model--zoo"
    return f"![{status}](https://img.shields.io/badge/{label}-{status}-{color})"


def queue_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in gallery.gallery_cases():
        clean = _result_for(case.clean_source, case.input_shapes, case.filename)
        buggy = _result_for(case.buggy_source, case.input_shapes, case.filename)
        row: Dict[str, object] = {
            "job_id": _job_id(case),
            "slug": case.slug,
            "title": case.title,
            "family": case.family,
            "source_sha256": _source_hash(case),
            "input_shapes": {k: list(v) for k, v in case.input_shapes.items()},
            "copy_config": case.copy_config,
            "expected_clean_verdict": "SAFE",
            "observed_clean_verdict": clean["verdict"],
            "clean_bug_count": clean["bug_count"],
            "clean_unknown_reasons": clean["unknown_reasons"],
            "expected_buggy_verdict": "UNSAFE",
            "observed_buggy_verdict": buggy["verdict"],
            "buggy_bug_count": buggy["bug_count"],
            "buggy_bug_categories": buggy["bug_categories"],
        }
        row["status"] = (
            "certified"
            if row["observed_clean_verdict"] == row["expected_clean_verdict"]
            and row["observed_buggy_verdict"] == row["expected_buggy_verdict"]
            else "failed"
        )
        row["failure_explanation"] = _explain_failure(row)
        row["badge_markdown"] = _badge(str(row["status"]))
        rows.append(row)
    return rows


def manifest() -> Dict[str, object]:
    rows = queue_rows()
    return {
        "schema": SCHEMA,
        "soundness_mode": SOUNDNESS_MODE,
        "queue": {
            "name": "TensorGuard public model-zoo certification queue",
            "source": "examples/model_gallery.py",
            "job_count": len(rows),
            "certified_count": sum(1 for row in rows if row["status"] == "certified"),
            "failed_count": sum(1 for row in rows if row["status"] == "failed"),
        },
        "freshness_policy": {
            "cadence": "monthly",
            "cron_utc": CRON_UTC,
            "workflow": WORKFLOW,
            "check_command": "PYTHONPATH=. python reproducibility/model_zoo_certification.py --check",
        },
        "rows": rows,
    }


def render_markdown(data: Dict[str, object]) -> str:
    queue = data["queue"]
    policy = data["freshness_policy"]
    lines = [
        "# TensorGuard model-zoo certification queue",
        "",
        "> Generated by `reproducibility/model_zoo_certification.py`. The queue",
        "> records deterministic verdicts only; no timestamps or wall-clock fields",
        "> are committed.",
        "",
        f"**Source.** `{queue['source']}` feeds {queue['job_count']} reproducible",
        f"certification jobs in `{data['soundness_mode']}` mode.",
        "",
        f"**Freshness.** `{policy['workflow']}` runs `{policy['cron_utc']}` UTC",
        f"and asserts `{policy['check_command']}`.",
        "",
        "| Job | Model | Family | Badge | Clean | Buggy | Failure explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in data["rows"]:
        failure = row["failure_explanation"] or "none"
        lines.append(
            f"| `{row['job_id']}` | `{row['slug']}` | {row['family']} | "
            f"{row['badge_markdown']} | "
            f"{row['observed_clean_verdict']} expected `{row['expected_clean_verdict']}` | "
            f"{row['observed_buggy_verdict']} expected `{row['expected_buggy_verdict']}` | "
            f"{failure} |"
        )
    lines.extend(
        [
            "",
            "## Queue contract",
            "",
            "A job is **certified** only when the clean model verifies `SAFE` and",
            "the paired buggy model verifies `UNSAFE` under the recorded soundness",
            "mode. Any drift becomes a `failed` row with a deterministic failure",
            "explanation and causes the monthly freshness workflow to fail.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts() -> Dict[str, object]:
    data = manifest()
    OUT_JSON.write_text(_dumps(data), encoding="utf-8")
    OUT_MD.write_text(render_markdown(data), encoding="utf-8")
    return data


def run(check: bool = False) -> int:
    data = manifest()
    new_json = _dumps(data)
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text(encoding="utf-8") != new_json:
            print("MISMATCH: model_zoo_certification.json differs", file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text(encoding="utf-8") != new_md:
            print("MISMATCH: model_zoo_certification.md differs", file=sys.stderr)
            ok = False
        print("model_zoo_certification --check:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    OUT_JSON.write_text(new_json, encoding="utf-8")
    OUT_MD.write_text(new_md, encoding="utf-8")
    queue = data["queue"]
    print(
        "Wrote reproducibility/model_zoo_certification.{json,md}: "
        f"{queue['certified_count']} certified, {queue['failed_count']} failed."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="assert committed artifacts are fresh")
    args = parser.parse_args()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
