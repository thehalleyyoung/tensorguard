"""
Real-source TensorGuard benchmark.

Runs the introspecting analyzer (verify_architecture) on each .py file
in benchmarks/realcode_corpus/ and classifies the verdict as one of:
  - real-bug-found      : analyzer flagged a bug AND the file is annotated as buggy
  - false-positive      : analyzer flagged a bug on a file NOT annotated as buggy
  - verified-safe       : analyzer reported no bugs on a non-buggy file
  - missed-bug          : analyzer reported no bugs on a buggy file (FN)
  - abstain-on-unknown  : analyzer ran but inferred no shapes (no calls had known transfers)

We extract the input shape from the leading `# input_shape: (...)` comment
and treat any file whose top has `# bug:` as ground-truth buggy.

Outputs benchmarks/realcode_results.json.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # type: ignore


CORPUS = ROOT / "benchmarks" / "realcode_corpus"
OUT = ROOT / "benchmarks" / "realcode_results.json"

INPUT_RE = re.compile(r"#\s*input_shape\s*:\s*\(([^)]*)\)", re.IGNORECASE)
BUG_RE = re.compile(r"#\s*bug\s*:", re.IGNORECASE)


def parse_input_shape(src: str):
    m = INPUT_RE.search(src)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            out.append(p)  # symbolic
    return tuple(out)


def is_buggy(src: str) -> bool:
    return bool(BUG_RE.search(src))


def classify(file: Path) -> dict:
    src = file.read_text()
    in_shape = parse_input_shape(src)
    buggy = is_buggy(src)
    record: dict = {
        "file": file.name,
        "input_shape": in_shape,
        "ground_truth_buggy": buggy,
    }

    if in_shape is None:
        record["verdict"] = "no-input-shape-annotation"
        return record

    t0 = time.perf_counter()
    try:
        result = verify_architecture(src, input_shapes={"x": in_shape})
    except Exception as e:
        record["verdict"] = "analyzer-error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["duration_ms"] = (time.perf_counter() - t0) * 1000
        return record
    record["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    bugs = list(getattr(result, "bugs", []) or [])
    inferred_any = bool(getattr(result, "lines_analyzed", 0))

    bug_msgs = [
        {"line": getattr(b.location, "line", None), "msg": b.message[:200]}
        for b in bugs
    ]
    record["analyzer_bugs"] = bug_msgs
    record["functions_analyzed"] = getattr(result, "functions_analyzed", 0)

    if bugs and buggy:
        record["verdict"] = "real-bug-found"
    elif bugs and not buggy:
        record["verdict"] = "false-positive"
    elif not bugs and buggy:
        record["verdict"] = "missed-bug"
    elif not bugs and inferred_any and result.functions_analyzed > 0:
        record["verdict"] = "verified-safe"
    else:
        record["verdict"] = "abstain-on-unknown"

    return record


def main():
    files = sorted(CORPUS.glob("*.py"))
    print(f"[realcode] running on {len(files)} files")
    records = [classify(f) for f in files]

    counts: dict = {}
    for r in records:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    summary = {
        "n_files": len(files),
        "verdict_counts": counts,
        "n_buggy_groundtruth": sum(1 for r in records if r["ground_truth_buggy"]),
    }

    out = {"summary": summary, "files": records}
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"[realcode] wrote {OUT}")


if __name__ == "__main__":
    main()
