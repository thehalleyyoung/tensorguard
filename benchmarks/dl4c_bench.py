#!/usr/bin/env python3
"""DL4C@ICML 2026 reproducibility benchmark for TensorGuard.

Re-runs the 10 paper examples already shipped under ``examples/paper_demo.py``
and emits a single JSON file consumed by ``docs/paper/icml_workshop.tex``.

Each row records: example name, expected status (SAFE/UNSAFE), the status
TensorGuard returned, the wall-clock duration reported by the analyser
(``AnalysisResult.duration_ms``), and the count of bugs reported.

Reproduce with::

    python3.11 benchmarks/dl4c_bench.py

Output: benchmarks/dl4c_bench_results.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.api import verify_module  # noqa: E402

# Reuse the 10 examples from examples/paper_demo.py as the benchmark suite.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "paper_demo", REPO_ROOT / "examples" / "paper_demo.py"
)
paper_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_demo)


def _materialise(ex: dict) -> str:
    """Return a path to the source file for example ``ex``.

    Examples in paper_demo.py either ship a ``"file"`` (existing on-disk
    example) or an inline ``"source"`` string we must drop into a temp file.
    """
    if "file" in ex:
        return ex["file"]
    fd, path = tempfile.mkstemp(prefix=f"tg_{ex['name']}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(ex["source"])
    return path


def run() -> dict:
    rows = []
    cases = [(ex, "source") for ex in paper_demo.EXAMPLES] + \
            [(ex, "path") for ex in paper_demo.EXAMPLE_FILES]
    for ex, kind in cases:
        path = ex["path"] if kind == "path" else _materialise(ex)
        # warm-up call so we report steady-state Z3 latency, not import cost
        verify_module(path, input_shapes=ex["input_shapes"])
        t0 = time.perf_counter()
        result = verify_module(path, input_shapes=ex["input_shapes"])
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "name": ex["name"],
            "expected": ex["expected"],
            "got": result.status,
            "agree": result.status == ex["expected"],
            "n_bugs": len(result.bugs),
            "duration_ms": round(result.duration_ms, 2),
            "wall_ms": round(wall_ms, 2),
        })
    times = [r["duration_ms"] for r in rows]
    summary = {
        "n_examples": len(rows),
        "n_agree": sum(1 for r in rows if r["agree"]),
        "n_disagree": sum(1 for r in rows if not r["agree"]),
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "max_ms": round(max(times), 2),
        "min_ms": round(min(times), 2),
        "rows": rows,
    }
    return summary


def main() -> int:
    summary = run()
    out = REPO_ROOT / "benchmarks" / "dl4c_bench_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"Agreement: {summary['n_agree']}/{summary['n_examples']}; "
          f"mean {summary['mean_ms']} ms, median {summary['median_ms']} ms, "
          f"max {summary['max_ms']} ms")
    return 0 if summary["n_agree"] == summary["n_examples"] else 1


if __name__ == "__main__":
    sys.exit(main())
