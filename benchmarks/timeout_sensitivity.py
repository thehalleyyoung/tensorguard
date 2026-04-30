#!/usr/bin/env python3
"""SMT time-budget sensitivity sweep.

Re-runs ``benchmarks/dl4c_bench.py``-style verification on the 10
paper-demo architectures while varying the SMT solver timeout knob
(``TensorShapeAnalyzer.timeout_ms`` and the CEGAR ``solver.set('timeout',
...)`` call sites) over::

    {50, 100, 250, 500, 1000, 5000} ms.

For each (timeout, architecture) pair we record:

* ``status``   — TensorGuard's verdict (SAFE/UNSAFE).
* ``agree``    — whether the verdict matches the ground truth label
                 (``ex["expected"]``) used by ``dl4c_bench.py``.
* ``duration`` — the wall-clock duration in ms.

We then summarise per-timeout *verification rate* (fraction of
architectures whose verdict agreed with ground truth) and median wall
time.  These numbers are written to
``benchmarks/timeout_sensitivity.json`` and plotted as
``docs/paper/figs/timeout_sensitivity.pdf``.

The sweep is implemented by *monkey-patching* every ``solver.set`` call
inside ``z3.Solver`` and friends so that the requested SMT timeout is
clamped to the per-run budget *T*; the
``TensorShapeAnalyzer(timeout_ms=...)`` constructor argument is also
overridden by passing it explicitly through ``_run_shape_analysis``.

Reproduce::

    python3.11 benchmarks/timeout_sensitivity.py
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import paper_demo by file path so we share the exact 10-example suite.
_spec = importlib.util.spec_from_file_location(
    "paper_demo", REPO_ROOT / "examples" / "paper_demo.py"
)
paper_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_demo)

TIMEOUTS_MS = [50, 100, 250, 500, 1000, 5000]


def _materialise(ex: dict) -> str:
    if "file" in ex:
        return ex["file"]
    fd, path = tempfile.mkstemp(prefix=f"tg_{ex['name']}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(ex["source"])
    return path


def _patch_z3_timeout(budget_ms: int):
    """Force every ``Solver.set('timeout', ...)`` call to clamp to budget.

    Returns an undo callable.
    """
    import z3
    original = z3.Solver.set

    def patched(self, *args, **kwargs):
        # accept either set("timeout", v) or set(timeout=v)
        if args and args[0] == "timeout" and len(args) >= 2:
            args = ("timeout", min(int(args[1]), budget_ms)) + args[2:]
        if "timeout" in kwargs:
            kwargs["timeout"] = min(int(kwargs["timeout"]), budget_ms)
        return original(self, *args, **kwargs)

    z3.Solver.set = patched
    return lambda: setattr(z3.Solver, "set", original)


def _patch_analyzer_timeout(budget_ms: int):
    """Force ``TensorShapeAnalyzer.__init__`` to use ``budget_ms``."""
    from src import tensor_shapes
    original = tensor_shapes.TensorShapeAnalyzer.__init__

    def patched(self, timeout_ms=budget_ms, *a, **kw):
        return original(self, timeout_ms=budget_ms, *a, **kw)

    tensor_shapes.TensorShapeAnalyzer.__init__ = patched
    return lambda: setattr(
        tensor_shapes.TensorShapeAnalyzer, "__init__", original
    )


def run_one(budget_ms: int) -> dict:
    undo_z3 = _patch_z3_timeout(budget_ms)
    undo_an = _patch_analyzer_timeout(budget_ms)
    try:
        # Force a fresh import of api so the patched analyzer takes effect
        # in any cached references; the api uses TensorShapeAnalyzer at call
        # time, so this is safe.
        from src.api import verify_module  # noqa: WPS433
        cases = ([(ex, "source") for ex in paper_demo.EXAMPLES] +
                 [(ex, "path")  for ex in paper_demo.EXAMPLE_FILES])
        rows = []
        for ex, kind in cases:
            path = ex["path"] if kind == "path" else _materialise(ex)
            # one warm-up
            verify_module(path, input_shapes=ex["input_shapes"])
            t0 = time.perf_counter()
            r = verify_module(path, input_shapes=ex["input_shapes"])
            wall_ms = (time.perf_counter() - t0) * 1000.0
            rows.append({
                "name": ex["name"],
                "expected": ex["expected"],
                "got": r.status,
                "agree": r.status == ex["expected"],
                "n_bugs": len(r.bugs),
                "duration_ms": round(r.duration_ms, 2),
                "wall_ms": round(wall_ms, 2),
            })
    finally:
        undo_an()
        undo_z3()
    walls = [r["wall_ms"] for r in rows]
    return {
        "timeout_ms": budget_ms,
        "n_examples": len(rows),
        "n_agree": sum(1 for r in rows if r["agree"]),
        "verification_rate": round(
            sum(1 for r in rows if r["agree"]) / len(rows), 3
        ),
        "median_wall_ms": round(statistics.median(walls), 2),
        "mean_wall_ms": round(statistics.mean(walls), 2),
        "max_wall_ms": round(max(walls), 2),
        "rows": rows,
    }


def plot(summary: dict, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweeps = summary["sweeps"]
    xs = [s["timeout_ms"] for s in sweeps]
    rates = [100 * s["verification_rate"] for s in sweeps]
    medians = [s["median_wall_ms"] for s in sweeps]

    fig, ax1 = plt.subplots(figsize=(6.4, 2.6))
    ax2 = ax1.twinx()
    line1, = ax1.plot(xs, rates, "o-", color="#2c7fb8",
                      label="verification rate (%)")
    line2, = ax2.plot(xs, medians, "s--", color="#d62728",
                      label="median wall-clock (ms)")
    ax1.set_xscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs], fontsize=8)
    ax1.set_xlabel("SMT timeout budget (ms, log scale)")
    ax1.set_ylabel("verification rate (%)", color="#2c7fb8")
    ax2.set_ylabel("median wall (ms)", color="#d62728")
    ax1.set_ylim(0, 105)
    ax1.grid(True, which="both", alpha=0.25, linestyle=":")
    ax1.set_title("Time-budget sensitivity on the 10 paper-demo models",
                  fontsize=10)
    ax1.legend(handles=[line1, line2], loc="lower right",
               fontsize=8, frameon=False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    sweeps = [run_one(t) for t in TIMEOUTS_MS]
    summary = {
        "timeouts_ms": TIMEOUTS_MS,
        "sweeps": sweeps,
    }
    out = REPO_ROOT / "benchmarks" / "timeout_sensitivity.json"
    out.write_text(json.dumps(summary, indent=2))
    plot(summary, REPO_ROOT / "docs" / "paper" / "figs"
         / "timeout_sensitivity.pdf")
    print(f"wrote {out}")
    print("timeout_ms  rate  median_wall_ms")
    for s in sweeps:
        print(f"  {s['timeout_ms']:5d}     {s['verification_rate']:.2f}  "
              f"{s['median_wall_ms']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
