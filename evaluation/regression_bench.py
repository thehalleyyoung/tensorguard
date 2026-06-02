"""Step 54 -- Criterion-style latency regression benchmark.

A latency *budget* (Step 46) catches absolute blow-ups; this harness catches
*relative* regressions -- a change that makes verification 10 percent slower
than the committed baseline -- the way Rust's Criterion or Google Benchmark do.

The hard problem is machine independence: raw wall-clock times differ across
CPUs, so a committed baseline of seconds is meaningless on another box. We solve
this exactly as a relative benchmark should: every measurement is **normalized
by a calibration workload measured in the same run** (the median time to verify
a fixed anchor model). If the whole machine is slower, the calibration is slower
too and the normalized ratio is unchanged; only an *algorithmic* slowdown in a
specific case moves its ratio. The committed baseline therefore stores portable
ratios, and `--gate` fails CI when any case's freshly-measured ratio exceeds its
baseline by more than the tolerance (default 10 percent).

Modes:
  * (default) ``--update`` -- measure and (re)write the committed baseline.
  * ``--check``            -- verify the committed baseline file is well-formed
                             and self-consistent (no live timing).
  * ``--gate``             -- re-measure and fail on any >tolerance regression.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "regression_bench.json")
MD_PATH = os.path.join(HERE, "regression_bench.md")

SAMPLES = 7
TOLERANCE = 0.10  # 10 percent regression budget


def _stack(n_layers: int, dim: int = 64) -> str:
    init = "\n".join(
        "        self.l%da = nn.Linear(%d, %d)\n"
        "        self.l%db = nn.Linear(%d, %d)" % (i, dim, dim, i, dim, dim)
        for i in range(n_layers)
    )
    body = "\n".join(
        "        x = self.l%db(nn.functional.relu(self.l%da(x)))" % (i, i)
        for i in range(n_layers)
    )
    return (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "%s\n"
        "    def forward(self, x):\n"
        "%s\n"
        "        return x\n" % (init, body)
    )


# The calibration anchor: a fixed model whose verification time normalizes every
# other measurement. Chosen small and stable so it tracks raw machine speed.
_ANCHOR = _stack(3)
_ANCHOR_SHAPES = {"x": ("b", 64)}


# (name, source, input_shapes)
def cases() -> List[Tuple[str, str, Dict[str, tuple]]]:
    return [
        ("stack_6", _stack(6), {"x": ("b", 64)}),
        ("stack_12", _stack(12), {"x": ("b", 64)}),
        ("stack_24", _stack(24), {"x": ("b", 64)}),
    ]


def _median_verify_time(source: str, shapes: Dict[str, tuple],
                        samples: int = SAMPLES) -> float:
    from src.model_checker import verify_model

    times: List[float] = []
    # One warm-up run (JIT of Z3 internals, import caches) excluded from stats.
    verify_model(source, input_shapes=shapes)
    for _ in range(samples):
        t0 = time.perf_counter()
        verify_model(source, input_shapes=shapes)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def measure_ratios() -> Dict[str, float]:
    """Return each case's calibration-normalized cost (machine-independent)."""
    calib = _median_verify_time(_ANCHOR, _ANCHOR_SHAPES)
    if calib <= 0:
        calib = 1e-9
    ratios: Dict[str, float] = {}
    for name, src, shapes in cases():
        ratios[name] = _median_verify_time(src, shapes) / calib
    return ratios


def _extract_steps(source: str) -> int:
    from src.model_checker import extract_computation_graph

    return len(extract_computation_graph(source).steps)


def baseline_document(ratios: Dict[str, float]) -> Dict[str, object]:
    return {
        "meta": {
            "generated_by": "evaluation/regression_bench.py",
            "command": "PYTHONPATH=. python3 evaluation/regression_bench.py",
            "python_version": "%d.%d" % sys.version_info[:2],
            "samples": SAMPLES,
            "tolerance": TOLERANCE,
            "note": ("Ratios are each case's median verification time divided "
                     "by the anchor model's median time, measured in the same "
                     "run; this cancels absolute machine speed so the gate "
                     "detects relative (algorithmic) regressions only."),
        },
        "anchor": {"name": "stack_3", "steps": _extract_steps(_ANCHOR)},
        "cases": {
            name: {"steps": _extract_steps(src),
                   "baseline_ratio": round(ratios[name], 4)}
            for name, src, _shapes in cases()
        },
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(doc: Dict[str, object]) -> str:
    lines = [
        "# Latency regression benchmark (Criterion-style)",
        "",
        ("Calibration-normalized verification cost for each benchmark case. "
         "Each ratio is the case's median verification time divided by the "
         "anchor model's median time in the same run, so the value is "
         "independent of absolute machine speed. `make regression-bench-gate` "
         "fails CI when any case regresses by more than the committed "
         "tolerance."),
        "",
        "| Case | Steps | Baseline ratio |",
        "|------|-------|----------------|",
    ]
    cdoc = doc["cases"]
    for name in sorted(cdoc):
        c = cdoc[name]
        lines.append("| `%s` | %d | %.4f |" % (
            name, c["steps"], c["baseline_ratio"]))
    lines.append("")
    lines.append("Tolerance: %d percent regression budget." % int(
        round(doc["meta"]["tolerance"] * 100)))
    lines.append("")
    return "\n".join(lines)


def _load_baseline() -> Dict[str, object]:
    with open(JSON_PATH) as fh:
        return json.load(fh)


def evaluate(doc: Dict[str, object], fresh: Dict[str, float],
             tol: float) -> List[str]:
    """Pure comparison: return a regression message for every case whose fresh
    ratio exceeds its baseline by more than *tol*. No timing here, so it is
    deterministic and unit-testable."""
    regressions: List[str] = []
    for name in sorted(doc["cases"]):
        base = float(doc["cases"][name]["baseline_ratio"])
        cur = fresh.get(name, float("inf"))
        limit = base * (1.0 + tol)
        if cur > limit:
            regressions.append(
                "%s: ratio %.4f exceeds limit %.4f (baseline %.4f, "
                "tolerance %d percent)" % (
                    name, cur, limit, base, int(round(tol * 100))))
    return regressions


def gate() -> int:
    if not os.path.exists(JSON_PATH):
        print("regression_bench.json missing; run "
              "`make regression-bench` first")
        return 1
    doc = _load_baseline()
    tol = float(doc["meta"]["tolerance"])
    fresh = measure_ratios()
    for name in sorted(doc["cases"]):
        base = float(doc["cases"][name]["baseline_ratio"])
        cur = fresh.get(name, float("inf"))
        limit = base * (1.0 + tol)
        flag = "ok" if cur <= limit else "REGRESSION"
        print("  [%s] %-10s ratio %.4f (baseline %.4f, limit %.4f)" % (
            flag, name, cur, base, limit))
    regressions = evaluate(doc, fresh, tol)
    if regressions:
        print("REGRESSION BENCHMARK GATE FAILED:")
        for r in regressions:
            print("  - %s" % r)
        return 1
    print("regression benchmark gate PASS: no case over %d percent" % int(
        round(tol * 100)))
    return 0


def check() -> int:
    """Validate the committed baseline is well-formed (no live timing)."""
    if not os.path.exists(JSON_PATH):
        print("regression_bench.json missing; run `make regression-bench`")
        return 1
    doc = _load_baseline()
    expected_cases = {name for name, _s, _sh in cases()}
    got_cases = set(doc.get("cases", {}))
    if got_cases != expected_cases:
        print("baseline cases %s != harness cases %s" % (
            sorted(got_cases), sorted(expected_cases)))
        return 1
    # Step counts in the baseline must match the current source (the source is
    # deterministic, so this is a byte-stable, machine-independent invariant).
    for name, src, _shapes in cases():
        want = _extract_steps(src)
        got = int(doc["cases"][name]["steps"])
        if got != want:
            print("case %s step count drifted: baseline %d != current %d" % (
                name, got, want))
            return 1
        if float(doc["cases"][name]["baseline_ratio"]) <= 0:
            print("case %s has non-positive baseline ratio" % name)
            return 1
    # Markdown must be in sync with the JSON.
    md = render_markdown(doc)
    if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
        print("regression_bench.md is stale; run `make regression-bench`")
        return 1
    print("regression benchmark baseline well-formed (%d cases)" % len(
        expected_cases))
    return 0


def update() -> int:
    ratios = measure_ratios()
    doc = baseline_document(ratios)
    with open(JSON_PATH, "w") as fh:
        fh.write(_dumps(doc))
    with open(MD_PATH, "w") as fh:
        fh.write(render_markdown(doc))
    print("regression benchmark baseline written: %d cases" % len(doc["cases"]))
    for name in sorted(ratios):
        print("  %-10s ratio %.4f" % (name, ratios[name]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Validate the committed baseline (no live timing).")
    ap.add_argument("--gate", action="store_true",
                    help="Re-measure and fail on any >tolerance regression.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    if args.check:
        return check()
    return update()


if __name__ == "__main__":
    raise SystemExit(main())
