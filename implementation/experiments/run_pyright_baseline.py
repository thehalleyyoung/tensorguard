#!/usr/bin/env python3
"""
Real pyright + mypy baseline comparison for Suite D benchmarks.

Unlike prior scripts that *simulated* tool behaviour, this script
**actually executes** pyright and mypy on every benchmark, parses their
real output, and reports precision / recall / F1.

This addresses the reviewer concern (Sinha): "Reimplemented PyTea baseline
creates straw-man comparison risk — 5.4× improvement may reflect
implementation fidelity."  By comparing against two widely-used,
unmodified, third-party type checkers we eliminate reimplementation bias.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from experiments.external_pytorch_benchmark import EXTERNAL_PYTORCH_BENCHMARKS

OUTPUT_FILE = SCRIPT_DIR / "pyright_baseline_results.json"

TIMEOUT_SECS = 60


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------

def run_pyright(source: str) -> dict:
    """Run pyright on *source* and return parsed results."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(source)
        f.flush()
        tmp = f.name

    try:
        r = subprocess.run(
            ["pyright", "--outputjson", tmp],
            capture_output=True, text=True, timeout=TIMEOUT_SECS,
        )
        data = json.loads(r.stdout) if r.stdout.strip() else {}
        diags = data.get("generalDiagnostics", [])
        errors = [d for d in diags if d.get("severity") == "error"]
        warnings = [d for d in diags if d.get("severity") in ("warning", "information")]
        return {
            "errors": len(errors),
            "warnings": len(warnings),
            "error_messages": [d.get("message", "") for d in errors],
            "warning_messages": [d.get("message", "") for d in warnings],
            "verdict": "buggy" if errors else "safe",
        }
    except subprocess.TimeoutExpired:
        return {"errors": 0, "warnings": 0, "verdict": "safe", "timeout": True}
    except Exception as e:
        return {"errors": 0, "warnings": 0, "verdict": "safe", "exception": str(e)}
    finally:
        os.unlink(tmp)


def run_mypy(source: str) -> dict:
    """Run mypy on *source* and return parsed results."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, dir="/tmp"
    ) as f:
        f.write(source)
        f.flush()
        tmp = f.name

    try:
        r = subprocess.run(
            ["mypy", "--ignore-missing-imports", "--no-incremental",
             "--no-error-summary", tmp],
            capture_output=True, text=True, timeout=TIMEOUT_SECS,
        )
        lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        errors = [l for l in lines if ": error:" in l]
        warnings = [l for l in lines if ": warning:" in l or ": note:" in l]
        return {
            "errors": len(errors),
            "warnings": len(warnings),
            "error_messages": errors,
            "warning_messages": warnings,
            "verdict": "buggy" if errors else "safe",
        }
    except subprocess.TimeoutExpired:
        return {"errors": 0, "warnings": 0, "verdict": "safe", "timeout": True}
    except Exception as e:
        return {"errors": 0, "warnings": 0, "verdict": "safe", "exception": str(e)}
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        predicted_buggy = r["verdict"] == "buggy"
        actual_buggy = r["is_buggy"]
        if actual_buggy and predicted_buggy:
            tp += 1
        elif actual_buggy and not predicted_buggy:
            fn += 1
        elif not actual_buggy and predicted_buggy:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Real Pyright + Mypy Baseline on Suite D (50 benchmarks)")
    print("  Running ACTUAL tools — no simulation / reimplementation")
    print("=" * 70)

    benchmarks = list(EXTERNAL_PYTORCH_BENCHMARKS.items())
    n = len(benchmarks)

    pyright_results = []
    mypy_results = []

    for i, (name, info) in enumerate(benchmarks, 1):
        label = "BUG" if info["is_buggy"] else "OK "
        print(f"\n[{i:2d}/{n}] {name} (ground truth: {label})")

        # --- Pyright ---
        t0 = time.time()
        pr = run_pyright(info["source"])
        pr_time = time.time() - t0
        pr["name"] = name
        pr["is_buggy"] = info["is_buggy"]
        pr["category"] = info["category"]
        pr["time_s"] = round(pr_time, 2)
        pyright_results.append(pr)

        pr_mark = "✓" if (pr["verdict"] == "buggy") == info["is_buggy"] else "✗"
        print(f"  pyright: {pr['verdict']:5s}  errors={pr['errors']}  "
              f"({pr_time:.1f}s) {pr_mark}")

        # --- Mypy ---
        t0 = time.time()
        mr = run_mypy(info["source"])
        mr_time = time.time() - t0
        mr["name"] = name
        mr["is_buggy"] = info["is_buggy"]
        mr["category"] = info["category"]
        mr["time_s"] = round(mr_time, 2)
        mypy_results.append(mr)

        mr_mark = "✓" if (mr["verdict"] == "buggy") == info["is_buggy"] else "✗"
        print(f"  mypy:    {mr['verdict']:5s}  errors={mr['errors']}  "
              f"({mr_time:.1f}s) {mr_mark}")

    # --- Summary ---
    pyright_metrics = compute_metrics(pyright_results)
    mypy_metrics = compute_metrics(mypy_results)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for tool_name, metrics in [("Pyright", pyright_metrics), ("Mypy", mypy_metrics)]:
        print(f"\n  {tool_name}:")
        print(f"    TP={metrics['TP']}  FP={metrics['FP']}  "
              f"TN={metrics['TN']}  FN={metrics['FN']}")
        print(f"    Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  F1={metrics['F1']:.4f}")
        print(f"    Accuracy={metrics['accuracy']:.4f}")

    print("\n  Note: Both tools perform *type* checking, not *shape/value*")
    print("  checking. Shape dimension mismatches (e.g. Linear(128, 10)")
    print("  fed 256-dim input) involve correct Python types (int) but")
    print("  wrong values — invisible to type checkers.")

    # --- Save ---
    output = {
        "description": (
            "Real pyright and mypy baseline on Suite D benchmarks. "
            "Tools were actually executed (not simulated). "
            "This addresses reviewer concern about reimplementation bias."
        ),
        "pyright_version": _get_version("pyright"),
        "mypy_version": _get_version("mypy"),
        "num_benchmarks": n,
        "pyright": {
            "metrics": pyright_metrics,
            "per_benchmark": pyright_results,
        },
        "mypy": {
            "metrics": mypy_metrics,
            "per_benchmark": mypy_results,
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")


def _get_version(tool: str) -> str:
    try:
        r = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
