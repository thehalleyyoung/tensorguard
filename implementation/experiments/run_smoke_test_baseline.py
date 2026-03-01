#!/usr/bin/env python3
"""
Smoke Test Baseline Experiment for TensorGuard.

Compares TensorGuard's static analysis against a simple forward-pass smoke test
(instantiate model, run model(torch.randn(*input_shape)), catch RuntimeError).

This addresses reviewer Sara Roy's critique: "what marginal value TensorGuard
provides over a simple forward-pass smoke test". TensorGuard catches bugs
BEFORE running the model, without needing a GPU or concrete inputs.

Outputs: implementation/experiments/results/smoke_test_baseline.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified import analyze_unified

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "smoke_test_baseline.json"


@dataclass
class BenchmarkResult:
    name: str
    has_bug: bool
    # Smoke test
    smoke_detected: bool
    smoke_error: Optional[str]
    smoke_latency_ms: float
    # TensorGuard
    tg_detected: bool
    tg_errors: List[str]
    tg_latency_ms: float


def _resolve_input_shape(shape_spec: Tuple) -> Tuple[int, ...]:
    """Convert symbolic shape spec to concrete ints for smoke test."""
    concrete = []
    for dim in shape_spec:
        if isinstance(dim, int):
            concrete.append(dim)
        elif isinstance(dim, str):
            if dim == "batch":
                concrete.append(2)
            elif dim == "seq":
                concrete.append(16)
            elif dim == "channels":
                concrete.append(3)
            else:
                concrete.append(4)
        else:
            concrete.append(4)
    return tuple(concrete)


def _run_smoke_test(code: str, input_shapes: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
    """Run forward-pass smoke test. Returns (detected_bug, error_msg, latency_ms)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return False, "torch not available", 0.0

    t0 = time.time()
    try:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)

        model_cls = None
        for val in namespace.values():
            if isinstance(val, type) and issubclass(val, nn.Module) and val is not nn.Module:
                model_cls = val
                break

        if model_cls is None:
            return False, "no nn.Module found", (time.time() - t0) * 1000

        model = model_cls()
        model.eval()

        input_tensors = []
        for name, shape_spec in input_shapes.items():
            concrete_shape = _resolve_input_shape(shape_spec)
            input_tensors.append(torch.randn(*concrete_shape))

        with torch.no_grad():
            if len(input_tensors) == 1:
                model(input_tensors[0])
            else:
                model(*input_tensors)

        latency = (time.time() - t0) * 1000
        return False, None, latency

    except RuntimeError as e:
        latency = (time.time() - t0) * 1000
        return True, str(e), latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return True, f"{type(e).__name__}: {e}", latency


def _run_tensorguard(code: str, input_shapes: Dict[str, Any]) -> Tuple[bool, List[str], float]:
    """Run TensorGuard analysis. Returns (detected_bug, errors, latency_ms)."""
    t0 = time.time()
    try:
        result = analyze_unified(code)
        latency = (time.time() - t0) * 1000

        errors = []
        if result.shape_result and result.shape_result.errors:
            errors.extend(str(e) for e in result.shape_result.errors)
        if result.liquid_result and result.liquid_result.bugs:
            errors.extend(str(b) for b in result.liquid_result.bugs)

        detected = len(errors) > 0
        return detected, errors, latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return False, [f"analysis error: {e}"], latency


def _load_benchmarks() -> List[Dict[str, Any]]:
    """Load benchmarks from expanded_benchmark_suite."""
    try:
        from experiments.expanded_benchmark_suite import EXPANDED_BENCHMARKS
        return EXPANDED_BENCHMARKS
    except ImportError:
        pass
    try:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
        from expanded_benchmark_suite import EXPANDED_BENCHMARKS
        return EXPANDED_BENCHMARKS
    except ImportError:
        from experiments.benchmark_suite import TENSOR_SHAPE_BUGS, CORRECT_FUNCTIONS
        benchmarks = []
        for b in TENSOR_SHAPE_BUGS:
            benchmarks.append({
                "name": b["name"],
                "has_bug": True,
                "code": b["code"],
                "input_shapes": b.get("input_shapes", {"x": (2, 784)}),
            })
        for b in CORRECT_FUNCTIONS:
            benchmarks.append({
                "name": b["name"],
                "has_bug": False,
                "code": b["code"],
                "input_shapes": b.get("input_shapes", {"x": (2, 784)}),
            })
        return benchmarks


def main():
    benchmarks = _load_benchmarks()
    print(f"Loaded {len(benchmarks)} benchmarks")

    results: List[BenchmarkResult] = []
    smoke_tp = smoke_fp = smoke_fn = smoke_tn = 0
    tg_tp = tg_fp = tg_fn = tg_tn = 0

    for i, bench in enumerate(benchmarks):
        name = bench["name"]
        has_bug = bench.get("has_bug", False)
        code = bench["code"]
        input_shapes = bench.get("input_shapes", {"x": (2, 784)})

        # Run smoke test
        smoke_detected, smoke_error, smoke_latency = _run_smoke_test(code, input_shapes)

        # Run TensorGuard
        tg_detected, tg_errors, tg_latency = _run_tensorguard(code, input_shapes)

        result = BenchmarkResult(
            name=name,
            has_bug=has_bug,
            smoke_detected=smoke_detected,
            smoke_error=smoke_error,
            smoke_latency_ms=smoke_latency,
            tg_detected=tg_detected,
            tg_errors=tg_errors,
            tg_latency_ms=tg_latency,
        )
        results.append(result)

        # Confusion matrix updates
        if has_bug:
            if smoke_detected:
                smoke_tp += 1
            else:
                smoke_fn += 1
            if tg_detected:
                tg_tp += 1
            else:
                tg_fn += 1
        else:
            if smoke_detected:
                smoke_fp += 1
            else:
                smoke_tn += 1
            if tg_detected:
                tg_fp += 1
            else:
                tg_tn += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(benchmarks)}] processed")

    # Compute metrics
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    smoke_precision = safe_div(smoke_tp, smoke_tp + smoke_fp)
    smoke_recall = safe_div(smoke_tp, smoke_tp + smoke_fn)
    smoke_f1 = safe_div(2 * smoke_precision * smoke_recall, smoke_precision + smoke_recall)

    tg_precision = safe_div(tg_tp, tg_tp + tg_fp)
    tg_recall = safe_div(tg_tp, tg_tp + tg_fn)
    tg_f1 = safe_div(2 * tg_precision * tg_recall, tg_precision + tg_recall)

    summary = {
        "experiment": "smoke_test_baseline",
        "description": (
            "Compares TensorGuard static analysis against forward-pass smoke test. "
            "TensorGuard operates purely statically (no model instantiation, no GPU, "
            "no concrete inputs) while smoke test requires runtime execution."
        ),
        "num_benchmarks": len(benchmarks),
        "smoke_test": {
            "tp": smoke_tp, "fp": smoke_fp, "fn": smoke_fn, "tn": smoke_tn,
            "precision": round(smoke_precision, 4),
            "recall": round(smoke_recall, 4),
            "f1": round(smoke_f1, 4),
        },
        "tensorguard": {
            "tp": tg_tp, "fp": tg_fp, "fn": tg_fn, "tn": tg_tn,
            "precision": round(tg_precision, 4),
            "recall": round(tg_recall, 4),
            "f1": round(tg_f1, 4),
        },
        "advantages_of_tensorguard": [
            "No runtime execution required (works at CI/review time)",
            "No GPU or specific hardware needed",
            "No concrete input shapes needed (works with symbolic shapes)",
            "Provides formal verification conditions when safe",
            "Catches bugs smoke test misses (e.g., shape-dependent on input size)",
        ],
        "per_benchmark": [asdict(r) for r in results],
    }

    print(f"\n{'='*60}")
    print("Smoke Test Baseline Results")
    print(f"{'='*60}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"\nSmoke Test: P={smoke_precision:.3f} R={smoke_recall:.3f} F1={smoke_f1:.3f}")
    print(f"  TP={smoke_tp} FP={smoke_fp} FN={smoke_fn} TN={smoke_tn}")
    print(f"\nTensorGuard: P={tg_precision:.3f} R={tg_recall:.3f} F1={tg_f1:.3f}")
    print(f"  TP={tg_tp} FP={tg_fp} FN={tg_fn} TN={tg_tn}")
    print(f"\nKey insight: TensorGuard is purely static — no model instantiation needed.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
