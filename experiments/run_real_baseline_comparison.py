#!/usr/bin/env python3
"""
Real baseline comparison: TensorGuard vs TorchScript vs mypy.

Runs each tool on the same benchmark suite and collects:
  - bugs found (true positives, false positives, false negatives)
  - analysis time
  - coverage (what fraction of bug classes each tool catches)

This produces experiments/results/baseline_real_comparison.json
"""

import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shape_bug_benchmarks import ALL_BENCHMARKS, BenchmarkModel


# ─── Result types ────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    detected_bug: bool
    error_message: str = ""
    analysis_time_ms: float = 0.0
    tool_error: str = ""  # if the tool itself crashed


@dataclass
class BenchmarkResult:
    model_name: str
    has_bug: bool
    bug_description: str
    tensorguard: Optional[ToolResult] = None
    torchscript: Optional[ToolResult] = None
    mypy: Optional[ToolResult] = None


# ─── TensorGuard runner ─────────────────────────────────────────────────────

def run_tensorguard(model: BenchmarkModel) -> ToolResult:
    """Run TensorGuard verification on a model."""
    try:
        from src.model_checker import verify_model

        # Determine input shapes based on model architecture
        source = model.source
        input_shapes = _guess_input_shapes(source)

        start = time.perf_counter()
        result = verify_model(source, input_shapes=input_shapes)
        elapsed = (time.perf_counter() - start) * 1000

        detected = not result.safe
        msg = ""
        if not result.safe and result.counterexample:
            msg = result.counterexample.pretty()

        return ToolResult(
            detected_bug=detected,
            error_message=msg[:500],
            analysis_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        return ToolResult(
            detected_bug=False,
            analysis_time_ms=0,
            tool_error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _guess_input_shapes(source: str) -> Dict[str, tuple]:
    """Guess input shapes from model source code."""
    if "Conv2d(3," in source or "Conv2d(3 ," in source:
        return {"x": ("batch", 3, 32, 32)}
    if "Conv2d(64," in source and "Conv2d(3," not in source:
        return {"x": ("batch", 64, 32, 32)}
    if "nn.Linear(784," in source:
        return {"x": ("batch", 784)}
    if "nn.Linear(768," in source:
        return {"x": ("batch", 768)}
    if "nn.Linear(512," in source or "d_model=512" in source:
        return {"x": ("batch", 10, 512)}
    if "d_model=256" in source:
        return {"x": ("batch", 10, 256)}
    if "input_size=256" in source:
        return {"x": ("batch", 20, 256)}
    if "input_size=128" in source:
        return {"x": ("batch", 20, 128)}
    return {"x": ("batch", 784)}


# ─── TorchScript runner ──────────────────────────────────────────────────────

def run_torchscript(model: BenchmarkModel) -> ToolResult:
    """Run TorchScript type checking (torch.jit.script) on a model."""
    try:
        import torch
        import torch.nn as nn

        # Write source to temp file and execute to get the class
        temp_ns: Dict = {}
        exec(model.source, {"torch": torch, "nn": nn, "__name__": "__main__"}, temp_ns)

        # Find the nn.Module subclass
        model_cls = None
        for name, obj in temp_ns.items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                model_cls = obj
                break

        if model_cls is None:
            return ToolResult(
                detected_bug=False,
                tool_error="Could not find nn.Module subclass",
            )

        start = time.perf_counter()
        try:
            instance = model_cls()
            scripted = torch.jit.script(instance)
            # TorchScript compiles but doesn't verify shapes — we also
            # try to trace with a sample input to trigger shape checking
            input_shapes = _guess_input_shapes(model.source)
            shape_tuple = input_shapes.get("x", (1, 784))
            # Replace symbolic dims with concrete values
            concrete = tuple(1 if isinstance(d, str) else d for d in shape_tuple)
            sample = torch.randn(*concrete)
            try:
                _ = scripted(sample)
                detected = False
                msg = ""
            except RuntimeError as e:
                detected = True
                msg = str(e)[:500]
        except Exception as e:
            detected = True
            msg = str(e)[:500]

        elapsed = (time.perf_counter() - start) * 1000
        return ToolResult(
            detected_bug=detected,
            error_message=msg,
            analysis_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        return ToolResult(
            detected_bug=False,
            tool_error=f"{type(e).__name__}: {str(e)[:200]}",
        )


# ─── mypy runner ─────────────────────────────────────────────────────────────

def run_mypy(model: BenchmarkModel) -> ToolResult:
    """Run mypy type checking on model source."""
    try:
        from mypy import api as mypy_api

        # Write source to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(model.source)
            f.flush()
            tmp_path = f.name

        try:
            start = time.perf_counter()
            stdout, stderr, exit_code = mypy_api.run(
                ["--no-error-summary", "--ignore-missing-imports", tmp_path]
            )
            elapsed = (time.perf_counter() - start) * 1000

            # mypy will report type errors but NOT shape errors
            has_errors = exit_code != 0 and "error:" in stdout
            msg = stdout[:500] if has_errors else ""

            return ToolResult(
                detected_bug=has_errors,
                error_message=msg,
                analysis_time_ms=round(elapsed, 2),
            )
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return ToolResult(
            detected_bug=False,
            tool_error=f"{type(e).__name__}: {str(e)[:200]}",
        )


# ─── Main comparison ─────────────────────────────────────────────────────────

def compute_metrics(results: List[BenchmarkResult], tool_name: str) -> Dict:
    """Compute precision, recall, F1 for a tool."""
    tp = fp = fn = tn = 0
    times = []
    errors = 0

    for r in results:
        tool_result: Optional[ToolResult] = getattr(r, tool_name)
        if tool_result is None or tool_result.tool_error:
            errors += 1
            if r.has_bug:
                fn += 1
            continue

        if tool_result.analysis_time_ms > 0:
            times.append(tool_result.analysis_time_ms)

        if r.has_bug and tool_result.detected_bug:
            tp += 1
        elif r.has_bug and not tool_result.detected_bug:
            fn += 1
        elif not r.has_bug and tool_result.detected_bug:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_time_ms": round(sum(times) / len(times), 2) if times else 0,
        "median_time_ms": round(sorted(times)[len(times) // 2], 2) if times else 0,
        "tool_errors": errors,
    }


def main():
    print("=" * 70)
    print("TensorGuard Baseline Comparison")
    print("=" * 70)
    print(f"Benchmarks: {len(ALL_BENCHMARKS)} models")
    print(f"  Buggy: {sum(1 for b in ALL_BENCHMARKS if b.has_bug)}")
    print(f"  Correct: {sum(1 for b in ALL_BENCHMARKS if not b.has_bug)}")
    print()

    results: List[BenchmarkResult] = []

    for i, bench in enumerate(ALL_BENCHMARKS):
        print(f"[{i+1:2d}/{len(ALL_BENCHMARKS)}] {bench.name:<35s}", end=" ", flush=True)

        br = BenchmarkResult(
            model_name=bench.name,
            has_bug=bench.has_bug,
            bug_description=bench.bug_description,
        )

        # Run TensorGuard
        br.tensorguard = run_tensorguard(bench)
        tg_status = "✓" if br.tensorguard.detected_bug == bench.has_bug else "✗"

        # Run TorchScript
        br.torchscript = run_torchscript(bench)
        ts_status = "✓" if br.torchscript.detected_bug == bench.has_bug else "✗"

        # Run mypy
        br.mypy = run_mypy(bench)
        my_status = "✓" if br.mypy.detected_bug == bench.has_bug else "✗"

        label = "BUG" if bench.has_bug else "OK "
        print(f"[{label}] TG:{tg_status} TS:{ts_status} MY:{my_status}  "
              f"({br.tensorguard.analysis_time_ms:.0f}/{br.torchscript.analysis_time_ms:.0f}/{br.mypy.analysis_time_ms:.0f}ms)")

        results.append(br)

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    metrics = {}
    for tool in ["tensorguard", "torchscript", "mypy"]:
        m = compute_metrics(results, tool)
        metrics[tool] = m
        print(f"\n{tool.upper():>12s}:  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})  "
              f"Time={m['mean_time_ms']:.1f}ms")

    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "description": "Real baseline comparison: TensorGuard vs TorchScript vs mypy on shape bug benchmarks",
        "num_benchmarks": len(ALL_BENCHMARKS),
        "num_buggy": sum(1 for b in ALL_BENCHMARKS if b.has_bug),
        "num_correct": sum(1 for b in ALL_BENCHMARKS if not b.has_bug),
        "metrics": metrics,
        "per_model": [
            {
                "name": r.model_name,
                "has_bug": r.has_bug,
                "bug_description": r.bug_description,
                "tensorguard": asdict(r.tensorguard) if r.tensorguard else None,
                "torchscript": asdict(r.torchscript) if r.torchscript else None,
                "mypy": asdict(r.mypy) if r.mypy else None,
            }
            for r in results
        ],
    }

    out_path = output_dir / "baseline_real_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (for paper)")
    print("=" * 70)
    print(f"{'Tool':<15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Time(ms)':>10s} {'FP':>5s}")
    print("-" * 60)
    for tool, m in metrics.items():
        print(f"{tool:<15s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} "
              f"{m['mean_time_ms']:>10.1f} {m['fp']:>5d}")

    return metrics


if __name__ == "__main__":
    main()
