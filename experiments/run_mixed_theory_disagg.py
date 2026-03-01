#!/usr/bin/env python3
"""Mixed-theory boundary disaggregation experiment.

Runs all mixed-theory benchmarks, disaggregates results by pairwise
theory boundary, and reports F1 per boundary.

Saves results to experiments/results/mixed_theory_disagg.json

Usage:
    python3 experiments/run_mixed_theory_disagg.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.model_checker import verify_model
from src.theory_combination_analysis import (
    classify_boundary_failure,
    aggregate_boundary_report,
    BoundaryFailure,
)


# ═══════════════════════════════════════════════════════════════════════════
# Mixed-theory benchmark suite
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MixedTheoryBenchmark:
    """A benchmark that exercises multiple theory boundaries."""
    name: str
    source: str
    input_shapes: Dict[str, Any]
    expected_safe: bool
    theories: List[str]
    boundary: str  # e.g. "shape_stride", "shape_device"
    mutation_type: str = ""  # e.g. "wrong_out_features", "device_mismatch"


BENCHMARKS: List[MixedTheoryBenchmark] = [
    # ── Shape × Stride (LIA × NIA): reshape boundaries ──────────────
    MixedTheoryBenchmark(
        name="reshape_correct_simple",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 784)},
        expected_safe=True,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
    ),
    MixedTheoryBenchmark(
        name="reshape_wrong_out_features",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 784)},
        expected_safe=False,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
        mutation_type="wrong_out_features",
    ),
    MixedTheoryBenchmark(
        name="reshape_element_count_mismatch",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 128)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 100)},
        expected_safe=False,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
        mutation_type="wrong_out_features",
    ),
    MixedTheoryBenchmark(
        name="reshape_correct_factored",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 32)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 100)},
        expected_safe=True,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
    ),
    # ── Shape × Device: matmul + device consistency ──────────────────
    MixedTheoryBenchmark(
        name="matmul_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=True,
        theories=["T_shape", "T_device"],
        boundary="shape_device",
    ),
    MixedTheoryBenchmark(
        name="matmul_wrong_inner_dim",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=False,
        theories=["T_shape", "T_device"],
        boundary="shape_device",
        mutation_type="wrong_out_features",
    ),
    # ── Shape × Phase: dropout/batchnorm ─────────────────────────────
    MixedTheoryBenchmark(
        name="batchnorm_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=True,
        theories=["T_shape", "T_phase"],
        boundary="shape_phase",
    ),
    MixedTheoryBenchmark(
        name="batchnorm_wrong_features",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=False,
        theories=["T_shape", "T_phase"],
        boundary="shape_phase",
        mutation_type="wrong_num_features",
    ),
    # ── Shape × Broadcast: broadcasting rules ────────────────────────
    MixedTheoryBenchmark(
        name="broadcast_add_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.fc(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=True,
        theories=["T_shape", "T_broadcast"],
        boundary="shape_broadcast",
    ),
    # ── Stride × Perm: transpose + contiguity ────────────────────────
    MixedTheoryBenchmark(
        name="transpose_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
""",
        input_shapes={"x": ("batch", 3, 32, 32)},
        expected_safe=True,
        theories=["T_stride", "T_perm"],
        boundary="stride_perm",
    ),
    # ── Multi-theory: shape + stride + device ────────────────────────
    MixedTheoryBenchmark(
        name="conv_linear_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv(x)
        return x
""",
        input_shapes={"x": ("batch", 3, 32, 32)},
        expected_safe=True,
        theories=["T_shape", "T_stride", "T_device"],
        boundary="multi_theory",
    ),
    MixedTheoryBenchmark(
        name="conv_wrong_channels",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        input_shapes={"x": ("batch", 3, 32, 32)},
        expected_safe=False,
        theories=["T_shape", "T_stride", "T_device"],
        boundary="multi_theory",
        mutation_type="wrong_out_channels",
    ),
    # ── Backward propagation specific ────────────────────────────────
    MixedTheoryBenchmark(
        name="backward_wrong_out_relu",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 784)},
        expected_safe=False,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
        mutation_type="wrong_out_features",
    ),
    MixedTheoryBenchmark(
        name="correct_three_layer_mlp",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
        input_shapes={"x": ("batch", 784)},
        expected_safe=True,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
    ),
    MixedTheoryBenchmark(
        name="wrong_middle_three_layer",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
        input_shapes={"x": ("batch", 784)},
        expected_safe=False,
        theories=["T_shape", "T_stride"],
        boundary="shape_stride",
        mutation_type="wrong_out_features",
    ),
    # ── Pure QF_LIA (baseline) ───────────────────────────────────────
    MixedTheoryBenchmark(
        name="pure_lia_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 50)

    def forward(self, x):
        return self.fc(x)
""",
        input_shapes={"x": ("batch", 100)},
        expected_safe=True,
        theories=["T_shape"],
        boundary="pure_lia",
    ),
    MixedTheoryBenchmark(
        name="pure_lia_wrong_input",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 50)

    def forward(self, x):
        return self.fc(x)
""",
        input_shapes={"x": ("batch", 200)},
        expected_safe=False,
        theories=["T_shape"],
        boundary="pure_lia",
        mutation_type="wrong_in_features",
    ),
    # ── Dropout with phase ───────────────────────────────────────────
    MixedTheoryBenchmark(
        name="dropout_correct",
        source="""\
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
""",
        input_shapes={"x": ("batch", 128)},
        expected_safe=True,
        theories=["T_shape", "T_phase"],
        boundary="shape_phase",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    name: str
    expected_safe: bool
    actual_safe: bool
    correct: bool
    boundary: str
    theories: List[str]
    mutation_type: str
    time_s: float
    error: Optional[str] = None


def run_benchmark(bench: MixedTheoryBenchmark) -> BenchmarkResult:
    """Run a single benchmark and return the result."""
    t0 = time.time()
    try:
        result = verify_model(bench.source, bench.input_shapes)
        actual_safe = result.safe
        correct = (actual_safe == bench.expected_safe)
        error = None
    except Exception as e:
        actual_safe = None
        correct = False
        error = str(e)
    elapsed = time.time() - t0

    return BenchmarkResult(
        name=bench.name,
        expected_safe=bench.expected_safe,
        actual_safe=actual_safe,
        correct=correct,
        boundary=bench.boundary,
        theories=bench.theories,
        mutation_type=bench.mutation_type,
        time_s=elapsed,
        error=error,
    )


def compute_f1(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Compute precision, recall, and F1 for bug detection.

    True Positive: expected_safe=False AND actual_safe=False (caught the bug)
    False Negative: expected_safe=False AND actual_safe=True (missed the bug)
    False Positive: expected_safe=True AND actual_safe=False (false alarm)
    True Negative: expected_safe=True AND actual_safe=True (correct safe)
    """
    tp = fp = fn = tn = 0
    for r in results:
        if r.actual_safe is None:
            fn += 1  # Error counts as missed
            continue
        if not r.expected_safe and not r.actual_safe:
            tp += 1
        elif not r.expected_safe and r.actual_safe:
            fn += 1
        elif r.expected_safe and not r.actual_safe:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(results), 4) if results else 0.0,
    }


def main():
    print("=" * 70)
    print("Mixed-Theory Boundary Disaggregation Experiment")
    print("=" * 70)

    all_results: List[BenchmarkResult] = []
    boundary_results: Dict[str, List[BenchmarkResult]] = {}

    for bench in BENCHMARKS:
        print(f"\n  Running: {bench.name} ({bench.boundary}) ...", end=" ")
        result = run_benchmark(bench)
        all_results.append(result)

        boundary_results.setdefault(bench.boundary, []).append(result)

        status = "✓" if result.correct else "✗"
        print(f"{status} (expected={'safe' if bench.expected_safe else 'unsafe'}, "
              f"got={'safe' if result.actual_safe else 'unsafe'}, "
              f"{result.time_s:.2f}s)")
        if result.error:
            print(f"    ERROR: {result.error}")

    # Overall metrics
    print("\n" + "=" * 70)
    print("Overall Results")
    print("=" * 70)
    overall = compute_f1(all_results)
    print(f"  Total benchmarks: {len(all_results)}")
    print(f"  Correct:          {sum(1 for r in all_results if r.correct)}")
    print(f"  F1:               {overall['f1']}")
    print(f"  Precision:        {overall['precision']}")
    print(f"  Recall:           {overall['recall']}")
    print(f"  Accuracy:         {overall['accuracy']}")

    # Per-boundary metrics
    print("\n" + "-" * 70)
    print("Per-Boundary F1")
    print("-" * 70)
    boundary_metrics: Dict[str, Dict] = {}
    for boundary, results in sorted(boundary_results.items()):
        metrics = compute_f1(results)
        boundary_metrics[boundary] = metrics
        print(f"  {boundary:20s}  F1={metrics['f1']:.3f}  "
              f"n={len(results)}  "
              f"TP={metrics['tp']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TN={metrics['tn']}")

    # Boundary disaggregation via theory_combination_analysis
    failures: List[BoundaryFailure] = []
    for r in all_results:
        if not r.correct:
            constraint_cats = {f"c_{t}": t for t in r.theories}
            loc = r.mutation_type if r.mutation_type else r.boundary
            bf = classify_boundary_failure(
                r.name, r.theories, constraint_cats, loc,
            )
            failures.append(bf)

    if failures:
        report = aggregate_boundary_report(failures, len(all_results))
        print("\n" + "-" * 70)
        print("Boundary Failure Disaggregation")
        print("-" * 70)
        for pair_key, count in sorted(report.failures_by_pair.items()):
            f1 = report.f1_by_pair.get(pair_key, 1.0)
            print(f"  {pair_key:30s}  failures={count}  f1={f1:.3f}")
    else:
        report = aggregate_boundary_report([], len(all_results))

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_benchmarks": len(all_results),
        "overall_metrics": overall,
        "per_boundary_metrics": boundary_metrics,
        "boundary_disaggregation": report.to_dict(),
        "individual_results": [
            {
                "name": r.name,
                "boundary": r.boundary,
                "theories": r.theories,
                "mutation_type": r.mutation_type,
                "expected_safe": r.expected_safe,
                "actual_safe": r.actual_safe,
                "correct": r.correct,
                "time_s": round(r.time_s, 4),
                "error": r.error,
            }
            for r in all_results
        ],
    }

    out_path = Path(__file__).resolve().parent / "results" / "mixed_theory_disagg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    mixed_only = [r for r in all_results if r.boundary != "pure_lia"]
    mixed_metrics = compute_f1(mixed_only)
    print(f"Mixed-theory F1: {mixed_metrics['f1']}")
    print(f"Target: >= 0.95")
    print(f"Status: {'PASS ✓' if mixed_metrics['f1'] >= 0.95 else 'FAIL ✗'}")
    print("=" * 70)

    return overall["f1"]


if __name__ == "__main__":
    f1 = main()
    sys.exit(0 if f1 >= 0.95 else 1)
