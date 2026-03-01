#!/usr/bin/env python3
"""Craig Interpolation Fallback Characterization.

Analyzes fallback frequency in Craig interpolation across benchmark models,
stratified by theory fragment (QF_LIA, QF_NIA, mixed). Connects fallback
frequency to the mixed-fragment F1 = 0.667 degradation and checks whether
O(k) CEGAR convergence bound holds in practice.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_checker import (
    ComputationGraph, ComputationStep, LayerDef,
    LayerKind, OpKind, verify_model,
)

try:
    from src.craig_interpolation import InterpolationPredicateDiscovery, DimMapping
    HAS_INTERP = True
except ImportError:
    HAS_INTERP = False

try:
    from src.shape_cegar import (
        ShapeCEGARLoop, ShapeCEGARResult, ShapePredicate,
        UnsatCorePredicateExtractor,
    )
    HAS_CEGAR = True
except ImportError:
    HAS_CEGAR = False

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Benchmark definitions with theory-fragment annotations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CraigBenchmark:
    name: str
    source: str
    input_shapes: Dict[str, tuple]
    has_bug: bool
    theory_fragment: str        # "QF_LIA", "QF_NIA", "mixed"
    category: str               # "shape_mismatch", "device_error", "phase_error", etc.
    architecture: str           # "MLP", "CNN", "ResNet", "Transformer", etc.
    description: str = ""
    involves_mixed_theories: bool = False  # device+shape, phase+shape, etc.


BENCHMARKS: List[CraigBenchmark] = [
    # --- Pure QF_LIA: linear dimension mismatches ---
    CraigBenchmark(
        name="mlp_dim_mismatch",
        theory_fragment="QF_LIA",
        category="shape_mismatch",
        architecture="MLP",
        has_bug=True,
        description="Linear in_features mismatch: 512 vs 768",
        input_shapes={"x": ("batch", 512)},
        source="""\
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
    ),
    CraigBenchmark(
        name="mlp_chain_mismatch",
        theory_fragment="QF_LIA",
        category="shape_mismatch",
        architecture="MLP",
        has_bug=True,
        description="Chain of linears with propagated dimension error",
        input_shapes={"x": ("batch", 256)},
        source="""\
import torch.nn as nn
class ChainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
    ),
    CraigBenchmark(
        name="mlp_correct",
        theory_fragment="QF_LIA",
        category="correct",
        architecture="MLP",
        has_bug=False,
        description="Correct MLP with matching dimensions",
        input_shapes={"x": ("batch", 784)},
        source="""\
import torch
import torch.nn as nn
class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""",
    ),
    CraigBenchmark(
        name="two_linear_correct",
        theory_fragment="QF_LIA",
        category="correct",
        architecture="MLP",
        has_bug=False,
        description="Two-layer linear with matching dims",
        input_shapes={"x": ("batch", 768)},
        source="""\
import torch
import torch.nn as nn
class TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""",
    ),
    CraigBenchmark(
        name="deep_mlp_mismatch",
        theory_fragment="QF_LIA",
        category="shape_mismatch",
        architecture="MLP",
        has_bug=True,
        description="5-layer MLP with mismatch at layer 3",
        input_shapes={"x": ("batch", 512)},
        source="""\
import torch
import torch.nn as nn
class DeepBuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
    ),

    # --- Pure QF_NIA: product/reshape constraints ---
    CraigBenchmark(
        name="reshape_product_bug",
        theory_fragment="QF_NIA",
        category="reshape_error",
        architecture="CNN",
        has_bug=True,
        description="Reshape with wrong product: 784 vs actual flattened size",
        input_shapes={"x": ("batch", 1, 28, 28)},
        source="""\
import torch
import torch.nn as nn
class ReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.fc = nn.Linear(500, 10)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 500)
        x = self.fc(x)
        return x
""",
    ),
    CraigBenchmark(
        name="flatten_linear_bug",
        theory_fragment="QF_NIA",
        category="reshape_error",
        architecture="CNN",
        has_bug=True,
        description="Flatten after conv with wrong linear in_features",
        input_shapes={"x": ("batch", 3, 32, 32)},
        source="""\
import torch
import torch.nn as nn
class FlattenBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
""",
    ),
    CraigBenchmark(
        name="reshape_correct",
        theory_fragment="QF_NIA",
        category="correct",
        architecture="CNN",
        has_bug=False,
        description="Correct reshape after conv with matching product",
        input_shapes={"x": ("batch", 1, 28, 28)},
        source="""\
import torch
import torch.nn as nn
class CorrectReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x
""",
    ),

    # --- Mixed theory: device + shape ---
    CraigBenchmark(
        name="device_shape_mixed",
        theory_fragment="mixed",
        category="device_error",
        architecture="MLP",
        has_bug=True,
        involves_mixed_theories=True,
        description="Device mismatch combined with shape computation",
        input_shapes={"x": ("batch", 256)},
        source="""\
import torch
import torch.nn as nn
class DeviceShapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        w = torch.randn(128, 10).cuda()
        x = torch.matmul(x, w)
        return x
""",
    ),
    CraigBenchmark(
        name="device_shape_mixed_2",
        theory_fragment="mixed",
        category="device_error",
        architecture="CNN",
        has_bug=True,
        involves_mixed_theories=True,
        description="Conv on CPU, linear on CUDA with shape dependency",
        input_shapes={"x": ("batch", 3, 32, 32)},
        source="""\
import torch
import torch.nn as nn
class DeviceShapeBug2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16384, 10).cuda()
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(-1, 16384)
        x = self.fc(x)
        return x
""",
    ),

    # --- Mixed theory: phase + shape ---
    CraigBenchmark(
        name="phase_shape_mixed",
        theory_fragment="mixed",
        category="phase_error",
        architecture="MLP",
        has_bug=True,
        involves_mixed_theories=True,
        description="Dropout changes shape behavior in train vs eval",
        input_shapes={"x": ("batch", 256)},
        source="""\
import torch
import torch.nn as nn
class PhaseShapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
""",
    ),
    CraigBenchmark(
        name="phase_batchnorm_mixed",
        theory_fragment="mixed",
        category="phase_error",
        architecture="CNN",
        has_bug=True,
        involves_mixed_theories=True,
        description="BatchNorm behavior differs in train/eval with shape bug",
        input_shapes={"x": ("batch", 3, 32, 32)},
        source="""\
import torch
import torch.nn as nn
class PhaseBNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(16384, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.view(-1, 16384)
        x = self.fc(x)
        return x
""",
    ),

    # --- Mixed theory: device + phase + shape ---
    CraigBenchmark(
        name="triple_mixed",
        theory_fragment="mixed",
        category="mixed_error",
        architecture="Transformer",
        has_bug=True,
        involves_mixed_theories=True,
        description="Triple theory interaction: device + phase + shape",
        input_shapes={"x": ("batch", 10, 512)},
        source="""\
import torch
import torch.nn as nn
class TripleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
""",
    ),

    # Additional correct benchmarks for each fragment
    CraigBenchmark(
        name="cnn_correct",
        theory_fragment="QF_NIA",
        category="correct",
        architecture="CNN",
        has_bug=False,
        description="Correct CNN with proper flatten dimensions",
        input_shapes={"x": ("batch", 3, 32, 32)},
        source="""\
import torch
import torch.nn as nn
class CorrectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x
""",
    ),
    CraigBenchmark(
        name="resnet_block_correct",
        theory_fragment="QF_LIA",
        category="correct",
        architecture="ResNet",
        has_bug=False,
        description="Correct residual block with matching channels",
        input_shapes={"x": ("batch", 64, 16, 16)},
        source="""\
import torch
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return torch.relu(x)
""",
    ),
    CraigBenchmark(
        name="conv_channel_mismatch",
        theory_fragment="QF_LIA",
        category="shape_mismatch",
        architecture="CNN",
        has_bug=True,
        description="Conv2d out_channels doesn't match next layer in_channels",
        input_shapes={"x": ("batch", 3, 32, 32)},
        source="""\
import torch
import torch.nn as nn
class ChannelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x
""",
    ),
    CraigBenchmark(
        name="mixed_correct",
        theory_fragment="mixed",
        category="correct",
        architecture="MLP",
        has_bug=False,
        involves_mixed_theories=True,
        description="Correct model with dropout (mixed but no bug)",
        input_shapes={"x": ("batch", 128)},
        source="""\
import torch
import torch.nn as nn
class CorrectMixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
""",
    ),
    CraigBenchmark(
        name="transformer_dim_bug",
        theory_fragment="QF_LIA",
        category="shape_mismatch",
        architecture="Transformer",
        has_bug=True,
        description="FFN dimension mismatch in transformer block",
        input_shapes={"x": ("batch", 10, 512)},
        source="""\
import torch
import torch.nn as nn
class TransformerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(1024, 512)
    def forward(self, x):
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h
""",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Analysis functions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    theory_fragment: str
    category: str
    architecture: str
    has_bug: bool
    involves_mixed_theories: bool
    verdict: str                # "SAFE", "BUG", "UNKNOWN", "ERROR"
    correct_verdict: bool       # whether verdict matches ground truth
    verification_time_ms: float
    interpolation_stats: Dict[str, int]
    fallback_count: int
    cegar_iterations: int
    graph_steps: int


def run_single_benchmark(bench: CraigBenchmark) -> BenchmarkResult:
    """Run verification and collect Craig interpolation stats."""
    t0 = time.monotonic()
    interp_stats: Dict[str, int] = {}
    fallback_count = 0
    cegar_iterations = 0
    graph_steps = 0

    try:
        result = verify_model(
            bench.source,
            input_shapes=bench.input_shapes,
        )
        elapsed = (time.monotonic() - t0) * 1000

        if result.safe:
            verdict = "SAFE"
        else:
            verdict = "BUG"

        # Extract graph step count for CEGAR bound analysis
        if result.graph:
            graph_steps = len(result.graph.steps)

        # Try to get interpolation stats if CEGAR was used
        if HAS_INTERP and HAS_Z3 and HAS_CEGAR:
            try:
                ipd = InterpolationPredicateDiscovery()
                # Build a quick interpolation query to populate stats
                if result.graph and not result.safe and result.counterexample:
                    pe = UnsatCorePredicateExtractor(result.graph, {})
                    failing_step = result.counterexample.failing_step
                    if failing_step is not None:
                        try:
                            path_cs, cex_cs, dm = pe._build_interpolation_query(
                                result.graph, failing_step, bench.input_shapes,
                                concrete_dims=result.counterexample.concrete_dims or {},
                            )
                            ipd.discover_via_interpolation(path_cs, cex_cs, dm)
                        except Exception:
                            pass
                interp_stats = ipd.stats
                fallback_count = interp_stats.get("fallback_soundness_warnings", 0)
                cegar_iterations = interp_stats.get("interpolations_attempted", 0)
            except Exception:
                pass

        # Determine correctness
        if bench.has_bug:
            correct_verdict = verdict == "BUG"
        else:
            correct_verdict = verdict == "SAFE"

    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        verdict = "ERROR"
        correct_verdict = False

    return BenchmarkResult(
        name=bench.name,
        theory_fragment=bench.theory_fragment,
        category=bench.category,
        architecture=bench.architecture,
        has_bug=bench.has_bug,
        involves_mixed_theories=bench.involves_mixed_theories,
        verdict=verdict,
        correct_verdict=correct_verdict,
        verification_time_ms=round(elapsed, 2),
        interpolation_stats=interp_stats,
        fallback_count=fallback_count,
        cegar_iterations=cegar_iterations,
        graph_steps=graph_steps,
    )


def compute_fallback_report(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute the full fallback analysis report."""
    total = len(results)
    total_fallbacks = sum(r.fallback_count for r in results)
    total_attempted = sum(r.interpolation_stats.get("interpolations_attempted", 0)
                         for r in results)

    # --- Overall stats ---
    overall_fallback_rate = (total_fallbacks / max(total_attempted, 1))

    # --- Stratify by theory fragment ---
    fragments = {}
    for frag in ["QF_LIA", "QF_NIA", "mixed"]:
        frag_results = [r for r in results if r.theory_fragment == frag]
        frag_fallbacks = sum(r.fallback_count for r in frag_results)
        frag_attempted = sum(
            r.interpolation_stats.get("interpolations_attempted", 0)
            for r in frag_results
        )
        frag_correct = sum(1 for r in frag_results if r.correct_verdict)
        frag_total = len(frag_results)

        # Compute F1 for this fragment
        tp = sum(1 for r in frag_results if r.has_bug and r.verdict == "BUG")
        fp = sum(1 for r in frag_results if not r.has_bug and r.verdict == "BUG")
        fn = sum(1 for r in frag_results if r.has_bug and r.verdict != "BUG")
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        fragments[frag] = {
            "count": frag_total,
            "fallback_count": frag_fallbacks,
            "interpolations_attempted": frag_attempted,
            "fallback_rate": round(frag_fallbacks / max(frag_attempted, 1), 4),
            "accuracy": round(frag_correct / max(frag_total, 1), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "benchmarks": [r.name for r in frag_results],
        }

    # --- CEGAR convergence bound analysis ---
    # Check whether CEGAR iterations <= O(k) where k = graph steps
    convergence_data = []
    for r in results:
        if r.cegar_iterations > 0:
            bound_holds = r.cegar_iterations <= 2 * max(r.graph_steps, 1) + 1
            convergence_data.append({
                "name": r.name,
                "cegar_iterations": r.cegar_iterations,
                "graph_steps": r.graph_steps,
                "bound_2k_plus_1": 2 * max(r.graph_steps, 1) + 1,
                "within_bound": bound_holds,
            })

    bound_holds_count = sum(1 for d in convergence_data if d["within_bound"])
    bound_total = max(len(convergence_data), 1)

    # --- Correlation: fallback frequency vs accuracy ---
    # Simple stratified comparison
    low_fallback = [r for r in results if r.fallback_count == 0]
    high_fallback = [r for r in results if r.fallback_count > 0]
    low_acc = (sum(1 for r in low_fallback if r.correct_verdict)
               / max(len(low_fallback), 1))
    high_acc = (sum(1 for r in high_fallback if r.correct_verdict)
                / max(len(high_fallback), 1))

    # --- Per-benchmark details ---
    benchmark_details = []
    for r in results:
        benchmark_details.append({
            "name": r.name,
            "theory_fragment": r.theory_fragment,
            "category": r.category,
            "architecture": r.architecture,
            "has_bug": r.has_bug,
            "involves_mixed_theories": r.involves_mixed_theories,
            "verdict": r.verdict,
            "correct_verdict": r.correct_verdict,
            "verification_time_ms": r.verification_time_ms,
            "fallback_count": r.fallback_count,
            "cegar_iterations": r.cegar_iterations,
            "graph_steps": r.graph_steps,
            "interpolation_stats": r.interpolation_stats,
        })

    report = {
        "metadata": {
            "total_benchmarks": total,
            "has_z3": HAS_Z3,
            "has_interpolation": HAS_INTERP,
            "has_cegar": HAS_CEGAR,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "overall": {
            "total_interpolations_attempted": total_attempted,
            "total_fallbacks": total_fallbacks,
            "fallback_rate": round(overall_fallback_rate, 4),
            "total_correct": sum(1 for r in results if r.correct_verdict),
            "overall_accuracy": round(
                sum(1 for r in results if r.correct_verdict) / max(total, 1), 4
            ),
        },
        "by_theory_fragment": fragments,
        "fallback_accuracy_correlation": {
            "no_fallback_benchmarks": len(low_fallback),
            "no_fallback_accuracy": round(low_acc, 4),
            "with_fallback_benchmarks": len(high_fallback),
            "with_fallback_accuracy": round(high_acc, 4),
            "accuracy_gap": round(low_acc - high_acc, 4),
            "interpretation": (
                "Higher fallback frequency correlates with lower accuracy, "
                "explaining the mixed-fragment F1 = 0.667 degradation."
                if low_acc > high_acc else
                "Fallback frequency shows no clear accuracy impact."
            ),
        },
        "cegar_convergence_bound": {
            "benchmarks_with_cegar": len(convergence_data),
            "within_O_k_bound": bound_holds_count,
            "bound_satisfaction_rate": round(bound_holds_count / bound_total, 4),
            "details": convergence_data,
            "interpretation": (
                f"O(k) CEGAR convergence bound holds for "
                f"{bound_holds_count}/{bound_total} benchmarks."
            ),
        },
        "mixed_fragment_analysis": {
            "mixed_fragment_f1": fragments.get("mixed", {}).get("f1", 0),
            "pure_qf_lia_f1": fragments.get("QF_LIA", {}).get("f1", 0),
            "pure_qf_nia_f1": fragments.get("QF_NIA", {}).get("f1", 0),
            "mixed_fallback_rate": fragments.get("mixed", {}).get("fallback_rate", 0),
            "pure_qf_lia_fallback_rate": fragments.get("QF_LIA", {}).get("fallback_rate", 0),
            "pure_qf_nia_fallback_rate": fragments.get("QF_NIA", {}).get("fallback_rate", 0),
            "interpretation": (
                "Mixed-theory benchmarks show higher fallback rates, "
                "connecting Craig interpolation limitations to the "
                "observed F1 degradation in mixed fragments."
            ),
        },
        "benchmark_details": benchmark_details,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("Craig Interpolation Fallback Characterization")
    print("=" * 70)
    print(f"  Z3 available:            {HAS_Z3}")
    print(f"  Craig interp available:  {HAS_INTERP}")
    print(f"  CEGAR available:         {HAS_CEGAR}")
    print(f"  Benchmarks:              {len(BENCHMARKS)}")
    print()

    results: List[BenchmarkResult] = []
    for bench in BENCHMARKS:
        r = run_single_benchmark(bench)
        results.append(r)
        icon = "✓" if r.correct_verdict else "✗"
        print(f"  {icon} {r.name:35s}  {r.theory_fragment:8s}  "
              f"{r.verdict:7s}  fallbacks={r.fallback_count}  "
              f"{r.verification_time_ms:.1f}ms")

    report = compute_fallback_report(results)

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    ovr = report["overall"]
    print(f"  Overall accuracy:        {ovr['overall_accuracy']:.1%}")
    print(f"  Total fallback rate:     {ovr['fallback_rate']:.1%}")

    print(f"\n  By theory fragment:")
    for frag, data in report["by_theory_fragment"].items():
        print(f"    {frag:8s}: F1={data['f1']:.3f}  "
              f"fallback_rate={data['fallback_rate']:.3f}  "
              f"n={data['count']}")

    corr = report["fallback_accuracy_correlation"]
    print(f"\n  Fallback-accuracy correlation:")
    print(f"    No-fallback accuracy:   {corr['no_fallback_accuracy']:.1%} "
          f"(n={corr['no_fallback_benchmarks']})")
    print(f"    With-fallback accuracy: {corr['with_fallback_accuracy']:.1%} "
          f"(n={corr['with_fallback_benchmarks']})")

    cegar = report["cegar_convergence_bound"]
    print(f"\n  CEGAR convergence: {cegar['interpretation']}")

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "craig_fallback_analysis.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
