"""
Comprehensive Scalability Analysis for TensorGuard.

Measures verification time, memory usage, Z3 query count, and state-space size
across models of increasing depth (2, 5, 10, 20, 50, 100 layers) and increasing
branching factor. Exercises both bounded model checking and IC3/PDR, and tracks
UserPropagator callback frequency.

Results are saved to experiments/scalability_comprehensive_results.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    verify_model,
    extract_computation_graph,
    extract_kripke_structure,
    Device,
    Phase,
)
from src.bmc_baseline import verify_model_bmc
from src.ic3_pdr import ic3_verify

RESULTS_FILE = Path(__file__).parent / "scalability_comprehensive_results.json"

DEPTHS = [2, 5, 10, 20, 50, 100]
BRANCHING_FACTORS = [2, 4, 8, 16]
NUM_TRIALS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Model generators
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mlp(depth: int, hidden: int = 256) -> str:
    layers = []
    fwd = []
    for i in range(depth):
        layers.append(f"        self.fc{i} = nn.Linear({hidden}, {hidden})")
        fwd.append(f"        x = self.fc{i}(x)")
    return (
        "import torch.nn as nn\n"
        f"class MLP(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(layers) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd) + "\n"
        "        return x\n"
    )


def generate_cnn(depth: int, channels: int = 64) -> str:
    layers = []
    fwd = []
    for i in range(depth):
        layers.append(f"        self.conv{i} = nn.Conv2d({channels}, {channels}, 3, padding=1)")
        layers.append(f"        self.bn{i} = nn.BatchNorm2d({channels})")
        fwd.append(f"        x = self.bn{i}(self.conv{i}(x))")
    return (
        "import torch.nn as nn\n"
        f"class CNN(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(layers) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd) + "\n"
        "        return x\n"
    )


def generate_branching(branches: int, depth_per_branch: int = 5) -> str:
    layers = []
    fwd = []
    for b in range(branches):
        for d in range(depth_per_branch):
            layers.append(f"        self.b{b}_fc{d} = nn.Linear(256, 256)")
        branch_lines = [f"        x{b} = x"]
        for d in range(depth_per_branch):
            branch_lines.append(f"        x{b} = self.b{b}_fc{d}(x{b})")
        fwd.extend(branch_lines)
    sum_expr = " + ".join(f"x{b}" for b in range(branches))
    return (
        "import torch\nimport torch.nn as nn\n"
        "class BranchModel(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(layers) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd) + "\n"
        f"        return {sum_expr}\n"
    )


def generate_buggy_mlp(depth: int, hidden: int = 256) -> str:
    layers = []
    fwd = []
    for i in range(depth):
        layers.append(f"        self.fc{i} = nn.Linear({hidden}, {hidden})")
        fwd.append(f"        x = self.fc{i}(x)")
    layers.append(f"        self.head = nn.Linear({hidden + 128}, 10)")
    fwd.append("        x = self.head(x)")
    return (
        "import torch.nn as nn\n"
        f"class BuggyMLP(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(layers) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd) + "\n"
        "        return x\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    else:
        std = 0.0
    cv = std / mean if mean > 0 else 0.0
    t_crit = {2: 4.303, 3: 3.182, 5: 2.776, 10: 2.262}.get(n, 2.0)
    ci_half = t_crit * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "cv": round(cv, 4),
        "ci_95_lower": round(mean - ci_half, 2),
        "ci_95_upper": round(mean + ci_half, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def measure_kripke(source: str, input_shapes: dict) -> Dict[str, Any]:
    """Extract Kripke structure and measure state-space size."""
    try:
        graph = extract_computation_graph(source)
        ks = extract_kripke_structure(
            graph,
            input_shapes=input_shapes,
            initial_device=Device.CPU,
            initial_phase=Phase.EVAL,
        )
        return {
            "num_states": ks.num_states,
            "num_transitions": ks.num_transitions,
            "is_safe": ks.is_safe(),
            "num_layers": len(graph.layers),
            "num_steps": graph.num_steps,
        }
    except Exception as e:
        return {"error": str(e)}


def measure_bmc(source: str, input_shapes: dict) -> Dict[str, Any]:
    """Run bounded model checking and collect metrics."""
    tracemalloc.start()
    t0 = time.monotonic()
    try:
        result = verify_model(source, input_shapes=input_shapes, return_kripke=True)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        z3_queries = 0
        if hasattr(result, 'certificate') and result.certificate:
            z3_queries = getattr(result.certificate, 'z3_queries', 0)

        kripke_info = {}
        if result.kripke_structure:
            kripke_info = {
                "num_states": result.kripke_structure.num_states,
                "num_transitions": result.kripke_structure.num_transitions,
            }

        return {
            "safe": result.safe,
            "time_ms": round(elapsed_ms, 2),
            "peak_memory_kb": round(peak_kb / 1024, 2),
            "z3_queries": z3_queries,
            "num_errors": len(result.errors) if result.errors else 0,
            "confidence": result.confidence.value if hasattr(result.confidence, 'value') else str(result.confidence),
            "kripke": kripke_info,
        }
    except Exception as e:
        tracemalloc.stop()
        return {"safe": None, "error": str(e), "time_ms": round((time.monotonic() - t0) * 1000, 2)}


def measure_bmc_baseline(source: str, input_shapes: dict) -> Dict[str, Any]:
    """Run monolithic BMC baseline and collect metrics."""
    tracemalloc.start()
    t0 = time.monotonic()
    try:
        result = verify_model_bmc(source, input_shapes=input_shapes, timeout=30)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "verdict": result.verdict.name,
            "safe": result.safe,
            "time_ms": round(elapsed_ms, 2),
            "peak_memory_kb": round(peak_kb / 1024, 2),
            "num_constraints": result.num_constraints,
            "num_steps": result.num_steps,
            "z3_queries": result.z3_queries,
        }
    except Exception as e:
        tracemalloc.stop()
        return {"safe": None, "error": str(e), "time_ms": round((time.monotonic() - t0) * 1000, 2)}


def measure_ic3(source: str, input_shapes: dict) -> Dict[str, Any]:
    """Run IC3/PDR and collect metrics."""
    tracemalloc.start()
    t0 = time.monotonic()
    try:
        result = ic3_verify(
            source,
            input_shapes=input_shapes,
            max_frames=50,
            solver_timeout_ms=5000,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "safe": result.safe,
            "time_ms": round(elapsed_ms, 2),
            "peak_memory_kb": round(peak_kb / 1024, 2),
            "frames_computed": result.frames_computed,
            "z3_queries": result.z3_queries,
            "num_blocked_cubes": result.num_blocked_cubes,
            "invariant_clauses": len(result.invariant_clauses),
        }
    except Exception as e:
        tracemalloc.stop()
        return {"safe": None, "error": str(e), "time_ms": round((time.monotonic() - t0) * 1000, 2)}


def measure_propagator_callbacks(source: str, input_shapes: dict) -> Dict[str, Any]:
    """Measure UserPropagator callback frequency during verification."""
    try:
        from src.model_checker import ConstraintVerifier, extract_computation_graph
        graph = extract_computation_graph(source)
        verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
        result = verifier.verify()
        # Stats live on the _Z3Context (verifier.ctx)
        stats = verifier.ctx.get_stats()
        return {
            "z3_queries": stats.get("z3_queries", 0),
            "broadcast_propagations": stats.get("broadcast_propagations", 0),
            "broadcast_conflicts": stats.get("broadcast_conflicts", 0),
            "stride_constraints": stats.get("stride_constraints", 0),
            "stride_reshapes": stats.get("stride_reshapes", 0),
            "device_same_pairs": stats.get("device_same_pairs", 0),
            "device_transfer_triples": stats.get("device_transfer_triples", 0),
            "phase_dropout_constraints": stats.get("phase_dropout_constraints", 0),
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_comprehensive_scalability():
    results = {
        "experiment": "comprehensive_scalability_analysis",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_trials": NUM_TRIALS,
        "depth_scaling": [],
        "branching_scaling": [],
        "buggy_depth_scaling": [],
        "propagator_callback_scaling": [],
        "summary": {},
    }

    mlp_shapes = {"x": ("batch", 256)}
    cnn_shapes = {"x": ("batch", 64, 32, 32)}

    # ─── Depth scaling (MLP + CNN) ────────────────────────────────────────
    print("=" * 70)
    print("1. Depth Scaling Analysis (BMC + IC3/PDR)")
    print("=" * 70)

    for depth in DEPTHS:
        print(f"\n  depth={depth}")
        source_mlp = generate_mlp(depth)

        # Multi-trial BMC
        bmc_times = []
        bmc_result = None
        for _ in range(NUM_TRIALS):
            r = measure_bmc(source_mlp, mlp_shapes)
            bmc_times.append(r.get("time_ms", 0))
            bmc_result = r

        # Kripke structure
        kripke = measure_kripke(source_mlp, mlp_shapes)

        # IC3/PDR (single run — typically slower)
        ic3_result = measure_ic3(source_mlp, mlp_shapes)

        # BMC baseline
        bmc_baseline_result = measure_bmc_baseline(source_mlp, mlp_shapes)

        entry = {
            "depth": depth,
            "model_type": "mlp",
            "kripke": kripke,
            "bmc": {
                "timing_stats": compute_stats(bmc_times),
                "last_run": bmc_result,
            },
            "bmc_baseline": bmc_baseline_result,
            "ic3_pdr": ic3_result,
        }
        results["depth_scaling"].append(entry)

        bmc_mean = compute_stats(bmc_times)["mean"]
        ic3_t = ic3_result.get("time_ms", "N/A")
        states = kripke.get("num_states", "?")
        print(f"    BMC: {bmc_mean:.1f}ms (mean)  IC3: {ic3_t}ms  "
              f"States: {states}  Safe(bmc): {bmc_result.get('safe')}")

    # ─── CNN depth scaling (skip 100 layers — too slow) ────────────────────
    print("\n" + "=" * 70)
    print("2. CNN Depth Scaling (BMC only)")
    print("=" * 70)

    for depth in [d for d in DEPTHS if d <= 50]:
        print(f"\n  cnn depth={depth}")
        source_cnn = generate_cnn(depth)
        bmc_times = []
        bmc_result = None
        for _ in range(NUM_TRIALS):
            r = measure_bmc(source_cnn, cnn_shapes)
            bmc_times.append(r.get("time_ms", 0))
            bmc_result = r

        kripke = measure_kripke(source_cnn, cnn_shapes)
        entry = {
            "depth": depth,
            "model_type": "cnn",
            "kripke": kripke,
            "bmc": {
                "timing_stats": compute_stats(bmc_times),
                "last_run": bmc_result,
            },
        }
        results["depth_scaling"].append(entry)
        bmc_mean = compute_stats(bmc_times)["mean"]
        print(f"    BMC: {bmc_mean:.1f}ms (mean)  States: {kripke.get('num_states', '?')}")

    # ─── Branching scaling ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. Branching Factor Scaling")
    print("=" * 70)

    for branches in BRANCHING_FACTORS:
        print(f"\n  branches={branches}")
        source = generate_branching(branches)
        total_layers = branches * 5

        bmc_times = []
        bmc_result = None
        for _ in range(NUM_TRIALS):
            r = measure_bmc(source, mlp_shapes)
            bmc_times.append(r.get("time_ms", 0))
            bmc_result = r

        kripke = measure_kripke(source, mlp_shapes)
        ic3_result = measure_ic3(source, mlp_shapes)

        entry = {
            "branches": branches,
            "depth_per_branch": 5,
            "total_layers": total_layers,
            "kripke": kripke,
            "bmc": {
                "timing_stats": compute_stats(bmc_times),
                "last_run": bmc_result,
            },
            "ic3_pdr": ic3_result,
        }
        results["branching_scaling"].append(entry)

        bmc_mean = compute_stats(bmc_times)["mean"]
        print(f"    BMC: {bmc_mean:.1f}ms (mean)  IC3: {ic3_result.get('time_ms', 'N/A')}ms  "
              f"States: {kripke.get('num_states', '?')}")

    # ─── Buggy model depth scaling ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. Buggy Model Detection Scaling")
    print("=" * 70)

    for depth in DEPTHS:
        print(f"\n  buggy depth={depth}")
        source = generate_buggy_mlp(depth)

        bmc_times = []
        bmc_result = None
        for _ in range(NUM_TRIALS):
            r = measure_bmc(source, mlp_shapes)
            bmc_times.append(r.get("time_ms", 0))
            bmc_result = r

        ic3_result = measure_ic3(source, mlp_shapes)

        entry = {
            "depth": depth,
            "bmc": {
                "timing_stats": compute_stats(bmc_times),
                "detected_bug": not bmc_result.get("safe", True) if bmc_result else False,
            },
            "ic3_pdr": {
                "time_ms": ic3_result.get("time_ms"),
                "detected_bug": not ic3_result.get("safe", True),
                "z3_queries": ic3_result.get("z3_queries", 0),
            },
        }
        results["buggy_depth_scaling"].append(entry)
        bmc_mean = compute_stats(bmc_times)["mean"]
        print(f"    BMC: {bmc_mean:.1f}ms  detected={entry['bmc']['detected_bug']}  "
              f"IC3: {ic3_result.get('time_ms', 'N/A')}ms  detected={entry['ic3_pdr']['detected_bug']}")

    # ─── UserPropagator callback frequency ─────────────────────────────────
    print("\n" + "=" * 70)
    print("5. UserPropagator Callback Frequency Scaling")
    print("=" * 70)

    for depth in DEPTHS:
        print(f"\n  depth={depth}")
        source = generate_mlp(depth)
        cb_stats = measure_propagator_callbacks(source, mlp_shapes)
        entry = {"depth": depth, "callbacks": cb_stats}
        results["propagator_callback_scaling"].append(entry)
        print(f"    z3_queries={cb_stats.get('z3_queries', '?')}  "
              f"broadcast_props={cb_stats.get('broadcast_propagations', '?')}  "
              f"device_pairs={cb_stats.get('device_same_pairs', '?')}")

    # ─── Summary ───────────────────────────────────────────────────────────
    mlp_entries = [e for e in results["depth_scaling"] if e.get("model_type") == "mlp"]
    if mlp_entries:
        max_bmc = max(e["bmc"]["timing_stats"]["mean"] for e in mlp_entries)
        ic3_entries = [e for e in mlp_entries if "ic3_pdr" in e and "time_ms" in e.get("ic3_pdr", {})]
        max_ic3 = max((e["ic3_pdr"]["time_ms"] for e in ic3_entries), default=0)
    else:
        max_bmc = max_ic3 = 0

    branching_entries = results["branching_scaling"]
    if branching_entries:
        max_branch_bmc = max(e["bmc"]["timing_stats"]["mean"] for e in branching_entries)
    else:
        max_branch_bmc = 0

    results["summary"] = {
        "depths_tested": DEPTHS,
        "branching_factors_tested": BRANCHING_FACTORS,
        "max_mlp_bmc_time_ms": round(max_bmc, 2),
        "max_mlp_ic3_time_ms": round(max_ic3, 2),
        "max_branch_bmc_time_ms": round(max_branch_bmc, 2),
        "all_buggy_detected_bmc": all(
            e["bmc"]["detected_bug"] for e in results["buggy_depth_scaling"]
        ),
        "all_buggy_detected_ic3": all(
            e["ic3_pdr"]["detected_bug"] for e in results["buggy_depth_scaling"]
        ),
    }

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")

    return results


if __name__ == "__main__":
    run_comprehensive_scalability()
