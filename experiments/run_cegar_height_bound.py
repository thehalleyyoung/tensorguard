#!/usr/bin/env python3
"""CEGAR convergence height bound analysis.

Computes the explicit predicate-universe bound |P_prog| for each
benchmark in the CEGAR ablation suite and compares it against the
actual number of iterations observed.  The tightness ratio
(actual / theoretical) quantifies how conservative the bound is.

Addresses the critique: "CEGAR convergence depends on implicit lattice
finiteness without explicit height bound — Worst-case iteration
complexity unknown without explicit |P_prog| analysis."
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Ensure the implementation package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import compute_predicate_universe_bound

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture → (num_layers, max_dims_per_layer) mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Per-benchmark model info derived from the source code in
# run_cegar_ablation_v5.py.  ``num_layers`` counts parameterised layers
# (Linear / Conv2d) and ``max_dims_per_layer`` is the maximum number of
# shape dimensions constrained per layer (2 for Linear [in, out],
# 4 for Conv2d [in_ch, out_ch, kH, kW]).

_LAYER_PATTERN = re.compile(
    r"nn\.(Linear|Conv[12]d|ConvTranspose[12]d)\s*\(", re.MULTILINE
)


def _count_layers(code: str) -> int:
    """Count parameterised layers (nn.Linear / nn.Conv*d) in source."""
    return len(_LAYER_PATTERN.findall(code))


def _max_dims_for_code(code: str) -> int:
    """Heuristic: 4 if any Conv layer is present, else 2 (Linear)."""
    if "Conv" in code:
        return 4
    return 2


# ═══════════════════════════════════════════════════════════════════════════════
# Load benchmark definitions from run_cegar_ablation_v5.py
# ═══════════════════════════════════════════════════════════════════════════════

def _load_benchmark_source() -> Dict[str, str]:
    """Return {benchmark_name: source_code} from run_cegar_ablation_v5.py."""
    ablation_path = os.path.join(os.path.dirname(__file__), "run_cegar_ablation_v5.py")
    # Import the TEST_CASES list
    import importlib.util
    spec = importlib.util.spec_from_file_location("ablation_v5", ablation_path)
    mod = importlib.util.module_from_spec(spec)
    # We need torch.nn stubs — the module defines classes but doesn't run them
    spec.loader.exec_module(mod)
    return {tc["name"]: tc["code"] for tc in mod.TEST_CASES}


# ═══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    results_path = os.path.join(os.path.dirname(__file__), "cegar_ablation_v5_results.json")
    with open(results_path) as f:
        ablation = json.load(f)

    # Use the cegar_filtered config (the primary CEGAR mode)
    per_benchmark: List[Dict[str, Any]] = ablation["configs"]["cegar_filtered"]["per_benchmark"]

    # Load source code per benchmark
    benchmark_src = _load_benchmark_source()

    analysis: List[Dict[str, Any]] = []

    for bm in per_benchmark:
        name = bm["name"]
        actual_iters = bm["iterations"]
        code = benchmark_src.get(name, "")

        num_layers = _count_layers(code)
        max_dims = _max_dims_for_code(code)

        bound_info = compute_predicate_universe_bound({
            "num_layers": num_layers,
            "max_dims_per_layer": max_dims,
        })

        theoretical_bound = bound_info["bound"]
        tightness = (
            round(actual_iters / theoretical_bound, 4)
            if theoretical_bound > 0 else 0.0
        )

        analysis.append({
            "name": name,
            "arch": bm["arch"],
            "has_bug": bm["has_bug"],
            "status": bm["status"],
            "actual_iterations": actual_iters,
            "num_layers": num_layers,
            "max_dims_per_layer": max_dims,
            "predicate_kinds": bound_info["predicate_kinds"],
            "theoretical_bound": theoretical_bound,
            "formula": bound_info["formula"],
            "tightness_ratio": tightness,
        })

    # ── Aggregate statistics ──────────────────────────────────────────────
    ratios = [a["tightness_ratio"] for a in analysis]
    bounds = [a["theoretical_bound"] for a in analysis]
    iters = [a["actual_iterations"] for a in analysis]

    summary = {
        "description": (
            "Explicit CEGAR height bound analysis. The predicate universe "
            "P_prog is bounded by |layers| × |dims_per_layer| × |predicate_kinds|. "
            "Each CEGAR iteration adds ≥1 new predicate (strict monotonicity), "
            "so convergence is guaranteed in ≤ |P_prog| iterations."
        ),
        "num_benchmarks": len(analysis),
        "mean_theoretical_bound": round(sum(bounds) / len(bounds), 2),
        "max_theoretical_bound": max(bounds),
        "mean_actual_iterations": round(sum(iters) / len(iters), 2),
        "max_actual_iterations": max(iters),
        "mean_tightness_ratio": round(sum(ratios) / len(ratios), 4),
        "max_tightness_ratio": max(ratios),
        "min_tightness_ratio": min(ratios),
        "all_within_bound": all(
            a["actual_iterations"] <= a["theoretical_bound"]
            for a in analysis
        ),
        "convergence_theorem": (
            "For any nn.Module with L layers, D max dims/layer, and "
            "K=7 predicate kinds, the CEGAR loop terminates in at most "
            "L × D × K iterations.  Proof: the accumulated predicate set "
            "P grows strictly monotonically (|P_{i+1}| > |P_i|) and "
            "P ⊆ P_prog where |P_prog| = L × D × K is finite."
        ),
    }

    output = {
        "experiment": "cegar_height_bound_analysis",
        "summary": summary,
        "per_benchmark": analysis,
    }

    out_path = os.path.join(os.path.dirname(__file__), "cegar_height_bound_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Console report ────────────────────────────────────────────────────
    print("=" * 72)
    print("CEGAR Height Bound Analysis")
    print("=" * 72)
    print(f"  Benchmarks analysed:      {summary['num_benchmarks']}")
    print(f"  Mean theoretical bound:   {summary['mean_theoretical_bound']}")
    print(f"  Max theoretical bound:    {summary['max_theoretical_bound']}")
    print(f"  Mean actual iterations:   {summary['mean_actual_iterations']}")
    print(f"  Max actual iterations:    {summary['max_actual_iterations']}")
    print(f"  Mean tightness ratio:     {summary['mean_tightness_ratio']}")
    print(f"  All within bound:         {summary['all_within_bound']}")
    print("-" * 72)
    print(f"  {'Benchmark':<30} {'Bound':>6} {'Actual':>7} {'Ratio':>8}")
    print("-" * 72)
    for a in analysis:
        print(
            f"  {a['name']:<30} {a['theoretical_bound']:>6} "
            f"{a['actual_iterations']:>7} {a['tightness_ratio']:>8.4f}"
        )
    print("=" * 72)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
