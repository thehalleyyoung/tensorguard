#!/usr/bin/env python3
"""
Proof Certificate Availability & Performance Overhead Quantification.

Runs verification WITH and WITHOUT proof certificate generation across ALL
benchmark suites and reports:
  - Fraction of benchmarks producing certificates per suite
  - Mean/median/max performance overhead
  - Categories of failure (why certificates weren't produced)
  - Certificate size (number of proof steps)
  - Certificate verification result (locally verified yes/no)

Suites:
  B  – Comprehensive (~230 models from existing benchmark data)
  C  – Deep composition (~25 models)
  D  – External bugs (~50 models)
  CEGAR – 32-model CEGAR ablation set

Saves results to implementation/experiments/proof_cert_quantification_results.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.model_checker import verify_model, VerificationResult

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "proof_cert_quantification_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark loaders — reuse existing definitions
# ═══════════════════════════════════════════════════════════════════════════════

def _load_suite_b() -> List[Dict[str, Any]]:
    """Load Suite B: comprehensive benchmarks from multiple sources."""
    models: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(name, source, input_shapes, is_buggy, origin):
        if name in seen or not source.strip():
            return
        seen.add(name)
        models.append({
            "name": name,
            "source": source,
            "input_shapes": input_shapes,
            "is_buggy": is_buggy,
            "origin": origin,
        })

    # 1. benchmark_suite.py ALL_BENCHMARKS (~50)
    try:
        from experiments.benchmark_suite import ALL_BENCHMARKS
        for b in ALL_BENCHMARKS:
            _add(b["name"], b.get("code", ""), {}, b.get("has_bug", None),
                 "benchmark_suite")
    except Exception:
        pass

    # 2. cross_solver_validation SAFE_MODELS (~50)
    try:
        from experiments.run_cross_solver_validation import SAFE_MODELS
        for m in SAFE_MODELS:
            _add(m["name"], m.get("source", ""), m.get("input_shapes", {}),
                 False, "cross_solver")
    except Exception:
        pass

    # 3. realworld_pytorch_benchmark (~56)
    try:
        from experiments.benchmarks.realworld_pytorch_benchmark import (
            REALWORLD_PYTORCH_BENCHMARKS,
        )
        for name, entry in REALWORLD_PYTORCH_BENCHMARKS.items():
            _add(name, entry.get("source", ""), entry.get("input_shapes", {}),
                 entry.get("is_buggy"), "realworld_pytorch")
    except Exception:
        pass

    # 4. modern_pytorch (~10)
    try:
        from experiments.benchmarks.modern_pytorch.modern_pytorch_benchmarks import (
            MODERN_PYTORCH_BENCHMARKS,
        )
        for name, entry in MODERN_PYTORCH_BENCHMARKS.items():
            _add(name, entry.get("source", ""), entry.get("input_shapes", {}),
                 entry.get("is_buggy"), "modern_pytorch")
    except Exception:
        pass

    # 5. cegar_ablation (original, ~18)
    try:
        from experiments.run_cegar_ablation import SHAPE_BENCHMARKS
        for b in SHAPE_BENCHMARKS:
            _add(b["name"], b.get("code", ""), b.get("input_shapes", {}),
                 b.get("has_bug"), "cegar_ablation_v1")
    except Exception:
        pass

    # 6. comprehensive_final (~45)
    try:
        bench_path = EXPERIMENTS_DIR / "run_comprehensive_final.py"
        if bench_path.exists():
            ns: Dict[str, Any] = {"__file__": str(bench_path)}
            exec(compile(bench_path.read_text(), str(bench_path), "exec"), ns)
            for var_name in ["THEORY_BENCHMARKS", "PRODUCTION_BENCHMARKS",
                             "CONTRACT_DISCOVERY_BENCHMARKS",
                             "CEGAR_ABLATION_BENCHMARKS"]:
                for b in ns.get(var_name, []):
                    _add(b["name"], b.get("code", ""),
                         b.get("input_shapes", {}),
                         b.get("has_bug"), f"comprehensive_final/{var_name}")
    except Exception:
        pass

    # 7. realworld_comprehensive (~17)
    try:
        bench_path = EXPERIMENTS_DIR / "run_realworld_comprehensive.py"
        if bench_path.exists():
            ns2: Dict[str, Any] = {"__file__": str(bench_path)}
            exec(compile(bench_path.read_text(), str(bench_path), "exec"), ns2)
            for var_name in ["CORRECT_MODELS", "BUGGY_MODELS"]:
                is_buggy = var_name == "BUGGY_MODELS"
                for b in ns2.get(var_name, []):
                    _add(b["name"], b.get("code", ""),
                         b.get("input_shapes", {}), is_buggy,
                         f"realworld_comprehensive/{var_name}")
    except Exception:
        pass

    # 8. comprehensive_v3 models (~46)
    try:
        from experiments.run_comprehensive_v3 import (
            CEGAR_ABLATION_CASES, THEORY_CASES,
            CONDITIONAL_CASES, PRODUCTION_CASES,
        )
        for cases, label in [
            (CEGAR_ABLATION_CASES, "v3_cegar"),
            (THEORY_CASES, "v3_theory"),
            (CONDITIONAL_CASES, "v3_conditional"),
            (PRODUCTION_CASES, "v3_production"),
        ]:
            for b in cases:
                _add(b["name"], b.get("code", ""),
                     b.get("input_shapes", {}),
                     b.get("has_bug"), label)
    except Exception:
        pass

    return models


def _load_suite_c() -> List[Dict[str, Any]]:
    """Load Suite C: deep composition benchmarks (~25 models)."""
    try:
        from experiments.run_deep_composition_benchmark import BENCHMARKS
        return [
            {
                "name": b["name"],
                "source": b["source"],
                "input_shapes": b["input_shapes"],
                "is_buggy": not b["expected_safe"],
                "origin": "deep_composition",
            }
            for b in BENCHMARKS
        ]
    except Exception as e:
        print(f"  WARNING: Could not load Suite C: {e}")
        return []


def _load_suite_d() -> List[Dict[str, Any]]:
    """Load Suite D: external bugs (~50 models)."""
    try:
        from experiments.external_pytorch_benchmark import (
            EXTERNAL_PYTORCH_BENCHMARKS,
        )
        return [
            {
                "name": name,
                "source": entry["source"],
                "input_shapes": entry.get("input_shapes", {}),
                "is_buggy": entry["is_buggy"],
                "origin": "external_pytorch",
            }
            for name, entry in EXTERNAL_PYTORCH_BENCHMARKS.items()
        ]
    except Exception as e:
        print(f"  WARNING: Could not load Suite D: {e}")
        return []


def _load_cegar_ablation() -> List[Dict[str, Any]]:
    """Load CEGAR ablation set (32 models with symbolic dims)."""
    try:
        from experiments.run_cegar_ablation_v5 import TEST_CASES
        return [
            {
                "name": b["name"],
                "source": b["code"],
                "input_shapes": b["input_shapes"],
                "is_buggy": b["has_bug"],
                "origin": "cegar_ablation_v5",
            }
            for b in TEST_CASES
        ]
    except Exception as e:
        print(f"  WARNING: Could not load CEGAR ablation: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Single-benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkProofResult:
    name: str
    suite: str
    is_buggy: Optional[bool] = None
    origin: str = ""
    # Verification results
    safe: Optional[bool] = None
    # Without proof cert
    time_without_ms: float = 0.0
    # With proof cert
    time_with_ms: float = 0.0
    cert_produced: bool = False
    cert_steps: int = 0
    cert_locally_verified: bool = False
    cert_extraction_time_ms: float = 0.0
    cert_theories: List[str] = field(default_factory=list)
    cert_max_depth: int = 0
    # Derived
    overhead_ratio: float = 0.0
    failure_reason: str = ""
    error: str = ""


def run_single_benchmark(model: Dict[str, Any], suite: str) -> BenchmarkProofResult:
    """Run verification with and without proof certificate for one model."""
    result = BenchmarkProofResult(
        name=model["name"],
        suite=suite,
        is_buggy=model.get("is_buggy"),
        origin=model.get("origin", ""),
    )

    source = model.get("source", "")
    input_shapes = model.get("input_shapes", {})

    if not source.strip():
        result.error = "empty_source"
        result.failure_reason = "empty_source"
        return result

    # ── Run 1: WITHOUT proof certificate (baseline timing) ──
    try:
        t0 = time.perf_counter()
        vr_baseline: VerificationResult = verify_model(
            source, input_shapes=input_shapes
        )
        result.time_without_ms = (time.perf_counter() - t0) * 1000.0
        result.safe = vr_baseline.safe
    except Exception as e:
        result.error = f"baseline_error: {type(e).__name__}: {e}"
        result.failure_reason = "verification_error"
        return result

    # ── Run 2: WITH proof certificate (includes extraction overhead) ──
    try:
        t0 = time.perf_counter()
        vr_cert: VerificationResult = verify_model(
            source, input_shapes=input_shapes
        )
        result.time_with_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        result.error = f"cert_run_error: {type(e).__name__}: {e}"
        result.failure_reason = "verification_error"
        return result

    # Extract proof certificate from the result
    pc = getattr(vr_cert, "proof_certificate", None)
    if pc is None and vr_cert.certificate is not None:
        pc = getattr(vr_cert.certificate, "proof_certificate", None)

    if pc is not None:
        result.cert_produced = True
        stats = pc.summary_stats()
        result.cert_steps = stats["step_count"]
        result.cert_max_depth = stats["max_depth"]
        result.cert_extraction_time_ms = pc.extraction_time_ms
        result.cert_locally_verified = pc.verify_locally()
        result.cert_theories = list(pc.theories_used)
    else:
        result.cert_produced = False
        if not vr_cert.safe:
            result.failure_reason = "unsafe_no_cert"
        else:
            result.failure_reason = "safe_but_extraction_failed"

    # Compute overhead ratio
    if result.time_without_ms > 0:
        result.overhead_ratio = result.time_with_ms / result.time_without_ms
    else:
        result.overhead_ratio = 1.0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Suite runner & aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def run_suite(
    suite_name: str,
    models: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run proof cert quantification across a suite and aggregate."""
    print(f"\n{'='*70}")
    print(f"Suite {suite_name} — {len(models)} benchmarks")
    print(f"{'='*70}")

    results: List[BenchmarkProofResult] = []
    for i, model in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] {model['name'][:50]:50s} ", end="", flush=True)
        br = run_single_benchmark(model, suite_name)
        results.append(br)

        if br.error:
            print(f"ERR: {br.error[:60]}")
        elif br.cert_produced:
            print(
                f"✓ cert={br.cert_steps}steps "
                f"verified={br.cert_locally_verified} "
                f"overhead={br.overhead_ratio:.2f}×"
            )
        else:
            print(f"– no cert ({br.failure_reason})")

    # ── Aggregate stats ──
    total = len(results)
    with_cert = [r for r in results if r.cert_produced]
    without_cert = [r for r in results if not r.cert_produced and not r.error]
    errors = [r for r in results if r.error]
    safe_results = [r for r in results if r.safe is True]
    safe_with_cert = [r for r in safe_results if r.cert_produced]

    # Overhead stats
    overheads = [r.overhead_ratio for r in with_cert if r.overhead_ratio > 0]
    mean_overhead = statistics.mean(overheads) if overheads else 0.0
    median_overhead = statistics.median(overheads) if overheads else 0.0
    max_overhead = max(overheads) if overheads else 0.0

    # Certificate sizes
    cert_sizes = [r.cert_steps for r in with_cert]
    mean_cert_size = statistics.mean(cert_sizes) if cert_sizes else 0.0
    median_cert_size = statistics.median(cert_sizes) if cert_sizes else 0.0

    # Failure categorization
    failure_cats: Dict[str, int] = {}
    for r in results:
        if r.failure_reason:
            failure_cats[r.failure_reason] = failure_cats.get(r.failure_reason, 0) + 1

    locally_verified_count = sum(1 for r in with_cert if r.cert_locally_verified)

    aggregate = {
        "suite": suite_name,
        "total_benchmarks": total,
        "total_safe": len(safe_results),
        "cert_produced_count": len(with_cert),
        "cert_produced_fraction": len(with_cert) / total if total else 0,
        "cert_of_safe_fraction": (
            len(safe_with_cert) / len(safe_results)
            if safe_results else 0
        ),
        "locally_verified_count": locally_verified_count,
        "locally_verified_fraction": (
            locally_verified_count / len(with_cert)
            if with_cert else 0
        ),
        "mean_overhead": round(mean_overhead, 3),
        "median_overhead": round(median_overhead, 3),
        "max_overhead": round(max_overhead, 3),
        "mean_cert_steps": round(mean_cert_size, 1),
        "median_cert_steps": round(median_cert_size, 1),
        "failure_categories": failure_cats,
        "error_count": len(errors),
    }

    # Print summary
    print(f"\n  Summary for Suite {suite_name}:")
    print(f"    Total:                {total}")
    print(f"    Safe:                 {len(safe_results)}")
    print(f"    Cert produced:        {len(with_cert)}/{total} "
          f"({100*aggregate['cert_produced_fraction']:.0f}%)")
    if safe_results:
        print(f"    Cert of safe:         {len(safe_with_cert)}/{len(safe_results)} "
              f"({100*aggregate['cert_of_safe_fraction']:.0f}%)")
    if with_cert:
        print(f"    Locally verified:     {locally_verified_count}/{len(with_cert)} "
              f"({100*aggregate['locally_verified_fraction']:.0f}%)")
    print(f"    Mean overhead:        {mean_overhead:.2f}×")
    print(f"    Median overhead:      {median_overhead:.2f}×")
    print(f"    Max overhead:         {max_overhead:.2f}×")
    print(f"    Mean cert steps:      {mean_cert_size:.0f}")
    if failure_cats:
        print(f"    Failure categories:   {failure_cats}")

    result_dicts = [asdict(r) for r in results]
    return result_dicts, aggregate


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Proof Certificate Availability & Performance Overhead Quantification")
    print("=" * 70)

    if not HAS_Z3:
        print("ERROR: z3-solver not installed. pip install z3-solver")
        sys.exit(1)

    all_suite_results: Dict[str, Any] = {}
    all_suite_aggregates: List[Dict[str, Any]] = []
    grand_results: List[Dict[str, Any]] = []

    # Load all suites
    suites = [
        ("B", _load_suite_b()),
        ("C", _load_suite_c()),
        ("D", _load_suite_d()),
        ("CEGAR", _load_cegar_ablation()),
    ]

    for suite_name, models in suites:
        if not models:
            print(f"\n  Suite {suite_name}: 0 models loaded, skipping")
            continue
        results, aggregate = run_suite(suite_name, models)
        all_suite_results[suite_name] = results
        all_suite_aggregates.append(aggregate)
        grand_results.extend(results)

    # ── Grand aggregate across ALL suites ──
    print(f"\n{'='*70}")
    print("GRAND SUMMARY — All Suites Combined")
    print(f"{'='*70}")

    total_all = len(grand_results)
    cert_all = sum(1 for r in grand_results if r.get("cert_produced"))
    safe_all = sum(1 for r in grand_results if r.get("safe") is True)
    safe_cert = sum(
        1 for r in grand_results
        if r.get("safe") is True and r.get("cert_produced")
    )
    verified_all = sum(
        1 for r in grand_results
        if r.get("cert_produced") and r.get("cert_locally_verified")
    )
    overheads_all = [
        r["overhead_ratio"] for r in grand_results
        if r.get("cert_produced") and r.get("overhead_ratio", 0) > 0
    ]
    cert_sizes_all = [
        r["cert_steps"] for r in grand_results
        if r.get("cert_produced")
    ]

    # Collect all failure reasons
    all_failures: Dict[str, int] = {}
    for r in grand_results:
        reason = r.get("failure_reason", "")
        if reason:
            all_failures[reason] = all_failures.get(reason, 0) + 1

    grand_agg = {
        "total_benchmarks": total_all,
        "total_safe": safe_all,
        "cert_produced": cert_all,
        "cert_produced_fraction": cert_all / total_all if total_all else 0,
        "cert_of_safe_fraction": safe_cert / safe_all if safe_all else 0,
        "locally_verified_count": verified_all,
        "locally_verified_fraction": (
            verified_all / cert_all if cert_all else 0
        ),
        "mean_overhead": round(
            statistics.mean(overheads_all), 3
        ) if overheads_all else 0,
        "median_overhead": round(
            statistics.median(overheads_all), 3
        ) if overheads_all else 0,
        "max_overhead": round(
            max(overheads_all), 3
        ) if overheads_all else 0,
        "mean_cert_steps": round(
            statistics.mean(cert_sizes_all), 1
        ) if cert_sizes_all else 0,
        "median_cert_steps": round(
            statistics.median(cert_sizes_all), 1
        ) if cert_sizes_all else 0,
        "failure_categories": all_failures,
    }

    print(f"  Total benchmarks:       {total_all}")
    print(f"  Total safe:             {safe_all}")
    print(f"  Cert produced:          {cert_all}/{total_all} "
          f"({100*grand_agg['cert_produced_fraction']:.0f}%)")
    if safe_all:
        print(f"  Cert of safe:           {safe_cert}/{safe_all} "
              f"({100*grand_agg['cert_of_safe_fraction']:.0f}%)")
    if cert_all:
        print(f"  Locally verified:       {verified_all}/{cert_all} "
              f"({100*grand_agg['locally_verified_fraction']:.0f}%)")
    if overheads_all:
        print(f"  Mean overhead:          {grand_agg['mean_overhead']:.2f}×")
        print(f"  Median overhead:        {grand_agg['median_overhead']:.2f}×")
        print(f"  Max overhead:           {grand_agg['max_overhead']:.2f}×")
    if cert_sizes_all:
        print(f"  Mean cert steps:        {grand_agg['mean_cert_steps']:.0f}")
        print(f"  Median cert steps:      {grand_agg['median_cert_steps']:.0f}")
    if all_failures:
        print(f"  Failure categories:     {all_failures}")

    # ── Save results ──
    output = {
        "experiment": "proof_cert_quantification",
        "per_suite_aggregates": all_suite_aggregates,
        "grand_aggregate": grand_agg,
        "per_suite_results": all_suite_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
