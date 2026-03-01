#!/usr/bin/env python3
"""
CVC5 Alethe Proof Coverage Evaluation for TensorGuard.

Runs benchmark models through both Z3 and CVC5 backends, measures proof
certificate extraction success rate for each, and saves comparative results
to .benchmarks/cvc5_proof_coverage.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    import cvc5
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

from src.proof_certificate import (
    ProofCertificate,
    ProofExtractor,
    get_proof_status,
)
from src.smt.solver import (
    SatResult,
    Comparison,
    ComparisonOp,
    Var,
    Const,
    BinOp,
    ArithOp,
    And,
    Not,
    BoolLit,
    Sort,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = EXPERIMENTS_DIR / ".benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = BENCHMARKS_DIR / "cvc5_proof_coverage.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark constraint sets (shape verification scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_CONSTRAINTS: List[Dict[str, Any]] = [
    {
        "name": "SimpleMLP_shape",
        "description": "Linear layer input shape must match weight dimensions",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("batch"), Const(32)),
            Comparison(ComparisonOp.EQ, Var("in_features"), Const(784)),
            Comparison(ComparisonOp.EQ, Var("out_features"), Const(128)),
            Comparison(ComparisonOp.GT, Var("batch"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.GT, Var("in_features"), Const(0))),
    },
    {
        "name": "MatMul_broadcast",
        "description": "Matrix multiply dimension compatibility",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")),
            Comparison(ComparisonOp.GT, Var("a_rows"), Const(0)),
            Comparison(ComparisonOp.GT, Var("a_cols"), Const(0)),
            Comparison(ComparisonOp.GT, Var("b_cols"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows"))),
    },
    {
        "name": "Conv2d_shape",
        "description": "Conv2d output spatial dimension computation",
        "constraints": [
            Comparison(ComparisonOp.GT, Var("H"), Const(0)),
            Comparison(ComparisonOp.GT, Var("W"), Const(0)),
            Comparison(ComparisonOp.GT, Var("kernel"), Const(0)),
            Comparison(ComparisonOp.GE, Var("H"), Var("kernel")),
        ],
        "negation": Not(Comparison(ComparisonOp.GE, Var("H"), Var("kernel"))),
    },
    {
        "name": "Reshape_product",
        "description": "Reshape preserves total element count",
        "constraints": [
            Comparison(
                ComparisonOp.EQ,
                BinOp(ArithOp.MUL, Var("d1"), Var("d2")),
                BinOp(ArithOp.MUL, Var("d3"), Var("d4")),
            ),
            Comparison(ComparisonOp.GT, Var("d1"), Const(0)),
            Comparison(ComparisonOp.GT, Var("d2"), Const(0)),
        ],
        "negation": Not(
            Comparison(
                ComparisonOp.EQ,
                BinOp(ArithOp.MUL, Var("d1"), Var("d2")),
                BinOp(ArithOp.MUL, Var("d3"), Var("d4")),
            )
        ),
    },
    {
        "name": "BatchNorm_dims",
        "description": "BatchNorm requires >= 2D input",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("ndim"), Const(2)),
            Comparison(ComparisonOp.LE, Var("ndim"), Const(5)),
        ],
        "negation": Not(Comparison(ComparisonOp.GE, Var("ndim"), Const(2))),
    },
    {
        "name": "Concatenate_axis",
        "description": "Concat axis must be valid",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("axis"), Const(0)),
            Comparison(ComparisonOp.LT, Var("axis"), Var("ndim")),
            Comparison(ComparisonOp.GT, Var("ndim"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.LT, Var("axis"), Var("ndim"))),
    },
    {
        "name": "Embedding_index",
        "description": "Embedding index must be in vocab range",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("idx"), Const(0)),
            Comparison(ComparisonOp.LT, Var("idx"), Var("vocab_size")),
            Comparison(ComparisonOp.GT, Var("vocab_size"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.LT, Var("idx"), Var("vocab_size"))),
    },
    {
        "name": "Attention_dims",
        "description": "Multi-head attention dimension divisibility",
        "constraints": [
            Comparison(ComparisonOp.GT, Var("d_model"), Const(0)),
            Comparison(ComparisonOp.GT, Var("n_heads"), Const(0)),
            Comparison(
                ComparisonOp.EQ,
                BinOp(ArithOp.MOD, Var("d_model"), Var("n_heads")),
                Const(0),
            ),
        ],
        "negation": Not(
            Comparison(
                ComparisonOp.EQ,
                BinOp(ArithOp.MOD, Var("d_model"), Var("n_heads")),
                Const(0),
            )
        ),
    },
    {
        "name": "Pooling_kernel",
        "description": "Pooling kernel must fit in spatial dims",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("H"), Var("kH")),
            Comparison(ComparisonOp.GE, Var("W"), Var("kW")),
            Comparison(ComparisonOp.GT, Var("kH"), Const(0)),
            Comparison(ComparisonOp.GT, Var("kW"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.GE, Var("H"), Var("kH"))),
    },
    {
        "name": "Transpose_perm",
        "description": "Transpose permutation length matches ndim",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("perm_len"), Var("ndim")),
            Comparison(ComparisonOp.GT, Var("ndim"), Const(0)),
        ],
        "negation": Not(Comparison(ComparisonOp.EQ, Var("perm_len"), Var("ndim"))),
    },
]


@dataclass
class BenchmarkResult:
    name: str
    z3_sat_result: str = ""
    z3_proof_extracted: bool = False
    z3_proof_steps: int = 0
    z3_extraction_time_ms: float = 0.0
    cvc5_sat_result: str = ""
    cvc5_proof_extracted: bool = False
    cvc5_proof_steps: int = 0
    cvc5_extraction_time_ms: float = 0.0
    error: str = ""


def run_z3_benchmark(bench: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark through the Z3 backend."""
    from src.smt.solver import Z3Solver
    result: Dict[str, Any] = {
        "sat_result": "",
        "proof_extracted": False,
        "proof_steps": 0,
        "extraction_time_ms": 0.0,
    }
    try:
        solver = Z3Solver(timeout_ms=10000)
        for c in bench["constraints"]:
            solver.assert_formula(c)
        solver.assert_formula(bench["negation"])
        sat = solver.check_sat()
        result["sat_result"] = sat.value

        if sat == SatResult.UNSAT:
            try:
                import z3 as z3_mod
                ctx = z3_mod.Context("proof", "true")
                s2 = z3_mod.Solver(ctx=ctx)
                s2.set("timeout", 10000)
                for c in bench["constraints"]:
                    s2.add(solver._encode_predicate(c).translate(ctx))
                s2.add(solver._encode_predicate(bench["negation"]).translate(ctx))
                if s2.check() == z3_mod.unsat:
                    extractor = ProofExtractor(s2, list(s2.assertions()))
                    cert = extractor.extract(bench["name"], ["shape_safety"])
                    if cert is not None:
                        result["proof_extracted"] = True
                        result["proof_steps"] = len(cert.steps)
                        result["extraction_time_ms"] = cert.extraction_time_ms
            except Exception:
                pass
    except Exception as e:
        result["error"] = str(e)
    return result


def run_cvc5_benchmark(bench: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark through the CVC5 backend."""
    from src.smt.cvc5_backend import CVC5Solver
    result: Dict[str, Any] = {
        "sat_result": "",
        "proof_extracted": False,
        "proof_steps": 0,
        "extraction_time_ms": 0.0,
    }
    try:
        solver = CVC5Solver(timeout_ms=10000, produce_proofs=True)
        for c in bench["constraints"]:
            solver.assert_formula(c)
        solver.assert_formula(bench["negation"])
        sat = solver.check_sat()
        result["sat_result"] = sat.value

        if sat == SatResult.UNSAT:
            cert = solver.get_proof_certificate(bench["name"], ["shape_safety"])
            if cert is not None:
                result["proof_extracted"] = True
                result["proof_steps"] = len(cert.steps)
                result["extraction_time_ms"] = cert.extraction_time_ms
    except Exception as e:
        result["error"] = str(e)
    return result


def main() -> None:
    print("=" * 72)
    print("CVC5 Alethe Proof Coverage Evaluation")
    print("=" * 72)
    print(f"Z3 available:  {HAS_Z3}")
    print(f"CVC5 available: {HAS_CVC5}")
    print(f"Benchmarks:    {len(BENCHMARK_CONSTRAINTS)}")
    print()

    results: List[Dict[str, Any]] = []

    for bench in BENCHMARK_CONSTRAINTS:
        print(f"  Running: {bench['name']}...")
        br: Dict[str, Any] = {"name": bench["name"]}

        if HAS_Z3:
            z3_res = run_z3_benchmark(bench)
            br["z3"] = z3_res
        else:
            br["z3"] = {"sat_result": "unavailable", "proof_extracted": False}

        if HAS_CVC5:
            cvc5_res = run_cvc5_benchmark(bench)
            br["cvc5"] = cvc5_res
        else:
            br["cvc5"] = {"sat_result": "unavailable", "proof_extracted": False}

        proof_status = "unverified"
        if br["cvc5"].get("proof_extracted") or br["z3"].get("proof_extracted"):
            proof_status = "certified"
        elif br["z3"].get("sat_result") == "unsat" or br["cvc5"].get("sat_result") == "unsat":
            proof_status = "solver_verified"
        br["proof_status"] = proof_status

        results.append(br)
        z3_status = "✓ proof" if br["z3"].get("proof_extracted") else br["z3"].get("sat_result", "n/a")
        cvc5_status = "✓ proof" if br["cvc5"].get("proof_extracted") else br["cvc5"].get("sat_result", "n/a")
        print(f"    Z3: {z3_status}  |  CVC5: {cvc5_status}  |  Status: {proof_status}")

    # Summary
    total = len(results)
    z3_proofs = sum(1 for r in results if r["z3"].get("proof_extracted"))
    cvc5_proofs = sum(1 for r in results if r["cvc5"].get("proof_extracted"))
    combined_proofs = sum(
        1
        for r in results
        if r["z3"].get("proof_extracted") or r["cvc5"].get("proof_extracted")
    )
    z3_coverage = z3_proofs / total * 100 if total > 0 else 0
    cvc5_coverage = cvc5_proofs / total * 100 if total > 0 else 0
    combined_coverage = combined_proofs / total * 100 if total > 0 else 0

    summary = {
        "total_benchmarks": total,
        "z3_proof_coverage": z3_coverage,
        "z3_proofs_extracted": z3_proofs,
        "cvc5_proof_coverage": cvc5_coverage,
        "cvc5_proofs_extracted": cvc5_proofs,
        "combined_proof_coverage": combined_coverage,
        "combined_proofs_extracted": combined_proofs,
        "improvement_over_z3_pct": combined_coverage - z3_coverage,
    }

    output = {
        "summary": summary,
        "benchmarks": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Z3  proof coverage:      {z3_coverage:.1f}% ({z3_proofs}/{total})")
    print(f"  CVC5 proof coverage:     {cvc5_coverage:.1f}% ({cvc5_proofs}/{total})")
    print(f"  Combined coverage:       {combined_coverage:.1f}% ({combined_proofs}/{total})")
    print(f"  Improvement over Z3:     +{combined_coverage - z3_coverage:.1f}%")
    print(f"  Results saved to:        {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
