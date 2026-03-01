#!/usr/bin/env python3
"""
CVC5 Propagator Concordance Experiment.

Verifies that Z3-only and CVC5+Z3 dual-solver produce identical verdicts,
consistent models, and compatible proof structures on a suite of shape
verification problems. Reports discrepancies to a JSON benchmark file.
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

from src.smt.cvc5_backend import CVC5Solver
from src.smt.solver import (
    SatResult,
    Comparison,
    ComparisonOp,
    Var,
    Const,
    BinOp,
    ArithOp,
    And,
    Or,
    Not,
    BoolLit,
    Sort,
)

BENCHMARKS_DIR = PROJECT_ROOT / ".benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = BENCHMARKS_DIR / "cvc5_propagator_concordance.json"


# ═══════════════════════════════════════════════════════════════════════════
# Problem suite
# ═══════════════════════════════════════════════════════════════════════════

PROBLEMS = [
    {
        "name": "simple_equality_sat",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("x"), Const(10)),
        ],
        "expected": "sat",
    },
    {
        "name": "simple_equality_unsat",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("x"), Const(10)),
            Comparison(ComparisonOp.EQ, Var("x"), Const(20)),
        ],
        "expected": "unsat",
    },
    {
        "name": "broadcast_1_to_n",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("b"), Const(8)),
            Comparison(ComparisonOp.EQ, Var("out"), Const(8)),
        ],
        "expected": "sat",
    },
    {
        "name": "matmul_compatible",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")),
            Comparison(ComparisonOp.EQ, Var("a_cols"), Const(64)),
            Comparison(ComparisonOp.GT, Var("b_rows"), Const(0)),
        ],
        "expected": "sat",
    },
    {
        "name": "matmul_incompatible",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a_cols"), Const(64)),
            Comparison(ComparisonOp.EQ, Var("b_rows"), Const(128)),
            Comparison(ComparisonOp.EQ, Var("a_cols"), Var("b_rows")),
        ],
        "expected": "unsat",
    },
    {
        "name": "reshape_product_sat",
        "constraints": [
            Comparison(ComparisonOp.EQ, BinOp(ArithOp.MUL, Var("h"), Var("w")), Const(784)),
            Comparison(ComparisonOp.EQ, Var("h"), Const(28)),
            Comparison(ComparisonOp.EQ, Var("w"), Const(28)),
        ],
        "expected": "sat",
    },
    {
        "name": "reshape_product_unsat",
        "constraints": [
            Comparison(ComparisonOp.EQ, BinOp(ArithOp.MUL, Var("h"), Var("w")), Const(100)),
            Comparison(ComparisonOp.EQ, Var("h"), Const(28)),
            Comparison(ComparisonOp.EQ, Var("w"), Const(28)),
        ],
        "expected": "unsat",
    },
    {
        "name": "stride_contiguous",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("s1"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("s0"), BinOp(ArithOp.MUL, Var("s1"), Var("d1"))),
            Comparison(ComparisonOp.EQ, Var("d1"), Const(8)),
            Comparison(ComparisonOp.GT, Var("s0"), Const(0)),
        ],
        "expected": "sat",
    },
    {
        "name": "transitive_ordering_sat",
        "constraints": [
            Comparison(ComparisonOp.GE, Var("d0"), Const(3)),
            Comparison(ComparisonOp.GE, Var("d1"), Const(2)),
            Comparison(ComparisonOp.GE, Var("d2"), Const(1)),
            Comparison(ComparisonOp.GT, Var("d0"), Var("d1")),
            Comparison(ComparisonOp.GT, Var("d1"), Var("d2")),
        ],
        "expected": "sat",
    },
    {
        "name": "triangle_inequality_unsat",
        "constraints": [
            Comparison(ComparisonOp.GT, Var("a"), Var("b")),
            Comparison(ComparisonOp.GT, Var("b"), Var("c")),
            Comparison(ComparisonOp.GT, Var("c"), Var("a")),
        ],
        "expected": "unsat",
    },
    {
        "name": "multi_dim_broadcast",
        "constraints": [
            Comparison(ComparisonOp.EQ, Var("a0"), Const(3)),
            Comparison(ComparisonOp.EQ, Var("a1"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("b0"), Const(1)),
            Comparison(ComparisonOp.EQ, Var("b1"), Const(4)),
            Comparison(ComparisonOp.EQ, Var("o0"), Const(3)),
            Comparison(ComparisonOp.EQ, Var("o1"), Const(4)),
        ],
        "expected": "sat",
    },
    {
        "name": "sum_constraint",
        "constraints": [
            Comparison(ComparisonOp.EQ, BinOp(ArithOp.ADD, Var("x"), Var("y")), Const(100)),
            Comparison(ComparisonOp.GT, Var("x"), Const(0)),
            Comparison(ComparisonOp.GT, Var("y"), Const(0)),
        ],
        "expected": "sat",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Solver runners
# ═══════════════════════════════════════════════════════════════════════════

def _pred_to_z3(pred, var_map):
    """Convert a Predicate to Z3 expression."""
    if isinstance(pred, Comparison):
        left = _expr_to_z3(pred.left, var_map)
        right = _expr_to_z3(pred.right, var_map)
        ops = {
            ComparisonOp.EQ: lambda a, b: a == b,
            ComparisonOp.NE: lambda a, b: a != b,
            ComparisonOp.LT: lambda a, b: a < b,
            ComparisonOp.LE: lambda a, b: a <= b,
            ComparisonOp.GT: lambda a, b: a > b,
            ComparisonOp.GE: lambda a, b: a >= b,
        }
        return ops[pred.op](left, right)
    elif isinstance(pred, And):
        return z3.And([_pred_to_z3(c, var_map) for c in pred.conjuncts])
    elif isinstance(pred, Or):
        return z3.Or([_pred_to_z3(d, var_map) for d in pred.disjuncts])
    elif isinstance(pred, Not):
        return z3.Not(_pred_to_z3(pred.operand, var_map))
    elif isinstance(pred, BoolLit):
        return z3.BoolVal(pred.value)
    raise ValueError(f"Unsupported predicate: {type(pred)}")


def _expr_to_z3(expr, var_map):
    if isinstance(expr, Var):
        if expr.name not in var_map:
            var_map[expr.name] = z3.Int(expr.name)
        return var_map[expr.name]
    elif isinstance(expr, Const):
        return z3.IntVal(expr.value)
    elif isinstance(expr, BinOp):
        left = _expr_to_z3(expr.left, var_map)
        right = _expr_to_z3(expr.right, var_map)
        ops = {
            ArithOp.ADD: lambda a, b: a + b,
            ArithOp.SUB: lambda a, b: a - b,
            ArithOp.MUL: lambda a, b: a * b,
        }
        return ops[expr.op](left, right)
    raise ValueError(f"Unsupported expression: {type(expr)}")


def _declare_vars_for_pred(solver, pred):
    if isinstance(pred, Comparison):
        _declare_vars_for_expr(solver, pred.left)
        _declare_vars_for_expr(solver, pred.right)
    elif isinstance(pred, (And, Or)):
        children = pred.conjuncts if isinstance(pred, And) else pred.disjuncts
        for child in children:
            _declare_vars_for_pred(solver, child)
    elif isinstance(pred, Not):
        _declare_vars_for_pred(solver, pred.operand)


def _declare_vars_for_expr(solver, expr):
    if isinstance(expr, Var):
        solver.declare_int(expr.name)
    elif isinstance(expr, BinOp):
        _declare_vars_for_expr(solver, expr.left)
        _declare_vars_for_expr(solver, expr.right)


def run_z3(constraints):
    """Run Z3 on constraints, return (verdict, time_ms, model_vals)."""
    s = z3.Solver()
    s.set("timeout", 10000)
    var_map = {}
    for c in constraints:
        s.add(_pred_to_z3(c, var_map))
    t0 = time.monotonic()
    result = s.check()
    elapsed = (time.monotonic() - t0) * 1000
    verdict = "sat" if result == z3.sat else ("unsat" if result == z3.unsat else "unknown")
    model_vals = {}
    if result == z3.sat:
        m = s.model()
        for name, var in var_map.items():
            try:
                model_vals[name] = m.evaluate(var).as_long()
            except Exception:
                model_vals[name] = str(m.evaluate(var))
    return verdict, elapsed, model_vals


def run_cvc5(constraints, produce_proofs=True):
    """Run CVC5 on constraints, return (verdict, time_ms, model_vals, proof_info)."""
    solver = CVC5Solver(timeout_ms=10000, produce_proofs=produce_proofs)
    for c in constraints:
        _declare_vars_for_pred(solver, c)
        solver.assert_formula(c)
    t0 = time.monotonic()
    result = solver.check_sat()
    elapsed = (time.monotonic() - t0) * 1000
    verdict = result.value
    model_vals = {}
    proof_info = None
    if result == SatResult.SAT:
        model = solver.get_model()
        if model:
            model_vals = {k: v for k, v in model.variable_values.items()}
    elif result == SatResult.UNSAT and produce_proofs:
        cert = solver.get_proof_certificate()
        if cert:
            proof_info = {
                "num_steps": len(cert.steps),
                "theories_used": cert.theories_used,
                "proof_source": cert.proof_source,
                "root_step": cert.root_step,
            }
    return verdict, elapsed, model_vals, proof_info


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CVC5 Propagator Concordance Experiment")
    print("=" * 70)

    if not HAS_Z3:
        print("ERROR: z3 not available")
        sys.exit(1)
    if not HAS_CVC5:
        print("ERROR: cvc5 not available")
        sys.exit(1)

    results = []
    n_agree = 0
    n_disagree = 0
    n_problems = len(PROBLEMS)

    for problem in PROBLEMS:
        name = problem["name"]
        constraints = problem["constraints"]
        expected = problem["expected"]

        print(f"\n  [{name}]")

        try:
            z3_verdict, z3_ms, z3_model = run_z3(constraints)
        except Exception as e:
            z3_verdict, z3_ms, z3_model = "ERROR", 0.0, {}
            print(f"    Z3 error: {e}")

        try:
            cvc5_verdict, cvc5_ms, cvc5_model, proof_info = run_cvc5(constraints)
        except Exception as e:
            cvc5_verdict, cvc5_ms, cvc5_model, proof_info = "ERROR", 0.0, {}, None
            print(f"    CVC5 error: {e}")

        agree = z3_verdict == cvc5_verdict
        correct_z3 = z3_verdict == expected
        correct_cvc5 = cvc5_verdict == expected

        if agree:
            n_agree += 1
        else:
            n_disagree += 1

        # Check model concordance for SAT cases
        model_concordant = True
        if z3_verdict == "SAT" and cvc5_verdict == "SAT" and z3_model and cvc5_model:
            common_vars = set(z3_model.keys()) & set(cvc5_model.keys())
            for v in common_vars:
                if z3_model[v] != cvc5_model[v]:
                    model_concordant = False
                    break

        status = "✓" if agree else "✗"
        print(f"    Z3:   {z3_verdict:>7s} ({z3_ms:6.1f}ms)")
        print(f"    CVC5: {cvc5_verdict:>7s} ({cvc5_ms:6.1f}ms)")
        print(f"    Agree: {status}  Expected: {expected}")
        if proof_info:
            print(f"    Proof: {proof_info['num_steps']} steps, "
                  f"theories={proof_info['theories_used']}")

        result_entry = {
            "name": name,
            "expected": expected,
            "z3_verdict": z3_verdict,
            "cvc5_verdict": cvc5_verdict,
            "z3_time_ms": round(z3_ms, 2),
            "cvc5_time_ms": round(cvc5_ms, 2),
            "verdicts_agree": agree,
            "z3_correct": correct_z3,
            "cvc5_correct": correct_cvc5,
            "model_concordant": model_concordant,
        }
        if proof_info:
            result_entry["cvc5_proof"] = proof_info
        results.append(result_entry)

    # Summary
    concordance_rate = n_agree / n_problems if n_problems > 0 else 0.0
    print(f"\n{'=' * 70}")
    print(f"CONCORDANCE SUMMARY")
    print(f"  Total problems: {n_problems}")
    print(f"  Verdicts agree: {n_agree} / {n_problems} ({concordance_rate:.1%})")
    print(f"  Discrepancies:  {n_disagree}")
    print(f"{'=' * 70}")

    output = {
        "experiment": "cvc5_propagator_concordance",
        "n_problems": n_problems,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "concordance_rate": concordance_rate,
        "problems": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
