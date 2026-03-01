#!/usr/bin/env python3
"""
K-Z-K Idempotence Verification.

Verifies that the 3-phase normalization pipeline from knuth_bendix.py
(Z3.simplify → KB-normalize → Z3.simplify, i.e. K-Z-K) is idempotent:

    KZK(KZK(e)) == KZK(e)   for all test expressions e

Generates concrete test expressions spanning all 7 KB rewrite rules
(R1–R7) plus AC-normalization, applies the pipeline once and twice,
and checks equality.

Results are saved to ``experiments/results/kzk_idempotence_results.json``.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_PATH = OUTPUT_DIR / "kzk_idempotence_results.json"

from src.knuth_bendix import (
    Term,
    RewriteRule,
    get_completed_rules,
    full_normalize,
    normalize,
    ac_normalize,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test expression generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_expressions() -> List[Dict[str, Any]]:
    """Generate test expressions spanning all KB rewrite rules.

    Each entry has a 'name', 'expr' (Term), and 'target_rules' list.
    """
    a = Term.var("a")
    b = Term.var("b")
    c = Term.var("c")
    s = Term.var("s")
    t = Term.var("t")
    d0 = Term.var("d0")
    d1 = Term.var("d1")
    h = Term.var("h")
    k = Term.var("k")
    _0 = Term.const(0)
    _1 = Term.const(1)
    _3 = Term.const(3)
    _5 = Term.const(5)
    _7 = Term.const(7)
    _8 = Term.const(8)
    _16 = Term.const(16)
    _32 = Term.const(32)
    _64 = Term.const(64)

    tests = []

    # --- R1: bc(a, 1) → a ---
    tests.append({
        "name": "R1_bc_identity_right_var",
        "expr": Term.bc(a, _1),
        "target_rules": ["R1"],
    })
    tests.append({
        "name": "R1_bc_identity_right_const",
        "expr": Term.bc(_32, _1),
        "target_rules": ["R1"],
    })

    # --- R2: bc(1, b) → b ---
    tests.append({
        "name": "R2_bc_identity_left_var",
        "expr": Term.bc(_1, b),
        "target_rules": ["R2"],
    })
    tests.append({
        "name": "R2_bc_identity_left_const",
        "expr": Term.bc(_1, _64),
        "target_rules": ["R2"],
    })

    # --- R3: bc(a, a) → a ---
    tests.append({
        "name": "R3_bc_idempotent_var",
        "expr": Term.bc(a, a),
        "target_rules": ["R3"],
    })
    tests.append({
        "name": "R3_bc_idempotent_const",
        "expr": Term.bc(_16, _16),
        "target_rules": ["R3"],
    })

    # --- R4: transp(transp(s,d0,d1),d0,d1) → s ---
    tests.append({
        "name": "R4_double_transpose_var",
        "expr": Term.transp(Term.transp(s, d0, d1), d0, d1),
        "target_rules": ["R4"],
    })
    tests.append({
        "name": "R4_double_transpose_const",
        "expr": Term.transp(Term.transp(_32, _0, _1), _0, _1),
        "target_rules": ["R4"],
    })

    # --- R5: numel(reshape(s,t)) → numel(s) ---
    tests.append({
        "name": "R5_reshape_numel_var",
        "expr": Term.numel(Term.reshape(s, t)),
        "target_rules": ["R5"],
    })
    tests.append({
        "name": "R5_reshape_numel_const",
        "expr": Term.numel(Term.reshape(_32, _8)),
        "target_rules": ["R5"],
    })

    # --- R6: conv(h,k,1,0) → add(sub(h,k),1) ---
    tests.append({
        "name": "R6_conv_basic_var",
        "expr": Term.conv(h, k, _1, _0),
        "target_rules": ["R6"],
    })
    tests.append({
        "name": "R6_conv_basic_const",
        "expr": Term.conv(_32, _3, _1, _0),
        "target_rules": ["R6"],
    })

    # --- R7: pool(h,k,k,0) → floor_div(h,k) ---
    tests.append({
        "name": "R7_pool_stride_eq_kernel_var",
        "expr": Term.pool(h, k, k, _0),
        "target_rules": ["R7"],
    })
    tests.append({
        "name": "R7_pool_stride_eq_kernel_const",
        "expr": Term.pool(_64, _8, _8, _0),
        "target_rules": ["R7"],
    })

    # --- AC normalization: bc(b, a) → bc(a, b) when repr(b) > repr(a) ---
    tests.append({
        "name": "AC_commutativity",
        "expr": Term.bc(b, a),
        "target_rules": ["AC"],
    })

    # --- Nested / combined expressions ---
    # bc(bc(a, 1), b) → bc(a, b) via R1 inside
    tests.append({
        "name": "nested_R1_inside_bc",
        "expr": Term.bc(Term.bc(a, _1), b),
        "target_rules": ["R1"],
    })

    # numel(reshape(reshape(s, t), a)) → numel(reshape(s, t)) → numel(s)
    tests.append({
        "name": "nested_R5_double_reshape",
        "expr": Term.numel(Term.reshape(Term.reshape(s, t), a)),
        "target_rules": ["R5"],
    })

    # transp(transp(bc(a,1), d0, d1), d0, d1) → bc(a,1) → a
    tests.append({
        "name": "combined_R4_R1",
        "expr": Term.transp(Term.transp(Term.bc(a, _1), d0, d1), d0, d1),
        "target_rules": ["R4", "R1"],
    })

    # conv(bc(h,1), k, 1, 0) → conv(h, k, 1, 0) → add(sub(h,k),1)
    tests.append({
        "name": "combined_R1_R6",
        "expr": Term.conv(Term.bc(h, _1), k, _1, _0),
        "target_rules": ["R1", "R6"],
    })

    # pool(bc(h,h), k, k, 0) → pool(h, k, k, 0) → floor_div(h, k)
    tests.append({
        "name": "combined_R3_R7",
        "expr": Term.pool(Term.bc(h, h), k, k, _0),
        "target_rules": ["R3", "R7"],
    })

    # Already-normal expressions (should be unchanged)
    tests.append({
        "name": "already_normal_var",
        "expr": a,
        "target_rules": [],
    })
    tests.append({
        "name": "already_normal_const",
        "expr": _32,
        "target_rules": [],
    })
    tests.append({
        "name": "already_normal_add",
        "expr": Term.add(a, b),
        "target_rules": [],
    })
    tests.append({
        "name": "already_normal_bc_distinct",
        "expr": Term.bc(a, b),
        "target_rules": [],
    })

    # Deep nesting
    tests.append({
        "name": "deep_bc_chain",
        "expr": Term.bc(Term.bc(Term.bc(a, _1), _1), _1),
        "target_rules": ["R1"],
    })

    # numel of numel(reshape(...))
    tests.append({
        "name": "numel_of_conv",
        "expr": Term.numel(Term.conv(h, k, _1, _0)),
        "target_rules": ["R6"],
    })

    return tests


# ═══════════════════════════════════════════════════════════════════════════════
# KZK pipeline (term-level, no Z3 dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def kzk_normalize(term: Term, rules: List[RewriteRule]) -> Term:
    """Apply the K-Z-K pipeline at term level.

    This mirrors normalize_z3_expr but operates purely on Terms:
      Phase 1: AC-normalize (analogous to Z3 simplify for commutativity)
      Phase 2: KB rewrite to normal form
      Phase 3: AC-normalize again (analogous to final Z3 simplify)
    """
    # Phase 1
    step1 = ac_normalize(term)
    # Phase 2
    step2 = full_normalize(step1, rules)
    # Phase 3
    step3 = ac_normalize(step2)
    return step3


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_verification() -> Dict[str, Any]:
    rules = get_completed_rules()
    test_exprs = generate_test_expressions()

    results_list: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    total = len(test_exprs)

    for test in test_exprs:
        name = test["name"]
        expr = test["expr"]
        target_rules = test["target_rules"]

        t0 = time.perf_counter()
        once = kzk_normalize(expr, rules)
        t1 = time.perf_counter()
        twice = kzk_normalize(once, rules)
        t2 = time.perf_counter()

        is_idempotent = (once == twice)
        changed = (expr != once)

        entry = {
            "name": name,
            "input": repr(expr),
            "after_once": repr(once),
            "after_twice": repr(twice),
            "idempotent": is_idempotent,
            "changed_from_input": changed,
            "target_rules": target_rules,
            "time_first_ms": round((t1 - t0) * 1000, 3),
            "time_second_ms": round((t2 - t1) * 1000, 3),
        }
        results_list.append(entry)

        if not is_idempotent:
            violations.append(entry)

    n_passed = sum(1 for r in results_list if r["idempotent"])
    n_changed = sum(1 for r in results_list if r["changed_from_input"])

    return {
        "summary": {
            "total_tests": total,
            "passed": n_passed,
            "violations": len(violations),
            "idempotence_verified": len(violations) == 0,
            "expressions_that_changed": n_changed,
            "expressions_already_normal": total - n_changed,
        },
        "per_test": results_list,
        "violations": violations,
        "rules_tested": [
            {"id": r.id, "name": r.name, "lhs": repr(r.lhs), "rhs": repr(r.rhs)}
            for r in rules
        ],
    }


def main():
    results = run_verification()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("K-Z-K Idempotence Verification")
    print("=" * 60)
    print(f"Total test expressions: {s['total_tests']}")
    print(f"Passed (idempotent):    {s['passed']}")
    print(f"Violations:             {s['violations']}")
    print(f"Changed from input:     {s['expressions_that_changed']}")
    print(f"Already normal:         {s['expressions_already_normal']}")
    print(f"\nVerdict: {'PASS ✓' if s['idempotence_verified'] else 'FAIL ✗'}")

    if results["violations"]:
        print("\nViolations:")
        for v in results["violations"]:
            print(f"  {v['name']}: {v['input']}")
            print(f"    after once:  {v['after_once']}")
            print(f"    after twice: {v['after_twice']}")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
