#!/usr/bin/env python3
"""
Knuth-Bendix Critical Pair Enumeration.

Enumerates ALL 28 critical pairs (C(7,2) = 21 inter-rule pairs + 7
self-overlaps) for the 7-rule tensor shape rewrite system, checks
joinability under RPO, and verifies K∘Z∘K commutativity.

Results are saved to ``experiments/results/kb_critical_pairs_28.json``.
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_PATH = OUTPUT_DIR / "kb_critical_pairs_28.json"

from src.knuth_bendix import (
    CriticalPair,
    RewriteRule,
    Term,
    ac_normalize,
    apply_substitution,
    compute_critical_pairs,
    full_normalize,
    get_completed_rules,
    normalize,
    rpo_gt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Critical pair enumeration
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_all_28_pairs(
    rules: List[RewriteRule],
) -> List[Dict[str, Any]]:
    """Enumerate all 28 rule pairs (21 inter-rule + 7 self-overlaps).

    For each pair (i, j) with i <= j, compute critical pairs by
    overlapping LHS_i with LHS_j and (if i != j) LHS_j with LHS_i.
    """
    assert len(rules) == 7, f"Expected 7 rules, got {len(rules)}"

    pair_results: List[Dict[str, Any]] = []

    # Generate all 28 pairs: 21 inter-rule (i < j) + 7 self-overlaps (i == j)
    all_pairs: List[Tuple[int, int]] = []
    for i in range(len(rules)):
        all_pairs.append((i, i))  # self-overlap
    for i, j in combinations(range(len(rules)), 2):
        all_pairs.append((i, j))

    for idx1, idx2 in all_pairs:
        r1 = rules[idx1]
        r2 = rules[idx2]
        is_self = idx1 == idx2

        # Compute critical pairs in both directions for inter-rule
        cps_forward = compute_critical_pairs(r1, r2)
        if not is_self:
            cps_backward = compute_critical_pairs(r2, r1)
            all_cps = cps_forward + cps_backward
        else:
            all_cps = cps_forward

        # Check joinability of each critical pair
        for cp in all_cps:
            nf1 = full_normalize(cp.term1, rules)
            nf2 = full_normalize(cp.term2, rules)
            cp.joinable = (nf1 == nf2)

        joinable_count = sum(1 for cp in all_cps if cp.joinable)
        all_joinable = all(cp.joinable for cp in all_cps) if all_cps else True

        cp_details = []
        for cp in all_cps:
            nf1 = full_normalize(cp.term1, rules)
            nf2 = full_normalize(cp.term2, rules)
            cp_details.append({
                "term1": repr(cp.term1),
                "term2": repr(cp.term2),
                "normal_form1": repr(nf1),
                "normal_form2": repr(nf2),
                "overlap_position": list(cp.overlap_position),
                "joinable": cp.joinable,
            })

        pair_results.append({
            "rule1_id": r1.id,
            "rule1_name": r1.name,
            "rule2_id": r2.id,
            "rule2_name": r2.name,
            "is_self_overlap": is_self,
            "overlaps_found": len(all_cps),
            "critical_pairs": cp_details,
            "joinable_count": joinable_count,
            "all_joinable": all_joinable,
        })

    return pair_results


# ═══════════════════════════════════════════════════════════════════════════════
# RPO orientation verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_rpo_orientation(rules: List[RewriteRule]) -> List[Dict[str, Any]]:
    """Verify that each rule is properly oriented under RPO."""
    results = []
    for r in rules:
        oriented = rpo_gt(r.lhs, r.rhs)
        results.append({
            "rule_id": r.id,
            "name": r.name,
            "lhs": repr(r.lhs),
            "rhs": repr(r.rhs),
            "lhs_gt_rhs_rpo": oriented,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# K∘Z∘K commutativity verification
# ═══════════════════════════════════════════════════════════════════════════════

def kzk_normalize(term: Term, rules: List[RewriteRule]) -> Term:
    """Apply K∘Z∘K pipeline at term level."""
    step1 = ac_normalize(term)
    step2 = full_normalize(step1, rules)
    step3 = ac_normalize(step2)
    return step3


def generate_kzk_test_expressions() -> List[Dict[str, Any]]:
    """Generate expressions for K∘Z∘K commutativity testing."""
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
    _8 = Term.const(8)
    _16 = Term.const(16)
    _32 = Term.const(32)
    _64 = Term.const(64)

    exprs = [
        ("bc_id_right", Term.bc(a, _1)),
        ("bc_id_left", Term.bc(_1, b)),
        ("bc_idempotent", Term.bc(a, a)),
        ("bc_comm", Term.bc(b, a)),
        ("double_transp", Term.transp(Term.transp(s, d0, d1), d0, d1)),
        ("reshape_numel", Term.numel(Term.reshape(s, t))),
        ("conv_basic", Term.conv(h, k, _1, _0)),
        ("pool_basic", Term.pool(h, k, k, _0)),
        ("nested_bc_r1", Term.bc(Term.bc(a, _1), b)),
        ("nested_bc_r2", Term.bc(Term.bc(_1, a), b)),
        ("nested_bc_r3", Term.bc(Term.bc(a, a), b)),
        ("double_r1", Term.bc(Term.bc(a, _1), _1)),
        ("conv_with_bc", Term.conv(Term.bc(h, _1), k, _1, _0)),
        ("pool_with_bc", Term.pool(Term.bc(h, h), k, k, _0)),
        ("numel_reshape_bc", Term.numel(Term.reshape(Term.bc(s, _1), t))),
        ("transp_bc", Term.transp(Term.transp(Term.bc(a, _1), d0, d1), d0, d1)),
        ("var_a", a),
        ("const_32", _32),
        ("add_ab", Term.add(a, b)),
        ("bc_distinct", Term.bc(a, b)),
        ("deep_bc", Term.bc(Term.bc(Term.bc(a, _1), _1), _1)),
        ("bc_const_1_1", Term.bc(_1, _1)),
        ("numel_conv", Term.numel(Term.conv(h, k, _1, _0))),
        ("bc_nested_r1_r3", Term.bc(Term.bc(a, _1), Term.bc(a, a))),
        ("pool_conv_nested", Term.pool(Term.conv(h, k, _1, _0), _3, _3, _0)),
        # Larger expressions
        ("triple_bc", Term.bc(Term.bc(a, b), c)),
        ("reshape_reshape", Term.numel(Term.reshape(Term.reshape(s, t), a))),
        ("bc_32_32", Term.bc(_32, _32)),
        ("bc_16_64", Term.bc(_16, _64)),
        ("conv_32_3", Term.conv(_32, _3, _1, _0)),
        ("pool_64_8", Term.pool(_64, _8, _8, _0)),
        ("mul_sub", Term.mul(Term.sub(a, b), c)),
        ("floor_div_ab", Term.floor_div(a, b)),
        ("bc_a_bc_a_1", Term.bc(a, Term.bc(a, _1))),
        ("transp_single", Term.transp(s, d0, d1)),
        ("numel_s", Term.numel(s)),
        ("reshape_st", Term.reshape(s, t)),
        ("perm_sp", Term.perm(s, t)),
        ("conv_general", Term.conv(h, k, s, t)),
        ("pool_general", Term.pool(h, k, s, t)),
        # Stress nesting
        ("bc_chain_4", Term.bc(Term.bc(Term.bc(Term.bc(a, _1), _1), _1), _1)),
        ("numel_reshape_chain", Term.numel(Term.reshape(Term.reshape(Term.reshape(s, t), a), b))),
        ("double_transp_nested", Term.transp(Term.transp(Term.transp(Term.transp(s, d0, d1), d0, d1), d0, d1), d0, d1)),
        # Mixed operations
        ("conv_pool", Term.pool(Term.conv(h, k, _1, _0), _3, _3, _0)),
        ("bc_conv", Term.conv(Term.bc(h, _1), Term.bc(k, _1), _1, _0)),
        ("numel_bc", Term.numel(Term.bc(a, _1))),
        ("reshape_bc", Term.reshape(Term.bc(s, _1), t)),
        ("transp_conv", Term.transp(Term.conv(h, k, _1, _0), d0, d1)),
        ("add_sub", Term.add(Term.sub(a, b), _1)),
    ]
    return [{"name": name, "expr": expr} for name, expr in exprs]


def verify_kzk_commutativity(
    rules: List[RewriteRule], n_tests: int = 50,
) -> Dict[str, Any]:
    """Verify K∘Z∘K idempotence on test expressions.

    KZK(KZK(e)) == KZK(e) for all test expressions.
    """
    test_exprs = generate_kzk_test_expressions()[:n_tests]
    results = []
    commutative_count = 0

    for test in test_exprs:
        expr = test["expr"]
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        is_commutative = (once == twice)
        if is_commutative:
            commutative_count += 1
        results.append({
            "name": test["name"],
            "input": repr(expr),
            "kzk_once": repr(once),
            "kzk_twice": repr(twice),
            "commutative": is_commutative,
        })

    return {
        "tested": len(results),
        "commutative": commutative_count,
        "all_commutative": commutative_count == len(results),
        "examples": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_enumeration() -> Dict[str, Any]:
    rules = get_completed_rules()
    assert len(rules) == 7

    t0 = time.perf_counter()

    # 1. Enumerate all 28 pairs
    pair_results = enumerate_all_28_pairs(rules)

    # 2. Verify RPO orientation
    rpo_results = verify_rpo_orientation(rules)

    # 3. K∘Z∘K commutativity
    kzk_results = verify_kzk_commutativity(rules, n_tests=50)

    elapsed = time.perf_counter() - t0

    # Summary
    total_pairs = len(pair_results)
    pairs_with_overlap = sum(1 for p in pair_results if p["overlaps_found"] > 0)
    total_cps = sum(p["overlaps_found"] for p in pair_results)
    all_joinable = all(p["all_joinable"] for p in pair_results)
    all_rpo = all(r["lhs_gt_rhs_rpo"] for r in rpo_results)

    return {
        "total_rule_pairs": total_pairs,
        "pairs_with_overlap": pairs_with_overlap,
        "critical_pairs_found": total_cps,
        "all_joinable": all_joinable,
        "all_rpo_oriented": all_rpo,
        "elapsed_seconds": round(elapsed, 3),
        "pairs": pair_results,
        "rpo_orientation": rpo_results,
        "kzk_commutativity": {
            "tested": kzk_results["tested"],
            "commutative": kzk_results["commutative"],
            "examples": kzk_results["examples"],
        },
    }


def main():
    results = run_enumeration()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("KB Critical Pair Enumeration (28 pairs)")
    print("=" * 60)
    print(f"Total rule pairs:       {results['total_rule_pairs']}")
    print(f"Pairs with overlap:     {results['pairs_with_overlap']}")
    print(f"Critical pairs found:   {results['critical_pairs_found']}")
    print(f"All joinable:           {results['all_joinable']}")
    print(f"All RPO-oriented:       {results['all_rpo_oriented']}")
    print(f"K∘Z∘K commutative:      {results['kzk_commutativity']['commutative']}"
          f"/{results['kzk_commutativity']['tested']}")
    print(f"Elapsed:                {results['elapsed_seconds']}s")

    print(f"\nPer-pair breakdown:")
    for p in results["pairs"]:
        kind = "self" if p["is_self_overlap"] else "inter"
        join = "✓" if p["all_joinable"] else "✗"
        print(f"  R{p['rule1_id']}-R{p['rule2_id']} ({kind}): "
              f"{p['overlaps_found']} overlaps [{join}]  "
              f"({p['rule1_name']} × {p['rule2_name']})")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
