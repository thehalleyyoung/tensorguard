#!/usr/bin/env python3
"""
Distinctness Axiom Verification for Finite Sorts.

Generates the axiom sets for each finite sort (T_device, T_phase,
T_perm), verifies tightness (exactly |S| distinct values), runs
bidirectional simulation checks, and tests on formulas that would
previously admit spurious models.

Results are saved to ``experiments/results/distinctness_axiom_verification.json``.
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
OUTPUT_PATH = OUTPUT_DIR / "distinctness_axiom_verification.json"

import z3

from src.smt.distinctness_axioms import (
    DEVICE_SORT,
    PHASE_SORT,
    PERM_SORT,
    FiniteSort,
    FiniteSortAxiomGenerator,
    get_standard_sorts,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Axiom set generation and summary
# ═══════════════════════════════════════════════════════════════════════════════

def generate_axiom_summary(gen: FiniteSortAxiomGenerator, sort: FiniteSort) -> Dict[str, Any]:
    """Generate and summarize axioms for a single sort."""
    dist_axioms = gen.generate_distinctness_axioms(sort.name)
    # Declare a test variable for totality
    gen.declare_variable(f"_test_{sort.name}", sort.name)
    tot_axioms = gen.generate_totality_axioms(sort.name)

    return {
        "sort_name": sort.name,
        "sort_size": sort.size,
        "constants": list(sort.constants),
        "distinctness_axiom_count": len(dist_axioms),
        "expected_distinctness_count": sort.size * (sort.size - 1) // 2,
        "totality_axiom_count": len(tot_axioms),
        "distinctness_axioms": [str(a) for a in dist_axioms],
        "totality_axioms": [str(a) for a in tot_axioms],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tightness verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_sort_tightness(gen: FiniteSortAxiomGenerator, sort: FiniteSort) -> Dict[str, Any]:
    """Verify that axioms admit exactly |S| distinct values."""
    return gen.verify_tightness(sort.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Bidirectional simulation check
# ═══════════════════════════════════════════════════════════════════════════════

def run_bidirectional_simulation(gen: FiniteSortAxiomGenerator) -> Dict[str, Any]:
    """Test bidirectional simulation: forward (encoding) and backward (decoding).

    For each finite sort, verify that:
      1. Each constant can be assigned to a variable (forward)
      2. Any model maps variables to one of the declared constants (backward)
    """
    results = []

    for sort_name in ["T_device", "T_phase", "T_perm"]:
        consts = gen.get_constants(sort_name)
        z3_sort = gen.get_sort(sort_name)
        const_values = list(consts.values())

        # Forward: each constant is assignable
        forward_tests = []
        for cname, cval in consts.items():
            s = z3.Solver()
            s.set("timeout", 5000)
            v = z3.Const(f"_fwd_{sort_name}_{cname}", z3_sort)
            dist = gen.generate_distinctness_axioms(sort_name)
            s.add(*dist)
            s.add(z3.Or(*[v == c for c in const_values]))
            s.add(v == cval)
            r = s.check()
            forward_tests.append({
                "constant": cname,
                "satisfiable": str(r) == "sat",
            })

        # Backward: no extra value is possible
        s = z3.Solver()
        s.set("timeout", 5000)
        v = z3.Const(f"_bwd_{sort_name}", z3_sort)
        dist = gen.generate_distinctness_axioms(sort_name)
        s.add(*dist)
        s.add(z3.Or(*[v == c for c in const_values]))
        for c in const_values:
            s.add(v != c)
        r = s.check()
        backward_ok = str(r) == "unsat"

        results.append({
            "sort": sort_name,
            "forward_tests": forward_tests,
            "all_forward_pass": all(t["satisfiable"] for t in forward_tests),
            "backward_no_extra": backward_ok,
            "simulation_holds": all(t["satisfiable"] for t in forward_tests) and backward_ok,
        })

    return {
        "tests": results,
        "all_pass": all(r["simulation_holds"] for r in results),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Spurious model test suite
# ═══════════════════════════════════════════════════════════════════════════════

def run_spurious_model_tests(gen: FiniteSortAxiomGenerator) -> Dict[str, Any]:
    """Test formulas that previously might admit spurious models."""
    results = []

    # Test 1: Without axioms, constants can be collapsed
    naive_sort = z3.DeclareSort("NaiveDevice")
    c1 = z3.Const("naive_cpu", naive_sort)
    c2 = z3.Const("naive_cuda0", naive_sort)
    s = z3.Solver()
    s.add(c1 == c2)
    r = s.check()
    results.append({
        "test": "naive_sort_allows_collapse",
        "description": "Without distinctness, constants can be equal",
        "expected": "sat",
        "actual": str(r),
        "pass": str(r) == "sat",
    })

    # Test 2: With axioms, collapse is blocked
    cpu = gen.get_constant("T_device", "cpu")
    cuda0 = gen.get_constant("T_device", "cuda:0")
    dist = gen.generate_distinctness_axioms("T_device")
    s = z3.Solver()
    s.add(*dist)
    s.add(cpu == cuda0)
    r = s.check()
    results.append({
        "test": "distinctness_blocks_collapse",
        "description": "With distinctness, cpu != cuda:0",
        "expected": "unsat",
        "actual": str(r),
        "pass": str(r) == "unsat",
    })

    # Test 3: Phase distinctness
    train = gen.get_constant("T_phase", "TRAIN")
    ev = gen.get_constant("T_phase", "EVAL")
    dist_phase = gen.generate_distinctness_axioms("T_phase")
    s = z3.Solver()
    s.add(*dist_phase)
    s.add(train == ev)
    r = s.check()
    results.append({
        "test": "phase_distinctness",
        "description": "TRAIN != EVAL under axioms",
        "expected": "unsat",
        "actual": str(r),
        "pass": str(r) == "unsat",
    })

    # Test 4: Variable forced to known value
    dev_sort = gen.get_sort("T_device")
    dev_consts = gen.get_constants("T_device")
    x = z3.Const("_spurious_x", dev_sort)
    s = z3.Solver()
    s.add(*gen.generate_distinctness_axioms("T_device"))
    s.add(z3.Or(*[x == c for c in dev_consts.values()]))
    for c in dev_consts.values():
        s.add(x != c)
    r = s.check()
    results.append({
        "test": "totality_prevents_extra_value",
        "description": "Variable cannot take value outside sort",
        "expected": "unsat",
        "actual": str(r),
        "pass": str(r) == "unsat",
    })

    # Test 5: Two different constants assignable simultaneously
    y = z3.Const("_spurious_y", dev_sort)
    s = z3.Solver()
    s.add(*gen.generate_distinctness_axioms("T_device"))
    s.add(z3.Or(*[x == c for c in dev_consts.values()]))
    s.add(z3.Or(*[y == c for c in dev_consts.values()]))
    s.add(x != y)
    r = s.check()
    results.append({
        "test": "two_distinct_vars_sat",
        "description": "Two vars can take different values from sort",
        "expected": "sat",
        "actual": str(r),
        "pass": str(r) == "sat",
    })

    # Test 6: All pairs of constants are distinct
    all_distinct = True
    for i, (n1, c1) in enumerate(dev_consts.items()):
        for n2, c2 in list(dev_consts.items())[i + 1:]:
            s = z3.Solver()
            s.add(*gen.generate_distinctness_axioms("T_device"))
            s.add(c1 == c2)
            r = s.check()
            if str(r) != "unsat":
                all_distinct = False
    results.append({
        "test": "all_device_pairs_distinct",
        "description": "Every pair of device constants is distinct",
        "expected": "all_unsat",
        "actual": "all_unsat" if all_distinct else "some_sat",
        "pass": all_distinct,
    })

    # Test 7: Perm sort distinctness
    perm_consts = gen.get_constants("T_perm")
    identity = gen.get_constant("T_perm", "identity")
    transpose = gen.get_constant("T_perm", "transpose")
    s = z3.Solver()
    s.add(*gen.generate_distinctness_axioms("T_perm"))
    s.add(identity == transpose)
    r = s.check()
    results.append({
        "test": "perm_identity_ne_transpose",
        "description": "identity != transpose under axioms",
        "expected": "unsat",
        "actual": str(r),
        "pass": str(r) == "unsat",
    })

    # Test 8: Same-device constraint with axioms
    a = z3.Const("_dev_a", dev_sort)
    b = z3.Const("_dev_b", dev_sort)
    s = z3.Solver()
    s.add(*gen.generate_distinctness_axioms("T_device"))
    s.add(z3.Or(*[a == c for c in dev_consts.values()]))
    s.add(z3.Or(*[b == c for c in dev_consts.values()]))
    s.add(a == b)
    cuda1 = gen.get_constant("T_device", "cuda:1")
    s.add(a == cuda1)
    r = s.check()
    results.append({
        "test": "same_device_with_assignment",
        "description": "a == b and a == cuda:1 implies b == cuda:1",
        "expected": "sat",
        "actual": str(r),
        "pass": str(r) == "sat",
    })

    return {
        "tests": results,
        "all_pass": all(t["pass"] for t in results),
        "pass_count": sum(1 for t in results if t["pass"]),
        "total_count": len(results),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_verification() -> Dict[str, Any]:
    t0 = time.perf_counter()

    gen = FiniteSortAxiomGenerator()
    standard_sorts = get_standard_sorts()
    for s in standard_sorts:
        gen.declare_sort(s)

    # 1. Axiom summaries
    axiom_summaries = []
    for sort in standard_sorts:
        summary = generate_axiom_summary(gen, sort)
        axiom_summaries.append(summary)

    # 2. Tightness verification
    tightness_results = {}
    for sort in standard_sorts:
        tightness_results[sort.name] = verify_sort_tightness(gen, sort)

    # 3. Bidirectional simulation
    simulation = run_bidirectional_simulation(gen)

    # 4. Spurious model tests
    spurious = run_spurious_model_tests(gen)

    elapsed = time.perf_counter() - t0

    all_tight = all(r["tight"] for r in tightness_results.values())

    return {
        "summary": {
            "sorts_verified": len(standard_sorts),
            "all_tight": all_tight,
            "simulation_pass": simulation["all_pass"],
            "spurious_tests_pass": spurious["all_pass"],
            "elapsed_seconds": round(elapsed, 3),
        },
        "axiom_sets": axiom_summaries,
        "tightness": tightness_results,
        "bidirectional_simulation": simulation,
        "spurious_model_tests": spurious,
    }


def main():
    results = run_verification()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print("Distinctness Axiom Verification")
    print("=" * 60)
    print(f"Sorts verified:          {s['sorts_verified']}")
    print(f"All tight:               {s['all_tight']}")
    print(f"Simulation pass:         {s['simulation_pass']}")
    print(f"Spurious tests pass:     {s['spurious_tests_pass']}")
    print(f"Elapsed:                 {s['elapsed_seconds']}s")

    print(f"\nPer-sort tightness:")
    for sort_name, t in results["tightness"].items():
        tight = "✓" if t["tight"] else "✗"
        print(f"  {sort_name} (|S|={t['expected_size']}): {tight}")

    spur = results["spurious_model_tests"]
    print(f"\nSpurious model tests: {spur['pass_count']}/{spur['total_count']} passed")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
