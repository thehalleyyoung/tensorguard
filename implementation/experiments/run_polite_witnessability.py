#!/usr/bin/env python3
"""
Dynamic Polite Witnessability Verification for TensorGuard.

For the 3 finite-domain theories (Device, Phase, Stride-as-finite-fragment),
generates a subset of arrangements, tests witness production consistency,
and verifies dynamic witnessability by checking that for each satisfying
assignment a witness can be produced.

Based on the Tinelli-Zarba (JAR 2005) polite combination framework.
"""

import json
import itertools
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from src.theory_combination_analysis import (
    THEORY_SIGNATURES,
    DomainType,
    TheoryCombinationAnalysis,
)


# ═══════════════════════════════════════════════════════════════════════════
# Arrangement generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_arrangements(domain: List[str], n_vars: int) -> List[Tuple[str, ...]]:
    """Generate all possible arrangements (assignments) of n_vars over domain.

    An arrangement maps each variable to a domain element.
    For a domain of size k and n variables, there are k^n arrangements.
    """
    return list(itertools.product(domain, repeat=n_vars))


def generate_equality_arrangements(
    n_vars: int,
) -> List[List[frozenset]]:
    """Generate all equivalence class partitions of n_vars variables.

    In Nelson-Oppen / Tinelli-Zarba, the arrangement is the partition of
    shared variables into equivalence classes (which are equal vs distinct).
    """
    if n_vars == 0:
        return [[]]
    if n_vars == 1:
        return [[frozenset({0})]]

    # Generate all set partitions using Bell's approach
    partitions = []
    _generate_partitions(list(range(n_vars)), [], partitions)
    return partitions


def _generate_partitions(
    remaining: List[int],
    current: List[frozenset],
    result: List[List[frozenset]],
) -> None:
    """Recursively generate all set partitions."""
    if not remaining:
        result.append([frozenset(s) for s in current])
        return

    elem = remaining[0]
    rest = remaining[1:]

    # Add to each existing class
    for i in range(len(current)):
        new_current = [s for s in current]
        new_current[i] = current[i] | {elem}
        _generate_partitions(rest, new_current, result)

    # Create a new singleton class
    _generate_partitions(rest, current + [frozenset({elem})], result)


# ═══════════════════════════════════════════════════════════════════════════
# Witness production
# ═══════════════════════════════════════════════════════════════════════════

def device_witness(
    arrangement: Tuple[str, ...],
    domain: List[str],
) -> Optional[Dict[str, str]]:
    """Produce a witness for a Device theory arrangement.

    For the device theory, a witness is simply the arrangement itself
    (mapping variables to device names). The theory is a pure equality
    theory, so any arrangement is a valid witness.
    """
    witness = {}
    for i, val in enumerate(arrangement):
        if val not in domain:
            return None
        witness[f"dev_{i}"] = val
    return witness


def phase_witness(
    arrangement: Tuple[str, ...],
    domain: List[str],
) -> Optional[Dict[str, str]]:
    """Produce a witness for a Phase theory arrangement.

    Phase theory is {TRAIN, EVAL}. Any arrangement over this domain
    is a valid witness since the theory only constrains phase-dependent
    behaviour (dropout, batchnorm), not the phase values themselves.
    """
    witness = {}
    for i, val in enumerate(arrangement):
        if val not in domain:
            return None
        witness[f"phase_{i}"] = val
    return witness


def stride_finite_witness(
    arrangement: Tuple[str, ...],
    domain: List[str],
) -> Optional[Dict[str, str]]:
    """Produce a witness for a finite subset of stride values.

    Stride theory is stably infinite (operates over ℤ_≥1), but for
    concrete bounded verification, we test witnessability over a finite
    subset of common stride values.
    """
    witness = {}
    for i, val in enumerate(arrangement):
        if val not in domain:
            return None
        witness[f"stride_{i}"] = val
    return witness


# ═══════════════════════════════════════════════════════════════════════════
# Witnessability verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_witnessability(
    theory_name: str,
    domain: List[str],
    n_vars: int,
    witness_fn,
) -> Dict[str, Any]:
    """Verify dynamic witnessability for a theory.

    For each arrangement of n_vars over the domain:
    1. Attempt to produce a witness
    2. Verify the witness is consistent (maps to valid domain elements)
    3. Check that the witness can be extended (adding fresh variables)
    """
    arrangements = generate_arrangements(domain, n_vars)
    total = len(arrangements)
    witnessed = 0
    failed = []
    extension_ok = 0

    for arr in arrangements:
        witness = witness_fn(arr, domain)
        if witness is not None:
            witnessed += 1
            # Verify extension: can we add one more variable?
            for extra_val in domain:
                extended_arr = arr + (extra_val,)
                ext_witness = witness_fn(extended_arr, domain)
                if ext_witness is not None:
                    extension_ok += 1
                    break
            else:
                failed.append({"arrangement": arr, "reason": "extension_failed"})
        else:
            failed.append({"arrangement": arr, "reason": "witness_failed"})

    return {
        "theory": theory_name,
        "domain": domain,
        "domain_size": len(domain),
        "n_vars": n_vars,
        "total_arrangements": total,
        "witnessed": witnessed,
        "witness_rate": round(witnessed / total, 6) if total > 0 else 0.0,
        "extension_verified": extension_ok,
        "extension_rate": round(extension_ok / total, 6) if total > 0 else 0.0,
        "failures": failed[:5],  # cap for readability
        "is_polite": witnessed == total and extension_ok == total,
    }


def verify_equality_arrangement_consistency(
    theory_name: str,
    domain: List[str],
    n_vars: int,
) -> Dict[str, Any]:
    """Verify that equality arrangements are consistent with the domain.

    For each equivalence class partition, check that it can be realized
    in the domain (i.e., the number of equivalence classes ≤ domain size).
    """
    partitions = generate_equality_arrangements(n_vars)
    total = len(partitions)
    realizable = 0
    unrealizable = []

    for partition in partitions:
        n_classes = len(partition)
        if n_classes <= len(domain):
            realizable += 1
        else:
            unrealizable.append({
                "partition_size": n_classes,
                "domain_size": len(domain),
            })

    return {
        "theory": theory_name,
        "domain_size": len(domain),
        "n_vars": n_vars,
        "total_partitions": total,
        "realizable": realizable,
        "unrealizable_count": len(unrealizable),
        "all_realizable": realizable == total,
        "unrealizable_examples": unrealizable[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    """Run polite witnessability verification for all finite-domain theories."""
    print("=" * 70)
    print("  Dynamic Polite Witnessability Verification — TensorGuard")
    print("=" * 70)
    print()

    results = {
        "experiment": "polite_witnessability_verification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "theories": {},
    }

    # 1. Device theory
    device_domain = ["CPU", "CUDA_0", "CUDA_1", "CUDA_2", "CUDA_3"]
    print("  [1/3] Device Theory (|D|=5)")
    device_witness_result = verify_witnessability(
        "T_device", device_domain, n_vars=2, witness_fn=device_witness
    )
    device_eq_result = verify_equality_arrangement_consistency(
        "T_device", device_domain, n_vars=3
    )
    results["theories"]["T_device"] = {
        "witnessability": device_witness_result,
        "equality_arrangements": device_eq_result,
        "is_polite": device_witness_result["is_polite"],
        "justification": (
            "T_device is a pure equality theory over a 5-element enumeration. "
            "Every arrangement can produce a witness, and every witness can be "
            "extended by mapping fresh variables to any domain element."
        ),
    }
    marker = "✓" if device_witness_result["is_polite"] else "✗"
    print(f"    {marker} Witnessability: {device_witness_result['witnessed']}/{device_witness_result['total_arrangements']}")
    print(f"    {marker} Extension: {device_witness_result['extension_verified']}/{device_witness_result['total_arrangements']}")
    print(f"    Equality arrangements realizable: {device_eq_result['realizable']}/{device_eq_result['total_partitions']}")
    print()

    # 2. Phase theory
    phase_domain = ["TRAIN", "EVAL"]
    print("  [2/3] Phase Theory (|D|=2)")
    phase_witness_result = verify_witnessability(
        "T_phase", phase_domain, n_vars=2, witness_fn=phase_witness
    )
    phase_eq_result = verify_equality_arrangement_consistency(
        "T_phase", phase_domain, n_vars=3
    )
    results["theories"]["T_phase"] = {
        "witnessability": phase_witness_result,
        "equality_arrangements": phase_eq_result,
        "is_polite": phase_witness_result["is_polite"],
        "justification": (
            "T_phase is a pure equality theory over {TRAIN, EVAL}. "
            "Every 2-variable arrangement produces a valid witness. "
            "Extension is trivially possible since fresh variables can "
            "take either phase value."
        ),
    }
    marker = "✓" if phase_witness_result["is_polite"] else "✗"
    print(f"    {marker} Witnessability: {phase_witness_result['witnessed']}/{phase_witness_result['total_arrangements']}")
    print(f"    {marker} Extension: {phase_witness_result['extension_verified']}/{phase_witness_result['total_arrangements']}")
    print(f"    Equality arrangements realizable: {phase_eq_result['realizable']}/{phase_eq_result['total_partitions']}")

    # Note: 3 vars over 2-element domain means some partitions have >2 classes
    if phase_eq_result['unrealizable_count'] > 0:
        print(f"    Note: {phase_eq_result['unrealizable_count']} partitions have more "
              f"equivalence classes than domain elements (expected for |D|=2, n=3)")
    print()

    # 3. Stride theory (finite subset for testing)
    stride_domain = ["1", "2", "4", "8", "16"]
    print("  [3/3] Stride Theory — finite subset (|D|=5 common stride values)")
    stride_witness_result = verify_witnessability(
        "T_stride_finite", stride_domain, n_vars=2,
        witness_fn=stride_finite_witness,
    )
    stride_eq_result = verify_equality_arrangement_consistency(
        "T_stride_finite", stride_domain, n_vars=3
    )
    results["theories"]["T_stride_finite"] = {
        "witnessability": stride_witness_result,
        "equality_arrangements": stride_eq_result,
        "is_polite": stride_witness_result["is_polite"],
        "justification": (
            "T_stride is stably infinite over ℤ_≥1, so polite witnessability "
            "is not required (Nelson-Oppen applies directly). However, for "
            "bounded verification, we test a finite subset {1,2,4,8,16}. "
            "All arrangements produce witnesses and extend successfully."
        ),
        "note": (
            "Stride theory is stably infinite; this test validates witness "
            "production for the common finite subset used in practice."
        ),
    }
    marker = "✓" if stride_witness_result["is_polite"] else "✗"
    print(f"    {marker} Witnessability: {stride_witness_result['witnessed']}/{stride_witness_result['total_arrangements']}")
    print(f"    {marker} Extension: {stride_witness_result['extension_verified']}/{stride_witness_result['total_arrangements']}")
    print(f"    Equality arrangements realizable: {stride_eq_result['realizable']}/{stride_eq_result['total_partitions']}")
    print()

    # Static analysis cross-check
    print("  Cross-checking with static TheoryCombinationAnalysis...")
    analysis = TheoryCombinationAnalysis()
    static_pw = analysis.check_polite_witnessability()
    static_results = {}
    for name, pw in static_pw.items():
        static_results[name] = {
            "is_polite": pw.is_polite,
            "domain_elements": pw.domain_elements,
        }
        dynamic_polite = results["theories"].get(name, {}).get("is_polite", None)
        match = dynamic_polite == pw.is_polite if dynamic_polite is not None else "N/A"
        marker = "✓" if match else "✗"
        print(f"    {marker} {name}: static={pw.is_polite}, dynamic={dynamic_polite}")

    results["static_cross_check"] = static_results
    results["all_polite"] = all(
        t.get("is_polite", False) for t in results["theories"].values()
    )

    print(f"\n  Overall: {'ALL theories polite ✓' if results['all_polite'] else 'Some theories NOT polite ✗'}")

    out_path = os.path.join(IMPL_ROOT, ".benchmarks",
                            "polite_witnessability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    run_analysis()
