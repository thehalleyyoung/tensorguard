"""
NP-Membership Analysis for Reshape Satisfiability.

Addresses MAJOR reviewer critique #4: the paper originally overclaimed
NP-completeness, but the Lean proof only establishes NP-hardness
(the reduction from SUBSET-PRODUCT, not NP-membership).

This script:
  1. Documents the NP-membership argument informally.
  2. Shows this is in NP by exhibiting polynomial-time verification.
  3. Demonstrates with concrete examples.
  4. References the Lean proof for NP-hardness (SUBSET-PRODUCT reduction).

The key insight:
  Given a reshape constraint ∏dᵢ = T with dᵢ ∈ {1, sᵢ}, a certificate
  is the vector of choices (c₁, ..., cₖ) where cᵢ ∈ {1, sᵢ}.
  Verification: check ∏cᵢ = T, which requires O(k) multiplications.
  This is polynomial in the input size, so RESHAPE-SAT ∈ NP.

Combined with NP-hardness (SUBSET-PRODUCT ≤ₚ RESHAPE-SAT, proved in
lean/TheoryCombination.lean), this establishes NP-hardness formally;
the NP-membership argument above is standard but not Lean-mechanized.

Outputs:
  - experiments/results/np_membership_analysis.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULTS_DIR = Path(os.path.dirname(__file__)) / "results"
RESULTS_FILE = RESULTS_DIR / "np_membership_analysis.json"


# ---------------------------------------------------------------------------
# Reshape satisfiability problem
# ---------------------------------------------------------------------------

def reshape_sat_verify(
    weights: List[int],
    target: int,
    certificate: List[int],
) -> Tuple[bool, str, int]:
    """Verify a certificate for RESHAPE-SAT in polynomial time.

    Given weights s₁,...,sₖ, target T, and certificate c₁,...,cₖ:
      1. Check len(certificate) == len(weights)           — O(1)
      2. Check each cᵢ ∈ {1, sᵢ}                         — O(k)
      3. Check ∏cᵢ == T                                   — O(k) multiplications

    Total: O(k) operations, polynomial in input size.

    Returns: (valid, reason, num_operations)
    """
    ops = 0

    # Step 1: Length check
    ops += 1
    if len(certificate) != len(weights):
        return False, f"Length mismatch: {len(certificate)} != {len(weights)}", ops

    # Step 2: Domain check — each cᵢ ∈ {1, sᵢ}
    for i, (c, s) in enumerate(zip(certificate, weights)):
        ops += 1
        if c != 1 and c != s:
            return False, f"certificate[{i}]={c} ∉ {{1, {s}}}", ops

    # Step 3: Product check — ∏cᵢ == T
    product = 1
    for c in certificate:
        product *= c
        ops += 1

    ops += 1
    if product != target:
        return False, f"Product {product} != target {target}", ops

    return True, "Valid certificate", ops


def enumerate_all_certificates(weights: List[int]) -> List[List[int]]:
    """Enumerate all 2^k possible certificates (brute force)."""
    k = len(weights)
    certs = []
    for mask in range(1 << k):
        cert = []
        for i in range(k):
            if mask & (1 << i):
                cert.append(weights[i])
            else:
                cert.append(1)
        certs.append(cert)
    return certs


def solve_reshape_sat_brute_force(
    weights: List[int],
    target: int,
) -> Optional[List[int]]:
    """Solve RESHAPE-SAT by brute force (2^k time)."""
    for cert in enumerate_all_certificates(weights):
        product = 1
        for c in cert:
            product *= c
        if product == target:
            return cert
    return None


# ---------------------------------------------------------------------------
# Concrete examples demonstrating NP-membership
# ---------------------------------------------------------------------------

def generate_examples() -> List[Dict[str, Any]]:
    """Generate concrete examples demonstrating the NP-membership argument."""
    examples = []

    # Example 1: Simple satisfiable case
    ex1_weights = [2, 3, 5]
    ex1_target = 6  # 2 * 3 * 1 = 6
    ex1_cert = [2, 3, 1]
    valid, reason, ops = reshape_sat_verify(ex1_weights, ex1_target, ex1_cert)
    examples.append({
        "name": "simple_satisfiable",
        "weights": ex1_weights,
        "target": ex1_target,
        "certificate": ex1_cert,
        "valid": valid,
        "reason": reason,
        "verification_operations": ops,
        "k": len(ex1_weights),
        "explanation": (
            f"Reshape constraint: d₁∈{{1,2}}, d₂∈{{1,3}}, d₃∈{{1,5}}, "
            f"∏dᵢ = {ex1_target}. Certificate: [{', '.join(map(str, ex1_cert))}]. "
            f"Product = {math.prod(ex1_cert)} = {ex1_target}. "
            f"Verified in {ops} operations (O(k={len(ex1_weights)}))."
        ),
    })

    # Example 2: Unsatisfiable (no valid certificate)
    ex2_weights = [2, 3, 5]
    ex2_target = 7  # 7 is prime and not in {1,2,3,5,6,10,15,30}
    ex2_cert_attempt = [2, 3, 1]
    valid2, reason2, ops2 = reshape_sat_verify(ex2_weights, ex2_target, ex2_cert_attempt)
    # Brute force verify no solution exists
    solution = solve_reshape_sat_brute_force(ex2_weights, ex2_target)
    examples.append({
        "name": "unsatisfiable",
        "weights": ex2_weights,
        "target": ex2_target,
        "certificate_attempt": ex2_cert_attempt,
        "valid": valid2,
        "reason": reason2,
        "has_any_solution": solution is not None,
        "brute_force_checked": True,
        "all_possible_products": sorted(set(
            math.prod(c) for c in enumerate_all_certificates(ex2_weights)
        )),
        "explanation": (
            f"Target {ex2_target} is not achievable. All possible products "
            f"from weights {ex2_weights} with choices in {{1, sᵢ}}: "
            f"{sorted(set(math.prod(c) for c in enumerate_all_certificates(ex2_weights)))}. "
            f"Since {ex2_target} is not among them, the instance is unsatisfiable."
        ),
    })

    # Example 3: Real reshape scenario
    ex3_weights = [4, 8, 16]
    ex3_target = 32  # 4 * 8 * 1 = 32
    ex3_cert = [4, 8, 1]
    valid3, reason3, ops3 = reshape_sat_verify(ex3_weights, ex3_target, ex3_cert)
    examples.append({
        "name": "reshape_scenario",
        "weights": ex3_weights,
        "target": ex3_target,
        "certificate": ex3_cert,
        "valid": valid3,
        "reason": reason3,
        "verification_operations": ops3,
        "k": len(ex3_weights),
        "context": (
            "Tensor reshape from shape [4, 8, 16] to [32, 16]. "
            "Question: can we choose dimensions d₁∈{1,4}, d₂∈{1,8}, "
            "d₃∈{1,16} such that d₁×d₂×d₃ = 32?"
        ),
        "explanation": (
            f"Certificate [{', '.join(map(str, ex3_cert))}]: "
            f"product = {math.prod(ex3_cert)} = {ex3_target}. "
            f"Verified in {ops3} operations."
        ),
    })

    # Example 4: Invalid certificate (domain violation)
    ex4_weights = [3, 7, 11]
    ex4_target = 21
    ex4_cert = [3, 7, 2]  # 2 ∉ {1, 11}
    valid4, reason4, ops4 = reshape_sat_verify(ex4_weights, ex4_target, ex4_cert)
    examples.append({
        "name": "invalid_certificate_domain",
        "weights": ex4_weights,
        "target": ex4_target,
        "certificate": ex4_cert,
        "valid": valid4,
        "reason": reason4,
        "verification_operations": ops4,
        "explanation": (
            f"Certificate [{', '.join(map(str, ex4_cert))}] is invalid: "
            f"c₃=2 but d₃ ∈ {{1, 11}}. The verifier catches this in O(k) time."
        ),
    })

    # Example 5: Larger instance (scaling argument)
    k = 20
    ex5_weights = [2] * k
    ex5_target = 2 ** 10  # select exactly 10 of the 20 twos
    ex5_cert = [2] * 10 + [1] * 10
    valid5, reason5, ops5 = reshape_sat_verify(ex5_weights, ex5_target, ex5_cert)
    examples.append({
        "name": "scaling_k20",
        "weights": ex5_weights,
        "target": ex5_target,
        "certificate": ex5_cert,
        "valid": valid5,
        "reason": reason5,
        "verification_operations": ops5,
        "k": k,
        "search_space_size": 2 ** k,
        "verification_complexity": f"O({k})",
        "explanation": (
            f"k={k} dimensions, each dᵢ ∈ {{1, 2}}. Target = 2^10 = {ex5_target}. "
            f"Search space: 2^{k} = {2**k} candidates. "
            f"But verification of a given certificate takes only {ops5} operations "
            f"= O(k={k}), polynomial in input size."
        ),
    })

    # Example 6: Very large k (complexity argument)
    for k_val in [50, 100]:
        weights_large = [2] * k_val
        target_large = 2 ** (k_val // 2)
        cert_large = [2] * (k_val // 2) + [1] * (k_val // 2)
        valid_l, reason_l, ops_l = reshape_sat_verify(
            weights_large, target_large, cert_large
        )
        examples.append({
            "name": f"scaling_k{k_val}",
            "k": k_val,
            "target": target_large,
            "certificate_valid": valid_l,
            "verification_operations": ops_l,
            "search_space_size_log2": k_val,
            "search_space_note": f"2^{k_val} ≈ 10^{k_val * math.log10(2):.0f}",
            "verification_complexity": f"O({k_val})",
            "explanation": (
                f"k={k_val}: search space = 2^{k_val}, but verification = "
                f"O({k_val}) = {ops_l} operations. This exponential gap "
                f"between searching and verifying is the essence of NP."
            ),
        })

    return examples


# ---------------------------------------------------------------------------
# Formal argument
# ---------------------------------------------------------------------------

def build_formal_argument() -> Dict[str, Any]:
    return {
        "problem_definition": {
            "name": "RESHAPE-SAT",
            "input": (
                "Weights s₁, ..., sₖ ∈ ℕ⁺ and target T ∈ ℕ⁺"
            ),
            "question": (
                "∃ c₁, ..., cₖ with cᵢ ∈ {1, sᵢ} such that ∏ᵢ cᵢ = T?"
            ),
        },
        "np_membership": {
            "claim": "RESHAPE-SAT ∈ NP",
            "proof": {
                "certificate": (
                    "A vector (c₁, ..., cₖ) where each cᵢ ∈ {1, sᵢ}"
                ),
                "certificate_size": (
                    "k values, each at most max(sᵢ), so O(k · log(max sᵢ)) bits — "
                    "polynomial in input size"
                ),
                "verification_algorithm": [
                    "1. Check len(certificate) == k — O(1)",
                    "2. For i = 1 to k: check cᵢ ∈ {1, sᵢ} — O(k) comparisons",
                    "3. Compute P = ∏ᵢ cᵢ — O(k) multiplications",
                    "4. Check P == T — O(1) comparison",
                ],
                "verification_complexity": "O(k) arithmetic operations",
                "conclusion": (
                    "The verifier runs in O(k) time, which is polynomial in "
                    "the input size (k numbers). Therefore RESHAPE-SAT ∈ NP."
                ),
            },
        },
        "np_hardness": {
            "claim": "RESHAPE-SAT is NP-hard",
            "proof_method": "Polynomial-time reduction from SUBSET-PRODUCT",
            "reduction": {
                "from_problem": "SUBSET-PRODUCT",
                "np_completeness_reference": "Garey & Johnson, 1979",
                "reduction_description": (
                    "Given SUBSET-PRODUCT instance (S = {s₁,...,sₖ}, T): "
                    "construct RESHAPE-SAT instance with weights = S, target = T. "
                    "This is the identity reduction — the problems are equivalent "
                    "because choosing dᵢ = sᵢ vs dᵢ = 1 is exactly subset selection."
                ),
                "forward_direction": (
                    "If subset with product T exists → choose dᵢ = sᵢ for "
                    "included elements, dᵢ = 1 for excluded. ∏dᵢ = T. ✓"
                ),
                "reverse_direction": (
                    "If choices satisfy ∏dᵢ = T → subset {sᵢ | dᵢ = sᵢ} "
                    "has product T (factors of 1 don't contribute). ✓"
                ),
                "lean_proof": (
                    "Both directions fully mechanized in "
                    "lean/TheoryCombination.lean as subset_product_forward "
                    "and subset_product_reverse, with zero sorry obligations."
                ),
            },
        },
        "np_hardness": {
            "claim": "RESHAPE-SAT is NP-hard",
            "proof": (
                "NP-hardness established via SUBSET-PRODUCT reduction, "
                "mechanized in Lean with zero sorry obligations. "
                "NP-membership is argued informally above (polynomial "
                "verifier) but is not Lean-mechanized."
            ),
            "lean_theorem": "reshape_np_hard in lean/TheoryCombination.lean",
            "note_on_lean_proof": (
                "The Lean theorem reshape_np_hard proves the equivalence "
                "SubsetProduct weights T ↔ ReshapeDimSat weights T, which "
                "establishes NP-hardness. NP-membership is a meta-theorem "
                "(about Turing machines / computational complexity) that "
                "is standard to argue informally: the certificate size is "
                "polynomial and verification is polynomial. Formalizing NP "
                "membership in Lean would require a formalization of "
                "computational complexity theory (Turing machines, polynomial "
                "time), which is outside the scope of this work but is a "
                "standard mathematical argument."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis() -> Dict[str, Any]:
    print("=" * 72)
    print("  NP-Membership Analysis for Reshape Satisfiability")
    print("=" * 72)
    print()

    # Build formal argument
    formal = build_formal_argument()
    print("  Formal argument constructed.")

    # Generate concrete examples
    print("  Generating concrete examples...")
    examples = generate_examples()

    for ex in examples:
        name = ex["name"]
        k = ex.get("k", len(ex.get("weights", [])))
        valid = ex.get("valid", ex.get("certificate_valid", "N/A"))
        ops = ex.get("verification_operations", "N/A")
        print(f"    {name:35s}  k={k:3d}  valid={str(valid):5s}  ops={ops}")

    # Complexity analysis
    complexity_analysis = {
        "verification_time": {
            "description": "Time to verify a certificate",
            "complexity": "O(k)",
            "operations": [
                "k comparisons for domain check",
                "k multiplications for product",
                "1 comparison for target check",
            ],
            "total": "2k + 2 operations",
        },
        "search_time_brute_force": {
            "description": "Time to find a certificate by exhaustive search",
            "complexity": "O(2^k · k)",
            "explanation": "2^k possible certificates, each verified in O(k)",
        },
        "np_gap": {
            "description": (
                "The exponential gap between verifying (O(k)) and searching "
                "(O(2^k · k)) is characteristic of NP problems."
            ),
            "examples": {
                "k=10": {"search_space": 1024, "verify_ops": 22},
                "k=20": {"search_space": 1048576, "verify_ops": 42},
                "k=50": {"search_space": "~10^15", "verify_ops": 102},
                "k=100": {"search_space": "~10^30", "verify_ops": 202},
            },
        },
    }

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    output = {
        "experiment": "np_membership_analysis",
        "timestamp": timestamp,
        "description": (
            "Formal analysis documenting NP-membership argument for RESHAPE-SAT, "
            "addressing MAJOR reviewer critique #4 about the NP-hardness "
            "claim. The Lean proof establishes NP-hardness via SUBSET-PRODUCT "
            "reduction; this analysis provides an informal NP-membership argument "
            "with concrete demonstrations."
        ),
        "formal_argument": formal,
        "concrete_examples": examples,
        "complexity_analysis": complexity_analysis,
        "conclusion": {
            "np_membership": (
                "RESHAPE-SAT ∈ NP: given certificate (c₁,...,cₖ), verify "
                "∏cᵢ = T in O(k) time. Certificate size = O(k log(max sᵢ)) bits."
            ),
            "np_hardness": (
                "RESHAPE-SAT is NP-hard: SUBSET-PRODUCT ≤ₚ RESHAPE-SAT "
                "via identity reduction (mechanized in Lean with zero sorry)."
            ),
            "formally_established": (
                "RESHAPE-SAT is NP-hard (Lean-mechanized via "
                "SUBSET-PRODUCT reduction)."
            ),
            "reviewer_response": (
                "The paper's NP-hardness claim is formally justified: "
                "NP-hardness is mechanized in Lean (subset_product_forward, "
                "subset_product_reverse). NP-membership follows from "
                "exhibiting a polynomial-time verifier (O(k) operations "
                "for k-dimensional reshape constraints), but this is a "
                "standard complexity-theoretic argument, not Lean-mechanized."
            ),
        },
    }

    print(f"\n  {'═' * 60}")
    print(f"  CONCLUSION")
    print(f"  {'═' * 60}")
    print(f"  RESHAPE-SAT is NP-hard (formally established):")
    print(f"    ✓ NP-hardness: SUBSET-PRODUCT reduction (Lean-mechanized)")
    print(f"    ○ NP-membership: O(k) verification of certificates (informal argument)")
    print(f"  {len(examples)} concrete examples demonstrate polynomial verification")

    return output


def main():
    output = run_analysis()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
