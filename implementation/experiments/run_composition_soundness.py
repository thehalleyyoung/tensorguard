"""
Run the 5-theory product domain composition soundness verification.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.composition_soundness import (
    verify_product_domain_soundness,
    verify_composition_properties_z3,
    compute_arrangement_complexity,
    ALL_THEORIES,
)

if __name__ == "__main__":
    print("=" * 60)
    print("5-Theory Product Domain Soundness Verification")
    print("=" * 60)

    # Run main soundness verification
    verdict = verify_product_domain_soundness()
    print(f"\nVerdict: {'SOUND ✓' if verdict.sound else 'UNSOUND ✗'}")
    print(f"Combination method: {verdict.combination_method}")
    print(f"Complexity bound: {verdict.complexity_bound}")
    print(f"Verification time: {verdict.verification_time_ms:.2f} ms")

    print("\nPreconditions:")
    for p in verdict.preconditions:
        status = "✓" if p.satisfied else "✗"
        print(f"  {status} {p.name}: {p.details[:80]}...")

    # Run Z3 verification
    print("\n" + "=" * 60)
    print("Z3 Composition Property Verification")
    print("=" * 60)

    z3_results = verify_composition_properties_z3()
    for key, val in z3_results.items():
        if key == "verification_time_ms":
            print(f"\nZ3 verification time: {val:.2f} ms")
        else:
            print(f"  {key}: {val}")

    # Complexity analysis
    print("\n" + "=" * 60)
    print("Arrangement Complexity Analysis")
    print("=" * 60)

    complexity = compute_arrangement_complexity(ALL_THEORIES)
    print(f"Total arrangements: {complexity['total_arrangements']}")
    print(f"Tractable: {complexity['tractable']}")
    for sort_name, info in complexity["per_sort"].items():
        print(f"  {sort_name}: {info['formula']}")

    # Save results
    output = {
        "soundness_verdict": {
            "sound": verdict.sound,
            "combination_method": verdict.combination_method,
            "complexity_bound": verdict.complexity_bound,
            "verification_time_ms": verdict.verification_time_ms,
            "preconditions": [
                {
                    "name": p.name,
                    "satisfied": p.satisfied,
                    "details": p.details,
                }
                for p in verdict.preconditions
            ],
        },
        "z3_properties": {
            k: v for k, v in z3_results.items()
        },
        "complexity_analysis": complexity,
        "proof_sketch": verdict.proof_sketch,
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "composition_soundness_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")
