"""
Experiment: Nelson-Oppen/Tinelli-Zarba Precondition Verification.

Generates the formal precondition verification report for all five
theories in TensorGuard's SMT combination, addressing Chang's and
Sinha's concerns about unverified combination preconditions.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smt.theory_combination import verify_combination_preconditions


def main():
    print("=" * 70)
    print("Nelson-Oppen/Tinelli-Zarba Precondition Verification")
    print("=" * 70)

    report = verify_combination_preconditions()

    print(f"\nAll preconditions satisfied: {report.all_satisfied}")
    print()

    print("1. STABLE INFINITENESS")
    print("-" * 40)
    for name, entry in sorted(report.stable_infiniteness.items()):
        status = "✓" if entry["satisfied"] else "✗"
        print(f"  {status} {name} (sort: {entry['sort']})")
        print(f"    {entry['justification'][:120]}")
    print()

    print("2. POLITE WITNESSABILITY (Tinelli-Zarba)")
    print("-" * 40)
    for name, entry in sorted(report.polite_witnessability.items()):
        status = "✓" if entry["satisfied"] else "✗"
        print(f"  {status} {name} (|D|={entry['domain_size']}, elements={entry['elements']})")
        print(f"    {entry['justification'][:120]}")
    print()

    print("3. SIGNATURE DISJOINTNESS")
    print("-" * 40)
    for pair_key, entry in sorted(report.signature_disjointness.items()):
        status = "✓" if entry["disjoint"] else "✗"
        shared = entry["shared_symbols"]
        print(f"  {status} {pair_key}: {'disjoint' if entry['disjoint'] else f'SHARED: {shared}'}")
    print()

    print("4. SHARED SORT ANALYSIS")
    print("-" * 40)
    for sort_name, info in sorted(report.shared_sort_analysis.items()):
        print(f"  Sort '{sort_name}': {info['theories']}")
        print(f"    Combination method: {info['combination_method']}")
        if info.get("note"):
            print(f"    Note: {info['note'][:120]}")
    print()

    # Save results
    results = report.to_dict()
    results["summary"] = {
        "num_theories": 5,
        "num_pairs_checked": len(report.signature_disjointness),
        "all_disjoint": all(
            e["disjoint"] for e in report.signature_disjointness.values()
        ),
        "finite_theories": ["device", "phase"],
        "infinite_theories": ["broadcast", "stride", "permutation"],
    }

    outpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "nelson_oppen_precondition_results.json",
    )
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {outpath}")


if __name__ == "__main__":
    main()
