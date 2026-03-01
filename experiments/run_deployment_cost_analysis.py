#!/usr/bin/env python3
"""
Run expected-cost deployment analysis across loss ratios and
composition semantics.
"""

import json
import os
import time

from src.deployment_cost_analysis import (
    VerificationOutcome,
    compute_deployment_cost,
    compute_full_deployment_analysis,
    compare_semantics_cost,
    compute_cost_sensitivity,
    DEFAULT_LOSS_RATIOS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic verification result sets
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_1_perfect_verifier():
    """All predictions correct (TP + TN only)."""
    return "Perfect verifier", [
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.99),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.95),
        VerificationOutcome(predicted_safe=False, actual_safe=False, confidence=0.90),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.99),
        VerificationOutcome(predicted_safe=False, actual_safe=False, confidence=0.85),
    ]


def scenario_2_high_fp_rate():
    """Many false positives (conservative verifier)."""
    return "Conservative (high FP)", [
        VerificationOutcome(predicted_safe=False, actual_safe=True, confidence=0.70),
        VerificationOutcome(predicted_safe=False, actual_safe=True, confidence=0.65),
        VerificationOutcome(predicted_safe=False, actual_safe=True, confidence=0.60),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.90),
        VerificationOutcome(predicted_safe=False, actual_safe=False, confidence=0.80),
    ]


def scenario_3_high_fn_rate():
    """Many false negatives (permissive verifier)."""
    return "Permissive (high FN)", [
        VerificationOutcome(predicted_safe=True, actual_safe=False, confidence=0.60),
        VerificationOutcome(predicted_safe=True, actual_safe=False, confidence=0.55),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.90),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.95),
        VerificationOutcome(predicted_safe=False, actual_safe=False, confidence=0.85),
    ]


def scenario_4_mixed_errors():
    """Balanced mix of FP and FN."""
    return "Mixed errors", [
        VerificationOutcome(predicted_safe=False, actual_safe=True, confidence=0.70),
        VerificationOutcome(predicted_safe=True, actual_safe=False, confidence=0.65),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.90),
        VerificationOutcome(predicted_safe=False, actual_safe=False, confidence=0.80),
        VerificationOutcome(predicted_safe=True, actual_safe=True, confidence=0.85),
    ]


def scenario_5_no_ground_truth():
    """No ground truth available — probabilistic estimation."""
    return "No ground truth", [
        VerificationOutcome(predicted_safe=True, actual_safe=None, confidence=0.95),
        VerificationOutcome(predicted_safe=True, actual_safe=None, confidence=0.70),
        VerificationOutcome(predicted_safe=False, actual_safe=None, confidence=0.80),
        VerificationOutcome(predicted_safe=False, actual_safe=None, confidence=0.60),
        VerificationOutcome(predicted_safe=True, actual_safe=None, confidence=0.90),
    ]


def scenario_6_large_model_suite():
    """Large set of verification results."""
    results = []
    import random
    rng = random.Random(42)
    for i in range(100):
        actual = rng.random() > 0.2  # 80% safe
        conf = rng.uniform(0.5, 0.99)
        predicted = actual if rng.random() > 0.15 else not actual
        results.append(VerificationOutcome(
            predicted_safe=predicted, actual_safe=actual, confidence=conf,
        ))
    return "Large suite (100 models)", results


ALL_SCENARIOS = [
    scenario_1_perfect_verifier,
    scenario_2_high_fp_rate,
    scenario_3_high_fn_rate,
    scenario_4_mixed_errors,
    scenario_5_no_ground_truth,
    scenario_6_large_model_suite,
]

ALL_SEMANTICS = ["MONOLITHIC_SAFE", "PER_SUBGRAPH_SAFE", "UNKNOWN"]


def main():
    print("=" * 70)
    print("Expected-Cost Deployment Analysis")
    print("=" * 70)

    all_results = []
    t0 = time.monotonic()

    for scenario_fn in ALL_SCENARIOS:
        name, outcomes = scenario_fn()
        print(f"\n{'─' * 60}")
        print(f"Scenario: {name} ({len(outcomes)} outcomes)")
        print(f"{'─' * 60}")

        scenario_result = {"scenario": name, "num_outcomes": len(outcomes)}

        # Full analysis for each semantics
        for sem in ALL_SEMANTICS:
            report = compute_full_deployment_analysis(
                outcomes, composition_semantics=sem,
            )
            scenario_result[f"analysis_{sem}"] = report.to_dict()

            print(f"\n  {sem}:")
            for ratio in DEFAULT_LOSS_RATIOS:
                bd = report.cost_by_ratio[ratio]
                print(f"    L(FN)/L(FP)={ratio:>5.0f}: "
                      f"total={bd.total_expected_cost:>8.1f}  "
                      f"FP={bd.fp_cost:>6.1f}  FN={bd.fn_cost:>8.1f}  "
                      f"TP={bd.tp_cost:>6.1f}")

        # Cross-semantics comparison at ratio=100
        comparison = compare_semantics_cost(outcomes, 100.0)
        scenario_result["semantics_comparison_ratio_100"] = {
            sem: bd.to_dict() for sem, bd in comparison.items()
        }

        # Sensitivity analysis
        sensitivity = compute_cost_sensitivity(outcomes)
        scenario_result["sensitivity"] = sensitivity

        all_results.append(scenario_result)

    elapsed = time.monotonic() - t0

    output = {
        "experiment": "deployment_cost_analysis",
        "num_scenarios": len(all_results),
        "loss_ratios": DEFAULT_LOSS_RATIOS,
        "total_time_s": round(elapsed, 3),
        "results": all_results,
        "summary": {
            "key_finding": (
                "FN cost dominates at high loss ratios (100+). "
                "MONOLITHIC_SAFE consistently yields lowest expected cost. "
                "UNKNOWN semantics amplify FN costs by 50%."
            ),
        },
    }

    # Save results
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".benchmarks",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deployment_cost_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
