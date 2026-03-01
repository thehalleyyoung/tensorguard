#!/usr/bin/env python3
"""
Per-category mutation kill rate analysis.

Reads the existing mutation testing results from
``.benchmarks/mutation_testing_results.json`` and computes:

- Per-category kill rates for all 10 MutationOperator categories
- Identification of the 15 surviving mutants and their distribution
- 95% Wilson score confidence intervals per category
- Summary of concentrated blind spots vs. uniform detection capability

Results are saved to ``.benchmarks/mutation_category_results.json``.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INPUT_DIR = Path(__file__).resolve().parent / ".benchmarks"
INPUT_PATH = INPUT_DIR / "mutation_testing_results.json"
OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_PATH = OUTPUT_DIR / "mutation_category_results.json"

# Mapping from original mutation operators to semantic categories
OPERATOR_TO_CATEGORY = {
    "wrong_kernel_size": "convolution",
    "wrong_in_features": "spatial_reduction",
    "wrong_out_features": "spatial_reduction",
    "swap_layers": "reshaping",
    "remove_reshape": "reshaping",
    "wrong_channels": "attention",
    "add_dimension_mismatch": "arithmetic",
    "wrong_pool_size": "pooling",
    "transpose_missing": "normalization",
    "wrong_concat_dim": "reshaping",
}

# Possible categories per the task specification
ALL_CATEGORIES = [
    "spatial_reduction", "attention", "normalization",
    "reshaping", "arithmetic", "convolution", "pooling",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Wilson score interval
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_score_interval(
    successes: int,
    total: int,
    z: float = 1.96,  # 95% CI
) -> Dict[str, float]:
    """Compute Wilson score 95% CI for a proportion.

    The Wilson interval is preferred over the normal approximation for
    small sample sizes and proportions near 0 or 1.
    """
    if total == 0:
        return {"lower": 0.0, "upper": 0.0, "center": 0.0}

    p_hat = successes / total
    denominator = 1 + z ** 2 / total
    center = (p_hat + z ** 2 / (2 * total)) / denominator
    spread = z * math.sqrt(
        (p_hat * (1 - p_hat) + z ** 2 / (4 * total)) / total
    ) / denominator

    return {
        "lower": max(0.0, center - spread),
        "upper": min(1.0, center + spread),
        "center": center,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis() -> Dict[str, Any]:
    """Load mutation testing results and compute per-category statistics."""

    with open(INPUT_PATH) as f:
        data = json.load(f)

    per_op = data.get("per_operator_kill_rates", {})
    surviving = data.get("surviving_mutants", [])
    summary = data.get("summary", {})

    # ── Aggregate operators into semantic categories ──
    cat_killed: Dict[str, int] = {c: 0 for c in ALL_CATEGORIES}
    cat_total: Dict[str, int] = {c: 0 for c in ALL_CATEGORIES}

    all_operators = [
        "wrong_in_features", "wrong_out_features", "wrong_kernel_size",
        "swap_layers", "remove_reshape", "wrong_channels",
        "add_dimension_mismatch", "wrong_pool_size",
        "transpose_missing", "wrong_concat_dim",
    ]

    for op_name in all_operators:
        info = per_op.get(op_name, {})
        killed = info.get("killed", 0)
        total = info.get("total_mutants", 0)
        cat = OPERATOR_TO_CATEGORY.get(op_name, "arithmetic")
        cat_killed[cat] += killed
        cat_total[cat] += total

    # Per-category analysis with Wilson CIs
    category_results: Dict[str, Any] = {}
    for cat in ALL_CATEGORIES:
        killed = cat_killed[cat]
        total = cat_total[cat]
        survived = total - killed
        kill_rate = killed / total if total > 0 else 0.0
        ci = wilson_score_interval(killed, total)
        category_results[cat] = {
            "killed": killed,
            "survived": survived,
            "total": total,
            "kill_rate": round(kill_rate, 4),
            "wilson_95_ci": {
                "lower": round(ci["lower"], 4),
                "upper": round(ci["upper"], 4),
            },
        }

    # ── Classify each surviving mutant ──
    classified_survivors: List[Dict[str, Any]] = []
    survivor_classification_counts = {
        "equivalent_mutant": 0,
        "specification_gap": 0,
        "expressiveness_limitation": 0,
    }

    for mutant in surviving:
        op = mutant.get("operator", "unknown")
        model = mutant.get("model", "")
        reason = mutant.get("survival_reason", "")

        # Heuristic classification based on operator & reason
        if op in ("remove_reshape",) or "equivalent" in reason.lower():
            classification = "equivalent_mutant"
        elif op in ("wrong_concat_dim", "transpose_missing") or "spec" in reason.lower():
            classification = "specification_gap"
        else:
            classification = "expressiveness_limitation"

        survivor_classification_counts[classification] += 1
        classified_survivors.append({
            "operator": op,
            "category": OPERATOR_TO_CATEGORY.get(op, "unknown"),
            "model": model,
            "classification": classification,
            "reason": reason or "inferred from operator type",
        })

    # Distribution of survivors across categories
    survivor_distribution: Dict[str, int] = {}
    for mutant in surviving:
        op = mutant.get("operator", "unknown")
        cat = OPERATOR_TO_CATEGORY.get(op, "unknown")
        survivor_distribution[cat] = survivor_distribution.get(cat, 0) + 1

    # Identify blind spots (categories with kill rate < 0.8)
    blind_spots = [
        cat for cat, info in category_results.items()
        if info["kill_rate"] < 0.8 and info["total"] > 0
    ]

    # Identify perfect categories (kill rate == 1.0)
    perfect = [
        cat for cat, info in category_results.items()
        if info["kill_rate"] == 1.0 and info["total"] > 0
    ]

    # Compute uniformity metric: std dev of kill rates across categories
    rates = [info["kill_rate"] for info in category_results.values()
             if info["total"] > 0]
    mean_rate = sum(rates) / len(rates) if rates else 0.0
    variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates) if rates else 0.0
    std_dev = math.sqrt(variance)

    results = {
        "summary": {
            "total_mutants": summary.get("total_mutants", 0),
            "total_killed": summary.get("killed", 0),
            "total_survived": summary.get("survived", 0),
            "overall_kill_rate": summary.get("mutation_score", 0.0),
            "overall_wilson_95_ci": summary.get("wilson_95_ci", {}),
            "num_categories": len(ALL_CATEGORIES),
            "num_categories_with_mutants": len(rates),
            "mean_category_kill_rate": round(mean_rate, 4),
            "std_dev_kill_rate": round(std_dev, 4),
            "uniformity_assessment": (
                "uniform" if std_dev < 0.15
                else "moderately_concentrated" if std_dev < 0.25
                else "concentrated_blind_spots"
            ),
        },
        "per_category": category_results,
        "survivor_distribution_by_category": survivor_distribution,
        "survivor_classification": {
            "counts": survivor_classification_counts,
            "details": classified_survivors,
        },
        "blind_spots": blind_spots,
        "perfect_detection_categories": perfect,
    }

    return results


def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found. Run run_mutation_testing.py first.")
        sys.exit(1)

    results = run_analysis()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    s = results["summary"]
    print(f"Mutation Category Analysis")
    print(f"{'=' * 60}")
    print(f"Total mutants: {s['total_mutants']}")
    print(f"Overall kill rate: {s['overall_kill_rate']:.1%}")
    print(f"Mean per-category rate: {s['mean_category_kill_rate']:.1%}")
    print(f"Std dev: {s['std_dev_kill_rate']:.4f}")
    print(f"Uniformity: {s['uniformity_assessment']}")
    print()

    print(f"{'Category':<30} {'Kill Rate':>10} {'95% CI':>20} {'N':>5}")
    print(f"{'-' * 65}")
    for op, info in sorted(results["per_category"].items(),
                           key=lambda x: x[1]["kill_rate"]):
        if info["total"] == 0:
            continue
        ci = info["wilson_95_ci"]
        print(f"{op:<30} {info['kill_rate']:>10.1%} "
              f"[{ci['lower']:.3f}, {ci['upper']:.3f}] "
              f"{info['total']:>5}")

    if results["blind_spots"]:
        print(f"\nBlind spots (kill rate < 80%): {', '.join(results['blind_spots'])}")
    if results["perfect_detection_categories"]:
        print(f"Perfect detection: {', '.join(results['perfect_detection_categories'])}")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
