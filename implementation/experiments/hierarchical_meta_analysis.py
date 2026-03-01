"""
Hierarchical Bayesian Meta-Analysis across Evaluation Suites.

Implements a random-effects meta-analysis using DerSimonian-Laird estimation
to integrate F1 results across evaluation suites B, C, D, and the deep
composition benchmark.  Produces:
  - Wilson score confidence intervals per suite
  - DerSimonian-Laird pooled estimate with 95% CI
  - I² heterogeneity statistic
  - Cochran's Q test
  - Forest plot data (numeric, no matplotlib dependency)

Per-suite inputs:
  Suite B (CEGAR ablation):     F1 = 0.966, n = 32
  Suite C (external bugs):      F1 = 0.875, n = 34
  Suite D (standard):           F1 = 0.917, n = 50
  Deep composition:             F1 = 1.000, n = 25
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy import stats

RESULTS_FILE = Path(__file__).parent / "hierarchical_results.json"


# ---------------------------------------------------------------------------
# Wilson score interval for a proportion
# ---------------------------------------------------------------------------

def wilson_interval(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    More accurate than Wald intervals for extreme p or small n.
    """
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)
    return (round(lo, 4), round(hi, 4))


# ---------------------------------------------------------------------------
# Variance of F1 via delta method (logit transform)
# ---------------------------------------------------------------------------

def f1_variance(f1: float, n: int) -> float:
    """
    Approximate sampling variance of F1 score.

    Uses the delta-method approximation treating F1 as a proportion-like
    statistic: Var(F1) ≈ F1*(1-F1)/n.  For F1=1.0, we apply a continuity
    correction: F1_adj = (n*F1 + 0.5)/(n+1) to avoid zero variance.
    """
    if f1 >= 1.0:
        f1_adj = (n * f1 + 0.5) / (n + 1)
    elif f1 <= 0.0:
        f1_adj = 0.5 / (n + 1)
    else:
        f1_adj = f1
    return f1_adj * (1 - f1_adj) / n


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis
# ---------------------------------------------------------------------------

def dersimonian_laird(
    estimates: List[float],
    variances: List[float],
) -> Dict[str, Any]:
    """
    DerSimonian-Laird random-effects pooled estimator.

    Returns:
        pooled_estimate, pooled_se, pooled_ci_lo, pooled_ci_hi,
        tau_squared, Q, I_squared, p_heterogeneity
    """
    k = len(estimates)
    assert k == len(variances) and k >= 2

    # Fixed-effect weights: w_i = 1/v_i
    weights = [1.0 / v for v in variances]
    W = sum(weights)

    # Fixed-effect pooled estimate
    theta_fe = sum(w * e for w, e in zip(weights, estimates)) / W

    # Cochran's Q statistic
    Q = sum(w * (e - theta_fe) ** 2 for w, e in zip(weights, estimates))

    # Degrees of freedom
    df = k - 1

    # p-value for heterogeneity test
    p_het = 1.0 - stats.chi2.cdf(Q, df)

    # DerSimonian-Laird tau² estimate
    C = W - sum(w * w for w in weights) / W
    tau_sq = max(0.0, (Q - df) / C)

    # Random-effects weights: w_i* = 1/(v_i + tau²)
    re_weights = [1.0 / (v + tau_sq) for v in variances]
    W_star = sum(re_weights)

    # Random-effects pooled estimate
    theta_re = sum(w * e for w, e in zip(re_weights, estimates)) / W_star

    # Standard error of pooled estimate
    se_re = math.sqrt(1.0 / W_star)

    # 95% CI
    z = 1.96
    ci_lo = theta_re - z * se_re
    ci_hi = theta_re + z * se_re

    # I² heterogeneity
    I_sq = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    return {
        "pooled_estimate": round(theta_re, 4),
        "pooled_se": round(se_re, 4),
        "pooled_ci_95": [round(max(0, ci_lo), 4), round(min(1, ci_hi), 4)],
        "tau_squared": round(tau_sq, 6),
        "cochrans_Q": round(Q, 4),
        "Q_df": df,
        "p_heterogeneity": round(p_het, 4),
        "I_squared_pct": round(I_sq, 1),
        "fixed_effect_estimate": round(theta_fe, 4),
        "method": "DerSimonian-Laird",
    }


# ---------------------------------------------------------------------------
# Forest plot data (numeric representation)
# ---------------------------------------------------------------------------

def forest_plot_data(
    suite_results: List[Dict[str, Any]],
    pooled: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate forest plot data as a list of rows.

    Each row: label, estimate, ci_lo, ci_hi, weight, n
    Plus a 'pooled' row at the end.
    """
    rows = []
    total_w = sum(s["re_weight"] for s in suite_results)
    for s in suite_results:
        rows.append({
            "label": s["suite"],
            "estimate": s["f1"],
            "ci_lo": s["wilson_ci"][0],
            "ci_hi": s["wilson_ci"][1],
            "weight_pct": round(s["re_weight"] / total_w * 100, 1),
            "n": s["n"],
        })
    rows.append({
        "label": "Pooled (RE)",
        "estimate": pooled["pooled_estimate"],
        "ci_lo": pooled["pooled_ci_95"][0],
        "ci_hi": pooled["pooled_ci_95"][1],
        "weight_pct": 100.0,
        "n": sum(s["n"] for s in suite_results),
    })
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _build_interpretation(pooled: Dict[str, Any], het_interp: str) -> str:
    """Build the natural-language interpretation string."""
    base = (
        f"The random-effects pooled F1 across four evaluation suites is "
        f"{pooled['pooled_estimate']:.3f} "
        f"(95% CI: [{pooled['pooled_ci_95'][0]:.3f}, {pooled['pooled_ci_95'][1]:.3f}]). "
        f"I² = {pooled['I_squared_pct']:.1f}% indicates {het_interp}. "
    )
    p_het = pooled["p_heterogeneity"]
    if p_het > 0.05:
        base += (
            f"The non-significant Q test (p={p_het:.3f}) suggests that "
            f"differences across suites may be attributable to sampling variability."
        )
    else:
        base += (
            "The significant Q test suggests genuine heterogeneity across "
            "suites, likely reflecting the intentional difficulty gradient "
            "from curated to external benchmarks."
        )
    return base


def run_analysis() -> Dict[str, Any]:
    """Run hierarchical random-effects meta-analysis."""
    print("=" * 72)
    print("  Hierarchical Random-Effects Meta-Analysis")
    print("=" * 72)

    # Per-suite data
    suites = [
        {"suite": "Suite B (CEGAR ablation)", "f1": 0.966, "n": 32},
        {"suite": "Suite C (external bugs)",  "f1": 0.875, "n": 34},
        {"suite": "Suite D (standard)",       "f1": 0.917, "n": 50},
        {"suite": "Deep composition",         "f1": 1.000, "n": 25},
    ]

    # Step 1: Wilson intervals and variances
    print("\n  Per-Suite Results with Wilson 95% CIs:")
    print(f"  {'Suite':<30s} {'F1':>6s}  {'95% CI':>16s}  {'n':>4s}  {'Var(F1)':>10s}")
    print(f"  {'─' * 70}")

    estimates = []
    variances = []

    for s in suites:
        ci = wilson_interval(s["f1"], s["n"])
        v = f1_variance(s["f1"], s["n"])
        s["wilson_ci"] = ci
        s["variance"] = v
        estimates.append(s["f1"])
        variances.append(v)
        print(f"  {s['suite']:<30s} {s['f1']:>6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  {s['n']:>4d}  {v:>10.6f}")

    # Step 2: DerSimonian-Laird random-effects meta-analysis
    pooled = dersimonian_laird(estimates, variances)

    # Compute RE weights for forest plot
    for s in suites:
        s["re_weight"] = 1.0 / (s["variance"] + pooled["tau_squared"])

    print(f"\n  Random-Effects Pooled Estimate (DerSimonian-Laird):")
    print(f"  {'─' * 50}")
    print(f"    θ_RE  = {pooled['pooled_estimate']:.4f}")
    print(f"    SE    = {pooled['pooled_se']:.4f}")
    print(f"    95% CI: [{pooled['pooled_ci_95'][0]:.4f}, {pooled['pooled_ci_95'][1]:.4f}]")
    print(f"    τ²    = {pooled['tau_squared']:.6f}")
    print(f"\n  Heterogeneity:")
    print(f"    Q     = {pooled['cochrans_Q']:.4f} (df={pooled['Q_df']})")
    print(f"    p     = {pooled['p_heterogeneity']:.4f}")
    print(f"    I²    = {pooled['I_squared_pct']:.1f}%")

    # Interpret I²
    if pooled["I_squared_pct"] < 25:
        het_interp = "low heterogeneity"
    elif pooled["I_squared_pct"] < 50:
        het_interp = "moderate heterogeneity"
    elif pooled["I_squared_pct"] < 75:
        het_interp = "substantial heterogeneity"
    else:
        het_interp = "considerable heterogeneity"
    print(f"    Interpretation: {het_interp}")
    pooled["heterogeneity_interpretation"] = het_interp

    # Step 3: Forest plot data
    forest = forest_plot_data(suites, pooled)

    print(f"\n  Forest Plot Data:")
    print(f"  {'Label':<30s} {'Est':>6s}  {'[95% CI]':>16s}  {'Wt%':>5s}")
    print(f"  {'─' * 60}")
    for row in forest:
        ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        marker = "◆" if row["label"].startswith("Pooled") else "●"
        print(f"  {marker} {row['label']:<28s} {row['estimate']:>6.3f}  {ci_str:>16s}  {row['weight_pct']:>5.1f}")

    # Build output
    output = {
        "experiment": "hierarchical_random_effects_meta_analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "DerSimonian-Laird random-effects",
        "per_suite": [
            {
                "suite": s["suite"],
                "f1": s["f1"],
                "n": s["n"],
                "wilson_ci_95": s["wilson_ci"],
                "variance": round(s["variance"], 6),
                "re_weight": round(s["re_weight"], 4),
            }
            for s in suites
        ],
        "pooled": pooled,
        "forest_plot": forest,
        "interpretation": _build_interpretation(pooled, het_interp),
    }

    print(f"\n  Analysis complete.")
    return output


if __name__ == "__main__":
    output = run_analysis()
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
