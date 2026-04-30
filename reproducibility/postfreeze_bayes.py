#!/usr/bin/env python3.11
"""Round-5 one-step-away: One-sided Wilson intervals + Bayes factor on N=15.

The unfiltered post-freeze N=15 sample shows TG 5/15 vs FT 2/15 vs
Pytea 3/15.  This computes:
  (a) One-sided Wilson 95% lower bounds for each tool's catch rate.
  (b) A Bayes factor (Beta-binomial prior-ratio test) for TG > FT and
      TG > Pytea, framing the data as evidence rather than just point
      estimates.

Run:
    PYTHONPATH=. python3.11 reproducibility/postfreeze_bayes.py
"""
from __future__ import annotations

import json
import math
import pathlib


# ------------------------------------------------------------- helpers ----

def wilson_lower(k: int, n: int, z: float = 1.6449) -> float:
    """One-sided Wilson lower bound at 95% (z=1.645)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - spread) / denom


def wilson_two_sided(k: int, n: int, z: float = 1.96):
    """Two-sided Wilson 95% CI."""
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return lo, hi


def log_beta(a: float, b: float) -> float:
    """log B(a,b) = lgamma(a)+lgamma(b)-lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_posterior_mean(k: int, n: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return (alpha + k) / (alpha + beta + n)


def bayes_factor_greater(
    k1: int, n1: int, k2: int, n2: int,
    alpha: float = 1.0, beta: float = 1.0
) -> float:
    """
    Bayes factor BF_10 for H1: p1 > p2 vs H0: p1 <= p2, using
    a symmetric Jeffreys-like Beta(alpha,beta) prior on both p1 and p2.

    Approximation via sampling from posterior Beta distributions.
    """
    import random
    random.seed(42)
    # Beta posterior for p1: Beta(alpha+k1, beta+n1-k1)
    # Beta posterior for p2: Beta(alpha+k2, beta+n2-k2)
    a1, b1 = alpha + k1, beta + n1 - k1
    a2, b2 = alpha + k2, beta + n2 - k2

    # P(p1 > p2 | data) via Monte Carlo
    n_samples = 100_000
    # Use gammas to sample from Beta
    count_p1_gt_p2 = 0
    for _ in range(n_samples):
        g1a = random.gammavariate(a1, 1)
        g1b = random.gammavariate(b1, 1)
        p1 = g1a / (g1a + g1b)
        g2a = random.gammavariate(a2, 1)
        g2b = random.gammavariate(b2, 1)
        p2 = g2a / (g2a + g2b)
        if p1 > p2:
            count_p1_gt_p2 += 1

    prob_h1 = count_p1_gt_p2 / n_samples
    # Prior probability P(p1 > p2) = 0.5 by symmetry with uniform priors
    prior_h1 = 0.5
    if prob_h1 <= 0 or prob_h1 >= 1:
        return float("inf") if prob_h1 > 0 else 0.0
    bf = (prob_h1 / (1 - prob_h1)) / (prior_h1 / (1 - prior_h1))
    return bf


# ------------------------------------------------------------- main ------

def main():
    # N=15 post-freeze unfiltered data
    data = {
        "TG":    {"k": 5, "n": 15},
        "FT":    {"k": 2, "n": 15},
        "Pytea": {"k": 3, "n": 15},
    }

    results = {}
    for tool, d in data.items():
        k, n = d["k"], d["n"]
        lo2, hi2 = wilson_two_sided(k, n)
        lo1 = wilson_lower(k, n)
        results[tool] = {
            "k": k, "n": n,
            "rate": k / n,
            "wilson_two_sided_95_ci": [round(lo2, 4), round(hi2, 4)],
            "wilson_one_sided_95_lower": round(lo1, 4),
            "posterior_mean_jeffreys": round(beta_posterior_mean(k, n, 0.5, 0.5), 4),
        }

    # Bayes factors
    bf_tg_vs_ft = bayes_factor_greater(
        data["TG"]["k"], data["TG"]["n"],
        data["FT"]["k"], data["FT"]["n"],
    )
    bf_tg_vs_pytea = bayes_factor_greater(
        data["TG"]["k"], data["TG"]["n"],
        data["Pytea"]["k"], data["Pytea"]["n"],
    )

    output = {
        "data": data,
        "per_tool_intervals": results,
        "bayes_factor_TG_gt_FT": round(bf_tg_vs_ft, 2),
        "bayes_factor_TG_gt_Pytea": round(bf_tg_vs_pytea, 2),
        "interpretation": (
            f"BF(TG>FT)={bf_tg_vs_ft:.2f}: the data are {bf_tg_vs_ft:.1f}x more "
            f"consistent with TG's catch rate exceeding FT's than with the reverse. "
            f"BF(TG>Pytea)={bf_tg_vs_pytea:.2f}: similarly for Pytea. "
            "BF < 3 is 'weak evidence', 3-10 is 'moderate evidence'. "
            "On N=15 the data are consistent with a TG advantage but do not "
            "constitute strong Bayesian evidence (BF >> 10) for superiority over "
            "either baseline."
        ),
    }

    out_json = pathlib.Path("reproducibility/postfreeze_bayes.json")
    out_md = pathlib.Path("reproducibility/postfreeze_bayes.md")

    out_json.write_text(json.dumps(output, indent=2))

    tg = results["TG"]
    ft = results["FT"]
    py = results["Pytea"]
    md = f"""# Bayesian analysis of N=15 post-freeze unfiltered sample

One-sided Wilson lower bounds and Bayes factors for the
N=15 unfiltered pre-registered post-freeze real-PR sample.

## One-sided Wilson 95% lower bounds

| Tool | Catches | Rate | One-sided 95% lower | Two-sided 95% CI |
|---|---|---|---|---|
| TG | {tg['k']}/{tg['n']} | {tg['rate']:.1%} | {tg['wilson_one_sided_95_lower']:.1%} | [{tg['wilson_two_sided_95_ci'][0]:.1%}, {tg['wilson_two_sided_95_ci'][1]:.1%}] |
| FakeTensorMode | {ft['k']}/{ft['n']} | {ft['rate']:.1%} | {ft['wilson_one_sided_95_lower']:.1%} | [{ft['wilson_two_sided_95_ci'][0]:.1%}, {ft['wilson_two_sided_95_ci'][1]:.1%}] |
| Pytea | {py['k']}/{py['n']} | {py['rate']:.1%} | {py['wilson_one_sided_95_lower']:.1%} | [{py['wilson_two_sided_95_ci'][0]:.1%}, {py['wilson_two_sided_95_ci'][1]:.1%}] |

## Bayes factors (H1: TG > baseline, H0: TG ≤ baseline; Beta(1,1) prior)

| Comparison | BF₁₀ | Evidence level |
|---|---|---|
| TG vs FakeTensorMode | **{bf_tg_vs_ft:.2f}** | {"weak (<3)" if bf_tg_vs_ft < 3 else "moderate (3-10)" if bf_tg_vs_ft < 10 else "strong (>10)"} |
| TG vs Pytea | **{bf_tg_vs_pytea:.2f}** | {"weak (<3)" if bf_tg_vs_pytea < 3 else "moderate (3-10)" if bf_tg_vs_pytea < 10 else "strong (>10)"} |

## Interpretation

At N=15 the posterior mean for TG ({tg['posterior_mean_jeffreys']:.1%}) exceeds those of
FT ({ft['posterior_mean_jeffreys']:.1%}) and Pytea ({py['posterior_mean_jeffreys']:.1%}), and the Bayes factors indicate
{output['interpretation']}

The one-sided Wilson lower bounds confirm that all three tools' catch
rates are plausibly above zero, but the CIs for TG and Pytea overlap,
consistent with the non-significant Fisher-exact p-values.

## Reproduce

    PYTHONPATH=. python3.11 reproducibility/postfreeze_bayes.py
"""
    out_md.write_text(md)
    print(f"Wrote {out_json} and {out_md}")
    print(f"BF(TG>FT)={bf_tg_vs_ft:.2f}, BF(TG>Pytea)={bf_tg_vs_pytea:.2f}")


if __name__ == "__main__":
    main()
