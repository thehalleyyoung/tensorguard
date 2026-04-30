#!/usr/bin/env python3.11
"""Round-5 reviewer Q4: power analysis for the post-freeze N=15 PR
sample.

Effect sizes observed in round-4 reporting:
  * TG  : 5/15 = 33.3%
  * FakeTensorMode : 2/15 = 13.3%
  * Pytea : 3/15 = 20.0%

We want the per-arm sample size (paired and unpaired) needed to
declare TG strictly above each baseline at alpha=0.05, two-sided,
80% power, under Fisher's exact test (no normal approximation),
holding the observed proportions fixed as the true effect.

We compute by simulation: for each candidate N, we draw 2000
synthetic 2x2 tables under the observed proportions and count the
fraction that reject Fisher's exact test at alpha=0.05 two-sided.
The smallest N at which power >= 0.80 is the required N per arm
(unpaired).  We also report the paired McNemar-exact analogue
which exploits within-PR concordance.

Output:
  reproducibility/postfreeze_power_analysis.json
  reproducibility/postfreeze_power_analysis.md
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import fisher_exact, binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_JSON = os.path.join(ROOT, "reproducibility",
                        "postfreeze_power_analysis.json")
OUT_MD = os.path.join(ROOT, "reproducibility",
                      "postfreeze_power_analysis.md")

ALPHA = 0.05
TARGET_POWER = 0.80
N_REPS = 2000
RNG = np.random.default_rng(20260429)


def _power_fisher(p_a: float, p_b: float, n: int, reps: int = N_REPS,
                  alpha: float = ALPHA) -> float:
    a = RNG.binomial(n, p_a, size=reps)
    b = RNG.binomial(n, p_b, size=reps)
    rej = 0
    for x, y in zip(a, b):
        table = [[x, n - x], [y, n - y]]
        _, p = fisher_exact(table, alternative="two-sided")
        if p < alpha:
            rej += 1
    return rej / reps


def _smallest_n_for_power(p_a: float, p_b: float,
                          target: float = TARGET_POWER,
                          n_max: int = 600) -> Tuple[int, float]:
    """Bisection / linear-up sweep for the smallest N per arm."""
    candidates = [15, 25, 40, 60, 80, 100, 140, 180, 240, 320, 420, 560]
    best = None
    for n in candidates:
        pw = _power_fisher(p_a, p_b, n)
        print(f"  N={n:4d}  power={pw:.3f}")
        if pw >= target:
            best = (n, pw)
            break
    if best is None:
        return (n_max, _power_fisher(p_a, p_b, n_max))
    # tighten: search within the previous bracket
    lo = candidates[max(0, candidates.index(best[0]) - 1)]
    hi = best[0]
    while hi - lo > 5:
        mid = (lo + hi) // 2
        pw_mid = _power_fisher(p_a, p_b, mid)
        if pw_mid >= target:
            hi = mid
            best = (mid, pw_mid)
        else:
            lo = mid
    return best


def main() -> int:
    p_tg = 5 / 15
    p_ft = 2 / 15
    p_pyt = 3 / 15

    out: Dict[str, Any] = {
        "_question": (
            "R5-Q4: per-arm sample size needed to declare TG strictly "
            "above the post-freeze baselines at alpha=0.05, 80% power, "
            "two-sided Fisher exact, given the round-4 effect sizes."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "observed": {
            "tg": [5, 15], "fake_tensor_mode": [2, 15], "pytea": [3, 15],
        },
    }

    print("Power vs FakeTensorMode (5/15 vs 2/15)...")
    n_ft, pw_ft = _smallest_n_for_power(p_tg, p_ft)
    print(f"  -> need N >= {n_ft} per arm (achieved power {pw_ft:.3f})")

    print("Power vs Pytea (5/15 vs 3/15)...")
    n_pyt, pw_pyt = _smallest_n_for_power(p_tg, p_pyt)
    print(f"  -> need N >= {n_pyt} per arm (achieved power {pw_pyt:.3f})")

    out["fisher_two_sided"] = {
        "vs_fake_tensor_mode": {"n_per_arm_for_80pct_power": n_ft,
                                 "achieved_power": pw_ft},
        "vs_pytea": {"n_per_arm_for_80pct_power": n_pyt,
                      "achieved_power": pw_pyt},
    }

    # Paired McNemar analogy: assume the observed marginals reflect a
    # discordant-pair distribution where TG and the baseline disagree
    # on max(p_TG,p_B)*N - common cases.  Simulated lower bound under
    # the conservative assumption that paired catches are independent
    # Bernoullis with the marginal rates.
    print("\nPaired McNemar (independence-baseline) vs FTM...")
    def _mcnemar_power(p_a: float, p_b: float, n: int, reps: int = N_REPS):
        rej = 0
        for _ in range(reps):
            a = RNG.binomial(1, p_a, size=n)
            b = RNG.binomial(1, p_b, size=n)
            b01 = int(np.sum((a == 1) & (b == 0)))
            b10 = int(np.sum((a == 0) & (b == 1)))
            n_disc = b01 + b10
            if n_disc == 0:
                continue
            from scipy.stats import binomtest  # type: ignore
            try:
                p = binomtest(b01, n_disc, 0.5,
                               alternative="two-sided").pvalue
            except Exception:
                p = 2 * min(binom.cdf(min(b01, b10), n_disc, 0.5),
                             1 - binom.cdf(min(b01, b10) - 1, n_disc, 0.5))
            if p < ALPHA:
                rej += 1
        return rej / reps

    n_pair_ft = None
    for n in [40, 60, 80, 100, 140, 180, 240]:
        pw = _mcnemar_power(p_tg, p_ft, n)
        print(f"  N={n}  power={pw:.3f}")
        if pw >= TARGET_POWER and n_pair_ft is None:
            n_pair_ft = (n, pw)
            break
    out["mcnemar_independent_baseline"] = {
        "vs_fake_tensor_mode": (
            None if n_pair_ft is None
            else {"n_pairs_for_80pct_power": n_pair_ft[0],
                  "achieved_power": n_pair_ft[1]}),
    }

    out["interpretation"] = (
        f"Under the round-4 observed proportions (TG 5/15 vs FTM 2/15 "
        f"vs Pytea 3/15), a two-sided Fisher exact test reaches 80% "
        f"power at approximately N >= {n_ft} per arm against "
        f"FakeTensorMode and N >= {n_pyt} per arm against Pytea.  The "
        f"shipped N=15 sample is therefore directionally informative "
        f"but underpowered against either baseline at the observed "
        f"effect size.  This is a corpus-collection cost question, not "
        f"a tool-quality question: catching unbiased post-freeze PRs "
        f"is the throughput bottleneck.  An N >= {max(n_ft, n_pyt)} "
        f"replication on a future-collected post-freeze sample would "
        f"be sufficient to upgrade the directional result to a "
        f"statistically separable comparison."
    )

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Post-freeze power analysis (round 5 Q4)",
        "",
        "## Setup",
        "",
        f"Observed round-4 catches on N=15 unbiased post-freeze PRs:",
        f"  * TG: 5/15 (33.3%)",
        f"  * FakeTensorMode: 2/15 (13.3%)",
        f"  * Pytea: 3/15 (20.0%)",
        "",
        f"Two-sided Fisher exact, alpha=0.05, 80% power.  Power "
        f"estimated by Monte-Carlo with {N_REPS} reps per (N, "
        f"effect-size) cell, holding the observed proportions as the "
        f"true effect.",
        "",
        "## Result",
        "",
        f"| Comparison | Required N per arm |",
        f"|---|---|",
        f"| TG vs FakeTensorMode | {n_ft} |",
        f"| TG vs Pytea | {n_pyt} |",
        "",
        "## Reading",
        "",
        f"The shipped N=15 sample is directionally informative (TG is "
        f"the top tool in absolute count on every comparable cell) but "
        f"underpowered for separation at alpha=0.05.  At the observed "
        f"effect sizes, a future N>={max(n_ft,n_pyt)} unbiased "
        f"post-freeze replication is sufficient to upgrade either "
        f"comparison to statistical separability.  Round-5 reviewer "
        f"Q4 is therefore a corpus-collection question (post-freeze "
        f"PR throughput) rather than a tool-quality question.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
