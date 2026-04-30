"""Power calculation for the pre-registered unfiltered corpus.

Question (round-3 reviewer): on the 5/15 pre-registered corpus, what
sample size N would be required to achieve 80% power (alpha=0.05) to
reject H0 (equal proportions) under the observed proportions
p_TG = 5/15 = 0.333, p_baseline = 2/15 = 0.133, two-sided
two-proportion z-test?

Result: N >= 69 per arm (138 total) under the standard normal-
approximation power formula
N = ((z_{alpha/2} sqrt(2 p_bar (1-p_bar)) + z_{beta} sqrt(p1(1-p1)+p2(1-p2)))/(p1-p2))**2
with z_{alpha/2}=1.96, z_{beta}=0.8416, p_bar=(p1+p2)/2.

We report this number in the paper alongside the existing 5/15
pre-registered result so the reader can see the gap between
n=15 (current pre-registered sample) and n=69 (required for 80%
power against the observed effect size).
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from scipy import stats


def power_n(p1: float, p2: float,
            alpha: float = 0.05, power: float = 0.80) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = z_alpha * sqrt(2 * p_bar * (1 - p_bar)) \
        + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(round((num / (p1 - p2)) ** 2))


def main() -> None:
    p1, p2 = 5 / 15, 2 / 15
    n_per_arm = power_n(p1, p2)
    out = {
        "p_tg": p1,
        "p_baseline": p2,
        "alpha": 0.05,
        "power": 0.80,
        "test": "two-sided two-proportion z-test (normal approximation)",
        "n_per_arm_required": n_per_arm,
        "n_total_required": 2 * n_per_arm,
        "n_observed_per_arm": 15,
        "shortfall_per_arm": n_per_arm - 15,
        "interpretation": (
            "The 5/15 vs 2/15 result is consistent with a true "
            "effect of about 20 percentage points but cannot reject "
            "equality at the n=15 sample size; reaching 80% power "
            f"against this effect requires {n_per_arm} pre-registered "
            "modules per arm."
        ),
    }
    out_path = Path(__file__).resolve().parent / "preregistered_power_calc.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
