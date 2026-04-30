#!/usr/bin/env python3.11
"""Pre-registered second-wave power calculation (R4-W4 / R4-Q5).

Reviewer R4 asks: on the post-freeze unfiltered surface (N=15
observations, TG 5/15, FT 2/15, Pytea 3/15), what is the smallest
*second-wave* N (additional items, drawn from the same
pre-registered GitHub-search query) at which the union (N=15 + new
N) yields Fisher exact p<0.05 on at least one of the two pairwise
comparisons (TG vs FT or TG vs Pytea), assuming the new items
follow the observed point estimates?

We sweep new-N from 5 to 200 and, at each N, build the union 2x2
contingency table assuming the second wave matches the observed
hit rates exactly:

    TG      hits = 5 + round(N * 5/15)        misses = (15+N) - hits
    FT      hits = 2 + round(N * 2/15)        misses = (15+N) - hits
    Pytea   hits = 3 + round(N * 3/15)        misses = (15+N) - hits

For each N we compute Fisher exact p for TG-vs-FT and TG-vs-Pytea
(two-sided), and report the smallest N at which either drops below
0.05, 0.01.  We also report the Bayes Factor (BF10) under a
Beta(1,1) prior on each rate.

Output:
    reproducibility/postfreeze_second_wave_power.json
    reproducibility/postfreeze_second_wave_power.md
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OUT_JSON = os.path.join(ROOT, "reproducibility",
                        "postfreeze_second_wave_power.json")
OUT_MD = os.path.join(ROOT, "reproducibility",
                      "postfreeze_second_wave_power.md")


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table.

    Table:  row 1: a hits, b misses    row 2: c hits, d misses
    """
    from math import comb

    n = a + b + c + d
    r1 = a + b
    r2 = c + d
    c1 = a + c
    c2 = b + d
    if c1 == 0 or c2 == 0 or r1 == 0 or r2 == 0:
        return 1.0

    # Probability of observing exactly k hits in row 1 given marginals.
    def hyper_pmf(k: int) -> float:
        return comb(c1, k) * comb(c2, r1 - k) / comb(n, r1)

    obs = hyper_pmf(a)
    p = 0.0
    for k in range(max(0, r1 - c2), min(r1, c1) + 1):
        pmf = hyper_pmf(k)
        if pmf <= obs + 1e-12:
            p += pmf
    return min(p, 1.0)


def bf10_beta11(a: int, b: int, c: int, d: int) -> float:
    """Bayes factor BF10 for H1: rates differ vs H0: rates equal,
    under Beta(1,1) priors on each rate."""
    from math import lgamma, log, exp

    def log_beta(x: float, y: float) -> float:
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    # Marginal under H1: independent Beta(1,1) priors -> Beta-binomial
    # P(D|H1) = B(a+1, b+1) / B(1,1)   *   B(c+1, d+1) / B(1,1)
    log_p_h1 = (log_beta(a + 1, b + 1) - log_beta(1, 1) +
                log_beta(c + 1, d + 1) - log_beta(1, 1))
    # Marginal under H0: shared rate p ~ Beta(1,1)
    # P(D|H0) = B(a+c+1, b+d+1) / B(1,1)
    log_p_h0 = log_beta(a + c + 1, b + d + 1) - log_beta(1, 1)
    return exp(log_p_h1 - log_p_h0)


def union_table(N_new: int) -> Tuple[int, int, int, int, int, int]:
    """Return (tg_hits, tg_miss, ft_hits, ft_miss, py_hits, py_miss)
    for the union of N=15 + N_new under the observed hit rates."""
    base_n = 15
    base_tg, base_ft, base_py = 5, 2, 3
    tg_new = round(N_new * base_tg / base_n)
    ft_new = round(N_new * base_ft / base_n)
    py_new = round(N_new * base_py / base_n)
    total = base_n + N_new
    tg_h = base_tg + tg_new
    ft_h = base_ft + ft_new
    py_h = base_py + py_new
    return (tg_h, total - tg_h,
            ft_h, total - ft_h,
            py_h, total - py_h)


def main() -> int:
    sweeps = []
    for N_new in range(5, 401):
        tg_h, tg_m, ft_h, ft_m, py_h, py_m = union_table(N_new)
        p_tg_ft = fisher_exact(tg_h, tg_m, ft_h, ft_m)
        p_tg_py = fisher_exact(tg_h, tg_m, py_h, py_m)
        bf_tg_ft = bf10_beta11(tg_h, tg_m, ft_h, ft_m)
        bf_tg_py = bf10_beta11(tg_h, tg_m, py_h, py_m)
        sweeps.append({
            "N_new": N_new,
            "total_N": 15 + N_new,
            "tg_hits": tg_h, "ft_hits": ft_h, "py_hits": py_h,
            "p_tg_ft": p_tg_ft, "p_tg_py": p_tg_py,
            "bf10_tg_ft": bf_tg_ft, "bf10_tg_py": bf_tg_py,
        })

    def first_below(predicate) -> int:
        for s in sweeps:
            if predicate(s):
                return s["N_new"]
        return -1

    smallest_p05_either = first_below(
        lambda s: s["p_tg_ft"] < 0.05 or s["p_tg_py"] < 0.05)
    smallest_p05_tg_ft = first_below(lambda s: s["p_tg_ft"] < 0.05)
    smallest_p05_tg_py = first_below(lambda s: s["p_tg_py"] < 0.05)
    smallest_p01_either = first_below(
        lambda s: s["p_tg_ft"] < 0.01 or s["p_tg_py"] < 0.01)
    smallest_bf10_either = first_below(
        lambda s: s["bf10_tg_ft"] >= 10.0 or s["bf10_tg_py"] >= 10.0)

    out = {
        "_question": (
            "R4-W4 / R4-Q5: smallest second-wave N (drawn from the "
            "frozen 2026-04-08 GitHub-search query) such that the "
            "union (N=15 + N_new) yields Fisher exact p<0.05 on at "
            "least one of the two pairwise comparisons "
            "(TG vs FakeTensorMode or TG vs Pytea), under the "
            "observed point estimates (TG 5/15, FT 2/15, Pytea 3/15) "
            "extended at the same hit rates."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "base_observations": {"N": 15, "TG_hits": 5,
                               "FT_hits": 2, "Pytea_hits": 3},
        "smallest_N_new": {
            "fisher_p_lt_0.05_either": smallest_p05_either,
            "fisher_p_lt_0.05_tg_vs_ft": smallest_p05_tg_ft,
            "fisher_p_lt_0.05_tg_vs_pytea": smallest_p05_tg_py,
            "fisher_p_lt_0.01_either": smallest_p01_either,
            "bf10_geq_10_either": smallest_bf10_either,
        },
        "interpretation": (
            f"At the observed point estimates (TG 33.3%, FT 13.3%, "
            f"Pytea 20.0%), a pre-registered second wave of "
            f"N_new={smallest_p05_either} fresh items would push the "
            f"union (N={15+smallest_p05_either}) to Fisher p<0.05 on "
            f"at least one pairwise comparison.  Reaching p<0.01 "
            f"requires N_new={smallest_p01_either}, and BF10>=10 on "
            f"either comparison requires N_new={smallest_bf10_either}."
            f"  TG vs FT separates first; TG vs Pytea requires "
            f"N_new={smallest_p05_tg_py} for p<0.05.  These numbers "
            f"are upper bounds in the sense that if the second wave "
            f"out-performs the observed point estimates the union "
            f"separates earlier; conversely, if the second wave "
            f"under-performs the union may not separate at the same N."
        ),
        "sweeps": sweeps,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Post-freeze second-wave power calculation",
        "",
        "## Command",
        "",
        "```",
        "python3.11 reproducibility/postfreeze_second_wave_power.py",
        "```",
        "",
        "## Question",
        "",
        "On the post-freeze unfiltered surface (N=15, TG 5/15, FT 2/15, "
        "Pytea 3/15), what is the smallest second-wave N (additional "
        "items from the frozen 2026-04-08 GitHub-search query) at "
        "which the union (N=15 + N_new) yields Fisher exact p<0.05 on "
        "at least one pairwise comparison (TG vs FT or TG vs Pytea), "
        "assuming the second wave matches the observed point estimates?",
        "",
        "## Result",
        "",
        f"| Threshold | Smallest N_new | Total N |",
        f"|---|---|---|",
        f"| Fisher p<0.05 (either pair) | {smallest_p05_either} | {15+smallest_p05_either} |",
        f"| Fisher p<0.05 (TG vs FT)    | {smallest_p05_tg_ft} | {15+smallest_p05_tg_ft} |",
        f"| Fisher p<0.05 (TG vs Pytea) | {smallest_p05_tg_py} | {15+smallest_p05_tg_py} |",
        f"| Fisher p<0.01 (either pair) | {smallest_p01_either} | {15+smallest_p01_either} |",
        f"| BF10 >= 10 (either pair)    | {smallest_bf10_either} | {15+smallest_bf10_either} |",
        "",
        "## Paper claim closed",
        "",
        "Round-4 reviewer W4/Q5 asks whether a pre-registered N>=30 "
        "second wave is feasible before camera-ready, and on the "
        "observed point estimates what is the smallest second-wave N "
        "at which the union yields Fisher p<0.05.  This artefact "
        "answers the second question; whether the wave is feasible "
        "before camera-ready is recorded in the internal review "
        "response log.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(f"Smallest N_new for p<0.05 either: {smallest_p05_either}")
    print(f"  TG vs FT: {smallest_p05_tg_ft}")
    print(f"  TG vs Pytea: {smallest_p05_tg_py}")
    print(f"  BF10>=10 either: {smallest_bf10_either}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
