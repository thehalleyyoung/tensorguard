#!/usr/bin/env python3
"""
Cluster-bootstrap confidence intervals.

Implements cluster-bootstrap resampling with architectural families as
clusters, computing both naïve (i.i.d.) and cluster-corrected CIs for:

- F1 score
- Mutation kill rate
- IC3 speedup over bounded model checking

Shows effective sample size reduction under intra-class correlation.

Results are saved to ``.benchmarks/cluster_bootstrap_results.json``.
"""

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_PATH = OUTPUT_DIR / "cluster_bootstrap_results.json"

N_BOOTSTRAP = 10_000


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_f1_data() -> Tuple[Dict[str, List[Dict]], float]:
    """Load per-benchmark results for F1 computation, grouped by category."""
    path = ROOT / "experiments" / "comprehensive_final_results.json"
    if not path.exists():
        return {}, 0.0

    with open(path) as f:
        data = json.load(f)

    overall_f1 = data.get("overall", {}).get("tensorguard", {}).get(
        "metrics", {}).get("f1", 0.0)

    # Group per-benchmark results by category (architectural family)
    clusters: Dict[str, List[Dict]] = {}
    tg_benchmarks = data.get("per_benchmark", {}).get("tensorguard", [])
    for bench in tg_benchmarks:
        cat = bench.get("category", "unknown")
        clusters.setdefault(cat, []).append(bench)

    return clusters, overall_f1


def load_mutation_data() -> Tuple[Dict[str, List[Dict]], float]:
    """Load per-model mutation results, grouped by model family."""
    path = ROOT / "experiments" / ".benchmarks" / "mutation_testing_results.json"
    if not path.exists():
        return {}, 0.0

    with open(path) as f:
        data = json.load(f)

    overall_rate = data.get("summary", {}).get("mutation_score", 0.0)

    # Group per-model scores by model name prefix (architectural family)
    clusters: Dict[str, List[Dict]] = {}
    per_model = data.get("per_model_scores", {})
    if isinstance(per_model, dict):
        for model_name, model_data in per_model.items():
            family = model_name.rstrip("0123456789").rstrip("_")
            entry = dict(model_data) if isinstance(model_data, dict) else {}
            entry["model"] = model_name
            entry.setdefault("total_mutants",
                             entry.get("killed", 0) + entry.get("survived", 0))
            clusters.setdefault(family, []).append(entry)

    return clusters, overall_rate


def load_ic3_data() -> Tuple[Dict[str, List[Dict]], float]:
    """Load IC3 benchmark results, grouped by category."""
    path = ROOT / ".benchmarks" / "ic3_comprehensive_results.json"
    if not path.exists():
        return {}, 0.0

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    clusters: Dict[str, List[Dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        clusters.setdefault(cat, []).append(r)

    # Average speedup
    speedups = []
    for r in results:
        bmc_t = r.get("bounded_time_ms", 0)
        ic3_t = r.get("ic3_time_ms", 0)
        if ic3_t > 0:
            speedups.append(bmc_t / ic3_t)
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    return clusters, avg_speedup


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap machinery
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f1_from_list(benchmarks: List[Dict]) -> float:
    """Compute F1 from a list of benchmark dicts."""
    tp = sum(1 for b in benchmarks
             if b.get("detected_bug", False) and b.get("ground_truth", False))
    fp = sum(1 for b in benchmarks
             if b.get("detected_bug", False) and not b.get("ground_truth", False))
    fn = sum(1 for b in benchmarks
             if not b.get("detected_bug", False) and b.get("ground_truth", False))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_kill_rate(model_results: List[Dict]) -> float:
    """Compute mutation kill rate from per-model results."""
    total = sum(r.get("total_mutants", 0) for r in model_results)
    killed = sum(r.get("killed", 0) for r in model_results)
    return killed / total if total > 0 else 0.0


def compute_ic3_speedup(results: List[Dict]) -> float:
    """Compute average IC3 speedup from results."""
    speedups = []
    for r in results:
        bmc_t = r.get("bounded_time_ms", 0)
        ic3_t = r.get("ic3_time_ms", 0)
        if ic3_t > 0:
            speedups.append(bmc_t / ic3_t)
    return sum(speedups) / len(speedups) if speedups else 0.0


def naive_bootstrap(
    all_items: List[Dict],
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, float]:
    """Standard i.i.d. bootstrap (ignores clustering)."""
    n = len(all_items)
    if n == 0:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}

    stats = []
    for _ in range(n_bootstrap):
        sample = random.choices(all_items, k=n)
        stats.append(metric_fn(sample))

    stats.sort()
    return {
        "lower": stats[int(n_bootstrap * 0.025)],
        "upper": stats[int(n_bootstrap * 0.975)],
        "mean": sum(stats) / len(stats),
    }


def cluster_bootstrap(
    clusters: Dict[str, List[Dict]],
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, float]:
    """Cluster bootstrap: resample entire clusters with replacement."""
    cluster_list = list(clusters.values())
    n_clusters = len(cluster_list)
    if n_clusters == 0:
        return {"lower": 0.0, "upper": 0.0, "mean": 0.0}

    stats = []
    for _ in range(n_bootstrap):
        sampled_clusters = random.choices(cluster_list, k=n_clusters)
        combined = [item for cl in sampled_clusters for item in cl]
        stats.append(metric_fn(combined))

    stats.sort()
    return {
        "lower": stats[int(n_bootstrap * 0.025)],
        "upper": stats[int(n_bootstrap * 0.975)],
        "mean": sum(stats) / len(stats),
    }


def compute_icc(clusters: Dict[str, List[float]]) -> float:
    """Compute intra-class correlation coefficient (ICC).

    Uses the one-way random effects ANOVA formulation.
    """
    all_vals = [v for vals in clusters.values() for v in vals]
    if not all_vals:
        return 0.0

    grand_mean = sum(all_vals) / len(all_vals)
    k = len(clusters)
    if k <= 1:
        return 0.0

    # Between-group variance
    cluster_means = {c: sum(v) / len(v) for c, v in clusters.items() if v}
    cluster_sizes = {c: len(v) for c, v in clusters.items() if v}
    n_total = sum(cluster_sizes.values())

    ms_between = sum(
        cluster_sizes[c] * (cluster_means[c] - grand_mean) ** 2
        for c in cluster_means
    ) / (k - 1)

    # Within-group variance
    ms_within = sum(
        sum((v - cluster_means[c]) ** 2 for v in vals)
        for c, vals in clusters.items() if vals
    ) / max(1, n_total - k)

    # Average cluster size
    n_bar = n_total / k

    if ms_within == 0 and ms_between == 0:
        return 0.0

    icc = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within)
    return max(0.0, min(1.0, icc))


def effective_sample_size(n: int, icc: float, avg_cluster_size: float) -> float:
    """Compute effective sample size under clustering."""
    deff = 1 + (avg_cluster_size - 1) * icc
    return n / deff if deff > 0 else n


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis() -> Dict[str, Any]:
    results: Dict[str, Any] = {"metrics": {}}

    # --- F1 ---
    f1_clusters, overall_f1 = load_f1_data()
    if f1_clusters:
        all_f1 = [b for bs in f1_clusters.values() for b in bs]
        naive_ci = naive_bootstrap(all_f1, compute_f1_from_list)
        cluster_ci = cluster_bootstrap(f1_clusters, compute_f1_from_list)

        # ICC for F1: use binary detection as the value
        icc_clusters = {}
        for cat, benchmarks in f1_clusters.items():
            icc_clusters[cat] = [
                1.0 if b.get("detected_bug", False) else 0.0
                for b in benchmarks
            ]
        icc = compute_icc(icc_clusters)
        n = len(all_f1)
        avg_cs = n / len(f1_clusters) if f1_clusters else 1
        ess = effective_sample_size(n, icc, avg_cs)

        results["metrics"]["f1"] = {
            "point_estimate": round(overall_f1, 4),
            "naive_ci_95": {k: round(v, 4) for k, v in naive_ci.items()},
            "cluster_ci_95": {k: round(v, 4) for k, v in cluster_ci.items()},
            "icc": round(icc, 4),
            "n_observations": n,
            "n_clusters": len(f1_clusters),
            "avg_cluster_size": round(avg_cs, 2),
            "effective_sample_size": round(ess, 2),
        }
    else:
        results["metrics"]["f1"] = {"error": "no data available"}

    # --- Mutation Kill Rate ---
    mut_clusters, overall_kill = load_mutation_data()
    if mut_clusters:
        all_mut = [m for ms in mut_clusters.values() for m in ms]
        naive_ci = naive_bootstrap(all_mut, compute_kill_rate)
        cluster_ci = cluster_bootstrap(mut_clusters, compute_kill_rate)

        icc_clusters = {}
        for cat, models in mut_clusters.items():
            rates = []
            for m in models:
                tot = m.get("total_mutants", 0)
                k = m.get("killed", 0)
                rates.append(k / tot if tot > 0 else 0.0)
            icc_clusters[cat] = rates
        icc = compute_icc(icc_clusters)
        n = len(all_mut)
        avg_cs = n / len(mut_clusters) if mut_clusters else 1
        ess = effective_sample_size(n, icc, avg_cs)

        results["metrics"]["mutation_kill_rate"] = {
            "point_estimate": round(overall_kill, 4),
            "naive_ci_95": {k: round(v, 4) for k, v in naive_ci.items()},
            "cluster_ci_95": {k: round(v, 4) for k, v in cluster_ci.items()},
            "icc": round(icc, 4),
            "n_observations": n,
            "n_clusters": len(mut_clusters),
            "avg_cluster_size": round(avg_cs, 2),
            "effective_sample_size": round(ess, 2),
        }
    else:
        results["metrics"]["mutation_kill_rate"] = {"error": "no data available"}

    # --- IC3 Speedup ---
    ic3_clusters, overall_speedup = load_ic3_data()
    if ic3_clusters:
        all_ic3 = [r for rs in ic3_clusters.values() for r in rs]
        naive_ci = naive_bootstrap(all_ic3, compute_ic3_speedup)
        cluster_ci = cluster_bootstrap(ic3_clusters, compute_ic3_speedup)

        icc_clusters = {}
        for cat, rs in ic3_clusters.items():
            speedups = []
            for r in rs:
                bmc = r.get("bounded_time_ms", 0)
                ic3 = r.get("ic3_time_ms", 0)
                if ic3 > 0:
                    speedups.append(bmc / ic3)
            icc_clusters[cat] = speedups
        icc = compute_icc(icc_clusters)
        n = len(all_ic3)
        avg_cs = n / len(ic3_clusters) if ic3_clusters else 1
        ess = effective_sample_size(n, icc, avg_cs)

        results["metrics"]["ic3_speedup"] = {
            "point_estimate": round(overall_speedup, 4),
            "naive_ci_95": {k: round(v, 4) for k, v in naive_ci.items()},
            "cluster_ci_95": {k: round(v, 4) for k, v in cluster_ci.items()},
            "icc": round(icc, 4),
            "n_observations": n,
            "n_clusters": len(ic3_clusters),
            "avg_cluster_size": round(avg_cs, 2),
            "effective_sample_size": round(ess, 2),
        }
    else:
        results["metrics"]["ic3_speedup"] = {"error": "no data available"}

    return results


def main():
    results = run_analysis()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Cluster-Bootstrap Confidence Interval Analysis")
    print("=" * 60)
    for metric_name, info in results["metrics"].items():
        if "error" in info:
            print(f"\n{metric_name}: {info['error']}")
            continue
        print(f"\n{metric_name}:")
        print(f"  Point estimate: {info['point_estimate']:.4f}")
        print(f"  Naïve 95% CI:   [{info['naive_ci_95']['lower']:.4f}, "
              f"{info['naive_ci_95']['upper']:.4f}]")
        print(f"  Cluster 95% CI: [{info['cluster_ci_95']['lower']:.4f}, "
              f"{info['cluster_ci_95']['upper']:.4f}]")
        print(f"  ICC:            {info['icc']:.4f}")
        print(f"  N:              {info['n_observations']}")
        print(f"  Clusters:       {info['n_clusters']}")
        print(f"  Effective N:    {info['effective_sample_size']:.1f}")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
