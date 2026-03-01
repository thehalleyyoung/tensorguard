"""
Scalability Benchmark with Variance Characterization for TensorGuard.

Runs each scalability configuration 10 times to characterize Z3 solver
nondeterminism. Reports mean, std, CV, and 95% confidence intervals.

Addresses Cheng critique: "scalability measurements lack variance
characterization despite Z3's nondeterminism (Cheng)."
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

RESULTS_FILE = Path(__file__).parent / "scalability_variance_results.json"

NUM_RUNS = 10
DEPTHS = [5, 10, 20, 50, 100]


def generate_mlp_model(num_layers: int, hidden_dim: int = 256) -> str:
    lines = [
        "import torch.nn as nn",
        f"class MLP_{num_layers}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(num_layers):
        lines.append(f"        self.fc{i} = nn.Linear({hidden_dim}, {hidden_dim})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    lines.append(f"        self.head = nn.Linear({hidden_dim}, 10)")
    lines.append("    def forward(self, x):")
    for i in range(num_layers):
        lines.append(f"        x = self.act{i}(self.fc{i}(x))")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def generate_cnn_model(num_layers: int, channels: int = 64) -> str:
    lines = [
        "import torch.nn as nn",
        f"class CNN_{num_layers}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(num_layers):
        lines.append(
            f"        self.conv{i} = nn.Conv2d({channels}, {channels}, 3, padding=1)"
        )
        lines.append(f"        self.bn{i} = nn.BatchNorm2d({channels})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    lines.append("    def forward(self, x):")
    for i in range(num_layers):
        lines.append(f"        x = self.act{i}(self.bn{i}(self.conv{i}(x)))")
    lines.append("        return x")
    return "\n".join(lines)


def generate_transformer_model(num_layers: int, d_model: int = 512) -> str:
    lines = [
        "import torch.nn as nn",
        f"class Transformer_{num_layers}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.encoder = nn.TransformerEncoder(",
        f"            nn.TransformerEncoderLayer(d_model={d_model}, nhead=8),",
        f"            num_layers={num_layers})",
        f"        self.fc = nn.Linear({d_model}, 10)",
        "    def forward(self, x):",
        "        x = self.encoder(x)",
        "        return self.fc(x)",
    ]
    return "\n".join(lines)


def generate_buggy_mlp_model(num_layers: int, hidden_dim: int = 256) -> str:
    lines = [
        "import torch.nn as nn",
        f"class BuggyMLP_{num_layers}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(num_layers):
        lines.append(f"        self.fc{i} = nn.Linear({hidden_dim}, {hidden_dim})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    lines.append(f"        self.head = nn.Linear({hidden_dim + 128}, 10)")
    lines.append("    def forward(self, x):")
    for i in range(num_layers):
        lines.append(f"        x = self.act{i}(self.fc{i}(x))")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def compute_stats(times: List[float]) -> Dict[str, float]:
    """Compute mean, std, CV, and 95% CI for a list of timing measurements."""
    n = len(times)
    mean = sum(times) / n
    if n > 1:
        variance = sum((t - mean) ** 2 for t in times) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    cv = std / mean if mean > 0 else 0.0
    # 95% CI using t-distribution approximation (t_{9,0.025} ≈ 2.262)
    t_crit = 2.262
    ci_half = t_crit * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "cv": round(cv, 4),
        "ci_95_lower": round(mean - ci_half, 2),
        "ci_95_upper": round(mean + ci_half, 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "n_runs": n,
        "raw_times_ms": [round(t, 2) for t in times],
    }


def benchmark_with_variance(
    source: str,
    input_shapes: Dict[str, tuple],
    model_type: str,
    depth: int,
    n_runs: int = NUM_RUNS,
) -> Dict[str, Any]:
    """Run verification n_runs times and compute timing statistics."""
    times = []
    safe_results = []

    for run_i in range(n_runs):
        t0 = time.monotonic()
        result = verify_model(source, input_shapes=input_shapes)
        elapsed_ms = (time.monotonic() - t0) * 1000
        times.append(elapsed_ms)
        safe_results.append(result.safe)

    stats = compute_stats(times)
    # All runs should agree on verdict
    verdict_consistent = len(set(safe_results)) == 1

    return {
        "model_type": model_type,
        "depth": depth,
        "safe": safe_results[0],
        "verdict_consistent": verdict_consistent,
        "timing": stats,
    }


def run_scalability_variance_benchmarks() -> Dict[str, Any]:
    """Run full scalability benchmark suite with variance characterization."""
    results = {
        "experiment": "scalability_variance_benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs_per_config": NUM_RUNS,
        "benchmarks": [],
        "summary_table": [],
    }

    print("=" * 70)
    print(f"TensorGuard Scalability Benchmark with Variance ({NUM_RUNS} runs each)")
    print("=" * 70)

    configs = [
        ("mlp_correct", generate_mlp_model, {"x": ("batch", 256)}),
        ("cnn_correct", generate_cnn_model, {"x": ("batch", 64, 32, 32)}),
        ("transformer_correct", generate_transformer_model, {"x": ("batch", 32, 512)}),
        ("mlp_buggy", generate_buggy_mlp_model, {"x": ("batch", 256)}),
    ]

    for model_type, gen_fn, shapes in configs:
        print(f"\n--- {model_type} ---")
        for depth in DEPTHS:
            # Skip 100-layer CNN (too slow for 10 runs)
            if model_type == "cnn_correct" and depth == 100:
                print(f"  {model_type}-{depth:3d}: SKIPPED (>150s per run)")
                continue

            source = gen_fn(depth)
            r = benchmark_with_variance(source, shapes, model_type, depth)
            results["benchmarks"].append(r)
            t = r["timing"]
            print(
                f"  {model_type}-{depth:3d}: "
                f"mean={t['mean_ms']:8.1f}ms ± {t['std_ms']:5.1f}ms  "
                f"CV={t['cv']:.3f}  "
                f"95%CI=[{t['ci_95_lower']:.0f}, {t['ci_95_upper']:.0f}]  "
                f"consistent={r['verdict_consistent']}"
            )

            results["summary_table"].append({
                "model_type": model_type,
                "depth": depth,
                "mean_ms": t["mean_ms"],
                "std_ms": t["std_ms"],
                "cv": t["cv"],
                "ci_95": f"[{t['ci_95_lower']:.0f}, {t['ci_95_upper']:.0f}]",
            })

    # Summary statistics
    all_cvs = [b["timing"]["cv"] for b in results["benchmarks"]]
    results["variance_summary"] = {
        "mean_cv": round(sum(all_cvs) / len(all_cvs), 4) if all_cvs else 0,
        "max_cv": round(max(all_cvs), 4) if all_cvs else 0,
        "min_cv": round(min(all_cvs), 4) if all_cvs else 0,
        "all_verdicts_consistent": all(
            b["verdict_consistent"] for b in results["benchmarks"]
        ),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Mean CV across configs: {results['variance_summary']['mean_cv']:.4f}")
    print(f"Max CV: {results['variance_summary']['max_cv']:.4f}")
    print(f"All verdicts consistent: {results['variance_summary']['all_verdicts_consistent']}")

    return results


if __name__ == "__main__":
    run_scalability_variance_benchmarks()
