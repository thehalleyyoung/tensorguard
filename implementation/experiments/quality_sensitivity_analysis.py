"""
Quality Scoring Sensitivity Analysis.

Varies the exponents (α, β, γ, δ) in the predicate quality function
  q(p) = g(p)^α · c(p)^β · k(p)^γ · m(p)^δ
and measures impact on CEGAR convergence and final verification outcome.

Tests whether the quality scoring function is robust to perturbations,
addressing the reviewer concern that exponent choices are ad hoc.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import (
    PredicateQualityScorer,
    ShapePredicate,
    PredicateKind,
    run_shape_cegar,
)


# Benchmarks with known outcomes for sensitivity testing
SENSITIVITY_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "mlp_mismatch",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "mlp_correct",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "conv_mismatch",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(32, 128, 3)
    def forward(self, x):
        return self.conv2(self.conv1(x))
""",
        "input_shapes": {"x": ("batch", "c", "h", "w")},
    },
    {
        "name": "conv_correct",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class CorrectConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
    def forward(self, x):
        return self.conv2(self.conv1(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "chain_correct_3",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class Chain3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "chain_buggy_3",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyChain3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "projection_correct",
        "has_bug": False,
        "code": """\
import torch.nn as nn
class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(512, 64)
        self.k = nn.Linear(512, 64)
    def forward(self, x):
        return self.q(x) + self.k(x)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "projection_buggy",
        "has_bug": True,
        "code": """\
import torch.nn as nn
class BuggyProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(512, 64)
        self.k = nn.Linear(512, 128)
    def forward(self, x):
        return self.q(x) + self.k(x)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
]


# Exponent configurations to test
EXPONENT_CONFIGS = [
    # (name, α=generality, β=coverage, γ=consistency, δ=mutual_info)
    ("default",      0.3, 0.2, 0.2, 0.3),
    ("paper_v1",     0.25, 0.35, 0.25, 0.15),
    ("uniform",      0.25, 0.25, 0.25, 0.25),
    ("gen_heavy",    0.5, 0.1, 0.1, 0.3),
    ("cov_heavy",    0.1, 0.5, 0.1, 0.3),
    ("cons_heavy",   0.1, 0.1, 0.5, 0.3),
    ("mi_heavy",     0.1, 0.1, 0.1, 0.7),
    ("no_quality",   0.0, 0.0, 0.0, 0.0),  # all predicates accepted
    ("gen_cov_only", 0.5, 0.5, 0.0, 0.0),
    ("gen_mi_only",  0.5, 0.0, 0.0, 0.5),
]


def run_with_exponents(
    bench: Dict[str, Any],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> Dict[str, Any]:
    """Run CEGAR with modified quality scoring exponents."""
    # Monkey-patch the scorer temporarily
    original_score = PredicateQualityScorer.score

    def patched_score(self, pred):
        g = self._generality_score(pred)
        c = self._coverage_score(pred)
        k = self._consistency_score(pred)
        m = self._mutual_information_score(pred)
        if alpha == 0 and beta == 0 and gamma == 0 and delta == 0:
            return 1.0  # accept all
        return (g ** alpha) * (c ** beta) * (k ** gamma) * (m ** delta)

    PredicateQualityScorer.score = patched_score
    try:
        t0 = time.monotonic()
        result = run_shape_cegar(
            bench["code"],
            input_shapes=bench["input_shapes"],
            max_iterations=10,
        )
        elapsed = (time.monotonic() - t0) * 1000

        detected_bug = result.final_status.name == "REAL_BUG_FOUND"
        correct = (detected_bug == bench["has_bug"]) or (
            not detected_bug and not bench["has_bug"]
        )

        return {
            "status": result.final_status.name,
            "iterations": result.iterations,
            "predicates": len(result.discovered_predicates),
            "time_ms": round(elapsed, 1),
            "detected_bug": detected_bug,
            "correct": correct,
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "correct": False,
        }
    finally:
        PredicateQualityScorer.score = original_score


def run_sensitivity_analysis() -> Dict[str, Any]:
    """Run full sensitivity analysis across all configs and benchmarks."""
    results = {
        "configs": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for config_name, alpha, beta, gamma, delta in EXPONENT_CONFIGS:
        print(f"\nConfig: {config_name} (α={alpha}, β={beta}, γ={gamma}, δ={delta})")
        config_results = {
            "name": config_name,
            "exponents": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
            "benchmarks": [],
            "accuracy": 0,
            "total_iterations": 0,
            "total_time_ms": 0,
        }

        correct_count = 0
        for bench in SENSITIVITY_BENCHMARKS:
            r = run_with_exponents(bench, alpha, beta, gamma, delta)
            r["benchmark"] = bench["name"]
            r["has_bug"] = bench["has_bug"]
            config_results["benchmarks"].append(r)
            if r.get("correct", False):
                correct_count += 1
            config_results["total_iterations"] += r.get("iterations", 0)
            config_results["total_time_ms"] += r.get("time_ms", 0)
            status_mark = "✓" if r.get("correct") else "✗"
            print(f"  {status_mark} {bench['name']}: {r.get('status', 'ERR')} "
                  f"({r.get('iterations', '?')} iters, {r.get('predicates', '?')} preds)")

        config_results["accuracy"] = round(correct_count / len(SENSITIVITY_BENCHMARKS), 3)
        config_results["total_time_ms"] = round(config_results["total_time_ms"], 1)
        results["configs"].append(config_results)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Quality Scoring Sensitivity Analysis")
    print("=" * 60)

    results = run_sensitivity_analysis()

    out_path = os.path.join(
        os.path.dirname(__file__),
        "quality_sensitivity_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<18} {'Accuracy':>8} {'Iters':>6} {'Time(ms)':>10}")
    print("-" * 44)
    for cfg in results["configs"]:
        print(f"{cfg['name']:<18} {cfg['accuracy']:>8.3f} "
              f"{cfg['total_iterations']:>6} {cfg['total_time_ms']:>10.1f}")

    print(f"\nResults saved to: {out_path}")
