"""
Reshape Solver Timeout Distribution Analysis for TensorGuard.

Characterizes the solver's behavior on the NP-hard reshape fragment,
measuring timeout rates and timing distributions across varying complexity.

Addresses Cheng critique: "Solver timeout distribution for NP-hard
reshape fragment uncharacterized."
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

RESULTS_FILE = Path(__file__).parent / "reshape_timeout_distribution_results.json"

NUM_RUNS = 5  # Per configuration


def generate_reshape_model(n_reshapes: int, dims: List[int]) -> str:
    """Generate a model with n_reshapes reshape operations of increasing complexity."""
    lines = [
        "import torch",
        "import torch.nn as nn",
        f"class ReshapeModel_{n_reshapes}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    # Add a linear layer to establish initial shape
    in_dim = dims[0] if dims else 256
    lines.append(f"        self.fc_in = nn.Linear({in_dim}, {in_dim})")

    # Add layers between reshapes
    for i in range(n_reshapes):
        lines.append(f"        self.fc{i} = nn.Linear({in_dim}, {in_dim})")

    lines.append("    def forward(self, x):")
    lines.append("        x = self.fc_in(x)")

    for i in range(n_reshapes):
        # Reshape: flatten and reshape back
        target_shape = ", ".join(str(d) for d in dims[1:]) if len(dims) > 1 else str(in_dim)
        lines.append(f"        x = x.view(-1, {target_shape})")
        lines.append(f"        x = self.fc{i}(x)")

    lines.append("        return x")
    return "\n".join(lines)


def generate_valid_reshape_chain(depth: int) -> tuple:
    """Generate a model with a chain of valid reshapes."""
    dim = 256
    lines = [
        "import torch",
        "import torch.nn as nn",
        f"class ValidReshapeChain_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.fc_in = nn.Linear({dim}, {dim})",
    ]
    for i in range(depth):
        lines.append(f"        self.fc{i} = nn.Linear({dim}, {dim})")

    lines.append("    def forward(self, x):")
    lines.append("        x = self.fc_in(x)")

    for i in range(depth):
        # Valid reshape: 256 = 16 * 16 = 4 * 64 = 8 * 32
        if i % 3 == 0:
            lines.append(f"        x = x.view(-1, 16, 16)")
            lines.append(f"        x = x.view(-1, {dim})")
        elif i % 3 == 1:
            lines.append(f"        x = x.view(-1, 4, 64)")
            lines.append(f"        x = x.view(-1, {dim})")
        else:
            lines.append(f"        x = x.view(-1, 8, 32)")
            lines.append(f"        x = x.view(-1, {dim})")
        lines.append(f"        x = self.fc{i}(x)")

    lines.append("        return x")
    return "\n".join(lines), {"x": ("batch", dim)}


def generate_buggy_reshape(depth: int) -> tuple:
    """Generate a model with a buggy reshape (wrong dimensions)."""
    dim = 256
    lines = [
        "import torch",
        "import torch.nn as nn",
        f"class BuggyReshape_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.fc_in = nn.Linear({dim}, {dim})",
    ]
    for i in range(depth):
        lines.append(f"        self.fc{i} = nn.Linear({dim}, {dim})")

    lines.append("    def forward(self, x):")
    lines.append("        x = self.fc_in(x)")

    for i in range(depth - 1):
        lines.append(f"        x = x.view(-1, 16, 16)")
        lines.append(f"        x = x.view(-1, {dim})")
        lines.append(f"        x = self.fc{i}(x)")

    # Last reshape is buggy: 256 != 15 * 17 = 255
    lines.append(f"        x = x.view(-1, 15, 17)")
    lines.append(f"        x = x.view(-1, {dim})")
    lines.append(f"        x = self.fc{depth - 1}(x)")
    lines.append("        return x")
    return "\n".join(lines), {"x": ("batch", dim)}


def generate_symbolic_reshape(depth: int) -> tuple:
    """Generate reshape with symbolic dimensions (triggers NP-hard fragment)."""
    lines = [
        "import torch",
        "import torch.nn as nn",
        f"class SymbolicReshape_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.fc_in = nn.Linear(256, 256)",
    ]
    for i in range(depth):
        lines.append(f"        self.fc{i} = nn.Linear(256, 256)")

    lines.append("    def forward(self, x):")
    lines.append("        x = self.fc_in(x)")

    for i in range(depth):
        lines.append(f"        x = x.view(-1, 16, 16)")
        lines.append(f"        x = x.view(-1, 256)")
        lines.append(f"        x = self.fc{i}(x)")

    lines.append("        return x")
    # Use symbolic batch dimension
    return "\n".join(lines), {"x": ("batch", 256)}


def benchmark_reshape(
    source: str, shapes: dict, label: str, n_runs: int = NUM_RUNS
) -> Dict[str, Any]:
    """Run verification n_runs times, collecting timing and timeout info."""
    times = []
    verdicts = []
    timeouts = 0

    for _ in range(n_runs):
        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=shapes)
            elapsed_ms = (time.monotonic() - t0) * 1000
            times.append(elapsed_ms)
            verdicts.append("safe" if result.safe else "unsafe")
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            times.append(elapsed_ms)
            if "timeout" in str(e).lower():
                timeouts += 1
                verdicts.append("timeout")
            else:
                verdicts.append(f"error: {str(e)[:50]}")

    n = len(times)
    mean = sum(times) / n if n > 0 else 0
    std = math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1)) if n > 1 else 0
    cv = std / mean if mean > 0 else 0

    return {
        "label": label,
        "n_runs": n_runs,
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "cv": round(cv, 4),
        "min_ms": round(min(times), 2) if times else 0,
        "max_ms": round(max(times), 2) if times else 0,
        "timeout_rate": round(timeouts / n_runs, 3),
        "verdicts": verdicts,
        "verdict_consistent": len(set(verdicts)) == 1,
    }


def run_reshape_timeout_analysis() -> Dict[str, Any]:
    """Run reshape timeout distribution analysis."""
    results = {
        "experiment": "reshape_timeout_distribution",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs_per_config": NUM_RUNS,
        "benchmarks": [],
    }

    print("=" * 70)
    print("Reshape Solver Timeout Distribution Analysis")
    print("=" * 70)

    # 1. Valid reshape chains of increasing depth
    print("\n--- Valid Reshape Chains ---")
    for depth in [1, 2, 5, 10, 20]:
        source, shapes = generate_valid_reshape_chain(depth)
        r = benchmark_reshape(source, shapes, f"valid_chain_{depth}")
        results["benchmarks"].append(r)
        print(
            f"  depth={depth:3d}: mean={r['mean_ms']:8.1f}ms ± {r['std_ms']:5.1f}ms  "
            f"CV={r['cv']:.3f}  timeout_rate={r['timeout_rate']}"
        )

    # 2. Buggy reshape chains
    print("\n--- Buggy Reshape Chains ---")
    for depth in [1, 2, 5, 10, 20]:
        source, shapes = generate_buggy_reshape(depth)
        r = benchmark_reshape(source, shapes, f"buggy_chain_{depth}")
        results["benchmarks"].append(r)
        print(
            f"  depth={depth:3d}: mean={r['mean_ms']:8.1f}ms ± {r['std_ms']:5.1f}ms  "
            f"CV={r['cv']:.3f}  verdict={r['verdicts'][0]}"
        )

    # 3. Symbolic dimension reshapes
    print("\n--- Symbolic Dimension Reshapes ---")
    for depth in [1, 2, 5, 10]:
        source, shapes = generate_symbolic_reshape(depth)
        r = benchmark_reshape(source, shapes, f"symbolic_reshape_{depth}")
        results["benchmarks"].append(r)
        print(
            f"  depth={depth:3d}: mean={r['mean_ms']:8.1f}ms ± {r['std_ms']:5.1f}ms  "
            f"CV={r['cv']:.3f}  verdict={r['verdicts'][0]}"
        )

    # Summary
    all_timeouts = sum(b["timeout_rate"] > 0 for b in results["benchmarks"])
    all_consistent = all(b["verdict_consistent"] for b in results["benchmarks"])
    max_time = max(b["max_ms"] for b in results["benchmarks"])

    results["summary"] = {
        "configs_with_timeouts": all_timeouts,
        "total_configs": len(results["benchmarks"]),
        "all_verdicts_consistent": all_consistent,
        "max_time_ms": round(max_time, 2),
        "zero_timeout_rate": all_timeouts == 0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Configs with timeouts: {all_timeouts}/{len(results['benchmarks'])}")
    print(f"All verdicts consistent: {all_consistent}")
    print(f"Max time: {max_time:.1f}ms")

    return results


if __name__ == "__main__":
    run_reshape_timeout_analysis()
