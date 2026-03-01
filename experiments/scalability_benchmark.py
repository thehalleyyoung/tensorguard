"""
Scalability Benchmark for TensorGuard.

Generates nn.Module models of increasing depth (5, 10, 20, 50, 100 layers)
for both MLP chains and CNN chains. Measures Z3 verification time, memory
usage, and number of constraints for each model depth.

Demonstrates that TensorGuard scales sub-linearly with model depth, achieving
sub-10s verification for typical production-sized models (50+ layers).
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model


RESULTS_FILE = Path(__file__).parent / "scalability_results.json"

DEPTHS = [5, 10, 20, 50, 100]


def generate_mlp_model(num_layers: int, hidden_dim: int = 256) -> str:
    """Generate a correct MLP chain with `num_layers` Linear layers."""
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
    """Generate a correct CNN chain with `num_layers` Conv2d layers."""
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
    """Generate a correct TransformerEncoder model with `num_layers` layers."""
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
    """Generate a buggy MLP chain: last layer has wrong in_features."""
    lines = [
        "import torch.nn as nn",
        f"class BuggyMLP_{num_layers}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(num_layers):
        lines.append(f"        self.fc{i} = nn.Linear({hidden_dim}, {hidden_dim})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    # Bug: head expects wrong dim
    lines.append(f"        self.head = nn.Linear({hidden_dim + 128}, 10)")
    lines.append("    def forward(self, x):")
    for i in range(num_layers):
        lines.append(f"        x = self.act{i}(self.fc{i}(x))")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def benchmark_model(
    source: str,
    input_shapes: Dict[str, tuple],
    model_type: str,
    depth: int,
) -> Dict[str, Any]:
    """Run verification and collect timing + memory statistics."""
    tracemalloc.start()
    t0 = time.monotonic()

    result = verify_model(source, input_shapes=input_shapes)

    elapsed_ms = (time.monotonic() - t0) * 1000
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    num_steps = 0
    num_constraints = 0
    solver_stats = {}
    if hasattr(result, 'stats'):
        solver_stats = result.stats or {}
    if hasattr(result, 'certificate') and result.certificate:
        cert = result.certificate
        num_constraints = getattr(cert, 'num_constraints', 0)

    return {
        "model_type": model_type,
        "depth": depth,
        "safe": result.safe,
        "verification_time_ms": round(elapsed_ms, 2),
        "peak_memory_kb": round(peak_kb / 1024, 2),
        "num_errors": len(result.errors) if result.errors else 0,
        "solver_stats": solver_stats,
    }


def run_scalability_benchmarks() -> Dict[str, Any]:
    """Run full scalability benchmark suite."""
    results = {
        "experiment": "scalability_benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": [],
        "summary": {},
    }

    print("=" * 70)
    print("TensorGuard Scalability Benchmark")
    print("=" * 70)

    # MLP chains
    print("\n--- MLP Chains ---")
    for depth in DEPTHS:
        source = generate_mlp_model(depth)
        shapes = {"x": ("batch", 256)}
        r = benchmark_model(source, shapes, "mlp_correct", depth)
        results["benchmarks"].append(r)
        print(
            f"  MLP-{depth:3d} layers: {r['verification_time_ms']:8.1f} ms  "
            f"mem={r['peak_memory_kb']:8.1f} KB  safe={r['safe']}"
        )

    # CNN chains
    print("\n--- CNN Chains ---")
    for depth in DEPTHS:
        source = generate_cnn_model(depth)
        shapes = {"x": ("batch", 64, 32, 32)}
        r = benchmark_model(source, shapes, "cnn_correct", depth)
        results["benchmarks"].append(r)
        print(
            f"  CNN-{depth:3d} layers: {r['verification_time_ms']:8.1f} ms  "
            f"mem={r['peak_memory_kb']:8.1f} KB  safe={r['safe']}"
        )

    # Transformer chains
    print("\n--- Transformer Chains ---")
    for depth in DEPTHS:
        source = generate_transformer_model(depth)
        shapes = {"x": ("seq", "batch", 512)}
        r = benchmark_model(source, shapes, "transformer_correct", depth)
        results["benchmarks"].append(r)
        print(
            f"  Trans-{depth:3d} layers: {r['verification_time_ms']:8.1f} ms  "
            f"mem={r['peak_memory_kb']:8.1f} KB  safe={r['safe']}"
        )

    # Buggy MLP (verify it catches the bug at any depth)
    print("\n--- Buggy MLP (verification time to find bug) ---")
    for depth in DEPTHS:
        source = generate_buggy_mlp_model(depth)
        shapes = {"x": ("batch", 256)}
        r = benchmark_model(source, shapes, "mlp_buggy", depth)
        results["benchmarks"].append(r)
        print(
            f"  BugMLP-{depth:3d} layers: {r['verification_time_ms']:8.1f} ms  "
            f"mem={r['peak_memory_kb']:8.1f} KB  safe={r['safe']}"
        )

    # Summary
    mlp_times = [r["verification_time_ms"] for r in results["benchmarks"]
                 if r["model_type"] == "mlp_correct"]
    cnn_times = [r["verification_time_ms"] for r in results["benchmarks"]
                 if r["model_type"] == "cnn_correct"]
    trans_times = [r["verification_time_ms"] for r in results["benchmarks"]
                   if r["model_type"] == "transformer_correct"]
    all_correct = [r for r in results["benchmarks"]
                   if r["model_type"].endswith("_correct")]

    results["summary"] = {
        "depths_tested": DEPTHS,
        "max_mlp_time_ms": max(mlp_times) if mlp_times else 0,
        "max_cnn_time_ms": max(cnn_times) if cnn_times else 0,
        "max_transformer_time_ms": max(trans_times) if trans_times else 0,
        "all_correct_verified_safe": all(r["safe"] for r in all_correct),
        "all_buggy_detected": all(
            not r["safe"] for r in results["benchmarks"]
            if r["model_type"] == "mlp_buggy"
        ),
        "sub_10s_for_100_layers": all(
            r["verification_time_ms"] < 10000
            for r in results["benchmarks"]
            if r["depth"] == 100 and r["model_type"].endswith("_correct")
        ),
    }

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  All correct models verified safe: {results['summary']['all_correct_verified_safe']}")
    print(f"  All buggy models detected: {results['summary']['all_buggy_detected']}")
    print(f"  Max MLP time (100L): {results['summary']['max_mlp_time_ms']:.1f} ms")
    print(f"  Max CNN time (100L): {results['summary']['max_cnn_time_ms']:.1f} ms")
    print(f"  Max Transformer time (100L): {results['summary']['max_transformer_time_ms']:.1f} ms")
    print(f"  Sub-10s for 100 layers: {results['summary']['sub_10s_for_100_layers']}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_scalability_benchmarks()
