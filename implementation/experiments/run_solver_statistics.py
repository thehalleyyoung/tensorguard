"""
Z3 Solver Statistics Analysis for TensorGuard.

Captures per-benchmark solver statistics (queries, sat/unsat counts,
propagator activity, theory combination) to characterize Z3's behavior
across different architecture types and model sizes.

Provides depth to the scalability section by showing the solver's
internal workload distribution.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

RESULTS_FILE = Path(__file__).parent / "solver_statistics_analysis_results.json"


def generate_mlp(depth: int, dim: int = 256) -> str:
    lines = [
        "import torch.nn as nn",
        f"class MLP_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(f"        self.fc{i} = nn.Linear({dim}, {dim})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    lines.append(f"        self.head = nn.Linear({dim}, 10)")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        x = self.act{i}(self.fc{i}(x))")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def generate_cnn(depth: int, ch: int = 64) -> str:
    lines = [
        "import torch.nn as nn",
        f"class CNN_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(
            f"        self.conv{i} = nn.Conv2d({ch}, {ch}, 3, padding=1)"
        )
        lines.append(f"        self.bn{i} = nn.BatchNorm2d({ch})")
        lines.append(f"        self.act{i} = nn.ReLU()")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        x = self.act{i}(self.bn{i}(self.conv{i}(x)))")
    lines.append("        return x")
    return "\n".join(lines)


def generate_transformer(depth: int) -> str:
    lines = [
        "import torch.nn as nn",
        f"class Transformer_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        f"        self.encoder = nn.TransformerEncoder(",
        f"            nn.TransformerEncoderLayer(d_model=512, nhead=8),",
        f"            num_layers={depth})",
        f"        self.fc = nn.Linear(512, 10)",
        "    def forward(self, x):",
        "        x = self.encoder(x)",
        "        return self.fc(x)",
    ]
    return "\n".join(lines)


def generate_broadcast_model(depth: int) -> str:
    """Model with broadcast operations between parallel branches."""
    lines = [
        "import torch.nn as nn",
        f"class BroadcastModel_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(f"        self.fc_a{i} = nn.Linear(256, 256)")
        lines.append(f"        self.fc_b{i} = nn.Linear(256, 256)")
    lines.append(f"        self.head = nn.Linear(256, 10)")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        a = self.fc_a{i}(x)")
        lines.append(f"        b = self.fc_b{i}(x)")
        lines.append(f"        x = a + b")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def generate_device_model(depth: int) -> str:
    """Model with device transfer operations."""
    lines = [
        "import torch",
        "import torch.nn as nn",
        f"class DeviceModel_{depth}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    for i in range(depth):
        lines.append(f"        self.fc{i} = nn.Linear(256, 256)")
    lines.append(f"        self.head = nn.Linear(256, 10)")
    lines.append("    def forward(self, x):")
    for i in range(depth):
        lines.append(f"        x = self.fc{i}(x)")
    lines.append("        return self.head(x)")
    return "\n".join(lines)


def collect_stats(source: str, shapes: dict) -> Dict[str, Any]:
    """Run verification and collect solver statistics from certificate."""
    t0 = time.monotonic()
    result = verify_model(source, input_shapes=shapes)
    elapsed_ms = (time.monotonic() - t0) * 1000

    stats = {}
    if result.safe and result.certificate:
        cert = result.certificate
        stats = {
            "z3_queries": getattr(cert, "z3_queries", 0),
            "z3_total_time_ms": getattr(cert, "z3_total_time_ms", 0.0),
            "z3_sat_count": getattr(cert, "z3_sat_count", 0),
            "z3_unsat_count": getattr(cert, "z3_unsat_count", 0),
            "checked_steps": getattr(cert, "checked_steps", 0),
            "theories_used": getattr(cert, "theories_used", []),
        }

    return {
        "safe": result.safe,
        "time_ms": round(elapsed_ms, 2),
        "num_violations": len(result.errors) if result.errors else 0,
        "solver_stats": stats,
    }


def run_solver_statistics_analysis() -> Dict[str, Any]:
    """Run comprehensive solver statistics analysis."""
    results = {
        "experiment": "solver_statistics_analysis",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": [],
    }

    print("=" * 70)
    print("Z3 Solver Statistics Analysis")
    print("=" * 70)

    configs = [
        ("MLP", generate_mlp, {"x": ("batch", 256)}, [5, 10, 20, 50]),
        ("CNN", generate_cnn, {"x": ("batch", 64, 32, 32)}, [5, 10, 20]),
        ("Transformer", generate_transformer, {"x": ("batch", 32, 512)}, [5, 10, 20]),
        ("Broadcast", generate_broadcast_model, {"x": ("batch", 256)}, [5, 10, 20]),
        ("Device", generate_device_model, {"x": ("batch", 256)}, [5, 10, 20]),
    ]

    for model_type, gen_fn, shapes, depths in configs:
        print(f"\n--- {model_type} ---")
        for depth in depths:
            source = gen_fn(depth)
            r = collect_stats(source, shapes)
            entry = {
                "model_type": model_type,
                "depth": depth,
                **r,
            }
            results["benchmarks"].append(entry)
            stats = r["solver_stats"]
            print(
                f"  {model_type}-{depth:3d}: "
                f"time={r['time_ms']:8.1f}ms  "
                f"safe={r['safe']}  "
                f"queries={stats.get('z3_queries', 'N/A')}  "
                f"sat={stats.get('z3_sat_count', 'N/A')}  "
                f"unsat={stats.get('z3_unsat_count', 'N/A')}  "
                f"solve_ms={stats.get('z3_total_time_ms', 'N/A')}"
            )

    # Summary: queries per computation step
    print("\n--- Summary ---")
    for b in results["benchmarks"]:
        stats = b.get("solver_stats", {})
        queries = stats.get("z3_queries", 0)
        if queries:
            print(f"  {b['model_type']}-{b['depth']}: "
                  f"{queries} queries, "
                  f"broadcast_props={stats.get('broadcast_propagations', 0)}, "
                  f"device_pairs={stats.get('device_same_pairs', 0)}, "
                  f"stride_constraints={stats.get('stride_constraints', 0)}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    run_solver_statistics_analysis()
