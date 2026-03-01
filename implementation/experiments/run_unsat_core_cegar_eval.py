#!/usr/bin/env python3
"""
Experiment: compare standard CEGAR vs UNSAT-core-enhanced CEGAR.

Metrics per benchmark:
  - iterations to convergence
  - predicates discovered
  - core-derived vs template-derived ratio
  - wall-clock time (ms)

Results are saved to .benchmarks/unsat_core_cegar_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shape_cegar import ShapeCEGARLoop, ShapeCEGARResult, CEGARStatus
from src.unsat_core_cegar import EnhancedShapeCEGARLoop, run_enhanced_cegar

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark suite (20+ models)
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "single_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "two_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
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
        "name": "three_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "linear_concrete_safe",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": (32, 768)},
    },
    {
        "name": "conv2d_simple",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "conv2d_concrete",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": (1, 3, 32, 32)},
    },
    {
        "name": "batchnorm_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc(self.bn(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "layernorm_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        return self.fc(self.ln(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "embedding_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc(self.embed(x))
""",
        "input_shapes": {"x": ("batch", "seq_len")},
    },
    {
        "name": "linear_bug_mismatch",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "deep_linear_4",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc4(self.fc3(self.fc2(self.fc1(x))))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "conv_bn_relu",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        return self.bn(self.conv(x))
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "single_linear_large",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 1024)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "groupnorm_conv",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.gn = nn.GroupNorm(8, 32)
    def forward(self, x):
        return self.gn(self.conv(x))
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "mlp_3_layer",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "single_conv1d",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, 3)
    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": ("batch", "channels", "length")},
    },
    {
        "name": "identity_safe",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "linear_no_shapes",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {},
    },
    {
        "name": "double_conv2d",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
    def forward(self, x):
        return self.conv2(self.conv1(x))
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "instancenorm_conv",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.in_norm = nn.InstanceNorm2d(16)
    def forward(self, x):
        return self.in_norm(self.conv(x))
""",
        "input_shapes": {"x": ("batch", "channels", "h", "w")},
    },
    {
        "name": "wide_linear",
        "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, 2048)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
]


def _run_one(
    bench: Dict[str, Any],
    use_enhanced: bool,
) -> Dict[str, Any]:
    """Run a single benchmark with either standard or enhanced CEGAR."""
    source = bench["source"]
    input_shapes = bench["input_shapes"]

    t0 = time.monotonic()
    try:
        if use_enhanced:
            loop = EnhancedShapeCEGARLoop(
                source, input_shapes=input_shapes, max_iterations=10,
            )
        else:
            loop = ShapeCEGARLoop(
                source, input_shapes=input_shapes, max_iterations=10,
            )
        result = loop.run()
        elapsed = (time.monotonic() - t0) * 1000

        stats = result.interpolation_stats or {}
        return {
            "name": bench["name"],
            "mode": "enhanced" if use_enhanced else "standard",
            "status": result.final_status.name,
            "verdict": result.verdict.name,
            "iterations": result.iterations,
            "predicates": len(result.discovered_predicates),
            "core_predicates": stats.get("core_predicates_count", 0),
            "template_predicates": stats.get("template_predicates_count", 0),
            "solver_reuse": stats.get("solver_reuse_count", 0),
            "time_ms": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "name": bench["name"],
            "mode": "enhanced" if use_enhanced else "standard",
            "status": "ERROR",
            "verdict": "ERROR",
            "iterations": 0,
            "predicates": 0,
            "core_predicates": 0,
            "template_predicates": 0,
            "solver_reuse": 0,
            "time_ms": round(elapsed, 2),
            "error": str(e),
        }


def main() -> None:
    results: List[Dict[str, Any]] = []

    for bench in BENCHMARKS:
        print(f"  [{bench['name']}] standard ...", end=" ", flush=True)
        std = _run_one(bench, use_enhanced=False)
        print(f"{std['time_ms']:.0f}ms  {std['status']}")
        results.append(std)

        print(f"  [{bench['name']}] enhanced ...", end=" ", flush=True)
        enh = _run_one(bench, use_enhanced=True)
        print(f"{enh['time_ms']:.0f}ms  {enh['status']}")
        results.append(enh)

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Benchmark':<30} {'Std ms':>8} {'Enh ms':>8} {'Std It':>7} {'Enh It':>7} {'Core':>5}")
    print("-" * 72)
    for i in range(0, len(results), 2):
        std, enh = results[i], results[i + 1]
        print(
            f"{std['name']:<30} "
            f"{std['time_ms']:>8.1f} "
            f"{enh['time_ms']:>8.1f} "
            f"{std['iterations']:>7} "
            f"{enh['iterations']:>7} "
            f"{enh['core_predicates']:>5}"
        )

    # Save
    out_dir = Path(__file__).resolve().parents[1] / ".benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unsat_core_cegar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
