"""
IC3/PDR Evaluation: Bounded vs. Unbounded Verification Comparison.

Runs IC3/PDR on 10+ models of varying depth and compares with
bounded model checking.  Results are saved to .benchmarks/ic3_results.json.
"""

import json
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ic3_pdr import ic3_verify
from src.model_checker import verify_model

# ---------------------------------------------------------------------------
# Benchmark models
# ---------------------------------------------------------------------------

MODELS = {
    "linear_1layer": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "depth": 1,
    },
    "linear_2layer": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "depth": 2,
    },
    "linear_3layer": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 100)},
        "depth": 3,
    },
    "linear_5layer": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "depth": 5,
    },
    "mlp_with_relu": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
        "depth": 3,
    },
    "mlp_with_dropout": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 512)},
        "depth": 4,
    },
    "conv_2layer": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "depth": 2,
    },
    "wide_mlp": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 1024)},
        "depth": 3,
    },
    "unsafe_mismatch": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "depth": 2,
    },
    "unsafe_deep": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(99, 4)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "depth": 4,
    },
    "batchnorm_chain": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "depth": 5,
    },
    "identity_passthrough": {
        "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "depth": 1,
    },
}


def run_evaluation() -> dict:
    """Run IC3/PDR vs bounded verification on all benchmark models."""
    results = []

    for name, spec in MODELS.items():
        print(f"  [{name}] (depth={spec['depth']}) ... ", end="", flush=True)
        entry = {
            "model": name,
            "depth": spec["depth"],
        }

        # --- Bounded verification ---
        t0 = time.monotonic()
        bounded_result = verify_model(
            spec["source"],
            input_shapes=spec["input_shapes"],
        )
        bounded_ms = (time.monotonic() - t0) * 1000
        entry["bounded_safe"] = bounded_result.safe
        entry["bounded_time_ms"] = round(bounded_ms, 2)

        # --- IC3/PDR unbounded verification ---
        ic3_result = ic3_verify(
            spec["source"],
            symbolic_dims={"batch": "batch_size"},
            input_shapes=spec["input_shapes"],
        )
        entry["ic3_safe"] = ic3_result.safe
        entry["ic3_time_ms"] = round(ic3_result.verification_time_ms, 2)
        entry["ic3_frames"] = ic3_result.frames_computed
        entry["ic3_z3_queries"] = ic3_result.z3_queries
        entry["ic3_blocked_cubes"] = ic3_result.num_blocked_cubes
        entry["ic3_has_invariant"] = ic3_result.invariant is not None
        entry["ic3_cex_depth"] = ic3_result.counterexample_depth

        # Verdicts should agree
        entry["verdicts_agree"] = bounded_result.safe == ic3_result.safe

        status = "SAFE" if ic3_result.safe else "UNSAFE"
        agree = "✓" if entry["verdicts_agree"] else "✗"
        print(
            f"{status}  bounded={bounded_ms:.1f}ms  ic3={ic3_result.verification_time_ms:.1f}ms  "
            f"frames={ic3_result.frames_computed}  agree={agree}"
        )

        results.append(entry)

    # --- Summary ---
    total = len(results)
    agreed = sum(1 for r in results if r["verdicts_agree"])
    safe_count = sum(1 for r in results if r["ic3_safe"])
    unsafe_count = total - safe_count
    avg_ic3_ms = sum(r["ic3_time_ms"] for r in results) / total
    avg_bounded_ms = sum(r["bounded_time_ms"] for r in results) / total

    summary = {
        "total_models": total,
        "safe_models": safe_count,
        "unsafe_models": unsafe_count,
        "verdicts_agreed": agreed,
        "avg_ic3_time_ms": round(avg_ic3_ms, 2),
        "avg_bounded_time_ms": round(avg_bounded_ms, 2),
    }

    return {"results": results, "summary": summary}


def main() -> None:
    print("=" * 60)
    print("IC3/PDR Evaluation: Bounded vs Unbounded Verification")
    print("=" * 60)
    print()

    data = run_evaluation()

    print()
    print("-" * 60)
    s = data["summary"]
    print(f"Total models: {s['total_models']}")
    print(f"  Safe: {s['safe_models']}  Unsafe: {s['unsafe_models']}")
    print(f"  Verdicts agreed: {s['verdicts_agreed']}/{s['total_models']}")
    print(f"  Avg IC3 time:    {s['avg_ic3_time_ms']:.1f}ms")
    print(f"  Avg bounded time: {s['avg_bounded_time_ms']:.1f}ms")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ic3_results.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
