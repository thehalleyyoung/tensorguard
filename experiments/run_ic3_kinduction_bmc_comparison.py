"""
IC3/PDR vs K-Induction vs BMC Comparison Experiment.

Compares three unbounded/bounded verification approaches on the same
benchmarks to establish IC3/PDR's advantages quantitatively.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ic3_pdr import ic3_verify
from src.k_induction import k_induction_verify, KInductionVerdict
from src.bmc_baseline import verify_model_bmc as bmc_verify, BMCVerdict


BENCHMARKS = [
    {
        "name": "safe_mlp_2layer",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 128)},
        "expected": "SAFE",
        "category": "linear_chain",
    },
    {
        "name": "buggy_mlp_mismatch",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 128)},
        "expected": "UNSAFE",
        "category": "linear_chain",
    },
    {
        "name": "safe_deep_5layer",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "expected": "SAFE",
        "category": "deep_chain",
    },
    {
        "name": "buggy_deep_5layer",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(32, 16)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "expected": "UNSAFE",
        "category": "deep_chain",
    },
    {
        "name": "safe_cnn_chain",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected": "SAFE",
        "category": "cnn",
    },
    {
        "name": "buggy_cnn_channel",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected": "UNSAFE",
        "category": "cnn",
    },
    {
        "name": "safe_transformer_block",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.norm2(x)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
        "expected": "SAFE",
        "category": "transformer",
    },
    {
        "name": "buggy_transformer_proj",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.norm2(x)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
        "expected": "UNSAFE",
        "category": "transformer",
    },
    {
        "name": "safe_uniform_10layer",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        return self.fc10(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "expected": "SAFE",
        "category": "deep_uniform",
    },
    {
        "name": "safe_bottleneck",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 256)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 256)},
        "expected": "SAFE",
        "category": "bottleneck",
    },
    {
        "name": "safe_norm_dropout",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "expected": "SAFE",
        "category": "norm_chain",
    },
    {
        "name": "buggy_norm_mismatch",
        "source": """
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "expected": "UNSAFE",
        "category": "norm_chain",
    },
]


def run_experiment():
    results = []
    summary = {
        "ic3_correct": 0,
        "kinduction_correct": 0,
        "bmc_correct": 0,
        "total": len(BENCHMARKS),
        "ic3_total_ms": 0.0,
        "kinduction_total_ms": 0.0,
        "bmc_total_ms": 0.0,
        "agreements": {"ic3_ki": 0, "ic3_bmc": 0, "ki_bmc": 0, "all_three": 0},
    }

    for bench in BENCHMARKS:
        print(f"\n--- {bench['name']} (expected: {bench['expected']}) ---")

        # IC3/PDR
        ic3_result = ic3_verify(
            bench["source"],
            input_shapes=bench["input_shapes"],
            symbolic_dims={"batch": "B"},
        )
        ic3_verdict = "SAFE" if ic3_result.safe else "UNSAFE"
        ic3_ms = ic3_result.verification_time_ms

        # K-Induction
        ki_result = k_induction_verify(
            bench["source"],
            input_shapes=bench["input_shapes"],
            symbolic_dims={"batch": "B"},
        )
        ki_verdict = ki_result.verdict.name
        ki_ms = ki_result.verification_time_ms

        # BMC
        bmc_result = bmc_verify(bench["source"], input_shapes=bench["input_shapes"])
        bmc_verdict = bmc_result.verdict.name
        bmc_ms = bmc_result.time_ms

        ic3_correct = ic3_verdict == bench["expected"]
        ki_correct = ki_verdict == bench["expected"]
        bmc_correct = bmc_verdict == bench["expected"]

        summary["ic3_correct"] += int(ic3_correct)
        summary["kinduction_correct"] += int(ki_correct)
        summary["bmc_correct"] += int(bmc_correct)
        summary["ic3_total_ms"] += ic3_ms
        summary["kinduction_total_ms"] += ki_ms
        summary["bmc_total_ms"] += bmc_ms

        if ic3_verdict == ki_verdict:
            summary["agreements"]["ic3_ki"] += 1
        if ic3_verdict == bmc_verdict:
            summary["agreements"]["ic3_bmc"] += 1
        if ki_verdict == bmc_verdict:
            summary["agreements"]["ki_bmc"] += 1
        if ic3_verdict == ki_verdict == bmc_verdict:
            summary["agreements"]["all_three"] += 1

        entry = {
            "name": bench["name"],
            "category": bench["category"],
            "expected": bench["expected"],
            "ic3": {
                "verdict": ic3_verdict,
                "time_ms": round(ic3_ms, 2),
                "frames": ic3_result.frames_computed,
                "z3_queries": ic3_result.z3_queries,
                "correct": ic3_correct,
            },
            "k_induction": {
                "verdict": ki_verdict,
                "time_ms": round(ki_ms, 2),
                "k": ki_result.k,
                "z3_queries": ki_result.z3_queries,
                "correct": ki_correct,
            },
            "bmc": {
                "verdict": bmc_verdict,
                "time_ms": round(bmc_ms, 2),
                "correct": bmc_correct,
            },
        }
        results.append(entry)

        print(
            f"  IC3: {ic3_verdict} ({ic3_ms:.1f}ms, {ic3_result.frames_computed} frames)"
        )
        print(f"  K-Ind: {ki_verdict} ({ki_ms:.1f}ms, k={ki_result.k})")
        print(f"  BMC: {bmc_verdict} ({bmc_ms:.1f}ms)")

    # Compute speedup ratios
    if summary["kinduction_total_ms"] > 0:
        summary["ic3_vs_kinduction_speedup"] = round(
            summary["kinduction_total_ms"] / max(summary["ic3_total_ms"], 0.01), 2
        )
    if summary["bmc_total_ms"] > 0:
        summary["ic3_vs_bmc_speedup"] = round(
            summary["bmc_total_ms"] / max(summary["ic3_total_ms"], 0.01), 2
        )
        summary["kinduction_vs_bmc_speedup"] = round(
            summary["bmc_total_ms"] / max(summary["kinduction_total_ms"], 0.01), 2
        )

    output = {
        "experiment": "IC3/PDR vs K-Induction vs BMC comparison",
        "num_benchmarks": len(BENCHMARKS),
        "summary": summary,
        "results": results,
    }

    outpath = os.path.join(
        os.path.dirname(__file__), "results", "ic3_kinduction_bmc_comparison.json"
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")
    print(f"\nSummary:")
    print(f"  IC3:   {summary['ic3_correct']}/{summary['total']} correct, {summary['ic3_total_ms']:.1f}ms total")
    print(f"  K-Ind: {summary['kinduction_correct']}/{summary['total']} correct, {summary['kinduction_total_ms']:.1f}ms total")
    print(f"  BMC:   {summary['bmc_correct']}/{summary['total']} correct, {summary['bmc_total_ms']:.1f}ms total")
    print(f"  Agreement: {summary['agreements']}")
    return output


if __name__ == "__main__":
    run_experiment()
