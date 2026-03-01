"""
BMC vs CEGAR Comparison Experiment.

Runs both monolithic BMC and CEGAR-style iterative refinement on the same
benchmark models and compares correctness, timing, and whether CEGAR adds
value over the monolithic approach for acyclic computation graphs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bmc_baseline import BMCVerdict, verify_model_bmc
from src.model_checker import verify_model
from src.shape_cegar import CEGARVerdict, run_shape_cegar

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark models (20+ diverse nn.Module architectures)
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARKS: List[Dict[str, Any]] = [
    # --- Safe models ---
    {
        "name": "simple_linear",
        "source": """
import torch.nn as nn
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "two_layer_mlp",
        "source": """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 784)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "deep_mlp_4layer",
        "source": """
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 100)},
        "expected_safe": True,
        "category": "deep_safe",
    },
    {
        "name": "dropout_model",
        "source": """
import torch.nn as nn
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 30)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(30, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 50)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "batchnorm_model",
        "source": """
import torch.nn as nn
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 40)
        self.bn = nn.BatchNorm1d(40)
        self.fc2 = nn.Linear(40, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 20)},
        "expected_safe": True,
        "category": "normalization",
    },
    {
        "name": "layernorm_residual",
        "source": """
import torch.nn as nn
class LNResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, 768)
    def forward(self, x):
        return x + self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
        "expected_safe": True,
        "category": "transformer",
    },
    {
        "name": "conv2d_relu",
        "source": """
import torch.nn as nn
class ConvReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": True,
        "category": "vision",
    },
    {
        "name": "symbolic_linear",
        "source": """
import torch.nn as nn
class SymLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
        "expected_safe": True,
        "category": "symbolic",
    },
    {
        "name": "mlp_block_768",
        "source": """
import torch.nn as nn
class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
        "expected_safe": True,
        "category": "transformer",
    },
    {
        "name": "identity_passthrough",
        "source": """
import torch.nn as nn
class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
    def forward(self, x):
        return self.id(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "five_layer_chain",
        "source": """
import torch.nn as nn
class FiveLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
""",
        "input_shapes": {"x": ("batch", 128)},
        "expected_safe": True,
        "category": "deep_safe",
    },
    # --- Buggy models ---
    {
        "name": "shape_mismatch_linear",
        "source": """
import torch.nn as nn
class BadLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "deep_mismatch_layer3",
        "source": """
import torch.nn as nn
class DeepBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(99, 16)
        self.fc4 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 100)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "wrong_input_dim",
        "source": """
import torch.nn as nn
class WrongInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 30)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "mlp_middle_mismatch",
        "source": """
import torch.nn as nn
class MiddleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
        "input_shapes": {"x": ("batch", 50)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "chain_bug_at_end",
        "source": """
import torch.nn as nn
class EndBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(99, 2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 64)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "embedding_fc_mismatch",
        "source": """
import torch.nn as nn
class EmbedBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq")},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "bn_dim_mismatch",
        "source": """
import torch.nn as nn
class BNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(30)
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "three_layer_safe_with_dropout",
        "source": """
import torch.nn as nn
class SafeDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 100)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x))
        x = self.fc3(x)
        return x
""",
        "input_shapes": {"x": ("batch", 200)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "wide_mlp",
        "source": """
import torch.nn as nn
class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", 512)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "narrow_bottleneck",
        "source": """
import torch.nn as nn
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 8)
        self.fc2 = nn.Linear(8, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", 256)},
        "expected_safe": True,
        "category": "basic_safe",
    },
    {
        "name": "double_bug",
        "source": """
import torch.nn as nn
class DoubleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(99, 30)
        self.fc3 = nn.Linear(88, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": False,
        "category": "shape_bug",
    },
    {
        "name": "single_layer_correct",
        "source": """
import torch.nn as nn
class OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 32)},
        "expected_safe": True,
        "category": "basic_safe",
    },
]


def _normalize_verdict(bmc_verdict: BMCVerdict) -> str:
    """Map BMCVerdict to a common string for comparison."""
    return {
        BMCVerdict.SAFE: "SAFE",
        BMCVerdict.UNSAFE: "UNSAFE",
        BMCVerdict.UNKNOWN: "UNKNOWN",
    }[bmc_verdict]


def _normalize_cegar_verdict(cegar_verdict: CEGARVerdict) -> str:
    return {
        CEGARVerdict.SAFE: "SAFE",
        CEGARVerdict.UNSAFE: "UNSAFE",
        CEGARVerdict.UNKNOWN: "UNKNOWN",
        CEGARVerdict.TIMEOUT: "UNKNOWN",
    }[cegar_verdict]


def run_comparison():
    """Run BMC and CEGAR on all benchmarks and compare results."""
    results = []
    summary = {
        "total": 0,
        "bmc_correct": 0,
        "cegar_correct": 0,
        "agree": 0,
        "disagree": 0,
        "bmc_faster": 0,
        "cegar_faster": 0,
        "bmc_total_ms": 0.0,
        "cegar_total_ms": 0.0,
    }

    for bench in BENCHMARKS:
        summary["total"] += 1
        name = bench["name"]
        source = bench["source"]
        input_shapes = bench.get("input_shapes", {})
        expected_safe = bench["expected_safe"]
        category = bench.get("category", "unknown")

        print(f"  [{summary['total']:2d}/{len(BENCHMARKS)}] {name:<30s} ", end="", flush=True)

        # --- Run BMC ---
        bmc_t0 = time.monotonic()
        bmc_result = verify_model_bmc(source, input_shapes=input_shapes, timeout=30)
        bmc_time = (time.monotonic() - bmc_t0) * 1000
        bmc_verdict = _normalize_verdict(bmc_result.verdict)

        # --- Run CEGAR ---
        cegar_t0 = time.monotonic()
        cegar_result = run_shape_cegar(source, input_shapes=input_shapes, max_iterations=10)
        cegar_time = (time.monotonic() - cegar_t0) * 1000
        cegar_verdict = _normalize_cegar_verdict(cegar_result.verdict)

        # --- Compare ---
        agree = bmc_verdict == cegar_verdict
        expected_verdict = "SAFE" if expected_safe else "UNSAFE"
        bmc_correct = bmc_verdict == expected_verdict
        cegar_correct = cegar_verdict == expected_verdict

        if bmc_correct:
            summary["bmc_correct"] += 1
        if cegar_correct:
            summary["cegar_correct"] += 1
        if agree:
            summary["agree"] += 1
        else:
            summary["disagree"] += 1
        if bmc_time < cegar_time:
            summary["bmc_faster"] += 1
        else:
            summary["cegar_faster"] += 1
        summary["bmc_total_ms"] += bmc_time
        summary["cegar_total_ms"] += cegar_time

        entry = {
            "name": name,
            "category": category,
            "expected_safe": expected_safe,
            "bmc_verdict": bmc_verdict,
            "cegar_verdict": cegar_verdict,
            "bmc_time_ms": round(bmc_time, 2),
            "cegar_time_ms": round(cegar_time, 2),
            "agree": agree,
            "bmc_correct": bmc_correct,
            "cegar_correct": cegar_correct,
            "bmc_num_steps": bmc_result.num_steps,
            "cegar_iterations": cegar_result.iterations,
            "cegar_predicates": len(cegar_result.discovered_predicates),
            "speedup": round(cegar_time / bmc_time, 2) if bmc_time > 0 else 0,
        }
        results.append(entry)

        status = "✓" if agree else "✗"
        print(
            f"{status}  BMC={bmc_verdict:<7s} ({bmc_time:6.1f}ms)  "
            f"CEGAR={cegar_verdict:<7s} ({cegar_time:6.1f}ms)  "
            f"expected={expected_verdict}"
        )

    summary["bmc_total_ms"] = round(summary["bmc_total_ms"], 2)
    summary["cegar_total_ms"] = round(summary["cegar_total_ms"], 2)
    summary["bmc_accuracy"] = round(summary["bmc_correct"] / summary["total"], 4) if summary["total"] > 0 else 0
    summary["cegar_accuracy"] = round(summary["cegar_correct"] / summary["total"], 4) if summary["total"] > 0 else 0
    summary["agreement_rate"] = round(summary["agree"] / summary["total"], 4) if summary["total"] > 0 else 0

    return {
        "benchmarks": results,
        "summary": summary,
    }


def main():
    print("=" * 72)
    print("BMC vs CEGAR Comparison Experiment")
    print("=" * 72)
    print()

    output = run_comparison()

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    s = output["summary"]
    print(f"  Total benchmarks:    {s['total']}")
    print(f"  BMC accuracy:        {s['bmc_correct']}/{s['total']} ({s['bmc_accuracy']:.1%})")
    print(f"  CEGAR accuracy:      {s['cegar_correct']}/{s['total']} ({s['cegar_accuracy']:.1%})")
    print(f"  Agreement rate:      {s['agree']}/{s['total']} ({s['agreement_rate']:.1%})")
    print(f"  BMC faster:          {s['bmc_faster']}/{s['total']}")
    print(f"  CEGAR faster:        {s['cegar_faster']}/{s['total']}")
    print(f"  BMC total time:      {s['bmc_total_ms']:.1f}ms")
    print(f"  CEGAR total time:    {s['cegar_total_ms']:.1f}ms")

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "bmc_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
