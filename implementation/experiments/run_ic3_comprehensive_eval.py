"""
Comprehensive IC3/PDR Evaluation: Parametric, Deep, Branching, and Scalability Benchmarks.

Tests IC3/PDR on challenging models including:
  - Parametric architectures (ResNet(n), Transformer(n_layers, n_heads, d_model))
  - Deep chains (10, 20, 50 layers)
  - Shape mismatches at various depths
  - Branching (residual connections, skip connections)
  - Timing comparison: IC3/PDR vs bounded model checking
  - Scalability: how IC3 performance scales with depth

Results saved to .benchmarks/ic3_comprehensive_results.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ic3_pdr import ic3_verify
from src.model_checker import verify_model


# ---------------------------------------------------------------------------
# Helpers to generate model source code
# ---------------------------------------------------------------------------

def _make_linear_chain(n_layers: int, widths: list[int]) -> str:
    """Generate an n-layer linear chain model."""
    assert len(widths) == n_layers + 1
    init_lines = []
    fwd_lines = []
    for i in range(n_layers):
        init_lines.append(f"        self.fc{i} = nn.Linear({widths[i]}, {widths[i+1]})")
        if i == 0:
            fwd_lines.append(f"        x = self.fc{i}(x)")
        else:
            fwd_lines.append(f"        x = self.fc{i}(x)")
    fwd_lines.append("        return x")
    return (
        "import torch\nimport torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(init_lines) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd_lines) + "\n"
    )


def _make_linear_chain_with_relu(n_layers: int, widths: list[int]) -> str:
    """Generate an n-layer linear chain with ReLU activations."""
    assert len(widths) == n_layers + 1
    init_lines = []
    fwd_lines = []
    for i in range(n_layers):
        init_lines.append(f"        self.fc{i} = nn.Linear({widths[i]}, {widths[i+1]})")
        fwd_lines.append(f"        x = self.fc{i}(x)")
        if i < n_layers - 1:
            fwd_lines.append("        x = torch.relu(x)")
    fwd_lines.append("        return x")
    return (
        "import torch\nimport torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(init_lines) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd_lines) + "\n"
    )


def _make_mismatch_at_depth(total_layers: int, mismatch_at: int) -> str:
    """Generate a chain where layer `mismatch_at` has a shape mismatch."""
    widths = [64] * (total_layers + 1)
    # Introduce mismatch: layer mismatch_at outputs 64 but layer mismatch_at+1 expects 99
    if mismatch_at < total_layers - 1:
        widths[mismatch_at + 1] = 99
    init_lines = []
    fwd_lines = []
    for i in range(total_layers):
        in_w = widths[i] if i != mismatch_at + 1 else 99
        out_w = widths[i + 1] if i != mismatch_at else 64
        if i == mismatch_at:
            init_lines.append(f"        self.fc{i} = nn.Linear({widths[i]}, 64)")
        elif i == mismatch_at + 1:
            init_lines.append(f"        self.fc{i} = nn.Linear(99, {widths[i+1]})")
        else:
            init_lines.append(f"        self.fc{i} = nn.Linear({widths[i]}, {widths[i+1]})")
        fwd_lines.append(f"        x = self.fc{i}(x)")
    fwd_lines.append("        return x")
    return (
        "import torch\nimport torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        + "\n".join(init_lines) + "\n"
        "    def forward(self, x):\n"
        + "\n".join(fwd_lines) + "\n"
    )


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, dict] = {}

# --- Category 1: Parametric architectures ---

# ResNet-style basic block (simplified)
BENCHMARKS["resnet_basic_block"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return torch.relu(out)
""",
    "input_shapes": {"x": ("batch", 64, 32, 32)},
    "depth": 6,
    "category": "parametric",
    "expect_safe": True,
}

# ResNet downsample block (channel mismatch without projection)
BENCHMARKS["resnet_downsample_mismatch"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return torch.relu(out)
""",
    "input_shapes": {"x": ("batch", 64, 32, 32)},
    "depth": 6,
    "category": "parametric",
    "expect_safe": False,
}

# Transformer encoder layer (simplified)
BENCHMARKS["transformer_encoder_layer"] = {
    "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        x2 = self.self_attn(x, x, x)[0]
        x = self.norm1(x + x2)
        x2 = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        return x
""",
    "input_shapes": {"x": ("seq_len", "batch", 512)},
    "depth": 7,
    "category": "parametric",
    "expect_safe": True,
}

# Transformer with mismatched d_model
BENCHMARKS["transformer_dmodel_mismatch"] = {
    "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
    def forward(self, x):
        x2 = self.self_attn(x, x, x)[0]
        x = self.norm1(x + x2)
        x2 = self.linear2(torch.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        return x
""",
    "input_shapes": {"x": ("seq_len", "batch", 512)},
    "depth": 7,
    "category": "parametric",
    "expect_safe": False,
}

# --- Category 2: Deep chains (10, 20, 50 layers) ---

for n in [10, 20, 50]:
    w = [128] + [64] * n
    BENCHMARKS[f"deep_chain_{n}layers"] = {
        "source": _make_linear_chain(n, w),
        "input_shapes": {"x": ("batch", 128)},
        "depth": n,
        "category": "deep_chain",
        "expect_safe": True,
    }

# Deep chain with ReLU activations
for n in [10, 20]:
    w = [256] + [128] * n
    BENCHMARKS[f"deep_relu_chain_{n}layers"] = {
        "source": _make_linear_chain_with_relu(n, w),
        "input_shapes": {"x": ("batch", 256)},
        "depth": n * 2 - 1,  # layers + activations
        "category": "deep_chain",
        "expect_safe": True,
    }

# --- Category 3: Shape mismatches at various depths ---

for total, mismatch_pos in [(10, 1), (10, 5), (10, 9), (20, 10), (20, 19), (50, 25), (50, 49)]:
    BENCHMARKS[f"mismatch_at_{mismatch_pos}_of_{total}"] = {
        "source": _make_mismatch_at_depth(total, mismatch_pos),
        "input_shapes": {"x": ("batch", 64)},
        "depth": total,
        "category": "mismatch_depth",
        "expect_safe": False,
    }

# --- Category 4: Branching architectures ---

BENCHMARKS["skip_connection_2block"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
    def forward(self, x):
        r1 = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) + r1
        r2 = x
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
    "input_shapes": {"x": ("batch", 128)},
    "depth": 6,
    "category": "branching",
    "expect_safe": True,
}

BENCHMARKS["skip_connection_mismatch"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 128)
    def forward(self, x):
        r = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) + r
        return x
""",
    "input_shapes": {"x": ("batch", 128)},
    "depth": 3,
    "category": "branching",
    "expect_safe": True,
}

BENCHMARKS["dual_path_merge"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_a1 = nn.Linear(64, 32)
        self.path_a2 = nn.Linear(32, 16)
        self.path_b1 = nn.Linear(64, 48)
        self.path_b2 = nn.Linear(48, 16)
        self.merge = nn.Linear(32, 10)
    def forward(self, x):
        a = torch.relu(self.path_a1(x))
        a = self.path_a2(a)
        b = torch.relu(self.path_b1(x))
        b = self.path_b2(b)
        merged = torch.cat([a, b], dim=-1)
        return self.merge(merged)
""",
    "input_shapes": {"x": ("batch", 64)},
    "depth": 6,
    "category": "branching",
    "expect_safe": True,
}

BENCHMARKS["dual_path_shape_mismatch"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_a = nn.Linear(64, 16)
        self.path_b = nn.Linear(64, 20)
        self.merge = nn.Linear(32, 10)
    def forward(self, x):
        a = self.path_a(x)
        b = self.path_b(x)
        merged = torch.cat([a, b], dim=-1)
        return self.merge(merged)
""",
    "input_shapes": {"x": ("batch", 64)},
    "depth": 4,
    "category": "branching",
    "expect_safe": False,
}

BENCHMARKS["deep_residual_10block"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
"""
    + "".join(
        f"        self.fc{i} = nn.Linear(64, 64)\n" for i in range(10)
    )
    + """\
    def forward(self, x):
"""
    + "".join(
        f"        x = torch.relu(self.fc{i}(x)) + x\n" for i in range(10)
    )
    + "        return x\n",
    "input_shapes": {"x": ("batch", 64)},
    "depth": 20,
    "category": "branching",
    "expect_safe": True,
}

# --- Category 5: Scalability series (for depth scaling analysis) ---

SCALABILITY_DEPTHS = [1, 2, 5, 10, 15, 20, 30, 50]
for n in SCALABILITY_DEPTHS:
    widths = [64] * (n + 1)
    BENCHMARKS[f"scalability_{n}layers"] = {
        "source": _make_linear_chain(n, widths),
        "input_shapes": {"x": ("batch", 64)},
        "depth": n,
        "category": "scalability",
        "expect_safe": True,
    }

# --- Category 6: Mixed complexity ---

BENCHMARKS["conv_bn_relu_pool_chain"] = {
    "source": """\
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        return x
""",
    "input_shapes": {"x": ("batch", 3, 32, 32)},
    "depth": 9,
    "category": "mixed",
    "expect_safe": True,
}

BENCHMARKS["layernorm_chain"] = {
    "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 128)
    def forward(self, x):
        x = self.ln1(self.fc1(x))
        x = self.ln2(self.fc2(x))
        x = self.ln3(self.fc3(x))
        return self.fc4(x)
""",
    "input_shapes": {"x": ("batch", 256)},
    "depth": 7,
    "category": "mixed",
    "expect_safe": True,
}

BENCHMARKS["embedding_to_classifier"] = {
    "source": """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
    "input_shapes": {"x": ("batch", "seq_len")},
    "depth": 4,
    "category": "mixed",
    "expect_safe": True,
}


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def run_single_benchmark(name: str, spec: dict) -> dict:
    """Run IC3/PDR and bounded verification on a single benchmark."""
    entry = {
        "model": name,
        "depth": spec["depth"],
        "category": spec["category"],
        "expect_safe": spec["expect_safe"],
    }

    # --- Bounded model checking ---
    t0 = time.monotonic()
    bounded_result = verify_model(
        spec["source"],
        input_shapes=spec["input_shapes"],
    )
    bounded_ms = (time.monotonic() - t0) * 1000
    entry["bounded_safe"] = bounded_result.safe
    entry["bounded_time_ms"] = round(bounded_ms, 2)

    # --- IC3/PDR ---
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
    entry["verdicts_agree"] = bounded_result.safe == ic3_result.safe

    return entry


def run_evaluation() -> dict:
    """Run all benchmarks and produce categorized results."""
    results = []
    categories: dict[str, list] = {}

    print(f"Running {len(BENCHMARKS)} benchmarks...\n")

    for name, spec in BENCHMARKS.items():
        print(f"  [{name}] (depth={spec['depth']}, cat={spec['category']}) ... ", end="", flush=True)
        entry = run_single_benchmark(name, spec)
        results.append(entry)

        cat = spec["category"]
        categories.setdefault(cat, [])
        categories[cat].append(entry)

        status = "SAFE" if entry["ic3_safe"] else "UNSAFE"
        agree = "✓" if entry["verdicts_agree"] else "✗"
        print(
            f"{status}  bmc={entry['bounded_time_ms']:.1f}ms  "
            f"ic3={entry['ic3_time_ms']:.1f}ms  "
            f"frames={entry['ic3_frames']}  "
            f"blocked={entry['ic3_blocked_cubes']}  "
            f"agree={agree}"
        )

    return {"results": results, "categories": categories}


def compute_summary(data: dict) -> dict:
    """Compute aggregate statistics."""
    results = data["results"]
    categories = data["categories"]
    total = len(results)

    summary = {
        "total_benchmarks": total,
        "safe_count": sum(1 for r in results if r["ic3_safe"]),
        "unsafe_count": sum(1 for r in results if not r["ic3_safe"]),
        "verdicts_agreed": sum(1 for r in results if r["verdicts_agree"]),
        "avg_ic3_time_ms": round(sum(r["ic3_time_ms"] for r in results) / total, 2),
        "avg_bounded_time_ms": round(sum(r["bounded_time_ms"] for r in results) / total, 2),
        "max_ic3_time_ms": round(max(r["ic3_time_ms"] for r in results), 2),
        "max_bounded_time_ms": round(max(r["bounded_time_ms"] for r in results), 2),
        "max_depth_tested": max(r["depth"] for r in results),
        "max_frames": max(r["ic3_frames"] for r in results),
        "max_blocked_cubes": max(r["ic3_blocked_cubes"] for r in results),
    }

    # Per-category breakdown
    cat_summary = {}
    for cat, items in categories.items():
        cat_summary[cat] = {
            "count": len(items),
            "safe": sum(1 for r in items if r["ic3_safe"]),
            "unsafe": sum(1 for r in items if not r["ic3_safe"]),
            "agreed": sum(1 for r in items if r["verdicts_agree"]),
            "avg_ic3_ms": round(sum(r["ic3_time_ms"] for r in items) / len(items), 2),
            "avg_bmc_ms": round(sum(r["bounded_time_ms"] for r in items) / len(items), 2),
            "max_depth": max(r["depth"] for r in items),
        }
    summary["by_category"] = cat_summary

    # Scalability data (depth vs time)
    scalability = []
    for r in results:
        if r["category"] == "scalability":
            scalability.append({
                "depth": r["depth"],
                "ic3_time_ms": r["ic3_time_ms"],
                "bounded_time_ms": r["bounded_time_ms"],
                "ic3_frames": r["ic3_frames"],
                "ic3_blocked_cubes": r["ic3_blocked_cubes"],
                "ic3_z3_queries": r["ic3_z3_queries"],
            })
    scalability.sort(key=lambda x: x["depth"])
    summary["scalability_curve"] = scalability

    # Timing comparison: IC3 speedup over BMC
    speedups = []
    for r in results:
        if r["bounded_time_ms"] > 0:
            speedups.append(r["bounded_time_ms"] / max(r["ic3_time_ms"], 0.01))
    summary["ic3_avg_speedup_over_bmc"] = round(sum(speedups) / len(speedups), 2) if speedups else 0
    summary["ic3_median_speedup_over_bmc"] = round(sorted(speedups)[len(speedups) // 2], 2) if speedups else 0

    # Mismatch detection depth analysis
    mismatch_analysis = []
    for r in results:
        if r["category"] == "mismatch_depth":
            mismatch_analysis.append({
                "model": r["model"],
                "total_depth": r["depth"],
                "ic3_cex_depth": r["ic3_cex_depth"],
                "ic3_time_ms": r["ic3_time_ms"],
                "bounded_time_ms": r["bounded_time_ms"],
            })
    summary["mismatch_detection"] = mismatch_analysis

    return summary


def print_report(summary: dict) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE IC3/PDR EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal benchmarks: {summary['total_benchmarks']}")
    print(f"  Safe: {summary['safe_count']}  Unsafe: {summary['unsafe_count']}")
    print(f"  Verdicts agreed: {summary['verdicts_agreed']}/{summary['total_benchmarks']}")
    print(f"  Max depth tested: {summary['max_depth_tested']}")

    print(f"\nTiming:")
    print(f"  Avg IC3 time:     {summary['avg_ic3_time_ms']:.1f}ms")
    print(f"  Avg BMC time:     {summary['avg_bounded_time_ms']:.1f}ms")
    print(f"  Max IC3 time:     {summary['max_ic3_time_ms']:.1f}ms")
    print(f"  Max BMC time:     {summary['max_bounded_time_ms']:.1f}ms")
    print(f"  IC3 avg speedup:  {summary['ic3_avg_speedup_over_bmc']:.1f}x")
    print(f"  IC3 median speedup: {summary['ic3_median_speedup_over_bmc']:.1f}x")

    print(f"\nIC3 Statistics:")
    print(f"  Max frames:       {summary['max_frames']}")
    print(f"  Max blocked cubes: {summary['max_blocked_cubes']}")

    print(f"\nPer-Category Breakdown:")
    for cat, cs in summary["by_category"].items():
        print(f"  [{cat}] {cs['count']} benchmarks, "
              f"{cs['safe']} safe / {cs['unsafe']} unsafe, "
              f"agreed={cs['agreed']}/{cs['count']}, "
              f"avg IC3={cs['avg_ic3_ms']:.1f}ms, "
              f"avg BMC={cs['avg_bmc_ms']:.1f}ms, "
              f"max depth={cs['max_depth']}")

    print(f"\nScalability (depth → IC3 time):")
    for s in summary["scalability_curve"]:
        bar = "█" * max(1, int(s["ic3_time_ms"] / 2))
        print(f"  depth={s['depth']:3d}  ic3={s['ic3_time_ms']:8.1f}ms  "
              f"bmc={s['bounded_time_ms']:8.1f}ms  "
              f"frames={s['ic3_frames']}  "
              f"cubes={s['ic3_blocked_cubes']}  "
              f"|{bar}")

    print(f"\nMismatch Detection Analysis:")
    for m in summary["mismatch_detection"]:
        print(f"  {m['model']}: cex at depth {m['ic3_cex_depth']}, "
              f"ic3={m['ic3_time_ms']:.1f}ms, bmc={m['bounded_time_ms']:.1f}ms")


def main() -> None:
    print("=" * 70)
    print("Comprehensive IC3/PDR Evaluation")
    print(f"Benchmarks: {len(BENCHMARKS)}")
    print("=" * 70)
    print()

    data = run_evaluation()
    summary = compute_summary(data)

    print_report(summary)

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ic3_comprehensive_results.json")
    output = {
        "results": data["results"],
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
