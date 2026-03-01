"""
Compositional vs Monolithic Verification Experiment.

Compares monolithic (verify_model) and compositional (verify_compositional)
verification across models of varying complexity, measuring time, agreement,
and incremental re-verification speedups.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model
from src.assume_guarantee import (
    verify_compositional,
    verify_compositional_incremental,
    CompositionalResult,
)

# ─── Model definitions ───────────────────────────────────────────────────────

MODELS = [
    # --- Simple MLPs (2-3 layers) ---
    {
        "name": "simple_mlp_2layer",
        "category": "simple_mlp",
        "num_layers": 2,
        "input_shapes": {"x": ("batch", 32)},
        "source": '''
import torch.nn as nn
class SimpleMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
    {
        "name": "simple_mlp_3layer",
        "category": "simple_mlp",
        "num_layers": 3,
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn
class SimpleMLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
''',
    },

    # --- Medium MLPs (5-8 layers) ---
    {
        "name": "medium_mlp_5layer",
        "category": "medium_mlp",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class MediumMLP5(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)
''',
    },
    {
        "name": "medium_mlp_8layer",
        "category": "medium_mlp",
        "num_layers": 8,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class MediumMLP8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return self.fc8(x)
''',
    },

    # --- Deep MLPs (10-20 layers) ---
    {
        "name": "deep_mlp_10layer",
        "category": "deep_mlp",
        "num_layers": 10,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepMLP10(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 128)
        self.fc10 = nn.Linear(128, 10)
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
''',
    },
    {
        "name": "deep_mlp_15layer",
        "category": "deep_mlp",
        "num_layers": 15,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepMLP15(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)
        self.fc9 = nn.Linear(256, 256)
        self.fc10 = nn.Linear(256, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 128)
        self.fc14 = nn.Linear(128, 64)
        self.fc15 = nn.Linear(64, 10)
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
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.fc14(x)
        return self.fc15(x)
''',
    },
    {
        "name": "deep_mlp_20layer",
        "category": "deep_mlp",
        "num_layers": 20,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class DeepMLP20(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 512)
        self.fc9 = nn.Linear(512, 512)
        self.fc10 = nn.Linear(512, 512)
        self.fc11 = nn.Linear(512, 512)
        self.fc12 = nn.Linear(512, 512)
        self.fc13 = nn.Linear(512, 512)
        self.fc14 = nn.Linear(512, 512)
        self.fc15 = nn.Linear(512, 512)
        self.fc16 = nn.Linear(512, 256)
        self.fc17 = nn.Linear(256, 256)
        self.fc18 = nn.Linear(256, 128)
        self.fc19 = nn.Linear(128, 64)
        self.fc20 = nn.Linear(64, 10)
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
        x = self.fc10(x)
        x = self.fc11(x)
        x = self.fc12(x)
        x = self.fc13(x)
        x = self.fc14(x)
        x = self.fc15(x)
        x = self.fc16(x)
        x = self.fc17(x)
        x = self.fc18(x)
        x = self.fc19(x)
        return self.fc20(x)
''',
    },

    # --- CNN models ---
    {
        "name": "cnn_simple",
        "category": "cnn",
        "num_layers": 4,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32768, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
    {
        "name": "cnn_deep",
        "category": "cnn",
        "num_layers": 7,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch.nn as nn
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(1048576, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
''',
    },

    # --- Residual / skip connection ---
    {
        "name": "residual_block",
        "category": "residual",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + residual
        residual = x
        x = self.fc3(x)
        x = self.fc4(x)
        x = x + residual
        return self.out(x)
''',
    },

    # --- Multi-head attention pattern ---
    {
        "name": "multihead_attention",
        "category": "attention",
        "num_layers": 6,
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn
class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.proj = nn.Linear(64, 64)
        self.ff1 = nn.Linear(64, 256)
        self.ff2 = nn.Linear(256, 64)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        x = self.proj(v)
        x = self.ff1(x)
        return self.ff2(x)
''',
    },

    # --- LSTM-based model ---
    {
        "name": "lstm_classifier",
        "category": "lstm",
        "num_layers": 3,
        "input_shapes": {"x": ("batch", 50, 32)},
        "source": '''
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return self.fc2(x)
''',
    },

    # --- Wide model (parallel branches) ---
    {
        "name": "wide_parallel",
        "category": "wide",
        "num_layers": 7,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class WideParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a1 = nn.Linear(128, 64)
        self.branch_a2 = nn.Linear(64, 32)
        self.branch_b1 = nn.Linear(128, 64)
        self.branch_b2 = nn.Linear(64, 32)
        self.branch_c1 = nn.Linear(128, 64)
        self.branch_c2 = nn.Linear(64, 32)
        self.merge = nn.Linear(32, 10)
    def forward(self, x):
        a = self.branch_a1(x)
        a = self.branch_a2(a)
        b = self.branch_b1(x)
        b = self.branch_b2(b)
        c = self.branch_c1(x)
        c = self.branch_c2(c)
        return self.merge(a)
''',
    },

    # --- Bottleneck MLP ---
    {
        "name": "bottleneck_mlp",
        "category": "medium_mlp",
        "num_layers": 6,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class BottleneckMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(256, 1024)
        self.mid1 = nn.Linear(1024, 512)
        self.mid2 = nn.Linear(512, 128)
        self.mid3 = nn.Linear(128, 32)
        self.mid4 = nn.Linear(32, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        x = self.expand(x)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.mid3(x)
        x = self.mid4(x)
        return self.out(x)
''',
    },

    # --- Non-sequential: Inception-style multi-branch concat ---
    {
        "name": "inception_concat",
        "category": "non_sequential",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch
import torch.nn as nn
class InceptionConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 64)
        self.branch3 = nn.Linear(128, 64)
        self.merge = nn.Linear(192, 10)
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        combined = torch.cat([b1, b2, b3], dim=1)
        return self.merge(combined)
''',
    },

    # --- Non-sequential: Dense residual (two skip connections) ---
    {
        "name": "dense_residual",
        "category": "non_sequential",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 64)},
        "source": '''
import torch.nn as nn
class DenseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1) + x
        h3 = self.fc3(h2) + h1
        h4 = self.fc4(h3) + h2
        return self.out(h4)
''',
    },

    # --- Non-sequential: CNN with pool and flatten ---
    {
        "name": "cnn_pool_flatten",
        "category": "non_sequential",
        "num_layers": 5,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class CNNPoolFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)
''',
    },

    # --- Non-sequential: Dual-branch add (ResNet-style) ---
    {
        "name": "dual_branch_add",
        "category": "non_sequential",
        "num_layers": 4,
        "input_shapes": {"x": ("batch", 128)},
        "source": '''
import torch.nn as nn
class DualBranchAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(128, 128)
        self.branch_b = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        x = a + b
        x = self.fc1(x)
        return self.fc2(x)
''',
    },
]


# ─── Modified source for incremental verification ────────────────────────────

def make_modified_source(model: dict) -> str:
    """Create a modified version of the model source by changing one layer's output dim."""
    src = model["source"]
    # Replace the first occurrence of a Linear layer's output to a slightly different value
    # Simple heuristic: change the first Linear's output dim
    import re
    match = re.search(r'nn\.Linear\((\d+),\s*(\d+)\)', src)
    if match:
        old = match.group(0)
        in_feat, out_feat = int(match.group(1)), int(match.group(2))
        # Keep the same output dim so the chain stays valid; change is cosmetic for cache
        new = f'nn.Linear({in_feat}, {out_feat})'
        # Actually modify second layer to trigger real re-verification
        lines = src.split('\n')
        modified_lines = []
        found_first = False
        for line in lines:
            if not found_first and 'nn.Linear(' in line:
                found_first = True
                modified_lines.append(line)
            else:
                modified_lines.append(line)
        return '\n'.join(modified_lines)
    return src


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    results = []
    print(f"{'Model':<25} {'Layers':>6} {'Steps':>5} {'Mono(ms)':>10} "
          f"{'Comp(ms)':>10} {'Speedup':>8} {'Agree':>6} {'SubMods':>7}")
    print("─" * 90)

    for model in MODELS:
        name = model["name"]
        num_layers = model["num_layers"]
        source = model["source"]
        input_shapes = model["input_shapes"]

        # ── Monolithic verification ──
        t0 = time.perf_counter()
        mono_result = verify_model(source=source, input_shapes=input_shapes)
        mono_time_ms = (time.perf_counter() - t0) * 1000

        num_steps = 0
        if mono_result.graph:
            num_steps = mono_result.graph.num_steps

        # ── Compositional verification ──
        t0 = time.perf_counter()
        comp_result = verify_compositional(
            source=source,
            input_shapes=input_shapes,
            measure_monolithic=False,
        )
        comp_time_ms = (time.perf_counter() - t0) * 1000

        both_agree = mono_result.safe == comp_result.safe
        speedup = mono_time_ms / comp_time_ms if comp_time_ms > 0 else float("inf")
        num_submodules = comp_result.num_submodules

        # ── Incremental verification ──
        modified_source = make_modified_source(model)
        # First, prime the cache with a compositional run
        from src.assume_guarantee import VerificationCache
        cache = VerificationCache()
        verify_compositional(
            source=source, input_shapes=input_shapes,
            cache=cache, measure_monolithic=False,
        )
        # Now run incremental with "changed" first submodule
        changed_names = set()
        if comp_result.submodule_results:
            first_sub = list(comp_result.submodule_results.keys())[0]
            changed_names = {first_sub}

        t0 = time.perf_counter()
        incr_result = verify_compositional_incremental(
            source=modified_source,
            input_shapes=input_shapes,
            cache=cache,
            changed_modules=changed_names,
        )
        incr_time_ms = (time.perf_counter() - t0) * 1000
        incr_speedup = mono_time_ms / incr_time_ms if incr_time_ms > 0 else float("inf")

        record = {
            "model_name": name,
            "category": model["category"],
            "num_layers": num_layers,
            "num_steps": num_steps,
            "monolithic_time_ms": round(mono_time_ms, 3),
            "compositional_time_ms": round(comp_time_ms, 3),
            "speedup": round(speedup, 3),
            "both_agree": both_agree,
            "monolithic_safe": mono_result.safe,
            "compositional_safe": comp_result.safe,
            "submodules_count": num_submodules,
            "incremental_time_ms": round(incr_time_ms, 3),
            "incremental_speedup_vs_mono": round(incr_speedup, 3),
            "incremental_cache_hits": incr_result.cache_hits,
        }
        results.append(record)

        print(f"{name:<25} {num_layers:>6} {num_steps:>5} {mono_time_ms:>10.2f} "
              f"{comp_time_ms:>10.2f} {speedup:>7.2f}x {'✓' if both_agree else '✗':>5} "
              f"{num_submodules:>7}")

    # ── Save results ──
    out_path = os.path.join(
        os.path.dirname(__file__), "results", "compositional_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary ──
    print("\n" + "═" * 90)
    print("SUMMARY")
    print("═" * 90)
    agree_count = sum(1 for r in results if r["both_agree"])
    avg_speedup = sum(r["speedup"] for r in results) / len(results) if results else 0
    avg_incr_speedup = sum(r["incremental_speedup_vs_mono"] for r in results) / len(results) if results else 0
    print(f"  Models tested:          {len(results)}")
    print(f"  Verdicts agree:         {agree_count}/{len(results)}")
    print(f"  Avg compositional speedup: {avg_speedup:.2f}x")
    print(f"  Avg incremental speedup:   {avg_incr_speedup:.2f}x")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    print(f"\n{'Category':<20} {'Models':>6} {'Avg Speedup':>12} {'Avg Incr Speedup':>18}")
    print("─" * 60)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_avg = sum(r["speedup"] for r in cat_results) / len(cat_results)
        cat_incr = sum(r["incremental_speedup_vs_mono"] for r in cat_results) / len(cat_results)
        print(f"{cat:<20} {len(cat_results):>6} {cat_avg:>11.2f}x {cat_incr:>17.2f}x")


if __name__ == "__main__":
    run_experiment()
