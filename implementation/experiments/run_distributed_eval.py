#!/usr/bin/env python3
"""Evaluation script for distributed verification (FSDP, DeepSpeed, adapters).

Creates 10+ benchmark model configurations and runs FSDP sharding,
DeepSpeed ZeRO, and adapter composition verification on each.
Results are saved to ``.benchmarks/distributed_eval_results.json``.
"""

import json
import os
import sys
import time

# Ensure the implementation root is on sys.path
_impl_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _impl_root not in sys.path:
    sys.path.insert(0, _impl_root)

from src.distributed_verification import (
    AdapterComposition,
    AdapterCompositionVerifier,
    AdapterMergeStrategy,
    DeepSpeedConfig,
    DeepSpeedVerifier,
    FSDPConfig,
    FSDPShardingVerifier,
    WrapPolicy,
    ZeROStage,
    verify_distributed,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark model sources
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "SimpleMLP": '''
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
''',
    "DeepMLP": '''
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)
''',
    "TransformerEncoder": '''
import torch.nn as nn
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)
        self.ff1 = nn.Linear(768, 3072)
        self.ff2 = nn.Linear(3072, 768)
        self.ln1 = nn.LayerNorm(768)
        self.ln2 = nn.LayerNorm(768)
    def forward(self, x):
        q = self.q_proj(self.ln1(x))
        k = self.k_proj(x)
        v = self.v_proj(x)
        h = self.out_proj(q + v)
        return self.ff2(self.ff1(self.ln2(h)))
''',
    "LargeTransformer": '''
import torch.nn as nn
class LargeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(1024, 1024)
        self.k = nn.Linear(1024, 1024)
        self.v = nn.Linear(1024, 1024)
        self.o = nn.Linear(1024, 1024)
        self.up = nn.Linear(1024, 4096)
        self.down = nn.Linear(4096, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(1024)
    def forward(self, x):
        h = self.o(self.q(self.ln1(x)) + self.v(x))
        return self.down(self.up(self.ln2(h)))
''',
    "ConvClassifier": '''
import torch.nn as nn
class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc = nn.Linear(128, 100)
    def forward(self, x):
        h = self.conv2(self.conv1(x))
        return self.fc(h.view(h.size(0), -1))
''',
    "EmbeddingModel": '''
import torch.nn as nn
class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50000, 768)
        self.proj = nn.Linear(768, 768)
        self.head = nn.Linear(768, 50000)
    def forward(self, x):
        h = self.embed(x)
        return self.head(self.proj(h))
''',
    "ResidualBlock": '''
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.ln = nn.LayerNorm(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.fc2(self.relu(self.fc1(self.ln(x))))
        return h + x
''',
    "WideNet": '''
import torch.nn as nn
class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
''',
    "SmallNet": '''
import torch.nn as nn
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16)
    def forward(self, x):
        return self.fc(x)
''',
    "GPT2Like": '''
import torch.nn as nn
class GPT2Like(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.c_proj = nn.Linear(768, 768)
        self.mlp_fc = nn.Linear(768, 3072)
        self.mlp_proj = nn.Linear(3072, 768)
        self.ln_f = nn.LayerNorm(768)
    def forward(self, x):
        h = self.wte(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        h = self.c_proj(q + v)
        h = self.mlp_proj(self.mlp_fc(h))
        return self.ln_f(h)
''',
    "BERTLike": '''
import torch.nn as nn
class BERTLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30522, 768)
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)
        self.out = nn.Linear(768, 768)
        self.ff1 = nn.Linear(768, 3072)
        self.ff2 = nn.Linear(3072, 768)
        self.cls = nn.Linear(768, 2)
    def forward(self, x):
        h = self.embed(x)
        h = self.out(self.q(h) + self.v(h))
        h = self.ff2(self.ff1(h))
        return self.cls(h)
''',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation configurations
# ═══════════════════════════════════════════════════════════════════════════════

FSDP_CONFIGS = [
    ("FSDP_2gpu", FSDPConfig(world_size=2)),
    ("FSDP_4gpu", FSDPConfig(world_size=4)),
    ("FSDP_8gpu", FSDPConfig(world_size=8)),
    ("FSDP_8gpu_size_wrap", FSDPConfig(
        world_size=8,
        auto_wrap_policy=WrapPolicy.SIZE_BASED,
        min_num_params=100_000,
    )),
]

DS_CONFIGS = [
    ("ZeRO1_4gpu", DeepSpeedConfig(stage=ZeROStage.STAGE_1, dp_world_size=4)),
    ("ZeRO2_4gpu", DeepSpeedConfig(stage=ZeROStage.STAGE_2, dp_world_size=4)),
    ("ZeRO3_4gpu", DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=4)),
    ("ZeRO3_8gpu", DeepSpeedConfig(stage=ZeROStage.STAGE_3, dp_world_size=8)),
]

ADAPTER_CONFIGS = [
    ("add_2_adapters", AdapterComposition(
        adapters=[
            {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
            {"name": "A2", "in_features": 768, "out_features": 768, "rank": 16},
        ],
        strategy=AdapterMergeStrategy.ADD,
    ), True),
    ("stack_2_adapters", AdapterComposition(
        adapters=[
            {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
            {"name": "A2", "in_features": 256, "out_features": 128, "rank": 4},
        ],
        strategy=AdapterMergeStrategy.STACK,
    ), True),
    ("switch_3_adapters", AdapterComposition(
        adapters=[
            {"name": "A1", "in_features": 768, "out_features": 768, "rank": 4},
            {"name": "A2", "in_features": 768, "out_features": 768, "rank": 8},
            {"name": "A3", "in_features": 768, "out_features": 768, "rank": 16},
        ],
        strategy=AdapterMergeStrategy.SWITCH,
    ), True),
    ("incompatible_add", AdapterComposition(
        adapters=[
            {"name": "A1", "in_features": 768, "out_features": 768, "rank": 8},
            {"name": "A2", "in_features": 512, "out_features": 768, "rank": 8},
        ],
        strategy=AdapterMergeStrategy.ADD,
    ), False),
    ("incompatible_stack", AdapterComposition(
        adapters=[
            {"name": "A1", "in_features": 768, "out_features": 256, "rank": 8},
            {"name": "A2", "in_features": 512, "out_features": 128, "rank": 8},
        ],
        strategy=AdapterMergeStrategy.STACK,
    ), False),
]


def run_eval():
    results = []
    total_pass = 0
    total_fail = 0

    print("=" * 70)
    print("Distributed Verification Evaluation")
    print("=" * 70)

    # --- FSDP evaluation ---
    print("\n--- FSDP Sharding Verification ---")
    for model_name, source in MODELS.items():
        for fsdp_name, fsdp_cfg in FSDP_CONFIGS:
            test_name = f"{model_name}/{fsdp_name}"
            t0 = time.perf_counter()
            try:
                result = verify_distributed(
                    source=source,
                    input_shapes={"x": ("batch", 256)},
                    fsdp_config=fsdp_cfg,
                )
                fsdp_safe = result.fsdp_result.safe if result.fsdp_result else True
                elapsed = (time.perf_counter() - t0) * 1000
                status = "✓" if fsdp_safe else "✗"
                total_pass += 1 if fsdp_safe else 0
                total_fail += 0 if fsdp_safe else 1
                print(
                    f"  {status}  {test_name:45s}  safe={fsdp_safe!s:5s}  "
                    f"params={result.fsdp_result.params_checked if result.fsdp_result else 0}  "
                    f"time={elapsed:.1f}ms"
                )
                results.append({
                    "name": test_name,
                    "category": "fsdp",
                    "safe": fsdp_safe,
                    "time_ms": round(elapsed, 2),
                })
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                total_fail += 1
                print(f"  ✗  {test_name:45s}  ERROR: {e}")
                results.append({
                    "name": test_name,
                    "category": "fsdp",
                    "safe": False,
                    "error": str(e),
                    "time_ms": round(elapsed, 2),
                })

    # --- DeepSpeed evaluation ---
    print("\n--- DeepSpeed ZeRO Verification ---")
    for model_name, source in MODELS.items():
        for ds_name, ds_cfg in DS_CONFIGS:
            test_name = f"{model_name}/{ds_name}"
            t0 = time.perf_counter()
            try:
                result = verify_distributed(
                    source=source,
                    input_shapes={"x": ("batch", 256)},
                    deepspeed_config=ds_cfg,
                )
                ds_safe = result.deepspeed_result.safe if result.deepspeed_result else True
                elapsed = (time.perf_counter() - t0) * 1000
                status = "✓" if ds_safe else "✗"
                total_pass += 1 if ds_safe else 0
                total_fail += 0 if ds_safe else 1
                print(
                    f"  {status}  {test_name:45s}  safe={ds_safe!s:5s}  "
                    f"time={elapsed:.1f}ms"
                )
                results.append({
                    "name": test_name,
                    "category": "deepspeed",
                    "safe": ds_safe,
                    "time_ms": round(elapsed, 2),
                })
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                total_fail += 1
                print(f"  ✗  {test_name:45s}  ERROR: {e}")
                results.append({
                    "name": test_name,
                    "category": "deepspeed",
                    "safe": False,
                    "error": str(e),
                    "time_ms": round(elapsed, 2),
                })

    # --- Adapter composition evaluation ---
    print("\n--- Adapter Composition Verification ---")
    for ac_name, ac_comp, expected_safe in ADAPTER_CONFIGS:
        t0 = time.perf_counter()
        verifier = AdapterCompositionVerifier(ac_comp)
        ac_result = verifier.verify()
        elapsed = (time.perf_counter() - t0) * 1000
        correct = ac_result.safe == expected_safe
        status = "✓ PASS" if correct else "✗ FAIL"
        if correct:
            total_pass += 1
        else:
            total_fail += 1
        print(
            f"  {status}  {ac_name:45s}  safe={ac_result.safe!s:5s}  "
            f"expected={expected_safe!s:5s}  time={elapsed:.1f}ms"
        )
        results.append({
            "name": ac_name,
            "category": "adapter_composition",
            "safe": ac_result.safe,
            "expected_safe": expected_safe,
            "correct": correct,
            "time_ms": round(elapsed, 2),
        })

    total = total_pass + total_fail
    print("\n" + "=" * 70)
    print(f"Results: {total_pass}/{total} passed, {total_fail} failed")
    print("=" * 70)

    # Save results
    out_dir = os.path.join(_impl_root, ".benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "distributed_eval_results.json")

    report = {
        "total_tests": total,
        "passed": total_pass,
        "failed": total_fail,
        "accuracy": total_pass / total if total else 0,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return report


if __name__ == "__main__":
    run_eval()
