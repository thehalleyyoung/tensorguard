#!/usr/bin/env python3
"""Evaluation script for LoRA verification support.

Creates 10+ test models with LoRA adapters (simple transformers, MLPs)
and runs verification on each, collecting results into a JSON report.
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

import torch
import torch.nn as nn

from src.lora_verification import (
    LoRAAdapter,
    LoRAConfig,
    LoRAShapeContract,
    LoRAVerifier,
    QuantizationVerifier,
    verify_lora_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test models
# ═══════════════════════════════════════════════════════════════════════════════


class SmallMLP(nn.Module):
    """2-layer MLP with LoRA on both layers."""

    def __init__(self, d_in=128, d_hidden=256, d_out=10, rank=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc1.lora_A = nn.Parameter(torch.randn(rank, d_in))
        self.fc1.lora_B = nn.Parameter(torch.randn(d_hidden, rank))
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.fc2.lora_A = nn.Parameter(torch.randn(rank, d_hidden))
        self.fc2.lora_B = nn.Parameter(torch.randn(d_out, rank))
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x) + x @ self.fc1.lora_A.T @ self.fc1.lora_B.T
        h = self.relu(h)
        return self.fc2(h) + h @ self.fc2.lora_A.T @ self.fc2.lora_B.T


class DeepMLP(nn.Module):
    """4-layer MLP with LoRA on all layers."""

    def __init__(self, d=256, rank=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(4):
            layer = nn.Linear(d, d)
            layer.lora_A = nn.Parameter(torch.randn(rank, d))
            layer.lora_B = nn.Parameter(torch.randn(d, rank))
            self.layers.append(layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x) + x @ layer.lora_A.T @ layer.lora_B.T)
        return x


class TransformerBlock(nn.Module):
    """Single transformer attention block with LoRA on Q/V."""

    def __init__(self, d_model=512, rank=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.q_proj.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.q_proj.lora_B = nn.Parameter(torch.randn(d_model, rank))
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.v_proj.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.v_proj.lora_B = nn.Parameter(torch.randn(d_model, rank))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x) + x @ self.q_proj.lora_A.T @ self.q_proj.lora_B.T
        k = self.k_proj(x)
        v = self.v_proj(x) + x @ self.v_proj.lora_A.T @ self.v_proj.lora_B.T
        return self.out_proj(q + k + v)


class LoRAEmbedding(nn.Module):
    """Embedding + linear with LoRA on the linear."""

    def __init__(self, vocab=1000, d_model=256, rank=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj.lora_A = nn.Parameter(torch.randn(rank, d_model))
        self.proj.lora_B = nn.Parameter(torch.randn(d_model, rank))

    def forward(self, x):
        e = self.embed(x)
        return self.proj(e) + e @ self.proj.lora_A.T @ self.proj.lora_B.T


class LoRAClassifier(nn.Module):
    """Classifier with LoRA on the head."""

    def __init__(self, d_in=768, n_classes=100, rank=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, n_classes)
        self.head.lora_A = nn.Parameter(torch.randn(rank, 256))
        self.head.lora_B = nn.Parameter(torch.randn(n_classes, rank))

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat) + feat @ self.head.lora_A.T @ self.head.lora_B.T


class HighRankLoRA(nn.Module):
    """LoRA with rank close to full — valid but large."""

    def __init__(self, d=64, rank=60):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.fc.lora_A = nn.Parameter(torch.randn(rank, d))
        self.fc.lora_B = nn.Parameter(torch.randn(d, rank))

    def forward(self, x):
        return self.fc(x) + x @ self.fc.lora_A.T @ self.fc.lora_B.T


class BadRankModel(nn.Module):
    """LoRA with rank > min(d, k) — should fail."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
        self.fc.lora_A = nn.Parameter(torch.randn(100, 32))
        self.fc.lora_B = nn.Parameter(torch.randn(64, 100))

    def forward(self, x):
        return self.fc(x)


class MismatchedInnerDim(nn.Module):
    """LoRA with B columns != A rows — merge unsafe."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 128)
        self.fc.lora_A = nn.Parameter(torch.randn(8, 128))
        self.fc.lora_B = nn.Parameter(torch.randn(128, 16))

    def forward(self, x):
        return self.fc(x)


class NoLoRABaseline(nn.Module):
    """Plain model — no LoRA for baseline comparison."""

    def __init__(self, d=256):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class AsymmetricLoRA(nn.Module):
    """LoRA on an asymmetric linear layer (d_in != d_out)."""

    def __init__(self, d_in=768, d_out=3072, rank=16):
        super().__init__()
        self.up_proj = nn.Linear(d_in, d_out)
        self.up_proj.lora_A = nn.Parameter(torch.randn(rank, d_in))
        self.up_proj.lora_B = nn.Parameter(torch.randn(d_out, rank))

    def forward(self, x):
        return self.up_proj(x) + x @ self.up_proj.lora_A.T @ self.up_proj.lora_B.T


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_eval():
    models = [
        ("SmallMLP_rank4", SmallMLP(128, 256, 10, rank=4), True),
        ("DeepMLP_rank8", DeepMLP(256, rank=8), True),
        ("TransformerBlock_rank8", TransformerBlock(512, rank=8), True),
        ("LoRAEmbedding_rank4", LoRAEmbedding(1000, 256, rank=4), True),
        ("LoRAClassifier_rank8", LoRAClassifier(768, 100, rank=8), True),
        ("HighRank_60of64", HighRankLoRA(64, rank=60), True),
        ("BadRank_100of32", BadRankModel(), False),
        ("MismatchedInner", MismatchedInnerDim(), False),
        ("NoLoRABaseline", NoLoRABaseline(256), True),
        ("AsymmetricLoRA_rank16", AsymmetricLoRA(768, 3072, rank=16), True),
        ("SmallMLP_rank1", SmallMLP(128, 256, 10, rank=1), True),
        ("TransformerBlock_rank16", TransformerBlock(512, rank=16), True),
    ]

    results = []
    total_pass = 0
    total_fail = 0

    print("=" * 70)
    print("LoRA Verification Evaluation")
    print("=" * 70)

    for name, model, expected_safe in models:
        t0 = time.perf_counter()
        result = verify_lora_model(model)
        elapsed = (time.perf_counter() - t0) * 1000

        correct = result.safe == expected_safe
        status = "✓ PASS" if correct else "✗ FAIL"
        if correct:
            total_pass += 1
        else:
            total_fail += 1

        print(
            f"  {status}  {name:30s}  safe={result.safe!s:5s}  "
            f"expected={expected_safe!s:5s}  "
            f"adapters={len(result.adapters)}  "
            f"time={elapsed:.1f}ms"
        )

        results.append({
            "name": name,
            "safe": result.safe,
            "expected_safe": expected_safe,
            "correct": correct,
            "has_lora": result.has_lora,
            "num_adapters": len(result.adapters),
            "num_rank_violations": len(result.rank_violations),
            "merge_safe": result.merge_safe,
            "quantization": result.quantization.name,
            "time_ms": round(elapsed, 2),
        })

    print("=" * 70)
    print(f"Results: {total_pass}/{len(models)} passed, {total_fail} failed")
    print("=" * 70)

    # Z3 contract verification
    print("\nZ3 Contract Verification:")
    z3_tests = [
        ("valid_8_768", LoRAAdapter("fc", 768, 768, rank=8), True),
        ("valid_16_512", LoRAAdapter("fc", 512, 512, rank=16), True),
        ("invalid_100_32", LoRAAdapter("fc", 32, 64, rank=100), False),
        # Symbolic dims: Z3 can find d<8 as counterexample, so not provably safe
        ("symbolic_unbounded", LoRAAdapter("fc", "d", "d", rank=8), False),
    ]
    for name, adapter, expected in z3_tests:
        contract = LoRAShapeContract(adapter=adapter)
        safe, cex = contract.verify_z3()
        status = "✓" if (safe == expected) else "✗"
        print(f"  {status}  {name}: safe={safe}, expected={expected}")

    # Save results
    out_dir = os.path.join(_impl_root, ".benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lora_verification_results.json")

    report = {
        "total_models": len(models),
        "passed": total_pass,
        "failed": total_fail,
        "accuracy": total_pass / len(models) if models else 0,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return report


if __name__ == "__main__":
    run_eval()
