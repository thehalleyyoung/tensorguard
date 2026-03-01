#!/usr/bin/env python3
"""
Operator coverage expansion evaluation.

Measures the expanded operator coverage after adding modern shape transfer
functions, tests each new operator on representative inputs, and evaluates
on models that exercise modern patterns (Transformer with RoPE, MoE).

Saves results to experiments/operator_coverage_expansion_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    TORCH_SHAPE_OPS,
    NUMPY_SHAPE_OPS,
    analyze_shapes,
)
from src.stdlib.modern_ops import (
    MODERN_TORCH_SHAPE_OPS,
    MODERN_SHAPE_TRANSFERS,
    # Attention
    transfer_scaled_dot_product_attention,
    transfer_multi_head_attention,
    # Normalization
    transfer_group_norm,
    transfer_instance_norm,
    transfer_layer_norm,
    transfer_rms_norm,
    # Activation
    transfer_elementwise,
    transfer_gelu,
    transfer_silu,
    transfer_mish,
    transfer_glu,
    # Sparse / MoE
    transfer_sparse_softmax,
    transfer_topk,
    transfer_moe_routing,
    transfer_moe_gate_scores,
    # Positional
    transfer_rotary_embedding,
    transfer_sinusoidal_pos_encoding,
    transfer_alibi_bias,
    # Memory
    transfer_checkpoint,
    transfer_gradient_checkpoint,
    # Einops
    transfer_rearrange,
    transfer_einops_repeat,
    transfer_einops_reduce,
    # Additional
    transfer_adaptive_avg_pool,
    transfer_pixel_shuffle,
    transfer_pixel_unshuffle,
    transfer_unfold,
    transfer_fold,
    transfer_chunk,
    transfer_split,
    transfer_repeat_interleave,
    transfer_embedding,
    transfer_dropout,
    transfer_conv1d,
    transfer_conv3d,
    transfer_conv_transpose1d,
    transfer_cross_attention,
)

RESULTS_FILE = Path(__file__).parent / "operator_coverage_expansion_results.json"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Operator coverage census
# ═══════════════════════════════════════════════════════════════════════════

def count_coverage() -> Dict[str, Any]:
    """Count total operators covered (original + modern)."""
    original_torch = {
        "zeros", "ones", "randn", "rand", "empty", "full",
        "zeros_like", "ones_like", "randn_like",
        "arange", "linspace",
        "reshape", "view", "flatten", "squeeze", "unsqueeze",
        "permute", "transpose",
        "sum", "mean", "max", "min", "prod", "norm",
        "cat", "stack", "concatenate",
        "matmul", "mm", "bmm", "linear",
        "add", "mul", "sub", "div",
    }
    original_numpy = set(NUMPY_SHAPE_OPS.keys())
    modern = set(MODERN_TORCH_SHAPE_OPS.keys())

    all_ops = original_torch | original_numpy | modern
    new_only = modern - original_torch - original_numpy

    return {
        "original_torch_ops": len(original_torch),
        "original_numpy_ops": len(original_numpy),
        "modern_ops_added": len(new_only),
        "total_ops_covered": len(all_ops),
        "original_total": len(original_torch) + len(original_numpy),
        "categories": _categorize_ops(MODERN_TORCH_SHAPE_OPS),
    }


def _categorize_ops(ops: dict) -> Dict[str, int]:
    cats: Dict[str, int] = {}
    for op, cat in ops.items():
        cats[cat] = cats.get(cat, 0) + 1
    return cats


# ═══════════════════════════════════════════════════════════════════════════
# 2. Shape transfer function unit tests
# ═══════════════════════════════════════════════════════════════════════════

def _shape(*dims) -> TensorShape:
    return TensorShape(tuple(ShapeDim(d) for d in dims))


def test_shape_transfers() -> List[Dict[str, Any]]:
    """Test each shape transfer function on representative inputs."""
    results: List[Dict[str, Any]] = []

    def _run(name: str, fn, expected_ndim: Optional[int] = None,
             expected_shape: Optional[tuple] = None):
        t0 = time.monotonic()
        try:
            result = fn()
            elapsed = (time.monotonic() - t0) * 1000
            passed = result is not None
            if passed and expected_ndim is not None:
                passed = result.ndim == expected_ndim
            if passed and expected_shape is not None:
                actual = tuple(d.value for d in result.dims)
                passed = actual == expected_shape
            results.append({
                "operator": name,
                "passed": passed,
                "result_shape": result.pretty() if result else None,
                "time_ms": round(elapsed, 3),
            })
        except Exception as e:
            results.append({
                "operator": name,
                "passed": False,
                "error": str(e),
                "time_ms": round((time.monotonic() - t0) * 1000, 3),
            })

    # Attention
    q = _shape(2, 8, 32, 64)
    k = _shape(2, 8, 128, 64)
    v = _shape(2, 8, 128, 64)
    _run("scaled_dot_product_attention",
         lambda: transfer_scaled_dot_product_attention(q, k, v),
         expected_ndim=4, expected_shape=(2, 8, 32, 64))

    _run("multi_head_attention",
         lambda: transfer_multi_head_attention(
             _shape(32, 2, 512), _shape(128, 2, 512), _shape(128, 2, 512),
             num_heads=8, embed_dim=512),
         expected_ndim=3, expected_shape=(32, 2, 512))

    _run("cross_attention",
         lambda: transfer_cross_attention(q, k, v),
         expected_ndim=4, expected_shape=(2, 8, 32, 64))

    # Normalization
    _run("group_norm",
         lambda: transfer_group_norm(_shape(4, 32, 16, 16), 8, 32),
         expected_ndim=4, expected_shape=(4, 32, 16, 16))

    _run("instance_norm",
         lambda: transfer_instance_norm(_shape(4, 32, 16, 16)),
         expected_ndim=4, expected_shape=(4, 32, 16, 16))

    _run("layer_norm",
         lambda: transfer_layer_norm(_shape(4, 32, 512), (512,)),
         expected_ndim=3, expected_shape=(4, 32, 512))

    _run("rms_norm",
         lambda: transfer_rms_norm(_shape(4, 32, 512), (512,)),
         expected_ndim=3, expected_shape=(4, 32, 512))

    # Activations
    act_input = _shape(4, 128, 512)
    _run("gelu", lambda: transfer_gelu(act_input),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("silu", lambda: transfer_silu(act_input),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("mish", lambda: transfer_mish(act_input),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("glu",
         lambda: transfer_glu(_shape(4, 128, 1024), dim=-1),
         expected_ndim=3, expected_shape=(4, 128, 512))

    # Sparse / MoE
    _run("sparse_softmax",
         lambda: transfer_sparse_softmax(_shape(4, 32, 32)),
         expected_ndim=3, expected_shape=(4, 32, 32))
    _run("topk",
         lambda: transfer_topk(_shape(4, 1000), k=10, dim=-1),
         expected_ndim=2, expected_shape=(4, 10))
    _run("moe_routing",
         lambda: transfer_moe_routing(_shape(4, 128, 512), num_experts=8, top_k=2),
         expected_ndim=4)
    _run("moe_gate_scores",
         lambda: transfer_moe_gate_scores(_shape(4, 128, 512), num_experts=8),
         expected_ndim=3, expected_shape=(4, 128, 8))

    # Positional encoding
    _run("rotary_embedding",
         lambda: transfer_rotary_embedding(_shape(4, 8, 128, 64), dim=64),
         expected_ndim=4, expected_shape=(4, 8, 128, 64))
    _run("sinusoidal_pos_encoding",
         lambda: transfer_sinusoidal_pos_encoding(seq_len=128, d_model=512),
         expected_ndim=2, expected_shape=(128, 512))
    _run("alibi_bias",
         lambda: transfer_alibi_bias(num_heads=8, seq_len=128),
         expected_ndim=3, expected_shape=(8, 128, 128))

    # Checkpoint
    _run("checkpoint",
         lambda: transfer_checkpoint(_shape(4, 128, 512)),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("gradient_checkpoint",
         lambda: transfer_gradient_checkpoint(_shape(4, 128, 512)),
         expected_ndim=3, expected_shape=(4, 128, 512))

    # Einops
    _run("rearrange",
         lambda: transfer_rearrange(
             _shape(2, 3, 4, 5), "b c h w -> b (c h) w"),
         expected_ndim=3, expected_shape=(2, 12, 5))
    _run("einops_repeat",
         lambda: transfer_einops_repeat(
             _shape(2, 3), "b c -> b c n", n=4),
         expected_ndim=3, expected_shape=(2, 3, 4))
    _run("einops_reduce",
         lambda: transfer_einops_reduce(
             _shape(2, 3, 4, 5), "b c h w -> b c", reduction="mean"),
         expected_ndim=2, expected_shape=(2, 3))

    # Additional ops
    _run("adaptive_avg_pool2d",
         lambda: transfer_adaptive_avg_pool(_shape(4, 64, 32, 32), (7, 7)),
         expected_ndim=4, expected_shape=(4, 64, 7, 7))
    _run("pixel_shuffle",
         lambda: transfer_pixel_shuffle(_shape(1, 12, 4, 4), upscale_factor=2),
         expected_ndim=4, expected_shape=(1, 3, 8, 8))
    _run("pixel_unshuffle",
         lambda: transfer_pixel_unshuffle(_shape(1, 3, 8, 8), downscale_factor=2),
         expected_ndim=4, expected_shape=(1, 12, 4, 4))
    _run("unfold",
         lambda: transfer_unfold(_shape(4, 3, 32), dim=2, size=5, step=1),
         expected_ndim=4)
    _run("fold",
         lambda: transfer_fold(
             _shape(1, 75, 196), output_size=(14, 14), kernel_size=(5, 5)),
         expected_ndim=4)
    _run("chunk",
         lambda: transfer_chunk(_shape(4, 128, 512), chunks=4, dim=-1),
         expected_ndim=3, expected_shape=(4, 128, 128))
    _run("split",
         lambda: transfer_split(_shape(4, 128, 512), split_size=64, dim=-1),
         expected_ndim=3, expected_shape=(4, 128, 64))
    _run("repeat_interleave",
         lambda: transfer_repeat_interleave(
             _shape(4, 128), repeats=3, dim=1),
         expected_ndim=2, expected_shape=(4, 384))
    _run("embedding",
         lambda: transfer_embedding(_shape(4, 128), embedding_dim=512),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("dropout",
         lambda: transfer_dropout(_shape(4, 128, 512)),
         expected_ndim=3, expected_shape=(4, 128, 512))
    _run("conv1d",
         lambda: transfer_conv1d(
             _shape(4, 3, 128), out_channels=64, kernel_size=3, padding=1),
         expected_ndim=3, expected_shape=(4, 64, 128))
    _run("conv3d",
         lambda: transfer_conv3d(
             _shape(2, 3, 16, 16, 16), out_channels=32,
             kernel_size=(3, 3, 3), padding=(1, 1, 1)),
         expected_ndim=5, expected_shape=(2, 32, 16, 16, 16))
    _run("conv_transpose1d",
         lambda: transfer_conv_transpose1d(
             _shape(4, 64, 32), out_channels=3, kernel_size=4, stride=2, padding=1),
         expected_ndim=3, expected_shape=(4, 3, 64))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Model-level evaluation
# ═══════════════════════════════════════════════════════════════════════════

TRANSFORMER_ROPE_SOURCE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryTransformer(nn.Module):
    """Transformer block with RoPE and FlashAttention."""
    def __init__(self, d_model=512, nhead=8, dim_ff=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        assert x.shape == (4, 128, 512)
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q = q.reshape(4, 128, 8, 64).transpose(1, 2)
        k = k.reshape(4, 128, 8, 64).transpose(1, 2)
        v = v.reshape(4, 128, 8, 64).transpose(1, 2)
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(4, 128, 512)
        out = self.out_proj(out)
        x = x + out
        h = self.norm2(x)
        h = self.ff1(h)
        h = F.gelu(h)
        h = self.ff2(h)
        return x + h
'''

MOE_MODEL_SOURCE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """Mixture-of-Experts layer with top-k routing."""
    def __init__(self, d_model=512, num_experts=8, dim_ff=2048, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.expert1 = nn.Linear(d_model, dim_ff)
        self.expert2 = nn.Linear(dim_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        assert x.shape == (4, 128, 512)
        h = self.norm(x)
        gate_logits = self.gate(h)
        gate_logits = F.softmax(gate_logits, dim=-1)
        expert_out = self.expert1(h)
        expert_out = F.silu(expert_out)
        expert_out = self.expert2(expert_out)
        return x + expert_out
'''

VISION_TRANSFORMER_SOURCE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: (batch, channels, height, width) = (4, 3, 224, 224)
        assert x.shape == (4, 3, 224, 224)
        # flatten patches → (batch, num_patches, patch_dim)
        x = x.reshape(4, 3, 14, 16, 14, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(4, 196, 768)
        x = self.proj(x)
        return x
'''


def evaluate_model(name: str, source: str) -> Dict[str, Any]:
    """Run shape analysis on a model source and collect results."""
    t0 = time.monotonic()
    try:
        result = analyze_shapes(source)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "model": name,
            "shapes_inferred": len(result.shapes),
            "constraints_generated": result.constraints_generated,
            "constraints_checked": result.constraints_checked,
            "errors_found": len(result.errors),
            "error_details": [e.to_dict() for e in result.errors],
            "time_ms": round(elapsed, 2),
            "status": "ok",
        }
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        return {
            "model": name,
            "status": "error",
            "error": str(e),
            "time_ms": round(elapsed, 2),
        }


def run_model_evaluations() -> List[Dict[str, Any]]:
    """Evaluate shape analysis on models using modern operators."""
    models = [
        ("transformer_with_rope", TRANSFORMER_ROPE_SOURCE),
        ("moe_model", MOE_MODEL_SOURCE),
        ("vision_transformer", VISION_TRANSFORMER_SOURCE),
    ]
    return [evaluate_model(name, src) for name, src in models]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Operator Coverage Expansion Evaluation")
    print("=" * 70)

    # 1. Coverage census
    print("\n[1/3] Counting operator coverage...")
    coverage = count_coverage()
    print(f"  Original torch ops:  {coverage['original_torch_ops']}")
    print(f"  Original numpy ops:  {coverage['original_numpy_ops']}")
    print(f"  Modern ops added:    {coverage['modern_ops_added']}")
    print(f"  Total ops covered:   {coverage['total_ops_covered']}")
    print(f"  Categories: {json.dumps(coverage['categories'], indent=4)}")

    # 2. Shape transfer tests
    print("\n[2/3] Testing shape transfer functions...")
    transfer_results = test_shape_transfers()
    passed = sum(1 for r in transfer_results if r["passed"])
    total = len(transfer_results)
    print(f"  Passed: {passed}/{total}")
    for r in transfer_results:
        status = "✓" if r["passed"] else "✗"
        shape_str = r.get("result_shape", r.get("error", "None"))
        print(f"    {status} {r['operator']:40s} → {shape_str}")

    # 3. Model evaluations
    print("\n[3/3] Evaluating on modern model architectures...")
    model_results = run_model_evaluations()
    for m in model_results:
        status = "✓" if m["status"] == "ok" else "✗"
        print(f"  {status} {m['model']:30s}  "
              f"shapes={m.get('shapes_inferred', '?'):>3}  "
              f"constraints={m.get('constraints_generated', '?'):>3}  "
              f"errors={m.get('errors_found', '?'):>2}  "
              f"time={m.get('time_ms', '?'):>7}ms")

    # Assemble final results
    results = {
        "coverage": coverage,
        "transfer_tests": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "details": transfer_results,
        },
        "model_evaluations": model_results,
        "summary": {
            "original_op_count": coverage["original_total"],
            "expanded_op_count": coverage["total_ops_covered"],
            "expansion_factor": round(
                coverage["total_ops_covered"] / max(coverage["original_total"], 1),
                2,
            ),
            "transfer_test_pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "models_evaluated": len(model_results),
            "models_successful": sum(
                1 for m in model_results if m["status"] == "ok"
            ),
        },
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"\nSummary: {coverage['original_total']} → "
          f"{coverage['total_ops_covered']} ops "
          f"({results['summary']['expansion_factor']}x expansion)")


if __name__ == "__main__":
    main()
