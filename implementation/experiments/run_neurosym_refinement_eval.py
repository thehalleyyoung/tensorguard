#!/usr/bin/env python3
"""
Neuro-Symbolic CEGAR Refinement Evaluation.

Evaluates counterexample-guided LLM refinement on intentionally buggy models.
Compares:
  1. LLM alone (just the buggy code, no verification context)
  2. LLM + TensorGuard explanations (formal verification guides repair)

Measures: fix rate, iterations needed, explanation quality.

Usage (from implementation/):
    source ~/.bashrc
    python3 experiments/run_neurosym_refinement_eval.py

If OPENAI_API_KEY is not available, generates mock results (clearly labeled).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.neurosym_refinement import (
    NeurosymRefinementLoop,
    RefinementResult,
    run_neurosym_refinement,
    _build_repair_prompt,
)
from src.cegar_explanation import explain_verification, VerificationExplanation

# ═══════════════════════════════════════════════════════════════════════════════
# Buggy benchmark models
# ═══════════════════════════════════════════════════════════════════════════════

BUGGY_MODELS = {
    # ── Category 1: Deep chain bugs (bug in layer 5+ of 8+ layer network) ─────
    "deep_encoder_layer7_bug": {
        "source": """\
import torch
import torch.nn as nn

class DeepEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 384)
        self.bn2 = nn.BatchNorm1d(384)
        self.fc3 = nn.Linear(384, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        # Bug: fc6 expects 96 but fc5 outputs 64. The correct fix is fc6=Linear(64,32),
        # NOT changing fc5 to output 96 (that would break the progressive halving pattern).
        self.fc6 = nn.Linear(96, 32)
        self.fc7 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 768)},
        "bug_type": "deep_chain",
        "description": "fc6 expects 96 inputs but fc5 outputs 64; bug is at layer 8 of 10 in progressive halving encoder",
    },
    "deep_bottleneck_layer6_bug": {
        "source": """\
import torch
import torch.nn as nn

class DeepBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(1024, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)
        self.enc4 = nn.Linear(128, 64)
        # Bottleneck
        self.bottleneck = nn.Linear(64, 32)
        # Decoder - Bug: dec1 expects 64 but bottleneck outputs 32
        # Correct fix: dec1=Linear(32,64), NOT changing bottleneck output
        # because the bottleneck is intentionally the narrowest point.
        self.dec1 = nn.Linear(64, 128)
        self.dec2 = nn.Linear(128, 256)
        self.dec3 = nn.Linear(256, 512)
        self.out = nn.Linear(512, 1024)

    def forward(self, x):
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.enc3(x))
        x = torch.relu(self.enc4(x))
        x = torch.relu(self.bottleneck(x))
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        x = torch.relu(self.dec3(x))
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 1024)},
        "bug_type": "deep_chain",
        "description": "dec1 expects 64 but bottleneck outputs 32; bug at layer 6 in autoencoder with symmetric structure",
    },

    # ── Category 2: Reshape/view dimension calculation errors ─────────────────
    "reshape_product_mismatch": {
        "source": """\
import torch
import torch.nn as nn

class ReshapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 360)
        # Bug: reshape to (batch, 8, 7, 7) requires 8*7*7=392 but fc2 outputs 360
        # 360 = 8*45 or 6*60 or 5*72 but NOT 8*7*7
        # Correct fix: fc2=Linear(512, 392) to match reshape target
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.out = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 8, 7, 7)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "bug_type": "reshape_mismatch",
        "description": "fc2 outputs 360 but view(-1,8,7,7) needs 392; non-obvious product-of-dims error",
    },
    "permute_reshape_interaction": {
        "source": """\
import torch
import torch.nn as nn

class PermuteReshapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(512, 256)
        # Process as (batch, 8_heads, 32_seq, 32_dim) after reshape
        # Bug: 256 = 8*32 (two factors), but we reshape to (b,8,32,16)
        # which needs 8*32*16=4096, not 256.
        # The intended architecture: embed outputs 256, reshape to (b,8,32)
        # then an attention-like operation. The correct fix is either
        # embed=Linear(512,4096) or reshape to (b,8,32) not (b,8,32,16).
        self.proj = nn.Linear(16, 64)
        self.out = nn.Linear(8 * 32 * 64, 10)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), 8, 32, 16)
        x = self.proj(x)
        x = x.reshape(x.size(0), -1)
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 512)},
        "bug_type": "reshape_mismatch",
        "description": "embed outputs 256 but view(b,8,32,16) needs 4096; interplay of reshape dims and downstream linear",
    },

    # ── Category 3: Multi-head attention bugs ─────────────────────────────────
    "mha_head_dim_mismatch": {
        "source": """\
import torch
import torch.nn as nn

class MHANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 256
        self.num_heads = 6
        # Bug: embed_dim=256 is not divisible by num_heads=6
        # (256/6 = 42.67, not integer). head_dim must be integer.
        # Correct fix: num_heads=8 (256/8=32) or embed_dim=192 (192/6=32)
        # but changing embed_dim cascades to all projections.
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 256)

    def forward(self, x):
        B, S, _ = x.shape
        head_dim = self.embed_dim // self.num_heads
        q = self.q_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)
""",
        "input_shapes": {"x": ("batch", "seq_len", 256)},
        "bug_type": "attention_mismatch",
        "description": "embed_dim=256 not divisible by num_heads=6; non-obvious divisibility constraint in attention reshape",
    },
    "mha_qk_dim_mismatch": {
        "source": """\
import torch
import torch.nn as nn

class CrossAttentionBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 8
        self.head_dim = 64
        # Bug: Q projects to 8*64=512 but K projects to 8*48=384
        # Q @ K^T requires last dims to match for matmul
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 384)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, q_input, kv_input):
        B, S, _ = q_input.shape
        _, T, _ = kv_input.shape
        q = self.q_proj(q_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T, self.num_heads, 48).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)
""",
        "input_shapes": {"q_input": ("batch", "seq_len", 512), "kv_input": ("batch", "kv_len", 512)},
        "bug_type": "attention_mismatch",
        "description": "Q head_dim=64 but K head_dim=48; matmul Q@K^T fails because inner dims don't match",
    },

    # ── Category 4: Skip connection bugs ──────────────────────────────────────
    "resblock_channel_skip_mismatch": {
        "source": """\
import torch
import torch.nn as nn

class ResBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Bug: skip connection adds input (64 channels) to output (128 channels)
        # Missing a 1x1 projection conv for the shortcut
        # Correct fix: add self.shortcut = nn.Conv2d(64, 128, 1) and use it

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Bug: 128 channels + 64 channels
        return torch.relu(out)
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
        "bug_type": "skip_connection",
        "description": "Residual addition: main path outputs 128 channels but shortcut passes through 64 channels unchanged",
    },
    "dense_skip_dim_mismatch": {
        "source": """\
import torch
import torch.nn as nn

class DenseSkipBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        # Bug: skip connects fc1 output (256) to fc4 output (128) via addition
        # Two possible fixes: (a) fc3,fc4 output 256 (b) add a projection layer
        # Only (b) is architecturally correct (preserves compression)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        h4 = torch.relu(self.fc4(h3))
        # Skip connection from h1 (256-dim) to h4 (128-dim)
        out = h4 + h1  # Bug: 128 + 256 shape mismatch
        return self.out(out)
""",
        "input_shapes": {"x": ("batch", 512)},
        "bug_type": "skip_connection",
        "description": "Skip from fc1 output (256) to fc4 output (128); requires projection but multiple plausible fixes exist",
    },

    # ── Category 5: Conv→linear transition bugs (spatial dimension calc) ──────
    "conv_stride_flatten_bug": {
        "source": """\
import torch
import torch.nn as nn

class ConvStrideFlattenBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        # Input 32x32 -> conv1(s2,p1) -> 16x16 -> conv2(s2,p1) -> 8x8
        # -> conv3(s2,p0) -> floor((8-3)/2)+1 = 3x3
        # Bug: fc expects 128*4*4=2048 but actual is 128*3*3=1152
        self.fc = nn.Linear(2048, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.out(torch.relu(self.fc(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "bug_type": "conv_linear_transition",
        "description": "fc expects 2048 (128*4*4) but conv3 stride/padding yields 3x3 spatial, so actual is 1152 (128*3*3)",
    },
    "pool_conv_flatten_bug": {
        "source": """\
import torch
import torch.nn as nn

class PoolConvFlattenBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Input 64x64 -> conv1(k5,p2) -> 64x64 -> pool -> 32x32
        # -> conv2(k3,p1) -> 32x32 -> pool -> 16x16
        # -> conv3(k3,p0) -> 14x14 -> pool -> 7x7
        # Bug: classifier expects 256*8*8=16384 but actual is 256*7*7=12544
        self.classifier = nn.Linear(16384, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "bug_type": "conv_linear_transition",
        "description": "classifier expects 16384 (256*8*8) but conv3(p=0)+pool yields 7x7 spatial, actual is 12544 (256*7*7)",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    model_name: str
    bug_type: str
    description: str
    llm_only_fixed: bool
    llm_only_iterations: int
    guided_fixed: bool
    guided_iterations: int
    llm_model: str
    is_mock: bool


def _get_api_key() -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        result = subprocess.run(
            ["bash", "-c", "source ~/.bashrc 2>/dev/null && echo $OPENAI_API_KEY"],
            capture_output=True, text=True, timeout=5,
        )
        key = result.stdout.strip()
        if key:
            return key
    except Exception:
        pass
    return None


def _generate_mock_results() -> List[BenchmarkResult]:
    """Generate mock results when no API key is available.

    NOTE: These are MOCK results for demonstrating the evaluation harness.
    Guided fixes ~80% (8/10), LLM-only fixes ~40% (4/10).
    Rationale: simple chain bugs are fixable without guidance, but reshape,
    attention, skip-connection, and conv→linear transition bugs require
    the TensorGuard explanation to identify the correct fix location.
    """
    # Deterministic mock outcomes per category:
    #   deep_chain: LLM-only fixes 1/2, guided fixes 2/2
    #   reshape: LLM-only fixes 0/2, guided fixes 1/2
    #   attention: LLM-only fixes 1/2, guided fixes 2/2
    #   skip_connection: LLM-only fixes 1/2, guided fixes 1/2
    #   conv_linear: LLM-only fixes 1/2, guided fixes 2/2
    mock_outcomes = {
        "deep_encoder_layer7_bug":      (True,  3, True,  1),
        "deep_bottleneck_layer6_bug":   (False, 5, True,  2),
        "reshape_product_mismatch":     (False, 5, True,  2),
        "permute_reshape_interaction":  (False, 5, False, 5),
        "mha_head_dim_mismatch":        (False, 5, True,  1),
        "mha_qk_dim_mismatch":          (True,  4, True,  2),
        "resblock_channel_skip_mismatch": (False, 5, True,  2),
        "dense_skip_dim_mismatch":      (True,  4, False, 5),
        "conv_stride_flatten_bug":      (False, 5, True,  1),
        "pool_conv_flatten_bug":        (True,  5, True,  2),
    }
    results = []
    for name, spec in BUGGY_MODELS.items():
        llm_fixed, llm_iters, guided_fixed, guided_iters = mock_outcomes[name]
        results.append(BenchmarkResult(
            model_name=name,
            bug_type=spec["bug_type"],
            description=spec["description"],
            llm_only_fixed=llm_fixed,
            llm_only_iterations=llm_iters,
            guided_fixed=guided_fixed,
            guided_iterations=guided_iters,
            llm_model="gpt-4.1-nano",
            is_mock=True,
        ))
    return results


def _run_real_eval(api_key: str, llm_model: str = "gpt-4.1-nano") -> List[BenchmarkResult]:
    """Run real evaluation with OpenAI API.

    LLM-only baseline: gets ONLY source code + generic "Shape mismatch detected"
    Guided mode: gets full TensorGuard explanation with exact layer, dims, counterexample
    """
    results = []
    for name, spec in BUGGY_MODELS.items():
        print(f"  Evaluating {name}...")

        # 1. LLM alone: only source code + generic error, NO specific location
        def llm_alone_call(source, explanation):
            """Call LLM with only the source and a generic error message."""
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": (
                            "You are a PyTorch model repair assistant. "
                            "Fix the shape bug in this model. "
                            "Output ONLY the corrected Python class (with imports)."
                        )},
                        {"role": "user", "content": (
                            f"```python\n{source}\n```\n\n"
                            "Error: Shape mismatch detected. "
                            "One or more tensor operations receive incompatible shapes. "
                            "Fix the model so all shapes are compatible."
                        )},
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                )
                content = resp.choices[0].message.content or ""
                import re
                match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
                return match.group(1).strip() if match else content.strip()
            except Exception as e:
                print(f"    LLM-alone call failed: {e}")
                return None

        llm_only_result = run_neurosym_refinement(
            spec["source"],
            max_iterations=5,
            llm_model=llm_model,
            llm_call=llm_alone_call,
            input_shapes=spec["input_shapes"],
        )

        # 2. Guided: LLM + full TensorGuard explanation (default behavior)
        guided_result = run_neurosym_refinement(
            spec["source"],
            max_iterations=5,
            llm_model=llm_model,
            input_shapes=spec["input_shapes"],
        )

        results.append(BenchmarkResult(
            model_name=name,
            bug_type=spec["bug_type"],
            description=spec["description"],
            llm_only_fixed=llm_only_result.final_verdict == "SAFE",
            llm_only_iterations=llm_only_result.iterations,
            guided_fixed=guided_result.final_verdict == "SAFE",
            guided_iterations=guided_result.iterations,
            llm_model=llm_model,
            is_mock=False,
        ))
    return results


def main():
    print("=" * 70)
    print("Neuro-Symbolic CEGAR Refinement Evaluation")
    print("=" * 70)

    api_key = _get_api_key()
    if api_key:
        print("OpenAI API key found. Running real evaluation...")
        results = _run_real_eval(api_key)
    else:
        print("No OPENAI_API_KEY found. Generating MOCK results.")
        print("(Set OPENAI_API_KEY to run real LLM evaluation)")
        results = _generate_mock_results()

    # Compute summary statistics
    n = len(results)
    llm_only_fixes = sum(1 for r in results if r.llm_only_fixed)
    guided_fixes = sum(1 for r in results if r.guided_fixed)
    llm_only_iters = [r.llm_only_iterations for r in results if r.llm_only_fixed]
    guided_iters = [r.guided_iterations for r in results if r.guided_fixed]

    summary = {
        "is_mock_data": any(r.is_mock for r in results),
        "num_models": n,
        "llm_only_fix_rate": llm_only_fixes / n if n else 0,
        "guided_fix_rate": guided_fixes / n if n else 0,
        "llm_only_avg_iterations": sum(llm_only_iters) / len(llm_only_iters) if llm_only_iters else 0,
        "guided_avg_iterations": sum(guided_iters) / len(guided_iters) if guided_iters else 0,
        "llm_model": results[0].llm_model if results else "unknown",
        "results": [asdict(r) for r in results],
    }

    # Print summary
    print()
    print(f"Models tested: {n}")
    is_mock = summary["is_mock_data"]
    label = " (MOCK)" if is_mock else ""
    print(f"LLM-only fix rate{label}: {summary['llm_only_fix_rate']:.1%} ({llm_only_fixes}/{n})")
    print(f"Guided fix rate{label}:   {summary['guided_fix_rate']:.1%} ({guided_fixes}/{n})")
    if llm_only_iters:
        print(f"LLM-only avg iterations (when fixed): {summary['llm_only_avg_iterations']:.1f}")
    if guided_iters:
        print(f"Guided avg iterations (when fixed):   {summary['guided_avg_iterations']:.1f}")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.bug_type
        if cat not in categories:
            categories[cat] = {"llm_fixed": 0, "guided_fixed": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.llm_only_fixed:
            categories[cat]["llm_fixed"] += 1
        if r.guided_fixed:
            categories[cat]["guided_fixed"] += 1
    print(f"\nPer-category breakdown:")
    for cat, stats in categories.items():
        print(f"  {cat:30s}  LLM-only: {stats['llm_fixed']}/{stats['total']}  Guided: {stats['guided_fixed']}/{stats['total']}")

    # Save results
    out_dir = PROJECT_ROOT / ".benchmarks"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "neurosym_refinement_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
