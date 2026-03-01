#!/usr/bin/env python3
"""
Craig Interpolation Suite — Cross-Domain Evaluation (CD-Eval).

Addresses the primary reviewer concern: Craig interpolation was only
evaluated on 5 benchmarks.  This experiment evaluates whether
interpolation closes the 4.3% CEGAR–BMC accuracy gap on 24 architectures
where CEGAR template-based discovery typically fails but BMC succeeds.

Benchmark categories (architectures that stress template limits):
  1. Complex flatten/reshape after conv layers (product-equality constraints)
  2. Skip connections / residual blocks (cross-branch constraints)
  3. Multi-head attention patterns (embed_dim = heads × head_dim)
  4. Deep composition chains (conv→pool→flatten→linear with arithmetic deps)
  5. Models requiring predicates beyond the 7-kind template grammar

For each benchmark, three verification modes are compared:
  (a) CEGAR without interpolation (template-only)
  (b) CEGAR with interpolation enabled
  (c) BMC baseline (monolithic)

Outputs: experiments/craig_suite_cd_eval_results.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    CEGARVerdict,
    run_shape_cegar,
)
from src.bmc_baseline import BMCVerdict, verify_model_bmc

RESULTS_FILE = Path(__file__).parent / "craig_suite_cd_eval_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark definitions — 24 architectures that stress template limits
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ── Category 1: Complex flatten/reshape after conv (product-equality) ──
    {
        "name": "conv_flatten_linear_correct",
        "category": "flatten_reshape",
        "has_bug": False,
        "description": "Conv2d→Flatten→Linear: flatten dim = out_ch * H' * W'",
        "code": """\
import torch.nn as nn
class ConvFlatLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "conv_flatten_linear_buggy",
        "category": "flatten_reshape",
        "has_bug": True,
        "description": "Conv→Flatten→Linear: fc in_features wrong (expects 32*8*8 but gets 32*H*W)",
        "code": """\
import torch.nn as nn
class BuggyConvFlatLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "conv_pool_flatten_correct",
        "category": "flatten_reshape",
        "has_bug": False,
        "description": "Conv→MaxPool→Flatten→Linear: pooling halves spatial dims",
        "code": """\
import torch.nn as nn
class ConvPoolFlat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 4 * 4, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "conv_pool_flatten_buggy",
        "category": "flatten_reshape",
        "has_bug": True,
        "description": "Conv→Pool→Flatten: fc expects 64*8*8 but pool halves dims",
        "code": """\
import torch.nn as nn
class BuggyConvPoolFlat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "reshape_view_chain_correct",
        "category": "flatten_reshape",
        "has_bug": False,
        "description": "Reshape chain: view→permute-like Linear chain, correct",
        "code": """\
import torch.nn as nn
class ReshapeChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 256)
        self.head = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.proj(x))
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", "c", "h", "w")},
    },

    # ── Category 2: Skip connections / residual blocks ─────────────────────
    {
        "name": "resblock_conv_correct",
        "category": "skip_connection",
        "has_bug": False,
        "description": "Conv residual block: conv→conv+skip, channels match",
        "code": """\
import torch.nn as nn
class ConvResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    {
        "name": "resblock_conv_buggy",
        "category": "skip_connection",
        "has_bug": True,
        "description": "Conv residual: conv2 out=128 but skip is 64; add fails",
        "code": """\
import torch.nn as nn
class BuggyConvResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    {
        "name": "double_skip_correct",
        "category": "skip_connection",
        "has_bug": False,
        "description": "Two stacked residual blocks with skip, correct",
        "code": """\
import torch.nn as nn
class DoubleSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        r1 = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x) + r1
        r2 = x
        x = self.relu(self.fc3(x))
        return self.fc4(x) + r2
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "double_skip_buggy",
        "category": "skip_connection",
        "has_bug": True,
        "description": "Double skip: fc4 out=128 but skip r2 is 256; add fails",
        "code": """\
import torch.nn as nn
class BuggyDoubleSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        r1 = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x) + r1
        r2 = x
        x = self.relu(self.fc3(x))
        return self.fc4(x) + r2
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "downsample_skip_correct",
        "category": "skip_connection",
        "has_bug": False,
        "description": "Skip with downsample projection, correct",
        "code": """\
import torch.nn as nn
class DownsampleSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.downsample = nn.Linear(512, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = self.downsample(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.relu(x + residual)
""",
        "input_shapes": {"x": ("batch", "d")},
    },

    # ── Category 3: Multi-head attention patterns ──────────────────────────
    {
        "name": "multihead_qkv_correct",
        "category": "multihead_attention",
        "has_bug": False,
        "description": "Multi-head Q/K/V proj: embed_dim → num_heads × head_dim",
        "code": """\
import torch.nn as nn
class MultiHeadQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "multihead_qkv_buggy",
        "category": "multihead_attention",
        "has_bug": True,
        "description": "Multi-head: v_proj out=256 but out_proj in=512",
        "code": """\
import torch.nn as nn
class BuggyMultiHeadQKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 256)
        self.out_proj = nn.Linear(512, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "multihead_split_merge_correct",
        "category": "multihead_attention",
        "has_bug": False,
        "description": "Split into heads → project each → merge, correct",
        "code": """\
import torch.nn as nn
class SplitMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(256, 64)
        self.head2 = nn.Linear(256, 64)
        self.head3 = nn.Linear(256, 64)
        self.head4 = nn.Linear(256, 64)
        self.merge = nn.Linear(64, 128)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        h3 = self.head3(x)
        h4 = self.head4(x)
        return self.merge(h1 + h2 + h3 + h4)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "multihead_split_merge_buggy",
        "category": "multihead_attention",
        "has_bug": True,
        "description": "4 heads: head3 out=128 but others are 64; add fails",
        "code": """\
import torch.nn as nn
class BuggySplitMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(256, 64)
        self.head2 = nn.Linear(256, 64)
        self.head3 = nn.Linear(256, 128)
        self.head4 = nn.Linear(256, 64)
        self.merge = nn.Linear(64, 128)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        h3 = self.head3(x)
        h4 = self.head4(x)
        return self.merge(h1 + h2 + h3 + h4)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "mha_ffn_block_correct",
        "category": "multihead_attention",
        "has_bug": False,
        "description": "Attention+FFN transformer block, correct",
        "code": """\
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_proj = nn.Linear(768, 768)
        self.ffn1 = nn.Linear(768, 3072)
        self.ffn2 = nn.Linear(3072, 768)
        self.relu = nn.ReLU()
    def forward(self, x):
        attn = self.attn_proj(x)
        x = x + attn
        ff = self.relu(self.ffn1(x))
        return x + self.ffn2(ff)
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },

    # ── Category 4: Deep composition chains ────────────────────────────────
    {
        "name": "deep_conv_pool_fc_correct",
        "category": "deep_composition",
        "has_bug": False,
        "description": "Conv→Conv→Pool→Flatten→Linear→Linear, correct",
        "code": """\
import torch.nn as nn
class DeepConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "deep_conv_pool_fc_buggy",
        "category": "deep_composition",
        "has_bug": True,
        "description": "Deep chain: fc1 in_features=64*8*8 but conv2→pool gives 64*4*4",
        "code": """\
import torch.nn as nn
class BuggyDeepConvFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "five_layer_chain_correct",
        "category": "deep_composition",
        "has_bug": False,
        "description": "5-layer linear chain with consistent dims",
        "code": """\
import torch.nn as nn
class FiveLayerChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", "d_in")},
    },
    {
        "name": "five_layer_chain_buggy",
        "category": "deep_composition",
        "has_bug": True,
        "description": "5-layer chain: fc3 in=512 but fc2 out=256",
        "code": """\
import torch.nn as nn
class BuggyFiveLayerChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)
""",
        "input_shapes": {"x": ("batch", "d_in")},
    },
    {
        "name": "encoder_decoder_deep_correct",
        "category": "deep_composition",
        "has_bug": False,
        "description": "Deep encoder-decoder: 3 encode + 3 decode layers",
        "code": """\
import torch.nn as nn
class DeepEncDec(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(1024, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        return self.dec3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },

    # ── Category 5: Beyond-template predicates ─────────────────────────────
    {
        "name": "cross_branch_add_correct",
        "category": "beyond_template",
        "has_bug": False,
        "description": "Two branches merged with addition, correct",
        "code": """\
import torch.nn as nn
class CrossBranchAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(512, 256)
        self.branch_b = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.relu(self.branch_a(x))
        b = self.relu(self.branch_b(x))
        return self.out(a + b)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "cross_branch_add_buggy",
        "category": "beyond_template",
        "has_bug": True,
        "description": "Two branches: branch_a out=256, branch_b out=128; add fails",
        "code": """\
import torch.nn as nn
class BuggyCrossBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(512, 256)
        self.branch_b = nn.Linear(512, 128)
        self.out = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.relu(self.branch_a(x))
        b = self.relu(self.branch_b(x))
        return self.out(a + b)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "project_gate_correct",
        "category": "beyond_template",
        "has_bug": False,
        "description": "Gated projection: proj * gate (element-wise), correct",
        "code": """\
import torch.nn as nn
class GatedProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 256)
        self.gate = nn.Linear(512, 256)
        self.out = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        p = self.proj(x)
        g = self.sigmoid(self.gate(x))
        return self.out(p * g)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "project_gate_buggy",
        "category": "beyond_template",
        "has_bug": True,
        "description": "Gated proj: proj out=256 but gate out=128; mul fails",
        "code": """\
import torch.nn as nn
class BuggyGatedProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 256)
        self.gate = nn.Linear(512, 128)
        self.out = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        p = self.proj(x)
        g = self.sigmoid(self.gate(x))
        return self.out(p * g)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
]

assert len(TEST_CASES) >= 20, f"Need ≥20 benchmarks, have {len(TEST_CASES)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_bmc(v: BMCVerdict) -> str:
    return {BMCVerdict.SAFE: "SAFE", BMCVerdict.UNSAFE: "UNSAFE"}.get(v, "UNKNOWN")


def _normalize_cegar(v: CEGARVerdict) -> str:
    return {
        CEGARVerdict.SAFE: "SAFE",
        CEGARVerdict.UNSAFE: "UNSAFE",
        CEGARVerdict.TIMEOUT: "UNKNOWN",
        CEGARVerdict.UNKNOWN: "UNKNOWN",
    }.get(v, "UNKNOWN")


def _is_correct(verdict_str: str, has_bug: bool) -> bool:
    expected = "UNSAFE" if has_bug else "SAFE"
    return verdict_str == expected


def compute_f1(results: List[Dict[str, Any]], verdict_key: str) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in results:
        has_bug = r["has_bug"]
        detected = r[verdict_key] == "UNSAFE"
        if has_bug and detected:
            tp += 1
        elif not has_bug and detected:
            fp += 1
        elif has_bug and not detected:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


# ═══════════════════════════════════════════════════════════════════════════════
# Mode runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_cegar_no_interp(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Mode (a): CEGAR with interpolation DISABLED (template-only)."""
    t0 = time.monotonic()
    try:
        loop = ShapeCEGARLoop(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_interpolation=False,
        )
        result = loop.run()
        verdict = _normalize_cegar(result.verdict)
        n_preds = len(result.discovered_predicates)
        n_iters = result.iterations
        status = result.final_status.name
        interp_stats = result.interpolation_stats or {}
    except Exception as e:
        verdict = "UNKNOWN"
        n_preds = 0
        n_iters = 0
        status = f"ERROR: {e}"
        interp_stats = {}
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "verdict": verdict,
        "predicates": n_preds,
        "iterations": n_iters,
        "status": status,
        "interp_preds": interp_stats.get("predicates_from_interpolation", 0),
        "time_ms": round(elapsed, 2),
    }


def run_cegar_with_interp(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Mode (b): CEGAR with interpolation ENABLED."""
    t0 = time.monotonic()
    try:
        loop = ShapeCEGARLoop(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_interpolation=True,
        )
        result = loop.run()
        verdict = _normalize_cegar(result.verdict)
        n_preds = len(result.discovered_predicates)
        n_iters = result.iterations
        status = result.final_status.name
        interp_stats = result.interpolation_stats or {}
    except Exception as e:
        verdict = "UNKNOWN"
        n_preds = 0
        n_iters = 0
        status = f"ERROR: {e}"
        interp_stats = {}
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "verdict": verdict,
        "predicates": n_preds,
        "iterations": n_iters,
        "status": status,
        "interp_preds": interp_stats.get("predicates_from_interpolation", 0),
        "time_ms": round(elapsed, 2),
    }


def run_bmc_baseline(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Mode (c): BMC monolithic verification."""
    t0 = time.monotonic()
    try:
        result = verify_model_bmc(
            tc["code"],
            input_shapes=tc["input_shapes"],
            timeout=30,
        )
        verdict = _normalize_bmc(result.verdict)
        n_steps = result.num_steps
        n_constraints = result.num_constraints
    except Exception as e:
        verdict = "UNKNOWN"
        n_steps = 0
        n_constraints = 0
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "verdict": verdict,
        "num_steps": n_steps,
        "num_constraints": n_constraints,
        "time_ms": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    n = len(TEST_CASES)
    n_buggy = sum(1 for tc in TEST_CASES if tc["has_bug"])
    n_safe = n - n_buggy
    categories = sorted(set(tc["category"] for tc in TEST_CASES))

    print("=" * 80)
    print("  Craig Interpolation Suite — Cross-Domain Evaluation (CD-Eval)")
    print(f"  {n} benchmarks ({n_buggy} buggy, {n_safe} correct) × 3 modes")
    print(f"  Categories: {', '.join(categories)}")
    print("=" * 80)

    per_benchmark: List[Dict[str, Any]] = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n  [{i:2d}/{n}] {tc['name']:<40s} ({tc['category']})")

        r_no_interp = run_cegar_no_interp(tc)
        r_with_interp = run_cegar_with_interp(tc)
        r_bmc = run_bmc_baseline(tc)

        expected = "UNSAFE" if tc["has_bug"] else "SAFE"
        no_interp_correct = r_no_interp["verdict"] == expected
        with_interp_correct = r_with_interp["verdict"] == expected
        bmc_correct = r_bmc["verdict"] == expected

        # Determine if interpolation closes the gap for this benchmark
        gap_case = (not no_interp_correct) and bmc_correct
        interp_closes = gap_case and with_interp_correct

        entry = {
            "name": tc["name"],
            "category": tc["category"],
            "has_bug": tc["has_bug"],
            "expected": expected,
            "cegar_no_interp": r_no_interp,
            "cegar_with_interp": r_with_interp,
            "bmc_baseline": r_bmc,
            "cegar_no_interp_correct": no_interp_correct,
            "cegar_with_interp_correct": with_interp_correct,
            "bmc_correct": bmc_correct,
            "is_gap_case": gap_case,
            "interpolation_closes_gap": interp_closes,
        }
        per_benchmark.append(entry)

        mark_ni = "✓" if no_interp_correct else "✗"
        mark_wi = "✓" if with_interp_correct else "✗"
        mark_bmc = "✓" if bmc_correct else "✗"
        gap_str = " ← GAP CLOSED" if interp_closes else (" ← GAP CASE" if gap_case else "")
        print(
            f"    CEGAR-noInterp: {mark_ni} {r_no_interp['verdict']:<7s} "
            f"({r_no_interp['predicates']} preds, {r_no_interp['time_ms']:.0f}ms)"
        )
        print(
            f"    CEGAR+Interp:   {mark_wi} {r_with_interp['verdict']:<7s} "
            f"({r_with_interp['predicates']} preds, "
            f"interp_preds={r_with_interp['interp_preds']}, "
            f"{r_with_interp['time_ms']:.0f}ms)"
        )
        print(
            f"    BMC baseline:   {mark_bmc} {r_bmc['verdict']:<7s} "
            f"({r_bmc['time_ms']:.0f}ms){gap_str}"
        )

    # ── Aggregate results ──────────────────────────────────────────────────
    # Build per-mode verdict lists for F1
    for entry in per_benchmark:
        entry["cegar_no_interp_verdict"] = entry["cegar_no_interp"]["verdict"]
        entry["cegar_with_interp_verdict"] = entry["cegar_with_interp"]["verdict"]
        entry["bmc_verdict"] = entry["bmc_baseline"]["verdict"]

    metrics_no_interp = compute_f1(per_benchmark, "cegar_no_interp_verdict")
    metrics_with_interp = compute_f1(per_benchmark, "cegar_with_interp_verdict")
    metrics_bmc = compute_f1(per_benchmark, "bmc_verdict")

    # Gap analysis
    gap_cases = [e for e in per_benchmark if e["is_gap_case"]]
    gap_closed = [e for e in per_benchmark if e["interpolation_closes_gap"]]
    n_gap = len(gap_cases)
    n_closed = len(gap_closed)
    interp_success_rate = n_closed / n_gap if n_gap > 0 else 0.0
    wilson_lo, wilson_hi = wilson_ci(n_closed, n_gap)

    # Per-category breakdown
    category_results: Dict[str, Any] = {}
    for cat in categories:
        cat_entries = [e for e in per_benchmark if e["category"] == cat]
        cat_gap = [e for e in cat_entries if e["is_gap_case"]]
        cat_closed = [e for e in cat_entries if e["interpolation_closes_gap"]]
        category_results[cat] = {
            "total": len(cat_entries),
            "cegar_no_interp_correct": sum(1 for e in cat_entries if e["cegar_no_interp_correct"]),
            "cegar_with_interp_correct": sum(1 for e in cat_entries if e["cegar_with_interp_correct"]),
            "bmc_correct": sum(1 for e in cat_entries if e["bmc_correct"]),
            "gap_cases": len(cat_gap),
            "gaps_closed": len(cat_closed),
        }

    # Total interp preds
    total_interp_preds = sum(
        e["cegar_with_interp"]["interp_preds"] for e in per_benchmark
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  {'Mode':<35s}  {'F1':<7s}  {'Prec':<7s}  {'Rec':<7s}  {'Acc':<7s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'TN':>3s}")
    print(f"  {'─' * 75}")
    for label, m in [
        ("CEGAR (template-only, no interp)", metrics_no_interp),
        ("CEGAR (with interpolation)", metrics_with_interp),
        ("BMC (monolithic baseline)", metrics_bmc),
    ]:
        print(
            f"  {label:<35s}  {m['f1']:<7.4f}  {m['precision']:<7.4f}  "
            f"{m['recall']:<7.4f}  {m['accuracy']:<7.4f}  {m['tp']:>3d}  "
            f"{m['fp']:>3d}  {m['fn']:>3d}  {m['tn']:>3d}"
        )

    f1_delta = metrics_with_interp["f1"] - metrics_no_interp["f1"]
    acc_delta = metrics_with_interp["accuracy"] - metrics_no_interp["accuracy"]
    print(f"\n  Interpolation impact:")
    print(f"    ΔF1 (interp vs no-interp): {f1_delta:+.4f}")
    print(f"    ΔAccuracy:                 {acc_delta:+.4f}")
    print(f"    Total interp-discovered predicates: {total_interp_preds}")

    print(f"\n  Gap analysis (CEGAR fails but BMC succeeds):")
    print(f"    Gap cases:            {n_gap}")
    print(f"    Gaps closed by interp: {n_closed}")
    print(f"    Success rate:          {interp_success_rate:.4f}")
    print(f"    Wilson 95% CI:         [{wilson_lo:.4f}, {wilson_hi:.4f}]")

    if gap_cases:
        print(f"\n  Gap case details:")
        for e in gap_cases:
            closed = "CLOSED" if e["interpolation_closes_gap"] else "OPEN"
            print(f"    {e['name']:<40s}  {closed}")

    print(f"\n  Per-category breakdown:")
    print(f"  {'Category':<25s}  {'N':>3s}  {'NoI':>3s}  {'Interp':>6s}  {'BMC':>3s}  {'Gap':>3s}  {'Closed':>6s}")
    print(f"  {'─' * 60}")
    for cat in categories:
        cr = category_results[cat]
        print(
            f"  {cat:<25s}  {cr['total']:>3d}  {cr['cegar_no_interp_correct']:>3d}  "
            f"{cr['cegar_with_interp_correct']:>6d}  {cr['bmc_correct']:>3d}  "
            f"{cr['gap_cases']:>3d}  {cr['gaps_closed']:>6d}"
        )

    # ── Build output JSON ──────────────────────────────────────────────────
    # Clean entries for JSON (remove temporary keys)
    for entry in per_benchmark:
        entry.pop("cegar_no_interp_verdict", None)
        entry.pop("cegar_with_interp_verdict", None)
        entry.pop("bmc_verdict", None)

    output = {
        "experiment": "craig_suite_cd_eval",
        "description": (
            "Craig interpolation cross-domain evaluation: does interpolation "
            "close the CEGAR–BMC accuracy gap on architectures where template-based "
            "CEGAR typically fails?"
        ),
        "num_benchmarks": n,
        "num_buggy": n_buggy,
        "num_safe": n_safe,
        "categories": categories,
        "aggregate": {
            "cegar_no_interp": metrics_no_interp,
            "cegar_with_interp": metrics_with_interp,
            "bmc_baseline": metrics_bmc,
            "delta_f1_interp_vs_no_interp": round(f1_delta, 4),
            "delta_accuracy_interp_vs_no_interp": round(acc_delta, 4),
            "total_interpolation_predicates": total_interp_preds,
        },
        "gap_analysis": {
            "gap_cases": n_gap,
            "gaps_closed_by_interpolation": n_closed,
            "interpolation_success_rate": round(interp_success_rate, 4),
            "wilson_score_95_ci": [wilson_lo, wilson_hi],
            "gap_case_names": [e["name"] for e in gap_cases],
            "closed_case_names": [e["name"] for e in gap_closed],
        },
        "per_category": category_results,
        "per_benchmark": per_benchmark,
        "comparison_table": {
            "headers": ["Mode", "F1", "Precision", "Recall", "Accuracy", "TP", "FP", "FN", "TN"],
            "rows": [
                ["CEGAR (no interp)"] + [
                    metrics_no_interp[k] for k in ("f1", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn")
                ],
                ["CEGAR (with interp)"] + [
                    metrics_with_interp[k] for k in ("f1", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn")
                ],
                ["BMC (monolithic)"] + [
                    metrics_bmc[k] for k in ("f1", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn")
                ],
            ],
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
