#!/usr/bin/env python3
"""
Chain-of-Thought LLM Baseline Comparison for TensorGuard.

Compares TensorGuard against three GPT-4.1-nano prompting strategies:
  1. Simple prompting (existing baseline)
  2. Chain-of-thought (step-by-step shape tracing)
  3. Architecture-aware CoT (expert persona + structured analysis)

This is a "weak LLM" benchmark (gpt-4.1-nano) — the goal is to measure
how much CoT prompting can close the gap, not to claim LLMs cannot do this.

Usage (from implementation/):
    source ~/.bashrc
    python3 experiments/run_cot_llm_baseline.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified import analyze_unified

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "cot_llm_baseline_results.json"

MODEL = "gpt-4.1-nano"

# ═══════════════════════════════════════════════════════════════════════════════
# Prompting strategies
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES: Dict[str, Dict[str, str]] = {
    "simple": {
        "system": (
            "You are a code analyzer. The user will show you a PyTorch nn.Module. "
            "Determine if it has a shape bug. Answer YES or NO on the first line."
        ),
        "user_template": (
            "Does this PyTorch model have a shape bug? Answer YES or NO.\n\n"
            "```python\n{code}\n```"
        ),
    },
    "chain_of_thought": {
        "system": (
            "You are a careful code analyzer. Think step by step before answering."
        ),
        "user_template": (
            "Analyze this PyTorch model step by step. For each layer in __init__, "
            "note its input and output dimensions. Then trace the forward() method, "
            "tracking the tensor shape at each step. Does any dimension mismatch occur? "
            "Show your work, then answer YES or NO on the final line.\n\n"
            "```python\n{code}\n```"
        ),
    },
    "architecture_aware_cot": {
        "system": (
            "You are an expert PyTorch debugger specializing in tensor shape analysis. "
            "You methodically verify neural network architectures for correctness."
        ),
        "user_template": (
            "You are an expert PyTorch debugger. Analyze this nn.Module for shape bugs "
            "using the following procedure:\n"
            "1. List all layers defined in __init__ and their expected input/output shapes.\n"
            "2. Trace data flow through forward(), noting the tensor shape after each operation.\n"
            "3. Check each connection point for dimensional compatibility.\n"
            "4. Check for device assignment issues and train/eval behavioral differences.\n"
            "Show your full analysis, then on the final line answer YES (bug found) or "
            "NO (code is correct). If YES, state the specific bug location.\n\n"
            "```python\n{code}\n```"
        ),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 36 benchmarks (18 buggy / 18 clean) — representative nn.Module programs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    code: str
    has_bug: bool
    category: str

BENCHMARKS: List[Benchmark] = [
    # ── Buggy (has_bug=True) ──────────────────────────────────────────────
    Benchmark("linear_chain_mismatch", """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", True, "shape_linear"),

    Benchmark("conv_channel_mismatch", """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", True, "shape_conv"),

    Benchmark("autoencoder_decoder_mismatch", """\
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 128)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
""", True, "shape_linear"),

    Benchmark("mlp_hidden_mismatch", """\
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", True, "shape_linear"),

    Benchmark("resnet_shortcut_mismatch", """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x
""", True, "shape_conv"),

    Benchmark("transformer_mlp_mismatch", """\
import torch.nn as nn
class TransformerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(2048, 768)
    def forward(self, x):
        h = self.fc1(self.norm(x))
        return x + self.fc2(h)
""", True, "shape_linear"),

    Benchmark("conv_flatten_linear_mismatch", """\
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
""", True, "shape_flatten"),

    Benchmark("lstm_hidden_mismatch", """\
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
""", True, "shape_rnn"),

    Benchmark("attention_proj_mismatch", """\
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(256, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn / 22.6, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)
""", True, "shape_attention"),

    Benchmark("unet_skip_mismatch", """\
import torch
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.merge = nn.Conv2d(192, 64, 1)
    def forward(self, x):
        d = self.down(x)
        u = self.up(d)
        return self.merge(torch.cat([u, x], dim=1))
""", True, "shape_conv"),

    Benchmark("double_conv_channel_bug", """\
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.bn(x)
""", True, "shape_conv"),

    Benchmark("embedding_linear_mismatch", """\
import torch.nn as nn
class EmbedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(256, 64)
    def forward(self, x):
        x = self.embed(x)
        return self.fc(x)
""", True, "shape_embedding"),

    Benchmark("batchnorm_channel_mismatch", """\
import torch.nn as nn
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", True, "shape_norm"),

    Benchmark("gru_output_mismatch", """\
import torch.nn as nn
class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(50, 100, batch_first=True)
        self.fc = nn.Linear(200, 10)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
""", True, "shape_rnn"),

    Benchmark("three_layer_conv_bug", """\
import torch.nn as nn
class ThreeLayerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x
""", True, "shape_conv"),

    Benchmark("classifier_head_mismatch", """\
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, 10)
    def forward(self, x):
        x = self.features(x)
        return self.head(x)
""", True, "shape_linear"),

    Benchmark("vae_decoder_mismatch", """\
import torch.nn as nn
class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        return self.fc3(h)
""", True, "shape_linear"),

    Benchmark("multihead_concat_mismatch", """\
import torch
import torch.nn as nn
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(256, 64)
        self.head2 = nn.Linear(256, 64)
        self.merge = nn.Linear(256, 128)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        return self.merge(torch.cat([h1, h2], dim=-1))
""", True, "shape_linear"),

    # ── Clean (has_bug=False) ─────────────────────────────────────────────
    Benchmark("linear_chain_correct", """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", False, "shape_linear"),

    Benchmark("conv_chain_correct", """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", False, "shape_conv"),

    Benchmark("autoencoder_correct", """\
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
""", False, "shape_linear"),

    Benchmark("mlp_correct", """\
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", False, "shape_linear"),

    Benchmark("resnet_block_correct", """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x
""", False, "shape_conv"),

    Benchmark("transformer_mlp_correct", """\
import torch.nn as nn
class TransformerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        h = self.fc1(self.norm(x))
        return x + self.fc2(h)
""", False, "shape_linear"),

    Benchmark("lstm_classifier_correct", """\
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
""", False, "shape_rnn"),

    Benchmark("attention_correct", """\
import torch
import torch.nn as nn
class SelfAttention(nn.Module):
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
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn / 22.6, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)
""", False, "shape_attention"),

    Benchmark("embedding_net_correct", """\
import torch.nn as nn
class EmbedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(128, 64)
    def forward(self, x):
        x = self.embed(x)
        return self.fc(x)
""", False, "shape_embedding"),

    Benchmark("batchnorm_correct", """\
import torch.nn as nn
class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", False, "shape_norm"),

    Benchmark("gru_net_correct", """\
import torch.nn as nn
class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(50, 100, batch_first=True)
        self.fc = nn.Linear(100, 10)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
""", False, "shape_rnn"),

    Benchmark("three_layer_conv_correct", """\
import torch.nn as nn
class ThreeLayerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x
""", False, "shape_conv"),

    Benchmark("classifier_correct", """\
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Linear(256, 10)
    def forward(self, x):
        x = self.features(x)
        return self.head(x)
""", False, "shape_linear"),

    Benchmark("vae_decoder_correct", """\
import torch.nn as nn
class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 784)
    def forward(self, z):
        h = self.fc1(z)
        h = self.fc2(h)
        return self.fc3(h)
""", False, "shape_linear"),

    Benchmark("multihead_correct", """\
import torch
import torch.nn as nn
class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(256, 64)
        self.head2 = nn.Linear(256, 64)
        self.merge = nn.Linear(128, 128)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        return self.merge(torch.cat([h1, h2], dim=-1))
""", False, "shape_linear"),

    Benchmark("double_conv_correct", """\
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.bn(x)
""", False, "shape_conv"),

    Benchmark("deep_classifier_correct", """\
import torch.nn as nn
class DeepClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
    def forward(self, x):
        return self.net(x)
""", False, "shape_linear"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return (max(0.0, c - m), min(1.0, c + m))


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 4),
        "precision_ci_95": [round(x, 4) for x in wilson_ci(tp, tp + fp)],
        "recall": round(recall, 4),
        "recall_ci_95": [round(x, 4) for x in wilson_ci(tp, tp + fn)],
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def run_tensorguard(code: str) -> bool:
    """Return True if TensorGuard finds at least one bug."""
    try:
        result = analyze_unified(code)
        return len(result.bugs) > 0
    except Exception as e:
        print(f"    [TensorGuard error] {e}")
        return False


def query_llm(
    client, code: str, strategy_name: str, max_tokens: int = 1024
) -> Tuple[Optional[bool], str]:
    """Query gpt-4.1-nano with the given prompting strategy."""
    strat = STRATEGIES[strategy_name]
    user_msg = strat["user_template"].format(code=code)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": strat["system"]},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        predicted = parse_yes_no(text)
        return predicted, text
    except Exception as e:
        return None, f"ERROR: {e}"


def parse_yes_no(text: str) -> Optional[bool]:
    """Extract YES/NO prediction from LLM response.

    Checks the first line, last line, and full text for YES/NO signals.
    """
    if not text:
        return None
    upper = text.upper()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    # Check first word
    first_word = lines[0].split()[0].strip(".:,*#") if lines else ""
    if first_word in ("YES", "NO"):
        return first_word == "YES"

    # Check last line (CoT puts answer at end)
    if lines:
        last_line = lines[-1].upper()
        for marker in ("ANSWER: YES", "ANSWER:YES", "**YES**", "YES.", "YES,", "YES:"):
            if marker in last_line:
                return True
        for marker in ("ANSWER: NO", "ANSWER:NO", "**NO**", "NO.", "NO,", "NO:"):
            if marker in last_line:
                return False
        last_word = last_line.split()[-1].strip(".:,*#()") if last_line.split() else ""
        if last_word in ("YES", "NO"):
            return last_word == "YES"

    # Fallback: scan full text for strong signals
    if "BUG FOUND" in upper or "MISMATCH" in upper and "NO MISMATCH" not in upper:
        return True
    if "NO BUG" in upper or "CODE IS CORRECT" in upper or "NO SHAPE" in upper:
        return False

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Run `source ~/.bashrc` first.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    num_bm = len(BENCHMARKS)
    num_buggy = sum(1 for b in BENCHMARKS if b.has_bug)
    num_clean = num_bm - num_buggy

    print("=" * 78)
    print(f"Chain-of-Thought LLM Baseline  —  {MODEL} (weak LLM benchmark)")
    print(f"{num_bm} benchmarks ({num_buggy} buggy / {num_clean} clean)")
    print(f"Strategies: {', '.join(STRATEGIES.keys())}")
    print("=" * 78)

    # ── Run TensorGuard ──────────────────────────────────────────────────
    print("\n▶ Running TensorGuard on all benchmarks...")
    tg_results: Dict[str, bool] = {}
    for bm in BENCHMARKS:
        found = run_tensorguard(bm.code)
        tg_results[bm.name] = found
        label = ("TP" if found else "FN") if bm.has_bug else ("FP" if found else "TN")
        print(f"  {bm.name:<45s} expect={bm.has_bug}  found={found}  {label}")

    tg_tp = sum(1 for b in BENCHMARKS if b.has_bug and tg_results[b.name])
    tg_fp = sum(1 for b in BENCHMARKS if not b.has_bug and tg_results[b.name])
    tg_fn = sum(1 for b in BENCHMARKS if b.has_bug and not tg_results[b.name])
    tg_tn = sum(1 for b in BENCHMARKS if not b.has_bug and not tg_results[b.name])
    tg_metrics = compute_metrics(tg_tp, tg_fp, tg_fn, tg_tn)

    # ── Run LLM strategies ───────────────────────────────────────────────
    strategy_results: Dict[str, Dict[str, Any]] = {}

    for strat_name in STRATEGIES:
        print(f"\n▶ Running LLM strategy: {strat_name}")
        print("-" * 78)

        tp = fp = fn = tn = skipped = 0
        rows: List[Dict[str, Any]] = []

        for i, bm in enumerate(BENCHMARKS, 1):
            print(f"  [{i:2d}/{num_bm}] {bm.name:<45s}", end="", flush=True)

            t0 = time.time()
            predicted, raw = query_llm(client, bm.code, strat_name)
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            if predicted is None:
                label = "SKIP"
                skipped += 1
                print(f"  SKIP ({elapsed_ms}ms)")
            elif bm.has_bug:
                if predicted:
                    tp += 1; label = "TP"
                else:
                    fn += 1; label = "FN"
                print(f"  {label} ({elapsed_ms}ms)")
            else:
                if predicted:
                    fp += 1; label = "FP"
                else:
                    tn += 1; label = "TN"
                print(f"  {label} ({elapsed_ms}ms)")

            rows.append({
                "name": bm.name,
                "category": bm.category,
                "has_bug": bm.has_bug,
                "predicted": predicted,
                "label": label,
                "ms": elapsed_ms,
                "llm_response": raw,
            })

            # Rate limiting
            time.sleep(0.3)

        m = compute_metrics(tp, fp, fn, tn)

        # Per-category breakdown
        cats: Dict[str, Dict[str, int]] = {}
        for r in rows:
            c = r["category"]
            if c not in cats:
                cats[c] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            lbl = r["label"]
            if lbl in ("TP", "FP", "FN", "TN"):
                cats[c][lbl.lower()] += 1

        print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}  skipped={skipped}")
        print(f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}")

        strategy_results[strat_name] = {
            "strategy": strat_name,
            "model": MODEL,
            "metrics": m,
            "skipped": skipped,
            "per_category": {
                cat: compute_metrics(d["tp"], d["fp"], d["fn"], d["tn"])
                for cat, d in sorted(cats.items())
            },
            "benchmarks": rows,
        }

    # ── Summary comparison table ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("COMPARISON SUMMARY")
    print("=" * 78)

    header = f"{'Strategy':<25s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Acc':>7s}  {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s}"
    print(header)
    print("-" * len(header))

    # TensorGuard row
    print(f"{'TensorGuard':<25s} {tg_metrics['precision']:>7.4f} {tg_metrics['recall']:>7.4f} "
          f"{tg_metrics['f1']:>7.4f} {tg_metrics['accuracy']:>7.4f}  "
          f"{tg_tp:>3d} {tg_fp:>3d} {tg_fn:>3d} {tg_tn:>3d}")

    for strat_name, sr in strategy_results.items():
        m = sr["metrics"]
        display = f"LLM ({strat_name})"
        print(f"{display:<25s} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['accuracy']:>7.4f}  "
              f"{m['TP']:>3d} {m['FP']:>3d} {m['FN']:>3d} {m['TN']:>3d}")

    # ── Per-benchmark detail table ───────────────────────────────────────
    print(f"\n{'Benchmark':<45s} {'Bug':>4s} {'TG':>4s}", end="")
    for sn in STRATEGIES:
        print(f" {sn[:8]:>8s}", end="")
    print()
    print("-" * (45 + 4 + 4 + 8 * len(STRATEGIES) + len(STRATEGIES)))

    for bm in BENCHMARKS:
        bug_str = "BUG" if bm.has_bug else "OK"
        tg_label = ("TP" if tg_results[bm.name] else "FN") if bm.has_bug else ("FP" if tg_results[bm.name] else "TN")
        print(f"{bm.name:<45s} {bug_str:>4s} {tg_label:>4s}", end="")
        for sn in STRATEGIES:
            row = next(r for r in strategy_results[sn]["benchmarks"] if r["name"] == bm.name)
            print(f" {row['label']:>8s}", end="")
        print()

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "description": (
            f"Chain-of-thought LLM baseline comparison using {MODEL} (weak LLM). "
            "Compares simple prompting, chain-of-thought, and architecture-aware CoT "
            "against TensorGuard's formal verification."
        ),
        "model": MODEL,
        "note": (
            "gpt-4.1-nano is a weak/small model. This benchmark measures prompting "
            "strategy impact, not frontier LLM capability."
        ),
        "num_benchmarks": num_bm,
        "num_buggy": num_buggy,
        "num_clean": num_clean,
        "tensorguard_metrics": tg_metrics,
        "strategies": strategy_results,
        "comparison": {
            "tensorguard_f1": tg_metrics["f1"],
            **{
                f"{sn}_f1": strategy_results[sn]["metrics"]["f1"]
                for sn in STRATEGIES
            },
        },
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
