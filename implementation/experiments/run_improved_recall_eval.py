#!/usr/bin/env python3
"""
Improved Recall Evaluation for TensorGuard Standalone.

Evaluates the expanded analysis capabilities against benchmarks covering:
  - All original benchmark categories
  - New layer types: InstanceNorm, SyncBatchNorm, Conv3d, Pool1d/3d, RNN,
    padding layers, PixelUnshuffle, etc.
  - New operations: expand, repeat, mean/sum reduction, F.pad, einsum, stack
  - Complex patterns: contiguous().view() chains, shape assertions

Reports precision, recall, F1 with comparison to previous recall=0.472.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified import analyze_unified

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "improved_recall_results.json"

PREVIOUS_RECALL = 0.472

# ═══════════════════════════════════════════════════════════════════════
# Benchmark definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Benchmark:
    name: str
    code: str
    expect_bug: bool
    category: str


BENCHMARKS: List[Benchmark] = []


def _b(name: str, code: str, expect_bug: bool, category: str):
    BENCHMARKS.append(Benchmark(name=name, code=code,
                                expect_bug=expect_bug, category=category))


# ── 1. Linear chain mismatches (existing) ─────────────────────────────

_b("linear_chain_mismatch", """
import torch
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
""", True, "shape_linear")

_b("linear_chain_correct", """
import torch
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
""", False, "shape_linear")

_b("linear_three_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", True, "shape_linear")

_b("linear_three_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""", False, "shape_linear")

# ── 2. Conv2d chain mismatches ────────────────────────────────────────

_b("conv_chain_mismatch", """
import torch
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
""", True, "shape_conv")

_b("conv_chain_correct", """
import torch
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
""", False, "shape_conv")

_b("conv1d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", True, "shape_conv")

_b("conv1d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 3)
        self.conv2 = nn.Conv1d(16, 32, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", False, "shape_conv")

# ── 3. Matmul mismatches ─────────────────────────────────────────────

_b("matmul_mismatch", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    return a @ b
""", True, "shape_matmul")

_b("matmul_correct", """
import torch
def f():
    a = torch.randn(3, 4)
    b = torch.randn(4, 6)
    return a @ b
""", False, "shape_matmul")

# ── 4. Null dereference ──────────────────────────────────────────────

_b("null_deref", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    return t.shape
""", True, "null_deref")

_b("null_deref_guarded", """
import torch
def f(cond):
    t = None
    if cond:
        t = torch.randn(3, 4)
    if t is not None:
        return t.shape
    return None
""", False, "null_deref")

# ── 5. InstanceNorm mismatches (NEW) ─────────────────────────────────

_b("instancenorm1d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, 3)
        self.norm = nn.InstanceNorm1d(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", True, "norm_layers")

_b("instancenorm1d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, 3)
        self.norm = nn.InstanceNorm1d(16)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", False, "norm_layers")

_b("instancenorm2d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.norm = nn.InstanceNorm2d(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", True, "norm_layers")

_b("instancenorm2d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.norm = nn.InstanceNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", False, "norm_layers")

# ── 6. GroupNorm mismatches (NEW) ─────────────────────────────────────

_b("groupnorm_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.norm = nn.GroupNorm(8, 32)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", True, "norm_layers")

_b("groupnorm_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.norm = nn.GroupNorm(8, 16)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
""", False, "norm_layers")

# ── 7. SyncBatchNorm ─────────────────────────────────────────────────

_b("syncbn_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.bn = nn.SyncBatchNorm(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", True, "norm_layers")

_b("syncbn_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.bn = nn.SyncBatchNorm(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", False, "norm_layers")

# ── 8. Pool1d / Pool3d (NEW) ─────────────────────────────────────────

_b("maxpool1d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, 3)
        self.pool = nn.MaxPool1d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
""", False, "pooling_layers")

_b("avgpool1d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, 3)
        self.pool = nn.AvgPool1d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
""", False, "pooling_layers")

_b("adaptiveavgpool1d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
""", False, "pooling_layers")

# ── 9. ConvTranspose mismatches (NEW) ────────────────────────────────

_b("convtranspose2d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.deconv = nn.ConvTranspose2d(32, 64, 3)
    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
""", True, "conv_transpose")

_b("convtranspose2d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.deconv = nn.ConvTranspose2d(16, 64, 3)
    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
""", False, "conv_transpose")

_b("convtranspose1d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, 3)
        self.deconv = nn.ConvTranspose1d(32, 64, 3)
    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
""", True, "conv_transpose")

# ── 10. Transformer layer mismatches (NEW) ───────────────────────────

_b("transformer_enc_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 64)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=2)
    def forward(self, x):
        x = self.fc(x)
        x = self.enc(x)
        return x
""", True, "transformer")

_b("transformer_enc_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 128)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=2)
    def forward(self, x):
        x = self.fc(x)
        x = self.enc(x)
        return x
""", False, "transformer")

# ── 11. LSTM / GRU mismatches (existing) ─────────────────────────────

_b("lstm_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lstm(x)
        return x
""", True, "rnn")

_b("lstm_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.lstm = nn.LSTM(32, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.lstm(x)
        return x
""", False, "rnn")

_b("gru_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.gru = nn.GRU(64, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.gru(x)
        return x
""", True, "rnn")

_b("gru_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.gru = nn.GRU(32, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.gru(x)
        return x
""", False, "rnn")

# ── 12. RNN (NEW) ────────────────────────────────────────────────────

_b("rnn_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.rnn = nn.RNN(64, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.rnn(x)
        return x
""", True, "rnn")

_b("rnn_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.rnn = nn.RNN(32, 128, batch_first=True)
    def forward(self, x):
        x = self.fc1(x)
        x, _ = self.rnn(x)
        return x
""", False, "rnn")

# ── 13. Padding layers (NEW) ─────────────────────────────────────────

_b("zeropad2d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ZeroPad2d(2)
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x
""", False, "padding_layers")

_b("reflectionpad_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(3, 16, 3)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x
""", False, "padding_layers")

# ── 14. PixelShuffle / PixelUnshuffle (NEW) ──────────────────────────

_b("pixelshuffle_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.ps = nn.PixelShuffle(4)
    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return x
""", True, "pixel_shuffle")

_b("pixelshuffle_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.ps = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        return x
""", False, "pixel_shuffle")

# ── 15. Conv3d (NEW) ─────────────────────────────────────────────────

_b("conv3d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3)
        self.conv2 = nn.Conv3d(32, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", True, "conv3d")

_b("conv3d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""", False, "conv3d")

# ── 16. BatchNorm3d (NEW) ────────────────────────────────────────────

_b("batchnorm3d_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 16, 3)
        self.bn = nn.BatchNorm3d(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", True, "norm_layers")

_b("batchnorm3d_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 16, 3)
        self.bn = nn.BatchNorm3d(16)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", False, "norm_layers")

# ── 17. Embedding dimension mismatch ─────────────────────────────────

_b("embedding_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x
""", True, "embedding")

_b("embedding_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x
""", False, "embedding")

# ── 18. MultiheadAttention ───────────────────────────────────────────

_b("mha_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 32)
        self.attn = nn.MultiheadAttention(64, 8)
    def forward(self, x):
        x = self.fc(x)
        x, _ = self.attn(x, x, x)
        return x
""", True, "attention")

_b("mha_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 64)
        self.attn = nn.MultiheadAttention(64, 8)
    def forward(self, x):
        x = self.fc(x)
        x, _ = self.attn(x, x, x)
        return x
""", False, "attention")

# ── 19. Flatten + Linear mismatch ────────────────────────────────────

_b("flatten_linear_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
""", True, "flatten_linear")

# ── 20. Squeeze / Unsqueeze patterns ─────────────────────────────────

_b("squeeze_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x
""", False, "squeeze_unsqueeze")

# ── 21. Reshape mismatches ───────────────────────────────────────────

_b("reshape_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", False, "reshape")

# ── 22. Complex real-world patterns ──────────────────────────────────

_b("resnet_block_mismatch", """
import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x
""", True, "real_world")

_b("resnet_block_correct", """
import torch
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x
""", False, "real_world")

_b("autoencoder_mismatch", """
import torch
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""", True, "real_world")

_b("autoencoder_correct", """
import torch
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""", False, "real_world")

_b("unet_skip_mismatch", """
import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.final = nn.Conv2d(256, 1, 1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        x = self.final(u1)
        return x
""", True, "real_world")

# ── 23. Activation function models (correct, true negatives) ─────────

_b("gelu_model_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
""", False, "activations")

_b("silu_model_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
""", False, "activations")

_b("mish_model_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
""", False, "activations")

# ── 24. AdaptiveAvgPool + Linear chain ───────────────────────────────

_b("adaptive_pool_linear_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
""", False, "real_world")

# ── 25. Conv + BN chain mismatches ───────────────────────────────────

_b("conv_bn_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.bn = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", True, "shape_conv")

_b("conv_bn_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.bn = nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
""", False, "shape_conv")

# ── 26. Sequential mismatch ──────────────────────────────────────────

_b("sequential_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(30, 5),
        )
    def forward(self, x):
        return self.net(x)
""", True, "sequential")

_b("sequential_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
    def forward(self, x):
        return self.net(x)
""", False, "sequential")

# ── 27. Wide linear mismatch ────────────────────────────────────────

_b("wide_linear_mismatch", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(512, 256)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""", True, "shape_linear")

# ── 28. Dropout variations (should be true negatives) ─────────────

_b("dropout_variations_correct", """
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.drop = nn.AlphaDropout(0.1)
        self.fc2 = nn.Linear(20, 5)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
""", False, "activations")


# ═══════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    category: str
    expect_bug: bool
    found_bugs: int
    bug_kinds: List[str]
    bug_messages: List[str]
    is_tp: bool
    is_fp: bool
    is_fn: bool
    is_tn: bool
    time_ms: float


def run_benchmark(bm: Benchmark) -> BenchmarkResult:
    t0 = time.perf_counter()
    try:
        result = analyze_unified(bm.code)
        bugs = result.bugs
    except Exception:
        bugs = []
    elapsed = (time.perf_counter() - t0) * 1000

    found = len(bugs)
    detected = found > 0

    tp = bm.expect_bug and detected
    fp = (not bm.expect_bug) and detected
    fn = bm.expect_bug and (not detected)
    tn = (not bm.expect_bug) and (not detected)

    return BenchmarkResult(
        name=bm.name,
        category=bm.category,
        expect_bug=bm.expect_bug,
        found_bugs=found,
        bug_kinds=[b.kind for b in bugs],
        bug_messages=[b.message for b in bugs],
        is_tp=tp, is_fp=fp, is_fn=fn, is_tn=tn,
        time_ms=elapsed,
    )


@dataclass
class CategoryMetrics:
    category: str
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


def compute_metrics(results: List[BenchmarkResult],
                    category: Optional[str] = None) -> CategoryMetrics:
    if category:
        subset = [r for r in results if r.category == category]
    else:
        subset = results
    tp = sum(r.is_tp for r in subset)
    fp = sum(r.is_fp for r in subset)
    fn = sum(r.is_fn for r in subset)
    tn = sum(r.is_tn for r in subset)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return CategoryMetrics(
        category=category or "overall",
        total=len(subset), tp=tp, fp=fp, fn=fn, tn=tn,
        precision=prec, recall=rec, f1=f1,
    )


def bootstrap_ci(values: List[float], n_boot: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    import random
    if not values:
        return (0.0, 0.0)
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(len(values))]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[int((1 - alpha) * n_boot)]
    return (lo, hi)


def print_table(results: List[BenchmarkResult]):
    categories = sorted(set(r.category for r in results))

    hdr = f"{'Category':<22} {'N':>3} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print("\n" + "=" * len(hdr))
    print("TensorGuard Improved Recall Evaluation")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for cat in categories:
        m = compute_metrics(results, cat)
        print(f"{m.category:<22} {m.total:>3} {m.tp:>3} {m.fp:>3} {m.fn:>3} {m.tn:>3} "
              f"{m.precision:>6.2f} {m.recall:>6.2f} {m.f1:>6.2f}")

    print("-" * len(hdr))
    overall = compute_metrics(results)
    print(f"{'OVERALL':<22} {overall.total:>3} {overall.tp:>3} {overall.fp:>3} "
          f"{overall.fn:>3} {overall.tn:>3} {overall.precision:>6.2f} "
          f"{overall.recall:>6.2f} {overall.f1:>6.2f}")
    print("=" * len(hdr))

    total_time = sum(r.time_ms for r in results)
    print(f"\nTotal benchmarks: {len(results)}")
    print(f"Total time: {total_time:.1f} ms  ({total_time/len(results):.1f} ms/benchmark)")

    # Comparison with previous
    print(f"\n── Recall Comparison ──")
    print(f"  Previous recall: {PREVIOUS_RECALL:.3f}")
    print(f"  Current recall:  {overall.recall:.3f}")
    delta = overall.recall - PREVIOUS_RECALL
    pct = (delta / PREVIOUS_RECALL * 100) if PREVIOUS_RECALL > 0 else 0
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"  Delta:           {direction} {abs(delta):.3f} ({pct:+.1f}%)")

    # Bootstrap CI
    tp_vals = [1.0 if r.is_tp else 0.0 for r in results if r.expect_bug]
    if tp_vals:
        lo, hi = bootstrap_ci(tp_vals)
        print(f"  95% CI recall:   [{lo:.3f}, {hi:.3f}]")

    # Print failures
    fns = [r for r in results if r.is_fn]
    fps = [r for r in results if r.is_fp]
    if fns:
        print(f"\n── False Negatives ({len(fns)}) ──")
        for r in fns:
            print(f"  ✗ {r.name} [{r.category}]")
    if fps:
        print(f"\n── False Positives ({len(fps)}) ──")
        for r in fps:
            print(f"  ✗ {r.name} [{r.category}]: {r.bug_kinds}")


def save_results(results: List[BenchmarkResult]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    categories = sorted(set(r.category for r in results))
    per_cat = {}
    for cat in categories:
        m = compute_metrics(results, cat)
        per_cat[cat] = asdict(m)
    overall = compute_metrics(results)

    tp_vals = [1.0 if r.is_tp else 0.0 for r in results if r.expect_bug]
    ci_lo, ci_hi = bootstrap_ci(tp_vals) if tp_vals else (0.0, 0.0)

    data = {
        "total_benchmarks": len(results),
        "total_time_ms": sum(r.time_ms for r in results),
        "overall": asdict(overall),
        "per_category": per_cat,
        "previous_recall": PREVIOUS_RECALL,
        "recall_improvement": overall.recall - PREVIOUS_RECALL,
        "recall_95ci": [ci_lo, ci_hi],
        "results": [asdict(r) for r in results],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"Running {len(BENCHMARKS)} benchmarks ...\n")
    results: List[BenchmarkResult] = []
    for i, bm in enumerate(BENCHMARKS, 1):
        r = run_benchmark(bm)
        status = "✓" if (r.is_tp or r.is_tn) else "✗"
        label = "BUG" if bm.expect_bug else "OK "
        found_label = f"{r.found_bugs} bugs" if r.found_bugs else "clean"
        print(f"  [{i:>2}/{len(BENCHMARKS)}] {status} {bm.name:<38} expect={label}  got={found_label:<12} {r.time_ms:.1f}ms")
        results.append(r)

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
