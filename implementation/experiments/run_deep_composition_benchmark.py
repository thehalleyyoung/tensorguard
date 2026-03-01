"""
LLM failure benchmark: Deep composition chains where LLMs fail.

Tests cases specifically designed to expose LLM limitations on shape
verification, demonstrating formal verification's advantage on complex
multi-hop shape reasoning.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

# Benchmark cases: each has source code, input shapes, expected verdict,
# and difficulty annotation
BENCHMARKS = [
    # --- Deep composition chain (15 layers, bug at layer 12) ---
    {
        "name": "deep-chain-15-bug-at-12",
        "category": "deep_composition",
        "difficulty": "15-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepChain(nn.Module):
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
        self.fc12 = nn.Linear(128, 256)  # BUG: expects 128 but gets 256
        self.fc13 = nn.Linear(256, 256)
        self.fc14 = nn.Linear(256, 256)
        self.fc15 = nn.Linear(256, 10)
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
        return x
'''
    },

    # --- Reshape chain with subtle divisibility bug ---
    {
        "name": "reshape-divisibility-bug",
        "category": "reshape_arithmetic",
        "difficulty": "arithmetic",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class ReshapeChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.fc = nn.Linear(256 * 8 * 9, 10)  # BUG: 8*9=72 but should be 8*8=64
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Cross-branch broadcast with subtle mismatch ---
    {
        "name": "cross-branch-broadcast-subtle",
        "category": "cross_branch",
        "difficulty": "cross-branch",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class CrossBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 63)   # 63
        )
        self.branch_b = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)   # 64 -- subtle off-by-one vs 63
        )
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        combined = a + b  # BUG: 63 vs 64
        return self.out(combined)
'''
    },

    # --- Multi-head attention dimension mismatch (requires multiplication) ---
    {
        "name": "mha-dim-mismatch",
        "category": "transformer_arithmetic",
        "difficulty": "arithmetic",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 128, 512)},
        "source": '''
import torch.nn as nn
class TransformerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(256, 512)  # BUG: expects 256 but gets 512
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = q @ k.transpose(-2, -1)
        attn = v  # simplified
        out = self.out_proj(attn)
        return out
'''
    },

    # --- Safe deep chain (20 layers, all correct) ---
    {
        "name": "deep-chain-20-safe",
        "category": "deep_composition",
        "difficulty": "20-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class DeepSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(19)])
        self.out = nn.Linear(512, 10)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
'''
    },

    # --- Interleaved conv-flatten-linear with arithmetic ---
    {
        "name": "conv-flatten-arithmetic-bug",
        "category": "conv_arithmetic",
        "difficulty": "arithmetic",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch.nn as nn
class ConvFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=2)    # 64 -> 30
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)   # 30 -> 14
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)  # 14 -> 6
        self.fc = nn.Linear(128 * 7 * 7, 256)  # BUG: 7*7 but actual is 6*6
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Double reshape with product constraint ---
    {
        "name": "double-reshape-product-bug",
        "category": "reshape_arithmetic",
        "difficulty": "arithmetic",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 768)},
        "source": '''
import torch.nn as nn
class DoubleReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc_out = nn.Linear(192, 10)  # BUG: expects 192 but gets 256
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 3, 256)  # (batch, 3, 256)
        x = self.fc2(x)
        x = x.view(-1, 3, 256)  # (batch, 3, 256)
        x = x[:, 0, :]          # (batch, 256)
        x = self.fc_out(x)      # expects 192, gets 256
        return x
'''
    },

    # --- Nested sequential with dimension tracking ---
    {
        "name": "nested-sequential-bug",
        "category": "deep_composition",
        "difficulty": "10-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 1024)},
        "source": '''
import torch.nn as nn
class NestedSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1023),  # BUG: 1023 vs 1024
        )
        self.skip = nn.Linear(1024, 1024)
    def forward(self, x):
        identity = self.skip(x)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon + identity  # BUG: 1023 + 1024 shape mismatch
'''
    },

    # --- Embedding dimension cascade bug ---
    {
        "name": "embedding-cascade-bug",
        "category": "embedding_arithmetic",
        "difficulty": "cascade",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 128, 512)},
        "source": '''
import torch.nn as nn
class EmbeddingCascade(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(512, 256)
        self.proj2 = nn.Linear(256, 128)
        self.proj3 = nn.Linear(128, 64)
        self.proj4 = nn.Linear(64, 32)
        self.proj5 = nn.Linear(32, 16)
        self.proj6 = nn.Linear(16, 8)
        self.out = nn.Linear(9, 1)  # BUG: expects 9 but gets 8
    def forward(self, x):
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.proj3(x)
        x = self.proj4(x)
        x = self.proj5(x)
        x = self.proj6(x)
        x = self.out(x)
        return x
'''
    },

    # --- Safe complex architecture ---
    {
        "name": "complex-safe-unet-style",
        "category": "complex_safe",
        "difficulty": "15-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 3, 128, 128)},
        "source": '''
import torch.nn as nn
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        return self.out(d2)
'''
    },

    # --- Benchmark 11: 25-layer chain with dimension changes, bug buried at layer 18 ---
    {
        "name": "deep-chain-25-bug-at-18",
        "category": "deep_composition",
        "difficulty": "25-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class DeepChain25(nn.Module):
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
        self.fc10 = nn.Linear(512, 256)
        self.fc11 = nn.Linear(256, 256)
        self.fc12 = nn.Linear(256, 256)
        self.fc13 = nn.Linear(256, 256)
        self.fc14 = nn.Linear(256, 256)
        self.fc15 = nn.Linear(256, 256)
        self.fc16 = nn.Linear(256, 256)
        self.fc17 = nn.Linear(256, 256)
        self.fc18 = nn.Linear(512, 128)  # BUG: expects 512 but gets 256
        self.fc19 = nn.Linear(128, 128)
        self.fc20 = nn.Linear(128, 128)
        self.fc21 = nn.Linear(128, 128)
        self.fc22 = nn.Linear(128, 64)
        self.fc23 = nn.Linear(64, 64)
        self.fc24 = nn.Linear(64, 32)
        self.fc25 = nn.Linear(32, 10)
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
        x = self.fc20(x)
        x = self.fc21(x)
        x = self.fc22(x)
        x = self.fc23(x)
        x = self.fc24(x)
        x = self.fc25(x)
        return x
'''
    },

    # --- Benchmark 12: Conv + MaxPool chain with non-power-of-2 input causing subtle spatial bug ---
    {
        "name": "conv-maxpool-spatial-bug",
        "category": "pooling_arithmetic",
        "difficulty": "6-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 30, 30)},
        "source": '''
import torch.nn as nn
class ConvPoolSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 4 * 4, 10)  # BUG: 30->15->7->3, not 4
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Benchmark 13: Three branches merged with subtle off-by-one in third branch ---
    {
        "name": "triple-branch-merge-bug",
        "category": "multi_branch_merge",
        "difficulty": "3-branch",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch
import torch.nn as nn
class TripleBranchMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.branch_b = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.branch_c = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 65)
        )
        self.out = nn.Linear(192, 10)  # BUG: 64+64+65=193, not 192
    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        combined = torch.cat([a, b, c], dim=1)
        return self.out(combined)
'''
    },

    # --- Benchmark 14: ConvTranspose2d encoder-decoder with spatial off-by-one in flatten ---
    {
        "name": "transpose-conv-spatial-bug",
        "category": "transpose_conv_arithmetic",
        "difficulty": "6-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "source": '''
import torch.nn as nn
class TransConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.fc = nn.Linear(16 * 63 * 63, 10)  # BUG: actual spatial is 64, not 63
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Benchmark 15: Long conv-pool chain with non-standard input, wrong flatten size ---
    {
        "name": "long-conv-pool-flatten-bug",
        "category": "conv_arithmetic",
        "difficulty": "6-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 50, 50)},
        "source": '''
import torch.nn as nn
class LongConvPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 5 * 5, 256)  # BUG: 50->48->24->22->11->9->4, not 5
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Benchmark 16: Deep autoencoder with skip connections, off-by-one at final skip ---
    {
        "name": "autoencoder-skip-mismatch",
        "category": "deep_composition",
        "difficulty": "16-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 768)},
        "source": '''
import torch
import torch.nn as nn
class AutoencoderSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(768, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)
        self.enc4 = nn.Linear(128, 64)
        self.enc5 = nn.Linear(64, 32)
        self.dec5 = nn.Linear(32, 64)
        self.dec4 = nn.Linear(128, 128)
        self.dec3 = nn.Linear(256, 256)
        self.dec2 = nn.Linear(512, 512)
        self.dec1 = nn.Linear(1025, 768)  # BUG: cat(512,512)=1024, not 1025
        self.out = nn.Linear(768, 10)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d5 = self.dec5(e5)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out(d1)
'''
    },

    # --- Benchmark 17: Four conv branches with wrong aggregate channel count ---
    {
        "name": "four-branch-concat-channel-bug",
        "category": "multi_branch_merge",
        "difficulty": "4-branch",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 3, 16, 16)},
        "source": '''
import torch
import torch.nn as nn
class FourBranchConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.branch_a = nn.Sequential(
            nn.Conv2d(64, 48, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU()
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(64, 48, 1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU()
        )
        self.branch_c = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU()
        )
        self.branch_d = nn.Sequential(
            nn.Conv2d(64, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.merge = nn.Conv2d(256, 128, 1)  # BUG: 64+96+48+32=240, not 256
        self.out = nn.Conv2d(128, 10, 1)
    def forward(self, x):
        x = self.stem(x)
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        d = self.branch_d(x)
        merged = torch.cat([a, b, c, d], dim=1)
        x = self.merge(merged)
        return self.out(x)
'''
    },

    # --- Benchmark 18: Deep residual with shortcut dimension mismatch after many layers ---
    {
        "name": "deep-residual-shortcut-bug",
        "category": "cross_branch",
        "difficulty": "8-hop",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepResidualShortcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, 128)
        self.shortcut = nn.Linear(256, 127)  # BUG: 127 vs 128
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = x + identity  # BUG: 128 vs 127
        return self.out(x)
'''
    },

    # --- Benchmark 19: Reshape + linear chain with wrong flatten product ---
    {
        "name": "reshape-flatten-linear-bug",
        "category": "reshape_arithmetic",
        "difficulty": "arithmetic",
        "expected_safe": False,
        "input_shapes": {"x": ("batch", 384)},
        "source": '''
import torch.nn as nn
class ReshapeFlattenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 384)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(128, 10)  # BUG: flatten(3*32)=96, not 128
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 3, 128)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.flatten(1)
        x = self.fc_out(x)
        return x
'''
    },

    # --- Benchmark 20: Deep bottleneck autoencoder, all dimensions correct ---
    {
        "name": "deep-bottleneck-safe",
        "category": "deep_composition",
        "difficulty": "11-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 512)},
        "source": '''
import torch.nn as nn
class DeepBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 32)
        self.fc7 = nn.Linear(32, 64)
        self.fc8 = nn.Linear(64, 128)
        self.fc9 = nn.Linear(128, 256)
        self.fc10 = nn.Linear(256, 512)
        self.fc11 = nn.Linear(512, 10)
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
        return x
'''
    },

    # --- Benchmark 21: Conv + pool deep chain, power-of-2 input, all correct ---
    {
        "name": "conv-pool-chain-safe",
        "category": "pooling_arithmetic",
        "difficulty": "10-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class ConvPoolChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.fc = nn.Linear(256 * 2 * 2, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
'''
    },

    # --- Benchmark 22: Inception-style multi-scale merge, all correct ---
    {
        "name": "inception-merge-safe",
        "category": "multi_branch_merge",
        "difficulty": "3-branch",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 3, 16, 16)},
        "source": '''
import torch
import torch.nn as nn
class InceptionMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Conv2d(3, 64, 3, padding=1)
        self.branch1x1 = nn.Conv2d(64, 32, 1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(64, 48, 1), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(64, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2)
        )
        self.merge = nn.Conv2d(128, 10, 1)  # 32+64+32=128
    def forward(self, x):
        x = self.pre(x)
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        return self.merge(torch.cat([b1, b2, b3], dim=1))
'''
    },

    # --- Benchmark 23: ConvTranspose2d roundtrip encoder-decoder, all correct ---
    {
        "name": "transpose-conv-roundtrip-safe",
        "category": "transpose_conv_arithmetic",
        "difficulty": "6-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class TransConvRoundtrip(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        self.out = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.out(x)
'''
    },

    # --- Benchmark 24: Deep residual network with 8 correct skip blocks ---
    {
        "name": "deep-residual-8-block-safe",
        "category": "cross_branch",
        "difficulty": "24-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 256)},
        "source": '''
import torch.nn as nn
class DeepResidual8(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
            for _ in range(8)
        ])
        self.out = nn.Linear(256, 10)
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)
'''
    },

    # --- Benchmark 25: VGG-style deep conv-bn-relu chain, all correct ---
    {
        "name": "vgg-style-deep-safe",
        "category": "conv_arithmetic",
        "difficulty": "13-hop",
        "expected_safe": True,
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "source": '''
import torch.nn as nn
class VGGStyleDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(256 * 4 * 4, 10)
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
'''
    },
]


def run_tensorguard_benchmark():
    """Run TensorGuard on all benchmarks."""
    results = []

    for bench in BENCHMARKS:
        t0 = time.monotonic()
        try:
            result = verify_model(
                bench["source"],
                input_shapes=bench["input_shapes"],
            )
            elapsed = (time.monotonic() - t0) * 1000

            actual_safe = result.safe
            correct = actual_safe == bench["expected_safe"]

            rec = {
                "name": bench["name"],
                "category": bench["category"],
                "difficulty": bench["difficulty"],
                "expected_safe": bench["expected_safe"],
                "actual_safe": actual_safe,
                "correct": correct,
                "time_ms": round(elapsed, 1),
            }
            if not actual_safe and result.counterexample:
                rec["violation"] = result.counterexample.violations[0].message[:200] if result.counterexample.violations else ""
            results.append(rec)

            status = "✓" if correct else "✗"
            verdict = "SAFE" if actual_safe else "UNSAFE"
            print(f"  {status} {bench['name']}: {verdict} ({elapsed:.0f}ms)")

        except Exception as e:
            results.append({
                "name": bench["name"],
                "category": bench["category"],
                "expected_safe": bench["expected_safe"],
                "error": str(e),
                "correct": False,
            })
            print(f"  ✗ {bench['name']}: ERROR: {e}")

    return results


def run_llm_benchmark():
    """Run GPT-4.1-nano on the deep composition benchmarks."""
    try:
        import openai
        import subprocess
        subprocess.run(["bash", "-c", "source ~/.bashrc"], capture_output=True)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Try to source bashrc
            result = subprocess.run(
                ["bash", "-c", "source ~/.bashrc && echo $OPENAI_API_KEY"],
                capture_output=True, text=True
            )
            api_key = result.stdout.strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        if not api_key:
            print("  No OPENAI_API_KEY found, skipping LLM benchmark")
            return None

        client = openai.OpenAI(api_key=api_key)
    except ImportError:
        print("  openai package not installed, skipping LLM benchmark")
        return None

    results = []
    for bench in BENCHMARKS:
        prompt = f"""Analyze this PyTorch nn.Module for shape bugs. The input shapes are: {bench['input_shapes']}.

Respond with EXACTLY one of:
- "SAFE" if the model has no shape bugs
- "UNSAFE" if the model has shape bugs

Think step by step about the shapes at each layer.

```python
{bench['source'].strip()}
```

Your answer (SAFE or UNSAFE):"""

        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            elapsed = (time.monotonic() - t0) * 1000

            answer = response.choices[0].message.content.strip()
            # Extract verdict
            if "UNSAFE" in answer.upper():
                llm_safe = False
            elif "SAFE" in answer.upper():
                llm_safe = True
            else:
                llm_safe = True  # default to safe if unclear

            correct = llm_safe == bench["expected_safe"]
            results.append({
                "name": bench["name"],
                "category": bench["category"],
                "difficulty": bench["difficulty"],
                "expected_safe": bench["expected_safe"],
                "llm_safe": llm_safe,
                "correct": correct,
                "time_ms": round(elapsed, 1),
                "llm_answer_snippet": answer[:200],
            })
            status = "✓" if correct else "✗"
            verdict = "SAFE" if llm_safe else "UNSAFE"
            print(f"  {status} {bench['name']}: LLM={verdict} ({elapsed:.0f}ms)")

        except Exception as e:
            results.append({
                "name": bench["name"],
                "error": str(e),
                "correct": False,
            })
            print(f"  ✗ {bench['name']}: LLM ERROR: {e}")

    return results


def main():
    print("=" * 60)
    print("TensorGuard Deep Composition Benchmark")
    print("=" * 60)

    print("\n--- TensorGuard ---")
    tg_results = run_tensorguard_benchmark()
    tg_correct = sum(1 for r in tg_results if r.get("correct"))
    tg_total = len(tg_results)
    print(f"\nTensorGuard: {tg_correct}/{tg_total} correct")

    print("\n--- GPT-4.1-nano (CoT) ---")
    llm_results = run_llm_benchmark()

    summary = {
        "tensorguard": {
            "correct": tg_correct,
            "total": tg_total,
            "accuracy": round(tg_correct / max(tg_total, 1), 4),
            "results": tg_results,
        },
    }

    if llm_results is not None:
        llm_correct = sum(1 for r in llm_results if r.get("correct"))
        llm_total = len(llm_results)
        print(f"\nGPT-4.1-nano: {llm_correct}/{llm_total} correct")

        summary["llm"] = {
            "correct": llm_correct,
            "total": llm_total,
            "accuracy": round(llm_correct / max(llm_total, 1), 4),
            "results": llm_results,
        }

        # Identify cases where TG succeeds but LLM fails
        tg_wins = []
        llm_wins = []
        for tg_r, llm_r in zip(tg_results, llm_results):
            if tg_r.get("correct") and not llm_r.get("correct"):
                tg_wins.append(tg_r["name"])
            elif llm_r.get("correct") and not tg_r.get("correct"):
                llm_wins.append(tg_r["name"])

        summary["tg_wins_over_llm"] = tg_wins
        summary["llm_wins_over_tg"] = llm_wins

        print(f"\nTG wins (correct where LLM wrong): {tg_wins}")
        print(f"LLM wins (correct where TG wrong): {llm_wins}")

    out_path = os.path.join(os.path.dirname(__file__),
                            "deep_composition_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
