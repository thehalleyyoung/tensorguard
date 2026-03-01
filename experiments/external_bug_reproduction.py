"""
External Bug Reproduction Experiment for TensorGuard.

Reproduces 30 real-world PyTorch shape-mismatch patterns drawn from common
bug categories found in public repositories, StackOverflow, and PyTorch
forums.  Each benchmark is a minimal nn.Module that isolates the pattern.

Categories:
  1. Linear dimension mismatch (transfer learning head swap)
  2. Conv2d spatial dimension mismatch (pooling / stride errors)
  3. Embedding dimension mismatch
  4. Multi-head attention bugs (embed_dim % num_heads != 0)
  5. Concatenation bugs (wrong dim / mismatched non-cat dims)
  6. Transpose / permute bugs
  7. BatchNorm feature mismatch
  8. Skip-connection / residual addition bugs

Usage:
    cd implementation && python3 experiments/external_bug_reproduction.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import verify_model  # noqa: E402

# ── Benchmark registry ──────────────────────────────────────────────────

BENCHMARKS = []


def bench(name, source, input_shapes, is_buggy, category, description):
    BENCHMARKS.append({
        "name": name,
        "source": source,
        "input_shapes": input_shapes,
        "is_buggy": is_buggy,
        "category": category,
        "description": description,
    })


# =====================================================================
# Category 1: Linear dimension mismatch
# Pattern from PyTorch forum: transfer learning head mismatch
# =====================================================================

bench(
    "linear_mismatch_buggy",
    """
import torch
import torch.nn as nn

class TransferHead(nn.Module):
    # Pattern: user replaces final FC but forgets intermediate dim
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(512, 256)
        self.classifier = nn.Linear(128, 10)  # BUG: expects 128, gets 256

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.classifier(x)
""",
    {"x": ("batch", 512)},
    True,
    "linear_mismatch",
    "Pattern from PyTorch forum: transfer learning head replacement with wrong input dim",
)

bench(
    "linear_mismatch_safe",
    """
import torch
import torch.nn as nn

class TransferHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.features(x))
        return self.classifier(x)
""",
    {"x": ("batch", 512)},
    False,
    "linear_mismatch",
    "Corrected transfer learning head",
)

bench(
    "linear_chain_buggy",
    """
import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    # Pattern from GitHub: copy-paste error in deep MLP layers
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)  # BUG: should be 256 -> 128
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
""",
    {"x": ("batch", 784)},
    True,
    "linear_mismatch",
    "Pattern from GitHub issues: copy-paste error in MLP chain, fc3 input doesn't match fc2 output",
)

bench(
    "linear_chain_safe",
    """
import torch
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
""",
    {"x": ("batch", 784)},
    False,
    "linear_mismatch",
    "Corrected deep MLP chain",
)

# =====================================================================
# Category 2: Conv2d spatial dimension mismatch
# Pattern from StackOverflow: forgetting to account for pooling / stride
# =====================================================================

bench(
    "conv_spatial_buggy",
    """
import torch
import torch.nn as nn

class ConvClassifier(nn.Module):
    # Pattern from StackOverflow: wrong flatten size after conv+pool
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # After: 28->14->7, channels=64, so flatten=64*7*7=3136
        self.fc = nn.Linear(64 * 14 * 14, 10)  # BUG: uses pre-pool size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
    {"x": ("batch", 1, 28, 28)},
    True,
    "conv_spatial",
    "Pattern from StackOverflow: Linear input uses pre-pool spatial dims instead of post-pool",
)

bench(
    "conv_spatial_safe",
    """
import torch
import torch.nn as nn

class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
    {"x": ("batch", 1, 28, 28)},
    False,
    "conv_spatial",
    "Corrected conv classifier with correct flatten size",
)

bench(
    "conv_channel_buggy",
    """
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    # Pattern from GitHub: conv2 input channels don't match conv1 output
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)  # BUG: expects 16 channels, gets 32

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x
""",
    {"x": ("batch", 3, 32, 32)},
    True,
    "conv_spatial",
    "Pattern from GitHub: mismatched conv channel dimensions between layers",
)

bench(
    "conv_channel_safe",
    """
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x
""",
    {"x": ("batch", 3, 32, 32)},
    False,
    "conv_spatial",
    "Corrected conv channel chain",
)

# =====================================================================
# Category 3: Embedding dimension mismatch
# Pattern from StackOverflow: embedding dim vs. linear input
# =====================================================================

bench(
    "embedding_mismatch_buggy",
    """
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    # Pattern from StackOverflow: embedding_dim doesn't match downstream LSTM/Linear
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc = nn.Linear(64, 10)  # BUG: expects 64, embedding gives 128

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        return self.fc(x)
""",
    {"x": ("batch", 20)},
    True,
    "embedding_mismatch",
    "Pattern from StackOverflow: embedding dim doesn't match downstream linear layer input",
)

bench(
    "embedding_mismatch_safe",
    """
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        return self.fc(x)
""",
    {"x": ("batch", 20)},
    False,
    "embedding_mismatch",
    "Corrected text classifier",
)

bench(
    "embedding_lstm_buggy",
    """
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    # Pattern from PyTorch forum: LSTM hidden_size vs. final linear
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5000, 100)
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(128, 5)  # BUG: LSTM hidden=256, not 128

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
""",
    {"x": ("batch", 50)},
    True,
    "embedding_mismatch",
    "Pattern from PyTorch forum: LSTM hidden_size doesn't match subsequent Linear input",
)

bench(
    "embedding_lstm_safe",
    """
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5000, 100)
        self.lstm = nn.LSTM(100, 256, batch_first=True)
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
""",
    {"x": ("batch", 50)},
    False,
    "embedding_mismatch",
    "Corrected LSTM classifier",
)

# =====================================================================
# Category 4: Multi-head attention bugs
# Pattern from PyTorch issues: embed_dim not divisible by num_heads
# =====================================================================

bench(
    "mha_indivisible_buggy",
    """
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    # Pattern from PyTorch GitHub issues: embed_dim % num_heads != 0
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=100, num_heads=3)  # BUG: 100 % 3 != 0
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return self.fc(x)
""",
    {"x": ("seq", "batch", 100)},
    True,
    "multihead_attention",
    "Pattern from PyTorch GitHub issues: embed_dim=100 not divisible by num_heads=3",
)

bench(
    "mha_indivisible_safe",
    """
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=96, num_heads=3)
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        return self.fc(x)
""",
    {"x": ("seq", "batch", 96)},
    False,
    "multihead_attention",
    "Corrected attention block with embed_dim divisible by num_heads",
)

bench(
    "mha_proj_buggy",
    """
import torch
import torch.nn as nn

class TransformerMLP(nn.Module):
    # Pattern from HuggingFace issues: post-attention linear has wrong dim
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(512, 256)  # BUG: should be 1024 -> 256

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
""",
    {"x": ("seq", "batch", 256)},
    True,
    "multihead_attention",
    "Pattern from HuggingFace issues: FFN second layer input doesn't match first layer output",
)

bench(
    "mha_proj_safe",
    """
import torch
import torch.nn as nn

class TransformerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
""",
    {"x": ("seq", "batch", 256)},
    False,
    "multihead_attention",
    "Corrected transformer FFN",
)

# =====================================================================
# Category 5: Concatenation bugs
# Pattern from StackOverflow: cat along wrong dimension
# =====================================================================

bench(
    "cat_wrong_dim_buggy",
    """
import torch
import torch.nn as nn

class DualPath(nn.Module):
    # Pattern from StackOverflow: concatenating along wrong axis
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(64, 32)
        self.branch_b = nn.Linear(64, 48)
        self.merge = nn.Linear(80, 10)  # expects cat on dim=1 -> 32+48=80

    def forward(self, x):
        a = torch.relu(self.branch_a(x))
        b = torch.relu(self.branch_b(x))
        merged = torch.cat([a, b], dim=0)  # BUG: cat on batch dim, not feature dim
        return self.merge(merged)
""",
    {"x": ("batch", 64)},
    True,
    "concatenation",
    "Pattern from StackOverflow: torch.cat on dim=0 instead of dim=1 for feature concat",
)

bench(
    "cat_wrong_dim_safe",
    """
import torch
import torch.nn as nn

class DualPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(64, 32)
        self.branch_b = nn.Linear(64, 48)
        self.merge = nn.Linear(80, 10)

    def forward(self, x):
        a = torch.relu(self.branch_a(x))
        b = torch.relu(self.branch_b(x))
        merged = torch.cat([a, b], dim=1)
        return self.merge(merged)
""",
    {"x": ("batch", 64)},
    False,
    "concatenation",
    "Corrected dual-path model with correct concat dimension",
)

bench(
    "cat_size_buggy",
    """
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    # Pattern from GitHub: concat features with mismatched merge layer
    def __init__(self):
        super().__init__()
        self.path1 = nn.Linear(128, 64)
        self.path2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 10)  # BUG: should be 128 (64+64)

    def forward(self, x):
        f1 = torch.relu(self.path1(x))
        f2 = torch.relu(self.path2(x))
        fused = torch.cat([f1, f2], dim=1)
        return self.classifier(fused)
""",
    {"x": ("batch", 128)},
    True,
    "concatenation",
    "Pattern from GitHub: Linear after concat doesn't account for doubled feature dim",
)

bench(
    "cat_size_safe",
    """
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.path1 = nn.Linear(128, 64)
        self.path2 = nn.Linear(128, 64)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        f1 = torch.relu(self.path1(x))
        f2 = torch.relu(self.path2(x))
        fused = torch.cat([f1, f2], dim=1)
        return self.classifier(fused)
""",
    {"x": ("batch", 128)},
    False,
    "concatenation",
    "Corrected feature fusion with correct Linear input after cat",
)

# =====================================================================
# Category 6: Transpose / permute bugs
# Pattern from PyTorch forum: wrong dim ordering after transpose
# =====================================================================

bench(
    "transpose_buggy",
    """
import torch
import torch.nn as nn

class TransposeNet(nn.Module):
    # Pattern from PyTorch forum: transpose dims wrong before linear
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, 3, padding=1)
        self.fc = nn.Linear(16, 10)  # BUG: after transpose, feature dim is 32

    def forward(self, x):
        x = self.conv(x)          # (batch, 32, seq)
        x = x.transpose(1, 2)    # (batch, seq, 32)
        return self.fc(x)
""",
    {"x": ("batch", 16, 50)},
    True,
    "transpose_permute",
    "Pattern from PyTorch forum: linear input dim doesn't match transposed channel dim",
)

bench(
    "transpose_safe",
    """
import torch
import torch.nn as nn

class TransposeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.fc(x)
""",
    {"x": ("batch", 16, 50)},
    False,
    "transpose_permute",
    "Corrected transpose-then-linear model",
)

# =====================================================================
# Category 7: BatchNorm feature mismatch
# Pattern from StackOverflow: BN features don't match conv output channels
# =====================================================================

bench(
    "batchnorm_buggy",
    """
import torch
import torch.nn as nn

class ConvBN(nn.Module):
    # Pattern from StackOverflow: BatchNorm2d num_features != conv out_channels
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BUG: should be 64 to match conv1 output
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x
""",
    {"x": ("batch", 3, 32, 32)},
    True,
    "batchnorm_mismatch",
    "Pattern from StackOverflow: BatchNorm2d(32) but Conv2d output has 64 channels",
)

bench(
    "batchnorm_safe",
    """
import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x
""",
    {"x": ("batch", 3, 32, 32)},
    False,
    "batchnorm_mismatch",
    "Corrected Conv-BN chain",
)

bench(
    "batchnorm1d_buggy",
    """
import torch
import torch.nn as nn

class MLPBN(nn.Module):
    # Pattern from GitHub: BatchNorm1d features mismatch after Linear
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(256)  # BUG: should be 128
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return self.fc2(x)
""",
    {"x": ("batch", 256)},
    True,
    "batchnorm_mismatch",
    "Pattern from GitHub: BatchNorm1d features match fc1 input instead of output",
)

bench(
    "batchnorm1d_safe",
    """
import torch
import torch.nn as nn

class MLPBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return self.fc2(x)
""",
    {"x": ("batch", 256)},
    False,
    "batchnorm_mismatch",
    "Corrected MLP with BatchNorm1d",
)

# =====================================================================
# Category 8: Skip connection / residual addition bugs
# Pattern from deep learning repos: mismatched dims in residual add
# =====================================================================

bench(
    "skip_connection_buggy",
    """
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    # Pattern from GitHub: residual connection with dim mismatch
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)  # changes dim: 256 -> 128
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        residual = x               # shape: (batch, 256)
        x = torch.relu(self.fc1(x))  # shape: (batch, 128)
        x = self.fc2(x)              # shape: (batch, 128)
        return x + residual           # BUG: 128 + 256 dim mismatch
""",
    {"x": ("batch", 256)},
    True,
    "skip_connection",
    "Pattern from GitHub: residual add with mismatched feature dims (no projection)",
)

bench(
    "skip_connection_safe",
    """
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual
""",
    {"x": ("batch", 256)},
    False,
    "skip_connection",
    "Corrected residual block with matching dims",
)

bench(
    "skip_conv_buggy",
    """
import torch
import torch.nn as nn

class ConvResidual(nn.Module):
    # Pattern from ResNet implementations: skip connection without projection
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        residual = x                         # (batch, 64, H, W)
        x = torch.relu(self.bn1(self.conv1(x)))  # (batch, 128, H, W)
        x = self.bn2(self.conv2(x))               # (batch, 128, H, W)
        return torch.relu(x + residual)           # BUG: 128 vs 64 channels
""",
    {"x": ("batch", 64, 16, 16)},
    True,
    "skip_connection",
    "Pattern from ResNet: skip connection without 1x1 projection when channels change",
)

bench(
    "skip_conv_safe",
    """
import torch
import torch.nn as nn

class ConvResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)
""",
    {"x": ("batch", 64, 16, 16)},
    False,
    "skip_connection",
    "Corrected conv residual with matching channels",
)

# =====================================================================
# Additional linear mismatch: autoencoder pattern
# Pattern from StackOverflow: encoder/decoder dim mismatch
# =====================================================================

bench(
    "autoencoder_buggy",
    """
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    # Pattern from StackOverflow: decoder input doesn't match encoder bottleneck
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),   # BUG: should be 64, not 32
            nn.ReLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
    {"x": ("batch", 784)},
    True,
    "linear_mismatch",
    "Pattern from StackOverflow: autoencoder decoder input doesn't match encoder bottleneck dim",
)

bench(
    "autoencoder_safe",
    """
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
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
""",
    {"x": ("batch", 784)},
    False,
    "linear_mismatch",
    "Corrected autoencoder",
)

# =====================================================================
# Additional conv: GAN generator pattern
# Pattern from DCGAN tutorials: wrong ConvTranspose2d channel dims
# =====================================================================

bench(
    "gan_generator_buggy",
    """
import torch
import torch.nn as nn

class Generator(nn.Module):
    # Pattern from DCGAN tutorial bugs: ConvTranspose2d channel mismatch
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)  # BUG: expects 64, gets 128

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = torch.relu(self.deconv1(x))
        return torch.tanh(self.deconv2(x))
""",
    {"z": ("batch", 100)},
    True,
    "conv_spatial",
    "Pattern from DCGAN tutorials: ConvTranspose2d channel mismatch in generator",
)

bench(
    "gan_generator_safe",
    """
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = torch.relu(self.deconv1(x))
        return torch.tanh(self.deconv2(x))
""",
    {"z": ("batch", 100)},
    False,
    "conv_spatial",
    "Corrected DCGAN generator",
)


# ═════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════

def run_experiment():
    total = len(BENCHMARKS)
    buggy_count = sum(1 for b in BENCHMARKS if b["is_buggy"])
    safe_count = total - buggy_count

    print("=" * 72)
    print("External Bug Reproduction Experiment")
    print(f"  Total models : {total}")
    print(f"  Buggy        : {buggy_count}")
    print(f"  Safe         : {safe_count}")
    print("=" * 72)

    tp = fp = tn = fn = 0
    individual_results = []
    per_category = {}

    for b in BENCHMARKS:
        name = b["name"]
        source = b["source"]
        input_shapes = b["input_shapes"]
        is_buggy = b["is_buggy"]
        category = b["category"]

        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=input_shapes)
            detected = not result.safe
            error = None
        except Exception as exc:
            detected = False
            error = str(exc)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Classify
        if is_buggy and detected:
            verdict = "TP"; tp += 1
        elif is_buggy and not detected:
            verdict = "FN"; fn += 1
        elif not is_buggy and detected:
            verdict = "FP"; fp += 1
        else:
            verdict = "TN"; tn += 1

        ok = "✓" if verdict in ("TP", "TN") else "✗"
        print(f"  {ok} {verdict}  {name:<35s} [{category}]  {elapsed_ms:>7.1f}ms"
              + (f"  ERR: {error}" if error else ""))

        individual_results.append({
            "name": name,
            "category": category,
            "is_buggy": is_buggy,
            "detected": detected,
            "verdict": verdict,
            "time_ms": round(elapsed_ms, 1),
            "error": error,
            "description": b["description"],
        })

        # Per-category tracking
        if category not in per_category:
            per_category[category] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        per_category[category][verdict.lower()] += 1

    # Compute aggregate metrics
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    print()
    print("=" * 72)
    print("Results Summary")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print()

    # Per-category summary
    print("Per-category breakdown:")
    for cat, counts in sorted(per_category.items()):
        cat_tp = counts["tp"]; cat_fp = counts["fp"]
        cat_tn = counts["tn"]; cat_fn = counts["fn"]
        cat_total = cat_tp + cat_fp + cat_tn + cat_fn
        cat_correct = cat_tp + cat_tn
        print(f"  {cat:<25s}  {cat_correct}/{cat_total} correct  "
              f"(TP={cat_tp} FP={cat_fp} TN={cat_tn} FN={cat_fn})")

    # Build output
    output = {
        "total_models": total,
        "buggy_models": buggy_count,
        "safe_models": safe_count,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "per_category": per_category,
        "individual_results": individual_results,
    }

    out_path = Path(__file__).parent / "external_bug_reproduction_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    run_experiment()
