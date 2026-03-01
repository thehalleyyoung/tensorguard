"""
Cross-Solver Certificate Validation Experiment.

Verifies nn.Module models with TensorGuard's verify_model(), extracts
SMT-LIB 2.6 verification conditions, and cross-validates them with both
Z3 (subprocess) and CVC5 (subprocess or Python API).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import verify_model

RESULTS_FILE = Path(__file__).parent / "cross_solver_validation_results.json"

# Check solver availability
HAS_Z3_CLI = shutil.which("z3") is not None

HAS_CVC5_CLI = shutil.which("cvc5") is not None

HAS_CVC5_PY = False
try:
    import cvc5
    HAS_CVC5_PY = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Safe models to verify (10-15 models of increasing complexity)
# ---------------------------------------------------------------------------

SAFE_MODELS: List[Dict[str, Any]] = [
    {
        "name": "simple_linear",
        "description": "Single linear layer",
        "source": """\
import torch.nn as nn
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
    },
    {
        "name": "two_layer_mlp",
        "description": "Two-layer MLP",
        "source": """\
import torch.nn as nn
class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "three_layer_mlp",
        "description": "Three-layer MLP with ReLU",
        "source": """\
import torch.nn as nn
class ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "deep_mlp",
        "description": "Five-layer deep MLP",
        "source": """\
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "residual_block",
        "description": "Residual connection (skip connection)",
        "source": """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return h + residual
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "layernorm_block",
        "description": "LayerNorm + Linear",
        "source": """\
import torch.nn as nn
class LayerNormBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, 768)
    def forward(self, x):
        return self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "cnn_simple",
        "description": "Simple Conv2d + Linear classifier",
        "source": """\
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "two_conv_cnn",
        "description": "Two Conv2d layers + pooling + Linear",
        "source": """\
import torch.nn as nn
class TwoConvCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "transformer_ffn",
        "description": "Transformer feed-forward block",
        "source": """\
import torch.nn as nn
class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.norm(x)
        h = self.relu(self.fc1(h))
        return self.fc2(h) + x
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "bottleneck_mlp",
        "description": "Bottleneck MLP (wide -> narrow -> wide)",
        "source": """\
import torch.nn as nn
class BottleneckMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "sequential_chain",
        "description": "Long sequential chain of compatible layers",
        "source": """\
import torch.nn as nn
class SequentialChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 100)},
    },
    {
        "name": "classifier_head",
        "description": "Classifier head with dropout",
        "source": """\
import torch.nn as nn
class ClassifierHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 2048)},
    },
    # -----------------------------------------------------------------
    # 13-16: CNN variants
    # -----------------------------------------------------------------
    {
        "name": "conv2d_small_kernel",
        "description": "Conv2d with small 1x1 and 3x3 kernels",
        "source": """\
import torch.nn as nn
class Conv2dSmallKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "conv2d_strided",
        "description": "Conv2d with stride-2 downsampling",
        "source": """\
import torch.nn as nn
class Conv2dStrided(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "convtranspose2d_upsample",
        "description": "ConvTranspose2d for upsampling",
        "source": """\
import torch.nn as nn
class ConvTranspose2dUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.deconv1(x))
        h = self.relu(self.deconv2(h))
        return self.conv_out(h)
""",
        "input_shapes": {"x": ("batch", 64, 8, 8)},
    },
    {
        "name": "conv2d_deep",
        "description": "Deep Conv2d with 4 layers",
        "source": """\
import torch.nn as nn
class DeepConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # -----------------------------------------------------------------
    # 17-20: RNN variants
    # -----------------------------------------------------------------
    {
        "name": "lstm_classifier",
        "description": "LSTM for sequence classification",
        "source": """\
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", "seq", 64)},
    },
    {
        "name": "gru_classifier",
        "description": "GRU for sequence classification",
        "source": """\
import torch.nn as nn
class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 5)
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", "seq", 32)},
    },
    {
        "name": "bilstm_classifier",
        "description": "Bidirectional LSTM classifier",
        "source": """\
import torch.nn as nn
class BiLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(50, 100, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(200, 10)
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", "seq", 50)},
    },
    {
        "name": "stacked_gru",
        "description": "Stacked 2-layer GRU",
        "source": """\
import torch.nn as nn
class StackedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(40, 80, num_layers=2, batch_first=True)
        self.fc = nn.Linear(80, 10)
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", "seq", 40)},
    },
    # -----------------------------------------------------------------
    # 21-24: Transformer variants
    # -----------------------------------------------------------------
    {
        "name": "multihead_attention_block",
        "description": "MultiheadAttention with residual",
        "source": """\
import torch.nn as nn
class MHABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8)
        self.norm = nn.LayerNorm(256)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)
""",
        "input_shapes": {"x": ("seq", "batch", 256)},
    },
    {
        "name": "transformer_encoder_layer_model",
        "description": "TransformerEncoderLayer block",
        "source": """\
import torch.nn as nn
class TransEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    def forward(self, x):
        return self.encoder_layer(x)
""",
        "input_shapes": {"x": ("seq", "batch", 512)},
    },
    {
        "name": "transformer_ffn_gelu",
        "description": "Transformer FFN with GELU activation",
        "source": """\
import torch.nn as nn
class TransformerFFNGelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.gelu = nn.GELU()
    def forward(self, x):
        h = self.norm(x)
        h = self.gelu(self.fc1(h))
        return self.fc2(h) + x
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "pre_norm_transformer",
        "description": "Pre-norm transformer block with two sub-layers",
        "source": """\
import torch.nn as nn
class PreNormTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4)
        self.norm2 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h = self.norm2(x)
        h = self.relu(self.fc1(h))
        return x + self.fc2(h)
""",
        "input_shapes": {"x": ("seq", "batch", 256)},
    },
    # -----------------------------------------------------------------
    # 25-28: Normalization variants
    # -----------------------------------------------------------------
    {
        "name": "batchnorm1d_mlp",
        "description": "MLP with BatchNorm1d",
        "source": """\
import torch.nn as nn
class BN1dMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "batchnorm2d_conv",
        "description": "Conv2d with BatchNorm2d",
        "source": """\
import torch.nn as nn
class BN2dConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "groupnorm_conv",
        "description": "Conv2d with GroupNorm",
        "source": """\
import torch.nn as nn
class GroupNormConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.gn1(self.conv1(x)))
        h = self.relu(self.gn2(self.conv2(h)))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "instancenorm2d_conv",
        "description": "Conv2d with InstanceNorm2d",
        "source": """\
import torch.nn as nn
class InstanceNormConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.in1(self.conv1(x)))
        h = self.relu(self.in2(self.conv2(h)))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # -----------------------------------------------------------------
    # 29-32: Pooling variants
    # -----------------------------------------------------------------
    {
        "name": "maxpool_conv",
        "description": "Conv2d with MaxPool2d",
        "source": """\
import torch.nn as nn
class MaxPoolConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.pool1(self.relu(self.conv1(x)))
        h = self.pool2(self.relu(self.conv2(h)))
        h = self.pool_final(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "avgpool_conv",
        "description": "Conv2d with AvgPool2d",
        "source": """\
import torch.nn as nn
class AvgPoolConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.pool1(self.relu(self.conv1(x)))
        h = self.relu(self.conv2(h))
        h = self.pool_final(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "adaptive_avgpool_mlp",
        "description": "AdaptiveAvgPool2d into MLP head",
        "source": """\
import torch.nn as nn
class AdaptivePoolMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv(x))
        h = self.pool(h)
        h = h.flatten(1)
        h = self.relu(self.fc1(h))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "global_avgpool_classifier",
        "description": "Global average pooling classifier",
        "source": """\
import torch.nn as nn
class GlobalAvgPoolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    # -----------------------------------------------------------------
    # 33-36: Skip connection / residual variants
    # -----------------------------------------------------------------
    {
        "name": "double_residual",
        "description": "Two stacked residual blocks",
        "source": """\
import torch.nn as nn
class DoubleResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        x = x + h
        h = self.relu(self.fc3(x))
        h = self.fc4(h)
        return x + h
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "conv_residual_block",
        "description": "Conv2d residual block (same padding)",
        "source": """\
import torch.nn as nn
class ConvResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 64, 16, 16)},
    },
    {
        "name": "pre_activation_resblock",
        "description": "Pre-activation residual block",
        "source": """\
import torch.nn as nn
class PreActResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.conv1(self.relu(self.bn1(x)))
        h = self.conv2(self.relu(self.bn2(h)))
        return h + x
""",
        "input_shapes": {"x": ("batch", 32, 16, 16)},
    },
    {
        "name": "residual_mlp_layernorm",
        "description": "Residual MLP with LayerNorm",
        "source": """\
import torch.nn as nn
class ResidualMLPLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(self.norm1(x)))
        x = x + self.fc2(h)
        h = self.relu(self.fc3(self.norm2(x)))
        return x + h
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # -----------------------------------------------------------------
    # 37-39: Embedding models
    # -----------------------------------------------------------------
    {
        "name": "embedding_classifier",
        "description": "Embedding + Linear for classification",
        "source": """\
import torch.nn as nn
class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.embed(x)
        h = self.relu(self.fc1(h))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", "seq")},
    },
    {
        "name": "embedding_with_layernorm",
        "description": "Embedding + LayerNorm + MLP",
        "source": """\
import torch.nn as nn
class EmbeddingLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5000, 128)
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.norm(self.embed(x))
        h = self.relu(self.fc1(h))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", "seq")},
    },
    {
        "name": "embedding_large_vocab",
        "description": "Large vocabulary embedding model",
        "source": """\
import torch.nn as nn
class LargeVocabEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30000, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.embed(x)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", "seq")},
    },
    # -----------------------------------------------------------------
    # 40-43: Autoencoder variants
    # -----------------------------------------------------------------
    {
        "name": "linear_autoencoder",
        "description": "Linear autoencoder with bottleneck",
        "source": """\
import torch.nn as nn
class LinearAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        h = self.relu(self.dec1(h))
        return self.dec2(h)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "deep_autoencoder",
        "description": "Deep autoencoder with symmetric layers",
        "source": """\
import torch.nn as nn
class DeepAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(512, 256)
        self.enc2 = nn.Linear(256, 128)
        self.enc3 = nn.Linear(128, 32)
        self.dec1 = nn.Linear(32, 128)
        self.dec2 = nn.Linear(128, 256)
        self.dec3 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        h = self.relu(self.enc3(h))
        h = self.relu(self.dec1(h))
        h = self.relu(self.dec2(h))
        return self.dec3(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "conv_autoencoder",
        "description": "Convolutional autoencoder",
        "source": """\
import torch.nn as nn
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc_conv1(x))
        h = self.relu(self.enc_conv2(h))
        h = self.relu(self.dec_conv1(h))
        return self.dec_conv2(h)
""",
        "input_shapes": {"x": ("batch", 1, 28, 28)},
    },
    {
        "name": "variational_encoder",
        "description": "Encoder portion of a VAE",
        "source": """\
import torch.nn as nn
class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    # -----------------------------------------------------------------
    # 44-47: Multi-branch / parallel path models
    # -----------------------------------------------------------------
    {
        "name": "dual_branch_add",
        "description": "Two branches merged by addition",
        "source": """\
import torch.nn as nn
class DualBranchAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1_fc = nn.Linear(128, 64)
        self.branch2_fc = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        b1 = self.relu(self.branch1_fc(x))
        b2 = self.relu(self.branch2_fc(x))
        h = b1 + b2
        return self.fc_out(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dual_branch_concat",
        "description": "Two branches merged by concatenation",
        "source": """\
import torch
import torch.nn as nn
class DualBranchConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 32)
        self.fc_out = nn.Linear(96, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        h = torch.cat([b1, b2], dim=1)
        return self.fc_out(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "triple_branch_add",
        "description": "Three parallel branches merged by addition",
        "source": """\
import torch.nn as nn
class TripleBranchAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.relu(self.fc_a(x))
        b = self.relu(self.fc_b(x))
        c = self.relu(self.fc_c(x))
        h = a + b + c
        return self.fc_out(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "wide_narrow_merge",
        "description": "Wide and narrow paths merged by addition",
        "source": """\
import torch.nn as nn
class WideNarrowMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(256, 128)
        self.narrow_1 = nn.Linear(256, 64)
        self.narrow_2 = nn.Linear(64, 128)
        self.fc_out = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        w = self.relu(self.wide(x))
        n = self.relu(self.narrow_1(x))
        n = self.relu(self.narrow_2(n))
        h = w + n
        return self.fc_out(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # -----------------------------------------------------------------
    # 48-50: Miscellaneous architectures
    # -----------------------------------------------------------------
    {
        "name": "mlp_with_softmax",
        "description": "MLP ending with Softmax",
        "source": """\
import torch.nn as nn
class MLPSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.softmax(self.fc2(h))
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "identity_shortcut",
        "description": "Identity layer with shortcut path",
        "source": """\
import torch.nn as nn
class IdentityShortcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 200)
        self.identity = nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        shortcut = self.identity(x)
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return h + shortcut
""",
        "input_shapes": {"x": ("batch", 200)},
    },
    {
        "name": "tanh_sigmoid_mlp",
        "description": "MLP with Tanh and Sigmoid activations",
        "source": """\
import torch.nn as nn
class TanhSigmoidMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.tanh(self.fc1(x))
        h = self.tanh(self.fc2(h))
        return self.sigmoid(self.fc3(h))
""",
        "input_shapes": {"x": ("batch", 100)},
    },
]


# ---------------------------------------------------------------------------
# Solver validation helpers
# ---------------------------------------------------------------------------

def validate_with_z3_subprocess(smtlib: str) -> Tuple[str, float]:
    """Validate an SMT-LIB certificate via `z3 -smt2` subprocess."""
    if not HAS_Z3_CLI:
        return "UNAVAILABLE", 0.0
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f:
        f.write(smtlib)
        tmp_path = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            ["z3", "-smt2", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        output = proc.stdout.strip().lower()
        if "unsat" in output:
            return "unsat", elapsed_ms
        elif "sat" in output:
            return "sat", elapsed_ms
        else:
            return f"error: {proc.stderr.strip()[:100]}" if proc.stderr.strip() else f"unknown: {output[:100]}", elapsed_ms
    except subprocess.TimeoutExpired:
        return "timeout", 0.0
    finally:
        os.unlink(tmp_path)


def validate_with_cvc5_subprocess(smtlib: str) -> Tuple[str, float]:
    """Validate an SMT-LIB certificate via `cvc5` subprocess."""
    if not HAS_CVC5_CLI:
        return "UNAVAILABLE", 0.0
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False
    ) as f:
        f.write(smtlib)
        tmp_path = f.name
    try:
        t0 = time.monotonic()
        proc = subprocess.run(
            ["cvc5", "--lang", "smt2", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        output = proc.stdout.strip().lower()
        if "unsat" in output:
            return "unsat", elapsed_ms
        elif "sat" in output:
            return "sat", elapsed_ms
        else:
            return f"error: {proc.stderr.strip()[:100]}" if proc.stderr.strip() else f"unknown: {output[:100]}", elapsed_ms
    except subprocess.TimeoutExpired:
        return "timeout", 0.0
    finally:
        os.unlink(tmp_path)


def validate_with_cvc5_python(smtlib: str) -> Tuple[str, float]:
    """Validate an SMT-LIB certificate via cvc5 Python API."""
    if not HAS_CVC5_PY:
        return "UNAVAILABLE", 0.0
    try:
        t0 = time.monotonic()
        solver = cvc5.Solver()
        parser = cvc5.InputParser(solver)
        parser.setStringInput(
            cvc5.InputLanguage.SMT_LIB_2_6, smtlib, "certificate"
        )
        sm = parser.getSymbolManager()
        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            cmd.invoke(solver, sm)
        r = solver.checkSat()
        elapsed_ms = (time.monotonic() - t0) * 1000
        if r.isUnsat():
            return "unsat", elapsed_ms
        elif r.isSat():
            return "sat", elapsed_ms
        else:
            return "unknown", elapsed_ms
    except Exception as e:
        return f"error: {str(e)[:100]}", 0.0


def validate_smtlib_syntax(smtlib: str) -> bool:
    """Check basic SMT-LIB syntax (balanced parens, required commands)."""
    depth = 0
    in_comment = False
    for ch in smtlib:
        if ch == ";":
            in_comment = True
        elif ch == "\n":
            in_comment = False
        elif not in_comment:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
    if depth != 0:
        return False
    if "(set-logic" not in smtlib:
        return False
    if "(check-sat)" not in smtlib:
        return False
    return True


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> Dict[str, Any]:
    print("=" * 72)
    print("  Cross-Solver Certificate Validation Experiment")
    print("=" * 72)
    print(f"  Z3 CLI available:    {HAS_Z3_CLI}")
    print(f"  CVC5 CLI available:  {HAS_CVC5_CLI}")
    print(f"  CVC5 Python API:     {HAS_CVC5_PY}")
    has_cvc5 = HAS_CVC5_CLI or HAS_CVC5_PY
    if not has_cvc5:
        print("  NOTE: CVC5 not available. Will use Z3 subprocess as")
        print("        secondary validator to demonstrate infrastructure.")
    print()

    results: List[Dict[str, Any]] = []
    safe_count = 0
    z3_validated = 0
    cvc5_validated = 0
    secondary_validated = 0

    for model in SAFE_MODELS:
        name = model["name"]
        desc = model["description"]
        source = model["source"]
        input_shapes = model["input_shapes"]

        print(f"  [{name}] {desc}")

        # Step 1: Verify with TensorGuard
        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=input_shapes)
        except Exception as e:
            print(f"    ERROR during verify_model: {e}")
            results.append({
                "model": name,
                "description": desc,
                "tensorguard_safe": None,
                "error": str(e),
            })
            continue
        verify_time = (time.monotonic() - t0) * 1000

        entry: Dict[str, Any] = {
            "model": name,
            "description": desc,
            "tensorguard_safe": result.safe,
            "tensorguard_time_ms": round(verify_time, 2),
        }

        if not result.safe:
            print(f"    TensorGuard: UNSAFE (skipping certificate extraction)")
            if result.counterexample:
                entry["skip_reason"] = "model verified as unsafe"
            results.append(entry)
            continue

        safe_count += 1

        # Step 2: Extract SMT-LIB certificate
        cert = result.certificate
        if cert is None:
            print(f"    TensorGuard: SAFE but no certificate object")
            entry["skip_reason"] = "no certificate object"
            results.append(entry)
            continue

        smtlib = cert.smtlib_certificate()
        cert_size = len(smtlib)

        entry["certificate_size_bytes"] = cert_size
        entry["certificate_properties"] = cert.properties
        entry["certificate_k"] = cert.k
        entry["certificate_steps"] = cert.checked_steps
        entry["certificate_z3_queries"] = cert.z3_queries
        entry["certificate_theories"] = cert.theories_used

        # Step 3: Syntax check
        syntax_ok = validate_smtlib_syntax(smtlib)
        entry["smtlib_syntax_valid"] = syntax_ok
        if not syntax_ok:
            print(f"    SMT-LIB syntax: INVALID")
            results.append(entry)
            continue

        print(f"    TensorGuard: SAFE | cert={cert_size}B | "
              f"k={cert.k} steps={cert.checked_steps}")

        # Step 4: Z3 subprocess validation (secondary/independent validator)
        z3_result, z3_time = validate_with_z3_subprocess(smtlib)
        entry["z3_subprocess_result"] = z3_result
        entry["z3_subprocess_time_ms"] = round(z3_time, 2)
        z3_ok = z3_result == "unsat"
        if z3_ok:
            z3_validated += 1
            secondary_validated += 1
        status_icon = "\u2713" if z3_ok else ("\u2717" if z3_result != "UNAVAILABLE" else "-")
        print(f"    Z3 subprocess:  {z3_result:>8s}  ({z3_time:.1f}ms) {status_icon}")

        # Step 5: CVC5 validation
        if HAS_CVC5_CLI:
            cvc5_result, cvc5_time = validate_with_cvc5_subprocess(smtlib)
        elif HAS_CVC5_PY:
            cvc5_result, cvc5_time = validate_with_cvc5_python(smtlib)
        else:
            cvc5_result, cvc5_time = "UNAVAILABLE", 0.0

        entry["cvc5_result"] = cvc5_result
        entry["cvc5_time_ms"] = round(cvc5_time, 2)
        cvc5_ok = cvc5_result == "unsat"
        if cvc5_ok:
            cvc5_validated += 1
        status_icon = "\u2713" if cvc5_ok else ("\u2717" if cvc5_result != "UNAVAILABLE" else "-")
        print(f"    CVC5:           {cvc5_result:>8s}  ({cvc5_time:.1f}ms) {status_icon}")

        # Cross-solver agreement
        if z3_result not in ("UNAVAILABLE",) and cvc5_result not in ("UNAVAILABLE",):
            entry["cross_solver_agreement"] = z3_result == cvc5_result
        else:
            entry["cross_solver_agreement"] = None

        results.append(entry)
        print()

    # Summary
    total = len(SAFE_MODELS)
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Models tested:               {total}")
    print(f"  TensorGuard SAFE:            {safe_count}/{total}")
    print(f"  Z3 subprocess validated:     {z3_validated}/{safe_count}")
    if has_cvc5:
        print(f"  CVC5 validated:              {cvc5_validated}/{safe_count}")
        agree = sum(1 for r in results if r.get("cross_solver_agreement") is True)
        print(f"  Cross-solver agreement:      {agree}/{safe_count}")
    else:
        print(f"  CVC5:                        not available")
        print(f"  Z3 subprocess re-validation: {secondary_validated}/{safe_count}")
        print()
        print("  LIMITATION: CVC5 is not installed. Certificates were")
        print("  re-validated by Z3 subprocess (independent process) to")
        print("  demonstrate the cross-validation infrastructure. For full")
        print("  trust-minimized validation, install CVC5:")
        print("    brew install cvc5  OR  pip install cvc5")
    print("=" * 72)

    output = {
        "experiment": "cross_solver_certificate_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solvers": {
            "z3_cli": HAS_Z3_CLI,
            "cvc5_cli": HAS_CVC5_CLI,
            "cvc5_python": HAS_CVC5_PY,
        },
        "summary": {
            "models_tested": total,
            "tensorguard_safe": safe_count,
            "z3_subprocess_validated": z3_validated,
            "cvc5_validated": cvc5_validated,
            "secondary_validated": secondary_validated,
            "cvc5_available": has_cvc5,
        },
        "results": results,
    }
    if not has_cvc5:
        output["limitation"] = (
            "CVC5 not available. Certificates were re-validated by Z3 "
            "subprocess as a secondary independent validator. Install CVC5 "
            "for full cross-solver validation."
        )

    return output


if __name__ == "__main__":
    output = run_experiment()

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")
