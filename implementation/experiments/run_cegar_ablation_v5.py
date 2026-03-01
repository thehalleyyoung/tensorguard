"""
CEGAR Ablation Experiment v5 — Symbolic-Dimension Contract Discovery.

Addresses two flaws in prior ablations:
  - Old sensitivity analysis (n=8): all configs yield identical 0.875 accuracy
    because conv_mismatch is always a FN; statistically underpowered.
  - v3 ablation (n=10): uses concrete dims so CEGAR discovers 0 predicates
    and all three modes produce identical results.

This experiment uses SYMBOLIC input dimensions (strings) to force CEGAR
contract discovery.  With symbolic dims, the verifier cannot determine
input shapes a priori; CEGAR must iteratively discover predicates like
``x.shape[-1] == 768`` to prove safety or expose real bugs.

Compares three verification modes across 32 nn.Module architectures:
  (a) Single-pass         — max_iterations=1 (no refinement)
  (b) Unfiltered CEGAR    — enable_quality_filter=False
  (c) Quality-filtered CEGAR — enable_quality_filter=True

Outputs: experiments/cegar_ablation_v5_results.json
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import run_shape_cegar, ShapeCEGARResult, CEGARStatus

RESULTS_FILE = Path(__file__).parent / "cegar_ablation_v5_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark definitions — 32 nn.Module architectures with SYMBOLIC input dims
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ── MLP architectures ─────────────────────────────────────────────────
    {
        "name": "mlp_correct_2layer",
        "arch": "MLP",
        "has_bug": False,
        "description": "2-layer MLP, correct; CEGAR must discover features==768",
        "code": """\
import torch.nn as nn
class CorrectMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "mlp_buggy_2layer",
        "arch": "MLP",
        "has_bug": True,
        "description": "2-layer MLP: fc1 out=256 but fc2 in=128",
        "code": """\
import torch.nn as nn
class BuggyMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "mlp_correct_3layer",
        "arch": "MLP",
        "has_bug": False,
        "description": "3-layer MLP, correct; needs multiple refinements",
        "code": """\
import torch.nn as nn
class CorrectMLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "mlp_buggy_3layer",
        "arch": "MLP",
        "has_bug": True,
        "description": "3-layer MLP: fc2 out=128 but fc3 in=64",
        "code": """\
import torch.nn as nn
class BuggyMLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "mlp_correct_4layer",
        "arch": "MLP",
        "has_bug": False,
        "description": "4-layer deep MLP, correct; needs 3+ refinements",
        "code": """\
import torch.nn as nn
class DeepMLP4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
""",
        "input_shapes": {"x": ("batch", "d_in")},
    },
    {
        "name": "mlp_buggy_4layer",
        "arch": "MLP",
        "has_bug": True,
        "description": "4-layer MLP: fc3 out=128 but fc4 in=256",
        "code": """\
import torch.nn as nn
class BuggyDeepMLP4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
""",
        "input_shapes": {"x": ("batch", "d_in")},
    },

    # ── CNN architectures ─────────────────────────────────────────────────
    {
        "name": "cnn_correct_2conv",
        "arch": "CNN",
        "has_bug": False,
        "description": "2-conv CNN, correct; channel dims match",
        "code": """\
import torch.nn as nn
class CorrectCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.relu(self.conv2(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "cnn_buggy_channel",
        "arch": "CNN",
        "has_bug": True,
        "description": "2-conv CNN: conv1 out=64 but conv2 in=32",
        "code": """\
import torch.nn as nn
class BuggyCNNChan(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.relu(self.conv2(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "cnn_correct_3conv",
        "arch": "CNN",
        "has_bug": False,
        "description": "3-conv CNN, correct; deeper pipeline",
        "code": """\
import torch.nn as nn
class CorrectCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },
    {
        "name": "cnn_buggy_deep",
        "arch": "CNN",
        "has_bug": True,
        "description": "3-conv CNN: conv2 out=64 but conv3 in=128",
        "code": """\
import torch.nn as nn
class BuggyCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))
""",
        "input_shapes": {"x": ("batch", 3, "h", "w")},
    },

    # ── Transformer-style architectures ───────────────────────────────────
    {
        "name": "transformer_proj_correct",
        "arch": "Transformer",
        "has_bug": False,
        "description": "Transformer Q/K/V projections, correct",
        "code": """\
import torch.nn as nn
class CorrectTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 64)
        self.k_proj = nn.Linear(512, 64)
        self.v_proj = nn.Linear(512, 64)
        self.out_proj = nn.Linear(64, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "transformer_proj_buggy",
        "arch": "Transformer",
        "has_bug": True,
        "description": "Transformer: Q outputs 64 but K outputs 128; addition fails",
        "code": """\
import torch.nn as nn
class BuggyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 64)
        self.k_proj = nn.Linear(512, 128)
        self.out_proj = nn.Linear(64, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        return self.out_proj(q + k)
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "transformer_ffn_correct",
        "arch": "Transformer",
        "has_bug": False,
        "description": "Transformer FFN block, correct; expand then contract",
        "code": """\
import torch.nn as nn
class CorrectFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },
    {
        "name": "transformer_ffn_buggy",
        "arch": "Transformer",
        "has_bug": True,
        "description": "FFN: linear1 out=2048 but linear2 in=1024",
        "code": """\
import torch.nn as nn
class BuggyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", "d_model")},
    },

    # ── Autoencoder architectures ─────────────────────────────────────────
    {
        "name": "autoencoder_correct",
        "arch": "Autoencoder",
        "has_bug": False,
        "description": "Symmetric autoencoder, correct",
        "code": """\
import torch.nn as nn
class CorrectAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "autoencoder_buggy",
        "arch": "Autoencoder",
        "has_bug": True,
        "description": "Autoencoder: enc2 out=64 but dec1 in=128",
        "code": """\
import torch.nn as nn
class BuggyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(128, 256)
        self.dec2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "autoencoder_deep_correct",
        "arch": "Autoencoder",
        "has_bug": False,
        "description": "Deep autoencoder 5-layer, correct",
        "code": """\
import torch.nn as nn
class DeepAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(1024, 512)
        self.enc2 = nn.Linear(512, 128)
        self.dec1 = nn.Linear(128, 512)
        self.dec2 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", "d")},
    },

    # ── ResNet-style skip connection architectures ────────────────────────
    {
        "name": "resnet_skip_correct",
        "arch": "ResNet-skip",
        "has_bug": False,
        "description": "ResNet-style skip block, correct; out+input same dim",
        "code": """\
import torch.nn as nn
class CorrectResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "resnet_skip_buggy",
        "arch": "ResNet-skip",
        "has_bug": True,
        "description": "ResNet skip: fc2 out=128 but skip expects 256; add fails",
        "code": """\
import torch.nn as nn
class BuggyResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x + residual
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "resnet_deep_correct",
        "arch": "ResNet-skip",
        "has_bug": False,
        "description": "Two stacked residual blocks, correct",
        "code": """\
import torch.nn as nn
class DeepResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        r = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x) + r
        r2 = x
        x = self.relu(self.fc3(x))
        return x + r2
""",
        "input_shapes": {"x": ("batch", "d")},
    },

    # ── U-Net-style architectures ─────────────────────────────────────────
    {
        "name": "unet_style_correct",
        "arch": "U-Net-style",
        "has_bug": False,
        "description": "U-Net-style with encoder/decoder + skip, correct",
        "code": """\
import torch.nn as nn
class CorrectUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Linear(256, 128)
        self.down2 = nn.Linear(128, 64)
        self.up1 = nn.Linear(64, 128)
        self.up2 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        u1 = self.relu(self.up1(d2))
        return self.up2(u1)
""",
        "input_shapes": {"x": ("batch", "features")},
    },
    {
        "name": "unet_style_buggy",
        "arch": "U-Net-style",
        "has_bug": True,
        "description": "U-Net-style: up1 out=128 but up2 in=64",
        "code": """\
import torch.nn as nn
class BuggyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Linear(256, 128)
        self.down2 = nn.Linear(128, 64)
        self.up1 = nn.Linear(64, 128)
        self.up2 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        u1 = self.relu(self.up1(d2))
        return self.up2(u1)
""",
        "input_shapes": {"x": ("batch", "features")},
    },

    # ── GAN discriminator architectures ───────────────────────────────────
    {
        "name": "gan_disc_correct",
        "arch": "GAN-discriminator",
        "has_bug": False,
        "description": "GAN discriminator, correct; shrinking pipeline to scalar",
        "code": """\
import torch.nn as nn
class CorrectDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "img_flat")},
    },
    {
        "name": "gan_disc_buggy",
        "arch": "GAN-discriminator",
        "has_bug": True,
        "description": "GAN disc: fc2 out=256 but fc3 in=512",
        "code": """\
import torch.nn as nn
class BuggyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "img_flat")},
    },

    # ── Classifier head architectures ─────────────────────────────────────
    {
        "name": "classifier_correct",
        "arch": "Classifier",
        "has_bug": False,
        "description": "Simple classifier head, correct",
        "code": """\
import torch.nn as nn
class CorrectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2048, 512)
        self.classifier = nn.Linear(512, 1000)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.features(x))
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", "feat_dim")},
    },
    {
        "name": "classifier_buggy",
        "arch": "Classifier",
        "has_bug": True,
        "description": "Classifier: features out=512 but classifier in=256",
        "code": """\
import torch.nn as nn
class BuggyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2048, 512)
        self.classifier = nn.Linear(256, 1000)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.features(x))
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", "feat_dim")},
    },

    # ── Bottleneck architectures ──────────────────────────────────────────
    {
        "name": "bottleneck_correct",
        "arch": "Bottleneck",
        "has_bug": False,
        "description": "Bottleneck (compress-expand), correct",
        "code": """\
import torch.nn as nn
class CorrectBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(1024, 128)
        self.expand = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.expand(self.relu(self.compress(x)))
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "bottleneck_buggy",
        "arch": "Bottleneck",
        "has_bug": True,
        "description": "Bottleneck: compress out=128 but expand in=64",
        "code": """\
import torch.nn as nn
class BuggyBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(1024, 128)
        self.expand = nn.Linear(64, 1024)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.expand(self.relu(self.compress(x)))
""",
        "input_shapes": {"x": ("batch", "d")},
    },

    # ── Multi-head projection architectures ───────────────────────────────
    {
        "name": "multihead_correct",
        "arch": "MultiHead",
        "has_bug": False,
        "description": "Multi-head projection with matching output dims",
        "code": """\
import torch.nn as nn
class CorrectMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(512, 64)
        self.head2 = nn.Linear(512, 64)
        self.merge = nn.Linear(64, 256)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        return self.merge(h1 + h2)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },
    {
        "name": "multihead_buggy",
        "arch": "MultiHead",
        "has_bug": True,
        "description": "Multi-head: head1 out=64, head2 out=128; add fails",
        "code": """\
import torch.nn as nn
class BuggyMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head1 = nn.Linear(512, 64)
        self.head2 = nn.Linear(512, 128)
        self.merge = nn.Linear(64, 256)
    def forward(self, x):
        h1 = self.head1(x)
        h2 = self.head2(x)
        return self.merge(h1 + h2)
""",
        "input_shapes": {"x": ("batch", "seq", "d")},
    },

    # ── Wide architectures ────────────────────────────────────────────────
    {
        "name": "wide_correct",
        "arch": "Wide",
        "has_bug": False,
        "description": "Wide network (large hidden), correct",
        "code": """\
import torch.nn as nn
class CorrectWide(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "d")},
    },
    {
        "name": "wide_buggy",
        "arch": "Wide",
        "has_bug": True,
        "description": "Wide network: fc1 out=4096 but fc2 in=2048",
        "code": """\
import torch.nn as nn
class BuggyWide(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "d")},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute precision, recall, F1, and confusion counts."""
    tp = fp = fn = tn = 0
    for r in results:
        has_bug = r["has_bug"]
        detected = r["detected_bug"]
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% CIs for precision, recall, and F1 (10k resamples)."""
    rng = random.Random(42)
    samples: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstrap):
        sample = rng.choices(results, k=len(results))
        m = compute_metrics(sample)
        for key in samples:
            samples[key].append(m[key])
    cis: Dict[str, Tuple[float, float]] = {}
    alpha = (1 - ci) / 2
    for key, vals in samples.items():
        vals.sort()
        lo = vals[int(alpha * len(vals))]
        hi = vals[min(int((1 - alpha) * len(vals)), len(vals) - 1)]
        cis[key] = (round(lo, 4), round(hi, 4))
    return cis


# ═══════════════════════════════════════════════════════════════════════════════
# Mode runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_pass(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Mode (a): single-pass — max_iterations=1, no refinement."""
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=1,
            enable_quality_filter=True,
        )
        detected = result.has_real_bugs
        status = result.final_status.name
        n_preds = len(result.discovered_predicates)
        n_iters = result.iterations
        qr = result.predicate_quality_report
        n_rejected = qr.get("rejected", 0) if qr else 0
    except Exception as e:
        detected = False
        status = f"ERROR: {e}"
        n_preds = 0
        n_iters = 1
        n_rejected = 0
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "arch": tc["arch"],
        "has_bug": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "iterations": n_iters,
        "predicates_discovered": n_preds,
        "predicates_rejected": n_rejected,
        "time_ms": round(elapsed, 2),
    }


def run_cegar_mode(tc: Dict[str, Any], enable_quality_filter: bool) -> Dict[str, Any]:
    """Mode (b)/(c): CEGAR loop with quality filter on or off."""
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=enable_quality_filter,
        )
        detected = result.has_real_bugs
        status = result.final_status.name
        n_preds = len(result.discovered_predicates)
        n_iters = result.iterations
        qr = result.predicate_quality_report
        n_rejected = qr.get("rejected", 0) if qr else 0
    except Exception as e:
        detected = False
        status = f"ERROR: {e}"
        n_preds = 0
        n_iters = 0
        n_rejected = 0
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "arch": tc["arch"],
        "has_bug": tc["has_bug"],
        "detected_bug": detected,
        "status": status,
        "iterations": n_iters,
        "predicates_discovered": n_preds,
        "predicates_rejected": n_rejected,
        "time_ms": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    n = len(TEST_CASES)
    n_buggy = sum(1 for tc in TEST_CASES if tc["has_bug"])
    n_safe = n - n_buggy
    archs = sorted(set(tc["arch"] for tc in TEST_CASES))

    print("=" * 78)
    print("  CEGAR Ablation Experiment v5 — Symbolic-Dimension Contract Discovery")
    print(f"  {n} benchmarks ({n_buggy} buggy, {n_safe} correct) × 3 modes")
    print(f"  Architectures: {', '.join(archs)}")
    print("=" * 78)

    modes = [
        ("single_pass", "Single-pass (max_iter=1, no refinement)"),
        ("cegar_unfiltered", "CEGAR — quality filter OFF"),
        ("cegar_filtered", "CEGAR — quality filter ON"),
    ]

    all_configs: Dict[str, Any] = {}

    for mode_key, mode_label in modes:
        print(f"\n{'─' * 78}")
        print(f"  Mode: {mode_label}")
        print(f"{'─' * 78}")

        per_bench: List[Dict[str, Any]] = []
        for tc in TEST_CASES:
            try:
                if mode_key == "single_pass":
                    r = run_single_pass(tc)
                elif mode_key == "cegar_unfiltered":
                    r = run_cegar_mode(tc, enable_quality_filter=False)
                else:
                    r = run_cegar_mode(tc, enable_quality_filter=True)
            except Exception as exc:
                r = {
                    "name": tc["name"],
                    "arch": tc["arch"],
                    "has_bug": tc["has_bug"],
                    "detected_bug": False,
                    "status": f"ERROR: {exc}",
                    "iterations": 0,
                    "predicates_discovered": 0,
                    "predicates_rejected": 0,
                    "time_ms": 0.0,
                }

            correct = r["detected_bug"] == r["has_bug"]
            mark = "✓" if correct else "✗"
            label = "BUG" if r["has_bug"] else "SAFE"
            det_label = "det" if r["detected_bug"] else "safe"
            print(
                f"  {mark} {r['name']:30s}  "
                f"expected={label:<4s}  result={det_label:<4s}  "
                f"iters={r['iterations']}  preds={r['predicates_discovered']}  "
                f"rej={r['predicates_rejected']}  {r['time_ms']:.0f}ms"
            )
            per_bench.append(r)

        metrics = compute_metrics(per_bench)
        cis = bootstrap_ci(per_bench)
        total_preds = sum(r["predicates_discovered"] for r in per_bench)
        total_rej = sum(r["predicates_rejected"] for r in per_bench)
        total_iters = sum(r["iterations"] for r in per_bench)
        total_time = sum(r["time_ms"] for r in per_bench)

        print(
            f"\n  Aggregate: F1={metrics['f1']}  "
            f"Precision={metrics['precision']}  Recall={metrics['recall']}"
        )
        print(
            f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  "
            f"FN={metrics['fn']}  TN={metrics['tn']}"
        )
        print(
            f"  Totals:    predicates={total_preds}  rejected={total_rej}  "
            f"iterations={total_iters}  time={total_time:.0f}ms"
        )
        print(f"  95% CI:    F1={cis['f1']}  P={cis['precision']}  R={cis['recall']}")

        all_configs[mode_key] = {
            "label": mode_label,
            "metrics": metrics,
            "confidence_intervals_95_10k": {
                k: list(v) for k, v in cis.items()
            },
            "total_predicates": total_preds,
            "total_rejected": total_rej,
            "total_iterations": total_iters,
            "total_time_ms": round(total_time, 2),
            "per_benchmark": per_bench,
        }

    # ── Summary comparison table ────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 78}")
    header = (
        f"  {'Configuration':<45s}  {'F1':<7s}  {'Prec':<7s}  "
        f"{'Rec':<7s}  {'Preds':>5s}  {'TP':>3s}  {'FP':>3s}  "
        f"{'FN':>3s}  {'TN':>3s}"
    )
    print(header)
    print(f"  {'─' * 74}")
    for mode_key, mode_label in modes:
        m = all_configs[mode_key]["metrics"]
        tp_ = all_configs[mode_key]["total_predicates"]
        print(
            f"  {mode_label:<45s}  {m['f1']:<7.4f}  {m['precision']:<7.4f}  "
            f"{m['recall']:<7.4f}  {tp_:>5d}  {m['tp']:>3d}  {m['fp']:>3d}  "
            f"{m['fn']:>3d}  {m['tn']:>3d}"
        )

    sp_f1 = all_configs["single_pass"]["metrics"]["f1"]
    unf_f1 = all_configs["cegar_unfiltered"]["metrics"]["f1"]
    filt_f1 = all_configs["cegar_filtered"]["metrics"]["f1"]
    sp_preds = all_configs["single_pass"]["total_predicates"]
    cegar_preds = all_configs["cegar_filtered"]["total_predicates"]

    print(f"\n  Key comparisons:")
    print(f"    Quality-filtered CEGAR vs single-pass:  ΔF1 = {filt_f1 - sp_f1:+.4f}")
    print(f"    Quality-filtered CEGAR vs unfiltered:   ΔF1 = {filt_f1 - unf_f1:+.4f}")
    print(
        f"    Predicates: single-pass={sp_preds}, "
        f"CEGAR={cegar_preds} (Δ={cegar_preds - sp_preds:+d})"
    )
    print(
        f"    Predicates rejected by quality filter:  "
        f"{all_configs['cegar_filtered']['total_rejected']}"
    )

    # Verdict
    print(f"\n  {'─' * 74}")
    if cegar_preds > sp_preds:
        print(
            f"  ✓ CEGAR discovers more predicates than single-pass "
            f"({cegar_preds} vs {sp_preds})"
        )
    else:
        print(
            f"  ✗ CEGAR did not discover more predicates than single-pass "
            f"({cegar_preds} vs {sp_preds})"
        )
    if filt_f1 >= sp_f1:
        print(
            f"  ✓ Quality-filtered CEGAR does NOT degrade F1 "
            f"(F1={filt_f1:.4f} ≥ {sp_f1:.4f})"
        )
    else:
        delta = sp_f1 - filt_f1
        print(
            f"  ✗ Quality-filtered CEGAR degraded F1 by {delta:.4f} "
            f"(F1={filt_f1:.4f} < {sp_f1:.4f})"
        )
    if filt_f1 >= unf_f1:
        print(
            f"  ✓ Quality filter maintains/improves F1 vs unfiltered "
            f"(F1={filt_f1:.4f} ≥ {unf_f1:.4f})"
        )
    else:
        print(
            f"  ✗ Quality filter did not help vs unfiltered "
            f"(F1={filt_f1:.4f} < {unf_f1:.4f})"
        )

    # ── Write JSON results ──────────────────────────────────────────────
    output = {
        "experiment": "cegar_ablation_v5",
        "description": (
            "CEGAR ablation with SYMBOLIC input dimensions (n=32) to force "
            "contract discovery.  Compares single-pass, unfiltered CEGAR, "
            "and quality-filtered CEGAR.  Addresses reviewer critique that "
            "prior ablations were underpowered (n=8) or used concrete dims "
            "that bypassed CEGAR entirely."
        ),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_benchmarks": n,
        "num_buggy": n_buggy,
        "num_correct": n_safe,
        "architectures": archs,
        "bootstrap_resamples": 10000,
        "configs": all_configs,
        "summary": {
            "single_pass_f1": sp_f1,
            "cegar_unfiltered_f1": unf_f1,
            "cegar_filtered_f1": filt_f1,
            "filtered_vs_single_pass_delta_f1": round(filt_f1 - sp_f1, 4),
            "filtered_vs_unfiltered_delta_f1": round(filt_f1 - unf_f1, 4),
            "single_pass_total_predicates": sp_preds,
            "cegar_filtered_total_predicates": cegar_preds,
            "cegar_discovers_more_predicates": cegar_preds > sp_preds,
            "quality_filtered_cegar_does_not_degrade_f1": filt_f1 >= sp_f1,
            "total_rejected_predicates": all_configs["cegar_filtered"]["total_rejected"],
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
