"""
UNSAT-Core Synthesis Strategy Ablation Experiment.

Addresses reviewer concerns (Chang, Sinha) about whether the three
predicate synthesis strategies in EnhancedUnsatCorePredicateExtractor
are complementary or redundant.

The three strategies are:
  (1) Direct     — each MUS assertion → predicate
  (2) Interpolant — pairwise implied bounds from MUS subsets
  (3) Generalised — weaken concrete equalities to inequality bounds

This experiment runs UNSAT-core CEGAR with every non-empty subset of
{direct, interpolant, generalised} (7 configurations) across the same
32 benchmarks from the v5 ablation, recording:
  - iterations, predicates discovered, time, verdict correctness

Then computes complementarity / redundancy / dominance analysis.

Outputs: experiments/unsat_ablation_results.json
"""

from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.unsat_core_cegar import (
    EnhancedUnsatCorePredicateExtractor,
    UnsatCorePredicate,
    run_enhanced_cegar,
)
from src.shape_cegar import ShapeCEGARResult, CEGARStatus

RESULTS_FILE = Path(__file__).parent / "unsat_ablation_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarks — reuse from v5 ablation
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
# Strategy-selective extractor (monkey-patches individual strategies)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_NAMES = ["direct", "interpolant", "generalised"]


class SelectiveExtractor(EnhancedUnsatCorePredicateExtractor):
    """Wraps the extractor to selectively enable/disable strategies."""

    def __init__(
        self,
        enabled_strategies: FrozenSet[str],
        timeout_ms: int = 3000,
    ) -> None:
        super().__init__(timeout_ms=timeout_ms)
        self._enabled = enabled_strategies

    def extract_predicates(
        self,
        core_labels: List[str],
        assertion_map: Dict[str, Any],
        dim_map=None,
    ) -> List[UnsatCorePredicate]:
        """Extract predicates using only the enabled strategies."""
        try:
            import z3
            has_z3 = True
        except ImportError:
            has_z3 = False

        if not has_z3 or not core_labels:
            return []

        mus = self._extract_mus(core_labels, assertion_map)

        predicates: List[UnsatCorePredicate] = []

        if "direct" in self._enabled:
            predicates.extend(self._direct_predicates(mus, assertion_map, dim_map))

        if "interpolant" in self._enabled:
            predicates.extend(self._pairwise_predicates(mus, assertion_map, dim_map))

        if "generalised" in self._enabled:
            predicates.extend(self._generalised_predicates(mus, assertion_map, dim_map))

        # Deduplicate by formula string
        seen: Set[str] = set()
        unique: List[UnsatCorePredicate] = []
        for p in predicates:
            key = str(p.formula)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique


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


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_with_strategies(
    tc: Dict[str, Any],
    enabled: FrozenSet[str],
) -> Dict[str, Any]:
    """Run enhanced CEGAR with only the specified strategies enabled.

    Monkey-patches the core extractor inside EnhancedShapeCEGARLoop.
    """
    from src.unsat_core_cegar import EnhancedShapeCEGARLoop

    t0 = time.monotonic()
    try:
        loop = EnhancedShapeCEGARLoop(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        # Replace the core extractor with our selective one
        loop._core_extractor = SelectiveExtractor(enabled)
        result = loop.run()

        detected = result.has_real_bugs
        status = result.final_status.name
        n_preds = len(result.discovered_predicates)
        n_iters = result.iterations
        istats = result.interpolation_stats or {}
        core_preds = istats.get("core_predicates_count", 0)
        template_preds = istats.get("template_predicates_count", 0)
        mus_extractions = istats.get("mus_extractions", 0)
    except Exception as e:
        detected = False
        status = f"ERROR: {e}"
        n_preds = 0
        n_iters = 0
        core_preds = 0
        template_preds = 0
        mus_extractions = 0

    elapsed = (time.monotonic() - t0) * 1000
    return {
        "name": tc["name"],
        "arch": tc["arch"],
        "has_bug": tc["has_bug"],
        "detected_bug": detected,
        "correct": detected == tc["has_bug"],
        "status": status,
        "iterations": n_iters,
        "predicates_discovered": n_preds,
        "core_predicates": core_preds,
        "template_predicates": template_preds,
        "mus_extractions": mus_extractions,
        "time_ms": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis: complementarity / redundancy / dominance
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_strategies(all_configs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute complementarity, redundancy, and dominance metrics."""
    analysis: Dict[str, Any] = {}

    # Collect per-benchmark correctness vectors for each config
    config_correct: Dict[str, List[bool]] = {}
    config_preds: Dict[str, int] = {}
    config_iters: Dict[str, int] = {}
    config_time: Dict[str, float] = {}

    for cfg_key, cfg_data in all_configs.items():
        benches = cfg_data["per_benchmark"]
        config_correct[cfg_key] = [b["correct"] for b in benches]
        config_preds[cfg_key] = sum(b["predicates_discovered"] for b in benches)
        config_iters[cfg_key] = sum(b["iterations"] for b in benches)
        config_time[cfg_key] = sum(b["time_ms"] for b in benches)

    all_key = "D+I+G"
    n_benchmarks = len(config_correct.get(all_key, []))

    # Individual strategy contribution: benchmarks solved by all_three
    # but not by (all_three minus this strategy)
    singles = {"D": "direct", "I": "interpolant", "G": "generalised"}
    pairs = {"D+I": "direct+interpolant", "D+G": "direct+generalised",
             "I+G": "interpolant+generalised"}

    # Unique contributions: benchmarks correct in config X but not in config Y
    unique_contributions: Dict[str, Dict[str, int]] = {}
    for s_key, s_name in singles.items():
        pair_without = {k: v for k, v in pairs.items() if s_key[0] not in k}
        # Find the pair that excludes this strategy
        complement_pairs = [k for k in pairs if s_key not in k]
        if complement_pairs:
            complement_key = complement_pairs[0]
        else:
            complement_key = None

        # Benchmarks where the full config is correct but removing this
        # strategy (pair of the other two) is incorrect
        if all_key in config_correct and complement_key and complement_key in config_correct:
            unique_count = sum(
                1 for a, b in zip(config_correct[all_key], config_correct[complement_key])
                if a and not b
            )
        else:
            unique_count = 0

        unique_contributions[s_name] = {
            "uniquely_needed_benchmarks": unique_count,
            "solo_accuracy": (
                sum(config_correct.get(s_key, [])) / n_benchmarks
                if n_benchmarks > 0 else 0
            ),
            "solo_predicates": config_preds.get(s_key, 0),
        }

    # Pairwise complementarity: does adding a strategy to a pair help?
    pairwise_analysis: Dict[str, Any] = {}
    pair_keys = [("D", "I", "D+I"), ("D", "G", "D+G"), ("I", "G", "I+G")]
    for s1, s2, pair_key in pair_keys:
        if (s1 in config_correct and s2 in config_correct
                and pair_key in config_correct):
            # Complementarity: benchmarks correct in pair but not in either solo
            complementary = sum(
                1 for p, a, b in zip(
                    config_correct[pair_key],
                    config_correct[s1],
                    config_correct[s2],
                )
                if p and not (a and b)
            )
            # Redundancy: benchmarks where both solos agree with pair
            redundant = sum(
                1 for p, a, b in zip(
                    config_correct[pair_key],
                    config_correct[s1],
                    config_correct[s2],
                )
                if p == a == b
            )
            pairwise_analysis[pair_key] = {
                "complementary_benchmarks": complementary,
                "redundant_benchmarks": redundant,
                "pair_accuracy": (
                    sum(config_correct[pair_key]) / n_benchmarks
                    if n_benchmarks > 0 else 0
                ),
                "pair_pred_gain_over_best_solo": (
                    config_preds.get(pair_key, 0)
                    - max(config_preds.get(s1, 0), config_preds.get(s2, 0))
                ),
            }

    # Dominance: is one strategy strictly better than another?
    dominance: Dict[str, str] = {}
    for s_a, s_b in itertools.combinations(singles.keys(), 2):
        if s_a in config_correct and s_b in config_correct:
            a_wins = sum(
                1 for a, b in zip(config_correct[s_a], config_correct[s_b])
                if a and not b
            )
            b_wins = sum(
                1 for a, b in zip(config_correct[s_a], config_correct[s_b])
                if b and not a
            )
            if a_wins > 0 and b_wins == 0:
                dominance[f"{s_a}_vs_{s_b}"] = f"{singles[s_a]} dominates {singles[s_b]}"
            elif b_wins > 0 and a_wins == 0:
                dominance[f"{s_a}_vs_{s_b}"] = f"{singles[s_b]} dominates {singles[s_a]}"
            elif a_wins > 0 and b_wins > 0:
                dominance[f"{s_a}_vs_{s_b}"] = (
                    f"complementary ({singles[s_a]} wins {a_wins}, "
                    f"{singles[s_b]} wins {b_wins})"
                )
            else:
                dominance[f"{s_a}_vs_{s_b}"] = "equivalent (same correctness)"

    # Overall verdict
    all_acc = sum(config_correct.get(all_key, [])) / n_benchmarks if n_benchmarks else 0
    best_single_acc = max(
        (sum(config_correct.get(k, [])) / n_benchmarks if n_benchmarks else 0)
        for k in singles
    )
    best_pair_acc = max(
        (sum(config_correct.get(k, [])) / n_benchmarks if n_benchmarks else 0)
        for k in pairs
    )

    if all_acc > best_pair_acc:
        verdict = "COMPLEMENTARY: all three strategies together outperform any pair"
    elif all_acc > best_single_acc:
        verdict = "PARTIALLY_COMPLEMENTARY: pairs outperform singles, but all-three does not beat best pair"
    elif all_acc == best_single_acc:
        verdict = "REDUNDANT: individual strategies achieve the same accuracy as all three combined"
    else:
        verdict = "INTERFERENCE: combining strategies degrades accuracy"

    analysis = {
        "unique_contributions": unique_contributions,
        "pairwise_analysis": pairwise_analysis,
        "dominance": dominance,
        "accuracy_summary": {
            "all_three": round(all_acc, 4),
            "best_single": round(best_single_acc, 4),
            "best_pair": round(best_pair_acc, 4),
        },
        "verdict": verdict,
    }
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    n = len(TEST_CASES)
    n_buggy = sum(1 for tc in TEST_CASES if tc["has_bug"])
    n_safe = n - n_buggy
    archs = sorted(set(tc["arch"] for tc in TEST_CASES))

    print("=" * 78)
    print("  UNSAT-Core Synthesis Strategy Ablation Experiment")
    print(f"  {n} benchmarks ({n_buggy} buggy, {n_safe} correct) × 7 configs")
    print(f"  Architectures: {', '.join(archs)}")
    print("  Strategies: direct (D), interpolant (I), generalised (G)")
    print("=" * 78)

    # All non-empty subsets of {direct, interpolant, generalised}
    configs = [
        ("D",     frozenset({"direct"})),
        ("I",     frozenset({"interpolant"})),
        ("G",     frozenset({"generalised"})),
        ("D+I",   frozenset({"direct", "interpolant"})),
        ("D+G",   frozenset({"direct", "generalised"})),
        ("I+G",   frozenset({"interpolant", "generalised"})),
        ("D+I+G", frozenset({"direct", "interpolant", "generalised"})),
    ]

    all_configs: Dict[str, Any] = {}

    for cfg_key, enabled in configs:
        label = f"Strategies: {cfg_key}"
        print(f"\n{'─' * 78}")
        print(f"  Config: {label}")
        print(f"{'─' * 78}")

        per_bench: List[Dict[str, Any]] = []
        for tc in TEST_CASES:
            try:
                r = run_with_strategies(tc, enabled)
            except Exception as exc:
                r = {
                    "name": tc["name"],
                    "arch": tc["arch"],
                    "has_bug": tc["has_bug"],
                    "detected_bug": False,
                    "correct": not tc["has_bug"],
                    "status": f"ERROR: {exc}",
                    "iterations": 0,
                    "predicates_discovered": 0,
                    "core_predicates": 0,
                    "template_predicates": 0,
                    "mus_extractions": 0,
                    "time_ms": 0.0,
                }

            mark = "✓" if r["correct"] else "✗"
            label_str = "BUG" if r["has_bug"] else "SAFE"
            det_str = "det" if r["detected_bug"] else "safe"
            print(
                f"  {mark} {r['name']:30s}  "
                f"expected={label_str:<4s}  result={det_str:<4s}  "
                f"iters={r['iterations']}  preds={r['predicates_discovered']}  "
                f"core={r['core_predicates']}  {r['time_ms']:.0f}ms"
            )
            per_bench.append(r)

        metrics = compute_metrics(per_bench)
        total_preds = sum(r["predicates_discovered"] for r in per_bench)
        total_core = sum(r["core_predicates"] for r in per_bench)
        total_template = sum(r["template_predicates"] for r in per_bench)
        total_iters = sum(r["iterations"] for r in per_bench)
        total_time = sum(r["time_ms"] for r in per_bench)
        correct_count = sum(1 for r in per_bench if r["correct"])

        print(
            f"\n  Aggregate: F1={metrics['f1']}  "
            f"Precision={metrics['precision']}  Recall={metrics['recall']}  "
            f"Accuracy={correct_count}/{n}"
        )
        print(
            f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  "
            f"FN={metrics['fn']}  TN={metrics['tn']}"
        )
        print(
            f"  Totals:    predicates={total_preds} (core={total_core}, "
            f"template={total_template})  iterations={total_iters}  "
            f"time={total_time:.0f}ms"
        )

        all_configs[cfg_key] = {
            "label": f"Strategies: {cfg_key}",
            "enabled_strategies": sorted(enabled),
            "metrics": metrics,
            "accuracy": round(correct_count / n, 4) if n > 0 else 0,
            "total_predicates": total_preds,
            "total_core_predicates": total_core,
            "total_template_predicates": total_template,
            "total_iterations": total_iters,
            "total_time_ms": round(total_time, 2),
            "per_benchmark": per_bench,
        }

    # ── Summary comparison table ────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 78}")
    header = (
        f"  {'Config':<10s}  {'F1':<7s}  {'Prec':<7s}  {'Rec':<7s}  "
        f"{'Acc':>5s}  {'Preds':>5s}  {'Core':>4s}  "
        f"{'Iters':>5s}  {'Time':>7s}"
    )
    print(header)
    print(f"  {'─' * 70}")
    for cfg_key, _ in configs:
        c = all_configs[cfg_key]
        m = c["metrics"]
        print(
            f"  {cfg_key:<10s}  {m['f1']:<7.4f}  {m['precision']:<7.4f}  "
            f"{m['recall']:<7.4f}  {c['accuracy']:>5.3f}  "
            f"{c['total_predicates']:>5d}  {c['total_core_predicates']:>4d}  "
            f"{c['total_iterations']:>5d}  {c['total_time_ms']:>7.0f}ms"
        )

    # ── Strategy analysis ───────────────────────────────────────────────
    analysis = analyse_strategies(all_configs)

    print(f"\n{'=' * 78}")
    print("  STRATEGY ANALYSIS")
    print(f"{'=' * 78}")

    print(f"\n  Accuracy summary:")
    acc = analysis["accuracy_summary"]
    print(f"    All three strategies: {acc['all_three']:.4f}")
    print(f"    Best single strategy: {acc['best_single']:.4f}")
    print(f"    Best pair:            {acc['best_pair']:.4f}")

    print(f"\n  Unique contributions:")
    for strat, info in analysis["unique_contributions"].items():
        print(
            f"    {strat:14s}: solo_acc={info['solo_accuracy']:.3f}  "
            f"solo_preds={info['solo_predicates']}  "
            f"uniquely_needed={info['uniquely_needed_benchmarks']}"
        )

    print(f"\n  Pairwise analysis:")
    for pair, info in analysis["pairwise_analysis"].items():
        print(
            f"    {pair:5s}: acc={info['pair_accuracy']:.3f}  "
            f"complementary={info['complementary_benchmarks']}  "
            f"redundant={info['redundant_benchmarks']}  "
            f"pred_gain={info['pair_pred_gain_over_best_solo']:+d}"
        )

    print(f"\n  Dominance:")
    for pair, result in analysis["dominance"].items():
        print(f"    {pair}: {result}")

    print(f"\n  {'─' * 74}")
    print(f"  VERDICT: {analysis['verdict']}")
    print(f"  {'─' * 74}")

    # ── Write JSON results ──────────────────────────────────────────────
    output = {
        "experiment": "unsat_core_synthesis_strategy_ablation",
        "description": (
            "Ablation of the three UNSAT-core predicate synthesis strategies "
            "(direct, interpolant, generalised) in the enhanced CEGAR loop. "
            "Tests all 7 non-empty subsets across 32 benchmarks to determine "
            "complementarity vs redundancy. Requested by reviewers Chang & Sinha."
        ),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_benchmarks": n,
        "num_buggy": n_buggy,
        "num_correct": n_safe,
        "architectures": archs,
        "strategies": STRATEGY_NAMES,
        "configs": {
            k: {key: val for key, val in v.items() if key != "per_benchmark"}
            for k, v in all_configs.items()
        },
        "per_benchmark_results": {
            k: v["per_benchmark"] for k, v in all_configs.items()
        },
        "analysis": analysis,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
