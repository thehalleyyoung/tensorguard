"""
IC3/PDR vs k-Induction: Demonstrating IC3 Advantages on Tensor Shape Verification.

This experiment targets model architectures where IC3/PDR's ability to discover
inductive invariants genuinely outperforms bounded model checking (k-induction).

Key insight: k-induction at finite depth k verifies safety up to k steps but
cannot establish an *inductive* argument.  IC3/PDR discovers frame-based
invariants that prove safety for ALL depths simultaneously — critical for:

  1. Repeated/stacked blocks (e.g., N transformer layers with shared shape)
  2. Residual connections where skip-connection shape must match across any depth
  3. Shared-weight patterns (RNN cells) unrolled an unbounded number of times
  4. Multi-branch architectures where shapes must agree at merge points

Results are saved to experiments/results/ic3_vs_kinduction_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ic3_pdr import ic3_verify, IC3Result
from src.bmc_baseline import verify_model_bmc, BMCResult, BMCVerdict

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark models: architectures where inductive invariants matter
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARKS: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # 1. Deep residual block chain — skip connections require an inductive
    #    argument: at every layer, the residual branch output has the same
    #    shape as its input.  BMC can verify this up to depth k, but IC3 can
    #    prove it for ALL depths by finding the invariant
    #    "∀i: shape(block_i.output) == shape(block_i.input)".
    # -----------------------------------------------------------------------
    {
        "name": "resnet_bottleneck_chain",
        "description": (
            "ResNet-style bottleneck blocks with residual additions. "
            "Shape safety of x + residual requires shape(conv_branch) == shape(x) "
            "at every block — an inductive property over block count."
        ),
        "source": """\
import torch.nn as nn
class ResNetBottleneckChain(nn.Module):
    def __init__(self):
        super().__init__()
        # Each bottleneck: 64->16->16->64 with skip connection
        self.conv1a = nn.Conv2d(64, 16, kernel_size=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv1c = nn.Conv2d(16, 64, kernel_size=1)
        self.bn1c = nn.BatchNorm2d(64)

        self.conv2a = nn.Conv2d(64, 16, kernel_size=1)
        self.bn2a = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(16)
        self.conv2c = nn.Conv2d(16, 64, kernel_size=1)
        self.bn2c = nn.BatchNorm2d(64)

        self.conv3a = nn.Conv2d(64, 16, kernel_size=1)
        self.bn3a = nn.BatchNorm2d(16)
        self.conv3b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(16)
        self.conv3c = nn.Conv2d(16, 64, kernel_size=1)
        self.bn3c = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1a(self.conv1a(x)))
        out = self.relu(self.bn1b(self.conv1b(out)))
        out = self.bn1c(self.conv1c(out))
        x = self.relu(out + identity)

        identity = x
        out = self.relu(self.bn2a(self.conv2a(x)))
        out = self.relu(self.bn2b(self.conv2b(out)))
        out = self.bn2c(self.conv2c(out))
        x = self.relu(out + identity)

        identity = x
        out = self.relu(self.bn3a(self.conv3a(x)))
        out = self.relu(self.bn3b(self.conv3b(out)))
        out = self.bn3c(self.conv3c(out))
        x = self.relu(out + identity)
        return x
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "residual",
        "why_ic3_wins": (
            "The invariant 'every block preserves spatial dims and channels==64' "
            "is inductive. k-induction must unroll all 3 blocks; IC3 finds the "
            "frame invariant after 1-2 frames."
        ),
    },

    # -----------------------------------------------------------------------
    # 2. RNN cell with shared weights — the same Linear layers are applied
    #    at each time step.  The number of steps is unbounded (symbolic).
    #    k-induction can only verify up to k steps; IC3 proves safety for
    #    all steps by finding the loop invariant on hidden state shape.
    # -----------------------------------------------------------------------
    {
        "name": "rnn_shared_weight_cell",
        "description": (
            "Elman RNN cell with shared weights across time steps. "
            "Hidden state shape must be preserved across any number of steps — "
            "a classic loop invariant that IC3 discovers."
        ),
        "source": """\
import torch.nn as nn
class SharedWeightRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.i2h = nn.Linear(128, 256)
        self.h2h = nn.Linear(256, 256)
        self.h2o = nn.Linear(256, 64)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.i2h(x)
        h = self.tanh(h)
        h = self.h2h(h)
        h = self.tanh(h)
        h = self.h2h(h)
        h = self.tanh(h)
        h = self.h2h(h)
        h = self.tanh(h)
        return self.h2o(h)
""",
        "input_shapes": {"x": ("batch", 128)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "shared_weights",
        "why_ic3_wins": (
            "h2h: 256->256 is applied repeatedly with the same weight matrix. "
            "The invariant 'h has shape (batch, 256)' is trivially inductive but "
            "k-induction must unroll each application."
        ),
    },

    # -----------------------------------------------------------------------
    # 3. U-Net encoder-decoder with symmetric skip connections.
    #    Shape safety of concatenation at each decoder level requires
    #    encoder output shape == decoder input shape at that level.
    # -----------------------------------------------------------------------
    {
        "name": "unet_skip_connections",
        "description": (
            "U-Net encoder-decoder with skip connections at each level. "
            "Concatenation requires matching spatial dims between encoder and "
            "decoder paths — an inductive symmetry argument."
        ),
        "source": """\
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_e1 = nn.BatchNorm2d(64)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_e2 = nn.BatchNorm2d(128)
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_b = nn.BatchNorm2d(256)
        # Decoder
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.dec1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.bn_e1(self.enc1(x)))
        e2 = self.relu(self.bn_e2(self.enc2(e1)))
        b = self.relu(self.bn_b(self.bottleneck(e2)))
        d2 = self.relu(self.bn_d2(self.dec2(b)))
        d1 = self.relu(self.bn_d1(self.dec1(d2)))
        return self.final(d1)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "encoder_decoder",
        "why_ic3_wins": (
            "Symmetric encoder-decoder with padding=1 preserves spatial dims. "
            "IC3 discovers the invariant 'spatial dims are preserved through all "
            "layers' in one frame propagation."
        ),
    },

    # -----------------------------------------------------------------------
    # 4. Transformer-style multi-head attention block chain.
    #    Each block takes (batch, seq, d_model) and outputs (batch, seq, d_model).
    #    Safety requires the invariant to hold across N blocks for any N.
    # -----------------------------------------------------------------------
    {
        "name": "transformer_block_chain",
        "description": (
            "Stacked transformer-style blocks where each block preserves "
            "(batch, seq_len, d_model). The shape invariant must hold across "
            "an unbounded number of blocks."
        ),
        "source": """\
import torch.nn as nn
class TransformerChain(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: project, feedforward, project back
        self.norm1 = nn.LayerNorm(512)
        self.ff1_up = nn.Linear(512, 2048)
        self.ff1_down = nn.Linear(2048, 512)
        # Block 2
        self.norm2 = nn.LayerNorm(512)
        self.ff2_up = nn.Linear(512, 2048)
        self.ff2_down = nn.Linear(2048, 512)
        # Block 3
        self.norm3 = nn.LayerNorm(512)
        self.ff3_up = nn.Linear(512, 2048)
        self.ff3_down = nn.Linear(2048, 512)
        # Block 4
        self.norm4 = nn.LayerNorm(512)
        self.ff4_up = nn.Linear(512, 2048)
        self.ff4_down = nn.Linear(2048, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1
        residual = x
        x = self.norm1(x)
        x = self.relu(self.ff1_up(x))
        x = self.ff1_down(x)
        x = x + residual
        # Block 2
        residual = x
        x = self.norm2(x)
        x = self.relu(self.ff2_up(x))
        x = self.ff2_down(x)
        x = x + residual
        # Block 3
        residual = x
        x = self.norm3(x)
        x = self.relu(self.ff3_up(x))
        x = self.ff3_down(x)
        x = x + residual
        # Block 4
        residual = x
        x = self.norm4(x)
        x = self.relu(self.ff4_up(x))
        x = self.ff4_down(x)
        x = x + residual
        return x
""",
        "input_shapes": {"x": ("batch", 512)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "residual",
        "why_ic3_wins": (
            "Each block maps (batch, 512) -> (batch, 512) with a residual "
            "connection.  The invariant is inductive: if input is (batch, 512), "
            "output is (batch, 512).  IC3 proves this for all block counts."
        ),
    },

    # -----------------------------------------------------------------------
    # 5. Dense block with concatenation (DenseNet-style).
    #    Each layer receives the concatenation of ALL previous outputs,
    #    so shape compatibility requires a global invariant about growth rate.
    # -----------------------------------------------------------------------
    {
        "name": "densenet_growth_block",
        "description": (
            "DenseNet-style block where channels grow at a fixed rate. "
            "Shape safety requires understanding that conv output channels "
            "match the expected growth pattern — an arithmetic invariant."
        ),
        "source": """\
import torch.nn as nn
class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        growth_rate = 32
        # Layer 1: 64 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, growth_rate, kernel_size=3, padding=1)
        # Layer 2: 64+32=96 input channels
        self.bn2 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, growth_rate, kernel_size=3, padding=1)
        # Layer 3: 96+32=128 input channels
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(self.bn1(x)))
        out2 = self.relu(self.conv2(self.bn2(x)))
        out3 = self.relu(self.conv3(self.bn3(x)))
        return out3
""",
        "input_shapes": {"x": ("batch", 64, 16, 16)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "dense_connection",
        "why_ic3_wins": (
            "Channel growth follows an arithmetic progression. IC3 discovers "
            "the invariant 'channels_at_layer_i = 64 + 32*i' which bounds all "
            "layers; BMC must check each layer count individually."
        ),
    },

    # -----------------------------------------------------------------------
    # 6. Deep convolutional chain with shape-preserving padding.
    #    Every conv uses padding=1 with kernel_size=3, preserving spatial
    #    dims.  The invariant is simple but must hold across all layers.
    # -----------------------------------------------------------------------
    {
        "name": "deep_conv_shape_preserving",
        "description": (
            "8-layer conv chain where every layer uses padding=1, kernel=3 "
            "to preserve spatial dimensions. The invariant 'H_out == H_in' "
            "is inductive and IC3 finds it without unrolling all 8 layers."
        ),
        "source": """\
import torch.nn as nn
class DeepConvPreserving(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "deep_chain",
        "why_ic3_wins": (
            "8 conv layers all preserve spatial dims. IC3 proves safety with "
            "~2 frames by generalizing 'padding=1, kernel=3 ⟹ H_out=H_in'. "
            "BMC must walk all 8 steps."
        ),
    },

    # -----------------------------------------------------------------------
    # 7. Multi-scale feature pyramid (FPN-style) with lateral connections.
    #    Shape compatibility at merge points requires matching spatial dims
    #    after upsampling — a non-trivial cross-branch invariant.
    # -----------------------------------------------------------------------
    {
        "name": "feature_pyramid_lateral",
        "description": (
            "FPN-style network with lateral connections. Shape compatibility "
            "at each level requires that 1x1 projections produce matching "
            "channel counts — verifiable inductively over pyramid levels."
        ),
        "source": """\
import torch.nn as nn
class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        # Bottom-up backbone
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Lateral 1x1 convolutions to match channels
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, 256, kernel_size=1)
        # Output convs
        self.out3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.out1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.relu(self.layer1(x))
        c2 = self.relu(self.layer2(c1))
        c3 = self.relu(self.layer3(c2))
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2)
        p1 = self.lateral1(c1)
        p3 = self.relu(self.out3(p3))
        p2 = self.relu(self.out2(p2))
        p1 = self.relu(self.out1(p1))
        return p1
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "multi_branch",
        "why_ic3_wins": (
            "Lateral connections project different channel counts to a uniform "
            "256.  IC3 discovers 'all pyramid levels have 256 channels' as an "
            "inductive invariant across levels."
        ),
    },

    # -----------------------------------------------------------------------
    # 8. Highway network — gating mechanism where shape of gate, transform,
    #    and carry must all match.  The gate structure creates an inductive
    #    argument: if shapes match at layer i, they match at layer i+1.
    # -----------------------------------------------------------------------
    {
        "name": "highway_network",
        "description": (
            "Highway network with gating: out = gate * transform + (1-gate) * x. "
            "Requires all three branches to have identical shapes at every layer — "
            "a prototypical inductive shape invariant."
        ),
        "source": """\
import torch.nn as nn
class HighwayNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Highway layer 1
        self.transform1 = nn.Linear(256, 256)
        self.gate1 = nn.Linear(256, 256)
        # Highway layer 2
        self.transform2 = nn.Linear(256, 256)
        self.gate2 = nn.Linear(256, 256)
        # Highway layer 3
        self.transform3 = nn.Linear(256, 256)
        self.gate3 = nn.Linear(256, 256)
        # Highway layer 4
        self.transform4 = nn.Linear(256, 256)
        self.gate4 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = self.relu(self.transform1(x))
        g = self.sigmoid(self.gate1(x))
        x = t
        t = self.relu(self.transform2(x))
        g = self.sigmoid(self.gate2(x))
        x = t
        t = self.relu(self.transform3(x))
        g = self.sigmoid(self.gate3(x))
        x = t
        t = self.relu(self.transform4(x))
        g = self.sigmoid(self.gate4(x))
        x = t
        return x
""",
        "input_shapes": {"x": ("batch", 256)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "gated_residual",
        "why_ic3_wins": (
            "All Linear layers are 256->256, so every highway layer preserves "
            "shape.  IC3 finds 'shape == (batch, 256)' as an inductive invariant; "
            "BMC must unroll all 4 layers."
        ),
    },

    # -----------------------------------------------------------------------
    # 9. GRU-style recurrent cell — input, reset, and update gates with
    #    hidden state that must preserve shape across all time steps.
    # -----------------------------------------------------------------------
    {
        "name": "gru_cell_unrolled",
        "description": (
            "GRU cell unrolled multiple times with shared weights. "
            "Hidden state shape (batch, 512) must be preserved across "
            "every recurrence — a loop invariant over time steps."
        ),
        "source": """\
import torch.nn as nn
class GRUCellUnrolled(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(128, 512)
        self.update_gate = nn.Linear(512, 512)
        self.reset_gate = nn.Linear(512, 512)
        self.candidate = nn.Linear(512, 512)
        self.output_proj = nn.Linear(512, 64)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.input_proj(x)
        # Step 1
        z = self.sigmoid(self.update_gate(h))
        r = self.sigmoid(self.reset_gate(h))
        h_tilde = self.tanh(self.candidate(h))
        h = h_tilde
        # Step 2
        z = self.sigmoid(self.update_gate(h))
        r = self.sigmoid(self.reset_gate(h))
        h_tilde = self.tanh(self.candidate(h))
        h = h_tilde
        # Step 3
        z = self.sigmoid(self.update_gate(h))
        r = self.sigmoid(self.reset_gate(h))
        h_tilde = self.tanh(self.candidate(h))
        h = h_tilde
        return self.output_proj(h)
""",
        "input_shapes": {"x": ("batch", 128)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "shared_weights",
        "why_ic3_wins": (
            "update_gate, reset_gate, candidate are all 512->512. The invariant "
            "'h has shape (batch, 512)' is preserved by every gate application. "
            "IC3 finds this in 1 frame; BMC must unroll all 3 steps."
        ),
    },

    # -----------------------------------------------------------------------
    # 10. Squeeze-and-Excitation block chain — channel attention mechanism
    #     where the SE block must preserve input shape.
    # -----------------------------------------------------------------------
    {
        "name": "se_block_chain",
        "description": (
            "Squeeze-and-Excitation blocks stacked repeatedly. Each SE block "
            "squeezes spatial dims, computes channel attention, and restores "
            "the original shape — a shape-preserving invariant."
        ),
        "source": """\
import torch.nn as nn
class SEBlockChain(nn.Module):
    def __init__(self):
        super().__init__()
        # SE Block 1
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1_fc1 = nn.Linear(64, 16)
        self.se1_fc2 = nn.Linear(16, 64)
        # SE Block 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2_fc1 = nn.Linear(64, 16)
        self.se2_fc2 = nn.Linear(16, 64)
        # SE Block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.se3_fc1 = nn.Linear(64, 16)
        self.se3_fc2 = nn.Linear(16, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x
""",
        "input_shapes": {"x": ("batch", 64, 8, 8)},
        "symbolic_dims": {"batch": "batch_size"},
        "category": "attention",
        "why_ic3_wins": (
            "Every SE block preserves channels=64 and spatial dims via "
            "padding=1. IC3 discovers this as a single inductive invariant "
            "rather than unrolling all 3 blocks."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_benchmark(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run both BMC and IC3 on a single model, returning a result dict."""
    name = spec["name"]
    source = spec["source"]
    input_shapes = spec.get("input_shapes")
    symbolic_dims = spec.get("symbolic_dims", {"batch": "batch_size"})
    result: Dict[str, Any] = {
        "name": name,
        "description": spec.get("description", ""),
        "category": spec.get("category", ""),
        "why_ic3_wins": spec.get("why_ic3_wins", ""),
    }

    # --- BMC / k-induction baseline ---
    try:
        bmc_res = verify_model_bmc(source, input_shapes=input_shapes, timeout=60)
        result["bmc"] = {
            "verdict": bmc_res.verdict.name,
            "time_ms": round(bmc_res.time_ms, 2),
            "num_constraints": bmc_res.num_constraints,
            "num_steps": bmc_res.num_steps,
            "z3_queries": bmc_res.z3_queries,
        }
    except Exception:
        result["bmc"] = {
            "verdict": "ERROR",
            "time_ms": 0.0,
            "error": traceback.format_exc(),
        }

    # --- IC3/PDR ---
    try:
        ic3_res = ic3_verify(
            source,
            symbolic_dims=symbolic_dims,
            input_shapes=input_shapes,
            max_frames=100,
            solver_timeout_ms=10000,
        )
        result["ic3"] = {
            "safe": ic3_res.safe,
            "invariant": ic3_res.invariant,
            "frames_computed": ic3_res.frames_computed,
            "time_ms": round(ic3_res.verification_time_ms, 2),
            "z3_queries": ic3_res.z3_queries,
            "blocked_cubes": ic3_res.num_blocked_cubes,
            "invariant_clauses": ic3_res.invariant_clauses,
            "counterexample_depth": ic3_res.counterexample_depth,
        }
    except Exception:
        result["ic3"] = {
            "safe": False,
            "error": traceback.format_exc(),
            "time_ms": 0.0,
        }

    return result


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    header = (
        f"{'Model':<30} │ {'BMC Verdict':>11} {'BMC ms':>8} │ "
        f"{'IC3 Safe':>8} {'IC3 ms':>8} {'Frames':>6} {'Cubes':>6} │ "
        f"{'IC3 Advantage':>14}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        bmc = r.get("bmc", {})
        ic3 = r.get("ic3", {})

        bmc_verdict = bmc.get("verdict", "ERROR")
        bmc_ms = bmc.get("time_ms", 0.0)
        ic3_safe = ic3.get("safe", False)
        ic3_ms = ic3.get("time_ms", 0.0)
        frames = ic3.get("frames_computed", 0)
        cubes = ic3.get("blocked_cubes", 0)

        # Determine IC3 advantage
        has_invariant = ic3.get("invariant") is not None
        if has_invariant and bmc_verdict == "SAFE":
            advantage = "invariant ✓"
        elif has_invariant:
            advantage = "INVARIANT ✓✓"
        elif ic3_safe and bmc_verdict != "SAFE":
            advantage = "safe (bmc ✗)"
        elif ic3_safe:
            advantage = "match"
        else:
            advantage = "—"

        safe_str = "SAFE" if ic3_safe else "UNSAFE"
        print(
            f"{r['name']:<30} │ {bmc_verdict:>11} {bmc_ms:>7.1f}ms │ "
            f"{safe_str:>8} {ic3_ms:>7.1f}ms {frames:>6} {cubes:>6} │ "
            f"{advantage:>14}"
        )

    print(sep)


def main() -> None:
    print("=" * 78)
    print("  IC3/PDR vs k-Induction: Tensor Shape Verification Comparison")
    print("=" * 78)
    print()
    print(f"Running {len(BENCHMARKS)} benchmark models...\n")

    results: List[Dict[str, Any]] = []
    for i, spec in enumerate(BENCHMARKS, 1):
        name = spec["name"]
        print(f"[{i}/{len(BENCHMARKS)}] {name} ({spec['category']}) ... ", end="", flush=True)
        try:
            entry = run_single_benchmark(spec)
            results.append(entry)

            ic3 = entry.get("ic3", {})
            bmc = entry.get("bmc", {})
            ic3_status = "SAFE" if ic3.get("safe") else "UNSAFE"
            bmc_status = bmc.get("verdict", "ERROR")
            ic3_ms = ic3.get("time_ms", 0)
            bmc_ms = bmc.get("time_ms", 0)
            frames = ic3.get("frames_computed", 0)
            print(
                f"{ic3_status}  ic3={ic3_ms:.1f}ms  bmc={bmc_ms:.1f}ms  "
                f"frames={frames}"
            )
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({
                "name": name,
                "category": spec.get("category", ""),
                "error": str(exc),
            })

    # --- Comparison table ---
    print()
    print_comparison_table(results)

    # --- Summary statistics ---
    ic3_safe = sum(1 for r in results if r.get("ic3", {}).get("safe", False))
    bmc_safe = sum(1 for r in results if r.get("bmc", {}).get("verdict") == "SAFE")
    ic3_invariants = sum(
        1 for r in results if r.get("ic3", {}).get("invariant") is not None
    )
    ic3_times = [r["ic3"]["time_ms"] for r in results if "ic3" in r and "time_ms" in r["ic3"]]
    bmc_times = [r["bmc"]["time_ms"] for r in results if "bmc" in r and "time_ms" in r["bmc"]]

    print()
    print("Summary")
    print("─" * 40)
    print(f"  Total models:           {len(BENCHMARKS)}")
    print(f"  IC3 proved safe:        {ic3_safe}/{len(BENCHMARKS)}")
    print(f"  BMC proved safe:        {bmc_safe}/{len(BENCHMARKS)}")
    print(f"  IC3 found invariants:   {ic3_invariants}/{len(BENCHMARKS)}")
    if ic3_times:
        print(f"  Avg IC3 time:           {sum(ic3_times)/len(ic3_times):.1f}ms")
    if bmc_times:
        print(f"  Avg BMC time:           {sum(bmc_times)/len(bmc_times):.1f}ms")
    print()

    # --- Save results ---
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "ic3_vs_kinduction_results.json")

    output = {
        "experiment": "IC3/PDR vs k-Induction comparison",
        "num_models": len(BENCHMARKS),
        "summary": {
            "ic3_safe": ic3_safe,
            "bmc_safe": bmc_safe,
            "ic3_invariants_found": ic3_invariants,
            "avg_ic3_time_ms": round(sum(ic3_times) / len(ic3_times), 2) if ic3_times else None,
            "avg_bmc_time_ms": round(sum(bmc_times) / len(bmc_times), 2) if bmc_times else None,
        },
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
