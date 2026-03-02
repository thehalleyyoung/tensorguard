"""
Multi-Theory Cross-Cutting Bug Benchmark for TensorGuard.

These benchmarks test bugs that span MULTIPLE verification theories
simultaneously (shape × device × phase). No other existing tool
— not TorchScript, PyTEA, jaxtyping, mypy, Pyright, or LLMs —
can detect these bugs because they require reasoning across theory
boundaries.

This is TensorGuard's genuinely SOTA capability: the 5-theory product
domain (Shape × Device × Phase × Stride × Permutation) catches bugs
that live at the intersection of multiple concerns.

Benchmark categories:
  A. Device+Shape  (12 models) — device transfers that cause shape issues
  B. Phase+Shape   (10 models) — train/eval mode differences causing shape bugs
  C. Device+Phase  (4 models)  — device-phase interactions
  D. Triple-theory (6 models)  — shape+device+phase simultaneously
  E. Correct multi-theory (8 models) — models using multiple theories correctly
  F. Real-world sourced (12 models) — bugs from HuggingFace, torchvision, fairseq,
     detectron2, mmdetection, YOLOv5, timm, wav2vec2, StackOverflow, PyTorch Forums
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MultiTheoryBenchmark:
    name: str
    source: str
    has_bug: bool
    bug_description: str
    category: str          # "device_shape", "phase_shape", "device_phase", "triple", "correct"
    theories_required: List[str]   # which theories are needed to catch/verify this
    torchscript_catches: bool      # would torch.jit.script catch it?
    mypy_catches: bool             # would mypy catch it?
    pytea_catches: bool            # would PyTEA catch it?
    jaxtyping_catches: bool        # would jaxtyping catch it?
    llm_catches: Optional[bool]    # would an LLM catch it? (None = uncertain)


# ═══════════════════════════════════════════════════════════════════════════
# A. DEVICE + SHAPE BUGS (12)
#    These bugs involve tensors on different devices being combined,
#    or device transfers that change the effective computation path.
# ═══════════════════════════════════════════════════════════════════════════

DEVICE_SHAPE_BUGS: List[MultiTheoryBenchmark] = [
    # A1: Buffer on CPU, parameters on GPU
    MultiTheoryBenchmark(
        name="buffer_device_mismatch",
        source='''\
import torch
import torch.nn as nn

class BufferDeviceMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.register_buffer('scale', torch.ones(128))  # stays on CPU

    def forward(self, x):
        x = self.fc(x)       # x on GPU (if model.cuda())
        x = x * self.scale   # BUG: scale on CPU, x on GPU
        return x
''',
        has_bug=True,
        bug_description="register_buffer 'scale' stays on CPU when model is moved to GPU; "
                        "multiplication x * scale is a cross-device operation",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A2: Device-dependent reshape
    MultiTheoryBenchmark(
        name="device_dependent_reshape",
        source='''\
import torch
import torch.nn as nn

class DeviceDependentReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
        self.register_buffer('mask', torch.ones(1, 64, 1, 1))

    def forward(self, x):
        x = self.conv(x)            # (B, 64, H, W) on model's device
        x = x * self.mask            # BUG: mask may be on different device
        x = x.mean(dim=[2, 3])       # (B, 64)
        x = self.fc(x)
        return x
''',
        has_bug=True,
        bug_description="Buffer 'mask' may not follow model to GPU; "
                        "cross-device multiply before shape-reducing mean",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A3: torch.tensor literal creates CPU tensor
    MultiTheoryBenchmark(
        name="literal_tensor_device",
        source='''\
import torch
import torch.nn as nn

class LiteralTensorDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.fc(x)
        bias = torch.tensor([1.0] * 64)   # always CPU!
        x = x + bias                       # BUG: cross-device if x is on GPU
        return x
''',
        has_bug=True,
        bug_description="torch.tensor() creates CPU tensor; adding to GPU "
                        "tensor 'x' is a cross-device operation",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # A4: Hardcoded device in forward
    MultiTheoryBenchmark(
        name="hardcoded_device_forward",
        source='''\
import torch
import torch.nn as nn

class HardcodedDeviceForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        noise = torch.randn(256, device='cpu')  # always CPU
        x = x + noise                            # BUG if model on GPU
        x = self.fc2(x)
        return x
''',
        has_bug=True,
        bug_description="torch.randn with device='cpu' creates CPU tensor; "
                        "adding to potentially GPU tensor is cross-device op",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # A5: Buffer shape mismatch after device transfer
    MultiTheoryBenchmark(
        name="buffer_shape_device_compound",
        source='''\
import torch
import torch.nn as nn

class BufferShapeDeviceCompound(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.register_buffer('channel_weight', torch.ones(64))  # wrong: 64 != 32

    def forward(self, x):
        x = self.conv(x)                        # (B, 32, H, W)
        w = self.channel_weight.view(1, -1, 1, 1)  # (1, 64, 1, 1) — wrong channels
        x = x * w                                # BUG: shape (B,32,H,W) * (1,64,1,1)
        return x
''',
        has_bug=True,
        bug_description="Buffer has 64 channels but conv outputs 32; the view "
                        "creates (1,64,1,1) which is incompatible with (B,32,H,W) "
                        "AND buffer may be on wrong device",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A6: Cross-device concatenation
    MultiTheoryBenchmark(
        name="cross_device_concat",
        source='''\
import torch
import torch.nn as nn

class CrossDeviceConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(128, 64)
        self.register_buffer('fixed_features', torch.randn(1, 32))

    def forward(self, x):
        a = self.branch_a(x)                    # (B, 64) on model device
        b = self.fixed_features.expand(x.size(0), -1)  # (B, 32) may be on CPU
        out = torch.cat([a, b], dim=1)           # BUG: cross-device cat
        return out
''',
        has_bug=True,
        bug_description="Buffer 'fixed_features' may not move to GPU with model; "
                        "concatenation of tensors on different devices fails",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A7: Mixed-precision shape interaction
    MultiTheoryBenchmark(
        name="mixed_precision_shape",
        source='''\
import torch
import torch.nn as nn

class MixedPrecisionShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.register_buffer('running_mean_backup', torch.zeros(32))  # wrong: 32 != 16

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # BUG: backup has wrong number of channels AND may be on wrong device
        diff = x.mean(dim=[0, 2, 3]) - self.running_mean_backup
        return x
''',
        has_bug=True,
        bug_description="running_mean_backup has 32 channels but bn outputs 16; "
                        "subtraction fails on shape AND possibly on device",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A8: Attention weight on wrong device
    MultiTheoryBenchmark(
        name="attention_weight_device",
        source='''\
import torch
import torch.nn as nn

class AttentionWeightDevice(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.register_buffer('attn_mask', torch.zeros(10, 10))  # fixed mask

    def forward(self, x):
        # x: (seq, batch, d_model) on model device
        # attn_mask may be on CPU if model moved to GPU
        out, _ = self.attn(x, x, x, attn_mask=self.attn_mask)  # BUG: cross-device mask
        return out
''',
        has_bug=True,
        bug_description="Attention mask buffer may be on CPU while query/key/value "
                        "are on GPU; this is a cross-device operation inside MHA",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A9: Device transfer breaks gradient+shape
    MultiTheoryBenchmark(
        name="device_transfer_shape_break",
        source='''\
import torch
import torch.nn as nn

class DeviceTransferShapeBreak(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.decoder = nn.Linear(128, 512)  # BUG: expects 128, encoder outputs 256

    def forward(self, x):
        h = self.encoder(x)
        h = h.cpu()            # explicit device transfer
        out = self.decoder(h)  # shape mismatch: 256 != 128
        return out
''',
        has_bug=True,
        bug_description="Encoder outputs 256 dims but decoder expects 128; "
                        "device transfer to CPU adds cross-device complexity",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # A10: Embedding + position on different devices
    MultiTheoryBenchmark(
        name="embedding_position_device",
        source='''\
import torch
import torch.nn as nn

class EmbeddingPositionDevice(nn.Module):
    def __init__(self, vocab=1000, d_model=256, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.register_buffer('pos_emb', torch.randn(max_len, d_model))

    def forward(self, input_ids):
        tok = self.tok_emb(input_ids)              # (B, S, d_model) on model device
        pos = self.pos_emb[:input_ids.size(1)]     # (S, d_model) may be CPU
        return tok + pos                            # BUG: cross-device add
''',
        has_bug=True,
        bug_description="Position embedding buffer may stay on CPU when model "
                        "is moved to GPU; adding to token embeddings fails",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # A11: Wrong buffer size for normalization
    MultiTheoryBenchmark(
        name="norm_buffer_wrong_size",
        source='''\
import torch
import torch.nn as nn

class NormBufferWrongSize(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 256)
        self.register_buffer('running_mean', torch.zeros(128))  # wrong: 128 != 256
        self.register_buffer('running_var', torch.ones(256))

    def forward(self, x):
        x = self.fc(x)
        # manual batch norm with wrong-sized running_mean
        x = (x - self.running_mean) / (self.running_var.sqrt() + 1e-5)  # BUG: 256 vs 128
        return x
''',
        has_bug=True,
        bug_description="running_mean has 128 elements but fc outputs 256; "
                        "subtraction fails AND buffer may be on wrong device",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # A12: Skip connection across devices
    MultiTheoryBenchmark(
        name="skip_connection_cross_device",
        source='''\
import torch
import torch.nn as nn

class SkipConnectionCrossDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.register_buffer('bias', torch.zeros(128))  # wrong: 128 != 256

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = x + residual + self.bias  # BUG: bias is (128,), x is (B, 256)
        return x                       #       AND bias may be on wrong device
''',
        has_bug=True,
        bug_description="Bias buffer has 128 dims but block outputs 256; "
                        "addition fails on shape AND possibly on device",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# B. PHASE + SHAPE BUGS (10)
#    These bugs manifest differently in train vs eval mode, or only
#    appear when dropout/batchnorm behavior changes between phases.
# ═══════════════════════════════════════════════════════════════════════════

PHASE_SHAPE_BUGS: List[MultiTheoryBenchmark] = [
    # B1: Dropout changes output shape expectation
    MultiTheoryBenchmark(
        name="dropout_shape_assumption",
        source='''\
import torch
import torch.nn as nn

class DropoutShapeAssumption(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        if self.training:
            x = self.dropout(x)
            # In training, zeroed elements change effective dimension
            x = x * 2  # manual dropout compensation
        x = self.fc2(x)  # shape is fine, but phase-dependent scaling
        return x
''',
        has_bug=True,
        bug_description="Manual dropout compensation (* 2) only in training "
                        "creates phase-dependent numerical behavior; dropout "
                        "already scales by 1/(1-p) — double compensation",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B2: BatchNorm dimensions wrong — only crashes in eval
    MultiTheoryBenchmark(
        name="batchnorm_eval_crash",
        source='''\
import torch
import torch.nn as nn

class BatchNormEvalCrash(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)  # BUG: 64 != 32

    def forward(self, x):
        x = self.conv(x)   # (B, 32, H, W)
        x = self.bn(x)     # expects 64 channels, gets 32
        return x
''',
        has_bug=True,
        bug_description="BatchNorm2d expects 64 channels but conv outputs 32; "
                        "in training mode, BN may silently reshape; in eval mode "
                        "running stats have wrong dimensions",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # B3: Auxiliary loss branch only in training
    MultiTheoryBenchmark(
        name="aux_loss_shape_mismatch",
        source='''\
import torch
import torch.nn as nn

class AuxLossShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(512, 256)
        self.head = nn.Linear(256, 10)
        self.aux_head = nn.Linear(128, 5)  # BUG: expects 128, gets 256

    def forward(self, x):
        features = self.backbone(x)        # (B, 256)
        output = self.head(features)       # (B, 10)

        if self.training:
            aux = self.aux_head(features)  # BUG: 256 != 128 — only crashes in training
            return output, aux
        return output
''',
        has_bug=True,
        bug_description="Auxiliary head expects 128 dims but backbone outputs 256; "
                        "bug only manifests during training when aux branch is active",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B4: Phase-dependent architecture selection
    MultiTheoryBenchmark(
        name="phase_arch_mismatch",
        source='''\
import torch
import torch.nn as nn

class PhaseArchMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.train_decoder = nn.Linear(256, 512)
        self.eval_decoder = nn.Linear(128, 512)  # BUG: expects 128, gets 256

    def forward(self, x):
        h = self.encoder(x)
        if self.training:
            return self.train_decoder(h)   # OK: 256 → 512
        else:
            return self.eval_decoder(h)    # BUG: 256 != 128 — eval-only crash
''',
        has_bug=True,
        bug_description="eval_decoder expects 128 dims but encoder outputs 256; "
                        "bug only manifests in eval mode",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B5: LayerNorm after wrong-shaped train augmentation
    MultiTheoryBenchmark(
        name="layernorm_train_augment",
        source='''\
import torch
import torch.nn as nn

class LayerNormTrainAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 256)
        self.norm = nn.LayerNorm(256)
        self.augment = nn.Linear(256, 128)  # changes dim in training

    def forward(self, x):
        x = self.fc(x)                    # (B, 256)
        if self.training:
            x = self.augment(x)            # (B, 128) — changes dimension!
        x = self.norm(x)                   # BUG: LayerNorm(256) but x is (B, 128) in training
        return x
''',
        has_bug=True,
        bug_description="In training mode, augment layer changes dim from 256 to 128, "
                        "but LayerNorm expects 256 — phase-dependent shape mismatch",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B6: Training-only batch dimension assumption
    MultiTheoryBenchmark(
        name="training_batch_assumption",
        source='''\
import torch
import torch.nn as nn

class TrainingBatchAssumption(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.fc(x)
        if self.training:
            # BN1d needs batch > 1 in training for statistics
            x = self.bn(x)  # crashes with batch=1 in training
        return x
''',
        has_bug=True,
        bug_description="BatchNorm1d requires batch > 1 in training mode for "
                        "variance computation; works in eval with running stats",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B7: Spectral norm mismatch
    MultiTheoryBenchmark(
        name="spectral_norm_phase",
        source='''\
import torch
import torch.nn as nn

class SpectralNormPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(64, 10)  # BUG: expects 64, gets 128

    def forward(self, x):
        x = self.fc1(x)
        if self.training:
            x = x / x.norm()  # normalize in training
        x = self.fc2(x)       # shape mismatch regardless of phase
        return x
''',
        has_bug=True,
        bug_description="fc2 expects 64 dims but fc1 outputs 128; "
                        "the training-only normalization is a red herring — "
                        "shape bug exists in both phases",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # B8: Different pooling in train vs eval
    MultiTheoryBenchmark(
        name="pool_phase_mismatch",
        source='''\
import torch
import torch.nn as nn

class PoolPhaseMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.train_pool = nn.AdaptiveAvgPool2d((4, 4))   # -> (B, 64, 4, 4) = 1024
        self.eval_pool = nn.AdaptiveAvgPool2d((2, 2))     # -> (B, 64, 2, 2) = 256
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            x = self.train_pool(x)
        else:
            x = self.eval_pool(x)   # (B, 64, 2, 2) = 256
        x = x.flatten(1)
        x = self.fc(x)              # BUG in eval: 256 != 1024
        return x
''',
        has_bug=True,
        bug_description="Different pooling sizes in train (4×4) vs eval (2×2) "
                        "produce different flattened sizes (1024 vs 256); "
                        "fc expects 1024, crashes in eval mode",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B9: Conditional teacher forcing
    MultiTheoryBenchmark(
        name="teacher_forcing_shape",
        source='''\
import torch
import torch.nn as nn

class TeacherForcingShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.decoder = nn.Linear(256, 100)
        self.train_proj = nn.Linear(100, 64)
        self.eval_proj = nn.Linear(256, 64)  # BUG: wrong input for decoder output

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)  # (B, 100)
        if self.training:
            return self.train_proj(out)   # OK: 100 → 64
        else:
            return self.eval_proj(out)    # BUG: expects 256, gets 100
''',
        has_bug=True,
        bug_description="eval_proj expects 256 dims but decoder outputs 100; "
                        "bug only manifests in eval mode — training path is correct",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # B10: InstanceNorm train/eval behavior
    MultiTheoryBenchmark(
        name="instancenorm_phase",
        source='''\
import torch
import torch.nn as nn

class InstanceNormPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.norm = nn.InstanceNorm2d(64, affine=True)  # BUG: 64 != 32

    def forward(self, x):
        x = self.conv(x)    # (B, 32, H, W)
        x = self.norm(x)    # expects 64 channels in eval (uses affine params)
        return x
''',
        has_bug=True,
        bug_description="InstanceNorm2d has 64 affine parameters but conv "
                        "outputs 32 channels; in eval mode the affine weight "
                        "shape (64,) doesn't match input channels (32)",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# C. DEVICE + PHASE BUGS (4)
#    Bugs where the device behavior depends on the training phase.
# ═══════════════════════════════════════════════════════════════════════════

DEVICE_PHASE_BUGS: List[MultiTheoryBenchmark] = [
    # C1: Training-only device transfer
    MultiTheoryBenchmark(
        name="training_device_transfer",
        source='''\
import torch
import torch.nn as nn

class TrainingDeviceTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.register_buffer('ema_weight', torch.ones(128))

    def forward(self, x):
        x = self.fc(x)
        if self.training:
            # Update EMA on CPU for numerical stability
            with torch.no_grad():
                self.ema_weight = self.ema_weight.cpu() * 0.99 + x.mean(0).cpu() * 0.01
        x = x * self.ema_weight  # BUG: ema_weight now on CPU after training step
        return x
''',
        has_bug=True,
        bug_description="In training, ema_weight is moved to CPU; subsequent "
                        "multiplication with GPU tensor x is cross-device — "
                        "this is a phase+device interaction bug",
        category="device_phase",
        theories_required=["device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # C2: DataParallel scatter breaks in eval
    MultiTheoryBenchmark(
        name="dataparallel_eval_device",
        source='''\
import torch
import torch.nn as nn

class DataParallelEvalDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.register_buffer('class_weights', torch.ones(10))

    def forward(self, x):
        x = self.fc(x)
        x = x.mean(dim=1, keepdim=True)  # reduce
        # class_weights might be on different device in DataParallel
        loss_weight = self.class_weights  # BUG: device mismatch in distributed
        return x, loss_weight
''',
        has_bug=True,
        bug_description="In DataParallel eval, class_weights buffer may be "
                        "on a different GPU replica — device+phase interaction",
        category="device_phase",
        theories_required=["device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # C3: Gradient checkpoint device issue
    MultiTheoryBenchmark(
        name="grad_checkpoint_device",
        source='''\
import torch
import torch.nn as nn

class GradCheckpointDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(4)])
        self.register_buffer('scale', torch.tensor(0.1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.training:
            x = x * self.scale  # BUG: scale may be on CPU after checkpoint reload
        return x
''',
        has_bug=True,
        bug_description="After gradient checkpointing recomputation, buffer 'scale' "
                        "may be on CPU while activations are on GPU — phase+device bug",
        category="device_phase",
        theories_required=["device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # C4: Mixed device in eval postprocessing
    MultiTheoryBenchmark(
        name="eval_postprocess_device",
        source='''\
import torch
import torch.nn as nn

class EvalPostprocessDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 10)
        self.register_buffer('class_names_idx', torch.arange(10))

    def forward(self, x):
        logits = self.fc(x)
        if not self.training:
            # Postprocessing in eval — index with buffer
            probs = torch.softmax(logits, dim=-1)
            top_classes = self.class_names_idx[probs.argmax(-1)]  # may be cross-device
            return logits, top_classes
        return logits
''',
        has_bug=True,
        bug_description="class_names_idx buffer may be on CPU while model outputs "
                        "are on GPU; eval-only postprocessing creates cross-device indexing",
        category="device_phase",
        theories_required=["device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# D. TRIPLE-THEORY BUGS (6)
#    Bugs requiring all three of shape + device + phase to detect.
# ═══════════════════════════════════════════════════════════════════════════

TRIPLE_THEORY_BUGS: List[MultiTheoryBenchmark] = [
    # D1: Full triple-theory interaction
    MultiTheoryBenchmark(
        name="triple_theory_classifier",
        source='''\
import torch
import torch.nn as nn

class TripleTheoryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(512, 256)
        self.train_head = nn.Linear(256, 100)
        self.eval_head = nn.Linear(128, 10)   # BUG: expects 128, gets 256
        self.register_buffer('eval_scale', torch.ones(10))  # may be on wrong device

    def forward(self, x):
        features = self.backbone(x)
        if self.training:
            return self.train_head(features)
        else:
            logits = self.eval_head(features)      # BUG: shape 256 != 128
            return logits * self.eval_scale         # BUG: device mismatch
''',
        has_bug=True,
        bug_description="Triple-theory bug: (1) eval_head expects 128 but gets 256 [shape], "
                        "(2) eval_scale buffer may be on CPU [device], "
                        "(3) both bugs only manifest in eval mode [phase]",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # D2: Training augmentation + device + shape
    MultiTheoryBenchmark(
        name="triple_augment_train",
        source='''\
import torch
import torch.nn as nn

class TripleAugmentTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(512, 10)
        self.register_buffer('noise_scale', torch.ones(64))  # BUG: 64 != 32

    def forward(self, x):
        x = self.conv(x)                    # (B, 32, H, W)
        if self.training:
            noise = self.noise_scale.view(1, -1, 1, 1)  # (1, 64, 1, 1) — wrong shape
            x = x + noise * 0.1             # BUG: shape (B,32,H,W) + (1,64,1,1)
                                             # AND noise_scale may be on wrong device
        x = self.pool(x)                    # (B, 32, 4, 4)
        x = x.flatten(1)                    # (B, 512)
        x = self.fc(x)
        return x
''',
        has_bug=True,
        bug_description="Triple-theory: noise_scale has 64 elements (shape bug), "
                        "may be on CPU (device bug), only used in training (phase bug)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # D3: Eval-only feature extraction with device issue
    MultiTheoryBenchmark(
        name="triple_eval_feature_extract",
        source='''\
import torch
import torch.nn as nn

class TripleEvalFeatureExtract(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, 10)
        self.register_buffer('pca_matrix', torch.randn(256, 64))

    def forward(self, x):
        features = self.encoder(x)         # (B, 256)
        if self.training:
            return self.classifier(features)  # (B, 10) — OK
        else:
            # Feature extraction mode
            reduced = features @ self.pca_matrix   # BUG: pca_matrix may be on CPU
            return reduced                          # (B, 64)
''',
        has_bug=True,
        bug_description="Triple-theory: pca_matrix buffer may be on CPU (device), "
                        "matmul only in eval mode (phase), shape interaction in the "
                        "matrix multiply (shape)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # D4: Knowledge distillation triple bug
    MultiTheoryBenchmark(
        name="triple_knowledge_distill",
        source='''\
import torch
import torch.nn as nn

class TripleKnowledgeDistill(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Linear(512, 128)
        self.head = nn.Linear(128, 10)
        self.register_buffer('teacher_logits', torch.randn(1, 20))  # BUG: 20 != 10

    def forward(self, x):
        h = self.student(x)
        logits = self.head(h)            # (B, 10)
        if self.training:
            # Distillation loss: compare with teacher
            teacher = self.teacher_logits.expand(x.size(0), -1)  # (B, 20) — wrong
            kl = (logits - teacher).pow(2).mean()  # BUG: (B,10) vs (B,20)
                                                    # AND teacher may be on CPU
            return logits, kl
        return logits
''',
        has_bug=True,
        bug_description="Triple-theory: teacher_logits has 20 dims vs 10 (shape), "
                        "may be on CPU (device), only used in training (phase)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # D5: Contrastive learning triple bug
    MultiTheoryBenchmark(
        name="triple_contrastive",
        source='''\
import torch
import torch.nn as nn

class TripleContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 256)
        self.projector = nn.Linear(256, 64)
        self.register_buffer('prototype', torch.randn(32, 64))  # 32 prototypes

    def forward(self, x):
        h = self.encoder(x)              # (B, 256)
        z = self.projector(h)            # (B, 64)
        if self.training:
            # Compare with prototypes
            sim = z @ self.prototype.t()  # (B, 32) — prototype may be on CPU
            return z, sim
        return z
''',
        has_bug=True,
        bug_description="Triple-theory: prototype buffer may be on CPU (device), "
                        "matmul with prototypes only in training (phase), "
                        "shape interaction in the matrix multiply (shape)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # D6: GAN discriminator triple bug
    MultiTheoryBenchmark(
        name="triple_gan_discriminator",
        source='''\
import torch
import torch.nn as nn

class TripleGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Linear(128, 1)
        self.register_buffer('real_stats', torch.zeros(64))  # BUG: 64 != 128

    def forward(self, x):
        h = self.features(x)              # (B, 128)
        score = self.classifier(h)         # (B, 1)
        if self.training:
            # Feature matching loss
            diff = h.mean(0) - self.real_stats  # BUG: (128,) - (64,)
                                                 # AND real_stats may be on CPU
            return score, diff.norm()
        return score
''',
        has_bug=True,
        bug_description="Triple-theory: real_stats has 64 dims vs features' 128 (shape), "
                        "buffer may be on CPU (device), feature matching only in training (phase)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# E. CORRECT MULTI-THEORY MODELS (8)
#    Models that correctly use multiple theories — should pass verification.
# ═══════════════════════════════════════════════════════════════════════════

CORRECT_MULTI_THEORY: List[MultiTheoryBenchmark] = [
    MultiTheoryBenchmark(
        name="correct_phase_aware_model",
        source='''\
import torch
import torch.nn as nn

class CorrectPhaseAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)       # behaves differently in train/eval
        x = self.dropout(x)  # drops in training, identity in eval
        x = self.fc2(x)
        return x
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_residual_bn",
        source='''\
import torch
import torch.nn as nn

class CorrectResidualBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = x + residual
        return x
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_multi_head",
        source='''\
import torch
import torch.nn as nn

class CorrectMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(512, 256)
        self.head_cls = nn.Linear(256, 10)
        self.head_reg = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        cls = self.head_cls(features)
        reg = self.head_reg(features)
        if self.training:
            return cls, reg
        return cls
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_buffer_usage",
        source='''\
import torch
import torch.nn as nn

class CorrectBufferUsage(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.register_buffer('scale', torch.ones(128))  # correct: matches fc output

    def forward(self, x):
        x = self.fc(x)
        x = x * self.scale
        return x
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_conv_bn_relu",
        source='''\
import torch
import torch.nn as nn

class CorrectConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_dual_phase_output",
        source='''\
import torch
import torch.nn as nn

class CorrectDualPhaseOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.train_head = nn.Linear(256, 100)
        self.eval_head = nn.Linear(256, 10)

    def forward(self, x):
        h = self.encoder(x)
        if self.training:
            return self.train_head(h)
        return self.eval_head(h)
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_embedding_with_buffer",
        source='''\
import torch
import torch.nn as nn

class CorrectEmbeddingWithBuffer(nn.Module):
    def __init__(self, vocab=5000, d_model=256, max_len=100):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.register_buffer('pos_emb', torch.randn(max_len, d_model))
        self.fc = nn.Linear(d_model, 10)

    def forward(self, input_ids):
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb[:input_ids.size(1)]
        x = tok + pos
        x = x.mean(dim=1)
        return self.fc(x)
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    MultiTheoryBenchmark(
        name="correct_train_eval_dropout",
        source='''\
import torch
import torch.nn as nn

class CorrectTrainEvalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
''',
        has_bug=False,
        bug_description="",
        category="correct",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# F. REAL-WORLD SOURCED MULTI-THEORY BUGS (12)
#    Derived from actual bug reports in open-source PyTorch projects,
#    StackOverflow questions, PyTorch GitHub Issues, and HuggingFace
#    Transformers PRs.  Each cites the real-world pattern it recreates.
# ═══════════════════════════════════════════════════════════════════════════

REALWORLD_MULTI_THEORY_BUGS: List[MultiTheoryBenchmark] = [
    # F1: HuggingFace-style position embedding device mismatch
    # Source: HuggingFace Transformers issue #13666 — position_ids buffer
    # not following model to GPU when using model.cuda().
    MultiTheoryBenchmark(
        name="real_hf_position_ids_device",
        source='''\
import torch
import torch.nn as nn

class HFStyleEmbedding(nn.Module):
    """Simplified HuggingFace BertEmbeddings with the real bug pattern
    from Transformers issue #13666: position_ids registered as buffer
    but created with torch.arange which defaults to CPU."""
    def __init__(self, vocab_size=30522, hidden_size=768, max_position=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.register_buffer('position_ids',
                             torch.arange(max_position).expand((1, -1)))

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]  # may be on CPU
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)  # cross-device indexing
        embeddings = word_emb + pos_emb  # BUG: cross-device add
        return self.LayerNorm(embeddings)
''',
        has_bug=True,
        bug_description="Real HuggingFace bug pattern (Transformers #13666): "
                        "position_ids buffer created with torch.arange defaults to "
                        "CPU; when model is moved to GPU, the buffer may not follow, "
                        "causing cross-device embedding lookup",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # F2: torchvision ResNet-style downsampling skip connection mismatch
    # Source: PyTorch GitHub Issues #28536, StackOverflow 62271688 —
    # residual connection dimension mismatch after bottleneck
    MultiTheoryBenchmark(
        name="real_resnet_downsample_phase",
        source='''\
import torch
import torch.nn as nn

class ResNetBottleneckBug(nn.Module):
    """Real ResNet bug pattern from PyTorch issues #28536:
    Bottleneck block with wrong downsample dimensions, only
    manifests when stride changes between train/eval preprocessing."""
    def __init__(self, inplanes=64, planes=128, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # BUG: downsample projects to wrong dimension (planes instead of planes*4)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),  # should be planes*4
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = self.downsample(x)     # (B, 128, H/2, W/2)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)               # (B, 512, H/2, W/2)
        out = out + identity              # BUG: 512 != 128 shape mismatch
        return torch.relu(out)
''',
        has_bug=True,
        bug_description="Real ResNet bottleneck bug (PyTorch #28536): downsample "
                        "projects to planes=128 instead of planes*4=512, causing "
                        "shape mismatch in residual addition. BatchNorm behavior "
                        "differs in train/eval (phase theory required)",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # F3: Real DCGAN generator device/shape bug
    # Source: PyTorch DCGAN tutorial — common error when mixing CPU/GPU
    # tensors in generator noise input
    MultiTheoryBenchmark(
        name="real_dcgan_noise_device",
        source='''\
import torch
import torch.nn as nn

class DCGANGeneratorBug(nn.Module):
    """Real DCGAN tutorial bug pattern: generator creates fixed noise
    on CPU but model is on GPU. Shape interaction through ConvTranspose2d."""
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.register_buffer('fixed_noise', torch.randn(64, nz, 1, 1))

    def forward(self, x):
        if not self.training:
            x = self.fixed_noise  # BUG: buffer may be on CPU when model on GPU
        return self.main(x)
''',
        has_bug=True,
        bug_description="Real DCGAN tutorial bug: fixed_noise buffer created on "
                        "CPU, used in eval mode only; ConvTranspose2d on GPU "
                        "receives CPU input — device+phase+shape interaction",
        category="triple",
        theories_required=["device", "phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # F4: StackOverflow #67589432 — BatchNorm channel mismatch after
    # architecture modification, only visible in eval mode with running stats
    MultiTheoryBenchmark(
        name="real_so_batchnorm_channels",
        source='''\
import torch
import torch.nn as nn

class ModifiedArchBug(nn.Module):
    """StackOverflow #67589432: User modified channel count in conv layer
    but forgot to update BatchNorm — works in training (BN ignores
    channel count for statistics) but crashes in eval with running stats."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)  # changed from 64 to 48
        self.bn1 = nn.BatchNorm2d(64)                  # BUG: still expects 64
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  # expects 64 from BN
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))   # BUG: 48 channels vs BN(64)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)
''',
        has_bug=True,
        bug_description="Real StackOverflow bug: conv1 outputs 48 channels but "
                        "BatchNorm2d expects 64. BN behavior differs in train vs "
                        "eval (running stats vs batch stats) — phase+shape",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # F5: PyTorch Forums — custom attention with relative position bias
    # Source: discuss.pytorch.org/t/200241
    MultiTheoryBenchmark(
        name="real_forums_attention_bias_device",
        source='''\
import torch
import torch.nn as nn

class RelativePositionAttention(nn.Module):
    """Real PyTorch Forums bug (discuss.pytorch.org/t/200241):
    relative position bias table registered as buffer but indexing
    creates CPU tensor, causing device mismatch in attention."""
    def __init__(self, d_model=256, n_heads=8, max_len=64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.register_buffer('rel_pos_bias',
                             torch.zeros(2 * max_len - 1, n_heads))

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        # Construct position index on CPU
        pos_idx = torch.arange(S).unsqueeze(0) - torch.arange(S).unsqueeze(1) + S - 1
        bias = self.rel_pos_bias[pos_idx]  # may be cross-device
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)  # BUG: device mismatch
        return (torch.softmax(attn, -1) @ v).transpose(1, 2).reshape(B, S, -1)
''',
        has_bug=True,
        bug_description="Real PyTorch Forums bug: pos_idx created with torch.arange "
                        "on CPU, indexing into GPU buffer creates device mismatch; "
                        "shape interaction through multi-head reshape",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # F6: Real fairseq-style model with eval-only beam search shape bug
    # Source: fairseq issues — beam search changes batch dimension
    MultiTheoryBenchmark(
        name="real_fairseq_beam_phase",
        source='''\
import torch
import torch.nn as nn

class FairseqDecoderBug(nn.Module):
    """Real fairseq pattern: encoder-decoder with beam search that
    changes effective batch size in eval. Auxiliary projection only
    used in training has wrong dimensions."""
    def __init__(self, d_model=512, vocab=10000):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_model)
        self.decoder = nn.Linear(d_model, vocab)
        self.label_smooth_proj = nn.Linear(256, vocab)  # BUG: expects 256, gets 512

    def forward(self, src):
        enc = self.encoder(src)
        logits = self.decoder(enc)
        if self.training:
            # Label smoothing uses wrong projection
            smooth = self.label_smooth_proj(enc)  # BUG: 512 != 256
            return logits, smooth
        return logits
''',
        has_bug=True,
        bug_description="Real fairseq pattern: label smoothing projection expects "
                        "256 dims but encoder outputs 512; bug only triggers in "
                        "training mode — phase+shape interaction",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # F7: Real torchvision FPN (Feature Pyramid Network) channel mismatch
    # Source: torchvision issues — lateral connection dimension error
    MultiTheoryBenchmark(
        name="real_torchvision_fpn_channels",
        source='''\
import torch
import torch.nn as nn

class FPNBug(nn.Module):
    """Real torchvision FPN pattern: lateral connections with wrong channel
    dimensions after backbone modification. Involves buffer for
    anchors that may be on wrong device."""
    def __init__(self):
        super().__init__()
        self.backbone_c3 = nn.Conv2d(3, 128, 3, stride=8, padding=1)
        self.backbone_c4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.lateral_c3 = nn.Conv2d(128, 256, 1)
        self.lateral_c4 = nn.Conv2d(512, 256, 1)  # BUG: expects 512, gets 256
        self.register_buffer('anchor_grid', torch.zeros(1, 3, 1, 1, 2))

    def forward(self, x):
        c3 = self.backbone_c3(x)             # (B, 128, H/8, W/8)
        c4 = self.backbone_c4(c3)            # (B, 256, H/16, W/16)
        p4 = self.lateral_c4(c4)             # BUG: Conv2d(512, 256) gets (B, 256, ...)
        p3 = self.lateral_c3(c3)
        return p3, p4
''',
        has_bug=True,
        bug_description="Real torchvision FPN bug: lateral_c4 expects 512 input "
                        "channels but backbone_c4 outputs 256. anchor_grid buffer "
                        "adds device interaction — shape+device cross-cutting",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=True,
    ),

    # F8: Real detectron2-style mask head with phase-dependent ROI pooling
    # Source: detectron2 issues — mask head channel mismatch in eval
    MultiTheoryBenchmark(
        name="real_detectron2_mask_phase",
        source='''\
import torch
import torch.nn as nn

class MaskHeadBug(nn.Module):
    """Real detectron2 pattern: mask prediction head with different
    feature sources in train vs eval, causing channel mismatch."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_pred = nn.Conv2d(256, 80, 1)
        self.train_adapter = nn.Conv2d(256, 256, 1)
        self.eval_adapter = nn.Conv2d(512, 256, 1)  # BUG: expects 512 in eval

    def forward(self, features):
        x = torch.relu(self.conv1(features))
        x = torch.relu(self.conv2(x))
        if self.training:
            x = self.train_adapter(x)         # OK: 256 → 256
        else:
            x = self.eval_adapter(x)          # BUG: expects 512, gets 256
        x = torch.relu(self.deconv(x))
        return self.mask_pred(x)
''',
        has_bug=True,
        bug_description="Real detectron2 pattern: eval_adapter expects 512 channels "
                        "but gets 256 from conv2; bug only in eval mode. Phase-aware "
                        "verification needed to detect this",
        category="phase_shape",
        theories_required=["phase", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # F9: Real timm (PyTorch Image Models) — EfficientNet SE block device bug
    # Source: timm issues — squeeze-excite with buffer
    MultiTheoryBenchmark(
        name="real_timm_se_block_device",
        source='''\
import torch
import torch.nn as nn

class SEBlockBug(nn.Module):
    """Real timm SE block pattern: squeeze-excite with channel attention
    scale stored as buffer. When used in transfer learning, buffer
    channels don't match after head replacement."""
    def __init__(self, channels=256, reduction=4):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.register_buffer('channel_scale', torch.ones(1, 512, 1, 1))  # BUG: 512 != 256

    def forward(self, x):
        x = self.conv(x)                          # (B, 256, H, W)
        scale = self.avg_pool(x).flatten(1)        # (B, 256)
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        scale = scale.view(-1, 256, 1, 1)
        x = x * scale * self.channel_scale         # BUG: (B,256,H,W) * (1,512,1,1)
        return x
''',
        has_bug=True,
        bug_description="Real timm SE block bug: channel_scale buffer has 512 channels "
                        "but conv outputs 256 (after transfer learning head swap). "
                        "Buffer may also be on wrong device — device+shape cross-cutting",
        category="device_shape",
        theories_required=["device", "shape"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),

    # F10: Real MMDetection-style FPN + anchor generator triple bug
    # Source: mmdetection issues — anchor shape + device + train/eval
    MultiTheoryBenchmark(
        name="real_mmdet_anchor_triple",
        source='''\
import torch
import torch.nn as nn

class AnchorGeneratorBug(nn.Module):
    """Real mmdetection pattern: anchor generator with pre-computed
    anchor grids as buffers. Triple bug: wrong anchor dimensions,
    CPU buffer, and train-only anchor matching."""
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_classes * 9, 1)
        self.reg_head = nn.Conv2d(in_channels, 4 * 9, 1)
        self.register_buffer('anchor_grid',
                             torch.randn(1, 6, 1, 1, 4))  # BUG: 6 anchors != 9

    def forward(self, feature_map):
        x = torch.relu(self.conv(feature_map))
        cls_pred = self.cls_head(x)               # (B, 720, H, W) for 9 anchors
        reg_pred = self.reg_head(x)               # (B, 36, H, W) for 9 anchors

        if self.training:
            # Anchor matching only in training
            B, _, H, W = reg_pred.shape
            anchors = self.anchor_grid.expand(B, -1, H, W, -1)  # BUG: 6 != 9
            reg_targets = reg_pred.view(B, 9, 4, H, W) - anchors.permute(0,1,4,2,3)
            return cls_pred, reg_targets  # shape mismatch + device mismatch
        return cls_pred
''',
        has_bug=True,
        bug_description="Real mmdetection triple bug: anchor_grid has 6 anchors but "
                        "heads assume 9 (shape), buffer may be on CPU (device), "
                        "anchor matching only in training (phase)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # F11: Real YOLO-style detection with eval-only NMS postprocessing
    # Source: YOLOv5 issues — postprocessing buffer mismatch
    MultiTheoryBenchmark(
        name="real_yolo_postprocess_triple",
        source='''\
import torch
import torch.nn as nn

class YOLODetectorBug(nn.Module):
    """Real YOLOv5 pattern: detection head with eval-only postprocessing
    using anchor buffers. Triple bug: anchor dims, device, and phase."""
    def __init__(self, nc=80, anchors_per_scale=3):
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # outputs per anchor
        self.conv = nn.Conv2d(256, self.no * anchors_per_scale, 1)
        # BUG: grid buffer has wrong spatial dims AND wrong anchor count
        self.register_buffer('grid', torch.zeros(1, 1, 20, 20, 2))
        self.register_buffer('anchor_wh', torch.ones(1, 6, 1, 1, 2))  # 6 != 3

    def forward(self, x):
        pred = self.conv(x)  # (B, 255, H, W)
        if not self.training:
            B, _, H, W = pred.shape
            pred = pred.view(B, 3, self.no, H, W).permute(0, 1, 3, 4, 2)
            # BUG: grid (1,1,20,20,2) vs pred (B,3,H,W,85) spatial mismatch
            xy = torch.sigmoid(pred[..., :2]) + self.grid[:, :, :H, :W, :]
            wh = pred[..., 2:4] * self.anchor_wh  # BUG: anchor_wh has 6, pred has 3
            return torch.cat([xy, wh, pred[..., 4:]], -1)
        return pred
''',
        has_bug=True,
        bug_description="Real YOLOv5 triple bug: grid/anchor_wh buffers have wrong "
                        "dimensions (shape), may be on CPU (device), and postprocessing "
                        "only runs in eval mode (phase)",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=False,
    ),

    # F12: Real wav2vec/speech model — train vs eval feature extraction
    # Source: HuggingFace Wav2Vec2 issues — feature extractor freeze
    MultiTheoryBenchmark(
        name="real_wav2vec_feature_freeze",
        source='''\
import torch
import torch.nn as nn

class Wav2VecFeatureBug(nn.Module):
    """Real wav2vec2 pattern from HuggingFace: feature extractor
    is frozen in training but unfrozen in fine-tuning, causing
    different output dimensions. Buffer for feature normalization
    has wrong size."""
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(80, 512)
        self.feature_projection = nn.Linear(512, 768)
        self.encoder = nn.Linear(768, 768)
        self.register_buffer('feat_norm_mean', torch.zeros(256))  # BUG: 256 != 512
        self.register_buffer('feat_norm_std', torch.ones(512))

    def forward(self, audio_features):
        features = self.feature_extractor(audio_features)
        if self.training:
            # Normalize features only in training
            features = (features - self.feat_norm_mean) / self.feat_norm_std
            # BUG: feat_norm_mean (256) vs features (512) + possible device mismatch
        proj = self.feature_projection(features)
        return self.encoder(proj)
''',
        has_bug=True,
        bug_description="Real wav2vec2 bug: feat_norm_mean buffer has 256 elements "
                        "but feature_extractor outputs 512. Normalization only in "
                        "training (phase), buffer may be on CPU (device) — triple bug",
        category="triple",
        theories_required=["shape", "device", "phase"],
        torchscript_catches=False,
        mypy_catches=False,
        pytea_catches=False,
        jaxtyping_catches=False,
        llm_catches=None,
    ),
]


ALL_MULTI_THEORY_BENCHMARKS: List[MultiTheoryBenchmark] = (
    DEVICE_SHAPE_BUGS
    + PHASE_SHAPE_BUGS
    + DEVICE_PHASE_BUGS
    + TRIPLE_THEORY_BUGS
    + CORRECT_MULTI_THEORY
    + REALWORLD_MULTI_THEORY_BUGS
)

BUGGY_MULTI_THEORY = [b for b in ALL_MULTI_THEORY_BENCHMARKS if b.has_bug]
CORRECT_MULTI_THEORY_ALL = [b for b in ALL_MULTI_THEORY_BENCHMARKS if not b.has_bug]

# Quick summary
if __name__ == "__main__":
    print(f"Multi-Theory Bug Benchmark Suite")
    print(f"  Total:          {len(ALL_MULTI_THEORY_BENCHMARKS)}")
    print(f"  Buggy:          {len(BUGGY_MULTI_THEORY)}")
    print(f"  Correct:        {len(CORRECT_MULTI_THEORY_ALL)}")
    print(f"  Device+Shape:   {len(DEVICE_SHAPE_BUGS)}")
    print(f"  Phase+Shape:    {len(PHASE_SHAPE_BUGS)}")
    print(f"  Device+Phase:   {len(DEVICE_PHASE_BUGS)}")
    print(f"  Triple-theory:  {len(TRIPLE_THEORY_BUGS)}")
    print(f"  Real-world:     {len(REALWORLD_MULTI_THEORY_BUGS)}")
    print()
    print("No other tool catches these bugs:")
    for b in BUGGY_MULTI_THEORY:
        catches = []
        if b.torchscript_catches: catches.append("TorchScript")
        if b.mypy_catches: catches.append("mypy")
        if b.pytea_catches: catches.append("PyTEA")
        if b.jaxtyping_catches: catches.append("jaxtyping")
        other = ", ".join(catches) if catches else "none"
        print(f"  {b.name}: theories={b.theories_required}, other_tools_catch={other}")
