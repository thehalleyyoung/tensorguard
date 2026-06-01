"""Authoritative definition of the TensorGuard frozen ground-truth benchmark corpus.

This module is the single source of truth for ``real_benchmarks/``. Each entry
describes one self-contained ``nn.Module`` together with its ground-truth label
(``clean`` or ``buggy``), the concrete input shapes used to analyze it, the
verdict TensorGuard is expected to produce, and provenance.

The corpus is *frozen*: ``build_manifest.py`` materializes every entry into a
standalone repro file under ``clean/`` or ``buggy/`` and records a SHA-256 hash
of each file in ``manifest.json``. ``load.py`` re-verifies those hashes so the
corpus cannot silently drift. Changing the corpus requires bumping
``CORPUS_VERSION`` and regenerating the manifest.

Provenance types
----------------
* ``pytorch_issue``  -- the bug pattern is drawn from a real pytorch/pytorch
  issue (``source_url`` points at it). The repro is a minimal, idiomatic
  ``nn.Module`` that exhibits the same class of error.
* ``canonical_pattern`` -- a real-world failure mode that is ubiquitous in
  practice (e.g. CPU/CUDA device mismatch, gradient detach in a trainable path)
  but is not tied to a single tracked issue. ``source_url`` is ``null``.
* ``canonical_clean`` -- an idiomatic, correct architecture that should verify
  SAFE. ``source_url`` is ``null``.
"""

from __future__ import annotations

CORPUS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Clean models (ground truth: SAFE)
# --------------------------------------------------------------------------- #
_CLEAN = [
    {
        "id": "clean_mlp",
        "domain": "shape",
        "note": "Two-layer MLP with matching feature dimensions.",
        "input_shapes": {"x": (32, 784)},
        "source": '''import torch
import torch.nn as nn


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
''',
    },
    {
        "id": "clean_cnn",
        "domain": "shape",
        "note": "Small convolutional classifier with a correctly sized head.",
        "input_shapes": {"x": (8, 3, 32, 32)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
''',
    },
    {
        "id": "clean_resblock",
        "domain": "shape",
        "note": "Residual block; the skip connection shape matches the main path.",
        "input_shapes": {"x": (4, 64, 16, 16)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(h + x)
''',
    },
    {
        "id": "clean_layernorm_mlp",
        "domain": "shape",
        "note": "Token-wise MLP with LayerNorm over the feature dimension.",
        "input_shapes": {"x": (16, 32, 128)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        return x + self.fc2(h)
''',
    },
    {
        "id": "clean_conv_bn_pool",
        "domain": "shape",
        "note": "Conv -> BatchNorm -> ReLU -> MaxPool -> flatten -> Linear, all sized correctly.",
        "input_shapes": {"x": (4, 1, 28, 28)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)
''',
    },
    {
        "id": "clean_self_attention",
        "domain": "shape",
        "note": "Single-head scaled dot-product attention with consistent projections.",
        "input_shapes": {"x": (2, 10, 64)},
        "source": '''import torch
import torch.nn as nn


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 64)
        self.v = nn.Linear(64, 64)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
''',
    },
    {
        "id": "clean_groupnorm",
        "domain": "shape",
        "note": "Conv followed by GroupNorm whose channel count divides evenly.",
        "input_shapes": {"x": (2, 16, 8, 8)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.gn = nn.GroupNorm(4, 16)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)))
''',
    },
    {
        "id": "clean_dropout_mlp",
        "domain": "phase",
        "note": "Regression MLP using Dropout; correct in both train and eval phases.",
        "input_shapes": {"x": (64, 100)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
''',
    },
]


# --------------------------------------------------------------------------- #
# Buggy models (ground truth: UNSAFE)
# --------------------------------------------------------------------------- #
_BUGGY = [
    {
        "id": "buggy_linear_inout_mismatch",
        "domain": "shape",
        "category": "linear_inout_mismatch",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/179789",
        "note": "fc2 expects 128 in-features but receives 256 from fc1.",
        "expected_error_substring": "mat1 and mat2 shapes cannot be multiplied",
        "input_shapes": {"x": (32, 784)},
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 10)  # BUG: should be Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
''',
    },
    {
        "id": "buggy_view_total_size",
        "domain": "shape",
        "category": "view_reshape_total_size",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/177691",
        "note": "view target has a different total element count than the input.",
        "expected_error_substring": "is invalid for input of size",
        "input_shapes": {"x": (2048, 384)},
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def forward(self, x):
        return x.view(2048, 640)  # BUG: 2048*640 != 2048*384
''',
    },
    {
        "id": "buggy_conv_channel_mismatch",
        "domain": "shape",
        "category": "conv_channel_mismatch",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/179931",
        "note": "conv2 expects 8 input channels but conv1 emits 16.",
        "expected_error_substring": "channels",
        "input_shapes": {"x": (8, 3, 32, 32)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)  # BUG: in_channels should be 16

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))
''',
    },
    {
        "id": "buggy_flatten_fc_mismatch",
        "domain": "shape",
        "category": "linear_inout_mismatch",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/172739",
        "note": "Flattened conv features are 16*8*8=1024 but the head expects 999.",
        "expected_error_substring": "mat1 and mat2 shapes cannot be multiplied",
        "input_shapes": {"x": (8, 3, 8, 8)},
        "source": '''import torch
import torch.nn as nn
import torch.nn.functional as F


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(999, 10)  # BUG: flattened size is 16*8*8 = 1024

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)
''',
    },
    {
        "id": "buggy_cat_dim_mismatch",
        "domain": "shape",
        "category": "broadcasting",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/175683",
        "note": "Concatenation along dim 0 then added to one branch whose dim-0 differs.",
        "expected_error_substring": "size of tensor",
        "input_shapes": {"x": (3, 8)},
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 4)
        self.b = nn.Linear(8, 6)

    def forward(self, x):
        ha = self.a(x)
        hb = self.b(x)
        # cat along dim 0 -> (6, 4) but ha is (3, 4): add broadcasts incorrectly
        return torch.cat([ha, hb], dim=0) + ha
''',
    },
    {
        "id": "buggy_matmul_inner_mismatch",
        "domain": "shape",
        "category": "linear_inout_mismatch",
        "provenance_type": "pytorch_issue",
        "source_url": "https://github.com/pytorch/pytorch/issues/176230",
        "note": "Inner dimensions of the two matmul operands do not agree (8 vs 16).",
        "expected_error_substring": "mat1 and mat2 shapes cannot be multiplied",
        "input_shapes": {"x": (4, 8), "y": (16, 4)},
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)  # BUG: (4,8) @ (16,4) inner dims 8 != 16
''',
    },
    {
        "id": "buggy_device_mismatch",
        "domain": "device",
        "category": "device_mismatch",
        "provenance_type": "canonical_pattern",
        "source_url": None,
        "note": "A CUDA buffer is added to a CPU activation -- the canonical "
                "'Expected all tensors to be on the same device' failure. The "
                "device mismatch only raises at runtime on a CUDA-enabled host; "
                "TensorGuard detects it statically without any GPU.",
        "expected_error_substring": "Expected all tensors to be on the same device",
        "input_shapes": {"x": (4, 8)},
        "check_devices": True,
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros(8, device="cuda"))
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x) + self.bias  # BUG: cuda buffer added to cpu activation
''',
    },
    {
        "id": "buggy_gradient_detach",
        "domain": "gradient",
        "category": "gradient_detach",
        "provenance_type": "canonical_pattern",
        "source_url": None,
        "note": "An intermediate activation is detached, silently severing the "
                "gradient path to fc1 -- the canonical 'parameters never update' "
                "bug. This is a SILENT bug: it raises no runtime exception, so "
                "runtime testing misses it; TensorGuard flags it statically.",
        "expected_error_substring": None,
        "input_shapes": {"x": (4, 8)},
        "check_gradients": True,
        "source": '''import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        h = self.fc1(x)
        h = h.detach()  # BUG: severs gradient flow to fc1
        return self.fc2(h)
''',
    },
]


def all_entries():
    """Return the full corpus as a list of normalized entry dicts."""
    entries = []
    for e in _CLEAN:
        entries.append(_normalize(e, label="clean"))
    for e in _BUGGY:
        entries.append(_normalize(e, label="buggy"))
    return entries


def _normalize(entry, label):
    out = dict(entry)
    out["label"] = label
    out.setdefault("category", "clean" if label == "clean" else "unknown")
    out.setdefault("provenance_type", "canonical_clean" if label == "clean" else "unknown")
    out.setdefault("source_url", None)
    out.setdefault("expected_error_substring", None)
    out.setdefault("check_devices", False)
    out.setdefault("check_gradients", False)
    out["expected_verdict"] = "SAFE" if label == "clean" else "UNSAFE"
    # input_shapes -> JSON-friendly lists
    out["input_shapes"] = {k: list(v) for k, v in entry["input_shapes"].items()}
    return out
