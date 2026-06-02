"""Regression tests for soundness gaps found by mutation testing (Step 112).

Mutation testing of clean models surfaced three real holes that are fixed here:

1. A genuine shape mismatch *inside* an ``nn.Sequential`` was silently abstained
   to SAFE (the per-sub-layer transfer error was swallowed as "opaque").
2. Source-level dtype casts (``x.long()`` etc.) were not parsed, so the dtype
   domain could not see a known-integer activation.
3. A known-integer tensor fed into a floating layer (Linear/Conv/recurrent/
   attention/transformer/norm), including inside a Sequential, was not flagged.

All of these are *sound* detections: the mutated models genuinely raise under
eager PyTorch, and the corresponding clean models stay SAFE (no false alarms).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402


def _verdict(src, shapes, mode="sound"):
    return str(verify_architecture(src, input_shapes=shapes,
                                   soundness_mode=mode).verdict)


_SEQ_BROKEN = (
    "import torch\nimport torch.nn as nn\n\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(nn.Linear(50, 33), nn.ReLU(), "
    "nn.Linear(32, 10))\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n"
)
_SEQ_CLEAN = (
    "import torch\nimport torch.nn as nn\n\n"
    "class Net(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(nn.Linear(50, 32), nn.ReLU(), "
    "nn.Linear(32, 10))\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n"
)


def test_broken_sequential_shape_is_unsafe():
    assert _verdict(_SEQ_BROKEN, {"x": (8, 50)}) == "UNSAFE"


def test_clean_sequential_shape_is_safe():
    assert _verdict(_SEQ_CLEAN, {"x": (8, 50)}) == "SAFE"


def test_dtype_cast_into_sequential_is_unsafe():
    bug = _SEQ_CLEAN.replace("        return self.net(x)",
                             "        x = x.long()\n        return self.net(x)")
    assert _verdict(bug, {"x": (8, 50)}) == "UNSAFE"


def test_dtype_cast_into_direct_linear_is_unsafe():
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(50, 10)\n"
        "    def forward(self, x):\n"
        "        x = x.long()\n"
        "        return self.l(x)\n"
    )
    assert _verdict(src, {"x": (8, 50)}) == "UNSAFE"


def test_float_cast_into_linear_stays_safe():
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.l = nn.Linear(50, 10)\n"
        "    def forward(self, x):\n"
        "        x = x.float()\n"
        "        return self.l(x)\n"
    )
    assert _verdict(src, {"x": (8, 50)}) == "SAFE"


def test_embedding_first_sequential_int_input_is_safe():
    # Integer indices into an Embedding are correct; output is floating, so the
    # downstream Linear is fine. Must NOT false-alarm.
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.e = nn.Embedding(100, 16)\n"
        "        self.l = nn.Linear(16, 10)\n"
        "    def forward(self, x):\n"
        "        return self.l(self.e(x))\n"
    )
    assert _verdict(src, {"x": (8,)}) == "SAFE"


def test_int_input_into_attention_is_unsafe():
    src = (
        "import torch\nimport torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.enc = nn.TransformerEncoderLayer(\n"
        "            d_model=64, nhead=8, dim_feedforward=128, batch_first=True)\n"
        "    def forward(self, x):\n"
        "        x = x.long()\n"
        "        return self.enc(x)\n"
    )
    assert _verdict(src, {"x": (8, 20, 64)}) == "UNSAFE"
