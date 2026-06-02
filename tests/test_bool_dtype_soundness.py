"""Locks in the Step 117 bool-input soundness fix in model_checker.

A boolean tensor fed into a float-parameter layer (Linear / Conv / Sequential of
them) is a guaranteed eager-torch error: such layers require a floating-point
input. Previously TensorGuard recognised int casts but not ``bool`` (bool was in
neither the int nor the float dtype set), so it silently returned SAFE -- a real
soundness gap surfaced by the per-domain ablation corpus. These tests prove the
gap is closed and that clean float inputs are *not* false-alarmed.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402


def _verify(source, shape=(4, 8)):
    return verify_architecture(
        source,
        input_shapes={"x": tuple(shape)},
        soundness_mode="sound",
        max_cegar_iterations=0,
    )


BOOL_LINEAR = """
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = x.bool()
        return self.fc(x)
"""

BOOL_CONV = """
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 3)

    def forward(self, x):
        x = x.bool()
        return self.conv(x)
"""

BOOL_SEQUENTIAL = """
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 4), nn.ReLU())

    def forward(self, x):
        x = x.bool()
        return self.net(x)
"""

CLEAN_FLOAT = """
import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)
"""


def test_bool_into_linear_is_unsafe():
    res = _verify(BOOL_LINEAR)
    assert res.bug_count > 0
    assert res.verdict != "SAFE"


def test_bool_into_conv_is_unsafe():
    res = _verify(BOOL_CONV, shape=(1, 3, 16, 16))
    assert res.bug_count > 0


def test_bool_into_sequential_is_unsafe():
    res = _verify(BOOL_SEQUENTIAL)
    assert res.bug_count > 0


def test_clean_float_not_false_alarmed():
    res = _verify(CLEAN_FLOAT)
    assert res.bug_count == 0
    assert res.verdict == "SAFE"
