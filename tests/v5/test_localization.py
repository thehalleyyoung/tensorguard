"""Unit tests for src.v5.localization."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from src.v5.localization import localize, enrich_result, _extract_offending_vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(bugs):
    """Build a minimal AnalysisResult-like object."""
    import types

    class Loc:
        def __init__(self, line):
            self.line = line

    class Bug:
        def __init__(self, message, line):
            self.message = message
            self.location = Loc(line)

    r = types.SimpleNamespace()
    r.bugs = [Bug(msg, ln) for msg, ln in bugs]
    r.counterexample = None
    return r


# ---------------------------------------------------------------------------
# Test 1: view mismatch on a known line
# ---------------------------------------------------------------------------

def test_view_mismatch_localization():
    """localize should return the line that contains .view()."""
    source = """\
import torch
import torch.nn as nn

class BuggyModule(nn.Module):
    def forward(self, x):
        y = x.reshape(32, 64)
        return y.view(32, 128)
"""
    msg = "[SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(32, 64)) to (32, 128)"
    line = localize(source, msg, None)
    # .view() is on line 7
    assert line is not None
    assert abs(line - 7) <= 2, f"Expected ~7, got {line}"


# ---------------------------------------------------------------------------
# Test 2: Conv2d channel mismatch
# ---------------------------------------------------------------------------

def test_conv2d_channel_mismatch():
    """localize should find the Conv2d layer definition."""
    source = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)

    def forward(self, x):
        return self.conv(x)
"""
    msg = "[SHAPE-INCOMPATIBLE] Conv2d: expected input with 3 channels, got 4"
    line = localize(source, msg, None)
    assert line is not None
    # Conv2d is on line 7
    assert abs(line - 7) <= 2, f"Expected ~7, got {line}"


# ---------------------------------------------------------------------------
# Test 3: Linear in-feature mismatch
# ---------------------------------------------------------------------------

def test_linear_in_feature_mismatch():
    """localize should find the Linear layer definition."""
    source = """\
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 32)

    def forward(self, x):
        return self.fc(x)
"""
    msg = "[SHAPE-INCOMPATIBLE] mat1 and mat2 shapes cannot be multiplied (8x64 and 128x32)"
    line = localize(source, msg, None)
    assert line is not None
    # Linear is on line 7
    assert abs(line - 7) <= 2, f"Expected ~7, got {line}"


# ---------------------------------------------------------------------------
# Test 4: broadcast mismatch
# ---------------------------------------------------------------------------

def test_broadcast_mismatch():
    """localize should find the element-wise operation line."""
    source = """\
import torch
import torch.nn as nn

class ScaleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(128, 2))

    def forward(self, x):
        return x * self.scale
"""
    msg = "[SHAPE-INCOMPATIBLE] must match the size of tensor a (16) at non-singleton dimension 2"
    line = localize(source, msg, None)
    assert line is not None
    # multiplication on line 10
    assert abs(line - 10) <= 2, f"Expected ~10, got {line}"


# ---------------------------------------------------------------------------
# Test 5: no-op — empty message
# ---------------------------------------------------------------------------

def test_empty_message_returns_none():
    """localize with empty message should return None."""
    source = "x = 1\n"
    result = localize(source, "", None)
    assert result is None


# ---------------------------------------------------------------------------
# Test 6: counterexample line takes priority over AST
# ---------------------------------------------------------------------------

def test_counterexample_line_used_when_populated():
    """When counterexample has a non-zero line, localize should prefer it."""
    source = """\
import torch
x = torch.randn(2, 3, 4)
y = x.view(2, 3, 5)
"""
    msg = "[SHAPE-INCOMPATIBLE] Reshape incompatible"
    ce = {"violations": [{"kind": "shape_incompatible", "message": msg, "line": 3}]}
    line = localize(source, msg, ce)
    assert line == 3


# ---------------------------------------------------------------------------
# Test 7: multi-statement chain — bug originates 5 lines back
# ---------------------------------------------------------------------------

def test_multistatement_chain_finds_originating_line():
    """When reshape follows several transformations, localize finds the reshape."""
    source = """\
import torch
import torch.nn as nn

class Pipeline(nn.Module):
    def forward(self, x):
        a = x + 1          # line 6
        b = a * 2          # line 7
        c = b - 1          # line 8
        d = c.reshape(4, 4)# line 9
        e = d.view(4, 8)   # line 10  <-- total-size mismatch
        return e
"""
    msg = "[VIEW] shape '[4, 8]' is invalid for input of size 16"
    line = localize(source, msg, None)
    assert line is not None
    # view is on line 10 (reshape on 9 also valid); within ±2
    assert abs(line - 10) <= 2 or abs(line - 9) <= 2, f"Expected ~10, got {line}"


# ---------------------------------------------------------------------------
# Test 8: enrich_result patches zero-line bugs but not valid-line bugs
# ---------------------------------------------------------------------------

def test_enrich_result_patches_zero_line():
    source = """\
import torch
import torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        return x.view(10, 20)
"""
    result = _make_result([
        ("[SHAPE-INCOMPATIBLE] cannot reshape", 0),   # should be patched
        ("[SHAPE-INCOMPATIBLE] another bug", 4),      # line 4 is valid → keep
    ])
    enrich_result(result, source)
    # First bug should be localized to the .view() line (5)
    assert result.bugs[0].location.line > 0
    # Second bug should remain unchanged
    assert result.bugs[1].location.line == 4


# ---------------------------------------------------------------------------
# Test 9: _extract_offending_vars coverage
# ---------------------------------------------------------------------------

def test_extract_offending_vars_view():
    msg = "[SHAPE-INCOMPATIBLE] Reshape incompatible: cannot reshape TensorShape(dims=(2048, 384)) to (2048, 640)"
    info = _extract_offending_vars(msg)
    # shapes should contain at least one entry
    assert any("2048" in s for s in info["shapes"])


def test_extract_offending_vars_symbolic():
    msg = "[VIEW] shape '[B, 64]' is invalid for input of size B*128"
    info = _extract_offending_vars(msg)
    # Should detect 'B' as a symbolic dim
    assert "B" in info["dims"] or any("B" in s for s in info["shapes"])
