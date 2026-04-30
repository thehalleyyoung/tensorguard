"""Tests for newly-wired operators in TensorGuard's shape analyzer.

Each correct-usage test asserts that verify_architecture reports no
shape-incompatibility bugs.  A handful of tests at the end inject a
deliberate shape bug and assert TensorGuard catches it.
"""

from __future__ import annotations

import textwrap

import pytest

from src.api import verify_architecture
from src.tensor_shapes import TensorShapeAnalyzer


def _shapes(src: str) -> dict:
    a = TensorShapeAnalyzer()
    r = a.analyze_source(textwrap.dedent(src))
    return {k: tuple(d.value for d in v.dims) for k, v in r.shapes.items()}


def _ok(src: str, input_shapes: dict) -> None:
    """Assert verify_architecture finds no shape-incompatibility bugs."""
    r = verify_architecture(textwrap.dedent(src), input_shapes=input_shapes)
    sh = [b for b in r.bugs if "SHAPE" in b.message.upper()]
    assert sh == [], f"unexpected shape bugs:\n" + "\n".join(b.message for b in sh)


def _has_bug(src: str, input_shapes: dict) -> None:
    r = verify_architecture(textwrap.dedent(src), input_shapes=input_shapes)
    assert any("SHAPE" in b.message.upper() for b in r.bugs), \
        "expected a shape bug but found none"


# ─────────────────────────────────────────────────────────────────────
# Indexing / gather family
# ─────────────────────────────────────────────────────────────────────

def test_gather_attribute_call():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            idx = torch.zeros(2, 3, 4)
            y = x.gather(1, idx)
            return y
    """)
    assert sh["y"] == (2, 3, 4)


def test_gather_free_function():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            idx = torch.zeros(2, 5, 4)
            y = torch.gather(x, 1, idx)
            return y
    """)
    assert sh["y"] == (2, 5, 4)


def test_scatter_shape_preserve():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(4, 6)
            src = torch.zeros(4, 2)
            idx = torch.zeros(4, 2)
            y = x.scatter(1, idx, src)
            return y
    """)
    assert sh["y"] == (4, 6)


def test_scatter_add_shape_preserve():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(4, 6)
            src = torch.zeros(4, 2)
            idx = torch.zeros(4, 2)
            y = x.scatter_add(1, idx, src)
            return y
    """)
    assert sh["y"] == (4, 6)


def test_index_select_replaces_dim():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 8, 4)
            idx = torch.zeros(3)
            y = x.index_select(1, idx)
            return y
    """)
    assert sh["y"][0] == 2 and sh["y"][2] == 4
    # dim-1 must be 3 (size of idx)
    assert sh["y"][1] == 3


def test_index_select_free_function():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 8, 4)
            idx = torch.zeros(5)
            y = torch.index_select(x, 1, idx)
            return y
    """)
    assert sh["y"][1] == 5


def test_masked_select_returns_1d():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            mask = torch.zeros(2, 3, 4)
            y = x.masked_select(mask)
            return y
    """)
    assert len(sh["y"]) == 1


def test_take_along_dim_returns_index_shape():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(4, 8)
            idx = torch.zeros(4, 3)
            y = torch.take_along_dim(x, idx, 1)
            return y
    """)
    assert sh["y"] == (4, 3)


def test_narrow_replaces_dim():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 16, 4)
            y = x.narrow(1, 2, 5)
            return y
    """)
    assert sh["y"] == (2, 5, 4)


def test_roll_shape_preserve():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x.roll(1, 0)
            return y
    """)
    assert sh["y"] == (2, 3, 4)


def test_flip_shape_preserve():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x.flip(1)
            return y
    """)
    assert sh["y"] == (2, 3, 4)


def test_repeat_interleave_scales_dim():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 4, 8)
            y = x.repeat_interleave(3, dim=1)
            return y
    """)
    assert sh["y"] == (2, 12, 8)


def test_broadcast_to_returns_given_shape():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(1, 4)
            y = x.broadcast_to((3, 4))
            return y
    """)
    assert sh["y"] == (3, 4)


# ─────────────────────────────────────────────────────────────────────
# F.scaled_dot_product_attention
# ─────────────────────────────────────────────────────────────────────

def test_sdpa_shape():
    sh = _shapes("""
        import torch
        import torch.nn.functional as F
        def f():
            q = torch.zeros(2, 4, 8, 16)
            k = torch.zeros(2, 4, 8, 16)
            v = torch.zeros(2, 4, 8, 32)
            y = F.scaled_dot_product_attention(q, k, v)
            return y
    """)
    assert sh["y"] == (2, 4, 8, 32)


# ─────────────────────────────────────────────────────────────────────
# Reductions with dim
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["var", "std", "argmax", "argmin", "amax", "amin"])
def test_reduction_with_dim(op: str):
    sh = _shapes(f"""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x.{op}(1)
            return y
    """)
    assert sh["y"] == (2, 4)


def test_reduction_keepdim():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x.var(1, keepdim=True)
            return y
    """)
    assert sh["y"] == (2, 1, 4)


# ─────────────────────────────────────────────────────────────────────
# Arithmetic w/ python scalars and tensor broadcast
# ─────────────────────────────────────────────────────────────────────

def test_scalar_add_preserves_shape():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x + 1
            return y
    """)
    assert sh["y"] == (2, 3, 4)


def test_scalar_mul_preserves_shape():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = x * 0.5
            return y
    """)
    assert sh["y"] == (2, 3, 4)


def test_broadcast_subtract():
    sh = _shapes("""
        import torch
        def f():
            x = torch.zeros(2, 3, 4)
            y = torch.zeros(3, 4)
            z = x - y
            return z
    """)
    assert sh["z"] == (2, 3, 4)


# ─────────────────────────────────────────────────────────────────────
# Layer constructors and self.layer(x) dispatch
# ─────────────────────────────────────────────────────────────────────

def test_conv1d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv1d(3, 8, 3)
            self.fc = nn.Linear(14, 1)
        def forward(self, x):
            return self.fc(self.c(x))
    """
    _ok(src, {"x": (1, 3, 16)})


def test_conv3d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv3d(3, 8, 3)
        def forward(self, x):
            return self.c(x)
    """
    _ok(src, {"x": (1, 3, 4, 16, 16)})


def test_convT2d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.ConvTranspose2d(3, 8, 4, 2, 1)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            return self.fc(self.c(x))
    """
    _ok(src, {"x": (1, 3, 16, 16)})


def test_avgpool2d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.AvgPool2d(2, 2)
        def forward(self, x):
            return self.p(x)
    """
    _ok(src, {"x": (1, 4, 16, 16)})


def test_adaptive_max_pool2d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.AdaptiveMaxPool2d((4, 4))
            self.fc = nn.Linear(16, 1)
        def forward(self, x):
            x = self.p(x)
            return self.fc(x.flatten(2))
    """
    _ok(src, {"x": (1, 8, 16, 16)})


def test_batchnorm1d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.BatchNorm1d(8)
        def forward(self, x):
            return self.b(x)
    """
    _ok(src, {"x": (4, 8)})


def test_batchnorm3d_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.BatchNorm3d(8)
        def forward(self, x):
            return self.b(x)
    """
    _ok(src, {"x": (1, 8, 4, 8, 8)})


def test_groupnorm_layer():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GroupNorm(4, 8)
        def forward(self, x):
            return self.g(x)
    """
    _ok(src, {"x": (1, 8, 16, 16)})


# ─────────────────────────────────────────────────────────────────────
# Bug-injection tests
# ─────────────────────────────────────────────────────────────────────

def test_bug_linear_after_convT2d_wrong():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.ConvTranspose2d(3, 8, 4, 2, 1)
            self.fc = nn.Linear(7, 1)
        def forward(self, x):
            return self.fc(self.c(x))
    """
    _has_bug(src, {"x": (1, 3, 16, 16)})


def test_bug_linear_in_features_mismatch():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 5)
        def forward(self, x):
            return self.fc(x)
    """
    _has_bug(src, {"x": (1, 10)})


def test_bug_conv1d_wrong_in_channels():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv1d(3, 8, 3)
        def forward(self, x):
            return self.c(x)
    """
    _has_bug(src, {"x": (1, 5, 16)})


def test_bug_matmul_dim_mismatch():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def forward(self, a, b):
            return a @ b
    """
    _has_bug(src, {"a": (4, 8), "b": (5, 7)})


def test_bug_conv2d_wrong_in_channels():
    src = """
    import torch
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(3, 8, 3, padding=1)
        def forward(self, x):
            return self.c(x)
    """
    _has_bug(src, {"x": (1, 5, 16, 16)})
