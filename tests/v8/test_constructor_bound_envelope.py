"""tests/v8/test_constructor_bound_envelope.py
==================================================

Round-6 pinning tests for the three envelope-synthesiser patterns
extended in ``src/model_checker.py`` to lift the upstream-faithful
real-bug corpus from 5/10 → 7/10 RP@0.99:

  (A) Init-time local-scalar fold in ``_InitExtractor.visit_Assign``
      (e.g. ``sharded_inner = (h*d)//tp`` then
      ``nn.Linear(d_model, sharded_inner)``).
  (B) Single-dim shape alias ``B = x.shape[0]`` propagated into
      ``_shape_dim_map`` then consumed by a downstream view.
  (C) Shape-tuple variable built from
      ``x.size()[:-1] + (heads, 3*head_size)`` and consumed by
      ``view(*new_shape)`` via starred expansion.

Each test constructs a minimal ``nn.Module`` exhibiting only one of
the three patterns and asserts that TG produces a bug at
confidence >= 0.99 on the buggy variant. These tests are
behavioural (input source string + expected verdict) and do not
import private extractor symbols, so they remain valid even if the
extractor internals are refactored.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from src.api import verify_architecture


def _max_conf(src: str, input_shapes: dict) -> float:
    result = verify_architecture(src, input_shapes=input_shapes)
    return max((b.confidence for b in result.bugs), default=0.0)


# ---------------------------------------------------------------------------
# (A) Init-time local-scalar fold
# ---------------------------------------------------------------------------
SRC_LOCAL_SCALAR = '''
import torch
import torch.nn as nn

class BuggyModule(nn.Module):
    def __init__(self, d_model=128, num_heads=8, d_kv=64, tp_world_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        # Local scalar binding: must be folded into _param_map so the
        # nn.Linear out-features below is a known concrete dim.
        sharded_inner = (num_heads * d_kv) // tp_world_size
        self.q = nn.Linear(d_model, sharded_inner)
        self.tp = tp_world_size

    def forward(self, x):
        q = self.q(x)
        # Bug: reshape uses full num_heads*d_kv but q only has sharded_inner.
        b, s, _ = q.shape
        return q.reshape(b, s, self.num_heads, self.d_kv)
'''


def test_init_local_scalar_fold():
    conf = _max_conf(SRC_LOCAL_SCALAR, {"x": (2, 10, 128)})
    assert conf >= 0.99, (
        f"Pattern A (init local-scalar fold) regressed: "
        f"max confidence {conf} < 0.99"
    )


# ---------------------------------------------------------------------------
# (B) Single-dim shape alias  B = x.shape[0]
# ---------------------------------------------------------------------------
SRC_SHAPE_DIM_ALIAS = '''
import torch
import torch.nn as nn

class BuggyModule(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)

    def forward(self, x):
        B = x.shape[0]
        S = x.shape[1]
        qkv = self.qkv(x)
        # Bug: 3*d_model split into num_heads*head_dim drops the *3*
        # multiplicity, so reshape target volume disagrees with input.
        return qkv.reshape(B, S, self.num_heads, self.head_dim)
'''


def test_single_dim_shape_alias():
    conf = _max_conf(SRC_SHAPE_DIM_ALIAS, {"x": (2, 10, 64)})
    assert conf >= 0.99, (
        f"Pattern B (single-dim shape alias) regressed: "
        f"max confidence {conf} < 0.99"
    )


# ---------------------------------------------------------------------------
# (C) Shape-tuple variable through view(*new_shape)
# ---------------------------------------------------------------------------
SRC_SHAPE_TUPLE_STARRED = '''
import torch
import torch.nn as nn

class BuggyModule(nn.Module):
    def __init__(self, hidden_size=64, num_heads=5, head_size=85):
        super().__init__()
        # 5 * 85 = 425 != hidden_size (64) — buggy on purpose.
        # We pick odd numbers so trailing dim 3*85=255 mismatches 3*hidden.
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states):
        qkv = self.qkv(hidden_states)
        new_qkv_shape = qkv.size()[:-1] + (self.num_heads, 3 * self.head_size)
        return qkv.view(*new_qkv_shape)
'''


def test_shape_tuple_starred_view():
    conf = _max_conf(SRC_SHAPE_TUPLE_STARRED, {"hidden_states": (1, 5, 64)})
    assert conf >= 0.99, (
        f"Pattern C (shape-tuple starred view) regressed: "
        f"max confidence {conf} < 0.99"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
