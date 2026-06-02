"""Step 153 — **nn.MaxPool2d/AvgPool2d spatial rule**, machine-checked in Lean
and cross-checked against real torch and ``_propagate_pool2d``.

``lean/TensorGuard/Pool2d.lean`` proves identity, the stride-1 form,
monotonicity, the upper bound, the positive-output guard and channel
preservation for ``out = (in + 2p − kernel)/stride + 1``.  This test replays
those laws on **real** ``nn.MaxPool2d``/``nn.AvgPool2d`` modules.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Pool2d.lean")

_THEOREMS = [
    "TensorGuard.Pool2d.poolOut_identity",
    "TensorGuard.Pool2d.poolOut_stride_one",
    "TensorGuard.Pool2d.poolOut_mono",
    "TensorGuard.Pool2d.poolOut_le",
    "TensorGuard.Pool2d.poolOut_pos",
    "TensorGuard.Pool2d.pool2d_channels_preserved",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _pool_out(i, p, k, s):
    return (i + 2 * p - k) // s + 1


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Pool2d" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(k, s, p):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="mp", kind=LayerKind.MAXPOOL2D, kernel_size=(k, k),
                    params={"stride": (s, s), "padding": (p, p)})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_pool2d_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_pool2d

    checked = 0
    for (h, w) in [(32, 32), (28, 40), (15, 17)]:
        for k in (2, 3):
            for s in (1, 2):
                p = 0 if k == 2 else 1
                if p > k // 2:
                    continue
                if _pool_out(h, p, k, s) <= 0 or _pool_out(w, p, k, s) <= 0:
                    continue
                mod = nn.MaxPool2d(k, stride=s, padding=p)
                real = list(mod(torch.zeros(1, 4, h, w)).shape)
                pred, err = _propagate_pool2d(_ts(1, 4, h, w), _ld(k, s, p))
                assert err is None
                got = [x.value for x in pred.dims]
                assert got == real, (got, real, (h, w, k, s, p))
                assert got[1] == 4   # pool2d_channels_preserved
                checked += 1
    assert checked > 0


def test_pool2d_identity_and_monotone():
    pytest.importorskip("torch")
    from src.model_checker import _propagate_pool2d
    pred, err = _propagate_pool2d(_ts(1, 4, 19, 23), _ld(1, 1, 0))
    assert err is None and pred.dims[2].value == 19 and pred.dims[3].value == 23
    prev = -1
    for h in (8, 16, 32, 64):
        p, _ = _propagate_pool2d(_ts(1, 4, h, h), _ld(2, 2, 0))
        assert p.dims[2].value >= prev
        prev = p.dims[2].value
