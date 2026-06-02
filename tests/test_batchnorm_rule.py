"""Step 159 — **nn.BatchNorm rule**, machine-checked in Lean and cross-checked
against real torch and ``_propagate_batchnorm``.

``lean/TensorGuard/BatchNormRule.lean`` proves shape & numel preservation, the
channel index, and the feature-count guard (``input.dims[1] == num_features``).
This test replays those laws on **real** ``nn.BatchNorm1d/2d`` modules and
confirms a wrong channel count makes torch raise exactly when the Lean guard
flags it.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "BatchNormRule.lean")

_THEOREMS = [
    "TensorGuard.BatchNormRule.bn_preserves",
    "TensorGuard.BatchNormRule.bn_numel",
    "TensorGuard.BatchNormRule.featValid_iff",
    "TensorGuard.BatchNormRule.feat_mismatch_flagged",
    "TensorGuard.BatchNormRule.bn_channel_index",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.BatchNormRule" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(num_features, kind=None):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="bn", kind=kind or LayerKind.BATCHNORM2D,
                    num_features=num_features,
                    params={"num_features": num_features})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_batchnorm_preserves_shape_real_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_batchnorm, LayerKind

    checked = 0
    # BatchNorm2d: (N, C, H, W)
    for shape, feat in [((2, 6, 8, 8), 6), ((4, 3, 5, 5), 3)]:
        mod = nn.BatchNorm2d(feat)
        real = list(mod(torch.zeros(*shape)).shape)
        pred, err = _propagate_batchnorm(_ts(*shape), _ld(feat))
        assert err is None
        got = [d.value for d in pred.dims]
        assert got == real == list(shape)      # bn_preserves
        assert got[1] == feat                  # bn_channel_index
        checked += 1
    # BatchNorm1d: (N, C)
    mod = nn.BatchNorm1d(10)
    real = list(mod(torch.zeros(4, 10)).shape)
    pred, err = _propagate_batchnorm(_ts(4, 10), _ld(10, kind=LayerKind.BATCHNORM1D))
    assert err is None and [d.value for d in pred.dims] == real
    checked += 1
    assert checked > 0


def test_batchnorm_feature_mismatch_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_batchnorm

    mod = nn.BatchNorm2d(6)
    with pytest.raises(RuntimeError):
        mod(torch.zeros(2, 5, 8, 8))           # 5 channels != 6 features
    pred, err = _propagate_batchnorm(_ts(2, 5, 8, 8), _ld(6))
    assert err is not None and pred is None    # feat_mismatch_flagged direction
