"""Step 157 — **nn.AdaptiveAvgPool2d rule**, machine-checked in Lean and
cross-checked against real torch and ``_propagate_adaptive_avgpool2d``.

``lean/TensorGuard/AdaptivePool.lean`` proves target-size exactness (the spatial
output equals the requested size for *any* input), batch/channel preservation,
rank and idempotence.  This test replays those laws on **real**
``nn.AdaptiveAvgPool2d`` modules over a grid of input/target sizes.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "AdaptivePool.lean")

_THEOREMS = [
    "TensorGuard.AdaptivePool.ap_spatial_h",
    "TensorGuard.AdaptivePool.ap_spatial_w",
    "TensorGuard.AdaptivePool.ap_channels",
    "TensorGuard.AdaptivePool.ap_idempotent",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.AdaptivePool" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(out):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="ap", kind=LayerKind.ADAPTIVE_AVGPOOL2D,
                    output_size=out, params={"output_size": out})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_adaptivepool_target_exact_real_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_adaptive_avgpool2d

    checked = 0
    for (h, w) in [(32, 40), (15, 15), (50, 7)]:
        for out in [(7, 7), (1, 1), (4, 8)]:
            mod = nn.AdaptiveAvgPool2d(out)
            real = list(mod(torch.zeros(2, 3, h, w)).shape)
            pred, err = _propagate_adaptive_avgpool2d(_ts(2, 3, h, w), _ld(out))
            assert err is None
            got = [d.value for d in pred.dims]
            assert got == real
            # ap_spatial_*: output spatial is the target, independent of input.
            assert (got[2], got[3]) == out
            # ap_channels / batch preserved.
            assert got[0] == 2 and got[1] == 3
            checked += 1
    assert checked > 0


def test_adaptivepool_idempotent():
    pytest.importorskip("torch")
    from src.model_checker import _propagate_adaptive_avgpool2d
    # ap_idempotent: pooling to the size already held preserves the full shape.
    pred, err = _propagate_adaptive_avgpool2d(_ts(2, 3, 9, 9), _ld((9, 9)))
    assert err is None and [d.value for d in pred.dims] == [2, 3, 9, 9]
