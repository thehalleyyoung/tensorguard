"""Step 156 — **nn.PixelShuffle rule**, machine-checked in Lean and cross-checked
against real torch and ``_propagate_pixel_shuffle``.

``lean/TensorGuard/PixelShuffle.lean`` proves numel preservation, the channel
divisibility by ``r²``, the recovered channel count ``C_in / r²`` and the
divisibility refutation.  This test replays those laws on **real**
``nn.PixelShuffle`` modules and confirms a non-divisible channel count makes
torch raise exactly when the Lean guard flags it.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "PixelShuffle.lean")

_THEOREMS = [
    "TensorGuard.PixelShuffle.ps_numel",
    "TensorGuard.PixelShuffle.ps_divisible",
    "TensorGuard.PixelShuffle.ps_cout",
    "TensorGuard.PixelShuffle.psValid_iff",
    "TensorGuard.PixelShuffle.ps_construct_valid",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.PixelShuffle" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(r):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="ps", kind=LayerKind.PIXEL_SHUFFLE,
                    params={"upscale_factor": r})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_pixelshuffle_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_pixel_shuffle

    checked = 0
    for r in (2, 3, 4):
        for c in (1, 2, 5):
            for (h, w) in [(8, 8), (6, 10)]:
                cin = c * r * r
                mod = nn.PixelShuffle(r)
                real = list(mod(torch.zeros(1, cin, h, w)).shape)
                pred, err = _propagate_pixel_shuffle(_ts(1, cin, h, w), _ld(r))
                assert err is None
                got = [d.value for d in pred.dims]
                assert got == real, (got, real, (r, c, h, w))
                # ps_cout: recovered channels = C_in / r^2 = c.
                assert got[1] == c
                # ps_numel: rearrangement preserves total elements.
                assert _prod(got) == _prod([1, cin, h, w])
                checked += 1
    assert checked > 0


def test_pixelshuffle_nondivisible_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_pixel_shuffle
    # 10 channels not divisible by r^2 = 4 -> torch raises; verifier flags.
    with pytest.raises(RuntimeError):
        nn.PixelShuffle(2)(torch.zeros(1, 10, 8, 8))
    pred, err = _propagate_pixel_shuffle(_ts(1, 10, 8, 8), _ld(2))
    assert err is not None and pred is None   # psValid_iff direction
