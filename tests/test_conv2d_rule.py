"""Step 152 — **nn.Conv2d spatial rule**, machine-checked in Lean and
cross-checked against real torch and the verifier's ``_propagate_conv2d``.

``lean/TensorGuard/Conv2d.lean`` models the integer spatial map
``convOut = (in + 2p − eff)/stride + 1`` (eff = dilation·(kernel−1)+1) and proves
identity, the stride-1 form, monotonicity, the padded-input upper bound, the
positive-output guard and the 4-D channel assembly.  This test replays those laws
on **real** ``nn.Conv2d`` modules over a kernel/stride/padding/dilation grid and
confirms a non-positive prediction coincides with torch raising.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Conv2d.lean")

_THEOREMS = [
    "TensorGuard.Conv2d.convOut_identity",
    "TensorGuard.Conv2d.convOut_stride_one",
    "TensorGuard.Conv2d.convOut_mono",
    "TensorGuard.Conv2d.convOut_le",
    "TensorGuard.Conv2d.convOut_pos",
    "TensorGuard.Conv2d.conv2d_rank",
    "TensorGuard.Conv2d.conv2d_channels",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _conv_out(i, p, k, s, d):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Conv2d" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(cin, cout, k, s, p, d):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="c", kind=LayerKind.CONV2D, in_channels=cin,
                    out_channels=cout, kernel_size=(k, k),
                    params={"stride": (s, s), "padding": (p, p),
                            "dilation": (d, d), "groups": 1})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_conv2d_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_conv2d

    checked = 0
    for (h, w) in [(32, 32), (28, 40), (15, 15)]:
        for k in (1, 3, 5):
            for s in (1, 2):
                for p in (0, 1):
                    for d in (1, 2):
                        if _conv_out(h, p, k, s, d) <= 0 or _conv_out(w, p, k, s, d) <= 0:
                            continue
                        mod = nn.Conv2d(3, 6, k, stride=s, padding=p, dilation=d)
                        real = list(mod(torch.zeros(1, 3, h, w)).shape)
                        pred, err = _propagate_conv2d(_ts(1, 3, h, w), _ld(3, 6, k, s, p, d))
                        assert err is None, (h, w, k, s, p, d)
                        got = [x.value for x in pred.dims]
                        assert got == real, (got, real, (h, w, k, s, p, d))
                        # Lean shape assembly: 4-D, channels set to out=6.
                        assert len(got) == 4 and got[1] == 6
                        checked += 1
    assert checked > 0


def test_conv2d_identity_and_monotone():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_conv2d

    # convOut_identity: 1x1 stride-1 no-pad preserves spatial.
    pred, err = _propagate_conv2d(_ts(1, 3, 17, 23), _ld(3, 6, 1, 1, 0, 1))
    got = [x.value for x in pred.dims]
    assert err is None and got[2] == 17 and got[3] == 23
    # convOut_mono: larger input -> at-least-as-large output.
    prev = -1
    for h in (10, 20, 40, 80):
        p, _ = _propagate_conv2d(_ts(1, 3, h, h), _ld(3, 6, 3, 2, 1, 1))
        cur = p.dims[2].value
        assert cur >= prev
        prev = cur


def test_conv2d_nonpositive_output_refuted_like_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_conv2d

    # Kernel larger than input -> torch raises; verifier flags (convOut_pos
    # guard fails the valid regime).
    with pytest.raises(RuntimeError):
        nn.Conv2d(3, 6, 9)(torch.zeros(1, 3, 4, 4))
    pred, err = _propagate_conv2d(_ts(1, 3, 4, 4), _ld(3, 6, 9, 1, 0, 1))
    assert err is not None and pred is None
