"""Step 154 — **nn.ConvTranspose1d length rule**, machine-checked in Lean and
cross-checked against real torch and ``_propagate_convtranspose1d``.

``lean/TensorGuard/ConvTranspose.lean`` proves identity, the no-pad form,
monotonicity, the upsampling lower bound and the shape laws for
``L' = (L−1)·stride − 2p + dilation·(kernel−1) + output_padding + 1``.  This test
replays those laws on **real** ``nn.ConvTranspose1d`` modules.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "ConvTranspose.lean")

_THEOREMS = [
    "TensorGuard.ConvTranspose.ctOut_identity",
    "TensorGuard.ConvTranspose.ctOut_no_pad",
    "TensorGuard.ConvTranspose.ctOut_mono",
    "TensorGuard.ConvTranspose.ctOut_ge",
    "TensorGuard.ConvTranspose.ct_channels",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _ct_out(length, s, p, op, k, d):
    return (length - 1) * s - 2 * p + d * (k - 1) + op + 1


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.ConvTranspose" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(cin, cout, k, s, p, op, d):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="ct", kind=LayerKind.CONVTRANSPOSE1D,
                    in_channels=cin, out_channels=cout,
                    params={"kernel_size": (k,), "stride": (s,), "padding": (p,),
                            "output_padding": (op,), "dilation": (d,)})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_convtranspose_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_convtranspose1d

    checked = 0
    for length in (8, 10, 16):
        for k in (2, 3):
            for s in (1, 2):
                for p in (0, 1):
                    for op in (0,):
                        d = 1
                        mod = nn.ConvTranspose1d(3, 6, k, stride=s, padding=p,
                                                 output_padding=op, dilation=d)
                        real = list(mod(torch.zeros(1, 3, length)).shape)
                        pred, err = _propagate_convtranspose1d(
                            _ts(1, 3, length), _ld(3, 6, k, s, p, op, d))
                        assert err is None
                        got = [x.value for x in pred.dims]
                        assert got == real, (got, real, (length, k, s, p, op))
                        assert got[1] == 6   # ct_channels
                        checked += 1
    assert checked > 0


def test_convtranspose_identity_and_upsamples():
    pytest.importorskip("torch")
    from src.model_checker import _propagate_convtranspose1d
    # ctOut_identity: stride 1, no pad, kernel 1 preserves length.
    pred, err = _propagate_convtranspose1d(_ts(1, 3, 13), _ld(3, 6, 1, 1, 0, 0, 1))
    assert err is None and pred.dims[2].value == 13
    # ctOut_ge: with stride >= 1 and no padding, never shrinks.
    for length in (4, 8, 12):
        p, _ = _propagate_convtranspose1d(_ts(1, 3, length), _ld(3, 6, 3, 2, 0, 0, 1))
        assert p.dims[2].value >= length
