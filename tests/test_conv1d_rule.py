"""Step 160 — **nn.Conv1d spatial rule**, machine-checked in Lean and
cross-checked against real torch and ``_propagate_conv1d``.

``lean/TensorGuard/Conv1d.lean`` proves identity, the stride-1 form,
monotonicity, the padded-input upper bound, the positive-output guard and the
3-D channel assembly for ``L' = (L + 2p − eff)/stride + 1``.  This test replays
those laws on **real** ``nn.Conv1d`` modules.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Conv1d.lean")

_THEOREMS = [
    "TensorGuard.Conv1d.convOut_identity",
    "TensorGuard.Conv1d.convOut_stride_one",
    "TensorGuard.Conv1d.convOut_mono",
    "TensorGuard.Conv1d.convOut_le",
    "TensorGuard.Conv1d.convOut_pos",
    "TensorGuard.Conv1d.conv1d_channels",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _conv_out(i, p, k, s, d):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Conv1d" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def _ld(cin, cout, k, s, p, d):
    from src.model_checker import LayerDef, LayerKind
    return LayerDef(attr_name="c", kind=LayerKind.CONV1D, in_channels=cin,
                    out_channels=cout, kernel_size=(k,),
                    params={"stride": (s,), "padding": (p,), "dilation": (d,),
                            "groups": 1})


def _ts(*dims):
    from src.tensor_shapes import ShapeDim, TensorShape
    return TensorShape(tuple(ShapeDim(int(d)) for d in dims))


def test_conv1d_matches_real_torch_and_verifier():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.model_checker import _propagate_conv1d

    checked = 0
    for length in (20, 33, 50):
        for k in (1, 3, 5):
            for s in (1, 2):
                for p in (0, 1):
                    for d in (1, 2):
                        if _conv_out(length, p, k, s, d) <= 0:
                            continue
                        mod = nn.Conv1d(3, 6, k, stride=s, padding=p, dilation=d)
                        real = list(mod(torch.zeros(1, 3, length)).shape)
                        pred, err = _propagate_conv1d(_ts(1, 3, length), _ld(3, 6, k, s, p, d))
                        assert err is None
                        got = [x.value for x in pred.dims]
                        assert got == real, (got, real, (length, k, s, p, d))
                        assert len(got) == 3 and got[1] == 6
                        checked += 1
    assert checked > 0


def test_conv1d_identity_and_monotone():
    pytest.importorskip("torch")
    from src.model_checker import _propagate_conv1d
    pred, err = _propagate_conv1d(_ts(1, 3, 41), _ld(3, 6, 1, 1, 0, 1))
    assert err is None and pred.dims[2].value == 41         # convOut_identity
    prev = -1
    for length in (10, 20, 40, 80):
        p, _ = _propagate_conv1d(_ts(1, 3, length), _ld(3, 6, 3, 2, 1, 1))
        assert p.dims[2].value >= prev                      # convOut_mono
        prev = p.dims[2].value
