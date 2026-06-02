"""Step 28 -- convolution family precision (stride/padding/dilation/groups,
transposed conv, 1d/2d/3d, output_padding).

Before Step 28 several precision/soundness gaps existed in the convolution
family:

* The transposed-conv output formulas (``ConvTranspose1d/2d/3d``) used
  ``+ kernel_size`` instead of ``+ dilation * (kernel_size - 1) + 1``.  This is
  correct only when ``dilation == 1`` and silently wrong for dilated transposed
  convolutions (e.g. ``ConvTranspose2d(4, 6, 3, stride=2, padding=1,
  dilation=2, output_padding=1)`` on ``(1, 4, 8, 8)`` yields ``(1, 6, 18, 18)``
  in torch but the old formula produced ``16``).
* ``dilation`` and ``groups`` were not captured at all for transposed convs in
  either extraction path (live FX modules and AST/source), so any dilation was
  ignored.
* The forward ``Conv3d`` formula also ignored ``dilation`` and never validated
  ``groups`` divisibility (``Conv1d/2d`` already did).

Step 28 fixes the output formulas to include dilation, captures
``dilation``/``groups`` for the whole conv family in both the FX path
(``_extract_layer_params``) and the AST positional path, and adds ``groups``
divisibility checks for the transposed convs and ``Conv3d``.

These tests prove the behaviour with a large differential sweep against real
``torch`` convolution modules (1d/2d/3d, forward + transposed, random
stride/padding/dilation/groups/output_padding), end-to-end ``verify_module``
checks (including the originally-wrong dilated transposed conv and a DCGAN-style
generator built entirely from stacked transposed convolutions), and
source-level ``groups`` divisibility detection that cannot be exercised with a
constructed module (torch asserts at ``__init__``).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.tensor_shapes import ShapeDim, TensorShape
from src.fx_extractor import verify_module
from src.model_checker import (
    LayerDef,
    LayerKind,
    verify_model,
    _propagate_conv1d,
    _propagate_conv2d,
    _propagate_conv3d,
    _propagate_convtranspose1d,
    _propagate_convtranspose2d,
    _propagate_convtranspose3d,
)


def _shape(t: torch.Tensor) -> TensorShape:
    return TensorShape(tuple(ShapeDim(int(d)) for d in t.shape))


def _make_layer(kind: LayerKind, m: nn.Module) -> LayerDef:
    ld = LayerDef(attr_name="t", kind=kind)
    ld.in_channels = m.in_channels
    ld.out_channels = m.out_channels
    ld.kernel_size = m.kernel_size
    params = {
        "in_channels": m.in_channels,
        "out_channels": m.out_channels,
        "kernel_size": m.kernel_size,
        "stride": m.stride,
        "padding": m.padding,
        "dilation": m.dilation,
        "groups": m.groups,
    }
    if hasattr(m, "output_padding"):
        params["output_padding"] = m.output_padding
    ld.params = params
    return ld


_FWD = {
    1: (nn.Conv1d, LayerKind.CONV1D, _propagate_conv1d),
    2: (nn.Conv2d, LayerKind.CONV2D, _propagate_conv2d),
    3: (nn.Conv3d, LayerKind.CONV3D, _propagate_conv3d),
}
_TRANSPOSED = {
    1: (nn.ConvTranspose1d, LayerKind.CONVTRANSPOSE1D, _propagate_convtranspose1d),
    2: (nn.ConvTranspose2d, LayerKind.CONVTRANSPOSE2D, _propagate_convtranspose2d),
    3: (nn.ConvTranspose3d, LayerKind.CONVTRANSPOSE3D, _propagate_convtranspose3d),
}


def _predict(prop, kind, m, x):
    pred, err = prop(_shape(x), _make_layer(kind, m))
    assert err is None, f"unexpected error from propagator: {err}"
    return tuple(d.value for d in pred.dims)


def test_differential_forward_and_transposed_against_torch():
    """Predicted output shapes match torch across the whole conv family."""
    random.seed(20240828)
    torch.manual_seed(20240828)
    checked = 0
    for _ in range(2500):
        nd = random.choice([1, 2, 3])
        transposed = random.choice([True, False])
        groups = random.choice([1, 1, 2])
        cin = groups * random.randint(1, 3)
        cout = groups * random.randint(1, 3)
        ks = random.randint(1, 4)
        stride = random.randint(1, 3)
        pad = random.randint(0, 2)
        dil = random.randint(1, 3)
        spatial = random.randint(6, 14)
        try:
            if transposed:
                cls, kind, prop = _TRANSPOSED[nd]
                lo = min(stride, dil)
                outpad = random.randint(0, lo - 1) if lo > 1 else 0
                m = cls(cin, cout, ks, stride=stride, padding=pad,
                        dilation=dil, groups=groups, output_padding=outpad)
            else:
                cls, kind, prop = _FWD[nd]
                m = cls(cin, cout, ks, stride=stride, padding=pad,
                        dilation=dil, groups=groups)
            x = torch.randn(2, cin, *([spatial] * nd))
            y = m(x)
        except Exception:
            continue
        assert _predict(prop, kind, m, x) == tuple(y.shape)
        checked += 1
    assert checked > 1500, f"too few cases exercised: {checked}"


def test_dilated_transposed_conv2d_regression():
    """The exact case that was wrong before Step 28."""
    m = nn.ConvTranspose2d(4, 6, kernel_size=3, stride=2, padding=1,
                           dilation=2, output_padding=1)
    x = torch.randn(1, 4, 8, 8)
    assert tuple(m(x).shape) == (1, 6, 18, 18)
    assert _predict(_propagate_convtranspose2d, LayerKind.CONVTRANSPOSE2D, m, x) == (1, 6, 18, 18)


def test_dilated_transposed_conv1d_and_3d():
    m1 = nn.ConvTranspose1d(3, 5, kernel_size=4, stride=2, padding=1, dilation=3)
    x1 = torch.randn(2, 3, 7)
    assert _predict(_propagate_convtranspose1d, LayerKind.CONVTRANSPOSE1D, m1, x1) == tuple(m1(x1).shape)

    m3 = nn.ConvTranspose3d(2, 4, kernel_size=3, stride=2, padding=1, dilation=2, output_padding=1)
    x3 = torch.randn(1, 2, 5, 5, 5)
    assert _predict(_propagate_convtranspose3d, LayerKind.CONVTRANSPOSE3D, m3, x3) == tuple(m3(x3).shape)


def test_dilated_conv3d_forward_regression():
    m = nn.Conv3d(2, 4, kernel_size=3, dilation=2)
    x = torch.randn(1, 2, 10, 10, 10)
    assert _predict(_propagate_conv3d, LayerKind.CONV3D, m, x) == tuple(m(x).shape)


def test_end_to_end_dilated_transposed_conv_downstream():
    """A Linear consuming the (correct) dilated output is SAFE; the buggy
    width is correctly flagged UNSAFE."""

    class Good(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, dilation=2, output_padding=1)
            self.lin = nn.Linear(18, 5)

        def forward(self, x):
            return self.lin(self.t(x))

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.t = nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1, dilation=2, output_padding=1)
            self.lin = nn.Linear(16, 5)

        def forward(self, x):
            return self.lin(self.t(x))

    assert verify_module(Good(), input_shapes={"x": (1, 4, 8, 8)}).safe is True
    assert verify_module(Bad(), input_shapes={"x": (1, 4, 8, 8)}).safe is False


def test_dcgan_generator_stacked_transposed_convs():
    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(100, 256, 4, 1, 0), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)

    g = Gen()
    assert tuple(g(torch.randn(1, 100, 1, 1)).shape) == (1, 3, 32, 32)
    assert verify_module(g, input_shapes={"x": (1, 100, 1, 1)}).safe is True


def test_grouped_conv_valid_is_safe():
    class Grp(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(8, 16, 3, groups=4, padding=1)

        def forward(self, x):
            return self.c(x)

    assert verify_module(Grp(), input_shapes={"x": (1, 8, 16, 16)}).safe is True


def test_source_level_groups_divisibility_flagged():
    """Invalid groups cannot be constructed (torch asserts), so this must be
    caught by source-level analysis."""
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.ConvTranspose2d(10, 6, 3, stride=2, padding=1, groups=4)
    def forward(self, x):
        return self.c(x)
"""
    r = verify_model(src, input_shapes={"x": (1, 10, 8, 8)})
    assert r.safe is False
    assert r.counterexample is not None
    assert any("groups" in v.message and "divide" in v.message
               for v in r.counterexample.violations)


def test_source_level_positional_dilation_capture():
    """ConvTranspose2d with positional dilation (9th positional arg) is parsed
    correctly by the AST path."""
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.ConvTranspose2d(4, 6, 3, 2, 1, 1, 1, True, 2)
        self.lin = nn.Linear(18, 5)
    def forward(self, x):
        return self.lin(self.t(x))
"""
    assert verify_model(src, input_shapes={"x": (1, 4, 8, 8)}).safe is True
    src_bad = src.replace("nn.Linear(18, 5)", "nn.Linear(16, 5)")
    assert verify_model(src_bad, input_shapes={"x": (1, 4, 8, 8)}).safe is False
