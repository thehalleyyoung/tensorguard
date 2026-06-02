"""Step 56 -- zero-flag verification via structural input-shape inference.

``tensorguard verify <path>`` must work on the common case with **no** ``-s``
annotation.  When the source documents nothing, TensorGuard pins the input rank
and channel count from the first rank-determining layer (Conv/BatchNorm/
InstanceNorm), which can legally accept exactly one input rank.  This is sound:
the pinned rank is the only rank the layer would accept at runtime, so inference
can never *introduce* a false alarm -- it only sharpens an otherwise
fully-symbolic input so that downstream shape bugs become decidable.

These tests prove (a) the structural inference recovers the right rank/channels,
(b) it makes a downstream bug catchable that is *missed* without inference,
(c) it never flips a genuinely-safe model to UNSAFE (no false positive), and
(d) it abstains when the input's first use is not a rank-determining layer.
"""
import textwrap

import pytest

from src.model_checker import extract_computation_graph, verify_model
from src.input_spec_inference import infer_input_specs_from_graph


def _graph(src: str):
    return extract_computation_graph(textwrap.dedent(src))


CONV2D_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.fc = nn.Linear(16, 10)
        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=(2, 3))
            return self.fc(x)
"""

CONV1D_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(8, 16, 3)
        def forward(self, x):
            return self.conv(x)
"""

CONV3D_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 16, 3)
        def forward(self, x):
            return self.conv(x)
"""

BN2D_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(32)
        def forward(self, x):
            return self.bn(x)
"""

# First use of x is a Linear -> rank is genuinely ambiguous -> must abstain.
LINEAR_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 10)
        def forward(self, x):
            return self.fc(x)
"""

# First use of x is a transpose (non-layer op) -> channel position unknown ->
# must abstain to stay sound.
TRANSPOSE_FIRST_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(8, 16, 3)
        def forward(self, x):
            x = x.transpose(1, 2)
            return self.conv(x)
"""


# ---------------------------------------------------------------------------
# 1. The structural inference recovers the right rank and channels.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src,name,expected", [
    (CONV2D_NET, "x", ("batch", 3, "height", "width")),
    (CONV1D_NET, "x", ("batch", 8, "length")),
    (CONV3D_NET, "x", ("batch", 4, "depth", "height", "width")),
    (BN2D_NET, "x", ("batch", 32, "height", "width")),
])
def test_structural_inference_recovers_shape(src, name, expected):
    spec = infer_input_specs_from_graph(_graph(src))
    assert spec.shapes.get(name) == expected
    assert spec.sources.get(name, "").startswith("layer:")


# ---------------------------------------------------------------------------
# 2. Abstention: ambiguous-rank layers and non-layer first uses infer nothing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src", [LINEAR_NET, TRANSPOSE_FIRST_NET])
def test_structural_inference_abstains(src):
    spec = infer_input_specs_from_graph(_graph(src))
    assert spec.shapes == {}


# ---------------------------------------------------------------------------
# 3. Inference makes a downstream bug catchable that is missed without it.
# ---------------------------------------------------------------------------
BAD_NET = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)   # BUG: should be Linear(16, 10)
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)
"""

GOOD_NET = BAD_NET.replace("nn.Linear(32, 10)", "nn.Linear(16, 10)")


def test_inference_enables_catching_downstream_bug():
    src = textwrap.dedent(BAD_NET)
    # Without inference the input rank is unknown, so the flatten width stays
    # symbolic and the wrong Linear cannot be refuted: a false negative.
    no_infer = verify_model(src, infer_inputs=False)
    assert no_infer.safe is True
    # With inference the input is pinned to 4-D, the flatten yields a concrete
    # width of 16, and the Linear(32) mismatch is provably refuted.
    with_infer = verify_model(src)
    assert with_infer.safe is False
    assert with_infer.counterexample is not None
    assert with_infer.inferred_input_shapes.get("x") == ("batch", 3, "height", "width")


def test_inference_does_not_flip_safe_model():
    # Soundness: the corrected model must stay SAFE under inference -- structural
    # inference never introduces a false positive.
    src = textwrap.dedent(GOOD_NET)
    res = verify_model(src)
    assert res.safe is True
    assert res.inferred_input_shapes.get("x") == ("batch", 3, "height", "width")


# ---------------------------------------------------------------------------
# 4. Explicit shapes are never overridden by inference.
# ---------------------------------------------------------------------------
def test_explicit_shapes_disable_inference():
    src = textwrap.dedent(CONV2D_NET)
    res = verify_model(src, input_shapes={"x": ("b", 3, 8, 8)})
    # Caller supplied shapes -> inference must not run / record anything.
    assert res.inferred_input_shapes == {}


# ---------------------------------------------------------------------------
# 5. End-to-end through the public verify_architecture API (CLI path).
# ---------------------------------------------------------------------------
def test_verify_architecture_surfaces_inferred_shapes():
    from src.api import verify_architecture
    src = textwrap.dedent(GOOD_NET)
    result = verify_architecture(src)
    assert result.inferred_input_shapes.get("x") == ("batch", 3, "height", "width")
    assert result.inferred_input_sources.get("x", "").startswith("layer:")
    # And the escape hatch removes the inference.
    result2 = verify_architecture(src, infer_inputs=False)
    assert result2.inferred_input_shapes == {}
