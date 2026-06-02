"""Step 41 — pluggable shape-stub registry for third-party layers.

Third-party blocks (HuggingFace ``Conv1D``, ``timm`` ``Mlp``/``DropPath``,
user-defined imported modules) are matched *by name* to a registered shape
stub, giving a precise output shape instead of an opaque symbolic abstention —
while still soundly flagging mismatches. Built-in stubs are differentially
validated against the real third-party forward semantics (reimplemented locally
so no heavy dependency is required).
"""
import textwrap

import torch
import torch.nn as nn

import pytest

from src.model_checker import verify_model, extract_computation_graph, LayerKind
from src.shape_stub_registry import (
    register_shape_stub,
    register_last_dim_linear,
    register_shape_preserving,
    get_shape_stub,
    clear_user_stubs,
    registered_stub_names,
)


@pytest.fixture(autouse=True)
def _isolate_user_stubs():
    # Each test starts/ends with only built-in stubs present.
    clear_user_stubs()
    yield
    clear_user_stubs()


# --------------------------------------------------------------------------- #
# User-registered linear-like stub.
# --------------------------------------------------------------------------- #
FANCY_SRC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from thirdparty import FancyBlock
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = FancyBlock(8, 16)
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            x = self.block(x)
            return self.head(x)
""")


def test_user_stub_makes_block_precise():
    register_last_dim_linear(
        "FancyBlock", in_arg="in_features", out_arg="out_features",
        arg_names=("in_features", "out_features"))

    g = extract_computation_graph(FANCY_SRC)
    assert g.layers["block"].kind == LayerKind.STUB
    assert g.layers["block"].params["in_features"] == 8
    assert g.layers["block"].params["out_features"] == 16

    r = verify_model(FANCY_SRC, input_shapes={"x": (2, 8)})
    assert r.safe is True


def test_user_stub_flags_downstream_mismatch():
    register_last_dim_linear(
        "FancyBlock", in_arg="in_features", out_arg="out_features",
        arg_names=("in_features", "out_features"))
    bad = FANCY_SRC.replace("nn.Linear(16, 4)", "nn.Linear(99, 4)")
    r = verify_model(bad, input_shapes={"x": (2, 8)})
    assert r.safe is False


def test_user_stub_flags_input_mismatch():
    register_last_dim_linear(
        "FancyBlock", in_arg="in_features", out_arg="out_features",
        arg_names=("in_features", "out_features"))
    # Block expects last dim 8; feed it 5.
    r = verify_model(FANCY_SRC, input_shapes={"x": (2, 5)})
    assert r.safe is False


def test_user_stub_symbolic_dims():
    register_last_dim_linear(
        "FancyBlock", in_arg="in_features", out_arg="out_features",
        arg_names=("in_features", "out_features"))
    r = verify_model(FANCY_SRC, input_shapes={"x": ("B", "S", 8)})
    assert r.safe is True
    bad = FANCY_SRC.replace("nn.Linear(16, 4)", "nn.Linear(99, 4)")
    assert verify_model(bad, input_shapes={"x": ("B", "S", 8)}).safe is False


# --------------------------------------------------------------------------- #
# Built-in HuggingFace GPT-2 Conv1D stub, validated against real semantics.
# --------------------------------------------------------------------------- #
CONV1D_SRC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = Conv1D(32, 8)   # nf=32, nx=8: maps (...,8) -> (...,32)
            self.head = nn.Linear(32, 4)
        def forward(self, x):
            return self.head(self.c(x))
""")


def _real_conv1d_out(nf, nx, x):
    class Conv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.randn(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))

        def forward(self, x):
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            return x.view(size_out)

    return tuple(Conv1D(nf, nx)(x).shape)


def test_builtin_conv1d_matches_real_semantics():
    # Real GPT-2 Conv1D maps last dim nx -> nf.
    assert _real_conv1d_out(32, 8, torch.randn(2, 5, 8)) == (2, 5, 32)
    r = verify_model(CONV1D_SRC, input_shapes={"x": (2, 5, 8)})
    assert r.safe is True
    bad = CONV1D_SRC.replace("nn.Linear(32, 4)", "nn.Linear(99, 4)")
    assert verify_model(bad, input_shapes={"x": (2, 5, 8)}).safe is False


# --------------------------------------------------------------------------- #
# Built-in timm Mlp stub (out_features defaults to in_features).
# --------------------------------------------------------------------------- #
MLP_SRC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from timm.layers import Mlp
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Mlp(16, 64)     # out_features=None -> preserves 16
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            return self.head(self.mlp(x))
""")


def test_builtin_mlp_preserves_in_features():
    r = verify_model(MLP_SRC, input_shapes={"x": (2, 7, 16)})
    assert r.safe is True
    # A head expecting the hidden size (64) is wrong — Mlp returns 16.
    bad = MLP_SRC.replace("nn.Linear(16, 4)", "nn.Linear(64, 4)")
    assert verify_model(bad, input_shapes={"x": (2, 7, 16)}).safe is False


def test_builtin_mlp_explicit_out_features():
    src = MLP_SRC.replace("Mlp(16, 64)", "Mlp(16, 64, 8)").replace(
        "nn.Linear(16, 4)", "nn.Linear(8, 4)")
    assert verify_model(src, input_shapes={"x": (2, 7, 16)}).safe is True


# --------------------------------------------------------------------------- #
# Built-in shape-preserving stub (DropPath).
# --------------------------------------------------------------------------- #
DROPPATH_SRC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from timm.layers import DropPath
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
            self.drop = DropPath(0.1)
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            x = self.fc(x)
            x = self.drop(x)
            return self.head(x)
""")


def test_builtin_droppath_is_shape_preserving():
    r = verify_model(DROPPATH_SRC, input_shapes={"x": (2, 8)})
    assert r.safe is True
    # Head mismatch after the (shape-preserving) DropPath is still caught.
    bad = DROPPATH_SRC.replace("nn.Linear(16, 4)", "nn.Linear(7, 4)")
    assert verify_model(bad, input_shapes={"x": (2, 8)}).safe is False


# --------------------------------------------------------------------------- #
# Registry mechanics: precedence, custom transfer, isolation.
# --------------------------------------------------------------------------- #
def test_local_class_wins_over_stub():
    # A locally-defined class named like a stub must be analysed as a submodule,
    # not replaced by the stub.
    register_shape_preserving("MyBlock")
    src = textwrap.dedent("""
        import torch
        import torch.nn as nn
        class MyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 32)
            def forward(self, x):
                return self.lin(x)
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.b = MyBlock()
                self.head = nn.Linear(32, 4)
            def forward(self, x):
                return self.head(self.b(x))
    """)
    g = extract_computation_graph(src)
    # The local class is a SUBMODULE, not a STUB — so it maps 8 -> 32.
    assert g.layers["b"].kind == LayerKind.SUBMODULE
    assert verify_model(src, input_shapes={"x": (2, 8)}).safe is True


def test_custom_transfer_function():
    from src.tensor_shapes import TensorShape, ShapeDim

    def doubler(inp, params):
        # Pretend this block doubles the last dim.
        last = inp.dims[-1]
        if last.is_symbolic:
            return TensorShape(inp.dims[:-1] + (ShapeDim("_doubled"),)), None
        return TensorShape(inp.dims[:-1] + (ShapeDim(last.value * 2),)), None

    register_shape_stub("Doubler", doubler, arg_names=())
    src = textwrap.dedent("""
        import torch.nn as nn
        from tp import Doubler
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.d = Doubler()
                self.head = nn.Linear(16, 4)
            def forward(self, x):
                return self.head(self.d(x))
    """)
    # Input last dim 8 -> doubled to 16 -> head expects 16: safe.
    assert verify_model(src, input_shapes={"x": (2, 8)}).safe is True
    # Input last dim 5 -> 10 -> head expects 16: mismatch.
    assert verify_model(src, input_shapes={"x": (2, 5)}).safe is False


def test_clear_user_stubs_keeps_builtins():
    register_shape_preserving("TempBlock")
    assert get_shape_stub("TempBlock") is not None
    clear_user_stubs()
    assert get_shape_stub("TempBlock") is None
    # Built-ins survive.
    assert get_shape_stub("Conv1D") is not None
    assert get_shape_stub("DropPath") is not None


def test_dotted_constructor_matches_bare_registration():
    register_shape_preserving("MyNorm")
    assert get_shape_stub("custom.layers.MyNorm") is not None


def test_builtins_present():
    names = registered_stub_names()
    for expected in ("Conv1D", "Mlp", "DropPath"):
        assert expected in names
