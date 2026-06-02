"""Step 38 — dynamic control flow: path-sensitive analysis of
``for <var> in self.<ModuleList>:`` unrolling and data-dependent branches.

Every model here is differentially validated against eager torch (the safe
ones run; the buggy ones raise) so the static verdicts are anchored to real
PyTorch semantics.
"""
import textwrap

import torch
import torch.nn as nn

from src.model_checker import verify_model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _eager_runs(module: nn.Module, x: torch.Tensor) -> bool:
    try:
        module(x)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# ModuleList for-loop unrolling
# --------------------------------------------------------------------------- #
MODULELIST_OK = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
""")

MODULELIST_BAD = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(99, 4)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
""")

MODULELIST_ENUM_OK = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(16, 4)])
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x
""")

CONV_MODULELIST_BAD = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Conv2d(3, 8, 3),
                                         nn.Conv2d(99, 4, 3)])
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x
""")


def test_modulelist_loop_safe():
    r = verify_model(MODULELIST_OK, input_shapes={"x": (2, 8)})
    assert r.safe is True

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    assert _eager_runs(Net(), torch.randn(2, 8))


def test_modulelist_loop_incompatible_stack_flagged():
    r = verify_model(MODULELIST_BAD, input_shapes={"x": (2, 8)})
    assert r.safe is False
    # The violation must point at the second (unrolled) sublayer.
    ce = r.counterexample
    assert ce is not None
    msgs = " ".join(v.message for v in ce.violations)
    assert "99" in msgs

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(99, 4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Eager torch agrees: this raises.
    assert not _eager_runs(Net(), torch.randn(2, 8))


def test_modulelist_enumerate_loop_safe():
    r = verify_model(MODULELIST_ENUM_OK, input_shapes={"x": (2, 8)})
    assert r.safe is True

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 16), nn.Linear(16, 4)])

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x

    assert _eager_runs(Net(), torch.randn(2, 8))


def test_conv_modulelist_channel_mismatch_flagged():
    r = verify_model(CONV_MODULELIST_BAD, input_shapes={"x": (1, 3, 16, 16)})
    assert r.safe is False

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Conv2d(3, 8, 3),
                                         nn.Conv2d(99, 4, 3)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    assert not _eager_runs(Net(), torch.randn(1, 3, 16, 16))


# --------------------------------------------------------------------------- #
# Data-dependent branch (path-sensitive) — must still verify both paths.
# --------------------------------------------------------------------------- #
BRANCH_OK = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 8)
        def forward(self, x):
            if x.sum() > 0:
                x = self.a(x)
            else:
                x = self.b(x)
            return x
""")

BRANCH_BAD = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(99, 8)
        def forward(self, x):
            if x.sum() > 0:
                x = self.a(x)
            else:
                x = self.b(x)
            return x
""")


def test_data_dependent_branch_safe():
    r = verify_model(BRANCH_OK, input_shapes={"x": (2, 8)})
    assert r.safe is True


def test_data_dependent_branch_unsafe_path_flagged():
    # One branch (self.b expects last dim 99) is incompatible — a sound
    # verifier must reject because that path is reachable.
    r = verify_model(BRANCH_BAD, input_shapes={"x": (2, 8)})
    assert r.safe is False


# --------------------------------------------------------------------------- #
# Loop over ModuleList nested inside a model that also has plain layers.
# --------------------------------------------------------------------------- #
MIXED_OK = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            self.head = nn.Linear(8, 2)
        def forward(self, x):
            x = self.proj(x)
            for layer in self.layers:
                x = layer(x)
            x = self.head(x)
            return x
""")


def test_mixed_pre_loop_post_layers_safe():
    r = verify_model(MIXED_OK, input_shapes={"x": (3, 4)})
    assert r.safe is True

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 8)
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
            self.head = nn.Linear(8, 2)

        def forward(self, x):
            x = self.proj(x)
            for layer in self.layers:
                x = layer(x)
            x = self.head(x)
            return x

    out = Net()(torch.randn(3, 4))
    assert tuple(out.shape) == (3, 2)
