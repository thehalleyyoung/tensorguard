"""Step 39 — symbolic/dynamic batch and sequence dimensions as first-class
symbols (not concrete guesses).

A symbolic dimension supplied as a *name* (e.g. ``("B", "S", 8)``) is verified
once, parametrically, and the safety certificate must record the symbol it
proved over (``symbolic_bindings``) — evidence the verifier reasoned about the
symbol itself rather than silently substituting a concrete guess. Each safe
model below is differentially validated against eager torch at *several*
concrete instantiations of the symbol(s); each unsafe model is validated to
fail in eager too.
"""
import textwrap

import torch
import torch.nn as nn

from src.model_checker import verify_model


def _eager_ok(module: nn.Module, *inputs: torch.Tensor) -> bool:
    try:
        module(*inputs)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Symbol is carried into the certificate (proof it isn't concretized).
# --------------------------------------------------------------------------- #
LINEAR_CHAIN = textwrap.dedent("""
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)
        def forward(self, x):
            return self.fc2(self.fc1(x))
""")


def test_symbolic_dims_recorded_in_certificate():
    r = verify_model(LINEAR_CHAIN, input_shapes={"x": ("B", "S", 8)})
    assert r.safe is True
    cert = r.certificate
    assert cert is not None
    # The symbols B and S must appear in the certificate's symbolic bindings —
    # proof they were verified parametrically, not replaced by concrete ints.
    assert "B" in cert.symbolic_bindings
    assert "S" in cert.symbolic_bindings


def test_symbolic_chain_matches_eager_at_many_sizes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    net = Net()
    for b in (1, 2, 7):
        for s in (1, 3, 16):
            assert _eager_ok(net, torch.randn(b, s, 8))


# --------------------------------------------------------------------------- #
# A symbol-independent bug is caught regardless of the symbolic batch.
# --------------------------------------------------------------------------- #
CNN_BAD_FC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(999, 10)
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
""")

CNN_OK_FC = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 32 * 32, 10)
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
""")


def test_symbolic_batch_does_not_mask_feature_bug():
    bad = verify_model(CNN_BAD_FC, input_shapes={"x": ("B", 3, 32, 32)})
    assert bad.safe is False
    ok = verify_model(CNN_OK_FC, input_shapes={"x": ("B", 3, 32, 32)})
    assert ok.safe is True

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 32 * 32, 10)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    net = Net()
    for b in (1, 4):
        assert _eager_ok(net, torch.randn(b, 3, 32, 32))


# --------------------------------------------------------------------------- #
# Symbol consistency across multiple inputs.
# --------------------------------------------------------------------------- #
TWO_INPUT_ADD = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, y):
            return x + y
""")


def test_same_symbol_add_is_safe():
    r = verify_model(TWO_INPUT_ADD,
                     input_shapes={"x": ("B", 8), "y": ("B", 8)})
    assert r.safe is True

    class Net(nn.Module):
        def forward(self, x, y):
            return x + y

    net = Net()
    for b in (1, 5):
        assert _eager_ok(net, torch.randn(b, 8), torch.randn(b, 8))


def test_distinct_symbols_add_is_rejected():
    # B and C are unconstrained distinct symbols; x + y is only safe if they
    # are equal (or one is 1). A sound verifier cannot prove that, so it must
    # not certify the model as safe.
    r = verify_model(TWO_INPUT_ADD,
                     input_shapes={"x": ("B", 8), "y": ("C", 8)})
    assert r.safe is False

    # Eager torch confirms the danger: distinct batch sizes raise.
    class Net(nn.Module):
        def forward(self, x, y):
            return x + y

    assert not _eager_ok(Net(), torch.randn(3, 8), torch.randn(4, 8))


# --------------------------------------------------------------------------- #
# Symbolic dim that flows through a reshape that breaks it is caught.
# --------------------------------------------------------------------------- #
RESIDUAL_RESHAPE_BAD = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
        def forward(self, x):
            y = self.fc(x)
            y = y.reshape(x.size(0), -1, 16)
            return x + y
""")


def test_symbolic_residual_reshape_mismatch_flagged():
    r = verify_model(RESIDUAL_RESHAPE_BAD, input_shapes={"x": ("B", "S", 8)})
    assert r.safe is False

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            y = self.fc(x)
            y = y.reshape(x.size(0), -1, 16)
            return x + y

    assert not _eager_ok(Net(), torch.randn(2, 4, 8))
