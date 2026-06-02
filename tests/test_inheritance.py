"""Step 40 — subclassing, ``super().forward()``, mixins and multi-level
inheritance handled uniformly.

Inherited ``__init__`` layers are merged (so a base class's ``self.fc`` is
visible to the child), and ``super().forward(x)`` is inlined so the base
computation is actually verified instead of silently skipped. Every model is
differentially anchored to eager torch.
"""
import textwrap

import torch
import torch.nn as nn

from src.model_checker import verify_model, extract_computation_graph


def _eager_ok(module: nn.Module, x: torch.Tensor) -> bool:
    try:
        module(x)
        return True
    except Exception:
        return False


SUPER_FORWARD_OK = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
        def forward(self, x):
            return self.fc(x)
    class Child(Base):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            x = super().forward(x)
            return self.head(x)
""")

SUPER_FORWARD_BAD = SUPER_FORWARD_OK.replace(
    "self.head = nn.Linear(16, 4)", "self.head = nn.Linear(99, 4)")


def test_super_forward_inlined_safe():
    r = verify_model(SUPER_FORWARD_OK, input_shapes={"x": (2, 8)})
    assert r.safe is True

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)

        def forward(self, x):
            return self.fc(x)

    class Child(Base):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            x = super().forward(x)
            return self.head(x)

    out = Child()(torch.randn(2, 8))
    assert tuple(out.shape) == (2, 4)


def test_super_forward_inlined_bug_flagged():
    # Previously a false NEGATIVE — super().forward was skipped, so the base's
    # fc was invisible and the head mismatch was missed. Now it is caught.
    r = verify_model(SUPER_FORWARD_BAD, input_shapes={"x": (2, 8)})
    assert r.safe is False

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)

        def forward(self, x):
            return self.fc(x)

    class ChildBad(Base):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(99, 4)

        def forward(self, x):
            x = super().forward(x)
            return self.head(x)

    assert not _eager_ok(ChildBad(), torch.randn(2, 8))


FULLY_INHERITED = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
            self.out = nn.Linear(16, 4)
        def forward(self, x):
            x = self.fc(x)
            return self.out(x)
    class Child(Base):
        def __init__(self):
            super().__init__()
""")


def test_fully_inherited_forward_verified():
    # Child overrides neither forward nor adds layers: the inherited forward
    # must still be verified (not treated as empty).
    r = verify_model(FULLY_INHERITED, input_shapes={"x": (2, 8)})
    assert r.safe is True
    g = extract_computation_graph(FULLY_INHERITED)
    layer_calls = [s.layer_ref for s in g.steps if s.layer_ref]
    assert "fc" in layer_calls and "out" in layer_calls


MULTI_LEVEL = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)
        def forward(self, x):
            return self.fc(x)
    class B(A):
        def __init__(self):
            super().__init__()
            self.mid = nn.Linear(16, 32)
        def forward(self, x):
            x = super().forward(x)
            return self.mid(x)
    class C(B):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(32, 4)
        def forward(self, x):
            x = super().forward(x)
            return self.head(x)
""")


def test_multi_level_super_forward_chain():
    g = extract_computation_graph(MULTI_LEVEL)
    assert g.class_name == "C"
    layer_calls = [s.layer_ref for s in g.steps if s.layer_ref]
    # All three levels' layers must appear, in order.
    assert layer_calls == ["fc", "mid", "head"]

    r = verify_model(MULTI_LEVEL, input_shapes={"x": (2, 8)})
    assert r.safe is True

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 16)

        def forward(self, x):
            return self.fc(x)

    class B(A):
        def __init__(self):
            super().__init__()
            self.mid = nn.Linear(16, 32)

        def forward(self, x):
            return self.mid(super().forward(x))

    class C(B):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(32, 4)

        def forward(self, x):
            return self.head(super().forward(x))

    assert tuple(C()(torch.randn(2, 8)).shape) == (2, 4)


def test_multi_level_deep_base_bug_flagged():
    bad = MULTI_LEVEL.replace("self.mid = nn.Linear(16, 32)",
                              "self.mid = nn.Linear(999, 32)")
    r = verify_model(bad, input_shapes={"x": (2, 8)})
    assert r.safe is False


MIXIN = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class ProjMixin(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 16)
        def forward(self, x):
            return self.proj(x)
    class Net(ProjMixin):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            return self.head(super().forward(x))
""")


def test_mixin_super_forward_safe_and_bug():
    r = verify_model(MIXIN, input_shapes={"x": (2, 8)})
    assert r.safe is True
    bad = MIXIN.replace("self.head = nn.Linear(16, 4)",
                        "self.head = nn.Linear(7, 4)")
    rb = verify_model(bad, input_shapes={"x": (2, 8)})
    assert rb.safe is False


COMPOSITION = textwrap.dedent("""
    import torch
    import torch.nn as nn
    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(x)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = Block(8)
            self.head = nn.Linear(8, 4)
        def forward(self, x):
            return self.head(self.b(x))
""")


def test_composition_still_works():
    # Inheritance handling must not regress submodule composition.
    r = verify_model(COMPOSITION, input_shapes={"x": (2, 8)})
    assert r.safe is True
