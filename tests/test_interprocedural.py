"""Step 44 — interprocedural analysis with a sound call-summary cache.

When ``forward`` calls a sibling tensor-transform method (``self._block(x)``),
TensorGuard inlines the method's body so the call is analyzed precisely instead
of abstaining as an opaque ``UNKNOWN`` layer.  Helper bodies are extracted once
and reused across call sites (the call-summary cache), nested helper chains are
followed, and (mutual) recursion is guarded with a sound fallback.
"""
import textwrap

from src.model_checker import verify_model, extract_computation_graph, LayerKind


def _v(src, **kw):
    return verify_model(src, **kw)


SINGLE_HELPER = textwrap.dedent("""
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(8, 32)
            self.head = nn.Linear(32, 4)
        def _block(self, t):
            return self.up(t)
        def forward(self, x):
            h = self._block(x)
            return self.head(h)
""")


def test_helper_method_is_inlined_precisely():
    # Helper output (32) flows to head — provably safe only if inlined.
    assert _v(SINGLE_HELPER, input_shapes={"x": (2, 8)}).safe is True


def test_mismatch_downstream_of_helper_is_caught():
    bad = SINGLE_HELPER.replace("nn.Linear(32, 4)", "nn.Linear(99, 4)")
    assert _v(bad, input_shapes={"x": (2, 8)}).safe is False


def test_mismatch_inside_helper_is_caught():
    # Helper's `up` expects last dim 8; feeding 5 must be flagged.
    assert _v(SINGLE_HELPER, input_shapes={"x": (2, 5)}).safe is False


def test_symbolic_dims_through_helper():
    assert _v(SINGLE_HELPER, input_shapes={"x": ("B", 8)}).safe is True
    bad = SINGLE_HELPER.replace("nn.Linear(32, 4)", "nn.Linear(99, 4)")
    assert _v(bad, input_shapes={"x": ("B", 8)}).safe is False


NESTED = textwrap.dedent("""
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 16)
            self.b = nn.Linear(16, 16)
            self.head = nn.Linear(16, 4)
        def _inner(self, t):
            return self.b(t)
        def _outer(self, t):
            h = self.a(t)
            return self._inner(h)
        def forward(self, x):
            y = self._outer(x)
            z = self._outer(x)
            return self.head(y + z)
""")


def test_nested_helper_chain_and_cache():
    # _outer calls _inner; _outer is itself called twice (cache reuse).
    assert _v(NESTED, input_shapes={"x": (2, 8)}).safe is True


def test_nested_helper_mismatch_caught():
    bad = NESTED.replace("nn.Linear(16, 4)", "nn.Linear(7, 4)")
    assert _v(bad, input_shapes={"x": (2, 8)}).safe is False


def test_recursive_helper_is_guarded():
    # A self-recursive helper must not hang; verification still terminates.
    rec = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def _r(self, t):
                return self._r(t)
            def forward(self, x):
                return self.fc(x)
    """)
    assert _v(rec, input_shapes={"x": (2, 8)}).safe is True


def test_helper_distinct_call_sites_do_not_alias():
    # Two helper calls with different inputs must keep independent shapes.
    src = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Linear(8, 8)
                self.head = nn.Linear(8, 4)
            def _id(self, t):
                return self.p(t)
            def forward(self, a, b):
                x = self._id(a)
                y = self._id(b)
                return self.head(x + y)
    """)
    assert _v(src, input_shapes={"a": (2, 8), "b": (2, 8)}).safe is True


def test_unknown_self_attr_still_abstains():
    # A `self.X(...)` that is NOT a method nor a layer (e.g. an opaque runtime
    # attribute) must still abstain soundly rather than crash.
    src = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(16, 4)
            def forward(self, x):
                h = self.runtime_thing(x)
                return self.head(h)
    """)
    g = extract_computation_graph(src)
    assert g.layers["runtime_thing"].kind == LayerKind.UNKNOWN
    # Sound abstention path: extraction still succeeds and verification returns
    # a result without crashing (the opaque attr is not inlined as a method).
    r = _v(src, input_shapes={"x": (2, 8)})
    assert r is not None
    assert r.graph is not None
