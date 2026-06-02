"""Step 43 — graceful degradation for unanalyzable forward regions.

When a single ``forward`` statement cannot be extracted (an unsupported
construct or an internal extraction error), TensorGuard isolates that statement
— recording its line, reason and source, and rebinding its target to a sound
fully-symbolic tensor — and continues verifying the rest of the model, instead
of abandoning the whole module.  The analyzable remainder is still verified, and
genuine bugs on independent paths are still caught.

The extraction frontend is intentionally hard to make raise (it already abstains
on unknown ops), so these tests force the failure deterministically by patching
one extractor visitor to raise, exercising the isolation safety net directly.
"""
import ast
import textwrap

import pytest

import src.model_checker as mc
from src.model_checker import (
    extract_computation_graph,
    verify_model,
    OpKind,
)


@pytest.fixture
def boom_on_bad(monkeypatch):
    """Patch the call visitor so any ``self.bad(...)`` call raises."""
    orig = mc._ForwardExtractor._process_call

    def patched(self, node, target, line, col):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "bad":
            raise RuntimeError("boom in bad")
        return orig(self, node, target, line, col)

    monkeypatch.setattr(mc._ForwardExtractor, "_process_call", patched)
    return patched


MODEL = textwrap.dedent("""
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.Linear(8, 16)
            self.bad = nn.Linear(16, 16)
            self.head = nn.Linear(16, 4)
        def forward(self, x):
            h = self.good(x)
            b = self.bad(h)
            return self.head(h)
""")


def test_isolated_region_is_recorded(boom_on_bad):
    g = extract_computation_graph(MODEL)
    assert len(g.isolated_regions) == 1
    region = g.isolated_regions[0]
    assert "bad" in region["source"]
    assert region["reason"].startswith("RuntimeError")
    assert region["line"] > 0


def test_isolated_target_becomes_unsupported_step(boom_on_bad):
    g = extract_computation_graph(MODEL)
    # The failed statement's target `b` is rebound via a sound UNSUPPORTED step.
    iso_steps = [s for s in g.steps
                 if s.op == OpKind.UNSUPPORTED
                 and s.params.get("isolated")]
    assert any(s.output == "b" for s in iso_steps)


def test_rest_of_model_still_verifies(boom_on_bad):
    # head consumes `h` (from the analyzable `good`), so the model is provably
    # safe even though `bad` was isolated.
    r = verify_model(MODEL, input_shapes={"x": (2, 8)})
    assert r.safe is True
    assert len(r.isolated_regions) == 1


def test_downstream_bug_on_analyzable_path_still_caught(boom_on_bad):
    # head now expects 99 but receives 16 from `good` — independent of the
    # isolated `bad` — so the bug must still be flagged.
    bad_model = MODEL.replace("nn.Linear(16, 4)", "nn.Linear(99, 4)")
    r = verify_model(bad_model, input_shapes={"x": (2, 8)})
    assert r.safe is False
    assert len(r.isolated_regions) == 1


def test_isolated_output_is_sound_no_false_positive(boom_on_bad):
    # Route the head off the *isolated* output `b` (fully symbolic). The checker
    # must abstain rather than fabricate a violation (soundness: no false
    # positive on an unanalyzable value).
    src = MODEL.replace("return self.head(h)", "return self.head(b)")
    r = verify_model(src, input_shapes={"x": (2, 8)})
    assert r.safe is True


def test_no_isolation_for_clean_model():
    # A fully-analyzable model has no isolated regions.
    clean = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return self.fc(x)
    """)
    g = extract_computation_graph(clean)
    assert g.isolated_regions == []
    r = verify_model(clean, input_shapes={"x": (2, 8)})
    assert r.safe is True
    assert r.isolated_regions == []


def test_multiple_isolated_regions(monkeypatch):
    orig = mc._ForwardExtractor._process_call

    def patched(self, node, target, line, col):
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("bad1", "bad2"):
            raise ValueError("unsupported")
        return orig(self, node, target, line, col)

    monkeypatch.setattr(mc._ForwardExtractor, "_process_call", patched)
    src = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.good = nn.Linear(8, 16)
                self.bad1 = nn.Linear(16, 16)
                self.bad2 = nn.Linear(16, 16)
                self.head = nn.Linear(16, 4)
            def forward(self, x):
                h = self.good(x)
                a = self.bad1(h)
                c = self.bad2(h)
                return self.head(h)
    """)
    g = extract_computation_graph(src)
    assert len(g.isolated_regions) == 2
    r = verify_model(src, input_shapes={"x": (2, 8)})
    assert r.safe is True
