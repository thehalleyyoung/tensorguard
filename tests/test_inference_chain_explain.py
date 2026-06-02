"""Step 58 -- the "why" explainer (``--explain``).

A reported shape bug can be explained by its *inference chain*: the step-by-step
shape propagation from the forward inputs down to the failing op, reconstructed
from the verifier's counterexample trace.  These tests prove the chain is built
correctly from a trace, renders in plain and ANSI form, is attached by
``verify_architecture`` only when a bug is present, and is empty/defensive on
incomplete input.
"""
import textwrap

import pytest

from src.api import verify_architecture
from src.inference_chain import (
    ChainLink,
    InferenceChain,
    build_inference_chain,
    format_chain_plain,
    format_chain_ansi,
)


BAD = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(30, 5)   # expects 30, upstream gives 20
        def forward(self, x):
            h = self.fc1(x)
            return self.fc2(h)
"""

GOOD = BAD.replace("nn.Linear(30, 5)", "nn.Linear(20, 5)")


# ---------------------------------------------------------------------------
# 1. Defensive behavior on empty / partial input.
# ---------------------------------------------------------------------------
def test_empty_inputs_yield_empty_chain():
    assert not build_inference_chain(None, None)
    assert not build_inference_chain(object(), object())
    assert format_chain_plain(InferenceChain("m", -1)) == ""
    assert format_chain_ansi(InferenceChain("m", -1)) == ""


# ---------------------------------------------------------------------------
# 2. End-to-end: verify_architecture attaches the chain for a real bug.
# ---------------------------------------------------------------------------
def test_chain_attached_for_reported_bug():
    src = textwrap.dedent(BAD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert result.bugs
    chain = result.inference_chain
    assert chain is not None and chain
    # The failing step is the second Linear.
    fail = next(l for l in chain.links if l.is_failing)
    assert fail.layer == "fc2"
    # The upstream link produced (batch, 20) -- exactly what fc2 wrongly receives.
    up = chain.links[0]
    assert up.layer == "fc1"
    assert up.output_shape == "(batch, 20)"
    assert "(batch, 20)" in fail.input_shapes


def test_chain_links_are_ordered_and_cover_up_to_failure():
    src = textwrap.dedent(BAD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    chain = result.inference_chain
    idxs = [l.step_index for l in chain.links]
    assert idxs == sorted(idxs)
    assert chain.links[-1].is_failing
    assert chain.failing_step == chain.links[-1].step_index


def test_safe_model_has_no_chain():
    src = textwrap.dedent(GOOD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    assert not result.bugs
    assert result.inference_chain is None


# ---------------------------------------------------------------------------
# 3. Rendering.
# ---------------------------------------------------------------------------
def test_plain_render_shows_propagation_and_marks_failure():
    src = textwrap.dedent(BAD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    out = format_chain_plain(result.inference_chain)
    assert "Why: inference chain for Net" in out
    assert "self.fc1" in out and "self.fc2" in out
    assert "(batch, 10)" in out      # input
    assert "(batch, 20)" in out      # propagated shape
    # The failing link is marked distinctly from the passing links.
    assert "x " in out and "->" in out


def test_ansi_render_has_color_codes():
    src = textwrap.dedent(BAD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    out = format_chain_ansi(result.inference_chain)
    assert "\033[" in out
    assert "Why" in out
    assert "self.fc2" in out


def test_chain_carries_concrete_dims():
    src = textwrap.dedent(BAD)
    result = verify_architecture(src, input_shapes={"x": ("batch", 10)})
    chain = result.inference_chain
    # Z3 picks a concrete witness for the symbolic batch dim.
    assert isinstance(chain.concrete_dims, dict)
    out = format_chain_plain(chain)
    if chain.concrete_dims:
        assert "concrete dimensions" in out


def test_chain_works_with_zero_flag_inference():
    # Combined with Step 56: no -s given, conv-first model, downstream bug.
    src = textwrap.dedent("""
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 10)
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.flatten(1)
                return self.fc(x)
    """)
    result = verify_architecture(src)  # no input_shapes -> inferred
    assert result.bugs
    chain = result.inference_chain
    assert chain is not None and chain
    # The chain starts from the inferred 4-D input on the conv.
    first = chain.links[0]
    assert first.layer == "conv"
    assert "(batch, 3," in first.input_shapes[0]
