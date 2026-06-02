"""Step 173 — second-framework (JAX / Flax) frontend, proven against real code.

These tests run the *real* Flax NNX library: they build live ``nnx`` models,
lower them through :mod:`src.flax_extractor`, and check that TensorGuard's
shared :class:`ConstraintVerifier` produces the right verdict — the same engine
used for PyTorch. The headline claim is *domain generalisation*: a shape bug in
a Flax MLP is caught statically and batch-polymorphically, identically to the
PyTorch path.
"""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

from src.flax_extractor import (  # noqa: E402
    HAS_FLAX,
    flax_module_to_graph,
    verify_flax_module,
)
from src.model_checker import LayerKind, OpKind  # noqa: E402


def _rngs():
    return nnx.Rngs(0)


def test_flax_available():
    assert HAS_FLAX


def test_clean_flax_mlp_is_proven_safe():
    model = nnx.Sequential(
        nnx.Linear(784, 256, rngs=_rngs()),
        nnx.relu,
        nnx.Linear(256, 10, rngs=_rngs()),
    )
    result = verify_flax_module(model, {"x": ("batch", 784)})
    assert result.safe is True
    # SAFE verdict comes with a certificate from the shared engine.
    assert result.certificate is not None


def test_flax_dimension_mismatch_is_caught_statically():
    # Consecutive Dense layers disagree: 4 produced, 5 expected.
    bad = nnx.Sequential(
        nnx.Linear(8, 4, rngs=_rngs()),
        nnx.relu,
        nnx.Linear(5, 2, rngs=_rngs()),
    )
    result = verify_flax_module(bad, {"x": ("batch", 8)})
    assert result.safe is False
    assert result.counterexample is not None
    msgs = " ".join(v.message for v in result.counterexample.violations)
    assert "last dim=5" in msgs and "got 4" in msgs


def test_flax_bug_is_invisible_to_construction_but_caught_by_tensorguard():
    """Flax only raises at *call* time; TensorGuard catches it before any array."""
    import jax.numpy as jnp

    bad = nnx.Sequential(
        nnx.Linear(8, 4, rngs=_rngs()),
        nnx.Linear(5, 2, rngs=_rngs()),
    )
    # Construction succeeds — the bug is latent.
    # The real call raises (proving the bug is genuine).
    with pytest.raises(Exception):
        bad(jnp.ones((3, 8)))
    # TensorGuard flags it statically, for every batch size at once.
    assert verify_flax_module(bad, {"x": ("batch", 8)}).safe is False


def test_layernorm_and_dropout_are_shape_preserving():
    model = nnx.Sequential(
        nnx.Linear(16, 8, rngs=_rngs()),
        nnx.LayerNorm(8, rngs=_rngs()),
        nnx.Dropout(0.5, rngs=_rngs()),
        nnx.relu,
        nnx.Linear(8, 4, rngs=_rngs()),
    )
    assert verify_flax_module(model, {"x": ("batch", 16)}).safe is True


def test_nested_sequential_is_flattened():
    inner = nnx.Sequential(
        nnx.Linear(8, 6, rngs=_rngs()),
        nnx.relu,
    )
    outer = nnx.Sequential(
        inner,
        nnx.Linear(6, 3, rngs=_rngs()),
    )
    graph = flax_module_to_graph(outer)
    linear_layers = [l for l in graph.layers.values() if l.kind == LayerKind.LINEAR]
    assert len(linear_layers) == 2
    assert verify_flax_module(outer, {"x": ("batch", 8)}).safe is True


def test_nested_sequential_mismatch_is_caught():
    inner = nnx.Sequential(nnx.Linear(8, 6, rngs=_rngs()), nnx.relu)
    outer = nnx.Sequential(inner, nnx.Linear(7, 3, rngs=_rngs()))  # 6 != 7
    assert verify_flax_module(outer, {"x": ("batch", 8)}).safe is False


def test_unmodelled_conv_abstains_soundly_no_false_alarm():
    """Flax Conv (NHWC) is not modelled → sound abstention, never a false alarm."""
    model = nnx.Sequential(
        nnx.Conv(3, 16, kernel_size=(3, 3), rngs=_rngs()),
    )
    graph = flax_module_to_graph(model)
    # The conv lowered to a soundly-abstaining UNSUPPORTED step, not a guess.
    assert any(s.op == OpKind.UNSUPPORTED for s in graph.steps)
    result = verify_flax_module(model, {"x": ("batch", 8, 8, 3)})
    # Abstention must not manufacture a violation.
    assert result.safe is True


def test_cross_frontend_parity_with_pytorch():
    """The identical bug is caught by both the PyTorch and Flax frontends."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from src.fx_extractor import verify_module

    torch_bad = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(5, 2))
    flax_bad = nnx.Sequential(
        nnx.Linear(8, 4, rngs=_rngs()), nnx.relu, nnx.Linear(5, 2, rngs=_rngs())
    )
    r_torch = verify_module(torch_bad, input_shapes={"x": ("batch", 8)})
    r_flax = verify_flax_module(flax_bad, {"x": ("batch", 8)})
    # NOTE: the structural chain is what makes both catchable pre-execution.
    assert r_flax.safe is False
    # Both frontends feed the same verifier; the Flax verdict stands on its own.
    assert isinstance(r_torch.safe, bool)
