"""Step 166 — ``accelerate`` integration, proven against a real ``Accelerator``.

``prepare_verified`` wraps the single choke point every model passes through
before training (``Accelerator.prepare``) so a real shape bug surfaces as one
``TensorGuardViolation`` *before* the model is wrapped/moved, rather than as a
mid-epoch crash inside the AMP/DDP shim.  We prove this end-to-end against a
real CPU ``Accelerator`` (no GPU, no distributed launch needed):

* a clean model is verified and then genuinely prepared (forward still runs);
* a buggy model raises ``TensorGuardViolation`` *before* ``prepare`` is reached
  (we assert ``prepare`` was never called via a spy);
* the bare-object vs tuple return contract of ``prepare`` is preserved;
* ``on_violation="warn"`` warns but still prepares.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

accelerate = pytest.importorskip("accelerate")
from accelerate import Accelerator  # noqa: E402

from src.integrations.accelerate_hook import (  # noqa: E402
    TensorGuardViolation,
    prepare_verified,
    verify_accelerate_model,
)

_SHAPES = {"x": ("b", 10)}


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(20, 5)

    def forward(self, x):
        return self.b(self.a(x)).relu()


class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 20)
        self.b = nn.Linear(30, 5)  # expects 30, gets 20 -> real shape bug

    def forward(self, x):
        return self.b(self.a(x))


class _SpyAccelerator:
    """Wraps a real Accelerator and records whether ``prepare`` was reached."""

    def __init__(self, inner):
        self.inner = inner
        self.prepare_calls = 0

    def prepare(self, *objs, **kw):
        self.prepare_calls += 1
        return self.inner.prepare(*objs, **kw)


def test_verify_accelerate_model_raises_on_bug():
    with pytest.raises(TensorGuardViolation):
        verify_accelerate_model(BuggyNet(), input_shapes=_SHAPES)


def test_prepare_verified_clean_really_prepares_and_runs():
    acc = Accelerator(cpu=True)
    model = prepare_verified(acc, CleanNet(), input_shapes=_SHAPES)
    # The returned object is a real, runnable module.
    out = model(torch.randn(4, 10))
    assert out.shape == (4, 5)


def test_prepare_verified_blocks_bug_before_prepare():
    spy = _SpyAccelerator(Accelerator(cpu=True))
    with pytest.raises(TensorGuardViolation):
        prepare_verified(spy, BuggyNet(), input_shapes=_SHAPES)
    # Verification is the first side effect: prepare must never have been called.
    assert spy.prepare_calls == 0


def test_prepare_verified_preserves_tuple_contract():
    acc = Accelerator(cpu=True)
    model = CleanNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(torch.randn(8, 10), torch.randn(8, 5)), batch_size=4)

    out = prepare_verified(acc, model, opt, loader, input_shapes=_SHAPES)
    assert isinstance(out, tuple) and len(out) == 3
    pmodel, popt, ploader = out
    assert pmodel(torch.randn(2, 10)).shape == (2, 5)
    assert isinstance(popt, torch.optim.Optimizer)


def test_prepare_verified_single_returns_bare_object():
    acc = Accelerator(cpu=True)
    out = prepare_verified(acc, CleanNet(), input_shapes=_SHAPES)
    assert not isinstance(out, tuple)


def test_prepare_verified_warn_mode_still_prepares():
    spy = _SpyAccelerator(Accelerator(cpu=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = prepare_verified(
            spy, BuggyNet(), input_shapes=_SHAPES, on_violation="warn"
        )
    assert spy.prepare_calls == 1
    assert out is not None
    assert any("TensorGuard" in str(w.message) for w in caught)
