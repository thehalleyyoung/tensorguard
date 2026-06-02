"""Step 169 — Lightning adoption demo, proven against a real ``Trainer.fit``.

We run the runnable walkthrough (``examples/lightning_guarded_training.py``) end
to end against a genuine ``pytorch_lightning.Trainer``:

* the clean CNN trains a real ``fast_dev_run`` batch and the callback's verdict
  is not UNSAFE;
* the buggy CNN (wrong classifier-head size after flatten) is blocked at
  ``fit`` time with ``TensorGuardViolation`` **before any optimizer step runs**
  (a counter wired into ``training_step`` proves it was never reached);
* the static gate's verdict matches what eager PyTorch would do (the buggy
  forward really does raise at runtime).
"""

from __future__ import annotations

import warnings

import pytest
import torch

pl = pytest.importorskip("pytorch_lightning")

from examples.lightning_guarded_training import (  # noqa: E402
    INPUT_SHAPES,
    BuggyCNN,
    GuardedCNN,
    make_loader,
)
from src.framework_hooks import TensorGuardCallback, TensorGuardViolation  # noqa: E402


def _trainer(cb):
    return pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        callbacks=[cb],
    )


def test_clean_cnn_trains_under_guarded_trainer():
    cb = TensorGuardCallback(input_shapes=INPUT_SHAPES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _trainer(cb).fit(GuardedCNN(), make_loader())
    assert cb.last_result is not None
    assert not str(cb.last_result.verdict).upper().endswith("UNSAFE")


def test_buggy_cnn_blocked_before_any_optimizer_step(monkeypatch):
    steps = {"n": 0}
    orig = BuggyCNN.training_step

    def _counting(self, batch, batch_idx):
        steps["n"] += 1
        return orig(self, batch, batch_idx)

    monkeypatch.setattr(BuggyCNN, "training_step", _counting)

    cb = TensorGuardCallback(input_shapes=INPUT_SHAPES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TensorGuardViolation):
            _trainer(cb).fit(BuggyCNN(), make_loader())
    assert steps["n"] == 0  # never reached a training step
    # A TensorGuardViolation (not a bare RuntimeError) proves the *static* gate
    # fired at fit-start rather than the model crashing mid-batch.


def test_static_verdict_matches_runtime_for_buggy_model():
    # Ground truth: the buggy forward genuinely raises at runtime.
    with pytest.raises(RuntimeError):
        BuggyCNN()(torch.randn(2, 3, 32, 32))
    # The clean forward genuinely runs and produces (b, num_classes).
    assert GuardedCNN()(torch.randn(2, 3, 32, 32)).shape == (2, 10)
