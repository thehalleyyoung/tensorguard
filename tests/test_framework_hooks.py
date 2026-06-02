"""Step 75 — framework hooks, proven against real Lightning & HF Trainer."""

import warnings

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from transformers import TrainerCallback

from src.framework_hooks import (
    TensorGuardCallback,
    TensorGuardTrainerCallback,
    TensorGuardViolation,
    verify_before_training,
)


# --- real LightningModules (defined in a file so source is recoverable) ------
class BadLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)  # expects 30, gets 20

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def training_step(self, batch, idx):
        x, y = batch
        return ((self(x) - y) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class GoodLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))

    def training_step(self, batch, idx):
        x, y = batch
        return ((self(x) - y) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader():
    ds = TensorDataset(torch.randn(8, 10), torch.randn(8, 5))
    return DataLoader(ds, batch_size=4)


# --- core --------------------------------------------------------------------
def test_verify_before_training_raises():
    with pytest.raises(TensorGuardViolation):
        verify_before_training(BadLit(), input_shapes={"x": ("batch", 10)})


def test_verify_before_training_invalid_mode():
    with pytest.raises(ValueError):
        verify_before_training(GoodLit(), on_violation="boom")


# --- Lightning ---------------------------------------------------------------
def test_callback_is_real_lightning_callback():
    cb = TensorGuardCallback(input_shapes={"x": ("batch", 10)})
    assert isinstance(cb, pl.Callback)


def test_lightning_hook_raises_on_bad_module():
    cb = TensorGuardCallback(input_shapes={"x": ("batch", 10)})
    with pytest.raises(TensorGuardViolation):
        cb.on_fit_start(trainer=None, pl_module=BadLit())


def test_lightning_real_fit_good_model_runs_hook():
    cb = TensorGuardCallback(input_shapes={"x": ("batch", 10)})
    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        callbacks=[cb],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(GoodLit(), _loader())
    # the hook ran and verification did not flag the good model
    assert cb.last_result is not None
    assert not str(cb.last_result.verdict).upper().endswith("UNSAFE")


def test_lightning_real_fit_bad_model_raises_at_start():
    cb = TensorGuardCallback(input_shapes={"x": ("batch", 10)})
    trainer = pl.Trainer(
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
        callbacks=[cb],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TensorGuardViolation):
            trainer.fit(BadLit(), _loader())


# --- Hugging Face Trainer ----------------------------------------------------
def test_hf_callback_is_real_trainer_callback():
    cb = TensorGuardTrainerCallback(input_shapes={"x": ("batch", 10)})
    assert isinstance(cb, TrainerCallback)


def test_hf_hook_raises_on_bad_model():
    cb = TensorGuardTrainerCallback(input_shapes={"x": ("batch", 10)})
    with pytest.raises(TensorGuardViolation):
        cb.on_train_begin(None, None, None, model=BadLit())


def test_hf_hook_passes_good_model_and_sets_result():
    cb = TensorGuardTrainerCallback(input_shapes={"x": ("batch", 10)})
    control = object()
    out = cb.on_train_begin(None, None, control, model=GoodLit())
    assert out is control
    assert cb.last_result is not None
    assert not str(cb.last_result.verdict).upper().endswith("UNSAFE")


def test_hf_hook_warn_mode():
    cb = TensorGuardTrainerCallback(
        input_shapes={"x": ("batch", 10)}, on_violation="warn"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.on_train_begin(None, None, None, model=BadLit())
    assert any("verification issue" in str(w.message) for w in caught)
