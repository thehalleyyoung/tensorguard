"""Step 274 — production adapters around real framework choke points.

The tests use real ``torch.nn.Module`` models and verify two properties for each
adapter: a clean model reaches the framework call and executes, while a genuine
shape bug is rejected before the framework mutates/wraps the model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.integrations.production_adapters import (
    ADAPTERS,
    TensorGuardViolation,
    accelerate_prepare_verified,
    adapter_matrix,
    hf_train_verified,
    keras_fit_verified,
    lightning_fit_verified,
    ray_train_verified,
)

_SHAPES = {"x": ("batch", 10)}


class CleanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(30, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _FitTrainer:
    def __init__(self, model=None):
        self.model = model
        self.fit_calls = 0
        self.last_shape = None

    def fit(self, model=None, *args, **kwargs):
        self.fit_calls += 1
        active = model if model is not None else self.model
        self.last_shape = tuple(active(torch.randn(3, 10)).shape)
        return {"fit_calls": self.fit_calls, "shape": self.last_shape}


class _HFTrainer:
    def __init__(self, model):
        self.model = model
        self.train_calls = 0

    def train(self):
        self.train_calls += 1
        return tuple(self.model(torch.randn(2, 10)).shape)


class _Accelerator:
    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, model, *others, **kwargs):
        self.prepare_calls += 1
        model(torch.randn(4, 10))
        return (model, *others) if others else model


class _KerasCoreWrapper:
    def __init__(self, module):
        self.torch_module = module
        self.fit_calls = 0

    def fit(self, *args, **kwargs):
        self.fit_calls += 1
        return tuple(self.torch_module(torch.randn(5, 10)).shape)


class _RayTrainer:
    def __init__(self, model):
        self.module = model
        self.fit_calls = 0

    def fit(self):
        self.fit_calls += 1
        return tuple(self.module(torch.randn(6, 10)).shape)


def test_adapter_matrix_covers_required_frameworks():
    assert adapter_matrix() == ADAPTERS
    assert {adapter.framework for adapter in ADAPTERS} == {
        "lightning",
        "hf_trainer",
        "accelerate",
        "keras_core",
        "ray_train",
    }


def test_lightning_adapter_blocks_bug_before_fit_and_runs_clean_model():
    trainer = _FitTrainer()
    assert lightning_fit_verified(trainer, CleanNet(), input_shapes=_SHAPES)["shape"] == (
        3,
        5,
    )
    assert trainer.fit_calls == 1

    bad_trainer = _FitTrainer()
    with pytest.raises(TensorGuardViolation):
        lightning_fit_verified(bad_trainer, BuggyNet(), input_shapes=_SHAPES)
    assert bad_trainer.fit_calls == 0


def test_hf_trainer_adapter_blocks_bug_before_train_and_runs_clean_model():
    trainer = _HFTrainer(CleanNet())
    assert hf_train_verified(trainer, input_shapes=_SHAPES) == (2, 5)
    assert trainer.train_calls == 1

    bad_trainer = _HFTrainer(BuggyNet())
    with pytest.raises(TensorGuardViolation):
        hf_train_verified(bad_trainer, input_shapes=_SHAPES)
    assert bad_trainer.train_calls == 0


def test_accelerate_adapter_blocks_bug_before_prepare_and_preserves_return_shape():
    accelerator = _Accelerator()
    model = accelerate_prepare_verified(accelerator, CleanNet(), input_shapes=_SHAPES)
    assert isinstance(model, CleanNet)
    assert accelerator.prepare_calls == 1

    bad_accelerator = _Accelerator()
    with pytest.raises(TensorGuardViolation):
        accelerate_prepare_verified(bad_accelerator, BuggyNet(), input_shapes=_SHAPES)
    assert bad_accelerator.prepare_calls == 0


def test_keras_core_adapter_verifies_backing_torch_module_before_fit():
    wrapper = _KerasCoreWrapper(CleanNet())
    assert keras_fit_verified(wrapper, input_shapes=_SHAPES) == (5, 5)
    assert wrapper.fit_calls == 1

    bad_wrapper = _KerasCoreWrapper(BuggyNet())
    with pytest.raises(TensorGuardViolation):
        keras_fit_verified(bad_wrapper, input_shapes=_SHAPES)
    assert bad_wrapper.fit_calls == 0


def test_ray_train_adapter_blocks_bug_before_fit_and_accepts_explicit_model():
    trainer = _RayTrainer(CleanNet())
    assert ray_train_verified(trainer, input_shapes=_SHAPES) == (6, 5)
    assert trainer.fit_calls == 1

    bad_trainer = _RayTrainer(BuggyNet())
    with pytest.raises(TensorGuardViolation):
        ray_train_verified(bad_trainer, input_shapes=_SHAPES)
    assert bad_trainer.fit_calls == 0

    loop_calls = {"n": 0}

    def loop():
        loop_calls["n"] += 1
        return "trained"

    assert ray_train_verified(loop, model=CleanNet(), input_shapes=_SHAPES) == "trained"
    assert loop_calls["n"] == 1
