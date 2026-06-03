"""TensorGuard integration hooks for experiment trackers, CI, and IDEs."""

from __future__ import annotations

from .wandb_hook import WandbHook
from .mlflow_hook import MLflowHook
from .ci_hook import CIHook
from .pytest_plugin import TensorGuardPlugin
from .accelerate_hook import prepare_verified, verify_accelerate_model
from .hf_hook import guarded_from_pretrained, verify_pretrained_model
from .production_adapters import (
    ADAPTERS,
    ProductionAdapter,
    accelerate_prepare_verified,
    adapter_matrix,
    hf_train_verified,
    keras_fit_verified,
    lightning_fit_verified,
    ray_train_verified,
)

__all__ = [
    "WandbHook",
    "MLflowHook",
    "CIHook",
    "TensorGuardPlugin",
    "prepare_verified",
    "verify_accelerate_model",
    "guarded_from_pretrained",
    "verify_pretrained_model",
    "ADAPTERS",
    "ProductionAdapter",
    "accelerate_prepare_verified",
    "adapter_matrix",
    "hf_train_verified",
    "keras_fit_verified",
    "lightning_fit_verified",
    "ray_train_verified",
]
