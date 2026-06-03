"""Production trainer adapters for verifying models before framework execution.

The adapters in this module are intentionally import-light: importing
TensorGuard should not require Lightning, Transformers, Accelerate, Keras Core,
or Ray.  Each wrapper duck-types the framework object it is handed, verifies the
original ``torch.nn.Module`` before the framework mutates/wraps it, and then
delegates to the framework's normal choke point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from src.framework_hooks import verify_before_training
from src.integrations.accelerate_hook import prepare_verified
from src.torch_integration import TensorGuardViolation

InputShapes = Optional[Dict[str, Tuple]]


@dataclass(frozen=True)
class ProductionAdapter:
    """A documented production integration surface."""

    framework: str
    entrypoint: str
    verifies_before: str


ADAPTERS: Tuple[ProductionAdapter, ...] = (
    ProductionAdapter("lightning", "lightning_fit_verified", "Trainer.fit"),
    ProductionAdapter("hf_trainer", "hf_train_verified", "Trainer.train"),
    ProductionAdapter("accelerate", "accelerate_prepare_verified", "Accelerator.prepare"),
    ProductionAdapter("keras_core", "keras_fit_verified", "Model.fit"),
    ProductionAdapter("ray_train", "ray_train_verified", "Trainer.fit/train_loop"),
)


def adapter_matrix() -> Tuple[ProductionAdapter, ...]:
    """Return the stable list of production adapters TensorGuard ships."""

    return ADAPTERS


def _torch_module_from(candidate: Any, *, explicit_model: Any = None) -> nn.Module:
    model = explicit_model if explicit_model is not None else candidate
    if isinstance(model, nn.Module):
        return model

    for attr in ("module", "model", "torch_module", "_module", "network"):
        nested = getattr(model, attr, None)
        if isinstance(nested, nn.Module):
            return nested

    getter = getattr(model, "get_model", None)
    if callable(getter):
        nested = getter()
        if isinstance(nested, nn.Module):
            return nested

    raise ValueError(
        "TensorGuard production adapters require a torch.nn.Module, or a wrapper "
        "with one of: module, model, torch_module, _module, network, get_model()."
    )


def _verify(
    candidate: Any,
    *,
    explicit_model: Any = None,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
):
    model = _torch_module_from(candidate, explicit_model=explicit_model)
    return verify_before_training(
        model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )


def lightning_fit_verified(
    trainer: Any,
    lightning_module: Any,
    *fit_args: Any,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **fit_kwargs: Any,
):
    """Verify a Lightning module, then delegate to ``trainer.fit``."""

    _verify(
        lightning_module,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    return trainer.fit(lightning_module, *fit_args, **fit_kwargs)


def hf_train_verified(
    trainer: Any,
    *train_args: Any,
    model: Any = None,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **train_kwargs: Any,
):
    """Verify a Hugging Face Trainer model, then delegate to ``trainer.train``."""

    _verify(
        trainer,
        explicit_model=model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    return trainer.train(*train_args, **train_kwargs)


def accelerate_prepare_verified(
    accelerator: Any,
    model: Any,
    *others: Any,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **prepare_kwargs: Any,
):
    """Verify a model, then delegate to ``Accelerator.prepare``."""

    return prepare_verified(
        accelerator,
        model,
        *others,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
        **prepare_kwargs,
    )


def keras_fit_verified(
    keras_model: Any,
    *fit_args: Any,
    model: Any = None,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **fit_kwargs: Any,
):
    """Verify the PyTorch module backing a Keras Core model, then call ``fit``.

    Keras Core can run on several backends; TensorGuard's static verifier applies
    to PyTorch-backed models.  Pass the backing module explicitly with
    ``model=...`` or expose it as ``torch_module``/``module``/``model`` on the
    Keras wrapper.
    """

    _verify(
        keras_model,
        explicit_model=model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    return keras_model.fit(*fit_args, **fit_kwargs)


def ray_train_verified(
    trainer_or_loop: Any,
    *args: Any,
    model: Any = None,
    input_shapes: InputShapes = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **kwargs: Any,
):
    """Verify a model before a Ray Train ``fit`` call or train-loop function."""

    _verify(
        trainer_or_loop,
        explicit_model=model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    fit = getattr(trainer_or_loop, "fit", None)
    if callable(fit):
        return fit(*args, **kwargs)
    if callable(trainer_or_loop):
        return trainer_or_loop(*args, **kwargs)
    raise ValueError("ray_train_verified expects a Ray trainer with fit() or a train loop.")


__all__ = [
    "ADAPTERS",
    "ProductionAdapter",
    "TensorGuardViolation",
    "accelerate_prepare_verified",
    "adapter_matrix",
    "hf_train_verified",
    "keras_fit_verified",
    "lightning_fit_verified",
    "ray_train_verified",
]
