"""Step 166 — Hugging Face ``accelerate`` integration.

``accelerate`` is the dominant launcher for multi-GPU / mixed-precision PyTorch
training, and ``Accelerator.prepare(model, optimizer, dataloader, ...)`` is the
single choke point every model passes through before the first training step.
Wrapping it means a shape/device/phase bug is reported *at prepare time* — one
clear ``TensorGuardViolation`` — instead of as a mid-epoch crash inside an
``accelerate``-wrapped, AMP/DDP-shimmed module where the traceback no longer
points at the user's ``forward``.

Two entry points:

* :func:`verify_accelerate_model` — verify a live ``nn.Module`` (the *original*
  model, before any ``accelerate`` wrapping) and raise/warn/ignore per
  ``on_violation``.
* :func:`prepare_verified` — a drop-in wrapper for ``accelerator.prepare`` that
  verifies the model first and then delegates, preserving ``prepare``'s exact
  return contract (a bare object for ``prepare(model)``; a tuple for
  ``prepare(model, opt, loader)``).

The verification runs *before* ``prepare`` so it inspects the user's own
``forward`` rather than the AMP/DDP wrapper ``prepare`` would return.  This
proves the original module is statically shape/device safe; it does not model
runtime dtype/device transforms ``prepare`` itself applies (documented
limitation).

This module never imports ``accelerate`` at import time, so it is safe to import
in any environment; the wrapper duck-types the accelerator it is handed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.framework_hooks import verify_before_training
from src.torch_integration import TensorGuardViolation

__all__ = [
    "TensorGuardViolation",
    "verify_accelerate_model",
    "prepare_verified",
]


def verify_accelerate_model(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
):
    """Verify a live model before it is handed to ``accelerate``.

    Returns the ``AnalysisResult`` (or ``None`` when verification abstained
    because the source could not be recovered).  ``on_violation`` is ``"raise"``
    (default, raises :class:`TensorGuardViolation`), ``"warn"`` or ``"ignore"``.
    """
    return verify_before_training(
        model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )


def prepare_verified(
    accelerator: Any,
    model: Any,
    *others: Any,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **prepare_kwargs: Any,
):
    """Verify *model*, then call ``accelerator.prepare(model, *others, ...)``.

    A drop-in replacement for ``accelerator.prepare`` that runs TensorGuard's
    static verification on the *original* ``model`` first.  On a real bug (and
    ``on_violation="raise"``) it raises :class:`TensorGuardViolation` *before*
    ``prepare`` is called, so the model is never wrapped/moved.  Otherwise it
    delegates to ``accelerator.prepare`` and returns its result unchanged,
    preserving the bare-object-vs-tuple contract.

    Example::

        from accelerate import Accelerator
        from src.integrations.accelerate_hook import prepare_verified

        accelerator = Accelerator()
        model, optimizer, loader = prepare_verified(
            accelerator, model, optimizer, loader, input_shapes={"x": ("b", 10)}
        )
    """
    verify_accelerate_model(
        model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    return accelerator.prepare(model, *others, **prepare_kwargs)
