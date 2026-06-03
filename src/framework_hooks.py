"""Step 75 — framework-author API: verify a model before training starts.

Framework integrations (PyTorch Lightning, Hugging Face ``Trainer``) get a
one-line way to run TensorGuard's static verification *before* the first
training step, turning a mid-epoch shape/device crash into an immediate,
actionable error at ``fit``/``train`` time.

* :func:`verify_before_training` — the framework-agnostic core: verify a live
  ``nn.Module`` and raise/warn/ignore per ``on_violation``.
* :class:`TensorGuardCallback` — a ``pytorch_lightning.Callback`` that verifies
  the ``LightningModule`` in ``on_fit_start``.
* :class:`TensorGuardTrainerCallback` — a Hugging Face ``TrainerCallback`` that
  verifies the model in ``on_train_begin``.
* ``src.integrations.production_adapters`` — import-light wrappers around
  Lightning ``fit``, HF ``train``, Accelerate ``prepare``, Keras Core ``fit``,
  and Ray Train ``fit``/train-loop choke points.

The callbacks degrade gracefully: if Lightning / Transformers is not installed
they fall back to a plain object base, so importing this module never fails and
the duck-typed hooks remain directly callable (and unit-testable).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.torch_integration import TensorGuardViolation, _check

__all__ = [
    "TensorGuardViolation",
    "verify_before_training",
    "TensorGuardCallback",
    "TensorGuardTrainerCallback",
]


def verify_before_training(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
):
    """Verify a live module before training; raise/warn/ignore per ``on_violation``.

    Returns the ``AnalysisResult`` (or ``None`` if verification abstained because
    the source could not be recovered).  ``on_violation`` is ``"raise"`` (default,
    raises :class:`TensorGuardViolation`), ``"warn"`` or ``"ignore"``.
    """
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(
            f"on_violation must be raise/warn/ignore, got {on_violation!r}"
        )
    return _check(model, input_shapes, on_violation, soundness_mode)


# --------------------------------------------------------------------------- #
# PyTorch Lightning
# --------------------------------------------------------------------------- #
def _lightning_base():
    try:
        from pytorch_lightning import Callback  # type: ignore

        return Callback
    except Exception:  # pragma: no cover - exercised only without lightning
        try:
            from lightning.pytorch import Callback  # type: ignore

            return Callback
        except Exception:
            return object


class TensorGuardCallback(_lightning_base()):  # type: ignore[misc]
    """A Lightning callback that verifies the model at ``on_fit_start``.

    Example::

        trainer = pl.Trainer(callbacks=[TensorGuardCallback(input_shapes={"x": ("b", 10)})])
    """

    def __init__(
        self,
        *,
        input_shapes: Optional[Dict[str, Tuple]] = None,
        on_violation: str = "raise",
        soundness_mode: str = "balanced",
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.on_violation = on_violation
        self.soundness_mode = soundness_mode
        self.last_result = None

    def _run(self, pl_module: Any):
        self.last_result = verify_before_training(
            pl_module,
            input_shapes=self.input_shapes,
            on_violation=self.on_violation,
            soundness_mode=self.soundness_mode,
        )
        return self.last_result

    # Lightning hook signatures.
    def on_fit_start(self, trainer: Any = None, pl_module: Any = None):  # noqa: D401
        if pl_module is not None:
            self._run(pl_module)

    def setup(self, trainer: Any = None, pl_module: Any = None, stage: Any = None):
        # `setup` runs earlier than `on_fit_start`; verify once at fit setup.
        if pl_module is not None and stage in (None, "fit"):
            self._run(pl_module)


# --------------------------------------------------------------------------- #
# Hugging Face Transformers Trainer
# --------------------------------------------------------------------------- #
def _hf_base():
    try:
        from transformers import TrainerCallback  # type: ignore

        return TrainerCallback
    except Exception:  # pragma: no cover - exercised only without transformers
        return object


class TensorGuardTrainerCallback(_hf_base()):  # type: ignore[misc]
    """A HF ``TrainerCallback`` that verifies the model at ``on_train_begin``.

    Example::

        trainer = Trainer(model=model, callbacks=[TensorGuardTrainerCallback(
            input_shapes={"x": ("b", 10)})], ...)
    """

    def __init__(
        self,
        *,
        input_shapes: Optional[Dict[str, Tuple]] = None,
        on_violation: str = "raise",
        soundness_mode: str = "balanced",
    ):
        self.input_shapes = input_shapes
        self.on_violation = on_violation
        self.soundness_mode = soundness_mode
        self.last_result = None

    def on_train_begin(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        *,
        model: Any = None,
        **kwargs: Any,
    ):
        if model is None:
            model = kwargs.get("model")
        if model is not None:
            self.last_result = verify_before_training(
                model,
                input_shapes=self.input_shapes,
                on_violation=self.on_violation,
                soundness_mode=self.soundness_mode,
            )
        return control
