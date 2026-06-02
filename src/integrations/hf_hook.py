"""Step 170 — Hugging Face ``from_pretrained`` live-model gate.

``AutoModel.from_pretrained(...)`` / ``MyModel.from_pretrained(...)`` is the
single entry point through which virtually every Transformers user instantiates
a model.  Wrapping it means a shape/device/phase bug in a *custom*
``PreTrainedModel`` subclass (the case the Hub cannot pre-validate) is reported
as one :class:`TensorGuardViolation` the instant the weights are loaded —
before the model is ever moved to a device, wrapped by ``accelerate``, or fed
its first batch by ``Trainer``.

Entry points:

* :func:`verify_pretrained_model` — verify an already-loaded model (the returned
  ``PreTrainedModel`` subclass) and raise/warn/ignore per ``on_violation``.
* :func:`guarded_from_pretrained` — a drop-in for ``cls.from_pretrained`` /
  ``AutoModel.from_pretrained`` that loads the model and then verifies it before
  returning, so a bad checkpoint architecture never escapes the loader.

The verification runs on the model's own ``forward`` source (recovered with
``inspect.getsource``), so it reasons about the architecture the checkpoint
restores, not the weights.  This module never imports ``transformers`` at import
time; it duck-types the loader class it is handed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.framework_hooks import verify_before_training
from src.torch_integration import TensorGuardViolation

__all__ = [
    "TensorGuardViolation",
    "verify_pretrained_model",
    "guarded_from_pretrained",
]


def verify_pretrained_model(
    model: Any,
    *,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
):
    """Verify a loaded ``PreTrainedModel`` subclass instance.

    Returns the ``AnalysisResult`` (or ``None`` when verification abstained
    because the source could not be recovered — e.g. a model class defined in a
    REPL).  ``on_violation`` is ``"raise"`` (default), ``"warn"`` or
    ``"ignore"``.
    """
    return verify_before_training(
        model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )


def guarded_from_pretrained(
    loader: Any,
    *args: Any,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    on_violation: str = "raise",
    soundness_mode: str = "balanced",
    **from_pretrained_kwargs: Any,
):
    """Load a model with ``loader.from_pretrained`` then verify it.

    A drop-in replacement for ``cls.from_pretrained`` / ``AutoModel
    .from_pretrained`` that loads the checkpoint and runs TensorGuard's static
    verification on the *returned* model before handing it back.  On a real bug
    (and ``on_violation="raise"``) it raises :class:`TensorGuardViolation` so the
    misbuilt model is **never returned to the caller** — it cannot reach a
    training or inference pipeline.  (Verification necessarily runs after the
    loader has constructed the model object; it gates the *return value*, not the
    loader's own side effects.)

    ``loader`` is anything exposing ``from_pretrained`` (a ``PreTrainedModel``
    subclass or an ``AutoModel*`` class); ``*args`` / ``**from_pretrained_kwargs``
    are forwarded verbatim.

    Example::

        from transformers import AutoModelForSequenceClassification
        from src.integrations.hf_hook import guarded_from_pretrained

        model = guarded_from_pretrained(
            AutoModelForSequenceClassification,
            "my-org/my-checkpoint",
            input_shapes={"x": ("b", 10)},
        )
    """
    if not hasattr(loader, "from_pretrained"):
        raise TypeError(
            "loader must expose .from_pretrained (a PreTrainedModel subclass or "
            f"AutoModel* class); got {type(loader).__name__}"
        )
    model = loader.from_pretrained(*args, **from_pretrained_kwargs)
    verify_pretrained_model(
        model,
        input_shapes=input_shapes,
        on_violation=on_violation,
        soundness_mode=soundness_mode,
    )
    return model
