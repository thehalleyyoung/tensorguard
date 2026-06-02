"""Step 173 — a **second-framework** (JAX / Flax) frontend.

TensorGuard's verification core (:class:`src.model_checker.ConstraintVerifier`)
is deliberately framework-agnostic: it consumes an internal
:class:`~src.model_checker.ComputationGraph` and knows nothing about PyTorch.
Every PyTorch frontend (the AST extractor, ``torch.fx``, ``torch.export``,
TorchDynamo) is just a *lowering* into that IR.

This module adds a frontend for **Flax NNX** (`flax>=0.10`'s current module
system) to demonstrate that the domain — and therefore the soundness story —
generalises beyond PyTorch. A Flax ``nnx.Module`` exposes its sub-layers and,
for ``nnx.Sequential``, an *ordered* pipeline; we walk that structure, read the
live layer attributes (``in_features``/``out_features``/``num_features``/…),
and lower each layer into the very same ``LayerDef`` / ``ComputationStep`` IR
the PyTorch frontends produce. The identical :class:`ConstraintVerifier` then
checks the result — so a Flax MLP whose consecutive ``Dense`` layers disagree
on the feature dimension is flagged **statically, batch-polymorphically, before
any array is allocated**, exactly as for an ``nn.Module``.

Crucially this is a *structural* lowering, not a concrete trace: Flax (like
PyTorch) only raises the dimension-mismatch error when the model is finally
*called* on real data. TensorGuard catches it at construction time and for all
batch sizes at once — the same value proposition, now on a non-PyTorch
frontend.

Usage::

    from flax import nnx
    from src.flax_extractor import verify_flax_module

    model = nnx.Sequential(
        nnx.Linear(784, 256, rngs=nnx.Rngs(0)),
        nnx.relu,
        nnx.Linear(256, 10, rngs=nnx.Rngs(0)),
    )
    result = verify_flax_module(model, input_shapes={"x": ("batch", 784)})
    assert result.safe  # proven shape-safe for every batch size

Soundness note: layers whose shape-transfer function is not modelled here
(e.g. Flax convolutions, which use a channels-*last* NHWC layout that differs
from the verifier's NCHW convention) lower to a soundly-abstaining
``OpKind.UNSUPPORTED`` step rather than being guessed — the output shape goes
fully symbolic and downstream checks abstain, never producing a false alarm.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from flax import nnx  # type: ignore
    HAS_FLAX = True
except Exception:  # pragma: no cover - exercised only when flax is absent
    HAS_FLAX = False

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    ConstraintVerifier,
    Confidence,
    Device,
    LayerDef,
    LayerKind,
    OpKind,
    Phase,
    VerificationResult,
)


# ── activation function recognition ──────────────────────────────────────────
# Flax exposes activations as plain callables (``nnx.relu`` is ``jax.nn.relu``),
# so an entry in an ``nnx.Sequential`` pipeline that is callable but not an
# ``nnx.Module`` is treated as a shape-preserving activation.
_ACTIVATION_NAMES = frozenset({
    "relu", "relu6", "sigmoid", "tanh", "gelu", "elu", "selu", "celu",
    "softplus", "softsign", "silu", "swish", "log_sigmoid", "hard_tanh",
    "hard_sigmoid", "hard_silu", "hard_swish", "leaky_relu", "identity",
    "log_softmax",
})


def _layer_class_name(layer: Any) -> str:
    return type(layer).__name__


def _flax_layer_to_layerdef(name: str, layer: Any) -> Optional[LayerDef]:
    """Lower a single live Flax ``nnx`` layer into a :class:`LayerDef`.

    Returns ``None`` for entries that are not parametric layers (e.g. bare
    activation callables), which the caller turns into activation steps.
    Unrecognised ``nnx.Module`` layers lower to ``LayerKind.UNKNOWN`` so the
    verifier abstains soundly instead of guessing a shape transfer.
    """
    if not HAS_FLAX:
        return None

    cls = _layer_class_name(layer)

    if isinstance(layer, nnx.Linear):
        ldef = LayerDef(
            attr_name=name,
            kind=LayerKind.LINEAR,
            params={"in_features": int(layer.in_features),
                    "out_features": int(layer.out_features)},
        )
        ldef.in_features = int(layer.in_features)
        ldef.out_features = int(layer.out_features)
        return ldef

    if isinstance(layer, nnx.LayerNorm):
        nf = getattr(layer, "num_features", None)
        params: Dict[str, Any] = {}
        if nf is not None:
            params["normalized_shape"] = (int(nf),)
        return LayerDef(attr_name=name, kind=LayerKind.LAYERNORM, params=params)

    if isinstance(layer, nnx.BatchNorm):
        nf = getattr(layer, "num_features", None)
        return LayerDef(
            attr_name=name,
            kind=LayerKind.BATCHNORM1D,
            params={"num_features": int(nf)} if nf is not None else {},
            num_features=int(nf) if nf is not None else None,
        )

    if isinstance(layer, nnx.Dropout):
        return LayerDef(attr_name=name, kind=LayerKind.DROPOUT,
                        params={"p": float(getattr(layer, "rate", 0.0) or 0.0)})

    if isinstance(layer, nnx.Module):
        # A real parametric layer we don't model precisely (e.g. nnx.Conv,
        # which is NHWC and would be mis-checked under the NCHW convention).
        # Abstain soundly.
        logger.info("flax_extractor: abstaining on unmodelled layer %s", cls)
        return LayerDef(attr_name=name, kind=LayerKind.UNKNOWN,
                        params={"flax_class": cls})

    # Not an nnx.Module at all → caller handles (activation / passthrough).
    return None


def _activation_op(layer: Any) -> OpKind:
    """Map a bare callable Sequential entry to a shape-preserving op."""
    fn_name = getattr(layer, "__name__", "") or ""
    if fn_name in _ACTIVATION_NAMES:
        return OpKind.ACTIVATION
    # Unknown callable: treat as a shape-preserving activation conservatively.
    return OpKind.ACTIVATION


def _iter_sequential_layers(model: Any) -> Optional[List[Any]]:
    """Return the ordered pipeline of an ``nnx.Sequential``-like module.

    Accepts a real ``nnx.Sequential`` (reads ``.layers``) or any object that
    exposes an ordered ``layers`` list/tuple of callables. Returns ``None`` if
    the module does not expose an ordered pipeline.
    """
    if HAS_FLAX and isinstance(model, nnx.Sequential):
        return list(model.layers)
    layers = getattr(model, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        return list(layers)
    return None


def flax_module_to_graph(
    model: Any,
    input_name: str = "x",
    class_name: Optional[str] = None,
) -> ComputationGraph:
    """Lower a Flax ``nnx`` model into TensorGuard's ``ComputationGraph``.

    Supports ``nnx.Sequential`` (and any module exposing an ordered ``layers``
    pipeline), recursing into nested Sequentials. Each layer becomes either a
    ``LAYER_CALL`` step against a :class:`LayerDef` (parametric layers) or a
    shape-preserving activation/dropout step (bare callables).
    """
    if not HAS_FLAX:
        raise RuntimeError("flax is required for flax_module_to_graph")

    graph = ComputationGraph(class_name=class_name or _layer_class_name(model))
    graph.dynamic_features["frontend"] = "flax_nnx"
    graph.input_names.append(input_name)

    pipeline = _iter_sequential_layers(model)
    if pipeline is None:
        raise ValueError(
            "flax_module_to_graph currently lowers nnx.Sequential-style "
            "pipelines (a module exposing an ordered `.layers`). Wrap your "
            "layers in nnx.Sequential to verify them."
        )

    counter = [0]

    def _emit(layers: List[Any], prev_output: str) -> str:
        for layer in layers:
            nested = _iter_sequential_layers(layer)
            if nested is not None and HAS_FLAX and isinstance(layer, nnx.Sequential):
                prev_output = _emit(nested, prev_output)
                continue

            idx = counter[0]
            counter[0] += 1
            out = f"_t{idx}"

            ldef = _flax_layer_to_layerdef(f"l{idx}", layer)
            if ldef is None:
                # Bare callable → shape-preserving activation.
                graph.steps.append(ComputationStep(
                    op=_activation_op(layer),
                    inputs=[prev_output],
                    output=out,
                ))
            else:
                graph.layers[ldef.attr_name] = ldef
                if ldef.kind == LayerKind.DROPOUT:
                    op = OpKind.DROPOUT
                elif ldef.kind == LayerKind.UNKNOWN:
                    op = OpKind.UNSUPPORTED
                else:
                    op = OpKind.LAYER_CALL
                graph.steps.append(ComputationStep(
                    op=op,
                    inputs=[prev_output],
                    output=out,
                    layer_ref=ldef.attr_name,
                    params=dict(ldef.params),
                ))
            prev_output = out
        return prev_output

    last = _emit(pipeline, input_name)
    if graph.steps:
        graph.output_names.append(last)
    return graph


def verify_flax_module(
    model: Any,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.EVAL,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Any]] = None,
    high_confidence_only: bool = False,
    class_name: Optional[str] = None,
) -> VerificationResult:
    """Verify a Flax ``nnx`` model with TensorGuard's PyTorch-grade engine.

    This is the Flax counterpart to :func:`src.fx_extractor.verify_module`. It
    lowers the model to the shared ``ComputationGraph`` IR and runs the exact
    same :class:`ConstraintVerifier`, so the verdict (and its certificate /
    counterexample) is produced by identical, already-audited logic.

    Examples
    --------
    >>> from flax import nnx
    >>> good = nnx.Sequential(nnx.Linear(8, 4, rngs=nnx.Rngs(0)), nnx.relu,
    ...                       nnx.Linear(4, 2, rngs=nnx.Rngs(0)))
    >>> verify_flax_module(good, {"x": ("batch", 8)}).safe
    True
    """
    if not HAS_FLAX:
        raise RuntimeError("flax is required for verify_flax_module")

    t0 = time.monotonic()
    graph = flax_module_to_graph(model, class_name=class_name)
    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        constraints=constraints,
    )
    result = checker.verify()
    result.verification_time_ms = (time.monotonic() - t0) * 1000.0
    if high_confidence_only:
        result = result.filter_by_confidence(Confidence.HIGH)
    return result


__all__ = [
    "HAS_FLAX",
    "flax_module_to_graph",
    "verify_flax_module",
]
