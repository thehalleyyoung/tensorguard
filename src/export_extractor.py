"""Step 37 -- a ``torch.export`` frontend, reconciled with the ``torch.fx`` path.

``torch.export`` is PyTorch's officially-supported, ahead-of-time capture path
and the successor to the older ``torch._dynamo.export``. It produces a
normalised ATen-level graph (`linear.default`, `conv2d.default`,
`batch_norm.default`, ...) in which module parameters are *lifted* to graph
inputs. This module lowers that exported graph into TensorGuard's
``ComputationGraph`` so the very same :class:`ConstraintVerifier` engine can
check it -- giving TensorGuard a second, independent frontend whose verdicts can
be reconciled against the ``torch.fx`` frontend.

The lowering recovers each layer's static parameters not from the lifted weight
*tensors* (whose runtime values are irrelevant to shape safety) but by mapping
the lifted-parameter graph inputs back to the live ``nn.Module`` they came from
via ``ExportedProgram.graph_signature`` and reusing the same
``fx_extractor._make_layer_def`` machinery. ATen ops that carry no parameters
(activations, residual adds, flatten, pooling functionals) are mapped directly
to their :class:`OpKind`. Unknown ATen ops abstain soundly as
``OpKind.UNSUPPORTED`` (Step 34), exactly as the fx frontend does.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

try:
    from torch.export import export as _torch_export
    HAS_EXPORT = HAS_TORCH and True
except Exception:  # pragma: no cover
    HAS_EXPORT = False

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
from src.fx_extractor import (
    _make_layer_def,
    _extract_layer_params,
    _module_to_layer_kind,
    _build_example_inputs,
)


# ---------------------------------------------------------------------------
# ATen composite op -> OpKind for parameter-free ops.
# ---------------------------------------------------------------------------
# These are the ATen ops that torch.export emits for the high-level, composite
# operators TensorGuard reasons about that do NOT own nn parameters. Layer ops
# (linear/conv/batch_norm/...) are handled separately because they need a
# LayerDef recovered from the owning module.
_ATEN_ACTIVATION = frozenset({
    "relu", "relu_", "gelu", "gelu_", "silu", "silu_", "sigmoid", "sigmoid_",
    "tanh", "tanh_", "hardtanh", "hardtanh_", "hardswish", "hardswish_",
    "hardsigmoid", "leaky_relu", "leaky_relu_", "elu", "elu_", "selu",
    "mish", "clamp", "clamp_", "abs", "neg", "exp", "log", "sqrt", "rsqrt",
    "clamp_min", "clamp_max", "dropout", "alpha_dropout", "contiguous",
    "clone", "detach", "to", "_to_copy", "type_as", "round", "floor", "ceil",
})
_ATEN_ADD = frozenset({"add", "add_", "sub", "sub_", "rsub"})
_ATEN_MUL = frozenset({"mul", "mul_", "div", "div_"})
_ATEN_MATMUL = frozenset({"matmul", "bmm", "mm"})
_ATEN_FLATTEN = frozenset({"flatten"})
_ATEN_SOFTMAX = frozenset({"softmax", "_softmax", "log_softmax", "_log_softmax"})

# ATen ops that own nn parameters / are recognised layer ops. The value is the
# LayerKind to fall back to if the owning module cannot be recovered.
_ATEN_LAYER_OPS = frozenset({
    "linear", "conv1d", "conv2d", "conv3d", "convolution",
    "batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training",
    "native_batch_norm", "layer_norm", "native_layer_norm", "group_norm",
    "native_group_norm", "max_pool2d", "max_pool2d_with_indices",
    "adaptive_avg_pool2d", "_adaptive_avg_pool2d", "avg_pool2d", "embedding",
})

# Parameter-free pooling / shape functionals that map to a synthetic pooling
# LAYER_CALL only when an owning module is present; otherwise treated as
# shape-altering and handled via the module recovery path. Kept for clarity.
_ATEN_POOL = frozenset({
    "max_pool2d", "max_pool2d_with_indices", "adaptive_avg_pool2d",
    "_adaptive_avg_pool2d", "avg_pool2d",
})


def _aten_base_name(target) -> str:
    """Return the base ATen op name, e.g. ``conv2d`` for ``conv2d.default``."""
    name = getattr(target, "__name__", None)
    if name is None:
        name = getattr(target, "_opname", None) or str(target)
    # 'conv2d.default' -> 'conv2d'; 'aten::conv2d' -> 'conv2d'
    name = name.split(".")[0]
    if "::" in name:
        name = name.split("::")[-1]
    return name


def _op_display_name(target) -> str:
    mod = getattr(target, "__module__", "") or ""
    name = getattr(target, "__name__", None) or str(target)
    if mod:
        return f"{mod}.{name}"
    return name


def _build_param_owner_map(ep) -> Dict[str, str]:
    """Map each lifted-parameter graph-input name to its owning module path.

    e.g. ``{'p_fc1_weight': 'fc1', 'p_fc1_bias': 'fc1', 'b_bn_running_mean':
    'bn'}``. The owning path is the parameter fqn with its trailing component
    (``weight``/``bias``/``running_mean``/...) stripped.
    """
    sig = ep.graph_signature
    owner: Dict[str, str] = {}
    for inp_name, fqn in list(sig.inputs_to_parameters.items()) + list(
            sig.inputs_to_buffers.items()):
        # 'fc1.weight' -> 'fc1'; '0.weight' -> '0'; 'layer1.0.conv1.weight' ->
        # 'layer1.0.conv1'.
        path = fqn.rsplit(".", 1)[0] if "." in fqn else fqn
        owner[inp_name] = path
    return owner


def _as_int_pair(v, default) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return (int(v[0]), int(v[1]))
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return (int(v[0]), int(v[0]))
    if isinstance(v, int):
        return (v, v)
    return default


def _synthesize_pool_layer(base: str, node, graph, step_idx: int,
                           inputs: List[str], output_name: str) -> bool:
    """Emit a precise pooling LAYER_CALL for a parameter-free pool functional.

    Returns True if a step was emitted. ``adaptive_avg_pool2d`` / ``max_pool2d``
    / ``avg_pool2d`` appear in export as functional ATen calls with no owning
    module, but their shape behaviour is fully determined by their args.
    """
    args = list(node.args)
    flat = f"_pool{step_idx}"
    if base in ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
        out = _as_int_pair(args[1] if len(args) > 1 else None, (1, 1))
        ldef = LayerDef(attr_name=flat, kind=LayerKind.ADAPTIVE_AVGPOOL2D,
                        params={"output_size": out})
        ldef.output_size = out
    elif base in ("max_pool2d", "max_pool2d_with_indices", "avg_pool2d"):
        ks = _as_int_pair(args[1] if len(args) > 1 else None, (1, 1))
        stride = _as_int_pair(args[2] if len(args) > 2 and args[2] else ks, ks)
        pad = _as_int_pair(args[3] if len(args) > 3 else 0, (0, 0))
        kind = (LayerKind.AVGPOOL2D if base == "avg_pool2d"
                else LayerKind.MAXPOOL2D)
        ldef = LayerDef(attr_name=flat, kind=kind,
                        params={"kernel_size": ks, "stride": stride,
                                "padding": pad})
    else:
        return False
    graph.layers[flat] = ldef
    graph.steps.append(ComputationStep(
        op=OpKind.LAYER_CALL, inputs=inputs, output=output_name,
        layer_ref=flat, params=dict(ldef.params)))
    return True


def export_trace_to_graph(
    module: "nn.Module",
    example_inputs: Optional[Tuple] = None,
    class_name: Optional[str] = None,
) -> ComputationGraph:
    """Capture a :class:`ComputationGraph` from *module* via ``torch.export``.

    Raises ``RuntimeError`` if ``torch.export`` is unavailable.
    """
    if not HAS_EXPORT:
        raise RuntimeError(
            "torch.export is not available (requires a recent PyTorch). "
            "Use fx_trace_to_graph or verify_module as a fallback."
        )
    if example_inputs is None:
        example_inputs = (torch.randn(2, 64),)

    module.eval()
    _fake_logger = logging.getLogger("torch._subclasses.fake_tensor")
    _prev_level = _fake_logger.level
    _fake_logger.setLevel(logging.CRITICAL)
    try:
        ep = _torch_export(module, tuple(example_inputs))
    finally:
        _fake_logger.setLevel(_prev_level)
    gm = ep.graph_module
    sig = ep.graph_signature
    cname = class_name or type(module).__name__

    graph = ComputationGraph(class_name=cname)
    graph.dynamic_features["export_traced"] = True

    owner_map = _build_param_owner_map(ep)
    named_modules = dict(module.named_modules())
    user_inputs = set(sig.user_inputs)

    # node name -> ComputationGraph tensor name (None for param/buffer inputs)
    node_to_tensor: Dict[str, Optional[str]] = {}
    step_idx = 0

    def _recover_module(node) -> Optional["nn.Module"]:
        """Find the live nn.Module owning a parameter argument of *node*."""
        for arg in node.args:
            arg_name = getattr(arg, "name", None)
            if arg_name in owner_map:
                path = owner_map[arg_name]
                return named_modules.get(path)
        return None

    def _real_inputs(node) -> List[str]:
        """Activation-tensor inputs of *node* (skip lifted params/constants)."""
        names: List[str] = []
        for arg in node.args:
            arg_name = getattr(arg, "name", None)
            if arg_name is None:
                continue
            t = node_to_tensor.get(arg_name)
            if t is not None:
                names.append(t)
        return names

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in user_inputs:
                node_to_tensor[node.name] = node.name
                graph.input_names.append(node.name)
            else:
                # Lifted parameter / buffer -- not an activation tensor.
                node_to_tensor[node.name] = None
            continue

        if node.op in ("get_attr", "output"):
            node_to_tensor[node.name] = None
            continue

        if node.op != "call_function":
            node_to_tensor[node.name] = None
            continue

        base = _aten_base_name(node.target)
        output_name = f"_t{step_idx}"
        node_to_tensor[node.name] = output_name
        inputs = _real_inputs(node)

        if base in _ATEN_LAYER_OPS:
            owner = _recover_module(node)
            if owner is not None:
                kind = _module_to_layer_kind(owner)
                flat = None
                for arg in node.args:
                    an = getattr(arg, "name", None)
                    if an in owner_map:
                        flat = owner_map[an].replace(".", "_")
                        break
                flat = flat or f"_layer{step_idx}"
                if flat not in graph.layers:
                    graph.layers[flat] = _make_layer_def(flat, owner)
                if kind == LayerKind.FLATTEN:
                    op = OpKind.FLATTEN
                elif kind in (LayerKind.RELU, LayerKind.DROPOUT,
                              LayerKind.IDENTITY):
                    op = (OpKind.DROPOUT if kind == LayerKind.DROPOUT
                          else OpKind.ACTIVATION)
                else:
                    op = OpKind.LAYER_CALL
                step = ComputationStep(
                    op=op, inputs=inputs, output=output_name,
                    layer_ref=flat,
                    params=_extract_layer_params(owner, kind),
                )
                graph.steps.append(step)
                step_idx += 1
                continue
            # Parameter-free pooling functional (no owning module).
            if base in _ATEN_POOL:
                if _synthesize_pool_layer(base, node, graph, step_idx,
                                          inputs, output_name):
                    step_idx += 1
                    continue
                graph.steps.append(ComputationStep(
                    op=OpKind.UNSUPPORTED, inputs=inputs, output=output_name,
                    params={"op_name": _op_display_name(node.target)}))
                step_idx += 1
                continue

        op_kind: Optional[OpKind] = None
        if base in _ATEN_ACTIVATION:
            op_kind = OpKind.ACTIVATION
        elif base in _ATEN_ADD:
            op_kind = OpKind.ADD
        elif base in _ATEN_MUL:
            op_kind = OpKind.MULTIPLY
        elif base in _ATEN_MATMUL:
            op_kind = OpKind.MATMUL
        elif base in _ATEN_FLATTEN:
            op_kind = OpKind.FLATTEN
        elif base in _ATEN_SOFTMAX:
            op_kind = OpKind.SOFTMAX

        if op_kind is None:
            graph.steps.append(ComputationStep(
                op=OpKind.UNSUPPORTED, inputs=inputs, output=output_name,
                params={"op_name": _op_display_name(node.target)}))
        else:
            params: Dict[str, object] = {}
            if op_kind == OpKind.FLATTEN:
                ints = [a for a in node.args[1:] if isinstance(a, int)]
                if len(ints) >= 1:
                    params["start_dim"] = ints[0]
                if len(ints) >= 2:
                    params["end_dim"] = ints[1]
            graph.steps.append(ComputationStep(
                op=op_kind, inputs=inputs, output=output_name, params=params))
        step_idx += 1

    return graph


def verify_module_export(
    module: "nn.Module",
    input_shapes: Optional[Dict[str, tuple]] = None,
    example_inputs: Optional[Tuple] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, object]] = None,
    high_confidence_only: bool = False,
    class_name: Optional[str] = None,
) -> VerificationResult:
    """Verify an ``nn.Module`` using the ``torch.export`` frontend."""
    t0 = time.monotonic()
    if example_inputs is None and input_shapes:
        ex = _build_example_inputs(input_shapes)
        if ex is not None:
            example_inputs = tuple(ex.values()) if isinstance(ex, dict) else ex
    try:
        graph = export_trace_to_graph(module, example_inputs=example_inputs,
                                      class_name=class_name)
    except Exception as exc:
        return VerificationResult(
            safe=False,
            errors=[f"torch.export extraction failed: {exc}"],
            verification_time_ms=(time.monotonic() - t0) * 1000,
        )
    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        constraints=constraints,
    )
    result = checker.verify()
    if high_confidence_only:
        result = result.filter_by_confidence(Confidence.HIGH)
    return result
