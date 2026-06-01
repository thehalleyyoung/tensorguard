"""
torch.fx → TensorGuard ComputationGraph extraction.

Converts a ``torch.fx.Graph`` (obtained via ``torch.fx.symbolic_trace`` or
``torch.export``) into the internal ``ComputationGraph`` representation used
by ``ConstraintVerifier``.  This enables verification of *arbitrary* PyTorch
``nn.Module`` instances — not just source code — dramatically expanding
coverage to traced, compiled, and dynamically-constructed models.

Usage::

    import torch
    from src.fx_extractor import fx_trace_to_graph, verify_module

    model = torchvision.models.resnet18()
    result = verify_module(model, input_shapes={"x": ("batch", 3, 224, 224)})
    print(result.safe)  # True — shape-safe for all batch sizes
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.fx
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
    Device,
    Phase,
    ConstraintVerifier,
    VerificationResult,
    Confidence,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Module type → LayerKind mapping
# ═══════════════════════════════════════════════════════════════════════════════

_MODULE_KIND_MAP: Dict[type, LayerKind] = {}

def _init_module_kind_map():
    """Lazily initialize the module-to-LayerKind mapping (requires torch)."""
    global _MODULE_KIND_MAP
    if _MODULE_KIND_MAP:
        return
    if not HAS_TORCH:
        return
    _MODULE_KIND_MAP = {
        nn.Linear: LayerKind.LINEAR,
        nn.Conv2d: LayerKind.CONV2D,
        nn.Conv1d: LayerKind.CONV1D,
        nn.ConvTranspose2d: LayerKind.CONVTRANSPOSE2D,
        nn.BatchNorm1d: LayerKind.BATCHNORM1D,
        nn.BatchNorm2d: LayerKind.BATCHNORM2D,
        nn.LayerNorm: LayerKind.LAYERNORM,
        nn.GroupNorm: LayerKind.GROUPNORM,
        nn.InstanceNorm2d: LayerKind.INSTANCENORM2D,
        nn.Dropout: LayerKind.DROPOUT,
        nn.Dropout2d: LayerKind.DROPOUT,
        nn.ReLU: LayerKind.RELU,
        nn.GELU: LayerKind.RELU,
        nn.SiLU: LayerKind.RELU,
        nn.Tanh: LayerKind.RELU,
        nn.Sigmoid: LayerKind.RELU,
        nn.LeakyReLU: LayerKind.RELU,
        nn.ELU: LayerKind.RELU,
        nn.PReLU: LayerKind.RELU,
        nn.SELU: LayerKind.RELU,
        nn.Mish: LayerKind.RELU,
        nn.Softmax: LayerKind.SOFTMAX,
        nn.LogSoftmax: LayerKind.SOFTMAX,
        nn.Embedding: LayerKind.EMBEDDING,
        nn.LSTM: LayerKind.LSTM,
        nn.GRU: LayerKind.GRU,
        nn.MultiheadAttention: LayerKind.MULTIHEAD_ATTENTION,
        nn.MaxPool2d: LayerKind.MAXPOOL2D,
        nn.AvgPool2d: LayerKind.AVGPOOL2D,
        nn.AdaptiveAvgPool2d: LayerKind.ADAPTIVE_AVGPOOL2D,
        nn.Flatten: LayerKind.FLATTEN,
        nn.Sequential: LayerKind.SEQUENTIAL,
        nn.ModuleList: LayerKind.MODULELIST,
        nn.Identity: LayerKind.IDENTITY,
        nn.Upsample: LayerKind.UPSAMPLE,
        nn.TransformerEncoder: LayerKind.TRANSFORMER_ENCODER,
        nn.TransformerDecoder: LayerKind.TRANSFORMER_DECODER,
        nn.TransformerEncoderLayer: LayerKind.TRANSFORMER_ENCODER_LAYER,
        nn.TransformerDecoderLayer: LayerKind.TRANSFORMER_DECODER_LAYER,
        nn.ConvTranspose1d: LayerKind.CONVTRANSPOSE1D,
        nn.AdaptiveMaxPool2d: LayerKind.ADAPTIVE_MAXPOOL2D,
        nn.PixelShuffle: LayerKind.PIXEL_SHUFFLE,
        nn.Unfold: LayerKind.UNFOLD,
        nn.Fold: LayerKind.FOLD,
        nn.InstanceNorm1d: LayerKind.INSTANCENORM1D,
        nn.InstanceNorm3d: LayerKind.INSTANCENORM3D,
        nn.SyncBatchNorm: LayerKind.SYNCBATCHNORM,
        nn.BatchNorm3d: LayerKind.BATCHNORM3D,
        nn.MaxPool1d: LayerKind.MAXPOOL1D,
        nn.AvgPool1d: LayerKind.AVGPOOL1D,
        nn.MaxPool3d: LayerKind.MAXPOOL3D,
        nn.AdaptiveAvgPool1d: LayerKind.ADAPTIVE_AVGPOOL1D,
        nn.AdaptiveMaxPool1d: LayerKind.ADAPTIVE_MAXPOOL1D,
        nn.LPPool2d: LayerKind.LPPOOL2D,
        nn.FractionalMaxPool2d: LayerKind.FRACTIONALMAXPOOL2D,
        nn.RNN: LayerKind.RNN,
        nn.ReflectionPad2d: LayerKind.REFLECTIONPAD2D,
        nn.ReplicationPad2d: LayerKind.REPLICATIONPAD2D,
        nn.ZeroPad2d: LayerKind.ZEROPAD2D,
        nn.ConstantPad2d: LayerKind.CONSTANTPAD2D,
        nn.PixelUnshuffle: LayerKind.PIXEL_UNSHUFFLE,
        nn.AlphaDropout: LayerKind.ALPHADROPOUT,
        nn.Dropout3d: LayerKind.DROPOUT,
        nn.Conv3d: LayerKind.CONV3D,
        nn.ConvTranspose3d: LayerKind.CONVTRANSPOSE3D,
        nn.LogSoftmax: LayerKind.SOFTMAX,
        nn.Hardswish: LayerKind.RELU,
        nn.Hardsigmoid: LayerKind.RELU,
        nn.ReLU6: LayerKind.RELU,
    }


def _module_to_layer_kind(module: "nn.Module") -> LayerKind:
    """Map a live nn.Module instance to a LayerKind enum."""
    _init_module_kind_map()
    for cls, kind in _MODULE_KIND_MAP.items():
        if isinstance(module, cls):
            return kind
    if isinstance(module, nn.Module):
        return LayerKind.SUBMODULE
    return LayerKind.UNKNOWN


def _extract_layer_params(module: "nn.Module", kind: LayerKind) -> Dict[str, Any]:
    """Extract shape-relevant parameters from a live module instance."""
    params: Dict[str, Any] = {}
    if kind == LayerKind.LINEAR:
        params["in_features"] = module.in_features
        params["out_features"] = module.out_features
    elif kind in (LayerKind.CONV2D, LayerKind.CONV1D):
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
        if hasattr(module, 'dilation'):
            params["dilation"] = module.dilation
        if hasattr(module, 'groups'):
            params["groups"] = module.groups
    elif kind == LayerKind.CONVTRANSPOSE2D:
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
        params["output_padding"] = module.output_padding
    elif kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                  LayerKind.INSTANCENORM2D):
        params["num_features"] = module.num_features
    elif kind == LayerKind.LAYERNORM:
        params["normalized_shape"] = module.normalized_shape
    elif kind == LayerKind.GROUPNORM:
        params["num_groups"] = module.num_groups
        params["num_channels"] = module.num_channels
    elif kind == LayerKind.EMBEDDING:
        params["num_embeddings"] = module.num_embeddings
        params["embedding_dim"] = module.embedding_dim
    elif kind in (LayerKind.LSTM, LayerKind.GRU):
        params["input_size"] = module.input_size
        params["hidden_size"] = module.hidden_size
        params["num_layers"] = module.num_layers
        params["bidirectional"] = module.bidirectional
        params["batch_first"] = module.batch_first
    elif kind == LayerKind.MULTIHEAD_ATTENTION:
        params["embed_dim"] = module.embed_dim
        params["num_heads"] = module.num_heads
    elif kind == LayerKind.MAXPOOL2D:
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
    elif kind == LayerKind.AVGPOOL2D:
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
    elif kind == LayerKind.ADAPTIVE_AVGPOOL2D:
        params["output_size"] = module.output_size
    elif kind == LayerKind.UPSAMPLE:
        params["scale_factor"] = module.scale_factor
        params["size"] = module.size
    elif kind == LayerKind.FLATTEN:
        params["start_dim"] = module.start_dim
        params["end_dim"] = module.end_dim
    elif kind == LayerKind.SOFTMAX:
        params["dim"] = module.dim if hasattr(module, 'dim') else -1
    elif kind == LayerKind.DROPOUT:
        params["p"] = module.p
    elif kind == LayerKind.CONVTRANSPOSE1D:
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
        params["output_padding"] = module.output_padding
    elif kind == LayerKind.ADAPTIVE_MAXPOOL2D:
        params["output_size"] = module.output_size
    elif kind == LayerKind.PIXEL_SHUFFLE:
        params["upscale_factor"] = module.upscale_factor
    elif kind == LayerKind.UNFOLD:
        params["kernel_size"] = module.kernel_size
        params["dilation"] = module.dilation
        params["padding"] = module.padding
        params["stride"] = module.stride
    elif kind == LayerKind.FOLD:
        params["output_size"] = module.output_size
        params["kernel_size"] = module.kernel_size
        params["dilation"] = module.dilation
        params["padding"] = module.padding
        params["stride"] = module.stride
    elif kind in (LayerKind.INSTANCENORM1D, LayerKind.INSTANCENORM3D,
                  LayerKind.SYNCBATCHNORM, LayerKind.BATCHNORM3D):
        params["num_features"] = module.num_features
    elif kind in (LayerKind.MAXPOOL1D, LayerKind.AVGPOOL1D):
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
    elif kind == LayerKind.MAXPOOL3D:
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
    elif kind == LayerKind.ADAPTIVE_AVGPOOL1D:
        params["output_size"] = module.output_size
    elif kind == LayerKind.ADAPTIVE_MAXPOOL1D:
        params["output_size"] = module.output_size
    elif kind == LayerKind.LPPOOL2D:
        params["norm_type"] = module.norm_type
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
    elif kind == LayerKind.FRACTIONALMAXPOOL2D:
        params["kernel_size"] = module.kernel_size
        if hasattr(module, 'output_size'):
            params["output_size"] = module.output_size
    elif kind == LayerKind.RNN:
        params["input_size"] = module.input_size
        params["hidden_size"] = module.hidden_size
        params["num_layers"] = module.num_layers
        params["bidirectional"] = module.bidirectional
        params["batch_first"] = module.batch_first
    elif kind in (LayerKind.REFLECTIONPAD2D, LayerKind.REPLICATIONPAD2D,
                  LayerKind.ZEROPAD2D):
        params["padding"] = module.padding
    elif kind == LayerKind.CONSTANTPAD2D:
        params["padding"] = module.padding
        params["value"] = module.value
    elif kind == LayerKind.PIXEL_UNSHUFFLE:
        params["downscale_factor"] = module.downscale_factor
    elif kind == LayerKind.ALPHADROPOUT:
        params["p"] = module.p
    elif kind == LayerKind.CONV3D:
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
    elif kind == LayerKind.CONVTRANSPOSE3D:
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
        params["output_padding"] = module.output_padding
    return params


def _make_layer_def(name: str, module: "nn.Module") -> LayerDef:
    """Create a LayerDef from a live nn.Module instance."""
    kind = _module_to_layer_kind(module)
    params = _extract_layer_params(module, kind)
    ldef = LayerDef(attr_name=name, kind=kind, params=params)
    # Set shortcut fields
    if kind == LayerKind.LINEAR:
        ldef.in_features = params.get("in_features")
        ldef.out_features = params.get("out_features")
    elif kind in (LayerKind.CONV2D, LayerKind.CONV1D, LayerKind.CONVTRANSPOSE2D,
                  LayerKind.CONVTRANSPOSE1D):
        ldef.in_channels = params.get("in_channels")
        ldef.out_channels = params.get("out_channels")
        ks = params.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks) if kind not in (LayerKind.CONV1D, LayerKind.CONVTRANSPOSE1D) else (ks,)
        ldef.kernel_size = ks
    elif kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                  LayerKind.INSTANCENORM2D, LayerKind.INSTANCENORM1D,
                  LayerKind.INSTANCENORM3D, LayerKind.SYNCBATCHNORM,
                  LayerKind.BATCHNORM3D):
        ldef.num_features = params.get("num_features")
    elif kind == LayerKind.EMBEDDING:
        ldef.num_embeddings = params.get("num_embeddings")
        ldef.embedding_dim = params.get("embedding_dim")
    elif kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RNN):
        ldef.hidden_size = params.get("hidden_size")
        ldef.bidirectional = params.get("bidirectional", False)
        ldef.batch_first = params.get("batch_first", False)
    elif kind == LayerKind.MULTIHEAD_ATTENTION:
        ldef.num_heads = params.get("num_heads")
        ldef.in_features = params.get("embed_dim")
    elif kind == LayerKind.ADAPTIVE_AVGPOOL2D:
        out = params.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        ldef.output_size = out
    elif kind == LayerKind.ADAPTIVE_MAXPOOL2D:
        out = params.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        ldef.output_size = out
    elif kind == LayerKind.FOLD:
        out = params.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        ldef.output_size = out
    elif kind in (LayerKind.ADAPTIVE_AVGPOOL1D, LayerKind.ADAPTIVE_MAXPOOL1D):
        out = params.get("output_size")
        if isinstance(out, int):
            out = (out,)
        ldef.output_size = out
    elif kind in (LayerKind.CONV3D, LayerKind.CONVTRANSPOSE3D):
        ldef.in_channels = params.get("in_channels")
        ldef.out_channels = params.get("out_channels")
        ks = params.get("kernel_size")
        if isinstance(ks, int):
            ks = (ks, ks, ks)
        ldef.kernel_size = ks
    elif kind == LayerKind.FRACTIONALMAXPOOL2D:
        out = params.get("output_size")
        if isinstance(out, int):
            out = (out, out)
        ldef.output_size = out
    return ldef


# ═══════════════════════════════════════════════════════════════════════════════
# Function / method → OpKind mapping
# ═══════════════════════════════════════════════════════════════════════════════

def _function_to_op(fn) -> Optional[OpKind]:
    """Map a torch function to an OpKind."""
    if not HAS_TORCH:
        return None
    fn_map = {
        torch.add: OpKind.ADD,
        torch.matmul: OpKind.MATMUL,
        torch.mm: OpKind.MATMUL,
        torch.bmm: OpKind.MATMUL,
        torch.cat: OpKind.CAT,
        torch.stack: OpKind.STACK,
        torch.flatten: OpKind.FLATTEN,
        torch.reshape: OpKind.RESHAPE,
        torch.broadcast_to: OpKind.EXPAND,
        torch.relu: OpKind.ACTIVATION,
        torch.sigmoid: OpKind.ACTIVATION,
        torch.tanh: OpKind.ACTIVATION,
        torch.softmax: OpKind.SOFTMAX,
        torch.dropout: OpKind.DROPOUT,
        torch.where: OpKind.WHERE,
        torch.chunk: OpKind.CHUNK,
        torch.split: OpKind.SPLIT,
        torch.einsum: OpKind.EINSUM,
        torch.gather: OpKind.GATHER,
        torch.index_select: OpKind.INDEX_SELECT,
        torch.scatter: OpKind.SCATTER,
        torch.scatter_add: OpKind.SCATTER,
        torch.masked_select: OpKind.MASKED_SELECT,
        torch.narrow: OpKind.NARROW,
        torch.select: OpKind.SELECT_DIM,
        torch.take: OpKind.TAKE,
    }
    # Also handle torch.nn.functional
    import torch.nn.functional as F
    fn_map.update({
        F.relu: OpKind.ACTIVATION,
        F.gelu: OpKind.ACTIVATION,
        F.silu: OpKind.ACTIVATION,
        F.sigmoid: OpKind.ACTIVATION,
        F.tanh: OpKind.ACTIVATION,
        F.leaky_relu: OpKind.ACTIVATION,
        F.elu: OpKind.ACTIVATION,
        F.softmax: OpKind.SOFTMAX,
        F.log_softmax: OpKind.SOFTMAX,
        F.dropout: OpKind.DROPOUT,
        F.linear: OpKind.LAYER_CALL,
        F.conv2d: OpKind.LAYER_CALL,
        F.conv1d: OpKind.LAYER_CALL,
        F.batch_norm: OpKind.LAYER_CALL,
        F.layer_norm: OpKind.LAYER_CALL,
        F.group_norm: OpKind.LAYER_CALL,
        F.max_pool2d: OpKind.LAYER_CALL,
        F.avg_pool2d: OpKind.LAYER_CALL,
        F.adaptive_avg_pool2d: OpKind.LAYER_CALL,
        F.interpolate: OpKind.INTERPOLATE,
        F.pad: OpKind.PAD,
    })
    if hasattr(F, "scaled_dot_product_attention"):
        fn_map[F.scaled_dot_product_attention] = OpKind.SDPA
    # Handle operator module
    import operator
    fn_map.update({
        operator.add: OpKind.ADD,
        operator.mul: OpKind.MULTIPLY,
        operator.getitem: OpKind.ACTIVATION,  # indexing preserves shape info
    })
    return fn_map.get(fn)


_METHOD_OP_MAP = {
    "view": OpKind.RESHAPE,
    "reshape": OpKind.RESHAPE,
    "flatten": OpKind.FLATTEN,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "transpose": OpKind.TRANSPOSE,
    "permute": OpKind.PERMUTE,
    "contiguous": OpKind.CONTIGUOUS,
    "detach": OpKind.DETACH,
    "to": OpKind.TO_DEVICE,
    "cuda": OpKind.TO_DEVICE,
    "cpu": OpKind.TO_DEVICE,
    "add": OpKind.ADD,
    "mul": OpKind.MULTIPLY,
    "matmul": OpKind.MATMUL,
    "bmm": OpKind.MATMUL,
    "mm": OpKind.MATMUL,
    "softmax": OpKind.SOFTMAX,
    "relu": OpKind.ACTIVATION,
    "sigmoid": OpKind.ACTIVATION,
    "tanh": OpKind.ACTIVATION,
    "expand": OpKind.ACTIVATION,  # shape-preserving in most cases
    "repeat": OpKind.ACTIVATION,
    "chunk": OpKind.CHUNK,
    "split": OpKind.SPLIT,
    "expand": OpKind.EXPAND,
    "repeat": OpKind.REPEAT,
    "mean": OpKind.MEAN_REDUCE,
    "sum": OpKind.SUM_REDUCE,
    # mean/sum handled specially via _handle_reduction_method
}

# Methods that reduce dimensions (need special handling)
_REDUCTION_METHODS = {"mean", "sum"}


# ═══════════════════════════════════════════════════════════════════════════════
# Core: torch.fx.Graph → ComputationGraph
# ═══════════════════════════════════════════════════════════════════════════════

def fx_trace_to_graph(
    traced: "torch.fx.GraphModule",
    class_name: Optional[str] = None,
) -> ComputationGraph:
    """Convert a ``torch.fx.GraphModule`` to a ``ComputationGraph``.

    Parameters
    ----------
    traced : torch.fx.GraphModule
        The traced graph module (from ``torch.fx.symbolic_trace`` or
        ``torch.export``).
    class_name : str, optional
        Override for the class name in the graph. Defaults to the traced
        module's class name.

    Returns
    -------
    ComputationGraph
        The TensorGuard computation graph ready for verification.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for torch.fx integration")

    graph = ComputationGraph(
        class_name=class_name or type(traced).__name__,
    )
    graph.dynamic_features["fx_traced"] = True

    # Step 1: Extract all named modules as LayerDefs
    for name, module in traced.named_modules():
        if name == "":
            continue  # skip root
        # Flatten dotted names: 'layer1.0.conv1' → 'layer1_0_conv1'
        flat_name = name.replace(".", "_")
        ldef = _make_layer_def(flat_name, module)
        graph.layers[flat_name] = ldef

    # Step 2: Walk fx nodes and convert to ComputationSteps
    step_idx = 0
    node_to_tensor: Dict[str, str] = {}  # fx node name → tensor name

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            # Input tensor
            tensor_name = node.name
            node_to_tensor[node.name] = tensor_name
            graph.input_names.append(tensor_name)

        elif node.op == "get_attr":
            # Attribute access (weights, buffers) — skip for shape analysis
            node_to_tensor[node.name] = f"_attr_{node.name}"

        elif node.op == "call_module":
            # nn.Module submodule call
            target = str(node.target)
            flat_target = target.replace(".", "_")
            input_names = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_names.append(node_to_tensor.get(arg.name, arg.name))
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, torch.fx.Node):
                            input_names.append(
                                node_to_tensor.get(a.name, a.name)
                            )

            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            # Look up the actual module for parameter extraction
            submodule = traced.get_submodule(target)
            kind = _module_to_layer_kind(submodule)

            # Ensure layer def exists
            if flat_target not in graph.layers:
                graph.layers[flat_target] = _make_layer_def(flat_target, submodule)

            # Map some module types to specific OpKinds
            if kind == LayerKind.FLATTEN:
                step = ComputationStep(
                    op=OpKind.FLATTEN,
                    inputs=input_names,
                    output=output_name,
                    layer_ref=flat_target,
                    params=_extract_layer_params(submodule, kind),
                )
            elif kind in (LayerKind.RELU, LayerKind.DROPOUT,
                         LayerKind.IDENTITY):
                step = ComputationStep(
                    op=OpKind.ACTIVATION if kind != LayerKind.DROPOUT
                    else OpKind.DROPOUT,
                    inputs=input_names,
                    output=output_name,
                    layer_ref=flat_target,
                )
            else:
                step = ComputationStep(
                    op=OpKind.LAYER_CALL,
                    inputs=input_names,
                    output=output_name,
                    layer_ref=flat_target,
                    params=_extract_layer_params(submodule, kind),
                )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "call_function":
            op_kind = _function_to_op(node.target)
            if op_kind is None:
                # Unknown function — treat as shape-preserving activation
                op_kind = OpKind.ACTIVATION

            input_names = _collect_node_inputs(node, node_to_tensor)
            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            params = _extract_function_params(node, op_kind)

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "call_method":
            method_name = str(node.target)

            input_names = _collect_node_inputs(node, node_to_tensor)
            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            # Handle reduction methods (mean, sum) that reduce spatial dims
            if method_name in _REDUCTION_METHODS:
                steps_to_add = _handle_reduction_method(
                    node, input_names, output_name, step_idx, graph,
                )
                for s in steps_to_add:
                    graph.steps.append(s)
                step_idx += len(steps_to_add)
                continue

            op_kind = _METHOD_OP_MAP.get(method_name, OpKind.ACTIVATION)
            params = _extract_method_params(node, method_name, op_kind)

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "output":
            # Map output node(s)
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    tensor_name = node_to_tensor.get(arg.name, arg.name)
                    graph.output_names.append(tensor_name)
                elif isinstance(arg, (tuple, list)):
                    for a in arg:
                        if isinstance(a, torch.fx.Node):
                            tensor_name = node_to_tensor.get(a.name, a.name)
                            graph.output_names.append(tensor_name)

    return graph


def _collect_node_inputs(
    node: "torch.fx.Node",
    node_to_tensor: Dict[str, str],
) -> List[str]:
    """Collect input tensor names from an fx node's args."""
    inputs = []
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            inputs.append(node_to_tensor.get(arg.name, arg.name))
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if isinstance(a, torch.fx.Node):
                    inputs.append(node_to_tensor.get(a.name, a.name))
    return inputs


def _parse_reshape_dims(shape_args: Tuple) -> Optional[Tuple]:
    """Parse the shape-spec args of a reshape/view call into a dim tuple.

    Handles both the varargs form (``reshape(2, 3)`` →
    ``shape_args == (2, 3)``) and the single tuple/list form
    (``reshape((2, 3))`` or ``torch.reshape(x, (2, 3))`` →
    ``shape_args == ((2, 3),)``).

    Each element becomes an ``int`` (including ``-1``) when concrete, or a
    unique placeholder string ``"_dynN"`` when it is a dynamic / non-int
    argument (e.g. a traced ``Node`` such as ``x.shape[0]``).  Placeholders
    preserve the output rank while signalling an unknown symbolic size, so the
    verifier abstains on that dimension instead of dropping it (which would
    silently corrupt the rank and hide reshape bugs).
    """
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
        seq = list(shape_args[0])
    elif len(shape_args) == 1 and not isinstance(shape_args[0], (int, bool)):
        # A single non-int, non-sequence arg (e.g. a traced ``y.shape`` /
        # ``y.size()`` node) represents an *entire* shape tuple of unknown
        # rank.  Inventing a rank-1 ``_dyn0`` placeholder here would corrupt
        # the rank and can produce false positives (e.g. ``x.expand(y.shape)``
        # flagged as rank-mismatch).  Abstain instead.
        return None
    else:
        seq = list(shape_args)
    if not seq:
        return None
    dims: List[Any] = []
    dyn = 0
    for a in seq:
        if isinstance(a, bool):
            # bool is an int subclass but never a valid dim spec.
            dims.append(f"_dyn{dyn}")
            dyn += 1
        elif isinstance(a, int):
            dims.append(a)
        else:
            dims.append(f"_dyn{dyn}")
            dyn += 1
    return tuple(dims)


def _extract_indexing_params(
    node: "torch.fx.Node",
    op_kind: OpKind,
) -> Dict[str, Any]:
    """Extract `dim` / `start` / `length` for gather/scatter/index ops.

    For both fx ``call_method`` (``node.args[0]`` is the receiver tensor) and
    ``call_function`` (``node.args[0]`` is the input tensor), the ``dim``
    argument is consistently ``node.args[1]``; ``narrow`` additionally takes
    ``start`` at ``args[2]`` and ``length`` at ``args[3]``.  Tensor operands
    (index / mask / src) are collected separately as graph inputs, so only the
    scalar params are captured here.
    """
    params: Dict[str, Any] = {}
    args = node.args
    kwargs = node.kwargs

    def _int_arg(pos: int, key: str) -> Optional[int]:
        if len(args) > pos and isinstance(args[pos], int) and not isinstance(args[pos], bool):
            return args[pos]
        if key in kwargs and isinstance(kwargs[key], int) and not isinstance(kwargs[key], bool):
            return kwargs[key]
        return None

    if op_kind in (
        OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
        OpKind.SELECT_DIM, OpKind.NARROW,
    ):
        d = _int_arg(1, "dim")
        if d is not None:
            params["dim"] = d
    if op_kind == OpKind.NARROW:
        s = _int_arg(2, "start")
        if s is not None:
            params["start"] = s
        ln = _int_arg(3, "length")
        if ln is not None:
            params["length"] = ln
    return params


def _extract_function_params(
    node: "torch.fx.Node",
    op_kind: OpKind,
) -> Dict[str, Any]:
    """Extract shape-relevant params from a function call node."""
    params: Dict[str, Any] = {}
    if op_kind == OpKind.CAT:
        # dim argument
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["dim"] = node.args[1]
        elif "dim" in node.kwargs:
            params["dim"] = node.kwargs["dim"]
        else:
            params["dim"] = 0
    elif op_kind == OpKind.RESHAPE:
        # shape argument(s): torch.reshape(x, shape) — ``shape`` may be a
        # single tuple/list or (rarely) varargs.  Capture the full dim spec
        # (with placeholders for dynamic args) under ``dims`` so the model
        # checker's reshape handlers can see it; keep the int-only
        # ``target_shape`` for backward compatibility with thread_modular.
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
            int_dims = tuple(d for d in dims if isinstance(d, int))
            if int_dims:
                params["target_shape"] = int_dims
    elif op_kind == OpKind.EXPAND:
        # torch.broadcast_to(x, shape): ``shape`` is the dim spec.
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
        params["expand_kind"] = "broadcast_to"
    elif op_kind == OpKind.FLATTEN:
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["start_dim"] = node.args[1]
        if len(node.args) > 2 and isinstance(node.args[2], int):
            params["end_dim"] = node.args[2]
    elif op_kind == OpKind.EINSUM:
        # torch.einsum(equation, *tensors): the equation is the first arg.
        if node.args and isinstance(node.args[0], str):
            params["equation"] = node.args[0]
        elif "equation" in node.kwargs and isinstance(node.kwargs["equation"], str):
            params["equation"] = node.kwargs["equation"]
    elif op_kind in (
        OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
        OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
        OpKind.SELECT_DIM, OpKind.TAKE,
    ):
        params.update(_extract_indexing_params(node, op_kind))
    return params


def _extract_method_params(
    node: "torch.fx.Node",
    method_name: str,
    op_kind: OpKind,
) -> Dict[str, Any]:
    """Extract shape-relevant params from a method call node."""
    params: Dict[str, Any] = {}
    if method_name in ("view", "reshape"):
        # ``x.view(2, 3)`` / ``x.reshape((2, 3))`` / ``x.reshape(b, -1)``.
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
            int_dims = tuple(d for d in dims if isinstance(d, int))
            if int_dims:
                params["target_shape"] = int_dims
    elif method_name == "transpose":
        if len(node.args) >= 3:
            params["dim0"] = node.args[1] if isinstance(node.args[1], int) else 0
            params["dim1"] = node.args[2] if isinstance(node.args[2], int) else 1
    elif method_name == "permute":
        dims = []
        for arg in node.args[1:]:
            if isinstance(arg, int):
                dims.append(arg)
        if dims:
            params["dims"] = tuple(dims)
    elif method_name == "flatten":
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["start_dim"] = node.args[1]
        if len(node.args) > 2 and isinstance(node.args[2], int):
            params["end_dim"] = node.args[2]
    elif method_name in ("expand", "broadcast_to"):
        # ``x.expand(2, 3, 4)`` / ``x.expand((2, 3, 4))`` /
        # ``x.broadcast_to((2, 3, 4))``.  Same arg shapes as reshape: varargs or
        # a single tuple/list, with ``-1`` allowed (keep dim) and dynamic args
        # captured as ``_dynN`` placeholders.
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
        if method_name == "broadcast_to":
            params["expand_kind"] = "broadcast_to"
    elif method_name in ("squeeze", "unsqueeze"):
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["dim"] = node.args[1]
    elif op_kind in (
        OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
        OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
        OpKind.SELECT_DIM, OpKind.TAKE,
    ):
        params.update(_extract_indexing_params(node, op_kind))
    return params


def _handle_reduction_method(
    node: "torch.fx.Node",
    input_names: List[str],
    output_name: str,
    step_idx: int,
    graph: ComputationGraph,
) -> List[ComputationStep]:
    """Handle mean/sum with dimension args as adaptive pooling + flatten.

    When ``mean([2, 3])`` is called on a 4D tensor (common global average
    pooling pattern), convert to AdaptiveAvgPool2d(1,1) + flatten to enable
    correct shape propagation.
    """
    # Extract reduction dims from args
    reduce_dims = None
    if len(node.args) > 1:
        arg1 = node.args[1]
        if isinstance(arg1, (list, tuple)):
            reduce_dims = list(arg1)
        elif isinstance(arg1, int):
            reduce_dims = [arg1]

    # Pattern: mean([2, 3]) on 4D tensor = global average pooling
    if reduce_dims and set(reduce_dims) == {2, 3}:
        # Create synthetic AdaptiveAvgPool2d layer
        pool_name = f"_gap_{step_idx}"
        pool_ldef = LayerDef(
            attr_name=pool_name,
            kind=LayerKind.ADAPTIVE_AVGPOOL2D,
            params={"output_size": (1, 1)},
        )
        pool_ldef.output_size = (1, 1)
        graph.layers[pool_name] = pool_ldef

        # Pool step: (N, C, H, W) → (N, C, 1, 1)
        pool_out = f"_t{step_idx}_pool"
        pool_step = ComputationStep(
            op=OpKind.LAYER_CALL,
            inputs=input_names,
            output=pool_out,
            layer_ref=pool_name,
            params={"output_size": (1, 1)},
        )

        # Flatten step: (N, C, 1, 1) → (N, C)
        flatten_step = ComputationStep(
            op=OpKind.FLATTEN,
            inputs=[pool_out],
            output=output_name,
            params={"start_dim": 1},
        )
        return [pool_step, flatten_step]

    # Fallback: treat as shape-preserving
    return [ComputationStep(
        op=OpKind.ACTIVATION,
        inputs=input_names,
        output=output_name,
    )]


# ═══════════════════════════════════════════════════════════════════════════════
# High-level API: verify_module
# ═══════════════════════════════════════════════════════════════════════════════

def verify_module(
    module: "nn.Module",
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    high_confidence_only: bool = False,
    class_name: Optional[str] = None,
    backend: str = "auto",
) -> VerificationResult:
    """Verify an ``nn.Module`` instance by tracing it with ``torch.fx``.

    This is the torch.fx counterpart to ``verify_model`` (which takes source
    code).  It works on *any* traceable ``nn.Module`` instance, including
    dynamically-constructed models, torchvision models, and HuggingFace
    models.

    Parameters
    ----------
    module : nn.Module
        The model instance to verify.
    input_shapes : dict, optional
        Input shape specification (same format as ``verify_model``).
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth.
    constraints : dict, optional
        Relational constraints between symbolic dimensions.
    high_confidence_only : bool
        When True, only report Z3-proven violations.
    class_name : str, optional
        Override class name (defaults to module's class name).
    backend : str
        Graph capture backend: ``"auto"`` (try Dynamo then fx),
        ``"dynamo"`` (Dynamo only), or ``"fx"`` (torch.fx only).

        **Auto backend selection algorithm** (when ``backend="auto"``):

        The auto selector maximises *coverage* first, then *soundness*:

        1. **Dynamo** (if available and model is traceable): Preferred because
           it covers the most execution paths, including across graph breaks
           caused by data-dependent control flow.  Produces partial subgraphs
           that are individually sound but may miss cross-break constraints
           (PER_SUBGRAPH_SAFE semantics).

        2. **FX** (``torch.fx.symbolic_trace``): Falls back here when Dynamo
           is unavailable or fails.  Captures the full forward graph for
           simple models but misses data-dependent branches entirely — only
           the path taken during tracing is captured.

        3. **AST** (implicit fallback via ``_fallback_module_graph``): Always
           available.  Walks ``named_modules`` without tracing.  Most sound
           for static code structure but cannot resolve dynamic shapes or
           runtime conditionals.

        **Soundness relationships**:

        - *AST*: sees all syntactic branches; misses runtime conditionals,
          closures, dynamically-computed shapes.
        - *FX*: sound within a single trace; misses data-dependent branches
          and cannot handle graph breaks.
        - *Dynamo*: produces partial graphs but handles dynamic control flow;
          each subgraph is individually sound but cross-break shape
          dependencies may be invisible under PER_SUBGRAPH_SAFE.

        Within each backend, a reported violation is a genuine constraint
        failure (soundness).  The gap lies in *completeness*: operations or
        paths not captured cannot be checked.

    Returns

    Examples
    --------
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> result = verify_module(model, input_shapes={"x": ("batch", 784)})
    >>> result.safe
    True
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for verify_module")

    t0 = time.monotonic()

    # Step 0: Try TorchDynamo first (handles data-dependent control flow)
    if backend in ("dynamo", "auto"):
        try:
            from src.dynamo_extractor import (
                dynamo_trace_to_graph, HAS_DYNAMO,
            )
            if HAS_DYNAMO:
                _example = None
                if input_shapes:
                    _example = _build_example_inputs(input_shapes)
                    if _example is not None:
                        _example = tuple(_example.values())
                graph = dynamo_trace_to_graph(
                    module,
                    example_inputs=_example,
                    class_name=class_name or type(module).__name__,
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
        except Exception as dynamo_exc:
            logger.info("TorchDynamo extraction failed: %s; falling back to torch.fx", dynamo_exc)

    # Step 1: Trace the module — try torch.fx first, then torch.compile fallback
    traced = None
    try:
        module.eval()  # eval mode for deterministic tracing
        traced = torch.fx.symbolic_trace(module)
    except Exception as exc:
        logger.info("torch.fx.symbolic_trace failed: %s; trying fallback", exc)
        # Fallback: try tracing with concrete example inputs
        try:
            if input_shapes:
                example_inputs = _build_example_inputs(input_shapes)
                if example_inputs is not None:
                    tracer = torch.fx.Tracer()
                    traced_graph = tracer.trace(module, concrete_args=example_inputs)
                    traced = torch.fx.GraphModule(module, traced_graph)
        except Exception:
            pass

        if traced is None:
            # Final fallback: walk named_modules directly to build graph
            try:
                graph = _fallback_module_graph(module, class_name)
                if graph is not None and graph.steps:
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
            except Exception:
                pass

            return VerificationResult(
                safe=False,
                errors=[f"torch.fx tracing failed: {exc}"],
                verification_time_ms=(time.monotonic() - t0) * 1000,
            )

    # Step 2: Convert fx graph to ComputationGraph
    try:
        cname = class_name or type(module).__name__
        graph = fx_trace_to_graph(traced, class_name=cname)
    except Exception as exc:
        return VerificationResult(
            safe=False,
            errors=[f"FX graph conversion failed: {exc}"],
            verification_time_ms=(time.monotonic() - t0) * 1000,
        )

    # Step 3: Verify using existing ConstraintVerifier
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


# ═══════════════════════════════════════════════════════════════════════════════
# Graph statistics helper
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FXTraceStats:
    """Statistics about a torch.fx trace conversion."""
    traceable: bool
    trace_error: Optional[str] = None
    num_nodes: int = 0
    num_steps: int = 0
    num_layers: int = 0
    num_inputs: int = 0
    num_outputs: int = 0
    layer_kinds: Dict[str, int] = field(default_factory=dict)
    op_kinds: Dict[str, int] = field(default_factory=dict)


def trace_stats(module: "nn.Module") -> FXTraceStats:
    """Get tracing statistics for a module without running verification."""
    if not HAS_TORCH:
        return FXTraceStats(traceable=False, trace_error="PyTorch not available")

    try:
        module.eval()
        traced = torch.fx.symbolic_trace(module)
    except Exception as exc:
        return FXTraceStats(traceable=False, trace_error=str(exc))

    graph = fx_trace_to_graph(traced)

    layer_counts: Dict[str, int] = {}
    for ldef in graph.layers.values():
        k = ldef.kind.name
        layer_counts[k] = layer_counts.get(k, 0) + 1

    op_counts: Dict[str, int] = {}
    for step in graph.steps:
        k = step.op.name
        op_counts[k] = op_counts.get(k, 0) + 1

    return FXTraceStats(
        traceable=True,
        num_nodes=len(list(traced.graph.nodes)),
        num_steps=graph.num_steps,
        num_layers=len(graph.layers),
        num_inputs=len(graph.input_names),
        num_outputs=len(graph.output_names),
        layer_kinds=layer_counts,
        op_kinds=op_counts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback helpers for tracing failures
# ═══════════════════════════════════════════════════════════════════════════════

def _build_example_inputs(
    input_shapes: Dict[str, tuple],
) -> Optional[Dict[str, "torch.Tensor"]]:
    """Build concrete example inputs from shape specs for fallback tracing."""
    if not HAS_TORCH:
        return None
    examples = {}
    for name, shape in input_shapes.items():
        concrete = []
        for d in shape:
            if isinstance(d, int):
                concrete.append(d)
            elif isinstance(d, str):
                concrete.append(2)  # small concrete value for symbolic dims
            else:
                concrete.append(2)
        examples[name] = torch.randn(*concrete)
    return examples


def _fallback_module_graph(
    module: "nn.Module",
    class_name: Optional[str] = None,
) -> Optional[ComputationGraph]:
    """Build a ComputationGraph by walking named_modules when fx tracing fails.

    This produces a linear chain of LAYER_CALL steps — less precise than
    fx tracing but still enables shape verification for sequential-style
    models with data-dependent control flow.
    """
    if not HAS_TORCH:
        return None

    graph = ComputationGraph(
        class_name=class_name or type(module).__name__,
    )
    graph.dynamic_features["fx_traced"] = False
    graph.dynamic_features["fallback_extraction"] = True
    graph.input_names.append("x")

    step_idx = 0
    prev_output = "x"

    for name, child in module.named_children():
        flat_name = name.replace(".", "_")
        ldef = _make_layer_def(flat_name, child)
        graph.layers[flat_name] = ldef

        output_name = f"_t{step_idx}"

        kind = _module_to_layer_kind(child)
        if kind in (LayerKind.RELU, LayerKind.DROPOUT, LayerKind.IDENTITY):
            op = OpKind.ACTIVATION if kind != LayerKind.DROPOUT else OpKind.DROPOUT
        elif kind == LayerKind.FLATTEN:
            op = OpKind.FLATTEN
        else:
            op = OpKind.LAYER_CALL

        step = ComputationStep(
            op=op,
            inputs=[prev_output],
            output=output_name,
            layer_ref=flat_name,
            params=_extract_layer_params(child, kind),
        )
        graph.steps.append(step)
        prev_output = output_name
        step_idx += 1

    # Augment with operations detected via AST analysis of forward()
    _ast_augment_graph(graph, module)

    if graph.steps:
        graph.output_names.append(graph.steps[-1].output)

    return graph if graph.steps else None


# ── AST-based operation detection for fallback graphs ─────────────────────────

# Method name → OpKind for calls on tensors (e.g. x.view(...))
_METHOD_OP_MAP: Dict[str, OpKind] = {
    "view": OpKind.RESHAPE,
    "reshape": OpKind.RESHAPE,
    "flatten": OpKind.FLATTEN,
    "transpose": OpKind.TRANSPOSE,
    "permute": OpKind.PERMUTE,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "mean": OpKind.MEAN_REDUCE,
    "sum": OpKind.SUM_REDUCE,
    "contiguous": OpKind.CONTIGUOUS,
    "detach": OpKind.DETACH,
    "expand": OpKind.EXPAND,
    "expand_as": OpKind.EXPAND,
    "broadcast_to": OpKind.EXPAND,
    "gather": OpKind.GATHER,
    "index_select": OpKind.INDEX_SELECT,
    "scatter": OpKind.SCATTER,
    "scatter_": OpKind.SCATTER,
    "scatter_add": OpKind.SCATTER,
    "scatter_add_": OpKind.SCATTER,
    "masked_select": OpKind.MASKED_SELECT,
    "masked_fill": OpKind.MASKED_FILL,
    "masked_fill_": OpKind.MASKED_FILL,
    "narrow": OpKind.NARROW,
    "select": OpKind.SELECT_DIM,
    "take": OpKind.TAKE,
}

# torch.xxx(...) → OpKind
_TORCH_FUNC_MAP: Dict[str, OpKind] = {
    "cat": OpKind.CAT,
    "stack": OpKind.STACK,
}

# F.xxx(...) → OpKind
_F_FUNC_MAP: Dict[str, OpKind] = {
    "relu": OpKind.ACTIVATION,
    "gelu": OpKind.ACTIVATION,
    "silu": OpKind.ACTIVATION,
    "elu": OpKind.ACTIVATION,
    "leaky_relu": OpKind.ACTIVATION,
    "sigmoid": OpKind.ACTIVATION,
    "tanh": OpKind.ACTIVATION,
    "softmax": OpKind.SOFTMAX,
    "log_softmax": OpKind.SOFTMAX,
    "dropout": OpKind.DROPOUT,
}

# ast BinOp operator → OpKind
_BINOP_MAP: Dict[type, OpKind] = {
    ast.Add: OpKind.ADD,
    ast.Mult: OpKind.MULTIPLY,
    ast.MatMult: OpKind.MATMUL,
}


def _ast_augment_graph(graph: ComputationGraph, module: "nn.Module") -> None:
    """Parse ``forward()`` source via AST to detect ops missed by named_children walk."""
    try:
        source = inspect.getsource(module.forward)
    except (OSError, TypeError):
        return

    try:
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except SyntaxError:
        return

    # Collect the set of layer_ref names already in the graph so we skip those
    existing_layer_refs = {s.layer_ref for s in graph.steps if s.layer_ref}

    detected_ops: List[Tuple[int, OpKind]] = []  # (line, op_kind)

    for node in ast.walk(tree):
        # --- Binary ops: x + y, x * y, x @ y ---
        if isinstance(node, ast.BinOp):
            op = _BINOP_MAP.get(type(node.op))
            if op is not None:
                detected_ops.append((getattr(node, "lineno", 0), op))

        # --- Call nodes ---
        elif isinstance(node, ast.Call):
            func = node.func

            # Method calls: <expr>.method(...)
            if isinstance(func, ast.Attribute):
                method_name = func.attr

                # Skip calls to self.<layer>(...) — already captured
                if (
                    isinstance(func.value, ast.Attribute)
                    and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "self"
                ):
                    attr_name = func.value.attr.replace(".", "_")
                    if attr_name in existing_layer_refs:
                        continue

                # torch.cat / torch.stack
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                    and method_name in _TORCH_FUNC_MAP
                ):
                    detected_ops.append(
                        (getattr(node, "lineno", 0), _TORCH_FUNC_MAP[method_name])
                    )
                    continue

                # F.relu / F.softmax / F.dropout etc.
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "F"
                    and method_name in _F_FUNC_MAP
                ):
                    detected_ops.append(
                        (getattr(node, "lineno", 0), _F_FUNC_MAP[method_name])
                    )
                    continue

                # Tensor method calls: x.view(...), x.reshape(...), ...
                if method_name in _METHOD_OP_MAP:
                    detected_ops.append(
                        (getattr(node, "lineno", 0), _METHOD_OP_MAP[method_name])
                    )

    if not detected_ops:
        return

    # Sort by source line so insertion order matches forward() control flow
    detected_ops.sort(key=lambda t: t[0])

    # Determine starting step index (continue after existing steps)
    step_idx = len(graph.steps)
    prev_output = graph.steps[-1].output if graph.steps else "x"

    for line, op_kind in detected_ops:
        output_name = f"_t{step_idx}"
        step = ComputationStep(
            op=op_kind,
            inputs=[prev_output],
            output=output_name,
            line=line,
        )
        graph.steps.append(step)
        prev_output = output_name
        step_idx += 1
