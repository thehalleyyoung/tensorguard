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
    TensorShape,
    ShapeDim,
    TensorValueRange,
    _TENSOR_FACTORY_FNS,
    _canon_dtype,
    _is_int_dtype,
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
    for name, kind in (
        ("ConstantPad1d", LayerKind.CONSTANTPAD1D),
        ("ConstantPad3d", LayerKind.CONSTANTPAD3D),
        ("ZeroPad1d", LayerKind.ZEROPAD1D),
        ("ZeroPad3d", LayerKind.ZEROPAD3D),
        ("ReflectionPad1d", LayerKind.REFLECTIONPAD1D),
        ("ReflectionPad3d", LayerKind.REFLECTIONPAD3D),
        ("ReplicationPad1d", LayerKind.REPLICATIONPAD1D),
        ("ReplicationPad3d", LayerKind.REPLICATIONPAD3D),
        ("CircularPad1d", LayerKind.CIRCULARPAD1D),
        ("CircularPad2d", LayerKind.CIRCULARPAD2D),
        ("CircularPad3d", LayerKind.CIRCULARPAD3D),
    ):
        cls = getattr(nn, name, None)
        if cls is not None:
            _MODULE_KIND_MAP[cls] = kind


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
        if hasattr(module, 'dilation'):
            params["dilation"] = module.dilation
        if hasattr(module, 'groups'):
            params["groups"] = module.groups
    elif kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                  LayerKind.INSTANCENORM2D):
        params["num_features"] = module.num_features
        if hasattr(module, 'track_running_stats'):
            params["track_running_stats"] = module.track_running_stats
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
        params["kdim"] = getattr(module, "kdim", module.embed_dim)
        params["vdim"] = getattr(module, "vdim", module.embed_dim)
        params["batch_first"] = getattr(module, "batch_first", False)
        same_qkv = getattr(module, "_qkv_same_embed_dim", True)
        params["use_separate_proj_weight"] = not bool(same_qkv)
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
        params["mode"] = getattr(module, "mode", "nearest")
        params["align_corners"] = getattr(module, "align_corners", None)
        params["recompute_scale_factor"] = getattr(
            module, "recompute_scale_factor", None
        )
        if hasattr(module, "antialias"):
            params["antialias"] = module.antialias
        params["__interpolate_args_observed__"] = True
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
        if hasattr(module, 'dilation'):
            params["dilation"] = module.dilation
        if hasattr(module, 'groups'):
            params["groups"] = module.groups
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
        if hasattr(module, 'track_running_stats'):
            params["track_running_stats"] = module.track_running_stats
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
    elif kind in (LayerKind.REFLECTIONPAD1D, LayerKind.REFLECTIONPAD2D,
                  LayerKind.REFLECTIONPAD3D, LayerKind.REPLICATIONPAD1D,
                  LayerKind.REPLICATIONPAD2D, LayerKind.REPLICATIONPAD3D,
                  LayerKind.ZEROPAD1D, LayerKind.ZEROPAD2D,
                  LayerKind.ZEROPAD3D, LayerKind.CIRCULARPAD1D,
                  LayerKind.CIRCULARPAD2D, LayerKind.CIRCULARPAD3D):
        params["padding"] = module.padding
    elif kind in (LayerKind.CONSTANTPAD1D, LayerKind.CONSTANTPAD2D,
                  LayerKind.CONSTANTPAD3D):
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
        if hasattr(module, 'dilation'):
            params["dilation"] = module.dilation
        if hasattr(module, 'groups'):
            params["groups"] = module.groups
    elif kind == LayerKind.CONVTRANSPOSE3D:
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        params["stride"] = module.stride
        params["padding"] = module.padding
        params["output_padding"] = module.output_padding
        if hasattr(module, 'dilation'):
            params["dilation"] = module.dilation
        if hasattr(module, 'groups'):
            params["groups"] = module.groups
    # Record the layer's parameter dtype (used by the dtype algebra). Reading
    # the live weight captures .half()/.to(dtype=) casts applied to the module.
    w = getattr(module, "weight", None)
    if w is not None and hasattr(w, "dtype"):
        params["param_dtype"] = str(w.dtype).replace("torch.", "")
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
        ldef.batch_first = bool(params.get("batch_first", False))
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
        torch.squeeze: OpKind.SQUEEZE,
        torch.unsqueeze: OpKind.UNSQUEEZE,
        torch.movedim: OpKind.MOVEDIM,
        torch.broadcast_to: OpKind.EXPAND,
        torch.repeat_interleave: OpKind.REPEAT_INTERLEAVE,
        torch.tile: OpKind.TILE,
        torch.broadcast_tensors: OpKind.BROADCAST_TENSORS,
        torch.permute: OpKind.PERMUTE,
        torch.transpose: OpKind.TRANSPOSE,
        torch.swapaxes: OpKind.TRANSPOSE,
        torch.swapdims: OpKind.TRANSPOSE,
        torch.roll: OpKind.ROLL,
        torch.rot90: OpKind.ROT90,
        torch.flip: OpKind.FLIP,
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
        torch.nonzero: OpKind.NONZERO,
        torch.narrow: OpKind.NARROW,
        torch.select: OpKind.SELECT_DIM,
        torch.take: OpKind.TAKE,
        torch.take_along_dim: OpKind.TAKE_ALONG_DIM,
        torch.argsort: OpKind.ARGSORT,
        torch.sort: OpKind.SORT,
        torch.topk: OpKind.TOPK,
        torch.kthvalue: OpKind.KTHVALUE,
        torch.argmax: OpKind.ARG_REDUCE,
        torch.argmin: OpKind.ARG_REDUCE,
    }
    for alias in ("hstack", "vstack", "dstack", "column_stack", "row_stack"):
        alias_fn = getattr(torch, alias, None)
        if alias_fn is not None:
            fn_map[alias_fn] = OpKind.STACK
    for alias in ("moveaxis",):
        alias_fn = getattr(torch, alias, None)
        if alias_fn is not None:
            fn_map[alias_fn] = OpKind.MOVEDIM
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
        F.unfold: OpKind.LAYER_CALL,
        F.fold: OpKind.LAYER_CALL,
        F.interpolate: OpKind.INTERPOLATE,
        F.pad: OpKind.PAD,
    })
    if hasattr(F, "scaled_dot_product_attention"):
        fn_map[F.scaled_dot_product_attention] = OpKind.SDPA
    # Handle operator module
    import operator
    fn_map.update({
        operator.add: OpKind.ADD,
        operator.sub: OpKind.ADD,        # element-wise; same broadcast rule
        operator.mul: OpKind.MULTIPLY,
        operator.truediv: OpKind.MULTIPLY,  # element-wise broadcast
        operator.matmul: OpKind.MATMUL,  # the ``@`` operator
        operator.getitem: OpKind.ACTIVATION,  # refined per-node below
    })
    return fn_map.get(fn)


_METHOD_OP_MAP = {
    "view": OpKind.RESHAPE,
    "reshape": OpKind.RESHAPE,
    "flatten": OpKind.FLATTEN,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "movedim": OpKind.MOVEDIM,
    "moveaxis": OpKind.MOVEDIM,
    "transpose": OpKind.TRANSPOSE,
    "swapaxes": OpKind.TRANSPOSE,
    "swapdims": OpKind.TRANSPOSE,
    "permute": OpKind.PERMUTE,
    "roll": OpKind.ROLL,
    "rot90": OpKind.ROT90,
    "flip": OpKind.FLIP,
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
    "repeat_interleave": OpKind.REPEAT_INTERLEAVE,
    "tile": OpKind.TILE,
    "take_along_dim": OpKind.TAKE_ALONG_DIM,
    "argsort": OpKind.ARGSORT,
    "sort": OpKind.SORT,
    "topk": OpKind.TOPK,
    "kthvalue": OpKind.KTHVALUE,
    "argmax": OpKind.ARG_REDUCE,
    "argmin": OpKind.ARG_REDUCE,
    "mean": OpKind.MEAN_REDUCE,
    "sum": OpKind.SUM_REDUCE,
    # mean/sum handled specially via _handle_reduction_method
    "half": OpKind.DTYPE_CAST,
    "float": OpKind.DTYPE_CAST,
    "double": OpKind.DTYPE_CAST,
    "bfloat16": OpKind.DTYPE_CAST,
    "long": OpKind.DTYPE_CAST,
    "int": OpKind.DTYPE_CAST,
    "short": OpKind.DTYPE_CAST,
    "bool": OpKind.DTYPE_CAST,
    "type_as": OpKind.DTYPE_CAST,
}

# Methods that reduce dimensions (need special handling)
_REDUCTION_METHODS = {"mean", "sum"}


# Unary, element-wise operations whose output shape is provably identical to the
# input shape. Mapping these to a shape-preserving ACTIVATION is sound. Anything
# *not* on an allowlist and not otherwise modelled is treated as UNSUPPORTED (a
# sound abstention) rather than guessed to be shape-preserving.
_SHAPE_PRESERVING_METHODS = frozenset({
    # activations / nonlinearities not already in _METHOD_OP_MAP
    "relu", "relu_", "sigmoid", "sigmoid_", "tanh", "tanh_",
    "gelu", "silu", "elu", "selu", "celu", "hardswish", "hardsigmoid",
    "hardtanh", "leaky_relu", "mish", "softplus", "softsign", "relu6",
    "logsigmoid",
    # elementwise math (all unary, shape-preserving)
    "abs", "absolute", "neg", "negative", "exp", "exp2", "expm1", "log",
    "log2", "log10", "log1p", "sqrt", "rsqrt", "square", "reciprocal",
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "asinh",
    "acosh", "atanh", "erf", "erfc", "erfinv", "sign", "sgn", "signbit",
    "floor", "ceil", "round", "trunc", "frac", "clamp", "clamp_min",
    "clamp_max", "clip", "nan_to_num", "sigmoid_", "pow", "clone",
    "isnan", "isinf", "isfinite", "logical_not", "bitwise_not",
    "masked_fill", "tril", "triu", "deg2rad", "rad2deg", "positive",
})

if HAS_TORCH:
    import torch.nn.functional as _F

    def _maybe(mod, name):
        return getattr(mod, name, None)

    _SHAPE_PRESERVING_FUNCTIONS = frozenset(
        fn for fn in (
            _maybe(torch, "abs"), _maybe(torch, "neg"), _maybe(torch, "exp"),
            _maybe(torch, "expm1"), _maybe(torch, "log"), _maybe(torch, "log2"),
            _maybe(torch, "log10"), _maybe(torch, "log1p"),
            _maybe(torch, "sqrt"), _maybe(torch, "rsqrt"),
            _maybe(torch, "square"), _maybe(torch, "reciprocal"),
            _maybe(torch, "sin"), _maybe(torch, "cos"), _maybe(torch, "tan"),
            _maybe(torch, "erf"), _maybe(torch, "erfc"), _maybe(torch, "sign"),
            _maybe(torch, "floor"), _maybe(torch, "ceil"),
            _maybe(torch, "round"), _maybe(torch, "trunc"),
            _maybe(torch, "frac"), _maybe(torch, "clamp"),
            _maybe(torch, "clip"), _maybe(torch, "nan_to_num"),
            _maybe(torch, "logical_not"), _maybe(torch, "isnan"),
            _maybe(torch, "isinf"), _maybe(torch, "isfinite"),
            _maybe(torch, "tril"), _maybe(torch, "triu"),
            _maybe(_F, "gelu"), _maybe(_F, "silu"), _maybe(_F, "mish"),
            _maybe(_F, "hardswish"), _maybe(_F, "hardsigmoid"),
            _maybe(_F, "hardtanh"), _maybe(_F, "softplus"),
            _maybe(_F, "softsign"), _maybe(_F, "logsigmoid"),
            _maybe(_F, "log_softmax"), _maybe(_F, "celu"), _maybe(_F, "selu"),
        )
        if fn is not None
    )
else:
    _SHAPE_PRESERVING_FUNCTIONS = frozenset()


def _maybe_tensor_factory(node, output_name: str):
    """If ``node`` is a tensor-factory call with statically-known size
    (``torch.zeros(4, 6)``, ``torch.randn((2, 3))``, ``torch.full((2, 2), v)``,
    ``torch.randint(10, (3,))``, ``torch.randperm(n)``), build a NEW_TENSOR step
    carrying the fixed shape. Seed-independent: the values are random/zero but
    the shape is statically determined. Returns ``None`` for non-factory calls
    or dynamic sizes (→ caller abstains)."""
    target = node.target
    short = getattr(target, "__name__", None)
    if short is None:
        return None
    if short != "randperm" and short not in _TENSOR_FACTORY_FNS:
        return None

    def _as_dims(vals) -> Optional[list]:
        dims = []
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, int):
                return None
            dims.append(ShapeDim(v))
        return dims

    args = list(node.args)
    if short == "randperm":
        if not args or not isinstance(args[0], int) or isinstance(args[0], bool):
            return None
        shape = TensorShape((ShapeDim(args[0]),))
        dtype_family = "int"
    else:
        if short == "full":
            size_args = [args[0]] if args else []
        elif short == "randint":
            size_args = [args[-1]] if args else []
        else:
            size_args = args
        if len(size_args) == 1 and isinstance(size_args[0], (tuple, list)):
            elts = list(size_args[0])
        else:
            elts = size_args
        if not elts:
            return None
        dims = _as_dims(elts)
        if dims is None:
            return None
        shape = TensorShape(tuple(dims))
        dtype_family = _TENSOR_FACTORY_FNS.get(short, "")

    params: Dict[str, Any] = {"shape": shape, "dtype_family": dtype_family}
    dev = node.kwargs.get("device")
    if isinstance(dev, str):
        params["device"] = dev
    return ComputationStep(
        op=OpKind.NEW_TENSOR, inputs=[], output=output_name, params=params,
    )


def _op_display_name(target) -> str:
    """Human-readable name of an fx call target for the unsupported-op
    diagnostic, e.g. ``torch.fft.fft`` or ``Tensor.unfold``."""
    mod = getattr(target, "__module__", None)
    name = getattr(target, "__name__", None)
    if name:
        if mod and mod not in ("builtins", "_operator"):
            return f"{mod}.{name}"
        return name
    return repr(target)


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
    tuple_output_info: Dict[str, Dict[str, Any]] = {}
    stable_tensor_targets = {
        name for name, _ in traced.named_buffers(recurse=True)
    } | {
        name for name, _ in traced.named_parameters(recurse=True)
    }

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            # Input tensor
            tensor_name = node.name
            node_to_tensor[node.name] = tensor_name
            graph.input_names.append(tensor_name)

        elif node.op == "get_attr":
            # Attribute access (weights, buffers, fx-folded constants).
            tname = f"_attr_{node.name}"
            node_to_tensor[node.name] = tname
            # If the attribute is a constant tensor (e.g. torch.fx folded a
            # ``torch.rand(2, 4)`` written in forward into a constant), record
            # its shape so downstream ops are not forced to abstain.  The
            # value is random but the SHAPE is seed-independent.
            try:
                obj = traced
                for part in str(node.target).split("."):
                    obj = getattr(obj, part)
                if torch.is_tensor(obj) and obj.dim() > 0:
                    graph.const_shapes[tname] = TensorShape(
                        tuple(ShapeDim(int(d)) for d in obj.shape)
                    )
                    dt = _canon_dtype(obj.dtype)
                    if dt is not None:
                        graph.const_dtypes[tname] = dt
                    dev_type = obj.device.type
                    if dev_type == "cuda":
                        idx = obj.device.index or 0
                        graph.const_devices[tname] = Device.from_string(
                            f"cuda:{idx}"
                        )
                    else:
                        graph.const_devices[tname] = Device.CPU
                    target_name = str(node.target)
                    is_generated_constant = any(
                        part.startswith("_tensor_constant")
                        or part.startswith("_constant")
                        for part in target_name.split(".")
                    )
                    if (
                        target_name in stable_tensor_targets
                        and not is_generated_constant
                        and dt is not None
                        and _is_int_dtype(dt)
                        and obj.numel() > 0
                        and obj.numel() <= 1_000_000
                    ):
                        try:
                            vals = obj.detach().cpu()
                            graph.const_value_ranges[tname] = TensorValueRange(
                                int(vals.min().item()),
                                int(vals.max().item()),
                            )
                        except Exception:
                            pass
            except Exception:
                pass

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
            elif kind == LayerKind.MULTIHEAD_ATTENTION:
                mha_inputs, mha_params = _collect_mha_call_inputs(
                    node, node_to_tensor
                )
                layer_params = _extract_layer_params(submodule, kind)
                layer_params.update(mha_params)
                step = ComputationStep(
                    op=OpKind.LAYER_CALL,
                    inputs=mha_inputs,
                    output=output_name,
                    layer_ref=flat_target,
                    params=layer_params,
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
            # Functional embedding: F.embedding(input, weight) /
            # torch.embedding(weight, input) produces input.shape + (embed_dim,).
            # Mapping it to a shape-preserving ACTIVATION (the generic fallback)
            # would be confidently wrong, so build a synthetic Embedding layer
            # from the weight constant's shape and emit a LAYER_CALL.
            emb_step = _maybe_functional_embedding(
                node, node_to_tensor, graph, step_idx
            )
            if emb_step is not None:
                graph.steps.append(emb_step)
                node_to_tensor[node.name] = emb_step.output
                step_idx += 1
                continue

            fold_step = _maybe_functional_fold_unfold(
                node, node_to_tensor, graph, step_idx
            )
            if fold_step is not None:
                graph.steps.append(fold_step)
                node_to_tensor[node.name] = fold_step.output
                step_idx += 1
                continue

            op_kind = _function_to_op(node.target)
            output_name = f"_t{step_idx}"
            tuple_getitem_params: Dict[str, Any] = {}
            display_name = _op_display_name(node.target)
            if display_name == "getattr" and len(node.args) >= 2:
                base_arg = node.args[0]
                attr_arg = node.args[1]
                if (isinstance(base_arg, torch.fx.Node)
                        and base_arg.name in tuple_output_info
                        and attr_arg in ("values", "indices")):
                    op_kind = OpKind.ACTIVATION
                    tuple_getitem_params = dict(tuple_output_info[base_arg.name])
                    tuple_getitem_params["tuple_index"] = (
                        1 if attr_arg == "indices" else 0
                    )
            if display_name == "getitem" and len(node.args) >= 2:
                def _has_tensor_index(value: Any) -> bool:
                    if isinstance(value, torch.fx.Node):
                        return True
                    if isinstance(value, (tuple, list)):
                        return any(_has_tensor_index(v) for v in value)
                    return False

                base_arg = node.args[0]
                item_arg = node.args[1]
                if (isinstance(base_arg, torch.fx.Node)
                        and base_arg.name in tuple_output_info
                        and isinstance(item_arg, int)
                        and not isinstance(item_arg, bool)):
                    op_kind = OpKind.ACTIVATION
                    tuple_getitem_params = dict(tuple_output_info[base_arg.name])
                    tuple_getitem_params["tuple_index"] = item_arg
                elif _has_tensor_index(node.args[1]):
                    op_kind = OpKind.BOOLEAN_INDEX
            if op_kind is None:
                factory_step = _maybe_tensor_factory(node, output_name)
                if factory_step is not None:
                    # Tensor factory with statically-known shape (seed-independent).
                    graph.steps.append(factory_step)
                    node_to_tensor[node.name] = output_name
                    step_idx += 1
                    continue
                if node.target in _SHAPE_PRESERVING_FUNCTIONS:
                    # Provably shape-preserving unary elementwise op.
                    op_kind = OpKind.ACTIVATION
                else:
                    # Unknown function — do NOT guess that it preserves shape
                    # (unsound for shape-changing ops). Abstain soundly and
                    # record the op for an "unsupported op: …" diagnostic.
                    op_kind = OpKind.UNSUPPORTED

            input_names = _collect_node_inputs(node, node_to_tensor)
            node_to_tensor[node.name] = output_name

            params = _extract_function_params(node, op_kind)
            params.update(tuple_getitem_params)
            if op_kind == OpKind.UNSUPPORTED:
                params["op_name"] = _op_display_name(node.target)

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            if op_kind in (OpKind.SORT, OpKind.TOPK, OpKind.KTHVALUE):
                tuple_output_info[node.name] = {
                    "tuple_source_op": op_kind.name.lower(),
                }
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

            if method_name in _METHOD_OP_MAP:
                op_kind = _METHOD_OP_MAP[method_name]
            elif method_name in _SHAPE_PRESERVING_METHODS:
                op_kind = OpKind.ACTIVATION
            else:
                # Unknown method — abstain soundly rather than assume it
                # preserves shape, and surface it as an unsupported op.
                op_kind = OpKind.UNSUPPORTED
            params = _extract_method_params(node, method_name, op_kind)
            if op_kind == OpKind.UNSUPPORTED:
                params["op_name"] = f"Tensor.{method_name}"

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            if op_kind in (OpKind.SORT, OpKind.TOPK, OpKind.KTHVALUE):
                tuple_output_info[node.name] = {
                    "tuple_source_op": op_kind.name.lower(),
                }
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


def _collect_mha_call_inputs(
    node: "torch.fx.Node",
    node_to_tensor: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Collect q/k/v and mask operands for a MultiheadAttention module call.

    ``step.inputs`` is positional and otherwise loses whether a tensor came from
    ``query``, ``key``, ``value``, ``attn_mask`` or ``key_padding_mask``.  Store a
    parallel role list so the verifier can consume kwargs soundly.
    """
    inputs: List[str] = []
    roles: List[str] = []

    def add(role: str, value: Any) -> None:
        if isinstance(value, torch.fx.Node):
            inputs.append(node_to_tensor.get(value.name, value.name))
            roles.append(role)

    positional_roles = ("query", "key", "value")
    for role, arg in zip(positional_roles, node.args):
        add(role, arg)
    for role in positional_roles[len(node.args):]:
        if role in node.kwargs:
            add(role, node.kwargs[role])
    if len(node.args) > 3:
        add("key_padding_mask", node.args[3])
    if len(node.args) > 5:
        add("attn_mask", node.args[5])
    add("key_padding_mask", node.kwargs.get("key_padding_mask"))
    add("attn_mask", node.kwargs.get("attn_mask"))

    params: Dict[str, Any] = {"__mha_input_roles__": roles}
    if len(node.args) > 4 and isinstance(node.args[4], bool):
        params["need_weights"] = node.args[4]
    if len(node.args) > 6 and isinstance(node.args[6], bool):
        params["average_attn_weights"] = node.args[6]
    if len(node.args) > 7 and isinstance(node.args[7], bool):
        params["is_causal"] = node.args[7]
    for name in ("need_weights", "average_attn_weights", "is_causal"):
        if name in node.kwargs:
            params[name] = node.kwargs[name]
    return inputs, params


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


def _literal_or_dynamic(value: Any) -> Tuple[Any, bool]:
    """Return a literal FX argument or mark it dynamic if it depends on a node."""
    if isinstance(value, torch.fx.Node):
        return None, True
    if isinstance(value, (tuple, list)):
        items = []
        dynamic = False
        for item in value:
            if isinstance(item, torch.fx.Node):
                dynamic = True
                items.append(None)
            else:
                items.append(item)
        if dynamic:
            return None, True
        return tuple(items) if isinstance(value, tuple) else list(items), False
    return value, False


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
    if op_kind == OpKind.NONZERO:
        if len(args) > 1 and isinstance(args[1], bool):
            params["as_tuple"] = args[1]
        if "as_tuple" in kwargs and isinstance(kwargs["as_tuple"], bool):
            params["as_tuple"] = kwargs["as_tuple"]
    if op_kind == OpKind.TAKE_ALONG_DIM:
        d = _int_arg(2, "dim")
        if d is not None:
            params["dim"] = d
        elif "dim" in kwargs and kwargs["dim"] is None:
            params["dim"] = None
    if op_kind in (OpKind.ARGSORT, OpKind.SORT):
        d = _int_arg(1, "dim")
        params["dim"] = d if d is not None else -1
    if op_kind == OpKind.ARG_REDUCE:
        d = _int_arg(1, "dim")
        if d is not None:
            params["dim"] = d
        elif "dim" in kwargs:
            params["dim"] = kwargs["dim"] if kwargs["dim"] is None else params.get("dim")
        if len(args) > 2 and isinstance(args[2], bool):
            params["keepdim"] = args[2]
        if "keepdim" in kwargs and isinstance(kwargs["keepdim"], bool):
            params["keepdim"] = kwargs["keepdim"]
    if op_kind in (OpKind.TOPK, OpKind.KTHVALUE):
        k = _int_arg(1, "k")
        if k is not None:
            params["k"] = k
        d = _int_arg(2, "dim")
        params["dim"] = d if d is not None else -1
        if op_kind == OpKind.KTHVALUE:
            if len(args) > 3 and isinstance(args[3], bool):
                params["keepdim"] = args[3]
            if "keepdim" in kwargs and isinstance(kwargs["keepdim"], bool):
                params["keepdim"] = kwargs["keepdim"]
    return params


def _extract_function_params(
    node: "torch.fx.Node",
    op_kind: OpKind,
) -> Dict[str, Any]:
    """Extract shape-relevant params from a function call node."""
    params: Dict[str, Any] = {}

    def _set_literal(key: str, value: Any, dynamic_key: str) -> None:
        literal, dynamic = _literal_or_dynamic(value)
        params[key] = literal
        if dynamic:
            params[dynamic_key] = True

    if op_kind == OpKind.CAT:
        # dim argument
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["dim"] = node.args[1]
        elif "dim" in node.kwargs:
            params["dim"] = node.kwargs["dim"]
        else:
            params["dim"] = 0
    elif op_kind == OpKind.STACK:
        target_name = getattr(node.target, "__name__", "")
        params["stack_kind"] = target_name if target_name else "stack"
        if target_name == "stack":
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
    elif op_kind == OpKind.TILE:
        reps = _parse_reshape_dims(tuple(node.args[1:]))
        if reps is not None:
            params["reps"] = reps
    elif op_kind == OpKind.REPEAT_INTERLEAVE:
        if len(node.args) > 1:
            _set_literal("repeats", node.args[1], "__repeats_dynamic__")
        elif "repeats" in node.kwargs:
            _set_literal(
                "repeats", node.kwargs["repeats"], "__repeats_dynamic__")
        if len(node.args) > 2:
            _set_literal("dim", node.args[2], "__dim_dynamic__")
        elif "dim" in node.kwargs:
            _set_literal("dim", node.kwargs["dim"], "__dim_dynamic__")
        if "output_size" in node.kwargs:
            _set_literal(
                "output_size",
                node.kwargs["output_size"],
                "__output_size_dynamic__",
            )
    elif op_kind == OpKind.FLATTEN:
        if len(node.args) > 1 and isinstance(node.args[1], int):
            params["start_dim"] = node.args[1]
        if len(node.args) > 2 and isinstance(node.args[2], int):
            params["end_dim"] = node.args[2]
    elif op_kind == OpKind.SQUEEZE:
        if len(node.args) > 1:
            _set_literal("dim", node.args[1], "__dim_dynamic__")
        elif "dim" in node.kwargs:
            _set_literal("dim", node.kwargs["dim"], "__dim_dynamic__")
    elif op_kind == OpKind.UNSQUEEZE:
        if len(node.args) > 1:
            _set_literal("dim", node.args[1], "__dim_dynamic__")
        elif "dim" in node.kwargs:
            _set_literal("dim", node.kwargs["dim"], "__dim_dynamic__")
    elif op_kind == OpKind.MOVEDIM:
        if len(node.args) > 1:
            _set_literal("source", node.args[1], "__source_dynamic__")
        elif "source" in node.kwargs:
            _set_literal("source", node.kwargs["source"], "__source_dynamic__")
        if len(node.args) > 2:
            _set_literal(
                "destination", node.args[2], "__destination_dynamic__")
        elif "destination" in node.kwargs:
            _set_literal(
                "destination",
                node.kwargs["destination"],
                "__destination_dynamic__",
            )
    elif op_kind == OpKind.ROLL:
        if len(node.args) > 1:
            _set_literal("shifts", node.args[1], "__shifts_dynamic__")
        elif "shifts" in node.kwargs:
            _set_literal("shifts", node.kwargs["shifts"], "__shifts_dynamic__")
        if len(node.args) > 2:
            params["__dims_present__"] = True
            _set_literal("dims", node.args[2], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            params["__dims_present__"] = True
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif op_kind == OpKind.ROT90:
        if len(node.args) > 1:
            _set_literal("k", node.args[1], "__k_dynamic__")
        elif "k" in node.kwargs:
            _set_literal("k", node.kwargs["k"], "__k_dynamic__")
        if len(node.args) > 2:
            _set_literal("dims", node.args[2], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif op_kind == OpKind.FLIP:
        params["__requires_sequence_dims__"] = True
        if len(node.args) > 1:
            _set_literal("dims", node.args[1], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif op_kind == OpKind.INTERPOLATE:
        params["__interpolate_args_observed__"] = True

        def _set_arg(key: str, value: Any, dynamic_key: str) -> None:
            literal, dynamic = _literal_or_dynamic(value)
            params[key] = literal
            if dynamic:
                params[dynamic_key] = True

        if len(node.args) > 1:
            params["__size_present__"] = True
            _set_arg("size", node.args[1], "__size_dynamic__")
        if len(node.args) > 2:
            params["__scale_factor_present__"] = True
            _set_arg(
                "scale_factor",
                node.args[2],
                "__scale_factor_dynamic__",
            )
        for pos, key in (
            (3, "mode"),
            (4, "align_corners"),
            (5, "recompute_scale_factor"),
            (6, "antialias"),
        ):
            if len(node.args) > pos:
                params[key], _ = _literal_or_dynamic(node.args[pos])
        for key in (
            "size",
            "scale_factor",
            "mode",
            "align_corners",
            "recompute_scale_factor",
            "antialias",
        ):
            if key in node.kwargs:
                if key == "size":
                    params["__size_present__"] = True
                    _set_arg("size", node.kwargs[key], "__size_dynamic__")
                elif key == "scale_factor":
                    params["__scale_factor_present__"] = True
                    _set_arg(
                        "scale_factor",
                        node.kwargs[key],
                        "__scale_factor_dynamic__",
                    )
                else:
                    params[key], _ = _literal_or_dynamic(node.kwargs[key])
    elif op_kind == OpKind.PERMUTE:
        # torch.permute(x, dims): ``dims`` is a single tuple/list, but tolerate
        # the varargs spelling torch.permute(x, 0, 2, 1) as well.
        rest = node.args[1:]
        if len(rest) == 1 and isinstance(rest[0], (tuple, list)):
            rest = tuple(rest[0])
        dims = tuple(d for d in rest if isinstance(d, int))
        if dims:
            params["dims"] = dims
    elif op_kind == OpKind.TRANSPOSE:
        # torch.transpose / swapaxes / swapdims (x, dim0, dim1).
        if len(node.args) >= 3:
            _set_literal("dim0", node.args[1], "__dim0_dynamic__")
            _set_literal("dim1", node.args[2], "__dim1_dynamic__")
        for key, dynamic_key in (
            ("dim0", "__dim0_dynamic__"),
            ("dim1", "__dim1_dynamic__"),
        ):
            if key in node.kwargs:
                _set_literal(key, node.kwargs[key], dynamic_key)
    elif op_kind == OpKind.EINSUM:
        # torch.einsum(equation, *tensors): the equation is the first arg.
        if node.args and isinstance(node.args[0], str):
            params["equation"] = node.args[0]
        elif "equation" in node.kwargs and isinstance(node.kwargs["equation"], str):
            params["equation"] = node.kwargs["equation"]
    elif op_kind == OpKind.PAD:
        if len(node.args) > 1:
            params["pad"] = node.args[1]
        elif "pad" in node.kwargs:
            params["pad"] = node.kwargs["pad"]
        if len(node.args) > 2:
            params["mode"] = node.args[2]
        elif "mode" in node.kwargs:
            params["mode"] = node.kwargs["mode"]
        if len(node.args) > 3:
            params["value"] = node.args[3]
        elif "value" in node.kwargs:
            params["value"] = node.kwargs["value"]
    elif op_kind in (
        OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
        OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
        OpKind.SELECT_DIM, OpKind.TAKE, OpKind.NONZERO,
        OpKind.BOOLEAN_INDEX, OpKind.TAKE_ALONG_DIM, OpKind.ARGSORT,
        OpKind.SORT, OpKind.TOPK, OpKind.KTHVALUE, OpKind.ARG_REDUCE,
    ):
        params.update(_extract_indexing_params(node, op_kind))
        if op_kind == OpKind.NONZERO and "as_tuple" in node.kwargs:
            params["as_tuple"] = node.kwargs["as_tuple"]
    return params


def _extract_to_device(node: "torch.fx.Node") -> Optional[str]:
    """Extract a device target from a ``.to(...)`` call node, if statically known.

    Handles ``x.to('cuda')``, ``x.to(torch.device('cuda:1'))``,
    ``x.to(device='cpu')`` and the combined ``x.to(device, dtype)`` form.
    Returns ``None`` when no static device argument is present (pure dtype cast,
    or a device taken from another traced tensor such as ``x.to(y.device)``)."""
    import torch as _torch

    def _as_device_str(val: Any) -> Optional[str]:
        if isinstance(val, _torch.device):
            return str(val)
        if isinstance(val, str):
            s = val.strip().lower()
            # Only accept recognised device spellings; a bare dtype-like or
            # arbitrary string must not be misread as a device.
            if s == "cpu" or s.startswith("cuda"):
                return s
        return None

    # Positional device arg (first non-self positional that parses as a device).
    for arg in node.args[1:]:
        d = _as_device_str(arg)
        if d is not None:
            return d
    # Keyword device=...
    dev_kw = node.kwargs.get("device")
    if dev_kw is not None:
        return _as_device_str(dev_kw)
    return None


def _extract_to_dtype(node: "torch.fx.Node") -> Optional[str]:
    """Extract a dtype target from a ``.to(...)`` call node, if present.

    Handles ``x.to(torch.float16)``, ``x.to(dtype=torch.float16)`` and
    ``x.to(some_device, torch.float16)``.  Returns ``None`` when no static
    dtype argument is present (pure device move, or dtype taken from another
    traced tensor)."""
    import torch as _torch
    for arg in list(node.args[1:]) + list(node.kwargs.values()):
        if isinstance(arg, _torch.dtype):
            return str(arg).replace("torch.", "")
    return None


def _extract_method_params(
    node: "torch.fx.Node",
    method_name: str,
    op_kind: OpKind,
) -> Dict[str, Any]:
    """Extract shape-relevant params from a method call node."""
    params: Dict[str, Any] = {}

    def _set_literal(key: str, value: Any, dynamic_key: str) -> None:
        literal, dynamic = _literal_or_dynamic(value)
        params[key] = literal
        if dynamic:
            params[dynamic_key] = True

    if method_name in ("view", "reshape"):
        # ``x.view(2, 3)`` / ``x.reshape((2, 3))`` / ``x.reshape(b, -1)``.
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
            int_dims = tuple(d for d in dims if isinstance(d, int))
            if int_dims:
                params["target_shape"] = int_dims
    elif method_name in ("transpose", "swapaxes", "swapdims"):
        if len(node.args) >= 3:
            _set_literal("dim0", node.args[1], "__dim0_dynamic__")
            _set_literal("dim1", node.args[2], "__dim1_dynamic__")
        for key, dynamic_key in (
            ("dim0", "__dim0_dynamic__"),
            ("dim1", "__dim1_dynamic__"),
            ("axis0", "__dim0_dynamic__"),
            ("axis1", "__dim1_dynamic__"),
        ):
            if key in node.kwargs:
                target = "dim0" if key == "axis0" else "dim1" if key == "axis1" else key
                _set_literal(target, node.kwargs[key], dynamic_key)
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
    elif method_name == "repeat":
        dims = _parse_reshape_dims(tuple(node.args[1:]))
        if dims is not None:
            params["dims"] = dims
    elif method_name == "tile":
        reps = _parse_reshape_dims(tuple(node.args[1:]))
        if reps is not None:
            params["reps"] = reps
    elif method_name == "repeat_interleave":
        if len(node.args) > 1:
            _set_literal("repeats", node.args[1], "__repeats_dynamic__")
        elif "repeats" in node.kwargs:
            _set_literal(
                "repeats", node.kwargs["repeats"], "__repeats_dynamic__")
        if len(node.args) > 2:
            _set_literal("dim", node.args[2], "__dim_dynamic__")
        elif "dim" in node.kwargs:
            _set_literal("dim", node.kwargs["dim"], "__dim_dynamic__")
        if "output_size" in node.kwargs:
            _set_literal(
                "output_size",
                node.kwargs["output_size"],
                "__output_size_dynamic__",
            )
    elif method_name in ("squeeze", "unsqueeze"):
        if len(node.args) > 1:
            _set_literal("dim", node.args[1], "__dim_dynamic__")
        elif "dim" in node.kwargs:
            _set_literal("dim", node.kwargs["dim"], "__dim_dynamic__")
    elif method_name in ("movedim", "moveaxis"):
        if len(node.args) > 1:
            _set_literal("source", node.args[1], "__source_dynamic__")
        elif "source" in node.kwargs:
            _set_literal("source", node.kwargs["source"], "__source_dynamic__")
        if len(node.args) > 2:
            _set_literal(
                "destination", node.args[2], "__destination_dynamic__")
        elif "destination" in node.kwargs:
            _set_literal(
                "destination",
                node.kwargs["destination"],
                "__destination_dynamic__",
            )
    elif method_name == "roll":
        if len(node.args) > 1:
            _set_literal("shifts", node.args[1], "__shifts_dynamic__")
        elif "shifts" in node.kwargs:
            _set_literal("shifts", node.kwargs["shifts"], "__shifts_dynamic__")
        if len(node.args) > 2:
            params["__dims_present__"] = True
            _set_literal("dims", node.args[2], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            params["__dims_present__"] = True
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif method_name == "rot90":
        if len(node.args) > 1:
            _set_literal("k", node.args[1], "__k_dynamic__")
        elif "k" in node.kwargs:
            _set_literal("k", node.kwargs["k"], "__k_dynamic__")
        if len(node.args) > 2:
            _set_literal("dims", node.args[2], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif method_name == "flip":
        if len(node.args) > 1:
            _set_literal("dims", node.args[1], "__dims_dynamic__")
        elif "dims" in node.kwargs:
            _set_literal("dims", node.kwargs["dims"], "__dims_dynamic__")
    elif op_kind in (
        OpKind.GATHER, OpKind.INDEX_SELECT, OpKind.SCATTER,
        OpKind.MASKED_SELECT, OpKind.MASKED_FILL, OpKind.NARROW,
        OpKind.SELECT_DIM, OpKind.TAKE, OpKind.NONZERO,
        OpKind.BOOLEAN_INDEX, OpKind.TAKE_ALONG_DIM, OpKind.ARGSORT,
        OpKind.SORT, OpKind.TOPK, OpKind.KTHVALUE, OpKind.ARG_REDUCE,
    ):
        params.update(_extract_indexing_params(node, op_kind))
        if op_kind == OpKind.NONZERO and "as_tuple" in node.kwargs:
            params["as_tuple"] = node.kwargs["as_tuple"]
    elif op_kind == OpKind.DTYPE_CAST:
        # x.half()/x.float()/x.double()/x.bfloat16()/x.long()/... — the target
        # dtype is the method name itself (type_as has no static target).
        if method_name != "type_as":
            params["cast_dtype"] = method_name
    elif op_kind == OpKind.TO_DEVICE:
        # The single op-kind ``TO_DEVICE`` covers ``.to(...)``, ``.cuda()``,
        # ``.cpu()`` and ``.pin_memory()``.  A ``.to(...)`` call may change the
        # device, the dtype, or both; ``.cuda()/.cpu()`` change only the device;
        # ``.pin_memory()`` is device-preserving (returns a pinned CPU tensor).
        if method_name == "cuda":
            # x.cuda() / x.cuda(1) → cuda:0 (or cuda:<index> when constant).
            idx = node.args[1] if len(node.args) > 1 else None
            params["device"] = f"cuda:{idx}" if isinstance(idx, int) else "cuda:0"
        elif method_name == "cpu":
            params["device"] = "cpu"
        elif method_name == "pin_memory":
            # Device-preserving (stays on CPU); no device/dtype target.
            pass
        else:  # method_name == "to"
            dev = _extract_to_device(node)
            if dev is not None:
                params["device"] = dev
            dt = _extract_to_dtype(node)
            if dt is not None:
                params["cast_dtype"] = dt
    return params


def _maybe_functional_embedding(
    node: "torch.fx.Node",
    node_to_tensor: Dict[str, str],
    graph: ComputationGraph,
    step_idx: int,
) -> Optional[ComputationStep]:
    """Build a synthetic Embedding ``LAYER_CALL`` step for functional embedding.

    Handles ``F.embedding(input, weight, ...)`` and ``torch.embedding(weight,
    input, ...)``.  The output shape is ``input.shape + (embedding_dim,)`` where
    ``embedding_dim`` is the weight's last dimension; the weight constant's shape
    is taken from ``graph.const_shapes`` (recorded for every ``get_attr``
    tensor).  Returns ``None`` when the node is not an embedding or the weight
    shape is unavailable (sound abstention via the generic path).
    """
    if not HAS_TORCH:
        return None
    target = node.target
    name = getattr(target, "__name__", "")
    module = getattr(target, "__module__", "")
    is_emb = name == "embedding" and (
        "functional" in module or module.startswith("torch")
    )
    if not is_emb:
        return None

    node_args = [a for a in node.args if isinstance(a, torch.fx.Node)]
    if len(node_args) < 2:
        return None
    # Identify the weight (a 2-D constant) and the input (the indices).
    weight_node = None
    input_node = None
    for a in node_args[:2]:
        tname = node_to_tensor.get(a.name, a.name)
        cshape = graph.const_shapes.get(tname)
        if cshape is not None and cshape.ndim == 2:
            weight_node = (a, cshape)
        else:
            input_node = a
    if weight_node is None or input_node is None:
        return None
    _, wshape = weight_node
    emb_dim = wshape.dims[-1]
    num_emb = wshape.dims[0]
    if not isinstance(emb_dim.value, int):
        return None

    layer_name = f"_func_embedding_{step_idx}"
    ldef = LayerDef(
        attr_name=layer_name,
        kind=LayerKind.EMBEDDING,
        params={},
    )
    ldef.embedding_dim = emb_dim.value
    if isinstance(num_emb.value, int):
        ldef.num_embeddings = num_emb.value
    graph.layers[layer_name] = ldef

    return ComputationStep(
        op=OpKind.LAYER_CALL,
        inputs=[node_to_tensor.get(input_node.name, input_node.name)],
        output=f"_t{step_idx}",
        layer_ref=layer_name,
        params={"embedding_dim": emb_dim.value},
    )


def _fx_pair(raw: Any) -> Any:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return (raw, raw)
    if (isinstance(raw, (tuple, list)) and len(raw) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in raw)):
        return tuple(int(v) for v in raw)
    return raw


def _maybe_functional_fold_unfold(
    node: "torch.fx.Node",
    node_to_tensor: Dict[str, str],
    graph: ComputationGraph,
    step_idx: int,
) -> Optional[ComputationStep]:
    """Build synthetic nn.Fold/nn.Unfold layer calls for F.fold/F.unfold."""
    if not HAS_TORCH:
        return None
    target = node.target
    name = getattr(target, "__name__", "")
    module = getattr(target, "__module__", "")
    if name not in ("fold", "unfold") or "functional" not in module:
        return None
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return None

    def arg(pos: int, key: str, default: Any = None) -> Any:
        if len(node.args) > pos:
            return node.args[pos]
        return node.kwargs.get(key, default)

    layer_name = f"_func_{name}_{step_idx}"
    if name == "unfold":
        kernel_size = _fx_pair(arg(1, "kernel_size"))
        dilation = _fx_pair(arg(2, "dilation", 1))
        padding = _fx_pair(arg(3, "padding", 0))
        stride = _fx_pair(arg(4, "stride", 1))
        ldef = LayerDef(
            attr_name=layer_name,
            kind=LayerKind.UNFOLD,
            params={
                "kernel_size": kernel_size,
                "dilation": dilation,
                "padding": padding,
                "stride": stride,
            },
        )
        ldef.kernel_size = kernel_size
    else:
        output_size = _fx_pair(arg(1, "output_size"))
        kernel_size = _fx_pair(arg(2, "kernel_size"))
        dilation = _fx_pair(arg(3, "dilation", 1))
        padding = _fx_pair(arg(4, "padding", 0))
        stride = _fx_pair(arg(5, "stride", 1))
        ldef = LayerDef(
            attr_name=layer_name,
            kind=LayerKind.FOLD,
            params={
                "output_size": output_size,
                "kernel_size": kernel_size,
                "dilation": dilation,
                "padding": padding,
                "stride": stride,
            },
        )
        ldef.output_size = output_size
        ldef.kernel_size = kernel_size

    graph.layers[layer_name] = ldef
    input_node = node.args[0]
    return ComputationStep(
        op=OpKind.LAYER_CALL,
        inputs=[node_to_tensor.get(input_node.name, input_node.name)],
        output=f"_t{step_idx}",
        layer_ref=layer_name,
        params=dict(ldef.params),
    )


def _handle_reduction_method(
    node: "torch.fx.Node",
    input_names: List[str],
    output_name: str,
    step_idx: int,
    graph: ComputationGraph,
) -> List[ComputationStep]:
    """Handle mean/sum reductions.

    The global-average-pooling idiom ``mean([2, 3])`` on a 4D tensor is mapped
    to AdaptiveAvgPool2d(1,1) + flatten.  Every other reduction is emitted as a
    real ``MEAN_REDUCE`` / ``SUM_REDUCE`` step carrying its ``dim``/``keepdim``
    so the shape propagator computes the correct reduced shape instead of
    (unsoundly) treating the op as shape-preserving.
    """
    method_name = str(node.target)

    # Extract reduction dims from args OR kwargs (fx may place either way).
    reduce_dims = None
    dim_arg = None
    if len(node.args) > 1:
        dim_arg = node.args[1]
    elif "dim" in node.kwargs:
        dim_arg = node.kwargs["dim"]
    if isinstance(dim_arg, (list, tuple)):
        reduce_dims = list(dim_arg)
    elif isinstance(dim_arg, int) and not isinstance(dim_arg, bool):
        reduce_dims = [dim_arg]

    keepdim = False
    if len(node.args) > 2 and isinstance(node.args[2], bool):
        keepdim = node.args[2]
    elif "keepdim" in node.kwargs:
        keepdim = bool(node.kwargs["keepdim"])

    # Pattern: mean([2, 3]) on 4D tensor = global average pooling
    if reduce_dims and set(reduce_dims) == {2, 3} and not keepdim:
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

    op_kind = (OpKind.MEAN_REDUCE if method_name == "mean"
               else OpKind.SUM_REDUCE)
    params: Dict[str, Any] = {"keepdim": keepdim}
    if reduce_dims is not None:
        # The propagator handles a single int dim or a list of dims; pass a
        # bare int when there is exactly one for maximum precision.
        params["dim"] = reduce_dims[0] if len(reduce_dims) == 1 else reduce_dims
    # dim omitted entirely → full reduction to a scalar (handled downstream).
    return [ComputationStep(
        op=op_kind,
        inputs=input_names[:1],
        output=output_name,
        params=params,
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
    check_dtypes: bool = True,
    input_dtypes: Optional[Dict[str, str]] = None,
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
                    check_dtypes=check_dtypes,
                    input_dtypes=input_dtypes,
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
                        check_dtypes=check_dtypes,
                        input_dtypes=input_dtypes,
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
        check_dtypes=check_dtypes,
        input_dtypes=input_dtypes,
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
    "movedim": OpKind.MOVEDIM,
    "moveaxis": OpKind.MOVEDIM,
    "swapaxes": OpKind.TRANSPOSE,
    "swapdims": OpKind.TRANSPOSE,
    "roll": OpKind.ROLL,
    "rot90": OpKind.ROT90,
    "flip": OpKind.FLIP,
    "mean": OpKind.MEAN_REDUCE,
    "sum": OpKind.SUM_REDUCE,
    "contiguous": OpKind.CONTIGUOUS,
    "detach": OpKind.DETACH,
    "expand": OpKind.EXPAND,
    "expand_as": OpKind.EXPAND,
    "broadcast_to": OpKind.EXPAND,
    "repeat": OpKind.REPEAT,
    "repeat_interleave": OpKind.REPEAT_INTERLEAVE,
    "tile": OpKind.TILE,
    "gather": OpKind.GATHER,
    "index_select": OpKind.INDEX_SELECT,
    "scatter": OpKind.SCATTER,
    "scatter_": OpKind.SCATTER,
    "scatter_add": OpKind.SCATTER,
    "scatter_add_": OpKind.SCATTER,
    "masked_select": OpKind.MASKED_SELECT,
    "masked_fill": OpKind.MASKED_FILL,
    "masked_fill_": OpKind.MASKED_FILL,
    "nonzero": OpKind.NONZERO,
    "narrow": OpKind.NARROW,
    "select": OpKind.SELECT_DIM,
    "take": OpKind.TAKE,
    "take_along_dim": OpKind.TAKE_ALONG_DIM,
    "argsort": OpKind.ARGSORT,
    "sort": OpKind.SORT,
    "topk": OpKind.TOPK,
    "kthvalue": OpKind.KTHVALUE,
    "argmax": OpKind.ARG_REDUCE,
    "argmin": OpKind.ARG_REDUCE,
    "half": OpKind.DTYPE_CAST,
    "float": OpKind.DTYPE_CAST,
    "double": OpKind.DTYPE_CAST,
    "bfloat16": OpKind.DTYPE_CAST,
    "long": OpKind.DTYPE_CAST,
    "int": OpKind.DTYPE_CAST,
    "short": OpKind.DTYPE_CAST,
    "bool": OpKind.DTYPE_CAST,
    "type_as": OpKind.DTYPE_CAST,
    "to": OpKind.TO_DEVICE,
    "cuda": OpKind.TO_DEVICE,
    "cpu": OpKind.TO_DEVICE,
    "pin_memory": OpKind.TO_DEVICE,
    # Arithmetic methods (broadcast semantics) — modelled precisely rather than
    # left to the shape-preserving default, which is only correct when the
    # other operand does not broaden the result.
    "add": OpKind.ADD,
    "add_": OpKind.ADD,
    "sub": OpKind.ADD,
    "sub_": OpKind.ADD,
    "subtract": OpKind.ADD,
    "mul": OpKind.MULTIPLY,
    "mul_": OpKind.MULTIPLY,
    "multiply": OpKind.MULTIPLY,
    "div": OpKind.MULTIPLY,
    "div_": OpKind.MULTIPLY,
    "divide": OpKind.MULTIPLY,
    "true_divide": OpKind.MULTIPLY,
    "matmul": OpKind.MATMUL,
    "bmm": OpKind.MATMUL,
    "mm": OpKind.MATMUL,
    "softmax": OpKind.SOFTMAX,
    "log_softmax": OpKind.SOFTMAX,
    "type": OpKind.DTYPE_CAST,
    "chunk": OpKind.CHUNK,
    "split": OpKind.SPLIT,
    "repeat": OpKind.REPEAT,
    "t": OpKind.TRANSPOSE,
}

# torch.xxx(...) → OpKind
_TORCH_FUNC_MAP: Dict[str, OpKind] = {
    "cat": OpKind.CAT,
    "stack": OpKind.STACK,
    "hstack": OpKind.STACK,
    "vstack": OpKind.STACK,
    "dstack": OpKind.STACK,
    "column_stack": OpKind.STACK,
    "row_stack": OpKind.STACK,
    "squeeze": OpKind.SQUEEZE,
    "unsqueeze": OpKind.UNSQUEEZE,
    "movedim": OpKind.MOVEDIM,
    "moveaxis": OpKind.MOVEDIM,
    "swapaxes": OpKind.TRANSPOSE,
    "swapdims": OpKind.TRANSPOSE,
    "roll": OpKind.ROLL,
    "rot90": OpKind.ROT90,
    "flip": OpKind.FLIP,
    "repeat_interleave": OpKind.REPEAT_INTERLEAVE,
    "tile": OpKind.TILE,
    "broadcast_tensors": OpKind.BROADCAST_TENSORS,
    "where": OpKind.WHERE,
    "masked_select": OpKind.MASKED_SELECT,
    "nonzero": OpKind.NONZERO,
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
    "interpolate": OpKind.INTERPOLATE,
    "upsample": OpKind.INTERPOLATE,
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
