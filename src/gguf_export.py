"""GGUF / llama.cpp export shape gates for transformer checkpoints.

The checks in this module validate a checkpoint's logical tensor shapes and
metadata before a GGUF/llama.cpp converter writes an artifact.  They are pure:
real ``torch.Tensor`` state_dict entries are accepted, but no model code or
external converter is executed by the verifier.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Logical tensor descriptor used when a real tensor is unavailable."""

    shape: Tuple[int, ...]
    dtype: Optional[str] = None
    quant_type: Optional[str] = None
    block_size: Optional[int] = None


@dataclass(frozen=True)
class GGUFExportIssue:
    """One actionable GGUF/llama.cpp export contract issue."""

    category: str
    message: str
    tensor_name: Optional[str] = None
    metadata_key: Optional[str] = None
    expected_shape: Optional[Tuple[int, ...]] = None
    actual_shape: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class GGUFExportGateResult:
    """Result of TensorGuard's GGUF/llama.cpp export gate."""

    ok: bool
    issues: Tuple[GGUFExportIssue, ...]
    checked_tensors: Tuple[str, ...] = ()
    architecture: str = "llama"
    layer_count: int = 0


class TensorGuardGGUFExportError(ValueError):
    """Raised when a GGUF/llama.cpp export contract is rejected."""

    def __init__(self, issues: Sequence[GGUFExportIssue]):
        self.issues = tuple(issues)
        details = "; ".join(issue.message for issue in self.issues[:3])
        more = "" if len(self.issues) <= 3 else f" (+{len(self.issues) - 3} more)"
        super().__init__(
            f"TensorGuard rejected GGUF export with "
            f"{len(self.issues)} issue(s): {details}{more}"
        )


_QUANT_BLOCK_SIZES = {
    "Q4_0": 32,
    "Q4_1": 32,
    "Q5_0": 32,
    "Q5_1": 32,
    "Q8_0": 32,
    "Q8_1": 32,
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "Q8_K": 256,
    "IQ1_S": 256,
    "IQ1_M": 256,
    "IQ2_XXS": 256,
    "IQ2_XS": 256,
    "IQ2_S": 256,
    "IQ2_M": 256,
    "IQ3_XXS": 256,
    "IQ3_S": 256,
    "IQ4_NL": 32,
    "IQ4_XS": 256,
}

_FLOAT_TYPES = {"F32", "F16", "BF16", "FLOAT32", "FLOAT16", "BFLOAT16"}


def verify_gguf_export_contract(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    architecture: str = "llama",
) -> GGUFExportGateResult:
    """Validate transformer checkpoint shapes before GGUF/llama.cpp export.

    The gate supports HF-style LLaMA state_dict keys, Meta/LLaMA keys, and common
    packed-qkv Linear weights.  It checks metadata consistency, q/k/v packing,
    rotary dimensions, vocab projection shapes, quant block divisibility, and
    layer-count coherence without invoking an external converter.
    """

    normalized_meta = _normalize_metadata(metadata, architecture)
    issues: List[GGUFExportIssue] = []
    checked: List[str] = []

    hidden = _meta_int(normalized_meta, ("embedding_length", "n_embd", "hidden_size"), "embedding_length", issues)
    n_layers = _meta_int(normalized_meta, ("block_count", "num_hidden_layers", "n_layer"), "block_count", issues)
    n_heads = _meta_int(normalized_meta, ("attention.head_count", "num_attention_heads", "n_head"), "attention.head_count", issues)
    n_kv_heads = _meta_int(
        normalized_meta,
        ("attention.head_count_kv", "num_key_value_heads", "n_head_kv"),
        "attention.head_count_kv",
        issues,
        required=False,
    )
    if n_kv_heads is None and n_heads is not None:
        n_kv_heads = n_heads
    vocab_size = _metadata_vocab_size(normalized_meta, issues)
    ffn_dim = _meta_int(
        normalized_meta,
        ("feed_forward_length", "intermediate_size", "n_ff"),
        "feed_forward_length",
        issues,
        required=False,
    )
    explicit_head_dim = _meta_int(
        normalized_meta,
        ("attention.head_dim", "head_dim"),
        "attention.head_dim",
        issues,
        required=False,
    )

    head_dim: Optional[int] = None
    if hidden is not None and n_heads is not None:
        if explicit_head_dim is not None:
            head_dim = explicit_head_dim
        elif n_heads > 0 and hidden % n_heads == 0:
            head_dim = hidden // n_heads
        else:
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key="attention.head_count",
                    message=(
                        f"embedding length {hidden} is not divisible by "
                        f"attention head count {n_heads}"
                    ),
                )
            )
        if n_heads <= 0:
            issues.append(_positive_issue("attention.head_count", n_heads))
    if n_kv_heads is not None and n_heads is not None:
        if n_kv_heads <= 0:
            issues.append(_positive_issue("attention.head_count_kv", n_kv_heads))
        elif n_heads > 0 and n_heads % n_kv_heads != 0:
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key="attention.head_count_kv",
                    message=(
                        f"attention head count {n_heads} must be divisible by "
                        f"kv head count {n_kv_heads} for llama.cpp grouped-query export"
                    ),
                )
            )

    if head_dim is not None:
        _check_rotary_dim(normalized_meta, head_dim, issues)

    if hidden is not None and vocab_size is not None:
        checked.extend(_check_vocab_tensors(tensors, normalized_meta, hidden, vocab_size, issues))

    layer_indices = _layer_indices(tensors)
    if n_layers is not None:
        if n_layers <= 0:
            issues.append(_positive_issue("block_count", n_layers))
        if len(layer_indices) != n_layers:
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key="block_count",
                    message=(
                        f"metadata declares {n_layers} transformer block(s), "
                        f"but checkpoint exposes {len(layer_indices)} layer index(es)"
                    ),
                )
            )
        out_of_range = sorted(index for index in layer_indices if index < 0 or index >= n_layers)
        for index in out_of_range:
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key="block_count",
                    message=f"checkpoint contains layer {index}, outside metadata block_count={n_layers}",
                )
            )

    if (
        hidden is not None
        and n_heads is not None
        and n_kv_heads is not None
        and head_dim is not None
    ):
        for index in sorted(layer_indices):
            checked.extend(
                _check_attention_layer(
                    tensors,
                    index,
                    hidden=hidden,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    issues=issues,
                )
            )
            if ffn_dim is not None:
                checked.extend(_check_ffn_layer(tensors, index, hidden, ffn_dim, issues))

    checked.extend(_check_quant_blocks(tensors, issues))

    return GGUFExportGateResult(
        ok=not issues,
        issues=tuple(issues),
        checked_tensors=tuple(dict.fromkeys(checked)),
        architecture=architecture,
        layer_count=len(layer_indices),
    )


def guarded_gguf_export(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    exporter: Callable[..., Any],
    *export_args: Any,
    architecture: str = "llama",
    on_violation: str = "raise",
    **export_kwargs: Any,
) -> Any:
    """Run the GGUF contract before invoking a user-supplied exporter."""

    result = verify_gguf_export_contract(
        tensors,
        metadata,
        architecture=architecture,
    )
    _handle_gguf_gate_result(result, on_violation)
    return exporter(*export_args, **export_kwargs)


def _handle_gguf_gate_result(
    result: GGUFExportGateResult,
    on_violation: str,
) -> None:
    if on_violation not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_violation must be raise/warn/ignore, got {on_violation!r}")
    if result.ok or on_violation == "ignore":
        return
    error = TensorGuardGGUFExportError(result.issues)
    if on_violation == "raise":
        raise error
    warnings.warn(str(error), stacklevel=2)


def _normalize_metadata(metadata: Mapping[str, Any], architecture: str) -> dict:
    prefix = f"{architecture}."
    normalized = {}
    for key, value in metadata.items():
        text_key = str(key)
        if text_key.startswith(prefix):
            normalized[text_key[len(prefix):]] = value
        normalized[text_key] = value
    return normalized


def _meta_int(
    metadata: Mapping[str, Any],
    keys: Iterable[str],
    canonical_key: str,
    issues: List[GGUFExportIssue],
    *,
    required: bool = True,
) -> Optional[int]:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key=key,
                    message=f"metadata {key!r} must be an integer, got {value!r}",
                )
            )
            return None
        if parsed <= 0:
            issues.append(_positive_issue(key, parsed))
        return parsed
    if required:
        issues.append(
            GGUFExportIssue(
                category="metadata_consistency",
                metadata_key=canonical_key,
                message=f"missing required GGUF metadata key {canonical_key!r}",
            )
        )
    return None


def _positive_issue(key: str, value: Any) -> GGUFExportIssue:
    return GGUFExportIssue(
        category="metadata_consistency",
        metadata_key=key,
        message=f"metadata {key!r} must be positive, got {value!r}",
    )


def _metadata_vocab_size(
    metadata: Mapping[str, Any],
    issues: List[GGUFExportIssue],
) -> Optional[int]:
    for key in ("vocab_size", "tokenizer.ggml.token_count", "tokenizer.ggml.n_vocab"):
        if key in metadata:
            try:
                value = int(metadata[key])
            except (TypeError, ValueError):
                issues.append(
                    GGUFExportIssue(
                        category="metadata_consistency",
                        metadata_key=key,
                        message=f"metadata {key!r} must be an integer, got {metadata[key]!r}",
                    )
                )
                return None
            if value <= 0:
                issues.append(_positive_issue(key, value))
            return value
    if "tokenizer.ggml.tokens" in metadata:
        tokens = metadata["tokenizer.ggml.tokens"]
        try:
            value = len(tokens)
        except TypeError:
            issues.append(
                GGUFExportIssue(
                    category="metadata_consistency",
                    metadata_key="tokenizer.ggml.tokens",
                    message="metadata 'tokenizer.ggml.tokens' must be a sized token sequence",
                )
            )
            return None
        if value <= 0:
            issues.append(_positive_issue("tokenizer.ggml.tokens", value))
        return int(value)
    issues.append(
        GGUFExportIssue(
            category="metadata_consistency",
            metadata_key="vocab_size",
            message="missing required GGUF vocab metadata ('vocab_size' or tokenizer.ggml.tokens)",
        )
    )
    return None


def _tensor_info(value: Any) -> Optional[GGUFTensorInfo]:
    if isinstance(value, GGUFTensorInfo):
        return value
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    quant_type = getattr(value, "quant_type", None)
    block_size = getattr(value, "block_size", None)
    if shape is not None:
        try:
            return GGUFTensorInfo(
                shape=tuple(int(dim) for dim in shape),
                dtype=str(dtype) if dtype is not None else None,
                quant_type=str(quant_type) if quant_type is not None else None,
                block_size=int(block_size) if block_size is not None else None,
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        shape = value.get("shape")
        if shape is None:
            return None
        try:
            return GGUFTensorInfo(
                shape=tuple(int(dim) for dim in shape),
                dtype=str(value["dtype"]) if value.get("dtype") is not None else None,
                quant_type=(
                    str(value["quant_type"])
                    if value.get("quant_type") is not None
                    else None
                ),
                block_size=(
                    int(value["block_size"])
                    if value.get("block_size") is not None
                    else None
                ),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, (tuple, list)):
        try:
            return GGUFTensorInfo(shape=tuple(int(dim) for dim in value))
        except (TypeError, ValueError):
            return None
    return None


def _shape_of(tensors: Mapping[str, Any], name: Optional[str]) -> Optional[Tuple[int, ...]]:
    if name is None:
        return None
    info = _tensor_info(tensors[name])
    if info is None:
        return None
    return info.shape


def _find_tensor(tensors: Mapping[str, Any], templates: Sequence[str], layer: Optional[int] = None) -> Optional[str]:
    for template in templates:
        name = template.format(i=layer) if layer is not None else template
        if name in tensors:
            return name
    return None


def _layer_indices(tensors: Mapping[str, Any]) -> set:
    import re

    indices = set()
    patterns = (
        re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\."),
        re.compile(r"^blk\.(\d+)\."),
    )
    for name in tensors:
        for pattern in patterns:
            match = pattern.search(name)
            if match:
                indices.add(int(match.group(1)))
                break
    return indices


def _check_shape(
    tensors: Mapping[str, Any],
    name: Optional[str],
    expected: Tuple[int, ...],
    issues: List[GGUFExportIssue],
    *,
    category: str,
    description: str,
    allow_rows_at_least: bool = False,
) -> bool:
    if name is None:
        return False
    actual = _shape_of(tensors, name)
    if actual is None:
        issues.append(
            GGUFExportIssue(
                category=category,
                tensor_name=name,
                expected_shape=expected,
                message=f"{description} {name!r} has an unreadable shape descriptor",
            )
        )
        return True
    matched = actual == expected
    if allow_rows_at_least and len(actual) == len(expected) == 2:
        matched = actual[0] >= expected[0] and actual[1] == expected[1]
    if not matched:
        comparator = ">=" if allow_rows_at_least else "=="
        issues.append(
            GGUFExportIssue(
                category=category,
                tensor_name=name,
                expected_shape=expected,
                actual_shape=actual,
                message=(
                    f"{description} {name!r} shape {actual} must be "
                    f"{comparator} {expected} for llama.cpp GGUF export"
                ),
            )
        )
    return True


def _check_vocab_tensors(
    tensors: Mapping[str, Any],
    metadata: Mapping[str, Any],
    hidden: int,
    vocab_size: int,
    issues: List[GGUFExportIssue],
) -> List[str]:
    checked: List[str] = []
    embed = _find_tensor(
        tensors,
        (
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            "token_embd.weight",
            "token_embd",
        ),
    )
    if embed is None:
        issues.append(
            GGUFExportIssue(
                category="vocab_projection",
                tensor_name="model.embed_tokens.weight",
                expected_shape=(vocab_size, hidden),
                message="missing token embedding weight required by llama.cpp GGUF export",
            )
        )
    elif _check_shape(
        tensors,
        embed,
        (vocab_size, hidden),
        issues,
        category="vocab_projection",
        description="token embedding",
        allow_rows_at_least=True,
    ):
        checked.append(embed)

    output = _find_tensor(tensors, ("lm_head.weight", "output.weight", "output"))
    if output is not None:
        if _check_shape(
            tensors,
            output,
            (vocab_size, hidden),
            issues,
            category="vocab_projection",
            description="vocab projection",
            allow_rows_at_least=True,
        ):
            checked.append(output)
    elif not _truthy(metadata.get("tie_word_embeddings", metadata.get("tie_output", True))):
        issues.append(
            GGUFExportIssue(
                category="vocab_projection",
                tensor_name="lm_head.weight",
                expected_shape=(vocab_size, hidden),
                message=(
                    "missing output/lm_head weight while metadata says token "
                    "embeddings are not tied"
                ),
            )
        )
    return checked


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _check_rotary_dim(
    metadata: Mapping[str, Any],
    head_dim: int,
    issues: List[GGUFExportIssue],
) -> None:
    rope = _meta_int(
        metadata,
        ("rope.dimension_count", "rope_dimension_count", "rotary_dim", "n_rot"),
        "rope.dimension_count",
        issues,
        required=False,
    )
    if rope is None:
        return
    if rope % 2 != 0:
        issues.append(
            GGUFExportIssue(
                category="rotary_dims",
                metadata_key="rope.dimension_count",
                message=f"rotary dimension count {rope} must be even for rotary pairs",
            )
        )
    if rope > head_dim:
        issues.append(
            GGUFExportIssue(
                category="rotary_dims",
                metadata_key="rope.dimension_count",
                message=(
                    f"rotary dimension count {rope} exceeds attention head_dim "
                    f"{head_dim}"
                ),
            )
        )


def _check_attention_layer(
    tensors: Mapping[str, Any],
    index: int,
    *,
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    issues: List[GGUFExportIssue],
) -> List[str]:
    checked: List[str] = []
    q_out = n_heads * head_dim
    kv_out = n_kv_heads * head_dim
    q_name = _find_tensor(tensors, _q_templates(), index)
    k_name = _find_tensor(tensors, _k_templates(), index)
    v_name = _find_tensor(tensors, _v_templates(), index)
    packed_name = _find_tensor(tensors, _packed_qkv_templates(), index)

    if q_name and k_name and v_name:
        for name, expected, desc in (
            (q_name, (q_out, hidden), "query projection"),
            (k_name, (kv_out, hidden), "key projection"),
            (v_name, (kv_out, hidden), "value projection"),
        ):
            if _check_shape(tensors, name, expected, issues, category="qkv_projection", description=desc):
                checked.append(name)
    elif packed_name is not None:
        expected = (q_out + kv_out + kv_out, hidden)
        actual = _shape_of(tensors, packed_name)
        is_c_attn = packed_name.endswith(".c_attn.weight") or ".c_attn." in packed_name
        if is_c_attn:
            expected = (hidden, q_out + kv_out + kv_out)
        if actual != expected:
            issues.append(
                GGUFExportIssue(
                    category="qkv_projection",
                    tensor_name=packed_name,
                    expected_shape=expected,
                    actual_shape=actual,
                    message=(
                        f"packed q/k/v projection {packed_name!r} shape {actual} "
                        f"must equal {expected} for llama.cpp export"
                    ),
                )
            )
        checked.append(packed_name)
    else:
        issues.append(
            GGUFExportIssue(
                category="qkv_projection",
                tensor_name=f"layer {index} q/k/v",
                expected_shape=(q_out + kv_out + kv_out, hidden),
                message=(
                    f"layer {index} exposes neither separate q/k/v projection "
                    "weights nor a recognized packed qkv weight"
                ),
            )
        )

    o_name = _find_tensor(tensors, _o_templates(), index)
    if o_name is None:
        issues.append(
            GGUFExportIssue(
                category="qkv_projection",
                tensor_name=f"layer {index} output projection",
                expected_shape=(hidden, q_out),
                message=f"layer {index} is missing attention output projection weight",
            )
        )
    elif _check_shape(
        tensors,
        o_name,
        (hidden, q_out),
        issues,
        category="qkv_projection",
        description="attention output projection",
    ):
        checked.append(o_name)

    return checked


def _check_ffn_layer(
    tensors: Mapping[str, Any],
    index: int,
    hidden: int,
    ffn_dim: int,
    issues: List[GGUFExportIssue],
) -> List[str]:
    checked: List[str] = []
    specs = (
        (
            _find_tensor(tensors, _ffn_gate_templates(), index),
            (ffn_dim, hidden),
            "feed-forward gate projection",
        ),
        (
            _find_tensor(tensors, _ffn_up_templates(), index),
            (ffn_dim, hidden),
            "feed-forward up projection",
        ),
        (
            _find_tensor(tensors, _ffn_down_templates(), index),
            (hidden, ffn_dim),
            "feed-forward down projection",
        ),
    )
    for name, expected, desc in specs:
        if name is not None and _check_shape(
            tensors,
            name,
            expected,
            issues,
            category="metadata_consistency",
            description=desc,
        ):
            checked.append(name)
    return checked


def _check_quant_blocks(
    tensors: Mapping[str, Any],
    issues: List[GGUFExportIssue],
) -> List[str]:
    checked: List[str] = []
    for name, value in tensors.items():
        info = _tensor_info(value)
        if info is None or len(info.shape) < 2:
            continue
        raw_quant_type = info.quant_type
        quant_type = _normalize_quant_type(raw_quant_type or info.dtype)
        explicit_block = info.block_size
        if quant_type is None:
            continue
        if quant_type in _FLOAT_TYPES:
            continue
        expected_block = _QUANT_BLOCK_SIZES.get(quant_type)
        if expected_block is None and raw_quant_type is None:
            continue
        if expected_block is None:
            issues.append(
                GGUFExportIssue(
                    category="quant_block_size",
                    tensor_name=name,
                    message=f"unknown GGUF quantization type {quant_type!r} for {name!r}",
                )
            )
            checked.append(name)
            continue
        if explicit_block is not None and explicit_block != expected_block:
            issues.append(
                GGUFExportIssue(
                    category="quant_block_size",
                    tensor_name=name,
                    message=(
                        f"{name!r} declares block_size={explicit_block} for "
                        f"{quant_type}, but llama.cpp expects {expected_block}"
                    ),
                )
            )
        if info.shape[-1] % expected_block != 0:
            issues.append(
                GGUFExportIssue(
                    category="quant_block_size",
                    tensor_name=name,
                    actual_shape=info.shape,
                    message=(
                        f"quantized tensor {name!r} has last logical dimension "
                        f"{info.shape[-1]}, not divisible by {quant_type} block "
                        f"size {expected_block}"
                    ),
                )
            )
        checked.append(name)
    return checked


def _normalize_quant_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if "." in text:
        text = text.split(".")[-1]
    return text.upper()


def _q_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.self_attn.q_proj.weight",
        "layers.{i}.attention.wq.weight",
        "blk.{i}.attn_q.weight",
    )


def _k_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.self_attn.k_proj.weight",
        "layers.{i}.attention.wk.weight",
        "blk.{i}.attn_k.weight",
    )


def _v_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.self_attn.v_proj.weight",
        "layers.{i}.attention.wv.weight",
        "blk.{i}.attn_v.weight",
    )


def _o_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.self_attn.o_proj.weight",
        "layers.{i}.attention.wo.weight",
        "blk.{i}.attn_output.weight",
    )


def _packed_qkv_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.self_attn.qkv_proj.weight",
        "layers.{i}.attention.wqkv.weight",
        "transformer.h.{i}.attn.c_attn.weight",
        "h.{i}.attn.c_attn.weight",
        "blk.{i}.attn_qkv.weight",
    )


def _ffn_gate_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.mlp.gate_proj.weight",
        "layers.{i}.feed_forward.w1.weight",
        "blk.{i}.ffn_gate.weight",
    )


def _ffn_up_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.mlp.up_proj.weight",
        "layers.{i}.feed_forward.w3.weight",
        "blk.{i}.ffn_up.weight",
    )


def _ffn_down_templates() -> Tuple[str, ...]:
    return (
        "model.layers.{i}.mlp.down_proj.weight",
        "layers.{i}.feed_forward.w2.weight",
        "blk.{i}.ffn_down.weight",
    )


__all__ = [
    "GGUFExportGateResult",
    "GGUFExportIssue",
    "GGUFTensorInfo",
    "TensorGuardGGUFExportError",
    "guarded_gguf_export",
    "verify_gguf_export_contract",
]
