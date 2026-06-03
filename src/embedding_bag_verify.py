"""Static contracts for PyTorch and TorchRec embedding-bag lookups.

The rules model the runtime checks that depend only on tensor metadata:
input/offset/per-sample-weight ranks and dtypes, pooling-mode restrictions,
``include_last_offset`` bag counts, static offset values when available, and
embedding-table index bounds.  Jagged partitions whose boundaries are not
statically known are reported as UNKNOWN metadata rather than refuted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

Dim = Any
Shape = Tuple[Dim, ...]

_INT_INDEX_DTYPES = frozenset({"int32", "int64", "torch.int32", "torch.int64", "torch.int", "torch.long", "int", "long"})
_POOLING_MODES = frozenset({"sum", "mean", "max"})


@dataclass(frozen=True)
class EmbeddingBagVerdict:
    """Result of checking one embedding-bag contract."""

    ok: bool
    output_shape: Optional[Shape] = None
    output_dtype: Optional[str] = None
    error_kind: Optional[str] = None
    message: Optional[str] = None
    unknown_reason: Optional[str] = None


@dataclass(frozen=True)
class TorchRecJaggedSpec:
    """Static facts for a TorchRec-style jagged feature."""

    values_shape: Shape
    offsets_shape: Shape
    values_dtype: Optional[str] = None
    offsets_dtype: Optional[str] = None
    values_range: Optional[Tuple[int, int]] = None
    offsets_values: Optional[Tuple[int, ...]] = None


def _canon_dtype(dtype: Optional[Any]) -> Optional[str]:
    if dtype is None:
        return None
    text = str(dtype).strip().lower().replace("torch.", "")
    aliases = {"long": "int64", "int": "int32", "float": "float32", "double": "float64"}
    return aliases.get(text, text)


def _shape(shape: Optional[Sequence[Dim]]) -> Optional[Shape]:
    if shape is None:
        return None
    return tuple(shape)


def _is_known_int(value: object) -> bool:
    return type(value) is int


def _same_shape(left: Shape, right: Shape) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if _is_known_int(a) and _is_known_int(b) and a != b:
            return False
    return True


def _fail(kind: str, message: str) -> EmbeddingBagVerdict:
    return EmbeddingBagVerdict(False, error_kind=kind, message=message)


def _bag_count(offset_len: Dim, include_last_offset: bool) -> Tuple[Optional[Dim], Optional[str]]:
    if not include_last_offset:
        return offset_len, None
    if _is_known_int(offset_len):
        if offset_len < 1:
            return None, "include_last_offset requires offsets to contain at least one value"
        return offset_len - 1, None
    return f"({offset_len}-1)", None


def _check_int_dtype(dtype: Optional[Any], label: str) -> Optional[EmbeddingBagVerdict]:
    canon = _canon_dtype(dtype)
    if canon is not None and canon not in {"int32", "int64"}:
        return _fail("dtype", f"{label} must have dtype int32 or int64, got {canon}")
    return None


def _check_static_offsets(
    offsets_values: Optional[Sequence[int]],
    input_len: Dim,
    include_last_offset: bool,
) -> Optional[EmbeddingBagVerdict]:
    if offsets_values is None:
        return None
    vals = tuple(int(v) for v in offsets_values)
    if include_last_offset and len(vals) < 1:
        return _fail(
            "offsets",
            "include_last_offset requires offsets to contain at least one value",
        )
    if len(vals) == 0:
        return None
    if vals[0] != 0:
        return _fail("offsets", f"offsets[0] must be 0, got {vals[0]}")
    if any(v < 0 for v in vals):
        return _fail("offsets", f"offsets must be non-negative, got {vals}")
    for prev, cur in zip(vals, vals[1:]):
        if cur < prev:
            return _fail("offsets", f"offsets must be nondecreasing, got {vals}")
    if _is_known_int(input_len) and vals[-1] > input_len:
        return _fail(
            "offsets",
            f"offsets[-1] cannot exceed input length {input_len}, got {vals[-1]}",
        )
    if include_last_offset and _is_known_int(input_len) and vals[-1] != input_len:
        return _fail(
            "offsets",
            f"include_last_offset requires offsets[-1] == input length {input_len}, got {vals[-1]}",
        )
    return None


def verify_embedding_bag(
    input_shape: Sequence[Dim],
    *,
    embedding_dim: Optional[Dim] = None,
    num_embeddings: Optional[Dim] = None,
    weight_shape: Optional[Sequence[Dim]] = None,
    offsets_shape: Optional[Sequence[Dim]] = None,
    per_sample_weights_shape: Optional[Sequence[Dim]] = None,
    input_dtype: Optional[Any] = None,
    offsets_dtype: Optional[Any] = None,
    weight_dtype: Optional[Any] = None,
    per_sample_weights_dtype: Optional[Any] = None,
    input_value_range: Optional[Tuple[int, int]] = None,
    offsets_value_range: Optional[Tuple[int, int]] = None,
    offsets_values: Optional[Sequence[int]] = None,
    mode: str = "mean",
    include_last_offset: bool = False,
) -> EmbeddingBagVerdict:
    """Check the metadata contract for ``nn/F.embedding_bag``.

    Returns a concrete output shape whenever PyTorch's rank-level output is
    determined.  Value-dependent jagged partition facts that are not available
    are surfaced via ``unknown_reason`` instead of producing a false alarm.
    """

    input_shape = tuple(input_shape)
    weight = _shape(weight_shape)
    if weight is not None:
        if len(weight) != 2:
            return _fail("weight_shape", f"embedding_bag weight must be 2D, got rank {len(weight)}")
        if num_embeddings is None:
            num_embeddings = weight[0]
        if embedding_dim is None:
            embedding_dim = weight[1]
    if embedding_dim is None:
        return _fail("weight_shape", "embedding_dim is unknown")

    mode = str(mode).lower()
    if mode not in _POOLING_MODES:
        return _fail("mode", f"embedding_bag mode must be one of sum/mean/max, got {mode!r}")

    dtype_err = _check_int_dtype(input_dtype, "embedding_bag input")
    if dtype_err is not None:
        return dtype_err
    dtype_err = _check_int_dtype(offsets_dtype, "embedding_bag offsets")
    if dtype_err is not None:
        return dtype_err

    if len(input_shape) not in (1, 2):
        return _fail("input_rank", f"embedding_bag input must be 1D or 2D, got rank {len(input_shape)}")

    if input_value_range is not None and _is_known_int(num_embeddings):
        lo, hi = input_value_range
        if lo < 0 or hi >= num_embeddings:
            return _fail(
                "index_bounds",
                f"embedding_bag indices [{lo}, {hi}] are out of bounds for table size {num_embeddings}",
            )

    psw = _shape(per_sample_weights_shape)
    if psw is not None:
        if mode != "sum":
            return _fail(
                "per_sample_weights",
                f"per_sample_weights is only supported for mode='sum', got mode={mode!r}",
            )
        if not _same_shape(psw, input_shape):
            return _fail(
                "per_sample_weights",
                f"per_sample_weights shape {psw} must exactly match input shape {input_shape}",
            )
        wdt = _canon_dtype(weight_dtype)
        pdt = _canon_dtype(per_sample_weights_dtype)
        if wdt is not None and pdt is not None and wdt != pdt:
            return _fail(
                "dtype",
                f"per_sample_weights dtype {pdt} must match embedding weight dtype {wdt}",
            )

    output_dtype = _canon_dtype(weight_dtype) or "float32"
    if len(input_shape) == 2:
        if offsets_shape is not None:
            return _fail("offsets", "2D embedding_bag input requires offsets=None")
        return EmbeddingBagVerdict(True, (input_shape[0], embedding_dim), output_dtype=output_dtype)

    offsets = _shape(offsets_shape)
    if offsets is None:
        return _fail("offsets", "1D embedding_bag input requires a 1D offsets tensor")
    if len(offsets) != 1:
        return _fail("offsets", f"embedding_bag offsets must be 1D, got rank {len(offsets)}")

    input_len = input_shape[0]
    if offsets_value_range is not None:
        lo, hi = offsets_value_range
        if lo < 0:
            return _fail("offsets", f"offsets must be non-negative, got minimum {lo}")
        if _is_known_int(input_len) and hi > input_len:
            return _fail("offsets", f"offsets cannot exceed input length {input_len}, got maximum {hi}")

    static_offsets_err = _check_static_offsets(offsets_values, input_len, include_last_offset)
    if static_offsets_err is not None:
        return static_offsets_err

    bags, bag_err = _bag_count(offsets[0], include_last_offset)
    if bag_err is not None:
        return _fail("offsets", bag_err)
    unknown_reason = None
    if offsets_values is None:
        unknown_reason = (
            "embedding_bag jagged bag boundaries depend on offset tensor values; "
            "monotonicity and exact ragged partition are UNKNOWN"
        )
    return EmbeddingBagVerdict(
        True,
        (bags, embedding_dim),
        output_dtype=output_dtype,
        unknown_reason=unknown_reason,
    )


def verify_torchrec_embedding_bag(
    feature: TorchRecJaggedSpec,
    *,
    embedding_dim: Dim,
    num_embeddings: Optional[Dim] = None,
    pooling: str = "sum",
    weight_dtype: Optional[Any] = None,
) -> EmbeddingBagVerdict:
    """Check a TorchRec pooled jagged feature against an EmbeddingBag table."""

    offsets_range = None
    if feature.offsets_values:
        offsets_range = (min(feature.offsets_values), max(feature.offsets_values))
    input_range = feature.values_range
    return verify_embedding_bag(
        feature.values_shape,
        offsets_shape=feature.offsets_shape,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        input_dtype=feature.values_dtype,
        offsets_dtype=feature.offsets_dtype,
        weight_dtype=weight_dtype,
        input_value_range=input_range,
        offsets_value_range=offsets_range,
        offsets_values=feature.offsets_values,
        mode=pooling,
        include_last_offset=True,
    )

