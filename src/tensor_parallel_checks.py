"""
tensor_parallel_checks.py — static shape-consistency checks for tensor-parallel
(Megatron-style) sharding (100_STEPS.md Steps 97 and 218).

``src/distributed_verification.py`` already covers FSDP and DeepSpeed ZeRO
parameter sharding. The remaining pain point with no good static tool is
**tensor parallelism**: a column-parallel ``Linear`` shards its weight along the
output dimension and a row-parallel ``Linear`` shards along the input
dimension, and the two must be chained with exactly the right
``gather_output`` / ``input_is_parallel`` flags and divisible dimensions, or the
model silently produces wrong numbers or crashes at runtime on N GPUs that a
single-GPU smoke test never exercises.

This module models tensor-parallel linear stacks and attention blocks
symbolically and checks:

  * **Divisibility** — a column-parallel layer's ``out_features`` (and a
    row-parallel layer's ``in_features``) must be divisible by the
    tensor-parallel world size ``tp_size``; otherwise the shard is ragged.
  * **Inner-dim match** — chained layers must agree on the contracted dimension
    (``prev.out_features == next.in_features``), exactly as for ordinary linears.
  * **Communication-flag consistency** — the canonical no-extra-communication
    Megatron MLP is ColumnParallel(``gather_output=False``) →
    RowParallel(``input_is_parallel=True``). If a column layer gathers its
    output, the following row layer must NOT expect parallel input, and vice
    versa; a mismatch double-counts or mis-shapes the activation.
  * **Attention head partitioning** — MQA/GQA query heads must be evenly split
    across tensor-parallel ranks, key/value heads must either shard cleanly or
    follow the Megatron replication rule, and q/k/v/o projection matrices must
    match the exact ``heads * head_dim`` dimensions used by HuggingFace and
    Megatron-style implementations.
  * **Sequence-parallel LayerNorm** — sequence sharding may not split the
    normalized hidden axis, and the LayerNorm normalized shape must match the
    trailing activation dimensions it normalizes.

The checks are sound for the modeled stack: every reported issue corresponds to
a real ragged shard, dimension mismatch, or flag inconsistency. The companion
harness ``reproducibility/tensor_parallel_sharding.py`` proves the decomposition
against real PyTorch by hand-sharding reference linear and attention modules and
comparing the gathered/all-reduced output to the unsharded forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union


class TPKind(str, Enum):
    COLUMN = "column"   # shards weight along out_features (dim 0)
    ROW = "row"         # shards weight along in_features (dim 1)


class TPKVSharding(str, Enum):
    AUTO = "auto"
    SHARD = "shard"
    REPLICATE = "replicate"


class TPIssueKind(str, Enum):
    INDIVISIBLE_SHARD = "indivisible_shard"
    INNER_DIM_MISMATCH = "inner_dim_mismatch"
    COMM_FLAG_MISMATCH = "comm_flag_mismatch"
    HEAD_DIM_INCOMPATIBLE = "head_dim_incompatible"
    QUERY_HEAD_TP_INCOMPATIBLE = "query_head_tp_incompatible"
    KV_HEAD_TP_INCOMPATIBLE = "kv_head_tp_incompatible"
    GQA_GROUP_MISMATCH = "gqa_group_mismatch"
    PROJECTION_SHAPE_MISMATCH = "projection_shape_mismatch"
    SEQUENCE_PARALLEL_AXIS = "sequence_parallel_axis"
    LAYERNORM_SHAPE_MISMATCH = "layernorm_shape_mismatch"


@dataclass(frozen=True)
class TPLinearSpec:
    name: str
    in_features: int
    out_features: int
    kind: TPKind
    # column-parallel: whether the layer all-gathers its sharded output.
    gather_output: bool = False
    # row-parallel: whether the layer expects its input to already be sharded.
    input_is_parallel: bool = False


ShapeDim = Union[int, str]
Shape = Tuple[ShapeDim, ...]
MatrixShape = Tuple[int, int]


@dataclass(frozen=True)
class TPAttentionSpec:
    """Tensor-parallel attention contract for HF/Megatron-style blocks.

    ``hidden_size`` is the input/output embedding dimension. ``head_dim`` may be
    independent from ``hidden_size // num_attention_heads``; real HuggingFace
    configs allow this, so projection checks use ``heads * head_dim`` directly.
    Global projection shapes model ordinary HF modules, while ``*_shard_shape``
    fields model per-rank Megatron-style local linears.
    """

    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_sharding: TPKVSharding = TPKVSharding.AUTO
    q_proj_shape: Optional[MatrixShape] = None
    k_proj_shape: Optional[MatrixShape] = None
    v_proj_shape: Optional[MatrixShape] = None
    o_proj_shape: Optional[MatrixShape] = None
    q_shard_shape: Optional[MatrixShape] = None
    k_shard_shape: Optional[MatrixShape] = None
    v_shard_shape: Optional[MatrixShape] = None
    o_shard_shape: Optional[MatrixShape] = None
    sequence_parallel: bool = False
    activation_shape: Optional[Shape] = None
    sequence_parallel_axis: int = 1
    layer_norm_shape: Optional[Shape] = None


@dataclass(frozen=True)
class TPIssue:
    kind: TPIssueKind
    message: str
    layer: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "message": self.message,
                "layer": self.layer, "evidence": self.evidence}


@dataclass
class TPVerifyResult:
    tp_size: int
    ok: bool
    issues: List[TPIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"tp_size": self.tp_size, "ok": self.ok,
                "issues": [i.to_dict() for i in self.issues]}


def verify_tensor_parallel(specs: List[TPLinearSpec],
                           tp_size: int) -> TPVerifyResult:
    """Check a tensor-parallel linear stack for shape/flag consistency.

    ``specs`` is the ordered list of parallel linears (as they execute). The
    contracted (inner) dimension of layer i must equal the output dimension of
    layer i-1.
    """
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    issues: List[TPIssue] = []

    for spec in specs:
        sharded_dim, dim_name = (
            (spec.out_features, "out_features") if spec.kind is TPKind.COLUMN
            else (spec.in_features, "in_features")
        )
        if tp_size > 1 and sharded_dim % tp_size != 0:
            issues.append(TPIssue(
                kind=TPIssueKind.INDIVISIBLE_SHARD,
                message=(f"{spec.kind.value}-parallel layer shards {dim_name}="
                         f"{sharded_dim} across tp_size={tp_size}, but "
                         f"{sharded_dim} % {tp_size} = "
                         f"{sharded_dim % tp_size} != 0 (ragged shard)"),
                layer=spec.name,
                evidence=f"{dim_name}={sharded_dim}, tp_size={tp_size}",
            ))

    for prev, nxt in zip(specs, specs[1:]):
        if prev.out_features != nxt.in_features:
            issues.append(TPIssue(
                kind=TPIssueKind.INNER_DIM_MISMATCH,
                message=(f"{prev.name}.out_features={prev.out_features} != "
                         f"{nxt.name}.in_features={nxt.in_features}"),
                layer=nxt.name,
                evidence=f"{prev.out_features} vs {nxt.in_features}",
            ))
        # Communication-flag consistency only constrains a column -> row chain.
        if prev.kind is TPKind.COLUMN and nxt.kind is TPKind.ROW:
            # No extra communication iff column does NOT gather and row expects
            # parallel input. The two flags must therefore be opposite.
            if prev.gather_output == nxt.input_is_parallel:
                issues.append(TPIssue(
                    kind=TPIssueKind.COMM_FLAG_MISMATCH,
                    message=(
                        f"column layer {prev.name} gather_output="
                        f"{prev.gather_output} is inconsistent with row layer "
                        f"{nxt.name} input_is_parallel={nxt.input_is_parallel}: "
                        "a non-gathered (sharded) column output must feed a "
                        "row layer with input_is_parallel=True, and a gathered "
                        "output must feed input_is_parallel=False"),
                    layer=nxt.name,
                    evidence=f"gather_output={prev.gather_output}, "
                             f"input_is_parallel={nxt.input_is_parallel}",
                ))

    return TPVerifyResult(tp_size=tp_size, ok=not issues, issues=issues)


def megatron_mlp(hidden: int, ffn: int, *, gather_output: bool = False,
                 input_is_parallel: bool = True) -> List[TPLinearSpec]:
    """Construct the canonical two-layer Megatron MLP spec
    (ColumnParallel hidden->ffn, RowParallel ffn->hidden)."""
    return [
        TPLinearSpec("fc1", hidden, ffn, TPKind.COLUMN,
                     gather_output=gather_output),
        TPLinearSpec("fc2", ffn, hidden, TPKind.ROW,
                     input_is_parallel=input_is_parallel),
    ]


def verify_tensor_parallel_attention(
    specs: Union[TPAttentionSpec, Sequence[TPAttentionSpec]],
    tp_size: int,
) -> TPVerifyResult:
    """Check tensor-parallel attention contracts for MQA/GQA and SP LayerNorm.

    The verifier refutes only concrete invalid contracts.  If ``head_dim`` is
    omitted, it is inferred from ``hidden_size // num_attention_heads`` and
    divisibility is required.  If ``head_dim`` is explicit, it is treated as an
    independent model-config field, matching HuggingFace Llama-style modules.
    """

    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    spec_list = [specs] if isinstance(specs, TPAttentionSpec) else list(specs)
    issues: List[TPIssue] = []
    for spec in spec_list:
        _verify_attention_spec(spec, tp_size, issues)
    return TPVerifyResult(tp_size=tp_size, ok=not issues, issues=issues)


def llama_gqa_attention(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    *,
    head_dim: Optional[int] = None,
    sequence_parallel: bool = False,
) -> TPAttentionSpec:
    """Construct a HuggingFace-Llama-style attention spec."""

    resolved = _infer_head_dim(hidden_size, num_attention_heads, head_dim)
    q_dim = num_attention_heads * resolved if resolved is not None else 0
    kv_dim = num_key_value_heads * resolved if resolved is not None else 0
    activation = ("batch", "seq", hidden_size) if sequence_parallel else None
    return TPAttentionSpec(
        name="llama_attention",
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        q_proj_shape=(q_dim, hidden_size) if q_dim else None,
        k_proj_shape=(kv_dim, hidden_size) if kv_dim else None,
        v_proj_shape=(kv_dim, hidden_size) if kv_dim else None,
        o_proj_shape=(hidden_size, q_dim) if q_dim else None,
        sequence_parallel=sequence_parallel,
        activation_shape=activation,
        layer_norm_shape=(hidden_size,) if sequence_parallel else None,
    )


def megatron_attention(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    tp_size: int,
    *,
    head_dim: Optional[int] = None,
    kv_sharding: TPKVSharding = TPKVSharding.AUTO,
    sequence_parallel: bool = False,
) -> TPAttentionSpec:
    """Construct a Megatron-style local-shard attention spec."""

    resolved = _infer_head_dim(hidden_size, num_attention_heads, head_dim)
    q_dim = num_attention_heads * resolved if resolved is not None else 0
    kv_dim = num_key_value_heads * resolved if resolved is not None else 0
    q_local, kv_local = _local_attention_dims(
        num_attention_heads,
        num_key_value_heads,
        resolved,
        tp_size,
        kv_sharding,
    )
    activation = ("batch", "seq", hidden_size) if sequence_parallel else None
    return TPAttentionSpec(
        name="megatron_attention",
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        kv_sharding=kv_sharding,
        q_proj_shape=(q_dim, hidden_size) if q_dim else None,
        k_proj_shape=(kv_dim, hidden_size) if kv_dim else None,
        v_proj_shape=(kv_dim, hidden_size) if kv_dim else None,
        o_proj_shape=(hidden_size, q_dim) if q_dim else None,
        q_shard_shape=(q_local, hidden_size) if q_local is not None else None,
        k_shard_shape=(kv_local, hidden_size) if kv_local is not None else None,
        v_shard_shape=(kv_local, hidden_size) if kv_local is not None else None,
        o_shard_shape=(hidden_size, q_local) if q_local is not None else None,
        sequence_parallel=sequence_parallel,
        activation_shape=activation,
        layer_norm_shape=(hidden_size,) if sequence_parallel else None,
    )


def _infer_head_dim(
    hidden_size: int,
    num_attention_heads: int,
    head_dim: Optional[int],
) -> Optional[int]:
    if head_dim is not None:
        return head_dim
    if num_attention_heads <= 0 or hidden_size % num_attention_heads != 0:
        return None
    return hidden_size // num_attention_heads


def _issue(
    issues: List[TPIssue],
    kind: TPIssueKind,
    layer: str,
    message: str,
    evidence: str = "",
) -> None:
    issues.append(TPIssue(kind=kind, message=message, layer=layer, evidence=evidence))


def _verify_attention_spec(
    spec: TPAttentionSpec,
    tp_size: int,
    issues: List[TPIssue],
) -> None:
    if spec.hidden_size <= 0:
        _issue(
            issues,
            TPIssueKind.HEAD_DIM_INCOMPATIBLE,
            spec.name,
            f"hidden_size must be positive, got {spec.hidden_size}",
        )
    q_heads = spec.num_attention_heads
    kv_heads = spec.num_key_value_heads or q_heads
    if q_heads <= 0:
        _issue(
            issues,
            TPIssueKind.QUERY_HEAD_TP_INCOMPATIBLE,
            spec.name,
            f"num_attention_heads must be positive, got {q_heads}",
        )
        return
    if kv_heads <= 0:
        _issue(
            issues,
            TPIssueKind.KV_HEAD_TP_INCOMPATIBLE,
            spec.name,
            f"num_key_value_heads must be positive, got {kv_heads}",
        )
        return

    head_dim = _infer_head_dim(spec.hidden_size, q_heads, spec.head_dim)
    if head_dim is None:
        _issue(
            issues,
            TPIssueKind.HEAD_DIM_INCOMPATIBLE,
            spec.name,
            f"hidden_size={spec.hidden_size} is not divisible by "
            f"num_attention_heads={q_heads}; provide explicit head_dim",
            evidence=f"hidden_size={spec.hidden_size}, heads={q_heads}",
        )
        return
    if head_dim <= 0:
        _issue(
            issues,
            TPIssueKind.HEAD_DIM_INCOMPATIBLE,
            spec.name,
            f"head_dim must be positive, got {head_dim}",
        )
        return

    if tp_size > 1 and q_heads % tp_size != 0:
        _issue(
            issues,
            TPIssueKind.QUERY_HEAD_TP_INCOMPATIBLE,
            spec.name,
            f"query heads {q_heads} must divide evenly across tp_size={tp_size}",
            evidence=f"{q_heads} % {tp_size} = {q_heads % tp_size}",
        )

    if q_heads % kv_heads != 0:
        _issue(
            issues,
            TPIssueKind.GQA_GROUP_MISMATCH,
            spec.name,
            f"num_attention_heads={q_heads} must be a multiple of "
            f"num_key_value_heads={kv_heads}",
            evidence=f"{q_heads} % {kv_heads} = {q_heads % kv_heads}",
        )

    kv_strategy = _resolve_kv_strategy(spec, tp_size, kv_heads, issues)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    _check_matrix_shape(
        spec.q_proj_shape,
        (q_dim, spec.hidden_size),
        spec.name,
        "q_proj",
        issues,
    )
    _check_matrix_shape(
        spec.k_proj_shape,
        (kv_dim, spec.hidden_size),
        spec.name,
        "k_proj",
        issues,
    )
    _check_matrix_shape(
        spec.v_proj_shape,
        (kv_dim, spec.hidden_size),
        spec.name,
        "v_proj",
        issues,
    )
    _check_matrix_shape(
        spec.o_proj_shape,
        (spec.hidden_size, q_dim),
        spec.name,
        "o_proj",
        issues,
    )

    q_local, kv_local = _local_attention_dims(
        q_heads,
        kv_heads,
        head_dim,
        tp_size,
        kv_strategy or spec.kv_sharding,
    )
    _check_matrix_shape(
        spec.q_shard_shape,
        (q_local, spec.hidden_size) if q_local is not None else None,
        spec.name,
        "q_shard",
        issues,
    )
    _check_matrix_shape(
        spec.k_shard_shape,
        (kv_local, spec.hidden_size) if kv_local is not None else None,
        spec.name,
        "k_shard",
        issues,
    )
    _check_matrix_shape(
        spec.v_shard_shape,
        (kv_local, spec.hidden_size) if kv_local is not None else None,
        spec.name,
        "v_shard",
        issues,
    )
    _check_matrix_shape(
        spec.o_shard_shape,
        (spec.hidden_size, q_local) if q_local is not None else None,
        spec.name,
        "o_shard",
        issues,
    )
    _verify_sequence_parallel_layernorm(spec, issues)


def _resolve_kv_strategy(
    spec: TPAttentionSpec,
    tp_size: int,
    kv_heads: int,
    issues: List[TPIssue],
) -> Optional[TPKVSharding]:
    if tp_size == 1:
        return TPKVSharding.SHARD
    shard_ok = kv_heads % tp_size == 0
    replicate_ok = tp_size % kv_heads == 0
    requested = spec.kv_sharding
    if requested is TPKVSharding.SHARD:
        if not shard_ok:
            _issue(
                issues,
                TPIssueKind.KV_HEAD_TP_INCOMPATIBLE,
                spec.name,
                f"kv heads {kv_heads} cannot be sharded across tp_size={tp_size}",
                evidence=f"{kv_heads} % {tp_size} = {kv_heads % tp_size}",
            )
            return None
        return TPKVSharding.SHARD
    if requested is TPKVSharding.REPLICATE:
        if not replicate_ok:
            _issue(
                issues,
                TPIssueKind.KV_HEAD_TP_INCOMPATIBLE,
                spec.name,
                f"kv heads {kv_heads} cannot be replicated across tp_size={tp_size}; "
                "tp_size must be a multiple of num_key_value_heads",
                evidence=f"{tp_size} % {kv_heads} = {tp_size % kv_heads}",
            )
            return None
        return TPKVSharding.REPLICATE
    if shard_ok:
        return TPKVSharding.SHARD
    if replicate_ok:
        return TPKVSharding.REPLICATE
    _issue(
        issues,
        TPIssueKind.KV_HEAD_TP_INCOMPATIBLE,
        spec.name,
        f"kv heads {kv_heads} neither shard nor replicate cleanly across "
        f"tp_size={tp_size}",
        evidence=(
            f"{kv_heads} % {tp_size} = {kv_heads % tp_size}, "
            f"{tp_size} % {kv_heads} = {tp_size % kv_heads}"
        ),
    )
    return None


def _local_attention_dims(
    q_heads: int,
    kv_heads: int,
    head_dim: Optional[int],
    tp_size: int,
    kv_sharding: TPKVSharding,
) -> Tuple[Optional[int], Optional[int]]:
    if head_dim is None or tp_size < 1 or q_heads % tp_size != 0:
        return None, None
    q_local = (q_heads // tp_size) * head_dim
    if tp_size == 1:
        return q_local, kv_heads * head_dim
    if kv_sharding is TPKVSharding.SHARD:
        if kv_heads % tp_size != 0:
            return q_local, None
        return q_local, (kv_heads // tp_size) * head_dim
    if kv_sharding is TPKVSharding.REPLICATE:
        if tp_size % kv_heads != 0:
            return q_local, None
        return q_local, head_dim
    if kv_heads % tp_size == 0:
        return q_local, (kv_heads // tp_size) * head_dim
    if tp_size % kv_heads == 0:
        return q_local, head_dim
    return q_local, None


def _check_matrix_shape(
    actual: Optional[MatrixShape],
    expected: Optional[MatrixShape],
    layer: str,
    label: str,
    issues: List[TPIssue],
) -> None:
    if actual is None or expected is None:
        return
    if tuple(actual) != tuple(expected):
        _issue(
            issues,
            TPIssueKind.PROJECTION_SHAPE_MISMATCH,
            layer,
            f"{label} shape {tuple(actual)} does not match expected {tuple(expected)}",
            evidence=f"{label}: actual={tuple(actual)}, expected={tuple(expected)}",
        )


def _normalize_axis(axis: int, rank: int) -> Optional[int]:
    if rank <= 0:
        return None
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def _verify_sequence_parallel_layernorm(
    spec: TPAttentionSpec,
    issues: List[TPIssue],
) -> None:
    if not spec.sequence_parallel and spec.layer_norm_shape is None:
        return
    activation = spec.activation_shape or ("batch", "seq", spec.hidden_size)
    normalized = spec.layer_norm_shape or (spec.hidden_size,)
    rank = len(activation)
    axis = _normalize_axis(spec.sequence_parallel_axis, rank)
    if axis is None:
        _issue(
            issues,
            TPIssueKind.SEQUENCE_PARALLEL_AXIS,
            spec.name,
            f"sequence_parallel_axis={spec.sequence_parallel_axis} is invalid "
            f"for activation rank {rank}",
        )
        return

    if len(normalized) > rank:
        _issue(
            issues,
            TPIssueKind.LAYERNORM_SHAPE_MISMATCH,
            spec.name,
            f"LayerNorm normalized_shape {normalized} is longer than activation "
            f"shape {activation}",
            evidence=f"activation={activation}, normalized_shape={normalized}",
        )
        return

    trailing = activation[rank - len(normalized):]
    for got, want in zip(trailing, normalized):
        if isinstance(got, int) and isinstance(want, int) and got != want:
            _issue(
                issues,
                TPIssueKind.LAYERNORM_SHAPE_MISMATCH,
                spec.name,
                f"LayerNorm normalized_shape {normalized} does not match "
                f"activation trailing shape {trailing}",
                evidence=f"activation={activation}, normalized_shape={normalized}",
            )
            break

    if spec.sequence_parallel:
        normalized_axes = set(range(rank - len(normalized), rank))
        if axis in normalized_axes:
            _issue(
                issues,
                TPIssueKind.SEQUENCE_PARALLEL_AXIS,
                spec.name,
                f"sequence_parallel_axis={spec.sequence_parallel_axis} shards a "
                "LayerNorm-normalized axis; shard the sequence axis instead",
                evidence=(
                    f"activation={activation}, normalized_axes="
                    f"{tuple(sorted(normalized_axes))}"
                ),
            )


__all__ = [
    "TPKind",
    "TPKVSharding",
    "TPIssueKind",
    "TPLinearSpec",
    "TPAttentionSpec",
    "TPIssue",
    "TPVerifyResult",
    "verify_tensor_parallel",
    "verify_tensor_parallel_attention",
    "megatron_mlp",
    "megatron_attention",
    "llama_gqa_attention",
]
