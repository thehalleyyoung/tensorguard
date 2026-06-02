"""
tensor_parallel_checks.py — static shape-consistency checks for tensor-parallel
(Megatron-style) sharding (100_STEPS.md Step 97, Phase 10).

``src/distributed_verification.py`` already covers FSDP and DeepSpeed ZeRO
parameter sharding. The remaining pain point with no good static tool is
**tensor parallelism**: a column-parallel ``Linear`` shards its weight along the
output dimension and a row-parallel ``Linear`` shards along the input
dimension, and the two must be chained with exactly the right
``gather_output`` / ``input_is_parallel`` flags and divisible dimensions, or the
model silently produces wrong numbers or crashes at runtime on N GPUs that a
single-GPU smoke test never exercises.

This module models a tensor-parallel linear stack symbolically and checks:

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

The checks are sound for the modeled stack: every reported issue corresponds to
a real ragged shard, dimension mismatch, or flag inconsistency. The companion
harness ``reproducibility/tensor_parallel_sharding.py`` proves the decomposition
against real PyTorch by hand-sharding a reference linear stack and comparing the
gathered/all-reduced output to the unsharded forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TPKind(str, Enum):
    COLUMN = "column"   # shards weight along out_features (dim 0)
    ROW = "row"         # shards weight along in_features (dim 1)


class TPIssueKind(str, Enum):
    INDIVISIBLE_SHARD = "indivisible_shard"
    INNER_DIM_MISMATCH = "inner_dim_mismatch"
    COMM_FLAG_MISMATCH = "comm_flag_mismatch"


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
