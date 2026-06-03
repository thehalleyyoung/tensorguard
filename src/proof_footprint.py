"""Per-operator proof-footprint manifest (Steps 243-244).

The operator-confidence table answers "how much should I trust this transfer?"
This manifest is its auditable source of truth: each row records both that
confidence tag and the proof/evidence footprint backing it.  It covers every
operator registered in
``graph_compiler._UNIVERSAL_TRANSFER_REGISTRY`` and classifies each transfer as
one of:

* ``lean_theorem``: an explicit, per-operator Lean theorem is listed.
* ``pen_and_paper_rule``: a static rule is specified mathematically, but not
  machine checked per operator.
* ``tested_only_rule``: the transfer is validated by live/differential tests,
  without a compact proof rule.
* ``heuristic``: the output shape is data-dependent or intentionally
  approximate.

Lean-backed rows are intentionally allowlisted per operator.  Broad families
such as "matmul" or "reduction" are not promoted wholesale unless the listed
theorem directly covers the registered operator.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.confidence_tags import ConfidenceTag


SCHEMA = "tensorguard.proof_footprint/v1"


class ProofStatus(str, enum.Enum):
    """Evidence tier for one registered transfer function."""

    LEAN_THEOREM = "lean_theorem"
    PEN_AND_PAPER_RULE = "pen_and_paper_rule"
    TESTED_ONLY_RULE = "tested_only_rule"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class LeanFootprint:
    """Lean theorem evidence for an operator row."""

    modules: Sequence[str]
    theorems: Sequence[str]
    rule: str
    evidence: Sequence[str]
    rationale: str


@dataclass(frozen=True)
class RuleFootprint:
    """Non-Lean evidence for an operator row."""

    status: ProofStatus
    rule: str
    evidence: Sequence[str]
    rationale: str


_LEAN: Mapping[str, LeanFootprint] = {
    "F.relu": LeanFootprint(
        modules=("TensorGuard.Extended",),
        theorems=("TensorGuard.relu_shape_preserving",),
        rule="relu(input) has exactly the input shape",
        evidence=("lean/TensorGuard/Extended.lean", "tests/test_lean_conformance.py"),
        rationale="The ReLU identity shape rule is stated and proved directly in Lean.",
    ),
    "torch.relu": LeanFootprint(
        modules=("TensorGuard.Extended",),
        theorems=("TensorGuard.relu_shape_preserving",),
        rule="relu(input) has exactly the input shape",
        evidence=("lean/TensorGuard/Extended.lean", "tests/test_lean_conformance.py"),
        rationale="The ReLU identity shape rule is stated and proved directly in Lean.",
    ),
    "torch.clamp": LeanFootprint(
        modules=("TensorGuard.SoundnessV5",),
        theorems=("TensorGuard.V5.applyOp_sound_clamp",),
        rule="clamp(input, ...) has exactly the input shape",
        evidence=("lean/TensorGuard/SoundnessV5.lean", "tests/test_device_dtype_transfer.py"),
        rationale="Lean proves the V5 clamp transfer is shape-preserving.",
    ),
    "torch.squeeze": LeanFootprint(
        modules=("TensorGuard.SoundnessV5",),
        theorems=("TensorGuard.V5.applyOp_sound_squeeze",),
        rule="squeeze(dim) removes a selected unit dimension when the guard succeeds",
        evidence=("lean/TensorGuard/SoundnessV5.lean", "tests/test_structural_transfers.py"),
        rationale="Lean proves successful squeeze transfers are witnessed by the modeled rule.",
    ),
    "torch.unsqueeze": LeanFootprint(
        modules=("TensorGuard.SoundnessV5",),
        theorems=("TensorGuard.V5.applyOp_sound_unsqueeze",),
        rule="unsqueeze(dim) inserts one size-1 dimension when the guard succeeds",
        evidence=("lean/TensorGuard/SoundnessV5.lean", "tests/test_structural_transfers.py"),
        rationale="Lean proves successful unsqueeze transfers increase rank by one.",
    ),
    "torch.stack": LeanFootprint(
        modules=("TensorGuard.SoundnessV5", "TensorGuard.V5OperatorRules"),
        theorems=("TensorGuard.V5.applyOp_sound_stack", "TensorGuard.V5.stackOp_iface"),
        rule="stack equal-shaped tensors by inserting a new axis",
        evidence=("lean/TensorGuard/SoundnessV5.lean", "tests/test_stack_family.py"),
        rationale="Lean pins the stack guard and the Python tests replay stack-family cases.",
    ),
    "torch.gather": LeanFootprint(
        modules=("TensorGuard.SoundnessV5", "TensorGuard.V5OperatorRules"),
        theorems=("TensorGuard.V5.applyOp_sound_gather", "TensorGuard.V5.gather_eq"),
        rule="gather(input, index, dim) returns the index shape",
        evidence=(
            "lean/TensorGuard/SoundnessV5.lean",
            "tests/test_gather_scatter_index_bounds.py",
            "tests/test_indexing_gather_precise.py",
        ),
        rationale="Lean proves the successful gather transfer returns exactly the index shape.",
    ),
    "torch.scatter": LeanFootprint(
        modules=("TensorGuard.SoundnessV5", "TensorGuard.V5OperatorRules"),
        theorems=("TensorGuard.V5.applyOp_sound_scatter", "TensorGuard.V5.scatter_eq"),
        rule="scatter(input, index, src, dim) preserves the input shape",
        evidence=(
            "lean/TensorGuard/SoundnessV5.lean",
            "tests/test_gather_scatter_index_bounds.py",
            "tests/test_indexing_gather_precise.py",
        ),
        rationale="Lean proves the successful scatter transfer returns exactly the input shape.",
    ),
    "torch.index_select": LeanFootprint(
        modules=("TensorGuard.SoundnessV5",),
        theorems=("TensorGuard.V5.applyOp_sound_index_select",),
        rule="index_select replaces the selected axis by the static index length",
        evidence=("lean/TensorGuard/SoundnessV5.lean", "tests/test_indexing_gather_precise.py"),
        rationale="Lean proves successful index_select transfers are guarded by the modeled rule.",
    ),
    "torch.sum": LeanFootprint(
        modules=("TensorGuard.AssumeGuaranteeExtended", "TensorGuard.RankTransfer"),
        theorems=(
            "TensorGuard.applyOpExt_sound_sum_reduce",
            "TensorGuard.RankTransfer.rankRun_le",
            "TensorGuard.RankTransfer.rankRun_eq_sub_countDrop",
        ),
        rule="sum(dim, keepdim) follows the reduction rank-transfer rule",
        evidence=("lean/TensorGuard/RankTransfer.lean", "tests/test_rank_transfer.py"),
        rationale="Lean proves the sum-reduce guard and the reduction rank-transfer laws.",
    ),
    "torch.mean": LeanFootprint(
        modules=("TensorGuard.AssumeGuaranteeExtended", "TensorGuard.RankTransfer"),
        theorems=("TensorGuard.applyOpExt_sound_mean_reduce", "TensorGuard.RankTransfer.rankRun_le"),
        rule="mean(dim, keepdim) follows the reduction rank-transfer rule",
        evidence=("lean/TensorGuard/RankTransfer.lean", "tests/test_rank_transfer.py"),
        rationale="Lean proves the mean-reduce guard and the shared rank-transfer bound.",
    ),
    "torch.matmul": LeanFootprint(
        modules=("TensorGuard.MatmulSound",),
        theorems=(
            "TensorGuard.MatmulSound.applyOpExt_sound_matmul",
            "TensorGuard.MatmulSound.matmul_contraction_sound",
        ),
        rule="(..., m, k) x (..., k, n) contracts to (..., m, n)",
        evidence=("lean/TensorGuard/MatmulSound.lean", "tests/test_graph_compiler.py"),
        rationale="Lean proves the matmul guard and the two-input contraction theorem.",
    ),
    "torch.mm": LeanFootprint(
        modules=("TensorGuard.MatmulSound", "TensorGuard.SmtEncoding"),
        theorems=(
            "TensorGuard.MatmulSound.matmul_contraction_sound",
            "TensorGuard.SmtEncoding.dtype_smt_matches_dtMatmulBug",
        ),
        rule="2-D matrix multiplication contracts matching inner dimensions",
        evidence=("lean/TensorGuard/MatmulSound.lean", "tests/test_smt_dtype_faithful.py"),
        rationale="The 2-D matrix-multiply rule is covered by the Lean contraction lemma.",
    ),
    "torch.bmm": LeanFootprint(
        modules=("TensorGuard.MatmulSound", "TensorGuard.SmtEncoding"),
        theorems=(
            "TensorGuard.MatmulSound.matmul_contraction_sound",
            "TensorGuard.SmtEncoding.dtype_smt_matches_dtMatmulBug",
        ),
        rule="batched matrix multiplication contracts matching inner dimensions per batch",
        evidence=("lean/TensorGuard/MatmulSound.lean", "tests/test_smt_dtype_faithful.py"),
        rationale="The batched matrix-multiply rule uses the same Lean contraction law.",
    ),
    "torch.broadcast_shapes": LeanFootprint(
        modules=("TensorGuard.BroadcastChain", "TensorGuard.SmtEncoding"),
        theorems=(
            "TensorGuard.BroadcastChain.bcDim_none_iff",
            "TensorGuard.BroadcastChain.bcRun_append",
            "TensorGuard.SmtEncoding.broadcast_smt_sat_iff_bcDim_some",
        ),
        rule="broadcast each aligned dimension by equality-or-one, folding across operands",
        evidence=("lean/TensorGuard/BroadcastChain.lean", "tests/test_broadcast_dim_chain.py"),
        rationale="Lean proves the broadcast-dimension chain and SMT formula faithfulness.",
    ),
    "torch.broadcast_tensors": LeanFootprint(
        modules=("TensorGuard.BroadcastChain", "TensorGuard.SmtEncoding"),
        theorems=(
            "TensorGuard.BroadcastChain.bcDim_none_iff",
            "TensorGuard.BroadcastChain.bcRun_append",
            "TensorGuard.SmtEncoding.broadcast_smt_sat_iff_bcDim_some",
        ),
        rule="broadcast each tensor shape by the same equality-or-one dimension rule",
        evidence=("lean/TensorGuard/BroadcastChain.lean", "tests/test_broadcast_dim_chain.py"),
        rationale="Lean proves the broadcast-dimension chain and SMT formula faithfulness.",
    ),
}


_ACTIVATIONS = frozenset({
    "relu", "gelu", "silu", "mish", "hardswish", "hardsigmoid", "leaky_relu",
    "elu", "selu", "celu", "prelu", "rrelu", "softplus", "softsign",
    "tanhshrink", "softshrink", "hardshrink", "logsigmoid", "sigmoid", "tanh",
})

_ELEMENTWISE_UNARY = frozenset({
    "abs", "neg", "sign", "ceil", "floor", "round", "exp", "log", "log2",
    "log10", "sqrt", "rsqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "erf", "erfc", "clamp", "clip", "nan_to_num",
})

_COMPARISON = frozenset({
    "eq", "ne", "gt", "ge", "lt", "le", "equal", "isnan", "isinf", "isfinite",
})

_POINTWISE_BASES = _ACTIVATIONS | _ELEMENTWISE_UNARY | _COMPARISON

_REDUCTION_BASES = frozenset({
    "sum", "mean", "prod", "max", "min", "std", "var", "norm", "logsumexp",
    "any", "all", "amax", "amin",
})

_STRUCTURAL_BASES = frozenset({
    "squeeze", "unsqueeze", "movedim", "moveaxis", "swapaxes", "swapdims",
    "roll", "rot90", "flip", "repeat_interleave", "tile",
})

_STACK_BASES = frozenset({"stack", "hstack", "vstack", "dstack", "column_stack", "row_stack"})

_MATMUL_BASES = frozenset({"matmul", "bmm", "mm", "mv", "outer", "kron", "tensordot", "cross"})

_STRUCTURAL_EXACT = (
    _STRUCTURAL_BASES
    | _STACK_BASES
    | frozenset({
        "gather",
        "scatter",
        "index_select",
        "sort",
        "argsort",
        "topk",
        "broadcast_tensors",
        "broadcast_shapes",
        "bernoulli",
        "poisson",
        "cdist",
    })
)

_HEURISTIC = frozenset({"torch.einsum", "torch.unique", "torch.multinomial"})
_DATA_DEPENDENT = frozenset({"einsum", "unique", "multinomial"})
_EXACT_LINALG = frozenset({"cholesky", "eig", "inv", "qr", "solve", "svd"})
_TESTED_PREFIXES = ("torch.linalg.", "torch.fft.")

_TESTED_ONLY: Mapping[str, RuleFootprint] = {
    "sort_family": RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "sort/topk/argsort preserve or statically set selected output axes",
        ("tests/test_index_value_ops_precise.py", "tests/test_graph_compiler.py"),
        "Index-value operators are validated by tests but do not yet have per-op Lean theorems.",
    ),
    "linalg": RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "linear-algebra contracts enforce square/broadcast/multi-output shapes",
        ("tests/test_linalg_verify.py", "tests/test_operator_confidence.py"),
        "torch.linalg contracts are checked against live torch, without per-op Lean theorems.",
    ),
    "fft": RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "FFT contracts preserve transformed ranks and real/complex endpoint sizes",
        ("tests/test_complex_verify.py", "tests/test_operator_confidence.py"),
        "FFT contracts are differential-tested against torch, without per-op Lean theorems.",
    ),
    "stochastic": RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "stochastic samplers return statically known tensor shapes",
        ("tests/test_graph_compiler.py", "tests/test_operator_confidence.py"),
        "Stochastic shape contracts are registry-tested but not mechanized per operator.",
    ),
    "cdist": RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "cdist returns pairwise-distance matrix shapes from the two input point axes",
        ("tests/test_graph_compiler.py", "tests/test_operator_confidence.py"),
        "cdist is covered by registry/contract tests, without a per-op Lean theorem.",
    ),
}


def _base_name(op_name: str) -> str:
    return op_name.rsplit(".", 1)[-1]


def confidence_for(op_name: str) -> Tuple[ConfidenceTag, str]:
    """Return the confidence tag and confidence-specific rationale for an op."""

    if op_name.startswith("torch.linalg."):
        base = _base_name(op_name)
        if base in _EXACT_LINALG:
            return (
                ConfidenceTag.SOUND,
                "torch.linalg shape contract with exact rank, square, "
                "broadcasting and multi-output shape checks enforced soundly.",
            )
        return (
            ConfidenceTag.HEURISTIC,
            "torch.linalg operator without an exact TensorGuard shape contract; "
            "defaulting conservatively to heuristic.",
        )
    if op_name.startswith("torch.fft."):
        return (
            ConfidenceTag.SOUND,
            "FFT family: exact, well-defined output-shape rule (e.g. rfft maps "
            "the last dim n -> n//2 + 1) enforced soundly.",
        )

    base = _base_name(op_name)
    if base in _ACTIVATIONS:
        return (
            ConfidenceTag.COMPLETE,
            "Pointwise activation: output shape is identical to the input, so "
            "the transfer is exact (sound and complete).",
        )
    if base in _ELEMENTWISE_UNARY:
        return (
            ConfidenceTag.COMPLETE,
            "Elementwise unary op: shape-preserving, so the transfer is exact "
            "(sound and complete).",
        )
    if base in _COMPARISON:
        return (
            ConfidenceTag.COMPLETE,
            "Elementwise comparison: shape-preserving boolean output, so the "
            "transfer is exact (sound and complete).",
        )
    if base in _MATMUL_BASES:
        return (
            ConfidenceTag.SOUND,
            "Matmul-family op with an exact, well-defined contraction rule that "
            "is enforced soundly (full completeness not claimed for every "
            "broadcasting / zero-dim edge case).",
        )
    if base in _REDUCTION_BASES:
        return (
            ConfidenceTag.SOUND,
            "Reduction with an exact dim/keepdim shape rule enforced soundly "
            "(full completeness not claimed for every keepdim edge case).",
        )
    if base in _STRUCTURAL_EXACT:
        return (
            ConfidenceTag.SOUND,
            "Structural op whose output shape is an exact function of the input "
            "shapes and static integer arguments; enforced soundly.",
        )
    if base in _DATA_DEPENDENT:
        return (
            ConfidenceTag.HEURISTIC,
            "Output shape depends on runtime values or is approximated "
            "generically; best-effort, neither sound nor complete in general.",
        )

    return (
        ConfidenceTag.HEURISTIC,
        "No explicit confidence classification; defaulting conservatively to "
        "heuristic (best-effort).",
    )


def _registry_names() -> List[str]:
    from src.graph_compiler import _UNIVERSAL_TRANSFER_REGISTRY

    return sorted(_UNIVERSAL_TRANSFER_REGISTRY)


def _pen_and_paper(status_rule: str, evidence: Sequence[str], rationale: str) -> RuleFootprint:
    return RuleFootprint(ProofStatus.PEN_AND_PAPER_RULE, status_rule, evidence, rationale)


def _fallback_for(op_name: str) -> RuleFootprint:
    base = _base_name(op_name)
    if op_name in _HEURISTIC or confidence_for(op_name)[0] is ConfidenceTag.HEURISTIC:
        return RuleFootprint(
            ProofStatus.HEURISTIC,
            "no proof rule claimed for data-dependent or approximated output shape",
            ("src/proof_footprint.py", "tests/test_operator_confidence.py"),
            confidence_for(op_name)[1],
        )
    if op_name.startswith("torch.linalg."):
        return _TESTED_ONLY["linalg"]
    if op_name.startswith("torch.fft."):
        return _TESTED_ONLY["fft"]
    if base in _POINTWISE_BASES:
        return _pen_and_paper(
            "shape-preserving pointwise/comparison transfer: output shape equals input shape",
            ("src/graph_compiler.py", "tests/test_graph_compiler.py"),
            "The rule is an elementary shape identity; only explicit allowlisted ops claim Lean.",
        )
    if base in _REDUCTION_BASES:
        return _pen_and_paper(
            "single-axis reduction removes or retains the reduced axis depending on keepdim",
            ("src/graph_compiler.py", "tests/test_rank_transfer.py"),
            "The reduction rule is statically specified; non-allowlisted reductions rely on the paper rule.",
        )
    if base in _STACK_BASES:
        return _pen_and_paper(
            "stack-family operators combine equal-shaped tensors along statically determined axes",
            ("src/graph_compiler.py", "tests/test_stack_family.py"),
            "Stack-family variants are specified by static axis rules; only torch.stack is Lean-backed.",
        )
    if base in _STRUCTURAL_BASES:
        return _pen_and_paper(
            "structural dimension transform with output shape determined by input shape and static args",
            ("src/graph_compiler.py", "tests/test_structural_transfers.py"),
            "The rule is a finite dimension rewrite; only allowlisted structural ops claim Lean.",
        )
    if base in _MATMUL_BASES:
        return _pen_and_paper(
            "linear/tensor contraction rule determined by ranks, broadcast prefixes, and static axes",
            ("src/graph_compiler.py", "tests/test_graph_compiler.py"),
            "The operator has a static contraction rule; only matmul/mm/bmm are tied to Lean theorems.",
        )
    if base in {"sort", "topk", "argsort"}:
        return _TESTED_ONLY["sort_family"]
    if base in {"bernoulli", "poisson"}:
        return _TESTED_ONLY["stochastic"]
    if base == "cdist":
        return _TESTED_ONLY["cdist"]
    if op_name.startswith(_TESTED_PREFIXES):
        family = "linalg" if op_name.startswith("torch.linalg.") else "fft"
        return _TESTED_ONLY[family]
    return RuleFootprint(
        ProofStatus.TESTED_ONLY_RULE,
        "registered transfer with no stronger proof-footprint classification",
        ("src/graph_compiler.py", "tests/test_operator_confidence.py"),
        "The operator is covered by the registry and confidence tests, but no proof rule is claimed.",
    )


def footprint_for(op_name: str) -> Dict[str, object]:
    """Return the proof-footprint row for one operator name."""

    confidence, confidence_rationale = confidence_for(op_name)
    if op_name in _LEAN:
        fp = _LEAN[op_name]
        return {
            "operator": op_name,
            "proof_status": ProofStatus.LEAN_THEOREM.value,
            "confidence": confidence.value,
            "confidence_rationale": confidence_rationale,
            "rule": fp.rule,
            "lean_modules": list(fp.modules),
            "lean_theorems": list(fp.theorems),
            "evidence": sorted(fp.evidence),
            "rationale": fp.rationale,
        }
    fp2 = _fallback_for(op_name)
    return {
        "operator": op_name,
        "proof_status": fp2.status.value,
        "confidence": confidence.value,
        "confidence_rationale": confidence_rationale,
        "rule": fp2.rule,
        "lean_modules": [],
        "lean_theorems": [],
        "evidence": sorted(fp2.evidence),
        "rationale": fp2.rationale,
    }


def proof_footprint_table(names: Iterable[str] = ()) -> List[Dict[str, object]]:
    """Return sorted proof-footprint rows for the registry or provided names."""

    op_names = sorted(set(names) if names else _registry_names())
    return [footprint_for(name) for name in op_names]


def summary_for(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    summary: Dict[str, int] = {status.value: 0 for status in ProofStatus}
    for row in rows:
        summary[str(row["proof_status"])] += 1
    return summary


def to_payload() -> Dict[str, object]:
    rows = proof_footprint_table()
    return {
        "schema": SCHEMA,
        "registry": "src.graph_compiler._UNIVERSAL_TRANSFER_REGISTRY",
        "default_status": ProofStatus.HEURISTIC.value,
        "summary": summary_for(rows),
        "total": len(rows),
        "operators": rows,
    }


def to_json(indent: int = 2) -> str:
    """Serialize the proof-footprint manifest as deterministic JSON."""

    return json.dumps(to_payload(), indent=indent, sort_keys=False)


if __name__ == "__main__":  # pragma: no cover
    print(to_json())
