"""
Thread-Modular Verification for TorchDynamo Graph-Break Composition.

Formalizes graph-break composition as thread-modular verification
(Flanagan–Qadeer 2003 style): each Dynamo subgraph is a "thread"
with pre/postconditions on shape environments, and inter-break Python
code is modeled as an abstract transformer.

For each pair of adjacent subgraphs (G_i, G_{i+1}), verifies:

    post(G_i) ∘ T_inter ⊆ pre(G_{i+1})

where T_inter is the inter-break abstract transformer.

This upgrades PER_SUBGRAPH_SAFE to COMPOSITION_VERIFIED when all
contracts chain, or downgrades to GAP_DETECTED when a gap is found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
    LayerDef,
    LayerKind,
)
from src.dynamo_gap_analysis import (
    GapCategory,
    CrossBreakDependency,
    RiskLevel,
    _detect_cross_break_deps,
    _detect_non_monotonic_at_break,
    SHAPE_ALTERING_OPS,
    _is_shape_altering,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Shape environment representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShapeEnv:
    """Symbolic shape environment mapping tensor names to shape tuples.

    Shapes may contain concrete ints or symbolic strings (e.g. "batch").
    """
    shapes: Dict[str, Tuple] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Tuple]:
        return self.shapes.get(name)

    def set(self, name: str, shape: Tuple) -> None:
        self.shapes[name] = shape

    def names(self) -> Set[str]:
        return set(self.shapes.keys())

    def compatible_with(self, other: "ShapeEnv") -> bool:
        """Check if self's shapes are compatible with other's expectations."""
        for name in other.shapes:
            if name in self.shapes:
                s1 = self.shapes[name]
                s2 = other.shapes[name]
                if len(s1) != len(s2):
                    return False
                for d1, d2 in zip(s1, s2):
                    if isinstance(d1, int) and isinstance(d2, int):
                        if d1 != d2:
                            return False
                    # symbolic dims are compatible by default
        return True

    def __repr__(self) -> str:
        return f"ShapeEnv({self.shapes})"


# ═══════════════════════════════════════════════════════════════════════════════
# Monotonicity constraints
# ═══════════════════════════════════════════════════════════════════════════════

class MonotonicityKind(Enum):
    """Types of monotonicity constraints on shape transformations."""
    MONOTONE = auto()       # output dims ≥ input dims (or proportional)
    ANTI_MONOTONE = auto()  # output dims ≤ input dims
    IDENTITY = auto()       # output dims = input dims
    NON_MONOTONIC = auto()  # no monotonicity guarantee


@dataclass
class MonotonicityConstraint:
    """Constraint on how a dimension changes across a transformation."""
    tensor_name: str
    dim_index: int
    kind: MonotonicityKind
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Environment assumptions and subgraph contracts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnvironmentAssumption:
    """Assumption about the shape environment at a program point.

    Captures what shapes are expected as input, what shapes are
    produced as output, and any monotonicity constraints between them.
    """
    input_shapes: ShapeEnv = field(default_factory=ShapeEnv)
    output_shapes: ShapeEnv = field(default_factory=ShapeEnv)
    monotonicity: List[MonotonicityConstraint] = field(default_factory=list)

    def is_identity(self) -> bool:
        """True if input shapes equal output shapes (no transformation)."""
        if self.input_shapes.shapes.keys() != self.output_shapes.shapes.keys():
            return False
        for name in self.input_shapes.shapes:
            if self.input_shapes.shapes[name] != self.output_shapes.shapes.get(name):
                return False
        return True


@dataclass
class SubgraphContract:
    """Contract for a single Dynamo subgraph.

    The contract specifies:
      - precondition: shape environment expected at subgraph entry
      - postcondition: shape environment guaranteed at subgraph exit
      - environment_assumption: the inter-break transformer assumption
      - detected gap categories from non-monotonic pattern analysis
    """
    subgraph_index: int
    precondition: ShapeEnv
    postcondition: ShapeEnv
    environment_assumption: EnvironmentAssumption
    gap_categories: List[GapCategory] = field(default_factory=list)
    shape_altering_ops: List[str] = field(default_factory=list)
    is_safe: bool = True

    def pretty(self) -> str:
        lines = [f"SubgraphContract(index={self.subgraph_index})"]
        lines.append(f"  Pre:  {self.precondition}")
        lines.append(f"  Post: {self.postcondition}")
        lines.append(f"  Safe: {self.is_safe}")
        if self.gap_categories:
            lines.append(f"  Gaps: {[g.value for g in self.gap_categories]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Inter-break abstract transformer
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerKind(Enum):
    """Kind of abstract transformer for inter-break Python code."""
    IDENTITY = auto()           # shapes pass through unchanged
    CONSERVATIVE = auto()       # unknown transformation, assume worst case
    RESHAPE = auto()            # detected reshape pattern
    DIMENSION_ROUTING = auto()  # detected dynamic routing
    ACCUMULATOR = auto()        # detected accumulator pattern


@dataclass
class InterBreakTransformer:
    """Abstract transformer modeling Python code between graph breaks.

    Conservative default: identity transformer (shapes pass through).
    When specific patterns are detected (reshape, routing, accumulator),
    a more precise transformer is used.
    """
    kind: TransformerKind
    break_index: int
    input_env: ShapeEnv = field(default_factory=ShapeEnv)
    output_env: ShapeEnv = field(default_factory=ShapeEnv)
    detected_patterns: List[str] = field(default_factory=list)
    preserves_rank: bool = True
    preserves_batch_dim: bool = True

    def apply(self, env: ShapeEnv) -> ShapeEnv:
        """Apply this transformer to a shape environment."""
        if self.kind == TransformerKind.IDENTITY:
            return ShapeEnv(shapes=dict(env.shapes))

        if self.kind == TransformerKind.CONSERVATIVE:
            # Conservative: mark all shapes as symbolic (unknown)
            result = ShapeEnv()
            for name, shape in env.shapes.items():
                new_shape = tuple(
                    f"_unknown_{i}" if isinstance(d, int) else d
                    for i, d in enumerate(shape)
                )
                result.set(name, new_shape)
            return result

        if self.kind == TransformerKind.RESHAPE:
            result = ShapeEnv(shapes=dict(env.shapes))
            # If we have explicit output env, use it
            if self.output_env.shapes:
                result = ShapeEnv(shapes=dict(self.output_env.shapes))
            return result

        if self.kind == TransformerKind.DIMENSION_ROUTING:
            result = ShapeEnv()
            for name, shape in env.shapes.items():
                if len(shape) > 0 and self.preserves_batch_dim:
                    new_shape = (shape[0],) + tuple(
                        f"_routed_{i}" for i in range(1, len(shape))
                    )
                else:
                    new_shape = tuple(
                        f"_routed_{i}" for i in range(len(shape))
                    )
                result.set(name, new_shape)
            return result

        if self.kind == TransformerKind.ACCUMULATOR:
            result = ShapeEnv()
            for name, shape in env.shapes.items():
                new_shape = tuple(
                    f"_accum_{i}" if isinstance(d, int) and i > 0 else d
                    for i, d in enumerate(shape)
                )
                result.set(name, new_shape)
            return result

        return ShapeEnv(shapes=dict(env.shapes))


# ═══════════════════════════════════════════════════════════════════════════════
# Composition verdict
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionVerdict(Enum):
    """Verdict for thread-modular composition verification."""
    COMPOSITION_VERIFIED = auto()  # all contracts chain correctly
    GAP_DETECTED = auto()          # gaps found in contract chain
    MONOLITHIC_SAFE = auto()       # single subgraph, no breaks


@dataclass
class GapDetail:
    """Details about a specific gap detected in composition."""
    break_index: int
    category: GapCategory
    source_subgraph: int
    target_subgraph: int
    description: str
    risk: RiskLevel = RiskLevel.MEDIUM
    affected_tensors: List[str] = field(default_factory=list)


@dataclass
class CompositionResult:
    """Result of thread-modular composition verification."""
    verdict: CompositionVerdict
    num_subgraphs: int
    contracts: List[SubgraphContract] = field(default_factory=list)
    transformers: List[InterBreakTransformer] = field(default_factory=list)
    gaps: List[GapDetail] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.name,
            "num_subgraphs": self.num_subgraphs,
            "num_contracts": len(self.contracts),
            "num_gaps": len(self.gaps),
            "gaps": [
                {
                    "break_index": g.break_index,
                    "category": g.category.value,
                    "source_subgraph": g.source_subgraph,
                    "target_subgraph": g.target_subgraph,
                    "description": g.description,
                    "risk": g.risk.value,
                    "affected_tensors": g.affected_tensors,
                }
                for g in self.gaps
            ],
            "details": self.verification_details,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Contract inference
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_shape_from_layer(layer: LayerDef, input_shape: Tuple) -> Optional[Tuple]:
    """Infer output shape from a layer definition and input shape."""
    if layer.kind == LayerKind.LINEAR:
        if layer.out_features is not None and len(input_shape) >= 1:
            return input_shape[:-1] + (layer.out_features,)
    elif layer.kind in (LayerKind.RELU, LayerKind.DROPOUT, LayerKind.IDENTITY):
        return input_shape
    elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.LAYERNORM):
        return input_shape
    elif layer.kind == LayerKind.CONV2D:
        if layer.out_channels is not None and len(input_shape) == 4:
            return (input_shape[0], layer.out_channels, "H_out", "W_out")
    elif layer.kind == LayerKind.FLATTEN:
        if len(input_shape) >= 2:
            return (input_shape[0], "flattened")
    elif layer.kind == LayerKind.EMBEDDING:
        if layer.embedding_dim is not None:
            return input_shape + (layer.embedding_dim,)
    return None


def _infer_op_output_shape(step: ComputationStep, input_shapes: Dict[str, Tuple]) -> Optional[Tuple]:
    """Infer output shape from an operation step."""
    if not step.inputs:
        return None

    first_input = step.inputs[0]
    first_shape = input_shapes.get(first_input)

    if step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                   OpKind.DETACH, OpKind.CONTIGUOUS):
        return first_shape

    if step.op == OpKind.RESHAPE:
        target = step.params.get("target_shape")
        if target:
            return tuple(target)
        return None

    if step.op == OpKind.FLATTEN:
        if first_shape and len(first_shape) >= 2:
            return (first_shape[0], "flattened")
        return None

    if step.op == OpKind.TRANSPOSE:
        if first_shape and len(first_shape) >= 2:
            dims = list(first_shape)
            d0 = step.params.get("dim0", -2)
            d1 = step.params.get("dim1", -1)
            if d0 < 0:
                d0 = len(dims) + d0
            if d1 < 0:
                d1 = len(dims) + d1
            if 0 <= d0 < len(dims) and 0 <= d1 < len(dims):
                dims[d0], dims[d1] = dims[d1], dims[d0]
                return tuple(dims)
        return None

    if step.op == OpKind.MATMUL:
        if len(step.inputs) >= 2:
            shape_b = input_shapes.get(step.inputs[1])
            if first_shape and shape_b:
                if len(first_shape) >= 2 and len(shape_b) >= 2:
                    return first_shape[:-1] + (shape_b[-1],)
        return None

    if step.op in (OpKind.ADD, OpKind.MULTIPLY):
        return first_shape

    if step.op == OpKind.CAT:
        return first_shape  # conservative: same rank

    if step.op in (OpKind.MEAN_REDUCE, OpKind.SUM_REDUCE):
        if first_shape:
            dim = step.params.get("dim")
            if dim is not None and isinstance(dim, int):
                dims = list(first_shape)
                if dim < 0:
                    dim = len(dims) + dim
                if 0 <= dim < len(dims):
                    dims.pop(dim)
                    return tuple(dims) if dims else (1,)
            return ("reduced",)
        return None

    return first_shape


def infer_contract(
    subgraph: ComputationGraph,
    subgraph_index: int,
    input_shapes: Optional[Dict[str, Tuple]] = None,
) -> SubgraphContract:
    """Infer a SubgraphContract from a computation graph.

    Propagates shapes through the subgraph to determine pre/postconditions.
    """
    pre = ShapeEnv()
    post = ShapeEnv()
    shape_altering = []
    gap_cats: List[GapCategory] = []

    # Build precondition from input names
    current_shapes: Dict[str, Tuple] = {}
    for inp_name in subgraph.input_names:
        if input_shapes and inp_name in input_shapes:
            shape = input_shapes[inp_name]
        else:
            shape = ("batch", "dim")  # symbolic default
        pre.set(inp_name, shape)
        current_shapes[inp_name] = shape

    # Forward propagation through steps
    has_reshape = False
    has_dynamic_routing = False
    has_accumulator = False

    for step in subgraph.steps:
        if _is_shape_altering(step):
            shape_altering.append(step.op.name)

        if step.op == OpKind.RESHAPE:
            has_reshape = True
        if step.op in (OpKind.SUBSCRIPT, OpKind.WHERE):
            has_dynamic_routing = True

        # Try to infer output shape
        out_shape = None
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = subgraph.layers.get(step.layer_ref)
            if layer and step.inputs:
                inp_shape = current_shapes.get(step.inputs[0])
                if inp_shape:
                    out_shape = _infer_shape_from_layer(layer, inp_shape)
        else:
            out_shape = _infer_op_output_shape(step, current_shapes)

        if out_shape is None and step.inputs:
            out_shape = current_shapes.get(step.inputs[0])

        if out_shape is not None:
            current_shapes[step.output] = out_shape

    # Build postcondition from output names
    for out_name in subgraph.output_names:
        if out_name in current_shapes:
            post.set(out_name, current_shapes[out_name])
        else:
            post.set(out_name, ("batch", "unknown"))

    # Detect gap categories
    if has_reshape:
        gap_cats.append(GapCategory.CONDITIONAL_RESHAPE)
    if has_dynamic_routing:
        gap_cats.append(GapCategory.DYNAMIC_ROUTING)

    # Infer monotonicity
    mono_constraints = []
    for name in pre.shapes:
        if name in post.shapes:
            pre_shape = pre.shapes[name]
            post_shape = post.shapes[name]
            if pre_shape == post_shape:
                kind = MonotonicityKind.IDENTITY
            elif len(pre_shape) == len(post_shape):
                kind = MonotonicityKind.MONOTONE
            else:
                kind = MonotonicityKind.NON_MONOTONIC
            for i in range(min(len(pre_shape), len(post_shape))):
                mono_constraints.append(MonotonicityConstraint(
                    tensor_name=name,
                    dim_index=i,
                    kind=kind,
                    description=f"dim {i}: {pre_shape[i] if i < len(pre_shape) else '?'} -> {post_shape[i] if i < len(post_shape) else '?'}",
                ))

    env_assumption = EnvironmentAssumption(
        input_shapes=pre,
        output_shapes=post,
        monotonicity=mono_constraints,
    )

    return SubgraphContract(
        subgraph_index=subgraph_index,
        precondition=pre,
        postcondition=post,
        environment_assumption=env_assumption,
        gap_categories=gap_cats,
        shape_altering_ops=shape_altering,
        is_safe=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Non-monotonic pattern detection
# ═══════════════════════════════════════════════════════════════════════════════

class NonMonotonicPattern(Enum):
    """Types of non-monotonic patterns that break composition."""
    SHAPE_INVERSION = auto()      # reshape inverts dimension order
    DIMENSION_ROUTING = auto()    # dynamic expert/routing selection
    ACCUMULATOR = auto()          # growing dimension across breaks
    CONDITIONAL_RESHAPE = auto()  # reshape depends on runtime values
    DATA_DEPENDENT_DIM = auto()   # dimension selected by tensor values


def detect_non_monotonic_patterns(
    sg_before: ComputationGraph,
    sg_after: ComputationGraph,
    contract_before: SubgraphContract,
    contract_after: SubgraphContract,
) -> List[Tuple[NonMonotonicPattern, str]]:
    """Detect non-monotonic patterns between adjacent subgraphs."""
    patterns: List[Tuple[NonMonotonicPattern, str]] = []

    if not sg_before.steps or not sg_after.steps:
        return patterns

    before_ops = {s.op for s in sg_before.steps}
    after_ops = {s.op for s in sg_after.steps}

    # Shape inversion: reshape in one, inverse reshape in other
    reshape_ops = {OpKind.RESHAPE, OpKind.FLATTEN}
    if before_ops & reshape_ops and after_ops & reshape_ops:
        patterns.append((
            NonMonotonicPattern.SHAPE_INVERSION,
            "Reshape in both adjacent subgraphs — shape inversion possible. "
            "Output shape of subgraph may be incompatible after inverse reshape.",
        ))

    # Dimension routing: subscript/where used for routing
    routing_ops = {OpKind.SUBSCRIPT, OpKind.WHERE}
    if before_ops & routing_ops or after_ops & routing_ops:
        patterns.append((
            NonMonotonicPattern.DIMENSION_ROUTING,
            "Dynamic indexing/routing detected — batch dimension may vary "
            "between subgraphs based on runtime tensor values.",
        ))

    # Accumulator: cat/stack in one subgraph, consumption in the next
    accum_ops = {OpKind.CAT, OpKind.STACK}
    if before_ops & accum_ops:
        patterns.append((
            NonMonotonicPattern.ACCUMULATOR,
            "Concatenation/stacking in preceding subgraph — accumulated "
            "dimension may violate constraints in next subgraph.",
        ))

    # Conditional reshape: reshape with symbolic/dynamic target
    for step in sg_before.steps:
        if step.op == OpKind.RESHAPE:
            target = step.params.get("target_shape", ())
            if any(isinstance(d, str) for d in target):
                patterns.append((
                    NonMonotonicPattern.CONDITIONAL_RESHAPE,
                    f"Reshape with symbolic target {target} — output shape "
                    f"depends on runtime values.",
                ))
                break

    # Data-dependent dimension: topk, argmax etc.
    data_dep_ops = {OpKind.SUBSCRIPT}
    if before_ops & data_dep_ops and any(
        s.op in (OpKind.MATMUL, OpKind.LAYER_CALL) for s in sg_after.steps
    ):
        patterns.append((
            NonMonotonicPattern.DATA_DEPENDENT_DIM,
            "Data-dependent dimension selection feeds into shape-sensitive "
            "operation in next subgraph.",
        ))

    # Check postcondition/precondition rank mismatch
    for name in contract_before.postcondition.shapes:
        post_shape = contract_before.postcondition.shapes[name]
        for pre_name in contract_after.precondition.shapes:
            pre_shape = contract_after.precondition.shapes[pre_name]
            if len(post_shape) != len(pre_shape):
                patterns.append((
                    NonMonotonicPattern.SHAPE_INVERSION,
                    f"Rank mismatch: post({name})={len(post_shape)}D vs "
                    f"pre({pre_name})={len(pre_shape)}D — non-monotonic.",
                ))
                break

    return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# Inter-break transformer inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_inter_break_transformer(
    sg_before: ComputationGraph,
    sg_after: ComputationGraph,
    contract_before: SubgraphContract,
    contract_after: SubgraphContract,
    break_index: int,
) -> InterBreakTransformer:
    """Infer the inter-break abstract transformer between two subgraphs.

    Analyzes the relationship between sg_before's outputs and sg_after's
    inputs to determine the most precise transformer.
    """
    prev_outputs = set(sg_before.output_names)
    next_inputs = set(sg_after.input_names)

    # Check if outputs map directly to inputs (identity)
    outputs_feed_inputs = next_inputs.issubset(prev_outputs)
    has_external_inputs = bool(next_inputs - prev_outputs)

    detected_patterns: List[str] = []

    # Check for non-monotonic patterns
    nm_patterns = detect_non_monotonic_patterns(
        sg_before, sg_after, contract_before, contract_after
    )

    if nm_patterns:
        for pat, desc in nm_patterns:
            detected_patterns.append(f"{pat.name}: {desc}")

    # Determine transformer kind
    if has_external_inputs:
        kind = TransformerKind.CONSERVATIVE
        detected_patterns.append(
            "External inputs detected — conservative transformer."
        )
    elif nm_patterns:
        # Check specific patterns
        nm_types = {p[0] for p in nm_patterns}
        if NonMonotonicPattern.DIMENSION_ROUTING in nm_types:
            kind = TransformerKind.DIMENSION_ROUTING
        elif NonMonotonicPattern.ACCUMULATOR in nm_types:
            kind = TransformerKind.ACCUMULATOR
        elif NonMonotonicPattern.SHAPE_INVERSION in nm_types:
            kind = TransformerKind.RESHAPE
        else:
            kind = TransformerKind.CONSERVATIVE
    elif outputs_feed_inputs:
        kind = TransformerKind.IDENTITY
    else:
        kind = TransformerKind.IDENTITY

    return InterBreakTransformer(
        kind=kind,
        break_index=break_index,
        input_env=contract_before.postcondition,
        output_env=contract_after.precondition,
        detected_patterns=detected_patterns,
        preserves_rank=(kind == TransformerKind.IDENTITY),
        preserves_batch_dim=(kind != TransformerKind.DIMENSION_ROUTING),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Composition soundness checking
# ═══════════════════════════════════════════════════════════════════════════════

def check_contract_chain(
    contract_i: SubgraphContract,
    contract_j: SubgraphContract,
    transformer: InterBreakTransformer,
) -> Tuple[bool, List[GapDetail]]:
    """Check that post(G_i) ∘ T_inter ⊆ pre(G_{i+1}).

    Returns (sound, gaps) where sound is True if the chain is valid.
    """
    gaps: List[GapDetail] = []

    # Apply transformer to postcondition of contract_i
    transformed_env = transformer.apply(contract_i.postcondition)

    # Check compatibility with precondition of contract_j
    pre_j = contract_j.precondition

    # If transformer is conservative, we can't guarantee compatibility
    if transformer.kind == TransformerKind.CONSERVATIVE:
        gaps.append(GapDetail(
            break_index=transformer.break_index,
            category=GapCategory.SHAPE_PROPAGATION,
            source_subgraph=contract_i.subgraph_index,
            target_subgraph=contract_j.subgraph_index,
            description=(
                "Conservative transformer applied — inter-break Python code "
                "may modify shapes in ways not captured by per-subgraph "
                "verification."
            ),
            risk=RiskLevel.HIGH,
            affected_tensors=list(pre_j.names()),
        ))
        return False, gaps

    # Check rank compatibility
    for name in pre_j.shapes:
        transformed_shape = transformed_env.get(name)
        pre_shape = pre_j.shapes[name]

        if transformed_shape is None:
            # Input expected by next subgraph not produced by transformer
            gaps.append(GapDetail(
                break_index=transformer.break_index,
                category=GapCategory.SHAPE_PROPAGATION,
                source_subgraph=contract_i.subgraph_index,
                target_subgraph=contract_j.subgraph_index,
                description=(
                    f"Tensor '{name}' expected by subgraph "
                    f"{contract_j.subgraph_index} but not produced by "
                    f"transformer from subgraph {contract_i.subgraph_index}."
                ),
                risk=RiskLevel.HIGH,
                affected_tensors=[name],
            ))
            continue

        # Rank check
        if len(transformed_shape) != len(pre_shape):
            gaps.append(GapDetail(
                break_index=transformer.break_index,
                category=GapCategory.NON_MONOTONIC,
                source_subgraph=contract_i.subgraph_index,
                target_subgraph=contract_j.subgraph_index,
                description=(
                    f"Rank mismatch for '{name}': transformed has "
                    f"{len(transformed_shape)} dims, precondition expects "
                    f"{len(pre_shape)} dims."
                ),
                risk=RiskLevel.HIGH,
                affected_tensors=[name],
            ))
            continue

        # Dimension compatibility check
        for dim_idx, (t_dim, p_dim) in enumerate(
            zip(transformed_shape, pre_shape)
        ):
            if isinstance(t_dim, int) and isinstance(p_dim, int):
                if t_dim != p_dim:
                    gaps.append(GapDetail(
                        break_index=transformer.break_index,
                        category=GapCategory.CONSTRAINT_CHAIN,
                        source_subgraph=contract_i.subgraph_index,
                        target_subgraph=contract_j.subgraph_index,
                        description=(
                            f"Dimension {dim_idx} mismatch for '{name}': "
                            f"transformed={t_dim}, expected={p_dim}."
                        ),
                        risk=RiskLevel.HIGH,
                        affected_tensors=[name],
                    ))

    # Check for non-monotonic patterns in the transformer
    if transformer.detected_patterns:
        for pattern_desc in transformer.detected_patterns:
            cat = GapCategory.NON_MONOTONIC
            if "routing" in pattern_desc.lower():
                cat = GapCategory.DYNAMIC_ROUTING
            elif "accumul" in pattern_desc.lower():
                cat = GapCategory.ACCUMULATOR
            elif "reshape" in pattern_desc.lower():
                cat = GapCategory.CONDITIONAL_RESHAPE
            elif "data-dependent" in pattern_desc.lower():
                cat = GapCategory.DATA_DEPENDENT_DIM

            if transformer.kind != TransformerKind.IDENTITY:
                gaps.append(GapDetail(
                    break_index=transformer.break_index,
                    category=cat,
                    source_subgraph=contract_i.subgraph_index,
                    target_subgraph=contract_j.subgraph_index,
                    description=pattern_desc,
                    risk=RiskLevel.MEDIUM,
                    affected_tensors=[],
                ))

    return len(gaps) == 0, gaps


# ═══════════════════════════════════════════════════════════════════════════════
# ThreadModularVerifier — main class
# ═══════════════════════════════════════════════════════════════════════════════

class ThreadModularVerifier:
    """Thread-modular verifier for TorchDynamo graph-break composition.

    Each subgraph is treated as a "thread" with pre/postconditions.
    Inter-break Python code is modeled as an abstract transformer.
    The verifier checks that:

        ∀ i: post(G_i) ∘ T_inter_i ⊆ pre(G_{i+1})

    Parameters
    ----------
    subgraphs : list of ComputationGraph
        The individual subgraphs from Dynamo extraction.
    input_shapes : dict, optional
        Known input shapes for the first subgraph.
    """

    def __init__(
        self,
        subgraphs: List[ComputationGraph],
        input_shapes: Optional[Dict[str, Tuple]] = None,
    ) -> None:
        self.subgraphs = subgraphs
        self.input_shapes = input_shapes or {}
        self._contracts: List[SubgraphContract] = []
        self._transformers: List[InterBreakTransformer] = []

    def _infer_contracts(self) -> None:
        """Infer contracts for each subgraph."""
        self._contracts = []
        current_shapes = dict(self.input_shapes)

        for i, sg in enumerate(self.subgraphs):
            contract = infer_contract(sg, i, current_shapes)
            self._contracts.append(contract)

            # Update shapes for next subgraph from postcondition
            current_shapes = dict(contract.postcondition.shapes)

    def _infer_transformers(self) -> None:
        """Infer inter-break transformers between adjacent subgraphs."""
        self._transformers = []
        for i in range(len(self.subgraphs) - 1):
            t = infer_inter_break_transformer(
                self.subgraphs[i],
                self.subgraphs[i + 1],
                self._contracts[i],
                self._contracts[i + 1],
                break_index=i,
            )
            self._transformers.append(t)

    def verify(self) -> CompositionResult:
        """Run thread-modular composition verification.

        Returns a CompositionResult with verdict:
          - MONOLITHIC_SAFE if there's only one subgraph
          - COMPOSITION_VERIFIED if all contracts chain correctly
          - GAP_DETECTED if any gap is found
        """
        if len(self.subgraphs) <= 1:
            # Single subgraph — monolithic
            contracts = []
            if self.subgraphs:
                contracts = [infer_contract(
                    self.subgraphs[0], 0, self.input_shapes
                )]
            return CompositionResult(
                verdict=CompositionVerdict.MONOLITHIC_SAFE,
                num_subgraphs=len(self.subgraphs),
                contracts=contracts,
                verification_details={"reason": "single_subgraph"},
            )

        # Step 1: Infer contracts
        self._infer_contracts()

        # Step 2: Infer transformers
        self._infer_transformers()

        # Step 3: Check composition soundness
        all_gaps: List[GapDetail] = []
        for i in range(len(self._transformers)):
            sound, gaps = check_contract_chain(
                self._contracts[i],
                self._contracts[i + 1],
                self._transformers[i],
            )
            all_gaps.extend(gaps)

        # Step 4: Additional cross-break dependency analysis
        cross_deps = _detect_cross_break_deps(self.subgraphs)
        for dep in cross_deps:
            if dep.dependency_type == "transitive":
                all_gaps.append(GapDetail(
                    break_index=dep.source_subgraph,
                    category=GapCategory.CONSTRAINT_CHAIN,
                    source_subgraph=dep.source_subgraph,
                    target_subgraph=dep.target_subgraph,
                    description=dep.description,
                    risk=RiskLevel.HIGH,
                    affected_tensors=[dep.tensor_name],
                ))
            elif dep.dependency_type == "implicit":
                all_gaps.append(GapDetail(
                    break_index=max(0, dep.target_subgraph - 1),
                    category=GapCategory.DYNAMIC_ROUTING,
                    source_subgraph=dep.source_subgraph,
                    target_subgraph=dep.target_subgraph,
                    description=dep.description,
                    risk=RiskLevel.HIGH,
                    affected_tensors=[dep.tensor_name],
                ))

        # Determine verdict
        if all_gaps:
            verdict = CompositionVerdict.GAP_DETECTED
        else:
            verdict = CompositionVerdict.COMPOSITION_VERIFIED

        return CompositionResult(
            verdict=verdict,
            num_subgraphs=len(self.subgraphs),
            contracts=self._contracts,
            transformers=self._transformers,
            gaps=all_gaps,
            verification_details={
                "num_breaks": len(self.subgraphs) - 1,
                "num_identity_transformers": sum(
                    1 for t in self._transformers
                    if t.kind == TransformerKind.IDENTITY
                ),
                "num_conservative_transformers": sum(
                    1 for t in self._transformers
                    if t.kind == TransformerKind.CONSERVATIVE
                ),
                "gap_categories": list(set(
                    g.category.value for g in all_gaps
                )),
            },
        )

    @property
    def contracts(self) -> List[SubgraphContract]:
        return self._contracts

    @property
    def transformers(self) -> List[InterBreakTransformer]:
        return self._transformers


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: verify_thread_modular
# ═══════════════════════════════════════════════════════════════════════════════

def verify_thread_modular(
    subgraphs: List[ComputationGraph],
    input_shapes: Optional[Dict[str, Tuple]] = None,
) -> CompositionResult:
    """Convenience function for thread-modular verification."""
    verifier = ThreadModularVerifier(subgraphs, input_shapes)
    return verifier.verify()
