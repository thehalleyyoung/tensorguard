"""
TorchDynamo PER_SUBGRAPH_SAFE gap analysis.

Analyzes what cross-break shape dependencies might be missed when
subgraphs are verified independently under PER_SUBGRAPH_SAFE semantics.

Documents the auto backend selection algorithm and the three backends'
soundness relationships (AST vs fx vs dynamo).

Backend Soundness Relationships
-------------------------------
AST backend:
    - Parses Python source statically; no runtime.
    - Sound for the fragment it covers (linear, conv, reshape, etc.)
      but incomplete: cannot resolve dynamic control flow, closures,
      or runtime-computed shapes.
    - Most sound for static code: sees all syntactic branches.
    - Misses: runtime conditionals, closures, dynamically-computed shapes.

FX backend (torch.fx.symbolic_trace):
    - Traces a single forward pass symbolically.
    - Sound within a single trace, but misses data-dependent branches
      (if/else on tensor values) entirely — those paths are never seen.
    - Cannot handle graph breaks; raises on unsupported ops.
    - Misses: data-dependent branches, graph breaks, dynamic dispatch.

Dynamo backend (torch._dynamo):
    - Captures *all* reachable subgraphs, including across graph breaks.
    - Each subgraph is individually sound (same as FX within a subgraph).
    - Produces partial graphs but handles dynamic control flow.
    - Gap: cross-break shape dependencies may be invisible when
      subgraphs are verified independently (PER_SUBGRAPH_SAFE).
    - Misses: cross-subgraph constraint propagation, intermediate Python
      code between graph breaks, non-monotonic constraint patterns that
      span multiple subgraphs.

Auto Backend Selection Algorithm
--------------------------------
The auto selector maximises *coverage* first, then *soundness*:
  1. If TorchDynamo is available and the model is traceable → dynamo.
     Rationale: covers the most execution paths (including graph breaks).
  2. Else if torch.fx can trace the model → fx.
     Rationale: still captures the full forward graph for simple models.
  3. Else fall back to AST analysis of Python source.
     Rationale: always available; no runtime dependency.

Within each backend the verifier is *sound* for the fragment it covers:
a reported violation is a genuine constraint failure.  The gap lies in
*completeness*: operations or paths not captured cannot be checked.

PER_SUBGRAPH_SAFE False Negative Categories
--------------------------------------------
When a model is classified PER_SUBGRAPH_SAFE, shape errors may still
occur at runtime.  The following categories of false negatives exist:

1. **Non-monotonic constraint patterns**: A constraint that holds in
   subgraph A and subgraph C individually but fails when A's output
   feeds through intermediate Python code (subgraph B) that inverts
   or non-monotonically transforms the shape.
   Example: Transformer attention where seq_len is padded between
   graph breaks — each subgraph sees valid shapes, but the padding
   introduces a mismatch.

2. **Dynamic routing shape dependencies**: MoE (Mixture of Experts)
   models where expert selection in one subgraph determines the
   batch dimension seen by the next subgraph.  Per-subgraph
   verification assumes the batch dimension is unconstrained.

3. **Accumulator patterns**: Recurrent or iterative architectures
   where a shape accumulates across graph breaks (e.g. growing
   sequence length).  Each subgraph sees a valid fixed-size tensor,
   but the composed sequence violates downstream constraints.

4. **Conditional reshape chains**: A reshape in subgraph A produces
   shape [B, N*H] which is consumed by subgraph C as [B, N, H].
   If intermediate code alters N, the per-subgraph analysis cannot
   detect the inconsistency.

5. **Data-dependent dimension selection**: When a dimension is selected
   based on tensor values (e.g. ``topk`` output used as an index),
   each subgraph verifies with symbolic dimensions, but the actual
   runtime value may violate constraints in a downstream subgraph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Risk levels
# ═══════════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class GapCategory(Enum):
    """Classification of shape information lost at a graph break."""
    SHAPE_PROPAGATION = "shape_propagation"
    CONSTRAINT_CHAIN = "constraint_chain"
    DYNAMIC_ROUTING = "dynamic_routing"
    NON_MONOTONIC = "non_monotonic"
    ACCUMULATOR = "accumulator"
    CONDITIONAL_RESHAPE = "conditional_reshape"
    DATA_DEPENDENT_DIM = "data_dependent_dim"


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-break dependency representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossBreakDependency:
    """A detected cross-break shape dependency."""
    source_subgraph: int
    target_subgraph: int
    tensor_name: str
    dependency_type: str  # "direct", "transitive", "implicit"
    description: str = ""


@dataclass
class MissedDependency:
    """A potentially missed dependency (best-effort detection)."""
    subgraph_index: int
    tensor_name: str
    reason: str
    severity: str = "unknown"  # "low", "medium", "high"


@dataclass
class GraphBreakGap:
    """Classification of what shape information is lost at a graph break.

    Each graph break severs the constraint propagation chain between
    the preceding and following subgraphs.  This dataclass records
    *what kind* of information is lost.
    """
    break_index: int
    category: GapCategory
    lost_constraints: List[str] = field(default_factory=list)
    false_negative_example: str = ""
    risk: RiskLevel = RiskLevel.MEDIUM


@dataclass
class FalseNegativeScenario:
    """A concrete example of a false negative that PER_SUBGRAPH_SAFE could miss."""
    name: str
    category: GapCategory
    description: str
    subgraph_a_constraint: str
    subgraph_b_constraint: str
    combined_failure: str
    is_non_monotonic: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Gap analysis result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GapAnalysisResult:
    """Result of PER_SUBGRAPH_SAFE gap analysis."""
    num_graph_breaks: int
    num_subgraphs: int
    cross_break_dependencies: List[CrossBreakDependency] = field(default_factory=list)
    missed_dependencies: List[MissedDependency] = field(default_factory=list)
    break_gaps: List[GraphBreakGap] = field(default_factory=list)
    false_negative_scenarios: List[FalseNegativeScenario] = field(default_factory=list)
    risk_assessment: RiskLevel = RiskLevel.LOW
    composition_semantics: str = "MONOLITHIC_SAFE"
    backend: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_graph_breaks": self.num_graph_breaks,
            "num_subgraphs": self.num_subgraphs,
            "cross_break_dependencies": [
                {
                    "source_subgraph": d.source_subgraph,
                    "target_subgraph": d.target_subgraph,
                    "tensor_name": d.tensor_name,
                    "dependency_type": d.dependency_type,
                    "description": d.description,
                }
                for d in self.cross_break_dependencies
            ],
            "missed_dependencies": [
                {
                    "subgraph_index": m.subgraph_index,
                    "tensor_name": m.tensor_name,
                    "reason": m.reason,
                    "severity": m.severity,
                }
                for m in self.missed_dependencies
            ],
            "break_gaps": [
                {
                    "break_index": g.break_index,
                    "category": g.category.value,
                    "lost_constraints": g.lost_constraints,
                    "false_negative_example": g.false_negative_example,
                    "risk": g.risk.value,
                }
                for g in self.break_gaps
            ],
            "false_negative_scenarios": [
                {
                    "name": s.name,
                    "category": s.category.value,
                    "description": s.description,
                    "subgraph_a_constraint": s.subgraph_a_constraint,
                    "subgraph_b_constraint": s.subgraph_b_constraint,
                    "combined_failure": s.combined_failure,
                    "is_non_monotonic": s.is_non_monotonic,
                }
                for s in self.false_negative_scenarios
            ],
            "risk_assessment": self.risk_assessment.value,
            "composition_semantics": self.composition_semantics,
            "backend": self.backend,
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Shape-altering operation detection
# ═══════════════════════════════════════════════════════════════════════════════

SHAPE_ALTERING_OPS = {
    OpKind.RESHAPE, OpKind.FLATTEN, OpKind.TRANSPOSE,
    OpKind.MATMUL, OpKind.CAT, OpKind.LAYER_CALL,
}


def _is_shape_altering(step: ComputationStep) -> bool:
    """Check if a computation step alters tensor shape."""
    if step.op in SHAPE_ALTERING_OPS:
        return True
    # Check params for reshape-like operations
    if step.params.get("target_shape") or step.params.get("dim"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Subgraph analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_subgraph_io(
    graph: ComputationGraph,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Extract input names, output names, and tensor producer map."""
    produced_by: Dict[str, str] = {}
    for step in graph.steps:
        produced_by[step.output] = step.op.name
    return list(graph.input_names), list(graph.output_names), produced_by


def _detect_cross_break_deps(
    subgraphs: List[ComputationGraph],
) -> List[CrossBreakDependency]:
    """Detect cross-break shape dependencies between subgraphs."""
    deps: List[CrossBreakDependency] = []
    if len(subgraphs) <= 1:
        return deps

    # Build producer map: tensor_name -> (subgraph_index, op_name)
    produced_by: Dict[str, Tuple[int, str]] = {}
    for sg_idx, sg in enumerate(subgraphs):
        for step in sg.steps:
            produced_by[step.output] = (sg_idx, step.op.name)

    for sg_idx, sg in enumerate(subgraphs):
        if sg_idx == 0:
            continue
        for step in sg.steps:
            for inp in step.inputs:
                producer_info = produced_by.get(inp)
                if producer_info is None:
                    continue
                prod_sg, prod_op = producer_info
                if prod_sg < sg_idx - 1:
                    deps.append(CrossBreakDependency(
                        source_subgraph=prod_sg,
                        target_subgraph=sg_idx,
                        tensor_name=inp,
                        dependency_type="transitive",
                        description=(
                            f"Tensor '{inp}' produced by subgraph {prod_sg} "
                            f"({prod_op}) consumed by subgraph {sg_idx}, "
                            f"skipping {sg_idx - prod_sg - 1} intermediate subgraph(s)."
                        ),
                    ))
                elif prod_sg == sg_idx - 1:
                    deps.append(CrossBreakDependency(
                        source_subgraph=prod_sg,
                        target_subgraph=sg_idx,
                        tensor_name=inp,
                        dependency_type="direct",
                        description=(
                            f"Tensor '{inp}' flows directly from subgraph "
                            f"{prod_sg} to subgraph {sg_idx}."
                        ),
                    ))

    # Check for implicit dependencies (inputs not from any subgraph)
    all_outputs: set = set()
    for sg in subgraphs:
        for step in sg.steps:
            all_outputs.add(step.output)
        all_outputs.update(sg.output_names)

    for sg_idx in range(1, len(subgraphs)):
        for inp_name in subgraphs[sg_idx].input_names:
            if inp_name not in all_outputs:
                deps.append(CrossBreakDependency(
                    source_subgraph=-1,
                    target_subgraph=sg_idx,
                    tensor_name=inp_name,
                    dependency_type="implicit",
                    description=(
                        f"Subgraph {sg_idx} input '{inp_name}' not found in "
                        f"any previous subgraph output — may come from "
                        f"intermediate Python code between graph breaks."
                    ),
                ))

    return deps


def _detect_missed_deps(
    subgraphs: List[ComputationGraph],
    cross_deps: List[CrossBreakDependency],
) -> List[MissedDependency]:
    """Best-effort detection of missed dependencies."""
    missed: List[MissedDependency] = []
    if len(subgraphs) <= 1:
        return missed

    # 1. Check for shape-altering ops at subgraph boundaries
    for sg_idx, sg in enumerate(subgraphs):
        if not sg.steps:
            continue
        last_step = sg.steps[-1]
        if _is_shape_altering(last_step):
            missed.append(MissedDependency(
                subgraph_index=sg_idx,
                tensor_name=last_step.output,
                reason=(
                    f"Shape-altering op '{last_step.op.name}' at end of "
                    f"subgraph {sg_idx} — output shape may constrain "
                    f"subsequent subgraph inputs but is verified in isolation."
                ),
                severity="medium",
            ))

        first_step = sg.steps[0]
        if sg_idx > 0 and _is_shape_altering(first_step):
            missed.append(MissedDependency(
                subgraph_index=sg_idx,
                tensor_name=first_step.inputs[0] if first_step.inputs else "unknown",
                reason=(
                    f"Shape-altering op '{first_step.op.name}' at start of "
                    f"subgraph {sg_idx} — input shape assumptions may not "
                    f"match the previous subgraph's actual output shape."
                ),
                severity="medium",
            ))

    # 2. Check for unverified input constraints
    for sg_idx in range(1, len(subgraphs)):
        prev_output_set = set(subgraphs[sg_idx - 1].output_names)
        for inp_name in subgraphs[sg_idx].input_names:
            if inp_name not in prev_output_set:
                missed.append(MissedDependency(
                    subgraph_index=sg_idx,
                    tensor_name=inp_name,
                    reason=(
                        f"Input '{inp_name}' of subgraph {sg_idx} does not "
                        f"come from the immediately preceding subgraph's "
                        f"outputs — shape may be modified by Python code "
                        f"between graph breaks."
                    ),
                    severity="high",
                ))

    # 3. Detect dynamic dimension dependencies
    for sg_idx, sg in enumerate(subgraphs):
        for step in sg.steps:
            if step.op == OpKind.RESHAPE:
                target_shape = step.params.get("target_shape", ())
                if any(isinstance(d, str) for d in target_shape):
                    missed.append(MissedDependency(
                        subgraph_index=sg_idx,
                        tensor_name=step.output,
                        reason=(
                            f"Reshape with symbolic dimensions {target_shape} "
                            f"in subgraph {sg_idx} — dimension values may "
                            f"depend on cross-break computation."
                        ),
                        severity="high",
                    ))

    return missed


# ═══════════════════════════════════════════════════════════════════════════════
# Gap classification per graph break
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_break_gaps(
    subgraphs: List[ComputationGraph],
    cross_deps: List[CrossBreakDependency],
) -> List[GraphBreakGap]:
    """Classify what shape information is lost at each graph break.

    For each break point (boundary between subgraph N and N+1), determines
    the category of information loss and provides a concrete false negative
    example showing what per-subgraph verification could miss.
    """
    gaps: List[GraphBreakGap] = []
    if len(subgraphs) <= 1:
        return gaps

    for break_idx in range(len(subgraphs) - 1):
        sg_before = subgraphs[break_idx]
        sg_after = subgraphs[break_idx + 1]

        lost: List[str] = []

        # Check for shape-altering ops at the boundary
        has_boundary_reshape = False
        if sg_before.steps:
            last = sg_before.steps[-1]
            if _is_shape_altering(last):
                lost.append(
                    f"Output shape from {last.op.name} not propagated to next subgraph"
                )
                has_boundary_reshape = True
        if sg_after.steps:
            first = sg_after.steps[0]
            if _is_shape_altering(first):
                lost.append(
                    f"Input shape assumption for {first.op.name} not validated against previous output"
                )
                has_boundary_reshape = True

        # Check for transitive deps through this break
        transitive_here = [
            d for d in cross_deps
            if d.dependency_type == "transitive"
            and d.source_subgraph <= break_idx < d.target_subgraph
        ]
        implicit_here = [
            d for d in cross_deps
            if d.dependency_type == "implicit"
            and d.target_subgraph == break_idx + 1
        ]

        # Classify the gap
        if transitive_here:
            for d in transitive_here:
                lost.append(f"Transitive dependency on '{d.tensor_name}' spans this break")
            category = GapCategory.CONSTRAINT_CHAIN
            example = (
                f"Subgraph {break_idx} outputs shape [B, N] which constrains "
                f"subgraph {break_idx + 1}'s input, but intermediate Python code "
                f"may modify N — per-subgraph verification cannot detect this."
            )
            risk = RiskLevel.HIGH
        elif implicit_here:
            for d in implicit_here:
                lost.append(f"Implicit input '{d.tensor_name}' from external code")
            category = GapCategory.DYNAMIC_ROUTING
            example = (
                f"Subgraph {break_idx + 1} receives input from Python code "
                f"between breaks — shape may depend on runtime values."
            )
            risk = RiskLevel.HIGH
        elif has_boundary_reshape:
            category = GapCategory.CONDITIONAL_RESHAPE
            example = (
                f"Reshape at break boundary: output of subgraph {break_idx} "
                f"is reshaped before entering subgraph {break_idx + 1}. "
                f"If reshape target uses dynamic dims, constraint is lost."
            )
            risk = RiskLevel.MEDIUM
        else:
            category = GapCategory.SHAPE_PROPAGATION
            example = (
                f"Standard shape propagation break: subgraph {break_idx} "
                f"output shape is not cross-checked with subgraph "
                f"{break_idx + 1} input assumptions."
            )
            risk = RiskLevel.LOW

        # Detect non-monotonic patterns
        has_non_monotonic = _detect_non_monotonic_at_break(sg_before, sg_after)
        if has_non_monotonic:
            category = GapCategory.NON_MONOTONIC
            lost.append("Non-monotonic constraint pattern detected across break")
            example = (
                f"Non-monotonic pattern: subgraph {break_idx} applies a "
                f"constraint f(x) and subgraph {break_idx + 1} applies g(x) "
                f"where g(f(x)) is not monotonic — per-subgraph analysis "
                f"verifies f and g independently but misses the composition."
            )
            risk = RiskLevel.HIGH

        gaps.append(GraphBreakGap(
            break_index=break_idx,
            category=category,
            lost_constraints=lost,
            false_negative_example=example,
            risk=risk,
        ))

    return gaps


def _detect_non_monotonic_at_break(
    sg_before: ComputationGraph,
    sg_after: ComputationGraph,
) -> bool:
    """Detect non-monotonic constraint patterns across a graph break.

    A non-monotonic pattern occurs when the output constraint of one
    subgraph and the input constraint of the next are individually
    satisfiable but their composition is not.  Classic example:
    subgraph A outputs [B, N*H], subgraph B expects [B, N, H] —
    if N changes between breaks, the decomposition is invalid.
    """
    if not sg_before.steps or not sg_after.steps:
        return False

    before_ops = {s.op for s in sg_before.steps}
    after_ops = {s.op for s in sg_after.steps}

    reshape_ops = {OpKind.RESHAPE, OpKind.FLATTEN}
    if before_ops & reshape_ops and after_ops & reshape_ops:
        return True

    # Matmul dimension dependency across break
    if OpKind.MATMUL in before_ops and OpKind.MATMUL in after_ops:
        before_last = sg_before.steps[-1]
        after_first = sg_after.steps[0]
        if before_last.op == OpKind.MATMUL and after_first.op == OpKind.MATMUL:
            return True

    return False


def _generate_false_negative_scenarios(
    subgraphs: List[ComputationGraph],
    break_gaps: List[GraphBreakGap],
) -> List[FalseNegativeScenario]:
    """Generate concrete false negative scenarios based on detected gaps.

    Each scenario describes a specific situation where PER_SUBGRAPH_SAFE
    would report safe but a runtime shape error could occur.
    """
    scenarios: List[FalseNegativeScenario] = []

    for gap in break_gaps:
        if gap.category == GapCategory.NON_MONOTONIC:
            scenarios.append(FalseNegativeScenario(
                name=f"non_monotonic_break_{gap.break_index}",
                category=GapCategory.NON_MONOTONIC,
                description=(
                    f"Non-monotonic constraint composition at break {gap.break_index}: "
                    f"subgraphs verify independently but composed constraint fails."
                ),
                subgraph_a_constraint="output shape [B, N*H] where N>0, H>0",
                subgraph_b_constraint="input shape [B, M, H] where M>0, H>0",
                combined_failure=(
                    "If N*H != M*H (i.e. N != M due to intermediate padding/masking), "
                    "reshape fails at runtime despite each subgraph being independently safe."
                ),
                is_non_monotonic=True,
            ))
        elif gap.category == GapCategory.DYNAMIC_ROUTING:
            scenarios.append(FalseNegativeScenario(
                name=f"dynamic_routing_break_{gap.break_index}",
                category=GapCategory.DYNAMIC_ROUTING,
                description=(
                    f"Dynamic routing at break {gap.break_index}: expert/path selection "
                    f"in one subgraph determines tensor shapes in the next."
                ),
                subgraph_a_constraint="output: [B, D] routed to K experts",
                subgraph_b_constraint="input: [B/K, D] per expert",
                combined_failure=(
                    "If K does not evenly divide B, the per-expert batch dimension "
                    "is invalid — but each subgraph sees valid shapes independently."
                ),
                is_non_monotonic=False,
            ))
        elif gap.category == GapCategory.CONSTRAINT_CHAIN:
            scenarios.append(FalseNegativeScenario(
                name=f"constraint_chain_break_{gap.break_index}",
                category=GapCategory.CONSTRAINT_CHAIN,
                description=(
                    f"Transitive constraint chain broken at break {gap.break_index}."
                ),
                subgraph_a_constraint="output: [B, seq_len, D]",
                subgraph_b_constraint="input: [B, seq_len, D] (assumes same seq_len)",
                combined_failure=(
                    "Intermediate Python truncates/pads seq_len — each subgraph "
                    "verifies with symbolic seq_len but the concrete values differ."
                ),
                is_non_monotonic=False,
            ))
        elif gap.category == GapCategory.CONDITIONAL_RESHAPE:
            scenarios.append(FalseNegativeScenario(
                name=f"conditional_reshape_break_{gap.break_index}",
                category=GapCategory.CONDITIONAL_RESHAPE,
                description=(
                    f"Conditional reshape at break {gap.break_index}."
                ),
                subgraph_a_constraint="output: [B, C*H*W]",
                subgraph_b_constraint="input: [B, C, H, W]",
                combined_failure=(
                    "If C*H*W factorization is ambiguous (multiple valid (C,H,W)), "
                    "per-subgraph verification picks one but runtime uses another."
                ),
                is_non_monotonic=True,
            ))

    return scenarios


def _assess_risk(
    num_breaks: int,
    cross_deps: List[CrossBreakDependency],
    missed_deps: List[MissedDependency],
) -> RiskLevel:
    """Assess the overall risk of PER_SUBGRAPH_SAFE verification."""
    if num_breaks == 0:
        return RiskLevel.LOW

    # Count high-severity missed deps
    high_severity = sum(1 for m in missed_deps if m.severity == "high")
    transitive_deps = sum(
        1 for d in cross_deps if d.dependency_type == "transitive"
    )
    implicit_deps = sum(
        1 for d in cross_deps if d.dependency_type == "implicit"
    )

    if high_severity > 0 or transitive_deps > 0 or implicit_deps > 1:
        return RiskLevel.HIGH
    if len(cross_deps) > 2 or len(missed_deps) > 2 or implicit_deps > 0:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ═══════════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_per_subgraph_safe_gap(
    model: Any = None,
    input_shape: Optional[tuple] = None,
    *,
    graph: Optional[ComputationGraph] = None,
    subgraphs: Optional[List[ComputationGraph]] = None,
) -> GapAnalysisResult:
    """Analyze PER_SUBGRAPH_SAFE gaps for a model or precomputed graph.

    Parameters
    ----------
    model : nn.Module or source string, optional
        The model to analyze. If a string, parsed via AST backend.
    input_shape : tuple, optional
        Input tensor shape for tracing (e.g. ``(1, 3, 224, 224)``).
    graph : ComputationGraph, optional
        Pre-extracted computation graph (skips extraction).
    subgraphs : list of ComputationGraph, optional
        Pre-split subgraphs for direct analysis.

    Returns
    -------
    GapAnalysisResult
    """
    backend = "unknown"

    # If subgraphs provided directly, analyze them
    if subgraphs is not None:
        num_breaks = max(0, len(subgraphs) - 1)
        cross_deps = _detect_cross_break_deps(subgraphs)
        missed_deps = _detect_missed_deps(subgraphs, cross_deps)
        break_gaps = _classify_break_gaps(subgraphs, cross_deps)
        false_neg = _generate_false_negative_scenarios(subgraphs, break_gaps)
        risk = _assess_risk(num_breaks, cross_deps, missed_deps)

        if num_breaks == 0:
            semantics = "MONOLITHIC_SAFE"
        elif cross_deps and any(
            d.dependency_type in ("transitive", "implicit") for d in cross_deps
        ):
            semantics = "UNKNOWN"
        else:
            semantics = "PER_SUBGRAPH_SAFE"

        return GapAnalysisResult(
            num_graph_breaks=num_breaks,
            num_subgraphs=len(subgraphs),
            cross_break_dependencies=cross_deps,
            missed_dependencies=missed_deps,
            break_gaps=break_gaps,
            false_negative_scenarios=false_neg,
            risk_assessment=risk,
            composition_semantics=semantics,
            backend="precomputed",
            details={
                "num_steps_per_subgraph": [len(sg.steps) for sg in subgraphs],
                "num_shape_altering_boundary_ops": sum(
                    1 for m in missed_deps
                    if "Shape-altering" in m.reason
                ),
                "non_monotonic_breaks": sum(
                    1 for g in break_gaps
                    if g.category == GapCategory.NON_MONOTONIC
                ),
            },
        )

    # If graph provided, extract subgraph info from dynamic_features
    if graph is not None:
        num_breaks = graph.dynamic_features.get("graph_breaks", 0)
        num_subgraphs = graph.dynamic_features.get("num_dynamo_subgraphs", 1)
        semantics = graph.dynamic_features.get(
            "composition_semantics", "MONOLITHIC_SAFE"
        )
        has_cross = graph.dynamic_features.get("cross_break_dependencies", False)
        backend = "dynamo" if graph.dynamic_features.get("dynamo_traced") else "fx"

        # Reconstruct approximate subgraphs from the composed graph
        approx_subgraphs = _approximate_subgraphs(graph, num_subgraphs)
        cross_deps = _detect_cross_break_deps(approx_subgraphs)
        missed_deps = _detect_missed_deps(approx_subgraphs, cross_deps)

        if has_cross and not cross_deps:
            cross_deps.append(CrossBreakDependency(
                source_subgraph=0,
                target_subgraph=1,
                tensor_name="<detected-by-dynamo>",
                dependency_type="implicit",
                description="Cross-break dependency flagged by dynamo composer.",
            ))

        break_gaps = _classify_break_gaps(approx_subgraphs, cross_deps)
        false_neg = _generate_false_negative_scenarios(approx_subgraphs, break_gaps)
        risk = _assess_risk(num_breaks, cross_deps, missed_deps)

        return GapAnalysisResult(
            num_graph_breaks=num_breaks,
            num_subgraphs=num_subgraphs,
            cross_break_dependencies=cross_deps,
            missed_dependencies=missed_deps,
            break_gaps=break_gaps,
            false_negative_scenarios=false_neg,
            risk_assessment=risk,
            composition_semantics=semantics,
            backend=backend,
            details={
                "total_steps": len(graph.steps),
                "total_layers": len(graph.layers),
            },
        )

    # Try to extract graph from model
    if model is not None:
        return _analyze_from_model(model, input_shape)

    # No input provided
    return GapAnalysisResult(
        num_graph_breaks=0,
        num_subgraphs=0,
        risk_assessment=RiskLevel.LOW,
        composition_semantics="MONOLITHIC_SAFE",
        backend="none",
        details={"error": "No model, graph, or subgraphs provided"},
    )


def _approximate_subgraphs(
    graph: ComputationGraph, num_subgraphs: int
) -> List[ComputationGraph]:
    """Approximate subgraph split from a composed graph."""
    if num_subgraphs <= 1 or not graph.steps:
        return [graph]

    steps_per_sg = max(1, len(graph.steps) // num_subgraphs)
    subgraphs: List[ComputationGraph] = []

    for i in range(num_subgraphs):
        start = i * steps_per_sg
        end = start + steps_per_sg if i < num_subgraphs - 1 else len(graph.steps)
        sg = ComputationGraph(class_name=graph.class_name)
        sg.steps = list(graph.steps[start:end])

        if i == 0:
            sg.input_names = list(graph.input_names)
        else:
            # Inputs are outputs from previous chunk
            sg.input_names = [graph.steps[start - 1].output] if start > 0 else []

        if i == num_subgraphs - 1:
            sg.output_names = list(graph.output_names)
        else:
            sg.output_names = [graph.steps[end - 1].output] if end <= len(graph.steps) else []

        subgraphs.append(sg)

    return subgraphs


def _analyze_from_model(
    model: Any, input_shape: Optional[tuple]
) -> GapAnalysisResult:
    """Extract graph from model and analyze."""
    # Try dynamo first, then fx, then AST
    graph = None
    backend = "unknown"

    try:
        import torch
        import torch.nn as nn
        if isinstance(model, nn.Module):
            try:
                from src.dynamo_extractor import dynamo_trace_to_graph, HAS_DYNAMO
                if HAS_DYNAMO:
                    example = (torch.randn(*input_shape),) if input_shape else None
                    graph = dynamo_trace_to_graph(model, example_inputs=example)
                    backend = "dynamo"
            except Exception:
                pass

            if graph is None:
                try:
                    from src.fx_extractor import fx_trace_to_graph
                    graph = fx_trace_to_graph(model)
                    backend = "fx"
                except Exception:
                    pass
    except ImportError:
        pass

    if graph is None and isinstance(model, str):
        try:
            from src.model_checker import extract_graph
            graph = extract_graph(model)
            backend = "ast"
        except Exception:
            pass

    if graph is None:
        return GapAnalysisResult(
            num_graph_breaks=0,
            num_subgraphs=0,
            risk_assessment=RiskLevel.LOW,
            composition_semantics="MONOLITHIC_SAFE",
            backend="failed",
            details={"error": "Could not extract graph from model"},
        )

    return analyze_per_subgraph_safe_gap(graph=graph)


# ═══════════════════════════════════════════════════════════════════════════════
# Backend selection documentation
# ═══════════════════════════════════════════════════════════════════════════════

def get_backend_selection_info() -> Dict[str, Any]:
    """Return documentation about the auto backend selection algorithm."""
    return {
        "algorithm": "coverage-first, then soundness",
        "selection_order": [
            {
                "backend": "dynamo",
                "condition": "TorchDynamo available and model traceable",
                "maximizes": "coverage",
                "soundness": "per-subgraph (sound within each subgraph)",
                "completeness": "handles graph breaks, data-dependent control flow",
            },
            {
                "backend": "fx",
                "condition": "torch.fx.symbolic_trace succeeds",
                "maximizes": "soundness",
                "soundness": "monolithic (single trace, no breaks)",
                "completeness": "no graph breaks; fails on dynamic control flow",
            },
            {
                "backend": "ast",
                "condition": "always available (no runtime needed)",
                "maximizes": "availability",
                "soundness": "sound for supported fragment",
                "completeness": "limited to statically analyzable patterns",
            },
        ],
        "soundness_relationships": {
            "ast_vs_fx": (
                "FX covers more operations (runtime tracing) but misses "
                "branches not taken. AST sees all branches but may not "
                "resolve dynamic shapes."
            ),
            "fx_vs_dynamo": (
                "Dynamo subsumes FX: it uses FX internally for each "
                "subgraph but handles graph breaks that FX cannot. "
                "Dynamo's gap is cross-break dependencies."
            ),
            "ast_vs_dynamo": (
                "AST and Dynamo are complementary: AST sees all source "
                "code branches; Dynamo captures actual runtime subgraphs. "
                "Neither is strictly stronger than the other."
            ),
        },
    }
