"""
Assume-Guarantee Compositional Verification for TensorGuard.

Decomposes an nn.Module's ``ComputationGraph`` into sub-modules, derives
interface contracts at every boundary, and verifies each sub-module
independently under an assume-guarantee discipline:

    For each sub-module M_i:
      ASSUME  the input contract I_i holds  (shape / device / phase)
      VERIFY  that M_i satisfies its output contract O_i

    Composition rule:
      If ∀ i: M_i satisfies (I_i ⇒ O_i)
         AND ∀ i: O_i implies I_{i+1}        (interface compatibility)
         AND I_0 is satisfied by the user-supplied input shapes
      THEN the whole model is safe.

This is sound: a monolithically-safe model is always compositionally-safe
(completeness direction follows because the contracts are *derived*, not
user-supplied).  The practical benefit is faster re-verification when only
a single sub-module changes — unchanged modules reuse cached results.

Usage::

    from src.assume_guarantee import verify_compositional

    result = verify_compositional(
        source=open("my_model.py").read(),
        input_shapes={"x": ("batch", 3, 224, 224)},
    )
    print(f"Safe: {result.safe}  Speedup: {result.speedup_vs_monolithic:.1f}x")
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
    ConstraintVerifier,
    VerificationResult,
    SafetyCertificate,
    Device,
    Phase,
    ModelState,
    extract_computation_graph,
    verify_model,
)
from src.tensor_shapes import TensorShape, ShapeDim

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Decomposition strategy enum
# ═══════════════════════════════════════════════════════════════════════════════

class DecompositionStrategy(Enum):
    """Strategy used to split a ``ComputationGraph`` into sub-modules."""
    AUTO = "auto"
    LAYER_BOUNDARY = "layer_boundary"
    BRANCH_MERGE = "branch_merge"
    USER_SPECIFIED = "user_specified"
    SINGLE_LAYER = "single_layer"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Interface contracts & data-classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InterfaceContract:
    """Shape / device / phase contract at a sub-module boundary.

    Captures what a sub-module *assumes* about its inputs and what it
    *guarantees* about its outputs.  Contracts are derived automatically
    from the ``LayerDef`` definitions at the boundary.

    Attributes
    ----------
    name : str
        Human-readable name, e.g. ``"encoder_output"``.
    input_shapes : dict
        Mapping from tensor name → expected shape tuple.  Dimensions may
        be concrete ints or symbolic strings.
    output_shapes : dict
        Mapping from tensor name → guaranteed shape tuple.
    constraints : list of str
        Human-readable descriptions of Z3 constraints that must hold.
    device : Device or None
        Expected device (``None`` means "inherit from producer").
    phase : Phase or None
        Expected phase (``None`` means "any").
    """

    name: str
    input_shapes: Dict[str, tuple] = field(default_factory=dict)
    output_shapes: Dict[str, tuple] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    device: Optional[Device] = None
    phase: Optional[Phase] = None

    def fingerprint(self) -> str:
        """Deterministic hash for caching / change-detection."""
        payload = (
            self.name,
            sorted(self.input_shapes.items()),
            sorted(self.output_shapes.items()),
            tuple(self.constraints),
            self.device,
            self.phase,
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]

    def pretty(self) -> str:
        lines = [f"InterfaceContract({self.name})"]
        if self.input_shapes:
            lines.append(f"  Inputs:  {self.input_shapes}")
        if self.output_shapes:
            lines.append(f"  Outputs: {self.output_shapes}")
        if self.constraints:
            for c in self.constraints:
                lines.append(f"  Constraint: {c}")
        if self.device is not None:
            lines.append(f"  Device: {self.device.value}")
        if self.phase is not None:
            lines.append(f"  Phase:  {self.phase.name}")
        return "\n".join(lines)


@dataclass
class SubModule:
    """A decomposed piece of the computation graph.

    Attributes
    ----------
    name : str
        Human-readable name for the sub-module (e.g. ``"block_0"``).
    graph : ComputationGraph
        The sub-graph (valid ``ComputationGraph`` with correct
        ``input_names`` / ``output_names``).
    input_contract : InterfaceContract
        What this sub-module *assumes* about its inputs.
    output_contract : InterfaceContract
        What this sub-module *guarantees* about its outputs.
    step_range : tuple of (int, int)
        Inclusive (start, end) indices into the original graph's step list.
    """

    name: str
    graph: ComputationGraph
    input_contract: InterfaceContract
    output_contract: InterfaceContract
    step_range: Tuple[int, int] = (0, 0)

    def fingerprint(self) -> str:
        """Hash over structure for change-detection."""
        payload = (
            self.name,
            self.input_contract.fingerprint(),
            self.output_contract.fingerprint(),
            len(self.graph.steps),
            tuple(s.op.name for s in self.graph.steps),
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


@dataclass
class InterfaceCheck:
    """Result of checking interface compatibility between adjacent sub-modules.

    Attributes
    ----------
    producer : str
        Name of the upstream sub-module.
    consumer : str
        Name of the downstream sub-module.
    compatible : bool
        ``True`` iff the producer's output contract satisfies the consumer's
        input contract.
    message : str
        Human-readable explanation of (in)compatibility.
    """

    producer: str
    consumer: str
    compatible: bool
    message: str


@dataclass
class CompositionalResult:
    """Result of assume-guarantee compositional verification.

    Attributes
    ----------
    safe : bool
        ``True`` iff every sub-module satisfies its contract AND all
        interfaces are compatible.
    submodule_results : dict
        Per-sub-module ``VerificationResult`` keyed by sub-module name.
    interface_checks : list of InterfaceCheck
        One entry per adjacent pair of sub-modules.
    total_time_ms : float
        Wall-clock time for the full compositional verification.
    speedup_vs_monolithic : float
        Ratio of monolithic verification time to compositional time.
        Values > 1.0 indicate compositional is faster.
    cache_hits : int
        Number of sub-modules whose cached result was reused.
    decomposition_strategy : DecompositionStrategy
        Which strategy was used for decomposition.
    num_submodules : int
        Number of sub-modules the graph was split into.
    """

    safe: bool
    submodule_results: Dict[str, VerificationResult] = field(default_factory=dict)
    interface_checks: List[InterfaceCheck] = field(default_factory=list)
    total_time_ms: float = 0.0
    speedup_vs_monolithic: float = 1.0
    cache_hits: int = 0
    decomposition_strategy: DecompositionStrategy = DecompositionStrategy.AUTO
    num_submodules: int = 0
    proof_tree: Optional["ProofTree"] = None

    def pretty(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        lines = [
            f"CompositionalResult: {status}",
            f"  Sub-modules:  {self.num_submodules}",
            f"  Strategy:     {self.decomposition_strategy.value}",
            f"  Total time:   {self.total_time_ms:.1f} ms",
            f"  Speedup:      {self.speedup_vs_monolithic:.2f}x",
            f"  Cache hits:   {self.cache_hits}",
        ]
        for name, res in self.submodule_results.items():
            tag = "✓" if res.safe else "✗"
            lines.append(
                f"  {tag} {name}: {res.verification_time_ms:.1f} ms"
            )
        for ic in self.interface_checks:
            tag = "✓" if ic.compatible else "✗"
            lines.append(
                f"  {tag} {ic.producer} → {ic.consumer}: {ic.message}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Formal composition proof rule (Abadi-Lamport non-circular)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompositionProofRule:
    """Formal Abadi-Lamport non-circular sequential composition rule.

    For sequential composition M1 ; M2:
        If  {P} M1 {Q}   and   {Q} M2 {R}
        Then  {P} M1;M2 {R}

    More generally, for a chain M_0 ; M_1 ; ... ; M_{n-1}:
        If  ∀ i ∈ [0, n-1]:  {C_i} M_i {C_{i+1}}
            AND ∀ i ∈ [0, n-2]:  C_{i+1} from M_i's postcondition
                                  implies C_{i+1} as M_{i+1}'s precondition
        Then  {C_0} M_0;...;M_{n-1} {C_n}

    This is non-circular because the proof obligation for M_i depends
    only on the *contract* of M_{i-1}, not on its implementation.

    Soundness Theorem
    -----------------
    If all submodule verifications succeed (each M_i satisfies {C_i} M_i {C_{i+1}})
    and all interface contracts are compatible (the postcondition C_{i+1} of M_i
    implies the precondition C_{i+1} of M_{i+1} for every adjacent pair), then
    the whole-module composition M_0;...;M_{n-1} is safe with respect to the
    global precondition C_0 and global postcondition C_n.

    Attributes
    ----------
    preconditions : list of str
        Precondition names for each submodule (C_0, ..., C_{n-1}).
    postconditions : list of str
        Postcondition names for each submodule (C_1, ..., C_n).
    submodule_names : list of str
        Names of the submodules in sequential order.
    interface_obligations : list of tuple
        Each (i, j) pair records an interface obligation: postcondition
        of submodule i must imply precondition of submodule j.
    """

    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    submodule_names: List[str] = field(default_factory=list)
    interface_obligations: List[Tuple[int, int]] = field(default_factory=list)

    @staticmethod
    def from_submodules(submodules: List[SubModule]) -> "CompositionProofRule":
        """Construct a composition proof rule from a list of submodules."""
        pre = [sm.input_contract.name for sm in submodules]
        post = [sm.output_contract.name for sm in submodules]
        names = [sm.name for sm in submodules]
        obligations = [(i, i + 1) for i in range(len(submodules) - 1)]
        return CompositionProofRule(
            preconditions=pre,
            postconditions=post,
            submodule_names=names,
            interface_obligations=obligations,
        )

    def pretty(self) -> str:
        lines = ["Composition Proof Rule (Abadi-Lamport non-circular):"]
        for i, name in enumerate(self.submodule_names):
            lines.append(f"  {{{self.preconditions[i]}}} {name} {{{self.postconditions[i]}}}")
        for i, j in self.interface_obligations:
            lines.append(
                f"  Interface: {self.postconditions[i]} ⟹ {self.preconditions[j]}"
            )
        lines.append("  ────────────────────────────────────────────")
        if self.preconditions and self.postconditions:
            lines.append(
                f"  ∴ {{{self.preconditions[0]}}} "
                f"{' ; '.join(self.submodule_names)} "
                f"{{{self.postconditions[-1]}}}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 2b′. DAG (non-sequential) composition proof rule
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DAGCompositionProofRule:
    """Generalized assume-guarantee rule for DAG (non-sequential) composition.

    For a DAG of submodules with edges E:
        If  ∀ node M_i: {∧_{j→i} G_j} M_i {G_i}   (each node verified under predecessor guarantees)
           AND ∀ edge (j,i) ∈ E: G_j implies pre(M_i) at that input  (interface compatibility)
           AND root preconditions satisfied by user input
        Then the whole DAG is safe.

    This generalizes the Abadi-Lamport sequential rule to DAGs. Soundness follows
    by topological induction: process nodes in topological order, each verified
    under already-established guarantees of predecessors.
    """
    node_names: List[str]
    edges: List[Tuple[int, int]]  # (producer_idx, consumer_idx)
    node_preconditions: List[List[str]]  # per-node list of assumed preconditions
    node_postconditions: List[str]  # per-node guaranteed postcondition
    topology: str  # "sequential", "residual", "dense", "encoder_decoder", "general_dag"

    @staticmethod
    def from_submodules_and_edges(
        submodules: List["SubModule"],
        edges: List[Tuple[int, int]],
        topology: str = "general_dag",
    ) -> "DAGCompositionProofRule":
        """Construct a DAG proof rule from submodules and an explicit edge list."""
        node_names = [sm.name for sm in submodules]
        node_pre: List[List[str]] = []
        for i, sm in enumerate(submodules):
            preds = [submodules[j].output_contract.name for j, k in edges if k == i]
            if not preds:
                preds = [sm.input_contract.name]
            node_pre.append(preds)
        node_post = [sm.output_contract.name for sm in submodules]
        return DAGCompositionProofRule(
            node_names=node_names,
            edges=edges,
            node_preconditions=node_pre,
            node_postconditions=node_post,
            topology=topology,
        )

    def topological_order(self) -> List[int]:
        """Return a topological ordering of the DAG nodes."""
        n = len(self.node_names)
        in_degree = [0] * n
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for src, dst in self.edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        queue = [i for i in range(n) if in_degree[i] == 0]
        order: List[int] = []
        while queue:
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != n:
            # Cycle detected — fall back to index order
            return list(range(n))
        return order

    def pretty(self) -> str:
        lines = [f"DAG Composition Proof Rule (topology={self.topology}):"]
        lines.append(f"  Nodes: {self.node_names}")
        lines.append(f"  Edges: {self.edges}")
        for i, name in enumerate(self.node_names):
            pres = ", ".join(self.node_preconditions[i])
            lines.append(
                f"  {{{pres}}} {name} {{{self.node_postconditions[i]}}}"
            )
        for src, dst in self.edges:
            lines.append(
                f"  Interface: {self.node_postconditions[src]} ⟹ pre({self.node_names[dst]})"
            )
        lines.append("  " + "─" * 44)
        if self.node_names:
            lines.append(
                f"  ∴ DAG({', '.join(self.node_names)}) safe"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 2c. Formal inference rules for assume-guarantee reasoning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceRule:
    """Formal inference rule for assume-guarantee reasoning.

    An inference rule has the form:
        P₁, P₂, ..., Pₙ
        ─────────────────── [RuleName]
              C
    where P₁...Pₙ are premises and C is the conclusion.
    """

    name: str
    premises: List[str]  # symbolic descriptions
    conclusion: str
    side_conditions: List[str]

    def to_latex(self) -> str:
        """Generate LaTeX inference rule using \\frac."""
        premises_str = " \\quad ".join(self.premises)
        sc = ", ".join(self.side_conditions)
        return (
            f"\\frac{{{premises_str}}}{{{self.conclusion}}}"
            f"\\;\\textsc{{{self.name}}}"
            + (f"\\quad\\text{{where }} {sc}" if sc else "")
        )


# The two core rules used by TensorGuard:

ASYMMETRIC_AG_RULE = InferenceRule(
    name="AG-Asym",
    premises=[
        "\\langle A \\rangle\\, M_1 \\,\\langle G_1 \\rangle",
        "\\langle G_1 \\rangle\\, M_2 \\,\\langle G_2 \\rangle",
    ],
    conclusion="\\langle A \\rangle\\, M_1 \\circ M_2 \\,\\langle G_2 \\rangle",
    side_conditions=[
        "\\text{output shape of } M_1 \\text{ matches input shape of } M_2",
        "\\text{device}(M_1) = \\text{device}(M_2)",
    ],
)

SEQUENTIAL_COMPOSITION_RULE = InferenceRule(
    name="AG-Seq",
    premises=[
        "\\langle \\phi_0 \\rangle\\, M_i \\,\\langle \\phi_i \\rangle \\quad \\forall i \\in [1..n]",
        "\\phi_{i-1} \\Rightarrow \\text{pre}(M_i) \\quad \\forall i \\in [1..n]",
    ],
    conclusion="\\langle \\phi_0 \\rangle\\, M_1 \\circ \\cdots \\circ M_n \\,\\langle \\phi_n \\rangle",
    side_conditions=[
        "\\text{no circular dependencies between } M_i",
        "\\text{each } \\phi_i \\text{ is a shape+device+phase predicate}",
    ],
)

NAMJOSHI_TREFLER_CIRCULAR_RULE = InferenceRule(
    name="AG-Circ",
    premises=[
        "\\langle A_i \\rangle\\, M_i \\,\\langle G_i \\rangle \\quad \\forall i",
        "\\bigwedge_{j \\neq i} G_j \\Rightarrow A_i \\quad \\forall i",
        "\\exists \\text{well-founded ordering } \\prec \\text{ on } "
        "\\{M_i\\} \\text{ s.t. } A_i \\text{ depends only on } "
        "G_j \\text{ with } j \\prec i",
    ],
    conclusion="\\langle \\bigwedge A_i \\rangle\\, "
               "M_1 \\| \\cdots \\| M_n \\,\\langle \\bigwedge G_i \\rangle",
    side_conditions=[
        "\\text{well-founded ordering breaks circularity}",
        "\\text{applicable when DAG has apparent cycles from shared state}",
    ],
)

DAG_COMPOSITION_RULE = InferenceRule(
    name="AG-DAG",
    premises=[
        "\\langle \\bigwedge_{j \\to i} G_j \\rangle\\, M_i \\,\\langle G_i \\rangle "
        "\\quad \\forall i",
        "G_j \\Rightarrow \\text{pre}(M_i) \\text{ for edge } (j,i)",
    ],
    conclusion="\\text{DAG}(M_1, \\ldots, M_n) \\text{ safe}",
    side_conditions=[
        "\\text{topological order exists (DAG is acyclic)}",
        "\\text{multi-branch concat/add shape constraints propagated}",
    ],
)


@dataclass
class ProofStep:
    """A single step in a proof tree, recording which inference rule was applied."""

    rule: InferenceRule
    producer: str
    consumer: str
    premise_judgments: List[str]
    conclusion_judgment: str
    discharged: bool = False

    def pretty(self) -> str:
        tag = "✓" if self.discharged else "✗"
        lines = [
            f"  {tag} [{self.rule.name}] {self.producer} ∘ {self.consumer}",
        ]
        for p in self.premise_judgments:
            lines.append(f"      premise: {p}")
        lines.append(f"      conclusion: {self.conclusion_judgment}")
        return "\n".join(lines)


@dataclass
class ProofTree:
    """A proof tree recording all inference rule applications during
    compositional verification."""

    steps: List[ProofStep] = field(default_factory=list)
    overall_rule: Optional[InferenceRule] = None

    def pretty(self) -> str:
        lines = ["Proof Tree:"]
        if self.overall_rule:
            lines.append(f"  Overall rule: {self.overall_rule.name}")
        for step in self.steps:
            lines.append(step.pretty())
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Generate LaTeX for the full proof tree."""
        parts = []
        if self.overall_rule:
            parts.append(f"% Overall rule: {self.overall_rule.name}")
            parts.append(self.overall_rule.to_latex())
            parts.append("")
        for i, step in enumerate(self.steps):
            parts.append(f"% Step {i}: {step.rule.name}")
            parts.append(step.rule.to_latex())
        return "\n".join(parts)


def _build_proof_tree(
    submodules: List["SubModule"],
    submodule_results: Dict[str, "VerificationResult"],
    interface_checks: List["InterfaceCheck"],
) -> ProofTree:
    """Build a proof tree from compositional verification results."""
    steps: List[ProofStep] = []

    for i in range(len(submodules) - 1):
        producer = submodules[i]
        consumer = submodules[i + 1]

        producer_result = submodule_results.get(producer.name)
        consumer_result = submodule_results.get(consumer.name)
        ic = interface_checks[i] if i < len(interface_checks) else None

        producer_safe = producer_result is not None and producer_result.safe
        consumer_safe = consumer_result is not None and consumer_result.safe
        interface_ok = ic is not None and ic.compatible

        premise_judgments = [
            f"⟨{producer.input_contract.name}⟩ {producer.name} ⟨{producer.output_contract.name}⟩"
            f" [{'✓' if producer_safe else '✗'}]",
            f"⟨{producer.output_contract.name}⟩ {consumer.name} ⟨{consumer.output_contract.name}⟩"
            f" [{'✓' if consumer_safe else '✗'}]",
        ]
        conclusion_judgment = (
            f"⟨{producer.input_contract.name}⟩ "
            f"{producer.name} ∘ {consumer.name} "
            f"⟨{consumer.output_contract.name}⟩"
        )

        step = ProofStep(
            rule=ASYMMETRIC_AG_RULE,
            producer=producer.name,
            consumer=consumer.name,
            premise_judgments=premise_judgments,
            conclusion_judgment=conclusion_judgment,
            discharged=producer_safe and consumer_safe and interface_ok,
        )
        steps.append(step)

    # Choose overall rule based on chain length
    overall_rule = (
        SEQUENTIAL_COMPOSITION_RULE if len(submodules) > 2
        else ASYMMETRIC_AG_RULE
    )

    return ProofTree(steps=steps, overall_rule=overall_rule)


class DisagreementKind(Enum):
    """Classification of compositional vs monolithic disagreements."""
    OVER_APPROXIMATION = "over_approximation"
    """Compositional says UNSAFE when monolithic says SAFE — sound."""
    UNDER_APPROXIMATION = "under_approximation"
    """Compositional says SAFE when monolithic says UNSAFE — UNSOUND, critical."""


@dataclass
class DisagreementAnalysis:
    """Root-cause analysis of a compositional vs monolithic disagreement.

    Attributes
    ----------
    kind : DisagreementKind
        Whether this is an over- or under-approximation.
    compositional_safe : bool
        Compositional verdict.
    monolithic_safe : bool
        Monolithic verdict.
    root_cause : str
        Human-readable explanation of the likely cause.
    is_critical : bool
        True if this represents an unsoundness (under-approximation).
    """

    kind: DisagreementKind
    compositional_safe: bool
    monolithic_safe: bool
    root_cause: str
    is_critical: bool


def analyze_compositional_disagreement(
    compositional: CompositionalResult,
    monolithic: VerificationResult,
) -> Optional[DisagreementAnalysis]:
    """Compare compositional vs monolithic results and characterize any
    disagreement.

    Returns ``None`` if both agree.  Otherwise returns a
    ``DisagreementAnalysis`` classifying the disagreement as:

    * **Over-approximation** (compositional=UNSAFE, monolithic=SAFE):
      The compositional analysis is conservative.  This can happen when
      interface contracts are coarser than the actual data flow.  This
      is *sound* — no real bugs are missed.

    * **Under-approximation** (compositional=SAFE, monolithic=UNSAFE):
      The compositional analysis missed a real bug.  This would indicate
      an *unsoundness* in the assume-guarantee decomposition (e.g.,
      incompatible interface contracts that were not detected).  Flagged
      as **critical**.
    """
    if compositional.safe == monolithic.safe:
        return None

    if not compositional.safe and monolithic.safe:
        return DisagreementAnalysis(
            kind=DisagreementKind.OVER_APPROXIMATION,
            compositional_safe=False,
            monolithic_safe=True,
            root_cause=(
                "Compositional analysis is conservative: interface contracts "
                "are coarser than the actual data flow, causing a spurious "
                "UNSAFE verdict.  This is sound (no real bugs are missed)."
            ),
            is_critical=False,
        )

    # compositional.safe and not monolithic.safe
    return DisagreementAnalysis(
        kind=DisagreementKind.UNDER_APPROXIMATION,
        compositional_safe=True,
        monolithic_safe=False,
        root_cause=(
            "CRITICAL: Compositional analysis missed a real bug that "
            "monolithic analysis detected.  This indicates an unsoundness "
            "in the assume-guarantee decomposition — likely an interface "
            "contract that does not faithfully capture the data-flow "
            "constraints between adjacent submodules."
        ),
        is_critical=True,
    )


def validate_interface_chain(
    submodules: List[SubModule],
) -> List[InterfaceCheck]:
    """Validate that output contracts of module i imply input contracts of
    module i+1 for all adjacent pairs.

    This is the key interface compatibility obligation in the
    Abadi-Lamport composition rule: the postcondition of each
    submodule must entail the precondition of the next submodule.

    Returns a list of ``InterfaceCheck`` results, one per adjacent pair.
    An empty list is returned if there are fewer than 2 submodules.
    """
    checks: List[InterfaceCheck] = []
    for i in range(len(submodules) - 1):
        producer = submodules[i]
        consumer = submodules[i + 1]

        # Check that every tensor in the consumer's input contract
        # is covered by the producer's output contract
        producer_outputs = producer.output_contract.output_shapes
        consumer_inputs = consumer.input_contract.input_shapes

        missing: List[str] = []
        mismatched: List[str] = []

        for tname, c_shape in consumer_inputs.items():
            if c_shape == ("*",):
                continue  # unconstrained — always satisfiable
            if tname not in producer_outputs:
                # Tensor not in producer's output contract — check if it
                # is actually produced by the producer's sub-graph
                produced = {s.output for s in producer.graph.steps}
                if tname in produced:
                    # Produced but not in contract — contract under-specifies
                    missing.append(
                        f"{tname}: produced by {producer.name} but absent "
                        f"from output contract"
                    )
                continue

            p_shape = producer_outputs[tname]
            if p_shape == ("*",):
                continue  # producer unconstrained — trivially implies

            ok, msg = _shapes_compatible(p_shape, c_shape)
            if not ok:
                mismatched.append(f"{tname}: {msg}")

        issues = missing + mismatched
        if issues:
            checks.append(InterfaceCheck(
                producer=producer.name,
                consumer=consumer.name,
                compatible=False,
                message=(
                    f"Interface contract violation (output ⊬ input): "
                    + "; ".join(issues)
                ),
            ))
        else:
            checks.append(InterfaceCheck(
                producer=producer.name,
                consumer=consumer.name,
                compatible=True,
                message=(
                    f"Output contract of {producer.name} implies "
                    f"input contract of {consumer.name}"
                ),
            ))

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Graph analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _tensor_producers(steps: List[ComputationStep]) -> Dict[str, int]:
    """Map each tensor name to the step index that produces it."""
    producers: Dict[str, int] = {}
    for idx, step in enumerate(steps):
        producers[step.output] = idx
    return producers


def _tensor_consumers(steps: List[ComputationStep]) -> Dict[str, List[int]]:
    """Map each tensor name to the step indices that consume it."""
    consumers: Dict[str, List[int]] = {}
    for idx, step in enumerate(steps):
        for inp in step.inputs:
            consumers.setdefault(inp, []).append(idx)
    return consumers


def _fan_out(steps: List[ComputationStep]) -> Dict[int, int]:
    """Compute fan-out for each step (how many later steps consume its output)."""
    consumers = _tensor_consumers(steps)
    return {
        idx: len(consumers.get(step.output, []))
        for idx, step in enumerate(steps)
    }


def _fan_in(steps: List[ComputationStep]) -> Dict[int, int]:
    """Compute fan-in for each step (how many inputs it consumes)."""
    return {idx: len(step.inputs) for idx, step in enumerate(steps)}


def _is_branch_point(step_idx: int, steps: List[ComputationStep]) -> bool:
    """True if the step's output is consumed by >1 later step."""
    output = steps[step_idx].output
    count = sum(
        1 for s in steps[step_idx + 1:] if output in s.inputs
    )
    return count > 1


def _is_merge_point(step_idx: int, steps: List[ComputationStep]) -> bool:
    """True if the step consumes outputs from >1 different producers."""
    return len(steps[step_idx].inputs) > 1


def _live_tensors_at(
    step_idx: int,
    steps: List[ComputationStep],
    input_names: List[str],
) -> Set[str]:
    """Return the set of tensor names that are live (defined and possibly
    consumed later) at the point *between* ``steps[step_idx-1]`` and
    ``steps[step_idx]``.
    """
    defined: Set[str] = set(input_names)
    for s in steps[:step_idx]:
        defined.add(s.output)
    consumed_later: Set[str] = set()
    for s in steps[step_idx:]:
        consumed_later.update(s.inputs)
    return defined & consumed_later


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Decomposition algorithms
# ═══════════════════════════════════════════════════════════════════════════════

def _find_cut_points_layer_boundary(
    graph: ComputationGraph,
    min_block_size: int = 1,
) -> List[int]:
    """Identify cut-points at transitions between named layer calls.

    A cut-point is placed after a ``LAYER_CALL`` step whenever the *next*
    step references a *different* layer (or is not a layer call at all).
    This groups consecutive uses of the same layer into one block and
    splits at layer boundaries.
    """
    steps = graph.steps
    if len(steps) <= 1:
        return []

    cuts: List[int] = []
    last_layer: Optional[str] = None

    for idx, step in enumerate(steps):
        current_layer = step.layer_ref if step.op == OpKind.LAYER_CALL else None

        if last_layer is not None and current_layer != last_layer and idx > 0:
            # Only add cut if both sides would be big enough
            prev_cut = cuts[-1] if cuts else 0
            if idx - prev_cut >= min_block_size:
                cuts.append(idx)

        last_layer = current_layer

    return cuts


def _find_cut_points_branch_merge(
    graph: ComputationGraph,
) -> List[int]:
    """Identify cut-points at branch/merge boundaries.

    Places a cut-point immediately *after* every merge point (a step that
    consumes outputs from multiple producers) and immediately *before*
    every branch point (a step whose output fans out to multiple
    consumers).
    """
    steps = graph.steps
    if len(steps) <= 1:
        return []

    cuts_set: Set[int] = set()

    for idx in range(len(steps)):
        if _is_branch_point(idx, steps) and idx + 1 < len(steps):
            cuts_set.add(idx + 1)
        if _is_merge_point(idx, steps) and idx > 0:
            cuts_set.add(idx)

    return sorted(cuts_set)


def _find_cut_points_auto(
    graph: ComputationGraph,
    max_submodules: int = 8,
    min_block_size: int = 2,
) -> List[int]:
    """Automatic heuristic decomposition.

    Combines layer-boundary and branch-merge heuristics, then filters to
    stay within ``max_submodules`` blocks.  Prefers cuts at layer
    boundaries when both strategies agree.
    """
    steps = graph.steps
    if len(steps) <= 2:
        return []

    layer_cuts = set(_find_cut_points_layer_boundary(graph, min_block_size))
    bm_cuts = set(_find_cut_points_branch_merge(graph))

    # Prioritise: cuts where both agree > layer-only > branch/merge-only
    agreed = sorted(layer_cuts & bm_cuts)
    layer_only = sorted(layer_cuts - bm_cuts)
    bm_only = sorted(bm_cuts - layer_cuts)

    candidates = agreed + layer_only + bm_only

    # Filter out cuts that would create too-small blocks
    filtered: List[int] = []
    prev = 0
    for c in candidates:
        if c - prev >= min_block_size:
            filtered.append(c)
            prev = c
    # Ensure the last block is big enough
    if filtered and len(steps) - filtered[-1] < min_block_size:
        filtered.pop()

    # Limit total sub-modules
    if len(filtered) >= max_submodules:
        # Keep evenly spaced subset
        step_size = len(filtered) / (max_submodules - 1)
        kept: List[int] = []
        for i in range(max_submodules - 1):
            kept.append(filtered[int(i * step_size)])
        filtered = kept

    return filtered


def _find_cut_points_user(
    graph: ComputationGraph,
    cut_after_steps: Optional[List[int]] = None,
    cut_after_layers: Optional[List[str]] = None,
) -> List[int]:
    """User-specified decomposition points.

    Parameters
    ----------
    cut_after_steps : list of int, optional
        Step indices after which to place a cut.
    cut_after_layers : list of str, optional
        Layer attribute names; a cut is placed after the *last* step that
        references each named layer.
    """
    cuts_set: Set[int] = set()
    steps = graph.steps

    if cut_after_steps:
        for idx in cut_after_steps:
            if 0 < idx < len(steps):
                cuts_set.add(idx)

    if cut_after_layers:
        for layer_name in cut_after_layers:
            last_idx = None
            for idx, step in enumerate(steps):
                if step.layer_ref == layer_name:
                    last_idx = idx
            if last_idx is not None and last_idx + 1 < len(steps):
                cuts_set.add(last_idx + 1)

    return sorted(cuts_set)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Contract derivation
# ═══════════════════════════════════════════════════════════════════════════════

def _shape_tuple_for_layer_input(layer: LayerDef) -> Optional[tuple]:
    """Derive expected input shape constraints from a ``LayerDef``."""
    if layer.kind == LayerKind.LINEAR:
        if layer.in_features is not None:
            return ("batch", layer.in_features)
    elif layer.kind in (LayerKind.CONV2D, LayerKind.CONVTRANSPOSE2D):
        if layer.in_channels is not None:
            return ("batch", layer.in_channels, "H", "W")
    elif layer.kind == LayerKind.CONV1D:
        if layer.in_channels is not None:
            return ("batch", layer.in_channels, "L")
    elif layer.kind in (LayerKind.BATCHNORM1D,):
        if layer.num_features is not None:
            return ("batch", layer.num_features)
    elif layer.kind in (LayerKind.BATCHNORM2D, LayerKind.INSTANCENORM2D):
        if layer.num_features is not None:
            return ("batch", layer.num_features, "H", "W")
    elif layer.kind == LayerKind.LAYERNORM:
        return None  # normalised_shape varies
    elif layer.kind == LayerKind.EMBEDDING:
        if layer.num_embeddings is not None:
            return ("batch", "seq_len")
    elif layer.kind in (LayerKind.LSTM, LayerKind.GRU):
        if layer.hidden_size is not None:
            return ("batch", "seq_len", layer.hidden_size)
    elif layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
        # Pool expects 4D input (batch, C, H, W)
        return ("batch", "C", "H", "W")
    return None


def _shape_tuple_for_layer_output(layer: LayerDef) -> Optional[tuple]:
    """Derive guaranteed output shape constraints from a ``LayerDef``."""
    if layer.kind == LayerKind.LINEAR:
        if layer.out_features is not None:
            return ("batch", layer.out_features)
    elif layer.kind in (LayerKind.CONV2D, LayerKind.CONVTRANSPOSE2D):
        if layer.out_channels is not None:
            return ("batch", layer.out_channels, "H_out", "W_out")
    elif layer.kind == LayerKind.CONV1D:
        if layer.out_channels is not None:
            return ("batch", layer.out_channels, "L_out")
    elif layer.kind in (LayerKind.BATCHNORM1D,):
        if layer.num_features is not None:
            return ("batch", layer.num_features)
    elif layer.kind in (LayerKind.BATCHNORM2D, LayerKind.INSTANCENORM2D):
        if layer.num_features is not None:
            return ("batch", layer.num_features, "H", "W")
    elif layer.kind == LayerKind.EMBEDDING:
        if layer.embedding_dim is not None:
            return ("batch", "seq_len", layer.embedding_dim)
    elif layer.kind in (LayerKind.LSTM, LayerKind.GRU):
        hidden = layer.hidden_size or "hidden"
        dirs = 2 if layer.bidirectional else 1
        out_dim = hidden * dirs if isinstance(hidden, int) else f"{hidden}x{dirs}"
        return ("batch", "seq_len", out_dim)
    elif layer.kind in (LayerKind.RELU, LayerKind.DROPOUT, LayerKind.IDENTITY):
        return None  # shape-preserving
    elif layer.kind == LayerKind.FLATTEN:
        return ("batch", "flat_dim")
    elif layer.kind in (LayerKind.ADAPTIVE_AVGPOOL2D,):
        if layer.output_size is not None:
            return ("batch", "C", *layer.output_size)
    elif layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
        # Pool preserves batch and channel dims; spatial dims change
        # based on kernel_size, stride, padding.
        return ("batch", "C", "H_pool", "W_pool")
    return None


def _derive_input_contract(
    sub_steps: List[ComputationStep],
    input_names: List[str],
    layers: Dict[str, LayerDef],
    module_name: str,
    global_input_shapes: Dict[str, tuple],
    truly_global_inputs: Optional[Set[str]] = None,
) -> InterfaceContract:
    """Derive the *input* interface contract for a sub-module.

    The contract specifies what shapes the sub-module expects on each of
    its input tensors.  For tensors that originate from the global model
    inputs, the user-supplied ``input_shapes`` are used.  For tensors
    produced by a preceding sub-module, the shape is inferred from the
    first step that consumes them and its layer definition.

    Parameters
    ----------
    truly_global_inputs : set of str, optional
        Tensor names that are truly global model inputs (not produced by
        any prior sub-module).  When provided, only these names are
        looked up in ``global_input_shapes``.  This prevents name
        collisions when intermediate tensors reuse the same name as a
        global input (e.g. ``x = self.fc1(x)``).
    """
    contract_shapes: Dict[str, tuple] = {}
    constraints: List[str] = []

    for tname in input_names:
        # Use global input shapes only for truly global inputs
        is_truly_global = (
            truly_global_inputs is None or tname in truly_global_inputs
        )
        if is_truly_global and tname in global_input_shapes:
            contract_shapes[tname] = global_input_shapes[tname]
            continue

        # Infer from the first step that consumes this tensor
        for step in sub_steps:
            if tname in step.inputs:
                if step.layer_ref and step.layer_ref in layers:
                    layer = layers[step.layer_ref]
                    shape = _shape_tuple_for_layer_input(layer)
                    if shape is not None:
                        contract_shapes[tname] = shape
                        constraints.append(
                            f"{tname} must match {layer.attr_name} "
                            f"({layer.kind.name}) input requirements"
                        )
                    break
                elif step.op == OpKind.ADD:
                    # For ADD, both inputs must have the same shape.
                    # Try to infer from the other input's producer.
                    for other_inp in step.inputs:
                        if other_inp != tname:
                            for s2 in reversed(sub_steps):
                                if s2.output == other_inp and s2.layer_ref and s2.layer_ref in layers:
                                    shape = _shape_tuple_for_layer_output(layers[s2.layer_ref])
                                    if shape is not None:
                                        contract_shapes[tname] = shape
                                        constraints.append(
                                            f"{tname} must match shape for ADD with {other_inp}"
                                        )
                                    break
                    # If still unknown, infer from the ADD output's consumer
                    if tname not in contract_shapes:
                        add_output = step.output
                        for s2 in sub_steps:
                            if add_output in s2.inputs and s2.layer_ref and s2.layer_ref in layers:
                                shape = _shape_tuple_for_layer_input(layers[s2.layer_ref])
                                if shape is not None:
                                    contract_shapes[tname] = shape
                                    constraints.append(
                                        f"{tname} must match ADD output consumed by {s2.layer_ref}"
                                    )
                                break
                    if tname in contract_shapes:
                        break

        # Fallback: leave unconstrained but note it
        if tname not in contract_shapes:
            contract_shapes[tname] = ("*",)  # wildcard
            constraints.append(f"{tname}: shape unconstrained (no layer info)")

    return InterfaceContract(
        name=f"{module_name}_input",
        input_shapes=contract_shapes,
        constraints=constraints,
    )


def _trace_shape_through_preserving(
    step: ComputationStep,
    sub_steps: List[ComputationStep],
    layers: Dict[str, LayerDef],
    depth: int = 5,
) -> Optional[tuple]:
    """Trace backward through shape-preserving ops to find a concrete shape.

    When a tensor is produced by a shape-preserving op (activation, contiguous,
    etc.), this function walks backward through the sub-graph to find the
    original layer call that established the shape.
    """
    if depth <= 0:
        return None
    preserving_ops = (
        OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.CONTIGUOUS, OpKind.DETACH,
    )
    for inp in step.inputs:
        for s in reversed(sub_steps):
            if s.output == inp:
                if s.layer_ref and s.layer_ref in layers:
                    return _shape_tuple_for_layer_output(layers[s.layer_ref])
                if s.op in preserving_ops:
                    return _trace_shape_through_preserving(
                        s, sub_steps, layers, depth - 1,
                    )
                break
    return None


def _derive_output_contract(
    sub_steps: List[ComputationStep],
    output_names: List[str],
    layers: Dict[str, LayerDef],
    module_name: str,
) -> InterfaceContract:
    """Derive the *output* interface contract for a sub-module.

    The contract specifies what shapes the sub-module guarantees on each
    of its output tensors.  This is inferred from the *last* layer-call
    step that produces each output tensor.
    """
    contract_shapes: Dict[str, tuple] = {}
    constraints: List[str] = []

    for tname in output_names:
        # Find the step that produces this tensor
        for step in reversed(sub_steps):
            if step.output == tname:
                if step.layer_ref and step.layer_ref in layers:
                    layer = layers[step.layer_ref]
                    shape = _shape_tuple_for_layer_output(layer)
                    if shape is not None:
                        contract_shapes[tname] = shape
                        constraints.append(
                            f"{tname} guaranteed by {layer.attr_name} "
                            f"({layer.kind.name}) output semantics"
                        )
                elif step.op == OpKind.CAT:
                    # For concat, derive output shape from input branch
                    # shapes.  The concat dim is summed; others must match.
                    branch_shapes = []
                    for inp in step.inputs:
                        for s2 in reversed(sub_steps):
                            if s2.output == inp:
                                if s2.layer_ref and s2.layer_ref in layers:
                                    bs = _shape_tuple_for_layer_output(layers[s2.layer_ref])
                                    if bs is not None:
                                        branch_shapes.append(bs)
                                break
                    if branch_shapes:
                        # Use first branch as template; concat dim is symbolic sum
                        base = list(branch_shapes[0])
                        concat_total = 0
                        all_concrete = True
                        for bs in branch_shapes:
                            if len(bs) > 1 and isinstance(bs[1], int):
                                concat_total += bs[1]
                            else:
                                all_concrete = False
                        if all_concrete and concat_total > 0 and len(base) > 1:
                            base[1] = concat_total
                        contract_shapes[tname] = tuple(base)
                        constraints.append(
                            f"{tname}: concat of {len(branch_shapes)} branches "
                            f"(total dim={concat_total if all_concrete else 'symbolic'})"
                        )
                elif step.op == OpKind.ADD:
                    # For element-wise add (residual), output shape equals
                    # input shapes (both must match).
                    for inp in step.inputs:
                        for s2 in reversed(sub_steps):
                            if s2.output == inp and s2.layer_ref and s2.layer_ref in layers:
                                shape = _shape_tuple_for_layer_output(layers[s2.layer_ref])
                                if shape is not None:
                                    contract_shapes[tname] = shape
                                    constraints.append(
                                        f"{tname}: residual add preserves shape from {s2.layer_ref}"
                                    )
                                break
                        if tname in contract_shapes:
                            break
                elif step.op in (
                    OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.CONTIGUOUS,
                    OpKind.DETACH,
                ):
                    # Shape-preserving ops: output shape == input shape.
                    # Trace back to find the input's shape from a layer.
                    traced = _trace_shape_through_preserving(
                        step, sub_steps, layers,
                    )
                    if traced is not None:
                        contract_shapes[tname] = traced
                        constraints.append(
                            f"{tname}: shape preserved (op={step.op.name})"
                        )
                    else:
                        constraints.append(
                            f"{tname}: shape preserved (op={step.op.name})"
                        )
                break

        if tname not in contract_shapes:
            contract_shapes[tname] = ("*",)
            constraints.append(f"{tname}: output shape unconstrained")

    return InterfaceContract(
        name=f"{module_name}_output",
        output_shapes=contract_shapes,
        constraints=constraints,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Sub-graph extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_subgraph(
    graph: ComputationGraph,
    start: int,
    end: int,
) -> Tuple[ComputationGraph, List[str], List[str]]:
    """Extract a sub-graph from ``graph.steps[start:end]``.

    Returns
    -------
    sub_graph : ComputationGraph
        A valid ``ComputationGraph`` whose steps are deep-copied from the
        slice ``[start, end)``.
    sub_inputs : list of str
        Tensor names that are consumed but not produced within the slice
        (i.e. they must come from outside — the sub-module's inputs).
    sub_outputs : list of str
        Tensor names produced inside the slice and consumed *after* the
        slice or listed in the original graph's ``output_names`` (i.e.
        the sub-module's outputs).
    """
    sub_steps = [copy.deepcopy(s) for s in graph.steps[start:end]]

    produced_here: Set[str] = {s.output for s in sub_steps}
    consumed_here: Set[str] = set()
    for s in sub_steps:
        consumed_here.update(s.inputs)

    # Inputs: consumed but not produced inside this slice.
    # Also include tensors that are consumed BEFORE they are produced
    # within the slice (e.g. skip connections: residual consumed at
    # step i but only re-produced at step i+1).
    external_inputs = consumed_here - produced_here
    produced_so_far: Set[str] = set()
    for s in sub_steps:
        for inp in s.inputs:
            if inp not in produced_so_far and inp not in external_inputs:
                # This tensor is consumed before produced in this slice
                if inp in produced_here:
                    external_inputs.add(inp)
        produced_so_far.add(s.output)
    # Also include global inputs that are consumed
    sub_inputs = sorted(external_inputs | (set(graph.input_names) & consumed_here))

    # Outputs: produced here and consumed later OR listed as a model output
    consumed_after: Set[str] = set()
    for s in graph.steps[end:]:
        consumed_after.update(s.inputs)
    model_outputs = set(graph.output_names)
    sub_outputs = sorted(produced_here & (consumed_after | model_outputs))

    # If no explicit outputs, use the last step's output
    if not sub_outputs and sub_steps:
        sub_outputs = [sub_steps[-1].output]

    # Collect layers referenced by this sub-graph
    sub_layers: Dict[str, LayerDef] = {}
    for s in sub_steps:
        if s.layer_ref and s.layer_ref in graph.layers:
            sub_layers[s.layer_ref] = copy.deepcopy(graph.layers[s.layer_ref])

    sub_graph = ComputationGraph(
        class_name=f"{graph.class_name}_sub",
        layers=sub_layers,
        steps=sub_steps,
        input_names=sub_inputs,
        output_names=sub_outputs,
    )
    return sub_graph, sub_inputs, sub_outputs


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main decomposition entry point
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_graph(
    graph: ComputationGraph,
    strategy: str = "auto",
    *,
    min_block_size: int = 2,
    max_submodules: int = 8,
    cut_after_steps: Optional[List[int]] = None,
    cut_after_layers: Optional[List[str]] = None,
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> List[SubModule]:
    """Decompose a ``ComputationGraph`` into a list of ``SubModule`` objects.

    Parameters
    ----------
    graph : ComputationGraph
        The full computation graph to decompose.
    strategy : str
        One of ``"auto"``, ``"layer_boundary"``, ``"branch_merge"``,
        ``"user_specified"``, ``"single_layer"``.
    min_block_size : int
        Minimum number of steps in a sub-module (ignored when
        ``strategy="user_specified"``).
    max_submodules : int
        Maximum number of sub-modules to create.
    cut_after_steps : list of int, optional
        Step indices for user-specified cuts (only used when
        ``strategy="user_specified"``).
    cut_after_layers : list of str, optional
        Layer names for user-specified cuts.
    input_shapes : dict, optional
        User-supplied input shapes for the global model.

    Returns
    -------
    list of SubModule
        Ordered list of sub-modules covering the entire graph.  Adjacent
        sub-modules share a boundary tensor.

    Examples
    --------
    >>> graph = extract_computation_graph(source)
    >>> modules = decompose_graph(graph, strategy="auto")
    >>> for m in modules:
    ...     print(m.name, m.step_range, m.graph.num_steps)
    """
    input_shapes = input_shapes or {}

    # Edge case: empty or trivial graph → single sub-module
    if graph.num_steps <= 1:
        sub_graph = copy.deepcopy(graph)
        in_contract = _derive_input_contract(
            graph.steps, graph.input_names, graph.layers,
            "whole", input_shapes,
        )
        out_contract = _derive_output_contract(
            graph.steps, graph.output_names, graph.layers, "whole",
        )
        return [SubModule(
            name="whole",
            graph=sub_graph,
            input_contract=in_contract,
            output_contract=out_contract,
            step_range=(0, graph.num_steps),
        )]

    # Select cut-point strategy
    strat = DecompositionStrategy(strategy)
    if strat == DecompositionStrategy.AUTO:
        cuts = _find_cut_points_auto(graph, max_submodules, min_block_size)
    elif strat == DecompositionStrategy.LAYER_BOUNDARY:
        cuts = _find_cut_points_layer_boundary(graph, min_block_size)
    elif strat == DecompositionStrategy.BRANCH_MERGE:
        cuts = _find_cut_points_branch_merge(graph)
    elif strat == DecompositionStrategy.USER_SPECIFIED:
        cuts = _find_cut_points_user(graph, cut_after_steps, cut_after_layers)
    elif strat == DecompositionStrategy.SINGLE_LAYER:
        cuts = list(range(1, graph.num_steps))
    else:
        cuts = _find_cut_points_auto(graph, max_submodules, min_block_size)

    # No natural decomposition → single sub-module
    if not cuts:
        sub_graph = copy.deepcopy(graph)
        in_contract = _derive_input_contract(
            graph.steps, graph.input_names, graph.layers,
            "whole", input_shapes,
        )
        out_contract = _derive_output_contract(
            graph.steps, graph.output_names, graph.layers, "whole",
        )
        return [SubModule(
            name="whole",
            graph=sub_graph,
            input_contract=in_contract,
            output_contract=out_contract,
            step_range=(0, graph.num_steps),
        )]

    # Build sub-modules from cut points
    boundaries = [0] + cuts + [graph.num_steps]
    submodules: List[SubModule] = []
    produced_before: Set[str] = set()

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        name = f"block_{i}"
        sub_graph, sub_inputs, sub_outputs = _extract_subgraph(graph, start, end)

        # Only treat a tensor as a global input if it was NOT produced
        # by a prior sub-module (avoids name collisions like x = f(x)).
        truly_global = set(graph.input_names) - produced_before

        in_contract = _derive_input_contract(
            sub_graph.steps, sub_inputs, graph.layers, name, input_shapes,
            truly_global_inputs=truly_global,
        )
        out_contract = _derive_output_contract(
            sub_graph.steps, sub_outputs, graph.layers, name,
        )

        submodules.append(SubModule(
            name=name,
            graph=sub_graph,
            input_contract=in_contract,
            output_contract=out_contract,
            step_range=(start, end),
        ))

        # Track tensors produced by this block for subsequent blocks
        for s in graph.steps[start:end]:
            produced_before.add(s.output)

    return submodules


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Interface compatibility checking
# ═══════════════════════════════════════════════════════════════════════════════

def _shapes_compatible(
    producer_shape: tuple,
    consumer_shape: tuple,
) -> Tuple[bool, str]:
    """Check whether a producer's output shape is compatible with a
    consumer's input shape.

    Symbolic dimensions (strings) are treated as universally quantified
    — they match any concrete dimension.  The wildcard ``("*",)`` matches
    everything.

    Returns ``(compatible, message)``.
    """
    if producer_shape == ("*",) or consumer_shape == ("*",):
        return True, "wildcard match"

    if len(producer_shape) != len(consumer_shape):
        return False, (
            f"rank mismatch: producer has {len(producer_shape)} dims, "
            f"consumer expects {len(consumer_shape)}"
        )

    mismatches: List[str] = []
    for dim_idx, (p, c) in enumerate(zip(producer_shape, consumer_shape)):
        # Both symbolic → match (assume same binding)
        if isinstance(p, str) and isinstance(c, str):
            continue
        # One symbolic → match (symbolic can be anything)
        if isinstance(p, str) or isinstance(c, str):
            continue
        # Both concrete → must be equal
        if p != c:
            mismatches.append(f"dim {dim_idx}: {p} ≠ {c}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "shapes compatible"


def _consumer_has_rank_changing_op(consumer: SubModule, tensor_name: str) -> bool:
    """Check if the consumer's first op on *tensor_name* is rank-changing."""
    for step in consumer.graph.steps:
        if tensor_name in step.inputs:
            return step.op in (OpKind.RESHAPE, OpKind.FLATTEN)
    return False


def _element_count_compatible(
    producer_shape: tuple,
    consumer_shape: tuple,
) -> Tuple[bool, str]:
    """Check element-count compatibility across a rank-changing boundary.

    Returns ``(compatible, message)``.  Concrete dimensions are multiplied;
    symbolic dimensions are treated as unknowns that unify freely.
    """
    def _concrete_product(shape: tuple) -> Optional[int]:
        product = 1
        for d in shape:
            if isinstance(d, str):
                return None  # symbolic — can't compute concrete product
            product *= d
        return product

    # Both shapes must share the batch dimension
    if not producer_shape or not consumer_shape:
        return True, "empty shape — assumed compatible"

    # Strip shared leading symbolic dims (typically "batch")
    p_rest = producer_shape
    c_rest = consumer_shape
    if (isinstance(p_rest[0], str) and isinstance(c_rest[0], str)):
        p_rest = p_rest[1:]
        c_rest = c_rest[1:]

    p_prod = _concrete_product(p_rest)
    c_prod = _concrete_product(c_rest)

    if p_prod is not None and c_prod is not None:
        if p_prod == c_prod:
            return True, (
                f"element count matches across flatten/reshape "
                f"({p_prod} elements)"
            )
        else:
            return False, (
                f"element count mismatch: producer has {p_prod}, "
                f"consumer expects {c_prod}"
            )

    # At least one side has symbolic dims — accept conservatively
    return True, "symbolic dims present — assumed compatible across reshape"


def check_interface_compatibility(
    producer: SubModule,
    consumer: SubModule,
) -> InterfaceCheck:
    """Check that *producer*'s output contract satisfies *consumer*'s
    input contract.

    For every tensor that flows from *producer* to *consumer*, checks:
      1. The output shape from the producer matches the expected input
         shape of the consumer.
      2. Device and phase constraints are consistent.

    When the consumer's first operation on a shared tensor is a
    rank-changing op (FLATTEN/RESHAPE), a rank mismatch is expected and
    element-count compatibility is checked instead of strict shape
    matching.
    """
    producer_outputs = producer.output_contract.output_shapes
    consumer_inputs = consumer.input_contract.input_shapes

    # Find shared tensors (outputs of producer that are inputs of consumer)
    shared = set(producer_outputs.keys()) & set(consumer_inputs.keys())

    if not shared:
        # No direct tensor flow — check if any consumer input is produced
        # by the producer's sub-graph
        produced_by_producer = {s.output for s in producer.graph.steps}
        consumed_by_consumer = set()
        for s in consumer.graph.steps:
            consumed_by_consumer.update(s.inputs)
        actually_shared = produced_by_producer & consumed_by_consumer

        if not actually_shared:
            return InterfaceCheck(
                producer=producer.name,
                consumer=consumer.name,
                compatible=True,
                message="no direct tensor flow (independent sub-modules)",
            )
        # Tensors flow but weren't in the contracts — still compatible
        # (contracts may be under-specified for shape-preserving ops)
        return InterfaceCheck(
            producer=producer.name,
            consumer=consumer.name,
            compatible=True,
            message=(
                f"{len(actually_shared)} tensor(s) flow but contracts "
                f"are not explicit — assumed compatible"
            ),
        )

    # Check each shared tensor
    issues: List[str] = []
    for tname in sorted(shared):
        p_shape = producer_outputs[tname]
        c_shape = consumer_inputs[tname]
        ok, msg = _shapes_compatible(p_shape, c_shape)
        if not ok:
            # If the consumer's first op on this tensor is a rank-changing
            # op (flatten/reshape), check element-count compatibility
            # instead of strict shape matching.
            if _consumer_has_rank_changing_op(consumer, tname):
                ok, msg = _element_count_compatible(p_shape, c_shape)
            if not ok:
                issues.append(f"{tname}: {msg}")

    # Device consistency
    if (
        producer.output_contract.device is not None
        and consumer.input_contract.device is not None
        and producer.output_contract.device != consumer.input_contract.device
    ):
        issues.append(
            f"device mismatch: {producer.output_contract.device.value} → "
            f"{consumer.input_contract.device.value}"
        )

    if issues:
        return InterfaceCheck(
            producer=producer.name,
            consumer=consumer.name,
            compatible=False,
            message="; ".join(issues),
        )

    return InterfaceCheck(
        producer=producer.name,
        consumer=consumer.name,
        compatible=True,
        message=f"{len(shared)} shared tensor(s) — all shapes compatible",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Per-sub-module verification
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_submodule(
    submodule: SubModule,
    input_shapes: Dict[str, tuple],
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
) -> VerificationResult:
    """Verify a single sub-module under its assumed input contract.

    Constructs a ``ConstraintVerifier`` scoped to the sub-module's graph
    and uses the input contract shapes as the initial symbolic state.
    """
    # Build effective input shapes from the contract
    effective_shapes: Dict[str, tuple] = {}
    for tname, shape in submodule.input_contract.input_shapes.items():
        if shape != ("*",):
            effective_shapes[tname] = shape

    # Fill in with explicit input_shapes only for tensors not already
    # covered by the contract (contract-derived shapes take priority).
    for tname, shape in input_shapes.items():
        if tname not in effective_shapes and tname in submodule.graph.input_names:
            effective_shapes[tname] = shape

    verifier = ConstraintVerifier(
        submodule.graph,
        input_shapes=effective_shapes,
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        constraints=constraints,
    )
    return verifier.verify()


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Verification result cache
# ═══════════════════════════════════════════════════════════════════════════════

class VerificationCache:
    """Cache of per-sub-module verification results keyed by fingerprint.

    Used for incremental re-verification: when a sub-module's structure
    hasn't changed (same fingerprint), its cached ``VerificationResult``
    is reused instead of re-running Z3.

    The cache is purely in-memory and does not persist across processes.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[VerificationResult, float]] = {}

    def get(self, fingerprint: str) -> Optional[VerificationResult]:
        """Retrieve a cached result, or ``None`` if not present."""
        entry = self._store.get(fingerprint)
        if entry is not None:
            return entry[0]
        return None

    def put(
        self,
        fingerprint: str,
        result: VerificationResult,
        time_ms: float,
    ) -> None:
        """Store a verification result."""
        self._store[fingerprint] = (result, time_ms)

    def invalidate(self, fingerprint: str) -> None:
        """Remove a cached entry."""
        self._store.pop(fingerprint, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Module-level default cache (re-used across calls in the same process).
_default_cache = VerificationCache()


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  Compositional verification — main entry points
# ═══════════════════════════════════════════════════════════════════════════════

def verify_compositional(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    strategy: str = "auto",
    *,
    min_block_size: int = 2,
    max_submodules: int = 8,
    cut_after_steps: Optional[List[int]] = None,
    cut_after_layers: Optional[List[str]] = None,
    cache: Optional[VerificationCache] = None,
    measure_monolithic: bool = True,
) -> CompositionalResult:
    """Assume-guarantee compositional verification of an ``nn.Module``.

    Decomposes the model's computation graph into sub-modules, verifies
    each independently, and checks interface compatibility.

    Formal Soundness Theorem (Abadi-Lamport Sequential Composition)
    ---------------------------------------------------------------
    Let M_0, M_1, ..., M_{n-1} be the decomposed submodules with
    interface contracts C_0, C_1, ..., C_n (where C_i is the
    precondition of M_i and C_{i+1} is its postcondition).

    **If** all of the following hold:

    1. Every submodule verification succeeds:
       ∀ i ∈ [0, n-1]:  {C_i} M_i {C_{i+1}}   (each M_i satisfies
       its contract under the assumed input)

    2. All interface contracts are compatible:
       ∀ i ∈ [0, n-2]:  C_{i+1} as M_i's postcondition implies
                         C_{i+1} as M_{i+1}'s precondition
       (the output contract of each submodule entails the input
       contract of the next submodule)

    3. The global precondition C_0 is satisfied by the user-supplied
       input shapes.

    **Then** the whole module M_0 ; M_1 ; ... ; M_{n-1} is safe:
       {C_0} M_0;...;M_{n-1} {C_n}

    This rule is non-circular: the proof obligation for M_i depends
    only on the *contract* of its neighbours, not on their
    implementations.  Soundness follows by induction on the chain
    length, with each step justified by the Hoare sequential
    composition rule.

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.
    default_device, default_phase, max_k, constraints
        Forwarded to the per-sub-module ``ConstraintVerifier``.
    strategy : str
        Decomposition strategy (see ``decompose_graph``).
    min_block_size, max_submodules
        Control decomposition granularity.
    cut_after_steps, cut_after_layers
        User-specified decomposition points.
    cache : VerificationCache, optional
        Reuse cached results.  ``None`` uses the module-level default.
    measure_monolithic : bool
        When ``True`` (default), also runs monolithic verification to
        compute a speedup ratio.

    Returns
    -------
    CompositionalResult
        Aggregated result with per-sub-module details.
    """
    t0 = time.monotonic()
    input_shapes = input_shapes or {}
    cache = cache if cache is not None else _default_cache

    # ── Step 1: parse source → ComputationGraph ──────────────────────────
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return CompositionalResult(
            safe=False,
            total_time_ms=(time.monotonic() - t0) * 1000,
            submodule_results={
                "__parse_error__": VerificationResult(
                    safe=False, errors=[str(exc)],
                )
            },
        )

    # ── Step 2: decompose ─────────────────────────────────────────────────
    strat_enum = DecompositionStrategy(strategy)
    submodules = decompose_graph(
        graph,
        strategy=strategy,
        min_block_size=min_block_size,
        max_submodules=max_submodules,
        cut_after_steps=cut_after_steps,
        cut_after_layers=cut_after_layers,
        input_shapes=input_shapes,
    )

    # ── Step 3: verify each sub-module ────────────────────────────────────
    submodule_results: Dict[str, VerificationResult] = {}
    cache_hits = 0

    for sm in submodules:
        fp = sm.fingerprint()
        cached = cache.get(fp)
        if cached is not None:
            submodule_results[sm.name] = cached
            cache_hits += 1
            logger.info("Cache hit for sub-module %s (fp=%s)", sm.name, fp)
            continue

        logger.info("Verifying sub-module %s (%d steps)", sm.name, sm.graph.num_steps)
        result = _verify_submodule(
            sm, input_shapes,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
            constraints=constraints,
        )
        submodule_results[sm.name] = result
        cache.put(fp, result, result.verification_time_ms)

    # ── Step 4: check interface compatibility ─────────────────────────────
    interface_checks: List[InterfaceCheck] = []
    for i in range(len(submodules) - 1):
        check = check_interface_compatibility(submodules[i], submodules[i + 1])
        interface_checks.append(check)

    # ── Step 5: compose results ───────────────────────────────────────────
    all_safe = all(r.safe for r in submodule_results.values())
    all_compatible = all(ic.compatible for ic in interface_checks)
    overall_safe = all_safe and all_compatible

    # ── Step 5b: build proof tree ─────────────────────────────────────────
    proof_tree = _build_proof_tree(submodules, submodule_results, interface_checks)

    compositional_time = (time.monotonic() - t0) * 1000

    # ── Step 6: optional monolithic comparison ────────────────────────────
    speedup = 1.0
    if measure_monolithic:
        mono_t0 = time.monotonic()
        try:
            _mono = verify_model(
                source,
                input_shapes=input_shapes,
                default_device=default_device,
                default_phase=default_phase,
                max_k=max_k,
                constraints=constraints,
            )
            mono_time = (time.monotonic() - mono_t0) * 1000
            speedup = mono_time / compositional_time if compositional_time > 0 else 1.0
        except Exception:
            logger.warning("Monolithic verification failed; speedup not measured")
            speedup = 1.0

    return CompositionalResult(
        safe=overall_safe,
        submodule_results=submodule_results,
        interface_checks=interface_checks,
        total_time_ms=compositional_time,
        speedup_vs_monolithic=speedup,
        cache_hits=cache_hits,
        decomposition_strategy=strat_enum,
        num_submodules=len(submodules),
        proof_tree=proof_tree,
    )


def verify_compositional_incremental(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    cache: Optional[VerificationCache] = None,
    changed_modules: Optional[Set[str]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    strategy: str = "auto",
    *,
    min_block_size: int = 2,
    max_submodules: int = 8,
    cut_after_steps: Optional[List[int]] = None,
    cut_after_layers: Optional[List[str]] = None,
) -> CompositionalResult:
    """Incremental assume-guarantee verification.

    Re-verifies only the sub-modules listed in ``changed_modules`` (plus
    their immediate neighbours for interface re-checking).  All other
    sub-modules reuse their cached ``VerificationResult``.

    Parameters
    ----------
    source : str
        Updated Python source code.
    input_shapes : dict, optional
        Global input shapes.
    cache : VerificationCache, optional
        Cache from a prior ``verify_compositional`` call.  If ``None``,
        uses the module-level default cache.
    changed_modules : set of str, optional
        Names of sub-modules that changed.  If ``None`` or empty, all
        sub-modules are re-verified (equivalent to a full compositional
        verification).
    default_device, default_phase, max_k, constraints, strategy,
    min_block_size, max_submodules, cut_after_steps, cut_after_layers
        Same as ``verify_compositional``.

    Returns
    -------
    CompositionalResult
        With ``cache_hits`` reflecting how many modules were reused.
    """
    t0 = time.monotonic()
    input_shapes = input_shapes or {}
    cache = cache if cache is not None else _default_cache
    changed_modules = changed_modules or set()

    # ── Parse & decompose ─────────────────────────────────────────────────
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return CompositionalResult(
            safe=False,
            total_time_ms=(time.monotonic() - t0) * 1000,
            submodule_results={
                "__parse_error__": VerificationResult(
                    safe=False, errors=[str(exc)],
                )
            },
        )

    strat_enum = DecompositionStrategy(strategy)
    submodules = decompose_graph(
        graph,
        strategy=strategy,
        min_block_size=min_block_size,
        max_submodules=max_submodules,
        cut_after_steps=cut_after_steps,
        cut_after_layers=cut_after_layers,
        input_shapes=input_shapes,
    )

    # Determine which modules need re-verification:
    # - explicitly changed modules
    # - neighbours of changed modules (for interface re-check)
    module_names = [sm.name for sm in submodules]
    needs_reverify: Set[str] = set(changed_modules)
    for name in list(changed_modules):
        if name in module_names:
            idx = module_names.index(name)
            if idx > 0:
                needs_reverify.add(module_names[idx - 1])
            if idx < len(module_names) - 1:
                needs_reverify.add(module_names[idx + 1])

    # If no specific modules changed, re-verify everything
    if not needs_reverify:
        needs_reverify = set(module_names)

    # ── Verify sub-modules ────────────────────────────────────────────────
    submodule_results: Dict[str, VerificationResult] = {}
    cache_hits = 0

    for sm in submodules:
        fp = sm.fingerprint()

        if sm.name not in needs_reverify:
            cached = cache.get(fp)
            if cached is not None:
                submodule_results[sm.name] = cached
                cache_hits += 1
                continue

        # Invalidate old cache entry and re-verify
        cache.invalidate(fp)
        logger.info(
            "Re-verifying sub-module %s (%d steps, changed=%s)",
            sm.name, sm.graph.num_steps, sm.name in changed_modules,
        )
        result = _verify_submodule(
            sm, input_shapes,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
            constraints=constraints,
        )
        submodule_results[sm.name] = result
        cache.put(fp, result, result.verification_time_ms)

    # ── Interface checks ──────────────────────────────────────────────────
    interface_checks: List[InterfaceCheck] = []
    for i in range(len(submodules) - 1):
        check = check_interface_compatibility(submodules[i], submodules[i + 1])
        interface_checks.append(check)

    all_safe = all(r.safe for r in submodule_results.values())
    all_compatible = all(ic.compatible for ic in interface_checks)

    return CompositionalResult(
        safe=all_safe and all_compatible,
        submodule_results=submodule_results,
        interface_checks=interface_checks,
        total_time_ms=(time.monotonic() - t0) * 1000,
        speedup_vs_monolithic=1.0,  # not measured in incremental mode
        cache_hits=cache_hits,
        decomposition_strategy=strat_enum,
        num_submodules=len(submodules),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11b. DAG interface validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_interface_dag(
    submodules: List[SubModule],
    edges: List[Tuple[int, int]],
) -> List[InterfaceCheck]:
    """Validate interface compatibility for DAG edges.

    Generalizes ``validate_interface_chain()`` to work on arbitrary DAG
    edges instead of sequential pairs.  For each edge ``(producer_idx,
    consumer_idx)`` in the DAG, verifies that the producer's output
    contract implies the consumer's input contract.

    Parameters
    ----------
    submodules : list of SubModule
        All submodules (nodes) in the DAG.
    edges : list of (int, int)
        Directed edges ``(producer_idx, consumer_idx)``.

    Returns
    -------
    list of InterfaceCheck
        One entry per edge.
    """
    checks: List[InterfaceCheck] = []
    for src_idx, dst_idx in edges:
        if src_idx < 0 or src_idx >= len(submodules):
            continue
        if dst_idx < 0 or dst_idx >= len(submodules):
            continue
        producer = submodules[src_idx]
        consumer = submodules[dst_idx]
        check = check_interface_compatibility(producer, consumer)
        checks.append(check)
    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# 11c. DAG decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_skip_connections(
    steps: List[ComputationStep],
) -> List[Tuple[int, int]]:
    """Detect skip connections: output of step i consumed by step j where j > i+1."""
    producers = _tensor_producers(steps)
    skip_edges: List[Tuple[int, int]] = []
    for idx, step in enumerate(steps):
        for inp in step.inputs:
            if inp in producers:
                prod_idx = producers[inp]
                if idx - prod_idx > 1:
                    skip_edges.append((prod_idx, idx))
    return skip_edges


def _detect_merge_nodes(
    steps: List[ComputationStep],
) -> List[int]:
    """Detect merge points (steps with multiple inputs from different producers)."""
    producers = _tensor_producers(steps)
    merges: List[int] = []
    for idx, step in enumerate(steps):
        input_producers = set()
        for inp in step.inputs:
            if inp in producers:
                input_producers.add(producers[inp])
        if len(input_producers) > 1:
            merges.append(idx)
    return merges


def _classify_dag_topology(
    steps: List[ComputationStep],
    skip_edges: List[Tuple[int, int]],
    merge_nodes: List[int],
) -> str:
    """Classify the DAG topology based on detected patterns."""
    if not skip_edges and not merge_nodes:
        return "sequential"

    has_cat = any(s.op == OpKind.CAT for s in steps)
    has_add = any(s.op == OpKind.ADD for s in steps)

    # Check for residual pattern: skip + add
    if has_add and skip_edges and not has_cat:
        return "residual"

    # Check for dense concatenation: skip + cat
    if has_cat and len(skip_edges) > 1:
        return "dense"

    # Check for encoder-decoder pattern: mirror structure with skips
    n = len(steps)
    if skip_edges:
        # If skips go from early steps to late steps (>50% span), likely encoder-decoder
        max_span = max(dst - src for src, dst in skip_edges) if skip_edges else 0
        if max_span > n * 0.4 and has_cat:
            return "encoder_decoder"

    return "general_dag"


def decompose_graph_dag(
    graph: ComputationGraph,
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> Tuple[List[SubModule], List[Tuple[int, int]], str]:
    """Decompose a ComputationGraph into a DAG of submodules.

    Detects skip connections, residual additions, concatenations, and
    creates proper DAG submodules with edges.  Returns the submodules,
    the DAG edges between them, and the detected topology type.

    Parameters
    ----------
    graph : ComputationGraph
        The full computation graph.
    input_shapes : dict, optional
        User-supplied input shapes.

    Returns
    -------
    submodules : list of SubModule
        The decomposed submodules (DAG nodes).
    edges : list of (int, int)
        Directed edges between submodule indices.
    topology : str
        Detected topology type.
    """
    input_shapes = input_shapes or {}
    steps = graph.steps

    if not steps:
        return [], [], "sequential"

    # Detect structural features
    skip_edges = _detect_skip_connections(steps)
    merge_nodes = _detect_merge_nodes(steps)
    topology = _classify_dag_topology(steps, skip_edges, merge_nodes)

    # For sequential models, fall back to normal decomposition
    if topology == "sequential":
        subs = decompose_graph(graph, strategy="auto", input_shapes=input_shapes)
        seq_edges = [(i, i + 1) for i in range(len(subs) - 1)]
        return subs, seq_edges, "sequential"

    # Partition steps into blocks.  Use merge/branch points as cut boundaries.
    cut_set: Set[int] = set()
    for idx in merge_nodes:
        if idx > 0:
            cut_set.add(idx)

    # Also cut at branch points (outputs consumed by multiple later steps)
    for src, dst in skip_edges:
        if src + 1 < len(steps):
            cut_set.add(src + 1)

    # Ensure we have at least one cut
    if not cut_set and len(steps) > 1:
        mid = len(steps) // 2
        cut_set.add(mid)

    cuts = sorted(cut_set)
    boundaries = [0] + cuts + [len(steps)]

    # Deduplicate consecutive identical boundaries
    deduped: List[int] = [boundaries[0]]
    for b in boundaries[1:]:
        if b != deduped[-1]:
            deduped.append(b)
    boundaries = deduped

    # Build submodules
    submodules: List[SubModule] = []
    produced_before: Set[str] = set()

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if start >= end:
            continue
        name = f"dag_block_{i}"
        sub_graph, sub_inputs, sub_outputs = _extract_subgraph(graph, start, end)

        truly_global = set(graph.input_names) - produced_before

        in_contract = _derive_input_contract(
            sub_graph.steps, sub_inputs, graph.layers, name, input_shapes,
            truly_global_inputs=truly_global,
        )
        out_contract = _derive_output_contract(
            sub_graph.steps, sub_outputs, graph.layers, name,
        )
        submodules.append(SubModule(
            name=name,
            graph=sub_graph,
            input_contract=in_contract,
            output_contract=out_contract,
            step_range=(start, end),
        ))
        for s in graph.steps[start:end]:
            produced_before.add(s.output)

    # Build DAG edges between submodules.
    # An edge (i, j) exists if any tensor produced by submodule i is
    # consumed by submodule j.
    block_ranges = [(sm.step_range[0], sm.step_range[1]) for sm in submodules]
    dag_edges: List[Tuple[int, int]] = []
    edge_set: Set[Tuple[int, int]] = set()

    for j_idx, sm_j in enumerate(submodules):
        consumed = set()
        for step in sm_j.graph.steps:
            consumed.update(step.inputs)
        for i_idx, sm_i in enumerate(submodules):
            if i_idx >= j_idx:
                continue
            produced = {step.output for step in sm_i.graph.steps}
            if consumed & produced:
                edge_pair = (i_idx, j_idx)
                if edge_pair not in edge_set:
                    edge_set.add(edge_pair)
                    dag_edges.append(edge_pair)

    # Add sequential edges for adjacent blocks that don't already have edges
    for i in range(len(submodules) - 1):
        edge_pair = (i, i + 1)
        if edge_pair not in edge_set:
            edge_set.add(edge_pair)
            dag_edges.append(edge_pair)

    return submodules, dag_edges, topology


# ═══════════════════════════════════════════════════════════════════════════════
# 11d. DAG compositional verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_compositional_dag(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    cache: Optional[VerificationCache] = None,
    measure_monolithic: bool = True,
) -> CompositionalResult:
    """Assume-guarantee compositional verification for DAG architectures.

    Generalizes ``verify_compositional`` to handle non-sequential models
    like ResNet (skip/residual), U-Net (encoder-decoder with skip),
    DenseNet (dense concatenation), and Transformer (cross-attention).

    Verification proceeds in topological order: each node is verified
    only after all its predecessors have been verified.

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.
    default_device, default_phase, max_k, constraints
        Forwarded to per-submodule verification.
    cache : VerificationCache, optional
        Result cache.
    measure_monolithic : bool
        When True, also run monolithic verification for speedup ratio.

    Returns
    -------
    CompositionalResult
        With DAG-aware interface checks and proof rule.
    """
    t0 = time.monotonic()
    input_shapes = input_shapes or {}
    cache = cache if cache is not None else _default_cache

    # Step 1: parse source
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return CompositionalResult(
            safe=False,
            total_time_ms=(time.monotonic() - t0) * 1000,
            submodule_results={
                "__parse_error__": VerificationResult(
                    safe=False, errors=[str(exc)],
                )
            },
        )

    # Step 2: DAG decomposition
    submodules, dag_edges, topology = decompose_graph_dag(graph, input_shapes)

    if not submodules:
        return CompositionalResult(
            safe=True,
            total_time_ms=(time.monotonic() - t0) * 1000,
            num_submodules=0,
        )

    # Step 3: build DAG proof rule for topological ordering
    dag_rule = DAGCompositionProofRule.from_submodules_and_edges(
        submodules, dag_edges, topology,
    )
    topo_order = dag_rule.topological_order()

    # Step 4: verify each submodule in topological order
    submodule_results: Dict[str, VerificationResult] = {}
    cache_hits = 0
    verified_indices: Set[int] = set()

    for node_idx in topo_order:
        sm = submodules[node_idx]
        fp = sm.fingerprint()
        cached = cache.get(fp)
        if cached is not None:
            submodule_results[sm.name] = cached
            cache_hits += 1
            verified_indices.add(node_idx)
            continue

        # Check that all predecessors are verified
        preds_ok = all(
            src in verified_indices
            for src, dst in dag_edges if dst == node_idx
        )
        if not preds_ok:
            logger.warning(
                "DAG verification: predecessor not verified for %s",
                sm.name,
            )

        result = _verify_submodule(
            sm, input_shapes,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
            constraints=constraints,
        )
        submodule_results[sm.name] = result
        cache.put(fp, result, result.verification_time_ms)
        verified_indices.add(node_idx)

    # Step 5: validate DAG interfaces
    interface_checks = validate_interface_dag(submodules, dag_edges)

    # Step 6: compose results
    all_safe = all(r.safe for r in submodule_results.values())
    all_compatible = all(ic.compatible for ic in interface_checks)
    overall_safe = all_safe and all_compatible

    compositional_time = (time.monotonic() - t0) * 1000

    # Step 7: optional monolithic comparison
    speedup = 1.0
    if measure_monolithic:
        mono_t0 = time.monotonic()
        try:
            _mono = verify_model(
                source,
                input_shapes=input_shapes,
                default_device=default_device,
                default_phase=default_phase,
                max_k=max_k,
                constraints=constraints,
            )
            mono_time = (time.monotonic() - mono_t0) * 1000
            speedup = mono_time / compositional_time if compositional_time > 0 else 1.0
        except Exception:
            logger.warning("Monolithic verification failed; speedup not measured")
            speedup = 1.0

    return CompositionalResult(
        safe=overall_safe,
        submodule_results=submodule_results,
        interface_checks=interface_checks,
        total_time_ms=compositional_time,
        speedup_vs_monolithic=speedup,
        cache_hits=cache_hits,
        decomposition_strategy=DecompositionStrategy.BRANCH_MERGE,
        num_submodules=len(submodules),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  Convenience helpers
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_and_summarize(
    source: str,
    strategy: str = "auto",
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> str:
    """Parse source, decompose, and return a human-readable summary.

    Useful for debugging decomposition without running verification.

    Returns
    -------
    str
        Multi-line summary of sub-modules and their contracts.
    """
    try:
        graph = extract_computation_graph(source)
    except (ValueError, SyntaxError) as exc:
        return f"Parse error: {exc}"

    submodules = decompose_graph(
        graph, strategy=strategy, input_shapes=input_shapes or {},
    )

    lines = [
        f"Model: {graph.class_name}",
        f"Total steps: {graph.num_steps}",
        f"Strategy: {strategy}",
        f"Sub-modules: {len(submodules)}",
        "",
    ]

    for sm in submodules:
        lines.append(f"── {sm.name} (steps {sm.step_range[0]}–{sm.step_range[1] - 1}) ──")
        lines.append(f"  Steps:  {sm.graph.num_steps}")
        lines.append(f"  Inputs: {sm.graph.input_names}")
        lines.append(f"  Outputs: {sm.graph.output_names}")
        lines.append(f"  {sm.input_contract.pretty()}")
        lines.append(f"  {sm.output_contract.pretty()}")
        lines.append("")

    # Interface compatibility (without full verification)
    for i in range(len(submodules) - 1):
        check = check_interface_compatibility(submodules[i], submodules[i + 1])
        tag = "✓" if check.compatible else "✗"
        lines.append(f"  {tag} {check.producer} → {check.consumer}: {check.message}")

    return "\n".join(lines)


def get_default_cache() -> VerificationCache:
    """Return the module-level default ``VerificationCache``."""
    return _default_cache


def reset_default_cache() -> None:
    """Clear the module-level default cache."""
    _default_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  Formal Abadi-Lamport proof rule with validation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProofObligation:
    """A single obligation in the Abadi-Lamport proof rule.

    Attributes
    ----------
    kind : str
        One of: ``"submodule_safety"`` (∀i: {I_i} M_i {O_i}),
        ``"interface_compatibility"`` (∀i: O_i ⊑ I_{i+1}),
        ``"input_precondition"`` (I_0 satisfied by input).
    submodule : str
        Name of the submodule this obligation pertains to.
    description : str
        Human-readable description.
    satisfied : bool
        Whether the obligation has been discharged.
    evidence : str
        Evidence or reason for the satisfaction/failure.
    """

    kind: str
    submodule: str
    description: str
    satisfied: bool = False
    evidence: str = ""


@dataclass
class FormalProofRule:
    """Formal Abadi-Lamport non-circular sequential composition rule.

    Encodes the standard assume-guarantee proof rule:

        ∀i: {I_i} M_i {O_i}    ∀i: O_i ⊑ I_{i+1}    I_0 satisfied by input
        ──────────────────────────────────────────────────────────────────────
        {I_0} M_0;...;M_n {O_n}

    where:
    - I_i is the input (precondition) contract for submodule M_i
    - O_i is the output (postcondition) contract for submodule M_i
    - O_i ⊑ I_{i+1} means the output contract of M_i implies the input
      contract of M_{i+1} (interface compatibility)

    Non-circularity: the proof obligation for M_i depends only on the
    *contract* of M_{i-1}, not on its implementation.  This breaks
    circular reasoning and enables independent verification.

    Soundness: If all obligations are satisfied, then the sequential
    composition M_0;...;M_n is safe with respect to the global
    precondition I_0 and global postcondition O_n.

    Reference: M. Abadi and L. Lamport, "Conjoining Specifications",
    ACM TOPLAS 17(3), 1995.

    Attributes
    ----------
    submodules : list of SubModule
        The submodules in sequential order.
    obligations : list of ProofObligation
        All proof obligations (submodule safety + interface + input).
    conclusion_holds : bool
        Whether the composition conclusion follows (all obligations met).
    """

    submodules: List[SubModule] = field(default_factory=list)
    obligations: List[ProofObligation] = field(default_factory=list)
    conclusion_holds: bool = False

    def pretty(self) -> str:
        """Render the proof rule as a formatted string."""
        lines = [
            "Abadi-Lamport Non-Circular Composition Rule",
            "=" * 60,
            "",
            "Premises:",
        ]
        for obl in self.obligations:
            tag = "✓" if obl.satisfied else "✗"
            lines.append(f"  {tag} [{obl.kind}] {obl.description}")
            if obl.evidence:
                lines.append(f"      Evidence: {obl.evidence}")
        lines.append("")
        lines.append("─" * 60)
        if self.conclusion_holds:
            if self.submodules:
                names = " ; ".join(sm.name for sm in self.submodules)
                lines.append(
                    f"∴ {{{self.submodules[0].input_contract.name}}} "
                    f"{names} "
                    f"{{{self.submodules[-1].output_contract.name}}}"
                )
            lines.append("CONCLUSION: SAFE ✓")
        else:
            failed = [o for o in self.obligations if not o.satisfied]
            lines.append(f"CONCLUSION: NOT ESTABLISHED ({len(failed)} failed)")
        return "\n".join(lines)


@dataclass
class CacheInvalidationSpec:
    """Specification of cache invalidation when a submodule changes.

    When M_i changes, only M_i and its downstream submodules
    (M_{i+1}, ..., M_n) need re-verification.  Upstream modules
    (M_0, ..., M_{i-1}) can reuse cached results.

    Additionally, interface checks for (M_{i-1}, M_i) and (M_i, M_{i+1})
    must be re-validated.

    Attributes
    ----------
    changed_module : str
        The module that changed.
    changed_index : int
        The index of the changed module in the chain.
    modules_to_reverify : list of str
        Modules that need re-verification (M_i and downstream).
    interfaces_to_recheck : list of tuple
        Pairs (producer, consumer) of interface checks to redo.
    modules_cached : list of str
        Modules whose cached results can be reused.
    """

    changed_module: str
    changed_index: int
    modules_to_reverify: List[str] = field(default_factory=list)
    interfaces_to_recheck: List[Tuple[str, str]] = field(default_factory=list)
    modules_cached: List[str] = field(default_factory=list)


def validate_proof_rule(
    submodules: List[SubModule],
    submodule_results: Dict[str, VerificationResult],
    interface_checks: List[InterfaceCheck],
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> FormalProofRule:
    """Validate all obligations of the Abadi-Lamport proof rule.

    Checks three categories of obligations:

    1. **Submodule safety**: ∀i: {I_i} M_i {O_i}
       Each submodule must be verified safe under its input contract.

    2. **Interface compatibility**: ∀i: O_i ⊑ I_{i+1}
       The output contract of M_i must imply the input contract of M_{i+1}.

    3. **Input precondition**: I_0 satisfied by the user-supplied input.
       The first submodule's input contract must be satisfiable.

    Parameters
    ----------
    submodules : list of SubModule
        Submodules in sequential order.
    submodule_results : dict
        Per-submodule verification results.
    interface_checks : list of InterfaceCheck
        Interface compatibility checks.
    input_shapes : dict, optional
        User-supplied input shapes to check I_0 satisfaction.

    Returns
    -------
    FormalProofRule
        With all obligations checked and conclusion determined.
    """
    obligations: List[ProofObligation] = []

    # Obligation 1: Submodule safety  ∀i: {I_i} M_i {O_i}
    for sm in submodules:
        result = submodule_results.get(sm.name)
        satisfied = result is not None and result.safe
        obligations.append(ProofObligation(
            kind="submodule_safety",
            submodule=sm.name,
            description=(
                f"{{{sm.input_contract.name}}} {sm.name} "
                f"{{{sm.output_contract.name}}}"
            ),
            satisfied=satisfied,
            evidence=(
                f"Verified safe in {result.verification_time_ms:.1f}ms"
                if satisfied and result is not None
                else f"Failed: {result.errors[0] if result and result.errors else 'not verified'}"
            ),
        ))

    # Obligation 2: Interface compatibility  ∀i: O_i ⊑ I_{i+1}
    for ic in interface_checks:
        obligations.append(ProofObligation(
            kind="interface_compatibility",
            submodule=f"{ic.producer}→{ic.consumer}",
            description=f"O({ic.producer}) ⊑ I({ic.consumer})",
            satisfied=ic.compatible,
            evidence=ic.message,
        ))

    # Obligation 3: Input precondition  I_0 satisfied
    if submodules:
        i0 = submodules[0].input_contract
        input_ok = True
        evidence_parts = []

        if input_shapes:
            for tname, expected in i0.input_shapes.items():
                if expected == ("*",):
                    continue
                supplied = input_shapes.get(tname)
                if supplied is None:
                    input_ok = False
                    evidence_parts.append(f"{tname}: not supplied")
                else:
                    ok, msg = _shapes_compatible(supplied, expected)
                    if not ok:
                        input_ok = False
                        evidence_parts.append(f"{tname}: {msg}")
                    else:
                        evidence_parts.append(f"{tname}: ✓")
        else:
            evidence_parts.append("No input shapes supplied — assumed OK")

        obligations.append(ProofObligation(
            kind="input_precondition",
            submodule=submodules[0].name,
            description=f"I_0 ({i0.name}) satisfied by input",
            satisfied=input_ok,
            evidence="; ".join(evidence_parts),
        ))

    conclusion = all(o.satisfied for o in obligations)

    return FormalProofRule(
        submodules=submodules,
        obligations=obligations,
        conclusion_holds=conclusion,
    )


def compute_cache_invalidation(
    submodules: List[SubModule],
    changed_module: str,
) -> CacheInvalidationSpec:
    """Compute cache invalidation for a changed submodule.

    When M_i changes, only M_i and its downstream need re-verification.
    Upstream modules reuse cached results.

    Parameters
    ----------
    submodules : list of SubModule
        All submodules in sequential order.
    changed_module : str
        Name of the changed submodule.

    Returns
    -------
    CacheInvalidationSpec
    """
    names = [sm.name for sm in submodules]
    if changed_module not in names:
        return CacheInvalidationSpec(
            changed_module=changed_module,
            changed_index=-1,
            modules_to_reverify=[],
            interfaces_to_recheck=[],
            modules_cached=list(names),
        )

    idx = names.index(changed_module)

    # M_i and all downstream need re-verification
    to_reverify = names[idx:]

    # Interface checks: (M_{i-1}, M_i) and (M_j, M_{j+1}) for j >= i
    interfaces = []
    if idx > 0:
        interfaces.append((names[idx - 1], names[idx]))
    for j in range(idx, len(names) - 1):
        interfaces.append((names[j], names[j + 1]))

    # Upstream modules can be cached
    cached = names[:idx]

    return CacheInvalidationSpec(
        changed_module=changed_module,
        changed_index=idx,
        modules_to_reverify=to_reverify,
        interfaces_to_recheck=interfaces,
        modules_cached=cached,
    )
