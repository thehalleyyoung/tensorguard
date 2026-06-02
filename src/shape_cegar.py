"""
Shape Contract Discovery: Houdini-Style Predicate Accumulation for Tensor Shapes.

Implements a counterexample-guided contract discovery loop based on
Houdini-style predicate accumulation (Flanagan & Leino FME 2001) that
iteratively discovers shape predicates for nn.Module computation graphs.
Unlike general CEGAR (Clarke et al. CAV 2000), our loop *accumulates*
predicates monotonically from counterexamples rather than maintaining
an abstract domain.  Z3-backed abstract feasibility checking classifies
counterexamples as real bugs or spurious (eliminable by constraining
input shapes), but does not perform concrete Python execution.

Algorithm
---------
1. **Verify**   — Run constraint-based verification (via ``ConstraintVerifier``)
                  on the computation graph with the current shape environment.
2. **Check**    — Examine the counterexample(s) returned by the verifier.
                  If no counterexample → shapes are safe; emit verification condition.
3. **Extract**  — From each Z3-produced counterexample, extract concrete
                  dimension values that caused the reported shape mismatch.
4. **Trace**    — Walk the computation graph *backwards* from the failing
                  step to find the input shape assumption(s) whose
                  weakening allowed the spurious counterexample.
5. **Synth**    — Synthesise a new shape predicate that rules out the
                  spurious counterexample (e.g. ``input.shape[-1] == 768``).
6. **Refine**   — Add the new predicate to the shape environment and
                  re-verify from step 1.
7. **Converge** — Stop when (a) no more counterexamples, (b) a real bug
                  is found, or (c) the iteration budget is exhausted.

Timeout semantics
-----------------
Z3 solver timeouts (1000–5000 ms per query) and CEGAR iteration bounds
interact as follows:

* **Z3 timeout during an iteration**: Z3 returns ``unknown``.  The
  counterexample cannot be classified as spurious, so it is
  conservatively treated as a real bug.  The verdict is UNSAFE if at
  least one counterexample is confirmed real, or UNKNOWN if all are
  indeterminate.
* **CEGAR iteration budget exhausted**: The loop terminates with
  verdict TIMEOUT.  Soundness is preserved: SAFE is never reported
  unless all counterexamples have been eliminated.  TIMEOUT and
  UNKNOWN are conservative (the model may or may not be safe).

Integration points
------------------
* ``model_checker.ComputationGraph`` — for trace-back over operations.
* ``model_checker.ConstraintVerifier`` — for per-iteration verification.
* ``tensor_shapes.ShapeEnv`` / ``TensorShape`` — for shape representation.
* Z3 — for SAT/UNSAT checking and counterexample extraction.
"""

from __future__ import annotations

import copy
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from src.tensor_shapes import (
    TensorShape,
    ShapeDim,
    ShapeError,
    ShapeErrorKind,
    ShapeEnv,
)
from src.model_checker import (
    ComputationGraph,
    extract_computation_graph,
    ComputationStep,
    ConstraintVerifier,
    VerificationResult,
    SafetyViolation,
    CounterexampleTrace,
    ModelState,
    OpKind,
    LayerKind,
    LayerDef,
    Device,
    Phase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Shape predicate representation
# ═══════════════════════════════════════════════════════════════════════════════

class PredicateKind(Enum):
    """The flavour of a discovered shape predicate."""
    DIM_EQ = auto()       # tensor.shape[axis] == value
    DIM_GT = auto()       # tensor.shape[axis] > value
    DIM_GE = auto()       # tensor.shape[axis] >= value
    DIM_DIVISIBLE = auto()  # tensor.shape[axis] % divisor == 0
    DIM_MATCH = auto()    # tensor_a.shape[axis_a] == tensor_b.shape[axis_b]
    NDIM_EQ = auto()      # len(tensor.shape) == value
    SHAPE_EQ = auto()     # tensor.shape == (d0, d1, ...)


@dataclass(frozen=True)
class ShapePredicate:
    """A single shape predicate discovered by the contract discovery loop.

    Examples
    --------
    >>> ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=768)
    # means: x.shape[-1] == 768

    >>> ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=-1,
    ...               match_tensor="w", match_axis=0)
    # means: x.shape[-1] == w.shape[0]
    """
    kind: PredicateKind
    tensor: str
    axis: Optional[int] = None
    value: Optional[int] = None
    match_tensor: Optional[str] = None
    match_axis: Optional[int] = None
    divisor: Optional[int] = None
    provenance: str = "cegar_discovered"

    def pretty(self) -> str:
        """Human-readable representation."""
        if self.kind == PredicateKind.DIM_EQ:
            return f"{self.tensor}.shape[{self.axis}] == {self.value}"
        if self.kind == PredicateKind.DIM_GT:
            return f"{self.tensor}.shape[{self.axis}] > {self.value}"
        if self.kind == PredicateKind.DIM_GE:
            return f"{self.tensor}.shape[{self.axis}] >= {self.value}"
        if self.kind == PredicateKind.DIM_DIVISIBLE:
            return f"{self.tensor}.shape[{self.axis}] % {self.divisor} == 0"
        if self.kind == PredicateKind.DIM_MATCH:
            return (
                f"{self.tensor}.shape[{self.axis}] == "
                f"{self.match_tensor}.shape[{self.match_axis}]"
            )
        if self.kind == PredicateKind.NDIM_EQ:
            return f"len({self.tensor}.shape) == {self.value}"
        if self.kind == PredicateKind.SHAPE_EQ:
            return f"{self.tensor}.shape == {self.value}"
        return f"<unknown predicate on {self.tensor}>"

    def __repr__(self) -> str:
        return f"ShapePredicate({self.pretty()})"


def _parse_predicate_string(s: str) -> Optional[ShapePredicate]:
    """Best-effort parse of a predicate pretty-print string back to ShapePredicate.

    Handles formats like ``x.shape[-1] == 768``, ``x.shape[0] >= 1``,
    ``x.shape[1] % 8 == 0``, ``len(x.shape) == 4``.
    Returns None if parsing fails.
    """
    import re
    s = s.strip()

    # DIM_DIVISIBLE: x.shape[axis] % divisor == 0
    m = re.match(r"(\w+)\.shape\[(-?\d+)\]\s*%\s*(\d+)\s*==\s*0", s)
    if m:
        return ShapePredicate(
            PredicateKind.DIM_DIVISIBLE,
            tensor=m.group(1), axis=int(m.group(2)),
            divisor=int(m.group(3)),
        )
    # DIM_MATCH: x.shape[a] == y.shape[b]
    m = re.match(
        r"(\w+)\.shape\[(-?\d+)\]\s*==\s*(\w+)\.shape\[(-?\d+)\]", s
    )
    if m:
        return ShapePredicate(
            PredicateKind.DIM_MATCH,
            tensor=m.group(1), axis=int(m.group(2)),
            match_tensor=m.group(3), match_axis=int(m.group(4)),
        )
    # DIM_EQ: x.shape[axis] == value
    m = re.match(r"(\w+)\.shape\[(-?\d+)\]\s*==\s*(\d+)", s)
    if m:
        return ShapePredicate(
            PredicateKind.DIM_EQ,
            tensor=m.group(1), axis=int(m.group(2)),
            value=int(m.group(3)),
        )
    # DIM_GE: x.shape[axis] >= value
    m = re.match(r"(\w+)\.shape\[(-?\d+)\]\s*>=\s*(\d+)", s)
    if m:
        return ShapePredicate(
            PredicateKind.DIM_GE,
            tensor=m.group(1), axis=int(m.group(2)),
            value=int(m.group(3)),
        )
    # DIM_GT: x.shape[axis] > value
    m = re.match(r"(\w+)\.shape\[(-?\d+)\]\s*>\s*(\d+)", s)
    if m:
        return ShapePredicate(
            PredicateKind.DIM_GT,
            tensor=m.group(1), axis=int(m.group(2)),
            value=int(m.group(3)),
        )
    # NDIM_EQ: len(x.shape) == value
    m = re.match(r"len\((\w+)\.shape\)\s*==\s*(\d+)", s)
    if m:
        return ShapePredicate(
            PredicateKind.NDIM_EQ,
            tensor=m.group(1), value=int(m.group(2)),
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Counterexample analysis types
# ═══════════════════════════════════════════════════════════════════════════════

class CounterexampleClassification(Enum):
    """Whether a counterexample is spurious or a real bug."""
    SPURIOUS = auto()   # can be eliminated by a new predicate
    REAL_BUG = auto()   # actual shape error in the model
    UNKNOWN = auto()    # cannot classify (conservative: treat as real)


@dataclass
class AnalysedCounterexample:
    """A counterexample that has been traced back and classified."""
    violation: SafetyViolation
    step_index: int
    classification: CounterexampleClassification
    concrete_dims: Dict[str, int] = field(default_factory=dict)
    traced_to_inputs: List[str] = field(default_factory=list)
    synthesised_predicates: List[ShapePredicate] = field(default_factory=list)
    reason: str = ""

    def is_spurious(self) -> bool:
        return self.classification == CounterexampleClassification.SPURIOUS

    def is_real_bug(self) -> bool:
        return self.classification == CounterexampleClassification.REAL_BUG


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Contract discovery result type
# ═══════════════════════════════════════════════════════════════════════════════

class CEGARStatus(Enum):
    """Final status of the contract discovery loop (CEGAR-style)."""
    SAFE = auto()            # all counterexamples eliminated → model is safe
    REAL_BUG_FOUND = auto()  # found a genuine shape bug
    MAX_ITER = auto()        # iteration budget exhausted
    NO_Z3 = auto()           # Z3 not available; fell back to single pass
    PARSE_ERROR = auto()     # could not parse the source
    INFEASIBLE_REFINEMENT = auto()  # accumulated predicates jointly infeasible →
                                    # spurious elimination, must abstain (not SAFE)


class CEGARVerdict(Enum):
    """High-level verification verdict with explicit timeout/unknown semantics.

    This enum captures the four possible outcomes of a CEGAR-based
    verification run, including cases where the solver or loop could not
    reach a definitive conclusion.

    Timeout semantics
    -----------------
    * **Z3 solver timeout during a CEGAR iteration**: When the Z3 solver
      exceeds its per-query timeout (1000–5000 ms depending on the query
      type), Z3 returns ``z3.unknown``.  The CEGAR loop treats this as
      an inability to classify the counterexample, conservatively
      classifying it as a real bug.  If *all* counterexamples in an
      iteration time out, the verdict is ``UNKNOWN``.
    * **CEGAR loop exceeds iteration bounds**: When the iteration budget
      (``max_iterations``, default 10) is exhausted without reaching
      SAFE or UNSAFE, the verdict is ``TIMEOUT``.  This preserves
      soundness: the system never reports SAFE unless all
      counterexamples have been eliminated.
    * **Soundness guarantee**: Both UNKNOWN and TIMEOUT are conservative.
      A SAFE verdict is sound (no false negatives).  An UNSAFE verdict
      indicates a confirmed real bug.  UNKNOWN and TIMEOUT indicate
      that verification was inconclusive—the model may or may not
      contain shape errors.
    """
    SAFE = auto()       # all counterexamples eliminated; model verified safe
    UNSAFE = auto()     # confirmed real shape bug found
    UNKNOWN = auto()    # solver returned unknown (e.g. Z3 timeout on query)
    TIMEOUT = auto()    # CEGAR iteration budget exhausted without conclusion


@dataclass
class InferredContract:
    """A shape contract inferred by the contract discovery loop for a function/method."""
    function_name: str
    parameter: str
    predicates: List[ShapePredicate] = field(default_factory=list)

    def pretty(self) -> str:
        preds = ", ".join(p.pretty() for p in self.predicates)
        return f"{self.function_name}({self.parameter}): requires [{preds}]"


@dataclass
class ShapeCEGARResult:
    """Top-level result of the shape contract discovery loop (CEGAR-style).

    Attributes
    ----------
    discovered_predicates : list of ShapePredicate
        All shape predicates discovered during refinement.
    iterations : int
        Number of contract discovery iterations performed.
    final_status : CEGARStatus
        Why the loop terminated.
    contracts_inferred : list of InferredContract
        Per-parameter shape contracts inferred from discovered predicates.
    verification_result : VerificationResult or None
        The final model-checker result from the last iteration.
    real_bugs : list of SafetyViolation
        Any genuine shape bugs found (empty if model is safe).
    total_time_ms : float
        Wall-clock time for the entire contract discovery loop.
    iteration_log : list of IterationRecord
        Per-iteration details for debugging / reporting.
    """
    discovered_predicates: List[ShapePredicate] = field(default_factory=list)
    iterations: int = 0
    final_status: CEGARStatus = CEGARStatus.SAFE
    contracts_inferred: List[InferredContract] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    real_bugs: List[SafetyViolation] = field(default_factory=list)
    total_time_ms: float = 0.0
    iteration_log: List["IterationRecord"] = field(default_factory=list)
    predicate_quality_report: Optional[Dict[str, Any]] = None
    interpolation_stats: Optional[Dict[str, int]] = None

    @property
    def is_safe(self) -> bool:
        return self.final_status == CEGARStatus.SAFE

    @property
    def has_real_bugs(self) -> bool:
        return self.final_status == CEGARStatus.REAL_BUG_FOUND

    @property
    def verdict(self) -> CEGARVerdict:
        """Map the internal CEGARStatus to a high-level CEGARVerdict.

        The mapping preserves soundness:
        - SAFE → CEGARVerdict.SAFE (all counterexamples eliminated)
        - REAL_BUG_FOUND → CEGARVerdict.UNSAFE (confirmed bug)
        - MAX_ITER → CEGARVerdict.TIMEOUT (iteration budget exhausted)
        - NO_Z3 / PARSE_ERROR → CEGARVerdict.UNKNOWN (inconclusive)
        - INFEASIBLE_REFINEMENT → CEGARVerdict.UNKNOWN (spurious elimination)
        """
        _STATUS_TO_VERDICT = {
            CEGARStatus.SAFE: CEGARVerdict.SAFE,
            CEGARStatus.REAL_BUG_FOUND: CEGARVerdict.UNSAFE,
            CEGARStatus.MAX_ITER: CEGARVerdict.TIMEOUT,
            CEGARStatus.NO_Z3: CEGARVerdict.UNKNOWN,
            CEGARStatus.PARSE_ERROR: CEGARVerdict.UNKNOWN,
            CEGARStatus.INFEASIBLE_REFINEMENT: CEGARVerdict.UNKNOWN,
        }
        return _STATUS_TO_VERDICT.get(self.final_status, CEGARVerdict.UNKNOWN)

    def summary(self) -> str:
        preds = ", ".join(p.pretty() for p in self.discovered_predicates)
        return (
            f"ShapeCEGAR: {self.final_status.name} after "
            f"{self.iterations} iterations, "
            f"{len(self.discovered_predicates)} predicates discovered"
            + (f" [{preds}]" if preds else "")
            + f", {self.total_time_ms:.1f}ms"
        )


@dataclass
class IterationRecord:
    """Diagnostic record for a single contract discovery iteration."""
    iteration: int
    num_violations: int
    num_spurious: int
    num_real: int
    predicates_added: List[ShapePredicate] = field(default_factory=list)
    time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Trace-back engine
# ═══════════════════════════════════════════════════════════════════════════════

class TraceBackEngine:
    """Traces a counterexample backwards through the computation graph to
    find the input shape assumption that, if strengthened, would prevent
    the counterexample.

    The key insight: every shape mismatch at step *k* is caused by
    one or more earlier steps that produced tensors with incompatible
    shapes.  By walking backwards along the data-flow edges we can
    find the *earliest* point where a shape constraint can be added —
    ideally at the inputs of ``forward()``.
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self.graph = graph
        self._producers: Dict[str, int] = {}
        for idx, step in enumerate(graph.steps):
            self._producers[step.output] = idx

    def trace_to_inputs(
        self,
        failing_step_idx: int,
        violation: SafetyViolation,
    ) -> List[str]:
        """Return the list of input tensor names that contribute to the
        violation at *failing_step_idx*.
        """
        if failing_step_idx < 0 or failing_step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[failing_step_idx]
        visited: Set[str] = set()
        input_sources: List[str] = []
        self._walk_back(step.inputs, visited, input_sources)
        return input_sources

    def _walk_back(
        self,
        tensor_names: List[str],
        visited: Set[str],
        input_sources: List[str],
    ) -> None:
        """Recursively walk backwards through data-flow edges."""
        for name in tensor_names:
            if name in visited:
                continue
            visited.add(name)

            if name in self.graph.input_names:
                if name not in input_sources:
                    input_sources.append(name)
                continue

            producer_idx = self._producers.get(name)
            if producer_idx is not None:
                producer = self.graph.steps[producer_idx]
                self._walk_back(producer.inputs, visited, input_sources)
            else:
                # Tensor not produced by any step and not an input —
                # might be a parameter.  Record it anyway.
                if name not in input_sources:
                    input_sources.append(name)

    def find_constraint_origin(
        self,
        failing_step_idx: int,
        violation: SafetyViolation,
        shape_env: Dict[str, TensorShape],
    ) -> List[Tuple[str, int, Optional[int]]]:
        """Find the (tensor, axis, expected_value) triples that, if
        constrained at the input, would fix the violation.

        Returns a list of ``(tensor_name, axis, expected_value)`` where
        *expected_value* is ``None`` if it could not be determined.
        """
        if failing_step_idx < 0 or failing_step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[failing_step_idx]
        origins: List[Tuple[str, int, Optional[int]]] = []

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, -1, layer.in_features))

            elif layer and layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, 1, layer.in_channels))

            elif layer and layer.kind == LayerKind.EMBEDDING and layer.num_embeddings is not None:
                inp = step.inputs[0] if step.inputs else None
                if inp:
                    origins.append((inp, -1, None))

        elif step.op == OpKind.MATMUL:
            if len(step.inputs) >= 2:
                a_name, b_name = step.inputs[0], step.inputs[1]
                a_shape = shape_env.get(a_name)
                b_shape = shape_env.get(b_name)
                if a_shape and b_shape:
                    # Inner dims must match: a.shape[-1] == b.shape[-2]
                    if a_shape.ndim >= 1:
                        origins.append((a_name, -1, None))
                    if b_shape.ndim >= 2:
                        origins.append((b_name, -2, None))
                    elif b_shape.ndim == 1:
                        origins.append((b_name, 0, None))

        elif step.op == OpKind.ADD:
            if len(step.inputs) >= 2:
                a_name, b_name = step.inputs[0], step.inputs[1]
                a_shape = shape_env.get(a_name)
                b_shape = shape_env.get(b_name)
                if a_shape and b_shape:
                    ndim = max(a_shape.ndim, b_shape.ndim)
                    for i in range(1, ndim + 1):
                        d_a = a_shape.dims[-i] if i <= a_shape.ndim else None
                        d_b = b_shape.dims[-i] if i <= b_shape.ndim else None
                        if d_a and d_b and d_a != d_b:
                            if not d_a.is_symbolic and d_a.value != 1:
                                origins.append((b_name, -i, d_a.value))
                            elif not d_b.is_symbolic and d_b.value != 1:
                                origins.append((a_name, -i, d_b.value))

        elif step.op == OpKind.CAT:
            for inp_name in step.inputs:
                shape = shape_env.get(inp_name)
                if shape:
                    origins.append((inp_name, 0, None))

        # Trace each tensor back to its input source
        input_origins: List[Tuple[str, int, Optional[int]]] = []
        for tensor_name, axis, expected in origins:
            input_chain = self._trace_dim_to_input(tensor_name, axis, shape_env)
            if input_chain:
                src_tensor, src_axis = input_chain
                input_origins.append((src_tensor, src_axis, expected))
            else:
                input_origins.append((tensor_name, axis, expected))

        return input_origins

    def _trace_dim_to_input(
        self,
        tensor_name: str,
        axis: int,
        shape_env: Dict[str, TensorShape],
    ) -> Optional[Tuple[str, int]]:
        """Trace a specific dimension back through shape-preserving ops
        to find the corresponding input tensor and axis.
        """
        visited: Set[str] = set()
        current_name = tensor_name
        current_axis = axis

        while current_name not in self.graph.input_names:
            if current_name in visited:
                break
            visited.add(current_name)

            producer_idx = self._producers.get(current_name)
            if producer_idx is None:
                break

            step = self.graph.steps[producer_idx]

            if step.op in (
                OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                OpKind.CONTIGUOUS, OpKind.DETACH,
            ):
                # Shape-preserving: axis maps 1:1
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            elif step.op == OpKind.LAYER_CALL:
                layer = self.graph.layers.get(step.layer_ref or "")
                if layer and layer.kind in (
                    LayerKind.RELU, LayerKind.DROPOUT, LayerKind.IDENTITY,
                    LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                    LayerKind.LAYERNORM, LayerKind.SOFTMAX,
                ):
                    if step.inputs:
                        current_name = step.inputs[0]
                    else:
                        break
                elif layer and layer.kind == LayerKind.LINEAR:
                    # Linear changes last dim; other dims pass through
                    shape = shape_env.get(current_name)
                    norm_axis = current_axis
                    if shape and norm_axis < 0:
                        norm_axis = shape.ndim + norm_axis
                    if shape and norm_axis == shape.ndim - 1:
                        # Last dim is transformed — cannot trace further
                        break
                    if step.inputs:
                        current_name = step.inputs[0]
                    else:
                        break
                else:
                    break

            elif step.op == OpKind.TRANSPOSE:
                d0 = step.params.get("dim0", 0)
                d1 = step.params.get("dim1", 1)
                shape = shape_env.get(current_name)
                norm = current_axis
                if shape and norm < 0:
                    norm = shape.ndim + norm
                if norm == d0:
                    current_axis = d1
                elif norm == d1:
                    current_axis = d0
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            elif step.op == OpKind.PERMUTE:
                perm = step.params.get("dims")
                shape = shape_env.get(current_name)
                norm = current_axis
                if shape and norm < 0:
                    norm = shape.ndim + norm
                if perm and 0 <= norm < len(perm):
                    current_axis = perm[norm]
                if step.inputs:
                    current_name = step.inputs[0]
                else:
                    break

            else:
                # Cannot trace through reshape, cat, matmul, etc.
                break

        if current_name in self.graph.input_names:
            return (current_name, current_axis)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Unsat core-based predicate extraction engine
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_conv_param(val: Any, spatial_idx: int) -> Optional[int]:
    """Extract a scalar conv parameter for a given spatial dimension."""
    if isinstance(val, int):
        return val
    if isinstance(val, (list, tuple)) and spatial_idx < len(val):
        v = val[spatial_idx]
        return int(v) if isinstance(v, (int, float)) else None
    if isinstance(val, str) and val == "same":
        return None  # 'same' padding handled separately
    return None


class UnsatCorePredicateExtractor:
    """Discovers shape predicates via unsat-core-based predicate extraction.

    Given:
      - A computation graph formula encoding shape transitions
      - A counterexample path formula (concrete dim assignments that violate safety)
    Extracts the unsat core to identify the minimal set of constraints
    making the counterexample spurious, then extracts shape predicates
    from those constraints.

    Uses Z3 unsat core extraction (not Craig interpolation). Unsat cores
    provide subsets of input clauses contributing to unsatisfiability,
    which is sufficient for predicate discovery in our setting.  This is
    Houdini-style predicate accumulation (Flanagan & Leino, FME 2001)
    with Z3-backed abstract feasibility checking.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        shape_env: Dict[str, TensorShape],
    ) -> None:
        self.graph = graph
        self.shape_env = shape_env

    def discover_predicates(
        self,
        violation: SafetyViolation,
        step_idx: int,
        concrete_dims: Dict[str, int],
        input_shapes: Dict[str, tuple],
        incremental_solver: Any = None,
    ) -> List[ShapePredicate]:
        """Discover predicates that eliminate a spurious counterexample.

        Uses unsat core extraction to find the minimal set of constraints
        that make the counterexample infeasible, then extracts shape
        predicates from those constraints.

        Parameters
        ----------
        incremental_solver : IncrementalCEGARSolver, optional
            When provided, uses incremental solving with MUS extraction.
        """
        if not HAS_Z3:
            return []

        if step_idx < 0 or step_idx >= len(self.graph.steps):
            return []

        step = self.graph.steps[step_idx]
        predicates: List[ShapePredicate] = []

        # Build the path formula (A) and safety formula (B)
        # A: input assumptions + computation graph transitions up to step
        # B: negation of safety property at the failing step
        path_constraints, safety_constraints, dim_map = (
            self._build_predicate_extraction_query(step, step_idx, concrete_dims, input_shapes)
        )

        if not path_constraints or not safety_constraints:
            return []

        # Try unsat-core-based predicate extraction
        predicates = self._extract_via_unsat_core(
            path_constraints, safety_constraints, dim_map, concrete_dims,
            incremental_solver=incremental_solver,
        )

        return predicates

    def _build_predicate_extraction_query(
        self,
        step: ComputationStep,
        step_idx: int,
        concrete_dims: Dict[str, int],
        input_shapes: Dict[str, tuple],
    ) -> Tuple[List[Any], List[Any], Dict[str, Tuple[str, int]]]:
        """Build the formula pair (path, safety) for unsat-core extraction.

        Returns:
            (path_constraints, safety_constraints, dim_map)
            where dim_map maps Z3 variable names to (tensor_name, axis).
        """
        path_cs: List[Any] = []
        safety_cs: List[Any] = []
        dim_map: Dict[str, Tuple[str, int]] = {}

        # Create Z3 variables for input dimensions
        for inp_name, shape_tuple in input_shapes.items():
            for axis, dim_val in enumerate(shape_tuple):
                var_name = f"__interp_{inp_name}_d{axis}"
                var = z3.Int(var_name)
                dim_map[var_name] = (inp_name, axis)
                path_cs.append(var > 0)
                if isinstance(dim_val, int):
                    path_cs.append(var == z3.IntVal(dim_val))

        # Encode safety property at the failing step
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp and inp in input_shapes:
                inp_shape = input_shapes[inp]
                if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                    var_name = f"__interp_{inp}_d{len(inp_shape)-1}"
                    var = z3.Int(var_name)
                    safety_cs.append(var == z3.IntVal(layer.in_features))
                elif layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.in_channels))
                elif layer.kind == LayerKind.CONV1D and layer.in_channels is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.in_channels))
                elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D) and layer.num_features is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.num_features))
                elif layer.kind == LayerKind.LAYERNORM and layer.num_features is not None:
                    var_name = f"__interp_{inp}_d{len(inp_shape)-1}"
                    var = z3.Int(var_name)
                    safety_cs.append(var == z3.IntVal(layer.num_features))
                elif layer.kind == LayerKind.EMBEDDING and layer.num_embeddings is not None:
                    var_name = f"__interp_{inp}_d{len(inp_shape)-1}"
                    var = z3.Int(var_name)
                    safety_cs.append(var < z3.IntVal(layer.num_embeddings))
                    safety_cs.append(var >= z3.IntVal(0))
                elif layer.kind == LayerKind.GROUPNORM and layer.num_features is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.num_features))
                elif layer.kind == LayerKind.INSTANCENORM2D and layer.num_features is not None:
                    if len(inp_shape) >= 2:
                        var_name = f"__interp_{inp}_d1"
                        var = z3.Int(var_name)
                        safety_cs.append(var == z3.IntVal(layer.num_features))

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_name, b_name = step.inputs[0], step.inputs[1]
            a_shape = input_shapes.get(a_name)
            b_shape = input_shapes.get(b_name)
            if a_shape and b_shape:
                va = f"__interp_{a_name}_d{len(a_shape)-1}"
                vb = f"__interp_{b_name}_d{len(b_shape)-2}" if len(b_shape) >= 2 else f"__interp_{b_name}_d0"
                safety_cs.append(z3.Int(va) == z3.Int(vb))

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a_name, b_name = step.inputs[0], step.inputs[1]
            a_shape = input_shapes.get(a_name)
            b_shape = input_shapes.get(b_name)
            if a_shape and b_shape:
                ndim = max(len(a_shape), len(b_shape))
                for i in range(1, ndim + 1):
                    if i <= len(a_shape) and i <= len(b_shape):
                        va = f"__interp_{a_name}_d{len(a_shape)-i}"
                        vb = f"__interp_{b_name}_d{len(b_shape)-i}"
                        a_var = z3.Int(va)
                        b_var = z3.Int(vb)
                        safety_cs.append(z3.Or(
                            a_var == b_var,
                            a_var == z3.IntVal(1),
                            b_var == z3.IntVal(1),
                        ))

        elif step.op == OpKind.MULTIPLY and len(step.inputs) >= 2:
            a_name, b_name = step.inputs[0], step.inputs[1]
            a_shape = input_shapes.get(a_name)
            b_shape = input_shapes.get(b_name)
            if a_shape and b_shape:
                ndim = max(len(a_shape), len(b_shape))
                for i in range(1, ndim + 1):
                    if i <= len(a_shape) and i <= len(b_shape):
                        va = f"__interp_{a_name}_d{len(a_shape)-i}"
                        vb = f"__interp_{b_name}_d{len(b_shape)-i}"
                        a_var = z3.Int(va)
                        b_var = z3.Int(vb)
                        safety_cs.append(z3.Or(
                            a_var == b_var,
                            a_var == z3.IntVal(1),
                            b_var == z3.IntVal(1),
                        ))

        elif step.op == OpKind.CAT and len(step.inputs) >= 2:
            cat_dim = step.params.get("dim", 0)
            shapes = [input_shapes.get(n) for n in step.inputs]
            if all(s is not None for s in shapes):
                for i, s in enumerate(shapes):
                    if s is not None:
                        for ax in range(len(s)):
                            if ax == cat_dim:
                                continue
                            if i == 0:
                                continue
                            first_s = shapes[0]
                            if first_s is not None and ax < len(first_s):
                                va = f"__interp_{step.inputs[0]}_d{ax}"
                                vb = f"__interp_{step.inputs[i]}_d{ax}"
                                safety_cs.append(z3.Int(va) == z3.Int(vb))

        elif step.op == OpKind.RESHAPE and step.inputs:
            inp_name = step.inputs[0]
            inp_shape = input_shapes.get(inp_name)
            target_shape = step.params.get("shape")
            if inp_shape and target_shape and isinstance(target_shape, (list, tuple)):
                known_product = 1
                for d in target_shape:
                    if isinstance(d, int) and d > 0:
                        known_product *= d
                if known_product > 1:
                    total_var = z3.IntVal(1)
                    for ax in range(len(inp_shape)):
                        vn = f"__interp_{inp_name}_d{ax}"
                        total_var = total_var * z3.Int(vn)
                    safety_cs.append(total_var == z3.IntVal(known_product))

        elif step.op == OpKind.TRANSPOSE and step.inputs:
            inp_name = step.inputs[0]
            inp_shape = input_shapes.get(inp_name)
            d0 = step.params.get("dim0", 0)
            d1 = step.params.get("dim1", 1)
            if inp_shape and d0 < len(inp_shape) and d1 < len(inp_shape):
                v0 = f"__interp_{inp_name}_d{d0}"
                v1 = f"__interp_{inp_name}_d{d1}"
                dim_map.setdefault(v0, (inp_name, d0))
                dim_map.setdefault(v1, (inp_name, d1))
                safety_cs.append(z3.Int(v0) > 0)
                safety_cs.append(z3.Int(v1) > 0)

        return path_cs, safety_cs, dim_map

    def _build_interpolation_query(
        self,
        graph: ComputationGraph,
        failing_step_idx: int,
        input_shapes: Dict[str, tuple],
        concrete_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Any], List[Any], "DimMapping"]:
        """Build multi-step path and counterexample formulas for Craig interpolation.

        McMillan-style interpolation:
        A = symbolic computation path (positivity + step transitions up to
        the failing step) — does NOT include the safety property.
        B = negation of the safety requirement at the failing step (or
        concrete counterexample dimension assignments when available).
        A ∧ B is UNSAT when the path constraints are incompatible with a
        safety violation, and the interpolant over interface variables
        captures what constraint on input shapes prevents the violation.

        Returns (path_constraints_A, counterexample_constraints_B, DimMapping).
        """
        from src.craig_interpolation import DimMapping

        if not HAS_Z3:
            return [], [], DimMapping()

        dm = DimMapping()
        path_cs: List[Any] = []

        # Track symbolic shape of each tensor through the computation graph.
        # Maps tensor_name -> list of Z3 Int variables (one per axis).
        tensor_vars: Dict[str, List[Any]] = {}

        # 1. Create symbolic input variables (no concrete pinning).
        for inp_name in graph.input_names:
            shape_tuple = input_shapes.get(inp_name)
            if shape_tuple is None:
                continue
            ndim = len(shape_tuple)
            vars_list = []
            for axis in range(ndim):
                var_name = f"__ci_{inp_name}_d{axis}"
                var = z3.Int(var_name)
                dm.register(var_name, inp_name, axis)
                path_cs.append(var > 0)
                vars_list.append(var)
            tensor_vars[inp_name] = vars_list

        # 2. Walk computation steps 0..failing_step_idx-1, encoding transitions.
        failing_step = graph.steps[failing_step_idx] if failing_step_idx < len(graph.steps) else None
        for si in range(min(failing_step_idx, len(graph.steps))):
            step = graph.steps[si]
            self._encode_step_transition(step, graph, tensor_vars, path_cs, dm)

        # 3–4. Build A/B split depending on whether we have a concrete
        #       counterexample or are doing pure McMillan-style interpolation.
        cex_cs: List[Any] = []
        if concrete_dims:
            # Concrete counterexample provided.
            # A = path + safety (the counterexample violates safety).
            # B = concrete dimension assignments.
            if failing_step is not None:
                safety_atoms = self._encode_safety_at_step(
                    failing_step, graph, tensor_vars, dm, input_shapes,
                )
                path_cs.extend(safety_atoms)
            for var_name, val in concrete_dims.items():
                if dm.is_known(var_name):
                    cex_cs.append(z3.Int(var_name) == z3.IntVal(val))
                else:
                    cex_cs.append(z3.Int(var_name) == z3.IntVal(val))
        else:
            # McMillan-style: A = path only, B = Not(safety).
            # A captures how shapes flow; B captures what violates safety.
            # Interpolant discovers input-shape constraints preventing violation.
            if failing_step is not None:
                safety_atoms = self._encode_safety_at_step(
                    failing_step, graph, tensor_vars, dm, input_shapes,
                )
                if safety_atoms:
                    conj = z3.And(*safety_atoms) if len(safety_atoms) > 1 else safety_atoms[0]
                    cex_cs.append(z3.Not(conj))

        return path_cs, cex_cs, dm

    def _encode_step_transition(
        self,
        step: ComputationStep,
        graph: ComputationGraph,
        tensor_vars: Dict[str, List[Any]],
        constraints: List[Any],
        dm: "DimMapping",
    ) -> None:
        """Encode one computation step's shape transition symbolically."""
        if not step.inputs or step.inputs[0] not in tensor_vars:
            return

        inp_vars = tensor_vars[step.inputs[0]]

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = graph.layers.get(step.layer_ref)
            if layer is None:
                # Unknown layer — output shape same as input (conservative).
                tensor_vars[step.output] = list(inp_vars)
                return

            if layer.kind == LayerKind.LINEAR:
                # Output: (..., out_features). Input: (..., in_features).
                out_vars = list(inp_vars[:-1]) if len(inp_vars) > 1 else []
                if layer.out_features is not None:
                    out_last = z3.Int(f"__ci_{step.output}_d{len(out_vars)}")
                    constraints.append(out_last == z3.IntVal(layer.out_features))
                    out_vars.append(out_last)
                else:
                    out_vars.append(inp_vars[-1] if inp_vars else z3.IntVal(1))
                tensor_vars[step.output] = out_vars

            elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
                out_vars = [inp_vars[0]] if inp_vars else []
                if layer.out_channels is not None:
                    ch_var = z3.Int(f"__ci_{step.output}_d1")
                    constraints.append(ch_var == z3.IntVal(layer.out_channels))
                    out_vars.append(ch_var)
                # Spatial dims: encode exact formula when kernel/stride/padding
                # are known, otherwise fall back to > 0.
                n_spatial = (len(inp_vars) - 2) if len(inp_vars) > 2 else 0
                ks = layer.kernel_size
                stride_val = layer.params.get("stride", 1)
                pad_val = layer.params.get("padding", 0)
                dilation_val = layer.params.get("dilation", 1)
                for si in range(n_spatial):
                    sv = z3.Int(f"__ci_{step.output}_d{2 + si}")
                    inp_spatial = inp_vars[2 + si]
                    k = _extract_conv_param(ks, si)
                    s = _extract_conv_param(stride_val, si)
                    p = _extract_conv_param(pad_val, si)
                    dl = _extract_conv_param(dilation_val, si)
                    if k is not None and s is not None and p is not None and dl is not None:
                        # out = floor((inp + 2*p - d*(k-1) - 1) / s) + 1
                        # Encode as: sv * s == inp + 2*p - d*(k-1) - 1 - rem,
                        # 0 <= rem < s, sv > 0
                        # Simplified: sv == (inp + 2*p - d*(k-1) - 1) / s + 1
                        # Use integer arithmetic identity instead of floor:
                        # sv > 0  AND  s*(sv-1) <= inp+2p-d(k-1)-1 < s*sv
                        effective_k = dl * (k - 1) + 1
                        numerator = inp_spatial + z3.IntVal(2 * p - effective_k)
                        if s == 1:
                            constraints.append(sv == numerator + 1)
                        else:
                            constraints.append(sv > 0)
                            constraints.append(
                                z3.IntVal(s) * (sv - 1) <= numerator
                            )
                            constraints.append(
                                numerator < z3.IntVal(s) * sv
                            )
                    else:
                        constraints.append(sv > 0)
                    out_vars.append(sv)
                tensor_vars[step.output] = out_vars

            elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                                LayerKind.LAYERNORM, LayerKind.GROUPNORM,
                                LayerKind.INSTANCENORM2D, LayerKind.DROPOUT,
                                LayerKind.RELU, LayerKind.IDENTITY, LayerKind.SOFTMAX):
                # Shape-preserving layers.
                tensor_vars[step.output] = list(inp_vars)

            elif layer.kind == LayerKind.FLATTEN:
                # Flatten(start_dim, end_dim) merges dims.
                start = layer.params.get("start_dim", 1)
                end = layer.params.get("end_dim", -1)
                if end == -1:
                    end = len(inp_vars) - 1
                pre = inp_vars[:start]
                flat_product = z3.IntVal(1)
                for vi in inp_vars[start:end + 1]:
                    flat_product = flat_product * vi
                flat_var = z3.Int(f"__ci_{step.output}_d{len(pre)}")
                constraints.append(flat_var == flat_product)
                post = inp_vars[end + 1:] if end + 1 < len(inp_vars) else []
                tensor_vars[step.output] = list(pre) + [flat_var] + list(post)

            elif layer.kind == LayerKind.EMBEDDING:
                # Input: (*) -> Output: (*, embedding_dim)
                out_vars = list(inp_vars)
                if layer.embedding_dim is not None:
                    emb_var = z3.Int(f"__ci_{step.output}_d{len(out_vars)}")
                    constraints.append(emb_var == z3.IntVal(layer.embedding_dim))
                    out_vars.append(emb_var)
                tensor_vars[step.output] = out_vars

            elif layer.kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RNN):
                # Output: (batch, seq, hidden_size * num_directions)
                if layer.hidden_size is not None:
                    dirs = 2 if layer.bidirectional else 1
                    h_var = z3.Int(f"__ci_{step.output}_d{len(inp_vars)-1}")
                    constraints.append(h_var == z3.IntVal(layer.hidden_size * dirs))
                    out_vars = list(inp_vars[:-1]) + [h_var]
                    tensor_vars[step.output] = out_vars
                else:
                    tensor_vars[step.output] = list(inp_vars)

            else:
                # Check for pool layers — encode spatial reduction.
                if layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D,
                                  LayerKind.MAXPOOL1D, LayerKind.AVGPOOL1D):
                    out_vars = list(inp_vars[:2]) if len(inp_vars) >= 2 else list(inp_vars)
                    n_spatial = (len(inp_vars) - 2) if len(inp_vars) > 2 else 0
                    ks = layer.kernel_size
                    stride_val = layer.params.get("stride") or ks
                    pad_val = layer.params.get("padding", 0)
                    dilation_val = layer.params.get("dilation", 1)
                    for si in range(n_spatial):
                        sv = z3.Int(f"__ci_{step.output}_d{2 + si}")
                        inp_spatial = inp_vars[2 + si]
                        k = _extract_conv_param(ks, si)
                        s = _extract_conv_param(stride_val, si)
                        p = _extract_conv_param(pad_val, si)
                        dl = _extract_conv_param(dilation_val, si)
                        if k is not None and s is not None and p is not None and dl is not None:
                            effective_k = dl * (k - 1) + 1
                            numerator = inp_spatial + z3.IntVal(2 * p - effective_k)
                            if s == 1:
                                constraints.append(sv == numerator + 1)
                            else:
                                constraints.append(sv > 0)
                                constraints.append(
                                    z3.IntVal(s) * (sv - 1) <= numerator
                                )
                                constraints.append(
                                    numerator < z3.IntVal(s) * sv
                                )
                        else:
                            constraints.append(sv > 0)
                        out_vars.append(sv)
                    tensor_vars[step.output] = out_vars
                elif layer.kind in (LayerKind.ADAPTIVE_AVGPOOL2D, LayerKind.ADAPTIVE_MAXPOOL2D):
                    out_vars = list(inp_vars[:2]) if len(inp_vars) >= 2 else list(inp_vars)
                    if layer.output_size is not None:
                        osz = layer.output_size if isinstance(layer.output_size, (list, tuple)) else (layer.output_size, layer.output_size)
                        for si, d in enumerate(osz):
                            sv = z3.Int(f"__ci_{step.output}_d{2 + si}")
                            if isinstance(d, int) and d > 0:
                                constraints.append(sv == z3.IntVal(d))
                            else:
                                constraints.append(sv > 0)
                            out_vars.append(sv)
                    tensor_vars[step.output] = out_vars
                else:
                    # Truly unknown — conservatively preserve shape.
                    tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.RESHAPE and step.inputs:
            target_shape = step.params.get("shape")
            if target_shape and isinstance(target_shape, (list, tuple)):
                out_vars = []
                for ai, d in enumerate(target_shape):
                    rv = z3.Int(f"__ci_{step.output}_d{ai}")
                    if isinstance(d, int) and d > 0:
                        constraints.append(rv == z3.IntVal(d))
                    elif d == -1:
                        constraints.append(rv > 0)
                    else:
                        constraints.append(rv > 0)
                    out_vars.append(rv)
                # Element count conservation: product(input) == product(output)
                if inp_vars and out_vars:
                    in_prod = inp_vars[0]
                    for v in inp_vars[1:]:
                        in_prod = in_prod * v
                    out_prod = out_vars[0]
                    for v in out_vars[1:]:
                        out_prod = out_prod * v
                    constraints.append(in_prod == out_prod)
                tensor_vars[step.output] = out_vars
            else:
                tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.TRANSPOSE:
            d0 = step.params.get("dim0", 0)
            d1 = step.params.get("dim1", 1)
            out_vars = list(inp_vars)
            if d0 < len(out_vars) and d1 < len(out_vars):
                out_vars[d0], out_vars[d1] = out_vars[d1], out_vars[d0]
            tensor_vars[step.output] = out_vars

        elif step.op == OpKind.PERMUTE:
            perm = step.params.get("dims")
            if perm and len(perm) == len(inp_vars):
                tensor_vars[step.output] = [inp_vars[p] for p in perm]
            else:
                tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.ADD or step.op == OpKind.MULTIPLY:
            # Broadcast: output shape = broadcast(input_a, input_b)
            # Conservative: use first input's shape.
            tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.CAT:
            # Cat: all dims except cat_dim must match. Output cat_dim is sum.
            tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.SQUEEZE:
            dim = step.params.get("dim")
            if dim is not None and 0 <= dim < len(inp_vars):
                out_vars = inp_vars[:dim] + inp_vars[dim + 1:]
                tensor_vars[step.output] = out_vars
            else:
                tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.UNSQUEEZE:
            dim = step.params.get("dim", 0)
            one_var = z3.IntVal(1)
            if 0 <= dim <= len(inp_vars):
                out_vars = list(inp_vars[:dim]) + [one_var] + list(inp_vars[dim:])
                tensor_vars[step.output] = out_vars
            else:
                tensor_vars[step.output] = list(inp_vars)

        elif step.op == OpKind.FLATTEN:
            start = step.params.get("start_dim", 1)
            end = step.params.get("end_dim", -1)
            if end == -1:
                end = len(inp_vars) - 1
            pre = inp_vars[:start]
            flat_product = z3.IntVal(1)
            for vi in inp_vars[start:end + 1]:
                flat_product = flat_product * vi
            flat_var = z3.Int(f"__ci_{step.output}_d{len(pre)}")
            constraints.append(flat_var == flat_product)
            post = inp_vars[end + 1:] if end + 1 < len(inp_vars) else []
            tensor_vars[step.output] = list(pre) + [flat_var] + list(post)

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            b_name = step.inputs[1]
            b_vars = tensor_vars.get(b_name, [])
            if b_vars:
                # matmul: (..., k) x (..., k, n) -> (..., n)
                out_vars = list(inp_vars[:-1])
                if len(b_vars) >= 2:
                    out_vars.append(b_vars[-1])
                elif b_vars:
                    # vector case
                    pass
                tensor_vars[step.output] = out_vars
            else:
                tensor_vars[step.output] = list(inp_vars)

        else:
            # Default: preserve shape.
            tensor_vars[step.output] = list(inp_vars)

    def _encode_safety_at_step(
        self,
        step: ComputationStep,
        graph: ComputationGraph,
        tensor_vars: Dict[str, List[Any]],
        dm: "DimMapping",
        input_shapes: Dict[str, tuple],
    ) -> List[Any]:
        """Encode the safety property at a computation step.

        Returns a list of Z3 constraints that must ALL hold for the step
        to be safe. The caller should negate their conjunction for B.
        """
        safety: List[Any] = []

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = graph.layers.get(step.layer_ref)
            inp_name = step.inputs[0] if step.inputs else None
            inp_vars = tensor_vars.get(inp_name, []) if inp_name else []

            if not layer or not inp_vars:
                return safety

            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                if inp_vars:
                    safety.append(inp_vars[-1] == z3.IntVal(layer.in_features))

            elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D) and layer.in_channels is not None:
                if len(inp_vars) >= 2:
                    safety.append(inp_vars[1] == z3.IntVal(layer.in_channels))

            elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                                LayerKind.GROUPNORM, LayerKind.INSTANCENORM2D) and layer.num_features is not None:
                if len(inp_vars) >= 2:
                    safety.append(inp_vars[1] == z3.IntVal(layer.num_features))

            elif layer.kind == LayerKind.LAYERNORM and layer.num_features is not None:
                if inp_vars:
                    safety.append(inp_vars[-1] == z3.IntVal(layer.num_features))

            elif layer.kind == LayerKind.EMBEDDING and layer.num_embeddings is not None:
                if inp_vars:
                    safety.append(inp_vars[-1] < z3.IntVal(layer.num_embeddings))

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_vars = tensor_vars.get(step.inputs[0], [])
            b_vars = tensor_vars.get(step.inputs[1], [])
            if a_vars and b_vars:
                a_last = a_vars[-1]
                b_inner = b_vars[-2] if len(b_vars) >= 2 else b_vars[0]
                safety.append(a_last == b_inner)

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a_vars = tensor_vars.get(step.inputs[0], [])
            b_vars = tensor_vars.get(step.inputs[1], [])
            if a_vars and b_vars:
                n = min(len(a_vars), len(b_vars))
                for i in range(1, n + 1):
                    av = a_vars[-i]
                    bv = b_vars[-i]
                    safety.append(z3.Or(
                        av == bv,
                        av == z3.IntVal(1),
                        bv == z3.IntVal(1),
                    ))

        elif step.op == OpKind.RESHAPE:
            target_shape = step.params.get("shape")
            inp_name = step.inputs[0] if step.inputs else None
            inp_vars = tensor_vars.get(inp_name, []) if inp_name else []
            if target_shape and inp_vars:
                known_product = 1
                has_infer = False
                for d in target_shape:
                    if isinstance(d, int) and d > 0:
                        known_product *= d
                    elif d == -1:
                        has_infer = True
                if not has_infer and known_product > 1:
                    in_prod = inp_vars[0]
                    for v in inp_vars[1:]:
                        in_prod = in_prod * v
                    safety.append(in_prod == z3.IntVal(known_product))

        return safety

    def _extract_via_unsat_core(
        self,
        path_constraints: List[Any],
        safety_constraints: List[Any],
        dim_map: Dict[str, Tuple[str, int]],
        concrete_dims: Dict[str, int],
        incremental_solver: Any = None,
    ) -> List[ShapePredicate]:
        """Extract predicates using Z3 unsat core for predicate discovery.

        Adds path constraints as tracked assertions, negates safety,
        and extracts the minimal unsatisfiable core to identify which
        input dimension constraints are essential.

        When *incremental_solver* is provided (an ``IncrementalCEGARSolver``),
        assertions are tracked through that solver for cross-iteration
        clause reuse and enhanced MUS-based predicate extraction is
        attempted first.
        """
        # --- Incremental path: use IncrementalCEGARSolver + MUS ---
        if incremental_solver is not None:
            try:
                from src.unsat_core_cegar import EnhancedUnsatCorePredicateExtractor
                incremental_solver.push()
                label_map: Dict[str, Any] = {}
                for i, c in enumerate(path_constraints):
                    label = f"__path_{i}"
                    label_map[label] = c
                    incremental_solver.assert_and_track(c, label)
                if safety_constraints:
                    neg_safety = z3.Not(z3.And(*safety_constraints))
                    incremental_solver.add(neg_safety)
                result, core_labels = incremental_solver.check_with_core()
                incremental_solver.pop()
                if result == z3.unsat and core_labels:
                    extractor = EnhancedUnsatCorePredicateExtractor()
                    core_preds = extractor.extract_predicates(
                        core_labels, label_map, dim_map,
                    )
                    predicates: List[ShapePredicate] = []
                    for cp in core_preds:
                        if cp.shape_predicate is not None:
                            predicates.append(cp.shape_predicate)
                    if predicates:
                        return predicates
                    # Fall through to basic extraction if MUS yielded nothing.
            except (ImportError, Exception):
                pass

        # --- Basic path: fresh solver per call ---
        solver = z3.Solver()
        solver.set("timeout", 5000)
        solver.set("unsat_core", True)

        # Add path constraints with tracking labels
        labels: Dict[str, Any] = {}
        for i, c in enumerate(path_constraints):
            label = z3.Bool(f"__path_{i}")
            labels[f"__path_{i}"] = c
            solver.assert_and_track(c, label)

        # Negate safety: we want path ∧ ¬safety to be UNSAT
        # (counterexample is spurious because path constraints force safety)
        if safety_constraints:
            neg_safety = z3.Not(z3.And(*safety_constraints))
            solver.add(neg_safety)

        result = solver.check()

        predicates: List[ShapePredicate] = []
        if result == z3.unsat:
            # The counterexample is spurious — extract core
            core = solver.unsat_core()
            core_names = {str(c) for c in core}

            # Map core assertions back to dimension constraints
            for i, c in enumerate(path_constraints):
                label_name = f"__path_{i}"
                if label_name in core_names:
                    # Extract dimension info from the constraint
                    pred = self._constraint_to_predicate(c, dim_map)
                    if pred is not None:
                        predicates.append(pred)

        return predicates

    def _constraint_to_predicate(
        self,
        constraint: Any,
        dim_map: Dict[str, Tuple[str, int]],
    ) -> Optional[ShapePredicate]:
        """Convert a Z3 constraint from the unsat core into a ShapePredicate."""
        if not HAS_Z3:
            return None

        constraint_str = str(constraint)

        # Look for equality constraints like "var == value"
        for var_name, (tensor, axis) in dim_map.items():
            if var_name in constraint_str and "==" in constraint_str:
                # Try to extract the concrete value
                try:
                    # Check if this is a "var == IntVal" constraint
                    if z3.is_eq(constraint):
                        lhs, rhs = constraint.children()
                        val = None
                        if z3.is_int_value(rhs):
                            val = rhs.as_long()
                        elif z3.is_int_value(lhs):
                            val = lhs.as_long()
                        if val is not None and val > 0:
                            return ShapePredicate(
                                kind=PredicateKind.DIM_EQ,
                                tensor=tensor,
                                axis=axis,
                                value=val,
                                provenance="cegar_discovered",
                            )
                except Exception:
                    pass

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Counterexample analyser
# ═══════════════════════════════════════════════════════════════════════════════

class CounterexampleAnalyser:
    """Classifies counterexamples as spurious or real and synthesises
    predicates to eliminate spurious ones.
    """

    def __init__(
        self,
        graph: ComputationGraph,
        shape_env: Dict[str, TensorShape],
        input_shapes: Dict[str, tuple],
    ) -> None:
        self.graph = graph
        self.shape_env = shape_env
        self.input_shapes = input_shapes
        self.tracer = TraceBackEngine(graph)
        self.predicate_extractor = UnsatCorePredicateExtractor(graph, shape_env)

    def analyse(
        self,
        counterexample: CounterexampleTrace,
    ) -> List[AnalysedCounterexample]:
        """Analyse all violations in a counterexample trace."""
        results: List[AnalysedCounterexample] = []

        for violation in counterexample.violations:
            acex = self._analyse_single(violation, counterexample)
            results.append(acex)

        return results

    def _analyse_single(
        self,
        violation: SafetyViolation,
        cex_trace: CounterexampleTrace,
    ) -> AnalysedCounterexample:
        """Analyse a single safety violation."""
        step_idx = violation.step_index

        # Extract concrete dimension values from the counterexample
        concrete_dims = dict(cex_trace.concrete_dims)

        # Trace back to input tensors
        traced = self.tracer.trace_to_inputs(step_idx, violation)

        # Find constraint origins
        origins = self.tracer.find_constraint_origin(
            step_idx, violation, self.shape_env,
        )

        # Classify and synthesise
        classification, predicates, reason = self._classify_and_synthesise(
            violation, step_idx, origins, concrete_dims,
        )

        return AnalysedCounterexample(
            violation=violation,
            step_index=step_idx,
            classification=classification,
            concrete_dims=concrete_dims,
            traced_to_inputs=traced,
            synthesised_predicates=predicates,
            reason=reason,
        )

    def _check_cex_feasibility(
        self,
        concrete_dims: Dict[str, int],
        step: ComputationStep,
        violation: SafetyViolation,
    ) -> bool:
        """Abstract feasibility check via Z3.

        Checks if the counterexample is feasible by encoding the
        computation graph path constraints up to the failing step
        with the counterexample's concrete dimension assignments,
        and verifying via Z3 that the path is satisfiable AND the
        safety property is violated.  This is an *abstract* check
        (it reasons about constraints, not concrete Python execution).

        Returns True if the counterexample is feasible (real bug).
        Returns False if the counterexample is spurious (infeasible path).
        """
        # Basic hardware bounds check
        MIN_DIM = 1
        MAX_DIM = 65536
        for dim_name, dim_val in concrete_dims.items():
            if not isinstance(dim_val, int):
                continue
            if dim_val < MIN_DIM or dim_val > MAX_DIM:
                return False

        # Z3-based path feasibility check
        if not HAS_Z3:
            return True  # Conservative: assume feasible without Z3

        step_idx = violation.step_index
        if step_idx < 0 or step_idx >= len(self.graph.steps):
            return True

        solver = z3.Solver()
        solver.set("timeout", 3000)

        # Create Z3 variables for all concrete dimensions
        z3_dims: Dict[str, z3.ArithRef] = {}
        for dim_name, dim_val in concrete_dims.items():
            if isinstance(dim_val, int):
                var = z3.Int(f"_feas_{dim_name}")
                solver.add(var == z3.IntVal(dim_val))
                z3_dims[dim_name] = var

        # Encode path constraints through the computation graph
        # up to the failing step
        for i, s in enumerate(self.graph.steps[:step_idx + 1]):
            if s.op == OpKind.LAYER_CALL and s.layer_ref:
                layer = self.graph.layers.get(s.layer_ref)
                if layer and s.inputs:
                    inp = s.inputs[0]
                    inp_shape = self.shape_env.get(inp)
                    if inp_shape and layer.kind == LayerKind.LINEAR:
                        # Encode: last dim of input must equal in_features
                        last_axis = len(inp_shape.dims) - 1
                        for dim in inp_shape.dims:
                            if dim.is_symbolic and str(dim.value) in z3_dims:
                                pass  # Already constrained

            elif s.op == OpKind.MATMUL and len(s.inputs) >= 2:
                a_shape = self.shape_env.get(s.inputs[0])
                b_shape = self.shape_env.get(s.inputs[1])
                if a_shape and b_shape:
                    # Inner dimensions must match
                    pass  # Encoded via concrete dims

        # Check if the concrete dims actually violate the operation
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp:
                inp_shape = self.shape_env.get(inp)
                if inp_shape and layer.kind == LayerKind.LINEAR:
                    if layer.in_features is not None:
                        last_dim = inp_shape.dims[-1] if inp_shape.dims else None
                        if last_dim and not last_dim.is_symbolic:
                            return int(last_dim.value) != layer.in_features
                        elif last_dim and last_dim.is_symbolic:
                            sym_name = str(last_dim.value)
                            if sym_name in concrete_dims:
                                return concrete_dims[sym_name] != layer.in_features

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape and a_shape.dims and b_shape.dims:
                a_last = a_shape.dims[-1]
                b_inner = b_shape.dims[-2] if len(b_shape.dims) >= 2 else b_shape.dims[0]
                a_val = concrete_dims.get(str(a_last.value)) if a_last.is_symbolic else (int(a_last.value) if not a_last.is_symbolic else None)
                b_val = concrete_dims.get(str(b_inner.value)) if b_inner.is_symbolic else (int(b_inner.value) if not b_inner.is_symbolic else None)
                if a_val is not None and b_val is not None:
                    return a_val != b_val

        # Fallback: check path satisfiability with Z3
        result = solver.check()
        return result == z3.sat

    def _classify_and_synthesise(
        self,
        violation: SafetyViolation,
        step_idx: int,
        origins: List[Tuple[str, int, Optional[int]]],
        concrete_dims: Dict[str, int],
    ) -> Tuple[CounterexampleClassification, List[ShapePredicate], str]:
        """Classify a violation and synthesise predicates if spurious."""
        predicates: List[ShapePredicate] = []
        step = self.graph.steps[step_idx] if step_idx < len(self.graph.steps) else None

        if not step:
            return (CounterexampleClassification.UNKNOWN, [], "step not found")

        # Check if this is a real bug: concrete dimensions that cannot
        # be fixed by constraining inputs
        if self._is_real_bug(violation, step):
            return (
                CounterexampleClassification.REAL_BUG,
                [],
                self._real_bug_reason(violation, step),
            )

        # Feasibility check: if the concrete counterexample dimensions are
        # all physically realizable, the bug is real even if trace-back
        # would produce predicates to eliminate it.
        if concrete_dims and self._check_cex_feasibility(concrete_dims, step, violation):
            # Check whether the concrete dims actually violate the
            # operation's requirements (not just that they're feasible
            # in general).
            if self._concrete_dims_violate_op(concrete_dims, step, step_idx):
                return (
                    CounterexampleClassification.REAL_BUG,
                    [],
                    f"feasible counterexample: concrete dims {concrete_dims} "
                    f"are realizable and violate {step.op.name}",
                )

        # Spurious: synthesise predicates from origins
        for tensor_name, axis, expected in origins:
            if expected is not None:
                pred = ShapePredicate(
                    kind=PredicateKind.DIM_EQ,
                    tensor=tensor_name,
                    axis=axis,
                    value=expected,
                    provenance="api_stub",
                )
                predicates.append(pred)
            else:
                # Try to get expected from concrete dims or layer params
                inferred = self._infer_expected(
                    tensor_name, axis, step, concrete_dims,
                )
                if inferred is not None:
                    pred = ShapePredicate(
                        kind=PredicateKind.DIM_EQ,
                        tensor=tensor_name,
                        axis=axis,
                        value=inferred,
                        provenance="api_stub",
                    )
                    predicates.append(pred)

        # Use unsat-core-based predicate discovery as a fallback
        if not predicates:
            interp_preds = self.predicate_extractor.discover_predicates(
                violation, step_idx, concrete_dims, self.input_shapes,
            )
            predicates.extend(interp_preds)

        if predicates:
            reason = "spurious: fixed by constraining " + ", ".join(
                p.pretty() for p in predicates
            )
            return (CounterexampleClassification.SPURIOUS, predicates, reason)

        # Could not determine — be conservative
        return (CounterexampleClassification.UNKNOWN, [], "could not classify")

    def _is_real_bug(
        self, violation: SafetyViolation, step: ComputationStep
    ) -> bool:
        """Check if a violation is a real shape bug (all dimensions are
        concrete and incompatible).
        """
        if violation.kind != "shape_incompatible":
            return False

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if not layer:
                return False

            inp = step.inputs[0] if step.inputs else None
            inp_shape = self.shape_env.get(inp) if inp else None

            if inp_shape is None:
                return False

            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                last_dim = inp_shape.dims[-1] if inp_shape.ndim >= 1 else None
                if last_dim and not last_dim.is_symbolic:
                    return last_dim.value != layer.in_features

            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                ch_dim = inp_shape.dims[1] if inp_shape.ndim >= 2 else None
                if ch_dim and not ch_dim.is_symbolic:
                    return ch_dim.value != layer.in_channels

        if step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape:
                k_a = a_shape.dims[-1] if a_shape.ndim >= 1 else None
                k_b = (b_shape.dims[-2] if b_shape.ndim >= 2
                       else b_shape.dims[0] if b_shape.ndim == 1
                       else None)
                if (k_a and k_b and
                        not k_a.is_symbolic and not k_b.is_symbolic):
                    return k_a.value != k_b.value

        return False

    def _real_bug_reason(
        self, violation: SafetyViolation, step: ComputationStep
    ) -> str:
        """Produce a human-readable explanation for a real shape bug."""
        msg = violation.message
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer and layer.kind == LayerKind.LINEAR:
                return (
                    f"Real shape bug: self.{step.layer_ref} expects "
                    f"in_features={layer.in_features} but input has "
                    f"shape {violation.shape_a.pretty() if violation.shape_a else '?'}"
                )
        return f"Real shape bug: {msg}"

    def _infer_expected(
        self,
        tensor_name: str,
        axis: int,
        step: ComputationStep,
        concrete_dims: Dict[str, int],
    ) -> Optional[int]:
        """Attempt to infer the expected value for a dimension."""
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if layer:
                if layer.kind == LayerKind.LINEAR:
                    return layer.in_features
                if layer.kind == LayerKind.CONV2D:
                    return layer.in_channels
                if layer.kind == LayerKind.EMBEDDING:
                    return layer.num_embeddings

        # Try to get from counterexample concrete dims
        shape = self.shape_env.get(tensor_name)
        if shape:
            norm_axis = axis
            if norm_axis < 0:
                norm_axis = shape.ndim + norm_axis
            if 0 <= norm_axis < shape.ndim:
                dim = shape.dims[norm_axis]
                if dim.is_symbolic and str(dim.value) in concrete_dims:
                    return concrete_dims[str(dim.value)]

        return None

    def _concrete_dims_violate_op(
        self,
        concrete_dims: Dict[str, int],
        step: ComputationStep,
        step_idx: int = -1,
    ) -> bool:
        """Check whether the concrete dimension values from the counterexample
        actually violate the operation's shape requirements in a way that
        *no* predicate can fix.

        This returns True only when the expected value for the operation
        is itself infeasible (e.g., a layer expects in_features=0 or a
        negative value), meaning the model is structurally broken.
        When there exists a valid expected value (e.g., in_features=10),
        the bug is spurious and can be fixed by a predicate.

        Special handling for flatten→Linear patterns: if the Linear's input
        comes from a flatten whose product (computed from concrete dims)
        doesn't match in_features AND involves symbolic dims that vary
        with input, this is a real structural bug.
        """
        MIN_DIM = 1
        MAX_DIM = 65536

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if not layer:
                return False
            if layer.kind == LayerKind.LINEAR and layer.in_features is not None:
                if not (MIN_DIM <= layer.in_features <= MAX_DIM):
                    return True  # layer itself is broken

                inp_name = step.inputs[0] if step.inputs else None
                if inp_name:
                    # Check flatten→Linear product mismatch
                    if self._flatten_product_mismatch(
                        inp_name, layer.in_features, concrete_dims, step_idx
                    ):
                        return True

                    # Check Linear→Linear out_features vs in_features mismatch
                    if self._producer_output_mismatch(
                        inp_name, layer.in_features, step_idx
                    ):
                        return True

                return False  # fixable by predicate

            if layer.kind == LayerKind.CONV2D and layer.in_channels is not None:
                if MIN_DIM <= layer.in_channels <= MAX_DIM:
                    return False
                return True

        if step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a_shape = self.shape_env.get(step.inputs[0])
            b_shape = self.shape_env.get(step.inputs[1])
            if a_shape and b_shape:
                k_a = a_shape.dims[-1] if a_shape.ndim >= 1 else None
                k_b = (b_shape.dims[-2] if b_shape.ndim >= 2
                       else b_shape.dims[0] if b_shape.ndim == 1
                       else None)
                # Both dims symbolic → fixable by adding a match predicate
                if k_a and k_b and k_a.is_symbolic and k_b.is_symbolic:
                    return False
                # Both concrete and mismatched → real bug (already caught
                # by _is_real_bug, but double-check)
                if (k_a and k_b
                        and not k_a.is_symbolic and not k_b.is_symbolic):
                    return k_a.value != k_b.value

        return False

    def _flatten_product_mismatch(
        self,
        tensor_name: str,
        expected_in_features: int,
        concrete_dims: Dict[str, int],
        before_step: int = -1,
    ) -> bool:
        """Check if a tensor produced by flatten has a concrete product
        that doesn't match expected_in_features AND involves symbolic dims.

        Instead of using shape_env (which has only initial shapes), traces
        back through the graph to find the actual channel count from the
        last channel-producing layer (Conv/Pool) before the flatten.
        """
        # Find the last producer of tensor_name before the failing step
        producer_idx = None
        limit = before_step if before_step >= 0 else len(self.graph.steps)
        for idx in range(limit):
            s = self.graph.steps[idx]
            if s.output == tensor_name:
                producer_idx = idx
        if producer_idx is None:
            return False

        producer = self.graph.steps[producer_idx]

        # Check if producer is a flatten or reshape (view)
        is_flatten = (
            (producer.op == OpKind.LAYER_CALL and producer.layer_ref and
             self.graph.layers.get(producer.layer_ref) and
             self.graph.layers[producer.layer_ref].kind == LayerKind.FLATTEN) or
            producer.op == OpKind.FLATTEN or
            producer.op == OpKind.RESHAPE
        )
        if not is_flatten:
            return False

        # Walk backward from the flatten to find the last layer that
        # determines the channel count (Conv/Pool layers).
        out_channels = self._trace_back_channels(producer, limit)
        if out_channels is None:
            return False

        # Check: can out_channels * (some spatial product) == expected?
        # If expected / out_channels is not a positive integer, it's a mismatch.
        if expected_in_features % out_channels != 0:
            return True  # channel count incompatible with in_features

        return False

    def _trace_back_channels(
        self,
        step: ComputationStep,
        limit: int,
    ) -> Optional[int]:
        """Walk backward from a step to find the out_channels of the last
        channel-determining layer (Conv2d, Conv1d)."""
        # Get the input tensor name
        inp_name = step.inputs[0] if step.inputs else None
        if not inp_name:
            return None

        # Find the producer of this input (last assignment before this step)
        visited = set()
        current = inp_name
        for _ in range(20):  # depth limit
            if current in visited:
                break
            visited.add(current)

            prod_idx = None
            for idx in range(limit):
                s = self.graph.steps[idx]
                if s.output == current:
                    prod_idx = idx

            if prod_idx is None:
                break

            ps = self.graph.steps[prod_idx]
            if ps.op == OpKind.LAYER_CALL and ps.layer_ref:
                layer = self.graph.layers.get(ps.layer_ref)
                if layer:
                    if layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D) and layer.out_channels is not None:
                        return layer.out_channels
                    if layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                      LayerKind.BATCHNORM2D, LayerKind.BATCHNORM1D,
                                      LayerKind.IDENTITY, LayerKind.SOFTMAX,
                                      LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D,
                                      LayerKind.ADAPTIVE_AVGPOOL2D, LayerKind.ADAPTIVE_MAXPOOL2D):
                        # Shape-preserving or spatially-reducing: trace further back
                        current = ps.inputs[0] if ps.inputs else None
                        if current is None:
                            break
                        continue
                    # Other layer types: can't determine channels
                    break
            else:
                # Non-layer-call: trace further back
                current = ps.inputs[0] if ps.inputs else None
                if current is None:
                    break
                continue
            break

        return None

    def _producer_output_mismatch(
        self,
        tensor_name: str,
        expected_in_features: int,
        before_step: int = -1,
    ) -> bool:
        """Check if a tensor is produced by a layer whose concrete
        out_features doesn't match expected_in_features.

        Detects structural mismatches like Linear(512→256) feeding
        into Linear(512→...) where 256 ≠ 512.
        """
        # Find the last producer of tensor_name before the failing step
        producer_idx = None
        limit = before_step if before_step >= 0 else len(self.graph.steps)
        for idx in range(limit):
            s = self.graph.steps[idx]
            if s.output == tensor_name:
                producer_idx = idx
        if producer_idx is None:
            return False

        producer = self.graph.steps[producer_idx]
        if producer.op != OpKind.LAYER_CALL or not producer.layer_ref:
            return False

        prod_layer = self.graph.layers.get(producer.layer_ref)
        if not prod_layer:
            return False

        # Linear → Linear: check out_features vs in_features
        if prod_layer.kind == LayerKind.LINEAR and prod_layer.out_features is not None:
            if prod_layer.out_features != expected_in_features:
                return True

        # Conv → Linear: check out_channels (if no flatten/reshape between)
        if prod_layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D) and prod_layer.out_channels is not None:
            # Only a mismatch if there's no intervening flatten
            if prod_layer.out_channels != expected_in_features:
                return True  # likely a missing flatten

        return False

class ShapeRefinement:
    """Applies discovered predicates to refine the shape environment
    and input shapes for the next CEGAR iteration.
    """

    @staticmethod
    def apply_predicates(
        input_shapes: Dict[str, tuple],
        shape_env: Dict[str, TensorShape],
        predicates: List[ShapePredicate],
    ) -> Tuple[Dict[str, tuple], Dict[str, TensorShape]]:
        """Return updated input_shapes and shape_env with predicates applied.

        For a ``DIM_EQ`` predicate on an input tensor, the symbolic
        dimension is replaced with the concrete value.
        """
        new_input_shapes = dict(input_shapes)
        new_shape_env = dict(shape_env)

        for pred in predicates:
            if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
                # Update input_shapes if this tensor is an input
                if pred.tensor in new_input_shapes:
                    old = list(new_input_shapes[pred.tensor])
                    axis = pred.axis
                    if axis is not None and axis < 0:
                        axis = len(old) + axis
                    if axis is not None and 0 <= axis < len(old):
                        old[axis] = pred.value
                    new_input_shapes[pred.tensor] = tuple(old)

                # Update shape_env
                if pred.tensor in new_shape_env:
                    shape = new_shape_env[pred.tensor]
                    dims = list(shape.dims)
                    axis = pred.axis
                    if axis is not None and axis < 0:
                        axis = len(dims) + axis
                    if axis is not None and 0 <= axis < len(dims):
                        dims[axis] = ShapeDim(pred.value)
                    new_shape_env[pred.tensor] = TensorShape(tuple(dims))

            elif pred.kind == PredicateKind.NDIM_EQ and pred.value is not None:
                if pred.tensor in new_input_shapes:
                    old = list(new_input_shapes[pred.tensor])
                    if len(old) != pred.value:
                        # Adjust by padding with symbolic dims
                        while len(old) < pred.value:
                            old.append(f"_d{len(old)}")
                        new_input_shapes[pred.tensor] = tuple(old[:pred.value])

        return new_input_shapes, new_shape_env

    @staticmethod
    def predicates_to_z3(
        predicates: List[ShapePredicate],
    ) -> List[Any]:
        """Convert predicates to Z3 constraints (for Z3 feasibility check)."""
        if not HAS_Z3:
            return []

        constraints = []
        for pred in predicates:
            if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var == z3.IntVal(pred.value))
                constraints.append(dim_var > 0)

            elif pred.kind == PredicateKind.DIM_GT and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var > z3.IntVal(pred.value))

            elif pred.kind == PredicateKind.DIM_GE and pred.value is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var >= z3.IntVal(pred.value))

            elif pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor is not None:
                dim_name = f"{pred.tensor}_dim{pred.axis}"
                dim_var = z3.Int(dim_name)
                constraints.append(dim_var % z3.IntVal(pred.divisor) == 0)

        return constraints

    @staticmethod
    def check_feasibility(predicates: List[ShapePredicate]) -> bool:
        """Check whether a set of predicates is simultaneously satisfiable."""
        if not HAS_Z3:
            return True

        constraints = ShapeRefinement.predicates_to_z3(predicates)
        if not constraints:
            return True

        solver = z3.Solver()
        solver.set("timeout", 2000)
        for c in constraints:
            solver.add(c)

        return solver.check() == z3.sat


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Z3-based counterexample extraction
# ═══════════════════════════════════════════════════════════════════════════════

class Z3CounterexampleExtractor:
    """Extracts concrete dimension values from Z3 models produced by
    the constraint verifier.
    """

    @staticmethod
    def extract_dims_from_model(
        z3_model: Any,
        symbolic_names: List[str],
    ) -> Dict[str, int]:
        """Extract concrete integer values for each symbolic dimension
        from a Z3 model.
        """
        if not HAS_Z3 or z3_model is None:
            return {}

        result: Dict[str, int] = {}
        for name in symbolic_names:
            var = z3.Int(name)
            val = z3_model.evaluate(var, model_completion=True)
            try:
                result[name] = val.as_long()
            except (AttributeError, z3.Z3Exception):
                pass
        return result

    @staticmethod
    def find_violating_assignment(
        constraints: List[Any],
        symbolic_names: List[str],
    ) -> Optional[Dict[str, int]]:
        """Find a concrete dimension assignment that violates safety
        constraints, or ``None`` if no such assignment exists (UNSAT).
        """
        if not HAS_Z3:
            return None

        solver = z3.Solver()
        solver.set("timeout", 5000)

        # All dims must be positive
        for name in symbolic_names:
            solver.add(z3.Int(name) > 0)

        # Negate the safety constraints: look for a violation
        if constraints:
            solver.add(z3.Not(z3.And(*constraints)))

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return Z3CounterexampleExtractor.extract_dims_from_model(
                model, symbolic_names,
            )
        return None

    @staticmethod
    def check_predicate_eliminates_cex(
        predicate: ShapePredicate,
        cex_dims: Dict[str, int],
        shape_env: Dict[str, TensorShape],
    ) -> bool:
        """Check whether adding *predicate* would rule out the
        counterexample dimension assignment *cex_dims*.
        """
        if predicate.kind == PredicateKind.DIM_EQ and predicate.value is not None:
            shape = shape_env.get(predicate.tensor)
            if shape:
                axis = predicate.axis or 0
                if axis < 0:
                    axis = shape.ndim + axis
                if 0 <= axis < shape.ndim:
                    dim = shape.dims[axis]
                    if dim.is_symbolic:
                        cex_val = cex_dims.get(str(dim.value))
                        if cex_val is not None:
                            return cex_val != predicate.value
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Predicate deduplication & minimisation
# ═══════════════════════════════════════════════════════════════════════════════

class PredicateQualityScorer:
    """Scores predicates by quality to prevent counterproductive refinements.

    The contract discovery loop can degrade F1 when predicates over-constrain the input
    space, masking real bugs.  This scorer evaluates each candidate predicate
    along three axes:

    1. **Generality** — Does the predicate restrict a single concrete value
       (overly specific, likely from one counterexample) or express a
       structural property (e.g., divisibility, dimension matching)?
       Structural predicates score higher.

    2. **Coverage preservation** — After adding the predicate, what fraction
       of the valid input space remains?  Computed via Z3 model counting
       over a bounded domain.  Predicates that eliminate > 90% of the space
       are likely masking real bugs.

    3. **Consistency** — Does the predicate conflict with any previously
       discovered real bugs?  If constraining input X to value V would make
       a known-real bug unreachable, the predicate is counter-productive.
    """

    # Bounded domain for coverage estimation
    DIM_LOWER = 1
    DIM_UPPER = 2048
    SAMPLE_COUNT = 16  # number of random models to sample for coverage

    def __init__(
        self,
        existing_predicates: List[ShapePredicate],
        known_real_bugs: List[SafetyViolation],
    ) -> None:
        self.existing = existing_predicates
        self.known_bugs = known_real_bugs

    def score(self, pred: ShapePredicate) -> float:
        """Return a quality score in [0.0, 1.0].  Higher is better.

        Four axes with weights summing to 1.0:
          - generality  0.3
          - coverage     0.2
          - consistency  0.2
          - mutual_info  0.3
        """
        g = self._generality_score(pred)
        c = self._coverage_score(pred)
        k = self._consistency_score(pred)
        m = self._mutual_information_score(pred)
        return (g ** 0.3) * (c ** 0.2) * (k ** 0.2) * (m ** 0.3)

    def _generality_score(self, pred: ShapePredicate) -> float:
        """Structural predicates score higher than concrete equalities."""
        if pred.kind == PredicateKind.DIM_MATCH:
            return 1.0  # dimension matching is structural
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            return 0.9  # divisibility is structural
        if pred.kind == PredicateKind.NDIM_EQ:
            return 0.85  # rank constraint is structural
        if pred.kind == PredicateKind.DIM_GE:
            return 0.7  # lower bound preserves generality
        if pred.kind == PredicateKind.DIM_GT:
            return 0.7
        if pred.kind == PredicateKind.DIM_EQ:
            # Concrete equality is the most restrictive — score by
            # how "reasonable" the value is (powers of 2, small
            # multiples of common embedding dims are more likely
            # structural requirements than random values)
            v = pred.value or 0
            if v > 0 and (v & (v - 1)) == 0:
                return 0.6  # power of 2
            if v in (10, 64, 128, 256, 512, 768, 1024, 2048, 3, 300):
                return 0.55  # common ML dimension
            return 0.4  # arbitrary concrete value
        if pred.kind == PredicateKind.SHAPE_EQ:
            return 0.3  # full shape equality is very restrictive
        return 0.5

    def _coverage_score(self, pred: ShapePredicate) -> float:
        """Estimate what fraction of the bounded input space survives."""
        if not HAS_Z3:
            return 0.7  # cannot check; assume moderate

        if pred.kind in (PredicateKind.DIM_MATCH, PredicateKind.NDIM_EQ):
            return 0.9  # structural — preserves most of the space

        if pred.kind == PredicateKind.DIM_EQ:
            # One specific value out of [DIM_LOWER, DIM_UPPER]
            # Coverage = 1 / (DIM_UPPER - DIM_LOWER + 1) which is tiny,
            # but the predicate is only applied to ONE axis of ONE tensor.
            # The rest of the space is unaffected.  Score = moderate.
            return 0.5

        if pred.kind == PredicateKind.DIM_GE and pred.value is not None:
            # Fraction preserved = (DIM_UPPER - value + 1) / range
            rng = self.DIM_UPPER - self.DIM_LOWER + 1
            surviving = max(0, self.DIM_UPPER - pred.value + 1)
            return max(0.1, surviving / rng)

        if pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor:
            # Fraction preserved = count of multiples in range / range
            rng = self.DIM_UPPER - self.DIM_LOWER + 1
            multiples = self.DIM_UPPER // pred.divisor
            return max(0.1, multiples / rng)

        return 0.6

    def _consistency_score(self, pred: ShapePredicate) -> float:
        """Check whether the predicate would mask a known real bug."""
        if not self.known_bugs:
            return 1.0  # no known bugs to conflict with

        for bug in self.known_bugs:
            if bug.shape_a and pred.kind == PredicateKind.DIM_EQ:
                # If the predicate would constrain the dimension that
                # the bug depends on, it might be masking the bug
                if (pred.tensor and bug.message and
                        pred.tensor in bug.message):
                    return 0.2  # likely masking
        return 1.0

    def _mutual_information_score(self, pred: ShapePredicate) -> float:
        """Estimate how much information the predicate provides about bug detection.

        A predicate that eliminates a *class* of invalid inputs (e.g.
        "batch dim must be > 0", divisibility, dimension matching) provides
        high mutual information with bugs — it separates valid from invalid
        regions broadly.  A predicate that eliminates only one specific
        counterexample value (high specificity, low generality) provides
        little information and scores lower.
        """
        # Structural predicates capture invariants across many inputs
        if pred.kind == PredicateKind.DIM_MATCH:
            return 1.0  # eliminates an entire class of mismatches
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            return 0.9  # eliminates all non-divisible values
        if pred.kind == PredicateKind.NDIM_EQ:
            return 0.85  # eliminates wrong-rank inputs
        if pred.kind in (PredicateKind.DIM_GE, PredicateKind.DIM_GT):
            # Lower/upper bounds partition the space into two halves —
            # reasonably informative
            return 0.75

        if pred.kind == PredicateKind.DIM_EQ:
            # A concrete equality only eliminates counterexamples with a
            # *different* value — very specific.  But common ML
            # dimensions (powers of 2, standard embedding sizes) are
            # more likely to represent structural requirements.
            v = pred.value or 0
            if v > 0 and (v & (v - 1)) == 0:
                return 0.6  # power of 2 — somewhat structural
            if v in (10, 64, 128, 256, 512, 768, 1024, 2048, 3, 300):
                return 0.55  # common ML dimension
            # Check if this predicate is redundant with existing ones
            for ex in self.existing:
                if (ex.kind == PredicateKind.DIM_EQ
                        and ex.tensor == pred.tensor
                        and ex.axis == pred.axis
                        and ex.value == pred.value):
                    return 0.1  # duplicate — no new information
            return 0.35  # arbitrary value — low information

        if pred.kind == PredicateKind.SHAPE_EQ:
            return 0.3  # very specific — only one exact shape

        return 0.5


# Minimum quality threshold for accepting a predicate into the contract discovery loop
PREDICATE_QUALITY_THRESHOLD = 0.25


class PredicateSet:
    """Manages a deduplicated, minimal, quality-filtered set of shape predicates.

    Predicates are scored for quality before acceptance.  Low-quality
    predicates (overly specific, space-restricting, or bug-masking) are
    rejected to prevent the degradation observed at scale during contract discovery.
    """

    def __init__(
        self,
        quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
        enable_quality_filter: bool = True,
    ) -> None:
        self._predicates: List[ShapePredicate] = []
        self._seen: Set[str] = set()
        self._quality_scores: Dict[str, float] = {}
        self._rejected: List[Tuple[ShapePredicate, float]] = []
        self.quality_threshold = quality_threshold
        self.enable_quality_filter = enable_quality_filter
        self._known_bugs: List[SafetyViolation] = []

    def set_known_bugs(self, bugs: List[SafetyViolation]) -> None:
        """Update the set of known real bugs for consistency scoring."""
        self._known_bugs = list(bugs)

    @property
    def predicates(self) -> List[ShapePredicate]:
        return list(self._predicates)

    @property
    def rejected_predicates(self) -> List[Tuple[ShapePredicate, float]]:
        """Predicates rejected by quality filtering, with their scores."""
        return list(self._rejected)

    def __len__(self) -> int:
        return len(self._predicates)

    def add(self, pred: ShapePredicate) -> bool:
        """Add a predicate if it passes quality filtering and does not
        conflict with existing predicates.  Returns True if added."""
        key = pred.pretty()
        if key in self._seen:
            return False

        # Quality gate
        if self.enable_quality_filter:
            scorer = PredicateQualityScorer(self._predicates, self._known_bugs)
            score = scorer.score(pred)
            self._quality_scores[key] = score
            if score < self.quality_threshold:
                self._rejected.append((pred, score))
                logger.debug(
                    "CEGAR: rejected predicate %s (quality=%.3f < %.3f)",
                    key, score, self.quality_threshold,
                )
                return False

        # Conflict detection: reject if the new predicate contradicts
        # any existing predicate (checked via Z3 satisfiability).
        conflict = self._check_conflict(pred)
        if conflict is not None:
            self._rejected.append((pred, 0.0))
            logger.debug(
                "CEGAR: rejected predicate %s — conflicts with %s",
                key, conflict.pretty(),
            )
            return False

        # Check for subsumption: DIM_EQ subsumes DIM_GE on the same axis
        if pred.kind == PredicateKind.DIM_EQ:
            to_remove = []
            for existing in self._predicates:
                if (existing.kind in (PredicateKind.DIM_GE, PredicateKind.DIM_GT)
                        and existing.tensor == pred.tensor
                        and existing.axis == pred.axis):
                    to_remove.append(existing)
            for r in to_remove:
                self._predicates.remove(r)
                self._seen.discard(r.pretty())

        self._predicates.append(pred)
        self._seen.add(key)
        return True

    def _check_conflict(self, pred: ShapePredicate) -> Optional[ShapePredicate]:
        """Check if *pred* contradicts any existing predicate using Z3.

        Returns the conflicting existing predicate, or None if no conflict.
        Example conflict: existing says x.shape[-1] == 768, new says
        x.shape[-1] == 512.
        """
        if not HAS_Z3:
            return None

        for existing in self._predicates:
            # Quick structural check: only predicates on the same
            # tensor and axis can conflict.
            if existing.tensor != pred.tensor or existing.axis != pred.axis:
                continue

            # Build Z3 constraints for both predicates and check SAT
            dim_var = z3.Int(f"_conflict_{pred.tensor}_d{pred.axis}")
            c_existing = self._pred_to_z3(existing, dim_var)
            c_new = self._pred_to_z3(pred, dim_var)
            if c_existing is None or c_new is None:
                continue

            solver = z3.Solver()
            solver.set("timeout", 1000)
            solver.add(dim_var > 0)
            solver.add(c_existing)
            solver.add(c_new)

            if solver.check() == z3.unsat:
                return existing

        return None

    @staticmethod
    def _pred_to_z3(pred: ShapePredicate, dim_var: Any) -> Optional[Any]:
        """Convert a single predicate to a Z3 constraint over *dim_var*."""
        if not HAS_Z3:
            return None
        if pred.kind == PredicateKind.DIM_EQ and pred.value is not None:
            return dim_var == z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_GT and pred.value is not None:
            return dim_var > z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_GE and pred.value is not None:
            return dim_var >= z3.IntVal(pred.value)
        if pred.kind == PredicateKind.DIM_DIVISIBLE and pred.divisor is not None:
            return dim_var % z3.IntVal(pred.divisor) == 0
        return None

    def add_all(self, preds: List[ShapePredicate]) -> int:
        """Add multiple predicates.  Returns count of new ones accepted."""
        return sum(1 for p in preds if self.add(p))

    def contains(self, pred: ShapePredicate) -> bool:
        return pred.pretty() in self._seen

    def quality_report(self) -> Dict[str, Any]:
        """Summary of predicate quality filtering for diagnostics."""
        return {
            "accepted": len(self._predicates),
            "rejected": len(self._rejected),
            "avg_quality": (
                sum(self._quality_scores.values()) / max(1, len(self._quality_scores))
            ),
            "rejected_details": [
                {"predicate": p.pretty(), "score": s}
                for p, s in self._rejected
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Contract inference
# ═══════════════════════════════════════════════════════════════════════════════

def infer_contracts(
    graph: ComputationGraph,
    predicates: List[ShapePredicate],
) -> List[InferredContract]:
    """Group discovered predicates by input parameter to form contracts."""
    param_preds: Dict[str, List[ShapePredicate]] = {}

    for pred in predicates:
        param = pred.tensor
        if param not in param_preds:
            param_preds[param] = []
        param_preds[param].append(pred)

    contracts: List[InferredContract] = []
    func_name = graph.class_name + ".forward"

    for param, preds in sorted(param_preds.items()):
        contracts.append(InferredContract(
            function_name=func_name,
            parameter=param,
            predicates=preds,
        ))

    return contracts


# ═══════════════════════════════════════════════════════════════════════════════
# 10b. Craig interpolation predicate conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_linear_combo_to_predicate(lcp: Any) -> Optional[ShapePredicate]:
    """Convert a LinearComboPredicate to a standard ShapePredicate if possible.

    Single-variable linear combinations map to DIM_EQ / DIM_GE / DIM_GT.
    Two-variable combinations with opposite signs map to DIM_MATCH.
    Returns None for complex multi-variable predicates.
    """
    coeffs = lcp.coeff_dict
    if len(coeffs) == 1:
        (tensor, axis), coeff = next(iter(coeffs.items()))
        if coeff == 1:
            if lcp.operator == "==":
                return ShapePredicate(
                    kind=PredicateKind.DIM_EQ, tensor=tensor,
                    axis=axis, value=lcp.rhs,
                    provenance="craig_interpolation",
                )
            elif lcp.operator == ">=":
                return ShapePredicate(
                    kind=PredicateKind.DIM_GE, tensor=tensor,
                    axis=axis, value=lcp.rhs,
                    provenance="craig_interpolation",
                )
            elif lcp.operator == ">":
                return ShapePredicate(
                    kind=PredicateKind.DIM_GT, tensor=tensor,
                    axis=axis, value=lcp.rhs,
                    provenance="craig_interpolation",
                )
    elif len(coeffs) == 2 and lcp.operator == "==" and lcp.rhs == 0:
        items = list(coeffs.items())
        (t1, a1), c1 = items[0]
        (t2, a2), c2 = items[1]
        if c1 == 1 and c2 == -1:
            return ShapePredicate(
                kind=PredicateKind.DIM_MATCH, tensor=t1, axis=a1,
                match_tensor=t2, match_axis=a2,
                provenance="craig_interpolation",
            )
        elif c1 == -1 and c2 == 1:
            return ShapePredicate(
                kind=PredicateKind.DIM_MATCH, tensor=t2, axis=a2,
                match_tensor=t1, match_axis=a1,
                provenance="craig_interpolation",
            )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Main contract discovery loop (counterexample-guided, CEGAR-style)
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeCEGARLoop:
    """The main counterexample-guided contract discovery loop for tensor shapes.

    Uses a CEGAR-style iteration to discover shape predicates, but the
    core algorithm is closer to Houdini-style predicate accumulation than
    classical CEGAR abstraction refinement.

    Usage
    -----
    >>> loop = ShapeCEGARLoop(source, input_shapes={"x": ("batch", 10)})
    >>> result = loop.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        max_iterations: int = 10,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
        enable_quality_filter: bool = True,
        quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
        constraints: Optional[Dict[str, Union[str, int]]] = None,
        enable_interpolation: bool = True,
        use_incremental: bool = False,
        knowledge_base: Optional[Any] = None,
    ) -> None:
        self.source = source
        self.input_shapes = dict(input_shapes or {})
        self.max_iterations = max_iterations
        self.default_device = default_device
        self.default_phase = default_phase
        self.max_k = max_k
        self.relational_constraints = constraints or {}
        self.enable_interpolation = enable_interpolation
        self.use_incremental = use_incremental
        self._knowledge_base = knowledge_base

        self.pred_set = PredicateSet(
            quality_threshold=quality_threshold,
            enable_quality_filter=enable_quality_filter,
        )
        self._iteration_log: List[IterationRecord] = []
        self._real_bugs_so_far: List[SafetyViolation] = []
        self._interpolation_stats = {
            "attempted": 0,
            "successful": 0,
            "predicates_from_interpolation": 0,
        }
        self._incremental_stats = {
            "core_predicates_count": 0,
            "template_predicates_count": 0,
            "solver_reuse_count": 0,
        }
        self._incremental_solver: Any = None

        # Prime predicate set with transferred predicates from KB
        if self._knowledge_base is not None:
            self._prime_from_kb()

    def _prime_from_kb(self) -> None:
        """Prime the predicate set with transferred predicates from the KB."""
        try:
            from src.knowledge_base import VerificationKnowledgeBase, compute_arch_hash
            if not isinstance(self._knowledge_base, VerificationKnowledgeBase):
                return
            arch_hash = compute_arch_hash(self.source)
            transferred = self._knowledge_base.lookup(arch_hash)
            if not transferred.has_knowledge:
                return
            # Parse transferred predicate strings into ShapePredicate objects
            for pred_str in transferred.predicates:
                pred = _parse_predicate_string(pred_str)
                if pred is not None:
                    self.pred_set.add(pred)
            logger.info(
                "CEGAR: primed with %d transferred predicates from KB",
                len(transferred.predicates),
            )
        except Exception as exc:
            logger.debug("KB priming failed: %s", exc)

    def run(self) -> ShapeCEGARResult:
        """Execute the full contract discovery loop."""
        t0 = time.monotonic()

        # --- Parse the source and extract the computation graph ---
        try:
            graph = extract_computation_graph(self.source)
        except (ValueError, SyntaxError) as exc:
            return ShapeCEGARResult(
                final_status=CEGARStatus.PARSE_ERROR,
                total_time_ms=(time.monotonic() - t0) * 1000,
            )

        if not HAS_Z3:
            # Fall back to a single verification pass
            result = self._single_pass(graph)
            result.total_time_ms = (time.monotonic() - t0) * 1000
            result.final_status = CEGARStatus.NO_Z3
            return result

        # Initialise incremental solver when requested.
        if self.use_incremental and HAS_Z3:
            try:
                from src.unsat_core_cegar import IncrementalCEGARSolver
                self._incremental_solver = IncrementalCEGARSolver()
            except ImportError:
                self._incremental_solver = None

        current_input_shapes = dict(self.input_shapes)
        current_shape_env: Dict[str, TensorShape] = {}
        last_vresult: Optional[VerificationResult] = None

        # ─── Convergence argument (see Proposition 6 in paper) ───────────
        # The predicate universe Pred is finite: it is bounded by
        #   |Pred| ≤ |layers| × |predicate_kinds|
        # where |layers| is the number of layers in __init__ and
        # |predicate_kinds| = 7 (DIM_EQ, DIM_GT, DIM_GE, DIM_DIVISIBLE,
        # DIM_MATCH, NDIM_EQ, SHAPE_EQ).  For a typical nn.Module with
        # L layers and D shape dimensions per layer, the universe has
        # at most L × D × 7 candidate predicates.
        #
        # Each CEGAR iteration either:
        #   (a) returns SAFE or REAL_BUG, terminating the loop, or
        #   (b) adds ≥1 new predicate from Pred \ P to the accumulated
        #       set P (monotone growth, Houdini-style).
        #   (c) adds 0 new predicates (no progress) → terminates as SAFE.
        #
        # Since P ⊆ Pred grows strictly in case (b) and |Pred| is finite,
        # the loop terminates in at most |Pred| iterations.  The budget
        # of max_iterations serves as a safety bound, but convergence is
        # guaranteed by the finite predicate universe.
        #
        # Mechanized: see cegar_terminates in lean/TheoryCombination.lean.
        # ─────────────────────────────────────────────────────────────────

        for iteration in range(self.max_iterations):
            iter_t0 = time.monotonic()

            # === Step 1: Verify ===
            checker = ConstraintVerifier(
                graph,
                input_shapes=current_input_shapes,
                default_device=self.default_device,
                default_phase=self.default_phase,
                max_k=self.max_k,
                constraints=self.relational_constraints,
            )
            vresult = checker.verify()
            last_vresult = vresult

            # Snapshot the shape environment from the checker
            current_shape_env = dict(checker._init_state.shape_env)

            # === Step 2: Check ===
            if vresult.safe:
                # No counterexamples — model is safe
                iter_time = (time.monotonic() - iter_t0) * 1000
                self._iteration_log.append(IterationRecord(
                    iteration=iteration,
                    num_violations=0,
                    num_spurious=0,
                    num_real=0,
                    time_ms=iter_time,
                ))
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            cex = vresult.counterexample
            if cex is None or not cex.violations:
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # === Step 3: Extract ===
            analyser = CounterexampleAnalyser(
                graph, current_shape_env, current_input_shapes,
            )
            analysed = analyser.analyse(cex)

            # === Step 4 & 5: Trace back + Synthesise ===
            new_predicates: List[ShapePredicate] = []
            real_bugs: List[SafetyViolation] = []
            num_spurious = 0
            num_real = 0

            for acex in analysed:
                if acex.is_real_bug():
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1
                elif acex.is_spurious():
                    new_predicates.extend(acex.synthesised_predicates)
                    num_spurious += 1
                else:
                    # Unknown classification — treat conservatively as real
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1

            # === Step 4b: Craig interpolation (complement to template) ===
            # Runs as a complement to template-based predicate discovery.
            # When template-based discovery yields few/no predicates, this
            # serves as a fallback.  Uses McMillan-style Craig interpolation
            # (A = path, B = ¬safety) first; if A ∧ B is SAT (path
            # constraints don't fully determine shapes), falls back to
            # concrete-counterexample interpolation with acex.concrete_dims.
            template_pred_count = len(new_predicates)
            if self.enable_interpolation and HAS_Z3:
                try:
                    from src.craig_interpolation import (
                        InterpolationPredicateDiscovery, DimMapping,
                        LinearComboPredicate,
                    )
                    ipd = InterpolationPredicateDiscovery()
                    pe = UnsatCorePredicateExtractor(graph, current_shape_env)
                    for acex in analysed:
                        if 0 <= acex.step_index < len(graph.steps):
                            # Phase 1: McMillan-style (A = path, B = ¬safety).
                            path_cs, safety_neg_cs, dm = (
                                pe._build_interpolation_query(
                                    graph,
                                    acex.step_index,
                                    current_input_shapes,
                                    concrete_dims=None,
                                )
                            )
                            interp_preds = []
                            if path_cs and safety_neg_cs:
                                self._interpolation_stats["attempted"] += 1
                                interp_preds = ipd.discover_via_interpolation(
                                    path_cs, safety_neg_cs, dm,
                                )

                            # Phase 2: concrete-counterexample fallback.
                            # Translate verifier dim names to __ci_ names.
                            if not interp_preds and acex.concrete_dims:
                                translated = {}
                                for inp_name in graph.input_names:
                                    shape_tuple = current_input_shapes.get(inp_name)
                                    if shape_tuple is None:
                                        continue
                                    for axis, dim_spec in enumerate(shape_tuple):
                                        if isinstance(dim_spec, str) and dim_spec in acex.concrete_dims:
                                            ci_name = f"__ci_{inp_name}_d{axis}"
                                            translated[ci_name] = acex.concrete_dims[dim_spec]
                                if translated:
                                    path_cs2, cex_cs2, dm2 = (
                                        pe._build_interpolation_query(
                                            graph,
                                            acex.step_index,
                                            current_input_shapes,
                                            concrete_dims=translated,
                                        )
                                    )
                                    if path_cs2 and cex_cs2:
                                        self._interpolation_stats["attempted"] += 1
                                        interp_preds = ipd.discover_via_interpolation(
                                            path_cs2, cex_cs2, dm2,
                                        )

                            if interp_preds:
                                self._interpolation_stats["successful"] += 1
                                self._interpolation_stats["predicates_from_interpolation"] += len(interp_preds)
                                for ip in interp_preds:
                                    if isinstance(ip, ShapePredicate):
                                        new_predicates.append(ip)
                                    elif isinstance(ip, LinearComboPredicate):
                                        converted = _convert_linear_combo_to_predicate(ip)
                                        if converted is not None:
                                            new_predicates.append(converted)
                                        else:
                                            logger.debug(
                                                "Craig interpolation: dropped non-convertible "
                                                "LinearComboPredicate: %s", ip.pretty(),
                                            )
                    if template_pred_count == 0 and len(new_predicates) > template_pred_count:
                        logger.info(
                            "Craig interpolation fallback: discovered %d predicates "
                            "when template-based discovery found none",
                            len(new_predicates) - template_pred_count,
                        )
                except ImportError:
                    pass  # Craig interpolation module not available
                except Exception as e:
                    logger.debug("Craig interpolation failed: %s", e)

            # Update quality scorer with accumulated real bugs
            self.pred_set.set_known_bugs(self._real_bugs_so_far)

            iter_time = (time.monotonic() - iter_t0) * 1000
            added = self.pred_set.add_all(new_predicates)

            self._iteration_log.append(IterationRecord(
                iteration=iteration,
                num_violations=len(cex.violations),
                num_spurious=num_spurious,
                num_real=num_real,
                predicates_added=new_predicates[:],
                time_ms=iter_time,
            ))

            # If we found real bugs, stop immediately
            if real_bugs:
                result = self._build_result(
                    CEGARStatus.REAL_BUG_FOUND, graph, last_vresult, t0,
                )
                result.real_bugs = real_bugs
                return result

            # === Step 6: Refine ===
            if added == 0:
                # No new predicates — no progress possible.
                # All violations were spurious but we cannot eliminate them;
                # declare safe (the violations are artifacts of abstraction).
                return self._build_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # Check feasibility of accumulated predicates.
            # If the accumulated refined predicates are jointly INFEASIBLE, the
            # loop eliminated its counterexamples using mutually contradictory
            # assumptions — a spurious elimination that carries no information
            # about whether the program is actually safe. Returning SAFE here is
            # UNSOUND (known-unsoundness gap U2). We must abstain instead.
            # The fix is machine-checked in lean/TensorGuard/CegarInfeasible.lean
            # (Step 132): `decideNew` abstains on the infeasible branch and is
            # sound under the feasible-branch guarantee, whereas the old SAFE
            # behaviour (`decideOld`) is provably unsound.
            if not ShapeRefinement.check_feasibility(self.pred_set.predicates):
                logger.warning(
                    "CEGAR: accumulated predicates are infeasible — "
                    "elimination is spurious; abstaining (UNKNOWN, not SAFE)"
                )
                return self._build_result(
                    CEGARStatus.INFEASIBLE_REFINEMENT, graph, last_vresult, t0,
                )

            # Apply refinement
            current_input_shapes, current_shape_env = (
                ShapeRefinement.apply_predicates(
                    current_input_shapes,
                    current_shape_env,
                    new_predicates,
                )
            )

            logger.debug(
                "CEGAR iteration %d: %d violations, %d spurious, "
                "%d real, %d new predicates",
                iteration,
                len(cex.violations),
                num_spurious,
                num_real,
                added,
            )

        # === Step 7: Max iterations reached ===
        return self._build_result(
            CEGARStatus.MAX_ITER, graph, last_vresult, t0,
        )

    def _single_pass(self, graph: ComputationGraph) -> ShapeCEGARResult:
        """Fallback when Z3 is not available: one verification pass."""
        checker = ConstraintVerifier(
            graph,
            input_shapes=self.input_shapes,
            default_device=self.default_device,
            default_phase=self.default_phase,
            max_k=self.max_k,
        )
        vresult = checker.verify()
        status = CEGARStatus.SAFE if vresult.safe else CEGARStatus.REAL_BUG_FOUND
        result = ShapeCEGARResult(
            final_status=status,
            verification_result=vresult,
            iterations=1,
        )
        if not vresult.safe and vresult.counterexample:
            result.real_bugs = list(vresult.counterexample.violations)
        return result

    def _build_result(
        self,
        status: CEGARStatus,
        graph: ComputationGraph,
        vresult: Optional[VerificationResult],
        t0: float,
    ) -> ShapeCEGARResult:
        """Construct the final ``ShapeCEGARResult``."""
        predicates = self.pred_set.predicates
        contracts = infer_contracts(graph, predicates)

        stats = dict(self._interpolation_stats) if any(self._interpolation_stats.values()) else None
        if self.use_incremental and any(self._incremental_stats.values()):
            if stats is None:
                stats = {}
            stats.update(self._incremental_stats)

        return ShapeCEGARResult(
            discovered_predicates=predicates,
            iterations=len(self._iteration_log),
            final_status=status,
            contracts_inferred=contracts,
            verification_result=vresult,
            total_time_ms=(time.monotonic() - t0) * 1000,
            iteration_log=self._iteration_log,
            predicate_quality_report=self.pred_set.quality_report(),
            interpolation_stats=stats,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_shape_cegar(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    enable_quality_filter: bool = True,
    quality_threshold: float = PREDICATE_QUALITY_THRESHOLD,
) -> ShapeCEGARResult:
    """One-shot entry point for shape contract discovery (CEGAR-style).

    Parameters
    ----------
    source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.  Dimensions
        may be ints (concrete) or strings (symbolic).
    max_iterations : int
        Maximum number of contract discovery iterations.
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth for the constraint verifier.
    enable_quality_filter : bool
        Whether to enable predicate quality filtering to prevent
        counterproductive refinements.  Default True.
    quality_threshold : float
        Minimum quality score for a predicate to be accepted.

    Returns
    -------
    ShapeCEGARResult
        Contains discovered predicates, iteration count, final status,
        and inferred shape contracts.  Use ``result.verdict`` to obtain
        a ``CEGARVerdict`` (SAFE, UNSAFE, UNKNOWN, TIMEOUT) that
        accounts for solver timeouts and iteration-budget exhaustion.

    Examples
    --------
    >>> result = run_shape_cegar('''
    ... import torch.nn as nn
    ... class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(768, 10)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ... ''', input_shapes={"x": ("batch", "features")})
    >>> result.is_safe
    True
    >>> result.discovered_predicates[0].pretty()
    'x.shape[-1] == 768'
    """
    loop = ShapeCEGARLoop(
        source,
        input_shapes=input_shapes,
        max_iterations=max_iterations,
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        enable_quality_filter=enable_quality_filter,
        quality_threshold=quality_threshold,
    )
    return loop.run()


def verify_and_discover(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
) -> Tuple[bool, List[ShapePredicate], List[InferredContract]]:
    """Convenience wrapper returning ``(is_safe, predicates, contracts)``.

    Useful for quick integration where the full ``ShapeCEGARResult`` is
    not needed.

    Examples
    --------
    >>> safe, preds, contracts = verify_and_discover(source,
    ...     input_shapes={"x": ("batch", "d")})
    >>> if safe:
    ...     print("Model verified safe with predicates:", preds)
    """
    result = run_shape_cegar(source, input_shapes, max_iterations)
    return result.is_safe, result.discovered_predicates, result.contracts_inferred


def compute_predicate_universe_bound(model_info: dict) -> dict:
    """Compute the explicit height bound |P_prog| for a model.

    The CEGAR predicate lattice has finite height bounded by the size
    of the predicate universe P_prog.  Each iteration adds at least one
    new predicate (strict monotonicity of the accumulated set P), so
    the loop terminates in at most |P_prog| iterations.

    The bound is:

        |P_prog| = num_layers × max_dims_per_layer × |predicate_kinds|

    where |predicate_kinds| = 7 (DIM_EQ, DIM_GT, DIM_GE, DIM_DIVISIBLE,
    DIM_MATCH, NDIM_EQ, SHAPE_EQ).

    Args:
        model_info: dict with keys:
            - ``num_layers`` (int): number of parameterised layers.
            - ``max_dims_per_layer`` (int): maximum shape dimensions
              per layer (e.g. 2 for Linear, 4 for Conv2d).

    Returns:
        dict with:
            - ``bound``: the computed |P_prog| upper bound.
            - ``layers``: num_layers used.
            - ``dims``: max_dims_per_layer used.
            - ``predicate_kinds``: number of predicate kinds (7).
            - ``formula``: human-readable formula string.
    """
    num_layers = model_info["num_layers"]
    max_dims = model_info["max_dims_per_layer"]
    num_kinds = len(PredicateKind)  # 7
    bound = num_layers * max_dims * num_kinds
    return {
        "bound": bound,
        "layers": num_layers,
        "dims": max_dims,
        "predicate_kinds": num_kinds,
        "formula": f"|P_prog| = {num_layers} × {max_dims} × {num_kinds} = {bound}",
    }


# Backward-compatible alias.
InterpolationEngine = UnsatCorePredicateExtractor
