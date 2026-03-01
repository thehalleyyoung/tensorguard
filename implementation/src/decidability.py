"""
Decidability characterization for the combined theory T_shape × T_device × T_phase.

This module formally classifies the complexity of verification queries over the
product theory used by the TensorGuard verifier.  The three component theories are:

T_shape (Shape theory)
    Domain: Dim ⊆ Z_≥1 (positive integers).
    Fragments:
      • Linear arithmetic (dim equalities/inequalities, broadcasting via
        element-wise max of compatible dims, matmul inner-dim equality).
        Reduces to QF_LIA (Presburger arithmetic) — decidable in **P** for
        fixed-rank concrete shapes (polynomial in the number of constraints).
      • Reshape: product-equality constraint d1·d2·…·dk = d1'·d2'·…·dk'.
        This is non-linear integer multiplication (QF_NIA).  Unrestricted
        QF_NIA over unbounded integers is undecidable (Matiyasevich, 1970).
        However, TensorGuard operates in three sub-fragments:
          – Concrete–symbolic multiplication (one factor known): reducible
            to QF_LIA, decidable in P.
          – Bounded symbolic multiplication (all factors finitely bounded):
            decidable via bit-blasting, NP-hard.
          – Unbounded symbolic multiplication: enters the undecidable
            fragment (Matiyasevich's theorem / Hilbert's 10th problem).
        TensorGuard restricts reshape to the SUBSET-PRODUCT reduction
        (proven NP-hard in the companion Lean formalisation), so the
        reshape fragment is **NP-hard** when variables are bounded.

T_device (Device theory)
    Domain: {CPU, CUDA:0, CUDA:1, CUDA:2, CUDA:3} — 5 elements.
    Decidability: trivially decidable (finite model checking, O(1) per
    constraint).

T_phase (Phase theory)
    Domain: {TRAIN, EVAL} — 2 elements.
    Decidability: trivially decidable (finite model checking, O(1) per
    constraint).

Combined theory (Tinelli–Zarba arrangement)
    The Nelson–Oppen combination requires theories over disjoint signatures
    to be stably infinite.  Finite-domain theories (T_device, T_phase) are
    *not* stably infinite, so we use the Tinelli–Zarba extension (JAR 2005)
    which enumerates arrangements (equivalence-class partitions) over shared
    variables.  For a finite domain of size n and k shared variables, the
    number of arrangements is bounded by S(k, min(k, n)) (Stirling numbers
    of the second kind), which is polynomial for the small k and n that
    arise in practice.

    The combined theory inherits NP-hardness from T_shape (reshape).
    Without reshape constraints, the combined theory is in **P**.

Reference:
    C. Tinelli and C. Zarba, "Combining Nonstably Infinite Theories",
    Journal of Automated Reasoning 34(3), 2005.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import z3


# ── Complexity classes ───────────────────────────────────────────────────────

class ComplexityClass(Enum):
    """Complexity classification for a verification query."""
    P = "P"
    NP_HARD = "NP-hard"
    # Backward-compatible alias.
    NP_COMPLETE = "NP-hard"


class TheoryFragment(Enum):
    """Identifiers for the individual theory fragments."""
    T_SHAPE_LINEAR = auto()    # QF_LIA: dim equalities, inequalities
    T_SHAPE_BROADCAST = auto() # broadcasting (max of compatible dims) — linear
    T_SHAPE_MATMUL = auto()    # inner-dim equality — linear
    T_SHAPE_RESHAPE = auto()   # product-equality (NP-hard)
    T_DEVICE = auto()          # finite domain, 5 elements
    T_PHASE = auto()           # finite domain, 2 elements


# Operations that use only linear-arithmetic shape reasoning (P fragment).
_LINEAR_OPS: FrozenSet[str] = frozenset({
    "LAYER_CALL",
    "MATMUL",
    "ADD",
    "CAT",
    "TRANSPOSE",
    "PERMUTE",
    "SQUEEZE",
    "UNSQUEEZE",
    "ACTIVATION",
    "DROPOUT",
    "SOFTMAX",
    "TO_DEVICE",
    "DETACH",
    "CONTIGUOUS",
    "CONDITIONAL",
    "CUSTOM",
    "MULTIPLY",
    "INTERPOLATE",
    "RETURN",
})

# Operations that introduce the NP-hard reshape fragment.
_RESHAPE_OPS: FrozenSet[str] = frozenset({
    "RESHAPE",
    "FLATTEN",
})

# Device-related operations.
_DEVICE_OPS: FrozenSet[str] = frozenset({
    "TO_DEVICE",
})

# Phase-sensitive operations.
_PHASE_OPS: FrozenSet[str] = frozenset({
    "DROPOUT",
    "CONDITIONAL",
})


# ── Query representation ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class VerificationQuery:
    """A lightweight description of the operations appearing in a verification query.

    Attributes:
        operations: set of operation name strings (matching ``OpKind`` names
            from the model checker, e.g. ``"RESHAPE"``, ``"MATMUL"``).
        has_device_constraints: whether the query involves device-consistency
            constraints.
        has_phase_constraints: whether the query involves phase-sensitive
            constraints.
    """
    operations: FrozenSet[str] = field(default_factory=frozenset)
    has_device_constraints: bool = False
    has_phase_constraints: bool = False


# ── Theory-fragment identification ────────────────────────────────────────────

def identify_fragments(query: VerificationQuery) -> Set[TheoryFragment]:
    """Return the set of theory fragments exercised by *query*.

    Every query uses at least T_SHAPE_LINEAR (dimension tracking is always
    present).  Specific operations additionally exercise T_SHAPE_BROADCAST,
    T_SHAPE_MATMUL, or T_SHAPE_RESHAPE.
    """
    fragments: Set[TheoryFragment] = {TheoryFragment.T_SHAPE_LINEAR}

    for op in query.operations:
        if op in _RESHAPE_OPS:
            fragments.add(TheoryFragment.T_SHAPE_RESHAPE)
        if op in {"ADD", "MULTIPLY"}:
            fragments.add(TheoryFragment.T_SHAPE_BROADCAST)
        if op == "MATMUL":
            fragments.add(TheoryFragment.T_SHAPE_MATMUL)

    if query.has_device_constraints or (query.operations & _DEVICE_OPS):
        fragments.add(TheoryFragment.T_DEVICE)

    if query.has_phase_constraints or (query.operations & _PHASE_OPS):
        fragments.add(TheoryFragment.T_PHASE)

    return fragments


# ── Complexity classification ────────────────────────────────────────────────

def classify_query_complexity(query: VerificationQuery) -> ComplexityClass:
    """Classify the worst-case complexity of deciding *query*.

    Returns ``ComplexityClass.NP_HARD`` when the query involves a reshape
    or flatten operation (product-equality constraint, SUBSET-PRODUCT
    reduction).  Otherwise returns ``ComplexityClass.P`` — the linear-
    arithmetic fragment (QF_LIA) combined with finite-domain theories
    (T_device, T_phase) via Tinelli–Zarba is decidable in polynomial time.
    """
    fragments = identify_fragments(query)
    if TheoryFragment.T_SHAPE_RESHAPE in fragments:
        return ComplexityClass.NP_HARD
    return ComplexityClass.P


# ── Decidability summaries ───────────────────────────────────────────────────

@dataclass
class DecidabilitySummary:
    """Human-readable decidability characterization of a verification query.

    Attributes:
        complexity: the worst-case complexity class.
        fragments: the set of theory fragments exercised.
        explanation: a short textual explanation suitable for logging or
            display in verification conditions.
    """
    complexity: ComplexityClass
    fragments: Set[TheoryFragment]
    explanation: str


def summarize_decidability(query: VerificationQuery) -> DecidabilitySummary:
    """Produce a full decidability summary for *query*."""
    fragments = identify_fragments(query)
    complexity = classify_query_complexity(query)

    parts: List[str] = []

    # Shape fragment description.
    if TheoryFragment.T_SHAPE_RESHAPE in fragments:
        parts.append(
            "T_shape includes reshape (product-equality), "
            "making this fragment NP-hard (SUBSET-PRODUCT reduction)."
        )
    else:
        parts.append(
            "T_shape uses only linear arithmetic (QF_LIA), decidable in P."
        )

    # Device / phase.
    if TheoryFragment.T_DEVICE in fragments:
        parts.append(
            "T_device: finite domain (5 elements), trivially decidable."
        )
    if TheoryFragment.T_PHASE in fragments:
        parts.append(
            "T_phase: finite domain (2 elements), trivially decidable."
        )

    # Combination note.
    n_finite = sum(
        1
        for f in fragments
        if f in {TheoryFragment.T_DEVICE, TheoryFragment.T_PHASE}
    )
    if n_finite > 0:
        parts.append(
            "Combined via Tinelli-Zarba arrangement enumeration "
            "(polynomial in shared variables for finite domains)."
        )

    explanation = " ".join(parts)
    return DecidabilitySummary(
        complexity=complexity,
        fragments=fragments,
        explanation=explanation,
    )


# ── Relational-constraint SMT-fragment classification ────────────────────────

class RelationalConstraintClass(Enum):
    """SMT theory fragment for a relational shape constraint."""
    QF_LIA_REDUCIBLE = "QF_LIA-reducible"
    QF_NIA = "QF_NIA"


@dataclass(frozen=True)
class RelationalConstraintInfo:
    """Classification result for a single relational constraint.

    Attributes:
        lhs: the left-hand side dimension name (e.g. ``"embed_dim"``).
        expression: the right-hand side expression string (e.g.
            ``"heads * head_dim"``).
        classification: whether the constraint is QF_LIA-reducible or
            genuine QF_NIA.
        symbolic_vars: names of symbolic (non-constant) variables in the
            expression.
        reason: human-readable explanation of the classification.
    """
    lhs: str
    expression: str
    classification: RelationalConstraintClass
    symbolic_vars: List[str]
    reason: str


class RelationalConstraintClassifier:
    """Classifies relational shape constraints by SMT theory fragment.

    A constraint like ``embed_dim = heads * head_dim`` is classified as:

    * **QF_LIA_REDUCIBLE** when at most one variable in any multiplicative
      term is truly symbolic (i.e. the other factor is a concrete integer).
      Example: ``embed_dim = 8 * head_dim`` — the ``8`` is concrete, so
      this is a linear constraint over ``head_dim``.

    * **QF_NIA** when two or more symbolic variables are multiplied
      together.  Example: ``embed_dim = heads * head_dim`` with both
      ``heads`` and ``head_dim`` symbolic.

    Parameters:
        concrete_dims: mapping of dimension names to their concrete integer
            values.  Any dimension *not* in this mapping is treated as
            symbolic.
    """

    def __init__(self, concrete_dims: Optional[Dict[str, int]] = None) -> None:
        self.concrete_dims: Dict[str, int] = dict(concrete_dims or {})

    # ── public API ───────────────────────────────────────────────────────

    def classify(
        self,
        lhs: str,
        expr: Union[str, int],
    ) -> RelationalConstraintInfo:
        """Classify a single relational constraint ``lhs = expr``."""
        if isinstance(expr, int):
            return RelationalConstraintInfo(
                lhs=lhs,
                expression=str(expr),
                classification=RelationalConstraintClass.QF_LIA_REDUCIBLE,
                symbolic_vars=[],
                reason="Concrete integer assignment — trivially linear.",
            )

        tree = ast.parse(expr, mode="eval")
        symbolic = self._collect_symbolic_vars(tree.body)
        has_nonlinear = self._has_nonlinear_product(tree.body)

        if has_nonlinear:
            cls = RelationalConstraintClass.QF_NIA
            reason = (
                f"Multiplicative term with ≥2 symbolic variables "
                f"({', '.join(sorted(symbolic))}); enters QF_NIA."
            )
        else:
            cls = RelationalConstraintClass.QF_LIA_REDUCIBLE
            if symbolic:
                reason = (
                    "At most one symbolic variable per multiplicative term; "
                    "reducible to QF_LIA."
                )
            else:
                reason = "All variables are concrete — trivially linear."

        return RelationalConstraintInfo(
            lhs=lhs,
            expression=expr,
            classification=cls,
            symbolic_vars=sorted(symbolic),
            reason=reason,
        )

    def classify_all(
        self,
        constraints: Dict[str, Union[str, int]],
    ) -> List[RelationalConstraintInfo]:
        """Classify every constraint in a constraints dict."""
        return [self.classify(lhs, rhs) for lhs, rhs in constraints.items()]

    # ── AST helpers ──────────────────────────────────────────────────────

    def _is_symbolic(self, name: str) -> bool:
        return name not in self.concrete_dims

    def _collect_symbolic_vars(self, node: ast.AST) -> Set[str]:
        """Return the set of symbolic variable names in *node*."""
        if isinstance(node, ast.Name):
            return {node.id} if self._is_symbolic(node.id) else set()
        if isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.BinOp):
            return (self._collect_symbolic_vars(node.left)
                    | self._collect_symbolic_vars(node.right))
        if isinstance(node, ast.UnaryOp):
            return self._collect_symbolic_vars(node.operand)
        if isinstance(node, ast.Expression):
            return self._collect_symbolic_vars(node.body)
        return set()

    def _has_nonlinear_product(self, node: ast.AST) -> bool:
        """Return ``True`` if *node* contains ``sym * sym``."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_sym = self._collect_symbolic_vars(node.left)
            right_sym = self._collect_symbolic_vars(node.right)
            if left_sym and right_sym:
                return True
        # Recurse into sub-expressions.
        if isinstance(node, ast.BinOp):
            return (self._has_nonlinear_product(node.left)
                    or self._has_nonlinear_product(node.right))
        if isinstance(node, ast.UnaryOp):
            return self._has_nonlinear_product(node.operand)
        if isinstance(node, ast.Expression):
            return self._has_nonlinear_product(node.body)
        return False


def classify_relational_constraint(
    lhs: str,
    expr: Union[str, int],
    concrete_dims: Optional[Dict[str, int]] = None,
) -> RelationalConstraintInfo:
    """Convenience wrapper: classify a single relational constraint.

    Parameters:
        lhs: left-hand side dimension name.
        expr: right-hand side — integer literal or expression string.
        concrete_dims: dimensions with known concrete values.

    Returns:
        A :class:`RelationalConstraintInfo` with the classification.
    """
    return RelationalConstraintClassifier(concrete_dims).classify(lhs, expr)


# ── Z3-based NIA fragment analysis ──────────────────────────────────────────

@dataclass
class NIAAnalysisResult:
    """Result of running Z3 on an NIA-fragment constraint system.

    Attributes:
        status: Z3 check result as a string (``"sat"``, ``"unsat"``,
            ``"unknown"``).
        elapsed_s: wall-clock time in seconds.
        timed_out: whether the solver exceeded the timeout.
        model: if SAT, a dict mapping variable names to their Z3 model
            values (as Python ints); ``None`` otherwise.
    """
    status: str
    elapsed_s: float
    timed_out: bool
    model: Optional[Dict[str, int]] = None


def analyze_nia_fragment(
    constraints: Dict[str, Union[str, int]],
    *,
    extra_bounds: Optional[Dict[str, Tuple[int, int]]] = None,
    timeout_ms: int = 5000,
) -> NIAAnalysisResult:
    """Test Z3's behaviour on an NIA constraint system.

    Builds a Z3 solver in the QF_NIA logic, asserts each constraint in
    *constraints* (``lhs == parse(rhs)``), optionally adds variable bounds
    from *extra_bounds*, and checks satisfiability.

    Parameters:
        constraints: mapping ``dim_name -> expr_or_int`` (same format as
            :pymethod:`model_checker._Z3Context.build_relational_constraints`).
        extra_bounds: optional ``{var: (lo, hi)}`` inclusive bounds.
        timeout_ms: Z3 timeout in milliseconds.

    Returns:
        An :class:`NIAAnalysisResult` with status, timing, and model.
    """
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    sym: Dict[str, z3.ArithRef] = {}

    def _get(name: str) -> z3.ArithRef:
        if name not in sym:
            sym[name] = z3.Int(name)
        return sym[name]

    def _parse(node: ast.AST) -> z3.ArithRef:
        if isinstance(node, ast.Expression):
            return _parse(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return z3.IntVal(node.value)
        if isinstance(node, ast.Name):
            return _get(node.id)
        if isinstance(node, ast.BinOp):
            l, r = _parse(node.left), _parse(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.FloorDiv):
                return l / r
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_parse(node.operand)
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

    # Assert constraints.
    for dim_name, value in constraints.items():
        lhs = _get(dim_name)
        if isinstance(value, int):
            solver.add(lhs == z3.IntVal(value))
        else:
            tree = ast.parse(str(value), mode="eval")
            solver.add(lhs == _parse(tree))

    # Assert bounds (positive dims by default).
    bounds = dict(extra_bounds or {})
    for name in sym:
        if name not in bounds:
            bounds[name] = (1, 4096)
    for name, (lo, hi) in bounds.items():
        v = _get(name)
        solver.add(v >= lo, v <= hi)

    t0 = time.perf_counter()
    result = solver.check()
    elapsed = time.perf_counter() - t0

    status_str = str(result)
    timed_out = status_str == "unknown"

    model_dict: Optional[Dict[str, int]] = None
    if result == z3.sat:
        m = solver.model()
        model_dict = {}
        for name, var in sym.items():
            val = m.evaluate(var, model_completion=True)
            try:
                model_dict[name] = val.as_long()
            except Exception:
                model_dict[name] = str(val)

    return NIAAnalysisResult(
        status=status_str,
        elapsed_s=elapsed,
        timed_out=timed_out,
        model=model_dict,
    )


# ── QF_NIA decidable fragment syntactic characterization ─────────────────────

class NIADecidableFragment(Enum):
    """Classification of multiplication patterns by decidability.

    Characterizes whether a nonlinear integer arithmetic constraint
    falls in a decidable or undecidable fragment of QF_NIA.
    """

    CONCRETE_SYMBOLIC_MUL = "concrete_symbolic_mul"
    """One factor is a concrete integer, one is symbolic.
    Reducible to QF_LIA by substituting the concrete value.
    Example: ``8 * head_dim`` → linear in ``head_dim``.
    Always decidable (polynomial time via QF_LIA solvers).
    """

    BOUNDED_SYMBOLIC_MUL = "bounded_symbolic_mul"
    """Both factors are symbolic but have finite upper bounds.
    Decidable via bit-blasting: encode each bounded integer as
    a fixed-width bit-vector and reduce to SAT.
    Example: ``heads * head_dim`` with ``1 ≤ heads ≤ 128``,
    ``1 ≤ head_dim ≤ 512``.
    Decidable but potentially expensive (NP-hard via bit-blasting).
    """

    UNBOUNDED_SYMBOLIC_MUL = "unbounded_symbolic_mul"
    """Both factors are symbolic with no finite upper bound.
    Enters the full QF_NIA fragment, which is undecidable in general
    (Matiyasevich's theorem / Hilbert's 10th problem).
    Example: ``factor_a * factor_b`` with no bounds on either.
    No algorithm can decide satisfiability in the general case.
    """

    LINEAR = "linear"
    """No symbolic-symbolic multiplication present.
    The constraint is in QF_LIA (Presburger arithmetic), decidable in P.
    """


@dataclass(frozen=True)
class ConstraintFragmentInfo:
    """Classification of a single constraint's decidable fragment.

    Attributes
    ----------
    expression : str
        The constraint expression.
    fragment : NIADecidableFragment
        Which decidable fragment the constraint falls into.
    symbolic_factors : list of tuple
        Pairs of symbolic variables in multiplicative terms.
    bounded_factors : dict
        Factors with known finite bounds: ``{name: (lo, hi)}``.
    unbounded_factors : list of str
        Factors with no known finite bound.
    reason : str
        Human-readable explanation of the classification.
    decidable : bool
        Whether the constraint is in a decidable fragment.
    """

    expression: str
    fragment: NIADecidableFragment
    symbolic_factors: List[Tuple[str, str]] = field(default_factory=list)
    bounded_factors: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    unbounded_factors: List[str] = field(default_factory=list)
    reason: str = ""
    decidable: bool = True


@dataclass
class DecidableFragmentReport:
    """Statistics on the decidable fragment analysis of a constraint system.

    Attributes
    ----------
    total_constraints : int
        Total number of constraints analyzed.
    linear_count : int
        Constraints in QF_LIA (no symbolic multiplication).
    concrete_symbolic_count : int
        Constraints with one concrete, one symbolic factor.
    bounded_symbolic_count : int
        Constraints with bounded symbolic-symbolic multiplication.
    unbounded_symbolic_count : int
        Constraints with unbounded symbolic-symbolic multiplication.
    all_decidable : bool
        Whether all constraints fall in decidable fragments.
    constraint_details : list of ConstraintFragmentInfo
        Per-constraint classification details.
    warnings : list of str
        Warnings for constraints in undecidable fragments.
    """

    total_constraints: int = 0
    linear_count: int = 0
    concrete_symbolic_count: int = 0
    bounded_symbolic_count: int = 0
    unbounded_symbolic_count: int = 0
    all_decidable: bool = True
    constraint_details: List[ConstraintFragmentInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def enforce_decidable_fragment(
    constraints: Dict[str, Union[str, int]],
    concrete_dims: Optional[Dict[str, int]] = None,
    bounded_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    reject_unbounded: bool = False,
) -> DecidableFragmentReport:
    """Syntactically inspect constraints and classify each by NIA fragment.

    For each constraint, determines whether it falls in:
    - LINEAR (QF_LIA, no symbolic multiplication)
    - CONCRETE_SYMBOLIC_MUL (reducible to QF_LIA)
    - BOUNDED_SYMBOLIC_MUL (decidable via bit-blasting)
    - UNBOUNDED_SYMBOLIC_MUL (enters undecidable QF_NIA)

    Parameters
    ----------
    constraints : dict
        Mapping ``dim_name -> expr_or_int``.
    concrete_dims : dict, optional
        Known concrete dimension values.
    bounded_dims : dict, optional
        Known finite bounds: ``{name: (lo, hi)}``.
    reject_unbounded : bool
        If True, raises ValueError on unbounded symbolic multiplication.

    Returns
    -------
    DecidableFragmentReport
        Statistics and per-constraint details.
    """
    concrete = dict(concrete_dims or {})
    bounds = dict(bounded_dims or {})
    report = DecidableFragmentReport()
    details: List[ConstraintFragmentInfo] = []

    for dim_name, expr in constraints.items():
        expr_str = str(expr)
        report.total_constraints += 1

        if isinstance(expr, int):
            info = ConstraintFragmentInfo(
                expression=f"{dim_name} = {expr}",
                fragment=NIADecidableFragment.LINEAR,
                reason="Concrete integer assignment — trivially linear.",
                decidable=True,
            )
            details.append(info)
            report.linear_count += 1
            continue

        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError:
            info = ConstraintFragmentInfo(
                expression=f"{dim_name} = {expr_str}",
                fragment=NIADecidableFragment.LINEAR,
                reason=f"Unparseable expression — treated as linear.",
                decidable=True,
            )
            details.append(info)
            report.linear_count += 1
            continue

        classifier = RelationalConstraintClassifier(concrete)
        rc = classifier.classify(dim_name, expr_str)

        if rc.classification == RelationalConstraintClass.QF_LIA_REDUCIBLE:
            info = ConstraintFragmentInfo(
                expression=f"{dim_name} = {expr_str}",
                fragment=NIADecidableFragment.LINEAR,
                reason=rc.reason,
                decidable=True,
            )
            details.append(info)
            report.linear_count += 1
            continue

        # QF_NIA — further classify by boundedness
        sym_vars = rc.symbolic_vars
        mul_pairs = _find_symbolic_mul_pairs(tree.body, concrete)

        if not mul_pairs:
            # Has symbolic vars but no actual symbolic multiplication
            if any(v not in concrete for v in sym_vars):
                info = ConstraintFragmentInfo(
                    expression=f"{dim_name} = {expr_str}",
                    fragment=NIADecidableFragment.CONCRETE_SYMBOLIC_MUL,
                    reason="Symbolic variables present but no symbolic-symbolic product.",
                    decidable=True,
                )
                details.append(info)
                report.concrete_symbolic_count += 1
            else:
                info = ConstraintFragmentInfo(
                    expression=f"{dim_name} = {expr_str}",
                    fragment=NIADecidableFragment.LINEAR,
                    reason="All variables concrete.",
                    decidable=True,
                )
                details.append(info)
                report.linear_count += 1
            continue

        # Determine if all factors in multiplication are bounded
        all_bounded = True
        unbounded: List[str] = []
        bounded_in_expr: Dict[str, Tuple[int, int]] = {}
        for v1, v2 in mul_pairs:
            for v in (v1, v2):
                if v in concrete:
                    continue
                if v in bounds:
                    bounded_in_expr[v] = bounds[v]
                else:
                    all_bounded = False
                    if v not in unbounded:
                        unbounded.append(v)

        if all_bounded and not unbounded:
            # Check if one factor is concrete → CONCRETE_SYMBOLIC_MUL
            has_concrete_factor = any(
                v1 in concrete or v2 in concrete for v1, v2 in mul_pairs
            )
            if has_concrete_factor:
                info = ConstraintFragmentInfo(
                    expression=f"{dim_name} = {expr_str}",
                    fragment=NIADecidableFragment.CONCRETE_SYMBOLIC_MUL,
                    symbolic_factors=mul_pairs,
                    bounded_factors=bounded_in_expr,
                    reason=(
                        "One factor in each product is concrete; "
                        "reducible to QF_LIA."
                    ),
                    decidable=True,
                )
                details.append(info)
                report.concrete_symbolic_count += 1
            else:
                info = ConstraintFragmentInfo(
                    expression=f"{dim_name} = {expr_str}",
                    fragment=NIADecidableFragment.BOUNDED_SYMBOLIC_MUL,
                    symbolic_factors=mul_pairs,
                    bounded_factors=bounded_in_expr,
                    reason=(
                        "Both factors symbolic but bounded; decidable "
                        "via bit-blasting (encode as fixed-width BV)."
                    ),
                    decidable=True,
                )
                details.append(info)
                report.bounded_symbolic_count += 1
        else:
            # Unbounded symbolic-symbolic multiplication
            info = ConstraintFragmentInfo(
                expression=f"{dim_name} = {expr_str}",
                fragment=NIADecidableFragment.UNBOUNDED_SYMBOLIC_MUL,
                symbolic_factors=mul_pairs,
                bounded_factors=bounded_in_expr,
                unbounded_factors=sorted(unbounded),
                reason=(
                    f"Unbounded symbolic factors ({', '.join(sorted(unbounded))}); "
                    f"enters undecidable QF_NIA fragment "
                    f"(Matiyasevich, 1970)."
                ),
                decidable=False,
            )
            details.append(info)
            report.unbounded_symbolic_count += 1
            report.all_decidable = False
            report.warnings.append(
                f"Constraint '{dim_name} = {expr_str}' has unbounded "
                f"symbolic multiplication ({', '.join(sorted(unbounded))}). "
                f"Consider adding finite bounds to reduce to decidable fragment."
            )

            if reject_unbounded:
                raise ValueError(
                    f"Unbounded symbolic multiplication in '{dim_name} = {expr_str}': "
                    f"variables {sorted(unbounded)} have no finite upper bound. "
                    f"Add bounds via bounded_dims parameter."
                )

    report.constraint_details = details
    return report


def _find_symbolic_mul_pairs(
    node: ast.AST, concrete: Dict[str, int]
) -> List[Tuple[str, str]]:
    """Find pairs of symbolic variables in multiplicative terms."""
    pairs: List[Tuple[str, str]] = []

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_sym = RelationalConstraintClassifier(concrete)._collect_symbolic_vars(node.left)
        right_sym = RelationalConstraintClassifier(concrete)._collect_symbolic_vars(node.right)
        if left_sym and right_sym:
            for v1 in sorted(left_sym):
                for v2 in sorted(right_sym):
                    pairs.append((v1, v2))

    if isinstance(node, ast.BinOp):
        pairs.extend(_find_symbolic_mul_pairs(node.left, concrete))
        pairs.extend(_find_symbolic_mul_pairs(node.right, concrete))
    if isinstance(node, ast.UnaryOp):
        pairs.extend(_find_symbolic_mul_pairs(node.operand, concrete))
    if isinstance(node, ast.Expression):
        pairs.extend(_find_symbolic_mul_pairs(node.body, concrete))

    return pairs


# ── Public constraint-fragment classification API ────────────────────────────

def classify_constraint_fragment(
    constraint: Union[str, int],
    concrete_dims: Optional[Dict[str, int]] = None,
) -> str:
    """Determine if a constraint expression is QF_LIA or QF_NIA.

    Parameters
    ----------
    constraint : str or int
        The constraint expression, e.g. ``"heads * head_dim"`` or ``512``.
    concrete_dims : dict, optional
        Mapping of dimension names to known concrete integer values.
        Variables not in this mapping are treated as symbolic.

    Returns
    -------
    str
        ``"QF_LIA"`` if the constraint uses only linear arithmetic
        (no symbolic-symbolic multiplication), or ``"QF_NIA"`` if it
        contains a product of two or more symbolic variables.

    Examples
    --------
    >>> classify_constraint_fragment(512)
    'QF_LIA'
    >>> classify_constraint_fragment("8 * head_dim")
    'QF_LIA'
    >>> classify_constraint_fragment("heads * head_dim")
    'QF_NIA'
    >>> classify_constraint_fragment("heads * head_dim", {"heads": 8})
    'QF_LIA'
    """
    if isinstance(constraint, int):
        return "QF_LIA"

    classifier = RelationalConstraintClassifier(concrete_dims)
    # Use a dummy LHS — we only care about the RHS expression.
    info = classifier.classify("_dummy", str(constraint))
    if info.classification == RelationalConstraintClass.QF_NIA:
        return "QF_NIA"
    return "QF_LIA"


def is_nia_decidable(
    constraints: Dict[str, Union[str, int]],
    concrete_dims: Optional[Dict[str, int]] = None,
    bounded_dims: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Check whether QF_NIA constraints fall in the decidable subset.

    Z3 handles nonlinear integer arithmetic in several sub-fragments with
    different decidability characteristics:

    **Factor-pair enumeration (concrete c in a*b=c)**:
        When the product target ``c`` is a concrete integer, Z3 can
        enumerate all factor pairs ``(a, b)`` such that ``a * b = c``.
        This is always decidable (finite search).
        Example: ``batch * seq = 1024`` — Z3 tries (1,1024), (2,512), ...

    **Bounded symbolic (bit-blasting)**:
        When all symbolic variables have finite upper bounds, Z3 can
        encode them as fixed-width bit-vectors and reduce to SAT.
        Decidable but NP-hard (bit-blasting).
        Example: ``heads * head_dim = embed_dim`` with ``1 ≤ heads ≤ 128``

    **Fully symbolic (nlsat, incomplete)**:
        When variables are unbounded and all factors are symbolic, the
        problem enters the undecidable fragment of QF_NIA (Matiyasevich's
        theorem / Hilbert's 10th problem). Z3's nlsat tactic may return
        ``unknown``.
        Example: ``batch * seq = total`` with no bounds on any variable.

    Parameters
    ----------
    constraints : dict
        Mapping ``dim_name -> expr_or_int``.
    concrete_dims : dict, optional
        Known concrete dimension values.
    bounded_dims : dict, optional
        Known finite bounds ``{name: (lo, hi)}``.

    Returns
    -------
    dict
        A report with keys:
        - ``"all_decidable"`` (bool): True if all NIA constraints are
          in a decidable sub-fragment.
        - ``"fragment_counts"``: counts per sub-fragment.
        - ``"per_constraint"``: per-constraint classification.
        - ``"undecidable_constraints"``: list of constraints in the
          undecidable fragment (fully symbolic, unbounded).
    """
    concrete = dict(concrete_dims or {})
    bounds = dict(bounded_dims or {})

    per_constraint: List[Dict[str, Any]] = []
    counts = {
        "linear": 0,
        "concrete_factor_pair": 0,
        "bounded_symbolic": 0,
        "fully_symbolic_unbounded": 0,
    }
    undecidable: List[Dict[str, Any]] = []

    for dim_name, expr in constraints.items():
        expr_str = str(expr)

        # Step 1: Is it even NIA?
        fragment = classify_constraint_fragment(expr, concrete)

        if fragment == "QF_LIA":
            counts["linear"] += 1
            per_constraint.append({
                "constraint": f"{dim_name} = {expr_str}",
                "fragment": "QF_LIA",
                "sub_fragment": "linear",
                "decidable": True,
                "reason": "No symbolic-symbolic multiplication; pure QF_LIA.",
            })
            continue

        # Step 2: QF_NIA — classify the decidable sub-fragment.
        # Check if the LHS (target) is concrete → factor-pair enumeration.
        lhs_concrete = dim_name in concrete
        if lhs_concrete:
            counts["concrete_factor_pair"] += 1
            per_constraint.append({
                "constraint": f"{dim_name} = {expr_str}",
                "fragment": "QF_NIA",
                "sub_fragment": "concrete_factor_pair",
                "decidable": True,
                "reason": (
                    f"Product target '{dim_name}' has concrete value "
                    f"{concrete[dim_name]}; Z3 enumerates factor pairs."
                ),
            })
            continue

        # Check if all symbolic variables in the expression are bounded.
        if isinstance(expr, int):
            # Already handled above.
            continue

        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError:
            counts["linear"] += 1
            per_constraint.append({
                "constraint": f"{dim_name} = {expr_str}",
                "fragment": "QF_LIA",
                "sub_fragment": "unparseable",
                "decidable": True,
                "reason": "Unparseable expression — treated as linear.",
            })
            continue

        sym_vars = RelationalConstraintClassifier(concrete)._collect_symbolic_vars(tree.body)
        unbounded_vars = [v for v in sorted(sym_vars) if v not in bounds and v not in concrete]

        if not unbounded_vars:
            counts["bounded_symbolic"] += 1
            per_constraint.append({
                "constraint": f"{dim_name} = {expr_str}",
                "fragment": "QF_NIA",
                "sub_fragment": "bounded_symbolic",
                "decidable": True,
                "reason": (
                    "All symbolic factors have finite bounds; "
                    "decidable via bit-blasting."
                ),
            })
        else:
            counts["fully_symbolic_unbounded"] += 1
            entry = {
                "constraint": f"{dim_name} = {expr_str}",
                "fragment": "QF_NIA",
                "sub_fragment": "fully_symbolic_unbounded",
                "decidable": False,
                "unbounded_vars": unbounded_vars,
                "reason": (
                    f"Unbounded symbolic factors ({', '.join(unbounded_vars)}); "
                    f"enters undecidable QF_NIA (Matiyasevich 1970). "
                    f"Z3's nlsat may return unknown."
                ),
            }
            per_constraint.append(entry)
            undecidable.append(entry)

    all_decidable = counts["fully_symbolic_unbounded"] == 0

    return {
        "all_decidable": all_decidable,
        "fragment_counts": counts,
        "per_constraint": per_constraint,
        "undecidable_constraints": undecidable,
    }
