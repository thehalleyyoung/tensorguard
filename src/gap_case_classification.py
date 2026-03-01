"""Gap-case classification for Craig interpolation failures.

Replaces the imprecise "fundamentally undecidable" characterization of
verification gap cases with a precise three-way classification:

1. **SPECIFICATION_INCOMPLETE**: The verification query has unspecified
   dimensions — the user did not supply enough concrete shape information
   for the verifier to close the proof.  This is *not* an algorithmic
   limitation; providing the missing dimensions resolves the gap.

2. **SOLVER_INCOMPLETE**: The constraint system falls within a decidable
   SMT fragment (e.g., QF_NIA with bounded variables) but the solver
   (Z3 / CVC5) returns ``UNKNOWN`` due to internal heuristic limitations
   or timeout.  The problem is decidable in principle; a stronger solver
   or longer timeout could resolve it.

3. **TURING_UNDECIDABLE**: The constraint system involves unbounded
   symbolic–symbolic multiplication, which enters the full QF_NIA
   fragment.  Hilbert's 10th problem (Matiyasevich, 1970) establishes
   that satisfiability of Diophantine equations over the integers is
   undecidable.  This is the *only* case that deserves the label
   "undecidable", and it arises only when dimension variables are
   truly unbounded and appear in non-linear products.

References:
    - Y. Matiyasevich, "Enumerable sets are Diophantine", Doklady
      Akademii Nauk SSSR 191(2), 1970.
    - R. M. Robinson, "Undecidable rings", Trans. AMS 70(1), 1951.
    - C. Tinelli and C. Zarba, "Combining Nonstably Infinite Theories",
      JAR 34(3), 2005.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union


class GapCaseClass(Enum):
    """Precise classification of why a verification gap case arises.

    Replaces the imprecise "fundamentally undecidable" label with three
    distinct categories, each requiring a different mitigation strategy.
    """

    SPECIFICATION_INCOMPLETE = auto()
    """The query has unspecified/unknown dimensions.  Providing concrete
    values or tighter symbolic bounds resolves the gap.

    Example: ``embed_dim = heads * head_dim`` where none of the three
    variables have been given concrete values or upper bounds by the user.
    The verifier cannot determine satisfiability because the specification
    is incomplete, not because the problem is undecidable.

    Mitigation: supply the missing dimension specifications.
    """

    SOLVER_INCOMPLETE = auto()
    """The constraint system is in a decidable fragment (e.g., QF_NIA
    with bounded variables, which is decidable via bit-blasting) but
    the solver returns UNKNOWN due to heuristic limitations or timeout.

    Example: ``embed_dim = heads * head_dim`` with ``1 ≤ heads ≤ 128``
    and ``1 ≤ head_dim ≤ 512``.  This is a bounded QF_NIA system —
    decidable by enumeration or bit-blasting — but Z3's NIA tactic
    may time out on complex instances.

    Mitigation: increase solver timeout, use bit-blasting tactic, or
    switch to CVC5's NIA solver.
    """

    TURING_UNDECIDABLE = auto()
    """The constraint system involves unbounded symbolic–symbolic
    multiplication, entering the full QF_NIA fragment.  By Matiyasevich's
    theorem (negative resolution of Hilbert's 10th problem), satisfiability
    of systems of Diophantine equations over unbounded integers is
    Turing-undecidable.

    Example: ``embed_dim = heads * head_dim`` with both ``heads`` and
    ``head_dim`` ranging over all positive integers (no finite upper
    bound).  No algorithm can decide satisfiability in general.

    Mitigation: introduce finite bounds on symbolic dimensions to reduce
    to the SOLVER_INCOMPLETE case, or verify the constraint manually.
    """


@dataclass(frozen=True)
class GapCaseReport:
    """Detailed report for a single gap-case classification.

    Attributes
    ----------
    classification : GapCaseClass
        The precise gap-case category.
    description : str
        Human-readable description of the gap case.
    constraint_expr : str
        The constraint expression that caused the gap.
    unspecified_dims : list of str
        Dimension names that lack concrete values or bounds.
    bounded_dims : dict
        Dimension names that have finite bounds: ``{name: (lo, hi)}``.
    unbounded_symbolic_products : list of tuple
        Pairs of symbolic variables appearing in unbounded products.
    mitigation : str
        Recommended mitigation strategy.
    references : list of str
        Academic references supporting the classification.
    """

    classification: GapCaseClass
    description: str
    constraint_expr: str = ""
    unspecified_dims: List[str] = field(default_factory=list)
    bounded_dims: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    unbounded_symbolic_products: List[Tuple[str, str]] = field(default_factory=list)
    mitigation: str = ""
    references: List[str] = field(default_factory=list)


def classify_gap_case(
    constraint_expr: Union[str, int],
    concrete_dims: Optional[Dict[str, int]] = None,
    bounded_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    solver_returned_unknown: bool = False,
) -> GapCaseReport:
    """Classify a verification gap case with precise undecidability analysis.

    This replaces the imprecise "fundamentally undecidable" characterization.
    Given a constraint expression and information about which dimensions are
    concrete, bounded, or unbounded, returns a precise classification.

    Parameters
    ----------
    constraint_expr : str or int
        The constraint expression (e.g., ``"heads * head_dim"``).
        An integer literal is trivially satisfiable — no gap case.
    concrete_dims : dict, optional
        Mapping of dimension names to concrete integer values.
    bounded_dims : dict, optional
        Mapping of dimension names to ``(lo, hi)`` finite bounds.
    solver_returned_unknown : bool
        Whether the SMT solver returned UNKNOWN on this constraint.

    Returns
    -------
    GapCaseReport
        A detailed classification report.
    """
    concrete = dict(concrete_dims or {})
    bounds = dict(bounded_dims or {})
    expr_str = str(constraint_expr)

    # Trivial case: integer literal
    if isinstance(constraint_expr, int):
        return GapCaseReport(
            classification=GapCaseClass.SPECIFICATION_INCOMPLETE,
            description="Concrete integer — no gap case.",
            constraint_expr=expr_str,
            mitigation="None needed.",
        )

    # Parse the expression to find symbolic variables and products
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError:
        return GapCaseReport(
            classification=GapCaseClass.SPECIFICATION_INCOMPLETE,
            description=f"Unparseable expression: {expr_str}",
            constraint_expr=expr_str,
            mitigation="Fix the expression syntax.",
        )

    all_vars = _collect_symbolic_vars(tree.body, concrete)
    nonlinear_pairs = _collect_nonlinear_pairs(tree.body, concrete)

    # Determine which variables lack both concrete values and bounds
    unspecified = [v for v in all_vars if v not in concrete and v not in bounds]

    # Case 1: Unspecified dimensions → SPECIFICATION_INCOMPLETE
    if unspecified and not nonlinear_pairs:
        return GapCaseReport(
            classification=GapCaseClass.SPECIFICATION_INCOMPLETE,
            description=(
                f"Dimensions {unspecified} are unspecified (no concrete value "
                f"or finite bound). The verification query is incomplete."
            ),
            constraint_expr=expr_str,
            unspecified_dims=sorted(unspecified),
            mitigation=(
                "Supply concrete values or finite bounds for: "
                + ", ".join(sorted(unspecified))
            ),
        )

    # Check for nonlinear symbolic-symbolic products
    if nonlinear_pairs:
        # Determine if all factors are bounded
        unbounded_pairs = []
        bounded_pairs = []
        for v1, v2 in nonlinear_pairs:
            v1_bounded = v1 in concrete or v1 in bounds
            v2_bounded = v2 in concrete or v2 in bounds
            if v1_bounded and v2_bounded:
                bounded_pairs.append((v1, v2))
            else:
                unbounded_pairs.append((v1, v2))

        if not unbounded_pairs:
            # All nonlinear products have bounded factors
            if solver_returned_unknown:
                # Case 2: Decidable but solver returned UNKNOWN
                return GapCaseReport(
                    classification=GapCaseClass.SOLVER_INCOMPLETE,
                    description=(
                        "All symbolic variables in nonlinear products are "
                        "finitely bounded. The QF_NIA constraint system is "
                        "decidable via bit-blasting, but the solver returned "
                        "UNKNOWN (likely due to timeout or heuristic limits)."
                    ),
                    constraint_expr=expr_str,
                    bounded_dims={
                        k: v for k, v in bounds.items()
                        if k in all_vars
                    },
                    mitigation=(
                        "Increase solver timeout, enable bit-blasting tactic "
                        "(z3.Then('simplify', 'bit-blast', 'sat')), or try "
                        "CVC5's nonlinear arithmetic solver."
                    ),
                    references=[
                        "Z3 Tactics: https://microsoft.github.io/z3guide/docs/strategies/tactics",
                    ],
                )
            else:
                # Bounded and solver didn't return UNKNOWN — specification issue
                return GapCaseReport(
                    classification=GapCaseClass.SPECIFICATION_INCOMPLETE,
                    description=(
                        "Nonlinear products with bounded factors; additional "
                        "dimension specifications may resolve the gap."
                    ),
                    constraint_expr=expr_str,
                    unspecified_dims=sorted(unspecified),
                    bounded_dims={
                        k: v for k, v in bounds.items()
                        if k in all_vars
                    },
                    mitigation=(
                        "Provide concrete values for symbolic dimensions, or "
                        "verify that bounds are tight enough."
                    ),
                )
        else:
            # Case 3: Unbounded symbolic-symbolic multiplication
            return GapCaseReport(
                classification=GapCaseClass.TURING_UNDECIDABLE,
                description=(
                    "Unbounded symbolic–symbolic multiplication enters the "
                    "full QF_NIA fragment. By Matiyasevich's theorem "
                    "(negative resolution of Hilbert's 10th problem, 1970), "
                    "satisfiability of Diophantine equations over unbounded "
                    "integers is Turing-undecidable."
                ),
                constraint_expr=expr_str,
                unspecified_dims=sorted(unspecified),
                unbounded_symbolic_products=sorted(unbounded_pairs),
                mitigation=(
                    "Introduce finite upper bounds on the symbolic dimensions "
                    f"({', '.join(sorted(set(v for p in unbounded_pairs for v in p)))}) "
                    "to reduce to the decidable bounded-NIA fragment."
                ),
                references=[
                    "Y. Matiyasevich, 'Enumerable sets are Diophantine', "
                    "Doklady Akademii Nauk SSSR 191(2), 1970.",
                    "R. M. Robinson, 'Undecidable rings', "
                    "Trans. AMS 70(1), 1951.",
                ],
            )

    # No nonlinear products, all variables specified → no gap
    if solver_returned_unknown:
        return GapCaseReport(
            classification=GapCaseClass.SOLVER_INCOMPLETE,
            description=(
                "Linear constraint system (QF_LIA), decidable, but solver "
                "returned UNKNOWN — likely a timeout."
            ),
            constraint_expr=expr_str,
            mitigation="Increase solver timeout.",
        )

    return GapCaseReport(
        classification=GapCaseClass.SPECIFICATION_INCOMPLETE,
        description="No gap case identified — constraint appears tractable.",
        constraint_expr=expr_str,
        mitigation="None needed.",
    )


# ── Craig interpolation gap-case instances ────────────────────────────────────

# The three canonical gap cases from the Craig interpolation module,
# now with precise classifications.

CRAIG_INTERPOLATION_GAP_CASES: List[Dict[str, object]] = [
    {
        "name": "unspecified_dimensions",
        "description": (
            "Interpolation fails because the verification query contains "
            "dimension variables with no concrete value or symbolic bound. "
            "The interpolant cannot be computed because the constraint "
            "system is under-determined."
        ),
        "classification": GapCaseClass.SPECIFICATION_INCOMPLETE,
        "example_constraint": "embed_dim = heads * head_dim",
        "example_concrete_dims": {},
        "example_bounded_dims": {},
        "mitigation": (
            "Supply concrete values or type-level bounds for unspecified "
            "dimensions (e.g., heads=8, head_dim=64)."
        ),
    },
    {
        "name": "solver_unknown_on_decidable_qf_nia",
        "description": (
            "Interpolation fails because the underlying SMT solver "
            "(Z3/CVC5) returns UNKNOWN on a QF_NIA constraint system "
            "that is in fact decidable (all symbolic variables are "
            "finitely bounded). The solver's incomplete NIA heuristics "
            "cannot close the proof within the timeout."
        ),
        "classification": GapCaseClass.SOLVER_INCOMPLETE,
        "example_constraint": "total_elements = batch * channels * height * width",
        "example_concrete_dims": {},
        "example_bounded_dims": {
            "batch": (1, 256),
            "channels": (1, 2048),
            "height": (1, 1024),
            "width": (1, 1024),
        },
        "mitigation": (
            "Increase solver timeout, enable bit-blasting for bounded "
            "integer variables, or switch to CVC5's dedicated NIA solver."
        ),
    },
    {
        "name": "unbounded_symbolic_multiplication",
        "description": (
            "Interpolation fails on a constraint involving the product "
            "of two or more unbounded symbolic variables. This enters "
            "the undecidable fragment of QF_NIA (Matiyasevich, 1970). "
            "No algorithm can decide satisfiability in general."
        ),
        "classification": GapCaseClass.TURING_UNDECIDABLE,
        "example_constraint": "output_dim = factor_a * factor_b",
        "example_concrete_dims": {},
        "example_bounded_dims": {},
        "mitigation": (
            "Introduce finite bounds on symbolic dimensions to reduce "
            "to the decidable bounded-NIA fragment."
        ),
        "reference": (
            "Y. Matiyasevich, 'Enumerable sets are Diophantine', "
            "Doklady Akademii Nauk SSSR 191(2), 1970."
        ),
    },
]


# ── AST helpers ──────────────────────────────────────────────────────────────

def _collect_symbolic_vars(
    node: ast.AST, concrete: Dict[str, int]
) -> Set[str]:
    """Return symbolic (non-concrete) variable names in *node*."""
    if isinstance(node, ast.Name):
        return {node.id} if node.id not in concrete else set()
    if isinstance(node, ast.Constant):
        return set()
    if isinstance(node, ast.BinOp):
        return (
            _collect_symbolic_vars(node.left, concrete)
            | _collect_symbolic_vars(node.right, concrete)
        )
    if isinstance(node, ast.UnaryOp):
        return _collect_symbolic_vars(node.operand, concrete)
    if isinstance(node, ast.Expression):
        return _collect_symbolic_vars(node.body, concrete)
    return set()


def _collect_nonlinear_pairs(
    node: ast.AST, concrete: Dict[str, int]
) -> List[Tuple[str, str]]:
    """Return pairs of symbolic variables in multiplicative terms."""
    pairs: List[Tuple[str, str]] = []

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_sym = _collect_symbolic_vars(node.left, concrete)
        right_sym = _collect_symbolic_vars(node.right, concrete)
        if left_sym and right_sym:
            for v1 in sorted(left_sym):
                for v2 in sorted(right_sym):
                    pairs.append((v1, v2))

    # Recurse
    if isinstance(node, ast.BinOp):
        pairs.extend(_collect_nonlinear_pairs(node.left, concrete))
        pairs.extend(_collect_nonlinear_pairs(node.right, concrete))
    if isinstance(node, ast.UnaryOp):
        pairs.extend(_collect_nonlinear_pairs(node.operand, concrete))
    if isinstance(node, ast.Expression):
        pairs.extend(_collect_nonlinear_pairs(node.body, concrete))

    return pairs
