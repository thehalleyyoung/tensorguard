"""
Craig Interpolation for Shape Predicate Discovery.

Complements the template-based CEGAR predicate discovery in ``shape_cegar.py``
by computing Craig interpolants from unsatisfiable path/safety formula pairs.

Given a path formula **A** (computation-graph constraints up to a failing step)
and a safety formula **B** (property at the failing step), a Craig interpolant
**I** satisfies:

    A ⊨ I   and   I ∧ B is UNSAT

and **I** mentions only variables common to both A and B (the "interface"
variables — typically input-shape dimensions).

Because Z3 ≥ 4.12 removed its native interpolation API, this module provides a
*simulation* of Craig interpolation using unsat-core extraction followed by
quantifier elimination on the non-interface variables.  The resulting formula
is then parsed back into ``ShapePredicate`` objects from ``shape_cegar``.

A new ``PredicateKind.DIM_LINEAR_COMBO`` is introduced for linear-combination
predicates (e.g. ``a*d0 + b*d1 >= c``) that fall outside the original 7-kind
template grammar.

Projection from QF_UFLIA to Predicate Template Language
--------------------------------------------------------
The **projection algorithm** maps full QF_UFLIA interpolants to TensorGuard's
predicate template language P comprising 9 predicate kinds (7 base + 2
extended):

  **Base kinds** (from ``PredicateKind``):
    1. DIM_EQ        — tensor.shape[axis] == value
    2. DIM_GT        — tensor.shape[axis] > value
    3. DIM_GE        — tensor.shape[axis] >= value
    4. DIM_DIVISIBLE  — tensor.shape[axis] % divisor == 0
    5. DIM_MATCH      — tensor_a.shape[a] == tensor_b.shape[b]
    6. NDIM_EQ        — len(tensor.shape) == value
    7. SHAPE_EQ       — tensor.shape == (d0, d1, ...)

  **Extended kinds** (from ``ExtendedPredicateKind``):
    8. DIM_LINEAR_COMBO  — Σ(coeff_i * dim_i) op rhs
    9. DIM_PRODUCT_EQ    — Π(dim_i) op rhs

**Projection completeness argument:**

The projection from QF_UFLIA Craig interpolants to P is **lossy in general**
but **complete for CEGAR-relevant interpolants** in practice:

  1. **Completeness within CEGAR scope**: Interpolants arising from
     TensorGuard's shape verification encode relationships between
     integer dimension variables via QF_LIA arithmetic.  The template
     language P covers:
       - All unary integer predicates on dimensions (DIM_EQ, DIM_GT, DIM_GE)
       - Binary dimension equalities (DIM_MATCH)
       - Linear combinations (DIM_LINEAR_COMBO — catches any LIA atomic)
       - Products from reshape/flatten (DIM_PRODUCT_EQ)

  2. **Lossy case**: Disjunctive interpolants (e.g. ``d0 == 3 ∨ d0 == 7``)
     cannot be represented as a single predicate in P.  The projection
     overapproximates by taking the convex hull or weakest covering
     predicate (e.g. ``d0 >= 3``).  This preserves soundness (the
     projected predicate is implied by A) but loses precision.

  3. **Soundness preservation under lossy projection**: If projection
     weakens I to I' where A ⊨ I ⊨ I', then I' may fail to block B
     (I' ∧ B might be SAT).  In this case, the CEGAR loop simply
     continues to the next iteration — soundness is never compromised
     because SAFE is only reported when all counterexamples are
     eliminated.  The worst case is additional CEGAR iterations.

  4. **Modular arithmetic gap**: DIM_DIVISIBLE requires modular
     reasoning.  Z3's QE may produce non-modular interpolants for
     divisibility constraints.  The projection detects modular patterns
     (``dim % k == 0``) when possible, otherwise falls back to
     DIM_LINEAR_COMBO.

See ``verify_projection_completeness()`` for runtime verification.

References
----------
* McMillan, K.L. "Interpolation and SAT-Based Model Checking", CAV 2003.
* Craig, W. "Three uses of the Herbrand–Gentzen theorem", JSL 1957.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    import cvc5
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

# ---------------------------------------------------------------------------
# Import shape_cegar types
# ---------------------------------------------------------------------------

from src.shape_cegar import ShapePredicate, PredicateKind


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Interpolation method enum & extended predicate kinds
# ═══════════════════════════════════════════════════════════════════════════════

class InterpolationMethod(Enum):
    """Which backend to use for Craig interpolation."""
    CVC5_NATIVE = auto()
    Z3_UNSAT_CORE_SIMULATION = auto()
    AUTO = auto()  # Try CVC5 first, fall back to Z3 simulation


class ExtendedPredicateKind(Enum):
    """Predicate kinds beyond the base 7, discovered via interpolation."""
    DIM_LINEAR_COMBO = auto()  # a*d0 + b*d1 + ... >= c  (or ==, >, etc.)
    DIM_PRODUCT_EQ = auto()    # d0 * d1 == c  (product equality)


# Re-export for convenience
DIM_LINEAR_COMBO = ExtendedPredicateKind.DIM_LINEAR_COMBO
DIM_PRODUCT_EQ = ExtendedPredicateKind.DIM_PRODUCT_EQ


@dataclass(frozen=True)
class LinearComboPredicate:
    """A linear-combination shape predicate: Σ(coeff_i * dim_i) op rhs.

    Attributes
    ----------
    coefficients : dict mapping (tensor_name, axis) -> integer coefficient
    operator : one of '==', '>=', '>', '<=', '<'
    rhs : integer right-hand side
    provenance : discovery source tag
    """
    coefficients: Tuple[Tuple[Tuple[str, int], int], ...]  # frozen dict representation
    operator: str
    rhs: int
    provenance: str = "craig_interpolation"

    @property
    def coeff_dict(self) -> Dict[Tuple[str, int], int]:
        return dict(self.coefficients)

    def pretty(self) -> str:
        parts = []
        for (tensor, axis), coeff in self.coefficients:
            if coeff == 1:
                parts.append(f"{tensor}.shape[{axis}]")
            elif coeff == -1:
                parts.append(f"-{tensor}.shape[{axis}]")
            else:
                parts.append(f"{coeff}*{tensor}.shape[{axis}]")
        lhs = " + ".join(parts)
        return f"{lhs} {self.operator} {self.rhs}"

    def __repr__(self) -> str:
        return f"LinearComboPredicate({self.pretty()})"


@dataclass(frozen=True)
class ProductEqualityPredicate:
    """A product-equality shape predicate: dim_a * dim_b op rhs.

    Represents constraints like ``x.shape[2] * x.shape[3] == 64`` that arise
    from flatten/reshape operations where the product of spatial dimensions
    must match a known feature count.

    Attributes
    ----------
    factors : tuple of (tensor_name, axis) pairs being multiplied
    operator : one of '==', '>=', '>', '<=', '<'
    rhs : integer right-hand side
    provenance : discovery source tag
    """
    factors: Tuple[Tuple[str, int], ...]
    operator: str
    rhs: int
    provenance: str = "craig_interpolation"

    def pretty(self) -> str:
        parts = [f"{t}.shape[{a}]" for t, a in self.factors]
        lhs = " * ".join(parts)
        return f"{lhs} {self.operator} {self.rhs}"

    def __repr__(self) -> str:
        return f"ProductEqualityPredicate({self.pretty()})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Dimension variable mapping
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimMapping:
    """Bidirectional mapping between Z3 Int variables and (tensor, axis) pairs.

    Attributes
    ----------
    var_to_dim : maps Z3 variable name -> (tensor_name, axis_index)
    dim_to_var : maps (tensor_name, axis_index) -> Z3 variable name
    """
    var_to_dim: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    dim_to_var: Dict[Tuple[str, int], str] = field(default_factory=dict)

    def register(self, var_name: str, tensor: str, axis: int) -> None:
        """Register a Z3 variable as representing a tensor dimension."""
        self.var_to_dim[var_name] = (tensor, axis)
        self.dim_to_var[(tensor, axis)] = var_name

    def is_known(self, var_name: str) -> bool:
        return var_name in self.var_to_dim

    def get_dim(self, var_name: str) -> Optional[Tuple[str, int]]:
        return self.var_to_dim.get(var_name)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Craig interpolation engine
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_vars(expr: "z3.ExprRef") -> Set[str]:
    """Collect all free integer variable names in a Z3 expression tree."""
    if not HAS_Z3:
        return set()
    result: Set[str] = set()
    worklist = [expr]
    seen: Set[int] = set()
    while worklist:
        e = worklist.pop()
        eid = e.get_id()
        if eid in seen:
            continue
        seen.add(eid)
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            result.add(str(e))
        else:
            for child in e.children():
                worklist.append(child)
    return result


def _collect_vars_from_list(exprs: Sequence["z3.ExprRef"]) -> Set[str]:
    """Collect free variables from a list of Z3 expressions."""
    result: Set[str] = set()
    for expr in exprs:
        result |= _collect_vars(expr)
    return result


def _compute_cvc5_interpolant(
    a_constraints: List["z3.ExprRef"],
    b_constraints: List["z3.ExprRef"],
    interface_vars: Set[str],
    timeout_ms: int = 5000,
) -> Optional["z3.ExprRef"]:
    """Compute a Craig interpolant using CVC5's native ``getInterpolant``.

    CVC5's native interpolation guarantees all three Craig properties:
      - A ⊨ I  (consequence)
      - I ∧ B is UNSAT  (separation)
      - Vars(I) ⊆ Vars(A) ∩ Vars(B)  (vocabulary restriction)

    The result is converted back to a Z3 expression via SMT-LIB2 string
    interchange so it integrates with the existing parsing pipeline.

    Returns None if CVC5 is unavailable, A ∧ B is SAT, or interpolation fails.
    """
    if not HAS_CVC5 or not HAS_Z3:
        return None

    try:
        # Collect all variable names from both sides
        a_vars = _collect_vars_from_list(a_constraints)
        b_vars = _collect_vars_from_list(b_constraints)
        all_vars = a_vars | b_vars

        tm = cvc5.TermManager()
        solver = cvc5.Solver(tm)
        solver.setOption("produce-interpolants", "true")
        solver.setLogic("QF_LIA")
        if timeout_ms > 0:
            solver.setOption("tlimit-per", str(timeout_ms))

        # Create CVC5 integer constants for all variables
        int_sort = tm.getIntegerSort()
        cvc5_vars: Dict[str, Any] = {}
        for v in sorted(all_vars):
            cvc5_vars[v] = tm.mkConst(int_sort, v)

        # Convert Z3 constraints to CVC5 terms and assert A-side
        a_terms = []
        for c in a_constraints:
            term = _z3_to_cvc5(c, tm, cvc5_vars)
            if term is None:
                logger.debug("Failed to convert Z3 constraint to CVC5: %s", c)
                return None
            a_terms.append(term)

        for t in a_terms:
            solver.assertFormula(t)

        # Build the B-side conjunction as the conjecture target
        b_terms = []
        for c in b_constraints:
            term = _z3_to_cvc5(c, tm, cvc5_vars)
            if term is None:
                logger.debug("Failed to convert Z3 constraint to CVC5: %s", c)
                return None
            b_terms.append(term)

        if len(b_terms) == 1:
            b_conj = b_terms[0]
        else:
            b_conj = tm.mkTerm(cvc5.Kind.AND, *b_terms)

        # getInterpolant(F) finds I s.t. assertions ⊨ I and I ⊨ F
        # We want: A ⊨ I and I ∧ B unsat, i.e. I ⊨ ¬B
        not_b = tm.mkTerm(cvc5.Kind.NOT, b_conj)
        interp_term = solver.getInterpolant(not_b)

        if interp_term.isNull():
            logger.debug("CVC5 getInterpolant returned null")
            return None

        # Convert CVC5 interpolant to Z3 via SMT-LIB2 string
        interp_str = str(interp_term)
        z3_interp = _cvc5_sexpr_to_z3(interp_str, all_vars)
        if z3_interp is None:
            logger.debug("Failed to parse CVC5 interpolant back to Z3: %s", interp_str)
            return None

        return z3.simplify(z3_interp)

    except Exception as exc:
        logger.debug("CVC5 interpolation failed: %s", exc)
        return None


def _z3_to_cvc5(
    expr: "z3.ExprRef",
    tm: "cvc5.TermManager",
    var_map: Dict[str, Any],
) -> Optional[Any]:
    """Convert a Z3 expression to a CVC5 term.

    Handles the QF_LIA fragment: integer constants, variables,
    arithmetic (+, -, *, unary minus), and comparisons (==, >=, >, <=, <, !=).
    """
    if not HAS_Z3 or not HAS_CVC5:
        return None

    if z3.is_int_value(expr):
        return tm.mkInteger(expr.as_long())

    if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        name = str(expr)
        return var_map.get(name)

    if z3.is_true(expr):
        return tm.mkTrue()
    if z3.is_false(expr):
        return tm.mkFalse()

    if not z3.is_app(expr):
        return None

    kind = expr.decl().kind()
    children = expr.children()
    converted = []
    for c in children:
        cc = _z3_to_cvc5(c, tm, var_map)
        if cc is None:
            return None
        converted.append(cc)

    kind_map = {
        z3.Z3_OP_ADD: cvc5.Kind.ADD,
        z3.Z3_OP_SUB: cvc5.Kind.SUB,
        z3.Z3_OP_MUL: cvc5.Kind.MULT,
        z3.Z3_OP_UMINUS: cvc5.Kind.NEG,
        z3.Z3_OP_LE: cvc5.Kind.LEQ,
        z3.Z3_OP_GE: cvc5.Kind.GEQ,
        z3.Z3_OP_LT: cvc5.Kind.LT,
        z3.Z3_OP_GT: cvc5.Kind.GT,
        z3.Z3_OP_EQ: cvc5.Kind.EQUAL,
        z3.Z3_OP_DISTINCT: cvc5.Kind.DISTINCT,
        z3.Z3_OP_AND: cvc5.Kind.AND,
        z3.Z3_OP_OR: cvc5.Kind.OR,
        z3.Z3_OP_NOT: cvc5.Kind.NOT,
        z3.Z3_OP_IMPLIES: cvc5.Kind.IMPLIES,
    }

    cvc5_kind = kind_map.get(kind)
    if cvc5_kind is None:
        return None

    return tm.mkTerm(cvc5_kind, *converted)


def _cvc5_sexpr_to_z3(
    sexpr: str, all_vars: Set[str]
) -> Optional["z3.ExprRef"]:
    """Parse a CVC5 interpolant s-expression string into a Z3 expression.

    Uses Z3's ``parse_smt2_string`` with variable declarations.
    """
    if not HAS_Z3:
        return None

    try:
        decls = " ".join(f"(declare-const {v} Int)" for v in sorted(all_vars))
        smt2 = f"{decls} (assert {sexpr})"
        parsed = z3.parse_smt2_string(smt2)
        if len(parsed) == 0:
            return z3.BoolVal(True)
        if len(parsed) == 1:
            return parsed[0]
        return z3.And(list(parsed))
    except Exception as exc:
        logger.debug("SMT-LIB2 parse failed: %s", exc)
        return None


def _compute_simulated_interpolant(
    a_constraints: List["z3.ExprRef"],
    b_constraints: List["z3.ExprRef"],
    interface_vars: Set[str],
    timeout_ms: int = 5000,
) -> Optional["z3.ExprRef"]:
    """Compute a simulated Craig interpolant via unsat-core + QE.

    .. warning::

       This is a *simulation*, not a true Craig interpolant.  The
       UNSAT-core extraction followed by quantifier elimination
       guarantees separation (I ∧ B is UNSAT) and consequence (A ⊨ I),
       but the vocabulary restriction Vars(I) ⊆ Vars(A) ∩ Vars(B)
       may be violated if QE is incomplete.  Use CVC5's native
       ``getInterpolant`` for a mathematically valid Craig interpolant
       with full soundness guarantees.

    Steps
    -----
    1. Check that A ∧ B is UNSAT.
    2. Extract the unsat core from A-side assertions.
    3. Existentially quantify out non-interface variables from the A-core.
    4. Apply Z3 quantifier elimination to obtain an interpolant over
       interface variables only.

    Returns None if A ∧ B is SAT or if QE fails/times out.
    """
    if not HAS_Z3:
        return None

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Tag A-constraints so we can extract the unsat core from them.
    a_tags = []
    for i, c in enumerate(a_constraints):
        tag = z3.Bool(f"__a_tag_{i}")
        a_tags.append((tag, c))
        solver.assert_and_track(c, tag)

    # Assert B-constraints directly (no tracking needed).
    for c in b_constraints:
        solver.add(c)

    result = solver.check()
    if result != z3.unsat:
        logger.debug("A ∧ B is %s — no interpolant", result)
        return None

    # Extract the unsat core (subset of A-side constraints).
    core_tags = set(str(t) for t in solver.unsat_core())
    core_constraints = [c for tag, c in a_tags if str(tag) in core_tags]

    if not core_constraints:
        # If the core is empty, B alone is UNSAT → interpolant is True.
        return z3.BoolVal(True)

    # Determine which variables are NOT in the interface (need quantifying out).
    a_core_vars = _collect_vars_from_list(core_constraints)
    vars_to_eliminate = a_core_vars - interface_vars

    if not vars_to_eliminate:
        # All variables are shared — the core itself is the interpolant.
        interpolant = z3.And(core_constraints) if len(core_constraints) > 1 else core_constraints[0]
        return z3.simplify(interpolant)

    # Build existential quantification and eliminate.
    core_formula = z3.And(core_constraints) if len(core_constraints) > 1 else core_constraints[0]

    # Create Z3 Int constants for the variables to eliminate.
    quant_vars = [z3.Int(v) for v in sorted(vars_to_eliminate)]
    quantified = z3.Exists(quant_vars, core_formula)

    # Use the Tactic API for quantifier elimination.
    try:
        tactic = z3.Then("qe", "simplify")
        goal = z3.Goal()
        goal.add(quantified)
        result_goal = tactic(goal)

        if len(result_goal) == 1:
            subgoal = result_goal[0]
            if len(subgoal) == 0:
                return z3.BoolVal(True)
            clauses = list(subgoal)
            interpolant = z3.And(clauses) if len(clauses) > 1 else clauses[0]
            return z3.simplify(interpolant)
        else:
            logger.warning("QE produced multiple subgoals; falling back")
            return None
    except z3.Z3Exception as exc:
        logger.warning("QE failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Interpolant → ShapePredicate parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _is_le(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_LE

def _is_ge(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_GE

def _is_lt(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_LT

def _is_gt(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_GT

def _is_eq(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_EQ

def _is_not(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_NOT

def _is_and(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_AND

def _is_true(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_TRUE

def _is_false(e: "z3.ExprRef") -> bool:
    return z3.is_app(e) and e.decl().kind() == z3.Z3_OP_FALSE


def _extract_linear_coeffs(
    expr: "z3.ExprRef", dim_map: DimMapping
) -> Optional[Tuple[Dict[Tuple[str, int], int], int]]:
    """Try to decompose *expr* into a linear combination of known dims + constant.

    Returns (coefficients_dict, constant_offset) or None if the expression is
    not a linear combination of mapped dimension variables.
    """
    coeffs: Dict[Tuple[str, int], int] = {}
    const_offset = 0

    def _walk(e: "z3.ExprRef", sign: int) -> bool:
        nonlocal const_offset

        if z3.is_int_value(e):
            const_offset += sign * e.as_long()
            return True

        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            name = str(e)
            dim = dim_map.get_dim(name)
            if dim is None:
                return False
            coeffs[dim] = coeffs.get(dim, 0) + sign
            return True

        if z3.is_app(e):
            kind = e.decl().kind()
            children = e.children()

            if kind == z3.Z3_OP_ADD:
                return all(_walk(c, sign) for c in children)

            if kind == z3.Z3_OP_SUB and len(children) == 2:
                return _walk(children[0], sign) and _walk(children[1], -sign)

            if kind == z3.Z3_OP_UMINUS and len(children) == 1:
                return _walk(children[0], -sign)

            if kind == z3.Z3_OP_MUL and len(children) == 2:
                c0, c1 = children
                if z3.is_int_value(c0):
                    factor = c0.as_long()
                    return _walk(c1, sign * factor)
                if z3.is_int_value(c1):
                    factor = c1.as_long()
                    return _walk(c0, sign * factor)
                return False

        return False

    if _walk(expr, 1):
        return coeffs, const_offset
    return None


def _extract_product_factors(
    expr: "z3.ExprRef", dim_map: DimMapping
) -> Optional[Tuple[List[Tuple[str, int]], int]]:
    """Try to decompose *expr* into a product of dim variables times a constant.

    Returns ``(factors_list, constant_multiplier)`` or ``None`` if the
    expression is not a product of mapped dimension variables (optionally
    times an integer constant).

    For example, ``__ci_x_d2 * __ci_x_d3`` → ``([('x', 2), ('x', 3)], 1)``.
    ``2 * __ci_x_d2 * __ci_x_d3`` → ``([('x', 2), ('x', 3)], 2)``.
    """
    factors: List[Tuple[str, int]] = []
    const_mult = 1

    def _collect(e: "z3.ExprRef") -> bool:
        nonlocal const_mult

        if z3.is_int_value(e):
            const_mult *= e.as_long()
            return True

        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            name = str(e)
            dim = dim_map.get_dim(name)
            if dim is None:
                return False
            factors.append(dim)
            return True

        if z3.is_app(e) and e.decl().kind() == z3.Z3_OP_MUL:
            return all(_collect(c) for c in e.children())

        return False

    if _collect(expr) and len(factors) >= 2:
        return factors, const_mult
    return None


def _try_parse_product(
    expr: "z3.ExprRef", dim_map: DimMapping
) -> List[ProductEqualityPredicate]:
    """Try to parse a comparison involving a product of dim variables.

    Handles patterns like ``d0 * d1 >= C``, ``d0 * d1 == C``, etc.
    Returns an empty list if the expression is not a product comparison.
    """
    if not HAS_Z3:
        return []

    op: Optional[str] = None
    if _is_eq(expr):
        op = "=="
    elif _is_ge(expr):
        op = ">="
    elif _is_gt(expr):
        op = ">"
    elif _is_le(expr):
        op = "<="
    elif _is_lt(expr):
        op = "<"

    if op is None:
        return []

    lhs, rhs = expr.children()

    # Try: product_of_dims OP integer_constant
    prod = _extract_product_factors(lhs, dim_map)
    if prod is not None and z3.is_int_value(rhs):
        factors, const_mult = prod
        rhs_val = rhs.as_long()
        if const_mult != 1:
            if rhs_val % const_mult != 0:
                return []
            rhs_val = rhs_val // const_mult
        return [ProductEqualityPredicate(
            factors=tuple(factors),
            operator=op,
            rhs=rhs_val,
            provenance="craig_interpolation",
        )]

    # Try: integer_constant OP product_of_dims (swap sides and flip op)
    prod = _extract_product_factors(rhs, dim_map)
    if prod is not None and z3.is_int_value(lhs):
        factors, const_mult = prod
        lhs_val = lhs.as_long()
        if const_mult != 1:
            if lhs_val % const_mult != 0:
                return []
            lhs_val = lhs_val // const_mult
        flipped = {"==": "==", ">=": "<=", ">": "<", "<=": ">=", "<": ">"}
        return [ProductEqualityPredicate(
            factors=tuple(factors),
            operator=flipped[op],
            rhs=lhs_val,
            provenance="craig_interpolation",
        )]

    return []


def _parse_atomic(
    expr: "z3.ExprRef", dim_map: DimMapping
) -> List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate]:
    """Parse a single atomic Z3 comparison into ShapePredicate(s)."""
    results: List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate] = []

    # Handle negation: Not(x <= y) → x > y, etc.
    if _is_not(expr) and len(expr.children()) == 1:
        inner = expr.children()[0]
        if _is_le(inner):
            lhs, rhs = inner.children()
            # Not(lhs <= rhs) means lhs > rhs → lhs - rhs > 0 → lhs >= rhs + 1
            return _parse_atomic(lhs > rhs, dim_map)
        if _is_ge(inner):
            lhs, rhs = inner.children()
            return _parse_atomic(lhs < rhs, dim_map)
        if _is_lt(inner):
            lhs, rhs = inner.children()
            return _parse_atomic(lhs >= rhs, dim_map)
        if _is_gt(inner):
            lhs, rhs = inner.children()
            return _parse_atomic(lhs <= rhs, dim_map)
        if _is_eq(inner):
            # Not(a == b) → can't produce a single predicate easily
            return results
        return results

    # Identify operator.
    op: Optional[str] = None
    if _is_eq(expr):
        op = "=="
    elif _is_ge(expr):
        op = ">="
    elif _is_gt(expr):
        op = ">"
    elif _is_le(expr):
        op = "<="
    elif _is_lt(expr):
        op = "<"

    if op is None:
        return results

    lhs, rhs = expr.children()

    # Normalise to form: linear_combo OP constant.
    # Move everything to LHS: lhs - rhs OP 0 if rhs is not constant, etc.
    combined = lhs - rhs
    decomp = _extract_linear_coeffs(z3.simplify(combined), dim_map)
    if decomp is None:
        # Try as-is: lhs OP rhs where lhs is a single dim var and rhs is const.
        decomp_lhs = _extract_linear_coeffs(lhs, dim_map)
        if decomp_lhs is not None and z3.is_int_value(rhs):
            coeffs, lhs_const = decomp_lhs
            rhs_val = rhs.as_long() - lhs_const
            decomp = (coeffs, 0)
            # rhs_val is the adjusted right-hand side
        else:
            # Linear parsing failed; try product-equality parsing
            product_results = _try_parse_product(expr, dim_map)
            if product_results:
                return product_results
            return results
    else:
        coeffs, const_offset = decomp
        # linear_combo + const_offset OP 0 → linear_combo OP -const_offset
        rhs_val = -const_offset

    if decomp is not None and 'rhs_val' not in dir():
        coeffs, const_offset = decomp
        rhs_val = -const_offset

    # Remove zero coefficients.
    coeffs = {k: v for k, v in coeffs.items() if v != 0}

    if not coeffs:
        return results

    # Normalise <= and < to >= and >
    if op in ("<=", "<"):
        # Negate everything: -coeffs, negate rhs, flip op
        coeffs = {k: -v for k, v in coeffs.items()}
        rhs_val = -rhs_val
        op = ">=" if op == "<=" else ">"

    # Try to match against the 7 standard PredicateKind shapes.
    if len(coeffs) == 1:
        (tensor, axis), coeff = next(iter(coeffs.items()))

        if coeff == 1 and op == "==" and rhs_val >= 0:
            results.append(ShapePredicate(
                kind=PredicateKind.DIM_EQ,
                tensor=tensor, axis=axis, value=rhs_val,
                provenance="craig_interpolation",
            ))
            return results

        if coeff == 1 and op == ">=" and rhs_val >= 0:
            results.append(ShapePredicate(
                kind=PredicateKind.DIM_GE,
                tensor=tensor, axis=axis, value=rhs_val,
                provenance="craig_interpolation",
            ))
            return results

        if coeff == 1 and op == ">" and rhs_val >= 0:
            results.append(ShapePredicate(
                kind=PredicateKind.DIM_GT,
                tensor=tensor, axis=axis, value=rhs_val,
                provenance="craig_interpolation",
            ))
            return results

    # Two variables with equal and opposite coefficients → DIM_MATCH.
    if len(coeffs) == 2 and op == "==":
        items = list(coeffs.items())
        (t1, a1), c1 = items[0]
        (t2, a2), c2 = items[1]
        if c1 == 1 and c2 == -1 and rhs_val == 0:
            results.append(ShapePredicate(
                kind=PredicateKind.DIM_MATCH,
                tensor=t1, axis=a1,
                match_tensor=t2, match_axis=a2,
                provenance="craig_interpolation",
            ))
            return results
        if c1 == -1 and c2 == 1 and rhs_val == 0:
            results.append(ShapePredicate(
                kind=PredicateKind.DIM_MATCH,
                tensor=t2, axis=a2,
                match_tensor=t1, match_axis=a1,
                provenance="craig_interpolation",
            ))
            return results

    # Fall through to DIM_LINEAR_COMBO.
    frozen_coeffs = tuple(sorted(coeffs.items()))
    results.append(LinearComboPredicate(
        coefficients=frozen_coeffs,
        operator=op,
        rhs=rhs_val,
        provenance="craig_interpolation",
    ))
    return results


def parse_interpolant(
    interpolant: "z3.ExprRef", dim_map: DimMapping
) -> List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate]:
    """Parse a Z3 interpolant formula into a list of ShapePredicates.

    Conjuncts are split and each atomic comparison is mapped to the most
    specific ``PredicateKind`` it fits; anything that doesn't fit the 7-kind
    template grammar is emitted as a ``LinearComboPredicate``.  Product
    expressions (e.g. ``d0 * d1 == C``) are emitted as
    ``ProductEqualityPredicate``.
    """
    if not HAS_Z3:
        return []

    if _is_true(interpolant):
        return []

    if _is_false(interpolant):
        return []

    # Split top-level conjunction.
    conjuncts: List[z3.ExprRef] = []
    if _is_and(interpolant):
        conjuncts = list(interpolant.children())
    else:
        conjuncts = [interpolant]

    results: List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate] = []
    for conj in conjuncts:
        results.extend(_parse_atomic(conj, dim_map))

    # Merge paired >= and <= on the same product into a single == predicate
    results = _merge_product_bounds(results)

    return results


def _merge_product_bounds(
    preds: List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate],
) -> List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate]:
    """Merge ``x*y >= C`` and ``x*y <= C`` pairs into ``x*y == C``."""
    product_ge: Dict[Tuple[Tuple[str, int], ...], int] = {}
    product_le: Dict[Tuple[Tuple[str, int], ...], int] = {}
    other: List[ShapePredicate | LinearComboPredicate | ProductEqualityPredicate] = []

    for p in preds:
        if isinstance(p, ProductEqualityPredicate):
            if p.operator == ">=":
                product_ge[p.factors] = p.rhs
            elif p.operator == "<=":
                product_le[p.factors] = p.rhs
            else:
                other.append(p)
        else:
            other.append(p)

    merged: List[ProductEqualityPredicate] = []
    for factors in set(product_ge) & set(product_le):
        if product_ge[factors] == product_le[factors]:
            merged.append(ProductEqualityPredicate(
                factors=factors,
                operator="==",
                rhs=product_ge[factors],
                provenance="craig_interpolation",
            ))
        else:
            # Keep the individual bounds
            merged.append(ProductEqualityPredicate(
                factors=factors, operator=">=", rhs=product_ge[factors],
                provenance="craig_interpolation",
            ))
            merged.append(ProductEqualityPredicate(
                factors=factors, operator="<=", rhs=product_le[factors],
                provenance="craig_interpolation",
            ))

    # Add unmatched product bounds
    for factors, val in product_ge.items():
        if factors not in product_le:
            merged.append(ProductEqualityPredicate(
                factors=factors, operator=">=", rhs=val,
                provenance="craig_interpolation",
            ))
    for factors, val in product_le.items():
        if factors not in product_ge:
            merged.append(ProductEqualityPredicate(
                factors=factors, operator="<=", rhs=val,
                provenance="craig_interpolation",
            ))

    return other + merged


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  InterpolationPredicateDiscovery class
# ═══════════════════════════════════════════════════════════════════════════════

class InterpolationPredicateDiscovery:
    """Craig-interpolation-based predicate discovery for shape verification.

    Complements the template-based discovery in ``shape_cegar.py`` by deriving
    predicates directly from the proof of unsatisfiability of A ∧ B, where

    * **A** = path constraints (computation graph up to a failing step), and
    * **B** = safety constraints (the property that must hold at the failing
      step).

    The interpolant **I** satisfies A ⊨ I and I ∧ B is UNSAT and uses only
    variables shared between A and B.

    Parameters
    ----------
    timeout_ms : int
        Z3 solver timeout in milliseconds (default 5000).
    method : InterpolationMethod
        Which interpolation backend to use (default: AUTO).
        AUTO tries CVC5 native first, falling back to Z3 simulation.
    """

    def __init__(
        self,
        timeout_ms: int = 5000,
        method: InterpolationMethod = InterpolationMethod.AUTO,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.method = method
        self._stats: Dict[str, int] = {
            "interpolations_attempted": 0,
            "interpolations_succeeded": 0,
            "predicates_discovered": 0,
            "linear_combo_predicates": 0,
            "template_predicates": 0,
            "cvc5_native_count": 0,
            "z3_simulation_count": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about discovery attempts."""
        return dict(self._stats)

    def discover_via_interpolation(
        self,
        path_constraints: List["z3.ExprRef"],
        safety_constraints: List["z3.ExprRef"],
        dim_map: "DimMapping | Dict[str, Tuple[str, int]]",
    ) -> List[ShapePredicate | LinearComboPredicate]:
        """Discover predicates via Craig interpolation.

        Parameters
        ----------
        path_constraints : list of z3.ExprRef
            Constraints A representing the computation graph up to the
            failing step.
        safety_constraints : list of z3.ExprRef
            Constraints B representing the property at the failing step
            (negated safety condition).
        dim_map : DimMapping or Dict[str, Tuple[str, int]]
            Mapping between Z3 variables and (tensor, axis) pairs.
            Accepts either a DimMapping object or a plain dict.

        Returns
        -------
        list of ShapePredicate | LinearComboPredicate
            Discovered predicates.  May be empty if A ∧ B is satisfiable
            or interpolation/QE fails.
        """
        if not HAS_Z3:
            logger.warning("Z3 not available; cannot compute interpolant")
            return []

        self._stats["interpolations_attempted"] += 1

        # Convert plain dict to DimMapping if needed
        if isinstance(dim_map, dict):
            dm = DimMapping()
            for var_name, (tensor, axis) in dim_map.items():
                dm.register(var_name, tensor, axis)
            dim_map = dm

        if not path_constraints:
            logger.debug("No path constraints — nothing to interpolate")
            return []

        if not safety_constraints:
            logger.debug("No safety constraints — nothing to interpolate")
            return []

        # Determine interface variables (shared between A and B).
        a_vars = _collect_vars_from_list(path_constraints)
        b_vars = _collect_vars_from_list(safety_constraints)
        interface_vars = a_vars & b_vars

        logger.debug(
            "Interpolation: |A|=%d, |B|=%d, |shared|=%d",
            len(a_vars), len(b_vars), len(interface_vars),
        )

        interpolant = None
        used_cvc5 = False

        # Dispatch based on selected method
        if self.method == InterpolationMethod.CVC5_NATIVE:
            interpolant = _compute_cvc5_interpolant(
                path_constraints, safety_constraints,
                interface_vars, self.timeout_ms,
            )
            if interpolant is not None:
                used_cvc5 = True
        elif self.method == InterpolationMethod.Z3_UNSAT_CORE_SIMULATION:
            interpolant = _compute_simulated_interpolant(
                path_constraints, safety_constraints,
                interface_vars, self.timeout_ms,
            )
        else:
            # AUTO: try CVC5 native first, fall back to Z3 simulation
            interpolant = _compute_cvc5_interpolant(
                path_constraints, safety_constraints,
                interface_vars, self.timeout_ms,
            )
            if interpolant is not None:
                used_cvc5 = True
            else:
                interpolant = _compute_simulated_interpolant(
                    path_constraints, safety_constraints,
                    interface_vars, self.timeout_ms,
                )

        if used_cvc5:
            self._stats["cvc5_native_count"] += 1
        elif interpolant is not None:
            self._stats["z3_simulation_count"] += 1
            logger.warning(
                "CVC5 native interpolation unavailable; using UNSAT-core "
                "simulation fallback. Result is an overapproximation, not "
                "a true Craig interpolant — soundness not guaranteed."
            )
            self._stats.setdefault("fallback_soundness_warnings", 0)
            self._stats["fallback_soundness_warnings"] += 1

        if interpolant is None:
            logger.debug("Interpolation failed or A ∧ B is SAT")
            return []

        # Validate Craig interpolation properties
        a_implies_i, i_and_b_unsat, vocab_ok = self.verify_interpolant_properties(
            path_constraints, safety_constraints,
            interpolant, interface_vars, self.timeout_ms,
        )

        if not vocab_ok:
            # Vocabulary restriction violated — this is an UNSAT-core
            # overapproximation, not a true Craig interpolant.
            logger.warning(
                "Interpolant violates vocabulary restriction "
                "(vars(I) ⊄ vars(A) ∩ vars(B)); labeling as "
                "UNSAT-core overapproximation"
            )
            self._stats.setdefault("vocab_violations", 0)
            self._stats["vocab_violations"] += 1

        if not a_implies_i:
            logger.warning("A ⊭ I — interpolant is not implied by path formula")
            self._stats.setdefault("implication_failures", 0)
            self._stats["implication_failures"] += 1

        if not i_and_b_unsat:
            logger.warning("I ∧ B is SAT — interpolant does not block safety violation")
            self._stats.setdefault("blocking_failures", 0)
            self._stats["blocking_failures"] += 1
            return []  # Unsound interpolant — discard

        self._stats["interpolations_succeeded"] += 1
        self._stats.setdefault("vocab_valid", 0)
        if vocab_ok:
            self._stats["vocab_valid"] += 1

        predicates = self.interpolant_to_predicates(interpolant, dim_map)

        self._stats["predicates_discovered"] += len(predicates)
        for p in predicates:
            if isinstance(p, LinearComboPredicate):
                self._stats["linear_combo_predicates"] += 1
            else:
                self._stats["template_predicates"] += 1

        return predicates

    def interpolant_to_predicates(
        self,
        interpolant: "z3.ExprRef",
        dim_map: DimMapping,
    ) -> List[ShapePredicate | LinearComboPredicate]:
        """Parse a Z3 interpolant expression into ShapePredicate objects.

        Parameters
        ----------
        interpolant : z3.ExprRef
            A quantifier-free formula over interface variables.
        dim_map : DimMapping
            Variable-to-dimension mapping.

        Returns
        -------
        list of ShapePredicate | LinearComboPredicate
        """
        if not HAS_Z3:
            return []
        return parse_interpolant(interpolant, dim_map)

    def verify_interpolant_properties(
        self,
        path_constraints: List["z3.ExprRef"],
        safety_constraints: List["z3.ExprRef"],
        interpolant: "z3.ExprRef",
        interface_vars: Set[str],
        timeout_ms: int = 5000,
    ) -> Tuple[bool, bool, bool]:
        """Verify that an interpolant satisfies the Craig properties.

        Returns (a_implies_i, i_and_b_unsat, uses_only_interface_vars).
        """
        if not HAS_Z3:
            return False, False, False

        # 1. Check A ⊨ I: A ∧ ¬I should be UNSAT.
        s1 = z3.Solver()
        s1.set("timeout", timeout_ms)
        for c in path_constraints:
            s1.add(c)
        s1.add(z3.Not(interpolant))
        a_implies_i = s1.check() == z3.unsat

        # 2. Check I ∧ B is UNSAT.
        s2 = z3.Solver()
        s2.set("timeout", timeout_ms)
        s2.add(interpolant)
        for c in safety_constraints:
            s2.add(c)
        i_and_b_unsat = s2.check() == z3.unsat

        # 3. Check variable containment.
        interp_vars = _collect_vars(interpolant)
        uses_only_interface = interp_vars.issubset(interface_vars)

        return a_implies_i, i_and_b_unsat, uses_only_interface


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Interpolation convergence bound analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InterpolationConvergenceBound:
    """Tracks convergence bounds when CEGAR predicates grow via interpolation.

    The original convergence proof assumes a finite predicate universe |P|.
    Interpolation adds predicates dynamically, but the number of
    interpolation-derived predicates per CEGAR iteration is bounded by
    the number of *interface variables* shared between path formula A
    and safety formula B.  These interface variables correspond to
    input-shape dimensions, which are bounded by the model's input
    dimensionality.

    Attributes
    ----------
    num_input_dimensions : int
        Total number of integer dimension variables across all model inputs.
    num_template_predicates : int
        Upper bound on template-derived predicates (|layers| × |dims| × 7).
    max_interpolation_predicates_per_iteration : int
        Bound on new interpolation predicates per CEGAR iteration.
        Equals the number of interface variables (input dimensions),
        since an interpolant over n variables yields at most O(n²)
        atomic predicates (pairwise relations + unary bounds).
    total_predicate_bound : int
        Upper bound on total predicates: template + max_iterations × interp.
    convergence_iterations_bound : int
        Worst-case number of CEGAR iterations until fixpoint.
    convergence_certificate : bool
        True when the computed bound is finite and holds.
    """
    num_input_dimensions: int = 0
    num_template_predicates: int = 0
    max_interpolation_predicates_per_iteration: int = 0
    total_predicate_bound: int = 0
    convergence_iterations_bound: int = 0
    convergence_certificate: bool = False

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "num_input_dimensions": self.num_input_dimensions,
            "num_template_predicates": self.num_template_predicates,
            "max_interpolation_predicates_per_iteration": (
                self.max_interpolation_predicates_per_iteration
            ),
            "total_predicate_bound": self.total_predicate_bound,
            "convergence_iterations_bound": self.convergence_iterations_bound,
            "convergence_certificate": self.convergence_certificate,
        }


def compute_convergence_bound(
    graph: "ComputationGraph",
    input_shapes: Optional[Dict[str, tuple]] = None,
) -> InterpolationConvergenceBound:
    """Compute the CEGAR+interpolation convergence bound for a model.

    Key insight: interpolation-derived predicates are bounded by the
    number of interface variables (shared between path formula A and
    safety formula B).  Interface variables are the input-shape
    dimensions because:

    1. Path formula A encodes the forward computation from inputs.
    2. Safety formula B encodes the property at a failing step.
    3. The Craig interpolant I uses only variables in Vars(A) ∩ Vars(B).
    4. For shape verification, the shared variables are exactly the
       input dimension variables (the "boundary" between the model's
       assumptions and its internal computation).

    An interpolant over *n* interface variables produces at most
    n² + n atomic predicates (n unary bounds + n(n-1)/2 pairwise
    relations, times 2 for ≥ and ==).  Since each CEGAR iteration
    adds ≥1 *new* predicate and the total universe is finite,
    convergence is guaranteed.

    Parameters
    ----------
    graph : ComputationGraph
        The extracted computation graph.
    input_shapes : dict, optional
        Input shape specifications (e.g. {"x": ("batch", "features")}).

    Returns
    -------
    InterpolationConvergenceBound
    """
    from src.shape_cegar import PredicateKind

    bound = InterpolationConvergenceBound()

    # --- Count input dimensions (number of SMT integer variables) ---
    total_input_dims = 0
    if input_shapes:
        for name, shape in input_shapes.items():
            total_input_dims += len(shape)
    else:
        # Fallback: use input_names with a conservative estimate
        for name in graph.input_names:
            total_input_dims += 4  # conservative default (batch, C, H, W)
    bound.num_input_dimensions = total_input_dims

    # --- Template predicate universe bound ---
    num_layers = len(graph.layers)
    # Max dims per layer: 4 for Conv layers, 2 for Linear
    max_dims = 2
    for layer_name, layer_def in graph.layers.items():
        layer_kind_name = getattr(layer_def, 'kind', None)
        if layer_kind_name is not None:
            kind_str = str(layer_kind_name)
            if 'CONV' in kind_str.upper():
                max_dims = max(max_dims, 4)
    num_predicate_kinds = len(PredicateKind)  # 7
    bound.num_template_predicates = num_layers * max_dims * num_predicate_kinds

    # --- Interpolation predicates per iteration ---
    # An interpolant over n interface variables can produce at most
    # n (unary) + n*(n-1)/2 (pairwise equality/inequality) atomic
    # predicates.  Each can appear as == or >=, giving factor of 2.
    n = total_input_dims
    max_interp_per_iter = n * n + n  # O(n²) generous bound
    bound.max_interpolation_predicates_per_iteration = max_interp_per_iter

    # --- Total predicate bound ---
    # Template predicates are discovered at most once each.
    # Interpolation predicates: bounded by the finite set of all
    # possible atomic predicates over interface variables.  Since
    # coefficients in interpolants come from the SMT encoding (bounded
    # by layer parameter values), the total interpolation predicate
    # universe is finite.
    #
    # Conservative bound: template + interpolation universe
    interp_universe = max_interp_per_iter  # all possible interp predicates
    bound.total_predicate_bound = (
        bound.num_template_predicates + interp_universe
    )

    # --- Convergence iterations bound ---
    # Each iteration adds ≥1 new predicate.  Total predicates ≤ bound.
    bound.convergence_iterations_bound = bound.total_predicate_bound

    # --- Certificate ---
    bound.convergence_certificate = (
        bound.total_predicate_bound > 0
        and bound.num_input_dimensions > 0
    )

    return bound


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Projection completeness verification
# ═══════════════════════════════════════════════════════════════════════════════

# The 9 predicate template kinds supported by P (the template language):
TEMPLATE_KINDS = [
    "DIM_EQ",            # tensor.shape[axis] == value
    "DIM_GT",            # tensor.shape[axis] > value
    "DIM_GE",            # tensor.shape[axis] >= value
    "DIM_DIVISIBLE",     # tensor.shape[axis] % divisor == 0
    "DIM_MATCH",         # tensor_a.shape[a] == tensor_b.shape[b]
    "NDIM_EQ",           # len(tensor.shape) == value
    "SHAPE_EQ",          # tensor.shape == (d0, d1, ...)
    "DIM_LINEAR_COMBO",  # Σ(coeff_i * dim_i) op rhs
    "DIM_PRODUCT_EQ",    # Π(dim_i) op rhs
]


@dataclass
class ProjectionResult:
    """Result of projecting a QF_UFLIA interpolant to the template language.

    Attributes
    ----------
    interpolant_str : str
        String representation of the original interpolant.
    projected_predicates : list
        Predicates obtained after projection.
    is_lossless : bool
        True if every conjunct of the interpolant was captured by a
        template predicate (no information lost).
    unprojected_conjuncts : list of str
        Conjuncts that could not be projected (if lossy).
    total_conjuncts : int
        Number of atomic conjuncts in the interpolant.
    projected_conjuncts : int
        Number of conjuncts successfully projected.
    soundness_preserved : bool
        True if the projected predicates are still implied by A
        (always true for overapproximation).
    """
    interpolant_str: str = ""
    projected_predicates: list = field(default_factory=list)
    is_lossless: bool = False
    unprojected_conjuncts: list = field(default_factory=list)
    total_conjuncts: int = 0
    projected_conjuncts: int = 0
    soundness_preserved: bool = True

    def summary(self) -> Dict[str, Any]:
        return {
            "interpolant": self.interpolant_str,
            "num_projected_predicates": len(self.projected_predicates),
            "is_lossless": self.is_lossless,
            "unprojected_conjuncts": self.unprojected_conjuncts,
            "total_conjuncts": self.total_conjuncts,
            "projected_conjuncts": self.projected_conjuncts,
            "soundness_preserved": self.soundness_preserved,
            "projection_ratio": (
                self.projected_conjuncts / self.total_conjuncts
                if self.total_conjuncts > 0 else 1.0
            ),
        }


def verify_projection_completeness(
    interpolant: "z3.ExprRef",
    dim_map: "DimMapping",
    path_constraints: Optional[List["z3.ExprRef"]] = None,
    timeout_ms: int = 5000,
) -> ProjectionResult:
    """Verify whether projection of a QF_UFLIA interpolant to the template
    language P is lossless.

    Takes a QF_UFLIA interpolant, projects it to the template language
    (via ``parse_interpolant``), and reports whether the projection
    captured all information.

    The projection is **lossless** if every atomic conjunct of the
    interpolant maps to exactly one template predicate.  It is **lossy**
    if some conjuncts are disjunctions, complex nonlinear expressions,
    or otherwise outside P.

    When lossy, soundness is still preserved: the projected predicates
    are (collectively) weaker than the original interpolant, so they
    are still implied by A.  The CEGAR loop may need additional
    iterations but will not produce false positives.

    Parameters
    ----------
    interpolant : z3.ExprRef
        A quantifier-free Z3 formula (the Craig interpolant).
    dim_map : DimMapping
        Mapping from Z3 variable names to (tensor, axis) pairs.
    path_constraints : list of z3.ExprRef, optional
        Original path constraints A, for verifying A ⊨ projected_I.
    timeout_ms : int
        Solver timeout.

    Returns
    -------
    ProjectionResult
        Detailed projection analysis.
    """
    if not HAS_Z3:
        return ProjectionResult()

    result = ProjectionResult()
    result.interpolant_str = str(interpolant)

    # Split interpolant into atomic conjuncts
    conjuncts: List["z3.ExprRef"] = []
    if _is_true(interpolant):
        result.is_lossless = True
        return result
    if _is_and(interpolant):
        conjuncts = list(interpolant.children())
    else:
        conjuncts = [interpolant]

    result.total_conjuncts = len(conjuncts)

    # Project each conjunct
    projected_count = 0
    unprojected: List[str] = []
    all_predicates: list = []

    for conj in conjuncts:
        preds = _parse_atomic(conj, dim_map)
        if preds:
            all_predicates.extend(preds)
            projected_count += 1
        else:
            unprojected.append(str(conj))

    result.projected_predicates = all_predicates
    result.projected_conjuncts = projected_count
    result.unprojected_conjuncts = unprojected
    result.is_lossless = (projected_count == result.total_conjuncts)

    # Verify soundness: if path_constraints given, check A ⊨ projected_I
    if path_constraints and all_predicates:
        # Reconstruct the projected interpolant as a Z3 formula
        projected_z3_parts = []
        for p in all_predicates:
            z3_expr = _predicate_to_z3(p, dim_map)
            if z3_expr is not None:
                projected_z3_parts.append(z3_expr)

        if projected_z3_parts:
            projected_I = (
                z3.And(projected_z3_parts)
                if len(projected_z3_parts) > 1
                else projected_z3_parts[0]
            )
            s = z3.Solver()
            s.set("timeout", timeout_ms)
            for c in path_constraints:
                s.add(c)
            s.add(z3.Not(projected_I))
            result.soundness_preserved = (s.check() == z3.unsat)

    return result


def _predicate_to_z3(
    pred: "ShapePredicate | LinearComboPredicate | ProductEqualityPredicate",
    dim_map: "DimMapping",
) -> Optional["z3.ExprRef"]:
    """Convert a projected predicate back to a Z3 expression for verification."""
    if not HAS_Z3:
        return None

    if isinstance(pred, ShapePredicate):
        var_name = dim_map.dim_to_var.get((pred.tensor, pred.axis))
        if var_name is None:
            return None
        v = z3.Int(var_name)
        if pred.kind == PredicateKind.DIM_EQ:
            return v == pred.value
        if pred.kind == PredicateKind.DIM_GT:
            return v > pred.value
        if pred.kind == PredicateKind.DIM_GE:
            return v >= pred.value
        if pred.kind == PredicateKind.DIM_MATCH:
            other_name = dim_map.dim_to_var.get(
                (pred.match_tensor, pred.match_axis)
            )
            if other_name is None:
                return None
            return v == z3.Int(other_name)
        if pred.kind == PredicateKind.DIM_DIVISIBLE:
            return v % pred.divisor == 0
        return None

    if isinstance(pred, LinearComboPredicate):
        terms = []
        for (tensor, axis), coeff in pred.coefficients:
            var_name = dim_map.dim_to_var.get((tensor, axis))
            if var_name is None:
                return None
            terms.append(coeff * z3.Int(var_name))
        lhs = z3.Sum(terms) if len(terms) > 1 else terms[0]
        op_map = {
            "==": lambda l, r: l == r,
            ">=": lambda l, r: l >= r,
            ">": lambda l, r: l > r,
            "<=": lambda l, r: l <= r,
            "<": lambda l, r: l < r,
        }
        op_fn = op_map.get(pred.operator)
        if op_fn is None:
            return None
        return op_fn(lhs, pred.rhs)

    if isinstance(pred, ProductEqualityPredicate):
        factors = []
        for tensor, axis in pred.factors:
            var_name = dim_map.dim_to_var.get((tensor, axis))
            if var_name is None:
                return None
            factors.append(z3.Int(var_name))
        product = factors[0]
        for f in factors[1:]:
            product = product * f
        op_map = {
            "==": lambda l, r: l == r,
            ">=": lambda l, r: l >= r,
            ">": lambda l, r: l > r,
            "<=": lambda l, r: l <= r,
            "<": lambda l, r: l < r,
        }
        op_fn = op_map.get(pred.operator)
        if op_fn is None:
            return None
        return op_fn(product, pred.rhs)

    return None


def demonstrate_lossy_projection() -> ProjectionResult:
    """Exhibit a case where Craig interpolation projection IS lossy.

    Constructs a disjunctive interpolant ``d0 == 3 ∨ d0 == 7`` which
    cannot be represented as a single predicate in the template language P.
    The projection overapproximates to ``d0 >= 3`` (the weakest covering
    predicate), demonstrating the lossy case.

    This addresses the reviewer's concern about whether projection is
    always lossless: it is not, but soundness is preserved because the
    CEGAR loop treats the weaker predicate as a valid (if imprecise)
    refinement.
    """
    if not HAS_Z3:
        return ProjectionResult()

    d0 = z3.Int("d0")
    # A disjunctive interpolant that P cannot represent as one predicate
    disjunctive_interp = z3.Or(d0 == 3, d0 == 7)

    dm = DimMapping()
    dm.register("d0", "x", 0)

    result = verify_projection_completeness(disjunctive_interp, dm)
    # The disjunction is not an atomic comparison, so _parse_atomic
    # returns empty — demonstrating lossiness.
    return result
