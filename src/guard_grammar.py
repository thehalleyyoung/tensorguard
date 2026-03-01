r"""
Formal BNF Grammar for the Guard Predicate Language P.

This module provides a formal characterisation of the guard predicate language
used by the refinement-type inference system.  It addresses the gap identified
by reviewers Zhang and Sinha: "Guard predicate language lacks formal grammar
and semantic characterization — completeness, decidability boundaries, and
coding style sensitivity of the harvesting mechanism are unspecified."

BNF Grammar
===========

::

    ⟨sort⟩       ::= Int | Bool | Tag | Str

    ⟨var⟩        ::= IDENTIFIER                          (* program variable *)

    ⟨const⟩      ::= INTEGER_LITERAL
                    | BOOLEAN_LITERAL
                    | STRING_LITERAL

    ⟨term⟩       ::= ⟨var⟩
                    | ⟨const⟩
                    | 'len(' ⟨var⟩ ')'

    ⟨linear_expr⟩ ::= ⟨const⟩
                     | ⟨term⟩
                     | ⟨linear_expr⟩ '+' ⟨linear_expr⟩
                     | ⟨linear_expr⟩ '-' ⟨linear_expr⟩
                     | INTEGER_LITERAL '*' ⟨term⟩

    ⟨comp_op⟩    ::= '<' | '<=' | '>' | '>=' | '==' | '!='

    ⟨type_tag⟩   ::= IDENTIFIER                          (* e.g. int, str, list *)

    ⟨type_tag_list⟩ ::= ⟨type_tag⟩ ( ',' ⟨type_tag⟩ )*

    ⟨atom⟩       ::= ⟨linear_expr⟩ ⟨comp_op⟩ ⟨linear_expr⟩         (* Comparison      *)
                    | 'isinstance(' ⟨var⟩ ',' ⟨type_tag_list⟩ ')'   (* TypeTag          *)
                    | 'is_none(' ⟨var⟩ ')'                           (* Nullity          *)
                    | 'is_not_none(' ⟨var⟩ ')'                       (* Nullity          *)
                    | 'is_truthy(' ⟨var⟩ ')'                         (* Truthiness       *)
                    | 'hasattr(' ⟨var⟩ ',' STRING_LITERAL ')'        (* HasAttr          *)
                    | ⟨var⟩ ⟨comp_op⟩ 'len(' ⟨var⟩ ')'              (* LenComparison    *)
                    | ⟨var⟩ 'in' ⟨expr⟩                              (* Membership       *)
                    | 'callable(' ⟨var⟩ ')'                          (* Callable         *)
                    | ⟨var⟩ 'is' ⟨expr⟩                              (* Identity (pos)   *)
                    | ⟨var⟩ 'is not' ⟨expr⟩                          (* Identity (neg)   *)
                    | 'range_bound(' ⟨var⟩ ',' ⟨expr⟩? ',' ⟨expr⟩? ',' ⟨expr⟩? ')'
                                                                     (* RangeBound       *)
                    | 'except(' ⟨var⟩ ',' ⟨type_tag_list⟩ ')'       (* ExceptionType    *)
                    | 'match(' ⟨var⟩ ',' PATTERN_DESC ')'            (* PatternMatch     *)

    ⟨predicate⟩  ::= ⟨atom⟩
                    | ⟨predicate⟩ '∧' ⟨predicate⟩                   (* Conjunction      *)
                    | ⟨predicate⟩ '∨' ⟨predicate⟩                   (* Disjunction      *)
                    | '¬' ⟨predicate⟩                                (* Negation         *)
                    | 'assert(' ⟨predicate⟩ ',' STRING_LITERAL? ')' (* Assertion        *)

    ⟨shape_pred⟩ ::= ⟨var⟩ '.shape[' INTEGER '] == ' INTEGER        (* DIM_EQ           *)
                    | ⟨var⟩ '.shape[' INTEGER '] > '  INTEGER        (* DIM_GT           *)
                    | ⟨var⟩ '.shape[' INTEGER '] >= ' INTEGER        (* DIM_GE           *)
                    | ⟨var⟩ '.shape[' INTEGER '] % '  INTEGER ' == 0' (* DIM_DIVISIBLE   *)
                    | ⟨var⟩ '.shape[' INTEGER '] == ' ⟨var⟩ '.shape[' INTEGER ']'
                                                                     (* DIM_MATCH        *)
                    | 'ndim(' ⟨var⟩ ') == ' INTEGER                  (* NDIM_EQ          *)
                    | ⟨var⟩ '.shape == ' TUPLE                       (* SHAPE_EQ         *)

Decidability Classification
============================

**Decidable (P — polynomial time)**:
  TypeTag, Nullity, Truthiness, HasAttr, Callable, Identity, ExceptionType,
  PatternMatch, Membership (finite collections), Comparison over QF_LIA
  (linear integer arithmetic), LenComparison, RangeBound.
  Boolean combinations (∧, ∨, ¬, assert) of decidable predicates remain
  decidable.  Shape predicates DIM_EQ, DIM_GT, DIM_GE, DIM_MATCH, NDIM_EQ
  reduce to QF_LIA and are in P.

**NP-hard (reshape fragment)**:
  DIM_DIVISIBLE and SHAPE_EQ when involving product-equality constraints
  (d1·d2·…·dk = d1'·d2'·…·dk'), which reduces to SUBSET-PRODUCT.

**Undecidable**:
  Arbitrary non-linear integer arithmetic (NLIA) — e.g., comparisons whose
  operands involve products of two or more symbolic variables.  The system
  never generates such predicates, but they are expressible in the full
  grammar if extended with multiplication of non-constant terms.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from .guard_extractor import (
    AssertionPredicate,
    CallablePredicate,
    ComparisonOp,
    ComparisonPredicate,
    ConjunctionPredicate,
    DisjunctionPredicate,
    ExceptionTypePredicate,
    HasAttrPredicate,
    IdentityPredicate,
    LenComparisonPredicate,
    MembershipPredicate,
    NegationPredicate,
    NullityPredicate,
    PatternMatchPredicate,
    PredicateKind,
    PredicateTemplate,
    RangeBoundPredicate,
    TruthinessPredicate,
    TypeTagPredicate,
)

# Re-export PredicateKind values used by the CEGAR shape layer.
from .shape_cegar import PredicateKind as ShapePredicateKind
from .shape_cegar import ShapePredicate


# ---------------------------------------------------------------------------
# Decidability fragment classification
# ---------------------------------------------------------------------------

class DecidabilityClass(Enum):
    """Decidability classification for a guard predicate."""
    DECIDABLE_P = auto()         # polynomial-time decidable (QF_LIA / finite)
    NP_HARD = auto()             # NP-hard (reshape / product-equality)
    # Backward-compatible alias.
    NP_COMPLETE = NP_HARD
    UNDECIDABLE = auto()         # undecidable (arbitrary NLIA)


class GrammarCategory(Enum):
    """Syntactic category a predicate belongs to in the BNF."""
    ATOM_TYPETAG = auto()
    ATOM_NULLITY = auto()
    ATOM_TRUTHINESS = auto()
    ATOM_HASATTR = auto()
    ATOM_COMPARISON = auto()
    ATOM_LEN_COMPARISON = auto()
    ATOM_MEMBERSHIP = auto()
    ATOM_CALLABLE = auto()
    ATOM_IDENTITY = auto()
    ATOM_RANGE_BOUND = auto()
    ATOM_EXCEPTION_TYPE = auto()
    ATOM_PATTERN_MATCH = auto()
    CONNECTIVE_CONJUNCTION = auto()
    CONNECTIVE_DISJUNCTION = auto()
    CONNECTIVE_NEGATION = auto()
    CONNECTIVE_ASSERTION = auto()
    SHAPE_DIM_EQ = auto()
    SHAPE_DIM_GT = auto()
    SHAPE_DIM_GE = auto()
    SHAPE_DIM_DIVISIBLE = auto()
    SHAPE_DIM_MATCH = auto()
    SHAPE_NDIM_EQ = auto()
    SHAPE_SHAPE_EQ = auto()


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_PREDICATE_KIND_TO_CATEGORY = {
    PredicateKind.TypeTag: GrammarCategory.ATOM_TYPETAG,
    PredicateKind.Nullity: GrammarCategory.ATOM_NULLITY,
    PredicateKind.Truthiness: GrammarCategory.ATOM_TRUTHINESS,
    PredicateKind.HasAttr: GrammarCategory.ATOM_HASATTR,
    PredicateKind.Comparison: GrammarCategory.ATOM_COMPARISON,
    PredicateKind.LenComparison: GrammarCategory.ATOM_LEN_COMPARISON,
    PredicateKind.Membership: GrammarCategory.ATOM_MEMBERSHIP,
    PredicateKind.Callable: GrammarCategory.ATOM_CALLABLE,
    PredicateKind.Identity: GrammarCategory.ATOM_IDENTITY,
    PredicateKind.RangeBound: GrammarCategory.ATOM_RANGE_BOUND,
    PredicateKind.ExceptionType: GrammarCategory.ATOM_EXCEPTION_TYPE,
    PredicateKind.PatternMatch: GrammarCategory.ATOM_PATTERN_MATCH,
    PredicateKind.Conjunction: GrammarCategory.CONNECTIVE_CONJUNCTION,
    PredicateKind.Disjunction: GrammarCategory.CONNECTIVE_DISJUNCTION,
    PredicateKind.Negation: GrammarCategory.CONNECTIVE_NEGATION,
    PredicateKind.Assertion: GrammarCategory.CONNECTIVE_ASSERTION,
}

_SHAPE_KIND_TO_CATEGORY = {
    ShapePredicateKind.DIM_EQ: GrammarCategory.SHAPE_DIM_EQ,
    ShapePredicateKind.DIM_GT: GrammarCategory.SHAPE_DIM_GT,
    ShapePredicateKind.DIM_GE: GrammarCategory.SHAPE_DIM_GE,
    ShapePredicateKind.DIM_DIVISIBLE: GrammarCategory.SHAPE_DIM_DIVISIBLE,
    ShapePredicateKind.DIM_MATCH: GrammarCategory.SHAPE_DIM_MATCH,
    ShapePredicateKind.NDIM_EQ: GrammarCategory.SHAPE_NDIM_EQ,
    ShapePredicateKind.SHAPE_EQ: GrammarCategory.SHAPE_SHAPE_EQ,
}

# Valid comparison operators for the grammar.
_VALID_COMPARISON_OPS = frozenset({
    ComparisonOp.LT, ComparisonOp.LE, ComparisonOp.GT,
    ComparisonOp.GE, ComparisonOp.EQ, ComparisonOp.NE,
})


# ---------------------------------------------------------------------------
# Grammar validation
# ---------------------------------------------------------------------------

class GrammarError:
    """A single grammar violation."""

    def __init__(self, message: str, path: str = "") -> None:
        self.message = message
        self.path = path

    def __repr__(self) -> str:
        if self.path:
            return f"GrammarError({self.path}: {self.message})"
        return f"GrammarError({self.message})"


def validate_predicate(pred: object) -> List[GrammarError]:
    """Check whether *pred* conforms to the guard predicate grammar.

    Returns an empty list when the predicate is well-formed.
    """
    if isinstance(pred, ShapePredicate):
        return _validate_shape_predicate(pred)
    if isinstance(pred, PredicateTemplate):
        return _validate_guard_predicate(pred, "")
    return [GrammarError(f"Unknown predicate type: {type(pred).__name__}")]


def _validate_guard_predicate(
    pred: PredicateTemplate, path: str
) -> List[GrammarError]:
    errors: List[GrammarError] = []

    if isinstance(pred, TypeTagPredicate):
        if not pred.target_variable:
            errors.append(GrammarError("TypeTag: target_variable is empty", path))
        if not pred.type_names:
            errors.append(GrammarError("TypeTag: type_names is empty", path))
        elif not all(isinstance(t, str) and t for t in pred.type_names):
            errors.append(GrammarError("TypeTag: type_names contains invalid entries", path))

    elif isinstance(pred, NullityPredicate):
        if not pred.target_variable:
            errors.append(GrammarError("Nullity: target_variable is empty", path))
        if not isinstance(pred.is_none, bool):
            errors.append(GrammarError("Nullity: is_none must be bool", path))

    elif isinstance(pred, TruthinessPredicate):
        if not pred.target_variable:
            errors.append(GrammarError("Truthiness: target_variable is empty", path))

    elif isinstance(pred, HasAttrPredicate):
        if not pred.target_variable:
            errors.append(GrammarError("HasAttr: target_variable is empty", path))
        if not pred.attr_name:
            errors.append(GrammarError("HasAttr: attr_name is empty", path))

    elif isinstance(pred, ComparisonPredicate):
        if pred.op not in _VALID_COMPARISON_OPS:
            errors.append(GrammarError(
                f"Comparison: invalid op {pred.op!r}", path
            ))

    elif isinstance(pred, LenComparisonPredicate):
        if not pred.collection_variable:
            errors.append(GrammarError("LenComparison: collection_variable is empty", path))
        if pred.op not in _VALID_COMPARISON_OPS:
            errors.append(GrammarError(
                f"LenComparison: invalid op {pred.op!r}", path
            ))

    elif isinstance(pred, MembershipPredicate):
        if not pred.element_variable:
            errors.append(GrammarError("Membership: element_variable is empty", path))

    elif isinstance(pred, CallablePredicate):
        if not pred.target_variable:
            errors.append(GrammarError("Callable: target_variable is empty", path))

    elif isinstance(pred, IdentityPredicate):
        if not pred.left_variable:
            errors.append(GrammarError("Identity: left_variable is empty", path))
        if not isinstance(pred.is_positive, bool):
            errors.append(GrammarError("Identity: is_positive must be bool", path))

    elif isinstance(pred, RangeBoundPredicate):
        if not pred.loop_variable:
            errors.append(GrammarError("RangeBound: loop_variable is empty", path))

    elif isinstance(pred, ExceptionTypePredicate):
        if not pred.exception_variable:
            errors.append(GrammarError("ExceptionType: exception_variable is empty", path))
        if not pred.exception_types:
            errors.append(GrammarError("ExceptionType: exception_types is empty", path))

    elif isinstance(pred, PatternMatchPredicate):
        if not pred.subject_variable:
            errors.append(GrammarError("PatternMatch: subject_variable is empty", path))

    elif isinstance(pred, ConjunctionPredicate):
        if len(pred.children) < 2:
            errors.append(GrammarError("Conjunction: must have ≥2 children", path))
        for i, child in enumerate(pred.children):
            errors.extend(_validate_guard_predicate(child, f"{path}.children[{i}]"))

    elif isinstance(pred, DisjunctionPredicate):
        if len(pred.children) < 2:
            errors.append(GrammarError("Disjunction: must have ≥2 children", path))
        for i, child in enumerate(pred.children):
            errors.extend(_validate_guard_predicate(child, f"{path}.children[{i}]"))

    elif isinstance(pred, NegationPredicate):
        if pred.child is None:
            errors.append(GrammarError("Negation: child is None", path))
        else:
            errors.extend(_validate_guard_predicate(pred.child, f"{path}.child"))

    elif isinstance(pred, AssertionPredicate):
        if pred.inner is None:
            errors.append(GrammarError("Assertion: inner predicate is None", path))
        else:
            errors.extend(_validate_guard_predicate(pred.inner, f"{path}.inner"))

    else:
        errors.append(GrammarError(
            f"Unknown predicate kind: {type(pred).__name__}", path
        ))

    return errors


def _validate_shape_predicate(pred: ShapePredicate) -> List[GrammarError]:
    errors: List[GrammarError] = []
    if not pred.tensor:
        errors.append(GrammarError("ShapePredicate: tensor name is empty"))
    if pred.kind not in ShapePredicateKind.__members__.values():
        errors.append(GrammarError(f"ShapePredicate: unknown kind {pred.kind!r}"))
        return errors

    if pred.kind in (
        ShapePredicateKind.DIM_EQ, ShapePredicateKind.DIM_GT,
        ShapePredicateKind.DIM_GE,
    ):
        if pred.axis is None:
            errors.append(GrammarError(f"{pred.kind.name}: axis is required"))
        if pred.value is None:
            errors.append(GrammarError(f"{pred.kind.name}: value is required"))

    elif pred.kind == ShapePredicateKind.DIM_DIVISIBLE:
        if pred.axis is None:
            errors.append(GrammarError("DIM_DIVISIBLE: axis is required"))
        if pred.divisor is None or pred.divisor == 0:
            errors.append(GrammarError("DIM_DIVISIBLE: divisor must be non-zero int"))

    elif pred.kind == ShapePredicateKind.DIM_MATCH:
        if pred.axis is None:
            errors.append(GrammarError("DIM_MATCH: axis is required"))
        if not pred.match_tensor:
            errors.append(GrammarError("DIM_MATCH: match_tensor is required"))
        if pred.match_axis is None:
            errors.append(GrammarError("DIM_MATCH: match_axis is required"))

    elif pred.kind == ShapePredicateKind.NDIM_EQ:
        if pred.value is None:
            errors.append(GrammarError("NDIM_EQ: value is required"))

    elif pred.kind == ShapePredicateKind.SHAPE_EQ:
        if pred.value is None:
            errors.append(GrammarError("SHAPE_EQ: value (shape tuple) is required"))

    return errors


# ---------------------------------------------------------------------------
# Decidability classification
# ---------------------------------------------------------------------------

def classify_decidability(pred: object) -> DecidabilityClass:
    """Classify a predicate into its decidability fragment.

    * ``DECIDABLE_P``: the predicate belongs to a decidable theory fragment
      solvable in polynomial time (QF_LIA, finite-domain tag/null checks,
      Boolean combinations thereof).
    * ``NP_HARD``: the predicate involves product-equality constraints
      (reshape / SHAPE_EQ / DIM_DIVISIBLE with symbolic divisors), which
      reduce to SUBSET-PRODUCT (NP-hard; NP-membership not formalized).
    * ``UNDECIDABLE``: the predicate involves arbitrary non-linear integer
      arithmetic (NLIA).  The harvesting engine never produces these, but
      they are expressible in the extended grammar.
    """
    if isinstance(pred, ShapePredicate):
        return _classify_shape(pred)
    if isinstance(pred, PredicateTemplate):
        return _classify_guard(pred)
    return DecidabilityClass.UNDECIDABLE


def _classify_shape(pred: ShapePredicate) -> DecidabilityClass:
    # DIM_DIVISIBLE introduces modular arithmetic; with symbolic operands
    # this is NP-hard (product-equality / SUBSET-PRODUCT).
    if pred.kind == ShapePredicateKind.DIM_DIVISIBLE:
        return DecidabilityClass.NP_HARD
    # SHAPE_EQ may encode product-equality when reshape is involved.
    if pred.kind == ShapePredicateKind.SHAPE_EQ:
        return DecidabilityClass.NP_HARD
    # All other shape predicates are linear-arithmetic (QF_LIA).
    return DecidabilityClass.DECIDABLE_P


def _classify_guard(pred: PredicateTemplate) -> DecidabilityClass:
    # Connectives: worst-case of children.
    if isinstance(pred, ConjunctionPredicate):
        return _worst_class([_classify_guard(c) for c in pred.children])
    if isinstance(pred, DisjunctionPredicate):
        return _worst_class([_classify_guard(c) for c in pred.children])
    if isinstance(pred, NegationPredicate):
        if pred.child is None:
            return DecidabilityClass.DECIDABLE_P
        return _classify_guard(pred.child)
    if isinstance(pred, AssertionPredicate):
        if pred.inner is None:
            return DecidabilityClass.DECIDABLE_P
        return _classify_guard(pred.inner)

    # All atomic guard predicates are decidable in P:
    # TypeTag, Nullity, Truthiness, HasAttr, Comparison (QF_LIA),
    # LenComparison, Membership, Callable, Identity, RangeBound,
    # ExceptionType, PatternMatch.
    return DecidabilityClass.DECIDABLE_P


def _worst_class(classes: List[DecidabilityClass]) -> DecidabilityClass:
    if DecidabilityClass.UNDECIDABLE in classes:
        return DecidabilityClass.UNDECIDABLE
    if DecidabilityClass.NP_HARD in classes:
        return DecidabilityClass.NP_HARD
    return DecidabilityClass.DECIDABLE_P


# ---------------------------------------------------------------------------
# Grammar category lookup
# ---------------------------------------------------------------------------

def grammar_category(pred: object) -> Optional[GrammarCategory]:
    """Return the BNF grammar category for *pred*, or ``None`` if unknown."""
    if isinstance(pred, ShapePredicate):
        return _SHAPE_KIND_TO_CATEGORY.get(pred.kind)
    if isinstance(pred, PredicateTemplate):
        return _PREDICATE_KIND_TO_CATEGORY.get(pred.kind)
    return None


# ---------------------------------------------------------------------------
# Utility: collect all grammar categories covered by a predicate tree
# ---------------------------------------------------------------------------

def covered_categories(pred: object) -> Set[GrammarCategory]:
    """Recursively collect all grammar categories present in *pred*."""
    result: Set[GrammarCategory] = set()
    _collect_categories(pred, result)
    return result


def _collect_categories(pred: object, acc: Set[GrammarCategory]) -> None:
    cat = grammar_category(pred)
    if cat is not None:
        acc.add(cat)
    if isinstance(pred, (ConjunctionPredicate, DisjunctionPredicate)):
        for child in pred.children:
            _collect_categories(child, acc)
    elif isinstance(pred, NegationPredicate) and pred.child is not None:
        _collect_categories(pred.child, acc)
    elif isinstance(pred, AssertionPredicate) and pred.inner is not None:
        _collect_categories(pred.inner, acc)
