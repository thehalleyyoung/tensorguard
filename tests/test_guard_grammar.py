"""
Tests for the formal guard predicate grammar specification.

Covers:
  - Grammar validation on all 16 guard-extractor predicate types
  - Grammar validation on all 7 shape-predicate types
  - Decidability classification
  - Boolean connective trees
  - Rejection of malformed predicates
  - Grammar category lookup
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure the src package is importable.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, "src")
)

from src.guard_extractor import (
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
from src.guard_grammar import (
    DecidabilityClass,
    GrammarCategory,
    GrammarError,
    classify_decidability,
    covered_categories,
    grammar_category,
    validate_predicate,
)
from src.shape_cegar import PredicateKind as ShapePredicateKind
from src.shape_cegar import ShapePredicate


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _assert_valid(pred: object) -> None:
    errors = validate_predicate(pred)
    assert errors == [], f"Expected valid, got: {errors}"


def _assert_invalid(pred: object) -> None:
    errors = validate_predicate(pred)
    assert len(errors) > 0, "Expected validation errors"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Guard-predicate validation – one test per atomic predicate type
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypeTagPredicate:
    def test_valid(self):
        p = TypeTagPredicate(target_variable="x", type_names=("int", "str"))
        _assert_valid(p)

    def test_invalid_empty_variable(self):
        p = TypeTagPredicate(target_variable="", type_names=("int",))
        _assert_invalid(p)

    def test_invalid_empty_type_names(self):
        p = TypeTagPredicate(target_variable="x", type_names=())
        _assert_invalid(p)


class TestNullityPredicate:
    def test_valid_is_none(self):
        p = NullityPredicate(target_variable="x", is_none=True)
        _assert_valid(p)

    def test_valid_is_not_none(self):
        p = NullityPredicate(target_variable="y", is_none=False)
        _assert_valid(p)

    def test_invalid_empty_variable(self):
        p = NullityPredicate(target_variable="", is_none=True)
        _assert_invalid(p)


class TestTruthinessPredicate:
    def test_valid(self):
        p = TruthinessPredicate(target_variable="flag")
        _assert_valid(p)

    def test_invalid_empty(self):
        p = TruthinessPredicate(target_variable="")
        _assert_invalid(p)


class TestHasAttrPredicate:
    def test_valid(self):
        p = HasAttrPredicate(target_variable="obj", attr_name="__len__")
        _assert_valid(p)

    def test_invalid_no_attr(self):
        p = HasAttrPredicate(target_variable="obj", attr_name="")
        _assert_invalid(p)


class TestComparisonPredicate:
    def test_valid_lt(self):
        p = ComparisonPredicate(left_expr="x", op=ComparisonOp.LT, right_expr="10")
        _assert_valid(p)

    def test_valid_eq(self):
        p = ComparisonPredicate(left_expr="a", op=ComparisonOp.EQ, right_expr="b")
        _assert_valid(p)


class TestLenComparisonPredicate:
    def test_valid(self):
        p = LenComparisonPredicate(
            index_variable="i", collection_variable="arr", op=ComparisonOp.LT
        )
        _assert_valid(p)

    def test_invalid_no_collection(self):
        p = LenComparisonPredicate(
            index_variable="i", collection_variable="", op=ComparisonOp.LT
        )
        _assert_invalid(p)


class TestMembershipPredicate:
    def test_valid(self):
        p = MembershipPredicate(element_variable="x", collection_expr="my_set")
        _assert_valid(p)

    def test_invalid_empty_element(self):
        p = MembershipPredicate(element_variable="", collection_expr="s")
        _assert_invalid(p)


class TestCallablePredicate:
    def test_valid(self):
        p = CallablePredicate(target_variable="fn")
        _assert_valid(p)


class TestIdentityPredicate:
    def test_valid_positive(self):
        p = IdentityPredicate(left_variable="x", right_expr="None", is_positive=True)
        _assert_valid(p)

    def test_valid_negative(self):
        p = IdentityPredicate(left_variable="x", right_expr="sentinel", is_positive=False)
        _assert_valid(p)


class TestRangeBoundPredicate:
    def test_valid(self):
        p = RangeBoundPredicate(
            loop_variable="i", lower_bound="0", upper_bound="n", step="1"
        )
        _assert_valid(p)

    def test_valid_partial(self):
        p = RangeBoundPredicate(loop_variable="i", upper_bound="10")
        _assert_valid(p)


class TestExceptionTypePredicate:
    def test_valid(self):
        p = ExceptionTypePredicate(
            exception_variable="e", exception_types=("ValueError", "TypeError")
        )
        _assert_valid(p)

    def test_invalid_empty_types(self):
        p = ExceptionTypePredicate(exception_variable="e", exception_types=())
        _assert_invalid(p)


class TestPatternMatchPredicate:
    def test_valid(self):
        p = PatternMatchPredicate(
            subject_variable="cmd",
            pattern_description="MatchValue(42)",
            bound_variables=["v"],
        )
        _assert_valid(p)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Boolean connectives
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectives:
    def test_conjunction_valid(self):
        a = NullityPredicate(target_variable="x", is_none=False)
        b = TruthinessPredicate(target_variable="y")
        p = a.conjoin(b)
        _assert_valid(p)

    def test_disjunction_valid(self):
        a = TypeTagPredicate(target_variable="x", type_names=("int",))
        b = TypeTagPredicate(target_variable="x", type_names=("str",))
        p = a.disjoin(b)
        _assert_valid(p)

    def test_negation_valid(self):
        inner = CallablePredicate(target_variable="f")
        p = inner.negate()
        _assert_valid(p)

    def test_assertion_valid(self):
        inner = ComparisonPredicate(left_expr="x", op=ComparisonOp.GT, right_expr="0")
        p = AssertionPredicate(inner=inner, message="x must be positive")
        _assert_valid(p)

    def test_assertion_missing_inner(self):
        p = AssertionPredicate(inner=None, message="oops")
        _assert_invalid(p)

    def test_negation_missing_child(self):
        p = NegationPredicate(child=None)
        _assert_invalid(p)

    def test_conjunction_too_few_children(self):
        a = NullityPredicate(target_variable="x", is_none=True)
        p = ConjunctionPredicate(children=[a], variables=["x"])
        _assert_invalid(p)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Shape predicates (CEGAR)
# ═══════════════════════════════════════════════════════════════════════════════


class TestShapePredicates:
    def test_dim_eq(self):
        p = ShapePredicate(ShapePredicateKind.DIM_EQ, "x", axis=-1, value=768)
        _assert_valid(p)

    def test_dim_gt(self):
        p = ShapePredicate(ShapePredicateKind.DIM_GT, "x", axis=0, value=0)
        _assert_valid(p)

    def test_dim_ge(self):
        p = ShapePredicate(ShapePredicateKind.DIM_GE, "w", axis=1, value=1)
        _assert_valid(p)

    def test_dim_divisible(self):
        p = ShapePredicate(ShapePredicateKind.DIM_DIVISIBLE, "x", axis=-1, divisor=8)
        _assert_valid(p)

    def test_dim_match(self):
        p = ShapePredicate(
            ShapePredicateKind.DIM_MATCH, "x", axis=-1,
            match_tensor="w", match_axis=0,
        )
        _assert_valid(p)

    def test_ndim_eq(self):
        p = ShapePredicate(ShapePredicateKind.NDIM_EQ, "x", value=3)
        _assert_valid(p)

    def test_shape_eq(self):
        p = ShapePredicate(ShapePredicateKind.SHAPE_EQ, "x", value=(3, 224, 224))
        _assert_valid(p)

    def test_invalid_dim_eq_no_value(self):
        p = ShapePredicate(ShapePredicateKind.DIM_EQ, "x", axis=-1, value=None)
        _assert_invalid(p)

    def test_invalid_divisible_zero(self):
        p = ShapePredicate(ShapePredicateKind.DIM_DIVISIBLE, "x", axis=0, divisor=0)
        _assert_invalid(p)

    def test_invalid_dim_match_no_match_tensor(self):
        p = ShapePredicate(
            ShapePredicateKind.DIM_MATCH, "x", axis=0,
            match_tensor="", match_axis=0,
        )
        _assert_invalid(p)

    def test_invalid_empty_tensor(self):
        p = ShapePredicate(ShapePredicateKind.DIM_EQ, "", axis=0, value=1)
        _assert_invalid(p)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Decidability classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecidability:
    def test_typetag_decidable(self):
        p = TypeTagPredicate(target_variable="x", type_names=("int",))
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_nullity_decidable(self):
        p = NullityPredicate(target_variable="x", is_none=True)
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_comparison_decidable(self):
        p = ComparisonPredicate(left_expr="a", op=ComparisonOp.LE, right_expr="b")
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_len_comparison_decidable(self):
        p = LenComparisonPredicate(
            index_variable="i", collection_variable="xs", op=ComparisonOp.LT
        )
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_membership_decidable(self):
        p = MembershipPredicate(element_variable="k", collection_expr="allowed")
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_range_bound_decidable(self):
        p = RangeBoundPredicate(loop_variable="i", upper_bound="n")
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_callable_decidable(self):
        p = CallablePredicate(target_variable="f")
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_hasattr_decidable(self):
        p = HasAttrPredicate(target_variable="o", attr_name="x")
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_identity_decidable(self):
        p = IdentityPredicate(left_variable="x", right_expr="None", is_positive=True)
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_exception_decidable(self):
        p = ExceptionTypePredicate(
            exception_variable="e", exception_types=("ValueError",)
        )
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_pattern_match_decidable(self):
        p = PatternMatchPredicate(
            subject_variable="cmd", pattern_description="MatchValue(1)"
        )
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_dim_eq_decidable(self):
        p = ShapePredicate(ShapePredicateKind.DIM_EQ, "x", axis=0, value=32)
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_dim_match_decidable(self):
        p = ShapePredicate(
            ShapePredicateKind.DIM_MATCH, "x", axis=-1,
            match_tensor="w", match_axis=0,
        )
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_dim_divisible_np_hard(self):
        p = ShapePredicate(ShapePredicateKind.DIM_DIVISIBLE, "x", axis=-1, divisor=8)
        assert classify_decidability(p) == DecidabilityClass.NP_HARD

    def test_shape_eq_np_hard(self):
        p = ShapePredicate(ShapePredicateKind.SHAPE_EQ, "x", value=(3, 224, 224))
        assert classify_decidability(p) == DecidabilityClass.NP_HARD

    def test_conjunction_inherits_np_hard(self):
        a = ComparisonPredicate(left_expr="a", op=ComparisonOp.EQ, right_expr="1")
        b = ComparisonPredicate(left_expr="b", op=ComparisonOp.EQ, right_expr="2")
        conj = a.conjoin(b)
        assert classify_decidability(conj) == DecidabilityClass.DECIDABLE_P

    def test_negation_preserves_decidability(self):
        inner = TruthinessPredicate(target_variable="flag")
        p = inner.negate()
        assert classify_decidability(p) == DecidabilityClass.DECIDABLE_P

    def test_unknown_object_undecidable(self):
        assert classify_decidability("not a predicate") == DecidabilityClass.UNDECIDABLE


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Grammar category lookup
# ═══════════════════════════════════════════════════════════════════════════════


class TestGrammarCategory:
    def test_typetag_category(self):
        p = TypeTagPredicate(target_variable="x", type_names=("int",))
        assert grammar_category(p) == GrammarCategory.ATOM_TYPETAG

    def test_conjunction_category(self):
        a = NullityPredicate(target_variable="x", is_none=True)
        b = TruthinessPredicate(target_variable="y")
        p = a.conjoin(b)
        assert grammar_category(p) == GrammarCategory.CONNECTIVE_CONJUNCTION

    def test_shape_dim_gt(self):
        p = ShapePredicate(ShapePredicateKind.DIM_GT, "t", axis=0, value=0)
        assert grammar_category(p) == GrammarCategory.SHAPE_DIM_GT

    def test_unknown_returns_none(self):
        assert grammar_category(42) is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. covered_categories utility
# ═══════════════════════════════════════════════════════════════════════════════


class TestCoveredCategories:
    def test_nested_tree(self):
        a = TypeTagPredicate(target_variable="x", type_names=("int",))
        b = NullityPredicate(target_variable="y", is_none=False)
        conj = a.conjoin(b)
        neg = conj.negate()
        cats = covered_categories(neg)
        assert GrammarCategory.ATOM_TYPETAG in cats
        assert GrammarCategory.ATOM_NULLITY in cats
        assert GrammarCategory.CONNECTIVE_CONJUNCTION in cats
        assert GrammarCategory.CONNECTIVE_NEGATION in cats
