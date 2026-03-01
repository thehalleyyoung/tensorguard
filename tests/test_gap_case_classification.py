"""Tests for gap_case_classification module."""

from __future__ import annotations

import pytest

from src.gap_case_classification import (
    GapCaseClass,
    GapCaseReport,
    CRAIG_INTERPOLATION_GAP_CASES,
    classify_gap_case,
)


class TestGapCaseClassEnum:
    def test_three_classes_exist(self):
        assert hasattr(GapCaseClass, "TURING_UNDECIDABLE")
        assert hasattr(GapCaseClass, "SOLVER_INCOMPLETE")
        assert hasattr(GapCaseClass, "SPECIFICATION_INCOMPLETE")

    def test_classes_are_distinct(self):
        classes = [
            GapCaseClass.TURING_UNDECIDABLE,
            GapCaseClass.SOLVER_INCOMPLETE,
            GapCaseClass.SPECIFICATION_INCOMPLETE,
        ]
        assert len(set(classes)) == 3


class TestClassifyGapCase:
    def test_concrete_integer_is_spec_incomplete(self):
        report = classify_gap_case(42)
        assert report.classification == GapCaseClass.SPECIFICATION_INCOMPLETE
        assert "no gap" in report.description.lower() or "concrete" in report.description.lower()

    def test_unspecified_dims_linear(self):
        report = classify_gap_case("a + b")
        assert report.classification == GapCaseClass.SPECIFICATION_INCOMPLETE
        assert "a" in report.unspecified_dims or "b" in report.unspecified_dims

    def test_unbounded_symbolic_mul(self):
        report = classify_gap_case("heads * head_dim")
        assert report.classification == GapCaseClass.TURING_UNDECIDABLE
        assert len(report.unbounded_symbolic_products) > 0

    def test_bounded_symbolic_mul_solver_unknown(self):
        report = classify_gap_case(
            "heads * head_dim",
            bounded_dims={"heads": (1, 128), "head_dim": (1, 512)},
            solver_returned_unknown=True,
        )
        assert report.classification == GapCaseClass.SOLVER_INCOMPLETE

    def test_bounded_symbolic_mul_no_solver_unknown(self):
        report = classify_gap_case(
            "heads * head_dim",
            bounded_dims={"heads": (1, 128), "head_dim": (1, 512)},
            solver_returned_unknown=False,
        )
        # Bounded, no solver issue → spec incomplete (need more info)
        assert report.classification == GapCaseClass.SPECIFICATION_INCOMPLETE

    def test_concrete_times_symbolic(self):
        """8 * head_dim is linear, not NIA."""
        report = classify_gap_case(
            "8 * head_dim",
            concrete_dims={"8": 8},
        )
        # "8" is a literal, not a name — so head_dim is unspecified
        assert report.classification in (
            GapCaseClass.SPECIFICATION_INCOMPLETE,
            GapCaseClass.TURING_UNDECIDABLE,
        )

    def test_all_concrete(self):
        report = classify_gap_case(
            "heads * head_dim",
            concrete_dims={"heads": 8, "head_dim": 64},
        )
        assert report.classification == GapCaseClass.SPECIFICATION_INCOMPLETE
        assert "no gap" in report.description.lower() or "tractable" in report.description.lower()

    def test_mitigation_includes_bounds_advice(self):
        report = classify_gap_case("a * b")
        assert report.classification == GapCaseClass.TURING_UNDECIDABLE
        assert "bound" in report.mitigation.lower()

    def test_solver_unknown_linear(self):
        report = classify_gap_case(
            "a + b",
            concrete_dims={"a": 10},
            solver_returned_unknown=True,
        )
        # b is unspecified, but it's linear
        assert report.classification in (
            GapCaseClass.SPECIFICATION_INCOMPLETE,
            GapCaseClass.SOLVER_INCOMPLETE,
        )

    def test_syntax_error_expression(self):
        report = classify_gap_case("+++invalid")
        assert report.classification == GapCaseClass.SPECIFICATION_INCOMPLETE


class TestCraigInterpolationGapCases:
    def test_three_gap_cases_defined(self):
        assert len(CRAIG_INTERPOLATION_GAP_CASES) == 3

    def test_unspecified_dimensions_case(self):
        case = CRAIG_INTERPOLATION_GAP_CASES[0]
        assert case["classification"] == GapCaseClass.SPECIFICATION_INCOMPLETE
        assert case["name"] == "unspecified_dimensions"

    def test_solver_unknown_case(self):
        case = CRAIG_INTERPOLATION_GAP_CASES[1]
        assert case["classification"] == GapCaseClass.SOLVER_INCOMPLETE
        assert case["name"] == "solver_unknown_on_decidable_qf_nia"

    def test_unbounded_multiplication_case(self):
        case = CRAIG_INTERPOLATION_GAP_CASES[2]
        assert case["classification"] == GapCaseClass.TURING_UNDECIDABLE
        assert case["name"] == "unbounded_symbolic_multiplication"

    def test_all_cases_have_mitigations(self):
        for case in CRAIG_INTERPOLATION_GAP_CASES:
            assert "mitigation" in case
            assert len(case["mitigation"]) > 10

    def test_undecidable_case_has_reference(self):
        case = CRAIG_INTERPOLATION_GAP_CASES[2]
        assert "reference" in case
        assert "Matiyasevich" in case["reference"]


class TestGapCaseReport:
    def test_report_fields(self):
        report = classify_gap_case("x * y")
        assert isinstance(report.classification, GapCaseClass)
        assert isinstance(report.description, str)
        assert isinstance(report.constraint_expr, str)
        assert isinstance(report.mitigation, str)

    def test_report_frozen(self):
        report = classify_gap_case("x * y")
        with pytest.raises(AttributeError):
            report.classification = GapCaseClass.SOLVER_INCOMPLETE  # type: ignore
