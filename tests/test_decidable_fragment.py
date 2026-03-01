"""Tests for QF_NIA decidable fragment classification in decidability module."""

from __future__ import annotations

import pytest

from src.decidability import (
    ConstraintFragmentInfo,
    DecidableFragmentReport,
    NIADecidableFragment,
    enforce_decidable_fragment,
)


class TestNIADecidableFragment:
    def test_four_fragments_exist(self):
        assert hasattr(NIADecidableFragment, "LINEAR")
        assert hasattr(NIADecidableFragment, "CONCRETE_SYMBOLIC_MUL")
        assert hasattr(NIADecidableFragment, "BOUNDED_SYMBOLIC_MUL")
        assert hasattr(NIADecidableFragment, "UNBOUNDED_SYMBOLIC_MUL")


class TestEnforceDecidableFragment:
    def test_concrete_integer(self):
        report = enforce_decidable_fragment({"dim": 42})
        assert report.total_constraints == 1
        assert report.linear_count == 1
        assert report.all_decidable

    def test_linear_expression(self):
        report = enforce_decidable_fragment(
            {"dim": "a + b"},
            concrete_dims={"a": 3, "b": 5},
        )
        assert report.total_constraints == 1
        assert report.linear_count == 1

    def test_concrete_symbolic_mul(self):
        report = enforce_decidable_fragment(
            {"embed_dim": "8 * head_dim"},
            concrete_dims={"8": 8},
        )
        # "8" is parsed as an integer literal by Python AST, not a Name
        # So head_dim is the only symbolic var in the product
        assert report.total_constraints == 1
        assert report.all_decidable

    def test_bounded_symbolic_mul(self):
        report = enforce_decidable_fragment(
            {"embed_dim": "heads * head_dim"},
            bounded_dims={"heads": (1, 128), "head_dim": (1, 512)},
        )
        assert report.total_constraints == 1
        assert report.bounded_symbolic_count == 1
        assert report.all_decidable

    def test_unbounded_symbolic_mul(self):
        report = enforce_decidable_fragment(
            {"embed_dim": "heads * head_dim"},
        )
        assert report.total_constraints == 1
        assert report.unbounded_symbolic_count == 1
        assert not report.all_decidable
        assert len(report.warnings) > 0

    def test_reject_unbounded(self):
        with pytest.raises(ValueError, match="[Uu]nbounded"):
            enforce_decidable_fragment(
                {"embed_dim": "heads * head_dim"},
                reject_unbounded=True,
            )

    def test_mixed_constraints(self):
        report = enforce_decidable_fragment(
            {
                "out_features": 10,
                "hidden": "in_features + 5",
                "embed_dim": "heads * head_dim",
            },
            concrete_dims={"in_features": 768},
        )
        assert report.total_constraints == 3
        assert report.linear_count >= 1
        assert report.unbounded_symbolic_count >= 1

    def test_report_details(self):
        report = enforce_decidable_fragment({"dim": "a * b"})
        assert len(report.constraint_details) == 1
        detail = report.constraint_details[0]
        assert isinstance(detail, ConstraintFragmentInfo)
        assert detail.fragment == NIADecidableFragment.UNBOUNDED_SYMBOLIC_MUL
        assert not detail.decidable

    def test_all_decidable_with_bounds(self):
        report = enforce_decidable_fragment(
            {
                "embed_dim": "heads * head_dim",
                "total": "batch * seq_len",
            },
            bounded_dims={
                "heads": (1, 16),
                "head_dim": (1, 64),
                "batch": (1, 32),
                "seq_len": (1, 512),
            },
        )
        assert report.all_decidable
        assert report.unbounded_symbolic_count == 0

    def test_empty_constraints(self):
        report = enforce_decidable_fragment({})
        assert report.total_constraints == 0
        assert report.all_decidable

    def test_constraint_fragment_info_fields(self):
        report = enforce_decidable_fragment(
            {"dim": "a * b"},
            bounded_dims={"a": (1, 10), "b": (1, 10)},
        )
        detail = report.constraint_details[0]
        assert detail.decidable
        assert detail.fragment == NIADecidableFragment.BOUNDED_SYMBOLIC_MUL
        assert len(detail.bounded_factors) > 0

    def test_matiyasevich_reference_in_reason(self):
        report = enforce_decidable_fragment({"dim": "x * y"})
        detail = report.constraint_details[0]
        assert "Matiyasevich" in detail.reason or "undecidable" in detail.reason
