"""Tests for dynamic architecture failure root-cause analysis."""

import os
import sys

import pytest

IMPL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, IMPL_ROOT)

from experiments.run_dynamic_failure_analysis import (
    FAILURE_1_GNN_EDGE,
    FAILURE_2_ADAPTIVE_POOL,
    analyze_adaptive_pool_failure,
    analyze_gnn_edge_failure,
    check_jtms_applicability,
)


class TestGNNEdgeFailure:
    """Tests for gnn_edge_conditioned_bug false negative analysis."""

    def test_root_cause_is_specification_error(self):
        result = analyze_gnn_edge_failure()
        assert result["root_cause_category"] == "specification_error"

    def test_not_non_monotonic(self):
        result = analyze_gnn_edge_failure()
        assert result["is_non_monotonic_constraint"] is False

    def test_not_graph_break(self):
        result = analyze_gnn_edge_failure()
        assert result["is_graph_break"] is False

    def test_jtms_would_not_help(self):
        result = analyze_gnn_edge_failure()
        assert result["jtms_would_help"] is False

    def test_suggested_fix_exists(self):
        result = analyze_gnn_edge_failure()
        assert len(result["suggested_fix"]) > 0


class TestAdaptivePoolFailure:
    """Tests for adaptive_pooling_safe false positive analysis."""

    def test_root_cause_is_non_monotonic(self):
        result = analyze_adaptive_pool_failure()
        assert result["root_cause_category"] == "non_monotonic_constraint_pattern"

    def test_is_non_monotonic_constraint(self):
        result = analyze_adaptive_pool_failure()
        assert result["is_non_monotonic_constraint"] is True

    def test_jtms_would_help(self):
        result = analyze_adaptive_pool_failure()
        assert result["jtms_would_help"] is True


class TestJTMSApplicability:
    """Tests for JTMS integration assessment."""

    def test_jtms_helps_with_some(self):
        analyses = [analyze_gnn_edge_failure(), analyze_adaptive_pool_failure()]
        result = check_jtms_applicability(analyses)
        assert result["jtms_would_help"] >= 1
        assert result["jtms_would_not_help"] >= 1

    def test_total_matches_input(self):
        analyses = [analyze_gnn_edge_failure(), analyze_adaptive_pool_failure()]
        result = check_jtms_applicability(analyses)
        assert result["total_failures"] == 2
