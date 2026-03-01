"""Tests for TorchDynamo PER_SUBGRAPH_SAFE gap analysis."""

import pytest

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    OpKind,
)
from src.dynamo_gap_analysis import (
    analyze_per_subgraph_safe_gap,
    CrossBreakDependency,
    MissedDependency,
    GapAnalysisResult,
    RiskLevel,
    get_backend_selection_info,
    _detect_cross_break_deps,
    _detect_missed_deps,
    _assess_risk,
    _is_shape_altering,
    SHAPE_ALTERING_OPS,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_graph(class_name: str, steps=None, inputs=None, outputs=None, features=None):
    g = ComputationGraph(class_name=class_name)
    g.steps = steps or []
    g.input_names = inputs or []
    g.output_names = outputs or []
    g.dynamic_features = features or {}
    return g


def _make_step(op, inputs, output, params=None):
    return ComputationStep(op=op, inputs=inputs, output=output, params=params or {})


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestNoGraphBreaks:
    """Single subgraph → MONOLITHIC_SAFE, LOW risk."""

    def test_single_subgraph_no_breaks(self):
        sg = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
            _make_step(OpKind.ACTIVATION, ["_t0"], "_t1"),
        ], inputs=["x"], outputs=["_t1"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg])
        assert result.num_graph_breaks == 0
        assert result.risk_assessment == RiskLevel.LOW
        assert result.composition_semantics == "MONOLITHIC_SAFE"

    def test_empty_subgraph(self):
        sg = _make_graph("Empty", steps=[], inputs=[], outputs=[])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg])
        assert result.num_graph_breaks == 0
        assert result.risk_assessment == RiskLevel.LOW


class TestDirectDependencies:
    """Two subgraphs with clean chaining → PER_SUBGRAPH_SAFE."""

    def test_two_subgraphs_clean_chain(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"])
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.ACTIVATION, ["_t0"], "_t1"),
        ], inputs=["_t0"], outputs=["_t1"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1])
        assert result.num_graph_breaks == 1
        assert result.composition_semantics == "PER_SUBGRAPH_SAFE"
        # Direct dep from sg0→sg1
        direct = [d for d in result.cross_break_dependencies if d.dependency_type == "direct"]
        assert len(direct) >= 1

    def test_three_subgraphs_clean_chain(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"])
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.ACTIVATION, ["_t0"], "_t1"),
        ], inputs=["_t0"], outputs=["_t1"])
        sg2 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["_t1"], "_t2"),
        ], inputs=["_t1"], outputs=["_t2"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1, sg2])
        assert result.num_graph_breaks == 2
        assert result.num_subgraphs == 3


class TestTransitiveDependencies:
    """Transitive dependency → UNKNOWN, HIGH risk."""

    def test_skip_one_subgraph(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"])
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.ACTIVATION, ["_t0"], "_t1"),
        ], inputs=["_t0"], outputs=["_t1"])
        # sg2 uses _t0 from sg0, skipping sg1
        sg2 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["_t0"], "_t2"),
        ], inputs=["_t0"], outputs=["_t2"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1, sg2])
        assert result.composition_semantics == "UNKNOWN"
        assert result.risk_assessment == RiskLevel.HIGH
        transitive = [d for d in result.cross_break_dependencies if d.dependency_type == "transitive"]
        assert len(transitive) >= 1


class TestImplicitDependencies:
    """Input from outside any subgraph → implicit dependency."""

    def test_external_input(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"])
        # sg1 uses "external_tensor" not produced by sg0
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["external_tensor"], "_t1"),
        ], inputs=["external_tensor"], outputs=["_t1"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1])
        assert result.composition_semantics == "UNKNOWN"
        implicit = [d for d in result.cross_break_dependencies if d.dependency_type == "implicit"]
        assert len(implicit) >= 1


class TestMissedDependencies:
    """Boundary shape-altering ops → missed dependency detection."""

    def test_reshape_at_boundary(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
            _make_step(OpKind.RESHAPE, ["_t0"], "_t1", {"target_shape": (1, -1)}),
        ], inputs=["x"], outputs=["_t1"])
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["_t1"], "_t2"),
        ], inputs=["_t1"], outputs=["_t2"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1])
        assert len(result.missed_dependencies) >= 1
        reshape_missed = [m for m in result.missed_dependencies if "Shape-altering" in m.reason]
        assert len(reshape_missed) >= 1

    def test_flatten_at_start(self):
        sg0 = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"])
        sg1 = _make_graph("Net", steps=[
            _make_step(OpKind.FLATTEN, ["_t0"], "_t1"),
            _make_step(OpKind.LAYER_CALL, ["_t1"], "_t2"),
        ], inputs=["_t0"], outputs=["_t2"])
        result = analyze_per_subgraph_safe_gap(subgraphs=[sg0, sg1])
        shape_altering = [m for m in result.missed_dependencies if "Shape-altering" in m.reason]
        assert len(shape_altering) >= 1


class TestRiskAssessment:

    def test_low_risk_no_breaks(self):
        assert _assess_risk(0, [], []) == RiskLevel.LOW

    def test_medium_risk_few_deps(self):
        deps = [
            CrossBreakDependency(0, 1, "t", "direct", ""),
            CrossBreakDependency(0, 1, "t2", "direct", ""),
            CrossBreakDependency(0, 1, "t3", "direct", ""),
        ]
        assert _assess_risk(1, deps, []) == RiskLevel.MEDIUM

    def test_high_risk_transitive(self):
        deps = [CrossBreakDependency(0, 2, "t", "transitive", "")]
        assert _assess_risk(2, deps, []) == RiskLevel.HIGH

    def test_high_risk_high_severity_missed(self):
        missed = [MissedDependency(1, "t", "reason", "high")]
        assert _assess_risk(1, [], missed) == RiskLevel.HIGH


class TestShapeAlteringDetection:

    def test_reshape_is_shape_altering(self):
        step = _make_step(OpKind.RESHAPE, ["x"], "_t0")
        assert _is_shape_altering(step)

    def test_activation_is_not_shape_altering(self):
        step = _make_step(OpKind.ACTIVATION, ["x"], "_t0")
        assert not _is_shape_altering(step)

    def test_param_based_detection(self):
        step = _make_step(OpKind.ACTIVATION, ["x"], "_t0", {"target_shape": (1, -1)})
        assert _is_shape_altering(step)


class TestGraphBasedAnalysis:
    """Analysis from a pre-composed ComputationGraph."""

    def test_graph_with_dynamo_features(self):
        g = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
            _make_step(OpKind.ACTIVATION, ["_t0"], "_t1"),
        ], inputs=["x"], outputs=["_t1"], features={
            "dynamo_traced": True,
            "graph_breaks": 1,
            "num_dynamo_subgraphs": 2,
            "composition_semantics": "PER_SUBGRAPH_SAFE",
        })
        result = analyze_per_subgraph_safe_gap(graph=g)
        assert result.num_graph_breaks == 1
        assert result.backend == "dynamo"

    def test_graph_without_breaks(self):
        g = _make_graph("Net", steps=[
            _make_step(OpKind.LAYER_CALL, ["x"], "_t0"),
        ], inputs=["x"], outputs=["_t0"], features={
            "dynamo_traced": True,
            "graph_breaks": 0,
            "num_dynamo_subgraphs": 1,
            "composition_semantics": "MONOLITHIC_SAFE",
        })
        result = analyze_per_subgraph_safe_gap(graph=g)
        assert result.num_graph_breaks == 0
        assert result.composition_semantics == "MONOLITHIC_SAFE"


class TestBackendSelectionInfo:

    def test_returns_structured_info(self):
        info = get_backend_selection_info()
        assert info["algorithm"] == "coverage-first, then soundness"
        assert len(info["selection_order"]) == 3
        backends = [e["backend"] for e in info["selection_order"]]
        assert "dynamo" in backends
        assert "fx" in backends
        assert "ast" in backends

    def test_soundness_relationships(self):
        info = get_backend_selection_info()
        rels = info["soundness_relationships"]
        assert "ast_vs_fx" in rels
        assert "fx_vs_dynamo" in rels
        assert "ast_vs_dynamo" in rels


class TestResultSerialization:

    def test_to_dict(self):
        result = GapAnalysisResult(
            num_graph_breaks=2,
            num_subgraphs=3,
            cross_break_dependencies=[
                CrossBreakDependency(0, 2, "t", "transitive", "desc"),
            ],
            missed_dependencies=[
                MissedDependency(1, "t", "reason", "high"),
            ],
            risk_assessment=RiskLevel.HIGH,
            composition_semantics="UNKNOWN",
            backend="dynamo",
        )
        d = result.to_dict()
        assert d["num_graph_breaks"] == 2
        assert d["risk_assessment"] == "HIGH"
        assert len(d["cross_break_dependencies"]) == 1
        assert len(d["missed_dependencies"]) == 1


class TestNoInputProvided:

    def test_no_model_or_graph(self):
        result = analyze_per_subgraph_safe_gap()
        assert result.num_graph_breaks == 0
        assert result.risk_assessment == RiskLevel.LOW
        assert result.backend == "none"
