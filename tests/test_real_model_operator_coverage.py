"""Step 208 -- real-model frequency-weighted operator coverage."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

from evaluation import real_model_operator_coverage as rmoc
from src.fx_extractor import fx_trace_to_graph, verify_module
from src.model_checker import ConstraintVerifier, LayerKind, OpKind

try:
    from torchvision.ops import stochastic_depth
    torch.fx.wrap("stochastic_depth")
except Exception:  # pragma: no cover - optional dependency
    stochastic_depth = None


def _committed():
    with open(rmoc.JSON_PATH) as fh:
        return json.load(fh)


def _dims(shape):
    return tuple(dim.value for dim in shape.dims)


def _graph_state(model, input_shapes):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(model))
    violations, states, _ = ConstraintVerifier(
        graph, input_shapes=input_shapes
    )._bmc_base_case()
    assert violations == []
    return graph, states[-1]


def test_step208_artifacts_exist_and_are_deterministic():
    assert os.path.exists(rmoc.JSON_PATH)
    assert os.path.exists(rmoc.MD_PATH)
    text = open(rmoc.JSON_PATH).read()
    assert text == rmoc._dumps(json.loads(text))


def test_step208_before_after_matrix_clears_threshold_without_metadata_padding():
    rep = _committed()
    summary = rep["summary"]
    assert summary["before_frequency_coverage_ratio"] < rmoc.THRESHOLD
    assert summary["after_frequency_coverage_ratio"] >= rmoc.THRESHOLD
    excluded = {
        row["operator"].lower()
        for row in summary["metadata_ops_excluded_from_new_coverage"]
    }
    assert {"size", "dim", "floordiv"}.issubset(excluded)


def test_step208_hot_operators_are_newly_covered_and_real():
    rep = _committed()
    hot = {
        row["operator"]
        for row in rep["summary"]["newly_covered_hot_operators"]
    }
    assert {
        "stochastic_depth",
        "layer_norm",
        "adaptive_avg_pool2d",
        "scaled_dot_product_attention",
    }.issubset(hot)
    assert all(row["frequency"] > 0 for row in rep["summary"]["newly_covered_hot_operators"])


def test_functional_layer_norm_fx_builds_layer_and_matches_torch_shape():
    class M(nn.Module):
        def forward(self, x):
            return F.layer_norm(x, (8,))

    graph, state = _graph_state(M(), {"x": (2, 4, 8)})
    layers = [layer for layer in graph.layers.values() if layer.kind == LayerKind.LAYERNORM]
    assert layers and layers[0].params["normalized_shape"] == (8,)
    assert _dims(state.shape_env[graph.output_names[-1]]) == tuple(
        F.layer_norm(torch.randn(2, 4, 8), (8,)).shape
    )


def test_functional_adaptive_avg_pool2d_fx_builds_exact_layer():
    class M(nn.Module):
        def forward(self, x):
            return F.adaptive_avg_pool2d(x, (3, 5))

    graph, state = _graph_state(M(), {"x": (2, 4, 12, 10)})
    layers = [
        layer for layer in graph.layers.values()
        if layer.kind == LayerKind.ADAPTIVE_AVGPOOL2D
    ]
    assert layers and layers[0].output_size == (3, 5)
    assert _dims(state.shape_env[graph.output_names[-1]]) == tuple(
        F.adaptive_avg_pool2d(torch.randn(2, 4, 12, 10), (3, 5)).shape
    )


def test_torchvision_stochastic_depth_fx_is_shape_preserving_activation():
    if stochastic_depth is None:
        pytest.skip("torchvision.ops.stochastic_depth unavailable")

    class M(nn.Module):
        def forward(self, x):
            return stochastic_depth(x, p=0.2, mode="row", training=False)

    traced = torch.fx.symbolic_trace(M())
    assert any(
        node.op == "call_function" and node.target is stochastic_depth
        for node in traced.graph.nodes
    )
    graph = fx_trace_to_graph(traced)
    assert any(step.op == OpKind.ACTIVATION for step in graph.steps)
    result = verify_module(M(), input_shapes={"x": ("batch", 3, 8, 8)})
    assert result.safe, result.errors


def test_sdpa_counted_hot_op_is_real_fx_sdpa_contract():
    if not hasattr(F, "scaled_dot_product_attention"):
        pytest.skip("SDPA unavailable")

    class M(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    graph = fx_trace_to_graph(torch.fx.symbolic_trace(M()))
    assert any(step.op == OpKind.SDPA for step in graph.steps)
    result = verify_module(
        M(),
        input_shapes={
            "q": ("batch", 4, 5, 8),
            "k": ("batch", 4, 7, 8),
            "v": ("batch", 4, 7, 9),
        },
    )
    assert result.safe, result.errors


def test_step208_check_passes_against_committed_artifact():
    assert rmoc.run(check=True, write=False) == 0
