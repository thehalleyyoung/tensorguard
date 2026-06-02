"""Step 194: precise fold/unfold spatial contracts."""

from __future__ import annotations

import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fx_extractor import fx_trace_to_graph, verify_module
from src.model_checker import (
    ConstraintVerifier,
    LayerKind,
    extract_computation_graph,
    verify_model,
)


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _messages(result):
    if result.counterexample is None:
        return []
    return [v.message for v in result.counterexample.violations]


def _state_for_module(module: nn.Module, input_shapes):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _output_dims(state, graph):
    return _dims(state.shape_env[graph.output_names[-1]])


class _BatchedUnfold(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(
            kernel_size=(2, 3),
            dilation=(1, 2),
            padding=(1, 0),
            stride=(2, 1),
        )

    def forward(self, x):
        return self.unfold(x)


def test_nn_unfold_batched_spatial_formula_matches_torch():
    x = torch.randn(2, 3, 5, 7)
    real = _BatchedUnfold()(x)
    assert tuple(real.shape) == (2, 18, 9)

    violations, state, graph = _state_for_module(_BatchedUnfold(), {"x": tuple(x.shape)})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)
    assert graph.layers["unfold"].kind == LayerKind.UNFOLD


class _UnbatchedFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold = nn.Fold(
            output_size=(4, 5),
            kernel_size=(2, 2),
            padding=(0, 1),
            stride=(1, 2),
        )

    def forward(self, x):
        return self.fold(x)


def test_nn_fold_unbatched_spatial_formula_matches_torch():
    x = torch.randn(12, 9)
    real = _UnbatchedFold()(x)
    assert tuple(real.shape) == (3, 4, 5)

    violations, state, graph = _state_for_module(_UnbatchedFold(), {"x": tuple(x.shape)})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)
    assert graph.layers["fold"].kind == LayerKind.FOLD


class _FunctionalRoundTrip(nn.Module):
    def forward(self, x):
        cols = F.unfold(x, kernel_size=(2, 2), stride=(2, 2))
        return F.fold(cols, output_size=(4, 4), kernel_size=(2, 2), stride=(2, 2))


def test_functional_unfold_fold_fx_round_trip_matches_torch():
    x = torch.randn(1, 3, 4, 4)
    real = _FunctionalRoundTrip()(x)
    assert tuple(real.shape) == (1, 3, 4, 4)

    violations, state, graph = _state_for_module(_FunctionalRoundTrip(), {"x": tuple(x.shape)})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)
    assert [layer.kind for layer in graph.layers.values()] == [
        LayerKind.UNFOLD,
        LayerKind.FOLD,
    ]


def test_functional_unfold_fold_source_round_trip_matches_torch():
    source = """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class M(nn.Module):
            def forward(self, x):
                cols = F.unfold(x, kernel_size=(2, 2), stride=(2, 2))
                return F.fold(cols, output_size=(4, 4), kernel_size=(2, 2), stride=(2, 2))
    """
    source = textwrap.dedent(source)
    result = verify_model(source, input_shapes={"x": (1, 3, 4, 4)})
    assert result.safe

    graph = extract_computation_graph(source)
    verifier = ConstraintVerifier(graph, input_shapes={"x": (1, 3, 4, 4)})
    violations, states, _ = verifier._bmc_base_case()
    assert violations == []
    assert _output_dims(states[-1], graph) == (1, 3, 4, 4)


class _BadUnfoldGrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=5)

    def forward(self, x):
        return self.unfold(x)


class _BadFoldChannels(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold = nn.Fold(output_size=(4, 4), kernel_size=2)

    def forward(self, x):
        return self.fold(x)


class _BadFoldBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold = nn.Fold(output_size=(4, 4), kernel_size=2)

    def forward(self, x):
        return self.fold(x)


def test_unfold_static_runtime_error_matches_torch():
    with torch.no_grad():
        try:
            _BadUnfoldGrid()(torch.randn(1, 3, 4, 4))
            raise AssertionError("torch should reject an empty sliding-block grid")
        except RuntimeError as exc:
            assert "sliding blocks" in str(exc)

    result = verify_module(
        _BadUnfoldGrid(),
        input_shapes={"x": (1, 3, 4, 4)},
        backend="fx",
    )
    assert not result.safe
    assert any("non-positive sliding block grid" in m for m in _messages(result))


def test_fold_static_runtime_errors_match_torch():
    with torch.no_grad():
        try:
            _BadFoldChannels()(torch.randn(1, 10, 9))
            raise AssertionError("torch should reject non-divisible fold channels")
        except RuntimeError as exc:
            assert "divisible by the product of kernel_size" in str(exc)

        try:
            _BadFoldBlocks()(torch.randn(1, 12, 8))
            raise AssertionError("torch should reject wrong fold column count")
        except RuntimeError as exc:
            assert "calculated number of sliding blocks" in str(exc)

    bad_channels = verify_module(
        _BadFoldChannels(),
        input_shapes={"x": (1, 10, 9)},
        backend="fx",
    )
    assert not bad_channels.safe
    assert any("divisible by kernel_size product" in m for m in _messages(bad_channels))

    bad_blocks = verify_module(
        _BadFoldBlocks(),
        input_shapes={"x": (1, 12, 8)},
        backend="fx",
    )
    assert not bad_blocks.safe
    assert any("sliding-block product" in m for m in _messages(bad_blocks))
