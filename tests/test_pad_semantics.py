"""Step 195: PyTorch-faithful pad semantics."""

from __future__ import annotations

import textwrap

import pytest
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


@pytest.mark.parametrize(
    ("shape", "pad", "mode"),
    [
        ((2, 3), (), "constant"),
        ((3,), (1, 2), "constant"),
        ((2, 4), (1, 2), "reflect"),
        ((2, 3, 4), (1, 2, 3, 4), "replicate"),
        ((1, 2, 3, 4), (1, 0, 2, 0, 0, 1), "circular"),
    ],
)
def test_functional_pad_modes_source_and_fx_match_torch(shape, pad, mode):
    class M(nn.Module):
        def forward(self, x):
            return F.pad(x, pad, mode=mode)

    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    real = M()(x)

    violations, state, graph = _state_for_module(M(), {"x": shape})
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)

    source = f"""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class S(nn.Module):
            def forward(self, x):
                return F.pad(x, {pad!r}, mode={mode!r})
    """
    source = textwrap.dedent(source)
    result = verify_model(source, input_shapes={"x": shape})
    assert result.safe

    source_graph = extract_computation_graph(source)
    verifier = ConstraintVerifier(source_graph, input_shapes={"x": shape})
    source_violations, source_states, _ = verifier._bmc_base_case()
    assert source_violations == []
    assert _output_dims(source_states[-1], source_graph) == tuple(real.shape)


class _Reflect2dTupleLength(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReflectionPad2d((1, 2))

    def forward(self, x):
        return self.pad(x)


def test_module_pad_tuple_length_matches_torch_source_and_fx():
    x = torch.randn(2, 3, 4)
    real = _Reflect2dTupleLength()(x)
    assert tuple(real.shape) == (2, 3, 7)

    violations, state, graph = _state_for_module(
        _Reflect2dTupleLength(),
        {"x": tuple(x.shape)},
    )
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)
    assert graph.layers["pad"].kind == LayerKind.REFLECTIONPAD2D

    source = """
        import torch
        import torch.nn as nn

        class S(nn.Module):
            def __init__(self):
                super().__init__()
                self.pad = nn.ReflectionPad2d((1, 2))

            def forward(self, x):
                return self.pad(x)
    """
    source = textwrap.dedent(source)
    result = verify_model(source, input_shapes={"x": tuple(x.shape)})
    assert result.safe

    source_graph = extract_computation_graph(source)
    verifier = ConstraintVerifier(source_graph, input_shapes={"x": tuple(x.shape)})
    source_violations, source_states, _ = verifier._bmc_base_case()
    assert source_violations == []
    assert _output_dims(source_states[-1], source_graph) == tuple(real.shape)


class _CircularPad3dModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.CircularPad3d((1, 0, 0, 1, 1, 0))

    def forward(self, x):
        return self.pad(x)


def test_fx_maps_circular_pad3d_module_and_matches_torch():
    x = torch.randn(1, 2, 3, 4, 5)
    real = _CircularPad3dModule()(x)

    violations, state, graph = _state_for_module(
        _CircularPad3dModule(),
        {"x": tuple(x.shape)},
    )
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)
    assert graph.layers["pad"].kind == LayerKind.CIRCULARPAD3D


class _ReplicateLargePad(nn.Module):
    def forward(self, x):
        return F.pad(x, (4, 0), mode="replicate")


def test_replicate_allows_padding_larger_than_input_dimension():
    x = torch.randn(1, 2, 3)
    real = _ReplicateLargePad()(x)
    assert tuple(real.shape) == (1, 2, 7)

    result = verify_module(
        _ReplicateLargePad(),
        input_shapes={"x": tuple(x.shape)},
        backend="fx",
    )
    assert result.safe

    violations, state, graph = _state_for_module(
        _ReplicateLargePad(),
        {"x": tuple(x.shape)},
    )
    assert violations == []
    assert _output_dims(state, graph) == tuple(real.shape)


@pytest.mark.parametrize(
    ("case", "shape", "expected_torch", "expected_message"),
    [
        (
            "reflect_rank",
            (1, 2, 3, 4),
            "Padding size 2 is not supported",
            "expects 2D or 3D input",
        ),
        (
            "reflect_bound",
            (1, 2, 3),
            "Padding size should be less than the corresponding input dimension",
            "reflect padding",
        ),
        (
            "circular_wrap",
            (1, 2, 3),
            "wrapping around more than once",
            "wrap dimension",
        ),
        (
            "constant_overcrop",
            (1, 2, 3),
            "narrow",
            "cannot crop dimension",
        ),
    ],
)
def test_invalid_pad_contracts_match_torch_errors(case, shape, expected_torch, expected_message):
    if case == "reflect_rank":
        class Bad(nn.Module):
            def forward(self, x):
                return F.pad(x, (1, 1), mode="reflect")
    elif case == "reflect_bound":
        class Bad(nn.Module):
            def forward(self, x):
                return F.pad(x, (3, 0), mode="reflect")
    elif case == "constant_overcrop":
        class Bad(nn.Module):
            def forward(self, x):
                return F.pad(x, (-4, 2), mode="constant")
    else:
        class Bad(nn.Module):
            def forward(self, x):
                return F.pad(x, (4, 0), mode="circular")

    x = torch.randn(shape)
    with pytest.raises((RuntimeError, NotImplementedError), match=expected_torch):
        Bad()(x)

    result = verify_module(Bad(), input_shapes={"x": shape}, backend="fx")
    assert not result.safe
    assert any(expected_message in m for m in _messages(result))
