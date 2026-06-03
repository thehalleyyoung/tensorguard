"""Step 200: repeat/tile and broadcast transfer semantics."""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn

from src.fx_extractor import fx_trace_to_graph
from src.model_checker import ConstraintVerifier, OpKind, extract_computation_graph


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _fx_state(module: nn.Module, input_shapes):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _fx_output_shape(module: nn.Module, input_shapes):
    violations, state, graph = _fx_state(module, input_shapes)
    assert violations == []
    return _dims(state.shape_env[graph.output_names[-1]])


def _source_state(source: str, input_shapes):
    graph = extract_computation_graph(textwrap.dedent(source))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _source_output_shape(source: str, input_shapes):
    violations, state, graph = _source_state(source, input_shapes)
    assert violations == []
    return _dims(state.shape_env[graph.output_names[-1]])


def _messages(violations):
    return [v.message for v in violations]


def test_repeat_interleave_and_tile_match_eager_source_and_fx():
    class M(nn.Module):
        def forward(self, x):
            y = x.repeat_interleave(2, dim=1)
            y = torch.repeat_interleave(y, 2, dim=2, output_size=8)
            return y.tile((2, 1, 3))

    x = torch.randn(2, 3, 4)
    expected = tuple(M()(x).shape)
    assert expected == (4, 6, 24)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                y = x.repeat_interleave(2, dim=1)
                y = torch.repeat_interleave(y, 2, dim=2, output_size=8)
                return y.tile((2, 1, 3))
    """
    assert _source_output_shape(source, {"x": tuple(x.shape)}) == expected


def test_repeat_interleave_flattening_matches_eager_source_and_fx():
    class M(nn.Module):
        def forward(self, x):
            return torch.repeat_interleave(x, 3)

    x = torch.randn(2, 3, 4)
    expected = tuple(M()(x).shape)
    assert expected == (72,)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.repeat_interleave(x, 3)
    """
    assert _source_output_shape(source, {"x": tuple(x.shape)}) == expected


def test_data_dependent_repeat_interleave_uses_static_output_size():
    class M(nn.Module):
        def forward(self, x, repeats):
            return torch.repeat_interleave(
                x, repeats, dim=1, output_size=4)

    x = torch.randn(2, 3, 4)
    repeats = torch.tensor([1, 2, 1])
    expected = tuple(M()(x, repeats).shape)
    assert expected == (2, 4, 4)
    assert _fx_output_shape(
        M(), {"x": tuple(x.shape), "repeats": tuple(repeats.shape)}
    ) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(
                    x, repeats, dim=1, output_size=4)
    """
    assert _source_output_shape(
        source, {"x": tuple(x.shape), "repeats": tuple(repeats.shape)}
    ) == expected


def test_data_dependent_repeat_interleave_without_output_size_abstains():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, repeats):
                return torch.repeat_interleave(x, repeats, dim=1)
    """
    violations, state, graph = _source_state(
        source, {"x": (2, 3, 4), "repeats": (3,)})
    assert violations == []
    out = _dims(state.shape_env[graph.output_names[-1]])
    assert out[0] == 2 and out[2] == 4
    assert str(out[1]).startswith("_repeat_interleave")
    assert any("repeat_interleave" in r for r in verifier_unknowns(source))


def verifier_unknowns(source: str):
    graph = extract_computation_graph(textwrap.dedent(source))
    verifier = ConstraintVerifier(
        graph, input_shapes={"x": (2, 3, 4), "repeats": (3,)})
    verifier._bmc_base_case()
    return verifier.unknown_reasons


def test_tile_and_repeat_distinguish_underlength_reps():
    class TileM(nn.Module):
        def forward(self, x):
            return x.tile((2, 2))

    x = torch.randn(2, 3, 4)
    assert tuple(TileM()(x).shape) == (2, 6, 8)
    assert _fx_output_shape(TileM(), {"x": tuple(x.shape)}) == (2, 6, 8)

    class RepeatM(nn.Module):
        def forward(self, x):
            return x.repeat(2, 2)

    with pytest.raises(RuntimeError):
        RepeatM()(x)
    violations, _, _ = _fx_state(RepeatM(), {"x": tuple(x.shape)})
    assert any("repeat dims" in msg for msg in _messages(violations))


def test_invalid_repeat_interleave_output_size_matches_runtime():
    class M(nn.Module):
        def forward(self, x):
            return torch.repeat_interleave(x, 2, dim=1, output_size=99)

    x = torch.randn(2, 3, 4)
    with pytest.raises(RuntimeError):
        M()(x)
    violations, _, _ = _fx_state(M(), {"x": tuple(x.shape)})
    assert any("Invalid output_size" in msg for msg in _messages(violations))


def test_broadcast_tensors_match_eager_source_and_fx():
    class M(nn.Module):
        def forward(self, x, y):
            a, b = torch.broadcast_tensors(x, y)
            return a + b

    x = torch.randn(2, 1, 3)
    y = torch.randn(1, 4, 3)
    expected = tuple(M()(x, y).shape)
    assert expected == (2, 4, 3)
    assert _fx_output_shape(
        M(), {"x": tuple(x.shape), "y": tuple(y.shape)}
    ) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, y):
                a, b = torch.broadcast_tensors(x, y)
                return a + b
    """
    assert _source_output_shape(
        source, {"x": tuple(x.shape), "y": tuple(y.shape)}
    ) == expected


def test_broadcast_tensors_invalid_static_shapes_refute():
    class M(nn.Module):
        def forward(self, x, y):
            return torch.broadcast_tensors(x, y)[0]

    x = torch.randn(2, 3)
    y = torch.randn(4, 3)
    with pytest.raises(RuntimeError):
        M()(x, y)
    violations, _, _ = _fx_state(
        M(), {"x": tuple(x.shape), "y": tuple(y.shape)})
    assert any("broadcast" in msg for msg in _messages(violations))


def test_source_broadcast_shapes_feeds_expand_precisely():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, y):
                shape = torch.broadcast_shapes(x.shape, y.shape)
                return x.expand(shape)
    """
    assert _source_output_shape(
        source, {"x": (2, 1, 3), "y": (1, 4, 3)}
    ) == (2, 4, 3)


def test_new_ops_are_not_extracted_as_generic_activations():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, y, repeats):
                a = x.repeat_interleave(repeats, dim=1, output_size=4)
                b = torch.tile(a, (1, 2, 1))
                c, _ = torch.broadcast_tensors(b, y)
                return c
    """
    graph = extract_computation_graph(textwrap.dedent(source))
    ops = [step.op for step in graph.steps]
    assert OpKind.REPEAT_INTERLEAVE in ops
    assert OpKind.TILE in ops
    assert OpKind.BROADCAST_TENSORS in ops
    assert not any(
        step.op == OpKind.ACTIVATION and step.line in {7, 8, 9}
        for step in graph.steps
    )
