"""Step 199: first-class structural tensor transfers."""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn

from src.fx_extractor import fx_trace_to_graph
from src.model_checker import (
    ConstraintVerifier,
    OpKind,
    extract_computation_graph,
)


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


def test_method_structural_chain_matches_real_torch_source_and_fx():
    class M(nn.Module):
        def forward(self, x):
            y = x.squeeze((1, 3))
            y = y.unsqueeze(-1)
            y = y.movedim((0, 2), (2, 0))
            y = y.swapaxes(0, 1)
            y = y.roll((1, 2), (0, 2))
            y = y.rot90(1, (0, 2))
            return y.flip((0, 2))

    x = torch.randn(2, 1, 4, 1)
    expected = tuple(M()(x).shape)
    assert expected == (2, 1, 4)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                y = x.squeeze((1, 3))
                y = y.unsqueeze(-1)
                y = y.movedim((0, 2), (2, 0))
                y = y.swapaxes(0, 1)
                y = y.roll((1, 2), (0, 2))
                y = y.rot90(1, (0, 2))
                return y.flip((0, 2))
    """
    assert _source_output_shape(source, {"x": tuple(x.shape)}) == expected


def test_function_structural_chain_matches_real_torch_source_and_fx():
    class M(nn.Module):
        def forward(self, x):
            y = torch.squeeze(x, dim=(1, 3))
            y = torch.unsqueeze(y, 0)
            y = torch.movedim(y, (0, 2), (2, 0))
            y = torch.swapaxes(y, 0, -1)
            y = torch.roll(y, 1, 1)
            y = torch.rot90(y, 2, (0, 1))
            return torch.flip(y, (0, 2))

    x = torch.randn(2, 1, 4, 1)
    expected = tuple(M()(x).shape)
    assert expected == (1, 2, 4)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == expected

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                y = torch.squeeze(x, dim=(1, 3))
                y = torch.unsqueeze(y, 0)
                y = torch.movedim(y, (0, 2), (2, 0))
                y = torch.swapaxes(y, 0, -1)
                y = torch.roll(y, 1, 1)
                y = torch.rot90(y, 2, (0, 1))
                return torch.flip(y, (0, 2))
    """
    assert _source_output_shape(source, {"x": tuple(x.shape)}) == expected


@pytest.mark.parametrize(
    ("source", "destination"),
    [(s, d) for s in range(-4, 4) for d in range(-4, 4)]
    + [
        ((0, 1), (2, 3)),
        ((0, 3), (3, 0)),
        ((-1, 0), (1, -1)),
        ((1, 3, 0), (0, 2, 3)),
        ((), ()),
    ],
)
def test_movedim_exhaustive_rank4_cases_match_torch_fx(source, destination):
    class M(nn.Module):
        def forward(self, x):
            return torch.movedim(x, source, destination)

    x = torch.randn(2, 3, 4, 5)
    expected = tuple(M()(x).shape)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == expected


@pytest.mark.parametrize(
    ("module", "shape", "message"),
    [
        (lambda: type("M", (nn.Module,), {
            "forward": lambda self, x: x.unsqueeze(5),
        })(), (2, 3, 4), "unsqueeze"),
        (lambda: type("M", (nn.Module,), {
            "forward": lambda self, x: torch.movedim(x, (0, -3), (1, 2)),
        })(), (2, 3, 4), "movedim"),
        (lambda: type("M", (nn.Module,), {
            "forward": lambda self, x: x.flip((0, 0)),
        })(), (2, 3, 4), "flip"),
        (lambda: type("M", (nn.Module,), {
            "forward": lambda self, x: x.rot90(1, (0, 0)),
        })(), (2, 3, 4), "rot90"),
        (lambda: type("M", (nn.Module,), {
            "forward": lambda self, x: x.roll((1, 2), (0,)),
        })(), (2, 3, 4), "roll"),
    ],
)
def test_invalid_static_contracts_refute_and_match_runtime(module, shape, message):
    m = module()
    with pytest.raises((IndexError, RuntimeError, TypeError)):
        m(torch.randn(*shape))

    violations, _, _ = _fx_state(m, {"x": shape})
    assert any(message in msg for msg in _messages(violations))


def test_symbolic_dims_do_not_create_false_positive_structural_violations():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                y = x.squeeze(0)
                y = y.squeeze(1)
                y = y.movedim((0, 2), (2, 0))
                y = y.rot90(1, (0, 2))
                y = y.flip((0, 2))
                return y.roll(1, dims=0)
    """
    violations, state, graph = _source_state(
        source, {"x": ("B", 1, "C", 5)})
    assert violations == []
    assert _dims(state.shape_env[graph.output_names[-1]]) == ("B", "C", 5)


def test_structural_ops_are_not_extracted_as_generic_activations():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                a = x.squeeze(1)
                b = a.unsqueeze(0)
                c = b.movedim(0, 2)
                d = c.swapaxes(0, 1)
                e = d.roll(1, 0)
                f = e.rot90(1, (0, 2))
                return f.flip((0, 1))
    """
    graph = extract_computation_graph(textwrap.dedent(source))
    ops = [step.op for step in graph.steps]
    assert OpKind.SQUEEZE in ops
    assert OpKind.UNSQUEEZE in ops
    assert OpKind.MOVEDIM in ops
    assert OpKind.TRANSPOSE in ops
    assert OpKind.ROLL in ops
    assert OpKind.ROT90 in ops
    assert OpKind.FLIP in ops
    assert not any(
        step.op == OpKind.ACTIVATION and step.line in {7, 8, 9, 10, 11, 12, 13}
        for step in graph.steps
    )

    class M(nn.Module):
        def forward(self, x):
            a = x.squeeze(1)
            b = a.unsqueeze(0)
            c = b.movedim(0, 2)
            d = c.swapaxes(0, 1)
            e = d.roll(1, 0)
            f = e.rot90(1, (0, 2))
            return f.flip((0, 1))

    fx_graph = fx_trace_to_graph(torch.fx.symbolic_trace(M()))
    fx_ops = [step.op for step in fx_graph.steps]
    assert {OpKind.SQUEEZE, OpKind.UNSQUEEZE, OpKind.MOVEDIM,
            OpKind.TRANSPOSE, OpKind.ROLL, OpKind.ROT90,
            OpKind.FLIP}.issubset(set(fx_ops))
