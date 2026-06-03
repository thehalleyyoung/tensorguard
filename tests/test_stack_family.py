"""Step 198: exact torch.stack / hstack / vstack / dstack family rules."""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn

from src.fx_extractor import fx_trace_to_graph
from src.model_checker import ConstraintVerifier, extract_computation_graph, verify_model


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _tensor(shape):
    return torch.randn(shape)


def _state_for_module(module: nn.Module, input_shapes):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


def _fx_output_shape(module: nn.Module, input_shapes):
    violations, state, graph = _state_for_module(module, input_shapes)
    assert violations == []
    return _dims(state.shape_env[graph.output_names[-1]])


def _source_output_shape(source: str, input_shapes):
    graph = extract_computation_graph(textwrap.dedent(source))
    verifier = ConstraintVerifier(graph, input_shapes=input_shapes)
    violations, states, _ = verifier._bmc_base_case()
    assert violations == []
    return _dims(states[-1].shape_env[graph.output_names[-1]])


def _messages(violations):
    return [v.message for v in violations]


def test_stack_singleton_and_negative_dim_match_torch_source_and_fx():
    class M(nn.Module):
        def forward(self, x):
            return torch.stack((x,), dim=-1)

    x = _tensor((2, 3))
    real = M()(x)
    assert tuple(real.shape) == (2, 3, 1)
    assert _fx_output_shape(M(), {"x": tuple(x.shape)}) == tuple(real.shape)

    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, y):
                z = torch.stack((x, y), dim=-1)
                return z
    """
    real_source = torch.stack((_tensor((2, 3)), _tensor((2, 3))), dim=-1)
    assert _source_output_shape(
        source, {"x": (2, 3), "y": (2, 3)}
    ) == tuple(real_source.shape)
    assert verify_model(textwrap.dedent(source), input_shapes={"x": (2, 3), "y": (2, 3)}).safe


@pytest.mark.parametrize(
    ("name", "op", "shapes", "expected"),
    [
        ("hstack_0d_promotes_to_1d", torch.hstack, ((), ()), (2,)),
        ("hstack_1d_concats_dim0", torch.hstack, ((2,), (3,)), (5,)),
        ("hstack_2d_concats_dim1", torch.hstack, ((2, 3), (2, 4)), (2, 7)),
        ("vstack_0d_atleast_2d", torch.vstack, ((), ()), (2, 1)),
        ("vstack_1d_rows", torch.vstack, ((3,), (3,)), (2, 3)),
        ("vstack_mixed_1d_2d", torch.vstack, ((3,), (2, 3)), (3, 3)),
        ("dstack_0d_atleast_3d", torch.dstack, ((), ()), (1, 1, 2)),
        ("dstack_1d", torch.dstack, ((3,), (3,)), (1, 3, 2)),
        ("dstack_2d", torch.dstack, ((2, 3), (2, 3)), (2, 3, 2)),
        ("dstack_3d_concats_dim2", torch.dstack, ((2, 3, 4), (2, 3, 5)), (2, 3, 9)),
        ("column_stack_0d_columns", torch.column_stack, ((), ()), (1, 2)),
        ("column_stack_1d_columns", torch.column_stack, ((3,), (3,)), (3, 2)),
        ("column_stack_mixed_1d_2d", torch.column_stack, ((3,), (3, 2)), (3, 3)),
        ("row_stack_1d_alias", torch.row_stack, ((3,), (3,)), (2, 3)),
    ],
)
def test_stack_alias_rank_promotions_match_real_torch_fx(name, op, shapes, expected):
    del name

    class M(nn.Module):
        def forward(self, a, b):
            return op((a, b))

    a, b = (_tensor(shape) for shape in shapes)
    real = M()(a, b)
    assert tuple(real.shape) == expected
    assert _fx_output_shape(M(), {"a": shapes[0], "b": shapes[1]}) == expected


def test_source_extraction_handles_alias_tuple_inputs_and_column_stack_promotion():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, y):
                z = torch.column_stack((x, y))
                return z
    """
    real = torch.column_stack((_tensor((3,)), _tensor((3, 2))))
    assert tuple(real.shape) == (3, 3)
    assert _source_output_shape(
        source, {"x": (3,), "y": (3, 2)}
    ) == tuple(real.shape)
    assert verify_model(textwrap.dedent(source), input_shapes={"x": (3,), "y": (3, 2)}).safe


@pytest.mark.parametrize(
    ("op", "shapes", "message"),
    [
        (torch.stack, ((2, 3), (2, 4)), "stack: dimension 1 mismatch"),
        (torch.hstack, ((3,), (2, 3)), "hstack: tensors have different ndim"),
        (torch.dstack, ((2, 3), (4, 3)), "dstack: dimension 0 mismatch"),
        (torch.column_stack, ((3,), (4, 2)), "column_stack: dimension 0 mismatch"),
    ],
)
def test_stack_family_invalid_static_contracts_match_runtime_errors(op, shapes, message):
    class M(nn.Module):
        def forward(self, a, b):
            return op((a, b))

    a, b = (_tensor(shape) for shape in shapes)
    with pytest.raises(RuntimeError):
        M()(a, b)

    violations, _, _ = _state_for_module(M(), {"a": shapes[0], "b": shapes[1]})
    assert any(message in m for m in _messages(violations))
