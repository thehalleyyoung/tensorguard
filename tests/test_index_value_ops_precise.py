"""Step 193: exact index/value-output shape and dtype transfers."""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn

from src.fx_extractor import fx_trace_to_graph, verify_module
from src.model_checker import (
    ConstraintVerifier,
    SymbolicShapePropagator,
    extract_computation_graph,
    verify_model,
)


def _dims(shape):
    return tuple(d.value for d in shape.dims)


def _messages(result):
    if result.counterexample is None:
        return []
    return [v.message for v in result.counterexample.violations]


def _state_for_module(module: nn.Module, input_shapes, input_dtypes=None):
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(module))
    verifier = ConstraintVerifier(
        graph,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
    )
    violations, states, _ = verifier._bmc_base_case()
    return violations, states[-1], graph


class _TakeAlongDim(nn.Module):
    def forward(self, x, idx):
        return torch.take_along_dim(x, idx, dim=1)


class _TakeAlongDimNone(nn.Module):
    def forward(self, x, idx):
        return torch.take_along_dim(x, idx, dim=None)


def test_take_along_dim_matches_torch_and_checks_index_dtype():
    x = torch.randn(2, 3, 4)
    idx = torch.zeros(2, 5, 4, dtype=torch.long)
    assert tuple(torch.take_along_dim(x, idx, dim=1).shape) == (2, 5, 4)

    violations, state, _ = _state_for_module(
        _TakeAlongDim(),
        {"x": (2, 3, 4), "idx": (2, 5, 4)},
        {"x": "float32", "idx": "int64"},
    )
    assert violations == []
    assert _dims(state.shape_env["_t0"]) == (2, 5, 4)
    assert state.dtype_env["_t0"] == "float32"

    bad = verify_module(
        _TakeAlongDim(),
        input_shapes={"x": (2, 3, 4), "idx": (2, 5, 4)},
        input_dtypes={"x": "float32", "idx": "int32"},
        backend="fx",
    )
    assert not bad.safe
    assert any("take_along_dim" in m and "int64" in m for m in _messages(bad))


def test_take_along_dim_dim_none_flattens_indices_like_torch():
    x = torch.randn(2, 3, 4)
    idx = torch.zeros(2, 3, dtype=torch.long)
    assert tuple(torch.take_along_dim(x, idx, dim=None).shape) == (6,)

    violations, state, _ = _state_for_module(
        _TakeAlongDimNone(),
        {"x": (2, 3, 4), "idx": (2, 3)},
        {"x": "float32", "idx": "int64"},
    )
    assert violations == []
    assert _dims(state.shape_env["_t0"]) == (6,)


def test_take_along_dim_bad_rank_and_non_dim_broadcast_are_refuted():
    rank_bad = verify_module(
        _TakeAlongDim(),
        input_shapes={"x": (2, 3, 4), "idx": (2, 4)},
        input_dtypes={"idx": "int64"},
        backend="fx",
    )
    assert not rank_bad.safe
    assert any("same rank" in m for m in _messages(rank_bad))

    bcast_bad = verify_module(
        _TakeAlongDim(),
        input_shapes={"x": (2, 3, 4), "idx": (3, 5, 4)},
        input_dtypes={"idx": "int64"},
        backend="fx",
    )
    assert not bcast_bad.safe
    assert any("not broadcastable" in m for m in _messages(bcast_bad))


class _ArgsortArgReduce(nn.Module):
    def forward(self, x):
        return torch.argsort(x, dim=1), torch.argmax(x, dim=1), torch.argmin(x, keepdim=True)


def test_argsort_argmax_argmin_shapes_and_int64_dtype():
    x = torch.randn(2, 3, 4)
    real = _ArgsortArgReduce()(x)
    assert [tuple(t.shape) for t in real] == [(2, 3, 4), (2, 4), (1, 1, 1)]
    assert all(t.dtype == torch.int64 for t in real)

    violations, state, _ = _state_for_module(
        _ArgsortArgReduce(),
        {"x": (2, 3, 4)},
        {"x": "float32"},
    )
    assert violations == []
    output_shapes = sorted(_dims(state.shape_env[name]) for name in state.shape_env if name.startswith("_t"))
    assert (2, 3, 4) in output_shapes
    assert (2, 4) in output_shapes
    assert (1, 1, 1) in output_shapes
    int64_outputs = [name for name, dt in state.dtype_env.items() if dt == "int64"]
    assert len([n for n in int64_outputs if n.startswith("_t")]) >= 3


class _SortTopkKth(nn.Module):
    def forward(self, x):
        sv, si = torch.sort(x, dim=1)
        tv, ti = torch.topk(x, k=2, dim=1)
        kv, ki = torch.kthvalue(x, k=2, dim=1, keepdim=True)
        return sv, si, tv, ti, kv, ki


class _MethodIndexOps(nn.Module):
    def forward(self, x, idx):
        sv, si = x.sort(dim=1)
        tv, ti = x.topk(2, dim=1)
        kv, ki = x.kthvalue(2, dim=1, keepdim=True)
        return (
            x.take_along_dim(idx, dim=1),
            x.argsort(dim=-1),
            sv,
            si,
            tv,
            ti,
            kv,
            ki,
            x.argmax(dim=1),
            x.argmin(dim=2, keepdim=True),
        )


def test_sort_topk_kthvalue_value_and_index_outputs_match_torch():
    x = torch.randn(2, 3, 4)
    real = _SortTopkKth()(x)
    assert [tuple(t.shape) for t in real] == [
        (2, 3, 4), (2, 3, 4), (2, 2, 4), (2, 2, 4), (2, 1, 4), (2, 1, 4)
    ]
    assert [t.dtype for t in real] == [
        torch.float32, torch.int64, torch.float32, torch.int64, torch.float32, torch.int64
    ]

    violations, state, graph = _state_for_module(
        _SortTopkKth(),
        {"x": (2, 3, 4)},
        {"x": "float32"},
    )
    assert violations == []
    outputs = SymbolicShapePropagator(graph).propagate({"x": (2, 3, 4)})
    assert any(_dims(s) == (2, 2, 4) for s in outputs.values())
    assert any(_dims(s) == (2, 1, 4) for s in outputs.values())
    assert any(dt == "int64" for name, dt in state.dtype_env.items() if name.startswith("_t"))


def test_tensor_method_dispatch_preserves_shapes_and_index_dtypes():
    x = torch.randn(2, 3, 4)
    idx = torch.zeros(2, 2, 4, dtype=torch.long)
    real = _MethodIndexOps()(x, idx)
    assert [tuple(t.shape) for t in real] == [
        (2, 2, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 2, 4),
        (2, 2, 4), (2, 1, 4), (2, 1, 4), (2, 4), (2, 3, 1),
    ]
    assert [t.dtype for t in real] == [
        torch.float32, torch.int64, torch.float32, torch.int64,
        torch.float32, torch.int64, torch.float32, torch.int64,
        torch.int64, torch.int64,
    ]

    violations, state, _ = _state_for_module(
        _MethodIndexOps(),
        {"x": (2, 3, 4), "idx": (2, 2, 4)},
        {"x": "float32", "idx": "int64"},
    )
    assert violations == []
    shapes = [_dims(shape) for name, shape in state.shape_env.items() if name.startswith("_t")]
    assert (2, 2, 4) in shapes
    assert (2, 1, 4) in shapes
    assert (2, 4) in shapes
    assert (2, 3, 1) in shapes
    assert sum(1 for name, dtype in state.dtype_env.items() if name.startswith("_t") and dtype == "int64") >= 6


class _BadTopk(nn.Module):
    def forward(self, x):
        return torch.topk(x, k=5, dim=1).values


class _BadKth(nn.Module):
    def forward(self, x):
        return torch.kthvalue(x, k=0, dim=1).values


def test_topk_and_kthvalue_static_k_errors_match_dispatcher():
    with pytest.raises(RuntimeError):
        torch.topk(torch.randn(2, 3, 4), k=5, dim=1)
    with pytest.raises(RuntimeError):
        torch.kthvalue(torch.randn(2, 3, 4), k=0, dim=1)

    topk = verify_module(_BadTopk(), input_shapes={"x": (2, 3, 4)}, backend="fx")
    assert not topk.safe
    assert any("topk" in m and "exceeds" in m for m in _messages(topk))

    kth = verify_module(_BadKth(), input_shapes={"x": (2, 3, 4)}, backend="fx")
    assert not kth.safe
    assert any("kthvalue" in m and ">= 1" in m for m in _messages(kth))


def test_source_tuple_unpack_tracks_index_dtype_into_take_along_dim():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                values, indices = torch.topk(x, k=2, dim=1)
                return torch.take_along_dim(x, indices, dim=1)
    """
    result = verify_model(
        textwrap.dedent(source),
        input_shapes={"x": (2, 3, 4)},
        input_dtypes={"x": "float32"},
    )
    assert result.safe

    graph = extract_computation_graph(textwrap.dedent(source))
    verifier = ConstraintVerifier(
        graph,
        input_shapes={"x": (2, 3, 4)},
        input_dtypes={"x": "float32"},
    )
    violations, states, _ = verifier._bmc_base_case()
    assert violations == []
    assert states[-1].dtype_env["indices"] == "int64"
    assert _dims(states[-1].shape_env["indices"]) == (2, 2, 4)
