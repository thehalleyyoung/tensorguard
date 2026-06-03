"""Step 201: gather/scatter index dtype and static value bounds."""

from __future__ import annotations

from typing import Optional

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.fx_extractor import fx_trace_to_graph, verify_module  # noqa: E402


def _messages(result, kind: Optional[str] = None):
    if result.counterexample is None:
        return []
    return [
        v.message for v in result.counterexample.violations
        if kind is None or v.kind == kind
    ]


class _GatherBuffer(nn.Module):
    def __init__(self, idx: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", idx)

    def forward(self, x):
        return torch.gather(x, 1, self.idx)


class _ScatterBuffer(nn.Module):
    def __init__(self, idx: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", idx)

    def forward(self, x, src):
        return x.scatter(1, self.idx, src)


class _IndexSelectBuffer(nn.Module):
    def __init__(self, idx: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", idx)

    def forward(self, x):
        return torch.index_select(x, 0, self.idx)


class _GatherInput(nn.Module):
    def forward(self, x, idx):
        return x.gather(1, idx)


def test_gather_buffer_static_oob_is_index_bounds_violation():
    idx = torch.tensor([[0, 3], [1, 2]], dtype=torch.long)
    model = _GatherBuffer(idx).eval()

    with pytest.raises(RuntimeError):
        model(torch.randn(2, 3))

    result = verify_module(model, input_shapes={"x": (2, 3)}, backend="fx")
    assert not result.safe
    assert any("gather" in m and "out of bounds" in m for m in _messages(result, "index_bounds"))
    assert not _messages(result, "shape_incompatible")


def test_gather_buffer_static_in_bounds_is_safe_and_int32_is_valid():
    idx = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    model = _GatherBuffer(idx).eval()
    assert tuple(model(torch.randn(2, 3)).shape) == (2, 2)

    result = verify_module(model, input_shapes={"x": (2, 3)}, backend="fx")
    assert result.safe


def test_dynamic_gather_index_has_dtype_check_but_unknown_bounds_abstain():
    result = verify_module(
        _GatherInput(),
        input_shapes={"x": (2, 3), "idx": (2, 2)},
        input_dtypes={"x": "float32", "idx": "int64"},
        backend="fx",
    )
    assert result.safe

    bad_dtype = verify_module(
        _GatherInput(),
        input_shapes={"x": (2, 3), "idx": (2, 2)},
        input_dtypes={"x": "float32", "idx": "float32"},
        backend="fx",
    )
    assert not bad_dtype.safe
    assert any("int32 or int64" in m for m in _messages(bad_dtype, "dtype_error"))

    with pytest.raises(RuntimeError):
        _GatherInput()(torch.randn(2, 3), torch.zeros(2, 2, dtype=torch.float32))


def test_scatter_buffer_static_oob_is_index_bounds_violation():
    idx = torch.tensor([[0, 4], [1, 2]], dtype=torch.long)
    model = _ScatterBuffer(idx).eval()
    x = torch.zeros(2, 3)
    src = torch.ones(2, 2)

    with pytest.raises(RuntimeError):
        model(x, src)

    result = verify_module(
        model,
        input_shapes={"x": (2, 3), "src": (2, 2)},
        input_dtypes={"x": "float32", "src": "float32"},
        backend="fx",
    )
    assert not result.safe
    assert any("scatter" in m and "out of bounds" in m for m in _messages(result, "index_bounds"))


def test_index_select_buffer_static_oob_is_index_bounds_violation():
    idx = torch.tensor([0, 4], dtype=torch.long)
    model = _IndexSelectBuffer(idx).eval()

    with pytest.raises((RuntimeError, IndexError)):
        model(torch.randn(4, 3))

    result = verify_module(model, input_shapes={"x": (4, 3)}, backend="fx")
    assert not result.safe
    assert any("index_select" in m and "out of bounds" in m for m in _messages(result, "index_bounds"))


def test_fx_constant_metadata_records_stable_buffer_dtype_and_range():
    idx = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    graph = fx_trace_to_graph(torch.fx.symbolic_trace(_GatherBuffer(idx).eval()))
    assert graph.const_value_ranges
    rng = next(iter(graph.const_value_ranges.values()))
    assert (rng.min_value, rng.max_value) == (0, 2)
    assert set(graph.const_dtypes.values()) == {"int32"}


def test_fx_folded_random_index_values_are_not_treated_as_static_bounds():
    class M(nn.Module):
        def forward(self, x):
            idx = torch.randint(0, 100, (2, 2), dtype=torch.long)
            return torch.gather(x, 1, idx)

    result = verify_module(M().eval(), input_shapes={"x": (2, 3)}, backend="fx")
    assert result.safe
    assert not _messages(result, "index_bounds")
