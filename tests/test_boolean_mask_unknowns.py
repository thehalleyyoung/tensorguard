"""Step 191: boolean mask operators are either refuted or explicit UNKNOWN.

``masked_select``, ``nonzero``, single-argument ``where`` and boolean indexing
all expose tensor extents that depend on values.  TensorGuard should still catch
statically impossible mask shapes, but valid PyTorch programs must not be
rejected just because the selected element count is data-dependent.
"""

from __future__ import annotations

import textwrap

import pytest
import torch
import torch.nn as nn

from src.api import verify_architecture
from src.fx_extractor import verify_module
from src.model_checker import verify_model


def _kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def _messages(result):
    if result.counterexample is None:
        return []
    return [v.message for v in result.counterexample.violations]


def _unsafe_shape(result):
    return (not result.safe) and "shape_incompatible" in _kinds(result)


class _MaskedSelect(nn.Module):
    def forward(self, x, mask):
        return torch.masked_select(x, mask)


def test_masked_select_valid_broadcast_passes_with_unknown_length():
    x = torch.randn(4, 3)
    mask = torch.tensor([[True], [False], [True], [False]])
    assert torch.masked_select(x, mask).ndim == 1

    result = verify_module(
        _MaskedSelect(),
        input_shapes={"x": (4, 3), "mask": (4, 1)},
        input_dtypes={"mask": "bool"},
        backend="fx",
    )
    assert result.safe
    assert any("masked_select" in r for r in result.unknown_reasons)


def test_masked_select_bad_broadcast_is_refuted_and_matches_torch():
    with pytest.raises(RuntimeError):
        torch.masked_select(torch.randn(4, 3), torch.ones(5, 3, dtype=torch.bool))

    result = verify_module(
        _MaskedSelect(),
        input_shapes={"x": (4, 3), "mask": (5, 3)},
        input_dtypes={"mask": "bool"},
        backend="fx",
    )
    assert _unsafe_shape(result)
    assert any("masked_select" in m and "cannot broadcast" in m for m in _messages(result))


class _Nonzero(nn.Module):
    def forward(self, x):
        return torch.nonzero(x)


def test_nonzero_passes_with_data_dependent_row_count_unknown():
    out = torch.nonzero(torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]]))
    assert tuple(out.shape) == (3, 2)

    result = verify_module(_Nonzero(), input_shapes={"x": (2, 3)}, backend="fx")
    assert result.safe
    assert any("nonzero output row count" in r for r in result.unknown_reasons)


class _WhereIndices(nn.Module):
    def forward(self, mask):
        return torch.where(mask)


def test_single_argument_where_is_unknown_not_binary_broadcast():
    mask = torch.tensor([[True, False, True], [False, False, True]])
    where = torch.where(mask)
    assert len(where) == 2
    assert all(t.ndim == 1 for t in where)

    result = verify_module(
        _WhereIndices(),
        input_shapes={"mask": (2, 3)},
        input_dtypes={"mask": "bool"},
        backend="fx",
    )
    assert result.safe
    assert any("torch.where(condition)" in r for r in result.unknown_reasons)


class _WhereValues(nn.Module):
    def forward(self, cond, x, y):
        return torch.where(cond, x, y)


def test_three_argument_where_bad_condition_broadcast_is_refuted():
    with pytest.raises(RuntimeError):
        torch.where(
            torch.ones(5, 3, dtype=torch.bool),
            torch.randn(4, 3),
            torch.randn(4, 3),
        )

    result = verify_module(
        _WhereValues(),
        input_shapes={"cond": (5, 3), "x": (4, 3), "y": (4, 3)},
        input_dtypes={"cond": "bool"},
        backend="fx",
    )
    assert _unsafe_shape(result)
    assert any("torch.where" in m and "condition" in m for m in _messages(result))


class _BoolIndex(nn.Module):
    def forward(self, x, mask):
        return x[mask]


def test_boolean_index_valid_prefix_mask_passes_with_unknown_length():
    x = torch.randn(4, 3)
    mask = torch.tensor([True, False, True, False])
    assert tuple(x[mask].shape) == (2, 3)

    result = verify_module(
        _BoolIndex(),
        input_shapes={"x": (4, 3), "mask": (4,)},
        input_dtypes={"mask": "bool"},
        backend="fx",
    )
    assert result.safe
    assert any("boolean indexing output length" in r for r in result.unknown_reasons)


def test_boolean_index_bad_prefix_mask_is_refuted_and_matches_torch():
    with pytest.raises(IndexError):
        _ = torch.randn(4, 3)[torch.ones(5, dtype=torch.bool)]

    result = verify_module(
        _BoolIndex(),
        input_shapes={"x": (4, 3), "mask": (5,)},
        input_dtypes={"mask": "bool"},
        backend="fx",
    )
    assert _unsafe_shape(result)
    assert any("boolean_index" in m and "must match" in m for m in _messages(result))


class _LongIndex(nn.Module):
    def forward(self, x, idx):
        return x[idx]


def test_non_bool_tensor_index_abstains_instead_of_false_positiveing():
    x = torch.randn(5, 3)
    idx = torch.tensor([0, 2])
    assert tuple(x[idx].shape) == (2, 3)

    result = verify_module(
        _LongIndex(),
        input_shapes={"x": (5, 3), "idx": (2,)},
        input_dtypes={"idx": "long"},
        backend="fx",
    )
    assert result.safe
    assert any("tensor indexing dtype" in r for r in result.unknown_reasons)


def test_source_verify_architecture_surfaces_unknown_reasons():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x):
                return torch.nonzero(x)
    """
    result = verify_architecture(
        textwrap.dedent(source),
        input_shapes={"x": (4, 3)},
        max_cegar_iterations=0,
    )
    assert result.bugs == []
    assert result.verdict == "UNKNOWN"
    assert any("nonzero output row count" in r for r in result.unknown_reasons)


def test_source_verify_model_refutes_masked_select_shape_error():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def forward(self, x, mask):
                return torch.masked_select(x, mask)
    """
    result = verify_model(
        textwrap.dedent(source),
        input_shapes={"x": (4, 3), "mask": (5, 3)},
    )
    assert _unsafe_shape(result)
    assert any("masked_select" in m and "cannot broadcast" in m for m in _messages(result))
