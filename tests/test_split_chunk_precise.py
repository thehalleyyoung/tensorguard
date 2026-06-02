"""Precise torch.chunk / torch.split shape rules."""

from __future__ import annotations

import textwrap

import pytest

from src.model_checker import verify_model
from src.tensor_shapes import (
    ShapeErrorKind,
    TensorShape,
    TensorShapeAnalyzer,
    analyze_shapes,
    compute_chunk_shapes,
    compute_split_shapes,
)


def _dims(shape: TensorShape) -> tuple:
    return tuple(d.value for d in shape.dims)


def _shape_map(src: str) -> dict[str, tuple]:
    result = TensorShapeAnalyzer().analyze_source(textwrap.dedent(src))
    assert result.errors == []
    return {name: _dims(shape) for name, shape in result.shapes.items()}


def _model_source(linear_features: int, *, functional: bool = False) -> str:
    split_expr = (
        "a, b, c = torch.chunk(x, 3, dim=-1)"
        if functional
        else "a, b, c = x.chunk(3, dim=-1)"
    )
    return f"""
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear({linear_features}, 1)

            def forward(self, x):
                {split_expr}
                return self.fc(c)
    """


@pytest.mark.parametrize(
    "size,chunks,expected",
    [
        (13, 6, [(3, 4), (3, 4), (3, 4), (3, 4), (1, 4)]),
        (10, 3, [(4, 4), (4, 4), (2, 4)]),
        (5, 8, [(1, 4), (1, 4), (1, 4), (1, 4), (1, 4)]),
        (0, 3, [(0, 4), (0, 4), (0, 4)]),
    ],
)
def test_compute_chunk_shapes_match_real_torch(size, chunks, expected):
    torch = pytest.importorskip("torch")
    x = torch.zeros(size, 4)
    real = [tuple(y.shape) for y in torch.chunk(x, chunks, dim=0)]
    static = compute_chunk_shapes(TensorShape.from_tuple((size, 4)), chunks, 0)
    assert real == expected
    assert [_dims(s) for s in static] == real


@pytest.mark.parametrize(
    "size,spec,expected",
    [
        (13, 6, [(2, 6), (2, 6), (2, 1)]),
        (10, 3, [(2, 3), (2, 3), (2, 3), (2, 1)]),
        (0, 0, [(2, 0)]),
        (5, [2, 0, 3], [(2, 2), (2, 0), (2, 3)]),
        (0, [0, 0], [(2, 0), (2, 0)]),
    ],
)
def test_compute_split_shapes_match_real_torch(size, spec, expected):
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, size)
    real = [tuple(y.shape) for y in torch.split(x, spec, dim=-1)]
    static = compute_split_shapes(TensorShape.from_tuple((2, size)), spec, -1)
    assert real == expected
    assert [_dims(s) for s in static] == real


def test_tensor_analyzer_method_chunk_uses_uneven_final_chunk():
    shapes = _shape_map("""
        import torch
        def f():
            x = torch.zeros(2, 10)
            a, b, c = x.chunk(3, dim=-1)
            return c
    """)
    assert shapes["a"] == (2, 4)
    assert shapes["b"] == (2, 4)
    assert shapes["c"] == (2, 2)


def test_tensor_analyzer_functional_split_negative_dim_and_empty_section():
    shapes = _shape_map("""
        import torch
        def f():
            x = torch.zeros(2, 5)
            a, empty, c = torch.split(x, [2, 0, 3], dim=-1)
            y = torch.cat([a, empty, c], dim=-1)
            return y
    """)
    assert shapes["a"] == (2, 2)
    assert shapes["empty"] == (2, 0)
    assert shapes["c"] == (2, 3)
    assert shapes["y"] == (2, 5)


def test_tensor_analyzer_linear_catches_wrong_final_chunk_width():
    result = analyze_shapes(textwrap.dedent("""
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 1)

            def forward(self, x):
                x = torch.zeros(2, 10)
                a, b, c = x.chunk(3, dim=-1)
                return self.fc(c)
    """))
    assert any(e.kind == ShapeErrorKind.DIM_MISMATCH and "got 2" in e.message
               for e in result.errors)


def test_tensor_analyzer_reports_concrete_chunk_unpack_count_mismatch():
    result = analyze_shapes(textwrap.dedent("""
        import torch
        def f():
            x = torch.zeros(5, 4)
            a, b, c, d, e, f, g, h = torch.chunk(x, 8, dim=0)
            return a
    """))
    assert any("unpack expects 8 outputs" in e.message for e in result.errors)


def test_verify_model_chunk_final_chunk_drives_linear_safety():
    safe = verify_model(textwrap.dedent(_model_source(2)), input_shapes={"x": (5, 10)})
    unsafe = verify_model(textwrap.dedent(_model_source(4)), input_shapes={"x": (5, 10)})
    assert safe.safe, safe.errors
    assert not unsafe.safe
    assert any("Linear expects last dim=4, got 2" in v.message
               for v in unsafe.counterexample.violations)


def test_verify_model_functional_chunk_form_matches_method_form():
    safe = verify_model(
        textwrap.dedent(_model_source(2, functional=True)),
        input_shapes={"x": (5, 10)},
    )
    assert safe.safe, safe.errors


def test_verify_model_symbolic_chunk_count_abstains_from_unpack_error():
    source = """
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                a, b, c = x.chunk(3, dim=0)
                return self.fc(a)
    """
    result = verify_model(textwrap.dedent(source), input_shapes={"x": ("batch", 10)})
    messages = [
        v.message for v in (
            result.counterexample.violations if result.counterexample else []
        )
    ]
    assert not any("unpack expects" in message for message in messages)
    assert result.safe, messages
