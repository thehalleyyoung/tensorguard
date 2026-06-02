from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from src.complex_verify import (  # noqa: E402
    verify_fft,
    verify_view_as_complex,
    verify_view_as_real,
)


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _tensor(shape, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ones(tuple(shape), dtype=dtype)


def _call_real_fft(op, shape, dtype, kwargs):
    fn = getattr(torch.fft, op)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(_tensor(shape, dtype), **kwargs)


def _assert_fft_matches_torch(op, shape, dtype, kwargs):
    verifier_kwargs = dict(kwargs)
    verdict = verify_fft(op, shape, dtype, **verifier_kwargs)
    try:
        actual = _call_real_fft(op, shape, dtype, kwargs)
    except Exception:
        assert not verdict.ok, (op, shape, dtype, kwargs, verdict)
        return

    assert verdict.ok, (op, shape, dtype, kwargs, verdict)
    assert verdict.output_shape == tuple(actual.shape)
    if verdict.output_dtype is not None:
        assert verdict.output_dtype == _dtype_name(actual.dtype)


def _fft_dtypes():
    dtypes = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ]
    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None:
        dtypes.append(complex32)
    return dtypes


def test_view_as_real_dtype_map_matches_torch():
    for dtype in [getattr(torch, "complex32", None), torch.complex64, torch.complex128]:
        if dtype is None:
            continue
        verdict = verify_view_as_real((3, 4), dtype)
        actual = torch.view_as_real(_tensor((3, 4), dtype))
        assert verdict.ok
        assert verdict.output_shape == tuple(actual.shape)
        assert verdict.output_dtype == _dtype_name(actual.dtype)

    verdict = verify_view_as_real((3, 4), torch.float32)
    assert not verdict.ok
    assert verdict.error_kind == "dtype"


def test_view_as_complex_dtype_and_last_dim_contract_matches_torch():
    for dtype in [torch.float16, torch.float32, torch.float64]:
        verdict = verify_view_as_complex((3, 2), dtype)
        actual = torch.view_as_complex(_tensor((3, 2), dtype))
        assert verdict.ok
        assert verdict.output_shape == tuple(actual.shape)
        assert verdict.output_dtype == _dtype_name(actual.dtype)

    for dtype in [torch.bfloat16, torch.int32, torch.bool, torch.complex64]:
        verdict = verify_view_as_complex((3, 2), dtype)
        with pytest.raises(Exception):
            torch.view_as_complex(_tensor((3, 2), dtype))
        assert not verdict.ok
        assert verdict.error_kind == "dtype"

    assert not verify_view_as_complex((3, 3), torch.float32).ok
    assert not verify_view_as_complex((), torch.float32).ok
    symbolic = verify_view_as_complex(("batch", "two"), None)
    assert symbolic.ok
    assert symbolic.output_shape == ("batch",)
    assert symbolic.output_dtype is None


def test_single_axis_fft_differential_against_real_torch():
    cases = [
        ((4, 5), {}),
        ((4, 5), {"n": 3, "dim": 0}),
        ((4, 5), {"n": 0, "dim": -1}),
        ((4, 5), {"dim": 7}),
        ((0, 5), {"dim": 0}),
        ((), {}),
    ]
    for op in ("fft", "ifft", "rfft", "irfft"):
        for shape, kwargs in cases:
            for dtype in _fft_dtypes():
                _assert_fft_matches_torch(op, shape, dtype, kwargs)


def test_multi_axis_fft_differential_against_real_torch():
    cases = [
        ((4, 5), {}),
        ((4, 5), {"s": (3,), "dim": (0,)}),
        ((4, 5), {"s": (-1, 3)}),
        ((4, 5), {"s": (0,), "dim": (0,)}),
        ((4, 5), {"dim": ()}),
        ((4, 5), {"dim": (0, 0)}),
        ((4, 5), {"dim": (1, 0)}),
        ((4,), {"s": (4, 4)}),
        ((), {}),
    ]
    for op in ("fftn", "ifftn", "rfftn", "irfftn"):
        for shape, kwargs in cases:
            for dtype in _fft_dtypes():
                _assert_fft_matches_torch(op, shape, dtype, kwargs)


def test_fft_hand_edge_cases_are_explicit():
    assert verify_fft("rfft", (4,), torch.complex64).error_kind == "dtype"
    assert verify_fft("irfft", (5,), torch.float32).ok
    assert verify_fft("irfft", (1,), torch.float32).error_kind == "size"
    assert verify_fft("fftn", (4, 5), torch.float32, dim=()).ok
    assert verify_fft("rfftn", (4, 5), torch.float32, dim=()).error_kind == "dim"
    assert verify_fft("rfftn", (4, 5), torch.float32, dim=(1, 0)).output_shape == (3, 5)
    assert verify_fft("fftn", (4, 5), torch.float32, s=(-1, 3)).output_shape == (4, 3)
    assert verify_fft("fft", ("seq", 8), None).output_shape == ("seq", 8)
    assert verify_fft("rfft", ("seq", 8), None, dim=0).output_shape == ("rfft(seq)", 8)
    assert verify_fft("irfft", ("freq",), None).output_shape == ("irfft(freq)",)


def test_integer_fft_output_dtype_abstains_from_global_default_dtype():
    old_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = torch.fft.fft(torch.ones(4, dtype=torch.int32))
        assert actual.dtype == torch.complex128
        verdict = verify_fft("fft", (4,), torch.int32)
        assert verdict.ok
        assert verdict.output_shape == (4,)
        assert verdict.output_dtype is None
    finally:
        torch.set_default_dtype(old_default)


def test_public_package_exports_complex_verifiers():
    import tensorguard

    assert tensorguard.verify_fft is verify_fft
    assert tensorguard.verify_view_as_real is verify_view_as_real
    assert tensorguard.verify_view_as_complex is verify_view_as_complex
