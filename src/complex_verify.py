"""Shape/dtype rules for complex views and ``torch.fft``.

Complex-valued models often fail at the boundary between real-valued tensors,
complex views, and FFT transforms.  This module checks the shape and dtype
contracts for ``torch.view_as_real``, ``torch.view_as_complex`` and the core
``torch.fft`` family without constructing tensors.  It is intentionally
sound-by-abstention: unknown dtypes or symbolic lengths are never refuted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "ComplexVerdict",
    "verify_fft",
    "verify_view_as_complex",
    "verify_view_as_real",
]


@dataclass(frozen=True)
class ComplexVerdict:
    """Result of one complex-view or FFT contract check."""

    ok: bool
    output_shape: Optional[Shape] = None
    output_dtype: Optional[str] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


_REAL_TO_COMPLEX = {
    "float16": "complex32",
    "float32": "complex64",
    "float64": "complex128",
}
_COMPLEX_TO_REAL = {
    "complex32": "float16",
    "complex64": "float32",
    "complex128": "float64",
}
_FFT_UNSUPPORTED = {"float16", "bfloat16", "complex32"}
_INTEGER_OR_BOOL = {
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
}
_ALIASES = {
    "torch.half": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "float16": "float16",
    "torch.bfloat16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.float": "float32",
    "float": "float32",
    "torch.float32": "float32",
    "float32": "float32",
    "torch.double": "float64",
    "double": "float64",
    "torch.float64": "float64",
    "float64": "float64",
    "torch.chalf": "complex32",
    "chalf": "complex32",
    "torch.complex32": "complex32",
    "complex32": "complex32",
    "torch.cfloat": "complex64",
    "cfloat": "complex64",
    "torch.complex64": "complex64",
    "complex64": "complex64",
    "torch.cdouble": "complex128",
    "cdouble": "complex128",
    "torch.complex128": "complex128",
    "complex128": "complex128",
    "torch.bool": "bool",
    "bool": "bool",
    "torch.uint8": "uint8",
    "uint8": "uint8",
    "torch.int8": "int8",
    "int8": "int8",
    "torch.short": "int16",
    "short": "int16",
    "torch.int16": "int16",
    "int16": "int16",
    "torch.int": "int32",
    "int": "int32",
    "torch.int32": "int32",
    "int32": "int32",
    "torch.long": "int64",
    "long": "int64",
    "torch.int64": "int64",
    "int64": "int64",
}


def _dtype_name(dtype: object) -> Optional[str]:
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    if text in {"", "unknown", "none"}:
        return None
    return _ALIASES.get(text)


def _ok(shape: Sequence[Dim], dtype: Optional[str]) -> ComplexVerdict:
    return ComplexVerdict(True, output_shape=tuple(shape), output_dtype=dtype)


def _fail(kind: str, message: str) -> ComplexVerdict:
    return ComplexVerdict(False, error=message, error_kind=kind)


def _shape_tuple(shape: Sequence[Dim]) -> Shape:
    return tuple(shape)


def _symbolic_dim(label: str, dim: Dim) -> Dim:
    return f"{label}({dim})" if isinstance(dim, str) else dim


def verify_view_as_real(shape: Sequence[Dim], dtype: object) -> ComplexVerdict:
    """Verify ``torch.view_as_real`` over a shape/dtype pair.

    Known complex dtypes append a trailing real/imaginary axis of length ``2``.
    Known non-complex dtypes are refuted.  Unknown dtypes abstain while still
    exposing the conditional output shape if the runtime dtype is complex.
    """

    input_shape = _shape_tuple(shape)
    name = _dtype_name(dtype)
    if name is None:
        return _ok(input_shape + (2,), None)
    if name not in _COMPLEX_TO_REAL:
        return _fail("dtype", f"view_as_real expects a complex dtype, got {name}")
    return _ok(input_shape + (2,), _COMPLEX_TO_REAL[name])


def verify_view_as_complex(shape: Sequence[Dim], dtype: object) -> ComplexVerdict:
    """Verify ``torch.view_as_complex`` over a contiguous shape/dtype view.

    PyTorch accepts exactly float16/float32/float64 input dtypes and requires a
    trailing dimension of size 2.  Stride constraints are intentionally outside
    this shape/dtype-only rule and should be checked by a layout analysis.
    """

    input_shape = _shape_tuple(shape)
    if not input_shape:
        return _fail("rank", "view_as_complex requires rank >= 1 with trailing size 2")
    last = input_shape[-1]
    if isinstance(last, int) and last != 2:
        return _fail("last_dim", f"view_as_complex requires trailing dimension 2, got {last}")

    name = _dtype_name(dtype)
    if name is None:
        return _ok(input_shape[:-1], None)
    if name not in _REAL_TO_COMPLEX:
        return _fail(
            "dtype",
            f"view_as_complex expects float16, float32 or float64 input, got {name}",
        )
    return _ok(input_shape[:-1], _REAL_TO_COMPLEX[name])


def _normalise_dim(rank: int, dim: int) -> Optional[int]:
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    return dim


def _resolve_single_dim(
    rank: int,
    dim: Optional[int],
) -> Tuple[Optional[int], Optional[ComplexVerdict]]:
    if rank == 0:
        return None, _fail("dim", "FFT over the default last dimension requires rank >= 1")
    raw = -1 if dim is None else dim
    if not isinstance(raw, int):
        return None, _fail("dim", f"FFT dimension must be an integer, got {raw!r}")
    resolved = _normalise_dim(rank, raw)
    if resolved is None:
        return None, _fail("dim", f"FFT dimension {raw} is out of range for rank {rank}")
    return resolved, None


def _as_dim_tuple(dim: object) -> Optional[Tuple[int, ...]]:
    if dim is None:
        return None
    if isinstance(dim, int):
        return (dim,)
    if isinstance(dim, tuple):
        return dim
    if isinstance(dim, list):
        return tuple(dim)
    return None


def _as_s_tuple(s: object) -> Optional[Tuple[int, ...]]:
    if s is None:
        return None
    if isinstance(s, tuple):
        return s
    if isinstance(s, list):
        return tuple(s)
    return None


def _resolve_ndims(
    rank: int,
    dim: object,
    s: object,
    *,
    require_nonempty: bool,
) -> Tuple[
    Optional[Tuple[int, ...]],
    Optional[Tuple[int, ...]],
    Optional[ComplexVerdict],
]:
    dims_arg = _as_dim_tuple(dim)
    if dim is not None and dims_arg is None:
        return None, None, _fail(
            "dim",
            f"FFT dimensions must be an int or sequence of ints, got {dim!r}",
        )
    s_arg = _as_s_tuple(s)
    if s is not None and s_arg is None:
        return None, None, _fail("size", f"FFT sizes must be a sequence of ints, got {s!r}")

    if dims_arg is None:
        if s_arg is None:
            dims_arg = tuple(range(rank))
        else:
            if len(s_arg) > rank:
                return None, None, _fail("dim", f"FFT size rank {len(s_arg)} exceeds tensor rank {rank}")
            dims_arg = tuple(range(rank - len(s_arg), rank))
    elif s_arg is not None and len(s_arg) != len(dims_arg):
        return None, None, _fail("size", "FFT sizes and dimensions must have the same length")

    resolved = []
    for raw in dims_arg:
        if not isinstance(raw, int):
            return None, None, _fail("dim", f"FFT dimension must be an integer, got {raw!r}")
        normal = _normalise_dim(rank, raw)
        if normal is None:
            return None, None, _fail("dim", f"FFT dimension {raw} is out of range for rank {rank}")
        resolved.append(normal)
    if len(set(resolved)) != len(resolved):
        return None, None, _fail("dim", "FFT dimensions must be unique")
    if require_nonempty and not resolved:
        return None, None, _fail("dim", "real FFT transforms must include at least one axis")

    if s_arg is not None:
        for size in s_arg:
            if not isinstance(size, int):
                return None, None, _fail("size", f"FFT size must be an integer, got {size!r}")
            if size != -1 and size <= 0:
                return None, None, _fail("size", f"FFT size must be positive or -1, got {size}")

    return tuple(resolved), s_arg, None


def _fft_input_length(dim: Dim, explicit: Optional[int]) -> Optional[Dim]:
    if explicit is not None:
        return explicit
    if isinstance(dim, int):
        if dim <= 0:
            return None
        return dim
    return dim


def _rfft_length(dim: Dim, explicit: Optional[int]) -> Optional[Dim]:
    base = _fft_input_length(dim, explicit)
    if base is None:
        return None
    if isinstance(base, int):
        return base // 2 + 1
    return _symbolic_dim("rfft", base)


def _irfft_length(dim: Dim, explicit: Optional[int]) -> Optional[Dim]:
    if explicit is not None:
        return explicit
    if isinstance(dim, int):
        out = 2 * (dim - 1)
        return out if out > 0 else None
    return _symbolic_dim("irfft", dim)


def _fft_dtype(
    op: str,
    dtype: object,
    *,
    no_transform: bool = False,
) -> Tuple[Optional[str], Optional[ComplexVerdict]]:
    name = _dtype_name(dtype)
    if name is None:
        return None, None
    if no_transform and op in {"fftn", "ifftn"} and name == "complex32":
        return name, None
    if name in _FFT_UNSUPPORTED:
        return None, _fail("dtype", f"{op} does not support dtype {name}")

    if op in {"rfft", "rfftn"}:
        if name in _COMPLEX_TO_REAL:
            return None, _fail("dtype", f"{op} expects a real input dtype, got {name}")
        if name in _REAL_TO_COMPLEX:
            return _REAL_TO_COMPLEX[name], None
        if name in _INTEGER_OR_BOOL:
            return None, None
        return None, None

    if op in {"irfft", "irfftn"}:
        if name in _COMPLEX_TO_REAL:
            return _COMPLEX_TO_REAL[name], None
        if name in _REAL_TO_COMPLEX:
            return name, None
        if name in _INTEGER_OR_BOOL:
            return None, None
        return None, None

    if name in _COMPLEX_TO_REAL:
        return name, None
    if name in _REAL_TO_COMPLEX:
        return _REAL_TO_COMPLEX[name], None
    if name in _INTEGER_OR_BOOL:
        return None, None
    return None, None


def verify_fft(
    op: str,
    shape: Sequence[Dim],
    dtype: object = None,
    *,
    dim: object = None,
    n: Optional[int] = None,
    s: object = None,
) -> ComplexVerdict:
    """Verify a core ``torch.fft`` shape/dtype contract.

    Supported operations are ``fft``, ``ifft``, ``rfft``, ``irfft``, ``fftn``,
    ``ifftn``, ``rfftn`` and ``irfftn``.  The checker mirrors PyTorch's shape
    rules for ``n``/``s``/``dim`` and refutes only dtype/rank/length facts that
    are known to make the real dispatcher raise.
    """

    op = op.lower()
    if op not in {"fft", "ifft", "rfft", "irfft", "fftn", "ifftn", "rfftn", "irfftn"}:
        raise ValueError(f"unsupported FFT op: {op!r}")

    out = list(_shape_tuple(shape))
    rank = len(out)

    if op in {"fft", "ifft", "rfft", "irfft"}:
        out_dtype, dtype_error = _fft_dtype(op, dtype)
        if dtype_error is not None:
            return dtype_error
        if s is not None:
            return _fail("size", "single-axis FFT operations use n=..., not s=...")
        if n is not None and (not isinstance(n, int) or n <= 0):
            return _fail("size", f"FFT n must be a positive integer, got {n!r}")
        axis, error = _resolve_single_dim(rank, dim if dim is not None else None)
        if error is not None:
            return error
        assert axis is not None
        current = out[axis]
        explicit = n
        if op in {"fft", "ifft"}:
            new_len = _fft_input_length(current, explicit)
        elif op == "rfft":
            new_len = _rfft_length(current, explicit)
        else:
            new_len = _irfft_length(current, explicit)
        if new_len is None:
            return _fail("size", f"{op} has non-positive transform length on axis {axis}")
        out[axis] = new_len
        return _ok(out, out_dtype)

    if n is not None:
        return _fail("size", "multi-axis FFT operations use s=..., not n=...")
    require_nonempty = op in {"rfftn", "irfftn"}
    dims, sizes, error = _resolve_ndims(rank, dim, s, require_nonempty=require_nonempty)
    if error is not None:
        return error
    assert dims is not None
    out_dtype, dtype_error = _fft_dtype(op, dtype, no_transform=not dims)
    if dtype_error is not None:
        return dtype_error
    sizes_by_dim = {}
    if sizes is not None:
        sizes_by_dim = {
            axis: None if size == -1 else size
            for axis, size in zip(dims, sizes)
        }

    last_real_axis = dims[-1] if op in {"rfftn", "irfftn"} else None
    for axis in dims:
        current = out[axis]
        explicit = sizes_by_dim.get(axis) if sizes is not None else None
        if axis == last_real_axis and op == "rfftn":
            new_len = _rfft_length(current, explicit)
        elif axis == last_real_axis and op == "irfftn":
            new_len = _irfft_length(current, explicit)
        else:
            new_len = _fft_input_length(current, explicit)
        if new_len is None:
            return _fail("size", f"{op} has non-positive transform length on axis {axis}")
        out[axis] = new_len
    return _ok(out, out_dtype)
