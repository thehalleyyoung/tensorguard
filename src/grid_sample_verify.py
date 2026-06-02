"""Shape/dtype contracts for ``grid_sample`` and ``affine_grid``.

The sampler pair is easy to misuse: ``grid_sample`` has separate 2-D and 3-D
grid layouts, rejects empty input spatial axes but accepts empty output grids,
and requires the input/grid dtypes to agree.  ``affine_grid`` derives the grid
shape from its ``size`` tuple and rejects any non-positive requested output
extent.  These checks model those contracts without constructing tensors.
Symbolic dimensions are carried through and never refuted by arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = ["GridSampleVerdict", "verify_affine_grid", "verify_grid_sample"]

_GRID_SAMPLE_MODES = {"bilinear", "nearest", "bicubic"}
_GRID_SAMPLE_PADDING_MODES = {"zeros", "border", "reflection"}
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_DTYPE_ALIASES = {
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
    "torch.cfloat": "complex64",
    "cfloat": "complex64",
    "torch.complex64": "complex64",
    "complex64": "complex64",
    "torch.cdouble": "complex128",
    "cdouble": "complex128",
    "torch.complex128": "complex128",
    "complex128": "complex128",
}


@dataclass(frozen=True)
class GridSampleVerdict:
    """Result of checking one grid-sampling contract."""

    ok: bool
    output_shape: Optional[Shape] = None
    output_dtype: Optional[str] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    unknown_reason: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _ok(shape: Sequence[Dim], dtype: Optional[str]) -> GridSampleVerdict:
    return GridSampleVerdict(True, output_shape=tuple(shape), output_dtype=dtype)


def _fail(kind: str, message: str) -> GridSampleVerdict:
    return GridSampleVerdict(False, error=message, error_kind=kind)


def _is_dim(value: object) -> bool:
    return type(value) is int or isinstance(value, str)


def _is_known_int(value: Dim) -> bool:
    return type(value) is int


def _shape_tuple(name: str, shape: Sequence[Dim]) -> Tuple[Optional[Shape], Optional[GridSampleVerdict]]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
        return None, _fail("shape", f"{name} shape must be a tuple/list of dimensions")
    out = tuple(shape)
    for dim in out:
        if not _is_dim(dim):
            return None, _fail("shape", f"{name} shape contains invalid dimension {dim!r}")
    return out, None


def _dtype_name(dtype: object) -> Optional[str]:
    if dtype is None:
        return None
    text = str(dtype).strip().lower()
    if text in {"", "unknown", "none"}:
        return None
    return _DTYPE_ALIASES.get(text, text)


def _check_align_corners(value: object, op: str) -> Optional[GridSampleVerdict]:
    if value is not None and type(value) is not bool:
        return _fail("align_corners", f"{op} align_corners must be bool or None, got {value!r}")
    return None


def _check_floating_dtype(name: str, dtype: Optional[str]) -> Optional[GridSampleVerdict]:
    if dtype is not None and dtype not in _FLOAT_DTYPES:
        return _fail("dtype", f"{name} must have floating dtype, got {dtype}")
    return None


def verify_grid_sample(
    input_shape: Sequence[Dim],
    grid_shape: Sequence[Dim],
    *,
    input_dtype: object = None,
    grid_dtype: object = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> GridSampleVerdict:
    """Verify one ``torch.nn.functional.grid_sample`` call.

    Supported concrete contracts match PyTorch's 4-D and 5-D sampler forms:
    ``(N,C,H,W)`` with grid ``(N,H_out,W_out,2)`` and
    ``(N,C,D,H,W)`` with grid ``(N,D_out,H_out,W_out,3)``.  Known symbolic
    uncertainty is preserved rather than refuted.
    """

    x, err = _shape_tuple("input", input_shape)
    if err is not None:
        return err
    g, err = _shape_tuple("grid", grid_shape)
    if err is not None:
        return err
    assert x is not None and g is not None

    if mode not in _GRID_SAMPLE_MODES:
        return _fail("mode", f"grid_sample mode must be one of {sorted(_GRID_SAMPLE_MODES)}, got {mode!r}")
    if padding_mode not in _GRID_SAMPLE_PADDING_MODES:
        return _fail(
            "padding_mode",
            f"grid_sample padding_mode must be one of {sorted(_GRID_SAMPLE_PADDING_MODES)}, got {padding_mode!r}",
        )
    err = _check_align_corners(align_corners, "grid_sample")
    if err is not None:
        return err

    if len(x) not in (4, 5):
        return _fail("input_rank", f"grid_sample expects a 4-D or 5-D input, got rank {len(x)}")
    if len(g) != len(x):
        return _fail("grid_rank", f"grid rank {len(g)} must match input rank {len(x)}")
    if mode == "bicubic" and len(x) != 4:
        return _fail("mode_rank", "grid_sample bicubic interpolation only supports 4-D input")

    for index, dim in enumerate(x[2:], start=2):
        if _is_known_int(dim) and dim <= 0:
            return _fail(
                "input_spatial",
                f"grid_sample input spatial dimension {index} must be non-empty, got {dim}",
            )

    expected_grid_last = len(x) - 2
    actual_grid_last = g[-1]
    if _is_known_int(actual_grid_last) and actual_grid_last != expected_grid_last:
        return _fail(
            "grid_last_dim",
            f"grid_sample expects grid last dimension {expected_grid_last}, got {actual_grid_last}",
        )

    input_batch, grid_batch = x[0], g[0]
    if _is_known_int(input_batch) and _is_known_int(grid_batch) and input_batch != grid_batch:
        return _fail(
            "batch",
            f"grid_sample input/grid batch sizes must match, got {input_batch} and {grid_batch}",
        )

    in_dtype = _dtype_name(input_dtype)
    gr_dtype = _dtype_name(grid_dtype)
    err = _check_floating_dtype("input", in_dtype)
    if err is not None:
        return err
    err = _check_floating_dtype("grid", gr_dtype)
    if err is not None:
        return err
    if in_dtype is not None and gr_dtype is not None and in_dtype != gr_dtype:
        return _fail("dtype_mismatch", f"input dtype {in_dtype} must match grid dtype {gr_dtype}")

    out = (x[0], x[1]) + g[1:-1]
    return _ok(out, in_dtype)


def verify_affine_grid(
    theta_shape: Sequence[Dim],
    size: Sequence[Dim],
    *,
    theta_dtype: object = None,
    align_corners: Optional[bool] = None,
) -> GridSampleVerdict:
    """Verify one ``torch.nn.functional.affine_grid`` call.

    ``size`` follows PyTorch's output-tensor size convention:
    ``(N,C,H,W)`` for 2-D grids and ``(N,C,D,H,W)`` for 3-D grids.  The returned
    grid drops ``C`` and appends coordinate size 2 or 3.
    """

    theta, err = _shape_tuple("theta", theta_shape)
    if err is not None:
        return err
    out_size, err = _shape_tuple("size", size)
    if err is not None:
        return err
    assert theta is not None and out_size is not None

    err = _check_align_corners(align_corners, "affine_grid")
    if err is not None:
        return err

    if len(out_size) not in (4, 5):
        return _fail("size_rank", f"affine_grid only supports 4-D and 5-D sizes, got rank {len(out_size)}")
    for index, dim in enumerate(out_size):
        if _is_known_int(dim) and dim <= 0:
            return _fail(
                "size_positive",
                f"affine_grid size dimension {index} must be positive, got {dim}",
            )

    if len(theta) != 3:
        return _fail("theta_rank", f"affine_grid expects theta rank 3, got rank {len(theta)}")
    if _is_known_int(theta[0]) and theta[0] <= 0:
        return _fail("theta_batch", f"affine_grid theta batch must be positive, got {theta[0]}")

    expected_rows = len(out_size) - 2
    expected_cols = expected_rows + 1
    if _is_known_int(theta[1]) and theta[1] != expected_rows:
        return _fail("theta_matrix", f"theta second dimension must be {expected_rows}, got {theta[1]}")
    if _is_known_int(theta[2]) and theta[2] != expected_cols:
        return _fail("theta_matrix", f"theta third dimension must be {expected_cols}, got {theta[2]}")

    theta_batch, size_batch = theta[0], out_size[0]
    if _is_known_int(theta_batch) and _is_known_int(size_batch) and theta_batch != size_batch:
        return _fail(
            "batch",
            f"theta batch size {theta_batch} must match requested output batch {size_batch}",
        )

    dtype = _dtype_name(theta_dtype)
    err = _check_floating_dtype("theta", dtype)
    if err is not None:
        return err

    coord_dim = len(out_size) - 2
    output_shape = (out_size[0],) + out_size[2:] + (coord_dim,)
    return _ok(output_shape, dtype)
