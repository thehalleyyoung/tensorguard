"""Static shape contracts for the core ``torch.linalg`` family.

The checks in this module model dispatcher-visible shape behavior without
constructing tensors.  They intentionally refute only facts that are known from
shapes alone: value-dependent numerical failures such as singular matrices or
non-positive-definite Cholesky inputs are outside this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "Dim",
    "Shape",
    "LinalgVerdict",
    "verify_linalg",
    "verify_linalg_cholesky",
    "verify_linalg_eig",
    "verify_linalg_inv",
    "verify_linalg_qr",
    "verify_linalg_solve",
    "verify_linalg_svd",
]


@dataclass(frozen=True)
class LinalgVerdict:
    """Result of one ``torch.linalg`` shape-contract check."""

    ok: bool
    output_shapes: Tuple[Tuple[str, Shape], ...] = ()
    error: Optional[str] = None
    error_kind: Optional[str] = None
    unknown_reason: Optional[str] = None

    @property
    def output_shape(self) -> Optional[Shape]:
        """Return the sole output shape for single-output ops."""

        if len(self.output_shapes) == 1:
            return self.output_shapes[0][1]
        return None

    def shape(self, name: str) -> Optional[Shape]:
        """Return a named output shape, e.g. ``"U"`` for SVD."""

        for output_name, output_shape in self.output_shapes:
            if output_name == name:
                return output_shape
        return None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _shape(value: Sequence[Dim]) -> Shape:
    return tuple(value)


def _fail(kind: str, message: str) -> LinalgVerdict:
    return LinalgVerdict(False, error=message, error_kind=kind)


def _ok(*outputs: Tuple[str, Shape], unknown_reason: Optional[str] = None) -> LinalgVerdict:
    return LinalgVerdict(True, output_shapes=tuple(outputs), unknown_reason=unknown_reason)


def _unknown(reason: str) -> LinalgVerdict:
    return LinalgVerdict(True, unknown_reason=reason)


def _is_int_dim(value: object) -> bool:
    return type(value) is int


def _known_negative(value: Dim) -> bool:
    return _is_int_dim(value) and value < 0


def _known_mismatch(left: Dim, right: Dim) -> bool:
    return _is_int_dim(left) and _is_int_dim(right) and left != right


def _has_symbolic(shape: Sequence[Dim]) -> bool:
    return any(not _is_int_dim(dim) for dim in shape)


def _check_nonnegative(shape: Shape, label: str) -> Optional[LinalgVerdict]:
    for dim in shape:
        if _known_negative(dim):
            return _fail("negative_dim", f"{label} contains negative dimension {dim}")
    return None


def _square_matrix(shape: Shape, op: str) -> Tuple[Optional[Shape], Optional[Dim], Optional[LinalgVerdict], Optional[str]]:
    err = _check_nonnegative(shape, "input")
    if err is not None:
        return None, None, err, None
    if len(shape) < 2:
        return None, None, _fail("rank", f"{op} expects rank >= 2, got rank {len(shape)}"), None
    rows, cols = shape[-2], shape[-1]
    if _known_mismatch(rows, cols):
        return None, None, _fail("square", f"{op} expects square trailing dims, got {rows}x{cols}"), None
    reason = None
    if rows != cols and (_has_symbolic((rows, cols))):
        reason = f"{op} requires the symbolic trailing dims {rows!r} and {cols!r} to be equal"
    return shape[:-2], rows, None, reason


def _min_dim(left: Dim, right: Dim, label: str) -> Dim:
    if _is_int_dim(left) and _is_int_dim(right):
        return min(left, right)
    if left == right:
        return left
    return f"{label}({left},{right})"


def _broadcast_dim(left: Dim, right: Dim) -> Optional[Dim]:
    if _is_int_dim(left) and _is_int_dim(right):
        if left == right:
            return left
        if left == 1:
            return right
        if right == 1:
            return left
        return None
    if left == 1:
        return right
    if right == 1:
        return left
    if left == right:
        return left
    return left if not _is_int_dim(left) else right


def _broadcast_shapes(left: Shape, right: Shape) -> Optional[Shape]:
    out = []
    rank = max(len(left), len(right))
    for offset in range(1, rank + 1):
        a: Dim = 1
        b: Dim = 1
        if offset <= len(left):
            a = left[-offset]
        if offset <= len(right):
            b = right[-offset]
        dim = _broadcast_dim(a, b)
        if dim is None:
            return None
        out.append(dim)
    out.reverse()
    return tuple(out)


def _batch_exact_status(left: Shape, right: Shape) -> Optional[bool]:
    """Return True/False when exact batch equality is known, None if symbolic."""

    if len(left) != len(right):
        return False
    if left == right:
        return True
    for a, b in zip(left, right):
        if _known_mismatch(a, b):
            return False
    if _has_symbolic(left) or _has_symbolic(right):
        return None
    return False


def verify_linalg_inv(shape: Sequence[Dim]) -> LinalgVerdict:
    """Verify ``torch.linalg.inv(A)`` shape: ``(..., n, n) -> (..., n, n)``."""

    input_shape = _shape(shape)
    _, _, err, reason = _square_matrix(input_shape, "torch.linalg.inv")
    if err is not None:
        return err
    return _ok(("output", input_shape), unknown_reason=reason)


def verify_linalg_cholesky(shape: Sequence[Dim], *, upper: bool = False) -> LinalgVerdict:
    """Verify ``torch.linalg.cholesky(A)`` shape.

    Positive-definiteness is value-dependent and therefore not refuted here.
    """

    if type(upper) is not bool:
        return _fail("argument", f"upper must be bool, got {upper!r}")
    input_shape = _shape(shape)
    _, _, err, reason = _square_matrix(input_shape, "torch.linalg.cholesky")
    if err is not None:
        return err
    return _ok(("output", input_shape), unknown_reason=reason)


def verify_linalg_eig(shape: Sequence[Dim]) -> LinalgVerdict:
    """Verify ``torch.linalg.eig(A)`` shapes."""

    input_shape = _shape(shape)
    batch, n, err, reason = _square_matrix(input_shape, "torch.linalg.eig")
    if err is not None:
        return err
    assert batch is not None and n is not None
    return _ok(
        ("eigenvalues", batch + (n,)),
        ("eigenvectors", input_shape),
        unknown_reason=reason,
    )


def verify_linalg_qr(shape: Sequence[Dim], *, mode: str = "reduced") -> LinalgVerdict:
    """Verify ``torch.linalg.qr(A, mode=...)`` output shapes."""

    if mode not in {"reduced", "complete", "r"}:
        return _fail(
            "mode",
            f"qr mode must be one of 'reduced', 'complete', or 'r', got {mode!r}",
        )
    input_shape = _shape(shape)
    err = _check_nonnegative(input_shape, "input")
    if err is not None:
        return err
    if len(input_shape) < 2:
        return _fail("rank", f"torch.linalg.qr expects rank >= 2, got rank {len(input_shape)}")
    batch = input_shape[:-2]
    m, n = input_shape[-2], input_shape[-1]
    k = _min_dim(m, n, "min")

    if mode == "complete":
        return _ok(("Q", batch + (m, m)), ("R", batch + (m, n)))
    if mode == "r":
        return _ok(("Q", (0,)), ("R", batch + (k, n)))
    return _ok(("Q", batch + (m, k)), ("R", batch + (k, n)))


def verify_linalg_svd(shape: Sequence[Dim], *, full_matrices: bool = True) -> LinalgVerdict:
    """Verify ``torch.linalg.svd(A, full_matrices=...)`` output shapes."""

    if type(full_matrices) is not bool:
        return _fail("argument", f"full_matrices must be bool, got {full_matrices!r}")
    input_shape = _shape(shape)
    err = _check_nonnegative(input_shape, "input")
    if err is not None:
        return err
    if len(input_shape) < 2:
        return _fail("rank", f"torch.linalg.svd expects rank >= 2, got rank {len(input_shape)}")
    batch = input_shape[:-2]
    m, n = input_shape[-2], input_shape[-1]
    k = _min_dim(m, n, "min")

    if full_matrices:
        return _ok(("U", batch + (m, m)), ("S", batch + (k,)), ("Vh", batch + (n, n)))
    return _ok(("U", batch + (m, k)), ("S", batch + (k,)), ("Vh", batch + (k, n)))


def verify_linalg_solve(
    a_shape: Sequence[Dim],
    b_shape: Sequence[Dim],
    *,
    left: bool = True,
) -> LinalgVerdict:
    """Verify ``torch.linalg.solve(A, B, left=...)`` output shape.

    PyTorch treats some RHS shapes as vector systems only when the RHS batch
    exactly equals ``A``'s batch (plus the unbatched ``B.ndim == 1`` case); those
    vector batches do not broadcast.  Matrix RHS batches do broadcast.
    """

    if type(left) is not bool:
        return _fail("argument", f"left must be bool, got {left!r}")
    a = _shape(a_shape)
    b = _shape(b_shape)
    a_batch, n, err, reason = _square_matrix(a, "torch.linalg.solve")
    if err is not None:
        return err
    assert a_batch is not None and n is not None
    err = _check_nonnegative(b, "rhs")
    if err is not None:
        return err
    if not b:
        return _fail("rhs_rank", "torch.linalg.solve expects RHS rank >= 1")

    if left:
        if len(b) == 1:
            if _known_mismatch(b[0], n):
                return _fail("rhs_dim", f"vector RHS length {b[0]} does not match matrix dim {n}")
            return _ok(("output", a_batch + (n,)), unknown_reason=reason)

        if len(b) == len(a) - 1:
            status = _batch_exact_status(a_batch, b[:-1])
            if status is True and not _known_mismatch(b[-1], n):
                return _ok(("output", a_batch + (n,)), unknown_reason=reason)
            if status is None and not _known_mismatch(b[-1], n):
                return _unknown(
                    "symbolic RHS shape could be PyTorch's non-broadcasting vector branch "
                    "or the matrix-RHS branch"
                )

        rhs_rows = b[-2]
        rhs_cols = b[-1]
        if _known_mismatch(rhs_rows, n):
            return _fail("rhs_dim", f"matrix RHS row count {rhs_rows} does not match matrix dim {n}")
        batch = _broadcast_shapes(a_batch, b[:-2])
        if batch is None:
            return _fail("batch_broadcast", f"A batch {a_batch} does not broadcast with B batch {b[:-2]}")
        return _ok(("output", batch + (rhs_rows, rhs_cols)), unknown_reason=reason)

    if len(b) == 1:
        return _fail("rhs_vector", "left=False expects a matrix RHS with shape (..., k, n), not a vector")

    if len(b) == len(a) - 1:
        status = _batch_exact_status(a_batch, b[:-1])
        if status is True and not _known_mismatch(b[-1], n):
            return _fail(
                "rhs_vector",
                "left=False routes exact-batch (..., n) RHS shapes through PyTorch's invalid vector branch",
            )
        if status is None and not _known_mismatch(b[-1], n):
            return _unknown(
                "symbolic RHS shape could be PyTorch's invalid left=False vector branch "
                "or the matrix-RHS branch"
            )

    rhs_cols = b[-1]
    rhs_rows = b[-2]
    if _known_mismatch(rhs_cols, n):
        return _fail("rhs_dim", f"matrix RHS trailing dim {rhs_cols} does not match matrix dim {n}")
    batch = _broadcast_shapes(a_batch, b[:-2])
    if batch is None:
        return _fail("batch_broadcast", f"A batch {a_batch} does not broadcast with B batch {b[:-2]}")
    return _ok(("output", batch + (rhs_rows, rhs_cols)), unknown_reason=reason)


def verify_linalg(op: str, *shapes: Sequence[Dim], **kwargs: object) -> LinalgVerdict:
    """Dispatch a supported ``torch.linalg`` shape contract by operation name."""

    name = op.replace("torch.linalg.", "").replace("linalg.", "").lower()
    if name == "inv":
        if len(shapes) != 1:
            raise TypeError("verify_linalg('inv', ...) expects one shape")
        return verify_linalg_inv(shapes[0])
    if name == "cholesky":
        if len(shapes) != 1:
            raise TypeError("verify_linalg('cholesky', ...) expects one shape")
        return verify_linalg_cholesky(shapes[0], upper=kwargs.get("upper", False))
    if name == "eig":
        if len(shapes) != 1:
            raise TypeError("verify_linalg('eig', ...) expects one shape")
        return verify_linalg_eig(shapes[0])
    if name == "qr":
        if len(shapes) != 1:
            raise TypeError("verify_linalg('qr', ...) expects one shape")
        return verify_linalg_qr(shapes[0], mode=kwargs.get("mode", "reduced"))
    if name == "svd":
        if len(shapes) != 1:
            raise TypeError("verify_linalg('svd', ...) expects one shape")
        return verify_linalg_svd(shapes[0], full_matrices=kwargs.get("full_matrices", True))
    if name == "solve":
        if len(shapes) != 2:
            raise TypeError("verify_linalg('solve', ...) expects A and B shapes")
        return verify_linalg_solve(shapes[0], shapes[1], left=kwargs.get("left", True))
    raise ValueError(
        f"unsupported torch.linalg op {op!r}; supported: inv, cholesky, eig, qr, svd, solve"
    )
