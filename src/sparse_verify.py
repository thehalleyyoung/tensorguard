"""Static layout checks for ``torch.sparse`` tensor constructors.

PyTorch's compressed sparse constructors are deliberately permissive by
default; the stricter ``check_invariants=True`` path is the contract users need
when they want malformed CSR/CSC/BSR/BSC metadata caught at construction time.
This module models that validated layout contract without allocating tensors.

For compressed layouts we additionally flag value-dense tails that disagree
with the requested tensor size.  PyTorch 2.9 may accept those tensors at
construction even under invariant checking, but the first dense materialization
fails; TensorGuard treats them as unusable sparse layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

Dim = Union[int, str]
Shape = Tuple[Dim, ...]

__all__ = [
    "SparseLayoutSpec",
    "SparseVerdict",
    "verify_sparse_addmm",
    "verify_sparse_bsc",
    "verify_sparse_bsr",
    "verify_sparse_coalesce",
    "verify_sparse_compressed",
    "verify_sparse_coo",
    "verify_sparse_csc",
    "verify_sparse_csr",
    "verify_sparse_layout_conversion",
    "verify_sparse_mm",
    "verify_sparse_sampled_addmm",
    "verify_sparse_softmax",
    "verify_sparse_to_dense",
]


@dataclass(frozen=True)
class SparseLayoutSpec:
    """Resolved static layout facts for a sparse tensor."""

    layout: str
    shape: Shape
    sparse_shape: Shape
    dense_shape: Shape
    batch_shape: Shape = ()
    nnz: Optional[Dim] = None
    sparse_dim: Optional[int] = None
    blocksize: Optional[Tuple[Dim, Dim]] = None


@dataclass(frozen=True)
class SparseVerdict:
    """Result of one sparse-layout contract check."""

    ok: bool
    spec: Optional[SparseLayoutSpec] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    unknown_reason: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _shape(value: Sequence[Dim]) -> Shape:
    return tuple(value)


def _fail(kind: str, message: str) -> SparseVerdict:
    return SparseVerdict(False, error=message, error_kind=kind)


def _ok(spec: SparseLayoutSpec, unknown_reason: Optional[str] = None) -> SparseVerdict:
    return SparseVerdict(True, spec=spec, unknown_reason=unknown_reason)


def _is_int_dim(value: object) -> bool:
    return type(value) is int


def _known_negative(value: Dim) -> bool:
    return _is_int_dim(value) and value < 0


def _known_mismatch(left: Dim, right: Dim) -> bool:
    return _is_int_dim(left) and _is_int_dim(right) and left != right


def _check_nonnegative(shape: Shape, label: str) -> Optional[SparseVerdict]:
    for dim in shape:
        if _known_negative(dim):
            return _fail("negative_dim", f"{label} contains negative dimension {dim}")
    return None


def _check_all_nonnegative(*items: Tuple[str, Shape]) -> Optional[SparseVerdict]:
    for label, shape in items:
        err = _check_nonnegative(shape, label)
        if err is not None:
            return err
    return None


def _compare_shape(
    left: Shape,
    right: Shape,
    *,
    kind: str,
    left_name: str,
    right_name: str,
) -> Optional[SparseVerdict]:
    if len(left) != len(right):
        return _fail(
            kind,
            f"{left_name} rank {len(left)} does not match {right_name} rank {len(right)}",
        )
    for index, (a, b) in enumerate(zip(left, right)):
        if _known_mismatch(a, b):
            return _fail(
                kind,
                f"{left_name}[{index}]={a} does not match {right_name}[{index}]={b}",
            )
    return None


def _compare_dim(
    left: Dim,
    right: Dim,
    *,
    kind: str,
    left_name: str,
    right_name: str,
) -> Optional[SparseVerdict]:
    if _known_mismatch(left, right):
        return _fail(kind, f"{left_name}={left} does not match {right_name}={right}")
    return None


def _canonical_layout(layout: str) -> Optional[str]:
    normalized = str(layout).replace("torch.", "").replace("sparse_", "").lower()
    aliases = {
        "csr": "csr",
        "csc": "csc",
        "bsr": "bsr",
        "bsc": "bsc",
    }
    return aliases.get(normalized)


def _dense_spec(shape: Shape) -> SparseLayoutSpec:
    return SparseLayoutSpec(
        layout="dense",
        shape=shape,
        sparse_shape=(),
        dense_shape=shape,
    )


def _sparse_like_spec(source: SparseLayoutSpec, shape: Optional[Shape] = None) -> SparseLayoutSpec:
    out_shape = source.shape if shape is None else shape
    if source.layout == "coo" and shape is None:
        return SparseLayoutSpec(
            layout=source.layout,
            shape=source.shape,
            sparse_shape=source.sparse_shape,
            dense_shape=source.dense_shape,
            batch_shape=source.batch_shape,
            nnz=source.nnz,
            sparse_dim=source.sparse_dim,
            blocksize=source.blocksize,
        )
    if source.layout in {"csr", "csc", "bsr", "bsc"} and len(out_shape) >= 2:
        return SparseLayoutSpec(
            layout=source.layout,
            shape=out_shape,
            sparse_shape=out_shape[-2:],
            dense_shape=(),
            batch_shape=out_shape[:-2],
            nnz=source.nnz,
            blocksize=source.blocksize,
        )
    return SparseLayoutSpec(
        layout=source.layout,
        shape=out_shape,
        sparse_shape=out_shape,
        dense_shape=source.dense_shape,
        batch_shape=source.batch_shape,
        nnz=source.nnz,
        sparse_dim=source.sparse_dim,
        blocksize=source.blocksize,
    )


def _compare_shape_with_unknown(
    left: Shape,
    right: Shape,
    *,
    kind: str,
    left_name: str,
    right_name: str,
) -> Tuple[Optional[SparseVerdict], bool]:
    if len(left) != len(right):
        return (
            _fail(
                kind,
                f"{left_name} rank {len(left)} does not match {right_name} rank {len(right)}",
            ),
            False,
        )
    symbolic = False
    for index, (a, b) in enumerate(zip(left, right)):
        if _known_mismatch(a, b):
            return (
                _fail(
                    kind,
                    f"{left_name}[{index}]={a} does not match {right_name}[{index}]={b}",
                ),
                False,
            )
        if a != b and (not _is_int_dim(a) or not _is_int_dim(b)):
            symbolic = True
    return None, symbolic


def _compare_dim_with_unknown(
    left: Dim,
    right: Dim,
    *,
    kind: str,
    left_name: str,
    right_name: str,
) -> Tuple[Optional[SparseVerdict], bool]:
    if _known_mismatch(left, right):
        return _fail(kind, f"{left_name}={left} does not match {right_name}={right}"), False
    if left != right and (not _is_int_dim(left) or not _is_int_dim(right)):
        return None, True
    return None, False


def _broadcastable_to(source: Shape, target: Shape) -> Tuple[Optional[SparseVerdict], bool]:
    if len(source) > len(target):
        return (
            _fail(
                "broadcast",
                f"input rank {len(source)} cannot broadcast to target rank {len(target)}",
            ),
            False,
        )
    symbolic = False
    offset = len(target) - len(source)
    for axis, dim in enumerate(source):
        target_dim = target[offset + axis]
        if _is_int_dim(dim):
            if dim == 1:
                continue
            if _is_int_dim(target_dim):
                if dim != target_dim:
                    return (
                        _fail(
                            "broadcast",
                            f"input dim {dim} cannot broadcast to target dim {target_dim} at axis {axis}",
                        ),
                        False,
                    )
            elif dim != target_dim:
                symbolic = True
            continue
        if dim != target_dim:
            symbolic = True
    return None, symbolic


def _normalize_dim(dim: int, rank: int) -> Optional[int]:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        return None
    return normalized


def verify_sparse_coo(
    indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse_coo_tensor(indices, values, size)`` shapes.

    The check mirrors PyTorch's constructor-visible shape contract: ``indices``
    has shape ``(sparse_dim, nnz)``, ``values`` has shape
    ``(nnz, *dense_shape)``, and ``size`` has rank
    ``sparse_dim + len(dense_shape)``.
    """

    indices = _shape(indices_shape)
    values = _shape(values_shape)
    tensor_size = _shape(size)

    err = _check_all_nonnegative(
        ("indices_shape", indices),
        ("values_shape", values),
        ("size", tensor_size),
    )
    if err is not None:
        return err

    if len(indices) != 2:
        return _fail("indices_rank", f"COO indices must be rank 2, got rank {len(indices)}")
    if not values:
        return _fail("values_rank", "COO values must have rank >= 1 with nnz as the first dimension")

    sparse_dim, nnz = indices
    if not _is_int_dim(sparse_dim):
        spec = SparseLayoutSpec(
            layout="coo",
            shape=tensor_size,
            sparse_shape=tensor_size,
            dense_shape=(),
            nnz=nnz,
            sparse_dim=None,
        )
        return _ok(spec, "symbolic sparse_dim: cannot split sparse and dense axes")

    if sparse_dim > len(tensor_size):
        return _fail(
            "size_rank",
            f"COO sparse_dim {sparse_dim} exceeds tensor rank {len(tensor_size)}",
        )

    dense_shape = tensor_size[sparse_dim:]
    if len(values) != 1 + len(dense_shape):
        return _fail(
            "values_rank",
            "COO values rank must be 1 + dense_dim "
            f"(expected {1 + len(dense_shape)}, got {len(values)})",
        )

    err = _compare_dim(values[0], nnz, kind="nnz", left_name="values nnz", right_name="indices nnz")
    if err is not None:
        return err
    err = _compare_shape(
        values[1:],
        dense_shape,
        kind="dense_shape",
        left_name="values dense tail",
        right_name="size dense tail",
    )
    if err is not None:
        return err

    return _ok(
        SparseLayoutSpec(
            layout="coo",
            shape=tensor_size,
            sparse_shape=tensor_size[:sparse_dim],
            dense_shape=dense_shape,
            nnz=nnz,
            sparse_dim=sparse_dim,
        )
    )


def _axis_plus_one(axis_extent: Dim, compressed_length: Dim, label: str) -> Optional[SparseVerdict]:
    if _is_int_dim(axis_extent) and _is_int_dim(compressed_length):
        expected = axis_extent + 1
        if compressed_length != expected:
            return _fail(
                "compressed_indices",
                f"{label} length must be {expected}, got {compressed_length}",
            )
    return None


def _block_axis_plus_one(
    axis_extent: Dim,
    block_extent: Dim,
    compressed_length: Dim,
    label: str,
) -> Tuple[Optional[SparseVerdict], bool]:
    """Return an error plus whether symbolic arithmetic was encountered."""

    if _is_int_dim(block_extent):
        if block_extent <= 0:
            return _fail("blocksize", f"{label} block size must be positive, got {block_extent}"), False
    else:
        return None, True

    if not _is_int_dim(axis_extent):
        return None, True

    if axis_extent % block_extent != 0:
        return (
            _fail(
                "block_divisibility",
                f"{label} dimension {axis_extent} is not divisible by block size {block_extent}",
            ),
            False,
        )

    if _is_int_dim(compressed_length):
        expected = axis_extent // block_extent + 1
        if compressed_length != expected:
            return (
                _fail(
                    "compressed_indices",
                    f"{label} compressed length must be {expected}, got {compressed_length}",
                ),
                False,
            )
    else:
        return None, True
    return None, False


def verify_sparse_compressed(
    layout: str,
    compressed_indices_shape: Sequence[Dim],
    plain_indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify CSR/CSC/BSR/BSC sparse-compressed layout metadata.

    The contract matches PyTorch's ``check_invariants=True`` shape checks and
    adds an explicit ``unusable_dense`` error when the values' dense tail cannot
    materialize to the requested tensor shape.
    """

    canonical = _canonical_layout(layout)
    if canonical is None:
        return _fail("layout", f"unsupported sparse layout {layout!r}; expected csr/csc/bsr/bsc")

    compressed = _shape(compressed_indices_shape)
    plain = _shape(plain_indices_shape)
    values = _shape(values_shape)
    tensor_size = _shape(size)

    err = _check_all_nonnegative(
        ("compressed_indices_shape", compressed),
        ("plain_indices_shape", plain),
        ("values_shape", values),
        ("size", tensor_size),
    )
    if err is not None:
        return err

    if not compressed:
        return _fail("compressed_rank", "compressed indices must have rank >= 1")
    batch_rank = len(compressed) - 1
    if len(plain) != batch_rank + 1:
        return _fail(
            "plain_rank",
            f"plain indices rank must be batch_rank + 1 ({batch_rank + 1}), got {len(plain)}",
        )
    if len(tensor_size) < batch_rank + 2:
        return _fail(
            "size_rank",
            f"{canonical.upper()} size rank must be at least batch_rank + 2 ({batch_rank + 2}), "
            f"got {len(tensor_size)}",
        )

    blocked = canonical in {"bsr", "bsc"}
    min_values_rank = batch_rank + (3 if blocked else 1)
    if len(values) < min_values_rank:
        if blocked:
            return _fail(
                "values_rank",
                "blocked sparse values must have shape (*batch, nblocks, block_rows, block_cols, *dense)",
            )
        return _fail("values_rank", "compressed sparse values must have shape (*batch, nnz, *dense)")

    batch_shape = tensor_size[:batch_rank]
    for candidate, label in (
        (compressed[:-1], "compressed batch"),
        (plain[:-1], "plain-index batch"),
        (values[:batch_rank], "values batch"),
    ):
        err = _compare_shape(
            candidate,
            batch_shape,
            kind="batch_shape",
            left_name=label,
            right_name="size batch",
        )
        if err is not None:
            return err

    nnz = values[batch_rank]
    err = _compare_dim(
        plain[-1],
        nnz,
        kind="nnz",
        left_name="plain indices nnz",
        right_name="values nnz",
    )
    if err is not None:
        return err

    base_shape = tensor_size[batch_rank : batch_rank + 2]
    if blocked:
        block_rows = values[batch_rank + 1]
        block_cols = values[batch_rank + 2]
        dense_tail = values[batch_rank + 3 :]
        blocksize: Optional[Tuple[Dim, Dim]] = (block_rows, block_cols)
    else:
        dense_tail = values[batch_rank + 1 :]
        blocksize = None

    expected_rank = batch_rank + 2 + len(dense_tail)
    if len(tensor_size) != expected_rank:
        return _fail(
            "size_rank",
            f"{canonical.upper()} size rank must be batch + 2 + dense ({expected_rank}), "
            f"got {len(tensor_size)}",
        )

    err = _compare_shape(
        dense_tail,
        tensor_size[batch_rank + 2 :],
        kind="unusable_dense",
        left_name="values dense tail",
        right_name="size dense tail",
    )
    if err is not None:
        return err

    symbolic = False
    compressed_axis = 0 if canonical in {"csr", "bsr"} else 1
    compressed_label = "row" if compressed_axis == 0 else "column"
    if blocked:
        assert blocksize is not None
        block_extent = blocksize[0] if compressed_axis == 0 else blocksize[1]
        err, symbolic = _block_axis_plus_one(
            base_shape[compressed_axis],
            block_extent,
            compressed[-1],
            compressed_label,
        )
        if err is not None:
            return err

        other_axis = 1 - compressed_axis
        other_extent = blocksize[other_axis]
        if _is_int_dim(other_extent):
            if other_extent <= 0:
                return _fail(
                    "blocksize",
                    f"{'column' if other_axis == 1 else 'row'} block size must be positive, got {other_extent}",
                )
            if _is_int_dim(base_shape[other_axis]) and base_shape[other_axis] % other_extent != 0:
                return _fail(
                    "block_divisibility",
                    f"{'column' if other_axis == 1 else 'row'} dimension {base_shape[other_axis]} "
                    f"is not divisible by block size {other_extent}",
                )
            if not _is_int_dim(base_shape[other_axis]):
                symbolic = True
        else:
            symbolic = True
    else:
        err = _axis_plus_one(base_shape[compressed_axis], compressed[-1], compressed_label)
        if err is not None:
            return err
        symbolic = (
            not _is_int_dim(base_shape[compressed_axis])
            or not _is_int_dim(compressed[-1])
        )

    unknown_reason = "symbolic sparse dimension: block/axis length arithmetic not fully checked" if symbolic else None
    return _ok(
        SparseLayoutSpec(
            layout=canonical,
            shape=tensor_size,
            sparse_shape=base_shape,
            dense_shape=tensor_size[batch_rank + 2 :],
            batch_shape=batch_shape,
            nnz=nnz,
            blocksize=blocksize,
        ),
        unknown_reason=unknown_reason,
    )


def verify_sparse_csr(
    crow_indices_shape: Sequence[Dim],
    col_indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse_csr_tensor`` layout metadata."""

    return verify_sparse_compressed("csr", crow_indices_shape, col_indices_shape, values_shape, size)


def verify_sparse_csc(
    ccol_indices_shape: Sequence[Dim],
    row_indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse_csc_tensor`` layout metadata."""

    return verify_sparse_compressed("csc", ccol_indices_shape, row_indices_shape, values_shape, size)


def verify_sparse_bsr(
    crow_indices_shape: Sequence[Dim],
    col_indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse_bsr_tensor`` layout metadata."""

    return verify_sparse_compressed("bsr", crow_indices_shape, col_indices_shape, values_shape, size)


def verify_sparse_bsc(
    ccol_indices_shape: Sequence[Dim],
    row_indices_shape: Sequence[Dim],
    values_shape: Sequence[Dim],
    size: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse_bsc_tensor`` layout metadata."""

    return verify_sparse_compressed("bsc", ccol_indices_shape, row_indices_shape, values_shape, size)


def verify_sparse_mm(mat1: SparseLayoutSpec, mat2_shape: Sequence[Dim]) -> SparseVerdict:
    """Verify sparse-dense ``torch.sparse.mm(mat1, mat2)`` shape rules.

    The supported success path mirrors CPU PyTorch 2-D sparse-dense kernels for
    COO/CSR/CSC inputs. Block compressed BSR/BSC kernels are deliberately not
    accepted here because the tested PyTorch CPU build raises before producing a
    dense result.
    """

    mat2 = _shape(mat2_shape)
    err = _check_all_nonnegative(("mat1", mat1.shape), ("mat2", mat2))
    if err is not None:
        return err
    if mat1.layout not in {"coo", "csr", "csc"}:
        return _fail(
            "layout",
            f"torch.sparse.mm shape parity is modeled for COO/CSR/CSC, got {mat1.layout!r}",
        )
    if len(mat1.shape) != 2 or len(mat2) != 2:
        return _fail("rank", "torch.sparse.mm requires a rank-2 sparse matrix and a rank-2 dense matrix")

    err, symbolic = _compare_dim_with_unknown(
        mat1.shape[1],
        mat2[0],
        kind="inner_dim",
        left_name="sparse columns",
        right_name="dense rows",
    )
    if err is not None:
        return err
    output = (mat1.shape[0], mat2[1])
    unknown = "symbolic sparse mm inner dimension: equality not fully checked" if symbolic else None
    return _ok(_dense_spec(output), unknown)


def verify_sparse_addmm(
    input_shape: Sequence[Dim],
    mat1: SparseLayoutSpec,
    mat2_shape: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse.addmm(input, mat1, mat2)`` shape rules."""

    input_ = _shape(input_shape)
    err = _check_all_nonnegative(("input", input_))
    if err is not None:
        return err
    mm = verify_sparse_mm(mat1, mat2_shape)
    if not mm.ok or mm.spec is None:
        return mm

    err, symbolic_broadcast = _broadcastable_to(input_, mm.spec.shape)
    if err is not None:
        return err
    unknown = mm.unknown_reason
    if symbolic_broadcast:
        suffix = "symbolic addmm input broadcast: equality not fully checked"
        unknown = f"{unknown}; {suffix}" if unknown else suffix
    return _ok(mm.spec, unknown)


def verify_sparse_sampled_addmm(
    input_spec: SparseLayoutSpec,
    mat1_shape: Sequence[Dim],
    mat2_shape: Sequence[Dim],
) -> SparseVerdict:
    """Verify ``torch.sparse.sampled_addmm(input, mat1, mat2)`` shape rules.

    PyTorch requires a CSR sparse mask. Dense batch dimensions must match each
    other exactly. A rank-2 CSR mask may be reused across dense batches; batched
    CSR masks must have the same batch shape as the dense operands.
    """

    mat1 = _shape(mat1_shape)
    mat2 = _shape(mat2_shape)
    err = _check_all_nonnegative(("input", input_spec.shape), ("mat1", mat1), ("mat2", mat2))
    if err is not None:
        return err
    if input_spec.layout != "csr":
        return _fail("layout", f"torch.sparse.sampled_addmm requires a CSR sparse input, got {input_spec.layout!r}")
    if len(input_spec.shape) < 2 or len(mat1) < 2 or len(mat2) < 2:
        return _fail("rank", "sampled_addmm requires matrix-shaped input, mat1, and mat2 operands")

    dense_batch = mat1[:-2]
    err, symbolic = _compare_shape_with_unknown(
        dense_batch,
        mat2[:-2],
        kind="batch_shape",
        left_name="mat1 batch",
        right_name="mat2 batch",
    )
    if err is not None:
        return err

    input_batch = input_spec.shape[:-2]
    if input_batch:
        err, batch_symbolic = _compare_shape_with_unknown(
            input_batch,
            dense_batch,
            kind="batch_shape",
            left_name="sparse input batch",
            right_name="dense batch",
        )
        if err is not None:
            return err
        symbolic = symbolic or batch_symbolic

    checks = (
        (
            mat1[-1],
            mat2[-2],
            "inner_dim",
            "mat1 columns",
            "mat2 rows",
        ),
        (
            input_spec.shape[-2],
            mat1[-2],
            "output_shape",
            "sparse input rows",
            "mat1 rows",
        ),
        (
            input_spec.shape[-1],
            mat2[-1],
            "output_shape",
            "sparse input columns",
            "mat2 columns",
        ),
    )
    for left, right, kind, left_name, right_name in checks:
        err, dim_symbolic = _compare_dim_with_unknown(
            left,
            right,
            kind=kind,
            left_name=left_name,
            right_name=right_name,
        )
        if err is not None:
            return err
        symbolic = symbolic or dim_symbolic

    output_shape = (dense_batch + input_spec.shape[-2:]) if not input_batch else input_spec.shape
    unknown = "symbolic sampled_addmm dimension: equality not fully checked" if symbolic else None
    return _ok(_sparse_like_spec(input_spec, output_shape), unknown)


def verify_sparse_softmax(input_spec: SparseLayoutSpec, dim: int) -> SparseVerdict:
    """Verify ``torch.sparse.softmax(input, dim)`` shape/layout rules."""

    if input_spec.layout != "coo":
        return _fail("layout", f"torch.sparse.softmax requires a COO sparse tensor, got {input_spec.layout!r}")
    if not input_spec.shape:
        return _fail("rank", "torch.sparse.softmax requires at least one dimension")
    if _normalize_dim(dim, len(input_spec.shape)) is None:
        return _fail("dim", f"dim {dim} is out of range for rank {len(input_spec.shape)}")
    return _ok(_sparse_like_spec(input_spec))


def verify_sparse_coalesce(input_spec: SparseLayoutSpec) -> SparseVerdict:
    """Verify ``Tensor.coalesce()`` shape/layout rules for sparse COO tensors."""

    if input_spec.layout != "coo":
        return _fail("layout", f"coalesce requires a COO sparse tensor, got {input_spec.layout!r}")
    return _ok(_sparse_like_spec(input_spec))


def verify_sparse_to_dense(input_spec: SparseLayoutSpec) -> SparseVerdict:
    """Verify ``Tensor.to_dense()`` shape preservation for sparse tensors."""

    if input_spec.layout not in {"coo", "csr", "csc", "bsr", "bsc"}:
        return _fail("layout", f"to_dense requires a sparse tensor, got {input_spec.layout!r}")
    return _ok(_dense_spec(input_spec.shape))


def verify_sparse_layout_conversion(
    input_shape: Union[Sequence[Dim], SparseLayoutSpec],
    target_layout: str,
    blocksize: Optional[Tuple[Dim, Dim]] = None,
) -> SparseVerdict:
    """Verify ``to_sparse_*`` layout-conversion shape contracts.

    The conversion preserves the dense shape. COO accepts any rank. Compressed
    layouts require at least matrix rank; blocked layouts additionally require a
    positive blocksize that divides the two sparse matrix axes when known.
    """

    shape = input_shape.shape if isinstance(input_shape, SparseLayoutSpec) else _shape(input_shape)
    err = _check_nonnegative(shape, "input")
    if err is not None:
        return err
    target = _canonical_layout(target_layout)
    normalized_target = str(target_layout).replace("torch.", "").replace("sparse_", "").lower()
    if target is None and normalized_target == "coo":
        target = "coo"
    if target is None:
        return _fail("layout", f"unsupported sparse conversion target {target_layout!r}")

    if target == "coo":
        return _ok(
            SparseLayoutSpec(
                layout="coo",
                shape=shape,
                sparse_shape=shape,
                dense_shape=(),
                sparse_dim=len(shape),
            )
        )

    if len(shape) < 2:
        return _fail("rank", f"to_sparse_{target} requires rank >= 2, got rank {len(shape)}")

    batch_shape = shape[:-2]
    sparse_shape = shape[-2:]
    symbolic = False
    resolved_blocksize: Optional[Tuple[Dim, Dim]] = None
    if target in {"bsr", "bsc"}:
        if blocksize is None:
            return _fail("blocksize", f"to_sparse_{target} requires a blocksize")
        if len(blocksize) != 2:
            return _fail("blocksize", "blocksize must contain exactly two dimensions")
        rows, cols = blocksize
        for label, value in (("row", rows), ("column", cols)):
            if _is_int_dim(value):
                if value <= 0:
                    return _fail("blocksize", f"{label} block size must be positive, got {value}")
            else:
                symbolic = True
        for label, extent, block in (
            ("row", sparse_shape[0], rows),
            ("column", sparse_shape[1], cols),
        ):
            if _is_int_dim(extent) and _is_int_dim(block):
                if extent % block != 0:
                    return _fail(
                        "block_divisibility",
                        f"{label} dimension {extent} is not divisible by block size {block}",
                    )
            elif extent != block:
                symbolic = True
        resolved_blocksize = (rows, cols)
    elif blocksize is not None:
        return _fail("blocksize", f"to_sparse_{target} does not accept a blocksize")

    unknown = "symbolic sparse layout conversion: divisibility not fully checked" if symbolic else None
    return _ok(
        SparseLayoutSpec(
            layout=target,
            shape=shape,
            sparse_shape=sparse_shape,
            dense_shape=(),
            batch_shape=batch_shape,
            blocksize=resolved_blocksize,
        ),
        unknown,
    )
