from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from src.sparse_verify import (  # noqa: E402
    verify_sparse_bsc,
    verify_sparse_bsr,
    verify_sparse_coo,
    verify_sparse_csc,
    verify_sparse_csr,
)


def _long(shape, data=None):
    if data is not None:
        return torch.tensor(data, dtype=torch.int64)
    return torch.zeros(tuple(shape), dtype=torch.int64)


def _values(shape):
    return torch.ones(tuple(shape))


def _sparse_call(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_coo_layout_contract_matches_real_torch_constructor():
    indices = _long((2, 3), [[0, 1, 1], [1, 0, 2]])
    values = _values((3, 4))
    actual = _sparse_call(torch.sparse_coo_tensor, indices, values, (2, 3, 4))

    verdict = verify_sparse_coo((2, 3), (3, 4), (2, 3, 4))

    assert verdict.ok
    assert verdict.spec is not None
    assert verdict.spec.layout == "coo"
    assert verdict.spec.sparse_dim == 2
    assert verdict.spec.sparse_shape == (2, 3)
    assert verdict.spec.dense_shape == (4,)
    assert verdict.spec.shape == tuple(actual.shape)

    assert verify_sparse_coo((2, 3), (4, 4), (2, 3, 4)).error_kind == "nnz"
    with pytest.raises(RuntimeError, match="same nnz"):
        torch.sparse_coo_tensor(indices, _values((4, 4)), (2, 3, 4))

    assert verify_sparse_coo((2, 3), (3, 5), (2, 3, 4)).error_kind == "dense_shape"
    with pytest.raises(RuntimeError, match="incorrect size"):
        torch.sparse_coo_tensor(indices, _values((3, 5)), (2, 3, 4))

    assert verify_sparse_coo((2, 3), (3,), (2,)).error_kind == "size_rank"
    with pytest.raises(RuntimeError, match="number of dimensions"):
        torch.sparse_coo_tensor(indices, _values((3,)), (2,))


def test_csr_and_csc_validated_contracts_match_check_invariants():
    crow = _long((3,), [0, 2, 3])
    col = _long((3,), [0, 1, 1])
    csr = _sparse_call(
        torch.sparse_csr_tensor,
        crow,
        col,
        _values((3,)),
        size=(2, 3),
        check_invariants=True,
    )
    assert verify_sparse_csr((3,), (3,), (3,), (2, 3)).spec.shape == tuple(csr.shape)

    ccol = _long((4,), [0, 1, 3, 3])
    row = _long((3,), [0, 0, 1])
    csc = _sparse_call(
        torch.sparse_csc_tensor,
        ccol,
        row,
        _values((3,)),
        size=(2, 3),
        check_invariants=True,
    )
    assert verify_sparse_csc((4,), (3,), (3,), (2, 3)).spec.shape == tuple(csc.shape)

    assert verify_sparse_csr((2,), (3,), (3,), (2, 3)).error_kind == "compressed_indices"
    with pytest.raises(RuntimeError, match="number of rows"):
        torch.sparse_csr_tensor(_long((2,), [0, 2]), col, _values((3,)), size=(2, 3), check_invariants=True)

    assert verify_sparse_csr((3,), (2,), (3,), (2, 3)).error_kind == "nnz"
    with pytest.raises(RuntimeError, match="nnz"):
        torch.sparse_csr_tensor(crow, _long((2,), [0, 1]), _values((3,)), size=(2, 3), check_invariants=True)

    assert verify_sparse_csc((3,), (3,), (3,), (2, 3)).error_kind == "compressed_indices"
    with pytest.raises(RuntimeError, match="number of columns"):
        torch.sparse_csc_tensor(_long((3,), [0, 1, 3]), row, _values((3,)), size=(2, 3), check_invariants=True)

    assert verify_sparse_csr((3,), (3,), (3,), (2,)).error_kind == "size_rank"
    with pytest.raises(RuntimeError, match="tensor dimensionality"):
        torch.sparse_csr_tensor(crow, col, _values((3,)), size=(2,), check_invariants=True)


def test_block_sparse_rules_match_real_invariant_checker():
    crow = _long((3,), [0, 1, 2])
    col = _long((2,), [0, 0])
    bsr = _sparse_call(
        torch.sparse_bsr_tensor,
        crow,
        col,
        _values((2, 2, 3)),
        size=(4, 3),
        check_invariants=True,
    )
    verdict = verify_sparse_bsr((3,), (2,), (2, 2, 3), (4, 3))
    assert verdict.ok
    assert verdict.spec.blocksize == (2, 3)
    assert verdict.spec.shape == tuple(bsr.shape)

    ccol = _long((2,), [0, 1])
    row = _long((1,), [0])
    bsc = _sparse_call(
        torch.sparse_bsc_tensor,
        ccol,
        row,
        _values((1, 2, 3)),
        size=(2, 3),
        check_invariants=True,
    )
    assert verify_sparse_bsc((2,), (1,), (1, 2, 3), (2, 3)).spec.shape == tuple(bsc.shape)

    assert verify_sparse_bsr((3,), (2,), (2, 2, 3), (5, 3)).error_kind == "block_divisibility"
    with pytest.raises(RuntimeError, match="divisible"):
        torch.sparse_bsr_tensor(crow, col, _values((2, 2, 3)), size=(5, 3), check_invariants=True)

    assert verify_sparse_bsr((3,), (2,), (2, 2, 3), (4, 4)).error_kind == "block_divisibility"
    with pytest.raises(RuntimeError, match="divisible"):
        torch.sparse_bsr_tensor(crow, col, _values((2, 2, 3)), size=(4, 4), check_invariants=True)

    assert verify_sparse_bsr((3,), (2,), (2, 2), (4, 3)).error_kind == "values_rank"
    with pytest.raises(RuntimeError, match="values must have dimensionality"):
        torch.sparse_bsr_tensor(crow, col, _values((2, 2)), size=(4, 3), check_invariants=True)

    assert verify_sparse_bsc((3,), (1,), (1, 2, 3), (2, 3)).error_kind == "compressed_indices"
    with pytest.raises(RuntimeError, match="column blocks"):
        torch.sparse_bsc_tensor(_long((3,), [0, 1, 1]), row, _values((1, 2, 3)), size=(2, 3), check_invariants=True)


def test_batched_and_zero_nnz_layouts_are_not_false_positive():
    crow = _long((2, 3), [[0, 2, 3], [0, 1, 3]])
    col = _long((2, 3), [[0, 1, 1], [0, 0, 2]])
    values = _values((2, 3, 4))
    actual = _sparse_call(
        torch.sparse_csr_tensor,
        crow,
        col,
        values,
        size=(2, 2, 3, 4),
        check_invariants=True,
    )
    verdict = verify_sparse_csr((2, 3), (2, 3), (2, 3, 4), (2, 2, 3, 4))
    assert verdict.ok
    assert verdict.spec.batch_shape == (2,)
    assert verdict.spec.dense_shape == (4,)
    assert verdict.spec.shape == tuple(actual.shape)

    bcrow = _long((2, 3), [[0, 1, 2], [0, 1, 2]])
    bcol = _long((2, 2), [[0, 0], [0, 0]])
    bvals = _values((2, 2, 2, 3, 5))
    bsr = _sparse_call(
        torch.sparse_bsr_tensor,
        bcrow,
        bcol,
        bvals,
        size=(2, 4, 3, 5),
        check_invariants=True,
    )
    assert verify_sparse_bsr((2, 3), (2, 2), (2, 2, 2, 3, 5), (2, 4, 3, 5)).spec.shape == tuple(bsr.shape)

    zero = _sparse_call(
        torch.sparse_csr_tensor,
        _long((3,), [0, 0, 0]),
        _long((0,), []),
        _values((0,)),
        size=(2, 3),
        check_invariants=True,
    )
    assert verify_sparse_csr((3,), (0,), (0,), (2, 3)).ok
    assert zero._nnz() == 0


def test_compressed_dense_tail_mismatch_is_reported_as_unusable_layout():
    cases = [
        (
            verify_sparse_csr,
            torch.sparse_csr_tensor,
            (_long((3,), [0, 2, 3]), _long((3,), [0, 1, 1]), _values((3, 5)), (2, 3, 4)),
            ((3,), (3,), (3, 5), (2, 3, 4)),
        ),
        (
            verify_sparse_csc,
            torch.sparse_csc_tensor,
            (_long((4,), [0, 1, 3, 3]), _long((3,), [0, 0, 1]), _values((3, 5)), (2, 3, 4)),
            ((4,), (3,), (3, 5), (2, 3, 4)),
        ),
        (
            verify_sparse_bsr,
            torch.sparse_bsr_tensor,
            (_long((3,), [0, 1, 2]), _long((2,), [0, 0]), _values((2, 2, 3, 5)), (4, 3, 4)),
            ((3,), (2,), (2, 2, 3, 5), (4, 3, 4)),
        ),
        (
            verify_sparse_bsc,
            torch.sparse_bsc_tensor,
            (_long((2,), [0, 1]), _long((1,), [0]), _values((1, 2, 3, 5)), (2, 3, 4)),
            ((2,), (1,), (1, 2, 3, 5), (2, 3, 4)),
        ),
    ]

    for verifier, ctor, torch_args, static_args in cases:
        verdict = verifier(*static_args)
        assert not verdict.ok
        assert verdict.error_kind == "unusable_dense"

        tensor = _sparse_call(ctor, *torch_args[:3], size=torch_args[3], check_invariants=True)
        with pytest.raises(RuntimeError):
            tensor.to_dense()


def test_symbolic_sparse_dims_abstain_from_arithmetic_without_false_positive():
    coo = verify_sparse_coo((2, "NNZ"), ("NNZ", "C"), (8, 9, "C"))
    assert coo.ok
    assert coo.spec.dense_shape == ("C",)

    bsr = verify_sparse_bsr(("row_blocks_plus_one",), ("N",), ("N", "BR", "BC"), ("ROWS", "COLS"))
    assert bsr.ok
    assert bsr.unknown_reason is not None
    assert bsr.spec.blocksize == ("BR", "BC")

    csr = verify_sparse_csr(("B", "rows_plus_one"), ("B", "NNZ"), ("B", "NNZ"), ("B", "ROWS", "COLS"))
    assert csr.ok
    assert csr.unknown_reason is not None
    assert csr.spec.batch_shape == ("B",)


def test_public_package_exports_sparse_verifiers():
    import src
    import tensorguard

    assert src.verify_sparse_coo is verify_sparse_coo
    assert src.verify_sparse_csr is verify_sparse_csr
    assert tensorguard.verify_sparse_bsr is verify_sparse_bsr
    assert tensorguard.verify_sparse_bsc is verify_sparse_bsc
