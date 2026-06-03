from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from src.sparse_verify import (  # noqa: E402
    verify_sparse_addmm,
    verify_sparse_bsc,
    verify_sparse_bsr,
    verify_sparse_coalesce,
    verify_sparse_coo,
    verify_sparse_csc,
    verify_sparse_csr,
    verify_sparse_layout_conversion,
    verify_sparse_mm,
    verify_sparse_sampled_addmm,
    verify_sparse_softmax,
    verify_sparse_to_dense,
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


def _spec(verdict):
    assert verdict.ok
    assert verdict.spec is not None
    return verdict.spec


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


def test_sparse_mm_and_addmm_match_live_sparse_dense_kernels():
    indices = _long((2, 3), [[0, 1, 1], [1, 0, 2]])
    values = torch.tensor([2.0, 3.0, 4.0])
    coo = torch.sparse_coo_tensor(indices, values, (2, 3))
    csr = _sparse_call(coo.to_sparse_csr)
    csc = _sparse_call(coo.to_sparse_csc)
    rhs = torch.arange(12.0).reshape(3, 4)

    specs = [
        (_spec(verify_sparse_coo((2, 3), (3,), (2, 3))), coo),
        (_spec(verify_sparse_csr((3,), (3,), (3,), (2, 3))), csr),
        (_spec(verify_sparse_csc((4,), (3,), (3,), (2, 3))), csc),
    ]
    for spec, tensor in specs:
        actual = _sparse_call(torch.sparse.mm, tensor, rhs)
        verdict = verify_sparse_mm(spec, rhs.shape)
        assert verdict.ok
        assert verdict.spec.layout == "dense"
        assert verdict.spec.shape == tuple(actual.shape)
        assert verdict.spec.shape == tuple(actual.to_dense().shape if actual.layout != torch.strided else actual.shape)

    addmm = _sparse_call(torch.sparse.addmm, torch.ones(2, 1), coo, rhs)
    addmm_verdict = verify_sparse_addmm((2, 1), specs[0][0], rhs.shape)
    assert addmm_verdict.spec.shape == tuple(addmm.shape)
    assert verify_sparse_addmm((4,), specs[0][0], rhs.shape).ok
    assert verify_sparse_addmm((), specs[0][0], rhs.shape).ok
    assert verify_sparse_addmm((3, 4), specs[0][0], rhs.shape).error_kind == "broadcast"

    assert verify_sparse_mm(specs[0][0], (4, 5)).error_kind == "inner_dim"
    with pytest.raises(RuntimeError, match="Expected dim 0 size"):
        torch.sparse.mm(coo, torch.ones(4, 5))

    bsr = _sparse_call(torch.ones(4, 6).to_sparse_bsr, (2, 3))
    bsr_spec = _spec(verify_sparse_bsr((3,), (2,), (2, 2, 3), (4, 6)))
    assert verify_sparse_mm(bsr_spec, (6, 5)).error_kind == "layout"
    with pytest.raises(RuntimeError, match="SparseBsr|not implemented"):
        torch.sparse.mm(bsr, torch.ones(6, 5))


def test_sampled_addmm_matches_live_csr_batch_rules():
    csr = _sparse_call(
        torch.sparse_csr_tensor,
        _long((3,), [0, 1, 2]),
        _long((2,), [0, 2]),
        _values((2,)),
        size=(2, 3),
        check_invariants=True,
    )
    csr_spec = _spec(verify_sparse_csr((3,), (2,), (2,), (2, 3)))

    actual = _sparse_call(torch.sparse.sampled_addmm, csr, torch.ones(2, 5), torch.ones(5, 3))
    verdict = verify_sparse_sampled_addmm(csr_spec, (2, 5), (5, 3))
    assert verdict.ok
    assert verdict.spec.layout == "csr"
    assert verdict.spec.shape == tuple(actual.shape)
    assert verdict.spec.shape == tuple(actual.to_dense().shape)

    batched_from_unbatched = _sparse_call(
        torch.sparse.sampled_addmm,
        csr,
        torch.ones(2, 2, 5),
        torch.ones(2, 5, 3),
    )
    batched_verdict = verify_sparse_sampled_addmm(csr_spec, (2, 2, 5), (2, 5, 3))
    assert batched_verdict.ok
    assert batched_verdict.spec.batch_shape == (2,)
    assert batched_verdict.spec.shape == tuple(batched_from_unbatched.to_dense().shape)

    b_crow = _long((2, 3), [[0, 1, 2], [0, 0, 2]])
    b_col = _long((2, 2), [[0, 1], [0, 2]])
    b_val = _values((2, 2))
    batched_csr = _sparse_call(
        torch.sparse_csr_tensor,
        b_crow,
        b_col,
        b_val,
        size=(2, 2, 3),
        check_invariants=True,
    )
    batched_csr_spec = _spec(verify_sparse_csr((2, 3), (2, 2), (2, 2), (2, 2, 3)))
    actual_batched = _sparse_call(
        torch.sparse.sampled_addmm,
        batched_csr,
        torch.ones(2, 2, 5),
        torch.ones(2, 5, 3),
    )
    assert verify_sparse_sampled_addmm(batched_csr_spec, (2, 2, 5), (2, 5, 3)).spec.shape == tuple(
        actual_batched.to_dense().shape
    )

    coo_spec = _spec(verify_sparse_coo((2, 2), (2,), (2, 3)))
    assert verify_sparse_sampled_addmm(coo_spec, (2, 5), (5, 3)).error_kind == "layout"
    with pytest.raises((RuntimeError, NotImplementedError), match="sampled_addmm|SparseCPU"):
        torch.sparse.sampled_addmm(csr.to_sparse_coo(), torch.ones(2, 5), torch.ones(5, 3))

    assert verify_sparse_sampled_addmm(csr_spec, (2, 5), (4, 3)).error_kind == "inner_dim"
    assert verify_sparse_sampled_addmm(csr_spec, (1, 2, 5), (2, 5, 3)).error_kind == "batch_shape"
    assert verify_sparse_sampled_addmm(csr_spec, (2, 5), (5, 4)).error_kind == "output_shape"


def test_sparse_softmax_coalesce_and_conversions_have_to_dense_shape_parity():
    indices = _long((2, 3), [[0, 1, 1], [1, 0, 2]])
    values = torch.tensor([2.0, 3.0, 4.0])
    coo = torch.sparse_coo_tensor(indices, values, (2, 3))
    coo_spec = _spec(verify_sparse_coo((2, 3), (3,), (2, 3)))
    csr = _sparse_call(coo.to_sparse_csr)
    csr_spec = _spec(verify_sparse_csr((3,), (3,), (3,), (2, 3)))

    softmax = _sparse_call(torch.sparse.softmax, coo, dim=-1)
    softmax_verdict = verify_sparse_softmax(coo_spec, -1)
    assert softmax_verdict.ok
    assert softmax_verdict.spec.layout == "coo"
    assert softmax_verdict.spec.shape == tuple(softmax.to_dense().shape)
    assert verify_sparse_softmax(coo_spec, 2).error_kind == "dim"
    assert verify_sparse_softmax(csr_spec, 1).error_kind == "layout"
    with pytest.raises((RuntimeError, NotImplementedError), match="SparseCsr|sparse_softmax"):
        torch.sparse.softmax(csr, dim=1)

    coalesced = _sparse_call(coo.coalesce)
    coalesce_verdict = verify_sparse_coalesce(coo_spec)
    assert coalesce_verdict.ok
    assert coalesce_verdict.spec.shape == tuple(coalesced.to_dense().shape)
    assert verify_sparse_coalesce(csr_spec).error_kind == "layout"
    with pytest.raises(RuntimeError, match="coalesce expected sparse coordinate"):
        csr.coalesce()

    for spec, tensor in ((coo_spec, coo), (csr_spec, csr)):
        dense = tensor.to_dense()
        verdict = verify_sparse_to_dense(spec)
        assert verdict.ok
        assert verdict.spec.layout == "dense"
        assert verdict.spec.shape == tuple(dense.shape)

    dense = torch.ones(4, 6)
    conversion_cases = [
        ("coo", None, dense.to_sparse_coo()),
        ("csr", None, _sparse_call(dense.to_sparse_csr)),
        ("csc", None, _sparse_call(dense.to_sparse_csc)),
        ("bsr", (2, 3), _sparse_call(dense.to_sparse_bsr, (2, 3))),
        ("bsc", (2, 3), _sparse_call(dense.to_sparse_bsc, (2, 3))),
    ]
    for layout, blocksize, tensor in conversion_cases:
        verdict = verify_sparse_layout_conversion(dense.shape, layout, blocksize=blocksize)
        assert verdict.ok
        assert verdict.spec.layout == layout
        assert verdict.spec.shape == tuple(tensor.to_dense().shape)

    assert verify_sparse_layout_conversion((5, 6), "bsr", blocksize=(2, 3)).error_kind == "block_divisibility"
    with pytest.raises(RuntimeError, match="must be divisible"):
        torch.ones(5, 6).to_sparse_bsr((2, 3))
    assert verify_sparse_layout_conversion((3,), "csr").error_kind == "rank"
    with pytest.raises((RuntimeError, IndexError)):
        torch.ones(3).to_sparse_csr()


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
    assert src.verify_sparse_mm is verify_sparse_mm
    assert src.verify_sparse_sampled_addmm is verify_sparse_sampled_addmm
    assert tensorguard.verify_sparse_addmm is verify_sparse_addmm
    assert tensorguard.verify_sparse_layout_conversion is verify_sparse_layout_conversion
