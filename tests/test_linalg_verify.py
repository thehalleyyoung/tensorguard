from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from src.linalg_verify import (  # noqa: E402
    verify_linalg,
    verify_linalg_cholesky,
    verify_linalg_eig,
    verify_linalg_inv,
    verify_linalg_qr,
    verify_linalg_solve,
    verify_linalg_svd,
)
from src.operator_confidence import ConfidenceTag, heuristic_ops_in_source, tag_for  # noqa: E402
from src.stdlib.modern_ops import (  # noqa: E402
    transfer_linalg_cholesky,
    transfer_linalg_eig,
    transfer_linalg_inv,
    transfer_linalg_qr,
    transfer_linalg_solve,
    transfer_linalg_svd,
)
from src.tensor_shapes import TensorShape  # noqa: E402


def _shape_dict(verdict):
    return dict(verdict.output_shapes)


def _matrix(shape):
    return torch.randn(*shape)


def _spd(shape):
    n = shape[-1]
    base = torch.randn(*shape)
    eye = torch.eye(n).expand(*shape[:-2], n, n)
    return base @ base.mT + (n + 1) * eye


def _invertible(shape):
    n = shape[-1]
    eye = torch.eye(n).expand(*shape[:-2], n, n)
    return torch.randn(*shape) * 0.01 + eye


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (0, 0), (2, 0, 0)])
def test_inv_and_cholesky_square_shapes_match_real_torch(shape):
    inv = verify_linalg_inv(shape)
    chol = verify_linalg_cholesky(shape)

    assert inv.ok
    assert inv.output_shape == tuple(torch.linalg.inv(_invertible(shape)).shape)
    assert chol.ok
    assert chol.output_shape == tuple(torch.linalg.cholesky(_spd(shape)).shape)


@pytest.mark.parametrize("shape", [(3,), (3, 5), (2, 0, 3)])
def test_square_linalg_ops_reject_real_shape_failures(shape):
    for checker in (verify_linalg_inv, verify_linalg_cholesky, verify_linalg_eig):
        verdict = checker(shape)
        assert not verdict.ok


@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5), (5, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("full_matrices", [True, False])
def test_svd_shapes_match_real_torch(shape, full_matrices):
    actual = torch.linalg.svd(_matrix(shape), full_matrices=full_matrices)
    verdict = verify_linalg_svd(shape, full_matrices=full_matrices)

    assert verdict.ok
    assert _shape_dict(verdict) == {
        "U": tuple(actual.U.shape),
        "S": tuple(actual.S.shape),
        "Vh": tuple(actual.Vh.shape),
    }


@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 5), (5, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
def test_qr_shapes_match_real_torch_including_r_mode_empty_q(shape, mode):
    actual = torch.linalg.qr(_matrix(shape), mode=mode)
    verdict = verify_linalg_qr(shape, mode=mode)

    assert verdict.ok
    assert _shape_dict(verdict) == {
        "Q": tuple(actual.Q.shape),
        "R": tuple(actual.R.shape),
    }


@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3), (0, 0), (2, 0, 0)])
def test_eig_shapes_match_real_torch(shape):
    actual = torch.linalg.eig(_invertible(shape))
    verdict = verify_linalg_eig(shape)

    assert verdict.ok
    assert _shape_dict(verdict) == {
        "eigenvalues": tuple(actual.eigenvalues.shape),
        "eigenvectors": tuple(actual.eigenvectors.shape),
    }


@pytest.mark.parametrize(
    "a_shape,b_shape,left",
    [
        ((3, 3), (3,), True),
        ((3, 3), (3, 2), True),
        ((2, 3, 3), (2, 3), True),
        ((2, 3, 3), (3, 2), True),
        ((1, 3, 3), (2, 3, 4), True),
        ((2, 1, 3, 3), (4, 3, 5), True),
        ((3, 3), (2, 3), False),
        ((2, 3, 3), (1, 3), False),
        ((2, 3, 3), (2, 1, 3), False),
        ((2, 1, 3, 3), (4, 3), False),
    ],
)
def test_solve_valid_shapes_match_real_torch(a_shape, b_shape, left):
    actual = torch.linalg.solve(_invertible(a_shape), torch.randn(*b_shape), left=left)
    verdict = verify_linalg_solve(a_shape, b_shape, left=left)

    assert verdict.ok, verdict
    assert verdict.output_shape == tuple(actual.shape)


@pytest.mark.parametrize(
    "a_shape,b_shape,left,kind",
    [
        ((3, 3), (2, 3), True, "rhs_dim"),
        ((2, 3, 3), (1, 3), True, "rhs_dim"),
        ((2, 3, 3), (2, 1, 3), True, "rhs_dim"),
        ((3, 3), (3,), False, "rhs_vector"),
        ((2, 3, 3), (2, 3), False, "rhs_vector"),
        ((1, 3, 3), (1, 3), False, "rhs_vector"),
    ],
)
def test_solve_refutes_real_shape_failures(a_shape, b_shape, left, kind):
    verdict = verify_linalg_solve(a_shape, b_shape, left=left)
    assert not verdict.ok
    assert verdict.error_kind == kind
    with pytest.raises(Exception):
        torch.linalg.solve(_invertible(a_shape), torch.randn(*b_shape), left=left)


def test_solve_vector_branch_does_not_broadcast_batched_rhs():
    assert verify_linalg_solve((1, 3, 3), (2, 3)).error_kind == "rhs_dim"
    with pytest.raises(RuntimeError, match="Incompatible shapes"):
        torch.linalg.solve(_invertible((1, 3, 3)), torch.randn(2, 3))


def test_symbolic_linalg_shapes_are_not_refuted_when_runtime_may_be_valid():
    inv = verify_linalg_inv(("B", "M", "N"))
    assert inv.ok
    assert inv.output_shape == ("B", "M", "N")
    assert inv.unknown_reason

    solve = verify_linalg_solve(("B", "N", "N"), ("C", "N"))
    assert solve.ok
    assert solve.output_shapes == ()
    assert solve.unknown_reason

    svd = verify_linalg_svd(("B", "M", "N"), full_matrices=False)
    assert svd.ok
    assert svd.shape("S") == ("B", "min(M,N)")


def test_dispatch_and_argument_validation():
    assert verify_linalg("torch.linalg.inv", (3, 3)).output_shape == (3, 3)
    assert verify_linalg("svd", (3, 5), full_matrices=False).shape("Vh") == (3, 5)
    assert verify_linalg_qr((3, 5), mode="bad").error_kind == "mode"
    assert verify_linalg_svd((3, 5), full_matrices=1).error_kind == "argument"
    assert verify_linalg_cholesky((3, 3), upper=1).error_kind == "argument"


def test_linalg_differential_fuzz_for_decomposition_shapes():
    rng = random.Random(20260602)
    for _ in range(120):
        batch = rng.choice([(), (rng.randint(1, 3),), (rng.randint(1, 2), rng.randint(1, 3))])
        m = rng.randint(0, 5)
        n = rng.randint(0, 5)
        shape = batch + (m, n)
        tensor = _matrix(shape)

        for full_matrices in (True, False):
            actual = torch.linalg.svd(tensor, full_matrices=full_matrices)
            verdict = verify_linalg_svd(shape, full_matrices=full_matrices)
            assert _shape_dict(verdict) == {
                "U": tuple(actual.U.shape),
                "S": tuple(actual.S.shape),
                "Vh": tuple(actual.Vh.shape),
            }
        for mode in ("reduced", "complete", "r"):
            actual = torch.linalg.qr(tensor, mode=mode)
            verdict = verify_linalg_qr(shape, mode=mode)
            assert _shape_dict(verdict) == {
                "Q": tuple(actual.Q.shape),
                "R": tuple(actual.R.shape),
            }


def test_modern_ops_linalg_transfers_delegate_to_exact_contracts():
    a = TensorShape.from_tuple((2, 3, 3))
    b = TensorShape.from_tuple((3, 4))

    assert transfer_linalg_inv(TensorShape.from_tuple((3, 5))) is None
    assert transfer_linalg_cholesky(a).pretty() == "(2, 3, 3)"
    assert transfer_linalg_eig(a).pretty() == "(2, 3)"
    assert transfer_linalg_svd(TensorShape.from_tuple((2, 3, 5))).pretty() == "(2, 3)"
    assert transfer_linalg_qr(TensorShape.from_tuple((2, 5, 3))).pretty() == "(2, 5, 3)"
    assert transfer_linalg_solve(a, b).pretty() == "(2, 3, 4)"


def test_linalg_confidence_and_sound_mode_scan_are_per_op():
    assert tag_for("torch.linalg.svd") is ConfidenceTag.SOUND
    assert tag_for("torch.linalg.solve") is ConfidenceTag.SOUND
    assert tag_for("torch.linalg.inv") is ConfidenceTag.SOUND
    assert tag_for("torch.linalg.lstsq") is ConfidenceTag.HEURISTIC

    source = """
import torch
def f(a, b):
    x = torch.linalg.svd(a)
    y = torch.linalg.lstsq(a, b)
    return x, y
"""
    assert heuristic_ops_in_source(source) == ["torch.linalg.lstsq"]


def test_public_package_exports_linalg_checker():
    import tensorguard

    assert tensorguard.verify_linalg is verify_linalg
    assert tensorguard.verify_linalg_svd is verify_linalg_svd
