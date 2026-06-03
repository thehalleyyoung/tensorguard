"""Step 236 — Lean sparse-layout invariants checked against real torch.

``lean/TensorGuard/SparseLayouts.lean`` mechanizes TensorGuard's shape-only
contract for PyTorch COO/CSR/CSC/BSR/BSC sparse constructors.  For compressed
layouts the contract intentionally matches PyTorch's ``check_invariants=True``
path and adds the verifier's dense-tail usability check: if TensorGuard accepts a
constructor shape, ``to_dense`` has exactly the requested dense shape.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings

import pytest

torch = pytest.importorskip("torch")

from src.sparse_verify import (  # noqa: E402
    verify_sparse_bsc,
    verify_sparse_bsr,
    verify_sparse_coo,
    verify_sparse_csc,
    verify_sparse_csr,
    verify_sparse_to_dense,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "SparseLayouts.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.SparseLayouts.dense_materialization_shape_sound",
    "TensorGuard.SparseLayouts.mkAccepted_dense_shape_sound",
    "TensorGuard.SparseLayouts.coo234_accepts",
    "TensorGuard.SparseLayouts.csr23_accepts",
    "TensorGuard.SparseLayouts.csc23_accepts",
    "TensorGuard.SparseLayouts.bsr43_accepts",
    "TensorGuard.SparseLayouts.bsc23_accepts",
    "TensorGuard.SparseLayouts.batched_csr_accepts",
    "TensorGuard.SparseLayouts.batched_bsr_accepts",
    "TensorGuard.SparseLayouts.coo234_toDense_shape",
    "TensorGuard.SparseLayouts.csr23_toDense_shape",
    "TensorGuard.SparseLayouts.csc23_toDense_shape",
    "TensorGuard.SparseLayouts.bsr43_toDense_shape",
    "TensorGuard.SparseLayouts.bsc23_toDense_shape",
    "TensorGuard.SparseLayouts.batched_csr_toDense_shape",
    "TensorGuard.SparseLayouts.batched_bsr_toDense_shape",
    "TensorGuard.SparseLayouts.csr_bad_compressed_length_rejected",
    "TensorGuard.SparseLayouts.csc_bad_compressed_length_rejected",
    "TensorGuard.SparseLayouts.bsr_bad_row_divisibility_rejected",
    "TensorGuard.SparseLayouts.bsr_bad_column_divisibility_rejected",
    "TensorGuard.SparseLayouts.compressed_dense_tail_mismatch_rejected",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def _long(data):
    return torch.tensor(data, dtype=torch.int64)


def _values(shape):
    return torch.ones(tuple(shape))


def _sparse_call(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _spec(verdict):
    assert verdict.ok, verdict.error
    assert verdict.spec is not None
    return verdict.spec


def test_lean_sparse_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.SparseLayouts" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_sparse_file_has_no_sorry_or_admit():
    with open(_FILE) as fh:
        assert not re.search(r"\b(sorry|admit)\b", _strip_comments(fh.read()))


def test_constructor_dense_materialization_shape_parity_for_all_layouts():
    cases = [
        (
            "coo",
            verify_sparse_coo((2, 3), (3, 4), (2, 3, 4)),
            torch.sparse_coo_tensor(
                _long([[0, 1, 1], [1, 0, 2]]),
                _values((3, 4)),
                (2, 3, 4),
            ),
        ),
        (
            "csr",
            verify_sparse_csr((3,), (3,), (3,), (2, 3)),
            _sparse_call(
                torch.sparse_csr_tensor,
                _long([0, 2, 3]),
                _long([0, 1, 1]),
                _values((3,)),
                size=(2, 3),
                check_invariants=True,
            ),
        ),
        (
            "csc",
            verify_sparse_csc((4,), (3,), (3,), (2, 3)),
            _sparse_call(
                torch.sparse_csc_tensor,
                _long([0, 1, 3, 3]),
                _long([0, 0, 1]),
                _values((3,)),
                size=(2, 3),
                check_invariants=True,
            ),
        ),
        (
            "bsr",
            verify_sparse_bsr((3,), (2,), (2, 2, 3), (4, 3)),
            _sparse_call(
                torch.sparse_bsr_tensor,
                _long([0, 1, 2]),
                _long([0, 0]),
                _values((2, 2, 3)),
                size=(4, 3),
                check_invariants=True,
            ),
        ),
        (
            "bsc",
            verify_sparse_bsc((2,), (1,), (1, 2, 3), (2, 3)),
            _sparse_call(
                torch.sparse_bsc_tensor,
                _long([0, 1]),
                _long([0]),
                _values((1, 2, 3)),
                size=(2, 3),
                check_invariants=True,
            ),
        ),
    ]

    for layout, verdict, tensor in cases:
        spec = _spec(verdict)
        dense = tensor.to_dense()
        assert spec.layout == layout
        assert spec.shape == tuple(tensor.shape)
        assert verify_sparse_to_dense(spec).spec.shape == tuple(dense.shape)


def test_batched_sparse_dense_materialization_shape_parity():
    batched_csr = _sparse_call(
        torch.sparse_csr_tensor,
        _long([[0, 2, 3], [0, 1, 3]]),
        _long([[0, 1, 1], [0, 0, 2]]),
        _values((2, 3, 4)),
        size=(2, 2, 3, 4),
        check_invariants=True,
    )
    csr_spec = _spec(verify_sparse_csr((2, 3), (2, 3), (2, 3, 4), (2, 2, 3, 4)))
    assert csr_spec.batch_shape == (2,)
    assert csr_spec.dense_shape == (4,)
    assert verify_sparse_to_dense(csr_spec).spec.shape == tuple(batched_csr.to_dense().shape)

    batched_bsr = _sparse_call(
        torch.sparse_bsr_tensor,
        _long([[0, 1, 2], [0, 1, 2]]),
        _long([[0, 0], [0, 0]]),
        _values((2, 2, 2, 3, 5)),
        size=(2, 4, 3, 5),
        check_invariants=True,
    )
    bsr_spec = _spec(verify_sparse_bsr((2, 3), (2, 2), (2, 2, 2, 3, 5), (2, 4, 3, 5)))
    assert bsr_spec.batch_shape == (2,)
    assert bsr_spec.blocksize == (2, 3)
    assert verify_sparse_to_dense(bsr_spec).spec.shape == tuple(batched_bsr.to_dense().shape)


def test_rejection_examples_match_checked_torch_invariants_and_dense_usability():
    assert verify_sparse_csr((2,), (3,), (3,), (2, 3)).error_kind == "compressed_indices"
    with pytest.raises(RuntimeError, match="rows"):
        torch.sparse_csr_tensor(
            _long([0, 2]),
            _long([0, 1, 1]),
            _values((3,)),
            size=(2, 3),
            check_invariants=True,
        )

    assert verify_sparse_bsr((3,), (2,), (2, 2, 3), (5, 3)).error_kind == "block_divisibility"
    with pytest.raises(RuntimeError, match="divisible"):
        torch.sparse_bsr_tensor(
            _long([0, 1, 2]),
            _long([0, 0]),
            _values((2, 2, 3)),
            size=(5, 3),
            check_invariants=True,
        )

    dense_tail_mismatch = verify_sparse_csr((3,), (3,), (3, 5), (2, 3, 4))
    assert dense_tail_mismatch.error_kind == "unusable_dense"
    tensor = _sparse_call(
        torch.sparse_csr_tensor,
        _long([0, 2, 3]),
        _long([0, 1, 1]),
        _values((3, 5)),
        size=(2, 3, 4),
        check_invariants=True,
    )
    with pytest.raises(RuntimeError):
        tensor.to_dense()


@pytest.mark.slow
def test_sparse_layouts_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SparseLayouts"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_sparse_layouts_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.SparseLayouts"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_SparseLayoutsAxCheck.lean")
    body = "import TensorGuard.SparseLayouts\n" + "\n".join(f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_SparseLayoutsAxCheck.lean"],
            cwd=_LEAN,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "sorryAx" not in proc.stdout
    seen = set()
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", proc.stdout):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                seen.add(name)
    assert not (seen - _TRUSTED_AXIOMS), proc.stdout
