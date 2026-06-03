"""Step 242 — SMT faithfulness for shape and dtype constraints.

``lean/TensorGuard/SmtEncoding.lean`` now proves that the formulas TensorGuard
hands to Z3 for four high-value constraint families are faithful to the abstract
transfer rules:

* broadcast: the disjunctive equality-or-unit formula is SAT iff ``bcDim`` accepts;
* reshape/view ``-1``: the product/divisibility formula is SAT iff the reshape
  guard accepts;
* split/chunk partitioning: the section-sum formula is SAT iff the axis is
  reconstructed;
* dtype promotion: a finite dtype table (and its chain fold) is SAT iff the
  claimed output dtype equals the promotion transfer.

This test guards the Lean proof wiring and cross-checks every family against the
real Z3 solver plus TensorGuard's Python helpers / live torch where applicable.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import shutil
import subprocess

import pytest

from src.smt.reshape_theory import check_reshape_compatible
from src.tensor_shapes import (
    TensorShape,
    compute_broadcast_shape,
    compute_chunk_shapes,
    compute_reshape_shape,
    compute_split_shapes,
)

z3 = pytest.importorskip("z3")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "SmtEncoding.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.SmtEncoding.broadcast_smt_sat_iff_bcDim_some",
    "TensorGuard.SmtEncoding.broadcast_smt_unsat_iff_bcDim_none",
    "TensorGuard.SmtEncoding.broadcast_smt_unsat_iff_incompatible",
    "TensorGuard.SmtEncoding.divisibility_smt_sat_iff_reshapeValid",
    "TensorGuard.SmtEncoding.divisibility_smt_unsat_iff_invalid",
    "TensorGuard.SmtEncoding.partition_smt_sat_iff_sum_eq",
    "TensorGuard.SmtEncoding.partition_smt_matches_splitSectionsValid",
    "TensorGuard.SmtEncoding.partition_smt_unsat_iff_mismatch",
    "TensorGuard.SmtEncoding.dtype_promote_smt_sat_iff",
    "TensorGuard.SmtEncoding.dtype_promote_smt_unsat_iff_mismatch",
    "TensorGuard.SmtEncoding.dtype_promote_chain_smt_sat_iff",
    "TensorGuard.SmtEncoding.dtype_promote_chain_smt_unsat_iff_mismatch",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _dims(shape: TensorShape) -> tuple:
    return tuple(dim.value for dim in shape.dims)


def _prod(xs: list[int] | tuple[int, ...]) -> int:
    return math.prod(xs)


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited_for_step_242():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean"), encoding="utf-8") as fh:
        assert "import TensorGuard.SmtEncoding" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean"), encoding="utf-8") as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_step_242_has_no_sorry_or_admit():
    with open(_FILE, encoding="utf-8") as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Broadcast: Lean model == real Z3 == TensorGuard helper == torch.
# --------------------------------------------------------------------------- #
def _bc_dim(a: int, b: int) -> int | None:
    if a == 1:
        return b
    if b == 1:
        return a
    if a == b:
        return a
    return None


def _broadcast_z3(a: int, b: int) -> tuple[bool, int | None]:
    av, bv, out = z3.Ints("bc_a bc_b bc_out")
    solver = z3.Solver()
    solver.add(av == a, bv == b, out >= 1)
    solver.add(
        z3.Or(
            z3.And(av == 1, out == bv),
            z3.And(bv == 1, out == av),
            z3.And(av == bv, out == av),
        )
    )
    sat = solver.check() == z3.sat
    return sat, solver.model().evaluate(out).as_long() if sat else None


def _torch_broadcast_dim(a: int, b: int) -> int | None:
    torch = pytest.importorskip("torch")
    try:
        return torch.broadcast_shapes((a,), (b,))[0]
    except RuntimeError:
        return None


def test_broadcast_formula_matches_z3_helper_and_torch():
    checked = 0
    for a, b in itertools.product(range(1, 6), repeat=2):
        model = _bc_dim(a, b)
        sat, out = _broadcast_z3(a, b)
        helper = compute_broadcast_shape(TensorShape.from_tuple((a,)), TensorShape.from_tuple((b,)))
        live = _torch_broadcast_dim(a, b)

        assert sat == (model is not None), (a, b)
        assert out == model
        assert (_dims(helper) if helper is not None else None) == (
            (model,) if model is not None else None
        )
        assert live == model
        checked += 1
    assert checked == 25


# --------------------------------------------------------------------------- #
# 3. Reshape divisibility: Lean model == real Z3 == TensorGuard helper == torch.
# --------------------------------------------------------------------------- #
def _reshape_valid(total: int, known: list[int]) -> bool:
    known_prod = _prod(known)
    return known_prod > 0 and total % known_prod == 0


def _reshape_divisibility_z3(total: int, known: list[int]) -> bool:
    known_prod = _prod(known)
    inferred = z3.Int("reshape_inferred")
    solver = z3.Solver()
    solver.add(z3.BoolVal(known_prod > 0))
    # All tested TensorGuard dimensions are positive, so adding inferred >= 1 is
    # equivalent to the Lean divisibility theorem while matching PyTorch's runtime
    # reshape contract.
    solver.add(inferred >= 1)
    solver.add(known_prod * inferred == total)
    return solver.check() == z3.sat


def _torch_reshape_accepts(shape: tuple[int, ...], known: list[int]) -> bool:
    torch = pytest.importorskip("torch")
    x = torch.arange(_prod(shape)).reshape(shape)
    try:
        x.reshape(*known, -1)
        return True
    except RuntimeError:
        return False


def test_divisibility_formula_matches_z3_helpers_and_torch():
    cases = [
        ((2, 3, 4), [6]),
        ((2, 3, 4), [4]),
        ((2, 3, 4), [5]),
        ((8, 5), [10]),
        ((10,), [3]),
        ((3, 5, 7), [21]),
        ((3, 5, 7), [22]),
    ]
    for shape, known in cases:
        total = _prod(shape)
        model = _reshape_valid(total, known)
        new_dims = tuple(known + [-1])
        ts = TensorShape.from_tuple(shape)

        assert _reshape_divisibility_z3(total, known) is model
        assert (check_reshape_compatible(ts, new_dims) is None) is model
        assert (compute_reshape_shape(ts, new_dims) is not None) is model
        assert _torch_reshape_accepts(shape, known) is model


# --------------------------------------------------------------------------- #
# 4. Partition reconstruction: Lean model == real Z3 == split/chunk helpers.
# --------------------------------------------------------------------------- #
def _partition_z3(axis_size: int, sections: list[int]) -> bool:
    axis, recon = z3.Ints("partition_axis partition_recon")
    solver = z3.Solver()
    solver.add(axis == axis_size)
    solver.add(recon == sum(sections))
    solver.add(axis == recon)
    return solver.check() == z3.sat


def test_split_partition_formula_matches_z3_helper_and_torch():
    torch = pytest.importorskip("torch")
    cases = [
        ((2, 5), [2, 0, 3], -1),
        ((2, 5), [2, 0, 2], -1),
        ((4, 9), [4, 5], -1),
        ((4, 9), [4, 4], -1),
        ((2, 0), [0, 0], -1),
    ]
    for shape, sections, dim in cases:
        axis = shape[dim]
        model = sum(sections) == axis
        assert _partition_z3(axis, sections) is model
        assert (compute_split_shapes(TensorShape.from_tuple(shape), sections, dim) is not None) is model
        x = torch.zeros(shape)
        if model:
            pieces = torch.split(x, sections, dim=dim)
            assert sum(piece.shape[dim] for piece in pieces) == axis
            assert tuple(torch.cat(pieces, dim=dim).shape) == shape
        else:
            with pytest.raises(RuntimeError):
                torch.split(x, sections, dim=dim)


def test_chunk_partition_formula_matches_z3_helper_and_torch():
    torch = pytest.importorskip("torch")
    for shape, chunks, dim in [((2, 10, 4), 3, 1), ((5, 4), 8, 0), ((0, 4), 3, 0)]:
        static = compute_chunk_shapes(TensorShape.from_tuple(shape), chunks, dim)
        assert static is not None
        sections = [_dims(s)[dim] for s in static]
        axis = shape[dim]
        assert _partition_z3(axis, sections) is True

        x = torch.zeros(shape)
        pieces = list(torch.chunk(x, chunks, dim=dim))
        assert [piece.shape[dim] for piece in pieces] == sections
        assert tuple(torch.cat(pieces, dim=dim).shape) == shape


# --------------------------------------------------------------------------- #
# 5. Dtype promotion: Lean table == real Z3 finite relation == torch.
# --------------------------------------------------------------------------- #
_ALL_DTYPES = ["f16", "bf16", "f32", "f64", "i32", "i64", "bool", "unknown"]
_RANK = {"f64": 7, "f32": 6, "bf16": 5, "f16": 4, "i64": 3, "i32": 2, "bool": 1}
_DT_SORT, _DT_CONST_LIST = z3.EnumSort("DtPromoteStep242", _ALL_DTYPES)
_DT_CONSTS = dict(zip(_ALL_DTYPES, _DT_CONST_LIST))


def _dt_promote(a: str, b: str) -> str:
    if a == b:
        return a
    if a == "unknown" or b == "unknown":
        return "unknown"
    return a if _RANK[a] >= _RANK[b] else b


def _promote_run(acc: str, chain: list[str]) -> str:
    cur = acc
    for dtype in chain:
        cur = _dt_promote(cur, dtype)
    return cur


def _promote_relation(consts, left, right, out):
    return z3.Or(
        *[
            z3.And(left == consts[a], right == consts[b], out == consts[_dt_promote(a, b)])
            for a, b in itertools.product(_ALL_DTYPES, repeat=2)
        ]
    )


def _dtype_chain_z3(acc: str, chain: list[str], claimed: str) -> bool:
    consts = _DT_CONSTS
    solver = z3.Solver()
    cur = z3.Const("dt_acc_0", consts[acc].sort())
    solver.add(cur == consts[acc])
    for idx, operand in enumerate(chain):
        nxt = z3.Const(f"dt_acc_{idx + 1}", consts[acc].sort())
        solver.add(_promote_relation(consts, cur, consts[operand], nxt))
        cur = nxt
    solver.add(cur == consts[claimed])
    return solver.check() == z3.sat


def test_dtype_promotion_chain_formula_matches_z3_relation():
    checked = 0
    for acc in _ALL_DTYPES:
        for chain in [
            [],
            ["i32"],
            ["i32", "f32"],
            ["bool", "i64", "f64"],
            ["bf16", "f16"],
            ["f32", "unknown", "i64"],
        ]:
            expected = _promote_run(acc, chain)
            for claimed in _ALL_DTYPES:
                assert _dtype_chain_z3(acc, chain, claimed) is (claimed == expected), (
                    acc,
                    chain,
                    claimed,
                    expected,
                )
                checked += 1
    assert checked > 0


_TORCH_DTYPES = {
    "f64": "float64",
    "f32": "float32",
    "i64": "int64",
    "i32": "int32",
    "bool": "bool",
}


def _torch_promote_run(acc: str, chain: list[str]) -> str:
    torch = pytest.importorskip("torch")
    cur = getattr(torch, _TORCH_DTYPES[acc])
    for dtype in chain:
        cur = torch.promote_types(cur, getattr(torch, _TORCH_DTYPES[dtype]))
    reverse = {getattr(torch, name): key for key, name in _TORCH_DTYPES.items()}
    return reverse[cur]


def test_dtype_promotion_formula_matches_real_torch_supported_subset():
    alphabet = list(_TORCH_DTYPES)
    checked = 0
    for acc in alphabet:
        for length in range(0, 4):
            for chain in itertools.product(alphabet, repeat=length):
                chain = list(chain)
                expected = _promote_run(acc, chain)
                assert _torch_promote_run(acc, chain) == expected
                assert _dtype_chain_z3(acc, chain, expected) is True
                checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# 6. Toolchain-gated Lean build + axiom audit.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds_step_242_smt_encoding():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SmtEncoding"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_step_242_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.SmtEncoding"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stdout[-3000:] + build.stderr[-3000:]

    check = os.path.join(_LEAN, "_Step242SmtAxCheck.lean")
    body = "import TensorGuard.SmtEncoding\n" + "\n".join(
        f"#print axioms {thm}" for thm in _THEOREMS
    ) + "\n"
    with open(check, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_Step242SmtAxCheck.lean"],
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
    out = proc.stdout
    assert "sorryAx" not in out
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
