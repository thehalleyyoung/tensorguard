"""Step 136 — **cross-domain (shape × device) encoding faithfulness**,
machine-checked in Lean and cross-checked against the *real* Z3 encoder.

When an op spans the shape and device algebras the verifier emits a cross-domain
constraint set (``model_checker.py::_Z3Context.encode_cross_domain_constraint``):
a **device-transfer** op preserves the *shape* (adds ``shape_pre[i]==shape_post[i]``,
leaves the device free); a **non-transfer** op preserves the *device* (adds
``dev_pre==dev_post``, leaves the shape free).

``lean/TensorGuard/CrossDomain.lean`` proves that encoder faithful: the emitted
conjunction is satisfiable iff the component the op preserves is actually
preserved, and the other component is left genuinely free.

This test runs the **real** ``encode_cross_domain_constraint`` through Z3 on
concrete shape/device endpoints and asserts the live SAT/UNSAT verdict equals the
Lean prediction.
"""

import itertools
import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "CrossDomain.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.CrossDomain.transfer_sat_iff_shape_eq",
    "TensorGuard.CrossDomain.transfer_unsat_iff_shape_ne",
    "TensorGuard.CrossDomain.transfer_device_free",
    "TensorGuard.CrossDomain.nontransfer_sat_iff_dev_eq",
    "TensorGuard.CrossDomain.nontransfer_unsat_iff_dev_ne",
    "TensorGuard.CrossDomain.nontransfer_shape_free",
    "TensorGuard.CrossDomain.branch_selects_preserved",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_exists_imported_and_audited():
    assert os.path.exists(_FILE), "CrossDomain.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.CrossDomain" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real Z3 cross-domain encoder.
# --------------------------------------------------------------------------- #
z3 = pytest.importorskip("z3")
from src.model_checker import _Z3Context  # noqa: E402

_SHAPES = [(2, 4), (2, 5), (3, 4)]


def _sat(constraints, *pins):
    s = z3.Solver()
    for c in constraints:
        s.add(c)
    for p in pins:
        s.add(p)
    return s.check() == z3.sat


def test_transfer_preserves_shape_unsat_iff_shape_differs():
    ctx = _Z3Context()
    pre = [z3.Int(f"sp_{i}") for i in range(2)]
    post = [z3.Int(f"sq_{i}") for i in range(2)]
    dpre, dpost = ctx.DEV_CPU, ctx.DEV_CUDA0
    cons = ctx.encode_cross_domain_constraint(pre, post, dpre, dpost, True)
    for sa, sb in itertools.product(_SHAPES, _SHAPES):
        pins = [pre[i] == sa[i] for i in range(2)] + [
            post[i] == sb[i] for i in range(2)]
        sat = _sat(cons, *pins)
        # Lean transfer_sat_iff_shape_eq: SAT iff shapes equal.
        assert sat == (sa == sb), (sa, sb, sat)


def test_transfer_leaves_device_free():
    ctx = _Z3Context()
    pre = [z3.Int(f"tp_{i}") for i in range(2)]
    post = [z3.Int(f"tq_{i}") for i in range(2)]
    devs = list(ctx.device_consts)
    # Shapes equal ⇒ any device pair must remain satisfiable (device free).
    for da, db in itertools.product(devs, devs):
        cons = ctx.encode_cross_domain_constraint(pre, post, da, db, True)
        pins = [pre[i] == 2 for i in range(2)] + [post[i] == 2 for i in range(2)]
        assert _sat(cons, *pins), (da, db)


def test_nontransfer_preserves_device_unsat_iff_device_differs():
    ctx = _Z3Context()
    pre = [z3.Int(f"np_{i}") for i in range(2)]
    post = [z3.Int(f"nq_{i}") for i in range(2)]
    da_var = z3.Const("ndev_a", ctx.DeviceSort)
    db_var = z3.Const("ndev_b", ctx.DeviceSort)
    cons = ctx.encode_cross_domain_constraint(pre, post, da_var, db_var, False)
    devs = list(ctx.device_consts)
    for da, db in itertools.product(devs, devs):
        sat = _sat(cons, da_var == da, db_var == db)
        # Lean nontransfer_sat_iff_dev_eq: SAT iff devices equal.
        assert sat == (str(da) == str(db)), (da, db, sat)


def test_nontransfer_leaves_shape_free():
    ctx = _Z3Context()
    pre = [z3.Int(f"fp_{i}") for i in range(2)]
    post = [z3.Int(f"fq_{i}") for i in range(2)]
    da = ctx.DEV_CPU
    cons = ctx.encode_cross_domain_constraint(pre, post, da, da, False)
    # Device equal ⇒ any shape pair stays satisfiable (shape free).
    for sa, sb in itertools.product(_SHAPES, _SHAPES):
        pins = [pre[i] == sa[i] for i in range(2)] + [
            post[i] == sb[i] for i in range(2)]
        assert _sat(cons, *pins), (sa, sb)


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.CrossDomain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.CrossDomain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_CrossDomainAxCheck.lean")
    body = "import TensorGuard.CrossDomain\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_CrossDomainAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
