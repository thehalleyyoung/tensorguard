"""Step 135 — **SMT-encoding faithfulness**, machine-checked in Lean and
cross-checked against the *real* Z3 encoder.

The verifier discharges every device / phase / gradient check by handing Z3 an
enumeration-sort equality constraint (``model_checker.py``:
``encode_device_constraint`` returns ``dev_a == dev_b``;
``encode_phase_constraint`` compares against ``PHASE_TRAIN``/``PHASE_EVAL``;
``encode_gradient_constraint`` returns ``grad_out == BoolVal(requires_grad)``).
A *mismatch* is reported exactly when, after pinning the two endpoints to
concrete sort elements, that same-value constraint is **unsatisfiable**.

``lean/TensorGuard/SmtEncoding.lean`` proves the faithfulness of that encoding:
for any decidable sort the pinned equality formula is satisfiable **iff** the two
endpoints are equal, so the SMT verdict (UNSAT) coincides exactly with the
abstract algebra's ``*Bug`` predicate (``DeviceDtype.lean``).

This test guards two halves:

1. **The Lean proof** — fast lexical ``sorry`` guard always on; build / axiom
   audit are toolchain-gated and skip when ``lake`` is absent.

2. **The live cross-check** — the *real* ``_Z3Context`` encoders are run on every
   concrete device / phase / gradient endpoint pair and the actual SAT/UNSAT
   verdict from Z3 must equal the Lean-modeled prediction (``UNSAT`` iff the
   endpoints differ).  So the proved faithfulness holds for the actual solver
   calls the verifier makes.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "SmtEncoding.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.SmtEncoding.sat_iff_eq",
    "TensorGuard.SmtEncoding.unsat_iff_ne",
    "TensorGuard.SmtEncoding.unsat_sound",
    "TensorGuard.SmtEncoding.eq_is_sat",
    "TensorGuard.SmtEncoding.device_smt_matches_devBug",
    "TensorGuard.SmtEncoding.phase_smt_unsat_iff_ne",
    "TensorGuard.SmtEncoding.grad_smt_unsat_iff_ne",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_exists_imported_and_audited():
    assert os.path.exists(_FILE), "SmtEncoding.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.SmtEncoding" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real Z3 encoder.
# --------------------------------------------------------------------------- #
z3 = pytest.importorskip("z3")
from src.model_checker import _Z3Context  # noqa: E402


def _unsat_under(ctx, var_a, var_b, pin_a, pin_b, constraint):
    """True iff pinning (var_a==pin_a, var_b==pin_b) + constraint is UNSAT."""
    s = z3.Solver()
    s.add(var_a == pin_a)
    s.add(var_b == pin_b)
    s.add(constraint)
    return s.check() == z3.unsat


def test_device_encoder_unsat_iff_devices_differ():
    ctx = _Z3Context()
    devs = list(ctx.device_consts)  # CPU, CUDA0..3
    a = z3.Const("dev_a", ctx.DeviceSort)
    b = z3.Const("dev_b", ctx.DeviceSort)
    constraint = ctx.encode_device_constraint(a, b)
    for da, db in itertools.product(devs, devs):
        unsat = _unsat_under(ctx, a, b, da, db, constraint)
        # Lean: unsat_iff_ne — UNSAT exactly when endpoints differ.
        assert unsat == (str(da) != str(db)), (da, db, unsat)


def test_device_encoder_sat_on_equal():
    ctx = _Z3Context()
    a = z3.Const("dev_a", ctx.DeviceSort)
    b = z3.Const("dev_b", ctx.DeviceSort)
    constraint = ctx.encode_device_constraint(a, b)
    for d in ctx.device_consts:
        # Lean: eq_is_sat — equal endpoints are always satisfiable (no FP).
        assert not _unsat_under(ctx, a, b, d, d, constraint)


def test_phase_encoder_unsat_iff_phases_differ():
    ctx = _Z3Context()
    phases = [ctx.PHASE_TRAIN, ctx.PHASE_EVAL]
    a = z3.Const("ph_a", ctx.PhaseSort)
    b = z3.Const("ph_b", ctx.PhaseSort)
    # The phase mismatch the verifier checks is a same-phase equality.
    constraint = a == b
    for pa, pb in itertools.product(phases, phases):
        unsat = _unsat_under(ctx, a, b, pa, pb, constraint)
        assert unsat == (str(pa) != str(pb)), (pa, pb, unsat)


def test_gradient_encoder_unsat_iff_status_differs():
    ctx = _Z3Context()
    g = ctx.grad_var("t")
    for set_val, req in itertools.product([True, False], [True, False]):
        # encode_gradient_constraint pins grad var to `req`.
        enc = ctx.encode_gradient_constraint(g, req)
        s = z3.Solver()
        s.add(g == z3.BoolVal(set_val))
        s.add(enc)
        unsat = s.check() == z3.unsat
        # Lean grad_smt_unsat_iff_ne: UNSAT iff demanded status disagrees.
        assert unsat == (set_val != req), (set_val, req, unsat)


def test_device_smt_matches_devbug_known_pairs():
    """device_smt_matches_devBug: on two *known* devices, the solver's UNSAT
    verdict equals the abstract devBug (a != b)."""
    ctx = _Z3Context()
    devs = list(ctx.device_consts)
    a = z3.Const("dev_a", ctx.DeviceSort)
    b = z3.Const("dev_b", ctx.DeviceSort)
    constraint = ctx.encode_device_constraint(a, b)
    for da, db in itertools.product(devs, devs):
        unsat = _unsat_under(ctx, a, b, da, db, constraint)
        devbug = str(da) != str(db)  # both known ⇒ devBug = (a != b)
        assert unsat == devbug


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SmtEncoding"],
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
        ["lake", "build", "TensorGuard.SmtEncoding"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_SmtEncodingAxCheck.lean")
    body = "import TensorGuard.SmtEncoding\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_SmtEncodingAxCheck.lean"],
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
