"""Step 146 — **SMT dtype-matmul encoding faithfulness**, machine-checked in Lean
and cross-checked against the real torch matmul dispatcher.

``mm``/``bmm``/``matmul`` require identical operand dtypes.  The verifier emits a
dtype-equality SMT formula and reports a bug iff it is UNSAT.
``lean/TensorGuard/SmtEncoding.lean`` proves this encoding **faithful**: it is
UNSAT iff the dtypes differ (``dtype_smt_unsat_iff_ne``) and, for two *known*
dtypes, the solver verdict coincides exactly with the abstract ``dtMatmulBug``
(``dtype_smt_matches_dtMatmulBug``); equal dtypes are SAT (``dtype_same_sat``).

This test:

1. guards the Lean proof (lexical ``sorry`` guard; import + axiom-audit wiring;
   toolchain-gated build + axiom audit);
2. mirrors ``dtMatmulBug`` and the UNSAT predicate in Python;
3. replays **every** known-dtype pair on **real torch** ``a @ b`` and asserts the
   live dispatcher **raises iff** the encoding is UNSAT (i.e. iff
   ``dtMatmulBug`` flags) over the supported tested subset ``{f32,f64,i64,i32}``;
4. exercises the **real Z3 solver** on a dtype ``EnumSort`` with the verifier's
   pinned equality constraint, asserting UNSAT iff the dtypes differ — the
   generic encoding the Lean theorem proves faithful, run on the live solver.
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
    "TensorGuard.SmtEncoding.dtype_smt_unsat_iff_ne",
    "TensorGuard.SmtEncoding.dtype_smt_matches_dtMatmulBug",
    "TensorGuard.SmtEncoding.dtype_same_sat",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the abstract rule and the UNSAT predicate.
def _dt_matmul_bug(a, b):
    if a == "unknown" or b == "unknown":
        return False
    return a != b


def _unsat(a, b):
    # the pinned dtype-equality formula is UNSAT iff the dtypes differ
    return a != b


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.SmtEncoding" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Python-mirror faithfulness (known dtypes: UNSAT == dtMatmulBug)
# --------------------------------------------------------------------------- #
def test_mirror_unsat_iff_dtmatmulbug_on_known():
    known = ["f32", "f64", "i64", "i32"]
    for a in known:
        for b in known:
            assert _unsat(a, b) == _dt_matmul_bug(a, b)


def test_mirror_unknown_never_flags():
    for d in ["f32", "f64", "i64", "i32", "unknown"]:
        assert _dt_matmul_bug("unknown", d) is False
        assert _dt_matmul_bug(d, "unknown") is False


# --------------------------------------------------------------------------- #
# 3. Live cross-check against the real torch matmul dispatcher.
# --------------------------------------------------------------------------- #
_DT_NAMES = ["f32", "f64", "i64", "i32"]  # supported, tested dtype subset


def _torch_matmul_raises(a, b):
    torch = pytest.importorskip("torch")
    dt = {"f32": torch.float32, "f64": torch.float64,
          "i64": torch.int64, "i32": torch.int32}
    x = torch.ones(2, 2, dtype=dt[a])
    y = torch.ones(2, 2, dtype=dt[b])
    try:
        x @ y
        return False
    except RuntimeError:
        return True


def test_matmul_raises_iff_unsat_against_real_torch():
    # For the supported tested dtype subset {f32,f64,i64,i32}, torch matmul
    # raises iff the dtype-equality encoding is UNSAT iff dtMatmulBug flags.
    checked = 0
    for a, b in itertools.product(_DT_NAMES, repeat=2):
        raised = _torch_matmul_raises(a, b)
        assert raised == _unsat(a, b) == _dt_matmul_bug(a, b), (a, b)
        checked += 1
    assert checked == 16


def test_equal_dtype_never_raises_against_real_torch():
    for a in _DT_NAMES:
        assert _torch_matmul_raises(a, a) is False


# --------------------------------------------------------------------------- #
# 3b. Live cross-check against the **real Z3 solver**: the generic enum-equality
#     encoding the Lean theorem proves faithful, instantiated on a dtype sort.
# --------------------------------------------------------------------------- #
def test_dtype_equality_encoding_unsat_iff_differ_real_z3():
    z3 = pytest.importorskip("z3")
    DtSort, consts = z3.EnumSort("Dt", _DT_NAMES)
    a = z3.Const("dt_a", DtSort)
    b = z3.Const("dt_b", DtSort)
    # The matmul dtype constraint the verifier emits: the two operand dtypes
    # are pinned and required equal.
    for ca, cb in itertools.product(consts, consts):
        s = z3.Solver()
        s.add(a == ca)
        s.add(b == cb)
        s.add(a == b)
        unsat = s.check() == z3.unsat
        # Lean dtype_smt_unsat_iff_ne / dtype_smt_matches_dtMatmulBug:
        # UNSAT iff the (known) dtypes differ == dtMatmulBug.
        differ = str(ca) != str(cb)
        assert unsat == differ == _dt_matmul_bug(str(ca), str(cb)), (ca, cb)


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.SmtEncoding"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.SmtEncoding"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_SmtDtypeAxCheck.lean")
    body = "import TensorGuard.SmtEncoding\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_SmtDtypeAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
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
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
