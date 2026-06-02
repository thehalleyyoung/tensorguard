"""Step 137 — **dtype promotion lattice laws**, machine-checked in Lean and
cross-checked against real torch.

The elementwise dtype transfer (`add`/`cat`/…) never raises in torch because
torch *type-promotes* its operands.  `lean/TensorGuard/DeviceDtype.lean` models
that promotion join `dtPromote` and now proves it a well-defined semilattice
join: **commutative** (`dtPromote_comm`), **idempotent** (`dtPromote_idem`),
**associative** (`dtPromote_assoc`) and with `unknown` **absorbing**
(`dtPromote_unknown_absorbs_left/right`) — exactly the algebra that justifies the
order-independent, never-flagging elementwise dtype transfer
(`dtElementwiseBug ≡ false`).

This test guards:

1. **The Lean proof** — fast lexical ``sorry`` guard; toolchain-gated build +
   axiom audit.

2. **A Python mirror** of `dtPromote` is checked to be commutative / idempotent /
   associative / unknown-absorbing over the full 8-element dtype set.

3. **Real torch** — for every pair of *concretely promotable* dtypes a real
   ``torch.add`` is executed and asserted **not to raise**, validating the
   verifier's "elementwise dtype never flags" claim against the live dispatcher.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "DeviceDtype.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.DevDtype.dtPromote_comm",
    "TensorGuard.DevDtype.dtPromote_idem",
    "TensorGuard.DevDtype.dtPromote_assoc",
    "TensorGuard.DevDtype.dtPromote_unknown_absorbs_left",
    "TensorGuard.DevDtype.dtPromote_unknown_absorbs_right",
]

# Abstract dtype set mirroring the Lean `Dt` inductive.
_DT = ["f16", "bf16", "f32", "f64", "i32", "i64", "bool", "unknown"]


def _dt_promote(a, b):
    """Python mirror of Lean `dtPromote`."""
    if a == b:
        return a
    if a == "unknown" or b == "unknown":
        return "unknown"
    for cand in ("f64", "f32", "bf16", "f16", "i64", "i32"):
        if a == cand or b == cand:
            return cand
    return "bool"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Python mirror obeys the proved semilattice laws.
# --------------------------------------------------------------------------- #
def test_mirror_commutative_idempotent():
    for a in _DT:
        assert _dt_promote(a, a) == a
        for b in _DT:
            assert _dt_promote(a, b) == _dt_promote(b, a)


def test_mirror_associative():
    for a, b, c in itertools.product(_DT, _DT, _DT):
        assert _dt_promote(_dt_promote(a, b), c) == _dt_promote(a, _dt_promote(b, c))


def test_mirror_unknown_absorbing():
    for a in _DT:
        assert _dt_promote("unknown", a) == "unknown"
        assert _dt_promote(a, "unknown") == "unknown"


# --------------------------------------------------------------------------- #
# 3. Real torch: elementwise add of promotable dtypes never raises.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")

_TORCH_DT = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i32": torch.int32,
    "i64": torch.int64,
    "bool": torch.bool,
}


def test_real_torch_elementwise_add_never_raises():
    concrete = [k for k in _DT if k != "unknown"]
    checked = 0
    for a, b in itertools.product(concrete, concrete):
        ta = torch.ones((2, 2), dtype=_TORCH_DT[a])
        tb = torch.ones((2, 2), dtype=_TORCH_DT[b])
        # The verifier's `dtElementwiseBug ≡ false` claims this never errors.
        out = ta + tb
        # torch's actual result dtype is its own promotion; we only assert the
        # op is total (no raise), which is what soundness of "never flag" needs.
        assert out.shape == (2, 2)
        checked += 1
    assert checked == len(concrete) ** 2


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.DeviceDtype"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.DeviceDtype"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_DtPromoteAxCheck.lean")
    body = "import TensorGuard.DeviceDtype\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_DtPromoteAxCheck.lean"],
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
