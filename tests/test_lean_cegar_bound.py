"""Steps 129–130 — machine-checked shape-CEGAR termination & iteration bound.

Guards `lean/TensorGuard/CegarBound.lean`, the Lean mechanization of the
Houdini-style termination argument behind `src/cegar_convergence_theory.py` and
the empirical `reproducibility/cegar_convergence.py` harness:

* every productive run terminates inside a finite predicate universe
  (`cegar_terminates`), and
* obeys the tight bound ``iterations ≤ 1 + |discovered predicates|``
  (`cegar_iter_bound`) — the exact inequality the harness checks per model.

A fast lexical guard (no ``sorry``) is always on; the build / axiom-audit checks
are toolchain-gated and skip when ``lake`` is absent.
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "CegarBound.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.Cegar.length_le_lsum",
    "TensorGuard.Cegar.cegar_iter_bound",
    "TensorGuard.Cegar.cegar_terminates",
    "TensorGuard.Cegar.tight_below_naive",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def test_file_exists_and_imported():
    assert os.path.exists(_FILE), "CegarBound.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.CegarBound" in fh.read()


def test_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


def test_states_tight_bound_matching_harness():
    # The Lean file must state exactly the harness inequality and reference it.
    with open(_FILE) as fh:
        text = fh.read()
    assert "totalIters gains ≤ 1 + discovered gains" in text
    assert "cegar_convergence.py" in text


def test_axiom_audit_covers_cegar():
    audit = os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")
    with open(audit) as fh:
        text = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in text, f"audit missing {thm}"


@pytest.mark.slow
def test_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.CegarBound"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_CegarAxCheck.lean")
    body = "import TensorGuard.CegarBound\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_CegarAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a CEGAR proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"audit output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
