"""Step 126 — machine-checked reduced-product transfer functions (Lean).

Guards the new ``lean/TensorGuard/ReducedProduct.lean`` formalization that lifts
the Lean soundness story from the *operator* transfer functions to the
**reduced-product** abstract domain that ``src/domains/product.py`` implements.

* a fast, always-on lexical guard that no ``sorry`` survives in the proof file
  and that the model mirrors the Python reductions, and
* slow, toolchain-gated checks that the module compiles under the Lean kernel
  and that every reduced-product theorem is axiom-clean (no ``sorryAx``; only
  trusted kernel axioms), exactly like the Step 87 core-soundness audit.
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_RP = os.path.join(_LEAN, "TensorGuard", "ReducedProduct.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_RP_THEOREMS = [
    "TensorGuard.RP.reduceTagNul_reductive",
    "TensorGuard.RP.reduceNulTag_reductive",
    "TensorGuard.RP.reduce_reductive",
    "TensorGuard.RP.pmeet_le_left",
    "TensorGuard.RP.pmeet_le_right",
    "TensorGuard.RP.pmeet_mono",
    "TensorGuard.RP.reduce_mono_consistent",
    "TensorGuard.RP.gamma_mono",
    "TensorGuard.RP.pmeet_gamma",
    "TensorGuard.RP.reduce_gamma",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def test_reduced_product_file_exists_and_imported():
    assert os.path.exists(_RP), "ReducedProduct.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.ReducedProduct" in fh.read()


def test_no_sorry_in_reduced_product():
    with open(_RP) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\bsorry\b", code), "`sorry` found in ReducedProduct.lean"


def test_model_mirrors_python_reductions():
    # The Lean model must name the two reductions that src/domains/product.py has.
    with open(_RP) as fh:
        text = fh.read()
    assert "reduceTagNul" in text and "reduceNulTag" in text
    # And the reductive property must be stated for the full reduction.
    assert "reduce_reductive" in text


def test_axiom_audit_covers_reduced_product():
    audit = os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")
    with open(audit) as fh:
        text = fh.read()
    for thm in _RP_THEOREMS:
        assert f"#print axioms {thm}" in text, f"audit missing {thm}"


@pytest.mark.slow
def test_reduced_product_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.ReducedProduct"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_reduced_product_theorems_axiom_clean(tmp_path):
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_RPAxCheck.lean")
    body = "import TensorGuard.ReducedProduct\n" + "\n".join(
        f"#print axioms {t}" for t in _RP_THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_RPAxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a reduced-product proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _RP_THEOREMS:
        assert f"'{thm}'" in out, f"audit output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
