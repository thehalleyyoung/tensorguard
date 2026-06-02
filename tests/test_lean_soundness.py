"""
Step 87 — machine-checked soundness of the core transfer functions.

These tests are the regression guard behind the README's "sorry-free Lean build"
claim and behind the formal soundness story in
``docs/formalization/type_system.md``:

* the whole ``lean/TensorGuard`` library compiles under ``lake build`` (so every
  soundness proof is accepted by the Lean kernel), and
* an ``#print axioms`` audit of the core transfer-function soundness theorems
  shows they depend on **no** ``sorryAx`` and only on the trusted kernel axioms
  ``{propext, Classical.choice, Quot.sound}``.

The Lean build is genuinely expensive, so the build/audit tests are marked
``slow`` and skip cleanly when the Lean toolchain (``lake``) is not installed.
A fast, always-on lexical check asserts no real ``sorry`` token survives in the
imported proof files.
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")

# Trusted Lean kernel axioms. Anything outside this set — most importantly
# ``sorryAx`` — would mean a proof was not actually closed.
_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

# Core transfer-function soundness theorems audited for axiom cleanliness.
_AUDITED_THEOREMS = [
    "TensorGuard.applyOp_sound_linear",
    "TensorGuard.applyOp_sound_view",
    "TensorGuard.applyOp_sound_broadcast_add",
    "TensorGuard.applyOpExt_sound_matmul",
    "TensorGuard.applyOpExt_sound_transpose",
    "TensorGuard.applyOpExt_sound_permute",
    "TensorGuard.applyOpExt_sound_sum_reduce",
    "TensorGuard.V5.applyOp_sound_cross_entropy",
    "TensorGuard.V5.applyOp_sound_argmax",
    "TensorGuard.MatmulSound.matmul_contraction_sound",
    "TensorGuard.BroadcastAddSound.broadcast_add_total",
]


def _imported_lean_files():
    """The proof files actually reachable from the ``TensorGuard`` root."""
    root = os.path.join(_LEAN, "TensorGuard.lean")
    with open(root) as fh:
        imports = re.findall(r"import\s+TensorGuard\.(\w+)", fh.read())
    files = [os.path.join(_LEAN, "TensorGuard", f"{name}.lean") for name in imports]
    return [p for p in files if os.path.exists(p)]


def _strip_comments(src: str) -> str:
    # Remove block comments /- ... -/ (Lean comments do not nest arbitrarily for
    # our purposes) and line comments -- ... .
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def test_root_imports_all_soundness_modules():
    names = {os.path.basename(p) for p in _imported_lean_files()}
    for required in ("Soundness.lean", "MatmulSound.lean",
                     "BroadcastAddSound.lean", "SoundnessV5.lean",
                     "AssumeGuaranteeExtended.lean"):
        assert required in names, f"{required} not imported by TensorGuard root"


def test_no_real_sorry_in_imported_proofs():
    """Fast lexical guard: no ``sorry`` token outside comments/strings."""
    offenders = []
    for path in _imported_lean_files():
        with open(path) as fh:
            code = _strip_comments(fh.read())
        if re.search(r"\bsorry\b", code):
            offenders.append(os.path.relpath(path, _ROOT))
    assert not offenders, f"`sorry` found in proof code of: {offenders}"


def test_audit_file_covers_core_transfer_functions():
    audit = os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")
    assert os.path.exists(audit), "AxiomAudit.lean missing"
    with open(audit) as fh:
        text = fh.read()
    for thm in _AUDITED_THEOREMS:
        assert f"#print axioms {thm}" in text, f"audit missing {thm}"


@pytest.mark.slow
def test_lean_library_builds():
    """The whole soundness library compiles under the Lean kernel."""
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"lake build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.mark.slow
def test_core_theorems_are_axiom_clean():
    """`#print axioms` of every core soundness theorem omits `sorryAx` and
    contains only trusted kernel axioms."""
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    # Ensure the library is built so the audit file's imports resolve.
    build = subprocess.run(
        ["lake", "build", "TensorGuard"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
    env = dict(os.environ, LEAN_PATH=lean_path)
    proc = subprocess.run(
        ["lake", "env", "lean", "-R", ".", "TensorGuard/AxiomAudit.lean"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
    )
    assert proc.returncode == 0, (
        f"axiom audit failed to elaborate:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout

    assert "sorryAx" not in out, f"a soundness proof depends on sorryAx:\n{out}"

    # Every reported axiom must be in the trusted set.
    reported = set(re.findall(r"\b([A-Za-z_][\w.]*)\b", out))
    # Pull only the axiom names that appear in '[ ... ]' lists.
    axiom_lists = re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out)
    seen = set()
    for lst in axiom_lists:
        for name in (s.strip() for s in lst.split(",")):
            if name:
                seen.add(name)
    illegal = seen - _TRUSTED_AXIOMS
    assert not illegal, f"untrusted axioms in soundness proofs: {illegal}\n{out}"

    # Sanity: we actually audited every theorem we listed.
    for thm in _AUDITED_THEOREMS:
        assert f"'{thm}'" in out, f"audit output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
