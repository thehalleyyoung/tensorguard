"""Step 132 — close known-unsoundness gap U2 (SAFE-on-infeasible) with a
Lean-checked fix.

Two halves are guarded here:

1. **The real code fix** (`src/shape_cegar.py`): when the accumulated refined
   predicates are jointly infeasible, the shape-CEGAR loop must *abstain* rather
   than report SAFE. We assert the new terminal status
   `CEGARStatus.INFEASIBLE_REFINEMENT` exists, is **not** safe, and maps to
   `CEGARVerdict.UNKNOWN`; and that the soundness contract now records U2 as
   *closed* (still surfaced, with a Lean reference), while keeping U2 in the
   list.

2. **The Lean proof** (`lean/TensorGuard/CegarInfeasible.lean`): a fast lexical
   `sorry` guard is always on; the build / axiom-audit checks are
   toolchain-gated and skip when `lake` is absent. The proof shows the abstaining
   decision is sound under the feasible-branch guarantee
   (`decideNew_safeSound`) and the old SAFE behaviour is unsound
   (`decideOld_unsound`).
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "CegarInfeasible.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.CegarU2.decideNew_safeSound",
    "TensorGuard.CegarU2.decideOld_unsound",
    "TensorGuard.CegarU2.fix_abstains_on_infeasible",
    "TensorGuard.CegarU2.fix_keeps_safe_when_feasible",
    "TensorGuard.CegarU2.old_always_safe",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# 1. The real code fix
# --------------------------------------------------------------------------- #
def test_infeasible_status_exists_and_is_not_safe():
    from src.shape_cegar import CEGARStatus, CEGARVerdict, ShapeCEGARResult

    assert "INFEASIBLE_REFINEMENT" in CEGARStatus.__members__

    r = ShapeCEGARResult(final_status=CEGARStatus.INFEASIBLE_REFINEMENT)
    # The whole point of U2: this must NOT be reported as safe.
    assert r.is_safe is False
    assert r.has_real_bugs is False
    # And it abstains rather than claiming SAFE/UNSAFE/TIMEOUT.
    assert r.verdict is CEGARVerdict.UNKNOWN


def test_old_safe_on_infeasible_return_is_gone():
    # The buggy literal `CEGARStatus.SAFE` return on the infeasible branch must
    # no longer exist; the branch now returns INFEASIBLE_REFINEMENT.
    src = open(os.path.join(_ROOT, "src", "shape_cegar.py")).read()
    # Locate the feasibility guard and verify the nearby return is the new one.
    idx = src.index("check_feasibility(self.pred_set.predicates)")
    window = src[idx: idx + 600]
    assert "CEGARStatus.INFEASIBLE_REFINEMENT" in window
    assert "CEGARStatus.SAFE" not in window


def test_contract_marks_u2_closed_but_keeps_it():
    from src import soundness_contract as sc

    by_id = {g.id: g for g in sc.KNOWN_UNSOUNDNESS}
    assert "U1" in by_id and "U2" in by_id, "U2 must stay surfaced"

    u2 = by_id["U2"]
    assert u2.status == "closed"
    assert "Step 132" in u2.closed_by
    assert "CegarInfeasible.lean" in u2.closed_by
    # The remediation should reference the machine-checked proof.
    assert "CegarInfeasible.lean" in u2.remediation
    # U1 remains open (only U2 is closed here).
    assert by_id["U1"].status == "open"


def test_rendered_contract_in_sync():
    from src import soundness_contract as sc

    rendered = sc.render_markdown()
    path = os.path.join(_ROOT, "SOUNDNESS_CONTRACT.md")
    assert open(path).read().strip() == rendered.strip(), (
        "SOUNDNESS_CONTRACT.md out of sync — regenerate with "
        "`python -m src.soundness_contract > SOUNDNESS_CONTRACT.md`"
    )
    # Status column present and U2 shown closed.
    assert "| ID | Status |" in rendered
    assert "U2 | closed" in rendered


def test_soundness_boundary_serializes_status():
    from reproducibility import soundness_boundary as sb

    data = sb.measure()
    gaps = {g["id"]: g for g in data["contract"]["known_unsoundness_gaps"]}
    assert gaps["U2"]["status"] == "closed"
    assert "Step 132" in gaps["U2"]["closed_by"]
    assert gaps["U1"]["status"] == "open"


# --------------------------------------------------------------------------- #
# 2. The Lean proof
# --------------------------------------------------------------------------- #
def test_lean_file_exists_imported_and_audited():
    assert os.path.exists(_FILE), "CegarInfeasible.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.CegarInfeasible" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.CegarInfeasible"],
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
        ["lake", "build", "TensorGuard.CegarInfeasible"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_CegarU2AxCheck.lean")
    body = "import TensorGuard.CegarInfeasible\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_CegarU2AxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a U2 proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
