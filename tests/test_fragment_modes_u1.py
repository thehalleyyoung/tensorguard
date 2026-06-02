"""Step 133 — re-audit known-unsoundness U1 (the verifiable-fragment boundary is
**mode-dependent**) with a Lean-backed argument cross-checked against real code.

Two halves are guarded here:

1. **The Lean proof** (`lean/TensorGuard/FragmentModes.lean`): a fast lexical
   `sorry` guard is always on; build / axiom-audit checks are toolchain-gated and
   skip when `lake` is absent. The proof models the three-mode terminal decision
   and shows `sound` mode is sound (`sound_safeSound`), `balanced`/`heuristic`
   are unsound on a fragment violation hiding a bug (`balanced_unsound`,
   `heuristic_unsound`), the modes agree in-fragment and differ exactly on a
   fragment violation.

2. **The live cross-check**: a Python mirror of the Lean `decide` function is run
   against the **real** verifier (`verify_architecture`) over the curated
   boundary probes in `reproducibility/soundness_boundary.py`. Every observed
   three-mode verdict must equal what the Lean model predicts — i.e. the proved
   `modes_agree_in_fragment` / `modes_differ_iff_violation` predictions actually
   hold on real PyTorch modules.
"""

import os
import re
import shutil
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "FragmentModes.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.FragmentU1.sound_safeSound",
    "TensorGuard.FragmentU1.balanced_unsound",
    "TensorGuard.FragmentU1.heuristic_unsound",
    "TensorGuard.FragmentU1.modes_agree_in_fragment",
    "TensorGuard.FragmentU1.modes_differ_iff_violation",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# Python mirror of the Lean `decide` model (Verdict in verifier vocabulary).
# --------------------------------------------------------------------------- #
def _lean_decide(mode: str, out_of_fragment: bool, core_found_bug: bool) -> str:
    """Mirror of TensorGuard.FragmentU1.decide, in CEGARVerdict vocabulary."""
    if core_found_bug:
        return "UNSAFE"            # Verdict.bug
    if out_of_fragment:
        return "UNKNOWN" if mode == "sound" else "SAFE"  # abstain vs safe
    return "SAFE"                  # Verdict.safe


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_exists_imported_and_audited():
    assert os.path.exists(_FILE), "FragmentModes.lean missing"
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.FragmentModes" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check: Lean model predicts the real verifier's three-mode
#    verdicts on every boundary probe.
# --------------------------------------------------------------------------- #
def test_lean_model_matches_live_verifier_on_probes():
    from reproducibility import soundness_boundary as sb

    probes = sb._probes()
    # Each probe carries (out_of_fragment, core_found_bug) implicitly via its
    # region; classify from the documented region string.
    for p in probes:
        out_of_fragment = "fragment boundary" in p.region
        core_found_bug = "refutation soundness" in p.region  # the shape-bug probe
        for mode in sb.MODES:
            r = sb.verify_architecture(
                p.source, input_shapes=p.input_shapes,
                filename=f"<{p.name}>", soundness_mode=mode,
            )
            observed = _vname(r)
            predicted = _lean_decide(mode, out_of_fragment, core_found_bug)
            assert observed == predicted, (
                f"probe {p.name!r} mode {mode!r}: live verifier said "
                f"{observed}, Lean model predicted {predicted}"
            )


def _vname(r) -> str:
    v = r.verdict
    return v.name if hasattr(v, "name") else str(v)


def test_modes_agree_in_fragment_and_differ_on_violation():
    # Directly exercise the two structural theorems against the live verifier.
    from reproducibility import soundness_boundary as sb

    in_fragment = [p for p in sb._probes()
                   if "fragment boundary" not in p.region]
    violations = [p for p in sb._probes() if "fragment boundary" in p.region]
    assert in_fragment and violations

    for p in in_fragment:
        verdicts = set()
        for mode in sb.MODES:
            r = sb.verify_architecture(
                p.source, input_shapes=p.input_shapes,
                filename=f"<{p.name}>", soundness_mode=mode)
            verdicts.add(_vname(r))
        assert len(verdicts) == 1, (
            f"in-fragment probe {p.name!r} disagrees across modes: {verdicts}")

    for p in violations:
        rs = {mode: _vname(sb.verify_architecture(
                  p.source, input_shapes=p.input_shapes,
                  filename=f"<{p.name}>", soundness_mode=mode))
              for mode in sb.MODES}
        assert rs["sound"] == "UNKNOWN", (p.name, rs)
        assert rs["balanced"] == "SAFE" and rs["heuristic"] == "SAFE", (p.name, rs)
        assert rs["sound"] != rs["balanced"], (p.name, rs)


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build / axiom-audit.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.FragmentModes"],
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
        ["lake", "build", "TensorGuard.FragmentModes"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_FragmentU1AxCheck.lean")
    body = "import TensorGuard.FragmentModes\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        lean_path = os.path.join(_LEAN, ".lake", "build", "lib")
        env = dict(os.environ, LEAN_PATH=lean_path)
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_FragmentU1AxCheck.lean"],
            cwd=_LEAN, capture_output=True, text=True, timeout=900, env=env,
        )
    finally:
        if os.path.exists(check):
            os.remove(check)
    assert proc.returncode == 0, (
        f"axiom check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    out = proc.stdout
    assert "sorryAx" not in out, f"a U1 proof depends on sorryAx:\n{out}"
    for lst in re.findall(r"depends on axioms:\s*\[([^\]]*)\]", out):
        for name in (s.strip() for s in lst.split(",")):
            if name:
                assert name in _TRUSTED_AXIOMS, f"untrusted axiom {name}:\n{out}"
    for thm in _THEOREMS:
        assert f"'{thm}'" in out, f"axiom output missing {thm}:\n{out}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
