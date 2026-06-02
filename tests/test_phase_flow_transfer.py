"""Step 141 — **train/eval phase chain transfer**, machine-checked in Lean and
cross-checked against the real torch module-mode machinery.

``lean/TensorGuard/PhaseFlow.lean`` models the ``training`` bit a module carries
through a sequence of mode setters: ``keep`` propagates the bit,
``setTrain``/``setEval`` set it (absorbing, last setter wins).  It proves the
chain is compositional (``phaseRun_append``), a setter is absorbing
(``run_after_setter``, ``run_ends_at_value``) and phase is preserved on the
setter-free fragment (``run_noSetter_id``).

This test mirrors ``phaseStep``/``phaseRun`` in Python and replays **every**
chain on a real ``nn.Module`` (which recurses into children), asserting
``module.training`` equals the Lean ``phaseRun`` prediction — so the proved
transfer holds against the live torch mode machinery, including child
propagation.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "PhaseFlow.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.PhaseFlow.keep_id",
    "TensorGuard.PhaseFlow.setTrain_true",
    "TensorGuard.PhaseFlow.setEval_false",
    "TensorGuard.PhaseFlow.setter_absorbing",
    "TensorGuard.PhaseFlow.phaseRun_append",
    "TensorGuard.PhaseFlow.run_after_setter",
    "TensorGuard.PhaseFlow.run_ends_at_value",
    "TensorGuard.PhaseFlow.run_noSetter_id",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer.
def _phase_step(b, op):
    return {"keep": b, "setTrain": True, "setEval": False}[op]


def _phase_run(b0, ops):
    b = b0
    for op in ops:
        b = _phase_step(b, op)
    return b


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.PhaseFlow" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch module-mode machinery.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


def _make_module():
    # A module with children, so .train()/.eval() exercise recursion.
    return nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1), nn.ReLU())


def _torch_run(b0, ops):
    m = _make_module()
    m.train(b0)
    for op in ops:
        if op == "keep":
            _ = m  # phase-agnostic: no mode change
        elif op == "setTrain":
            m.train()
        elif op == "setEval":
            m.eval()
    return m.training, m


def test_phase_run_matches_real_module_all_chains():
    ops_alphabet = ["keep", "setTrain", "setEval"]
    checked = 0
    for b0 in (True, False):
        for length in range(0, 4):
            for chain in itertools.product(ops_alphabet, repeat=length):
                chain = list(chain)
                training, m = _torch_run(b0, chain)
                predicted = _phase_run(b0, chain)
                assert training == predicted, (b0, chain)
                # child modules track the same phase (.train()/.eval() recurse)
                for child in m.modules():
                    assert child.training == predicted, (b0, chain, child)
                checked += 1
    assert checked == 2 * sum(3 ** n for n in range(4))


def test_setter_free_chain_preserves_phase():
    # run_noSetter_id: a chain of only `keep` ops preserves the phase.
    for b0 in (True, False):
        training, _ = _torch_run(b0, ["keep", "keep", "keep"])
        assert training == _phase_run(b0, ["keep", "keep", "keep"]) == b0


def test_last_setter_wins():
    # run_after_setter / run_ends_at_value: the final phase is the last setter.
    training, _ = _torch_run(True, ["setEval", "setTrain", "setEval"])
    predicted = _phase_run(True, ["setEval", "setTrain", "setEval"])
    assert predicted is False
    assert training == predicted


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.PhaseFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.PhaseFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_PhaseFlowAxCheck.lean")
    body = "import TensorGuard.PhaseFlow\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_PhaseFlowAxCheck.lean"],
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
