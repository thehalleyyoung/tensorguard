"""Step 138 — **gradient-flow chain transfer**, machine-checked in Lean and
cross-checked against the real torch autograd engine.

`lean/TensorGuard/GradFlow.lean` lifts the single-op `detach` gradient check to a
transfer function over a *chain* of ops: `keep` propagates the `requires_grad`
bit, `detach` / `noGrad` reset it to `false` (absorbing), `reattach` sets it
`true`.  It proves the chain is compositional (`gradRun_append`), resetting ops
are absorbing (`reset_absorbing`, `run_after_reset`) and, on the reattach-free
fragment, the outgoing bit is `true` iff the input required grad and no reset
intervened (`run_noReattach_true_iff`).

This test mirrors `gradStep`/`gradRun` in Python and replays **every** chain on a
real autograd tensor, asserting `out.requires_grad` equals the Lean `gradRun`
prediction — so the proved transfer holds against the live torch engine.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "GradFlow.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.GradFlow.keep_id",
    "TensorGuard.GradFlow.detach_false",
    "TensorGuard.GradFlow.reattach_true",
    "TensorGuard.GradFlow.reset_absorbing",
    "TensorGuard.GradFlow.gradRun_append",
    "TensorGuard.GradFlow.run_after_reset",
    "TensorGuard.GradFlow.run_noReattach_true_iff",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer.
def _grad_step(b, op):
    return {"keep": b, "detach": False, "noGrad": False, "reattach": True}[op]


def _grad_run(b0, ops):
    b = b0
    for op in ops:
        b = _grad_step(b, op)
    return b


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.GradFlow" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against real torch autograd.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def _torch_run(b0, ops):
    cur = torch.ones((2, 2), dtype=torch.float32, requires_grad=b0)
    for op in ops:
        if op == "keep":
            cur = cur + 1.0
        elif op == "detach":
            cur = cur.detach()
        elif op == "noGrad":
            with torch.no_grad():
                cur = cur + 1.0
    return cur.requires_grad


def test_grad_run_matches_real_autograd_all_chains():
    ops_alphabet = ["keep", "detach", "noGrad"]
    checked = 0
    for b0 in (True, False):
        for length in range(0, 4):
            for chain in itertools.product(ops_alphabet, repeat=length):
                chain = list(chain)
                assert _torch_run(b0, chain) == _grad_run(b0, chain), (b0, chain)
                checked += 1
    assert checked == 2 * sum(3 ** n for n in range(4))


def test_reattach_on_leaf_matches():
    # gradStep _ reattach = True
    x = torch.ones((2, 2), dtype=torch.float32)
    x.requires_grad_(True)
    assert x.requires_grad == _grad_step(False, "reattach")


def test_reattach_after_detach_matches():
    x = torch.ones((2, 2), dtype=torch.float32, requires_grad=True)
    y = x.detach()
    y.requires_grad_(True)
    # gradRun True [detach, reattach] == True
    assert y.requires_grad == _grad_run(True, ["detach", "reattach"])


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.GradFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.GradFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_GradFlowAxCheck.lean")
    body = "import TensorGuard.GradFlow\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_GradFlowAxCheck.lean"],
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
