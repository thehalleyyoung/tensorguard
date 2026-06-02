"""Step 144 — **broadcast dim-chain transfer**, machine-checked in Lean and
cross-checked against real torch broadcasting.

``lean/TensorGuard/BroadcastChain.lean`` models torch's per-dimension broadcast
folded across a chain of operand sizes (``a + b + c``,
``torch.broadcast_shapes``).  It proves the rule **commutative** (``bcDim_comm``),
``1`` a two-sided identity (``bcDim_one_left/right``), the compatible broadcast
size exactly their ``max`` (``bcDim_compat_max``), and **refutation soundness** —
the rule flags (``= none``) iff the two sizes are genuinely incompatible
(``bcDim_none_iff``); plus the chain laws (``bcRun_append``, ``bcRun_none``).

This test:

1. guards the Lean proof (lexical ``sorry`` guard; import + axiom-audit wiring;
   toolchain-gated build + axiom audit);
2. mirrors ``bcDim``/``bcRun`` in Python;
3. replays **every** single-dim chain on **real torch** via
   ``torch.broadcast_shapes``, asserting the live broadcaster returns ``max`` when
   ``bcRun`` is ``some`` and **raises** exactly when ``bcRun`` is ``none``.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "BroadcastChain.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.BroadcastChain.bcDim_comm",
    "TensorGuard.BroadcastChain.bcDim_one_left",
    "TensorGuard.BroadcastChain.bcDim_one_right",
    "TensorGuard.BroadcastChain.bcDim_self",
    "TensorGuard.BroadcastChain.bcDim_compat_max",
    "TensorGuard.BroadcastChain.bcDim_none_iff",
    "TensorGuard.BroadcastChain.bcRun_append",
    "TensorGuard.BroadcastChain.bcRun_none",
    "TensorGuard.BroadcastChain.bcRun_ones",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer (None == incompatibility).
def _bc_dim(a, b):
    if a == 1:
        return b
    if b == 1:
        return a
    if a == b:
        return a
    return None


def _bc_run(acc, xs):
    for x in xs:
        if acc is None:
            return None
        acc = _bc_dim(acc, x)
    return acc


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.BroadcastChain" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Python-mirror algebra
# --------------------------------------------------------------------------- #
def test_mirror_commutative_and_identity():
    for a in range(1, 6):
        for b in range(1, 6):
            assert _bc_dim(a, b) == _bc_dim(b, a)
        assert _bc_dim(a, 1) == a and _bc_dim(1, a) == a


def test_mirror_compat_is_max_and_none_iff_incompatible():
    for a in range(1, 6):
        for b in range(1, 6):
            out = _bc_dim(a, b)
            if a == 1 or b == 1 or a == b:
                assert out == max(a, b)
            else:
                assert out is None


# --------------------------------------------------------------------------- #
# 3. Live cross-check against real torch broadcasting.
# --------------------------------------------------------------------------- #
def _torch_bc_run(acc, xs):
    """Fold torch broadcasting over a chain; return resulting size or None if it
    raises (an incompatibility)."""
    torch = pytest.importorskip("torch")
    cur = (acc,)
    for x in xs:
        try:
            cur = torch.broadcast_shapes(cur, (x,))
        except RuntimeError:
            return None
    return cur[0]


def test_bc_run_matches_real_torch_all_chains():
    sizes = [1, 2, 3]
    checked = 0
    for acc in sizes:
        for length in range(0, 4):
            for chain in itertools.product(sizes, repeat=length):
                chain = list(chain)
                model = _bc_run(acc, chain)
                live = _torch_bc_run(acc, chain)
                assert model == live, (acc, chain, model, live)
                checked += 1
    assert checked > 0


def test_incompatibility_raises_in_real_torch():
    # bcDim_none_iff: 2 vs 3 is incompatible -> torch raises, model is None.
    assert _bc_run(2, [3]) is None
    assert _torch_bc_run(2, [3]) is None


def test_order_independence_against_real_torch():
    # bcDim_comm: operand order does not change the broadcast size.
    chain = [1, 4, 1]
    base = _torch_bc_run(1, chain)
    for perm in itertools.permutations(chain):
        assert _torch_bc_run(1, list(perm)) == base


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.BroadcastChain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.BroadcastChain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_BroadcastChainAxCheck.lean")
    body = "import TensorGuard.BroadcastChain\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_BroadcastChainAxCheck.lean"],
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
