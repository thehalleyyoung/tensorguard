"""Step 142 — **reduction rank-transfer chain**, machine-checked in Lean and
cross-checked against the real torch shape machinery.

``lean/TensorGuard/RankTransfer.lean`` models how a tensor's *rank* changes under
a chain of reductions: ``reduceKeep`` (``keepdim=True``) preserves rank,
``reduceDrop`` (``keepdim=False``) lowers it by one (truncated at 0).  It proves
the chain is compositional (``rankRun_append``), **monotone non-increasing**
(``rankRun_le`` — a reduction never invents a dimension), equals the input rank
minus the number of ``keepdim=False`` reductions (``rankRun_eq_sub_countDrop``),
and is exact on no-underflow chains (``rankRun_exact``).

This test mirrors ``rankStep``/``rankRun`` in Python and replays **every**
no-underflow chain on a real tensor via ``torch.sum(dim=0, keepdim=...)``,
asserting ``out.dim()`` equals the Lean ``rankRun`` prediction — so the proved
transfer holds against the live torch shape engine.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "RankTransfer.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.RankTransfer.keep_id",
    "TensorGuard.RankTransfer.drop_pred",
    "TensorGuard.RankTransfer.step_le",
    "TensorGuard.RankTransfer.rankRun_append",
    "TensorGuard.RankTransfer.rankRun_eq_sub_countDrop",
    "TensorGuard.RankTransfer.rankRun_le",
    "TensorGuard.RankTransfer.rankRun_allKeep",
    "TensorGuard.RankTransfer.rankRun_exact",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer.
def _rank_step(r, op):
    return r if op == "reduceKeep" else max(r - 1, 0)


def _rank_run(r0, ops):
    r = r0
    for op in ops:
        r = _rank_step(r, op)
    return r


def _count_drop(ops):
    return sum(1 for op in ops if op == "reduceDrop")


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.RankTransfer" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch shape machinery.
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def _torch_run(r0, ops):
    # A real tensor of rank r0 (each dim size 2). Reduce dim 0 each step.
    cur = torch.ones(*([2] * r0)) if r0 > 0 else torch.tensor(1.0)
    for op in ops:
        keepdim = op == "reduceKeep"
        cur = cur.sum(dim=0, keepdim=keepdim)
    return cur.dim()


def _no_underflow(r0, ops):
    """True iff every reduceDrop happens at rank >= 1 (a legal torch reduction)."""
    r = r0
    for op in ops:
        if op == "reduceDrop":
            if r < 1:
                return False
            r -= 1
    return True


def test_rank_run_matches_real_torch_all_chains():
    ops_alphabet = ["reduceKeep", "reduceDrop"]
    checked = 0
    for r0 in range(0, 4):
        for length in range(0, 4):
            for chain in itertools.product(ops_alphabet, repeat=length):
                chain = list(chain)
                if not _no_underflow(r0, chain):
                    continue
                assert _torch_run(r0, chain) == _rank_run(r0, chain), (r0, chain)
                checked += 1
    assert checked > 0


def test_closed_form_matches_real_torch():
    # rankRun_eq_sub_countDrop on a no-underflow chain.
    r0, chain = 3, ["reduceDrop", "reduceKeep", "reduceDrop"]
    assert _no_underflow(r0, chain)
    out = _torch_run(r0, chain)
    assert out == r0 - _count_drop(chain) == _rank_run(r0, chain)


def test_keepdim_only_preserves_rank():
    # rankRun_allKeep
    r0, chain = 3, ["reduceKeep", "reduceKeep"]
    assert _torch_run(r0, chain) == r0 == _rank_run(r0, chain)


def test_reductions_never_raise_rank():
    # rankRun_le (monotone non-increasing)
    for r0 in range(0, 4):
        for length in range(0, 4):
            for chain in itertools.product(["reduceKeep", "reduceDrop"], repeat=length):
                assert _rank_run(r0, list(chain)) <= r0


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.RankTransfer"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.RankTransfer"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_RankTransferAxCheck.lean")
    body = "import TensorGuard.RankTransfer\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_RankTransferAxCheck.lean"],
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
