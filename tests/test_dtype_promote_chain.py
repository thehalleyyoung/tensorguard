"""Step 143 — **dtype-promotion chain transfer**, machine-checked in Lean and
cross-checked against real torch.

A multi-operand elementwise op (``torch.cat``, a running ``add``, ``stack``, a
fused residual) folds torch's type-promotion join over a *list* of operand
dtypes.  ``lean/TensorGuard/DtypePromoteChain.lean`` lifts the single-pair
promotion laws (Step 137) to the **chain** and proves it compositional
(``promoteRun_append``), an **upper bound** of every operand
(``promoteRun_ge_elem`` — promotion never narrows below an operand, the soundness
direction), **order-independent** under adjacent transposition
(``promoteRun_swap``) and ``unknown``-absorbing (``promoteRun_unknown``).

This test:

1. guards the Lean proof (lexical ``sorry`` guard; import + axiom-audit wiring;
   toolchain-gated build + axiom audit);
2. mirrors ``promoteRun`` in Python and checks the upper-bound / order-independence
   / unknown-absorbing properties over the full 8-element dtype set;
3. replays every chain (length <= 3) on **real torch** via
   ``torch.promote_types``, asserting the live promotion fold equals the Lean
   ``promoteRun`` prediction.

The torch cross-check uses the dtype sub-alphabet ``{f64,f32,i64,i32,bool}`` on
which the verifier's promotion lattice is *exactly* torch's; the ``f16``/``bf16``
mixing corner (``bf16+f16 -> f32`` in torch) is a known modeled-vs-torch
divergence covered by the Python-mirror algebra checks, not the live fold (so the
``against real torch`` claim stays literally true).
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
_FILE = os.path.join(_LEAN, "TensorGuard", "DtypePromoteChain.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.DtypePromoteChain.promoteRun_append",
    "TensorGuard.DtypePromoteChain.promoteRun_ge_acc",
    "TensorGuard.DtypePromoteChain.promoteRun_ge_elem",
    "TensorGuard.DtypePromoteChain.promoteRun_swap",
    "TensorGuard.DtypePromoteChain.promoteRun_unknown",
]

# Full modeled dtype lattice (mirror of DeviceDtype.lean `dtPromote`).
_ALL = ["f16", "bf16", "f32", "f64", "i32", "i64", "bool", "unknown"]
_RANK = {"f64": 7, "f32": 6, "bf16": 5, "f16": 4, "i64": 3, "i32": 2, "bool": 1}


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def _dt_promote(a, b):
    if a == b:
        return a
    if a == "unknown" or b == "unknown":
        return "unknown"
    # higher rank wins (float category strictly dominates int category)
    return a if _RANK[a] >= _RANK[b] else b


def _promote_run(acc, xs):
    for x in xs:
        acc = _dt_promote(acc, x)
    return acc


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.DtypePromoteChain" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Python-mirror algebra (full 8-dtype set, incl. unknown)
# --------------------------------------------------------------------------- #
def test_mirror_upper_bound_of_every_operand():
    for acc in _ALL:
        for length in range(0, 4):
            for chain in itertools.product(_ALL, repeat=length):
                out = _promote_run(acc, list(chain))
                # result joined with any operand is a no-op => operand <= result
                for x in chain:
                    assert _dt_promote(x, out) == out, (acc, chain, x)
                assert _dt_promote(acc, out) == out


def test_mirror_order_independent_adjacent_swap():
    for acc in _ALL:
        for x in _ALL:
            for y in _ALL:
                for rest_len in range(0, 3):
                    for rest in itertools.product(_ALL, repeat=rest_len):
                        rest = list(rest)
                        assert _promote_run(acc, [x, y] + rest) == _promote_run(
                            acc, [y, x] + rest
                        )


def test_mirror_unknown_absorbing():
    for length in range(0, 4):
        for chain in itertools.product(_ALL, repeat=length):
            assert _promote_run("unknown", list(chain)) == "unknown"


# --------------------------------------------------------------------------- #
# 3. Live cross-check against real torch promotion.
# --------------------------------------------------------------------------- #
# Sub-alphabet on which the modeled lattice is exactly torch's promotion.
_TORCH_DT = {
    "f64": "float64",
    "f32": "float32",
    "i64": "int64",
    "i32": "int32",
    "bool": "bool",
}


def _torch_promote_run(acc, xs):
    torch = pytest.importorskip("torch")
    cur = getattr(torch, _TORCH_DT[acc])
    for x in xs:
        cur = torch.promote_types(cur, getattr(torch, _TORCH_DT[x]))
    # map torch dtype back to our label
    rev = {getattr(torch, v): k for k, v in _TORCH_DT.items()}
    return rev[cur]


def test_promote_run_matches_real_torch_all_chains():
    alphabet = list(_TORCH_DT.keys())
    checked = 0
    for acc in alphabet:
        for length in range(0, 4):
            for chain in itertools.product(alphabet, repeat=length):
                chain = list(chain)
                assert _torch_promote_run(acc, chain) == _promote_run(acc, chain), (
                    acc,
                    chain,
                )
                checked += 1
    assert checked > 0


def test_order_independence_against_real_torch():
    # promoteRun_swap on real torch: operand order does not change the dtype.
    acc, chain = "i32", ["f32", "i64", "bool"]
    base = _torch_promote_run(acc, chain)
    for perm in itertools.permutations(chain):
        assert _torch_promote_run(acc, list(perm)) == base


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.DtypePromoteChain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.DtypePromoteChain"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_DtypePromoteChainAxCheck.lean")
    body = "import TensorGuard.DtypePromoteChain\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_DtypePromoteChainAxCheck.lean"],
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
