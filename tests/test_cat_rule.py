"""Step 148 — **concatenation (`torch.cat`) shape rule**, machine-checked in Lean
and cross-checked against the real torch cat engine.

``lean/TensorGuard/CatRule.lean`` models an operand's shape relative to the
concat axis as ``(pre, axisSize, post)`` and proves the rule the verifier relies
on: cat is admitted **iff** the non-axis dims coincide (``catValid_iff``), the
output axis is the **sum** of the operands' axis sizes (``catAxis_value``),
**numel is additive** under compatibility (``prod_cat``), and the axis sum is
commutative / associative (``cat_axis_comm`` / ``cat_assoc``).

This test mirrors the Lean model in Python and replays it on **real tensors** via
``torch.cat``: compatible operands produce the predicted shape with additive
numel, and an incompatible (non-axis dim differs) pair makes torch **raise** —
exactly when the Lean ``catValid`` flags ``false``.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "CatRule.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.CatRule.prod_append",
    "TensorGuard.CatRule.catValid_iff",
    "TensorGuard.CatRule.catValid_pre_mismatch",
    "TensorGuard.CatRule.catAxis_value",
    "TensorGuard.CatRule.prod_cat",
    "TensorGuard.CatRule.cat_axis_comm",
    "TensorGuard.CatRule.cat_assoc",
    "TensorGuard.CatRule.cat_zero_right",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


# Python mirror of the Lean model (split a shape at `axis`).
def _split(shape, axis):
    return list(shape[:axis]), shape[axis], list(shape[axis + 1:])


def _cat_valid(a, b, axis):
    apre, _, apost = _split(a, axis)
    bpre, _, bpost = _split(b, axis)
    return apre == bpre and apost == bpost


def _cat_shape(a, b, axis):
    apre, aax, apost = _split(a, axis)
    _, bax, _ = _split(b, axis)
    return apre + [aax + bax] + apost


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.CatRule" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch cat engine.
# --------------------------------------------------------------------------- #
def test_cat_compatible_matches_real_torch():
    torch = pytest.importorskip("torch")
    checked = 0
    cases = [
        ((2, 3, 4), (5, 3, 4), 0),
        ((2, 3, 4), (2, 6, 4), 1),
        ((2, 3, 4), (2, 3, 7), 2),
        ((4,), (3,), 0),
        ((2, 3), (2, 0), 1),  # zero-length axis (cat_zero_right)
    ]
    for a, b, axis in cases:
        assert _cat_valid(a, b, axis)
        y = torch.cat([torch.zeros(a), torch.zeros(b)], dim=axis)
        predicted = _cat_shape(a, b, axis)
        assert list(y.shape) == predicted, (a, b, axis)
        # numel additivity (prod_cat)
        assert y.numel() == _prod(a) + _prod(b)
        # axis additivity (catAxis_value)
        assert y.shape[axis] == a[axis] + b[axis]
        checked += 1
    assert checked > 0


def test_cat_incompatible_raises_iff_invalid():
    torch = pytest.importorskip("torch")
    # Non-axis dim differs => Lean catValid is false => torch must raise.
    bad = [
        ((2, 3, 4), (2, 3, 5), 1),  # post dim (4 vs 5) differs
        ((2, 3, 4), (3, 3, 4), 1),  # pre dim (2 vs 3) differs
        ((2, 3), (3, 3), 1),
    ]
    for a, b, axis in bad:
        assert not _cat_valid(a, b, axis)
        with pytest.raises(RuntimeError):
            torch.cat([torch.zeros(a), torch.zeros(b)], dim=axis)


def test_cat_commutativity_and_associativity_axis():
    torch = pytest.importorskip("torch")
    a, b, c, axis = (2, 3), (2, 4), (2, 5), 1
    yab = torch.cat([torch.zeros(a), torch.zeros(b)], dim=axis)
    yba = torch.cat([torch.zeros(b), torch.zeros(a)], dim=axis)
    # axis sizes equal (cat_axis_comm); other dims identical
    assert yab.shape[axis] == yba.shape[axis] == a[axis] + b[axis]
    left = torch.cat([torch.cat([torch.zeros(a), torch.zeros(b)], dim=axis),
                      torch.zeros(c)], dim=axis)
    right = torch.cat([torch.zeros(a),
                       torch.cat([torch.zeros(b), torch.zeros(c)], dim=axis)],
                      dim=axis)
    assert list(left.shape) == list(right.shape)  # cat_assoc


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.CatRule"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.CatRule"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_CatRuleAxCheck.lean")
    body = "import TensorGuard.CatRule\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_CatRuleAxCheck.lean"],
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
