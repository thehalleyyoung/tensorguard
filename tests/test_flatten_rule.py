"""Step 147 — **flatten shape rule**, machine-checked in Lean and cross-checked
against the real torch flatten engine and the verifier's own propagator.

``lean/TensorGuard/Flatten.lean`` models ``torch.flatten(x, start, end)`` as the
prefix / span / suffix split that ``src/model_checker.py::_propagate_flatten``
implements, and proves: **numel preservation** (``prod_flatten``), the **rank
law** ``|prefix| + 1 + |suffix|`` (``length_flatten``), **full flatten** yields
``[numel]`` (``flatten_full``) and the flattened dim equals the product of the
spanned sizes (``flatten_dim_value``).

This test mirrors the Lean transform in Python and replays **every** split on a
real tensor via ``torch.flatten`` *and* against the verifier's
``_propagate_flatten``, asserting the Lean predictions (numel preserved, rank
law, exact dims) hold against the live engine.
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
_FILE = os.path.join(_LEAN, "TensorGuard", "Flatten.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.Flatten.prod_append",
    "TensorGuard.Flatten.prod_flatten",
    "TensorGuard.Flatten.length_flatten",
    "TensorGuard.Flatten.flatten_full",
    "TensorGuard.Flatten.flatten_singleton",
    "TensorGuard.Flatten.flatten_dim_value",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# --------------------------------------------------------------------------- #
# Python mirror of the Lean transform.
# --------------------------------------------------------------------------- #
def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _flatten_shape(pre, span, suf):
    return list(pre) + [_prod(span)] + list(suf)


def _split(dims, start, end):
    """Resolve negative dims and split into (prefix, span, suffix) like torch."""
    nd = len(dims)
    s = start + nd if start < 0 else start
    e = end + nd if end < 0 else end
    return dims[:s], dims[s:e + 1], dims[e + 1:]


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Flatten" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch flatten engine.
# --------------------------------------------------------------------------- #
def test_flatten_matches_real_torch_all_splits():
    torch = pytest.importorskip("torch")
    checked = 0
    shapes = [(2, 3, 4), (5, 1, 2, 3), (2, 2, 2, 2), (6,), (3, 4)]
    for shape in shapes:
        nd = len(shape)
        x = torch.arange(_prod(shape)).reshape(shape)
        for start in range(nd):
            for end in range(start, nd):
                y = torch.flatten(x, start, end)
                pre, span, suf = _split(list(shape), start, end)
                predicted = _flatten_shape(pre, span, suf)
                assert list(y.shape) == predicted, (shape, start, end)
                # numel preservation (prod_flatten)
                assert _prod(predicted) == _prod(shape) == y.numel()
                # rank law (length_flatten)
                assert len(predicted) == len(pre) + 1 + len(suf)
                checked += 1
    assert checked > 0


def test_full_flatten_is_numel_dimension():
    torch = pytest.importorskip("torch")
    for shape in [(2, 3, 4), (5, 6), (7,)]:
        x = torch.zeros(shape)
        y = torch.flatten(x)  # full flatten == start=0,end=-1
        assert list(y.shape) == [_prod(shape)]  # flatten_full


# --------------------------------------------------------------------------- #
# 3. Cross-check against the verifier's own propagator.
# --------------------------------------------------------------------------- #
def test_flatten_matches_verifier_propagator():
    pytest.importorskip("torch")
    from src.tensor_shapes import ShapeDim, TensorShape
    from src.model_checker import _propagate_flatten

    shapes = [(2, 3, 4), (5, 1, 2, 3), (2, 2, 2, 2), (3, 4)]
    for shape in shapes:
        nd = len(shape)
        ts = TensorShape(tuple(ShapeDim(int(d)) for d in shape))
        for start in range(nd):
            for end in list(range(start, nd)) + [-1]:
                pred, err = _propagate_flatten(ts, start_dim=start, end_dim=end)
                assert err is None
                pre, span, suf = _split(list(shape), start, end)
                expected = _flatten_shape(pre, span, suf)
                got = [d.value for d in pred.dims]
                assert got == expected, (shape, start, end, got, expected)
                # numel preserved (the soundness direction)
                assert _prod(got) == _prod(shape)


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.Flatten"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.Flatten"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_FlattenAxCheck.lean")
    body = "import TensorGuard.Flatten\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_FlattenAxCheck.lean"],
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
