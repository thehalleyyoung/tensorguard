"""Step 150 — **reshape/`view` `-1`-inference rule**, machine-checked in Lean and
cross-checked against the real torch reshape engine and the verifier's
``compute_reshape_shape``.

``lean/TensorGuard/ReshapeInfer.lean`` proves the inferred ``-1`` dim equals
``numel / ∏ known`` (``inferDim_spec``), that reconstituting the shape conserves
numel (``prod_reshape_valid``), and that the reshape is admitted **iff**
``∏ known`` is positive and divides numel (``reshapeValid_iff`` /
``nondivisible_flagged``).

This test mirrors the Lean model in Python and replays it on **real tensors** via
``x.reshape(known…, -1)`` and against ``compute_reshape_shape``: the inferred dim
and total numel match the Lean prediction, and a **non-dividing specification
makes torch raise** — exactly when the Lean guard flags it.
"""

import os
import re
import shutil
import subprocess

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "ReshapeInfer.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.ReshapeInfer.prod_append",
    "TensorGuard.ReshapeInfer.reshapeValid_iff",
    "TensorGuard.ReshapeInfer.reshapeValid_imp_dvd",
    "TensorGuard.ReshapeInfer.nondivisible_flagged",
    "TensorGuard.ReshapeInfer.inferDim_spec",
    "TensorGuard.ReshapeInfer.prod_reshape_valid",
    "TensorGuard.ReshapeInfer.reshape_infer_position",
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


# Python mirror of the Lean model.
def _reshape_valid(total, known):
    kp = _prod(known)
    return kp > 0 and total % kp == 0


def _infer_dim(total, known):
    return total // _prod(known)


def _reshape_shape(total, known):
    return list(known) + [_infer_dim(total, known)]


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.ReshapeInfer" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry():
    with open(_FILE) as fh:
        assert not re.search(r"\bsorry\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch reshape engine.
# --------------------------------------------------------------------------- #
def test_reshape_infer_matches_real_torch():
    torch = pytest.importorskip("torch")
    checked = 0
    # (input shape, known specified dims for view(known…, -1))
    cases = [
        ((2, 3, 4), [6]),
        ((2, 3, 4), [4]),
        ((2, 3, 4), [2, 2]),
        ((12,), [3]),
        ((8, 5), [10]),
    ]
    for shape, known in cases:
        total = _prod(shape)
        assert _reshape_valid(total, known)
        x = torch.arange(total).reshape(shape)
        y = x.reshape(*known, -1)
        predicted = _reshape_shape(total, known)
        assert list(y.shape) == predicted, (shape, known)
        # inference correctness (inferDim_spec) and numel preservation
        assert y.shape[-1] == _infer_dim(total, known)
        assert y.numel() == total
        checked += 1
    assert checked > 0


def test_reshape_nondivisible_raises_iff_invalid():
    torch = pytest.importorskip("torch")
    # ∏ known does not divide numel => Lean guard flags => torch raises.
    bad = [((2, 3, 4), [5]), ((10,), [3]), ((2, 3), [4])]
    for shape, known in bad:
        total = _prod(shape)
        assert not _reshape_valid(total, known)
        x = torch.arange(total).reshape(shape)
        with pytest.raises(RuntimeError):
            x.reshape(*known, -1)


def test_reshape_matches_verifier_compute_reshape_shape():
    pytest.importorskip("torch")
    from src.tensor_shapes import ShapeDim, TensorShape, compute_reshape_shape

    cases = [((2, 3, 4), (6, -1)), ((2, 3, 4), (4, -1)), ((12,), (3, -1)),
             ((8, 5), (10, -1))]
    for shape, new_dims in cases:
        total = _prod(shape)
        known = [d for d in new_dims if d != -1]
        ts = TensorShape(tuple(ShapeDim(int(d)) for d in shape))
        res = compute_reshape_shape(ts, new_dims)
        assert res is not None
        got = [d.value for d in res.dims]
        assert got == _reshape_shape(total, known), (shape, new_dims, got)
        assert _prod(got) == total  # numel preserved


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.ReshapeInfer"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.ReshapeInfer"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_ReshapeInferAxCheck.lean")
    body = "import TensorGuard.ReshapeInfer\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_ReshapeInferAxCheck.lean"],
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
