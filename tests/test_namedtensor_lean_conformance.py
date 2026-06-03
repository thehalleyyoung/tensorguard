"""Step 232 — Lean-checked named-tensor refine/align rules.

``lean/TensorGuard/NamedTensor.lean`` mechanizes the no-ellipsis core of
PyTorch named-tensor ``refine_names`` and ``align_to``:

* existing concrete names are preserved by ``refine_names``;
* anonymous axes can be refined to concrete names;
* ``align_to`` permutes existing named axes and inserts singleton axes for fresh
  target names or explicit ``None`` targets;
* duplicate concrete names and unnamed no-ellipsis inputs are rejected.

The concrete cases below are generated from those theorem shapes and checked
against ``src.named_tensor_verify`` plus live PyTorch named tensors.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings

import pytest

torch = pytest.importorskip("torch")

from src.named_tensor_verify import (  # noqa: E402
    NamedTensorSpec,
    verify_align_to,
    verify_refine_names,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "NamedTensor.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.NamedTensor.containsNamed_head",
    "TensorGuard.NamedTensor.unique_named_duplicate_head",
    "TensorGuard.NamedTensor.uniqueNamed_allows_repeated_anon",
    "TensorGuard.NamedTensor.refine_existing_name_preserved",
    "TensorGuard.NamedTensor.refine_rename_rejected",
    "TensorGuard.NamedTensor.refine_demotion_rejected",
    "TensorGuard.NamedTensor.refine_duplicate_requested_rejected",
    "TensorGuard.NamedTensor.refine_duplicate_current_rejected",
    "TensorGuard.NamedTensor.refine_shape_preserved",
    "TensorGuard.NamedTensor.refine_fill_anon_example",
    "TensorGuard.NamedTensor.refine_preserve_existing_example",
    "TensorGuard.NamedTensor.refine_duplicate_names_rejected",
    "TensorGuard.NamedTensor.existing_name_dim_preserved",
    "TensorGuard.NamedTensor.fresh_name_inserts_singleton",
    "TensorGuard.NamedTensor.anon_target_inserts_singleton",
    "TensorGuard.NamedTensor.align_names_preserved",
    "TensorGuard.NamedTensor.align_duplicate_target_rejected",
    "TensorGuard.NamedTensor.align_duplicate_current_rejected",
    "TensorGuard.NamedTensor.align_unnamed_input_rejected",
    "TensorGuard.NamedTensor.align_permute_example",
    "TensorGuard.NamedTensor.align_singleton_insert_example",
    "TensorGuard.NamedTensor.align_anon_target_insert_example",
    "TensorGuard.NamedTensor.align_missing_name_rejected",
    "TensorGuard.NamedTensor.align_duplicate_names_rejected",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _named(shape, names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.randn(*shape).refine_names(*names)


def _real_refine(shape, names, requested):
    try:
        y = _named(shape, names).refine_names(*requested)
        return "ok", tuple(y.shape), tuple(y.names)
    except Exception:
        return "err", None, None


def _real_align(shape, names, target):
    try:
        y = _named(shape, names).align_to(*target)
        return "ok", tuple(y.shape), tuple(y.names)
    except Exception:
        return "err", None, None


def _check_refine(shape, names, requested):
    real_status, real_shape, real_names = _real_refine(shape, names, requested)
    verdict = verify_refine_names(shape, names, requested)
    assert ("ok" if verdict.ok else "err") == real_status, verdict
    if real_status == "ok":
        assert verdict.spec == NamedTensorSpec(real_shape, real_names)


def _check_align(shape, names, target):
    real_status, real_shape, real_names = _real_align(shape, names, target)
    verdict = verify_align_to(shape, names, target)
    assert ("ok" if verdict.ok else "err") == real_status, verdict
    if real_status == "ok":
        assert verdict.spec == NamedTensorSpec(real_shape, real_names)


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.NamedTensor" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated refine_names conformance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,names,requested,expected_names",
    [
        ((2, 3), (None, None), ("N", "C"), ("N", "C")),
        ((2, 3), ("N", "C"), ("N", "C"), ("N", "C")),
        ((2, 3), ("N", "C"), ("N", "D"), None),
        ((2, 3), ("N", "C"), (None, "C"), None),
        ((2, 3), (None, None), ("N", "N"), None),
    ],
)
def test_generated_refine_cases_match_real_torch(shape, names, requested, expected_names):
    _check_refine(shape, names, requested)
    verdict = verify_refine_names(shape, names, requested)
    if expected_names is None:
        assert not verdict.ok
    else:
        assert verdict.ok
        assert verdict.spec.names == expected_names
        assert verdict.spec.shape == shape


def test_refine_duplicate_current_names_rejected_like_torch():
    _check_refine((2, 3), ("N", "N"), ("N", "N"))
    verdict = verify_refine_names((2, 3), ("N", "N"), ("N", "N"))
    assert not verdict.ok
    assert verdict.error_kind == "duplicate"


# --------------------------------------------------------------------------- #
# 3. Generated align_to conformance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,names,target,expected_shape,expected_names",
    [
        ((2, 3), ("N", "C"), ("C", "N"), (3, 2), ("C", "N")),
        ((2, 3), ("N", "C"), ("N", "H", "C"), (2, 1, 3), ("N", "H", "C")),
        ((2, 3), ("N", "C"), ("N", None, "C"), (2, 1, 3), ("N", None, "C")),
        ((2, 3), ("N", "C"), ("N",), None, None),
        ((2, 3), ("N", "C"), ("N", "N", "C"), None, None),
        ((2, 3), (None, "C"), ("N", "C"), None, None),
    ],
)
def test_generated_align_cases_match_real_torch(
    shape,
    names,
    target,
    expected_shape,
    expected_names,
):
    _check_align(shape, names, target)
    verdict = verify_align_to(shape, names, target)
    if expected_shape is None:
        assert not verdict.ok
    else:
        assert verdict.ok
        assert verdict.spec.shape == expected_shape
        assert verdict.spec.names == expected_names


def test_align_duplicate_current_names_rejected_like_torch():
    _check_align((2, 3), ("N", "N"), ("N",))
    verdict = verify_align_to((2, 3), ("N", "N"), ("N",))
    assert not verdict.ok
    assert verdict.error_kind == "duplicate"


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.NamedTensor"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.NamedTensor"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_NamedTensorAxCheck.lean")
    body = "import TensorGuard.NamedTensor\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_NamedTensorAxCheck.lean"],
            cwd=_LEAN,
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
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
