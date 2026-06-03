"""Step 229 — Lean-checked einops decomposition and axis-bijection rules.

``lean/TensorGuard/Einops.lean`` mechanizes two obligations used by
``src.einops_verify.verify_einops``:

* grouped-axis decomposition is admitted iff the known factor product is
  positive and divides the consumed axis, and the inferred factor reconstructs
  that axis exactly;
* ``rearrange`` is a named-axis bijection: no drops, additions, or duplicates.

These tests generate concrete conformance cases from those theorem shapes, run
TensorGuard's static checker, and (when the real ``einops`` package is present)
ground every case against real execution.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from src.einops_source import find_einops_bugs
from src.einops_verify import verify_einops

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "Einops.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.Einops.prod_append",
    "TensorGuard.Einops.decompValid_iff",
    "TensorGuard.Einops.decompValid_imp_dvd",
    "TensorGuard.Einops.nondivisible_decomposition_flagged",
    "TensorGuard.Einops.inferSubaxis_spec",
    "TensorGuard.Einops.decomposedGroup_product",
    "TensorGuard.Einops.inferSubaxis_position",
    "TensorGuard.Einops.axisBijection_iff_counts",
    "TensorGuard.Einops.axisBijection_refl",
    "TensorGuard.Einops.axisBijection_sym",
    "TensorGuard.Einops.axisBijection_trans",
    "TensorGuard.Einops.adjacent_swap_axis_bijection",
    "TensorGuard.Einops.dropped_axis_not_bijection",
    "TensorGuard.Einops.added_axis_not_bijection",
    "TensorGuard.Einops.duplicated_axis_not_bijection",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _decomp_valid(axis: int, known_product: int) -> bool:
    return known_product > 0 and axis % known_product == 0


def _infer_subaxis(axis: int, known_product: int) -> int:
    return axis // known_product


def _real_rearrange(pattern: str, shape: tuple[int, ...], **axes):
    einops = pytest.importorskip("einops")
    x = np.zeros(shape, dtype=np.float32)
    try:
        out = einops.rearrange(x, pattern, **axes)
    except Exception:
        return "err", None
    return "ok", tuple(out.shape)


def _generated_decomposition_cases():
    for axis in (6, 7, 12, 20, 36):
        for known in (2, 3, 4, 5, 6, 9):
            yield axis, known


def _generated_axis_bijection_cases():
    names = ("a", "b", "c")
    base_shape = (2, 3, 5)
    for rhs in itertools.permutations(names):
        yield ("valid", " ".join(names), " ".join(rhs), base_shape, None)
    yield ("drop", "a b c", "a c", base_shape, "axis_set_mismatch")
    yield ("add", "a b", "a b c", (2, 3), "axis_set_mismatch")
    yield ("duplicate_lhs", "a a", "a", (2, 2), "duplicate")
    yield ("duplicate_rhs", "a b", "a a", (2, 3), "duplicate")


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.Einops" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated conformance: decomposition divisibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis,known", list(_generated_decomposition_cases()))
def test_generated_decomposition_cases_match_verify_einops_and_real(axis, known):
    pattern = "(h w) c -> h w c"
    shape = (axis, 5)
    verdict = verify_einops("rearrange", pattern, shape, h=known)
    real_status, real_shape = _real_rearrange(pattern, shape, h=known)

    if _decomp_valid(axis, known):
        expected = (known, _infer_subaxis(axis, known), 5)
        assert verdict.ok, verdict.error
        assert verdict.output_shape == expected
        assert verdict.axes["w"] == _infer_subaxis(axis, known)
        assert _prod(verdict.output_shape) == _prod(shape)
        assert real_status == "ok"
        assert real_shape == expected
    else:
        assert not verdict.ok
        assert verdict.error_kind == "non_divisible"
        assert real_status == "err"


def test_multifactor_decomposition_product_is_generated_and_checked():
    pattern = "(h p1 p2) c -> h p1 p2 c"
    axis = 60
    axes = {"p1": 3, "p2": 4}
    known_product = axes["p1"] * axes["p2"]
    assert _decomp_valid(axis, known_product)
    verdict = verify_einops("rearrange", pattern, (axis, 7), **axes)
    assert verdict.ok, verdict.error
    assert verdict.output_shape == (_infer_subaxis(axis, known_product), 3, 4, 7)
    assert _prod(verdict.output_shape) == axis * 7
    real_status, real_shape = _real_rearrange(pattern, (axis, 7), **axes)
    assert real_status == "ok"
    assert real_shape == verdict.output_shape


def test_symbolic_decomposition_abstains_instead_of_false_positive():
    verdict = verify_einops("rearrange", "(h w) c -> h w c", ("seq", 5), h=8)
    assert verdict.ok
    assert verdict.output_shape == (8, "(seq//8)", 5)


# --------------------------------------------------------------------------- #
# 3. Generated conformance: rearrange axis bijection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind,lhs,rhs,shape,error_kind",
    list(_generated_axis_bijection_cases()),
)
def test_generated_axis_bijection_cases_match_verify_einops_and_real(
    kind, lhs, rhs, shape, error_kind
):
    pattern = f"{lhs} -> {rhs}"
    verdict = verify_einops("rearrange", pattern, shape)
    real_status, real_shape = _real_rearrange(pattern, shape)

    if kind == "valid":
        order = rhs.split()
        source = dict(zip(lhs.split(), shape))
        expected = tuple(source[name] for name in order)
        assert verdict.ok, verdict.error
        assert verdict.output_shape == expected
        assert _prod(verdict.output_shape) == _prod(shape)
        assert real_status == "ok"
        assert real_shape == expected
    else:
        assert not verdict.ok
        assert verdict.error_kind == error_kind
        assert real_status == "err"


def test_source_bridge_reports_generated_lean_failure_shapes():
    src = """
from einops import rearrange

def f(x, y):
    bad_split = rearrange(x, "(h w) c -> h w c", h=5)
    bad_axis = rearrange(y, "a b -> a b c")
    return bad_split, bad_axis
"""
    bugs = find_einops_bugs(src, {"x": (14, 3), "y": (2, 4)})
    assert len(bugs) == 2
    assert any("can't divide axis" in bug.message for bug in bugs)
    assert any("identifiers must appear" in bug.message for bug in bugs)


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.Einops"],
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
        ["lake", "build", "TensorGuard.Einops"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_EinopsAxCheck.lean")
    body = "import TensorGuard.Einops\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_EinopsAxCheck.lean"],
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
