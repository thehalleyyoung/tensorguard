"""Step 233 — Lean-checked grid_sample / affine_grid shape rules.

``lean/TensorGuard/GridSample.lean`` mechanizes the concrete rank-4/rank-5
shape contracts used by ``src.grid_sample_verify``:

* ``grid_sample`` preserves batch/channel, takes output spatial extents from the
  grid, rejects wrong coordinate dimensions and empty input spatial axes, and
  still allows empty output grids;
* ``affine_grid`` maps positive rank-4/rank-5 output sizes and 2x3/3x4 theta
  matrices to coordinate grids with trailing coordinate dimension 2 or 3.

The cases below are generated from those theorem shapes and checked against the
Python verifier plus live PyTorch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.grid_sample_verify import verify_affine_grid, verify_grid_sample  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "GridSample.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.GridSample.gridSample2DValid_iff",
    "TensorGuard.GridSample.gridSample3DValid_iff",
    "TensorGuard.GridSample.gridSample2D_valid_link",
    "TensorGuard.GridSample.gridSample3D_valid_link",
    "TensorGuard.GridSample.gridSample2D_invalid_link",
    "TensorGuard.GridSample.gridSample3D_invalid_link",
    "TensorGuard.GridSample.gridSample2D_output_shape",
    "TensorGuard.GridSample.gridSample3D_output_shape",
    "TensorGuard.GridSample.gridSample2D_output_rank",
    "TensorGuard.GridSample.gridSample3D_output_rank",
    "TensorGuard.GridSample.gridSample_wrong_input_rank_rejected",
    "TensorGuard.GridSample.gridSample_grid_rank_mismatch_rejected",
    "TensorGuard.GridSample.gridSample2D_coord_dim_flagged",
    "TensorGuard.GridSample.gridSample3D_coord_dim_flagged",
    "TensorGuard.GridSample.gridSample2D_zero_height_flagged",
    "TensorGuard.GridSample.gridSample2D_zero_width_flagged",
    "TensorGuard.GridSample.gridSample3D_zero_depth_flagged",
    "TensorGuard.GridSample.gridSample3D_zero_height_flagged",
    "TensorGuard.GridSample.gridSample3D_zero_width_flagged",
    "TensorGuard.GridSample.gridSample2D_batch_mismatch_flagged",
    "TensorGuard.GridSample.gridSample_accepts_empty_output_grid",
    "TensorGuard.GridSample.affineGrid2DValid_iff",
    "TensorGuard.GridSample.affineGrid3DValid_iff",
    "TensorGuard.GridSample.affineGrid2D_valid_link",
    "TensorGuard.GridSample.affineGrid3D_valid_link",
    "TensorGuard.GridSample.affineGrid2D_invalid_link",
    "TensorGuard.GridSample.affineGrid3D_invalid_link",
    "TensorGuard.GridSample.affineGrid2D_output_shape",
    "TensorGuard.GridSample.affineGrid3D_output_shape",
    "TensorGuard.GridSample.affineGrid2D_output_rank",
    "TensorGuard.GridSample.affineGrid3D_output_rank",
    "TensorGuard.GridSample.affineGrid_size_rank_rejected",
    "TensorGuard.GridSample.affineGrid_theta_rank_rejected",
    "TensorGuard.GridSample.affineGrid2D_theta_rows_flagged",
    "TensorGuard.GridSample.affineGrid2D_theta_cols_flagged",
    "TensorGuard.GridSample.affineGrid3D_theta_rows_flagged",
    "TensorGuard.GridSample.affineGrid3D_theta_cols_flagged",
    "TensorGuard.GridSample.affineGrid2D_size_batch_positive_required",
    "TensorGuard.GridSample.affineGrid2D_size_channel_positive_required",
    "TensorGuard.GridSample.affineGrid2D_size_height_positive_required",
    "TensorGuard.GridSample.affineGrid2D_size_width_positive_required",
    "TensorGuard.GridSample.affineGrid3D_size_depth_positive_required",
    "TensorGuard.GridSample.affineGrid_theta_batch_positive_required",
    "TensorGuard.GridSample.affineGrid2D_batch_mismatch_flagged",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _tensor(shape):
    return torch.randn(*shape, dtype=torch.float32)


def _real_grid_sample(input_shape, grid_shape):
    try:
        out = F.grid_sample(_tensor(input_shape), _tensor(grid_shape), align_corners=False)
        return "ok", tuple(out.shape)
    except Exception:
        return "err", None


def _real_affine_grid(theta_shape, size):
    try:
        out = F.affine_grid(_tensor(theta_shape), tuple(size), align_corners=False)
        return "ok", tuple(out.shape)
    except Exception:
        return "err", None


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.GridSample" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated grid_sample conformance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "input_shape,grid_shape,expected_shape",
    [
        ((2, 3, 4, 5), (2, 6, 7, 2), (2, 3, 6, 7)),
        ((2, 3, 4, 5, 6), (2, 7, 8, 9, 3), (2, 3, 7, 8, 9)),
        ((0, 3, 4, 5), (0, 6, 7, 2), (0, 3, 6, 7)),
        ((1, 0, 4, 5), (1, 6, 7, 2), (1, 0, 6, 7)),
        ((1, 3, 4, 5), (1, 0, 7, 2), (1, 3, 0, 7)),
    ],
)
def test_generated_grid_sample_valid_cases_match_real_torch(input_shape, grid_shape, expected_shape):
    real_status, real_shape = _real_grid_sample(input_shape, grid_shape)
    verdict = verify_grid_sample(input_shape, grid_shape)

    assert real_status == "ok"
    assert verdict.ok
    assert real_shape == expected_shape
    assert verdict.output_shape == expected_shape


@pytest.mark.parametrize(
    "input_shape,grid_shape",
    [
        ((2, 3, 4), (2, 4, 2)),
        ((1, 3, 4, 5), (1, 6, 7, 2, 1)),
        ((1, 3, 4, 5), (1, 6, 7, 3)),
        ((1, 3, 4, 5, 6), (1, 6, 7, 8, 2)),
        ((1, 3, 0, 5), (1, 6, 7, 2)),
        ((1, 3, 4, 0), (1, 6, 7, 2)),
        ((1, 3, 0, 4, 5), (1, 6, 7, 8, 3)),
        ((1, 3, 4, 0, 5), (1, 6, 7, 8, 3)),
        ((1, 3, 4, 5, 0), (1, 6, 7, 8, 3)),
        ((2, 3, 4, 5), (1, 6, 7, 2)),
    ],
)
def test_generated_grid_sample_invalid_cases_match_real_torch(input_shape, grid_shape):
    real_status, _ = _real_grid_sample(input_shape, grid_shape)
    verdict = verify_grid_sample(input_shape, grid_shape)

    assert real_status == "err"
    assert not verdict.ok


# --------------------------------------------------------------------------- #
# 3. Generated affine_grid conformance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "theta_shape,size,expected_shape",
    [
        ((2, 2, 3), (2, 3, 6, 7), (2, 6, 7, 2)),
        ((2, 3, 4), (2, 3, 5, 6, 7), (2, 5, 6, 7, 3)),
    ],
)
def test_generated_affine_grid_valid_cases_match_real_torch(theta_shape, size, expected_shape):
    real_status, real_shape = _real_affine_grid(theta_shape, size)
    verdict = verify_affine_grid(theta_shape, size)

    assert real_status == "ok"
    assert verdict.ok
    assert real_shape == expected_shape
    assert verdict.output_shape == expected_shape


@pytest.mark.parametrize(
    "theta_shape,size",
    [
        ((2, 2, 3), (2, 3, 6)),
        ((2, 2), (2, 3, 6, 7)),
        ((2, 3, 3), (2, 3, 6, 7)),
        ((2, 2, 4), (2, 3, 6, 7)),
        ((2, 2, 4), (2, 3, 5, 6, 7)),
        ((2, 3, 3), (2, 3, 5, 6, 7)),
        ((0, 2, 3), (0, 3, 6, 7)),
        ((1, 2, 3), (1, 0, 6, 7)),
        ((1, 2, 3), (1, 3, 0, 7)),
        ((1, 2, 3), (1, 3, 6, 0)),
        ((1, 3, 4), (1, 3, 0, 6, 7)),
        ((0, 2, 3), (1, 3, 6, 7)),
        ((1, 2, 3), (2, 3, 6, 7)),
    ],
)
def test_generated_affine_grid_invalid_cases_match_real_torch(theta_shape, size):
    real_status, _ = _real_affine_grid(theta_shape, size)
    verdict = verify_affine_grid(theta_shape, size)

    assert real_status == "err"
    assert not verdict.ok


# --------------------------------------------------------------------------- #
# 4. Toolchain-gated Lean build + axiom audit (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.GridSample"],
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
        ["lake", "build", "TensorGuard.GridSample"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_GridSampleAxCheck.lean")
    body = "import TensorGuard.GridSample\n" + "\n".join(f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_GridSampleAxCheck.lean"],
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
