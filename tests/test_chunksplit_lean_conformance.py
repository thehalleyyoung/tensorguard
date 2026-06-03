"""Step 231 — Lean-checked chunk/split partition reconstruction.

``lean/TensorGuard/ChunkSplit.lean`` mechanizes the axis-local arithmetic used by
TensorGuard's precise ``torch.chunk`` / ``torch.split`` transfer functions:

* list split sections are valid iff they reconstruct the split axis;
* uneven integer splits and chunks preserve the exact final section size;
* ``torch.chunk`` may return fewer tensors than requested;
* zero-size axes and zero-size split sections match PyTorch behavior;
* concatenating every produced piece along the same axis reconstructs the
  original shape.

The concrete cases below are generated from those theorem shapes and checked
against ``src.tensor_shapes`` plus live ``torch`` execution.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import textwrap

import pytest

from src.model_checker import verify_model
from src.tensor_shapes import (
    TensorShape,
    compute_chunk_shapes,
    compute_split_shapes,
    symbolic_split_shape,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "ChunkSplit.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.ChunkSplit.sum_append",
    "TensorGuard.ChunkSplit.prod_append",
    "TensorGuard.ChunkSplit.splitValid_iff",
    "TensorGuard.ChunkSplit.split_list_mismatch_flagged",
    "TensorGuard.ChunkSplit.axisConcat_reconstruct",
    "TensorGuard.ChunkSplit.splitConcat_shape",
    "TensorGuard.ChunkSplit.splitConcat_numel",
    "TensorGuard.ChunkSplit.split_int_uneven_example",
    "TensorGuard.ChunkSplit.split_int_tail_example",
    "TensorGuard.ChunkSplit.split_int_zero_axis_example",
    "TensorGuard.ChunkSplit.split_list_with_empty_section_valid",
    "TensorGuard.ChunkSplit.split_list_mismatch_example",
    "TensorGuard.ChunkSplit.chunk_uneven_example",
    "TensorGuard.ChunkSplit.chunk_many_sections_example",
    "TensorGuard.ChunkSplit.chunk_fewer_than_requested_example",
    "TensorGuard.ChunkSplit.chunk_fewer_than_requested_len",
    "TensorGuard.ChunkSplit.chunk_zero_axis_returns_requested_empties",
    "TensorGuard.ChunkSplit.split_concat_reconstruct_example",
    "TensorGuard.ChunkSplit.chunk_concat_reconstruct_example",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", src)


def _dims(shape: TensorShape) -> tuple:
    return tuple(dim.value for dim in shape.dims)


def _tensor(shape: tuple[int, ...]):
    torch = pytest.importorskip("torch")
    numel = math.prod(shape)
    if numel == 0:
        return torch.zeros(shape)
    return torch.arange(numel, dtype=torch.float32).reshape(shape)


def _axis_size(shape: tuple[int, ...], dim: int) -> int:
    return shape[dim if dim >= 0 else len(shape) + dim]


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.ChunkSplit" in fh.read()
    with open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")) as fh:
        audit = fh.read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        code = _strip_comments(fh.read())
    assert not re.search(r"\b(sorry|admit)\b", code)


# --------------------------------------------------------------------------- #
# 2. Generated chunk conformance: helper == torch, then cat reconstructs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,chunks,dim",
    [
        ((13, 4), 6, 0),
        ((2, 10, 4), 3, 1),
        ((5, 4), 8, 0),
        ((0, 4), 3, 0),
    ],
)
def test_generated_chunk_cases_match_real_torch_and_reconstruct(shape, chunks, dim):
    torch = pytest.importorskip("torch")
    x = _tensor(shape)
    real = list(torch.chunk(x, chunks, dim=dim))
    static = compute_chunk_shapes(TensorShape.from_tuple(shape), chunks, dim)
    assert static is not None
    assert [_dims(s) for s in static] == [tuple(y.shape) for y in real]

    axis = dim if dim >= 0 else len(shape) + dim
    assert sum(y.shape[axis] for y in real) == _axis_size(shape, dim)
    assert tuple(torch.cat(real, dim=dim).shape) == shape
    assert torch.equal(torch.cat(real, dim=dim), x)


def test_chunk_returns_fewer_than_requested_sections_like_pytorch():
    torch = pytest.importorskip("torch")
    x = _tensor((5, 4))
    real = list(torch.chunk(x, 8, dim=0))
    static = compute_chunk_shapes(TensorShape.from_tuple((5, 4)), 8, 0)
    assert len(real) == 5
    assert static is not None and len(static) == 5
    assert [_dims(s) for s in static] == [tuple(y.shape) for y in real]


# --------------------------------------------------------------------------- #
# 3. Generated split conformance: helper == torch, then cat reconstructs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,spec,dim",
    [
        ((2, 13), 6, -1),
        ((2, 10), 3, -1),
        ((2, 0), 0, -1),
        ((2, 5), [2, 0, 3], -1),
        ((2, 0), [0, 0], -1),
    ],
)
def test_generated_split_cases_match_real_torch_and_reconstruct(shape, spec, dim):
    torch = pytest.importorskip("torch")
    x = _tensor(shape)
    real = list(torch.split(x, spec, dim=dim))
    static = compute_split_shapes(TensorShape.from_tuple(shape), spec, dim)
    assert static is not None
    assert [_dims(s) for s in static] == [tuple(y.shape) for y in real]

    axis = dim if dim >= 0 else len(shape) + dim
    assert sum(y.shape[axis] for y in real) == _axis_size(shape, dim)
    assert tuple(torch.cat(real, dim=dim).shape) == shape
    assert torch.equal(torch.cat(real, dim=dim), x)


def test_split_section_sum_mismatch_is_rejected_like_pytorch():
    torch = pytest.importorskip("torch")
    x = _tensor((2, 5))
    assert compute_split_shapes(TensorShape.from_tuple((2, 5)), [2, 0, 2], -1) is None
    with pytest.raises(RuntimeError):
        torch.split(x, [2, 0, 2], dim=-1)


def test_symbolic_axis_abstains_with_fresh_partition_axis():
    symbolic = TensorShape.from_tuple(("B", 10))
    assert compute_chunk_shapes(symbolic, 3, 0) is None
    assert compute_split_shapes(symbolic, 3, 0) is None
    abstained = symbolic_split_shape(symbolic, 0, "_chunk_out")
    assert abstained is not None
    assert _dims(abstained) == ("_chunk_out", 10)


# --------------------------------------------------------------------------- #
# 4. Verifier bridge: final uneven chunk width flows to downstream consumers.
# --------------------------------------------------------------------------- #
def test_model_checker_uses_exact_final_chunk_width_downstream():
    def source(features: int) -> str:
        return f"""
        import torch
        import torch.nn as nn

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear({features}, 1)

            def forward(self, x):
                a, b, c = x.chunk(3, dim=-1)
                return self.fc(c)
        """

    safe = verify_model(textwrap.dedent(source(2)), input_shapes={"x": (5, 10)})
    unsafe = verify_model(
        textwrap.dedent(source(4)),
        input_shapes={"x": (5, 10)},
    )
    assert safe.safe, safe.errors
    assert not unsafe.safe
    assert unsafe.counterexample is not None
    assert any(
        "Linear expects last dim=4, got 2" in violation.message
        for violation in unsafe.counterexample.violations
    )


# --------------------------------------------------------------------------- #
# 5. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.ChunkSplit"],
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
        ["lake", "build", "TensorGuard.ChunkSplit"],
        cwd=_LEAN,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]

    check = os.path.join(_LEAN, "_ChunkSplitAxCheck.lean")
    body = "import TensorGuard.ChunkSplit\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS
    ) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_ChunkSplitAxCheck.lean"],
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
