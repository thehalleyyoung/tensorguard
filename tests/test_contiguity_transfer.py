"""Step 145 / 235 — contiguity and concrete stride-layout transfer,
machine-checked in Lean and cross-checked against the real torch layout engine.

``lean/TensorGuard/ContigFlow.lean`` models how the contiguity bit flows through a
chain of layout ops, **scoped** to the concrete regime the verifier reasons about
— a freshly contiguous, rank-2, non-degenerate tensor under repeated ``.t()`` on
dims ``(0,1)``, ``.contiguous()`` and no-ops — where ``transpose`` toggles the
bit, ``.contiguous()`` forces it ``True`` and ``keep`` preserves it.  It proves
the chain compositional
(``ctgRun_append``), that ``.contiguous()`` **erases history**
(``run_after_contig``), that ``transpose`` is an **involution**
(``run_transpose_involution`` — ``t.t().t() == t``) and that a ``keep``-only chain
is identity (``run_allKeep``).

This test mirrors ``ctgStep``/``ctgRun`` in Python and replays **every** chain on
a real 2-D tensor via ``.t()`` / ``.contiguous()``.  It also cross-checks the
Step 235 rank-4 algebra for row-major strides, canonical non-degenerate
channels-last strides, storage offsets, and ``view(N, C*H*W)`` legality.
"""

import itertools
import os
import re
import shutil
import subprocess

import pytest

from src.smt.stride_theory import (
    StridedLayout,
    compute_channels_last_strides,
    compute_contiguous_strides,
    is_canonical_channels_last_contiguous,
    is_contiguous,
    narrow_channel_layout,
    permute_nchw_to_nhwc_layout,
    view_nchw_tail_layout,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_LEAN = os.path.join(_ROOT, "lean")
_FILE = os.path.join(_LEAN, "TensorGuard", "ContigFlow.lean")

_TRUSTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

_THEOREMS = [
    "TensorGuard.ContigFlow.ctgRun_append",
    "TensorGuard.ContigFlow.run_cons_contig",
    "TensorGuard.ContigFlow.run_after_contig",
    "TensorGuard.ContigFlow.run_transpose_involution",
    "TensorGuard.ContigFlow.run_allKeep",
    "TensorGuard.ContigFlow.LayoutAlgebra.contigLayout_is_contiguous",
    "TensorGuard.ContigFlow.LayoutAlgebra.rowMajorCHWTail_viewable",
    "TensorGuard.ContigFlow.LayoutAlgebra.rowMajorCHWTail_view_layout",
    "TensorGuard.ContigFlow.LayoutAlgebra.transpose01_preserves_storageOffset",
    "TensorGuard.ContigFlow.LayoutAlgebra.permuteNCHWtoNHWC_preserves_storageOffset",
    "TensorGuard.ContigFlow.LayoutAlgebra.rowMajor4_strides",
    "TensorGuard.ContigFlow.LayoutAlgebra.rowMajor4_view_tail",
    "TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_is_canonical_channels_last",
    "TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_not_row_major_contiguous",
    "TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_tail_not_viewable",
    "TensorGuard.ContigFlow.LayoutAlgebra.channelsLast4_view_tail_rejected",
    "TensorGuard.ContigFlow.LayoutAlgebra.narrowChannel_rowMajor_offset",
    "TensorGuard.ContigFlow.LayoutAlgebra.narrowChannel_channelsLast_offset",
    "TensorGuard.ContigFlow.LayoutAlgebra.canonicalChannelsLast4_degenerateC_abstains",
]


def _strip_comments(src: str) -> str:
    src = re.sub(r"/-.*?-/", "", src, flags=re.DOTALL)
    src = re.sub(r"--[^\n]*", "", src)
    return src


# Python mirror of the Lean transfer.
def _ctg_step(b, op):
    if op == "keep":
        return b
    if op == "transpose":
        return not b
    return True  # contiguous


def _ctg_run(b0, ops):
    b = b0
    for op in ops:
        b = _ctg_step(b, op)
    return b


# --------------------------------------------------------------------------- #
# 1. Lean proof guards
# --------------------------------------------------------------------------- #
def test_lean_file_imported_and_audited():
    assert os.path.exists(_FILE)
    with open(os.path.join(_LEAN, "TensorGuard.lean")) as fh:
        assert "import TensorGuard.ContigFlow" in fh.read()
    audit = open(os.path.join(_LEAN, "TensorGuard", "AxiomAudit.lean")).read()
    for thm in _THEOREMS:
        assert f"#print axioms {thm}" in audit, f"audit missing {thm}"


def test_lean_no_sorry_or_admit():
    with open(_FILE) as fh:
        assert not re.search(r"\b(sorry|admit)\b", _strip_comments(fh.read()))


# --------------------------------------------------------------------------- #
# 2. Live cross-check against the real torch layout engine.
# --------------------------------------------------------------------------- #
def _torch_run(ops):
    # Fresh contiguous 2-D tensor (distinct dim sizes so transpose is meaningful).
    torch = pytest.importorskip("torch")
    cur = torch.ones(2, 3)
    for op in ops:
        if op == "transpose":
            cur = cur.t()
        elif op == "contiguous":
            cur = cur.contiguous()
        # keep == no-op
    return cur.is_contiguous()


def _torch_layout(tensor) -> StridedLayout:
    return StridedLayout(
        shape=tuple(tensor.shape),
        strides=tuple(tensor.stride()),
        storage_offset=tensor.storage_offset(),
    )


def test_ctg_run_matches_real_torch_all_chains():
    ops_alphabet = ["keep", "transpose", "contiguous"]
    checked = 0
    for length in range(0, 5):
        for chain in itertools.product(ops_alphabet, repeat=length):
            chain = list(chain)
            # base tensor is contiguous => b0 = True
            assert _torch_run(chain) == _ctg_run(True, chain), chain
            checked += 1
    assert checked > 0


def test_transpose_involution_against_real_torch():
    # run_transpose_involution: two transposes cancel.
    for prefix in (["keep"], ["contiguous"], []):
        chain = prefix + ["transpose", "transpose"]
        assert _torch_run(chain) == _torch_run(prefix) == _ctg_run(True, chain)


def test_contiguous_erases_history_against_real_torch():
    # run_after_contig: result after a .contiguous() ignores the prefix.
    for prefix in itertools.product(["keep", "transpose"], repeat=2):
        chain = list(prefix) + ["contiguous", "transpose"]
        assert _torch_run(chain) == _ctg_run(True, ["contiguous", "transpose"]) is False


def test_rank4_contiguous_and_channels_last_layouts_match_real_torch():
    torch = pytest.importorskip("torch")
    row_major = torch.empty((2, 3, 4, 5))
    channels_last = row_major.contiguous(memory_format=torch.channels_last)

    row_layout = _torch_layout(row_major)
    cl_layout = _torch_layout(channels_last)

    assert compute_contiguous_strides(row_layout.shape) == row_layout.strides
    assert is_contiguous(row_layout.shape, row_layout.strides)
    assert row_major.is_contiguous()

    assert compute_channels_last_strides(cl_layout.shape) == cl_layout.strides
    assert is_canonical_channels_last_contiguous(cl_layout.shape, cl_layout.strides)
    assert channels_last.is_contiguous(memory_format=torch.channels_last)
    assert not channels_last.is_contiguous()


def test_nchw_tail_viewability_matches_real_torch():
    torch = pytest.importorskip("torch")
    row_major = torch.empty((2, 3, 4, 5))
    channels_last = row_major.contiguous(memory_format=torch.channels_last)

    row_view = row_major.view(2, 60)
    assert view_nchw_tail_layout(_torch_layout(row_major)) == StridedLayout(
        shape=(2, 60),
        strides=tuple(row_view.stride()),
        storage_offset=row_view.storage_offset(),
    )

    assert view_nchw_tail_layout(_torch_layout(channels_last)) is None
    with pytest.raises(RuntimeError):
        channels_last.view(2, 60)


def test_narrow_and_permute_storage_offsets_match_real_torch():
    torch = pytest.importorskip("torch")
    row_major = torch.empty((2, 3, 4, 5))
    channels_last = row_major.contiguous(memory_format=torch.channels_last)

    row_narrow = row_major.narrow(1, 1, 2)
    cl_narrow = channels_last.narrow(1, 1, 2)

    assert narrow_channel_layout(_torch_layout(row_major), 1, 2) == _torch_layout(row_narrow)
    assert row_narrow.storage_offset() == 20
    assert narrow_channel_layout(_torch_layout(channels_last), 1, 2) == _torch_layout(cl_narrow)
    assert cl_narrow.storage_offset() == 1

    permuted = row_narrow.permute(0, 2, 3, 1)
    assert permute_nchw_to_nhwc_layout(_torch_layout(row_narrow)) == _torch_layout(permuted)
    assert permuted.storage_offset() == row_narrow.storage_offset()


def test_channels_last_degenerate_caveat_is_conservative():
    torch = pytest.importorskip("torch")
    degenerate = torch.empty((2, 1, 4, 5)).contiguous(memory_format=torch.channels_last)
    layout = _torch_layout(degenerate)

    assert degenerate.is_contiguous(memory_format=torch.channels_last)
    assert degenerate.is_contiguous()
    assert compute_channels_last_strides(layout.shape) is None
    assert not is_canonical_channels_last_contiguous(layout.shape, layout.strides)


# --------------------------------------------------------------------------- #
# 3. Toolchain-gated Lean build + axiom audit (slow).
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_lean_builds():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    proc = subprocess.run(
        ["lake", "build", "TensorGuard.ContigFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.slow
def test_lean_theorems_axiom_clean():
    if shutil.which("lake") is None:
        pytest.skip("lake (Lean toolchain) not installed")
    build = subprocess.run(
        ["lake", "build", "TensorGuard.ContigFlow"],
        cwd=_LEAN, capture_output=True, text=True, timeout=900,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    check = os.path.join(_LEAN, "_ContigFlowAxCheck.lean")
    body = "import TensorGuard.ContigFlow\n" + "\n".join(
        f"#print axioms {t}" for t in _THEOREMS) + "\n"
    with open(check, "w") as fh:
        fh.write(body)
    try:
        env = dict(os.environ, LEAN_PATH=os.path.join(_LEAN, ".lake", "build", "lib"))
        proc = subprocess.run(
            ["lake", "env", "lean", "-R", ".", "_ContigFlowAxCheck.lean"],
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
