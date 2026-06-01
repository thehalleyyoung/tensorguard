"""Step 26 -- advanced indexing: gather / scatter / index_select / masked ops.

Before Step 26 none of the tensor indexing/gather ops were modelled: methods
like ``x.gather(...)``, ``x.index_select(...)``, ``x.masked_select(...)``,
``x.narrow(...)`` and ``x.select(...)`` fell through to the no-op ACTIVATION
fallback, so their (often rank-changing) shape effects were dropped and
downstream shape checks operated on the wrong shape.

Step 26 adds OpKinds GATHER / INDEX_SELECT / SCATTER / MASKED_SELECT /
MASKED_FILL / NARROW / SELECT_DIM / TAKE, a shared ``_apply_indexing`` handler
wired into both engine paths, and extraction (fx method + function forms) that
captures ``dim`` / ``start`` / ``length`` and collects the index/mask/src
tensors as graph inputs.

Soundness posture: a violation is emitted only when the relevant dimensions are
fully concrete and the error is provable (e.g. a concrete gather rank mismatch,
a 2-D ``index_select`` index, an out-of-range ``narrow``); symbolic dimensions
abstain. Output shapes match torch so downstream ops are checked correctly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.fx_extractor import verify_module


def _violation_kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def _is_unsafe(result):
    return (not result.safe) and "shape_incompatible" in _violation_kinds(result)


def _verify(module, **shapes):
    return verify_module(module, input_shapes=shapes)


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

class _Gather(nn.Module):
    def forward(self, x, idx):
        return x.gather(1, idx)


def test_gather_equal_rank_safe():
    assert _verify(_Gather(), x=(4, 3), idx=(4, 2)).safe


def test_gather_rank_mismatch_unsafe():
    assert _is_unsafe(_verify(_Gather(), x=(4, 3), idx=(4,)))


def test_gather_index_exceeds_input_unsafe():
    # index size 5 at dim 0 > input size 4 (non-`dim` axis) -> provable bug.
    assert _is_unsafe(_verify(_Gather(), x=(4, 3), idx=(5, 2)))


def test_gather_output_shape_flows_downstream():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 7)

        def forward(self, x, idx):
            return self.w(x.gather(1, idx))  # (4,3)/(4,2) -> (4,2)

    assert _verify(M(), x=(4, 3), idx=(4, 2)).safe

    class MBad(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(5, 7)

        def forward(self, x, idx):
            return self.w(x.gather(1, idx))  # -> (4,2), linear(5) mismatch

    assert not _verify(MBad(), x=(4, 3), idx=(4, 2)).safe


def test_gather_function_form():
    class M(nn.Module):
        def forward(self, x, idx):
            return torch.gather(x, 1, idx)

    assert _verify(M(), x=(4, 3), idx=(4, 2)).safe
    assert _is_unsafe(_verify(M(), x=(4, 3), idx=(4,)))


# ---------------------------------------------------------------------------
# index_select
# ---------------------------------------------------------------------------

class _IdxSel(nn.Module):
    def forward(self, x, idx):
        return x.index_select(0, idx)


def test_index_select_1d_index_safe():
    assert _verify(_IdxSel(), x=(5, 3), idx=(2,)).safe


def test_index_select_2d_index_unsafe():
    assert _is_unsafe(_verify(_IdxSel(), x=(5, 3), idx=(2, 2)))


def test_index_select_replaces_dim_length():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(3, 7)

        def forward(self, x, idx):
            return self.w(x.index_select(0, idx))  # (5,3)->(2,3)

    assert _verify(M(), x=(5, 3), idx=(2,)).safe


# ---------------------------------------------------------------------------
# scatter (conservative: output == input.shape)
# ---------------------------------------------------------------------------

def test_scatter_preserves_input_shape():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(3, 7)

        def forward(self, x, idx, src):
            return self.w(x.scatter(1, idx, src))  # output == x: (4,3)

    assert _verify(M(), x=(4, 3), idx=(4, 2), src=(4, 2)).safe


def test_scatter_rank_mismatch_unsafe():
    class M(nn.Module):
        def forward(self, x, idx, src):
            return x.scatter(1, idx, src)

    assert _is_unsafe(_verify(M(), x=(4, 3), idx=(4,), src=(4, 2)))


# ---------------------------------------------------------------------------
# masked_fill / masked_select
# ---------------------------------------------------------------------------

class _MaskedFill(nn.Module):
    def forward(self, x, m):
        return x.masked_fill(m, 0.0)


def test_masked_fill_broadcast_safe():
    assert _verify(_MaskedFill(), x=(4, 3), m=(4, 1)).safe


def test_masked_fill_bad_broadcast_unsafe():
    assert _is_unsafe(_verify(_MaskedFill(), x=(4, 3), m=(5, 3)))


def test_masked_fill_preserves_shape():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(3, 7)

        def forward(self, x, m):
            return self.w(x.masked_fill(m, 0.0))

    assert _verify(M(), x=(4, 3), m=(4, 3)).safe


def test_masked_select_is_rank1():
    class M(nn.Module):
        def forward(self, x, m):
            return torch.masked_select(x, m)

    # Output is rank-1 dynamic; a downstream rank-2 linear would mismatch, but
    # here we just assert the verify completes and stays safe (no FP) since the
    # length is symbolic.
    assert _verify(M(), x=(4, 3), m=(4, 3)).safe


# ---------------------------------------------------------------------------
# narrow / select / take
# ---------------------------------------------------------------------------

def test_narrow_safe_and_shape():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 7)

        def forward(self, x):
            return self.w(x.narrow(1, 0, 2))  # (4,5)->(4,2)

    assert _verify(M(), x=(4, 5)).safe


def test_narrow_out_of_bounds_unsafe():
    class M(nn.Module):
        def forward(self, x):
            return x.narrow(1, 4, 3)  # 4+3 > 5

    assert _is_unsafe(_verify(M(), x=(4, 5)))


def test_select_removes_dim():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(5, 7)

        def forward(self, x):
            return self.w(x.select(1, 2))  # (4,3,5)->(4,5)

    assert _verify(M(), x=(4, 3, 5)).safe


def test_select_dim_out_of_range_unsafe():
    class M(nn.Module):
        def forward(self, x):
            return x.select(5, 0)  # rank 2, dim 5 invalid

    assert _is_unsafe(_verify(M(), x=(4, 3)))


def test_take_output_matches_index_shape():
    class M(nn.Module):
        def forward(self, x, idx):
            return torch.take(x, idx)

    # take returns index-shaped output; just assert no FP.
    assert _verify(M(), x=(4, 3), idx=(2, 2)).safe


# ---------------------------------------------------------------------------
# Real-model smoke test: no new false positives
# ---------------------------------------------------------------------------

def test_real_models_no_new_false_positives():
    import torchvision.models as M

    for ctor in (M.resnet18, M.mobilenet_v2, M.shufflenet_v2_x0_5, M.convnext_tiny):
        r = verify_module(ctor().eval(), input_shapes={"x": (1, 3, 224, 224)})
        assert r.safe, f"{ctor.__name__} unexpectedly unsafe: {_violation_kinds(r)}"
