"""Step 29 -- normalization layers' phase / statistics semantics.

PyTorch normalization layers raise *runtime* errors that depend on the phase
(train/eval) and on ``track_running_stats``:

* ``nn.BatchNorm{1,2,3}d`` (and the lazy variants) raise
  ``ValueError: Expected more than 1 value per channel when training`` when the
  per-channel element count ``N * prod(spatial)`` equals 1 and the layer is
  using *batch* statistics.
* ``nn.InstanceNorm{1,2,3}d`` (and the lazy variants) raise
  ``ValueError: Expected more than 1 spatial element when training`` when the
  *spatial* element count ``prod(spatial)`` equals 1 and the layer is using
  *input* statistics.

A layer uses batch/input statistics when ``training or not
track_running_stats``.  BatchNorm defaults ``track_running_stats=True`` so its
error is TRAIN-only by default; InstanceNorm defaults
``track_running_stats=False`` so its error also fires in eval.  ``GroupNorm`` and
``LayerNorm`` have no such restriction and are exempt, and ``SyncBatchNorm`` is
deliberately excluded because under distributed training the *global*
per-channel count can exceed 1 even when the local count is 1.

Before Step 29 the verifier had phase SMT plumbing but never surfaced a
phase/statistics-dependent runtime error to the user.  Step 29 adds a sound
check (``_check_norm_stats_mode``) that emits a ``phase_error`` violation only
when the relevant element count is *provably* exactly 1 (every contributing dim
is concrete) and the layer is known to use batch/input statistics.

These tests prove the behaviour with a large differential sweep against real
``torch`` modules (the verifier's flag must agree exactly with whether torch
raises), plus targeted end-to-end cases covering both phases, both
``track_running_stats`` settings, the exempt layers, and ``SyncBatchNorm``
abstention.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.fx_extractor import verify_module
from src.model_checker import verify_model, Phase


class _Wrap(nn.Module):
    def __init__(self, norm: nn.Module):
        super().__init__()
        self.n = norm

    def forward(self, x):
        return self.n(x)


def _make(fam: str, c: int, trs: bool) -> nn.Module:
    return {
        "bn1": lambda: nn.BatchNorm1d(c, track_running_stats=trs),
        "bn2": lambda: nn.BatchNorm2d(c, track_running_stats=trs),
        "bn3": lambda: nn.BatchNorm3d(c, track_running_stats=trs),
        "in1": lambda: nn.InstanceNorm1d(c, track_running_stats=trs),
        "in2": lambda: nn.InstanceNorm2d(c, track_running_stats=trs),
        "in3": lambda: nn.InstanceNorm3d(c, track_running_stats=trs),
    }[fam]()


def _torch_raises(norm: nn.Module, shape, train: bool):
    norm.train(train)
    try:
        norm(torch.randn(*shape))
        return False
    except ValueError:
        return True
    except Exception:
        return None  # unrelated error -> skip this sample


def _tg_flags(norm: nn.Module, shape, train: bool) -> bool:
    phase = Phase.TRAIN if train else Phase.EVAL
    r = verify_module(_Wrap(norm), input_shapes={"x": tuple(shape)},
                      default_phase=phase)
    if r.safe or not r.counterexample:
        return False
    return any(v.kind == "phase_error" for v in r.counterexample.violations)


def test_differential_against_torch_raise_behaviour():
    """The verifier's phase_error flag agrees exactly with whether torch raises,
    across thousands of randomized norm configs at canonical ranks."""
    random.seed(7)
    torch.manual_seed(7)

    def dim():
        return random.choice([1, 1, 2, 3])

    checked = 0
    flagged = 0
    for _ in range(2000):
        fam = random.choice(["bn1", "bn2", "bn3", "in1", "in2", "in3"])
        c = random.randint(1, 4)
        trs = random.choice([True, False])
        train = random.choice([True, False])
        if fam == "bn1":
            shape = [dim(), c] + ([dim()] if random.random() < 0.5 else [])
        elif fam == "bn2":
            shape = [dim(), c, dim(), dim()]
        elif fam == "bn3":
            shape = [dim(), c, dim(), dim(), dim()]
        elif fam == "in1":
            shape = [dim(), c, dim()]
        elif fam == "in2":
            shape = [dim(), c, dim(), dim()]
        else:
            shape = [dim(), c, dim(), dim(), dim()]

        raises = _torch_raises(_make(fam, c, trs), shape, train)
        if raises is None:
            continue
        flags = _tg_flags(_make(fam, c, trs), shape, train)
        checked += 1
        if flags:
            flagged += 1
        # Soundness AND completeness at concrete canonical ranks: exact match.
        assert flags == raises, (
            f"disagreement: fam={fam} shape={shape} trs={trs} train={train} "
            f"torch_raises={raises} tg_flags={flags}"
        )
    assert checked > 1200, f"too few samples exercised: {checked}"
    assert flagged > 100, f"expected many positive cases, got {flagged}"


def test_batchnorm_batch_one_train_vs_eval():
    bn = nn.BatchNorm2d(4)
    assert _tg_flags(bn, (1, 4, 1, 1), train=True) is True
    # Default track_running_stats=True -> eval uses running stats -> safe.
    assert _tg_flags(nn.BatchNorm2d(4), (1, 4, 1, 1), train=False) is False
    # Spatial > 1 or batch > 1 is safe even in train.
    assert _tg_flags(nn.BatchNorm2d(4), (1, 4, 2, 2), train=True) is False
    assert _tg_flags(nn.BatchNorm2d(4), (4, 4, 1, 1), train=True) is False


def test_batchnorm_track_running_stats_false_eval_flags():
    """track_running_stats=False forces batch stats even in eval -> error."""
    bn = nn.BatchNorm2d(4, track_running_stats=False)
    assert _tg_flags(bn, (1, 4, 1, 1), train=False) is True


def test_batchnorm1d_and_3d():
    assert _tg_flags(nn.BatchNorm1d(4), (1, 4), train=True) is True
    assert _tg_flags(nn.BatchNorm1d(4), (2, 4), train=True) is False
    assert _tg_flags(nn.BatchNorm1d(4), (1, 4, 1), train=True) is True
    assert _tg_flags(nn.BatchNorm1d(4), (1, 4, 5), train=True) is False
    assert _tg_flags(nn.BatchNorm3d(4), (1, 4, 1, 1, 1), train=True) is True
    assert _tg_flags(nn.BatchNorm3d(4), (1, 4, 2, 1, 1), train=True) is False


def test_instancenorm_spatial_one_both_phases():
    """InstanceNorm (default track_running_stats=False) errors in train AND
    eval when the spatial element count is 1; batch size is irrelevant."""
    assert _tg_flags(nn.InstanceNorm2d(4), (2, 4, 1, 1), train=True) is True
    assert _tg_flags(nn.InstanceNorm2d(4), (2, 4, 1, 1), train=False) is True
    assert _tg_flags(nn.InstanceNorm2d(4), (2, 4, 3, 3), train=True) is False
    assert _tg_flags(nn.InstanceNorm1d(4), (2, 4, 1), train=True) is True


def test_groupnorm_and_layernorm_exempt():
    class GN(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GroupNorm(2, 4)

        def forward(self, x):
            return self.g(x)

    class LN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(4)

        def forward(self, x):
            return self.ln(x)

    assert verify_module(GN(), input_shapes={"x": (1, 4, 1, 1)},
                         default_phase=Phase.TRAIN).safe is True
    assert verify_module(LN(), input_shapes={"x": (1, 1, 4)},
                         default_phase=Phase.TRAIN).safe is True


def test_syncbatchnorm_abstains():
    """SyncBatchNorm is excluded (distributed global count may exceed 1)."""
    class SBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = nn.SyncBatchNorm(4)

        def forward(self, x):
            return self.b(x)

    r = verify_module(SBN(), input_shapes={"x": (1, 4, 1, 1)},
                      default_phase=Phase.TRAIN)
    assert r.safe is True


def test_source_level_and_check_phases_toggle():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.BatchNorm2d(8)
    def forward(self, x):
        return self.b(x)
"""
    assert verify_model(src, input_shapes={"x": (1, 8, 1, 1)},
                        default_phase=Phase.TRAIN).safe is False
    assert verify_model(src, input_shapes={"x": (1, 8, 1, 1)},
                        default_phase=Phase.EVAL).safe is True
    # check_phases=False suppresses the phase_error violation.
    assert verify_model(src, input_shapes={"x": (1, 8, 1, 1)},
                        default_phase=Phase.TRAIN, check_phases=False).safe is True


def test_symbolic_dims_abstain():
    """When the contributing dims are symbolic we must abstain (no flag)."""
    bn = nn.BatchNorm2d(4)
    r = verify_module(_Wrap(bn), input_shapes={"x": ("N", 4, "H", "W")},
                      default_phase=Phase.TRAIN)
    phase_errs = []
    if not r.safe and r.counterexample:
        phase_errs = [v for v in r.counterexample.violations
                      if v.kind == "phase_error"]
    assert phase_errs == []
