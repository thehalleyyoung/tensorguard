"""Step 52 -- anytime / time-budgeted verification with sound partial results.

A static *sound* verifier must never raise a false alarm and must never claim a
proof it did not complete. The budgeted mode honours both invariants:

  * Any violation discovered *before* the budget expires is genuine and is still
    reported (soundness of reported bugs).
  * If the budget expires before every step is checked and no violation was
    found, the result is flagged ``completed=False`` / ``timed_out=True`` with
    NO safety certificate and LOW confidence -- the absence of a violation is
    explicitly NOT a proof. CI gates must treat ``completed=False`` as untrusted.
"""
import textwrap

import pytest

from src.model_checker import Confidence, verify_model


_SAFE = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 10)
        def forward(self, x):
            return self.fc2(self.fc1(x))
"""

# Bug is at the FIRST forward step so any positive budget checks it.
_BUG_FIRST = """
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(99, 16)   # expects 99, input is 32
            self.second = nn.Linear(16, 8)
        def forward(self, x):
            return self.second(self.first(x))
"""


def test_no_budget_completes():
    res = verify_model(textwrap.dedent(_SAFE), input_shapes={"x": (4, 32)})
    assert res.safe is True
    assert res.completed is True
    assert res.timed_out is False
    assert res.steps_total > 0
    assert res.steps_checked == res.steps_total


def test_generous_budget_matches_unbudgeted():
    base = verify_model(textwrap.dedent(_SAFE), input_shapes={"x": (4, 32)})
    budgeted = verify_model(textwrap.dedent(_SAFE),
                            input_shapes={"x": (4, 32)},
                            time_budget_ms=60_000)
    assert budgeted.safe == base.safe is True
    assert budgeted.completed is True
    assert budgeted.timed_out is False
    # A completed budgeted run still issues a certificate.
    assert budgeted.certificate is not None


def test_zero_budget_is_sound_partial_not_a_proof():
    res = verify_model(textwrap.dedent(_SAFE),
                       input_shapes={"x": (4, 32)},
                       time_budget_ms=0.0)
    # No steps were checked, so this is NOT a safety proof.
    assert res.completed is False
    assert res.timed_out is True
    assert res.steps_checked == 0
    assert res.certificate is None
    assert res.confidence == Confidence.LOW
    assert any("incomplete" in e for e in res.errors)
    # `safe` may be True (no violation *found*), but callers must gate on
    # `completed`, never on `safe` alone, for an unfinished run.


def test_zero_budget_on_buggy_model_is_not_trusted_safe():
    # Even though a bug exists, a 0 ms budget checks nothing; the partial result
    # must advertise itself as incomplete so the bug is never silently "passed".
    res = verify_model(textwrap.dedent(_BUG_FIRST),
                       input_shapes={"x": (4, 32)},
                       time_budget_ms=0.0)
    assert res.completed is False
    assert res.timed_out is True
    # The honest contract: do not trust `safe` when `completed` is False.
    trusted_safe = res.safe and res.completed
    assert trusted_safe is False


def test_first_step_violation_survives_budgeting():
    # The bug is at forward-step 0, which any positive budget checks. The
    # violation must be reported regardless of whether later steps timed out.
    res = verify_model(textwrap.dedent(_BUG_FIRST),
                       input_shapes={"x": (4, 32)},
                       time_budget_ms=30_000)
    assert res.safe is False
    assert res.counterexample is not None
    kinds = {v.kind for v in res.counterexample.violations}
    assert "shape_incompatible" in kinds


def test_completed_true_default_on_plain_result():
    # Backward-compat: results without a budget default to completed=True.
    res = verify_model(textwrap.dedent(_BUG_FIRST),
                       input_shapes={"x": (4, 32)})
    assert res.completed is True
    assert res.timed_out is False
    assert res.safe is False
