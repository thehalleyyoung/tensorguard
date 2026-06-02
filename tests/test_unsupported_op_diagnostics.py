"""Step 34 — unsupported-operator diagnostics (no more guessing).

Before this step the ``torch.fx`` frontend mapped every operator it did not
recognise to a shape-preserving ``ACTIVATION``. For a shape-CHANGING unknown op
that is confidently wrong: it can both (a) fabricate a downstream "shape
mismatch" against a shape the op never actually produced, and (b) silently miss
a genuine bug. A sound verifier must instead *abstain* on what it cannot model
and *tell the user which operator it could not model*.

This suite proves:

* Unknown ops are mapped to ``OpKind.UNSUPPORTED`` and named in the
  ``UnsupportedOpTracker`` diagnostic (``Tensor.unfold``, ``torch.fft.fft``…),
  rather than guessed to be shape-preserving.
* Verification of a *correct* model that uses an unsupported, shape-changing op
  no longer produces a false positive — the unsupported output (and everything
  derived from it) is treated as opaque and shape checks abstain.
* Precision is retained: genuinely shape-preserving ops (``abs``, ``relu`` …)
  still propagate exactly, so a real downstream mismatch is still caught.
* Tensor factories (``torch.zeros`` …) are modelled as ``NEW_TENSOR`` with their
  static shape, not lumped in with unsupported ops.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.fx  # noqa: E402

from src.fx_extractor import fx_trace_to_graph, verify_module  # noqa: E402
from src.model_checker import OpKind, ConstraintVerifier  # noqa: E402


def _graph(module):
    return fx_trace_to_graph(torch.fx.symbolic_trace(module))


def _ops(graph):
    return [s.op.name for s in graph.steps]


# ---------------------------------------------------------------------------
# Unknown ops become UNSUPPORTED and are named in the diagnostic
# ---------------------------------------------------------------------------

class _Unfold(nn.Module):
    def forward(self, x):
        return x.unfold(1, 2, 2)


class _Fft(nn.Module):
    def forward(self, x):
        return torch.fft.fft(x)


def test_unknown_method_is_unsupported_and_named():
    g = _graph(_Unfold())
    unsup = [s for s in g.steps if s.op == OpKind.UNSUPPORTED]
    assert len(unsup) == 1
    assert unsup[0].params.get("op_name") == "Tensor.unfold"
    # It must NOT have been guessed as a shape-preserving activation.
    assert "ACTIVATION" not in _ops(g)


def test_unknown_function_is_unsupported_and_named():
    g = _graph(_Fft())
    unsup = [s for s in g.steps if s.op == OpKind.UNSUPPORTED]
    assert len(unsup) == 1
    assert "fft" in unsup[0].params.get("op_name", "")


def test_tracker_surfaces_unsupported_ops_in_result():
    r = verify_module(_Unfold(), input_shapes={"x": (4, 8)}, backend="fx")
    tracker = r.unsupported_op_tracker
    assert tracker is not None
    assert "Tensor.unfold" in tracker.unsupported_ops
    assert tracker.unsupported_counts["Tensor.unfold"] >= 1
    # coverage fraction is in [0, 1] and < 1 here (one op is unsupported)
    assert 0.0 <= tracker.coverage_fraction() < 1.0
    assert "unsupported op" in tracker.pretty().lower() or \
           "Tensor.unfold" in tracker.pretty()


# ---------------------------------------------------------------------------
# Soundness: an unsupported shape-changing op must not cause a false positive
# ---------------------------------------------------------------------------

class _SafeUnfoldThenLinear(nn.Module):
    """unfold((4,8))->(4,4,2); Linear(2,3) is actually correct."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x.unfold(1, 2, 2))


class _SafeUnfoldReluLinear(nn.Module):
    """Opacity must propagate through a supported op (relu) in between."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(torch.relu(x.unfold(1, 2, 2)))


def test_unsupported_op_does_not_fabricate_violation():
    real = _SafeUnfoldThenLinear()(torch.randn(4, 8))
    assert tuple(real.shape) == (4, 4, 3)  # the model is genuinely correct
    r = verify_module(_SafeUnfoldThenLinear(),
                      input_shapes={"x": (4, 8)}, backend="fx")
    assert r.safe, "abstaining on an unsupported op must not yield a false positive"


def test_opacity_propagates_through_supported_ops():
    r = verify_module(_SafeUnfoldReluLinear(),
                      input_shapes={"x": (4, 8)}, backend="fx")
    assert r.safe


# ---------------------------------------------------------------------------
# Precision retained: genuine ops still propagate, real bugs still caught
# ---------------------------------------------------------------------------

class _AbsThenWrongLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(99, 3)  # wrong: input last dim is 4

    def forward(self, x):
        return self.fc(torch.abs(x))


class _ReluMethodThenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x.relu())


def test_shape_preserving_op_stays_precise_and_catches_bug():
    g = _graph(_AbsThenWrongLinear())
    # abs is allow-listed → ACTIVATION, NOT unsupported
    assert OpKind.UNSUPPORTED not in [s.op for s in g.steps]
    assert "ACTIVATION" in _ops(g)
    r = verify_module(_AbsThenWrongLinear(),
                      input_shapes={"x": (2, 4)}, backend="fx")
    assert not r.safe, "a real downstream mismatch must still be detected"
    assert r.unsupported_op_tracker.unsupported_ops == []


def test_activation_method_is_not_unsupported():
    g = _graph(_ReluMethodThenLinear())
    assert OpKind.UNSUPPORTED not in [s.op for s in g.steps]
    r = verify_module(_ReluMethodThenLinear(),
                      input_shapes={"x": (5, 4)}, backend="fx")
    assert r.safe
    assert r.unsupported_op_tracker.unsupported_ops == []


# ---------------------------------------------------------------------------
# Factories are NEW_TENSOR, not UNSUPPORTED
# ---------------------------------------------------------------------------

class _Factory(nn.Module):
    def forward(self, x):
        return torch.zeros(4, 6) + torch.ones(4, 6)


def test_factory_is_new_tensor_not_unsupported():
    # torch.fx constant-folds literal factories into get_attr constants whose
    # shape is tracked (Step 32 const-shape path); the important guarantee for
    # Step 34 is that a factory is never misclassified as an unsupported op.
    g = _graph(_Factory())
    kinds = [s.op for s in g.steps]
    assert OpKind.UNSUPPORTED not in kinds
    r = verify_module(_Factory(), input_shapes={"x": (3, 6)}, backend="fx")
    assert r.unsupported_op_tracker.unsupported_ops == []


def test_unfolded_factory_call_is_new_tensor():
    # When a factory survives as a call_function (not constant-folded), the fx
    # frontend models it as NEW_TENSOR with its static shape, not UNSUPPORTED.
    from src.fx_extractor import _maybe_tensor_factory

    class _Node:
        def __init__(self, target, args, kwargs):
            self.target, self.args, self.kwargs = target, args, kwargs

    step = _maybe_tensor_factory(_Node(torch.zeros, (4, 6), {}), "_t0")
    assert step is not None and step.op == OpKind.NEW_TENSOR
    assert tuple(d.value for d in step.params["shape"].dims) == (4, 6)
    # dynamic size → abstain (returns None)
    assert _maybe_tensor_factory(_Node(torch.zeros, ("n", 6), {}), "_t0") is None


# ---------------------------------------------------------------------------
# Coverage accounting
# ---------------------------------------------------------------------------

def test_coverage_fraction_is_one_for_fully_supported_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(torch.relu(x))

    r = verify_module(M(), input_shapes={"x": (2, 4)}, backend="fx")
    assert r.unsupported_op_tracker.coverage_fraction() == 1.0
    assert r.unsupported_op_tracker.unsupported_ops == []
