"""Intent check — ``backward()`` on a detached / no-grad tensor.

``.detach()`` cuts a tensor from the autograd graph, so a subsequent
``.backward()`` computes no gradients and PyTorch raises ``RuntimeError: element
0 of tensors does not require grad and does not have a grad_fn`` — a classic
silent training-killer (``loss.detach().backward()``).

This is a **heuristic** finding: it fires only when the receiver's
``requires_grad`` is *positively* ``False`` (e.g. provably after ``detach()``),
never on tensors whose grad status is merely unknown, so it cannot produce a
false positive from missing information.  Suppressed in ``balanced``/``sound``.
"""

import pytest

from src.symexec.bugs import SymBugKind, _API_CATEGORY
from src.symexec.config import SymConfig
from src.symexec.engine import analyze_source

HEUR = SymConfig.heuristic()
BAL = SymConfig.balanced()
SOUND = SymConfig.sound()


def _kinds(src, cfg):
    return sorted(b.kind.value for b in analyze_source(src, config=cfg).bugs)


def _bugs(src, cfg):
    return list(analyze_source(src, config=cfg).bugs)


DETACH_ASSIGN = """
import torch
def train():
    loss = torch.randn(())
    loss = loss.detach()
    loss.backward()
"""

DETACH_CHAIN = """
import torch
def train():
    x = torch.randn(())
    x.detach().backward()
"""

OK = """
import torch
def train():
    loss = torch.randn(())
    loss.backward()
"""


# ---- fires in heuristic mode -------------------------------------------------

def test_detach_then_backward_fires():
    assert "backward_no_grad" in _kinds(DETACH_ASSIGN, HEUR)


def test_detach_chain_backward_fires():
    assert "backward_no_grad" in _kinds(DETACH_CHAIN, HEUR)


def test_requires_grad_false_explicit_fires():
    src = """
import torch
def train():
    loss = torch.zeros((), requires_grad=False)
    loss.backward()
"""
    # a tensor explicitly constructed without grad, backpropagated
    assert "backward_no_grad" in _kinds(src, HEUR)


# ---- suppressed in balanced / sound -----------------------------------------

@pytest.mark.parametrize("src", [DETACH_ASSIGN, DETACH_CHAIN])
def test_suppressed_in_balanced(src):
    assert "backward_no_grad" not in _kinds(src, BAL)


@pytest.mark.parametrize("src", [DETACH_ASSIGN, DETACH_CHAIN])
def test_suppressed_in_sound(src):
    assert "backward_no_grad" not in _kinds(src, SOUND)


# ---- true negatives ----------------------------------------------------------

def test_plain_backward_does_not_fire():
    assert "backward_no_grad" not in _kinds(OK, HEUR)


def test_unknown_grad_status_does_not_fire():
    # an opaque receiver (unknown requires_grad) must NOT fire — we only fire on
    # positive non-grad provenance, never on absence of information.
    src = """
def train(loss):
    loss.backward()
"""
    assert "backward_no_grad" not in _kinds(src, HEUR)


def test_model_output_unknown_does_not_fire():
    src = """
import torch
def train(model, x):
    loss = model(x).sum()
    loss.backward()
"""
    assert "backward_no_grad" not in _kinds(src, HEUR)


# ---- metadata ----------------------------------------------------------------

def test_finding_is_warning_with_fix():
    bug = next(b for b in _bugs(DETACH_ASSIGN, HEUR) if b.kind.value == "backward_no_grad")
    assert bug.severity == "warning"
    assert bug.fix_suggestion
    assert 0.0 < bug.confidence < 1.0


def test_api_category_registered():
    assert _API_CATEGORY[SymBugKind.BACKWARD_NO_GRAD] == "TYPE_ERROR"


def test_deterministic():
    assert _kinds(DETACH_ASSIGN, HEUR) == _kinds(DETACH_ASSIGN, HEUR)


def test_detach_clears_requires_grad_in_domain():
    from src.symexec.transfer import tensor_method
    from src.symexec.values import TensorVal

    t = TensorVal(rank=2, shape=None, dtype="float32", requires_grad=True, is_leaf=False)
    out = tensor_method(t, "detach", [], {})
    assert isinstance(out, TensorVal)
    assert out.requires_grad is False
    assert out.is_leaf is True
    assert out.rank == 2  # rank/dtype preserved
    assert out.dtype == "float32"
