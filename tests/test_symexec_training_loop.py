"""Intent pack — training-loop hygiene ("why isn't it training?").

A PyTorch training step that calls ``.backward()`` and/or ``optimizer.step()``
in a loop has three classic *silent* mistakes that never raise at runtime but
mean the model trains wrong or not at all:

* ``missing_zero_grad``     — backward + step but no zero_grad (grads accumulate)
* ``step_without_backward`` — zero_grad + step but no backward (no grads at all)
* ``backward_without_step`` — backward but no step (grads computed, never applied)

These are **heuristic-only** findings: they must appear in ``heuristic`` mode and
be suppressed in ``balanced``/``sound`` so their zero-false-positive guarantee is
preserved.  All assertions below pin both directions (fires in heuristic, silent
in balanced) plus a battery of true-negatives that must never produce a finding.
"""

import ast

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
    return [b for b in analyze_source(src, config=cfg).bugs]


# Canonical loop fragments ----------------------------------------------------

MISSING_ZG = """
import torch
def train(model, loader, optimizer):
    for x, y in loader:
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        optimizer.step()
"""

STEP_NO_BW = """
import torch
def train(loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        optimizer.step()
"""

BW_NO_STEP = """
import torch
def train(loader):
    for x, y in loader:
        loss = x.sum()
        loss.backward()
"""

CORRECT = """
import torch
def train(model, loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        optimizer.step()
"""


# ---- fires in heuristic mode -------------------------------------------------

def test_missing_zero_grad_fires_heuristic():
    assert "missing_zero_grad" in _kinds(MISSING_ZG, HEUR)


def test_step_without_backward_fires_heuristic():
    assert "step_without_backward" in _kinds(STEP_NO_BW, HEUR)


def test_backward_without_step_fires_heuristic():
    assert "backward_without_step" in _kinds(BW_NO_STEP, HEUR)


def test_while_loop_also_scanned():
    src = """
import torch
def train(optimizer):
    i = 0
    while i < 10:
        loss = compute()
        loss.backward()
        optimizer.step()
        i += 1
"""
    assert "missing_zero_grad" in _kinds(src, HEUR)


def test_main_demo_training_loop():
    src = """
import torch
if __name__ == "__main__":
    for epoch in range(5):
        loss = step()
        loss.backward()
        opt.step()
"""
    assert "missing_zero_grad" in _kinds(src, HEUR)


# ---- suppressed in balanced / sound -----------------------------------------

@pytest.mark.parametrize("src", [MISSING_ZG, STEP_NO_BW, BW_NO_STEP])
def test_suppressed_in_balanced(src):
    assert not any(
        k in _kinds(src, BAL)
        for k in ("missing_zero_grad", "step_without_backward", "backward_without_step")
    )


@pytest.mark.parametrize("src", [MISSING_ZG, STEP_NO_BW, BW_NO_STEP])
def test_suppressed_in_sound(src):
    assert not any(
        k in _kinds(src, SOUND)
        for k in ("missing_zero_grad", "step_without_backward", "backward_without_step")
    )


# ---- true negatives (must never fire) ---------------------------------------

def test_correct_loop_is_silent():
    assert not any(
        k in _kinds(CORRECT, HEUR)
        for k in ("missing_zero_grad", "step_without_backward", "backward_without_step")
    )


def test_scheduler_step_only_does_not_fire():
    # ``scheduler.step()`` with no backward and no zero_grad must not be mistaken
    # for an optimizer training step (no gradient signal at all in the loop).
    src = """
import torch
def go(scheduler):
    for e in range(10):
        scheduler.step()
"""
    assert _kinds(src, HEUR) == []


def test_step_without_zero_grad_or_backward_does_not_fire():
    # A bare ``.step()`` (could be a scheduler / RNG / iterator) with neither
    # backward nor zero_grad is too ambiguous — stay silent.
    src = """
def go(thing):
    for e in range(10):
        thing.step()
"""
    assert _kinds(src, HEUR) == []


def test_nested_def_calls_do_not_count():
    # backward()/step() inside a closure defined in the loop are a different
    # execution context and must not be attributed to the loop body.
    src = """
def go(loader, opt):
    for x in loader:
        def cb():
            x.backward()
            opt.step()
        opt.zero_grad()
"""
    assert _kinds(src, HEUR) == []


def test_grad_accumulation_with_zero_grad_is_silent():
    # zero_grad + backward + step all present, even spread across an if, is a
    # correct (grad-accumulation-style) loop and must not fire.
    src = """
def train(loader, opt, n):
    for i, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = x.sum()
        loss.backward()
        if i % n == 0:
            opt.step()
"""
    assert _kinds(src, HEUR) == []


def test_no_training_calls_is_silent():
    src = """
def go(loader):
    total = 0
    for x in loader:
        total = total + x
    return total
"""
    assert _kinds(src, HEUR) == []


# ---- finding metadata --------------------------------------------------------

def test_findings_are_warnings_with_fixes():
    for src, kind in (
        (MISSING_ZG, "missing_zero_grad"),
        (STEP_NO_BW, "step_without_backward"),
        (BW_NO_STEP, "backward_without_step"),
    ):
        bug = next(b for b in _bugs(src, HEUR) if b.kind.value == kind)
        assert bug.severity == "warning"
        assert bug.fix_suggestion
        assert 0.0 < bug.confidence < 1.0


def test_api_category_registered():
    for kind in (
        SymBugKind.MISSING_ZERO_GRAD,
        SymBugKind.STEP_WITHOUT_BACKWARD,
        SymBugKind.BACKWARD_WITHOUT_STEP,
    ):
        assert _API_CATEGORY[kind] == "TYPE_ERROR"


def test_findings_point_at_loop_line():
    bug = next(b for b in _bugs(MISSING_ZG, HEUR) if b.kind.value == "missing_zero_grad")
    # the ``for`` header line
    assert bug.line == 4


def test_deterministic():
    a = _kinds(MISSING_ZG, HEUR)
    b = _kinds(MISSING_ZG, HEUR)
    assert a == b


def test_to_api_bug_round_trips_new_kinds():
    bug = next(b for b in _bugs(STEP_NO_BW, HEUR) if b.kind.value == "step_without_backward")
    api_bug = bug.to_api_bug("train.py")
    assert api_bug.severity == "warning"
    assert api_bug.location.line == bug.line


def test_only_one_finding_per_loop():
    # the three checks are mutually exclusive (elif chain): a loop produces at
    # most one training-loop hygiene finding.
    tl = [
        b for b in _bugs(MISSING_ZG, HEUR)
        if b.kind.value in
        ("missing_zero_grad", "step_without_backward", "backward_without_step")
    ]
    assert len(tl) == 1


def test_collect_helper_skips_nested_scopes():
    from src.symexec.interpreter import Interpreter

    body = ast.parse(
        "for x in loader:\n"
        "    opt.zero_grad()\n"
        "    loss.backward()\n"
        "    def cb():\n"
        "        other.step()\n"
    ).body[0].body
    calls = Interpreter._loop_method_calls(body, {"backward", "step", "zero_grad"})
    assert len(calls["zero_grad"]) == 1
    assert len(calls["backward"]) == 1
    assert len(calls["step"]) == 0  # nested in cb()
