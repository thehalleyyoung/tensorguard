"""Tests for the proposed upstream verification hook (Step 100)."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import reproducibility.upstream_hook_demo as harness  # noqa: E402
from src.upstream_hook import (  # noqa: E402
    ShapeVerificationError,
    attach_verifier,
    verifiable,
    verify_nn_module,
)


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class _Buggy:
    pass


def _buggy_instance():
    torch, nn = _torch()

    class Buggy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(5, 2)

        def forward(self, x):
            return self.b(self.a(x))

    return Buggy()


def _clean_instance():
    torch, nn = _torch()

    class Clean(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x))

    return Clean()


# --- instance verification -------------------------------------------------
def test_verify_buggy_is_unsafe():
    r = verify_nn_module(
        _buggy_instance(), input_shapes={"x": (2, 8)}, soundness_mode="sound"
    )
    assert r.verdict == "UNSAFE"


def test_verify_clean_is_safe():
    r = verify_nn_module(
        _clean_instance(), input_shapes={"x": (2, 8)}, soundness_mode="sound"
    )
    assert r.verdict == "SAFE"


# --- attached hook ---------------------------------------------------------
def test_hook_rejects_buggy_before_forward():
    torch, _ = _torch()
    m = _buggy_instance()
    attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    with pytest.raises(ShapeVerificationError):
        m(torch.randn(2, 8))


def test_hook_transparent_on_clean():
    torch, _ = _torch()
    m = _clean_instance()
    attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    y = m(torch.randn(2, 8))
    assert tuple(y.shape) == (2, 2)


def test_hook_handle_removable():
    torch, _ = _torch()
    m = _buggy_instance()
    h = attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    h.remove()
    # after removal the hook no longer runs; real torch raises instead
    with pytest.raises(RuntimeError) as ei:
        m(torch.randn(2, 8))
    assert not isinstance(ei.value, ShapeVerificationError)


def test_hook_verifies_only_once():
    torch, _ = _torch()
    m = _clean_instance()
    attach_verifier(m, input_shapes={"x": (2, 8)}, soundness_mode="sound")
    # two forwards both succeed; verification is cached after the first
    assert tuple(m(torch.randn(2, 8)).shape) == (2, 2)
    assert tuple(m(torch.randn(2, 8)).shape) == (2, 2)


# --- decorator -------------------------------------------------------------
def test_decorator_rejects_buggy():
    torch, nn = _torch()

    @verifiable(input_shapes={"x": (2, 8)}, soundness_mode="sound")
    class BuggyD(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(5, 2)

        def forward(self, x):
            return self.b(self.a(x))

    with pytest.raises(ShapeVerificationError):
        BuggyD()(torch.randn(2, 8))


def test_decorator_accepts_clean():
    torch, nn = _torch()

    @verifiable(input_shapes={"x": (2, 8)}, soundness_mode="sound")
    class CleanD(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x))

    assert tuple(CleanD()(torch.randn(2, 8)).shape) == (2, 2)


# --- live equivalence ------------------------------------------------------
def test_static_matches_live():
    data = harness.measure()
    assert data["static_matches_live"] is True
    assert data["all_consistent"] is True


def test_buggy_real_forward_actually_raises():
    data = harness.measure()
    assert data["buggy_real_forward_errors"] is True


# --- determinism -----------------------------------------------------------
_VOLATILE_SUBSTRINGS = (
    "time", "elapsed", "timestamp", "wall", "clock",
    "_ms", "seconds", "duration", "date",
)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_artifact_has_no_volatile_fields():
    data = harness.measure()
    for key in _walk_keys(data):
        low = key.lower()
        for bad in _VOLATILE_SUBSTRINGS:
            assert bad not in low, f"volatile substring {bad!r} in key {key!r}"


def test_artifact_is_byte_deterministic():
    assert harness.run(check=True) == 0
