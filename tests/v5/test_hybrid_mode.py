"""
tests/v5/test_hybrid_mode.py
============================

Tests for src/v5/hybrid_mode.py — hybrid TG-static-first → FakeTensor mode.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.hybrid_mode import hybrid_check  # noqa: E402


# ── Fixtures / helpers ──────────────────────────────────────────────────

CLEAN_MODULE_SOURCE = """
class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
"""

BUGGY_CHANNEL_SOURCE = """
class BuggyConv(nn.Module):
    \"\"\"Input expects 64 channels but gets 3 — detectable by FakeTensor.\"\"\"
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 16, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
"""

NOT_IMPORTABLE_SOURCE = """
class NeedsArgs(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc(x)
"""

NOTIMPL_SOURCE = """
class NotImplForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("not implemented")
"""


# ── Fake TG results ─────────────────────────────────────────────────────

def _make_tg_result(abstained=False, bugs=None):
    """Build a minimal AnalysisResult-like mock."""
    m = MagicMock()
    m.abstained = abstained
    m.bug_count = len(bugs or [])
    m.bugs = bugs or []
    m.status = "SAFE" if m.bug_count == 0 else "UNSAFE"
    return m


# ══════════════════════════════════════════════════════════════════════
# Test 1: TG-Verified → returns immediately without fallback
# ══════════════════════════════════════════════════════════════════════

def test_tg_verified_short_circuits():
    """When TG returns Verified, no fallback should be attempted."""
    tg_result = _make_tg_result(abstained=False, bugs=[])
    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result) as mock_va:
        result = hybrid_check(CLEAN_MODULE_SOURCE,
                              input_shapes={"x": (1, 3, 8, 8)})
    assert result["verdict"] == "Verified"
    assert result["source"] == "tensorguard"
    assert result["fallback"] is None
    mock_va.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# Test 2: TG-Refuted → returns immediately without fallback
# ══════════════════════════════════════════════════════════════════════

def test_tg_refuted_short_circuits():
    """When TG finds a bug, result should be Refuted with no fallback."""
    from src.api import Bug, BugCategory, SourceLocation
    bug = Bug(
        category=BugCategory.TYPE_ERROR,
        message="Shape mismatch detected",
        location=SourceLocation(file="<test>", line=1, column=0),
        severity="error",
        confidence=0.9,
    )
    tg_result = _make_tg_result(abstained=False, bugs=[bug])
    tg_result.bugs = [bug]
    tg_result.bug_count = 1

    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result):
        result = hybrid_check(CLEAN_MODULE_SOURCE,
                              input_shapes={"x": (1, 3, 8, 8)})

    assert result["verdict"] == "Refuted"
    assert result["source"] == "tensorguard"
    assert result["fallback"] is None
    assert len(result["tg_bugs"]) == 1


# ══════════════════════════════════════════════════════════════════════
# Test 3: TG-Abstain + FakeTensor catches Conv channel bug
# ══════════════════════════════════════════════════════════════════════

def test_abstain_faketensor_catches_channel_bug():
    """TG abstains; FakeTensor should detect the 64-vs-3 channel mismatch."""
    tg_result = _make_tg_result(abstained=True)
    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result):
        result = hybrid_check(
            BUGGY_CHANNEL_SOURCE,
            input_shapes={"x": (1, 3, 8, 8)},  # 3 channels, but conv expects 64
        )

    assert result["verdict"] == "Refuted", (
        f"Expected Refuted but got {result['verdict']}; "
        f"fallback={result.get('fallback')}"
    )
    assert result["source"] == "fake_tensor"
    assert result["fallback"] is not None
    assert result["fallback"]["verdict"] == "Refuted"


# ══════════════════════════════════════════════════════════════════════
# Test 4: TG-Abstain + module not importable → Abstain, no crash
# ══════════════════════════════════════════════════════════════════════

def test_abstain_not_importable_returns_abstain():
    """A module whose __init__ requires args cannot be instantiated → Abstain."""
    tg_result = _make_tg_result(abstained=True)
    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result):
        result = hybrid_check(
            NOT_IMPORTABLE_SOURCE,
            input_shapes={"x": (1, 64, 8, 8)},
        )

    assert result["verdict"] == "Abstain"
    assert result["fallback"] is not None
    assert "ctor_failed" in (result["fallback"].get("error") or "")


# ══════════════════════════════════════════════════════════════════════
# Test 5: TG-Abstain + FakeTensor accepts clean module → Verified
# ══════════════════════════════════════════════════════════════════════

def test_abstain_faketensor_accepts_clean_module():
    """TG abstains; FakeTensor should succeed on a well-formed module."""
    tg_result = _make_tg_result(abstained=True)
    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result):
        result = hybrid_check(
            CLEAN_MODULE_SOURCE,
            input_shapes={"x": (1, 3, 8, 8)},
        )

    assert result["verdict"] == "Verified", (
        f"Expected Verified; fallback={result.get('fallback')}"
    )
    assert result["source"] == "fake_tensor"
    assert result["fallback"]["verdict"] == "Verified"
    assert result["fallback"]["error"] is None


# ══════════════════════════════════════════════════════════════════════
# Test 6: Crash isolation — NotImplementedError in forward → Abstain
# ══════════════════════════════════════════════════════════════════════

def test_notimpl_forward_isolated_as_abstain():
    """A forward that raises NotImplementedError must be recorded but not crash."""
    tg_result = _make_tg_result(abstained=True)
    with patch("src.v5.hybrid_mode.verify_architecture", return_value=tg_result):
        result = hybrid_check(
            NOTIMPL_SOURCE,
            input_shapes={"x": (1, 3, 8, 8)},
        )

    # Must not raise, must return Abstain
    assert result["verdict"] == "Abstain"
    assert result["fallback"] is not None
    error = result["fallback"].get("error") or ""
    assert "NotImplementedError" in error or "not implemented" in error.lower(), (
        f"Expected NotImplementedError in error, got: {error!r}"
    )


# ══════════════════════════════════════════════════════════════════════
# Test 7: Public alias tensorguard.hybrid_check is importable
# ══════════════════════════════════════════════════════════════════════

def test_public_alias_importable():
    """hybrid_check must be importable from the src.v5 package."""
    from src.v5 import hybrid_check as hc  # noqa: F401
    assert callable(hc)
