"""Regression tests for soundness modes and the three-valued verdict (Step 7)."""

import json
import os
import subprocess
import sys

import pytest

from src.api import AnalysisResult, Bug, BugCategory, SourceLocation, verify_architecture
from src.soundness_contract import SoundnessMode

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CLEAN = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        return self.fc(x)
"""

_OPAQUE = """
import torch, torch.nn as nn
class Weird(nn.Module):
    def forward(self, x):
        return x
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.weird = Weird()
    def forward(self, x):
        return self.weird(self.fc(x))
"""

_DDCF = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        if x.sum() > 0:
            x = self.fc(x)
        return x
"""

_HEURISTIC_OP = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
    def forward(self, x):
        x = self.fc(x)
        u = torch.unique(x)
        return x
"""

_BUG = """
import torch, torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 8)
    def forward(self, x):
        return self.fc(x)
"""


def _verify(src, shape, mode):
    return verify_architecture(
        src, input_shapes={"x": shape}, max_cegar_iterations=0, soundness_mode=mode
    )


# ── SoundnessMode enum ─────────────────────────────────────────────────────

def test_soundness_mode_from_str():
    assert SoundnessMode.from_str("sound") is SoundnessMode.SOUND
    assert SoundnessMode.from_str("BALANCED") is SoundnessMode.BALANCED
    assert SoundnessMode.from_str(SoundnessMode.HEURISTIC) is SoundnessMode.HEURISTIC


def test_soundness_mode_rejects_unknown():
    with pytest.raises(ValueError):
        SoundnessMode.from_str("strict")


# ── verdict property semantics (unit) ──────────────────────────────────────

def test_verdict_property_unit():
    bug = Bug(
        category=BugCategory.TYPE_ERROR,
        message="x", location=SourceLocation("f.py", 1, 0), severity="error", confidence=1.0,
    )
    # bug always UNSAFE regardless of mode
    for mode in ("sound", "balanced", "heuristic"):
        assert AnalysisResult(bugs=[bug], soundness_mode=mode).verdict == "UNSAFE"
    # abstained: UNKNOWN unless heuristic
    assert AnalysisResult(abstained=True, soundness_mode="sound").verdict == "UNKNOWN"
    assert AnalysisResult(abstained=True, soundness_mode="balanced").verdict == "UNKNOWN"
    assert AnalysisResult(abstained=True, soundness_mode="heuristic").verdict == "SAFE"
    # clean
    assert AnalysisResult(soundness_mode="sound").verdict == "SAFE"


# ── verdict matrix against real verification ───────────────────────────────

def test_clean_is_safe_in_all_modes():
    for mode in ("sound", "balanced", "heuristic"):
        assert _verify(_CLEAN, (4, 8), mode).verdict == "SAFE"


def test_opaque_layer_unknown_in_sound_and_balanced_safe_in_heuristic():
    assert _verify(_OPAQUE, (4, 8), "sound").verdict == "UNKNOWN"
    assert _verify(_OPAQUE, (4, 8), "balanced").verdict == "UNKNOWN"
    assert _verify(_OPAQUE, (4, 8), "heuristic").verdict == "SAFE"


def test_data_dependent_control_flow_only_abstains_in_sound():
    # This is the silent-SAFE soundness gap (U1) that sound mode closes.
    assert _verify(_DDCF, (4, 8), "sound").verdict == "UNKNOWN"
    assert _verify(_DDCF, (4, 8), "balanced").verdict == "SAFE"
    assert _verify(_DDCF, (4, 8), "heuristic").verdict == "SAFE"


def test_heuristic_operator_only_abstains_in_sound():
    r = _verify(_HEURISTIC_OP, (4, 8), "sound")
    assert r.verdict == "UNKNOWN"
    assert any("heuristic-tagged operator" in reason for reason in r.unknown_reasons)
    assert _verify(_HEURISTIC_OP, (4, 8), "balanced").verdict == "SAFE"


def test_real_bug_refuted_in_every_mode():
    # soundness_mode must NOT change which bugs are reported (no recall change).
    for mode in ("sound", "balanced", "heuristic"):
        r = _verify(_BUG, (4, 768), mode)
        assert r.verdict == "UNSAFE"
        assert r.bug_count >= 1


def test_unknown_reasons_populated_for_abstention():
    r = _verify(_OPAQUE, (4, 8), "sound")
    assert r.unknown_reasons
    assert r.abstained is True


# ── CLI ────────────────────────────────────────────────────────────────────

def _write(tmp_path, src):
    p = tmp_path / "mod.py"
    p.write_text(src)
    return str(p)


def _run_cli(path, mode, fmt="text"):
    return subprocess.run(
        [
            sys.executable, "-m", "src.cli.main", "verify", path,
            "-s", "x=4,8", "--soundness-mode", mode, "-f", fmt,
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )


def test_cli_sound_mode_fails_on_opaque(tmp_path):
    path = _write(tmp_path, _OPAQUE)
    out = _run_cli(path, "sound")
    assert out.returncode == 2, out.stdout + out.stderr
    assert "UNKNOWN" in out.stdout


def test_cli_balanced_mode_exit_zero_on_opaque(tmp_path):
    path = _write(tmp_path, _OPAQUE)
    out = _run_cli(path, "balanced")
    assert out.returncode == 0, out.stdout + out.stderr


def test_cli_json_has_verdict_fields(tmp_path):
    path = _write(tmp_path, _OPAQUE)
    out = _run_cli(path, "sound", fmt="json")
    payload = json.loads(out.stdout)
    assert payload["verdict"] == "UNKNOWN"
    assert payload["soundness_mode"] == "sound"
    assert payload["unknown_reasons"]
