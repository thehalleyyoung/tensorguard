"""Tests for the cross-version verdict-stability matrix (Step 106)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAB_JSON = REPO / "reproducibility" / "cross_version_stability.json"

_VOLATILE = ("time", "elapsed", "timestamp", "wall", "clock",
             "_ms", "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_artifact_no_volatile_fields():
    data = json.loads(STAB_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import cross_version_stability as cvs

    assert cvs.run(check=True) == 0


def test_verifier_is_static_no_torch_execution():
    data = json.loads(STAB_JSON.read_text())
    assert data["verifier_is_static_no_torch_execution"] is True
    assert data["torch_blocked_verdicts_match_baseline"] is True


def test_all_torch_versions_stable():
    data = json.loads(STAB_JSON.read_text())
    assert data["all_versions_verdict_stable"] is True
    for v, ok in data["per_version_matches_baseline"].items():
        assert ok is True, f"verdict drift at torch {v}"


def test_version_range_covers_2_1_to_2_9():
    data = json.loads(STAB_JSON.read_text())
    versions = data["torch_versions_tested"]
    assert "2.1.0" in versions
    assert "2.9.1" in versions
    assert len(versions) >= 8


def test_overall_stability_claim():
    data = json.loads(STAB_JSON.read_text())
    assert data["verdict_stable_across_torch_2_1_to_2_9"] is True


def test_blocked_torch_invariance_live():
    # Directly re-prove the core property in-process: blocking torch import
    # does not change a known verdict.
    import builtins

    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = nn.Linear(8, 4)\n"
        "        self.b = nn.Linear(5, 2)\n"
        "    def forward(self, x):\n"
        "        return self.b(self.a(x))\n"
    )
    from src.api import verify_architecture

    normal = str(verify_architecture(
        src, input_shapes={"x": (2, 8)}, soundness_mode="sound").verdict)

    real_import = builtins.__import__
    saved = {m: sys.modules[m] for m in list(sys.modules)
             if m == "torch" or m.startswith("torch.")}
    for m in saved:
        del sys.modules[m]

    def blocker(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("blocked")
        return real_import(name, *a, **k)

    builtins.__import__ = blocker
    try:
        blocked = str(verify_architecture(
            src, input_shapes={"x": (2, 8)}, soundness_mode="sound").verdict)
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)

    assert normal == blocked == "UNSAFE"
