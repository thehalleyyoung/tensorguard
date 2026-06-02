"""Tests for the cross-Python determinism proof (Step 107)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DET_JSON = REPO / "reproducibility" / "cross_python_determinism.json"
WORKER = REPO / "reproducibility" / "_pyhash_worker.py"

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
    data = json.loads(DET_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_artifact_is_byte_deterministic():
    from reproducibility import cross_python_determinism as cpd

    assert cpd.run(check=True) == 0


def test_verdict_invariant_under_hash_randomization():
    data = json.loads(DET_JSON.read_text())
    assert data["verdict_invariant_under_hash_randomization"] is True
    assert data["n_distinct_digests"] == 1
    assert data["deterministic_across_python_builds"] is True


def test_all_fixed_seed_digests_agree():
    data = json.loads(DET_JSON.read_text())
    digs = set(data["fixed_seed_digests"].values())
    assert len(digs) == 1, f"hash-seed drift: {digs}"


def test_random_runs_agree_with_fixed():
    data = json.loads(DET_JSON.read_text())
    fixed = set(data["fixed_seed_digests"].values())
    rand = set(data["random_run_digests"])
    assert fixed == rand, "random hash seeds diverged from fixed seeds"


def test_python_matrix_covers_supported_range():
    data = json.loads(DET_JSON.read_text())
    matrix = data["python_matrix_supported"]
    assert "3.9" in matrix
    assert "3.14" in matrix
    assert len(matrix) >= 6


def test_live_hash_seed_invariance():
    # Re-prove the property out-of-band: two fresh subprocesses with very
    # different PYTHONHASHSEED values must emit the same verdict digest.
    def _digest(seed: str) -> str:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, str(WORKER)],
            cwd=str(REPO), env=env, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr[-500:]
        return proc.stdout.strip()

    a = _digest("0")
    b = _digest("987654321")
    assert a == b and len(a) == 64
