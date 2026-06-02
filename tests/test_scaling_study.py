"""Tests for the scaling study (Step 109)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STUDY_JSON = REPO / "reproducibility" / "scaling_study.json"

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


def test_deterministic_artifact_no_volatile_fields():
    # The byte-compared artifact must carry no wall-clock fields.
    data = json.loads(STUDY_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        assert not any(v in low for v in _VOLATILE), f"volatile key: {key}"


def test_deterministic_artifact_is_byte_stable():
    # Re-run measure() (NOT the volatile wall-clock) and compare json/md only.
    from reproducibility import scaling_study as ss

    data = ss.measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert new_json == ss.OUT_JSON.read_text()
    assert ss.render_markdown(data) == ss.OUT_MD.read_text()


def test_work_is_linear_in_size():
    data = json.loads(STUDY_JSON.read_text())
    fit = data["work_vs_depth_fit"]
    assert data["work_is_linear_in_size"] is True
    assert fit["r_squared"] >= 0.999
    # lines = 2*depth + 6 for this family -> slope 2.
    assert abs(fit["slope"] - 2.0) < 1e-6


def test_every_size_decided_no_blowup():
    data = json.loads(STUDY_JSON.read_text())
    assert data["all_sizes_decided"] is True
    assert data["no_abstention_at_scale"] is True
    assert data["cegar_bounded_at_scale"] is True
    for row in data["rows"]:
        assert row["decided"] is True
        assert row["verdict"] != "UNKNOWN"


def test_sweep_reaches_large_depth():
    data = json.loads(STUDY_JSON.read_text())
    assert data["max_depth"] >= 64
    assert len(data["rows"]) >= 8


def test_walltime_scaling_is_polynomial_live():
    # Re-measure wall-clock and assert sub-cubic (polynomial, not exponential).
    from reproducibility import scaling_study as ss

    wt = ss.measure_walltime()
    assert wt["loglog_scaling_exponent"] < 3.0
    assert wt["is_polynomial_not_exponential"] is True


def test_params_grow_with_depth():
    data = json.loads(STUDY_JSON.read_text())
    rows = data["rows"]
    params = [r["params"] for r in rows]
    assert params == sorted(params)
    assert params[-1] > params[0]
