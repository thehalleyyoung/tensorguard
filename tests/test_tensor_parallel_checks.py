"""Tests for the tensor-parallel sharding checker (Step 97, Phase 10)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.tensor_parallel_checks import (  # noqa: E402
    verify_tensor_parallel, megatron_mlp, TPLinearSpec, TPKind, TPIssueKind,
)
import reproducibility.tensor_parallel_sharding as tps  # noqa: E402

VOLATILE_TOKENS = ("time", "elapsed", "timestamp", "wall", "clock", "_ms",
                   "seconds", "duration", "date")


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_correct_megatron_mlp_is_ok():
    assert verify_tensor_parallel(megatron_mlp(8, 16), 2).ok is True
    assert verify_tensor_parallel(megatron_mlp(64, 256), 8).ok is True


def test_indivisible_shard_flagged():
    res = verify_tensor_parallel(megatron_mlp(8, 15), 2)
    assert not res.ok
    assert TPIssueKind.INDIVISIBLE_SHARD in {i.kind for i in res.issues}


def test_inner_dim_mismatch_flagged():
    specs = [TPLinearSpec("fc1", 8, 16, TPKind.COLUMN),
             TPLinearSpec("fc2", 12, 8, TPKind.ROW, input_is_parallel=True)]
    res = verify_tensor_parallel(specs, 2)
    assert TPIssueKind.INNER_DIM_MISMATCH in {i.kind for i in res.issues}


def test_comm_flag_mismatch_flagged():
    # column gathers but row expects parallel input -> mismatch
    res = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=True, input_is_parallel=True), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH in {i.kind for i in res.issues}
    # the other inconsistent combo: column does not gather, row not parallel
    res2 = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=False, input_is_parallel=False), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH in {i.kind for i in res2.issues}


def test_correct_flag_combos_have_no_comm_issue():
    # gather + not-parallel is also valid (extra communication but consistent)
    res = verify_tensor_parallel(
        megatron_mlp(8, 16, gather_output=True, input_is_parallel=False), 2)
    assert TPIssueKind.COMM_FLAG_MISMATCH not in {i.kind for i in res.issues}


def test_tp_size_one_never_ragged():
    assert verify_tensor_parallel(megatron_mlp(8, 15), 1).ok is True


def test_tp_size_zero_rejected():
    try:
        verify_tensor_parallel(megatron_mlp(8, 16), 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- reproducibility harness: static verdict vs real sharded torch ---------
def test_static_matches_sharded_runtime():
    data = tps.measure()
    assert data["all_ok"] is True
    for c in data["cases"]:
        assert c["static_match"], f"{c['name']} static mismatch"
        assert c["live_match"], f"{c['name']} runtime mismatch"


def test_correct_megatron_reproduces_reference_at_runtime():
    sim = tps._simulate_megatron(8, 16, 2, gather_output=False,
                                 input_is_parallel=True)
    assert sim["ran"] and sim["matches_reference"]
    sim4 = tps._simulate_megatron(8, 16, 4, gather_output=False,
                                  input_is_parallel=True)
    assert sim4["ran"] and sim4["matches_reference"]


def test_artifact_is_byte_deterministic():
    assert tps.run(check=True) == 0


def test_artifact_has_no_volatile_fields():
    data = json.loads(tps.OUT_JSON.read_text())
    for key in _walk_keys(data):
        low = key.lower()
        for tok in VOLATILE_TOKENS:
            assert tok not in low, f"volatile key token {tok!r} in {key!r}"
