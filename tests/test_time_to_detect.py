"""Tests for Step 116: per-bug time-to-detect (static vs first failing forward)."""

from __future__ import annotations

import importlib

import pytest

harness = importlib.import_module("reproducibility.time_to_detect")

from corpus_extended.module_ast import Linear, ModuleAST, ReLU  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return harness.measure()


_VOLATILE = (
    "time",
    "elapsed",
    "timestamp",
    "wall",
    "clock",
    "_ms",
    "seconds",
    "duration",
    "date",
)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_no_volatile_keys(data):
    for key in _walk_keys(data):
        low = str(key).lower()
        assert not any(tok in low for tok in _VOLATILE), key


def test_static_catches_all_at_depth_zero(data):
    s = data["static"]
    assert s["detect_depth"] == 0
    assert s["requires_constructed_input"] is False
    assert s["requires_execution"] is False
    assert s["n_caught_unsafe"] == data["n_buggy_modules"]
    assert s["all_caught_at_depth_zero"] is True


def test_dynamic_requires_execution_and_prefix(data):
    d = data["dynamic"]
    assert d["requires_constructed_input"] is True
    assert d["requires_execution"] is True
    assert d["detect_depth_max"] >= 1
    # A meaningful fraction of bugs only surface after a successful prefix.
    assert d["n_requires_successful_prefix"] >= 1
    assert 0.0 < d["frac_requires_successful_prefix"] <= 1.0
    assert sum(d["detect_depth_histogram"].values()) == data["n_buggy_modules"]


def test_static_never_later_than_dynamic(data):
    c = data["comparison"]
    assert c["static_never_later_than_dynamic"] is True
    assert c["static_strictly_earlier_count"] == data["dynamic"][
        "n_requires_successful_prefix"
    ]
    assert c["ops_saved_total"] == data["dynamic"][
        "total_ops_executed_before_detection"
    ]


def test_first_failing_op_depth_is_precise():
    # Layer 0 is fine (16->8), layer 1 mismatches (7 != 8): first failure at op 1.
    ast = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(16, 8), Linear(7, 5)),
    )
    assert harness._first_failing_op_depth(ast) == 1

    # Immediate mismatch at op 0.
    ast0 = ModuleAST(
        regime="vec", input_shape=(4, 16), layers=(Linear(7, 8), Linear(8, 5))
    )
    assert harness._first_failing_op_depth(ast0) == 0

    # ReLU is a no-op layer between two good Linears that then mismatch.
    ast2 = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(16, 8), ReLU(), Linear(7, 5)),
    )
    assert harness._first_failing_op_depth(ast2) == 2

    # A clean module has no failing op.
    clean = ModuleAST(
        regime="vec", input_shape=(4, 16), layers=(Linear(16, 8), Linear(8, 5))
    )
    assert harness._first_failing_op_depth(clean) is None


def test_byte_determinism():
    assert harness.run(check=True) == 0
