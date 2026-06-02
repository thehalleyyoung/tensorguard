"""Tests for Step 114: property-based full-module-AST testing with shrinking.

Two layers of testing:

  * a fast, *real* Hypothesis property-based test that draws full module ASTs
    from the ``module_asts()`` strategy and asserts the sound verifier never
    commits a soundness violation or a false alarm against the live torch
    oracle (Hypothesis shrinks any counterexample automatically);
  * structural / determinism / minimality assertions over the recorded
    deterministic artifact and the standalone delta-debugging shrinker.
"""

from __future__ import annotations

import importlib
import logging

import pytest

module_ast = importlib.import_module("corpus_extended.module_ast")
harness = importlib.import_module("reproducibility.hypothesis_module_ast")

from corpus_extended.module_ast import (  # noqa: E402
    Linear,
    ModuleAST,
    ReLU,
    module_asts,
    render,
    shrink_to_minimal,
    size,
    torch_runs_clean,
)


def _verdict(ast: ModuleAST) -> str:
    from src.api import verify_architecture

    source, shapes = render(ast)
    return str(
        verify_architecture(
            source,
            input_shapes={k: tuple(v) for k, v in shapes.items()},
            soundness_mode="sound",
        ).verdict
    )


# --------------------------------------------------------------------------
# Real Hypothesis property-based test (the headline deliverable)
# --------------------------------------------------------------------------

try:
    from hypothesis import HealthCheck, given, settings

    _HAS_HYP = True
except Exception:  # pragma: no cover
    _HAS_HYP = False


@pytest.mark.skipif(not _HAS_HYP, reason="hypothesis not installed")
@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(module_asts())
def test_no_soundness_violation_or_false_alarm(ast):
    logging.disable(logging.CRITICAL)
    try:
        clean = torch_runs_clean(ast)
        verdict = _verdict(ast)
        if verdict == "UNKNOWN":
            return  # abstention is always permitted for a sound verifier
        safe = verdict == "SAFE"
        # Soundness: never SAFE when torch raises.
        assert not (safe and not clean), (
            "SOUNDNESS VIOLATION: verifier SAFE but torch raises\n"
            + render(ast)[0]
        )
        # Precision: never UNSAFE when torch runs clean.
        assert not ((not safe) and clean), (
            "FALSE ALARM: verifier UNSAFE but torch clean\n" + render(ast)[0]
        )
    finally:
        logging.disable(logging.NOTSET)


# --------------------------------------------------------------------------
# Deterministic shrinker properties
# --------------------------------------------------------------------------


def test_shrinker_reaches_single_layer_minimal():
    big = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(7, 8), ReLU(), Linear(8, 8), ReLU(), Linear(8, 5)),
    )
    assert torch_runs_clean(big) is False
    minimal = shrink_to_minimal(big, lambda a: not torch_runs_clean(a))
    # Minimal witness keeps the failure...
    assert torch_runs_clean(minimal) is False
    # ...and is strictly smaller than the start.
    assert size(minimal)[0] < size(big)[0]
    assert size(minimal)[0] == 1


def test_shrinker_result_is_locally_minimal():
    from dataclasses import replace

    from corpus_extended.module_ast import (
        _input_reductions,
        _layer_dim_reductions,
    )

    big = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(7, 8), ReLU(), Linear(8, 16), Linear(16, 5)),
    )
    minimal = shrink_to_minimal(big, lambda a: not torch_runs_clean(a))
    # No single deletion preserves the failure.
    for i in range(len(minimal.layers)):
        cand = replace(
            minimal, layers=minimal.layers[:i] + minimal.layers[i + 1 :]
        )
        assert torch_runs_clean(cand) is True
    # No single dim reduction preserves the failure.
    for i, layer in enumerate(minimal.layers):
        for repl in _layer_dim_reductions(layer):
            cand = replace(
                minimal, layers=minimal.layers[:i] + (repl,) + minimal.layers[i + 1 :]
            )
            assert torch_runs_clean(cand) is True
    # No single input reduction preserves the failure.
    for cand in _input_reductions(minimal):
        assert torch_runs_clean(cand) is True


def test_shrinker_is_deterministic():
    big = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(9, 8), ReLU(), Linear(8, 8), Linear(8, 5)),
    )
    pred = lambda a: not torch_runs_clean(a)  # noqa: E731
    a = shrink_to_minimal(big, pred)
    b = shrink_to_minimal(big, pred)
    assert a == b


def test_shrinker_rejects_non_failing_start():
    ok = ModuleAST(
        regime="vec", input_shape=(4, 16), layers=(Linear(16, 8), Linear(8, 5))
    )
    assert torch_runs_clean(ok) is True
    with pytest.raises(ValueError):
        shrink_to_minimal(ok, lambda a: not torch_runs_clean(a))


def test_real_verifier_catches_shrunk_witness():
    big = ModuleAST(
        regime="vec",
        input_shape=(4, 16),
        layers=(Linear(7, 8), ReLU(), Linear(8, 5)),
    )
    minimal = shrink_to_minimal(big, lambda a: not torch_runs_clean(a))
    assert _verdict(minimal) == "UNSAFE"


# --------------------------------------------------------------------------
# Renderer / oracle sanity
# --------------------------------------------------------------------------


def test_render_clean_vs_mismatch():
    clean = ModuleAST(
        regime="vec", input_shape=(4, 16), layers=(Linear(16, 8), Linear(8, 5))
    )
    bad = ModuleAST(
        regime="vec", input_shape=(4, 16), layers=(Linear(16, 8), Linear(7, 5))
    )
    assert torch_runs_clean(clean) is True
    assert torch_runs_clean(bad) is False
    src, shapes = render(clean)
    assert "class Net(nn.Module)" in src
    assert shapes == {"x": (4, 16)}


# --------------------------------------------------------------------------
# Deterministic artifact: structure, no volatile keys, byte-determinism
# --------------------------------------------------------------------------


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


def test_zero_violations_and_false_alarms(data):
    s = data["soundness"]
    assert s["n_soundness_violations"] == 0
    assert s["n_false_alarms"] == 0
    assert s["soundness_violation_sources"] == []
    assert s["false_alarm_sources"] == []
    assert s["perfect_decided_agreement"] is True


def test_scale_and_both_outcomes(data):
    assert data["n_generated"] >= 500
    o = data["oracle"]
    assert o["n_clean"] > 0 and o["n_raise"] > 0
    assert o["n_clean"] + o["n_raise"] == data["n_generated"]
    # Both regimes are exercised.
    assert set(data["regime_counts"]) == {"img", "vec"}


def test_shrinking_demo_minimal_and_caught(data):
    sh = data["shrinking_demo"]
    assert sh["minimal_n_layers"] < sh["start_n_layers"]
    assert sh["minimal_n_layers"] == 1
    assert sh["minimal_is_locally_minimal"] is True
    assert sh["real_verifier_catches_minimal"] is True
    assert sh["real_verifier_verdict_on_minimal"] == "UNSAFE"
    assert "class Net" in sh["minimal_counterexample_source"]


def test_byte_determinism():
    assert harness.run(check=True) == 0
