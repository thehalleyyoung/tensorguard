"""Step 15 regression tests -- differential fuzzing for false positives.

Pins the invariants of the random-architecture false-positive hunt:

* the fuzzer is deterministic per seed;
* every model it emits is dimensionally valid and *executes* in eager PyTorch;
* TensorGuard (sound mode) produces **zero false positives** on cleanly
  executing random models, and its SAFE coverage is non-trivial;
* the SAFE verdicts are non-vacuous (TG still Refutes a genuine mismatch);
* the on-disk artifact regenerates byte-for-byte.

The full 200-seed run is exercised once via the committed artifact (`--check`);
the live tests use a smaller seed budget to stay fast.
"""

from __future__ import annotations

import os

import pytest

from evaluation import diff_fuzz


def test_builder_is_deterministic_per_seed():
    for seed in (0, 5, 42, 199):
        a_src, a_sh = diff_fuzz.build_model(seed)
        b_src, b_sh = diff_fuzz.build_model(seed)
        assert a_src == b_src
        assert a_sh == b_sh


def test_emitted_models_execute_cleanly():
    for seed in range(40):
        src, shapes = diff_fuzz.build_model(seed)
        assert diff_fuzz.runtime_runs_clean(src, shapes), \
            "fuzzed model seed %d did not execute" % seed


def test_no_false_positives_live_small_run():
    a = diff_fuzz.run(check=False, n_models=40, write=False)
    s = a["summary"]
    assert s["admitted_clean_executing"] >= 1
    assert s["false_positives"] == 0
    assert not a["false_positive_models"]


def test_safe_coverage_is_nontrivial_live():
    a = diff_fuzz.run(check=False, n_models=40, write=False)
    s = a["summary"]
    # Sound mode is allowed to abstain, but it must verify some models SAFE,
    # otherwise "zero false positives" would be vacuous.
    assert s["verified_safe"] >= 1
    assert s["safe_coverage"] > 0.0


def test_safe_is_nonvacuous_tg_still_refutes_a_mismatch():
    # Pick a rank-2 fuzzed model with a Linear, feed a mismatched input dim,
    # and confirm TensorGuard Refutes it (so SAFE is a real verdict).
    for seed in range(200):
        src, shapes = diff_fuzz.build_model(seed)
        shp = next(iter(shapes.values()))
        if len(shp) == 2 and "nn.Linear" in src:
            assert diff_fuzz.tensorguard_verdict(src, shapes)[0] == "SAFE"
            bad = {"x": (shp[0], shp[1] + 7)}
            assert diff_fuzz.tensorguard_verdict(src, bad)[0] == "REFUTED"
            return
    pytest.fail("no rank-2 Linear model found in 200 seeds")


def test_committed_artifact_is_up_to_date():
    """The full 200-seed run is pinned and must regenerate byte-identically."""
    assert os.path.exists(diff_fuzz.OUT_JSON)
    diff_fuzz.run(check=True)


def test_committed_artifact_reports_zero_false_positives():
    import json
    with open(diff_fuzz.OUT_JSON, "r", encoding="utf-8") as fh:
        a = json.load(fh)
    s = a["summary"]
    assert s["seeds_attempted"] == diff_fuzz.N_MODELS
    assert s["false_positives"] == 0
    assert s["admitted_clean_executing"] >= 100
    assert s["verified_safe"] >= 1
