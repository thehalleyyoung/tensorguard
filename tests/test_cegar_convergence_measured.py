"""Step 92: validate the CEGAR convergence theorem against the *real* loop.

These tests assert that the committed measurement artifact
(``reproducibility/cegar_convergence.json``) is internally consistent, that the
tight convergence bound from ``src/cegar_convergence_theory.py`` holds on every
frozen-corpus model, and that the harness is byte-deterministic. They also
exercise the theory module directly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.cegar_convergence_theory import (  # noqa: E402
    compute_predicate_coverage,
    compute_tight_iteration_bound,
)

ART = os.path.join(REPO_ROOT, "reproducibility", "cegar_convergence.json")
SCRIPT = os.path.join(REPO_ROOT, "reproducibility", "cegar_convergence.py")


def _load():
    with open(ART, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ───────────── measured artifact ─────────────

def test_artifact_present_and_covers_corpus():
    art = _load()
    assert art["meta"]["n_models"] == len(art["per_model"]) >= 16


def test_tight_bound_holds_on_every_model():
    art = _load()
    for r in art["per_model"]:
        # the theorem's empirical prediction, per model
        assert r["iterations"] <= 1 + r["discovered_predicates"], r["id"]
        assert r["tight_bound_holds"] is True, r["id"]
        assert r["tight_bound_prediction"] == 1 + r["discovered_predicates"]
    assert art["summary"]["tight_bound_holds_all"] is True


def test_iterations_far_below_naive_bound():
    art = _load()
    for r in art["per_model"]:
        assert r["iterations"] <= r["naive_bound"]
        assert r["iterations_below_naive"] is True
        # naive bound is genuinely much looser (>= 5x here)
        assert r["naive_bound"] >= 5 * max(r["iterations"], 1)
    assert art["summary"]["iterations_below_naive_all"] is True


def test_observed_iterations_are_small():
    art = _load()
    s = art["summary"]
    assert s["max_iterations"] <= 3          # fast convergence on the corpus
    assert s["min_iterations"] >= 1
    assert sum(s["iteration_histogram"].values()) == s["n_models"]


def test_predicate_count_consistency():
    art = _load()
    for r in art["per_model"]:
        assert len(r["predicates"]) == r["discovered_predicates"]


def test_no_volatile_walltime_field():
    # determinism guard: the artifact must not embed a volatile numeric field
    art = _load()

    def scan(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = k.lower()
                assert not any(t in kl for t in
                               ("time_ms", "elapsed", "timestamp", "wallclock")), k
                scan(v)
        elif isinstance(obj, list):
            for v in obj:
                scan(v)

    scan(art)


def test_check_mode_is_byte_deterministic():
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ───────────── theory module ─────────────

def test_coverage_high_means_few_iterations():
    seed = {"a", "b", "c", "d"}
    final = {"a", "b", "c", "d", "e"}  # only one predicate not seeded
    cov = compute_predicate_coverage(seed, final, naive_bound=100)
    assert cov.tight_bound == 1                 # |final \ seed|
    assert cov.coverage == 0.8
    assert cov.improvement_factor == 100.0


def test_full_coverage_zero_iterations():
    s = {"a", "b"}
    cov = compute_predicate_coverage(s, s, naive_bound=50)
    assert cov.tight_bound == 0
    assert cov.coverage == 1.0


def test_tight_iteration_bound_recovers_naive_at_zero_coverage():
    b = compute_tight_iteration_bound(
        num_layers=4, max_dims_per_layer=4, num_predicate_kinds=7,
        estimated_coverage=0.0)
    assert b["naive_bound"] == 4 * 4 * 7
    assert b["tight_bound"] == b["naive_bound"]


def test_tight_iteration_bound_shrinks_with_coverage():
    b = compute_tight_iteration_bound(
        num_layers=4, max_dims_per_layer=4, num_predicate_kinds=7,
        estimated_coverage=0.9)
    assert b["tight_bound"] < b["naive_bound"]
