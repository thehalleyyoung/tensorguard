"""Tests for Step 91: effect-size estimators + localization-effort proxy.

Covers (a) the new effect-size functions in ``src.statistical_rigor`` against
textbook / hand-computed values and invariants, and (b) the
``evaluation/localization_effort.py`` harness: it runs, is deterministic
(``--check``), keeps the misleading bugs, and stays consistent with the
committed localization artifact it consumes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.statistical_rigor import (  # noqa: E402
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    hedges_g,
)


# ───────────────────────── effect sizes ─────────────────────────

def test_cliffs_delta_total_dominance():
    # every a > every b  ->  delta = +1
    d = cliffs_delta([10, 11, 12], [1, 2, 3])
    assert d.value == pytest.approx(1.0)
    assert d.magnitude == "large"
    # symmetry: swapping arguments negates delta
    d2 = cliffs_delta([1, 2, 3], [10, 11, 12])
    assert d2.value == pytest.approx(-1.0)


def test_cliffs_delta_identical_is_zero():
    d = cliffs_delta([1, 2, 3], [1, 2, 3])
    assert d.value == pytest.approx(0.0)
    assert d.magnitude == "negligible"


def test_cliffs_delta_hand_value():
    # a=[1,2,3], b=[2,2,5]; count over 9 pairs:
    #   a=1: <2,<2,<5  -> 3 lt
    #   a=2: =2,=2,<5  -> 1 lt
    #   a=3: >2,>2,<5  -> 2 gt, 1 lt
    # gt=2, lt=5 -> (2-5)/9 = -0.3333
    d = cliffs_delta([1, 2, 3], [2, 2, 5])
    assert d.value == pytest.approx(-3 / 9)
    assert d.magnitude == "medium"  # |0.333| is just over the 0.33 small/medium band


def test_cliffs_delta_magnitude_bands():
    assert cliffs_delta([1], [1]).magnitude == "negligible"
    assert cliffs_delta([5] * 3, [1, 2, 9]).magnitude in {
        "negligible", "small", "medium", "large"}


def test_cohens_d_known_one_sd():
    # two groups separated by exactly one pooled SD
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [x - 1.0 for x in a]  # shifted down by 1; same variance
    d = cohens_d(a, b)
    # mean diff = 1, pooled sd = sample sd of a (=sqrt(2.5)) -> d = 1/sqrt(2.5)
    assert d.value == pytest.approx(1.0 / (2.5 ** 0.5), rel=1e-9)


def test_hedges_g_shrinks_toward_zero():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 1.0, 2.0, 3.0]
    d = cohens_d(a, b).value
    g = hedges_g(a, b).value
    assert abs(g) < abs(d)            # bias correction shrinks magnitude
    assert (g > 0) == (d > 0)         # same sign


def test_cohens_d_degenerate_variance_is_zero():
    d = cohens_d([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])
    assert d.value == 0.0
    assert d.magnitude == "negligible"


def test_bootstrap_ci_is_seeded_and_brackets_point():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    import statistics
    ci1 = bootstrap_ci(vals, statistics.mean, n_resamples=2000, seed=0)
    ci2 = bootstrap_ci(vals, statistics.mean, n_resamples=2000, seed=0)
    assert ci1.ci_low == ci2.ci_low and ci1.ci_high == ci2.ci_high  # deterministic
    assert ci1.ci_low <= ci1.point_estimate <= ci1.ci_high
    assert ci1.point_estimate == pytest.approx(statistics.mean(vals))


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], lambda x: 0.0)
    assert ci.point_estimate == 0.0 and ci.n_resamples == 0


# ───────────────────────── harness ─────────────────────────

LOC_JSON = os.path.join(REPO_ROOT, "evaluation", "localization_effort.json")
SCRIPT = os.path.join(REPO_ROOT, "evaluation", "localization_effort.py")
SRC_JSON = os.path.join(
    REPO_ROOT, "reproducibility", "localization_marker_only_n30.json")


def _load():
    with open(LOC_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_artifact_present_and_shaped():
    art = _load()
    assert art["meta"]["n_bugs"] == len(art["per_bug"]) >= 10
    s = art["summary"]
    for k in ("cliffs_delta", "cohens_d", "hedges_g",
              "median_reduction_factor", "n_tg_helped", "n_tg_hurt"):
        assert k in s


def test_effort_unit_consistency_per_bug():
    art = _load()
    for r in art["per_bug"]:
        # assisted = dist + 1 ; unaided = (N+1)/2 ; both in line units
        assert r["assisted_effort_lines"] == r["dist_v5"] + 1
        assert r["unaided_effort_lines"] == (r["search_space_lines"] + 1) / 2.0
        assert r["tg_helped"] == (
            r["assisted_effort_lines"] < r["unaided_effort_lines"])


def test_misleading_bugs_are_kept_not_hidden():
    art = _load()
    s = art["summary"]
    # honesty invariant: hurt count is reported and equals the per-bug tally
    hurt = sum(1 for r in art["per_bug"] if not r["tg_helped"])
    assert s["n_tg_hurt"] == hurt
    assert s["n_tg_helped"] + s["n_tg_hurt"] == s["n_bugs"]


def test_effect_is_large_and_ci_excludes_null():
    art = _load()
    s = art["summary"]
    # Cliff's delta large & CI strictly above 0 (TG helps), reduction CI > 1
    assert s["cliffs_delta"] > 0.474
    assert s["cliffs_delta_magnitude"] == "large"
    assert s["cliffs_delta_ci"][0] > 0.0
    assert s["median_reduction_factor"] > 1.0
    assert s["median_reduction_factor_ci"][0] > 1.0


def test_consumes_only_committed_localized_bugs():
    art = _load()
    with open(SRC_JSON, "r", encoding="utf-8") as fh:
        src = json.load(fh)
    eligible = {it["id"] for it in src["per_item"]
                if it.get("refuted") and it.get("dist_v5") is not None}
    used = {r["id"] for r in art["per_bug"]}
    assert used.issubset(eligible)
    # dist values match the upstream artifact exactly (no re-derivation drift)
    src_dist = {it["id"]: it["dist_v5"] for it in src["per_item"]}
    for r in art["per_bug"]:
        assert r["dist_v5"] == src_dist[r["id"]]


def test_check_mode_is_byte_deterministic():
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
