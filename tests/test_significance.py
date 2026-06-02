"""Step 88 — significance testing of the precision/recall study.

Covers both the statistics primitives in ``src.statistical_rigor`` (McNemar
exact test, paired bootstrap) against closed-form/known values, and the
``evaluation/significance.py`` harness end-to-end on the committed confusion
artifact.
"""

import json
import math
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.statistical_rigor import (  # noqa: E402
    mcnemar_exact_test,
    mcnemar_from_correctness,
    paired_bootstrap_accuracy_diff,
)

CONFUSION = os.path.join(_ROOT, "evaluation", "confusion_matrices.json")
SIG_JSON = os.path.join(_ROOT, "evaluation", "significance.json")
SIG_PY = os.path.join(_ROOT, "evaluation", "significance.py")


# ── statistics primitives ────────────────────────────────────────────────────

def _exact_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


@pytest.mark.parametrize("b,c", [(0, 0), (1, 0), (5, 5), (8, 1), (10, 0), (3, 7), (12, 4)])
def test_mcnemar_matches_closed_form(b, c):
    r = mcnemar_exact_test(b, c)
    assert r.b == b and r.c == c
    assert r.n_discordant == b + c
    assert r.statistic == min(b, c)
    assert abs(r.p_value - _exact_two_sided(b, c)) < 1e-12


def test_mcnemar_no_discordant_is_p1():
    r = mcnemar_exact_test(0, 0)
    assert r.p_value == 1.0
    assert r.odds_ratio is None


def test_mcnemar_symmetry_in_swap():
    # Swapping b and c must not change the two-sided p-value.
    assert mcnemar_exact_test(8, 1).p_value == mcnemar_exact_test(1, 8).p_value


def test_mcnemar_lopsided_is_significant():
    # 10-0 split -> p = 2 * 0.5^10 ≈ 0.00195, clearly < 0.05.
    assert mcnemar_exact_test(10, 0).p_value < 0.05
    # 1-0 split is not.
    assert mcnemar_exact_test(1, 0).p_value > 0.05


def test_mcnemar_from_correctness_counts_discordant():
    a = [True, True, False, True, False]
    b = [True, False, False, False, True]
    # discordant: idx1 (a right, b wrong) -> b; idx3 (a right, b wrong) -> b;
    # idx4 (a wrong, b right) -> c. => b=2, c=1
    r = mcnemar_from_correctness(a, b)
    assert (r.b, r.c) == (2, 1)


def test_mcnemar_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar_from_correctness([True], [True, False])
    with pytest.raises(ValueError):
        mcnemar_exact_test(-1, 0)


def test_paired_bootstrap_is_deterministic_and_well_formed():
    a = [True] * 12 + [False] * 4
    b = [True] * 8 + [False] * 8
    r1 = paired_bootstrap_accuracy_diff(a, b, n_resamples=3000, seed=0)
    r2 = paired_bootstrap_accuracy_diff(a, b, n_resamples=3000, seed=0)
    assert r1.point_estimate == r2.point_estimate == pytest.approx(0.25)
    assert r1.ci_low == r2.ci_low and r1.ci_high == r2.ci_high  # seeded
    assert r1.ci_low <= r1.point_estimate <= r1.ci_high
    assert 0.0 <= r1.fraction_above_zero <= 1.0


def test_paired_bootstrap_identical_methods_is_zero():
    v = [True, False, True, True, False]
    r = paired_bootstrap_accuracy_diff(v, v, n_resamples=2000, seed=1)
    assert r.point_estimate == 0.0
    assert r.ci_low == 0.0 and r.ci_high == 0.0


# ── end-to-end harness ───────────────────────────────────────────────────────

def _run(args):
    env = dict(os.environ, PYTHONPATH=_ROOT)
    return subprocess.run(
        [sys.executable, SIG_PY, *args],
        cwd=_ROOT, capture_output=True, text=True, timeout=300, env=env,
    )


def test_significance_artifact_is_committed_and_fresh():
    assert os.path.exists(SIG_JSON), "run evaluation/significance.py first"
    proc = _run(["--check"])
    assert proc.returncode == 0, (
        f"significance.json is stale:\n{proc.stdout}\n{proc.stderr}"
    )


def test_significance_artifact_structure_and_claims():
    with open(SIG_JSON) as fh:
        art = json.load(fh)
    meta = art["meta"]
    assert meta["reference_method"] == "tensorguard"
    assert meta["alpha"] == 0.05
    with open(CONFUSION) as fh:
        conf = json.load(fh)
    assert meta["n_items"] == len(conf["per_model"])

    by_base = {c["baseline"]: c for c in art["comparisons"]}
    # The four non-reference methods are all compared.
    assert set(by_base) == {"runtime_forward", "runtime_backward", "pytea", "noop"}

    for c in art["comparisons"]:
        mc = c["mcnemar"]
        assert mc["n_discordant"] == mc["b_ref_right_base_wrong"] + mc["c_ref_wrong_base_right"]
        assert 0.0 <= mc["p_value"] <= 1.0
        bs = c["accuracy_diff_bootstrap"]
        assert bs["ci_low"] <= bs["point_estimate"] <= bs["ci_high"]

    # The trivial 'always clean' detector must be beaten significantly: on this
    # balanced corpus TensorGuard is right on every buggy model where noop is
    # wrong, an 8-0 discordant split (exact p ≈ 0.0078) -> significant post-Holm.
    noop = by_base["noop"]["mcnemar"]
    assert noop["c_ref_wrong_base_right"] == 0
    assert noop["p_value"] < 0.05
    assert noop["significant_at_alpha"] is True

    # Honesty: TensorGuard never loses a discordant pair to any baseline
    # (c == 0 everywhere) — it is a strict dominance, not a trade-off.
    for c in art["comparisons"]:
        assert c["mcnemar"]["c_ref_wrong_base_right"] == 0


def test_holm_correction_is_applied_to_usable_family():
    with open(SIG_JSON) as fh:
        art = json.load(fh)
    for c in art["comparisons"]:
        if c["usable"]:
            mc = c["mcnemar"]
            assert "holm_adjusted_p" in mc
            # Holm-adjusted p is never smaller than the raw p.
            assert mc["holm_adjusted_p"] >= mc["p_value"] - 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
