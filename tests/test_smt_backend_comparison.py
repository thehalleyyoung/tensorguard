"""Step 93: Z3 vs cvc5 backend comparison on shape verification conditions.

Validates the committed comparison artifact
(``reproducibility/smt_backend_comparison.json``): both SMT backends reach the
intended SAT/UNSAT verdict on every curated VC, they are fully concordant
(including on the NP-hard NIA reshape fragment), the decidability tags are
correct, and the harness is byte-deterministic. Also re-runs the two backends
live on a handful of VCs to guard against artifact staleness.
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

from src.smt.solver import (  # noqa: E402
    Comparison,
    ComparisonOp,
    Const,
    SatResult,
    Sort,
    Var,
    Z3Solver,
)

try:
    from src.smt.cvc5_backend import CVC5Solver
    HAS_CVC5 = True
except Exception:
    HAS_CVC5 = False

ART = os.path.join(REPO_ROOT, "reproducibility", "smt_backend_comparison.json")
SCRIPT = os.path.join(REPO_ROOT, "reproducibility", "smt_backend_comparison.py")


def _load():
    with open(ART, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_artifact_present_and_shaped():
    art = _load()
    assert art["meta"]["n_vcs"] == len(art["per_vc"]) >= 12


def test_z3_correct_on_every_vc():
    art = _load()
    for r in art["per_vc"]:
        assert r["z3_verdict"] == r["expected"], r["id"]
        assert r["z3_correct"] is True, r["id"]
    assert art["summary"]["z3_all_correct"] is True


@pytest.mark.skipif(not HAS_CVC5, reason="cvc5 not installed")
def test_full_concordance_including_np_hard():
    art = _load()
    s = art["summary"]
    assert s["cvc5_available"] is True
    assert s["full_concordance"] is True
    assert s["np_hard_concordant"] is True
    assert s["n_np_hard"] >= 2
    for r in art["per_vc"]:
        assert r["agree"] is True, r["id"]
        assert r["cvc5_verdict"] == r["expected"], r["id"]


def test_decidability_tags_consistent():
    art = _load()
    for r in art["per_vc"]:
        # the reshape/flatten VCs are the only NP-hard ones
        is_reshape = "T_SHAPE_RESHAPE" in r["fragment"]
        assert r["np_hard"] == is_reshape, r["id"]
        assert (r["complexity"] == "NP-hard") == is_reshape, r["id"]


def test_both_verdicts_present_per_vc():
    art = _load()
    if not HAS_CVC5:
        pytest.skip("cvc5 not installed")
    for r in art["per_vc"]:
        assert r["z3_verdict"] in {"sat", "unsat"}
        assert r["cvc5_verdict"] in {"sat", "unsat"}


def test_no_volatile_field_in_artifact():
    art = _load()

    def scan(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert not any(t in k.lower() for t in
                               ("time", "elapsed", "timestamp", "wallclock")), k
                scan(v)
        elif isinstance(o, list):
            for v in o:
                scan(v)

    scan(art)


def _check(solver_cls, decls_bool, decls_int, constraints):
    s = solver_cls()
    for d in decls_int:
        s.declare_int(d)
    for d in decls_bool:
        s.declare_bool(d)
    for c in constraints:
        s.assert_formula(c)
    return s.check_sat()


def test_live_matmul_mismatch_unsat_on_both():
    cons = [
        Comparison(ComparisonOp.EQ, Var("k1"), Const(64)),
        Comparison(ComparisonOp.EQ, Var("k2"), Const(32)),
        Comparison(ComparisonOp.EQ, Var("k1"), Var("k2")),
    ]
    assert _check(Z3Solver, [], ["k1", "k2"], cons) == SatResult.UNSAT
    if HAS_CVC5:
        assert _check(CVC5Solver, [], ["k1", "k2"], cons) == SatResult.UNSAT


def test_live_phase_contradiction_unsat_on_both():
    cons = [
        Comparison(ComparisonOp.EQ, Var("training", Sort.BOOL), Const(True, Sort.BOOL)),
        Comparison(ComparisonOp.EQ, Var("training", Sort.BOOL), Const(False, Sort.BOOL)),
    ]
    assert _check(Z3Solver, ["training"], [], cons) == SatResult.UNSAT
    if HAS_CVC5:
        assert _check(CVC5Solver, ["training"], [], cons) == SatResult.UNSAT


def test_check_mode_is_byte_deterministic():
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
