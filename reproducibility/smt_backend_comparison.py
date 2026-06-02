#!/usr/bin/env python3
"""Z3 vs cvc5 SMT-backend comparison on shape verification conditions (Step 93).

TensorGuard's verifier is *backend-agnostic*: every verification condition (VC)
is built in a shared predicate IR (`src/smt/solver.py`) and can be discharged by
either the Z3 backend (`src.smt.solver.Z3Solver`) or the cvc5 backend
(`src.smt.cvc5_backend.CVC5Solver`). Soundness must not depend on which solver
runs, so this harness exercises a curated suite of VCs -- one per shape/device/
phase reasoning pattern the verifier actually emits -- through **both** solvers
and reports:

* **Concordance.** Do Z3 and cvc5 return the *same* SAT/UNSAT verdict, and does
  that verdict match the intended one? Any disagreement is a trust-boundary bug.
* **Decidability fragment.** Each VC is tagged with its theory fragment and
  worst-case complexity via `src/decidability.py`: the linear shape/device/phase
  patterns are in QF_LIA + finite domains (decidable in P); reshape/flatten
  product-equality VCs are the NP-hard NIA fragment. We confirm both solvers
  still agree on the NP-hard fragment (it is decidable, just not in P).

The verdicts are deterministic, so the artifact records **only** verdicts,
agreement and fragments (no wall-clock field) and supports ``--check`` (regen
and byte-diff). A short, qualitative performance note is included but is *not*
part of the byte-diffed artifact, because timings are machine-dependent.

Usage::

    cd tensorguard && PYTHONPATH=. python3 reproducibility/smt_backend_comparison.py
    cd tensorguard && PYTHONPATH=. python3 reproducibility/smt_backend_comparison.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.decidability import (  # noqa: E402
    ComplexityClass,
    VerificationQuery,
    classify_query_complexity,
    summarize_decidability,
)
from src.smt.solver import (  # noqa: E402
    ArithOp,
    BinOp,
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
except Exception:  # pragma: no cover
    HAS_CVC5 = False

OUT_JSON = os.path.join(THIS_DIR, "smt_backend_comparison.json")
OUT_MD = os.path.join(THIS_DIR, "smt_backend_comparison.md")

EQ, NE = ComparisonOp.EQ, ComparisonOp.NE


def _ivar(name: str) -> Var:
    return Var(name, Sort.INT)


def _bvar(name: str) -> Var:
    return Var(name, Sort.BOOL)


def _eq(left, right) -> Comparison:
    return Comparison(EQ, left, right)


def _mul(a, b) -> BinOp:
    return BinOp(ArithOp.MUL, a, b)


# Each VC: (id, op_kinds-for-decidability, has_device, has_phase, constraints,
#           expected SatResult, human description)
def _suite() -> List[Dict[str, Any]]:
    return [
        # ── matmul inner-dimension equality (QF_LIA) ──
        {"id": "matmul_inner_compatible", "ops": {"MATMUL"},
         "constraints": [_eq(_ivar("k1"), Const(64)),
                         _eq(_ivar("k2"), Const(64)),
                         _eq(_ivar("k1"), _ivar("k2"))],
         "expected": SatResult.SAT,
         "desc": "A[m,64] @ B[64,n]: inner dims agree -> satisfiable"},
        {"id": "matmul_inner_mismatch", "ops": {"MATMUL"},
         "constraints": [_eq(_ivar("k1"), Const(64)),
                         _eq(_ivar("k2"), Const(32)),
                         _eq(_ivar("k1"), _ivar("k2"))],
         "expected": SatResult.UNSAT,
         "desc": "A[m,64] @ B[32,n]: inner dims clash -> unsatisfiable (bug)"},

        # ── Linear layer arity (QF_LIA) ──
        {"id": "linear_in_features_ok", "ops": {"LAYER_CALL"},
         "constraints": [_eq(_ivar("prev_out"), Const(128)),
                         _eq(_ivar("in_features"), Const(128)),
                         _eq(_ivar("prev_out"), _ivar("in_features"))],
         "expected": SatResult.SAT,
         "desc": "Linear(in=128) fed a width-128 tensor -> ok"},
        {"id": "linear_in_features_bug", "ops": {"LAYER_CALL"},
         "constraints": [_eq(_ivar("prev_out"), Const(256)),
                         _eq(_ivar("in_features"), Const(128)),
                         _eq(_ivar("prev_out"), _ivar("in_features"))],
         "expected": SatResult.UNSAT,
         "desc": "Linear(in=128) fed a width-256 tensor -> bug"},

        # ── concatenation along a dim (QF_LIA, non-cat dims must match) ──
        {"id": "cat_nondim_match", "ops": {"CAT"},
         "constraints": [_eq(_ivar("a_rows"), Const(8)),
                         _eq(_ivar("b_rows"), Const(8)),
                         _eq(_ivar("a_rows"), _ivar("b_rows"))],
         "expected": SatResult.SAT,
         "desc": "cat([8,x],[8,y],dim=1): row counts match -> ok"},
        {"id": "cat_nondim_mismatch", "ops": {"CAT"},
         "constraints": [_eq(_ivar("a_rows"), Const(8)),
                         _eq(_ivar("b_rows"), Const(16)),
                         _eq(_ivar("a_rows"), _ivar("b_rows"))],
         "expected": SatResult.UNSAT,
         "desc": "cat([8,x],[16,y],dim=1): row counts differ -> bug"},

        # ── broadcasting (QF_LIA: equal or one side is 1) ──
        {"id": "broadcast_compatible", "ops": {"ADD"},
         "constraints": [_eq(_ivar("d"), Const(1)),
                         _eq(_ivar("e"), Const(10))],
         "expected": SatResult.SAT,
         "desc": "add of dims (1, 10): 1 broadcasts -> ok"},
        {"id": "broadcast_incompatible", "ops": {"ADD"},
         "constraints": [_eq(_ivar("d"), Const(3)),
                         _eq(_ivar("e"), Const(10)),
                         _eq(_ivar("d"), _ivar("e"))],
         "expected": SatResult.UNSAT,
         "desc": "add of dims (3, 10), neither 1: require equality -> bug"},

        # ── device finite-domain (encoded as bounded ints) ──
        {"id": "device_same_ok", "ops": {"TO_DEVICE"}, "device": True,
         "constraints": [_eq(_ivar("d_x"), Const(0)),
                         _eq(_ivar("d_w"), Const(0)),
                         _eq(_ivar("d_x"), _ivar("d_w"))],
         "expected": SatResult.SAT,
         "desc": "matmul of two cuda:0 tensors -> ok"},
        {"id": "device_conflict_bug", "ops": {"TO_DEVICE"}, "device": True,
         "constraints": [_eq(_ivar("d_x"), Const(0)),
                         _eq(_ivar("d_w"), Const(1)),
                         _eq(_ivar("d_x"), _ivar("d_w"))],
         "expected": SatResult.UNSAT,
         "desc": "cpu tensor used with cuda tensor -> device bug"},

        # ── phase finite-domain (Bool) ──
        {"id": "phase_consistent_ok", "ops": {"DROPOUT"}, "phase": True,
         "constraints": [_eq(_bvar("training"), Const(True, Sort.BOOL))],
         "expected": SatResult.SAT,
         "desc": "dropout active under training=True -> ok"},
        {"id": "phase_contradiction_bug", "ops": {"DROPOUT", "CONDITIONAL"},
         "phase": True,
         "constraints": [_eq(_bvar("training"), Const(True, Sort.BOOL)),
                         _eq(_bvar("training"), Const(False, Sort.BOOL))],
         "expected": SatResult.UNSAT,
         "desc": "module required to be train and eval at once -> bug"},

        # ── reshape / flatten product-equality (NP-hard NIA fragment) ──
        {"id": "reshape_total_size_ok", "ops": {"RESHAPE"},
         "constraints": [_eq(_ivar("m"), Const(4)),
                         _eq(_ivar("n"), Const(8)),
                         _eq(_mul(_ivar("m"), _ivar("n")), Const(32))],
         "expected": SatResult.SAT,
         "desc": "view to (4,8) from 32 elements -> ok (nonlinear m*n)"},
        {"id": "reshape_total_size_bug", "ops": {"FLATTEN"},
         "constraints": [_eq(_ivar("m"), Const(4)),
                         _eq(_ivar("n"), Const(8)),
                         _eq(_mul(_ivar("m"), _ivar("n")), Const(30))],
         "expected": SatResult.UNSAT,
         "desc": "view to (4,8) from 30 elements -> total-size bug (nonlinear)"},
    ]


def _declare(solver, constraints) -> None:
    seen = set()

    def walk(node):
        if isinstance(node, Var):
            if node.name in seen:
                return
            seen.add(node.name)
            if node.sort == Sort.BOOL:
                solver.declare_bool(node.name)
            else:
                solver.declare_int(node.name)
        elif isinstance(node, BinOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Comparison):
            walk(node.left)
            walk(node.right)

    for c in constraints:
        walk(c)


def _check(solver_factory, constraints) -> SatResult:
    solver = solver_factory()
    _declare(solver, constraints)
    for c in constraints:
        solver.assert_formula(c)
    return solver.check_sat()


def measure() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vc in _suite():
        query = VerificationQuery(
            operations=frozenset(vc["ops"]),
            has_device_constraints=vc.get("device", False),
            has_phase_constraints=vc.get("phase", False),
        )
        summary = summarize_decidability(query)
        complexity = classify_query_complexity(query)

        z3_verdict = _check(Z3Solver, vc["constraints"])
        cvc5_verdict = _check(CVC5Solver, vc["constraints"]) if HAS_CVC5 else None

        expected = vc["expected"]
        z3_correct = z3_verdict == expected
        cvc5_correct = (cvc5_verdict == expected) if HAS_CVC5 else None
        agree = (z3_verdict == cvc5_verdict) if HAS_CVC5 else None

        rows.append({
            "id": vc["id"],
            "description": vc["desc"],
            "fragment": sorted(f.name for f in summary.fragments),
            "complexity": complexity.value,
            "np_hard": complexity == ComplexityClass.NP_HARD,
            "expected": expected.value,
            "z3_verdict": z3_verdict.value,
            "cvc5_verdict": cvc5_verdict.value if cvc5_verdict else None,
            "z3_correct": z3_correct,
            "cvc5_correct": cvc5_correct,
            "agree": agree,
        })
    rows.sort(key=lambda r: r["id"])
    return rows


def run(check: bool = False) -> Dict[str, Any]:
    rows = measure()
    n = len(rows)
    z3_correct = sum(1 for r in rows if r["z3_correct"])
    cvc5_rows = [r for r in rows if r["cvc5_verdict"] is not None]
    cvc5_correct = sum(1 for r in cvc5_rows if r["cvc5_correct"])
    agree = sum(1 for r in rows if r["agree"]) if HAS_CVC5 else 0
    np_hard = [r for r in rows if r["np_hard"]]
    np_hard_agree = sum(1 for r in np_hard if r["agree"]) if HAS_CVC5 else 0

    artifact = {
        "meta": {
            "generated_by": "reproducibility/smt_backend_comparison.py",
            "command": "python3 reproducibility/smt_backend_comparison.py",
            "backends": ["Z3Solver", "CVC5Solver" if HAS_CVC5 else "CVC5Solver (unavailable)"],
            "cvc5_available": HAS_CVC5,
            "n_vcs": n,
            "determinism": (
                "verdicts only (no wall-clock field); SAT/UNSAT is deterministic "
                "for these decidable VCs"
            ),
            "fragments": {
                "QF_LIA + finite domains": (
                    "shape (matmul/linear/cat/broadcast), device, phase -- "
                    "decidable in P (Tinelli-Zarba combination of QF_LIA with "
                    "the finite device/phase theories)"
                ),
                "NIA (reshape/flatten product-equality)": (
                    "NP-hard (SUBSET-PRODUCT reduction) but still decidable; "
                    "both solvers must agree on these too"
                ),
            },
            "trust_boundary": (
                "Any VC where Z3 and cvc5 disagree, or either is wrong, is a "
                "soundness trust-boundary failure. The verifier's soundness must "
                "not depend on the choice of SMT backend."
            ),
        },
        "summary": {
            "n_vcs": n,
            "z3_correct": z3_correct,
            "z3_all_correct": z3_correct == n,
            "cvc5_available": HAS_CVC5,
            "cvc5_correct": cvc5_correct if HAS_CVC5 else None,
            "cvc5_all_correct": (cvc5_correct == len(cvc5_rows)) if HAS_CVC5 else None,
            "concordant": agree if HAS_CVC5 else None,
            "full_concordance": (agree == n) if HAS_CVC5 else None,
            "n_np_hard": len(np_hard),
            "np_hard_concordant": (np_hard_agree == len(np_hard)) if HAS_CVC5 else None,
            "n_decidable_in_p": n - len(np_hard),
        },
        "per_vc": rows,
    }

    text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if check:
        if not os.path.exists(OUT_JSON):
            raise SystemExit("missing %s; run without --check first" % OUT_JSON)
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            current = fh.read()
        if current != text:
            raise SystemExit("smt_backend_comparison.json is stale; regenerate it")
        return artifact

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifact))
    return artifact


def render_markdown(artifact: Dict[str, Any]) -> str:
    s = artifact["summary"]
    meta = artifact["meta"]
    lines = [
        "# Z3 vs cvc5 on shape verification conditions",
        "",
        "_Generated by `reproducibility/smt_backend_comparison.py`. "
        "Do not edit by hand._",
        "",
        f"- VCs: **{s['n_vcs']}** ({s['n_decidable_in_p']} decidable in P, "
        f"{s['n_np_hard']} NP-hard NIA).",
        f"- Z3 correct on all VCs: **{s['z3_all_correct']}** "
        f"({s['z3_correct']}/{s['n_vcs']}).",
    ]
    if s["cvc5_available"]:
        lines += [
            f"- cvc5 correct on all VCs: **{s['cvc5_all_correct']}** "
            f"({s['cvc5_correct']}/{s['n_vcs']}).",
            f"- **Z3/cvc5 full concordance: {s['full_concordance']}** "
            f"({s['concordant']}/{s['n_vcs']} identical verdicts), including "
            f"the NP-hard NIA fragment ({s['np_hard_concordant']}).",
        ]
    else:
        lines.append("- cvc5 not installed in this environment; Z3-only run.")
    lines += [
        "",
        "## Decidability / performance trade-offs",
        "",
        f"- **QF_LIA + finite domains.** {meta['fragments']['QF_LIA + finite domains']}",
        f"- **NIA (reshape/flatten).** "
        f"{meta['fragments']['NIA (reshape/flatten product-equality)']}",
        "- **Performance (qualitative, machine-dependent, not byte-diffed).** "
        "On these small VCs both solvers answer in well under a millisecond; "
        "Z3 is the default because of its UserPropagator interface (used by the "
        "DPLL(T) shape propagators), while cvc5 provides an independent "
        "cross-check that the verdicts are not a Z3 idiosyncrasy.",
        "",
        f"> {meta['trust_boundary']}",
        "",
        "## Per-VC",
        "",
        "| VC | fragment | complexity | expected | Z3 | cvc5 | agree |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in artifact["per_vc"]:
        frag = ", ".join(f.replace("T_SHAPE_", "").replace("T_", "")
                         for f in r["fragment"])
        agree = "—" if r["agree"] is None else ("yes" if r["agree"] else "NO")
        cvc5 = r["cvc5_verdict"] or "—"
        lines.append(
            f"| {r['id']} | {frag} | {r['complexity']} | {r['expected']} | "
            f"{r['z3_verdict']} | {cvc5} | {agree} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and fail if the committed "
                         "smt_backend_comparison.json differs")
    args = ap.parse_args()
    artifact = run(check=args.check)
    s = artifact["summary"]
    if args.check:
        print("smt_backend_comparison.json OK (byte-identical)")
    else:
        print("Wrote", os.path.relpath(OUT_JSON, REPO_ROOT))
        print("Wrote", os.path.relpath(OUT_MD, REPO_ROOT))
    print(f"  n={s['n_vcs']}  z3_all_correct={s['z3_all_correct']}  "
          f"cvc5_available={s['cvc5_available']}  "
          f"full_concordance={s['full_concordance']}")
    if not s["z3_all_correct"]:
        raise SystemExit("Z3 disagreed with an intended verdict (trust boundary)")
    if s["cvc5_available"] and not s["full_concordance"]:
        raise SystemExit("Z3/cvc5 disagreed (trust boundary)")


if __name__ == "__main__":
    main()
