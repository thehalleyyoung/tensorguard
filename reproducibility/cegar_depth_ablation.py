"""CEGAR refinement-depth ablation: precision/runtime trade-off (Step 119).

The shape-CEGAR loop accumulates shape predicates from counterexamples; its
``max_iterations`` budget is the refinement *depth*. This harness sweeps that
budget over a labeled, real-PyTorch-validated corpus of infeasible-contract bugs
and clean controls (``corpus_extended/cegar_depth_corpus.py``) and measures, at
every depth, what the extra refinement buys and what it costs.

Three quantities per depth ``d``:

* **Detection (recall / FP).** The three-valued verdict on every case. On this
  corpus the direct shape check already refutes every infeasible contract at
  depth 0, so recall is depth-invariant at full -- and, crucially, raising the
  budget never flips a clean control to a false alarm. We report this invariance
  explicitly: CEGAR depth is a *diagnosis* knob, not a soundness knob.

* **Diagnostic precision.** The number of bugs for which TensorGuard emits the
  *contract-level* root-cause diagnosis (the ``CEGAR_REFINED_CONTRACT`` bug:
  "x cannot simultaneously be width A and width B"), which is strictly more
  informative than the generic per-step shape-incompatibility report. This is
  what refinement actually buys: it is zero at depth 0 and rises to full at the
  refinement knee.

* **Work.** The number of CEGAR iterations the loop actually performs at budget
  ``d`` (hardware-independent), summed over the corpus. Because the loop is a
  monotone Houdini-style accumulation it self-terminates at convergence, so work
  saturates at the empirical convergence bound and additional budget is free but
  useless.

The headline is the *knee*: the smallest depth at which diagnostic precision is
full, and the depth at which work saturates. A deterministic wall-clock
companion (``cegar_depth_walltime.json``, VOLATILE) records the same curve in
seconds to make the runtime axis literal.

Only counts and the (small, integer) knees are recorded in the deterministic
artifact, so it is byte-identical across machines.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from corpus_extended.cegar_depth_corpus import (  # noqa: E402
    build_corpus,
    cegar_depth_validate,
)
from src.api import BugCategory, verify_architecture  # noqa: E402
from src.shape_cegar import run_shape_cegar  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "cegar_depth_ablation.json"
OUT_MD = REPO / "reproducibility" / "cegar_depth_ablation.md"
OUT_WALL = REPO / "reproducibility" / "cegar_depth_walltime.json"

MAX_DEPTH = 6


def _verify(source, shapes, depth):
    return verify_architecture(
        source, input_shapes=shapes, max_cegar_iterations=depth,
        soundness_mode="sound",
    )


def _has_refined_contract(res) -> bool:
    return any(b.category == BugCategory.CEGAR_REFINED_CONTRACT for b in res.bugs)


def _iterations(source, shapes, depth) -> int:
    if depth == 0:
        return 0
    return run_shape_cegar(source, input_shapes=shapes, max_iterations=depth).iterations


def measure() -> dict:
    logging.disable(logging.CRITICAL)
    try:
        cases = build_corpus()
        cegar_depth_validate(cases)  # ground truth vs real PyTorch
        conflicts = [c for c in cases if c.family == "conflict"]
        cleans = [c for c in cases if c.family == "clean"]

        per_depth = []
        for d in range(0, MAX_DEPTH + 1):
            bugs_detected = 0  # conflict cases reported UNSAFE
            refined = 0  # conflict cases with contract-level diagnosis
            clean_false_alarms = 0
            total_iters = 0
            for c in conflicts:
                res = _verify(c.source, c.input_shapes, d)
                if res.verdict == "UNSAFE":
                    bugs_detected += 1
                if _has_refined_contract(res):
                    refined += 1
                total_iters += _iterations(c.source, c.input_shapes, d)
            for c in cleans:
                res = _verify(c.source, c.input_shapes, d)
                if res.verdict == "UNSAFE":
                    clean_false_alarms += 1
                total_iters += _iterations(c.source, c.input_shapes, d)
            per_depth.append({
                "depth": d,
                "bugs_detected": bugs_detected,
                "n_conflict": len(conflicts),
                "refined_contract_diagnoses": refined,
                "clean_false_alarms": clean_false_alarms,
                "n_clean": len(cleans),
                "total_cegar_iterations": total_iters,
            })

        recall_full_all_depths = all(
            r["bugs_detected"] == len(conflicts) for r in per_depth
        )
        zero_fp_all_depths = all(r["clean_false_alarms"] == 0 for r in per_depth)

        full_refined = len(conflicts)
        precision_knee = next(
            (r["depth"] for r in per_depth
             if r["refined_contract_diagnoses"] == full_refined),
            None,
        )
        # Work saturation: smallest depth past which total iterations stop rising.
        work_saturation_depth = None
        for i in range(1, len(per_depth)):
            if per_depth[i]["total_cegar_iterations"] == per_depth[i - 1]["total_cegar_iterations"]:
                work_saturation_depth = per_depth[i - 1]["depth"]
                break
        max_iters = max(r["total_cegar_iterations"] for r in per_depth)
        saturated_iters = per_depth[-1]["total_cegar_iterations"]

        data = {
            "step": 119,
            "max_depth": MAX_DEPTH,
            "n_conflict_cases": len(conflicts),
            "n_clean_cases": len(cleans),
            "per_depth": per_depth,
            "precision_knee_depth": precision_knee,
            "work_saturation_depth": work_saturation_depth,
            "refined_diagnoses_at_depth_0": per_depth[0]["refined_contract_diagnoses"],
            "refined_diagnoses_at_knee": full_refined,
            "max_total_iterations": max_iters,
            "saturated_total_iterations": saturated_iters,
            # Honest invariants.
            "recall_is_depth_invariant_full": recall_full_all_depths,
            "zero_false_alarms_all_depths": zero_fp_all_depths,
            "precision_rises_then_plateaus": (
                per_depth[0]["refined_contract_diagnoses"] == 0
                and precision_knee is not None
                and all(
                    per_depth[i]["refined_contract_diagnoses"] == full_refined
                    for i in range(precision_knee, len(per_depth))
                )
            ),
            "work_saturates_at_convergence": (
                work_saturation_depth is not None
                and saturated_iters == max_iters
            ),
        }
        return data
    finally:
        logging.disable(logging.NOTSET)


def measure_walltime(repeats: int = 3) -> dict:
    """Volatile companion: median wall-clock per depth over the corpus."""
    logging.disable(logging.CRITICAL)
    try:
        cases = build_corpus()
        rows = []
        for d in range(0, MAX_DEPTH + 1):
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                for c in cases:
                    _verify(c.source, c.input_shapes, d)
                samples.append(time.perf_counter() - t0)
            rows.append({"depth": d, "median_seconds": round(median(samples), 4)})
        return {"step": 119, "max_depth": MAX_DEPTH, "per_depth_walltime": rows,
                "note": "VOLATILE wall-clock; scientific content is the "
                        "deterministic iteration-count curve."}
    finally:
        logging.disable(logging.NOTSET)


def render_markdown(d: dict) -> str:
    lines = [
        "# CEGAR refinement-depth ablation: precision/runtime trade-off (Step 119)",
        "",
        f"Sweep of the shape-CEGAR budget over **{d['n_conflict_cases']}** "
        f"infeasible-contract bugs and **{d['n_clean_cases']}** clean controls, "
        "every case validated against real PyTorch.",
        "",
        "| depth | bugs detected | refined-contract diagnoses | clean false "
        "alarms | total CEGAR iterations |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in d["per_depth"]:
        lines.append(
            f"| {r['depth']} | {r['bugs_detected']}/{r['n_conflict']} "
            f"| {r['refined_contract_diagnoses']}/{r['n_conflict']} "
            f"| {r['clean_false_alarms']}/{r['n_clean']} "
            f"| {r['total_cegar_iterations']} |"
        )
    lines += [
        "",
        "## Trade-off",
        "",
        f"- detection (recall) is depth-invariant at full and clean false alarms "
        f"stay zero at every depth — CEGAR depth is a diagnosis knob, not a "
        f"soundness knob: **{d['recall_is_depth_invariant_full'] and d['zero_false_alarms_all_depths']}**",
        f"- contract-level diagnostic precision rises from "
        f"{d['refined_diagnoses_at_depth_0']} at depth 0 to full "
        f"({d['refined_diagnoses_at_knee']}) at the refinement knee "
        f"(depth {d['precision_knee_depth']}), then plateaus: "
        f"**{d['precision_rises_then_plateaus']}**",
        f"- work saturates at the convergence bound (depth "
        f"{d['work_saturation_depth']}, {d['saturated_total_iterations']} total "
        f"iterations); beyond it the budget is free but useless: "
        f"**{d['work_saturates_at_convergence']}**",
        "",
    ]
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("cegar_depth_ablation: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    OUT_WALL.write_text(json.dumps(measure_walltime(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_WALL} (volatile)")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
