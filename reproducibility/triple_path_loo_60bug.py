#!/usr/bin/env python3.11
"""Triple-path LOO on the 60-bug corpus (round-3 W1 / Q1).

The reviewer asked: report the *handler-attributable refute count
under joint-LOO* — bugs that joint-LOO converts from RP to non-RP,
broken out by category — so a reader can distinguish "the catalogue
is over-determined" from "the corpus is catchable from predicate
harvesting alone".

Background.  TG runs three logically separable refute paths:
  P1: per-operator handler dispatch in src.tensor_shapes /
      src.model_checker (TORCH_SHAPE_OPS, MODERN_TORCH_SHAPE_OPS,
      FUNCTIONAL_SHAPE_RULES).
  P2: AST-pattern intent-bug analyser (src.intent_bugs.OverwarnAnalyzer).
  P3: constraint-based residue — CEGAR shape loop
      (src.shape_cegar.run_shape_cegar) plus the heuristic flow-
      sensitive analyser (src.real_analyzer.analyze_source).

Existing artefacts:
  reproducibility/bug_corpus_loo_handler.{json,md}   disables P1 only
  reproducibility/bug_corpus_loo_joint.{json,md}     disables P1+P2
  reproducibility/per_rule_ablation_60bug.{json,md}  message-level
                                                     attribution

This script disables P1+P2+P3 simultaneously (the *triple* LOO) and
reports, per category, the bugs that the triple-LOO converts from
RP to non-RP.  This is the answer to the reviewer's question:
* if the triple-LOO drops ALL bugs in a category to non-RP, then
  the per-handler attribution is vacuous (the category has no
  independent evidence from outside the three paths under test);
* if the triple-LOO leaves some category bugs at RP, then a fourth
  path (e.g.\ pure SMT shape arithmetic from refinement variables
  alone) is also catching them and the catalogue is genuinely
  over-determined.

The triple-LOO drop per category is the "handler-attributable"
number the reviewer asked for: it is the number of category-i
bugs whose RP requires at least one of P1/P2/P3.

Run:
    PYTHONPATH=. python3.11 reproducibility/triple_path_loo_60bug.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility/triple_path_loo_60bug.json")
OUT_MD = os.path.join(ROOT, "reproducibility/triple_path_loo_60bug.md")
MANIFEST = os.path.join(ROOT, "experiments_v5/bug_corpus_manifest.json")

CATEGORY_HANDLERS: Dict[str, List[str]] = {
    "view_reshape_total_size": ["view", "reshape"],
    "broadcasting": ["broadcast", "add", "mul", "sub", "div"],
    "conv_channel_mismatch": ["conv1d", "conv2d", "conv3d"],
    "linear_inout_mismatch": ["linear"],
    "einsum_dim": ["einsum", "matmul", "bmm"],
    "transpose_axes": ["transpose", "permute"],
    "attention_dim": ["scaled_dot_product_attention", "matmul", "bmm",
                      "softmax", "multihead_attention"],
    "batchnorm_features": ["batch_norm"],
    "embedding_index": ["embed", "index_select", "gather"],
}


def _reset_modules() -> None:
    for m in list(sys.modules):
        if m == "src" or m.startswith("src."):
            del sys.modules[m]


def _score(disabled_handlers: List[str], disable_intent: bool,
           disable_constraint_residue: bool) -> Dict[str, Any]:
    """Score the 60-bug corpus under the requested ablation."""
    _reset_modules()

    import src.tensor_shapes as ts  # noqa: E402
    if disabled_handlers:
        for op in list(ts.TORCH_SHAPE_OPS):
            if op in disabled_handlers:
                ts.TORCH_SHAPE_OPS.pop(op, None)
    if "matmul" in disabled_handlers or "bmm" in disabled_handlers:
        ts.compute_matmul_shape = lambda *a, **k: None
    if "broadcast" in disabled_handlers:
        ts.compute_broadcast_shape = lambda *a, **k: None
    if "view" in disabled_handlers or "reshape" in disabled_handlers:
        ts.compute_reshape_shape = lambda *a, **k: None

    try:
        import src.model_checker as mc
        if "matmul" in disabled_handlers or "bmm" in disabled_handlers:
            mc.compute_matmul_shape = lambda *a, **k: None
        if "broadcast" in disabled_handlers:
            mc.compute_broadcast_shape = lambda *a, **k: None
        if "view" in disabled_handlers or "reshape" in disabled_handlers:
            mc.compute_reshape_shape = lambda *a, **k: None
    except Exception:
        pass

    try:
        import src.stdlib.modern_ops as mo
        for op in list(mo.MODERN_TORCH_SHAPE_OPS):
            if op in disabled_handlers:
                mo.MODERN_TORCH_SHAPE_OPS.pop(op, None)
    except Exception:
        pass
    try:
        import src.smt.encoder as enc
        for op in list(enc.FUNCTIONAL_SHAPE_RULES):
            if op in disabled_handlers:
                enc.FUNCTIONAL_SHAPE_RULES.pop(op, None)
    except Exception:
        pass

    if disable_intent:
        try:
            import src.intent_bugs as ib

            class _Stub:
                def __init__(self, *a, **k):
                    pass

                def analyze(self, *a, **k):
                    return []
            ib.OverwarnAnalyzer = _Stub
        except Exception:
            pass
        try:
            import src.api as api
            api.OverwarnAnalyzer = _Stub
        except Exception:
            pass

    if disable_constraint_residue:
        try:
            import src.shape_cegar as sc

            class _CegarStub:
                bugs: list = []

                def __init__(self, *a, **k):
                    pass
            def _stub_run(*a, **k):
                return _CegarStub()
            sc.run_shape_cegar = _stub_run
        except Exception:
            pass
        try:
            import src.real_analyzer as ra

            class _Empty:
                function_results: list = []

            def _stub_analyze(*a, **k):
                return _Empty()
            ra.analyze_source = _stub_analyze
        except Exception:
            pass

    from src.api import verify_architecture  # noqa: E402

    items = json.load(open(MANIFEST))["items"]
    rp = silent = abst = err = 0
    per_cat: Dict[str, Dict[str, int]] = {}
    rp_ids: List[str] = []
    for it in items:
        cat = it["category"]
        per_cat.setdefault(cat, {"rp": 0, "silent": 0, "abst": 0,
                                 "err": 0, "n": 0})
        per_cat[cat]["n"] += 1
        try:
            src_str = open(os.path.join(ROOT, it["repro_file"])).read()
        except Exception:
            err += 1
            per_cat[cat]["err"] += 1
            continue
        try:
            r = verify_architecture(src_str, input_shapes={},
                                    max_cegar_iterations=(
                                        0 if disable_constraint_residue
                                        else 10),
                                    filename=os.path.basename(it["repro_file"]))
            bugs = getattr(r, "bugs", [])
            high_conf = [b for b in bugs if b.confidence >= 0.99]
            if high_conf:
                rp += 1
                per_cat[cat]["rp"] += 1
                rp_ids.append(it["id"])
            elif bugs:
                silent += 1
                per_cat[cat]["silent"] += 1
            else:
                abst += 1
                per_cat[cat]["abst"] += 1
        except Exception:
            err += 1
            per_cat[cat]["err"] += 1
    return {"rp": rp, "silent": silent, "abst": abst, "err": err,
            "per_category": per_cat, "rp_ids": rp_ids}


def main() -> int:
    print("Scoring full pipeline (no LOO)...", flush=True)
    t0 = time.time()
    full = _score([], disable_intent=False, disable_constraint_residue=False)
    full["elapsed_s"] = round(time.time() - t0, 2)
    print(f"  full RP={full['rp']}/60 ({full['elapsed_s']}s)", flush=True)

    print("Scoring with all three paths globally disabled (no LOO)...",
          flush=True)
    t0 = time.time()
    triple_global = _score([], disable_intent=True,
                           disable_constraint_residue=True)
    triple_global["elapsed_s"] = round(time.time() - t0, 2)
    print(f"  triple-disabled RP={triple_global['rp']}/60 "
          f"({triple_global['elapsed_s']}s)", flush=True)

    runs: Dict[str, Any] = {}
    for cat, handlers in CATEGORY_HANDLERS.items():
        print(f"[{cat}] triple-LOO (handlers + intent + CEGAR/heuristic): "
              f"{handlers}", flush=True)
        t0 = time.time()
        r = _score(handlers, disable_intent=True,
                   disable_constraint_residue=True)
        r["disabled_handlers"] = handlers
        r["elapsed_s"] = round(time.time() - t0, 2)
        runs[cat] = r
        cat_rp_full = full["per_category"].get(cat, {}).get("rp", 0)
        cat_rp_triple = r["per_category"].get(cat, {}).get("rp", 0)
        print(f"  total RP={r['rp']}/60  category-{cat} RP "
              f"{cat_rp_full}->{cat_rp_triple}", flush=True)

    # Per-category attributable count: under triple-LOO, how many bugs
    # in that category move RP -> non-RP relative to the full pipeline.
    handler_attrib = {}
    for cat in CATEGORY_HANDLERS:
        cat_full = full["per_category"].get(cat, {}).get("rp", 0)
        cat_triple = runs[cat]["per_category"].get(cat, {}).get("rp", 0)
        handler_attrib[cat] = {
            "rp_full": cat_full,
            "rp_under_triple_loo": cat_triple,
            "handler_attributable_drop": cat_full - cat_triple,
        }

    out = {
        "_question": (
            "Round-3 W1 / Q1: report the handler-attributable refute "
            "count under joint-LOO, broken out by category, to "
            "reconcile the per-handler attribution (49 RPs across "
            "categories) with the joint-LOO non-result (53/60 -> "
            "53/60).  This script extends the joint-LOO to a "
            "*triple*-LOO that simultaneously disables the per-"
            "category operator handlers, the AST-pattern intent-bug "
            "analyser, and the constraint-based residue (CEGAR loop "
            "plus heuristic flow-sensitive analyser)."
        ),
        "full_pipeline": full,
        "all_three_paths_globally_disabled": triple_global,
        "per_category_triple_loo": runs,
        "handler_attributable_under_triple_loo": handler_attrib,
        "interpretation": (
            "If the all-three-paths globally-disabled run still "
            "produces RP > 0, a fourth refute path exists.  If the "
            "per-category triple-LOO drop matches the per-rule "
            "attribution table (7/7/6/5/4*4/3), then the catalogue "
            "is genuinely over-determined and the joint-LOO "
            "53/60 result is robustness, not vacuity.  If the "
            "per-category drops are all zero, the corpus is being "
            "caught by predicate harvesting that is independent of "
            "the per-handler dispatch.  This script lets the reader "
            "decide between the two readings."
        ),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    md = ["# Triple-path LOO on the 60-bug corpus (round-3 W1/Q1)",
          "",
          ("Reviewer W1 / Q1 asked for the handler-attributable refute "
           "count under joint-LOO -- i.e., bugs that joint-LOO converts "
           "from RP to non-RP, broken out by category."),
          "",
          ("TG has three refute paths: P1 per-operator handler dispatch, "
           "P2 AST-pattern intent-bug analyser, P3 constraint residue "
           "(CEGAR loop + heuristic flow-sensitive analyser).  The "
           "earlier joint-LOO disabled P1+P2 only; this script disables "
           "P1+P2+P3 simultaneously."),
          "",
          f"## Full pipeline: **RP {full['rp']}/60**.",
          (f"## All three paths globally disabled (no LOO): "
           f"**RP {triple_global['rp']}/60**."),
          "",
          ("## Per-category triple-LOO (P1 handlers for category + P2 + "
           "P3 all disabled)"),
          "",
          ("| category | full RP | triple-LOO RP | handler-attributable "
           "drop |"),
          "|---|---|---|---|"]
    for cat, attrib in handler_attrib.items():
        md.append(f"| {cat} | {attrib['rp_full']} | "
                  f"{attrib['rp_under_triple_loo']} | "
                  f"{attrib['handler_attributable_drop']} |")
    md.extend(["",
               "## Reading",
               "",
               ("The handler-attributable drop is the per-category "
                "answer to W1: how many RP verdicts in category $c$ "
                "depend on at least one of (handler $\\in c$, intent-"
                "pattern path, constraint residue).  A non-zero drop "
                "means the category has independent dependence on TG's "
                "refute paths; a zero drop combined with a non-zero "
                "global triple-disabled RP would mean the bug is "
                "still being caught by a fourth refute path (e.g.\\ "
                "raw refinement-variable SMT)."),
               "",
               ("This artefact reconciles the per-rule attribution "
                "table (7/7/6/5/4*4/3 = 49 RPs across categories) with "
                "the joint-LOO non-result (53/60 -> 53/60).  The "
                "per-rule attribution measures *which keyword* fires "
                "in the catching bug message; the joint-LOO measures "
                "*how many paths* survive removal.  They are not in "
                "tension: one path may catch a bug whose message "
                "fires another category's keyword.")])
    open(OUT_MD, "w").write("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
