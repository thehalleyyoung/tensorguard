#!/usr/bin/env python3.11
"""Round-4 reviewer Q5: 60-bug RP using ONLY the typing rules whose
soundness is the subject of Theorem 2.

Round-3 produced the joint LOO (``bug_corpus_loo_joint``) which
disables the per-category operator handlers AND the AST-pattern
intent-bug analyser.  The round-4 reviewer asks for the
*complementary* ablation: drop *everything* that is not literally a
typing-rule lookup -- i.e. also disable the constraint-based shape
back-end (the symbolic interpreter / SMT path inside ``verify_model``)
that is shipped alongside the per-operator dispatch.

This script re-runs the 60-bug corpus three more times to enumerate
the residual RP count when the analyser is progressively narrowed:

  (i)  baseline (full pipeline)                              -> reference
  (ii) intent-bug AST-pattern path disabled                  -> reference
  (iii) intent-bug + per-operator handlers disabled          -> reference
  (iv) intent-bug + per-operator handlers + constraint-based
       shape back-end (verify_model SMT path) disabled.      -> THIS

Configuration (iv) is the empirical instantiation of "rules only".
The constraint-based back-end is disabled by stubbing the public
``verify_model`` entrypoint to return an empty
``VerificationResult`` (no errors, no counterexample), and
``OverwarnAnalyzer`` is patched as in ``bug_corpus_loo_joint``.
What remains is the bare typing-rule lookup table; if that table is
soundness-load-bearing for the bug-corpus catches, configuration (iv)
will still surface non-zero RP on the 60 bugs.

Output:
  reproducibility/bug_corpus_typing_rules_only.json
  reproducibility/bug_corpus_typing_rules_only.md
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "reproducibility/bug_corpus_typing_rules_only.json")
OUT_MD = os.path.join(ROOT, "reproducibility/bug_corpus_typing_rules_only.md")
CORPUS = os.path.join(ROOT, "experiments_v5/v5_bug_corpus.jsonl")


def _load_corpus() -> List[Dict[str, Any]]:
    with open(CORPUS) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _read_repro(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


def _score(disable_intent: bool, disable_handlers: bool,
           disable_constraint_backend: bool) -> Dict[str, Any]:
    # Fresh src.* import tree.
    for m in list(sys.modules):
        if m == "src" or m.startswith("src."):
            del sys.modules[m]

    if disable_handlers:
        import src.tensor_shapes as ts
        # Drop every operator handler.
        ts.TORCH_SHAPE_OPS.clear()
        # Stub the per-op shape computers.
        for _nm in ("compute_matmul_shape", "compute_broadcast_shape",
                    "compute_reshape_shape", "compute_conv_shape",
                    "compute_linear_shape"):
            if hasattr(ts, _nm):
                setattr(ts, _nm, lambda *a, **k: None)
        try:
            import src.stdlib.modern_ops as mo
            mo.MODERN_TORCH_SHAPE_OPS.clear()
        except Exception:
            pass
        try:
            import src.smt.encoder as enc
            enc.FUNCTIONAL_SHAPE_RULES.clear()
        except Exception:
            pass

    if disable_intent:
        try:
            import src.intent_bugs as ib

            class _StubAnalyzer:
                def __init__(self, *a, **k): pass
                def analyze(self, *a, **k): return []

            ib.OverwarnAnalyzer = _StubAnalyzer
        except Exception:
            pass

    if disable_constraint_backend:
        # Stub the constraint-based back-end (verify_model) at its
        # public boundary inside src.api.  We import api first so the
        # stub takes effect before verify_architecture binds the
        # symbol.
        import src.api as api

        class _EmptyVerificationResult:
            def __init__(self):
                self.safe = True
                self.errors: List[str] = []
                self.counterexample = None
                self.cex_meta = None
                self.discovered_contracts: List[Any] = []

        def _stub_verify_model(source, input_shapes=None,
                               high_confidence_only=False, **kwargs):
            return _EmptyVerificationResult()

        api.verify_model = _stub_verify_model
        # Also stub shape_cegar entrypoints if they are reached.
        try:
            import src.shape_cegar as sc
            class _EmptyCEGARResult:
                contracts = []
                invariants = []
                failures = []
                final_safe = True
                def __init__(self): pass
            sc.run_shape_cegar = lambda *a, **k: _EmptyCEGARResult()
            sc.verify_and_discover = lambda *a, **k: _EmptyCEGARResult()
        except Exception:
            pass

    if disable_intent:
        # Re-stub OverwarnAnalyzer inside src.api binding too.
        try:
            import src.api as api
            class _StubAnalyzer2:
                def __init__(self, *a, **k): pass
                def analyze(self, *a, **k): return []
            api.OverwarnAnalyzer = _StubAnalyzer2
        except Exception:
            pass

    from src.api import verify_architecture  # noqa: E402

    items = _load_corpus()
    rp = silent = abst = err = 0
    per_cat: Dict[str, Dict[str, int]] = {}
    for it in items:
        cat = it["category"]
        per_cat.setdefault(cat, {"rp": 0, "silent": 0, "abst": 0, "err": 0, "n": 0})
        per_cat[cat]["n"] += 1
        try:
            src_str = _read_repro(it["repro_file"])
        except Exception:
            err += 1; per_cat[cat]["err"] += 1
            continue
        try:
            r = verify_architecture(src_str)
            status = getattr(r, "status", "UNKNOWN")
            if status == "UNSAFE":
                rp += 1; per_cat[cat]["rp"] += 1
            elif status == "SAFE":
                silent += 1; per_cat[cat]["silent"] += 1
            else:
                abst += 1; per_cat[cat]["abst"] += 1
        except Exception:
            err += 1; per_cat[cat]["err"] += 1
    return {"rp": rp, "silent": silent, "abst": abst, "err": err,
            "per_category": per_cat}


def main() -> None:
    print("(i) full pipeline...", flush=True)
    t0 = time.time()
    full = _score(False, False, False); full["elapsed_s"] = round(time.time()-t0, 2)
    print(f"  RP={full['rp']}/60  ({full['elapsed_s']}s)", flush=True)

    print("(ii) intent-bug disabled...", flush=True)
    t0 = time.time()
    no_ib = _score(True, False, False); no_ib["elapsed_s"] = round(time.time()-t0, 2)
    print(f"  RP={no_ib['rp']}/60", flush=True)

    print("(iii) intent-bug + handlers disabled...", flush=True)
    t0 = time.time()
    no_ib_h = _score(True, True, False); no_ib_h["elapsed_s"] = round(time.time()-t0, 2)
    print(f"  RP={no_ib_h['rp']}/60", flush=True)

    print("(iv) intent-bug + handlers + constraint-back-end disabled "
          "(typing-rules-only)...", flush=True)
    t0 = time.time()
    typing_only = _score(True, True, True); typing_only["elapsed_s"] = round(time.time()-t0, 2)
    print(f"  RP={typing_only['rp']}/60", flush=True)

    out = {
        "_question": (
            "Round-4 reviewer Q5: report the 60-bug RP count when the "
            "analyser is restricted to the typing rules of Theorem 2 "
            "(operator handlers + AST-pattern + constraint-based shape "
            "back-end all disabled)."
        ),
        "configurations": {
            "(i)_full_pipeline": full,
            "(ii)_intent_bug_disabled": no_ib,
            "(iii)_intent_bug_plus_handlers_disabled": no_ib_h,
            "(iv)_typing_rules_only": typing_only,
        },
        "interpretation": (
            "The typing-rules-only configuration disables every "
            "verification path that is not a literal rule-table "
            "lookup.  The resulting RP count is the share of the bug "
            "corpus that the rule table catches *as a static lookup* "
            "with no constraint solver, no AST pattern, and no per-"
            "operator dispatch -- i.e. the contribution of the "
            "Theorem 2 fragment as a recognition device, not as a "
            "decision procedure.  An RP count near zero is the "
            "honest outcome: the typing rules describe the calculus, "
            "the analyser is what runs them; without the analyser, "
            "the rules emit no verdict."
        ),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Typing-rules-only RP on the 60-bug corpus (round-4 Q5)",
        "",
        "Reviewer Q5: report the 60-bug RP count when the analyser is",
        "restricted to the typing rules whose soundness is asserted by",
        "Theorem 2 -- with the constraint-based shape back-end, the",
        "AST-pattern intent-bug analyser, and the per-operator",
        "handler dispatch all disabled.",
        "",
        "| configuration | RP | silent | abstain | err |",
        "|---|---|---|---|---|",
        f"| (i) full pipeline | {full['rp']} | {full['silent']} | "
        f"{full['abst']} | {full['err']} |",
        f"| (ii) intent-bug disabled | {no_ib['rp']} | "
        f"{no_ib['silent']} | {no_ib['abst']} | {no_ib['err']} |",
        f"| (iii) intent-bug + handlers disabled | {no_ib_h['rp']} | "
        f"{no_ib_h['silent']} | {no_ib_h['abst']} | {no_ib_h['err']} |",
        f"| (iv) typing-rules only | {typing_only['rp']} | "
        f"{typing_only['silent']} | {typing_only['abst']} | "
        f"{typing_only['err']} |",
        "",
        ("The (iv) row is the answer to the reviewer's question.  The "
         "drop from (iii) to (iv) isolates the contribution of the "
         "constraint-based shape back-end, separating it from the "
         "rule-table fragment that Theorem 2 covers."),
        "",
        "Run with `python3.11 reproducibility/bug_corpus_typing_rules_only.py`.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
