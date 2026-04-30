"""Reconcile the two 488-block headline counts that have appeared in
released artifacts.

Background
----------
Two distinct verdict-count tuples for the 488-block corpus have appeared
across released artifacts:

  * 57 Verified / 206 Refuted / 225 Abstain
    -- referenced by the paper text and by
       ``experiments_v5/feature_ablation.json`` (every ladder rung)
       and ``experiments_v5/hybrid_mode_results.json`` (the ``tg_only``
       row).  These all run the verifier with
       ``high_confidence_only=True`` (Z3-proven bugs only).

  * 50 Verified / 213 Refuted / 225 Abstain
    -- recorded by ``experiments_v5/v5_benchmark_results.json``,
       which calls the verifier with the public default
       ``high_confidence_only=False`` (Z3-proven bugs plus a
       lower-confidence heuristic post-pass).

The 7-row gap (Verified -> Refuted) is therefore not a bookkeeping
inconsistency: it is the heuristic post-pass.  The 225-Abstain count is
identical across both regimes, as expected (the abstain decision is
independent of the heuristic post-pass).

This script regenerates BOTH regimes against the *current* code base
and writes per-id verdicts so a reader can verify the diff is a strict
``Verified -> Refuted`` set under HCO=False.
"""
import contextlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)

BLOCK_JSONL = ROOT / "experiments_v5" / "v5_block_corpus.jsonl"
OUT_JSON = Path(__file__).with_suffix(".json")


def _bucket(rec, hco):
    src = PREAMBLE + rec["source"]
    cap = io.StringIO()
    try:
        with contextlib.redirect_stderr(cap), contextlib.redirect_stdout(cap):
            res = verify_architecture(
                src,
                input_shapes=rec["input_shapes"],
                max_cegar_iterations=3,
                high_confidence_only=hco,
                filename="<recon>",
            )
    except Exception:
        return "Abstain"
    if bool(res.abstained):
        return "Abstain"
    if int(res.bug_count) > 0:
        return "Refuted"
    return "Verified"


def main():
    items = []
    with open(BLOCK_JSONL) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(json.loads(ln))

    t0 = time.time()
    rows = []
    for i, rec in enumerate(items):
        v_true = _bucket(rec, True)
        v_false = _bucket(rec, False)
        rows.append({"id": rec.get("qualified_name", f"item_{i}"),
                     "hco_true": v_true, "hco_false": v_false})
    elapsed = time.time() - t0

    summary = {
        "n": len(rows),
        "hco_true": dict(Counter(r["hco_true"] for r in rows)),
        "hco_false": dict(Counter(r["hco_false"] for r in rows)),
        "transitions_true_to_false": dict(Counter(
            f"{r['hco_true']}->{r['hco_false']}"
            for r in rows if r["hco_true"] != r["hco_false"]
        )),
        "elapsed_s": round(elapsed, 1),
    }

    out = {
        "meta": {
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "purpose": "reconcile 50/213/225 vs 57/206/225 headline counts",
            "regime_hco_true": "high_confidence_only=True (Z3-proven bugs only); "
                               "matches paper headline and feature_ablation.json",
            "regime_hco_false": "high_confidence_only=False (default); adds "
                                "lower-confidence heuristic post-pass; matches "
                                "experiments_v5/v5_benchmark_results.json",
        },
        "summary": summary,
        "per_id": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
