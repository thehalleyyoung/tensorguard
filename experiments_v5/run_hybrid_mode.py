"""
run_hybrid_mode.py
==================

Runs TensorGuard's hybrid mode (TG-static-first → FakeTensor fallback) on
the v5 488-block corpus and reports the effect of the fallback.

Outputs:
  experiments_v5/hybrid_mode_results.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from src.v5.hybrid_mode import hybrid_check  # noqa: E402

BLOCK_JSONL = ROOT / "v5_block_corpus.jsonl"
OUT_JSON = ROOT / "hybrid_mode_results.json"

NOTES = (
    "FakeTensor 'Verified' means the symbolic forward pass completed without "
    "a shape/size RuntimeError. It does NOT prove correctness of the logic — "
    "only that tensor shapes are consistent with the declared input shapes. "
    "This may inflate Verified counts for blocks whose shape bugs are only "
    "triggered by specific runtime values, not by shape alone."
)


def main():
    records = [json.loads(l) for l in BLOCK_JSONL.open()]
    n = len(records)
    print(f"[hybrid] running {n} blocks ...")

    tg_counts: Counter = Counter()
    hybrid_counts: Counter = Counter()
    fallback_breakdown = {
        "abstain_to_verified": 0,
        "abstain_to_refuted": 0,
        "abstain_remained": 0,
        "abstain_unimportable": 0,
    }
    per_item = []

    t0 = time.time()
    for i, rec in enumerate(records):
        result = hybrid_check(
            rec["source"],
            input_shapes=rec["input_shapes"],
            qualified_name=rec.get("qualified_name"),
            filename=rec.get("library_path", "<hybrid>"),
        )

        tg_verdict: str
        # Determine pre-fallback TG verdict
        if result["fallback"] is None:
            # TG gave Verified or Refuted
            tg_verdict = result["verdict"]
        else:
            # TG abstained
            tg_verdict = "Abstain"

        hybrid_verdict = result["verdict"]
        tg_counts[tg_verdict] += 1
        hybrid_counts[hybrid_verdict] += 1

        fallback_used = result["fallback"] is not None
        fallback_error = None
        if fallback_used:
            fb = result["fallback"]
            fallback_error = fb.get("error")
            fb_verdict = fb["verdict"]
            if fb_verdict == "Verified":
                fallback_breakdown["abstain_to_verified"] += 1
            elif fb_verdict == "Refuted":
                fallback_breakdown["abstain_to_refuted"] += 1
            else:
                # Abstain in fallback — distinguish unimportable from other
                err = fallback_error or ""
                if "ctor_failed" in err or "exec_failed" in err or "no_nn_module" in err:
                    fallback_breakdown["abstain_unimportable"] += 1
                    # sub-category counters
                    if "ctor_failed" in err:
                        fallback_breakdown.setdefault("_ctor_failed", 0)
                        fallback_breakdown["_ctor_failed"] += 1
                    elif "exec_failed" in err:
                        fallback_breakdown.setdefault("_exec_failed", 0)
                        fallback_breakdown["_exec_failed"] += 1
                    else:
                        fallback_breakdown.setdefault("_no_module_found", 0)
                        fallback_breakdown["_no_module_found"] += 1
                else:
                    fallback_breakdown["abstain_remained"] += 1

        per_item.append({
            "id": rec["id"],
            "library": rec["library"],
            "tg_verdict": tg_verdict,
            "hybrid_verdict": hybrid_verdict,
            "fallback_used": fallback_used,
            "fallback_error": fallback_error,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n} ({elapsed:.0f}s)")

    elapsed_total = time.time() - t0

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": torch.__version__,
            "n_blocks": n,
            "elapsed_s": round(elapsed_total, 1),
            "notes": NOTES,
        },
        "tg_only": {
            "Verified": tg_counts["Verified"],
            "Refuted": tg_counts["Refuted"],
            "Abstain": tg_counts["Abstain"],
        },
        "hybrid": {
            "Verified": hybrid_counts["Verified"],
            "Refuted": hybrid_counts["Refuted"],
            "Abstain": hybrid_counts["Abstain"],
        },
        "fallback_breakdown": fallback_breakdown,
        "per_item": per_item,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")

    print("\n== TG-only ==")
    print(json.dumps(out["tg_only"], indent=2))
    print("\n== Hybrid (TG + FakeTensor fallback) ==")
    print(json.dumps(out["hybrid"], indent=2))
    print("\n== Fallback breakdown (of TG Abstains) ==")
    print(json.dumps(out["fallback_breakdown"], indent=2))
    print(
        f"\nSummary: {tg_counts['Abstain']} TG abstains → "
        f"{fallback_breakdown['abstain_to_verified']} Verified, "
        f"{fallback_breakdown['abstain_to_refuted']} Refuted, "
        f"{fallback_breakdown['abstain_remained']} Abstain (other), "
        f"{fallback_breakdown['abstain_unimportable']} unimportable"
    )


if __name__ == "__main__":
    main()
