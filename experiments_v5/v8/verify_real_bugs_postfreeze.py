"""Verify the post-freeze upstream-faithful corpus against TG.

This harness directly addresses reviewer Q2 (round 2): bug PRs filed
*strictly after* the catalogue freeze date are scored end-to-end with
no rule edits.

Freeze invariant
----------------
* Catalogue freeze date: ``2026-04-07`` (commit 040f6f3,
  ``Fix documentation code examples to match implementation``;
  the most-recent git-tracked commit touching the v5 rule-set
  before this round began).
* Catalogue content hash: SHA-256 over the sorted-tarball of
  ``src/v5/`` at freeze; recorded in
  ``reproducibility/postfreeze_catalogue_hash.txt``.
* Each post-freeze repro mirrors a real upstream bug-fix PR
  whose ``created_at`` is strictly after ``2026-04-07T00:00:00Z``.

The headline number is the *user-visible* (assume_M-empty) verdict
triple, NOT the assume_M-synthesised triple --- so an "RP" here is
unconditional.

Run
---
    PYTHONPATH=. python3 experiments_v5/v8/verify_real_bugs_postfreeze.py
"""
import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "real_bugs_postfreeze")
OUT = os.path.join(ROOT, "reproducibility", "real_bugs_postfreeze.json")

# Per-bug provenance (PR url, created_at, root cause class, expected verdict family).
PROVENANCE = {
    "rb_pf_001": {
        "pr": "https://github.com/huggingface/diffusers/pull/13494",
        "created_at": "2026-04-16",
        "class": "config_dependent_linear_chain",
    },
    "rb_pf_002": {
        "pr": "https://github.com/huggingface/transformers/pull/45540",
        "created_at": "2026-04-21",
        "class": "attention_mask_dim_mismatch",
    },
    "rb_pf_003": {
        "pr": "https://github.com/huggingface/peft/pull/3165",
        "created_at": "2026-04-15",
        "class": "lora_in_out_swap_3d",
    },
    "rb_pf_004": {
        "pr": "https://github.com/huggingface/transformers/pull/45473",
        "created_at": "2026-04-16",
        "class": "router_topk_vs_num_experts",
    },
    "rb_pf_005": {
        "pr": "https://github.com/huggingface/diffusers/pull/13490",
        "created_at": "2026-04-16",
        "class": "attention_mask_expand_off_by_one",
    },
    "rb_pf_006": {
        "pr": "https://github.com/huggingface/diffusers/pull/13441",
        "created_at": "2026-04-10",
        "class": "batch_ordering_chunk_mismatch",
    },
}


def main() -> int:
    records = []
    counts = {"RP_0.99": 0, "RP_0.80": 0, "silent_verified": 0, "load_err": 0}
    for fname in sorted(os.listdir(BASE)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(BASE, fname)
        bid = "_".join(fname.split("_")[:3])  # rb_pf_001
        with open(fpath) as f:
            src = f.read()
        ns = {"__name__": "__rb_pf__"}
        try:
            exec(compile(src, fpath, "exec"), ns)
        except Exception as e:
            print(f"LOAD_ERR {fname}: {e}")
            records.append({"id": bid, "status": "LOAD_ERR", "err": str(e)})
            counts["load_err"] += 1
            continue
        input_shapes = ns.get("INPUT_SHAPES")
        if input_shapes is None:
            continue
        try:
            result = verify_architecture(src, input_shapes=input_shapes)
            bugs = list(result.bugs)
            max_conf = max((b.confidence for b in bugs), default=0.0)
            verdicts = sorted({getattr(b, "verdict", "") for b in bugs} or {"none"})
        except Exception as e:
            traceback.print_exc()
            records.append({"id": bid, "status": "VERIFY_ERR", "err": str(e)})
            continue
        if max_conf >= 0.99:
            status = "RP_0.99"
        elif max_conf >= 0.80:
            status = "RP_0.80"
        else:
            status = "silent_verified"
        counts[status] += 1
        rec = {
            "id": bid,
            "file": os.path.relpath(fpath, ROOT),
            "input_shapes": {k: list(v) for k, v in input_shapes.items()},
            "n_bugs": len(bugs),
            "max_confidence": max_conf,
            "verdicts": verdicts,
            "status": status,
            **PROVENANCE.get(bid, {}),
        }
        records.append(rec)
        print(f"{status:18s} conf={max_conf:.2f}  {fname}")

    summary = {
        "regime": "no_synthesised_assume_M (user-visible)",
        "freeze_date": "2026-04-07",
        "freeze_commit": "040f6f3",
        "n_total": len(records),
        "headline_triple": {
            "Verified (silent)": counts["silent_verified"],
            "Refuted_Proof_at_0.99": counts["RP_0.99"],
            "Refuted_Proof_at_0.80": counts["RP_0.80"],
            "load_err": counts["load_err"],
        },
        "interpretation": (
            "Each row is a real upstream bug-fix PR filed strictly after "
            "the catalogue-freeze date 2026-04-07. TG is run end-to-end "
            "with no rule edits and no analyser changes between freeze "
            "and this run.  The headline triple is the user-visible "
            "(assume_M-empty) verdict; an RP at this regime is "
            "unconditional."
        ),
        "per_item": records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(summary["headline_triple"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
