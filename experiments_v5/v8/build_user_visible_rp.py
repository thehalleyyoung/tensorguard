"""User-visible unconditional-RP report (round-1 reviewer Q2).

The 488-block headline number includes Contract-Violation refutations
that are *sound under the synthesised caller-rely assume*.  A
practitioner running ``tensorguard`` on their own codebase, however,
would not see CVs as RPs: they would see them as conditional warnings
predicated on the contract envelope TG synthesised on their behalf.

This script computes the verdict triple a practitioner would actually
    seen if the synthesised assume_M were dropped (i.e. every
    "self.config.X" treated as a free symbolic variable with no
    upper-bound assumption).  Concretely we read the precomputed verdict
    classification at "experiments_v5/verdict_reclassification.json" and
    re-bucket every CV into LW (since without the assume the refutation
    no longer has the soundness theorem to lean on) and every previously
    "opaque-config" abstain into a still-abstain (no change).

Output: ``experiments_v5/v8/user_visible_rp.json`` with the triple
(V, RP, CV, LW, Abstain) under the no-assume regime.
"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)

VERDICT_PATH = os.path.join(_EXP, "verdict_reclassification.json")
RESULTS_PATH = os.path.join(_EXP, "v5_benchmark_results.json")
OUT_PATH = os.path.join(_HERE, "user_visible_rp.json")


def main() -> None:
    with open(VERDICT_PATH) as fh:
        verdict = json.load(fh)
    with open(RESULTS_PATH) as fh:
        results = json.load(fh)

    summary = results["block_corpus"]["summary"]
    n_total = summary["total"]
    v = summary["buckets"].get("Verified", 0)
    a = summary["buckets"].get("Abstain", 0)
    rp = verdict["block_corpus"]["REFUTED_PROOF"]
    cv = verdict["block_corpus"]["CONTRACT_VIOLATION"]
    lw = verdict["block_corpus"]["LIBRARY_WARN"]

    # Under the no-synthesised-assume regime, every CV becomes LW
    # (still a refutation, but no longer covered by the soundness
    # theorem because the caller-rely is no longer assumed); RP stays
    # RP; LW stays LW; Verified stays Verified iff the verifier did
    # not depend on the synthesised assume to discharge constraints.
    user_visible_rp = rp                        # 0
    user_visible_lw = lw + cv                   # 78 + 128 = 206

    # Verified-under-no-assume: any block whose Verified verdict
    # depended on the synthesised assume must be downgraded to
    # Abstain.  Heuristic: a block is "assume-dependent Verified"
    # iff it has any opaque_config_attr in its abstain_tags or its
    # source contains `self.config.` *and* is in the Verified
    # bucket.  We apply the conservative version: count *all*
    # transformer-Verified blocks as assume-dependent (a
    # transformer block with a Verified verdict almost always rests
    # on a config-driven hidden_size assume).
    transformer_v = summary["by_category"].get("transformer", {}).get("Verified", 0)
    user_visible_v = v - transformer_v
    user_visible_a = a + transformer_v

    out = {
        "regime": "no_synthesised_assume_M",
        "n_total": n_total,
        "user_visible_triple": {
            "Verified": user_visible_v,
            "Refuted_Proof": user_visible_rp,
            "Contract_Violation": 0,  # by construction; CVs collapse into LW
            "Library_Warn": user_visible_lw,
            "Abstain": user_visible_a,
        },
        "headline_triple_with_assume": {
            "Verified": v,
            "Refuted_Proof": rp,
            "Contract_Violation": cv,
            "Library_Warn": lw,
            "Abstain": a,
        },
        "delta": {
            "Verified": user_visible_v - v,
            "Refuted_Proof": 0,
            "Contract_Violation": -cv,
            "Library_Warn": user_visible_lw - lw,
            "Abstain": user_visible_a - a,
        },
        "interpretation": (
            "Under the no-synthesised-assume regime, the "
            "user-visible unconditional-RP count on the 488-block "
            "real-source corpus is 0.  The 128 CV refutations in "
            "the headline triple collapse into LW (conservative "
            "warnings); the 78 LW are unchanged; the "
            "transformer-Verified blocks that lean on a synthesised "
            "config-envelope assume become Abstain.  This is the "
            "number a practitioner would see when running TG on "
            "their own codebase without TG synthesising contracts "
            "for them.  See review_response.md (W1 / Q2)."
        ),
    }

    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
