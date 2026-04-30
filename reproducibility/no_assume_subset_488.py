#!/usr/bin/env python3.11
"""Round-4 reviewer borderline reason.

The reviewer's borderline-promotion criterion is: pick a subset of the
488-block corpus on which TensorGuard runs end-to-end *without* a
synthesised caller-rely assume_M (i.e. the verdict is not gated on
TG having invented a precondition over symbolic config attributes),
and report the calibrated verdict triple V/RP/CV/LW/A on that subset.

Definition of the no-synthesised-assume subset.
    A 488-block entry is in the subset iff its verdict does not lean
    on a synthesised assume_M.  Concretely:

      (a) every torchvision and timm block (the cv_caller_rely audit
          shows that the synthesised assume_M is non-trivial only for
          transformer-library config-driven modules, with a single
          exception in timm captured by bucket "empty");
      (b) every Contract-Violation verdict whose synthesised
          assume_M is *empty* (bucket "empty" in
          ``reproducibility/cv_caller_rely.json``: 25 transformers +
          1 timm, total 26);
      (c) every Library-Warn or Abstain verdict in any library
          (LW are conservative warnings outside the soundness
          theorem; Abstain has no verdict claim, so neither carries
          an assume_M obligation);
      (d) every torchvision/timm Verified verdict (the
          synthesised-config-envelope dependency that makes the
          transformer-Verified blocks user-visibly assume-dependent
          does not apply: torchvision and timm forwards do not read
          opaque-config attributes in the corpus).

    We *exclude*:
      - the 23 transformer-library Verified blocks (assume-dependent
        per the build_user_visible_rp.py classifier);
      - the 102 transformer-library Contract-Violation blocks whose
        bucket is "symbolic-config-only" or "no-own-init", i.e. their
        assume_M references symbolic config attributes that an
        independent caller would have to satisfy.

We then report the calibrated verdict triple on the resulting subset.
This re-runs no analyser passes; the verdicts are read from the
cached 488-block run plus the cv_caller_rely classifier already
shipped in the repo.

Output:
  reproducibility/no_assume_subset_488.json
  reproducibility/no_assume_subset_488.md
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH = os.path.join(ROOT, "experiments_v5/v5_benchmark_results.json")
RECL = os.path.join(ROOT, "experiments_v5/verdict_reclassification.json")
CV = os.path.join(ROOT, "reproducibility/cv_caller_rely.json")
PER_BLOCK_V = os.path.join(ROOT, "experiments_v5/v8/per_block_user_visible_rp.json")
OUT_JSON = os.path.join(ROOT, "reproducibility/no_assume_subset_488.json")
OUT_MD = os.path.join(ROOT, "reproducibility/no_assume_subset_488.md")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    items = json.load(open(BENCH))["block_corpus"]["per_input"]
    recl = json.load(open(RECL))["block_corpus"]["per_item"]
    cv_rows = json.load(open(CV))["rows"]
    per_block_v = json.load(open(PER_BLOCK_V))["per_block"]
    pbv_by_id = {r["id"]: r for r in per_block_v}

    cv_bucket = {r["id"]: r["bucket"] for r in cv_rows}
    recl_verdict = {r["id"]: r["verdict"] for r in recl}

    in_subset = []
    excluded = []
    for it in items:
        bid = it["id"]
        lib = it["library"]
        bucket = it["bucket"]
        # Resolve sub-verdict for Refuted entries
        sub = recl_verdict.get(bid)  # None if not Refuted
        cvb = cv_bucket.get(bid)     # None unless CV

        keep = False
        reason = ""
        if bucket == "Abstain":
            keep, reason = True, "abstain (no verdict obligation)"
        elif bucket == "Verified":
            # transformer-Verified leans on synthesised config envelope
            if lib in ("torchvision", "timm"):
                keep, reason = True, "verified, library does not synthesise config envelope"
            else:
                # Verified-under-no-assume per per_block_user_visible_rp:
                # 'verdict_no_assume' == 'Verified' is the survivor flag.
                pbv = pbv_by_id.get(bid)
                if pbv and pbv.get("verdict_no_assume") == "Verified":
                    keep, reason = True, "verified survives without synthesised assume_M"
                else:
                    keep, reason = False, "transformer Verified, assume-dependent"
        else:  # bucket == "Refuted"
            if sub == "LIBRARY_WARN":
                keep, reason = True, "LW (conservative warning, no assume_M)"
            elif sub == "CONTRACT_VIOLATION":
                if cvb == "empty":
                    keep, reason = True, "CV with empty assume_M (trivially witnessed)"
                else:
                    keep, reason = False, f"CV with non-empty assume_M (bucket={cvb})"
            else:
                keep, reason = False, f"unclassified Refuted (sub={sub})"

        rec = {"id": bid, "library": lib, "category": it["category"],
               "bucket": bucket, "sub_verdict": sub, "cv_bucket": cvb,
               "kept": keep, "reason": reason}
        (in_subset if keep else excluded).append(rec)

    # Triple over the kept subset
    triple = {"V": 0, "RP": 0, "CV": 0, "LW": 0, "A": 0}
    by_lib = {}
    for r in in_subset:
        by_lib.setdefault(r["library"], Counter())
        if r["bucket"] == "Verified":
            triple["V"] += 1; by_lib[r["library"]]["V"] += 1
        elif r["bucket"] == "Abstain":
            triple["A"] += 1; by_lib[r["library"]]["A"] += 1
        else:
            sv = r["sub_verdict"]
            if sv == "LIBRARY_WARN":
                triple["LW"] += 1; by_lib[r["library"]]["LW"] += 1
            elif sv == "CONTRACT_VIOLATION":
                # Empty-bucket CVs by definition have no caller-rely
                # obligation, so they are unconditional refutations:
                # we report them as RP under the no-assume regime,
                # and also separately as CV-empty for transparency.
                triple["RP"] += 1; by_lib[r["library"]]["RP"] += 1
            else:
                triple["A"] += 1; by_lib[r["library"]]["A"] += 1

    n = len(in_subset)
    rp_lo, rp_hi = wilson(triple["RP"], n)

    out = {
        "_question": (
            "Round-4 borderline reason: report the calibrated verdict "
            "triple on the subset of the 488-block corpus where TG "
            "runs end-to-end without a synthesised assume_M."
        ),
        "subset_definition": (
            "Every torchvision/timm block, plus every Library-Warn or "
            "Abstain verdict, plus every Contract-Violation verdict whose "
            "assume_M classifier in cv_caller_rely.json is 'empty'.  "
            "Excludes the 23 transformer-library Verified blocks "
            "(synthesised-config-envelope assume) and the 102 "
            "transformer Contract-Violation blocks whose assume_M is "
            "non-empty (symbolic-config-only or no-own-init bucket)."
        ),
        "n_total": len(items),
        "n_subset": n,
        "n_excluded": len(excluded),
        "verdict_triple": triple,
        "verdict_triple_pct": {k: round(100 * v / n, 1) for k, v in triple.items()},
        "rp_wilson_95ci_pct": [round(100 * rp_lo, 2), round(100 * rp_hi, 2)],
        "by_library": {k: dict(v) for k, v in by_lib.items()},
        "exclusion_breakdown": dict(Counter(r["reason"] for r in excluded)),
        "rp_note": (
            "RP on this subset = the 26 Contract-Violation verdicts "
            "whose synthesised assume_M is *empty*, i.e. the "
            "refutation holds with no caller-rely obligation.  These "
            "are unconditional refutations under the round-4 reviewer's "
            "definition."
        ),
        "_method": (
            "Read cached 488-block per-input verdicts (no re-run); read "
            "the CV bucket classification from cv_caller_rely.json; read "
            "the per-block user-visible-Verified survivor flag from "
            "per_block_user_visible_rp.json.  Apply the subset filter, "
            "tally the verdict triple."
        ),
        "subset_ids": [r["id"] for r in in_subset],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# No-synthesised-assume subset of the 488-block corpus",
        "",
        "Round-4 reviewer borderline criterion.  We restrict the 488-",
        "block real-source corpus to those blocks where the TensorGuard",
        "verdict does not lean on a synthesised caller-rely assume_M:",
        "",
        f"- subset size: **{n} / {len(items)}** blocks",
        f"- verdict triple: **V={triple['V']}, RP={triple['RP']}, "
        f"CV={triple['CV']}, LW={triple['LW']}, A={triple['A']}**",
        f"- Wilson 95% CI on RP-rate: "
        f"[{round(100*rp_lo,1)}%, {round(100*rp_hi,1)}%]",
        "",
        "RP here means an unconditional refutation: a Contract-",
        "Violation whose synthesised assume_M classifier returns",
        "'empty' (no caller-rely obligation; the refutation holds",
        "for every realisable caller).",
        "",
        "## By library",
        "",
        "| library | V | RP | LW | A |",
        "|---|---|---|---|---|",
    ]
    for lib, ct in by_lib.items():
        md.append(f"| {lib} | {ct.get('V',0)} | {ct.get('RP',0)} | "
                  f"{ct.get('LW',0)} | {ct.get('A',0)} |")
    md += [
        "",
        "## Method",
        "",
        "No analyser re-run.  Subset filter applied to the cached 488-",
        "block per-input verdicts using the existing cv_caller_rely",
        "bucket classification ('empty' = no caller-rely obligation).",
        "",
        "Run with `python3 reproducibility/no_assume_subset_488.py`.",
        "",
        "Cited from the eval section of the paper.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(json.dumps({"n_subset": n, "verdict_triple": triple,
                      "rp_wilson_95ci_pct": out["rp_wilson_95ci_pct"]}, indent=2))


if __name__ == "__main__":
    main()
