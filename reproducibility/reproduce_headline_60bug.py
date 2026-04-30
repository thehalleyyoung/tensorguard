"""Reproducer for the 60-bug historical corpus headline.

Single command:

    PYTHONPATH=. python3 reproducibility/reproduce_headline_60bug.py

Reports BOTH counts the reviewer asked for:

  * "Refuted-Proof"      : the paper headline.  Defined as
                           bugs flagged at confidence >= 0.99
                           by ``verify_architecture(src)`` with default
                           parameters (high_confidence_only=False but
                           gated post-hoc on max-bug-confidence).
                           Expected on current main: **53 / 60**.

  * "Raw refute count"   : ``verify_architecture(src, input_shapes=...,
                           max_cegar_iterations=3)`` with shapes lifted
                           from the per-bug repro (matches the regime
                           used in ``experiments_v5/feature_ablation.json``
                           and ``experiments_v5/v5_benchmark_results.json``).
                           Expected on current main: **56 / 60**.

The 3-bug gap is exactly the low-confidence heuristic post-pass plus
the input-shape-context refinements that fire on three additional bugs;
neither path is Z3-discharged so neither contributes to the headline RP.

Output: reproducibility/reproduce_headline_60bug.json (and stdout).
"""

from __future__ import annotations

import io
import contextlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture  # noqa: E402

MANIFEST = ROOT / "experiments_v5" / "bug_corpus_manifest.json"
BUG_JSONL = ROOT / "experiments_v5" / "v5_bug_corpus.jsonl"
OUT_JSON = ROOT / "reproducibility" / "reproduce_headline_60bug.json"

HIGH_CONFIDENCE = 0.99


def _silent_run(src: str, **kwargs):
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
        return verify_architecture(src, **kwargs)


def _classify(result, threshold: float) -> tuple[str, float, int]:
    bugs = list(result.bugs)
    if bugs:
        max_conf = max(b.confidence for b in bugs)
        verdict = "REFUTED_PROOF" if max_conf >= threshold else "REFUTED_LOW_CONF"
        return verdict, max_conf, len(bugs)
    if getattr(result, "abstained", False):
        return "ABSTAIN", 0.0, 0
    return "SILENT_MISS", 0.0, 0


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    jsonl_records = {json.loads(l)["id"]: json.loads(l)
                     for l in BUG_JSONL.read_text().splitlines() if l.strip()}

    headline = {"refuted_proof_high_confidence": 0,
                "refuted_low_confidence": 0,
                "silent_miss": 0, "abstain": 0, "error": 0}
    raw = {"refute_raw": 0, "refute_high_confidence": 0,
           "silent_miss": 0, "abstain": 0, "error": 0}
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "headline_rp": 0, "raw_refute": 0, "silent": 0}
    )
    per_id: list[dict] = []

    t0 = time.perf_counter()
    for item in items:
        cat = item.get("category", "other")
        by_cat[cat]["n"] += 1
        repro = ROOT / item["repro_file"]

        try:
            src = repro.read_text()
        except Exception as exc:  # pragma: no cover
            headline["error"] += 1
            raw["error"] += 1
            per_id.append({"id": item["id"], "category": cat,
                           "headline_verdict": "ERROR", "error": str(exc)[:200]})
            continue

        try:
            r_default = _silent_run(src)
        except Exception as exc:
            headline["error"] += 1
            r_default = None

        try:
            jsonl = jsonl_records.get(item["id"])
            shapes = {}
            if jsonl is not None:
                jsrc = (ROOT / jsonl["repro_file"]).read_text()
                m = re.search(r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", jsrc, flags=re.MULTILINE)
                if m:
                    try:
                        shapes = eval(m.group(1))
                    except Exception:
                        shapes = {}
            r_raw = _silent_run((jsrc if jsonl else src),
                                input_shapes=shapes,
                                max_cegar_iterations=3)
        except Exception:
            r_raw = None

        if r_default is None:
            head_verdict, head_conf, head_nb = "ERROR", 0.0, 0
        else:
            head_verdict, head_conf, head_nb = _classify(r_default, HIGH_CONFIDENCE)
            if head_verdict == "REFUTED_PROOF":
                headline["refuted_proof_high_confidence"] += 1
                by_cat[cat]["headline_rp"] += 1
            elif head_verdict == "REFUTED_LOW_CONF":
                headline["refuted_low_confidence"] += 1
            elif head_verdict == "SILENT_MISS":
                headline["silent_miss"] += 1
                by_cat[cat]["silent"] += 1
            elif head_verdict == "ABSTAIN":
                headline["abstain"] += 1

        if r_raw is None:
            raw_verdict, raw_conf, raw_nb = "ERROR", 0.0, 0
        else:
            raw_verdict, raw_conf, raw_nb = _classify(r_raw, HIGH_CONFIDENCE)
            if raw_verdict in ("REFUTED_PROOF", "REFUTED_LOW_CONF"):
                raw["refute_raw"] += 1
                by_cat[cat]["raw_refute"] += 1
                if raw_verdict == "REFUTED_PROOF":
                    raw["refute_high_confidence"] += 1
            elif raw_verdict == "SILENT_MISS":
                raw["silent_miss"] += 1
            elif raw_verdict == "ABSTAIN":
                raw["abstain"] += 1

        per_id.append({"id": item["id"], "category": cat,
                       "headline_verdict": head_verdict,
                       "headline_max_conf": round(head_conf, 4),
                       "headline_bugs": head_nb,
                       "raw_verdict": raw_verdict,
                       "raw_max_conf": round(raw_conf, 4),
                       "raw_bugs": raw_nb})

    elapsed = round(time.perf_counter() - t0, 2)

    summary = {
        "meta": {
            "command": "PYTHONPATH=. python3 reproducibility/reproduce_headline_60bug.py",
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "input_shapes_corpus": str(BUG_JSONL.relative_to(ROOT)),
            "high_confidence_threshold": HIGH_CONFIDENCE,
            "elapsed_s": elapsed,
            "n": len(items),
        },
        "headline_regime": {
            "description": "verify_architecture(src) with defaults; RP = max-bug-confidence >= 0.99",
            "refuted_proof_high_confidence": headline["refuted_proof_high_confidence"],
            "refuted_low_confidence": headline["refuted_low_confidence"],
            "silent_miss": headline["silent_miss"],
            "abstain": headline["abstain"],
            "error": headline["error"],
        },
        "raw_refute_regime": {
            "description": ("verify_architecture(src, input_shapes=lifted, "
                            "max_cegar_iterations=3) -- mirrors the regime "
                            "used by experiments_v5/feature_ablation.json and "
                            "experiments_v5/v5_benchmark_results.json"),
            "raw_refute_count": raw["refute_raw"],
            "raw_refute_high_confidence_subcount": raw["refute_high_confidence"],
            "silent_miss": raw["silent_miss"],
            "abstain": raw["abstain"],
            "error": raw["error"],
        },
        "by_category": dict(by_cat),
        "per_item": per_id,
        "paper_claims_resolved": {
            "53/60_refuted_proof_paper_headline":
                headline["refuted_proof_high_confidence"],
            "56/60_raw_refuted_in_feature_ablation":
                raw["refute_raw"],
            "interpretation": (
                "The paper's headline 'Refuted-Proof on N/60' is the "
                "high-confidence sub-count of the headline regime. "
                "The 'refuted' field in feature_ablation.json counts "
                "any flagged bug under the input-shape-lifted regime "
                "(which the per-feature ablation runs with "
                "max_cegar_iterations=3 and the per-bug INPUT_SHAPES). "
                "Both numbers are emitted by this single command, "
                "removing the apparent 53/56 inconsistency: they are "
                "two different verdicts on two different (clearly "
                "labelled) regimes."
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=False))

    print("=" * 64)
    print(f"60-bug historical corpus  (N={len(items)}, elapsed {elapsed}s)")
    print("=" * 64)
    print("[paper-headline regime: verify_architecture(src) defaults]")
    print(f"  Refuted-Proof  (max-conf >= {HIGH_CONFIDENCE})  : "
          f"{headline['refuted_proof_high_confidence']}/{len(items)}   <-- paper headline")
    print(f"  Refuted (low-conf only)               : "
          f"{headline['refuted_low_confidence']}/{len(items)}")
    print(f"  Silent miss                           : "
          f"{headline['silent_miss']}/{len(items)}")
    print()
    print("[ablation regime: input_shapes lifted, max_cegar_iterations=3]")
    print(f"  Raw refute count                      : "
          f"{raw['refute_raw']}/{len(items)}   <-- feature_ablation.json 'refuted' value")
    print(f"  Of which Z3-RP (max-conf >= {HIGH_CONFIDENCE}) : "
          f"{raw['refute_high_confidence']}/{len(items)}")
    print(f"  Silent miss                           : "
          f"{raw['silent_miss']}/{len(items)}")
    print()
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
