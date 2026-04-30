#!/usr/bin/env python3
"""Run the 488-block analysis with the new unbind handler.
Compares to the prior run to identify verdict changes.
"""
from __future__ import annotations
import json, sys, time, io, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import verify_architecture

PREAMBLE = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch import Tensor\n"
CORPUS = ROOT / "experiments_v5" / "v5_block_corpus.jsonl"
OUT_JSON = ROOT / "reproducibility" / "unbind_handler_488_run.json"
OUT_MD = ROOT / "reproducibility" / "unbind_handler_488_run.md"


def score_one(source: str, input_shapes: dict) -> dict:
    src = PREAMBLE + source
    t0 = time.perf_counter()
    captured = io.StringIO()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            res = verify_architecture(src, input_shapes=input_shapes, max_cegar_iterations=3)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed = (time.perf_counter() - t0) * 1000
    if err:
        return {"bucket": "Abstain", "elapsed_ms": elapsed, "err": err}
    if res.abstained:
        return {"bucket": "Abstain", "elapsed_ms": elapsed}
    if res.bug_count > 0:
        first_bug = res.bugs[0].message if res.bugs else ""
        # [MODEL_CHECK] No nn.Module subclass found — block extractor stripped
        # surrounding class context; reclassify as not-analysable Abstain rather
        # than counting as a genuine refutation.
        if first_bug.startswith("[MODEL_CHECK]"):
            return {"bucket": "Abstain", "elapsed_ms": elapsed, "note": "not_analyzable",
                    "first_bug": first_bug[:200]}
        return {"bucket": "Refuted", "elapsed_ms": elapsed, "bug_count": res.bug_count,
                "first_bug": first_bug[:200]}
    return {"bucket": "Verified", "elapsed_ms": elapsed}


def main():
    records = [json.loads(l) for l in CORPUS.open()]
    print(f"Running unbind-handler 488-block analysis on {len(records)} blocks...")
    results = []
    t0 = time.time()
    for i, rec in enumerate(records):
        r = score_one(rec["source"], rec.get("input_shapes", {}))
        r["id"] = rec["id"]
        r["class_name"] = rec["class_name"]
        r["library"] = rec["library"]
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)} ({time.time()-t0:.0f}s)")
    
    # Tally
    from collections import Counter
    tally = Counter(r["bucket"] for r in results)
    refuted = [r for r in results if r["bucket"] == "Refuted"]
    
    out = {
        "n_total": len(results),
        "Verified": tally["Verified"],
        "Refuted": tally["Refuted"],
        "Abstain": tally["Abstain"],
        "refuted_blocks": [{"id": r["id"], "class_name": r["class_name"],
                            "library": r["library"],
                            "first_bug": r.get("first_bug", "")} for r in refuted],
        "all_results": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    
    md = f"""# 488-block analysis: unbind handler

## Headline triple
- Verified: {tally['Verified']}
- Refuted-Proof (RP): {tally['Refuted']}
- Abstain: {tally['Abstain']}
- Total: {len(results)}

## Refuted blocks
"""
    for r in refuted:
        md += f"- `{r['class_name']}` ({r['library']}): {r.get('first_bug','')[:100]}\n"
    
    md += f"""
## Change from prior run
Prior headline: 57V / 0RP / 431A (0 unconditional RP).
New headline: {tally['Verified']}V / {tally['Refuted']}RP / {tally['Abstain']}A.
"""
    OUT_MD.write_text(md)
    print(f"\nDone. Refuted={tally['Refuted']}, Verified={tally['Verified']}, Abstain={tally['Abstain']}")
    print(f"Output: {OUT_JSON}")

if __name__ == "__main__":
    main()
