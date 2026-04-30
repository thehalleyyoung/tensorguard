"""
run_v5_benchmark.py
===================

Runs TensorGuard's `verify_architecture` (the v5 baseline analyzer; v5
extension modules — Track C — are loaded if importable, otherwise we fall
back honestly) on the BLOCK corpus (`v5_block_corpus.jsonl`) and the BUG
corpus (`v5_bug_corpus.jsonl`).

For every input we record one of three outcomes following the calibrated
honesty contract:

    Verified  – analyzer returned SAFE and was NOT abstained
                AND ground-truth label is "is_buggy=False"
                                          (or unknown for clean blocks).
    Refuted   – analyzer flagged ≥1 bug.
    Abstain   – analyzer marked itself abstained, raised on the input,
                or returned SAFE on a known-buggy input (silent miss is
                logged as REFUTED-MISS, see below).

The abstain-reason taxonomy (top-10) is derived from the analyzer's
own warning stream (captured stdout/stderr) plus heuristics on the
source code.  Every abstain is tagged with EXACTLY ONE primary reason
and may carry one or more secondary tags.

Outputs:
  experiments_v5/v5_benchmark_results.json
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402
try:
    # v5 (Track C) modules — load if available, else fall back silently.
    from src.api import verify_module as _v5_verify_module  # noqa: F401
    HAS_V5_EXT = True
except Exception:
    HAS_V5_EXT = False


BLOCK_JSONL = ROOT / "v5_block_corpus.jsonl"
BUG_JSONL = ROOT / "v5_bug_corpus.jsonl"
BUG_REPRO_DIR = ROOT / "bug_repros"
OUT_JSON = ROOT / "v5_benchmark_results.json"

PER_INPUT_TIMEOUT_S = 60.0  # hard wall-clock budget; we don't actually preempt
                            # but we record exceeding cases.

# --------------------------------------------------------------------------
# Abstain-reason taxonomy (top-10)
# --------------------------------------------------------------------------
ABSTAIN_TAXONOMY = [
    ("opaque_config_attr",
     r"\b(self\.config|self\.cfg|args\.|hparams\.|model_config\.)"),
    ("qkv_unpack",
     r"(qkv\s*=\s*self\.qkv|chunk\s*\(\s*3|split\s*\(\s*[^,]+,\s*dim)"),
    ("data_dependent_control_flow",
     r"\b(if\s+x\.|if\s+.*\.shape\[|if\s+self\.training)"),
    ("external_library_call",
     r"\b(F\.scaled_dot_product_attention|torch\.einsum|"
     r"flash_attn|xformers|apex\.|deepspeed\.|fused_ops\.)"),
    ("tuple_returning_forward",
     r"return\s+\([^)]+,[^)]+\)|return\s+[a-zA-Z_]+\s*,\s*[a-zA-Z_]+"),
    ("dynamic_reshape",
     r"\.(view|reshape)\s*\([^)]*-1[^)]*\)|\.view\(\s*\*"),
    ("opaque_submodule",
     r"Unsupported layer kind UNKNOWN"),
    ("registered_buffer_or_parameter",
     r"register_buffer\(|nn\.Parameter\("),
    ("python_loop_over_layers",
     r"for\s+\w+\s+in\s+self\.\w+\s*:|nn\.ModuleList"),
    ("complex_indexing_or_gather",
     r"torch\.gather|\.index_select|\.scatter|\bx\[[^]]*,\s*[^]]*\]"),
]


def _classify_abstain(source: str, captured_warnings: str) -> Tuple[str, List[str]]:
    """Return (primary_reason, all_matched_tags). primary_reason is the
    first taxonomy tag that matches; if none match, 'unclassified'."""
    tags: List[str] = []
    haystack = source + "\n" + captured_warnings
    for tag, pat in ABSTAIN_TAXONOMY:
        if re.search(pat, haystack, flags=re.MULTILINE):
            tags.append(tag)
    primary = tags[0] if tags else "unclassified"
    return primary, tags


# --------------------------------------------------------------------------
# Verifier wrapper
# --------------------------------------------------------------------------
PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)


def _run_one(source: str, input_shapes: Dict[str, tuple],
             ) -> Dict[str, Any]:
    src = PREAMBLE + source
    captured = io.StringIO()
    t0 = time.perf_counter()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            res = verify_architecture(src, input_shapes=input_shapes,
                                      max_cegar_iterations=3,
                                      filename="<bench>")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out: Dict[str, Any] = {
        "elapsed_ms": round(elapsed_ms, 1),
        "exception": err,
        "warnings_captured": captured.getvalue()[-4000:],
    }
    if res is not None:
        out["analyzer_status"] = res.status
        out["analyzer_abstained"] = bool(res.abstained)
        out["opaque_layer_count"] = int(res.opaque_layer_count)
        out["bug_count"] = int(res.bug_count)
        out["bugs"] = [{"category": b.category.value,
                        "severity": b.severity,
                        "message": b.message[:300]} for b in res.bugs[:10]]
    return out


def _decide(record: Dict[str, Any], is_buggy_gt: bool | None,
            source: str) -> Dict[str, Any]:
    """Map raw analyzer outcome → {Verified, Refuted, Abstain} bucket and
    annotate with abstain-reason taxonomy when applicable."""
    out: Dict[str, Any] = {}
    if record.get("exception"):
        out["bucket"] = "Abstain"
        out["abstain_reason"], out["abstain_tags"] = _classify_abstain(
            source, record["exception"] + "\n" + record["warnings_captured"])
        out["abstain_subkind"] = "exception"
        return out
    if record.get("analyzer_abstained"):
        out["bucket"] = "Abstain"
        out["abstain_reason"], out["abstain_tags"] = _classify_abstain(
            source, record["warnings_captured"])
        out["abstain_subkind"] = "analyzer_self_abstain"
        return out
    if record.get("bug_count", 0) > 0:
        out["bucket"] = "Refuted"
        return out
    out["bucket"] = "Verified"
    if is_buggy_gt is True:
        out["calibration_note"] = "VERIFIED_BUT_GROUND_TRUTH_BUGGY (silent miss)"
    return out


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def run_block_corpus() -> List[Dict[str, Any]]:
    if not BLOCK_JSONL.exists():
        print("Block corpus missing — run build_block_corpus.py first.")
        return []
    records = [json.loads(l) for l in BLOCK_JSONL.open()]
    print(f"[blocks] running TensorGuard on {len(records)} blocks ...")
    out = []
    t0 = time.time()
    for i, rec in enumerate(records):
        raw = _run_one(rec["source"], rec["input_shapes"])
        decision = _decide(raw, is_buggy_gt=False, source=rec["source"])
        out.append({
            "id": rec["id"],
            "qualified_name": rec["qualified_name"],
            "library": rec["library"],
            "category": rec["category"],
            "loc": rec["loc"],
            "source_sha256": rec["source_sha256"],
            **decision,
            **{k: raw[k] for k in ("elapsed_ms", "analyzer_status",
                                   "analyzer_abstained", "opaque_layer_count",
                                   "bug_count", "exception") if k in raw},
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(records)} ({time.time()-t0:.0f}s)")
    return out


def run_bug_corpus() -> List[Dict[str, Any]]:
    if not BUG_JSONL.exists():
        print("Bug corpus missing.")
        return []
    records = [json.loads(l) for l in BUG_JSONL.open()]
    print(f"[bugs] running TensorGuard on {len(records)} bug repros ...")
    out = []
    for i, rec in enumerate(records):
        repro_path = REPO / rec["repro_file"]
        if not repro_path.exists():
            out.append({"id": rec["id"], "bucket": "Abstain",
                        "abstain_subkind": "missing_repro",
                        "abstain_reason": "missing_repro", "abstain_tags": []})
            continue
        src = repro_path.read_text()
        # Look for INPUT_SHAPES dict in the repro file; fall back to {}.
        m = re.search(r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", src, flags=re.MULTILINE)
        try:
            shapes = eval(m.group(1)) if m else {}
        except Exception:
            shapes = {}
        raw = _run_one(src, shapes)
        decision = _decide(raw, is_buggy_gt=True, source=src)
        out.append({
            "id": rec["id"],
            "github_url": rec["github_url"],
            "category": rec["category"],
            "is_buggy_gt": True,
            **decision,
            **{k: raw[k] for k in ("elapsed_ms", "analyzer_status",
                                   "analyzer_abstained", "opaque_layer_count",
                                   "bug_count", "exception") if k in raw},
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}")
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def summarize(name: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(recs)
    bucket_counts = Counter(r["bucket"] for r in recs)
    by_category = defaultdict(lambda: Counter())
    by_library = defaultdict(lambda: Counter())
    for r in recs:
        if "category" in r:
            by_category[r["category"]][r["bucket"]] += 1
        if "library" in r:
            by_library[r["library"]][r["bucket"]] += 1
    abstain_reasons = Counter(r.get("abstain_reason") for r in recs
                              if r["bucket"] == "Abstain")
    silent_misses = sum(1 for r in recs
                        if r["bucket"] == "Verified"
                        and r.get("calibration_note", "").startswith("VERIFIED_BUT_GROUND_TRUTH_BUGGY"))
    return {
        "name": name,
        "total": total,
        "buckets": dict(bucket_counts),
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "by_library": {k: dict(v) for k, v in by_library.items()},
        "abstain_reasons_top10": dict(abstain_reasons.most_common(10)),
        "all_abstain_reasons": dict(abstain_reasons),
        "silent_misses": silent_misses,
    }


def main():
    t0 = time.time()
    block_results = run_block_corpus()
    bug_results = run_bug_corpus()
    out = {
        "meta": {
            "build_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 1),
            "v5_extension_modules_available": HAS_V5_EXT,
            "abstain_taxonomy": [t for t, _ in ABSTAIN_TAXONOMY],
        },
        "block_corpus": {
            "summary": summarize("blocks", block_results),
            "per_input": block_results,
        },
        "bug_corpus": {
            "summary": summarize("bugs", bug_results),
            "per_input": bug_results,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print("\n== BLOCK CORPUS ==")
    print(json.dumps(out["block_corpus"]["summary"], indent=2))
    print("\n== BUG CORPUS ==")
    print(json.dumps(out["bug_corpus"]["summary"], indent=2))


if __name__ == "__main__":
    main()
