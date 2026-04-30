"""
run_feature_ablation.py
=======================

Per-feature ablation table for TensorGuard v5.

ABLATION STRATEGY DECISION
--------------------------
After inspecting `src/v5/__init__.py`, `src/v5/attention_norms.py`, and
`experiments_v5/track_C_coverage.json`:

  * `src/v5/attention_norms.py` registers ops into `TORCH_SHAPE_OPS` at import
    time, but `verify_architecture` calls `verify_model` from `model_checker.py`
    which does *not* consult that dispatch table for its primary verdict.
  * Track C confirmed: delta = {VERIFIED:0, REFUTED:0, ABSTAIN:0} — v5 imports
    do NOT change `verify_architecture` verdict counts.
  * The other v5 modules (`symbolic_config`, `qkv_unpacking`, `reshape_neg1`,
    `backward_shape`, `grad_flag_verifier`) expose standalone helper APIs never
    called by `verify_architecture`.

Therefore we use the HONEST FALLBACK: ablate `verify_architecture`'s own kwargs.

Functional parameters actually wired up in `verify_architecture` (api.py):
  • high_confidence_only  — forwarded to verify_model → filter_by_confidence(HIGH)
  • max_cegar_iterations  — gates run_shape_cegar; predicates are stored as
                           metadata only (not fed back as Bug objects)

Parameters accepted but NOT forwarded to verify_model:
  • check_devices, check_phases, check_gradients

FEATURE LADDER (6 rows)
-----------------------
L0  base fragment only  — high_confidence_only=True, max_cegar_iterations=0,
                          check_devices=False, check_phases=False,
                          check_gradients=False
L1  + CEGAR             — high_confidence_only=True, max_cegar_iterations=3
L2  + device check      — add check_devices=True  (accepted; no-op in v5 impl)
L3  + phase check       — add check_phases=True   (accepted; no-op in v5 impl)
L4  + gradient check    — add check_gradients=True (accepted; no-op in v5 impl)
L5  full                — high_confidence_only=False (drops HIGH-confidence
                          filter → adds LOW-confidence violations), all checks on

Outputs
-------
experiments_v5/feature_ablation.json  — structured results
Markdown table printed to stdout at end of run.
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from src.api import verify_architecture  # noqa: E402

try:
    import torch
    TORCH_VERSION = torch.__version__
except Exception:
    TORCH_VERSION = "unknown"

BLOCK_JSONL = ROOT / "v5_block_corpus.jsonl"
BUG_JSONL = ROOT / "v5_bug_corpus.jsonl"
BUG_REPRO_DIR = ROOT / "bug_repros"
OUT_JSON = ROOT / "feature_ablation.json"

PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)

# ---------------------------------------------------------------------------
# Feature-level definitions
# ---------------------------------------------------------------------------

LEVELS: List[Dict[str, Any]] = [
    {
        "level": "L0",
        "label": "base fragment only (high-confidence, no CEGAR, no secondary checks)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 0,
            "check_devices": False,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L1",
        "label": "+ CEGAR (3 iterations, predicates-only)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": False,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L2",
        "label": "+ device consistency check (check_devices=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": False,
            "check_gradients": False,
        },
    },
    {
        "level": "L3",
        "label": "+ train/eval phase check (check_phases=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": False,
        },
    },
    {
        "level": "L4",
        "label": "+ gradient flow check (check_gradients=True)",
        "kwargs": {
            "high_confidence_only": True,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": True,
        },
    },
    {
        "level": "L5",
        "label": "full (high_confidence_only=False — adds low-confidence violations)",
        "kwargs": {
            "high_confidence_only": False,
            "max_cegar_iterations": 3,
            "check_devices": True,
            "check_phases": True,
            "check_gradients": True,
        },
    },
]

# ---------------------------------------------------------------------------
# Per-input runner
# ---------------------------------------------------------------------------

def _run_one(source: str, input_shapes: Dict[str, tuple],
             level_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    src = PREAMBLE + source
    captured = io.StringIO()
    t0 = time.perf_counter()
    err = None
    res = None
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            res = verify_architecture(
                src,
                input_shapes=input_shapes,
                filename="<bench>",
                **level_kwargs,
            )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out: Dict[str, Any] = {
        "elapsed_ms": round(elapsed_ms, 1),
        "exception": err,
        "warnings_captured": captured.getvalue()[-2000:],
    }
    if res is not None:
        out["analyzer_status"] = res.status
        out["analyzer_abstained"] = bool(res.abstained)
        out["opaque_layer_count"] = int(res.opaque_layer_count)
        out["bug_count"] = int(res.bug_count)
    return out


def _decide(record: Dict[str, Any], is_buggy_gt: Optional[bool]) -> str:
    """Return one of: Refuted | Verified | Abstain | SilentMiss"""
    if record.get("exception"):
        return "Abstain"
    if record.get("analyzer_abstained"):
        return "Abstain"
    if record.get("bug_count", 0) > 0:
        return "Refuted"
    # Verified — but was it a miss?
    if is_buggy_gt is True:
        return "SilentMiss"
    return "Verified"


# ---------------------------------------------------------------------------
# Corpus runners
# ---------------------------------------------------------------------------

def run_bug_corpus(level_kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    if not BUG_JSONL.exists():
        raise FileNotFoundError(f"Bug corpus not found: {BUG_JSONL}")
    records = [json.loads(line) for line in BUG_JSONL.open()]
    print(f"  [{label}] bug corpus: {len(records)} records")
    counts: Counter = Counter()
    t0 = time.time()
    for i, rec in enumerate(records):
        repro_path = REPO / rec["repro_file"]
        if not repro_path.exists():
            counts["abstain"] += 1
            continue
        src = repro_path.read_text()
        m = re.search(r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", src, flags=re.MULTILINE)
        try:
            shapes = eval(m.group(1)) if m else {}
        except Exception:
            shapes = {}
        raw = _run_one(src, shapes, level_kwargs)
        bucket = _decide(raw, is_buggy_gt=True)
        if bucket == "Refuted":
            counts["refuted"] += 1
        elif bucket == "SilentMiss":
            counts["silent_miss"] += 1
        elif bucket == "Abstain":
            counts["abstain"] += 1
        else:
            counts["exception"] += 1
        if raw.get("exception"):
            counts["exception"] += 1
            counts["abstain"] -= 1
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(records)} ({time.time()-t0:.0f}s)")
    n = len(records)
    return {
        "refuted": int(counts["refuted"]),
        "silent_miss": int(counts["silent_miss"]),
        "abstain": int(counts["abstain"]),
        "exception": int(counts["exception"]),
        "n": n,
    }


def run_block_corpus(level_kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    if not BLOCK_JSONL.exists():
        raise FileNotFoundError(f"Block corpus not found: {BLOCK_JSONL}")
    records = [json.loads(line) for line in BLOCK_JSONL.open()]
    print(f"  [{label}] block corpus: {len(records)} records")
    counts: Counter = Counter()
    t0 = time.time()
    for i, rec in enumerate(records):
        raw = _run_one(rec["source"], rec["input_shapes"], level_kwargs)
        bucket = _decide(raw, is_buggy_gt=False)
        if bucket == "Verified":
            counts["verified"] += 1
        elif bucket == "Refuted":
            counts["refuted"] += 1
        else:
            counts["abstain"] += 1
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(records)} ({time.time()-t0:.0f}s)")
    n = len(records)
    abstain_rate = round(counts["abstain"] / n, 4) if n > 0 else 0.0
    return {
        "verified": int(counts["verified"]),
        "refuted": int(counts["refuted"]),
        "abstain": int(counts["abstain"]),
        "abstain_rate": abstain_rate,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_total = time.time()
    ladder = []

    for level_def in LEVELS:
        lvl = level_def["level"]
        label = level_def["label"]
        kwargs = level_def["kwargs"]
        print(f"\n=== {lvl}: {label} ===")
        t0 = time.time()

        bug_result = run_bug_corpus(kwargs, f"{lvl}/bugs")
        block_result = run_block_corpus(kwargs, f"{lvl}/blocks")

        elapsed = round(time.time() - t0, 1)
        print(f"  [{lvl}] done in {elapsed}s | "
              f"bugs: refuted={bug_result['refuted']} miss={bug_result['silent_miss']} "
              f"abstain={bug_result['abstain']} | "
              f"blocks: abstain_rate={block_result['abstain_rate']:.1%}")

        ladder.append({
            "level": lvl,
            "label": label,
            "feature_kwargs": kwargs,
            "bug_corpus": bug_result,
            "block_corpus": block_result,
            "elapsed_s": elapsed,
        })

    # -------------------------------------------------------------------
    # Honest notes
    # -------------------------------------------------------------------
    notes = (
        "ABLATION STRATEGY: verify_architecture kwargs gating (Honest Fallback). "
        "v5 modules are not invoked by verify_architecture; Track C delta = "
        "{VERIFIED:0, REFUTED:0, ABSTAIN:0} confirmed no change from importing them. "
        "The two functional knobs in verify_architecture are: "
        "(1) high_confidence_only — forwarded to verify_model.filter_by_confidence(HIGH); "
        "(2) max_cegar_iterations — gates run_shape_cegar, but CEGAR predicates are stored "
        "as metadata only (not fed back as Bug objects). "
        "check_devices, check_phases, check_gradients are accepted by the API but "
        "NOT forwarded to verify_model in the current implementation; L2/L3/L4 rows "
        "therefore replicate L1 verdict counts and document this no-op behaviour."
    )

    feature_definitions = {
        lvl["level"]: lvl["label"] for lvl in LEVELS
    }

    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch_version": TORCH_VERSION,
            "python_version": sys.version.split()[0],
            "total_elapsed_s": round(time.time() - t_total, 1),
            "feature_definitions": feature_definitions,
            "notes": notes,
        },
        "ladder": ladder,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # -------------------------------------------------------------------
    # Human-readable table
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FEATURE ABLATION TABLE")
    print("=" * 80)
    print(f"{'Level':<5} {'Label':<52} {'Bugs-Det':<9} {'Abstain-Blk':<12} {'Abst%'}")
    print("-" * 80)
    for row in ladder:
        lvl = row["level"]
        lbl = row["label"][:50]
        refuted = row["bug_corpus"]["refuted"]
        abst_blk = row["block_corpus"]["abstain"]
        abst_pct = row["block_corpus"]["abstain_rate"]
        print(f"{lvl:<5} {lbl:<52} {refuted:<9} {abst_blk:<12} {abst_pct:.1%}")
    print("=" * 80)
    print("Bugs-Det = Refuted on 60-bug corpus (higher = more detections = better)")
    print("Abst%    = Abstain rate on 488-block corpus (lower = more coverage)")
    print()
    print("Markdown table:")
    print()
    print("| Level | Label | Bugs detected (Refuted/60) | Silent misses | Block abstain rate |")
    print("|-------|-------|---------------------------|---------------|-------------------|")
    for row in ladder:
        lvl = row["level"]
        lbl = row["label"][:60]
        bc = row["bug_corpus"]
        blk = row["block_corpus"]
        print(f"| {lvl} | {lbl} | {bc['refuted']}/{bc['n']} | "
              f"{bc['silent_miss']} | {blk['abstain_rate']:.1%} ({blk['abstain']}/{blk['n']}) |")


if __name__ == "__main__":
    main()
