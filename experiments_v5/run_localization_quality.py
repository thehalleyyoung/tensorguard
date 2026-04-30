"""
experiments_v5/run_localization_quality.py
==========================================

Measures localization quality of TensorGuard v4 (raw bug.location.line)
versus v5 (src.v5.localization.localize) on the v5 bug corpus.

For each known-buggy repro:
  1. Read the source file.
  2. Run verify_architecture (with PREAMBLE prefix).
  3. Determine the ground-truth bug line:
       - Parse repro for '# BUG_LINE: <N>' marker → gt_source="marker"
       - Fall back to first occurrence of a bug-introducing pattern → gt_source="heuristic"
  4. Apply v4 (raw bug.location.line) and v5 (localize) to get line estimates.
     Both estimates are in the combined (PREAMBLE + repro) coordinate space.
  5. Compute |estimate - gt| and tally within_1 / within_5 / within_10 / miss.

Output: experiments_v5/localization_quality.json
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

import torch
from src.api import verify_architecture
from src.v5.localization import localize

BUG_JSONL = ROOT / "v5_bug_corpus.jsonl"
BUG_REPRO_DIR = ROOT / "bug_repros"
OUT_JSON = ROOT / "localization_quality.json"

PREAMBLE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from typing import Optional, Tuple, List, Dict, Any\n"
)
PREAMBLE_LINES = len(PREAMBLE.splitlines())  # = 4

# Bug-introducing patterns ordered by precedence for heuristic GT.
# These cover every category in the v5 bug corpus.
_BUG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("view",           re.compile(r"\.view\s*\(")),
    ("reshape",        re.compile(r"\.reshape\s*\(")),
    ("Conv2d",         re.compile(r"\bConv\d*[dD]?\s*\(")),
    ("Linear",         re.compile(r"\bLinear\s*\(")),
    ("BatchNorm",      re.compile(r"\bBatchNorm\w*\s*\(")),
    ("MultiheadAttn",  re.compile(r"\bMultiheadAttention\s*\(")),
    ("LSTM",           re.compile(r"\bLSTM\s*\(|\bGRU\s*\(")),
    ("Embedding",      re.compile(r"\bEmbedding\s*\(")),
    ("cat",            re.compile(r"torch\.cat\s*\(")),
    ("stack",          re.compile(r"torch\.stack\s*\(")),
    ("matmul",         re.compile(r"torch\.matmul\s*\(|torch\.mm\s*\(|@\s")),
    ("einsum",         re.compile(r"torch\.einsum\s*\(")),
    ("sdpa",           re.compile(r"scaled_dot_product_attention\s*\(")),
    ("cross_entropy",  re.compile(r"CrossEntropyLoss\s*\(|cross_entropy\s*\(")),
    ("isclose",        re.compile(r"torch\.isclose\s*\(")),
    ("broadcast",      re.compile(r"\ba\s*\+\s*b|\*=\s*self\.|x\s*\*\s*self\.|a\s*\+\s*")),
    ("index",          re.compile(r"\[.*:.*\]|\.select\s*\(|\.narrow\s*\(")),
    ("permute",        re.compile(r"\.permute\s*\(|\.transpose\s*\(")),
    ("norm",           re.compile(r"\bLayerNorm\s*\(|\bGroupNorm\s*\(")),
]


def _find_bug_line_marker(source: str) -> Optional[int]:
    """Return N from '# BUG_LINE: N' if present."""
    m = re.search(r"#\s*BUG_LINE\s*:\s*(\d+)", source)
    if m:
        return int(m.group(1))
    return None


def _find_bug_line_heuristic(source: str) -> Optional[int]:
    """Find the first line in source matching a known bug-introducing pattern."""
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        for _, pat in _BUG_PATTERNS:
            if pat.search(line):
                return i
    return None


def _run_one(
    src: str,
    input_shapes: Dict[str, Any],
) -> Optional[Any]:
    """Run verify_architecture; return result or None on exception."""
    full_src = PREAMBLE + src
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            result = verify_architecture(
                full_src,
                input_shapes=input_shapes,
                max_cegar_iterations=3,
                filename="<bench>",
            )
        return result, full_src
    except Exception:
        return None, PREAMBLE + src


def _tally(distances: List[Optional[int]]) -> Dict[str, int]:
    """Convert a list of distances (or None) to within_* tallies."""
    t = {"within_1": 0, "within_5": 0, "within_10": 0, "miss": 0}
    for d in distances:
        if d is None:
            t["miss"] += 1
        elif d <= 1:
            t["within_1"] += 1
            t["within_5"] += 1
            t["within_10"] += 1
        elif d <= 5:
            t["within_5"] += 1
            t["within_10"] += 1
        elif d <= 10:
            t["within_10"] += 1
        else:
            t["miss"] += 1
    return t


def main() -> None:
    records = [json.loads(l) for l in BUG_JSONL.open()]
    print(f"[localization] {len(records)} bug records in corpus")

    per_item: List[Dict[str, Any]] = []
    n_refuted = 0
    n_bugs_total = len(records)

    for i, rec in enumerate(records):
        bug_id = rec["id"]
        repro_path = REPO / rec["repro_file"]
        if not repro_path.exists():
            print(f"  [{bug_id}] repro file missing — skip")
            continue

        src = repro_path.read_text()

        # Parse INPUT_SHAPES from repro file
        shapes_m = re.search(
            r"^INPUT_SHAPES\s*=\s*(\{[^}]*\})", src, flags=re.MULTILINE
        )
        try:
            input_shapes = eval(shapes_m.group(1)) if shapes_m else {}
        except Exception:
            input_shapes = {}

        # Determine ground-truth line
        marker_line = _find_bug_line_marker(src)
        if marker_line is not None:
            gt_line = marker_line + PREAMBLE_LINES
            gt_source = "marker"
        else:
            heur_line = _find_bug_line_heuristic(src)
            if heur_line is not None:
                gt_line = heur_line + PREAMBLE_LINES
                gt_source = "heuristic"
            else:
                gt_line = None
                gt_source = "unknown"

        # Run TensorGuard
        result, full_src = _run_one(src, input_shapes)

        if result is None or not result.bugs:
            # Not refuted — still record for completeness
            item = {
                "id": bug_id,
                "category": rec.get("category"),
                "gt_line": gt_line,
                "gt_source": gt_source,
                "refuted": False,
                "v4_line": None,
                "v5_line": None,
                "dist_v4": None,
                "dist_v5": None,
            }
            per_item.append(item)
            continue

        n_refuted += 1

        # v4: raw first bug line (in full_src coordinate space)
        v4_line = result.bugs[0].location.line if result.bugs[0].location.line > 0 else None

        # v5: localize using message + counterexample
        first_bug = result.bugs[0]
        v5_line = localize(full_src, first_bug.message, result.counterexample)

        # Distances (None if gt_line unknown or line estimate unknown)
        dist_v4 = abs(v4_line - gt_line) if (v4_line and gt_line) else None
        dist_v5 = abs(v5_line - gt_line) if (v5_line and gt_line) else None

        item = {
            "id": bug_id,
            "category": rec.get("category"),
            "gt_line": gt_line,
            "gt_source": gt_source,
            "refuted": True,
            "v4_line": v4_line,
            "v5_line": v5_line,
            "dist_v4": dist_v4,
            "dist_v5": dist_v5,
        }
        per_item.append(item)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_bugs_total} done")

    # Only summarize items that were refuted AND have a gt_line
    refuted_items = [it for it in per_item if it["refuted"] and it["gt_line"] is not None]
    print(f"\n[localization] {n_refuted} refuted / {n_bugs_total} total")
    print(f"[localization] {len(refuted_items)} refuted with known gt_line")

    v4_dists = [it["dist_v4"] for it in refuted_items]
    v5_dists = [it["dist_v5"] for it in refuted_items]

    summary = {
        "v4": _tally(v4_dists),
        "v5": _tally(v5_dists),
    }

    import datetime
    output = {
        "meta": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "torch_version": torch.__version__,
            "n_bugs": n_bugs_total,
            "n_refuted": n_refuted,
            "n_with_gt": len(refuted_items),
        },
        "summary": summary,
        "per_item": per_item,
    }

    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n[localization] wrote {OUT_JSON}")

    # Print summary
    print("\n=== LOCALIZATION QUALITY SUMMARY ===")
    print(f"Refuted with gt: {len(refuted_items)}")
    hdr = f"{'':12s}  {'within_1':>8}  {'within_5':>8}  {'within_10':>9}  {'miss':>6}"
    print(hdr)
    for ver, tag in [("v4", "v4 (raw)"), ("v5", "v5 (new)")]:
        t = summary[ver]
        n = len(refuted_items) or 1
        print(
            f"{tag:12s}  "
            f"{t['within_1']:>5} ({100*t['within_1']//n:2d}%)  "
            f"{t['within_5']:>5} ({100*t['within_5']//n:2d}%)  "
            f"{t['within_10']:>6} ({100*t['within_10']//n:2d}%)  "
            f"{t['miss']:>5} ({100*t['miss']//n:2d}%)"
        )


if __name__ == "__main__":
    main()
