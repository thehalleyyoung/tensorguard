"""
run_footprint_strict_488.py
===========================

Re-runs (or replays) the 488-block real-source corpus restricted to verdicts
whose entire derivation path lies strictly inside the Lean-or-pen-and-paper
audited operator footprint.

Footprint classification for each block:
  audited       – every operator found in source maps to lean_verified or
                  pen_and_paper in handler_soundness_scope.json
  tested-only   – at least one tested_only operator, but no out-of-scope ops
  out-of-scope  – at least one operator not in any handler entry

Verdicts are taken from the already-computed v5_benchmark_results.json
(so this is a classification pass, not a re-run of the expensive analyzer).

Outputs:
  experiments_v5/footprint_strict_488.csv
  experiments_v5/footprint_strict_488_summary.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

SCOPE_JSON = ROOT / "handler_soundness_scope.json"
BLOCK_JSONL = ROOT / "v5_block_corpus.jsonl"
RESULTS_JSON = ROOT / "v5_benchmark_results.json"
OUT_CSV = ROOT / "footprint_strict_488.csv"
OUT_JSON = ROOT / "footprint_strict_488_summary.json"


# ---------------------------------------------------------------------------
# Build operator scope maps from handler_soundness_scope.json
# ---------------------------------------------------------------------------
def _load_scope_maps() -> Tuple[Set[str], Set[str], Set[str]]:
    scope = json.loads(SCOPE_JSON.read_text())
    lean_verified: Set[str] = set()
    pen_and_paper: Set[str] = set()
    tested_only: Set[str] = set()
    for h in scope["handlers"]:
        name = h["name"]
        s = h["scope"]
        if s == "lean_verified":
            lean_verified.add(name)
        elif s == "pen_and_paper":
            pen_and_paper.add(name)
        elif s == "tested_only":
            tested_only.add(name)
    return lean_verified, pen_and_paper, tested_only


# ---------------------------------------------------------------------------
# Operator detection patterns
# Each entry maps a regex pattern → canonical handler name.
# Patterns are applied to the source in order; a match records that handler.
# ---------------------------------------------------------------------------
_OP_PATTERNS: List[Tuple[str, str]] = [
    # linalg (must come before generic matmul)
    (r"torch\.linalg\.svd|linalg\.svd", "linalg_svd"),
    (r"torch\.linalg\.qr|linalg\.qr", "linalg_qr"),
    (r"torch\.linalg\.solve|linalg\.solve", "linalg_solve"),
    (r"torch\.linalg\.eig|linalg\.eig", "linalg_eig"),
    # matmul / bmm
    (r"torch\.matmul|\.matmul\(", "matmul"),
    (r"torch\.bmm|\.bmm\(", "bmm"),
    # convolutions
    (r"nn\.Conv1d|F\.conv1d", "conv1d"),
    (r"nn\.Conv2d|F\.conv2d", "conv2d"),
    (r"nn\.Conv3d|F\.conv3d", "conv3d"),
    (r"nn\.ConvTranspose2d|F\.conv_transpose2d", "conv_transpose2d"),
    (r"nn\.ConvTranspose1d|F\.conv_transpose1d", "conv_transpose1d"),
    # view / reshape
    (r"\.view\(", "view"),
    (r"\.reshape\(", "reshape"),
    # permute / transpose / expand / repeat
    (r"\.permute\(", "permute"),
    (r"\.transpose\(|torch\.transpose", "transpose"),
    (r"\.expand\(", "expand"),
    (r"torch\.repeat_interleave|\.repeat_interleave\(", "repeat_interleave"),
    (r"\.repeat\(", "repeat"),
    # broadcast_to
    (r"torch\.broadcast_to|\.broadcast_to\(", "broadcast_to"),
    # cat / stack / split / chunk / unbind
    (r"torch\.cat\b", "cat"),
    (r"torch\.stack\b", "stack"),
    (r"\.split\(", "split"),
    (r"\.chunk\(", "chunk"),
    (r"\.unbind\(", "unbind"),
    # gather / scatter / index_select / narrow
    (r"torch\.gather|\.gather\(", "gather"),
    (r"torch\.scatter_add|\.scatter_add\b", "scatter_add"),
    (r"torch\.scatter|\.scatter\b|\.scatter_\(", "scatter"),
    (r"torch\.index_select|\.index_select\(", "index_select"),
    (r"\.narrow\(", "narrow"),
    # embedding
    (r"nn\.Embedding\b|F\.embedding\b", "embed"),
    # norms
    (r"nn\.LayerNorm|F\.layer_norm", "layer_norm"),
    (r"rms_norm|RMSNorm", "rms_norm"),
    (r"nn\.BatchNorm1d|nn\.BatchNorm2d|nn\.BatchNorm3d|F\.batch_norm", "batch_norm"),
    (r"nn\.GroupNorm|F\.group_norm", "group_norm"),
    (r"nn\.InstanceNorm1d|nn\.InstanceNorm2d|nn\.InstanceNorm3d|F\.instance_norm", "instance_norm"),
    # attention
    (r"F\.scaled_dot_product_attention", "scaled_dot_product_attention"),
    (r"nn\.MultiheadAttention", "multihead_attention"),
    # linear
    (r"nn\.Linear\b|F\.linear\b", "linear"),
    # squeeze / unsqueeze
    (r"\.squeeze\(", "squeeze"),
    (r"\.unsqueeze\(", "unsqueeze"),
    # flatten
    (r"\.flatten\(|torch\.flatten", "flatten"),
    # activations
    (r"F\.softmax\b|nn\.Softmax", "softmax"),
    (r"F\.relu\b|nn\.ReLU\b|\.relu\(", "relu"),
    (r"F\.gelu\b|nn\.GELU\b", "gelu"),
    (r"F\.silu\b|nn\.SiLU\b", "silu"),
    (r"\.tanh\(|F\.tanh\b|nn\.Tanh\b", "tanh"),
    (r"\.sigmoid\(|F\.sigmoid\b|nn\.Sigmoid\b", "sigmoid"),
    (r"F\.dropout\b|nn\.Dropout\b", "dropout"),
    # loss
    (r"F\.cross_entropy\b", "cross_entropy"),
    # pooling
    (r"F\.interpolate\b", "interpolate"),
    (r"F\.pixel_shuffle\b|nn\.PixelShuffle\b", "pixel_shuffle"),
    (r"F\.pixel_unshuffle\b", "pixel_unshuffle"),
    (r"nn\.AdaptiveAvgPool1d|nn\.AdaptiveAvgPool2d|nn\.AdaptiveAvgPool3d"
     r"|F\.adaptive_avg_pool1d|F\.adaptive_avg_pool2d|F\.adaptive_avg_pool3d",
     "adaptive_avg_pool"),
    (r"nn\.MaxPool2d|nn\.MaxPool1d|nn\.MaxPool3d|F\.max_pool2d", "max_pool2d"),
    (r"nn\.AvgPool2d|nn\.AvgPool1d|F\.avg_pool2d", "avg_pool2d"),
    # misc
    (r"torch\.topk\b|\.topk\(", "topk"),
    (r"F\.glu\b|nn\.GLU\b", "glu"),
    (r"\.unfold\(", "unfold"),
    (r"F\.fold\b", "fold"),
    (r"torch\.where\b", "where"),
    (r"torch\.masked_select\b", "masked_select"),
    (r"torch\.take_along_dim\b", "take_along_dim"),
    (r"torch\.roll\b|torch\.flip\b", "roll_flip"),
    (r"torch\.linalg\.svd|linalg_svd", "linalg_svd"),
    (r"apply_rotary|rotary_emb|rotary_embedding", "rotary_embedding"),
    (r"einops\.rearrange|rearrange\(", "einops_rearrange"),
    (r"torch\.stft\b", "stft"),
    (r"torch\.zeros\b|torch\.ones\b|torch\.randn\b|torch\.rand\b"
     r"|torch\.empty\b|torch\.full\b|torch\.arange\b|torch\.linspace\b",
     "creation_ops"),
    (r"\.detach\(", "detach"),
    (r"F\.pad\b|torch\.nn\.functional\.pad\b", "pad"),
    (r"torch\.addmm\b|\.addmm\(", "addmm"),
    (r"torch\.baddbmm\b", "baddbmm"),
    (r"torch\.outer\b", "outer"),
    (r"torch\.tensordot\b", "tensordot"),
    (r"F\.grid_sample\b", "grid_sample"),
    (r"checkpoint\(|torch\.utils\.checkpoint", "checkpoint"),
    (r"\.to\(device|\.to\(dtype|\.to\(torch\.", "to"),
    (r"\.contiguous\(", "contiguous"),
    (r"\.clamp\(|torch\.clamp\b", "clamp"),
    (r"\.argmax\(|torch\.argmax\b", "argmax"),
    (r"torch\.einsum\b", "einsum"),
    # element-wise binary arithmetic (catch-all for +, -, *, / between tensors)
    (r"torch\.add\b|torch\.sub\b|torch\.mul\b|torch\.div\b"
     r"|torch\.pow\b|torch\.exp\b|torch\.log\b|torch\.abs\b"
     r"|torch\.sqrt\b|torch\.rsqrt\b",
     "elementwise_binary"),
    # reduce ops
    (r"torch\.sum\b|torch\.mean\b|torch\.max\b|torch\.min\b"
     r"|torch\.prod\b|torch\.norm\b|torch\.std\b|torch\.var\b"
     r"|\.sum\(|\.mean\(|\.max\(|\.min\(",
     "reduce"),
]

# Compile patterns once.
_COMPILED = [(re.compile(pat, re.MULTILINE), name) for pat, name in _OP_PATTERNS]


def _extract_ops(source: str) -> Set[str]:
    found: Set[str] = set()
    for pat, name in _COMPILED:
        if pat.search(source):
            found.add(name)
    return found


def _classify_block(ops: Set[str],
                    lean_verified: Set[str],
                    pen_and_paper: Set[str],
                    tested_only: Set[str]) -> Tuple[str, List[str]]:
    """Return (footprint_class, sorted_ops_touched)."""
    audited_set = lean_verified | pen_and_paper
    known_set = audited_set | tested_only
    has_oos = any(op not in known_set for op in ops)
    has_tested_only = any(op in tested_only and op not in audited_set for op in ops)
    if has_oos:
        return "out-of-scope", sorted(ops)
    if has_tested_only:
        return "tested-only", sorted(ops)
    return "audited", sorted(ops)


def main() -> None:
    lean_verified, pen_and_paper, tested_only = _load_scope_maps()

    blocks: Dict[str, dict] = {}
    for line in BLOCK_JSONL.open():
        rec = json.loads(line)
        blocks[rec["id"]] = rec

    results = json.loads(RESULTS_JSON.read_text())
    per_input: List[dict] = results["block_corpus"]["per_input"]

    rows = []
    for entry in per_input:
        block_id = entry["id"]
        verdict_bucket = entry.get("bucket", "Abstain")
        source = blocks[block_id]["source"] if block_id in blocks else ""
        ops = _extract_ops(source)
        footprint_class, ops_touched = _classify_block(
            ops, lean_verified, pen_and_paper, tested_only)
        rows.append({
            "block_id": block_id,
            "verdict": verdict_bucket,
            "footprint_class": footprint_class,
            "ops_touched": "|".join(ops_touched),
        })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["block_id", "verdict", "footprint_class", "ops_touched"])
        writer.writeheader()
        writer.writerows(rows)

    # --- Aggregation ---
    summary: Dict[str, Dict[str, int]] = {}
    class_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        fc = row["footprint_class"]
        v = row["verdict"]
        if fc not in summary:
            summary[fc] = {"V": 0, "CV": 0, "RP": 0, "A": 0}
        class_counts[fc] += 1
        if v == "Verified":
            summary[fc]["V"] += 1
        elif v == "Refuted":
            summary[fc]["RP"] += 1
        else:
            summary[fc]["A"] += 1

    out = {
        "audited": summary.get("audited", {"V": 0, "CV": 0, "RP": 0, "A": 0}),
        "tested-only": summary.get("tested-only", {"V": 0, "CV": 0, "RP": 0, "A": 0}),
        "out-of-scope": summary.get("out-of-scope", {"V": 0, "CV": 0, "RP": 0, "A": 0}),
        "class_counts": dict(class_counts),
        "total_blocks": len(rows),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"[footprint-strict] wrote {OUT_CSV.name} ({len(rows)} rows)")
    print(f"[footprint-strict] wrote {OUT_JSON.name}")
    for fc, counts in out.items():
        if isinstance(counts, dict) and "V" in counts:
            n = class_counts.get(fc, 0)
            print(f"  {fc}: {n} blocks — V={counts['V']} RP={counts['RP']} A={counts['A']}")


if __name__ == "__main__":
    main()
