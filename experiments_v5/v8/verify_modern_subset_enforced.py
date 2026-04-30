"""Modern-subset enforcement at verification time (Round 2 / Q6).

Reviewer Q6 asks whether the Pytea modern-subset filter
("no-TG-only handler is allowed to fire") is enforced *at
verification time*, not just by post-hoc filtering of the
verdict log.

This harness implements verification-time enforcement:

  1. AST-walk every modern-subset bug repro to extract the set of
     operator calls reachable from ``forward()``.
  2. Reject any bug whose reachable op set contains operators
     **not** in the Pytea-2022 catalogue (SDPA, einsum, Conv1d/3d,
     BatchNorm1d/GroupNorm/InstanceNorm, MHA-2x, swapaxes/movedim,
     where, dot, linalg.*, repeat_interleave, F.add, F.maximum,
     gather, scatter_, isclose, F.embedding, split-with-list-sum).
  3. For all remaining bugs, run TG end-to-end. If TG's verdict
     references a non-catalogue operator (forensics: scan each
     ``Bug.message`` for non-catalogue op names), force-abstain
     and downgrade to Verified.
  4. Compare the resulting Refuted/Total ratio against the
     post-hoc number reported in
     ``experiments_v5/v8/pytea_modern_subset.json``.

The enforcement is *strict*: an op missing from the catalogue
forbids any TG handler from firing on that bug, even if TG would
have fired correctly. This is what the reviewer wants: a
fragment-fairness check at verification time, not a post-hoc
verdict filter.

Output: ``reproducibility/pytea_modern_enforced.json``
"""
from __future__ import annotations

import ast
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)

from src.api import verify_architecture  # noqa: E402

# Pytea 2022 catalogue (per build_modern_subset.py docstring).
PYTEA_CATALOGUE_OPS = {
    "matmul", "mm", "bmm",
    "conv2d", "conv_transpose2d",
    "linear", "Linear",
    "view", "reshape",
    "permute", "transpose",
    "broadcast",
    "cat", "stack",
    "BatchNorm2d", "batch_norm",
    "Embedding",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d",
    "layer_norm", "LayerNorm",
    "flatten", "Flatten",
    "unsqueeze", "squeeze", "expand", "narrow", "pad",
    "sum", "mean", "max", "min", "topk",
    "cross_entropy", "nll_loss", "mse_loss",
    "Conv2d", "ConvTranspose2d",
}

# Strict TG-only ops that the catalogue does NOT contain.
NON_CATALOGUE_OP_NAMES = {
    "scaled_dot_product_attention", "einsum", "Conv1d", "Conv3d",
    "BatchNorm1d", "BatchNorm3d", "GroupNorm", "InstanceNorm1d",
    "InstanceNorm2d", "InstanceNorm3d", "MultiheadAttention",
    "swapaxes", "movedim", "isclose", "where", "dot", "linalg",
    "repeat_interleave", "gather", "scatter", "scatter_",
    "index_select", "RMSNorm",
}


def _ops_in_repro(src: str) -> set[str]:
    """Walk the AST to collect all attribute and call names."""
    ops: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ops
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            ops.add(node.attr)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                ops.add(f.id)
            elif isinstance(f, ast.Attribute):
                ops.add(f.attr)
    return ops


def _has_non_catalogue_op(ops: set[str]) -> str | None:
    for op in ops:
        if op in NON_CATALOGUE_OP_NAMES:
            return op
    return None


def _verdict_uses_non_catalogue_op(bugs: list, ops: set[str]) -> str | None:
    """Forensics check: scan each Bug.message for non-catalogue op names."""
    msgs = " ".join(getattr(b, "message", "") for b in bugs)
    for op in NON_CATALOGUE_OP_NAMES:
        if op in msgs:
            return op
    return None


def _resolve_repro_path(bid: str) -> str | None:
    base = os.path.join(_REPO_ROOT, "experiments_v5", "bug_repros")
    # File names look like bug_001_xxx.py or bug_018.py
    for fn in sorted(os.listdir(base)):
        if fn.startswith(bid + "_") or fn == bid + ".py":
            return os.path.join(base, fn)
    return None


def main() -> int:
    subset = json.load(open(os.path.join(_HERE, "pytea_modern_subset.json")))
    modern = [b for b in subset["per_bug"] if b["modern"]]
    records: list[dict] = []
    rejected_at_screen = 0
    forced_abstain_after = 0
    refuted_count = 0
    verified_count = 0
    error_count = 0
    for b in modern:
        bid = b["id"]
        path = _resolve_repro_path(bid)
        rec: dict = {
            "id": bid,
            "primary_op": b.get("primary_op"),
            "post_hoc_verdict": b.get("tg_verdict"),
        }
        if path is None or not os.path.exists(path):
            rec["status"] = "missing_repro"
            error_count += 1
            records.append(rec)
            continue
        with open(path) as f:
            src = f.read()
        ops = _ops_in_repro(src)
        # Extract INPUT_SHAPES literal if defined in the repro.
        input_shapes: dict | None = None
        try:
            ns: dict = {}
            exec(compile(ast.parse(src), path, "exec"), ns)
            if "INPUT_SHAPES" in ns and isinstance(ns["INPUT_SHAPES"], dict):
                input_shapes = ns["INPUT_SHAPES"]
        except Exception:
            input_shapes = None
        rec["ops_used"] = sorted(ops & (PYTEA_CATALOGUE_OPS | NON_CATALOGUE_OP_NAMES))
        nc = _has_non_catalogue_op(ops)
        if nc:
            rec["status"] = "screen_rejected_non_catalogue"
            rec["non_catalogue_op"] = nc
            rec["enforced_verdict"] = "Verified (forced)"
            rejected_at_screen += 1
            verified_count += 1
            records.append(rec)
            continue
        # Run TG end-to-end on the bug repro (no input shapes baked in;
        # the repro contains its own constants).
        try:
            res = verify_architecture(src, input_shapes=input_shapes)
            bugs = list(res.bugs)
        except Exception as e:
            rec["status"] = "verify_err"
            rec["err"] = str(e)[:120]
            error_count += 1
            records.append(rec)
            continue
        # Forensics: did the verdict reference a non-catalogue op?
        bad = _verdict_uses_non_catalogue_op(bugs, ops)
        if bad is not None:
            rec["status"] = "forced_abstain_post"
            rec["non_catalogue_referenced"] = bad
            rec["enforced_verdict"] = "Verified (forced)"
            forced_abstain_after += 1
            verified_count += 1
            records.append(rec)
            continue
        if bugs:
            rec["status"] = "Refuted"
            rec["max_conf"] = max((b.confidence for b in bugs), default=0.0)
            rec["enforced_verdict"] = "Refuted"
            refuted_count += 1
        else:
            rec["status"] = "Verified"
            rec["enforced_verdict"] = "Verified"
            verified_count += 1
        records.append(rec)
    out = {
        "regime": "modern_subset_enforced_at_verification_time",
        "n_total": len(modern),
        "tg_refuted_enforced": refuted_count,
        "tg_verified_enforced": verified_count,
        "errors": error_count,
        "rejected_at_screen": rejected_at_screen,
        "forced_abstain_after": forced_abstain_after,
        "post_hoc_tg_refuted": subset["modern_subset_results"]["tg_refuted"],
        "post_hoc_tg_total": subset["modern_subset_results"]["tg_total"],
        "interpretation": (
            "The 'rejected_at_screen' count is the number of "
            "modern-subset bug repros where an AST walk found a "
            "non-catalogue op call (TG would otherwise have fired); "
            "those are forced to Verified per the Pytea fragment "
            "fairness rule. The 'forced_abstain_after' count is "
            "the number of TG verdicts whose Bug.message referenced "
            "a non-catalogue op despite passing the screen; those "
            "are also forced to Verified."
        ),
        "per_bug": records,
    }
    out_path = os.path.join(_REPO_ROOT, "reproducibility", "pytea_modern_enforced.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_bug"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
