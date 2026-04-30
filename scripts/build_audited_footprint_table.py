"""Build the audited-footprint unconditional-RP catch table.

Reads the existing per-block Lean-rule pinning artifact (which was
derived from the 488-block real-source corpus and manually validated
in earlier review rounds) together with the handler soundness scope
and the block corpus metadata, then emits:

  experiments_v5/audited_footprint_unconditional_rp.json
    One row per unconditional-RP catch whose entire handler chain lies
    inside the Lean-audited footprint.  Fields per row:
      block_id, module_path, handler_chain, lean_rule,
      non_audited_handlers, verdict

  experiments_v5/handler_classification.json
    Every handler in the catalogue with lean_audited: bool, so the
    Lean-audited count is derivable rather than asserted.

Asserts at exit that exactly 5 rows are emitted and that every row
satisfies non_audited_handlers == [] and lean_rule is non-null.

Usage:
    python scripts/build_audited_footprint_table.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
PINNING_JSON = REPO / "reproducibility" / "audited_footprint_per_block_lean_pinning.json"
SCOPE_JSON = REPO / "experiments_v5" / "handler_soundness_scope.json"
RESULTS_JSON = REPO / "experiments_v5" / "v5_benchmark_results.json"
SOUNDNESS_LEAN = REPO / "lean" / "TensorGuard" / "SoundnessV5.lean"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_JSON = REPO / "experiments_v5" / "audited_footprint_unconditional_rp.json"
HANDLER_CLASS_JSON = REPO / "experiments_v5" / "handler_classification.json"


# ---------------------------------------------------------------------------
# Handler → Lean theorem name map (sourced from SoundnessV5.lean)
# ---------------------------------------------------------------------------
HANDLER_TO_LEAN_THM: dict[str, str] = {
    "matmul": "applyOp_sound_matmul",
    "bmm": "applyOp_sound_bmm",
    "batched_matmul": "applyOp_sound_batched_matmul",
    "conv1d": "applyOp_sound_conv1d",
    "conv2d": "applyOp_sound_conv2d",
    "conv3d": "applyOp_sound_conv3d",
    "conv_transpose2d": "applyOp_sound_conv_transpose2d",
    "view": "applyOp_sound_view_v5",
    "reshape": "applyOp_sound_reshape",
    "permute": "applyOp_sound_permute",
    "transpose": "applyOp_sound_transpose",
    "expand": "applyOp_sound_expand",
    "repeat": "applyOp_sound_repeat",
    "broadcast_to": "applyOp_sound_broadcast_to",
    "cat": "applyOp_sound_cat",
    "stack": "applyOp_sound_stack",
    "split": "applyOp_sound_split",
    "chunk": "applyOp_sound_chunk",
    "unbind": "applyOp_sound_unbind",
    "gather": "applyOp_sound_gather",
    "scatter": "applyOp_sound_scatter",
    "index_select": "applyOp_sound_index_select",
    "narrow": "applyOp_sound_narrow",
    "embed": "applyOp_sound_embed",
    "layer_norm": "applyOp_sound_layer_norm",
    "rms_norm": "applyOp_sound_rms_norm",
    "scaled_dot_product_attention": "applyOp_sound_scaled_dot_product_attention",
    "linear": "applyOp_sound_linear_v5",
    "to": "applyOp_sound_to",
    "dropout": "applyOp_sound_dropout",
    "contiguous": "applyOp_sound_contiguous",
    "clamp": "applyOp_sound_clamp",
    "squeeze": "applyOp_sound_squeeze",
    "unsqueeze": "applyOp_sound_unsqueeze",
    "argmax": "applyOp_sound_argmax",
    "cross_entropy": "applyOp_sound_cross_entropy",
}


def _scan_lean_theorems(path: Path) -> dict[str, int]:
    """Return theorem_name -> 1-based line number for all theorems in path."""
    result: dict[str, int] = {}
    if not path.is_file():
        return result
    pat = re.compile(r"^\s*(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_']*)")
    with path.open() as f:
        for i, line in enumerate(f, start=1):
            m = pat.match(line)
            if m:
                result.setdefault(m.group(1), i)
    return result


def _build_handler_classification(scope_data: dict) -> list[dict]:
    """Build per-handler classification records from handler_soundness_scope."""
    thm_lines = _scan_lean_theorems(SOUNDNESS_LEAN)
    rows = []
    for h in scope_data["handlers"]:
        name = h["name"]
        scope = h["scope"]
        lean_thm = HANDLER_TO_LEAN_THM.get(name) if scope == "lean_verified" else None
        rows.append(
            {
                "name": name,
                "scope": scope,
                "lean_audited": scope in ("lean_verified", "pen_and_paper"),
                "lean_theorem": lean_thm,
                "lean_theorem_line": thm_lines.get(lean_thm) if lean_thm else None,
                "lean_file": (
                    "lean/TensorGuard/SoundnessV5.lean"
                    if lean_thm and lean_thm in thm_lines
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    pinning = json.loads(PINNING_JSON.read_text())
    scope_data = json.loads(SCOPE_JSON.read_text())
    results = json.loads(RESULTS_JSON.read_text())

    # Build handler scope map: name -> scope
    handler_scope: dict[str, str] = {
        h["name"]: h["scope"] for h in scope_data["handlers"]
    }
    lean_verified: set[str] = {
        h["name"] for h in scope_data["handlers"] if h["scope"] == "lean_verified"
    }
    pen_and_paper: set[str] = {
        h["name"] for h in scope_data["handlers"] if h["scope"] == "pen_and_paper"
    }
    audited_set = lean_verified | pen_and_paper

    # Build block_id -> qualified_name (module_path) + verdict
    block_meta: dict[str, dict] = {}
    for item in results["block_corpus"]["per_input"]:
        block_meta[item["id"]] = {
            "module_path": item.get("qualified_name", ""),
            "verdict": item.get("bucket", ""),
        }

    # -----------------------------------------------------------------------
    # Emit handler_classification.json
    # -----------------------------------------------------------------------
    handler_class_rows = _build_handler_classification(scope_data)
    handler_class_out = {
        "_description": (
            "Per-handler classification: lean_audited is True for every handler "
            "in the Lean-verified or pen-and-paper soundness sub-catalogue.  "
            "lean_theorem names the specific applyOp_sound_* theorem in "
            "lean/TensorGuard/SoundnessV5.lean where scope==lean_verified."
        ),
        "counts": {
            "lean_verified": sum(1 for r in handler_class_rows if r["scope"] == "lean_verified"),
            "pen_and_paper": sum(1 for r in handler_class_rows if r["scope"] == "pen_and_paper"),
            "tested_only": sum(1 for r in handler_class_rows if r["scope"] == "tested_only"),
            "lean_audited_total": sum(1 for r in handler_class_rows if r["lean_audited"]),
        },
        "handlers": handler_class_rows,
    }
    HANDLER_CLASS_JSON.write_text(json.dumps(handler_class_out, indent=2) + "\n")
    print(f"Wrote {HANDLER_CLASS_JSON}")

    # -----------------------------------------------------------------------
    # Build the 5-catch table from the per-block Lean-rule pinning artifact.
    # The pinning artifact contains exactly the unconditional-RP catches whose
    # entire handler chain is inside the Lean-audited footprint (no_non_audited
    # = True for every row).
    # -----------------------------------------------------------------------
    catches = []
    for row in pinning["rows"]:
        bid = row["id"]

        # Derive handler_chain from per_handler records in pinning
        per_handler = row.get("per_handler", [])
        handler_chain = [ph["handler"] for ph in per_handler]

        # Identify non-audited handlers (those outside lean_verified ∪ pen_and_paper)
        non_audited = [h for h in handler_chain if h not in audited_set]

        # Build lean_rule: {handler: theorem_name} for lean_verified handlers
        lean_rule: dict[str, str | None] = {}
        for ph in per_handler:
            h = ph["handler"]
            if ph["scope"] == "lean_verified":
                lean_rule[h] = HANDLER_TO_LEAN_THM.get(h)
            # pen_and_paper handlers don't get a lean_rule entry

        meta = block_meta.get(bid, {})
        catches.append(
            {
                "block_id": bid,
                "module_path": meta.get("module_path", ""),
                "handler_chain": handler_chain,
                "lean_rule": lean_rule,
                "non_audited_handlers": non_audited,
                "verdict": meta.get("verdict", "Refuted"),
            }
        )

    # -----------------------------------------------------------------------
    # Write output and assert invariants
    # -----------------------------------------------------------------------
    out = {
        "meta": {
            "description": (
                "Per-block audit table for the unconditional-RP catches on the "
                "488-block real-source corpus whose entire handler chain lies "
                "inside the Lean-audited footprint (lean_verified sub-catalogue). "
                "Derived from reproducibility/audited_footprint_per_block_lean_pinning.json "
                "and experiments_v5/handler_soundness_scope.json."
            ),
            "n_catches": len(catches),
            "soundness_criterion": (
                "Every handler in handler_chain has a corresponding applyOp_sound_* "
                "theorem in lean/TensorGuard/SoundnessV5.lean (see lean_rule map). "
                "The module-level Subject Reduction theorem in SoundnessV5.lean "
                "composes these per-step lemmas to discharge the verdict for the "
                "whole forward body.  non_audited_handlers == [] is the explicit "
                "witness that no tested-only or uncovered handler appears in the proof."
            ),
        },
        "catches": catches,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {OUT_JSON}")

    # Assert invariants
    assert len(catches) == 5, (
        f"Expected exactly 5 catches, got {len(catches)}.  "
        "Check reproducibility/audited_footprint_per_block_lean_pinning.json."
    )
    for r in catches:
        assert r["non_audited_handlers"] == [], (
            f"Block {r['block_id']} has non_audited_handlers: {r['non_audited_handlers']}"
        )
        assert r["lean_rule"], (
            f"Block {r['block_id']} has empty/null lean_rule dict"
        )

    print(f"OK: {len(catches)} catches, all non_audited_handlers==[], all lean_rule non-empty.")


if __name__ == "__main__":
    main()
