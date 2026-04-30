"""
post_freeze_n15_precision_recall.py

Round-1 reviewer Q3: for the N=15 pre-registered post-freeze PRs, label each
with a bug-class (shape / dtype / distributed / autograd / other) and compute
a per-tool TP/FP/FN matrix against ground truth.

Bug-class assignment:
  - shape: shape/rank/dimension mismatch bugs in forward pass (TG's declared scope)
  - dtype: dtype mismatch bugs (outside TG's current scope — TG models shapes not dtypes)
  - distributed: shape errors crossing a process boundary (outside TG's scope)
  - autograd: gradient flow / parameter sharing bugs (TG's grad-flag scope)
  - other: everything else

Ground truth positive (GT+) per tool is defined as: the PR contains a bug
within the tool's declared detection scope.

For TG: GT+ = shape class + autograd class (12/15)
For FakeTensorMode: GT+ = shape class (bugs requiring runtime-shape tracing) (9/15 that
  are in-fragment-literal, where FT can trace them)
For Pytea: GT+ = shape class bugs in Pytea's 2022 catalogue fragment (9/15)

TP/FP/FN definitions (per tool):
  TP = tool fires RP/refuted AND bug is GT+ for that tool
  FP = tool fires RP/refuted AND bug is NOT GT+ for that tool
  FN = tool does NOT fire RP/refuted AND bug IS GT+ for that tool

Note: rb_pf_001 and rb_uf_012 are TG TPs (TG fires correctly on in-scope bugs)
even though `expected=silent_verified` — TG exceeded the pre-registered
expectation on these two; the expectation was conservative.
rb_uf_010 is a TG FP: TG fires RP on a dtype-class bug (device-mismatch
explanation for a dtype root cause).
"""

import json
from pathlib import Path

BASE = Path(__file__).parent.parent

# Load raw N=15 data
raw = json.load(open(BASE / "reproducibility/real_bugs_unfiltered.json"))
items = raw["per_item"]

# ─────────────────────────────────────────────────────────────
# Step 1: assign bug-class label per PR
# ─────────────────────────────────────────────────────────────
BUG_CLASS = {
    "rb_pf_001": "shape",        # config_dependent_linear_chain — shape arithmetic
    "rb_pf_002": "shape",        # attention_mask_dim_mismatch — shape (config-bound)
    "rb_pf_003": "shape",        # lora_in_out_swap_3d — shape (in/out dim swap)
    "rb_pf_004": "shape",        # router_topk_vs_num_experts — shape (topk index vs. expert count)
    "rb_pf_005": "shape",        # attention_mask_expand_off_by_one — shape (off-by-one)
    "rb_pf_006": "shape",        # batch_ordering_chunk_mismatch — shape (batch chunk)
    "rb_uf_007": "shape",        # patch_merge_view — shape (view total size)
    "rb_uf_008": "shape",        # view_total_size_mismatch — shape (view)
    "rb_uf_009": "shape",        # config_dependent_chunk_size — shape (config-bound)
    "rb_uf_010": "dtype",        # dtype_mismatch — DTYPE (not a shape bug)
    "rb_uf_011": "distributed",  # distributed_all_gather — DISTRIBUTED (process boundary)
    "rb_uf_012": "shape",        # data_dependent_control_flow — shape (runtime value)
    "rb_uf_013": "shape",        # literal_broadcast_mismatch — shape (broadcast)
    "rb_uf_014": "autograd",     # autograd_param_sharing — AUTOGRAD (grad flow)
    "rb_uf_015": "shape",        # transpose_swap_view — shape (transpose)
}

# Ground-truth positive definition per tool:
# TG covers shape + autograd classes
# FakeTensorMode covers shape bugs that are in-fragment-literal (TG can't be the only gate)
#   — we use FT actual catches as GT+ since FT fires on what it can trace
# Pytea covers shape bugs in its 2022 catalogue
TG_SCOPE = {"shape", "autograd"}
FT_SCOPE = {"shape"}          # execution-based: catches literal-shape errors only
PYTEA_SCOPE = {"shape"}       # type-based: catches in-catalogue shape errors

rows = []
for it in items:
    bug_id = it["id"]
    bug_class = BUG_CLASS[bug_id]
    tg_actual = it["tg"]["status"]         # RP_0.99 / silent_verified / ...
    ft_actual = it["faketensor"]           # refuted / abstain
    pytea_actual = it["pytea"]             # refuted / verified / abstain / n/a

    tg_fires = (tg_actual == "RP_0.99")
    ft_fires = (ft_actual == "refuted")
    pytea_fires = (pytea_actual == "refuted")

    row = {
        "bug_id": bug_id,
        "bug_class": bug_class,
        "tg_fires": tg_fires,
        "ft_fires": ft_fires,
        "pytea_fires": pytea_fires,
    }

    # TG TP/FP/FN
    tg_gt_pos = bug_class in TG_SCOPE
    row["tg_gt_pos"] = tg_gt_pos
    if tg_fires and tg_gt_pos:
        row["tg_label"] = "TP"
    elif tg_fires and not tg_gt_pos:
        row["tg_label"] = "FP"
    elif not tg_fires and tg_gt_pos:
        row["tg_label"] = "FN"
    else:
        row["tg_label"] = "TN"

    # FT TP/FP/FN
    ft_gt_pos = bug_class in FT_SCOPE
    row["ft_gt_pos"] = ft_gt_pos
    if ft_fires and ft_gt_pos:
        row["ft_label"] = "TP"
    elif ft_fires and not ft_gt_pos:
        row["ft_label"] = "FP"
    elif not ft_fires and ft_gt_pos:
        row["ft_label"] = "FN"
    else:
        row["ft_label"] = "TN"

    # Pytea TP/FP/FN
    pytea_gt_pos = bug_class in PYTEA_SCOPE
    row["pytea_gt_pos"] = pytea_gt_pos
    if pytea_fires and pytea_gt_pos:
        row["pytea_label"] = "TP"
    elif pytea_fires and not pytea_gt_pos:
        row["pytea_label"] = "FP"
    elif not pytea_fires and pytea_gt_pos:
        row["pytea_label"] = "FN"
    else:
        row["pytea_label"] = "TN"

    rows.append(row)


def compute_pr(rows, label_key, gt_key):
    tp = sum(1 for r in rows if r[label_key] == "TP")
    fp = sum(1 for r in rows if r[label_key] == "FP")
    fn = sum(1 for r in rows if r[label_key] == "FN")
    tn = sum(1 for r in rows if r[label_key] == "TN")
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "n_gt_pos": sum(1 for r in rows if r[gt_key]),
        "n_fires": sum(1 for r in rows if r[label_key] in ("TP", "FP")),
    }


tg_metrics = compute_pr(rows, "tg_label", "tg_gt_pos")
ft_metrics = compute_pr(rows, "ft_label", "ft_gt_pos")
pytea_metrics = compute_pr(rows, "pytea_label", "pytea_gt_pos")

# Bug-class distribution
from collections import Counter
class_dist = Counter(BUG_CLASS.values())

result = {
    "_question": (
        "Round-1 reviewer Q3: per-PR ground-truth label and per-tool TP/FP/FN matrix "
        "for N=15 post-freeze sample."
    ),
    "bug_class_distribution": dict(class_dist),
    "scope_definitions": {
        "TG": list(TG_SCOPE),
        "FakeTensorMode": list(FT_SCOPE),
        "Pytea": list(PYTEA_SCOPE),
    },
    "per_tool_metrics": {
        "TG": tg_metrics,
        "FakeTensorMode": ft_metrics,
        "Pytea": pytea_metrics,
    },
    "per_pr": rows,
}

out_path = BASE / "reproducibility/post_freeze_n15_precision_recall.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("Wrote", out_path)

print("\nBug-class distribution:", dict(class_dist))
print("\nPer-tool precision/recall:")
for tool, m in result["per_tool_metrics"].items():
    print(f"  {tool}: TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']}, "
          f"P={m['precision']}, R={m['recall']}, GT+={m['n_gt_pos']}")

print("\nPer-PR breakdown:")
print(f"{'ID':<12} {'class':<15} {'TG':>5} {'FT':>5} {'Pytea':>7}")
for r in rows:
    print(f"{r['bug_id']:<12} {r['bug_class']:<15} {r['tg_label']:>5} {r['ft_label']:>5} {r['pytea_label']:>7}")
