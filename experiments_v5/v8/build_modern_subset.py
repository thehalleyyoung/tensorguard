"""
Build the "modern subset" — bugs whose repro touches ONLY operators in Pytea's 2022 catalogue.
Catalogue: matmul/mm/bmm, conv2d/conv_transpose2d, linear, view/reshape, permute/transpose,
           broadcast/elementwise (tensor.__add__ etc.), cat/stack, batch_norm (2D only),
           embed (shape only), pool2d, layer_norm, flatten, unsqueeze/squeeze, expand, narrow,
           pad, reduce/topk.

NOT in catalogue: einsum, SDPA/attention2x, Conv1d/Conv3d, BatchNorm1d/GroupNorm/InstanceNorm,
  swapaxes/movedim/index_select, where, torch.add (functional), torch.maximum (functional),
  gather/scatter (no TS handler), torch.dot, linalg.*, repeat_interleave,
  MultiheadAttention (2.x), isclose, split-with-list-sum.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# Classification: modern=True if repro only touches catalogued ops
BUG_MODERN_MAP = {
    # id: (modern, primary_op, note)
    "bug_001": (False, "F.scaled_dot_product_attention", "SDPA — PyTorch-2.x, no Pytea handler"),
    "bug_002": (False, "torch.isclose",                  "isclose — not in Pytea catalogue"),
    "bug_003": (True,  "Tensor.view",                    "view — index.ts:1165"),
    "bug_004": (True,  "Tensor.view",                    "view with empty tensor — index.ts:1165"),
    "bug_005": (True,  "broadcast (param + tensor)",     "broadcast — index.ts:331"),
    "bug_006": (True,  "nn.CrossEntropyLoss",            "cross_entropy — index.ts:1912"),
    "bug_007": (True,  "F.conv2d (dtype check)",         "conv2d — index.ts:1338 (dtype mismatch surfaced by shape)"),
    "bug_008": (True,  "nn.Conv2d",                      "conv2d channel mismatch — index.ts:1338"),
    "bug_009": (True,  "nn.Linear",                      "linear in/out mismatch — via matmul chain"),
    "bug_010": (True,  "Tensor.view",                    "view total size — index.ts:1165"),
    "bug_011": (True,  "broadcast (a + b)",              "broadcast — index.ts:331"),
    "bug_012": (False, "nn.MultiheadAttention",          "MHA embed_dim divisibility — PyTorch-2.x redesign, no Pytea handler"),
    "bug_013": (False, "torch.einsum",                   "einsum — no Pytea handler"),
    "bug_014": (True,  "Tensor.transpose",               "transpose dim OOR — index.ts:923 has dim-range check"),
    "bug_015": (True,  "nn.BatchNorm2d",                 "batchnorm2d num_features — index.ts:1871"),
    "bug_016": (True,  "nn.Embedding",                   "embedding index OOB — handler exists, bounds check absent (symbolic_fragment)"),
    "bug_017": (False, "nn.Conv1d",                      "Conv1d — no Pytea handler (only Conv2d)"),
    "bug_018": (True,  "matmul (@ operator)",            "matmul mismatch — Pytea frontend_parse_failed on bare @ stmt"),
    "bug_019": (True,  "Tensor.reshape",                 "reshape total size — delegated to view handler"),
    "bug_020": (True,  "torch.cat",                      "cat shape mismatch — index.ts:1984"),
    "bug_022": (False, "torch.einsum",                   "einsum — no Pytea handler"),
    "bug_024": (False, "nn.BatchNorm1d",                 "BatchNorm1d — no Pytea handler (only BatchNorm2d)"),
    "bug_026": (False, "nn.Conv3d",                      "Conv3d — no Pytea handler"),
    "bug_027": (True,  "torch.bmm",                      "bmm — index.ts:518"),
    "bug_028": (True,  "Tensor.view (wildcard -1)",      "view with -1 wildcard — index.ts:1165"),
    "bug_029": (False, "torch.where",                    "where — no Pytea handler"),
    "bug_031": (False, "torch.einsum",                   "einsum — no Pytea handler"),
    "bug_032": (False, "torch.swapaxes",                 "swapaxes — no Pytea handler"),
    "bug_033": (False, "nn.GroupNorm",                   "GroupNorm — no Pytea handler"),
    "bug_034": (True,  "nn.Embedding",                   "embedding negative-index OOB — same symbolic_fragment weakness"),
    "bug_035": (True,  "nn.ConvTranspose2d",             "conv_transpose2d — index.ts:1502"),
    "bug_037": (True,  "flatten + view",                 "flatten+view total size — both handlers present"),
    "bug_038": (True,  "torch.stack",                    "stack shape mismatch — index.ts:2081"),
    "bug_039": (True,  "F.softmax",                      "softmax dim OOR — handler present, no dim-range check (symbolic_fragment)"),
    "bug_040": (True,  "matmul (@ operator, batched)",   "batched matmul mismatch — Pytea frontend_parse_failed on bare @ stmt"),
    "bug_041": (False, "torch.movedim",                  "movedim — no Pytea handler"),
    "bug_042": (True,  "nn.CrossEntropyLoss",            "cross_entropy target-length — index.ts:1912"),
    "bug_043": (True,  "nn.MSELoss",                     "mse_loss shape mismatch — cross_entropy handler path"),
    "bug_044": (True,  "nn.NLLLoss",                     "nll_loss class-index OOB — handler exists, bounds absent (symbolic_fragment)"),
    "bug_045": (True,  "nn.Conv2d",                      "conv2d kernel too large — index.ts:1338"),
    "bug_047": (True,  "Tensor.unsqueeze",               "unsqueeze dim OOR — index.ts:2175"),
    "bug_048": (False, "torch.add (functional)",         "torch.add functional — no Pytea handler; only tensor.__add__ handled"),
    "bug_049": (True,  "F.layer_norm",                   "layer_norm normalized_shape mismatch — index.ts:2561"),
    "bug_050": (False, "torch.einsum",                   "einsum — no Pytea handler"),
    "bug_051": (False, "torch.gather",                   "gather — no Pytea TS handler (index.ts grep empty)"),
    "bug_052": (False, "Tensor.scatter_",                "scatter_ — no Pytea TS handler"),
    "bug_053": (True,  "Tensor.expand",                  "expand incompatible dims — index.ts:738"),
    "bug_054": (False, "nn.InstanceNorm2d",              "InstanceNorm2d — stub has no num_features check; not in catalogue"),
    "bug_055": (False, "F.embedding (functional)",       "F.embedding — not in nn/functional.py stubs; unimpl"),
    "bug_056": (True,  "nn.Conv2d",                      "conv2d groups divisibility — index.ts:1338"),
    "bug_057": (True,  "torch.mm",                       "mm rank/shape — index.ts:470"),
    "bug_058": (False, "torch.split (list)",             "split with list — symbolic sum() not evaluated; unimpl"),
    "bug_059": (False, "torch.maximum (functional)",     "torch.maximum — no Pytea handler (only tensor ops)"),
    "bug_060": (False, "nn.Conv1d",                      "Conv1d — no Pytea handler"),
    "bug_063": (True,  "Tensor.view (post-transpose)",   "view after transpose — handler present, no contiguity check (symbolic_fragment)"),
    "bug_064": (False, "torch.index_select",             "index_select — no Pytea handler"),
    "bug_065": (False, "torch.dot",                      "dot — no Pytea handler"),
    "bug_067": (False, "torch.linalg.inv",               "linalg.inv — no Pytea handler; no linalg subpackage"),
    "bug_068": (True,  "nn.MaxPool2d",                   "pool2d kernel too large — index.ts:1719"),
    "bug_069": (False, "torch.repeat_interleave",        "repeat_interleave — no Pytea handler"),
}

# Load TG and Pytea verdicts
with open(os.path.join(ROOT, "v5_benchmark_results.json")) as f:
    tg_data = json.load(f)
tg_by_id = {e["id"]: e for e in tg_data["bug_corpus"]["per_input"]}

import re
with open(os.path.join(ROOT, "pytea_baseline_results.json")) as f:
    pytea_data = json.load(f)
pytea_by_short = {}
for e in pytea_data["bug_corpus"]["per_input"]:
    m = re.match(r"(bug_\d+)", e["id"])
    if m:
        pytea_by_short[m.group(1)] = e

# Build modern-subset rows
modern_ids = [bid for bid, (modern, op, note) in BUG_MODERN_MAP.items() if modern]
not_modern_ids = [bid for bid, (modern, op, note) in BUG_MODERN_MAP.items() if not modern]

tg_refuted_modern = 0
tg_refuted_not_modern = 0
pytea_refuted_modern = 0
pytea_refuted_not_modern = 0

modern_rows = []
for bid in sorted(BUG_MODERN_MAP.keys()):
    modern, op, note = BUG_MODERN_MAP[bid]
    tg_entry = tg_by_id.get(bid, {})
    pytea_entry = pytea_by_short.get(bid, {})
    tg_verdict = tg_entry.get("bucket", "MISSING")
    pytea_verdict = pytea_entry.get("verdict", "MISSING")

    row = {
        "id": bid,
        "modern": modern,
        "primary_op": op,
        "note": note,
        "tg_verdict": tg_verdict,
        "pytea_verdict": pytea_verdict,
    }
    modern_rows.append(row)

    if modern:
        if tg_verdict == "Refuted": tg_refuted_modern += 1
        if pytea_verdict == "Refuted": pytea_refuted_modern += 1
    else:
        if tg_verdict == "Refuted": tg_refuted_not_modern += 1
        if pytea_verdict == "Refuted": pytea_refuted_not_modern += 1

total_modern = len(modern_ids)
total_not_modern = len(not_modern_ids)
total = len(BUG_MODERN_MAP)

output = {
    "meta": {
        "generated_by": "v8/build_modern_subset.py",
        "pytea_catalogue_ops": [
            "matmul/mm/bmm", "conv2d/conv_transpose2d", "nn.Linear",
            "view/reshape", "transpose (2D+, with dim check)", "unsqueeze/squeeze",
            "cat/stack", "broadcast/elementwise (tensor.__add__ etc.)",
            "BatchNorm2d (rank-4 only)", "Embedding (shape, no index bounds)",
            "pool2d", "layer_norm", "flatten", "expand/expand_as",
            "narrow", "pad", "reduce/topk", "cross_entropy/nll_loss/mse_loss",
        ],
        "catalogue_gaps_in_corpus": [
            "einsum (5 bugs: 013,022,031,050 + misrouted 040)",
            "Conv1d/Conv3d (3 bugs: 017,026,060)",
            "BatchNorm1d/GroupNorm/InstanceNorm (3 bugs: 024,033,054)",
            "SDPA/MHA-2x (2 bugs: 001,012)",
            "swapaxes/movedim (2 bugs: 032,041)",
            "torch.where (1 bug: 029)",
            "torch.dot (1 bug: 065)",
            "torch.linalg.* (1 bug: 067)",
            "repeat_interleave (1 bug: 069)",
            "torch.add functional (1 bug: 048)",
            "torch.maximum functional (1 bug: 059)",
            "index_select (1 bug: 064)",
            "gather (1 bug: 051)",
            "scatter_ (1 bug: 052)",
            "split-with-list-sum (1 bug: 058)",
            "F.embedding functional (1 bug: 055)",
            "isclose (1 bug: 002)",
        ],
    },
    "corpus_partition": {
        "total_bugs": total,
        "modern_subset_size": total_modern,
        "not_modern_size": total_not_modern,
    },
    "modern_subset_results": {
        "tg_refuted": tg_refuted_modern,
        "tg_total": total_modern,
        "tg_detection_rate": round(tg_refuted_modern / total_modern, 3),
        "pytea_refuted": pytea_refuted_modern,
        "pytea_total": total_modern,
        "pytea_detection_rate": round(pytea_refuted_modern / total_modern, 3),
        "tg_advantage_within_modern": tg_refuted_modern - pytea_refuted_modern,
    },
    "not_modern_subset_results": {
        "tg_refuted": tg_refuted_not_modern,
        "tg_total": total_not_modern,
        "pytea_refuted": pytea_refuted_not_modern,
        "pytea_total": total_not_modern,
    },
    "full_corpus_results": {
        "tg_refuted": tg_refuted_modern + tg_refuted_not_modern,
        "pytea_refuted": pytea_refuted_modern + pytea_refuted_not_modern,
        "total": total,
    },
    "per_bug": modern_rows,
}

OUT = os.path.join(HERE, "pytea_modern_subset.json")
with open(OUT, "w") as f:
    json.dump(output, f, indent=2)

print("=== MODERN SUBSET ===")
print(f"  TG  refuted: {tg_refuted_modern}/{total_modern} = {tg_refuted_modern/total_modern:.1%}")
print(f"  Pytea refuted: {pytea_refuted_modern}/{total_modern} = {pytea_refuted_modern/total_modern:.1%}")
print(f"  TG advantage within modern: +{tg_refuted_modern - pytea_refuted_modern}")
print()
print("=== NOT-MODERN SUBSET (catalogue-gap bugs) ===")
print(f"  TG  refuted: {tg_refuted_not_modern}/{total_not_modern}")
print(f"  Pytea refuted: {pytea_refuted_not_modern}/{total_not_modern}")
print()
print("=== FULL CORPUS ===")
print(f"  TG  refuted: {tg_refuted_modern+tg_refuted_not_modern}/{total}")
print(f"  Pytea refuted: {pytea_refuted_modern+pytea_refuted_not_modern}/{total}")
print("Written to", OUT)
