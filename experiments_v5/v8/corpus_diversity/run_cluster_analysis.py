"""
Corpus diversity clustering analysis for the 488-block tensorguard corpus.

Two orthogonal analyses:
  1. Handler-call multiset clustering (Jaccard, greedy at 0.85 threshold)
  2. AST-skeleton clustering (SHA-256 of normalised forward body)
"""

import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "experiments_v5" / "v5_block_corpus.jsonl"
OUT_DIR = REPO_ROOT / "experiments_v5" / "v8" / "corpus_diversity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------
records = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if l.strip()]
assert len(records) == 488, f"Expected 488, got {len(records)}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_forward_node(tree: ast.Module):
    """Return the `forward` FunctionDef node if present, else None."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "forward":
                return node
    return None


def canonical_op(call_node: ast.Call) -> str | None:
    """
    Return a canonical string like 'nn.Linear', 'F.softmax', 'torch.cat',
    'x.view', 'x.permute', etc.  Returns None for unrecognised call shapes.
    """
    func = call_node.func
    if isinstance(func, ast.Attribute):
        attr = func.attr
        obj = func.value
        if isinstance(obj, ast.Name):
            prefix = obj.id
        elif isinstance(obj, ast.Attribute):
            # e.g. torch.nn.functional.relu  → use last two parts
            prefix = obj.attr
        else:
            prefix = "_"
        # Normalise self.whatever(x) → SELF_CALL
        if prefix == "self":
            return f"self.{attr}"
        return f"{prefix}.{attr}"
    if isinstance(func, ast.Name):
        return func.id
    return None


def extract_handler_counter(source: str) -> Counter:
    """Parse source, walk forward body, return Counter of canonical op calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Counter()

    fwd = get_forward_node(tree)
    if fwd is None:
        # Fall back: walk everything
        walk_target = tree
    else:
        walk_target = fwd

    ctr: Counter = Counter()
    for node in ast.walk(walk_target):
        if isinstance(node, ast.Call):
            op = canonical_op(node)
            if op is not None:
                ctr[op] += 1
    return ctr


def jaccard_counter(a: Counter, b: Counter) -> float:
    """Jaccard similarity of two Counters (bag / multiset version)."""
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    if union == 0:
        return 1.0
    return intersection / union


def greedy_cluster(items, similarity_fn, threshold=0.85):
    """
    O(n^2) greedy clustering.  Each new item joins the first cluster whose
    representative has similarity >= threshold, otherwise starts a new cluster.
    Returns list of (representative_idx, [member_idxs]).
    """
    clusters: list[tuple[int, list[int]]] = []  # (rep_idx, members)
    for i, item in enumerate(items):
        assigned = False
        for rep_idx, members in clusters:
            if similarity_fn(item, items[rep_idx]) >= threshold:
                members.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append((i, [i]))
    return clusters


# ---------------------------------------------------------------------------
# Analysis 1: Handler-call multiset clustering
# ---------------------------------------------------------------------------
print("Computing handler-call multisets …", flush=True)
handler_counters = [extract_handler_counter(r["source"]) for r in records]

print("Greedy clustering at Jaccard ≥ 0.85 …", flush=True)
handler_clusters = greedy_cluster(
    handler_counters,
    similarity_fn=jaccard_counter,
    threshold=0.85,
)

K_handler = len(handler_clusters)
handler_sizes = sorted([len(m) for _, m in handler_clusters], reverse=True)
print(f"  K_handler = {K_handler}")

# Top-10 cluster reps
handler_top10 = [
    {
        "rank": rank + 1,
        "representative_id": records[rep_idx]["id"],
        "size": len(members),
        "top_ops": handler_counters[rep_idx].most_common(8),
    }
    for rank, (rep_idx, members) in enumerate(
        sorted(handler_clusters, key=lambda x: len(x[1]), reverse=True)[:10]
    )
]

# ---------------------------------------------------------------------------
# Analysis 2: AST-skeleton clustering
# ---------------------------------------------------------------------------
print("Computing AST skeleton hashes …", flush=True)


class SkeletonNormaliser(ast.NodeTransformer):
    """
    Replaces:
      - all integer/float constants with _INT_ / _FLOAT_
      - all string constants with _STR_
      - all Name nodes (variables) with positional _vN_ placeholders
      - all Attribute .attr parts with _ATTR_
    """

    def __init__(self):
        self._var_map: dict[str, str] = {}
        self._var_counter = 0

    def _var(self, name: str) -> str:
        if name not in self._var_map:
            self._var_counter += 1
            self._var_map[name] = f"_v{self._var_counter}_"
        return self._var_map[name]

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return ast.Constant(value="_INT_")
        if isinstance(node.value, float):
            return ast.Constant(value="_FLOAT_")
        if isinstance(node.value, str):
            return ast.Constant(value="_STR_")
        return node

    def visit_Name(self, node):
        return ast.Name(id=self._var(node.id), ctx=node.ctx)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = "_ATTR_"
        return node


def skeleton_hash(source: str) -> str:
    """Return SHA-256 of the normalised forward-body AST dump."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "PARSE_ERROR:" + hashlib.sha256(source.encode()).hexdigest()[:16]

    fwd = get_forward_node(tree)
    if fwd is None:
        target = tree
    else:
        target = fwd

    normaliser = SkeletonNormaliser()
    norm_tree = normaliser.visit(ast.parse(ast.unparse(target)))  # re-parse for clean AST
    dump = ast.dump(norm_tree, indent=None)
    return hashlib.sha256(dump.encode()).hexdigest()


hashes = [skeleton_hash(r["source"]) for r in records]

# Group by hash
hash_to_indices: dict[str, list[int]] = defaultdict(list)
for i, h in enumerate(hashes):
    hash_to_indices[h].append(i)

K_ast = len(hash_to_indices)
ast_sizes = sorted([len(v) for v in hash_to_indices.values()], reverse=True)
print(f"  K_ast = {K_ast}")

# ---------------------------------------------------------------------------
# Combined result
# ---------------------------------------------------------------------------
K_effective = max(K_handler, K_ast)
print(f"  K_effective = {K_effective}")

# ---------------------------------------------------------------------------
# Build size histograms (bucket counts)
# ---------------------------------------------------------------------------

def size_histogram(sizes: list[int]) -> dict[str, int]:
    buckets = {"1": 0, "2-3": 0, "4-7": 0, "8-15": 0, "16-31": 0, "32+": 0}
    for s in sizes:
        if s == 1:
            buckets["1"] += 1
        elif s <= 3:
            buckets["2-3"] += 1
        elif s <= 7:
            buckets["4-7"] += 1
        elif s <= 15:
            buckets["8-15"] += 1
        elif s <= 31:
            buckets["16-31"] += 1
        else:
            buckets["32+"] += 1
    return buckets


# ---------------------------------------------------------------------------
# representatives.txt: one block-id per cluster (top 50 by size)
# ---------------------------------------------------------------------------
all_clusters_sorted = sorted(handler_clusters, key=lambda x: len(x[1]), reverse=True)
rep_ids = [records[rep_idx]["id"] for rep_idx, _ in all_clusters_sorted[:50]]

# ---------------------------------------------------------------------------
# Serialize outputs
# ---------------------------------------------------------------------------

result = {
    "corpus_size": len(records),
    "handler_clustering": {
        "threshold": 0.85,
        "K_handler": K_handler,
        "cluster_sizes": handler_sizes,
        "size_histogram": size_histogram(handler_sizes),
        "top10_clusters": handler_top10,
    },
    "ast_skeleton_clustering": {
        "K_ast": K_ast,
        "cluster_sizes": ast_sizes,
        "size_histogram": size_histogram(ast_sizes),
    },
    "combined": {
        "K_effective": K_effective,
        "reviewer_claim_50_80": K_effective <= 80,
        "assessment": (
            f"K_effective = {K_effective}. "
            + (
                "This is within the reviewer's claimed 50–80 range, confirming limited shape-reasoning diversity."
                if K_effective <= 80
                else (
                    f"This substantially exceeds the reviewer's claim of 50–80 distinct challenges "
                    f"(factor {K_effective/80:.1f}× the upper bound)."
                    if K_effective > 120
                    else f"This is modestly above the reviewer's 50–80 claim but still indicates limited diversity."
                )
            )
        ),
    },
}

(OUT_DIR / "cluster_analysis.json").write_text(json.dumps(result, indent=2))
print(f"Wrote cluster_analysis.json")

(OUT_DIR / "representatives.txt").write_text("\n".join(rep_ids) + "\n")
print(f"Wrote representatives.txt  ({len(rep_ids)} entries)")

# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------

def hist_table(hist: dict[str, int]) -> str:
    lines = ["| Cluster size | Count |", "|---|---|"]
    for k, v in hist.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


summary_md = f"""# Corpus Diversity Analysis — 488-block corpus

## Headline Numbers

| Metric | Value |
|---|---|
| Total blocks | 488 |
| K_handler (op-call Jaccard ≥ 0.85) | **{K_handler}** |
| K_ast (skeleton SHA-256) | **{K_ast}** |
| **K_effective** | **{K_effective}** |
| Reviewer's claimed range | 50–80 |

## Assessment

{result['combined']['assessment']}

---

## Analysis 1 — Handler-Call Multiset Clustering

Greedy Jaccard clustering at threshold 0.85.  Each block's feature vector is
the multiset of canonical operator calls (e.g. `nn.Linear`, `F.softmax`,
`x.view`) extracted by walking the `forward` body AST.

### Cluster-size histogram

{hist_table(result['handler_clustering']['size_histogram'])}

### Top-10 clusters (by size)

| Rank | Representative block | Cluster size | Top ops |
|---|---|---|---|
""" + "\n".join(
    f"| {t['rank']} | `{t['representative_id']}` | {t['size']} | "
    + ", ".join(f"`{op}`×{cnt}" for op, cnt in t["top_ops"][:5])
    + " |"
    for t in handler_top10
) + f"""

---

## Analysis 2 — AST-Skeleton Clustering

`forward` body normalised: all integer/float literals → `_INT_`, all attribute
names → `_ATTR_`, all variable names → positional `_vN_` placeholders.
SHA-256 of the normalised `ast.dump` string groups syntactically identical
skeletons.

### Cluster-size histogram

{hist_table(result['ast_skeleton_clustering']['size_histogram'])}

*Unique skeletons (singletons): {result['ast_skeleton_clustering']['size_histogram']['1']}*

---

*Generated by `experiments_v5/v8/corpus_diversity/run_cluster_analysis.py`*
"""

(OUT_DIR / "SUMMARY.md").write_text(summary_md)
print(f"Wrote SUMMARY.md")

print("\n=== DONE ===")
print(f"  K_handler  = {K_handler}")
print(f"  K_ast      = {K_ast}")
print(f"  K_effective = {K_effective}")
print(f"  Reviewer's claim (50–80): {'CONFIRMED' if K_effective <= 80 else 'REFUTED'}")
