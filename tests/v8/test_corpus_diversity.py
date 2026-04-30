"""
Regression test: re-runs corpus diversity clustering and asserts headline
numbers are reproducible.

Run with:
    cd /Users/halleyyoung/Documents/div/mathdivergence/halley-labs/tensorguard
    PYTHONPATH=. python3.11 -m pytest tests/v8/test_corpus_diversity.py -v
"""

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "experiments_v5" / "v5_block_corpus.jsonl"
RESULTS_PATH = REPO_ROOT / "experiments_v5" / "v8" / "corpus_diversity" / "cluster_analysis.json"

# ---------------------------------------------------------------------------
# Helpers (duplicated from run_cluster_analysis.py so the test is self-contained)
# ---------------------------------------------------------------------------

def get_forward_node(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "forward":
                return node
    return None


def canonical_op(call_node):
    func = call_node.func
    if isinstance(func, ast.Attribute):
        attr = func.attr
        obj = func.value
        prefix = obj.id if isinstance(obj, ast.Name) else (obj.attr if isinstance(obj, ast.Attribute) else "_")
        if prefix == "self":
            return f"self.{attr}"
        return f"{prefix}.{attr}"
    if isinstance(func, ast.Name):
        return func.id
    return None


def extract_handler_counter(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Counter()
    fwd = get_forward_node(tree)
    walk_target = fwd if fwd is not None else tree
    ctr = Counter()
    for node in ast.walk(walk_target):
        if isinstance(node, ast.Call):
            op = canonical_op(node)
            if op is not None:
                ctr[op] += 1
    return ctr


def jaccard_counter(a, b):
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return 1.0 if union == 0 else intersection / union


def greedy_cluster(items, similarity_fn, threshold=0.85):
    clusters = []
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


class SkeletonNormaliser(ast.NodeTransformer):
    def __init__(self):
        self._var_map = {}
        self._var_counter = 0

    def _var(self, name):
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


def skeleton_hash(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "PARSE_ERROR:" + hashlib.sha256(source.encode()).hexdigest()[:16]
    fwd = get_forward_node(tree)
    target = fwd if fwd is not None else tree
    normaliser = SkeletonNormaliser()
    norm_tree = normaliser.visit(ast.parse(ast.unparse(target)))
    dump = ast.dump(norm_tree, indent=None)
    return hashlib.sha256(dump.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    records = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if l.strip()]
    return records


@pytest.fixture(scope="module")
def saved_results():
    return json.loads(RESULTS_PATH.read_text())


def test_corpus_size(corpus):
    assert len(corpus) == 488


def test_k_handler_reproducible(corpus, saved_results):
    counters = [extract_handler_counter(r["source"]) for r in corpus]
    clusters = greedy_cluster(counters, jaccard_counter, threshold=0.85)
    K = len(clusters)
    expected = saved_results["handler_clustering"]["K_handler"]
    assert K == expected, f"K_handler changed: got {K}, expected {expected}"


def test_k_ast_reproducible(corpus, saved_results):
    hashes = [skeleton_hash(r["source"]) for r in corpus]
    hash_groups = defaultdict(list)
    for i, h in enumerate(hashes):
        hash_groups[h].append(i)
    K = len(hash_groups)
    expected = saved_results["ast_skeleton_clustering"]["K_ast"]
    assert K == expected, f"K_ast changed: got {K}, expected {expected}"


def test_k_effective_substantially_exceeds_reviewer_claim(saved_results):
    """The reviewer claimed 50–80 effective challenges; assert the real number is higher."""
    K_eff = saved_results["combined"]["K_effective"]
    assert K_eff > 80, (
        f"K_effective={K_eff} is within the reviewer's 50–80 claim; "
        "re-evaluate the diversity assessment."
    )


def test_output_files_exist():
    base = REPO_ROOT / "experiments_v5" / "v8" / "corpus_diversity"
    assert (base / "cluster_analysis.json").exists()
    assert (base / "SUMMARY.md").exists()
    assert (base / "representatives.txt").exists()
