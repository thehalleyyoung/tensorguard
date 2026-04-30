#!/usr/bin/env python3
"""AST-based audit of tied/renamed-attribute parameter sharing in the 488-block corpus.

Scans every nn.Module block in experiments_v5/v5_block_corpus.jsonl for
patterns where two self-attributes share the same underlying parameter tensor,
via AST-level detection (not regex). Emits a Wilson-CI prevalence estimate and
a worst-case false-Verified deployment bound.

Detection patterns (block-level, all methods):
  R1  self.X = self.Y.weight
  R2  self.X = self.Y.bias
  R3  self.X.weight = self.Y.weight  (in-place weight rebind)
  R4  self.X = nn.Parameter(self.Y.weight ...)
  R5  self.X = self.Y  (direct self-attribute alias, Y assigned nn module)
  R6  setattr(self, name, self.Y)   where Y is a known nn attribute
  R7  setattr(self, name, self.Y.weight)

Output:
  experiments_v5/ast_tied_param_prevalence.json
Stdout:
  PREVALENCE_AUDIT prevalence=<x> wilson=[<lo>,<hi>] bound=<b>
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS_PATH = ROOT / "experiments_v5" / "v5_block_corpus.jsonl"
OUTPUT_PATH = ROOT / "experiments_v5" / "ast_tied_param_prevalence.json"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for k successes out of n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _is_self_attr(node: ast.AST) -> bool:
    """True iff node is `self.X`."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_self_dot_weight_or_bias(node: ast.AST) -> bool:
    """True iff node is `self.X.weight` or `self.X.bias`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in ("weight", "bias")
        and _is_self_attr(node.value)
    )


def _nn_attr_name(node: ast.Attribute) -> str:
    """Extract Y from self.Y or self.Y.weight — the first attribute name."""
    if isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    if isinstance(node.value, ast.Attribute) and _is_self_attr(node.value):
        return node.value.attr  # type: ignore[union-attr]
    return ""


def _is_nn_module_or_param_call(node: ast.AST) -> bool:
    """True iff node looks like nn.Something(...) or Parameter(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # nn.Linear(...), nn.Parameter(...), etc.
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in ("nn", "torch_nn")
    # Parameter(...) if imported directly
    if isinstance(func, ast.Name) and func.id in (
        "Parameter",
        "Linear",
        "Embedding",
        "Conv2d",
        "Conv1d",
        "LSTM",
        "GRU",
    ):
        return True
    return False


def _collect_method_stmts(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return all statements reachable in the function (via ast.walk)."""
    stmts: list[ast.stmt] = []
    for node in ast.walk(func_node):
        if node is func_node:
            continue
        if isinstance(node, ast.stmt):
            stmts.append(node)
    return stmts


def detect_tied_params(source: str) -> bool:
    """Return True if source contains any tied/renamed-attribute parameter sharing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        stmts = _collect_method_stmts(func_node)

        # Pass 1: collect self-attributes assigned to nn modules / parameters.
        param_attrs: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if _is_self_attr(tgt) and isinstance(tgt, ast.Attribute):
                        if _is_nn_module_or_param_call(stmt.value):
                            param_attrs.add(tgt.attr)
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if _is_self_attr(stmt.target) and isinstance(stmt.target, ast.Attribute):
                    if _is_nn_module_or_param_call(stmt.value):
                        param_attrs.add(stmt.target.attr)

        if not param_attrs:
            continue

        # Pass 2: detect any aliasing or weight-extraction.
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    val = stmt.value

                    # R1/R2: self.X = self.Y.weight | .bias
                    if _is_self_attr(tgt) and _is_self_dot_weight_or_bias(val):
                        assert isinstance(val, ast.Attribute)
                        src_mod = _nn_attr_name(val)
                        if src_mod and src_mod in param_attrs:
                            return True

                    # R3: self.X.weight = self.Y.weight (in-place rebind)
                    if (
                        isinstance(tgt, ast.Attribute)
                        and tgt.attr in ("weight", "bias")
                        and _is_self_attr(tgt.value)
                        and _is_self_dot_weight_or_bias(val)
                    ):
                        assert isinstance(val, ast.Attribute)
                        src_mod = _nn_attr_name(val)
                        if src_mod and src_mod in param_attrs:
                            return True

                    # R4: self.X = nn.Parameter(self.Y.weight ...)
                    if (
                        _is_self_attr(tgt)
                        and isinstance(val, ast.Call)
                    ):
                        func = val.func
                        is_param = (
                            isinstance(func, ast.Attribute) and func.attr == "Parameter"
                        ) or (isinstance(func, ast.Name) and func.id == "Parameter")
                        if is_param:
                            for arg in val.args:
                                if _is_self_dot_weight_or_bias(arg):
                                    assert isinstance(arg, ast.Attribute)
                                    src_mod = _nn_attr_name(arg)
                                    if src_mod and src_mod in param_attrs:
                                        return True

                    # R5: self.X = self.Y  (direct alias of a known nn attribute)
                    if _is_self_attr(tgt) and _is_self_attr(val):
                        assert isinstance(tgt, ast.Attribute)
                        assert isinstance(val, ast.Attribute)
                        if val.attr in param_attrs and val.attr != tgt.attr:
                            return True

            # R6/R7: setattr(self, name, self.Y) or setattr(self, name, self.Y.weight)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "setattr"
                    and len(call.args) == 3
                    and isinstance(call.args[0], ast.Name)
                    and call.args[0].id == "self"
                ):
                    val = call.args[2]
                    # R6: setattr(self, ?, self.Y)
                    if _is_self_attr(val) and isinstance(val, ast.Attribute):
                        if val.attr in param_attrs:
                            return True
                    # R7: setattr(self, ?, self.Y.weight)
                    if _is_self_dot_weight_or_bias(val) and isinstance(val, ast.Attribute):
                        src_mod = _nn_attr_name(val)
                        if src_mod and src_mod in param_attrs:
                            return True

    return False


def main() -> int:
    if not CORPUS_PATH.exists():
        print(
            "INFEASIBLE: real-source corpus path unresolved or unparseable; reverting",
            file=sys.stderr,
        )
        return 1

    n_blocks = 0
    n_flagged = 0
    flagged_ids: list[str] = []

    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            block = json.loads(line)
            source = block.get("source", "")
            block_id = block.get("id", f"block_{n_blocks}")
            n_blocks += 1
            if detect_tied_params(source):
                n_flagged += 1
                flagged_ids.append(block_id)

    if n_blocks == 0:
        print(
            "INFEASIBLE: real-source corpus path unresolved or unparseable; reverting",
            file=sys.stderr,
        )
        return 1

    prevalence = n_flagged / n_blocks
    wilson_low, wilson_high = wilson_ci(n_flagged, n_blocks)
    recomputed_bound = wilson_high * 0.25

    result = {
        "n_blocks": n_blocks,
        "n_flagged": n_flagged,
        "prevalence": prevalence,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "recomputed_bound": recomputed_bound,
        "flagged_ids": flagged_ids,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"PREVALENCE_AUDIT prevalence={prevalence:.4f} "
        f"wilson=[{wilson_low:.4f},{wilson_high:.4f}] "
        f"bound={recomputed_bound:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
