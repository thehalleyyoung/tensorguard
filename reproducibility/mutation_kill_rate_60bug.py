#!/usr/bin/env python3.11
"""AST mutation-testing kill rate on src/model_checker.py (R4-W5).

Each mutant is scored in a fresh subprocess to avoid Z3
process-global state leaking across in-process re-imports.

Output:
    reproducibility/mutation_kill_rate_60bug.json
    reproducibility/mutation_kill_rate_60bug.md
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

TARGET = os.path.join(ROOT, "src", "model_checker.py")
CORPUS = os.path.join(ROOT, "experiments_v5", "v5_bug_corpus.jsonl")
OUT_JSON = os.path.join(ROOT, "reproducibility",
                        "mutation_kill_rate_60bug.json")
OUT_MD = os.path.join(ROOT, "reproducibility",
                      "mutation_kill_rate_60bug.md")
N_MUTANTS = 50
SEED = 0
SUBPROC_TIMEOUT = 240

WORKER = r'''
import json, os, sys, traceback
ROOT = os.environ["TG_ROOT"]
sys.path.insert(0, ROOT)
from src.api import verify_architecture
items = []
with open(os.path.join(ROOT, "experiments_v5", "v5_bug_corpus.jsonl")) as f:
    for ln in f:
        items.append(json.loads(ln))
out = []
for it in items:
    try:
        with open(os.path.join(ROOT, it["repro_file"])) as f:
            src = f.read()
        r = verify_architecture(src)
        s = getattr(r, "status", None)
        sn = s.name if hasattr(s, "name") else str(s)
        v = ("RP" if sn == "UNSAFE" else
             "V"  if sn == "SAFE"   else "ABST")
    except Exception as e:
        v = "ERR:" + type(e).__name__
    out.append({"id": it["id"], "verdict": v})
print("__BEGIN__")
print(json.dumps(out))
print("__END__")
'''


# ── Mutation operators ────────────────────────────────────────────────
class _Mutator(ast.NodeTransformer):
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.applied = 0
        self.description = ""

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.applied:
            return node
        flips = {ast.Lt: ast.Gt, ast.LtE: ast.GtE,
                 ast.Gt: ast.Lt, ast.GtE: ast.LtE,
                 ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
        new_ops, flipped = [], False
        for op in node.ops:
            t = type(op)
            if not flipped and t in flips and self.rng.random() < 0.04:
                new_ops.append(flips[t]())
                flipped = True
            else:
                new_ops.append(op)
        if flipped:
            self.applied = 1
            self.description = (
                f"M1 compare flip ({type(node.ops[0]).__name__})"
            )
            node.ops = new_ops
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.applied:
            return node
        if isinstance(node.op, (ast.And, ast.Or)) and self.rng.random() < 0.05:
            new_op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = 1
            self.description = (
                f"M2 bool flip ({type(node.op).__name__}->"
                f"{type(new_op).__name__})"
            )
            node.op = new_op
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.applied:
            return node
        if isinstance(node.op, ast.Add) and self.rng.random() < 0.04:
            self.applied = 1
            self.description = "M3 +->-"
            node.op = ast.Sub()
        elif isinstance(node.op, ast.Mult) and self.rng.random() < 0.04:
            self.applied = 1
            self.description = "M3 *->+"
            node.op = ast.Add()
        return node

    def visit_Constant(self, node):
        if self.applied:
            return node
        if isinstance(node.value, bool) and self.rng.random() < 0.04:
            self.applied = 1
            self.description = (
                f"M5 bool const flip ({node.value}->{not node.value})"
            )
            return ast.copy_location(
                ast.Constant(value=not node.value), node)
        if (isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and 0 <= node.value <= 4
                and self.rng.random() < 0.04):
            self.applied = 1
            new_val = node.value + 1
            self.description = (
                f"M4 int const +1 ({node.value}->{new_val})"
            )
            return ast.copy_location(
                ast.Constant(value=new_val), node)
        return node


def _mutate(source: str, seed: int) -> Tuple[str, str]:
    for off in range(20):
        tree = ast.parse(source)
        m = _Mutator(seed + off * 1000)
        new_tree = m.visit(tree)
        ast.fix_missing_locations(new_tree)
        if m.applied:
            try:
                return ast.unparse(new_tree), m.description
            except Exception:
                continue
    return source, "<no mutation>"


def _run_subproc(env_root: str) -> Dict[str, Any]:
    env = os.environ.copy()
    env["TG_ROOT"] = env_root
    try:
        p = subprocess.run(
            [sys.executable, "-c", WORKER],
            env=env, capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            timeout=SUBPROC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"error": "subproc_timeout", "per_bug": []}
    out = p.stdout
    if "__BEGIN__" not in out or "__END__" not in out:
        tail = (p.stderr or "")[-200:].replace("\n", " | ")
        return {"error": "subproc_load_error",
                "stderr_tail": tail, "per_bug": []}
    body = out.split("__BEGIN__", 1)[1].split("__END__", 1)[0].strip()
    try:
        per = json.loads(body)
    except Exception as e:
        return {"error": f"json: {e}", "per_bug": []}
    return {"error": None, "per_bug": per}


def _diff(a, b):
    by = {r["id"]: r["verdict"] for r in a}
    return [f"{r['id']}:{by.get(r['id'])}->{r['verdict']}"
            for r in b if by.get(r["id"]) != r["verdict"]]


def main() -> int:
    rng = random.Random(SEED)
    src = open(TARGET).read()
    print(f"Source size: {len(src):,} bytes")

    print("Scoring clean baseline (subprocess) ...")
    t0 = time.time()
    base_res = _run_subproc(ROOT)
    if base_res["error"]:
        print(f"BASELINE FAILED: {base_res['error']}")
        return 1
    baseline = base_res["per_bug"]
    base_rp = sum(1 for r in baseline if r["verdict"] == "RP")
    print(f"  clean baseline RP={base_rp}/60 ({time.time()-t0:.1f}s)")

    runs, killed, syntax_errs, no_op = [], 0, 0, 0
    for i in range(N_MUTANTS):
        seed_i = rng.randint(0, 10**9)
        mutated, desc = _mutate(src, seed_i)
        if desc == "<no mutation>":
            no_op += 1
            continue
        with open(TARGET, "w") as f:
            f.write(mutated)
        try:
            t0 = time.time()
            r = _run_subproc(ROOT)
            elapsed = time.time() - t0
        finally:
            with open(TARGET, "w") as f:
                f.write(src)

        if r["error"] == "subproc_load_error":
            syntax_errs += 1
            killed += 1
            print(f"  m{i+1:02d}: {desc[:55]:55s} LOAD_ERR  "
                  f"({elapsed:5.1f}s)")
            runs.append({"i": i, "description": desc,
                         "load_error": r.get("stderr_tail", ""),
                         "killed": True, "elapsed_s": round(elapsed, 2)})
            continue
        if r["error"]:
            print(f"  m{i+1:02d}: {desc[:55]:55s} ERR={r['error']}  "
                  f"({elapsed:5.1f}s)")
            runs.append({"i": i, "description": desc,
                         "subproc_error": r["error"],
                         "killed": True, "elapsed_s": round(elapsed, 2)})
            killed += 1
            continue

        diffs = _diff(baseline, r["per_bug"])
        is_killed = len(diffs) > 0
        if is_killed:
            killed += 1
        print(f"  m{i+1:02d}: {desc[:55]:55s} diffs={len(diffs):2d} "
              f"killed={is_killed}  ({elapsed:5.1f}s)")
        runs.append({"i": i, "description": desc,
                     "n_verdict_changes": len(diffs),
                     "verdict_changes": diffs[:20],
                     "killed": is_killed,
                     "elapsed_s": round(elapsed, 2)})

    n_actual = len(runs)
    kill_rate = killed / n_actual if n_actual else 0.0

    out = {
        "_question": (
            "R4-W5: kill rate of an AST-rewrite mutation-testing "
            "sweep on src/model_checker.py against the 60-bug "
            "regression.  Each mutant is scored in a fresh "
            "subprocess to avoid Z3 process-global state leakage."
        ),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "target_file": "src/model_checker.py",
        "target_size_bytes": len(src),
        "n_mutants_attempted": N_MUTANTS,
        "n_mutants_with_real_mutation": n_actual,
        "n_no_op_seeds": no_op,
        "n_syntax_load_errors": syntax_errs,
        "n_killed": killed,
        "kill_rate": kill_rate,
        "clean_baseline_rp": base_rp,
        "subproc_timeout_s": SUBPROC_TIMEOUT,
        "mutants": runs,
        "interpretation": (
            f"Of {n_actual} subprocess-isolated mutants "
            f"(single-edit AST mutations of comparison / boolean "
            f"/ arithmetic operators and small-integer / boolean "
            f"constants in src/model_checker.py), {killed} were "
            f"killed by the 60-bug regression, giving a kill "
            f"rate of {kill_rate*100:.1f}% (of which "
            f"{syntax_errs} failed to load the analyser at all)."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Mutation kill rate on the 60-bug corpus",
        "",
        "## Command",
        "",
        "```",
        "python3.11 reproducibility/mutation_kill_rate_60bug.py",
        "```",
        "",
        "## Mutation operators (single-edit AST rewrites)",
        "",
        "- M1: comparison flip (`<`->`>`, `<=`->`>=`, `==`->`!=`)",
        "- M2: boolean operator flip (`and`->`or`)",
        "- M3: arithmetic op swap (`+`->`-`, `*`->`+`)",
        "- M4: small-integer constant `+1` (constants 0..4)",
        "- M5: boolean constant flip (`True`->`False`)",
        "",
        "Each mutant is scored in a fresh Python 3.11 subprocess "
        "to avoid Z3 process-global state leaking across mutants.  "
        "A mutant is *killed* iff at least one of the 60 bugs has "
        "a different verdict from the clean baseline (or the "
        "mutated source fails to load).",
        "",
        "## Result",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mutants attempted | {N_MUTANTS} |",
        f"| Mutants with real mutation | {n_actual} |",
        f"| Killed (>=1 verdict change OR load error) | {killed} |",
        f"| Strong kills (analyser fails to load) | {syntax_errs} |",
        f"| **Kill rate** | **{kill_rate*100:.1f}%** |",
        f"| Clean baseline RP | {base_rp}/60 |",
        "",
        "## Paper claim closed",
        "",
        "Round-4 W5 noted that the four-fault TCB exposure / "
        "measured-flip pair is hand-picked, and suggested an "
        "automated mutation-testing sweep as the natural next "
        "instrument.  This artefact reports the mutation kill "
        f"rate on the 60-bug regression: {killed}/{n_actual} = "
        f"{kill_rate*100:.1f}%.  Combined with the four-fault "
        "exposure / measured-flip pair, this provides an "
        "automated lower bound on analyser-level robustness "
        "that is not limited to the four hand-picked TCB faults.",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")
    print(f"Kill rate: {killed}/{n_actual} = {kill_rate*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
