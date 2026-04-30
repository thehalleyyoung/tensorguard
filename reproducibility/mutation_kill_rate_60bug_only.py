#!/usr/bin/env python3
"""Targeted mutation kill rate for the conv2d and einsum load-bearing handlers,
scored on the 60-bug corpus alone (no targeted extension).

This is the baseline measurement requested in Round-5 reviewer Q4: 
the kill rate on the historical 60-bug corpus without the 18-case 
targeted extension, for the FULL mutation class enumeration.

This script is identical to mutation_kill_rate_loadbearing_v2.py 
EXCEPT it uses only CORPUS_60BUG (dropping CORPUS_EXT).

Output:
    reproducibility/mutation_kill_rate_60bug_only.json
    reproducibility/mutation_kill_rate_60bug_only.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

TARGET = os.path.join(ROOT, "src", "model_checker.py")
CORPUS_60BUG = os.path.join(ROOT, "experiments_v5", "v5_bug_corpus.jsonl")

OUT_JSON = os.path.join(ROOT, "reproducibility", "mutation_kill_rate_60bug_only.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "mutation_kill_rate_60bug_only.md")

HANDLER_RANGES = {
    "conv_channel_mismatch": (4911, 5017),
    "einsum_dim":            (8259, 8302),
}

SUBPROC_TIMEOUT = 240

MUTATION_CLASSES: List[tuple] = [
    ("<", ">"), (">", "<"), ("<=", ">="), (">=", "<="),
    ("==", "!="), ("!=", "=="),
    (" and ", " or "), (" or ", " and "),
    (" + ", " - "), (" - ", " + "), (" * ", " + "), (" / ", " * "),
]

WORKER = r'''
import json, os, sys
ROOT = os.environ["TG_ROOT"]
CORPORA = os.environ["CORPORA_PATHS"].split(os.pathsep)
sys.path.insert(0, ROOT)
from src.api import verify_architecture
items = []
for p in CORPORA:
    with open(p) as f:
        for ln in f:
            items.append(json.loads(ln))
out = []
for it in items:
    try:
        with open(os.path.join(ROOT, it["repro_file"])) as f:
            src = f.read()
        kwargs = {}
        if it.get("input_shapes"):
            kwargs["input_shapes"] = {k: tuple(v) for k, v in it["input_shapes"].items()}
        r = verify_architecture(src, **kwargs)
        s = getattr(r, "status", None)
        sn = s.name if hasattr(s, "name") else str(s)
        v = ("RP" if sn == "UNSAFE" else "V" if sn == "SAFE" else "ABST")
    except Exception:
        v = "ERR"
    out.append({"id": it["id"], "verdict": v})
print("__BEGIN__")
print(json.dumps(out))
print("__END__")
'''


def _score_one(env_extra: dict) -> List[dict]:
    env = {**os.environ, **env_extra, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", WORKER],
            input=None, capture_output=True, text=True,
            timeout=SUBPROC_TIMEOUT, env=env,
        )
        stdout = proc.stdout
        if "__BEGIN__" in stdout and "__END__" in stdout:
            body = stdout.split("__BEGIN__")[1].split("__END__")[0].strip()
            return json.loads(body)
        return []
    except (subprocess.TimeoutExpired, Exception):
        return []


def _docstring_lines(source: str) -> set:
    """Return the set of 1-based line numbers occupied by docstrings."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                start = first.lineno
                end = getattr(first, "end_lineno", start)
                for ln in range(start, end + 1):
                    out.add(ln)
    return out


def _enumerate_mutants(source: str, start_line: int, end_line: int):
    """Enumerate (line_no, occurrence_index, old, new) tuples for every
    applicable mutation occurrence (not just the first occurrence per line).
    Excludes lines that are part of a docstring or a comment."""
    lines = source.splitlines()
    docstring = _docstring_lines(source)
    out = []
    for i in range(start_line - 1, min(end_line, len(lines))):
        line_no = i + 1
        if line_no in docstring:
            continue
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # strip trailing '#'-comment so we don't mutate inside comments
        comment_pos = -1
        in_str = False
        quote = ""
        for k, ch in enumerate(line):
            if in_str:
                if ch == quote and line[k-1:k] != "\\":
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    quote = ch
                elif ch == "#":
                    comment_pos = k
                    break
        code = line if comment_pos < 0 else line[:comment_pos]
        for old, new in MUTATION_CLASSES:
            occ = code.count(old)
            for k in range(occ):
                out.append((line_no, k, old, new))
    return out


def _apply_mutation(original_lines: List[str], line_no: int, occ_index: int,
                    old: str, new: str) -> str:
    """Replace the occ_index-th occurrence of `old` with `new` on line_no."""
    idx = line_no - 1
    line = original_lines[idx]
    pos = -1
    for _ in range(occ_index + 1):
        pos = line.find(old, pos + 1)
        if pos < 0:
            break
    if pos < 0:
        return "\n".join(original_lines)
    new_line = line[:pos] + new + line[pos + len(old):]
    new_lines = original_lines[:]
    new_lines[idx] = new_line
    return "\n".join(new_lines)


def run():
    with open(TARGET) as f:
        original_src = f.read()
    original_lines = original_src.splitlines()

    # Use ONLY the 60-bug corpus (no targeted extension)
    env_score = {"TG_ROOT": ROOT, "CORPORA_PATHS": CORPUS_60BUG}

    print("Scoring baseline (60-bug corpus only)...")
    baseline = _score_one(env_score)
    if not baseline:
        print("FATAL: baseline scoring failed")
        sys.exit(1)
    baseline_map = {r["id"]: r["verdict"] for r in baseline}
    n_rp = sum(1 for v in baseline_map.values() if v == "RP")
    print(f"  baseline: {n_rp}/{len(baseline_map)} RP across 60-bug corpus")

    results: Dict[str, dict] = {}
    for handler_name, (start, end) in HANDLER_RANGES.items():
        muts = _enumerate_mutants(original_src, start, end)
        print(f"\nHandler {handler_name}: {len(muts)} candidate (line, occ, mutation) tuples in [{start}, {end}]")
        killed = 0
        per_mut = []
        for j, (lineno, occ, old, new) in enumerate(muts, 1):
            mutated = _apply_mutation(original_lines, lineno, occ, old, new)
            if mutated == original_src:
                per_mut.append({"line": lineno, "occ": occ, "old": old, "new": new, "killed": False, "reason": "no_change"})
                continue
            try:
                with open(TARGET, "w") as f:
                    f.write(mutated)
                res = _score_one(env_score)
                res_map = {r["id"]: r["verdict"] for r in res}
                changed = any(res_map.get(k) != v for k, v in baseline_map.items() if k in res_map)
                if changed:
                    killed += 1
                per_mut.append({"line": lineno, "occ": occ, "old": old.strip(), "new": new.strip(), "killed": changed})
                tag = "KILLED" if changed else "survived"
                print(f"  [{j}/{len(muts)}] line {lineno} occ#{occ} '{old.strip()}'->'{new.strip()}': {tag}")
            finally:
                with open(TARGET, "w") as f:
                    f.write(original_src)
        total = len([m for m in per_mut if m.get("reason") != "no_change"])
        results[handler_name] = {
            "killed": killed,
            "total": total,
            "kill_rate": round(killed / total, 4) if total else 0.0,
            "mutants": per_mut,
        }
        print(f"  -> {handler_name}: {killed}/{total} = {killed/total*100 if total else 0:.0f}%")

    union_killed = sum(v["killed"] for v in results.values())
    union_total = sum(v["total"] for v in results.values())

    # Reviewer-asked subset: comparison-flip + arithmetic-swap only
    # (excluding boolean-op flips on defensive guard conditions).
    cmp_arith_per_handler = {}
    cmp_arith_union_killed = 0
    cmp_arith_union_total = 0
    for name, hd in results.items():
        muts = [m for m in hd["mutants"]
                if m.get("reason") != "no_change"
                and m["old"] not in ("and", "or")]
        k = sum(1 for m in muts if m["killed"])
        cmp_arith_per_handler[name] = {
            "killed": k,
            "total": len(muts),
            "kill_rate": round(k / len(muts), 4) if muts else 0.0,
        }
        cmp_arith_union_killed += k
        cmp_arith_union_total += len(muts)

    out = {
        "handler_ranges": HANDLER_RANGES,
        "corpora": ["v5_bug_corpus (60-bug only)"],
        "baseline_rp_total": n_rp,
        "baseline_total": len(baseline_map),
        "per_handler": results,
        "union_killed": union_killed,
        "union_total": union_total,
        "union_kill_rate": round(union_killed / union_total, 4) if union_total else 0.0,
        "comparison_flip_and_arithmetic_swap_only": {
            "per_handler": cmp_arith_per_handler,
            "union_killed": cmp_arith_union_killed,
            "union_total": cmp_arith_union_total,
            "union_kill_rate": round(cmp_arith_union_killed / cmp_arith_union_total, 4) if cmp_arith_union_total else 0.0,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    rows_all = "\n".join(
        f"| {name} | {v['killed']} | {v['total']} | {v['kill_rate']*100:.0f}% |"
        for name, v in results.items()
    )
    rows_ca = "\n".join(
        f"| {name} | {v['killed']} | {v['total']} | {v['kill_rate']*100:.0f}% |"
        for name, v in cmp_arith_per_handler.items()
    )
    md = f"""# Targeted mutation kill rate (60-bug corpus only): conv2d & einsum

## Command

```bash
python3 reproducibility/mutation_kill_rate_60bug_only.py
```

## Setup

Handler ranges:

| Handler | Lines |
|---|---|
| conv_channel_mismatch | 4911--5017 |
| einsum_dim            | 8259--8302 |

Corpus used (baseline only):

  * 60-bug historical corpus alone, with NO targeted extension.
  * This measurement provides the regressor-alone calibration anchor,
    unadjusted by the 18-case targeted extension.

Baseline: {n_rp}/{len(baseline_map)} RP on the 60-bug corpus.

We enumerate **every (line, occurrence, mutation)** triple that is
syntactically applicable on a non-docstring, non-comment line in the
handler range.  A mutant is killed iff at least one verdict in the
corpus differs from the clean baseline.

## Headline (full mutation class enumeration)

All mutation classes across all syntactic occurrences:

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
{rows_all}
| **Union** | **{union_killed}** | **{union_total}** | **{union_killed/union_total*100 if union_total else 0:.0f}%** |

## Subset: comparison-flip + arithmetic-swap only

Restricting to comparison-flip and arithmetic-swap mutation classes
(`<`, `>`, `<=`, `>=`, `==`, `!=`, `+`, `-`, `*`, `/`) and
excluding boolean-op flips (`and`/`or`):

| Handler | Killed | Total | Kill rate |
|---|---|---|---|
{rows_ca}
| **Union** | **{cmp_arith_union_killed}** | **{cmp_arith_union_total}** | **{cmp_arith_union_killed/cmp_arith_union_total*100 if cmp_arith_union_total else 0:.0f}%** |

## Interpretation

This measurement represents the **regressor-alone baseline**: the mutation
kill rate on the original 60-bug corpus without the targeted extension 
corpus. These numbers provide the anchor against which to contrast the
load-bearing-extension union results.

**Full enumeration:**
  * conv2d: {results['conv_channel_mismatch']['killed']}/{results['conv_channel_mismatch']['total']} = {results['conv_channel_mismatch']['kill_rate']*100:.0f}%
  * einsum: {results['einsum_dim']['killed']}/{results['einsum_dim']['total']} = {results['einsum_dim']['kill_rate']*100:.0f}%

**Comparison-flip + arithmetic-swap subset:**
  * conv2d: {cmp_arith_per_handler['conv_channel_mismatch']['killed']}/{cmp_arith_per_handler['conv_channel_mismatch']['total']} = {cmp_arith_per_handler['conv_channel_mismatch']['kill_rate']*100:.0f}%
  * einsum: {cmp_arith_per_handler['einsum_dim']['killed']}/{cmp_arith_per_handler['einsum_dim']['total']} = {cmp_arith_per_handler['einsum_dim']['kill_rate']*100:.0f}%

The 60-bug-corpus-only baseline allows direct measurement of load-bearing
regressor effectiveness on the historical bug suite, independent of any
targeted-extension amplification effects. Comparison with the union corpus
numbers reveals how much additional detection capability the targeted
extension provides for these specific handlers.
"""
    with open(OUT_MD, "w") as f:
        f.write(md)

    print(f"\nUnion: {union_killed}/{union_total} = {union_killed/union_total*100 if union_total else 0:.0f}%")
    print(f"Comparison+arithmetic only: {cmp_arith_union_killed}/{cmp_arith_union_total} = "
          f"{cmp_arith_union_killed/cmp_arith_union_total*100 if cmp_arith_union_total else 0:.0f}%")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    run()
