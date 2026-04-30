#!/usr/bin/env python3.11
"""Unified mutation kill rate: same 50-mutant sweep run against the
union of (60-bug ∪ 488-block sample ∪ 25-stress ∪ targeted-extension)
corpora.

The reviewer's round-18 W4 explicitly asked for the targeted extension
corpus to be plugged into the same 50-mutant sweep so the resulting
kill rate is directly comparable to the headline 7/50 union number.
This artifact does exactly that: it reuses the mutator and the three
existing workers from mutation_kill_rate_corpora.py, adds a fourth
worker for the targeted-extension corpus
(experiments_v5/bug_repros_loadbearing_ext/), and reports a single
union kill rate.

Output:
    reproducibility/mutation_kill_rate_unified.json
    reproducibility/mutation_kill_rate_unified.md
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Reuse the mutator + worker plumbing from the existing artifact.
spec = importlib.util.spec_from_file_location(
    "mkc",
    os.path.join(ROOT, "reproducibility", "mutation_kill_rate_corpora.py"),
)
mkc = importlib.util.module_from_spec(spec)
sys.modules["mkc"] = mkc
spec.loader.exec_module(mkc)  # type: ignore

OUT_JSON = os.path.join(ROOT, "reproducibility", "mutation_kill_rate_unified.json")
OUT_MD = os.path.join(ROOT, "reproducibility", "mutation_kill_rate_unified.md")

WORKER_TARGETED_EXT = r'''
import json, os, sys
ROOT = os.environ["TG_ROOT"]
sys.path.insert(0, ROOT)
from src.api import verify_architecture

ext_dir = os.path.join(ROOT, "experiments_v5", "bug_repros_loadbearing_ext")
files = sorted(f for f in os.listdir(ext_dir) if f.endswith(".py"))
out = []
for fn in files:
    path = os.path.join(ext_dir, fn)
    try:
        with open(path) as fh:
            src = fh.read()
        r = verify_architecture(src)
        s = getattr(r, "status", None)
        sn = s.name if hasattr(s, "name") else str(s)
        v = ("RP" if sn == "UNSAFE" else
             "V"  if sn == "SAFE"   else "ABST")
    except Exception as e:
        v = "ERR:" + type(e).__name__
    out.append({"id": fn, "verdict": v})
print("__BEGIN__")
print(json.dumps(out))
print("__END__")
'''

N_MUTANTS = mkc.N_MUTANTS
SEED = mkc.SEED
SAMPLE_488 = mkc.SAMPLE_488
TARGET = mkc.TARGET


def main() -> int:
    rng = random.Random(SEED)
    src_text = open(TARGET).read()
    print(f"Source size: {len(src_text):,} bytes")
    print(f"N_MUTANTS={N_MUTANTS}, SEED={SEED}, SAMPLE_488={SAMPLE_488}")

    print("\nScoring clean baselines (4 corpora) ...")
    baselines = {}
    for tag, worker, env in [
        ("60bug", mkc.WORKER_60BUG, None),
        ("488block", mkc.WORKER_488BLOCK, {"SAMPLE_SIZE": str(SAMPLE_488)}),
        ("25stress", mkc.WORKER_25STRESS, None),
        ("targeted_ext", WORKER_TARGETED_EXT, None),
    ]:
        t0 = time.time()
        r = mkc._run_subproc(worker, env)
        if r["error"]:
            print(f"BASELINE {tag} FAILED: {r['error']}")
            return 1
        baselines[tag] = r["per_item"]
        rp = sum(1 for it in r["per_item"] if it["verdict"] == "RP")
        print(f"  {tag:13s}: n={len(r['per_item'])}, RP={rp} ({time.time()-t0:.1f}s)")

    runs = []
    killed = {tag: 0 for tag in baselines}
    killed["union"] = 0
    syntax_errs = 0
    no_op = 0

    for i in range(N_MUTANTS):
        seed_i = rng.randint(0, 10**9)
        mutated, desc = mkc._mutate(src_text, seed_i)
        if desc == "<no mutation>":
            no_op += 1
            continue

        with open(TARGET, "w") as f:
            f.write(mutated)
        try:
            t0 = time.time()
            results = {}
            for tag, worker, env in [
                ("60bug", mkc.WORKER_60BUG, None),
                ("488block", mkc.WORKER_488BLOCK, {"SAMPLE_SIZE": str(SAMPLE_488)}),
                ("25stress", mkc.WORKER_25STRESS, None),
                ("targeted_ext", WORKER_TARGETED_EXT, None),
            ]:
                results[tag] = mkc._run_subproc(worker, env)
            elapsed = time.time() - t0
        finally:
            with open(TARGET, "w") as f:
                f.write(src_text)

        if any(r["error"] == "subproc_load_error" for r in results.values()):
            syntax_errs += 1
            for tag in baselines:
                killed[tag] += 1
            killed["union"] += 1
            print(f"  m{i+1:02d}: {desc[:50]:50s} LOAD_ERR ({elapsed:5.1f}s)")
            runs.append({"i": i, "description": desc, "load_error": True,
                         **{f"killed_{t}": True for t in baselines},
                         "killed_union": True, "elapsed_s": round(elapsed, 2)})
            continue

        if any(r["error"] for r in results.values()):
            err_msg = "; ".join(f"{t}:{r['error']}" for t, r in results.items() if r["error"])
            for tag in baselines:
                killed[tag] += 1
            killed["union"] += 1
            print(f"  m{i+1:02d}: {desc[:50]:50s} ERR={err_msg[:30]} ({elapsed:5.1f}s)")
            runs.append({"i": i, "description": desc, "subproc_error": err_msg,
                         **{f"killed_{t}": True for t in baselines},
                         "killed_union": True, "elapsed_s": round(elapsed, 2)})
            continue

        diffs = {tag: mkc._diff(baselines[tag], results[tag]["per_item"]) for tag in baselines}
        kflags = {tag: len(diffs[tag]) > 0 for tag in baselines}
        kunion = any(kflags.values())
        for tag, k in kflags.items():
            if k:
                killed[tag] += 1
        if kunion:
            killed["union"] += 1

        diff_str = ",".join(f"{tag[:3]}:{len(diffs[tag])}" for tag in baselines)
        kstr = ",".join(f"{tag[:3]}:{int(k)}" for tag, k in kflags.items())
        print(f"  m{i+1:02d}: {desc[:50]:50s} d=({diff_str}) k=({kstr},U:{int(kunion)}) ({elapsed:5.1f}s)")

        runs.append({
            "i": i, "description": desc,
            **{f"n_diffs_{t}": len(diffs[t]) for t in baselines},
            **{f"killed_{t}": kflags[t] for t in baselines},
            "killed_union": kunion,
            "elapsed_s": round(elapsed, 2),
        })

    n_run = N_MUTANTS - no_op
    summary = {
        "n_mutants_attempted": N_MUTANTS,
        "n_no_op": no_op,
        "n_real_mutants": n_run,
        "n_syntax_errs": syntax_errs,
        **{f"killed_{tag}": killed[tag] for tag in baselines},
        "killed_union": killed["union"],
        **{f"kill_rate_{tag}": (killed[tag] / n_run if n_run else 0.0) for tag in baselines},
        "kill_rate_union": (killed["union"] / n_run if n_run else 0.0),
        "baseline_RP": {tag: sum(1 for it in items if it["verdict"] == "RP")
                        for tag, items in baselines.items()},
        "baseline_N": {tag: len(items) for tag, items in baselines.items()},
    }

    out = {"meta": {
        "command": "python3.11 reproducibility/mutation_kill_rate_unified.py",
        "n_mutants": N_MUTANTS, "seed": SEED, "sample_488": SAMPLE_488,
    }, "summary": summary, "runs": runs}

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    _write_md(summary)
    print("\n" + "=" * 70)
    print(json.dumps(summary, indent=2))
    return 0


def _write_md(s):
    lines = []
    lines.append("# Unified mutation kill rate (4-corpus union, 50-mutant sweep)")
    lines.append("")
    lines.append("## Command")
    lines.append("")
    lines.append("    python3.11 reproducibility/mutation_kill_rate_unified.py")
    lines.append("")
    lines.append("## Reviewer obligation")
    lines.append("")
    lines.append(
        "Round-18 W4: the global mutation union (7/50) and the targeted "
        "per-handler measurement (conv2d 53%, einsum 100%) are not directly "
        "comparable because the latter is reported on a separate 18-case "
        "targeted corpus.  This artifact reruns the *same* 50-mutant sweep "
        "against the union of all four corpora "
        "(60-bug ∪ 488-block sample ∪ 25-stress ∪ targeted-extension)."
    )
    lines.append("")
    lines.append("## Mutation operators")
    lines.append("")
    lines.append("Same as `mutation_kill_rate_corpora.py`: M1 comparison flip, "
                 "M2 boolean-op flip, M3 arithmetic-op swap, M4 small-int +1, "
                 "M5 boolean constant flip.")
    lines.append("")
    lines.append("## Per-corpus and union kill rates (one mutant either kills or it does not)")
    lines.append("")
    lines.append("| Corpus | Baseline RP / N | Killed / 50 | Kill rate |")
    lines.append("|---|---|---:|---:|")
    for tag in ["60bug", "488block", "25stress", "targeted_ext"]:
        lines.append(
            f"| {tag} | {s['baseline_RP'][tag]} / {s['baseline_N'][tag]} | "
            f"{s[f'killed_{tag}']} / {s['n_real_mutants']} | "
            f"{100*s[f'kill_rate_{tag}']:.1f}% |"
        )
    lines.append(
        f"| **Union (any corpus)** | --- | **{s['killed_union']} / {s['n_real_mutants']}** | "
        f"**{100*s['kill_rate_union']:.1f}%** |"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"Adding the targeted-extension corpus to the same 50-mutant sweep raises "
        f"the union kill rate from the 4-corpus measurement to "
        f"**{s['killed_union']}/{s['n_real_mutants']} = {100*s['kill_rate_union']:.1f}%**.  "
        f"The targeted-extension corpus alone, scored against the same 50 mutants, "
        f"yields {s['killed_targeted_ext']}/{s['n_real_mutants']} = "
        f"{100*s['kill_rate_targeted_ext']:.1f}%."
    )
    lines.append("")
    lines.append("## Paper claim cited by this artifact")
    lines.append("")
    lines.append(
        "- Eval section paragraph on mutation-testing robustness: a unified, "
        "directly-comparable union kill rate against the 4-corpus union, "
        "supplementing the 7/50 = 14% union reported in "
        "`mutation_kill_rate_corpora.md`."
    )
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
