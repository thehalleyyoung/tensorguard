"""Head-to-head benchmark: TensorGuard vs torch.compile vs Pytea on the
pre-registered post-freeze N=15 unfiltered corpus.

Addresses reviewer questions:
  - "per-bug verdict pair against torch.compile"
  - "abandoned Pytea baseline"

Usage
-----
    PYTHONPATH=. python -m benchmarks.torch_compile_headtohead \
        --out benchmarks/results/headtohead_n15.csv

Outputs
-------
  benchmarks/results/headtohead_n15.csv   - per-bug verdict table
  benchmarks/results/headtohead_n15_stats.json - pairwise McNemar + BH-Fisher
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.api import verify_architecture  # noqa: E402

HERE = os.path.dirname(__file__)
UF_DIR = os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_unfiltered")
PF_DIR = os.path.join(ROOT, "experiments_v5", "v8", "real_bugs_postfreeze")
MANIFEST_PATH = os.path.join(UF_DIR, "manifest.json")

# ---------------------------------------------------------------------------
# Pytea catalogue cache
# Pytea (commit 2022-04-26) is not pip-installable in this environment.
# Verdicts are hand-filled from the existing Pytea harness runs for the
# subset covered by Pytea's operator catalogue.  All other entries are "n/a".
# Source: experiments_v5/pytea_baseline_results.json + verify_real_bugs_unfiltered.py
# ---------------------------------------------------------------------------
PYTEA_CACHE: dict[str, str] = {
    "rb_pf_003": "refuted",   # 3D LoRA + matmul: Pytea Linear-in-features check
    "rb_pf_004": "verified",  # softmax over wrong dim is a value bug, Pytea silent
    "rb_uf_007": "refuted",   # view + Linear: catalogue-covered
    "rb_uf_008": "n/a",       # 3D conv decoder: outside Pytea's nn.Conv3d coverage
    "rb_uf_013": "verified",  # broadcast multiply: attribute lookup skip
    "rb_uf_015": "refuted",   # transpose+view+Linear: catalogue-covered
}


def _resolve_path(item: dict) -> str:
    f = item["file"]
    if f.startswith("../real_bugs_postfreeze/"):
        return os.path.join(PF_DIR, os.path.basename(f))
    return os.path.join(UF_DIR, os.path.basename(f))


def _load_repro(repro_path: str) -> tuple[str, dict]:
    """Return (source_code, INPUT_SHAPES) from a repro file."""
    src = open(repro_path).read()
    ns: dict = {}
    try:
        exec(compile(src, repro_path, "exec"), ns)
    except Exception:
        pass
    return src, ns.get("INPUT_SHAPES", {})


# ---------------------------------------------------------------------------
# TensorGuard verdict
# ---------------------------------------------------------------------------

def _tg_verdict(src: str, input_shapes: dict, repro_path: str) -> str:
    """Run TensorGuard and return a verdict string."""
    try:
        result = verify_architecture(
            src,
            input_shapes=input_shapes,
            high_confidence_only=False,
            filename=os.path.basename(repro_path),
        )
        n_bugs = len(result.bugs)
        max_conf = max((b.confidence for b in result.bugs), default=0.0)
        if max_conf >= 0.99:
            return "RP_0.99"
        if max_conf >= 0.80:
            return "RP_0.80"
        if n_bugs == 0:
            return "silent_verified"
        return "low_confidence"
    except Exception as e:
        return f"abstain"


# ---------------------------------------------------------------------------
# torch.compile / FakeTensorMode verdict
# ---------------------------------------------------------------------------

def _compile_verdict(repro_path: str, input_shapes: dict) -> str:
    """Try to catch shape bugs via FakeTensorMode and torch.compile tracing.

    Strategy:
      1. First pass: FakeTensorMode (static symbolic shapes, cheapest).
      2. Second pass: torch.compile with eager fallback (dynamo graph
         capture can surface shape mismatches at trace time).

    Returns one of {"refuted", "verified", "abstain"}.
    """
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
    except Exception:
        return "abstain"

    # Load the repro module
    ns: dict = {}
    try:
        src = open(repro_path).read()
        exec(compile(src, repro_path, "exec"), ns)
    except Exception:
        return "abstain"

    BuggyModule = ns.get("BuggyModule")
    if BuggyModule is None:
        return "abstain"

    # --- Pass 1: FakeTensorMode -------------------------------------------
    try:
        mod = BuggyModule()
        with FakeTensorMode():
            fake_inputs = {k: torch.empty(*shape) for k, shape in input_shapes.items()}
            try:
                _ = mod(**fake_inputs)
                faketensor_ok = True
            except (RuntimeError, ValueError) as e:
                msg = str(e)
                if any(s in msg for s in ("shape", "size", "dim", "broadcast",
                                           "matmul", "scalar type", "dtype",
                                           "expected", "must be")):
                    return "refuted"
                faketensor_ok = False
            except Exception:
                faketensor_ok = False
    except Exception:
        faketensor_ok = False

    # --- Pass 2: torch.compile in eager-fallback mode --------------------
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        mod2 = BuggyModule()
        compiled = torch.compile(mod2, mode="reduce-overhead", fullgraph=False)
        real_inputs = {k: torch.zeros(*shape) for k, shape in input_shapes.items()}
        try:
            _ = compiled(**real_inputs)
            return "verified"
        except (RuntimeError, ValueError) as e:
            msg = str(e)
            if any(s in msg for s in ("shape", "size", "dim", "broadcast",
                                       "matmul", "scalar type", "dtype",
                                       "expected", "must be")):
                return "refuted"
            return "abstain"
        except Exception:
            return "abstain"
    except Exception:
        # torch.compile failed to set up; fall back to FakeTensorMode result
        if faketensor_ok:
            return "verified"
        return "abstain"


# ---------------------------------------------------------------------------
# Pytea verdict
# ---------------------------------------------------------------------------

def _pytea_verdict(bug_id: str) -> str:
    return PYTEA_CACHE.get(bug_id, "n/a")


# ---------------------------------------------------------------------------
# Statistics: McNemar exact + BH-corrected Fisher
# ---------------------------------------------------------------------------

def _compute_stats(records: list[dict]) -> dict:
    """Compute pairwise McNemar exact p-values and BH-adjusted Fisher p-values.

    Ground truth: expected == "RP_0.99" → positive (bug present);
                  expected == "silent_verified" → negative (no bug);
                  expected == "abstain" → excluded from statistics.

    Tool verdict mapping to detected/not-detected:
      TG: "RP_0.99" or "RP_0.80" → detected; else → not detected
      compile: "refuted" → detected; else → not detected
      pytea: "refuted" → detected; else → not detected
    """
    try:
        import numpy as np
        from scipy.stats import fisher_exact
    except ImportError:
        return {"error": "scipy not available"}

    def _is_detected_tg(v: str) -> Optional[bool]:
        if v in ("RP_0.99", "RP_0.80"):
            return True
        if v in ("silent_verified", "low_confidence", "abstain"):
            return False
        return None

    def _is_detected_compile(v: str) -> Optional[bool]:
        if v == "refuted":
            return True
        if v in ("verified", "abstain", "n/a"):
            return False
        return None

    def _is_detected_pytea(v: str) -> Optional[bool]:
        if v == "refuted":
            return True
        if v in ("verified", "n/a", "abstain"):
            return False
        return None

    detectors = {
        "tg": _is_detected_tg,
        "compile": _is_detected_compile,
        "pytea": _is_detected_pytea,
    }
    verdict_keys = {
        "tg": "tg_verdict",
        "compile": "compile_verdict",
        "pytea": "pytea_verdict",
    }

    def _gt_positive(r: dict) -> Optional[bool]:
        if r["expected"] == "RP_0.99":
            return True
        if r["expected"] == "silent_verified":
            return False
        return None  # abstain items excluded

    # Pairwise McNemar (exact binomial) and Fisher
    tool_pairs = [("tg", "compile"), ("tg", "pytea"), ("compile", "pytea")]
    pairwise: dict = {}

    from scipy.stats import binomtest

    for t1, t2 in tool_pairs:
        n10 = n01 = n11 = n00 = 0
        for r in records:
            gt = _gt_positive(r)
            if gt is None:
                continue
            d1 = detectors[t1](r[verdict_keys[t1]])
            d2 = detectors[t2](r[verdict_keys[t2]])
            if d1 is None or d2 is None:
                continue
            if d1 and not d2:
                n10 += 1
            elif not d1 and d2:
                n01 += 1
            elif d1 and d2:
                n11 += 1
            else:
                n00 += 1
        # McNemar exact: Pr(X >= n10 or X <= n01) under Bin(n10+n01, 0.5)
        discordant = n10 + n01
        if discordant == 0:
            mcnemar_p = 1.0
        else:
            res = binomtest(min(n10, n01), discordant, 0.5, alternative="two-sided")
            mcnemar_p = float(res.pvalue)
        pairwise[f"{t1}_vs_{t2}"] = {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "mcnemar_exact_p": mcnemar_p,
        }

    # Fisher exact p-values for each tool vs ground truth, then BH
    fisher_ps: dict[str, float] = {}
    for tool in ("tg", "compile", "pytea"):
        tp = fp = tn = fn = 0
        for r in records:
            gt = _gt_positive(r)
            if gt is None:
                continue
            det = detectors[tool](r[verdict_keys[tool]])
            if det is None:
                continue
            if gt and det:
                tp += 1
            elif not gt and det:
                fp += 1
            elif gt and not det:
                fn += 1
            else:
                tn += 1
        table = [[tp, fp], [fn, tn]]
        _, p = fisher_exact(table, alternative="two-sided")
        fisher_ps[tool] = float(p)

    # BH correction
    tools_order = list(fisher_ps.keys())
    raw_ps = [fisher_ps[t] for t in tools_order]
    m = len(raw_ps)
    sorted_idx = sorted(range(m), key=lambda i: raw_ps[i])
    bh_ps = [1.0] * m
    for rank, idx in enumerate(sorted_idx, 1):
        bh_ps[idx] = min(1.0, raw_ps[idx] * m / rank)
    # Make monotone
    for i in range(m - 2, -1, -1):
        bh_ps[sorted_idx[i]] = min(bh_ps[sorted_idx[i]], bh_ps[sorted_idx[i + 1]])

    bh_corrected = {tools_order[i]: bh_ps[i] for i in range(m)}

    return {
        "pairwise_mcnemar": pairwise,
        "fisher_vs_groundtruth": fisher_ps,
        "bh_adjusted_fisher": bh_corrected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: TensorGuard vs torch.compile vs Pytea"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "benchmarks", "results", "headtohead_n15.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--stats-out",
        default=os.path.join(ROOT, "benchmarks", "results", "headtohead_n15_stats.json"),
        help="Output stats JSON path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    rows = []
    print(f"{'bug_id':12s}  {'expected':20s}  {'tg_verdict':20s}  {'compile_verdict':16s}  {'pytea_verdict'}")
    for item in manifest["items"]:
        bug_id = item["id"]
        repro_path = _resolve_path(item)
        src, input_shapes = _load_repro(repro_path)

        tg = _tg_verdict(src, input_shapes, repro_path)
        comp = _compile_verdict(repro_path, input_shapes)
        pytea = _pytea_verdict(bug_id)

        # Ensure all verdicts are non-empty strings
        tg = tg or "abstain"
        comp = comp or "abstain"
        pytea = pytea or "n/a"

        row = {
            "bug_id": bug_id,
            "expected": item.get("expected", ""),
            "bug_class": item.get("class", ""),
            "fragment": item.get("fragment", ""),
            "tg_verdict": tg,
            "compile_verdict": comp,
            "pytea_verdict": pytea,
        }
        rows.append(row)
        print(f"{bug_id:12s}  {item.get('expected',''):20s}  {tg:20s}  {comp:16s}  {pytea}")

    # Write CSV
    fieldnames = ["bug_id", "expected", "bug_class", "fragment",
                  "tg_verdict", "compile_verdict", "pytea_verdict"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)")

    # Compute and write stats
    stats = _compute_stats(rows)
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {args.stats_out}")

    # Summary
    print("\n=== Summary ===")
    from collections import Counter
    for tool, key in [("TG", "tg_verdict"), ("compile", "compile_verdict"), ("Pytea", "pytea_verdict")]:
        c = Counter(r[key] for r in rows)
        print(f"  {tool}: {dict(c)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
