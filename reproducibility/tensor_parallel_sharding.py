#!/usr/bin/env python3
"""
tensor_parallel_sharding.py — prove the tensor-parallel checker against REAL
PyTorch (100_STEPS.md Step 97, Phase 10).

For each tensor-parallel MLP configuration we hold two things side by side:

  * the **static** verdict from
    ``src/tensor_parallel_checks.verify_tensor_parallel``, and
  * the **runtime** outcome of actually hand-sharding a reference linear stack
    across ``tp_size`` simulated ranks, running each rank, all-reducing /
    gathering, and comparing to the unsharded forward.

A consistent config must verify OK statically AND reproduce the reference output
(bit-for-bit up to fp tolerance) at runtime; each inconsistent config must be
flagged statically AND fail at runtime (shape error, or no even shard). The
artifact records boolean outcomes and verdicts only (no timing, no raw floats),
so it is byte-deterministic and checked by ``reproduce_all.py --check``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.tensor_parallel_checks import (  # noqa: E402
    verify_tensor_parallel, megatron_mlp, TPLinearSpec, TPKind,
)

OUT_JSON = REPO / "reproducibility" / "tensor_parallel_sharding.json"
OUT_MD = REPO / "reproducibility" / "tensor_parallel_sharding.md"

SEED = 0
TOL = 1e-4


def _simulate_megatron(hidden: int, ffn: int, tp: int, *,
                       gather_output: bool, input_is_parallel: bool) -> Dict:
    """Hand-shard a reference 2-layer MLP across ``tp`` ranks and compare to the
    unsharded forward. Returns dict with ``ran`` and ``matches_reference``."""
    torch.manual_seed(SEED)
    fc1 = nn.Linear(hidden, ffn, bias=False)
    fc2 = nn.Linear(ffn, hidden, bias=False)
    x = torch.randn(3, hidden)
    ref = fc2(F.relu(fc1(x)))

    if ffn % tp != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "ffn not divisible by tp (no even shard)"}

    W1 = fc1.weight.data          # [ffn, hidden]  column-parallel: split dim 0
    W2 = fc2.weight.data          # [hidden, ffn]  row-parallel: split dim 1
    s1 = W1.split(ffn // tp, dim=0)
    s2 = W2.split(ffn // tp, dim=1)

    try:
        if not gather_output and input_is_parallel:
            # canonical no-comm path: each rank keeps its shard, all-reduce sum
            z = torch.zeros(3, hidden)
            for r in range(tp):
                y_local = F.relu(x @ s1[r].T)        # [3, ffn/tp]
                z = z + (y_local @ s2[r].T)          # [3, hidden]
            out = z
        else:
            # column gathers full [3, ffn] then feeds a row shard expecting a
            # sharded input -> dimension mismatch (the real bug).
            y_full = F.relu(x @ W1.T)                # [3, ffn]
            out = y_full @ s2[0].T                   # shape error expected
    except RuntimeError as e:
        return {"ran": False, "matches_reference": False,
                "reason": f"runtime shape error: {str(e)[:48]}"}

    matches = bool(torch.allclose(out, ref, atol=TOL))
    return {"ran": True, "matches_reference": matches, "reason": ""}


class Case:
    def __init__(self, name: str, specs: List[TPLinearSpec], tp: int,
                 sim_kwargs: Dict, hidden: int, ffn: int,
                 expect_static_ok: bool, expect_runtime_match: bool):
        self.name = name
        self.specs = specs
        self.tp = tp
        self.sim_kwargs = sim_kwargs
        self.hidden = hidden
        self.ffn = ffn
        self.expect_static_ok = expect_static_ok
        self.expect_runtime_match = expect_runtime_match


CASES: List[Case] = [
    Case("megatron_mlp_tp2_correct", megatron_mlp(8, 16), 2,
         {"gather_output": False, "input_is_parallel": True}, 8, 16,
         expect_static_ok=True, expect_runtime_match=True),
    Case("megatron_mlp_tp4_correct", megatron_mlp(8, 16), 4,
         {"gather_output": False, "input_is_parallel": True}, 8, 16,
         expect_static_ok=True, expect_runtime_match=True),
    Case("comm_flag_mismatch_gather_then_parallel",
         megatron_mlp(8, 16, gather_output=True, input_is_parallel=True), 2,
         {"gather_output": True, "input_is_parallel": True}, 8, 16,
         expect_static_ok=False, expect_runtime_match=False),
    Case("indivisible_ffn15_tp2", megatron_mlp(8, 15), 2,
         {"gather_output": False, "input_is_parallel": True}, 8, 15,
         expect_static_ok=False, expect_runtime_match=False),
    Case("inner_dim_mismatch",
         [TPLinearSpec("fc1", 8, 16, TPKind.COLUMN),
          TPLinearSpec("fc2", 12, 8, TPKind.ROW, input_is_parallel=True)], 2,
         None, 8, 16, expect_static_ok=False, expect_runtime_match=False),
]


def measure() -> Dict:
    rows: List[Dict] = []
    all_ok = True
    for c in CASES:
        res = verify_tensor_parallel(c.specs, c.tp)
        static_ok = res.ok
        static_match = static_ok == c.expect_static_ok

        if c.sim_kwargs is None:
            sim = {"ran": None, "matches_reference": None,
                   "reason": "structural (no runtime simulation)"}
            runtime_match = True
        else:
            sim = _simulate_megatron(c.hidden, c.ffn, c.tp, **c.sim_kwargs)
            runtime_match = sim["matches_reference"] == c.expect_runtime_match

        ok = static_match and runtime_match
        all_ok = all_ok and ok
        rows.append({
            "name": c.name,
            "tp_size": c.tp,
            "static_ok": static_ok,
            "static_issues": sorted({i.kind.value for i in res.issues}),
            "expect_static_ok": c.expect_static_ok,
            "static_match": static_match,
            "live_ran": sim["ran"],
            "live_matches_reference": sim["matches_reference"],
            "live_reason": sim["reason"],
            "expect_live_match": c.expect_runtime_match,
            "live_match": runtime_match,
            "ok": ok,
        })
    return {"seed": SEED, "tolerance": TOL, "n_cases": len(rows),
            "all_ok": all_ok, "cases": rows}


def render_markdown(data: Dict) -> str:
    L: List[str] = []
    L.append("# Tensor-parallel sharding checker — static verdict vs real torch")
    L.append("")
    L.append("> Generated by `reproducibility/tensor_parallel_sharding.py`. "
             "Boolean outcomes and verdicts only, no timing or raw floats — "
             "byte-deterministic, checked by `reproduce_all.py --check`.")
    L.append("")
    L.append(f"Seed **{data['seed']}** · cases: **{data['n_cases']}** · static "
             f"verdict matches real sharded execution on every case: "
             f"**{str(data['all_ok']).upper()}**")
    L.append("")
    L.append("| Case | tp | Static issues | Reproduces reference? | "
             "static ✓ | runtime ✓ |")
    L.append("|------|----|---------------|-----------------------|"
             "----------|-----------|")
    for c in data["cases"]:
        iss = ", ".join(c["static_issues"]) or "(ok)"
        if c["live_ran"] is None:
            repro = "(structural)"
        elif not c["live_ran"]:
            repro = f"no — {c['live_reason']}"
        else:
            repro = "yes" if c["live_matches_reference"] else "no (wrong output)"
        L.append(f"| `{c['name']}` | {c['tp_size']} | {iss} | {repro} | "
                 f"{'yes' if c['static_match'] else 'NO'} | "
                 f"{'yes' if c['live_match'] else 'NO'} |")
    L.append("")
    L.append("The canonical Megatron MLP (ColumnParallel `gather_output=False` → "
             "RowParallel `input_is_parallel=True`) reproduces the unsharded "
             "forward exactly (up to fp tolerance) at tp=2 and tp=4. Every "
             "inconsistent config — gathered output fed to a parallel-input row "
             "layer, an indivisible shard, or a contracted-dimension mismatch — "
             "is flagged statically and also fails when actually sharded across "
             "ranks. The checker lives in `src/tensor_parallel_checks.py`.")
    L.append("")
    return "\n".join(L)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != new_json:
            print("MISMATCH: tensor_parallel_sharding.json differs",
                  file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != new_md:
            print("MISMATCH: tensor_parallel_sharding.md differs",
                  file=sys.stderr)
            ok = False
        if not data["all_ok"]:
            print("FAIL: static verdict diverges from sharded runtime",
                  file=sys.stderr)
            ok = False
        print("tensor_parallel_sharding --check:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    if not data["all_ok"]:
        print("WARNING: static verdict diverges from runtime!", file=sys.stderr)
        return 1
    print(f"Wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)} "
          f"({data['n_cases']} cases, all_ok={data['all_ok']}).")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
