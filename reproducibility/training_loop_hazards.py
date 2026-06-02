#!/usr/bin/env python3
"""
training_loop_hazards.py — validate the training-loop hazard analyzer against
REAL PyTorch execution (100_STEPS.md Step 96, Phase 10).

For each curated training-step pattern we hold two things side by side:

  * the **static** verdict from ``src/training_loop_checks.analyze_training_loop``
    on the source text, and
  * the **runtime** symptom obtained by actually executing the equivalent loop
    in seeded PyTorch (backward raised? parameters changed? gradients
    accumulated across steps?).

A clean loop must produce no hazard statically AND train at runtime; each buggy
loop must produce exactly the expected hazard statically AND exhibit the
matching real failure at runtime. The artifact records the boolean outcomes and
verdicts only (no timing), so it is byte-deterministic and checked by
``reproduce_all.py --check``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.training_loop_checks import analyze_training_loop, HazardKind  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "training_loop_hazards.json"
OUT_MD = REPO / "reproducibility" / "training_loop_hazards.md"

SEED = 0


def _fresh():
    torch.manual_seed(SEED)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    return model, opt, x, y


# --- runtime reference implementations (one per case) ----------------------
def _rt_clean() -> Dict:
    model, opt, x, y = _fresh()
    before = model.weight.detach().clone()
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    return {"backward_raised": False,
            "param_changed": bool(not torch.allclose(before, model.weight))}


def _rt_detach() -> Dict:
    model, opt, x, y = _fresh()
    opt.zero_grad()
    loss = ((model(x).detach() - y) ** 2).mean()
    raised = False
    try:
        loss.backward()
    except RuntimeError:
        raised = True
    return {"backward_raised": raised, "param_changed": False}


def _rt_nograd_block() -> Dict:
    model, opt, x, y = _fresh()
    opt.zero_grad()
    with torch.no_grad():
        loss = ((model(x) - y) ** 2).mean()
    raised = False
    try:
        loss.backward()
    except RuntimeError:
        raised = True
    return {"backward_raised": raised, "param_changed": False}


def _rt_missing_zero_grad() -> Dict:
    model, opt, x, y = _fresh()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    g1 = model.weight.grad.detach().clone()
    loss2 = ((model(x) - y) ** 2).mean()
    loss2.backward()                       # no zero_grad in between
    g2 = model.weight.grad.detach().clone()
    return {"backward_raised": False,
            "grads_accumulated": bool(not torch.allclose(g1, g2))}


def _rt_missing_step() -> Dict:
    model, opt, x, y = _fresh()
    before = model.weight.detach().clone()
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()                        # no opt.step()
    return {"backward_raised": False,
            "param_changed": bool(not torch.allclose(before, model.weight))}


def _rt_badorder() -> Dict:
    model, opt, x, y = _fresh()
    before = model.weight.detach().clone()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.zero_grad()                        # wipes grads before step
    opt.step()
    return {"backward_raised": False,
            "param_changed": bool(not torch.allclose(before, model.weight))}


# --- cases -----------------------------------------------------------------
CLEAN_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        optimizer.step()
'''
DETACH_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = ((out.detach() - y) ** 2).mean()
        loss.backward()
        optimizer.step()
'''
NOGRAD_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        with torch.no_grad():
            loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
'''
NOZERO_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        out = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        optimizer.step()
'''
NOSTEP_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
'''
BADORDER_SRC = '''
def train(model, loader, optimizer):
    for x, y in loader:
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
'''


class Case:
    def __init__(self, name: str, source: str, expected_kinds: List[HazardKind],
                 runtime: Callable[[], Dict], runtime_expect: Dict,
                 runtime_supported: bool = True):
        self.name = name
        self.source = source
        self.expected_kinds = expected_kinds
        self.runtime = runtime
        self.runtime_expect = runtime_expect
        self.runtime_supported = runtime_supported


CASES: List[Case] = [
    Case("clean", CLEAN_SRC, [], _rt_clean,
         {"backward_raised": False, "param_changed": True}),
    Case("gradient_flow_break_detach", DETACH_SRC,
         [HazardKind.GRADIENT_FLOW_BREAK], _rt_detach,
         {"backward_raised": True, "param_changed": False}),
    Case("gradient_flow_break_no_grad_block", NOGRAD_SRC,
         [HazardKind.GRADIENT_FLOW_BREAK], _rt_nograd_block,
         {"backward_raised": True, "param_changed": False}),
    Case("missing_zero_grad", NOZERO_SRC,
         [HazardKind.MISSING_ZERO_GRAD], _rt_missing_zero_grad,
         {"backward_raised": False, "grads_accumulated": True}),
    Case("missing_optimizer_step", NOSTEP_SRC,
         [HazardKind.MISSING_OPTIMIZER_STEP], _rt_missing_step,
         {"backward_raised": False, "param_changed": False}),
    Case("backward_before_zero_grad", BADORDER_SRC,
         [HazardKind.BACKWARD_BEFORE_ZERO_GRAD], _rt_badorder,
         {"backward_raised": False, "param_changed": False}),
]


def measure() -> Dict:
    rows: List[Dict] = []
    all_ok = True
    for c in CASES:
        hazards = analyze_training_loop(c.source)
        static_kinds = sorted({h.kind.value for h in hazards})
        expected = sorted({k.value for k in c.expected_kinds})
        static_ok = static_kinds == expected

        runtime_obs = c.runtime() if c.runtime_supported else {}
        runtime_ok = all(runtime_obs.get(k) == v
                         for k, v in c.runtime_expect.items()) \
            if c.runtime_supported else True

        ok = static_ok and runtime_ok
        all_ok = all_ok and ok
        rows.append({
            "name": c.name,
            "expected_hazards": expected,
            "static_hazards": static_kinds,
            "static_ok": static_ok,
            "live_supported": c.runtime_supported,
            "live_expected": c.runtime_expect,
            "live_observed": runtime_obs,
            "live_ok": runtime_ok,
            "ok": ok,
        })
    return {
        "seed": SEED,
        "n_cases": len(rows),
        "all_ok": all_ok,
        "cases": rows,
    }


def render_markdown(data: Dict) -> str:
    L: List[str] = []
    L.append("# Training-loop hazard analyzer — static verdict vs real PyTorch")
    L.append("")
    L.append("> Generated by `reproducibility/training_loop_hazards.py`. "
             "Boolean outcomes and verdicts only, no timing — "
             "byte-deterministic, checked by `reproduce_all.py --check`.")
    L.append("")
    L.append(f"Seed **{data['seed']}** · cases: **{data['n_cases']}** · "
             f"static verdict matches real runtime behaviour on every case: "
             f"**{str(data['all_ok']).upper()}**")
    L.append("")
    L.append("| Case | Static hazards | Runtime symptom (real torch) | "
             "static ✓ | runtime ✓ |")
    L.append("|------|----------------|------------------------------|"
             "----------|-----------|")
    for c in data["cases"]:
        haz = ", ".join(c["static_hazards"]) or "(none)"
        sym = ", ".join(f"{k}={v}" for k, v in c["live_observed"].items()) \
            or ("(cuda-only)" if not c["live_supported"] else "(none)")
        L.append(f"| `{c['name']}` | {haz} | {sym} | "
                 f"{'yes' if c['static_ok'] else 'NO'} | "
                 f"{'yes' if c['live_ok'] else 'NO'} |")
    L.append("")
    L.append("Every clean loop is silent statically and trains at runtime; "
             "every buggy loop raises exactly the expected hazard and exhibits "
             "the matching real failure (backward raises, parameters never "
             "change, or gradients accumulate across steps). The analyzer lives "
             "in `src/training_loop_checks.py`.")
    L.append("")
    return "\n".join(L)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != new_json:
            print("MISMATCH: training_loop_hazards.json differs", file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != new_md:
            print("MISMATCH: training_loop_hazards.md differs", file=sys.stderr)
            ok = False
        if not data["all_ok"]:
            print("FAIL: static verdict diverges from runtime", file=sys.stderr)
            ok = False
        print("training_loop_hazards --check:", "OK" if ok else "FAILED")
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
