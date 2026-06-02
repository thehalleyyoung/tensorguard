"""Step 46 -- end-to-end verification latency budgets.

A static verifier has to be *fast enough to live in CI*.  This harness profiles
the full `verify_model` pipeline (source parse + graph extraction + bounded
model checking + Z3) across three size tiers and enforces a per-model **latency
budget**:

  * **small**  -- a handful of layers (classifiers, tiny CNNs);    budget   3 s
  * **medium** -- a dozen-block transformer-style stack;           budget  12 s
  * **large**  -- a deep 40-block stack (~120 computation steps);  budget  30 s

Two artifacts are produced.  The committed JSON/MD *manifest* records only
**deterministic** facts -- each model's tier, extracted step count, and budget
-- because the source/AST frontend is torch-version-independent, so the manifest
is byte-reproducible everywhere (`--check`).  The wall-clock latency itself is
machine-dependent and is therefore measured live by `--gate`, which fails the
build if any model blows its budget, and printed for information but never
committed.

Steps 47-50 (Z3 context reuse, constraint simplification, incremental and
parallel verification) tighten these budgets; this step establishes the
baseline and the regression gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

JSON_PATH = os.path.join(HERE, "latency_budgets.json")
MD_PATH = os.path.join(HERE, "latency_budgets.md")


def _mlp_block_model(n_layers: int, dim: int = 64) -> str:
    """A deterministic n-block Linear/ReLU stack used for medium/large tiers."""
    init = "\n".join(
        "        self.l%da = nn.Linear(%d, %d)\n"
        "        self.l%db = nn.Linear(%d, %d)" % (i, dim, dim, i, dim, dim)
        for i in range(n_layers)
    )
    body = "\n".join(
        "        x = self.l%db(nn.functional.relu(self.l%da(x)))" % (i, i)
        for i in range(n_layers)
    )
    return (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "%s\n"
        "    def forward(self, x):\n"
        "%s\n"
        "        return x\n" % (init, body)
    )


_SMALL_MLP = (
    "import torch.nn as nn\n"
    "class M(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(16, 32)\n"
    "        self.fc2 = nn.Linear(32, 10)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(nn.functional.relu(self.fc1(x)))\n"
)

_SMALL_CNN = (
    "import torch.nn as nn\n"
    "class M(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.c1 = nn.Conv2d(3, 16, 3, padding=1)\n"
    "        self.c2 = nn.Conv2d(16, 32, 3, padding=1)\n"
    "        self.pool = nn.AdaptiveAvgPool2d((1, 1))\n"
    "        self.fc = nn.Linear(32, 10)\n"
    "    def forward(self, x):\n"
    "        x = nn.functional.relu(self.c1(x))\n"
    "        x = nn.functional.relu(self.c2(x))\n"
    "        x = self.pool(x).flatten(1)\n"
    "        return self.fc(x)\n"
)


# (name, tier, source, input_shapes, budget_seconds)
def corpus() -> List[Tuple[str, str, str, Dict[str, tuple], float]]:
    return [
        ("small_mlp", "small", _SMALL_MLP, {"x": ("b", 16)}, 3.0),
        ("small_cnn", "small", _SMALL_CNN, {"x": ("b", 3, 32, 32)}, 3.0),
        ("medium_stack_12", "medium", _mlp_block_model(12),
         {"x": ("b", 64)}, 12.0),
        ("large_stack_40", "large", _mlp_block_model(40),
         {"x": ("b", 64)}, 30.0),
    ]


def _extract_steps(source: str) -> int:
    from src.model_checker import extract_computation_graph

    return len(extract_computation_graph(source).steps)


def measure() -> List[Dict[str, object]]:
    """Time end-to-end verification for every model; never raises."""
    from src.model_checker import verify_model

    out: List[Dict[str, object]] = []
    for name, tier, src, shapes, budget in corpus():
        steps = _extract_steps(src)
        t0 = time.perf_counter()
        err = None
        try:
            verify_model(src, input_shapes=shapes)
        except Exception as exc:  # pragma: no cover - defensive
            err = "%s: %s" % (type(exc).__name__, str(exc)[:160])
        elapsed = time.perf_counter() - t0
        out.append({
            "model": name, "tier": tier, "steps": steps,
            "budget_s": budget, "latency_s": round(elapsed, 3),
            "within_budget": err is None and elapsed <= budget,
            "error": err,
        })
    return out


def manifest() -> Dict[str, object]:
    """Deterministic, byte-reproducible manifest (no timings)."""
    rows = [
        {"model": name, "tier": tier, "steps": _extract_steps(src),
         "budget_s": budget}
        for name, tier, src, _shapes, budget in corpus()
    ]
    rows.sort(key=lambda r: (r["tier"], r["model"]))
    return {
        "meta": {
            "generated_by": "evaluation/latency_budgets.py",
            "command": "PYTHONPATH=. python3 evaluation/latency_budgets.py",
            "python_version": "%d.%d" % sys.version_info[:2],
            "note": ("Manifest records deterministic budgets and graph sizes "
                     "only; wall-clock latency is machine-dependent and is "
                     "checked live by --gate."),
        },
        "tiers": sorted({r["tier"] for r in rows}),
        "models": rows,
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(man: Dict[str, object]) -> str:
    lines = [
        "# End-to-end verification latency budgets",
        "",
        ("Per-model latency budgets for the full `verify_model` pipeline "
         "(source parse, graph extraction, bounded model checking, Z3), grouped "
         "by size tier. The committed manifest is deterministic (budgets and "
         "extracted step counts only); measured wall-clock latency is enforced "
         "live by `make latency-budgets-gate`."),
        "",
        "| Model | Tier | Steps | Budget (s) |",
        "|-------|------|-------|------------|",
    ]
    for r in man["models"]:
        lines.append("| `%s` | %s | %d | %.1f |" % (
            r["model"], r["tier"], r["steps"], r["budget_s"]))
    lines.append("")
    return "\n".join(lines)


def gate() -> int:
    rows = measure()
    over = [r for r in rows if not r["within_budget"]]
    for r in rows:
        flag = "ok" if r["within_budget"] else "OVER"
        print("  [%s] %-16s tier=%-6s steps=%3d  %.3fs / %.1fs budget" % (
            flag, r["model"], r["tier"], r["steps"],
            r["latency_s"], r["budget_s"]))
    if over:
        print("LATENCY BUDGET GATE FAILED: %d model(s) over budget" % len(over))
        for r in over:
            extra = (" (error: %s)" % r["error"]) if r["error"] else ""
            print("  - %s: %.3fs > %.1fs%s" % (
                r["model"], r["latency_s"], r["budget_s"], extra))
        return 1
    print("latency budget gate PASS: %d model(s) within budget" % len(rows))
    return 0


def run(check: bool = False, write: bool = True) -> int:
    man = manifest()
    text = _dumps(man)

    if check:
        if not os.path.exists(JSON_PATH):
            print("latency_budgets.json missing; run the harness first")
            return 1
        if open(JSON_PATH).read() != text:
            print("latency_budgets.json is stale; run `make latency-budgets`")
            return 1
        md = render_markdown(man)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("latency_budgets.md is stale; run `make latency-budgets`")
            return 1
        print("latency budgets manifest up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(man))
    print("latency budgets manifest written: %d models across tiers %s" % (
        len(man["models"]), ", ".join(man["tiers"])))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed manifest is byte-identical.")
    ap.add_argument("--gate", action="store_true",
                    help="Measure latency live and fail on any budget breach.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
