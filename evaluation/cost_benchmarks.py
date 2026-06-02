"""Step 53 -- pip-install / import / analysis cost benchmarks.

TensorGuard's headline dependency claim is that **static analysis does not pull
in the deep-learning runtime**: a model is verified from its *source* (AST +
Z3), never by importing torch and instantiating it. This harness measures, in
fresh subprocesses for isolation:

  * **import cost** -- time to `from src.model_checker import verify_model`, and
    crucially an assertion that doing so does NOT import `torch` (the heavy
    dependency, ~1 s on its own). This is the load-bearing fact behind the
    "negligible PyTorch import cost" story.
  * **analysis cost** -- end-to-end `verify_model` latency on a small and a
    medium model, with the same subprocess still never having imported torch.

Two artifacts are produced (mirroring the latency-budgets harness, Step 46).
The committed JSON/MD *manifest* records only deterministic, machine-independent
facts -- which costs are gated, the torch-free invariant, and the import-time
ceiling -- so it is byte-reproducible everywhere (`--check`). Wall-clock costs
are machine-dependent and are measured live by `--gate`, which fails the build
if the core import exceeds its ceiling or if importing the analysis API drags in
torch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

JSON_PATH = os.path.join(HERE, "cost_benchmarks.json")
MD_PATH = os.path.join(HERE, "cost_benchmarks.md")

# Ceilings (seconds). Generous relative to observed values (~0.16 s import,
# ~0.03 s small analysis) so the gate flags only genuine regressions, not
# machine jitter.
IMPORT_CEILING_S = 2.0
SMALL_ANALYSIS_CEILING_S = 5.0
MEDIUM_ANALYSIS_CEILING_S = 20.0


_SMALL = (
    "import torch.nn as nn\n"
    "class M(nn.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(32, 64)\n"
    "        self.fc2 = nn.Linear(64, 10)\n"
    "    def forward(self, x):\n"
    "        return self.fc2(nn.functional.relu(self.fc1(x)))\n"
)


def _medium(n_layers: int = 12, dim: int = 64) -> str:
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


# A self-contained probe script run in a *fresh* interpreter. It emits a single
# JSON line so the parent can read import/analysis costs and the torch-free
# invariant without contaminating its own import state.
_PROBE = r"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, %(root)r)

t = time.perf_counter()
from src.model_checker import verify_model
import_s = time.perf_counter() - t
torch_after_import = "torch" in sys.modules

small = %(small)r
medium = %(medium)r

t = time.perf_counter()
verify_model(small, input_shapes={"x": (4, 32)})
small_s = time.perf_counter() - t

t = time.perf_counter()
verify_model(medium, input_shapes={"x": ("b", 64)})
medium_s = time.perf_counter() - t

torch_after_analysis = "torch" in sys.modules

print(json.dumps({
    "import_s": import_s,
    "small_analysis_s": small_s,
    "medium_analysis_s": medium_s,
    "torch_after_import": torch_after_import,
    "torch_after_analysis": torch_after_analysis,
}))
"""


def _probe() -> Dict[str, object]:
    """Run the cost probe in a fresh subprocess; return its parsed JSON."""
    script = _PROBE % {
        "root": ROOT,
        "small": _SMALL,
        "medium": _medium(),
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    # The probe prints exactly one JSON line on stdout; ignore any stderr noise.
    line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(line)
    return data


def manifest() -> Dict[str, object]:
    """Deterministic, byte-reproducible manifest (no timings)."""
    return {
        "meta": {
            "generated_by": "evaluation/cost_benchmarks.py",
            "command": "PYTHONPATH=. python3 evaluation/cost_benchmarks.py",
            "python_version": "%d.%d" % sys.version_info[:2],
            "note": ("Manifest records deterministic invariants and ceilings "
                     "only; wall-clock import/analysis costs are machine-"
                     "dependent and are checked live by --gate."),
        },
        "invariants": {
            "import_is_torch_free": True,
            "analysis_is_torch_free": True,
        },
        "ceilings_s": {
            "import": IMPORT_CEILING_S,
            "small_analysis": SMALL_ANALYSIS_CEILING_S,
            "medium_analysis": MEDIUM_ANALYSIS_CEILING_S,
        },
    }


def _dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def render_markdown(man: Dict[str, object]) -> str:
    c = man["ceilings_s"]
    lines = [
        "# Import / analysis cost benchmarks",
        "",
        ("TensorGuard verifies a model from its *source* (AST plus Z3) and "
         "never imports the deep-learning runtime to do so. This harness "
         "measures import and analysis cost in fresh subprocesses and asserts "
         "the torch-free invariant. The committed manifest is deterministic "
         "(invariants and ceilings only); measured wall-clock cost is enforced "
         "live by `make cost-benchmarks-gate`."),
        "",
        "| Cost | Ceiling (s) | torch-free |",
        "|------|-------------|-----------|",
        "| import `verify_model` | %.1f | yes |" % c["import"],
        "| analyze small model | %.1f | yes |" % c["small_analysis"],
        "| analyze medium model | %.1f | yes |" % c["medium_analysis"],
        "",
    ]
    return "\n".join(lines)


def gate() -> int:
    data = _probe()
    failures: List[str] = []

    if not data["torch_after_import"]:
        print("  [ok]   importing verify_model does NOT import torch")
    else:
        print("  [FAIL] importing verify_model imported torch")
        failures.append("import pulled in torch")

    if not data["torch_after_analysis"]:
        print("  [ok]   analysis does NOT import torch")
    else:
        print("  [FAIL] analysis imported torch")
        failures.append("analysis pulled in torch")

    checks = [
        ("import", data["import_s"], IMPORT_CEILING_S),
        ("small_analysis", data["small_analysis_s"], SMALL_ANALYSIS_CEILING_S),
        ("medium_analysis", data["medium_analysis_s"],
         MEDIUM_ANALYSIS_CEILING_S),
    ]
    for name, val, ceil in checks:
        flag = "ok" if val <= ceil else "FAIL"
        print("  [%s] %-16s %.3fs / %.1fs ceiling" % (flag, name, val, ceil))
        if val > ceil:
            failures.append("%s %.3fs over %.1fs" % (name, val, ceil))

    if failures:
        print("COST BENCHMARK GATE FAILED:")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("cost benchmark gate PASS")
    return 0


def run(check: bool = False, write: bool = True) -> int:
    man = manifest()
    text = _dumps(man)

    if check:
        if not os.path.exists(JSON_PATH):
            print("cost_benchmarks.json missing; run the harness first")
            return 1
        if open(JSON_PATH).read() != text:
            print("cost_benchmarks.json is stale; run `make cost-benchmarks`")
            return 1
        md = render_markdown(man)
        if not os.path.exists(MD_PATH) or open(MD_PATH).read() != md:
            print("cost_benchmarks.md is stale; run `make cost-benchmarks`")
            return 1
        print("cost benchmarks manifest up to date")
        return 0

    if write:
        with open(JSON_PATH, "w") as fh:
            fh.write(text)
        with open(MD_PATH, "w") as fh:
            fh.write(render_markdown(man))
    print("cost benchmarks manifest written")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Verify the committed manifest is byte-identical.")
    ap.add_argument("--gate", action="store_true",
                    help="Measure costs live and fail on a ceiling/invariant "
                         "breach.")
    args = ap.parse_args()
    if args.gate:
        return gate()
    return run(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
