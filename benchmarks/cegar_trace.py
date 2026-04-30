#!/usr/bin/env python3
"""Generate a CEGAR refinement trace for the paper.

We pick a small, illustrative example whose forward shape is
parametric in an unstated input precondition and run the
``run_shape_cegar`` loop with ``max_iterations=10``.  For each
iteration we record:

* iteration index,
* number of counterexamples returned by the constraint verifier,
* how many were classified spurious vs. real,
* the predicates added in this iteration (pretty-printed),
* per-iteration wall time in ms.

We also dump the final discovered predicates and the loop verdict.

The example is a single ``nn.Linear(768, 10)`` consumer of a tensor
``x`` with symbolic input shape ``(batch, features)``.  CEGAR must
discover the precondition ``features == 768``.  This is the toy
example used in the docstring of ``run_shape_cegar`` and is a
faithful, minimal demonstration of the loop's behaviour.

Output: ``benchmarks/cegar_trace.json``.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.shape_cegar import run_shape_cegar  # noqa: E402

SOURCE = textwrap.dedent("""\
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(768, 10)
        def forward(self, x):
            return self.fc(x)
""")


def main() -> int:
    result = run_shape_cegar(
        SOURCE,
        input_shapes={"x": ("batch", "features")},
        max_iterations=10,
    )
    trace = []
    for rec in result.iteration_log:
        trace.append({
            "iteration": rec.iteration,
            "num_violations": rec.num_violations,
            "num_spurious": rec.num_spurious,
            "num_real": rec.num_real,
            "predicates_added": [p.pretty() for p in rec.predicates_added],
            "time_ms": round(rec.time_ms, 2),
        })
    out = {
        "example": "nn.Linear(768, 10) on input x : (batch, features)",
        "max_iterations": 10,
        "verdict": result.verdict.name,
        "final_status": result.final_status.name,
        "iterations_run": result.iterations,
        "discovered_predicates": [p.pretty()
                                   for p in result.discovered_predicates],
        "total_time_ms": round(result.total_time_ms, 2),
        "iteration_log": trace,
    }
    p = REPO_ROOT / "benchmarks" / "cegar_trace.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
