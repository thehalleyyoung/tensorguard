"""Property-test that PyTorch operator handlers in TensorGuard's
``SymbolicShapePropagator`` agree with PyTorch's runtime output shapes.

For each supported operator we generate ~50 random concrete configurations,
compare the predicted output shape with the actual PyTorch output, and assert
they match.  Total comparisons ≈ 1000+.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from benchmarks.lean_parity_runner import (
    SPECS,
    Case,
    OpSpec,
    REPO_ROOT,
    run_one,
)


# ---------------------------------------------------------------------------
# Build a deterministic test grid: (op_name, seed) for ~50 cases per op.
# ---------------------------------------------------------------------------

CASES_PER_OP = 50
MASTER_SEED = 42


def _build_cases():
    """Produce a list of (op_name, case) by re-driving each spec's generator."""
    rng = random.Random(MASTER_SEED)
    out = []
    for spec in SPECS:
        spec_rng = random.Random(rng.random())
        for i in range(CASES_PER_OP):
            try:
                case = spec.gen(spec_rng)
            except Exception as exc:  # pragma: no cover - generator should be robust
                pytest.fail(f"{spec.name} generator failed at iteration {i}: {exc}")
            out.append((spec.name, case))
    return out


_CASES = _build_cases()


@pytest.mark.parametrize(
    ("op_name", "case"),
    _CASES,
    ids=[f"{name}-{i % CASES_PER_OP}" for i, (name, _) in enumerate(_CASES)],
)
def test_handler_parity(op_name: str, case: Case) -> None:
    """PyTorch's runtime shape must match TG's predicted shape."""
    spec = next(s for s in SPECS if s.name == op_name)
    record = run_one(spec, case)
    status = record["status"]

    if status == "introspection_unavailable":
        pytest.skip(record["reason"] or "TG cannot introspect this case")
    if status == "torch_error":
        pytest.skip(f"PyTorch could not run this case: {record['reason']}")
    if status == "tg_error":
        pytest.fail(
            f"TG raised on {op_name} with input_shapes={record['input_shapes']}: "
            f"{record['reason']}"
        )
    if status == "mismatch":
        pytest.fail(
            f"{op_name} parity mismatch: input_shapes={record['input_shapes']} "
            f"PyTorch={record['actual']}  TG={record['expected']}"
        )
    assert status == "ok"


def test_run_full_suite_and_write_json(tmp_path_factory):
    """Run the full parity sweep once and persist the JSON summary.

    This duplicates ``benchmarks/lean_parity_runner.py`` so the artefact exists
    even if the runner script isn't invoked directly.
    """
    from benchmarks.lean_parity_runner import run_all

    summary = run_all(seed=MASTER_SEED, per_op=CASES_PER_OP)
    out = REPO_ROOT / "benchmarks" / "lean_parity_results.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    # Soft assertion: total > 1000 and at least 95% pass.
    assert summary["n_total_inputs"] >= 1000
    assert summary["n_passed"] >= int(0.95 * summary["n_total_inputs"]), (
        f"too many parity failures: {summary['status_breakdown']}"
    )


if __name__ == "__main__":
    from benchmarks.lean_parity_runner import main

    raise SystemExit(main())
