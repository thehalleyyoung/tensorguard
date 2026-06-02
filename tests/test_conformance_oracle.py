"""Step 33 — conformance oracle in CI.

Cross-checks TensorGuard's shape transfer functions against real PyTorch
execution: for a battery of single-op modules it samples concrete input shapes,
runs the genuine ``torch`` forward pass for the ground-truth output shape, and
compares it to TensorGuard's predicted shape.

The contract enforced here is the one a *sound* verifier must satisfy:

* **Zero DISAGREE** — TensorGuard must never predict a concrete output shape
  that differs from what PyTorch actually produces. A disagreement is a
  soundness bug.
* **Zero TRACE_FAIL** on the supported battery.
* **High conformance** — the transfer functions must actually resolve shapes
  for the core ops rather than abstaining everywhere (a degenerate verifier
  that abstains on everything would be sound but useless).

This oracle is what drove the fixes in this step: it caught the fx frontend
modelling ``x.mean(dim=...)``/``x.sum(dim=...)`` as shape-preserving, ``x @ w``
as an activation, and functional ``embedding`` as shape-preserving — each a
confidently-wrong shape.
"""

import pytest

torch = pytest.importorskip("torch")

from reproducibility.conformance_oracle import (  # noqa: E402
    _default_cases,
    run_oracle,
)


@pytest.fixture(scope="module")
def report():
    return run_oracle(_default_cases())


def test_no_disagreements(report):
    """TensorGuard never predicts a concrete shape that torch contradicts."""
    assert report.disagreements == [], (
        "Conformance disagreements (predicted concrete shape != torch):\n"
        + "\n".join(
            f"  {r.name} in={r.input_shape} pred={r.predicted} actual={r.actual}"
            for r in report.disagreements
        )
    )


def test_no_trace_failures(report):
    assert report.trace_fails == [], (
        "Unexpected trace failures: "
        + ", ".join(r.name for r in report.trace_fails)
    )


def test_high_conformance(report):
    # The battery covers ~25 distinct ops; the great majority must resolve to a
    # concrete, correct shape (not merely abstain).
    assert report.conformant >= 25, report.summary()


def test_every_case_ran(report):
    # Sanity: the battery is non-trivial and every result is one of the
    # expected statuses.
    assert len(report.results) >= 30
    valid = {"CONFORMANT", "ABSTAINED", "DISAGREE", "TRACE_FAIL"}
    assert all(r.status in valid for r in report.results)


@pytest.mark.parametrize("op_name", [
    "linear", "conv2d", "conv1d", "maxpool2d", "avgpool2d",
    "adaptive_avgpool2d", "batchnorm2d", "layernorm", "flatten", "embedding",
    "matmul", "cat_dim1", "stack_dim0", "mean_keepdim", "sum_reduce",
    "transpose", "permute", "unsqueeze", "softmax", "factory_zeros",
])
def test_core_op_is_conformant(report, op_name):
    """Each core transfer function must be conformant on at least one sample
    (and never disagree)."""
    statuses = [r.status for r in report.results if r.name == op_name]
    assert statuses, f"{op_name} not in battery"
    assert "DISAGREE" not in statuses
    assert "CONFORMANT" in statuses, f"{op_name}: {statuses}"
