"""Step 59 — abstain accounting.

The engine is sound by abstention: outside the modeled fragment a detector
returns ``Top`` and emits no report.  These tests verify that those abstain
decisions are now *recorded* (a structured reason with category + location +
detector), that the ledger reports coverage metrics, and — crucially — that
recording is side-effect-only: it never changes which bugs fire (soundness and
the zero-regression contract are preserved).
"""

from __future__ import annotations

import ast

from src.symexec.abstain import AbstainCategory, AbstainLedger, AbstainReason
from src.symexec.engine import analyze_source
from src.symexec.interpreter import Interpreter


# -- ledger unit tests ---------------------------------------------------


def test_ledger_starts_empty():
    led = AbstainLedger()
    assert led.total == 0
    assert led.coverage() == {}
    assert led.by_detector() == {}
    assert led.summary() == "abstentions: 0"


def test_ledger_records_and_counts():
    led = AbstainLedger()
    led.record(AbstainReason(AbstainCategory.UNKNOWN_RANK, "matmul", line=3))
    led.record(AbstainReason(AbstainCategory.UNKNOWN_RANK, "matmul", line=5))
    led.record(AbstainReason(AbstainCategory.ELLIPSIS_PATTERN, "einsum", line=9))
    assert led.total == 3
    assert led.coverage() == {
        AbstainCategory.UNKNOWN_RANK: 2,
        AbstainCategory.ELLIPSIS_PATTERN: 1,
    }
    assert led.by_detector() == {"matmul": 2, "einsum": 1}
    assert led.categories() == {
        AbstainCategory.UNKNOWN_RANK,
        AbstainCategory.ELLIPSIS_PATTERN,
    }


def test_ledger_record_returns_none():
    led = AbstainLedger()
    assert led.record(AbstainReason(AbstainCategory.UNKNOWN_DIM, "broadcast")) is None


def test_ledger_summary_is_deterministic():
    led = AbstainLedger()
    led.record(AbstainReason(AbstainCategory.UNKNOWN_RANK, "matmul"))
    led.record(AbstainReason(AbstainCategory.ELLIPSIS_PATTERN, "einsum"))
    led.record(AbstainReason(AbstainCategory.UNKNOWN_RANK, "matmul"))
    # categories rendered sorted by value name → stable regardless of insert order
    assert led.summary() == "abstentions: 3 (ellipsis_pattern=1, unknown_rank=2)"


# -- interpreter helper --------------------------------------------------


def _run(src: str) -> Interpreter:
    interp = Interpreter(ast.parse(src))
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
            interp.run_function(node, args={}, self_val=None)
    return interp


def test_abstain_helper_records_and_returns_none():
    interp = Interpreter(ast.parse("x = 1"))
    node = ast.parse("y = 1").body[0]
    out = interp._abstain(AbstainCategory.UNKNOWN_DIM, "demo", node, "detail here")
    assert out is None
    assert interp._abstentions.total == 1
    r = interp._abstentions.reasons[0]
    assert r.category is AbstainCategory.UNKNOWN_DIM
    assert r.detector == "demo"
    assert r.detail == "detail here"
    assert r.line == node.lineno


# -- detector abstain sites record a reason ------------------------------


def test_matmul_unknown_rank_records_abstention():
    src = (
        "import torch\n"
        "def f(a: torch.Tensor, b: torch.Tensor):\n"
        "    return a @ b\n"  # ranks unknown → abstain
    )
    interp = _run(src)
    cats = interp._abstentions.categories()
    assert AbstainCategory.UNKNOWN_RANK in cats
    assert "matmul" in interp._abstentions.by_detector()


def test_einsum_ellipsis_records_abstention():
    src = (
        "import torch\n"
        "def f(a: torch.Tensor, b: torch.Tensor):\n"
        "    return torch.einsum('...ij,...jk->...ik', a, b)\n"
    )
    interp = _run(src)
    assert AbstainCategory.ELLIPSIS_PATTERN in interp._abstentions.categories()
    assert "einsum" in interp._abstentions.by_detector()


def test_reshape_unknown_shape_records_abstention():
    src = (
        "import torch\n"
        "def f(x: torch.Tensor):\n"
        "    return x.reshape(2, 3)\n"  # x shape/rank unknown → abstain
    )
    interp = _run(src)
    assert AbstainCategory.UNKNOWN_SHAPE in interp._abstentions.categories()
    assert "reshape" in interp._abstentions.by_detector()


def test_cat_non_literal_sequence_records_abstention():
    src = (
        "import torch\n"
        "def f(xs):\n"
        "    return torch.cat(xs, dim=0)\n"  # xs not a literal seq → abstain
    )
    interp = _run(src)
    assert AbstainCategory.NON_LITERAL_PATTERN in interp._abstentions.categories()
    assert "cat_stack" in interp._abstentions.by_detector()


# -- recording is side-effect-only (soundness preserved) -----------------


def test_recording_does_not_change_bugs_reported():
    # A genuine forced broadcast failure must still be reported, and no
    # abstain-recording at the same site suppresses it.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3)\n"
        "    b = torch.zeros(2)\n"
        "    return a + b\n"  # forced broadcast mismatch (3 vs 2)
    )
    res = analyze_source(src)
    kinds = {b.kind.name for b in res.bugs}
    assert "BROADCAST_MISMATCH" in kinds


def test_no_false_positive_when_abstaining():
    # Wholly-unknown dims must abstain (record) and NOT report a bug.
    src = (
        "import torch\n"
        "def f(a: torch.Tensor, b: torch.Tensor):\n"
        "    return a @ b\n"
    )
    res = analyze_source(src)
    assert all(b.kind.name != "MATMUL_DIM_MISMATCH" for b in res.bugs)
    # but the abstention was accounted for
    assert res.abstentions.total >= 1


def test_result_exposes_abstention_ledger():
    src = (
        "import torch\n"
        "def f(a: torch.Tensor, b: torch.Tensor):\n"
        "    return a @ b\n"
    )
    res = analyze_source(src)
    assert isinstance(res.abstentions, AbstainLedger)
    assert res.abstentions.total >= 1
    # summary is renderable
    assert res.abstentions.summary().startswith("abstentions:")


def test_clean_concrete_program_few_or_no_abstentions():
    # Fully concrete, modeled code: detectors stay inside the fragment.
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(3, 4)\n"
        "    return a @ b\n"  # both contracting dims known → no matmul abstain
    )
    res = analyze_source(src)
    assert all(
        r.detector != "matmul" for r in res.abstentions.reasons
    )
