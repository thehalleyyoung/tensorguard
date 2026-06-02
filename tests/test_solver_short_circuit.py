"""Step 47 -- solver-call avoidance via a fast syntactic short-circuit.

The bounded model checker discharges many per-step well-formedness and combined
safety obligations.  When the conjunction of safety constraints is valid on its
own (Z3's cheap simplifier rewrites it to literal True), its negation is
unsatisfiable in any context, so the expensive `solver.check()` is skipped.
This must (a) actually fire, and (b) never change a verdict.
"""
import textwrap

import pytest

from src.model_checker import (
    ConstraintVerifier,
    extract_computation_graph,
    verify_model,
)


def _stack(n_layers: int, dim: int = 32) -> str:
    init = "\n".join(
        "        self.l%da = nn.Linear(%d, %d)\n"
        "        self.l%db = nn.Linear(%d, %d)" % (i, dim, dim, i, dim, dim)
        for i in range(n_layers)
    )
    body = "\n".join(
        "        x = self.l%db(nn.functional.relu(self.l%da(x)))" % (i, i)
        for i in range(n_layers)
    )
    return ("import torch.nn as nn\n"
            "class M(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "%s\n"
            "    def forward(self, x):\n"
            "%s\n"
            "        return x\n" % (init, body))


def test_stats_exposes_syntactic_skip_counter():
    g = extract_computation_graph(_stack(3))
    v = ConstraintVerifier(g, {"x": (8, 32)})
    v.verify()
    stats = v.ctx.get_stats()
    assert "syntactic_skips" in stats


def test_short_circuit_actually_fires():
    # A concrete-shape stack produces many ground well-formedness obligations
    # that the simplifier discharges without the solver.
    g = extract_computation_graph(_stack(6))
    v = ConstraintVerifier(g, {"x": (8, 32)})
    v.verify()
    assert v.ctx.get_stats()["syntactic_skips"] > 0


def test_safe_model_still_safe():
    res = verify_model(_stack(4), input_shapes={"x": ("b", 32)})
    assert res.safe


def test_unsafe_model_still_unsafe():
    src = textwrap.dedent(
        """
        import torch.nn as nn
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(32, 16)
                self.b = nn.Linear(8, 4)   # 16 != 8 -> shape violation
            def forward(self, x):
                return self.b(self.a(x))
        """
    )
    res = verify_model(src, input_shapes={"x": ("batch", 32)})
    assert not res.safe


def test_certificate_mode_disables_short_circuit():
    # With certificate extraction enabled the short-circuit is bypassed so the
    # per-step proof replay path is preserved; the verdict is unchanged.
    g = extract_computation_graph(_stack(3))
    v = ConstraintVerifier(g, {"x": (8, 32)}, produce_certificates=True)
    v.verify()
    assert v.ctx.get_stats()["syntactic_skips"] == 0
