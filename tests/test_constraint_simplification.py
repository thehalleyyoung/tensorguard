"""Step 49 -- pre-solve constraint simplification pass.

Before transition constraints reach Z3 they are constant-folded; conjuncts that
fold to literal True are vacuous and dropped, shrinking the accumulated SMT
assertion set.  This must fire on real models and never change a verdict.
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


def test_stats_exposes_folded_counter():
    g = extract_computation_graph(_stack(2))
    v = ConstraintVerifier(g, {"x": ("b", 32)})
    v.verify()
    assert "folded_constraints" in v.ctx.get_stats()


def test_folding_actually_fires():
    g = extract_computation_graph(_stack(8))
    v = ConstraintVerifier(g, {"x": ("b", 32)})
    v.verify()
    assert v.ctx.get_stats()["folded_constraints"] > 0


def test_safe_verdict_preserved():
    assert verify_model(_stack(5), input_shapes={"x": ("b", 32)}).safe


def test_unsafe_verdict_preserved():
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
    assert not verify_model(src, input_shapes={"x": ("batch", 32)}).safe


def test_concrete_and_symbolic_agree():
    # Folding is most aggressive on concrete dims; the verdict must match the
    # symbolic run.
    src = _stack(4)
    assert (verify_model(src, input_shapes={"x": (8, 32)}).safe
            == verify_model(src, input_shapes={"x": ("b", 32)}).safe)
