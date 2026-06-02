"""Step 55 -- optimized reshape divisibility constraints.

The reshape element-count check is `prod(inputs) == prod(targets)` with every
dimension >= 1. Products of integer *variables* are nonlinear integer
arithmetic, which is what makes Z3 blow up on high-rank reshapes. This step adds
an algebraic reduction -- fold concrete dims into one integer and cancel shared
symbolic factors (each >= 1, hence nonzero, so cancellation is exact) -- so the
hot paths become a pure divisibility test that needs no solver, and only a
genuinely under-determined symbolic remainder reaches Z3 on a smaller formula.

These tests prove (a) the reduction never changes a verdict (soundness:
incompatible only when truly impossible), and (b) the concrete and single-infer
hot paths, and fully-cancelling symbolic reshapes, cost ZERO solver calls.
"""
import textwrap

import pytest

from src.tensor_shapes import ShapeDim, TensorShape
from src.smt import reshape_theory as rt
from src.model_checker import verify_model


def _C(t):
    return TensorShape.from_tuple(t)


def _S(*names):
    return TensorShape(tuple(ShapeDim(n) for n in names))


def _compat(inp, dims):
    return rt.check_reshape_compatible(inp, dims) is None


# ---------------------------------------------------------------------------
# 1. Verdict correctness across concrete / infer / symbolic cases.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,dims,ok", [
    (_C((2, 3, 4)), (6, 4), True),       # 24 == 24
    (_C((2, 3, 4)), (24,), True),        # full flatten
    (_C((2, 3, 4)), (-1, 4), True),      # infer 6
    (_C((2, 3, 4)), (5, 5), False),      # 24 != 25
    (_C((2, 3, 4)), (-1, 5), False),     # 24 % 5 != 0
    (_C((2, 3, 4)), (-1, 7), False),     # 24 % 7 != 0
    (_C((2, 3, 4)), (2, -1), True),      # infer 12
    (_S("B", 5), ("B", 3), False),       # cancel B -> 5 != 3
    (_S("B", 6), ("B", 2, 3), True),     # cancel B -> 6 == 6
    (_S("B", 6), ("B", -1), True),       # cancel B -> infer 6
    (_S("a", "a"), (3,), False),         # a*a == 3 impossible
    (_S("a", "a"), (4,), True),          # a=2 works
])
def test_verdicts(inp, dims, ok):
    assert _compat(inp, dims) is ok


# ---------------------------------------------------------------------------
# 2. Concrete reshapes are decided WITHOUT the solver.
# ---------------------------------------------------------------------------

def test_concrete_reshape_is_solver_free():
    rt.reset_reshape_counters()
    inp = _C((8, 16, 32, 32, 3, 4))
    for _ in range(50):
        rt.check_reshape_compatible(inp, (8, -1))
        rt.check_reshape_compatible(inp, (8 * 16 * 32 * 32 * 3 * 4,))
        rt.check_reshape_compatible(inp, (7,))  # incompatible
    assert rt.reshape_solver_call_count() == 0
    assert rt.reshape_analytic_decision_count() == 150


def test_single_infer_is_solver_free():
    rt.reset_reshape_counters()
    assert _compat(_C((4, 6)), (-1, 3)) is True       # infer 8
    assert _compat(_C((4, 6)), (-1, 5)) is False      # 24 % 5
    assert rt.reshape_solver_call_count() == 0


# ---------------------------------------------------------------------------
# 3. Shared symbolic factors cancel; fully-cancelling cases are solver-free.
# ---------------------------------------------------------------------------

def test_shared_factor_cancellation_is_solver_free():
    rt.reset_reshape_counters()
    # All symbolic dims shared and identical -> cancels to 1 == 1, no solver.
    assert _compat(_S("B", "H", "W", "C"), ("B", "H", "W", "C")) is True
    # Shared B, residual constants -> still analytic.
    assert _compat(_S("B", 6), ("B", 2, 3)) is True
    assert _compat(_S("B", 5), ("B", 3)) is False
    assert rt.reshape_solver_call_count() == 0
    assert rt.reshape_analytic_decision_count() == 3


def test_irreducible_symbolic_uses_solver_but_is_correct():
    rt.reset_reshape_counters()
    # a*a == 3 cannot be cancelled to constants -> solver, and is UNSAT.
    assert _compat(_S("a", "a"), (3,)) is False
    assert rt.reshape_solver_call_count() == 1


# ---------------------------------------------------------------------------
# 4. Soundness guard: independent opaque (_-prefixed) dims never falsely flag.
# ---------------------------------------------------------------------------

def test_opaque_dims_do_not_couple_and_never_false_positive():
    # Two unrelated opaque dims must NOT be cancelled/coupled; the reshape is
    # satisfiable for some assignment, so we must abstain (return compatible).
    inp = TensorShape((ShapeDim("_dyn0"), ShapeDim(2), ShapeDim("_dyn1")))
    assert _compat(inp, ("_dyn2", "_dyn3")) is True


# ---------------------------------------------------------------------------
# 5. End-to-end: a real model with a genuinely bad reshape is still caught.
# ---------------------------------------------------------------------------

def test_end_to_end_bad_reshape_caught():
    src = """
        import torch
        import torch.nn as nn
        class M(nn.Module):
            def forward(self, x):
                # x is (4, 10); 40 elements cannot become (4, 11) = 44
                return x.reshape(4, 11)
    """
    res = verify_model(textwrap.dedent(src), input_shapes={"x": (4, 10)})
    assert res.safe is False


def test_end_to_end_good_reshape_passes():
    src = """
        import torch
        import torch.nn as nn
        class M(nn.Module):
            def forward(self, x):
                return x.reshape(2, 20)   # 40 == 40
    """
    res = verify_model(textwrap.dedent(src), input_shapes={"x": (4, 10)})
    assert res.safe is True
