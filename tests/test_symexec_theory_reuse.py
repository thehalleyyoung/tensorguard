"""Step 53 — theory reuse: wire the SMT shape theories into the executor.

Two theories are wired into the symbolic executor so that *symbolic*-dimension
shape faults — which the purely-concrete checks abstain on — are decided by Z3:

* **Reshape**: when the element count is not concretely known, the receiver
  shape and target sizes are lowered into the existing
  ``src.smt.reshape_theory.check_reshape_compatible`` contract, which proves a
  reshape impossible for *every* positive concretization (e.g. ``[a, b]`` →
  ``[a, b, 2]``).
* **Broadcast**: a symbolic trailing-dim pair ``(da, db)`` is broadcastable iff
  ``da == db ∨ da == 1 ∨ db == 1`` is satisfiable under the path constraints;
  when all three disjuncts are infeasible the mismatch is forced.

Soundness: every symbolic report rests on a Z3-proved impossibility under
dimensions ``>= 1``; free symbolic dims (no constraints) never fire.
"""

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec import smt_bridge as B

import pytest


RESHAPE = SymBugKind.RESHAPE_SIZE_MISMATCH
BROADCAST = SymBugKind.BROADCAST_MISMATCH

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


# --------------------------------------------------------------------------
# reshape theory reuse (symbolic element count)
# --------------------------------------------------------------------------

@z3only
def test_reshape_symbolic_add_factor_is_incompatible():
    # [a, b] -> [a, b, 2] : numel a*b vs 2*a*b, impossible for all a,b >= 1.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    return x.reshape(x.size(0), x.size(1), 2)
"""
    assert RESHAPE in _kinds(src)


@z3only
def test_reshape_symbolic_permutation_is_compatible():
    # [a, b] -> [b, a] preserves the element count: no report.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    return x.reshape(x.size(1), x.size(0))
"""
    assert _kinds(src) == []


@z3only
def test_reshape_symbolic_infer_slot_is_compatible():
    # [a, b] -> [a, -1] is always satisfiable.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    return x.reshape(x.size(0), -1)
"""
    assert _kinds(src) == []


@z3only
def test_reshape_symbolic_view_alias_also_checked():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b c"]):
    return x.view(x.size(0), x.size(1), x.size(2), 3)
"""
    assert RESHAPE in _kinds(src)


def test_reshape_fully_symbolic_unconstrained_no_false_positive():
    # x.reshape(a, b) where a, b are independent fresh dims is satisfiable.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"], y: Float[Tensor, "c d"]):
    return x.reshape(y.size(0), y.size(1))
"""
    assert RESHAPE not in _kinds(src)


# --------------------------------------------------------------------------
# broadcast theory reuse (symbolic trailing dims under path constraints)
# --------------------------------------------------------------------------

@z3only
def test_broadcast_symbolic_forced_mismatch():
    # a != b, a != 1, b != 1  ⇒  x + y cannot broadcast for any sizes.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a"], y: Float[Tensor, "b"]):
    if x.size(0) != y.size(0):
        if x.size(0) != 1:
            if y.size(0) != 1:
                return x + y
    return x
"""
    assert BROADCAST in _kinds(src)


@z3only
def test_broadcast_symbolic_neq_only_is_not_forced():
    # a != b alone does not force a mismatch: a == 1 (or b == 1) still broadcasts.
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a"], y: Float[Tensor, "b"]):
    if x.size(0) != y.size(0):
        return x + y
    return x
"""
    assert BROADCAST not in _kinds(src)


def test_broadcast_free_symbolic_no_false_positive():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a"], y: Float[Tensor, "b"]):
    return x + y
"""
    assert BROADCAST not in _kinds(src)


@z3only
def test_broadcast_concrete_mismatch_still_fires():
    # The concrete path must remain intact alongside the symbolic one.
    src = """
import torch
if __name__ == "__main__":
    a = torch.zeros(3, 4)
    b = torch.zeros(3, 5)
    c = a + b
"""
    assert BROADCAST in _kinds(src)
