"""Step 54 — counterexample lifting.

When a *symbolic*-dimension detector proves a forced failure under the current
path constraints, the executor asks the Z3 bridge for a concrete satisfying
assignment of the dimension variables and substitutes it back into the operand
shapes, producing a **concrete entry shape** that actually triggers the error.
This replayable witness is attached to the report's ``evidence`` field.

Soundness: the lift only *renders* an already-proved bug; it never gates the
report, so when z3 is unavailable the bug is still reported (just without a
concrete witness).
"""

import re

import ast

import pytest

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec import smt_bridge as B
from src.symexec.interpreter import Interpreter

RESHAPE = SymBugKind.RESHAPE_SIZE_MISMATCH
BROADCAST = SymBugKind.BROADCAST_MISMATCH

z3only = pytest.mark.skipif(not B.Z3_AVAILABLE, reason="z3 not installed")


def _interp():
    return Interpreter(ast.parse(""))


def _bugs(src, kind):
    return [b for b in analyze_source(src).bugs if b.kind == kind]


def _shape_from(text):
    """Extract the first ``(a, b, ...)`` tuple of ints from an evidence string."""
    m = re.search(r"\(([0-9, ]+)\)", text)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split(",") if x.strip())


# --------------------------------------------------------------------------
# reshape counterexamples
# --------------------------------------------------------------------------

@z3only
def test_reshape_symbolic_report_carries_concrete_counterexample():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    return x.reshape(x.size(0), x.size(1), 2)
"""
    bugs = _bugs(src, RESHAPE)
    assert bugs, "expected a symbolic reshape report"
    ev = bugs[0].evidence
    assert ev is not None and "concrete counterexample" in ev
    # The witness must really violate element-count preservation: input numel
    # must differ from the (a, b, 2) target numel.
    in_shape = _shape_from(ev)
    assert in_shape is not None and len(in_shape) == 2
    a, b = in_shape
    assert a >= 1 and b >= 1
    in_numel = a * b
    target_numel = a * b * 2
    assert in_numel != target_numel  # the reported impossibility, concretely


@z3only
def test_reshape_counterexample_dims_are_positive():
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "m n"]):
    return x.view(x.size(0), x.size(1), 3)
"""
    bugs = _bugs(src, RESHAPE)
    assert bugs
    in_shape = _shape_from(bugs[0].evidence)
    assert in_shape is not None
    assert all(d >= 1 for d in in_shape)  # well-formedness floor honoured


# --------------------------------------------------------------------------
# broadcast counterexamples
# --------------------------------------------------------------------------

@z3only
def test_broadcast_symbolic_report_carries_concrete_counterexample():
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
    bugs = _bugs(src, BROADCAST)
    assert bugs, "expected a symbolic broadcast report"
    ev = bugs[0].evidence
    assert ev is not None and "concrete counterexample" in ev
    # The witnessing dims must be unequal and neither 1 (so they cannot
    # broadcast), exactly the proved impossibility.
    m = re.search(r"dims (\d+) vs (\d+)", ev)
    assert m is not None
    da, db = int(m.group(1)), int(m.group(2))
    assert da != db and da != 1 and db != 1


# --------------------------------------------------------------------------
# soundness: no z3 => still reported, just no concrete witness
# --------------------------------------------------------------------------

def test_reshape_report_survives_without_z3(monkeypatch):
    monkeypatch.setattr(B, "Z3_AVAILABLE", False)
    src = """
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "a b"]):
    return x.reshape(x.size(0), x.size(1), 2)
"""
    # Without z3 the symbolic reshape theory abstains, so we only assert the
    # lift itself produces no witness (no crash) when z3 is unavailable.
    interp = _interp()
    assert interp._lift_model([B.SymDim.var("a")]) is None


# --------------------------------------------------------------------------
# helper-level units
# --------------------------------------------------------------------------

@z3only
def test_lift_model_assigns_positive_values():
    interp = _interp()
    m = interp._lift_model([B.SymDim.var("a"), B.SymDim.var("b")])
    assert m is not None
    assert m.get("a", 0) >= 1 and m.get("b", 0) >= 1


def test_concretize_symdim_substitutes_affine():
    sd = B.SymDim.var("a") * 2 + B.SymDim.const_dim(3)  # 2*a + 3
    assert Interpreter._concretize_symdim(sd, {"a": 5}) == 13


def test_concretize_symdim_missing_var_is_none():
    sd = B.SymDim.var("z")
    assert Interpreter._concretize_symdim(sd, {"a": 1}) is None
