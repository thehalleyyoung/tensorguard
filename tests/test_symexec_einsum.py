"""Step 26 — torch.einsum modeling: arity & repeated-index dim-mismatch
detection, output shape/rank inference, and sound abstention on the parts of
the einsum language we don't model (ellipsis, non-literal equation)."""

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec.interpreter import _parse_einsum_eq, _einsum_implicit_out
from src.symexec.values import StrVal, TensorVal
from src.symexec.symdim import SymDim


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def test_repeated_index_dim_mismatch():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 5)
    return torch.einsum("ij,jk->ik", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH in _kinds(src)


def test_subscript_rank_arity_mismatch():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    return torch.einsum("ijk", a)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH in _kinds(src)


def test_implicit_output_dim_mismatch():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 5)
    return torch.einsum("ij,jk", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# No false positives
# --------------------------------------------------------------------------
def test_valid_matmul_no_fp():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(3, 5)
    return torch.einsum("ij,jk->ik", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH not in _kinds(src)


def test_valid_batched_no_fp():
    src = """
import torch
def f():
    a = torch.zeros(8, 2, 3)
    b = torch.zeros(8, 3, 5)
    return torch.einsum("bij,bjk->bik", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH not in _kinds(src)


def test_ellipsis_abstains():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 5)
    return torch.einsum("...ij,...jk->...ik", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH not in _kinds(src)


def test_symbolic_dims_abstain():
    # unknown operand shapes -> no statically-known conflict -> no report
    src = """
import torch
def f(a, b):
    return torch.einsum("ij,jk->ik", a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH not in _kinds(src)


# --------------------------------------------------------------------------
# Output shape / rank inference (no false positive, correct propagation)
# --------------------------------------------------------------------------
def test_output_shape_inferred():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(3, 5)
    c = torch.einsum("ij,jk->ik", a, b)
    return c.reshape(2, 5)  # element-count must match 2*5=10
"""
    # reshape preserves count: no RESHAPE bug, no einsum bug
    assert _kinds(src) == []


def test_output_shape_mismatched_reshape_flags():
    src = """
import torch
def f():
    a = torch.zeros(2, 3)
    b = torch.zeros(3, 5)
    c = torch.einsum("ij,jk->ik", a, b)  # -> (2, 5), 10 elems
    return c.reshape(3, 4)  # 12 elems
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


# --------------------------------------------------------------------------
# Parse-helper units
# --------------------------------------------------------------------------
def test_parse_explicit():
    assert _parse_einsum_eq("ij,jk->ik") == (["ij", "jk"], "ik")


def test_parse_implicit():
    assert _parse_einsum_eq("ij,jk") == (["ij", "jk"], None)


def test_parse_ellipsis_none():
    assert _parse_einsum_eq("...ij,...jk") is None


def test_parse_strips_spaces():
    assert _parse_einsum_eq(" i j , j k -> i k ") == (["ij", "jk"], "ik")


def test_implicit_out_alphabetical_singletons():
    assert _einsum_implicit_out(["ij", "jk"]) == "ik"
    assert _einsum_implicit_out(["bij", "bjk"]) == "ik"  # b twice -> contracted


def test_non_literal_equation_abstains():
    src = """
import torch
def f(eq):
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 5)
    return torch.einsum(eq, a, b)
"""
    assert SymBugKind.EINSUM_DIM_MISMATCH not in _kinds(src)
