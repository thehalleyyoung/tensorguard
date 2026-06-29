"""Step 48 — input-shape inference fallback: when a function/method has no
caller or demo, seed its entry parameters from their type annotations (a sound
contract) so rank/shape checks engage.  Also analyze ``forward``/``__call__``
of demo-less classes as entry points, with ``forward``'s first positional
argument seeded as a (rank-unknown) tensor by nn.Module convention."""

import ast

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec.interpreter import (
    _infer_from_annotation,
    _tensor_from_shape_spec,
)
from src.symexec.values import TensorVal, IntVal, FloatVal, BoolVal, StrVal


RANK = SymBugKind.RANK_INDEX_ERROR


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


def _ann(expr_src):
    """Parse a single annotation expression to an AST node."""
    return ast.parse(expr_src, mode="eval").body


# --------------------------------------------------------------------------
# Annotation -> abstract value (unit)
# --------------------------------------------------------------------------
def test_plain_tensor_annotation():
    v = _infer_from_annotation(_ann("Tensor"))
    assert isinstance(v, TensorVal) and v.rank is None


def test_torch_attribute_tensor():
    v = _infer_from_annotation(_ann("torch.Tensor"))
    assert isinstance(v, TensorVal) and v.rank is None


def test_jaxtyping_rank_and_dims():
    v = _infer_from_annotation(_ann('Float[Tensor, "b c h w"]'))
    assert isinstance(v, TensorVal) and v.rank == 4


def test_jaxtyping_const_dim():
    v = _infer_from_annotation(_ann('Float[Tensor, "b 3 h"]'))
    assert v.rank == 3
    assert v.dim(1).value == 3  # the literal 3 is a concrete dim
    assert v.dim(0).value is None  # named dim has unknown size (sound)


def test_jaxtyping_variadic_unknown_rank():
    v = _infer_from_annotation(_ann('Float[Tensor, "*batch c"]'))
    assert isinstance(v, TensorVal) and v.rank is None


def test_torchtyping_rank():
    v = _infer_from_annotation(_ann('TensorType["b", "c", "h"]'))
    assert isinstance(v, TensorVal) and v.rank == 3


def test_scalar_annotations():
    assert isinstance(_infer_from_annotation(_ann("int")), IntVal)
    assert isinstance(_infer_from_annotation(_ann("float")), FloatVal)
    assert isinstance(_infer_from_annotation(_ann("bool")), BoolVal)
    assert isinstance(_infer_from_annotation(_ann("str")), StrVal)


def test_unknown_annotation_abstains():
    assert _infer_from_annotation(_ann("Optional[Tensor]")) is None
    assert _infer_from_annotation(_ann("List[int]")) is None
    assert _infer_from_annotation(_ann("MyType")) is None


def test_shape_spec_helper():
    v = _tensor_from_shape_spec("b c h w")
    assert v.rank == 4
    assert _tensor_from_shape_spec("").rank == 0
    assert _tensor_from_shape_spec("a ... b").rank is None


# --------------------------------------------------------------------------
# Free function: annotation seeds entry analysis
# --------------------------------------------------------------------------
def test_free_function_jaxtyping_rank_bug():
    src = """
import torch
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "b c h w"]):
    return x[0, 1, 2, 3, 4]   # 5 indices on rank-4 -> bug
"""
    assert RANK in _kinds(src)


def test_free_function_plain_tensor_no_fp():
    # rank unknown -> over-indexing cannot be forced -> no report
    src = """
import torch
from torch import Tensor
def f(x: Tensor):
    return x[0, 1, 2, 3, 4]
"""
    assert RANK not in _kinds(src)


def test_free_function_jaxtyping_valid_no_fp():
    src = """
import torch
from jaxtyping import Float
from torch import Tensor
def f(x: Float[Tensor, "b c h w"]):
    return x[0, 1]
"""
    assert RANK not in _kinds(src)


def test_torchtyping_free_function_bug():
    src = """
import torch
from torchtyping import TensorType
def f(x: TensorType["b", "c", "h"]):
    return x[0, 1, 2, 3]   # 4 indices on rank-3 -> bug
"""
    assert RANK in _kinds(src)


# --------------------------------------------------------------------------
# Method / forward as an entry point (no demo)
# --------------------------------------------------------------------------
def test_forward_annotation_rank_bug():
    src = """
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Float[Tensor, "b c h"]):
        return x[0, 1, 2, 3]   # 4 indices on rank-3
"""
    assert RANK in _kinds(src)


def test_forward_unannotated_tensor_seed_downstream_bug():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.reshape(2, 3, 4)   # rank-3 result
        return y[0, 1, 2, 3]     # 4 indices -> bug
"""
    assert RANK in _kinds(src)


def test_forward_valid_no_fp():
    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.reshape(2, 3, 4)[0, 1]
"""
    assert RANK not in _kinds(src)


def test_call_method_entry():
    src = """
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
class Op(nn.Module):
    def __call__(self, x: Float[Tensor, "b c"]):
        return x[0, 1, 2]   # 3 indices on rank-2
"""
    assert RANK in _kinds(src)


def test_no_forward_entry_when_demo_present():
    # A shipped demo means forward is exercised by the demo path, not seeded.
    # The demo passes a rank-2 tensor, which is valid for the single index.
    src = """
import torch
import torch.nn as nn
class Net(nn.Module):
    def forward(self, x):
        return x[0]
if __name__ == "__main__":
    Net().forward(torch.zeros(3, 4))
"""
    assert RANK not in _kinds(src)
