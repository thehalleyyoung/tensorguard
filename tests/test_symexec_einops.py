"""Step 30 — einops library-contract transfers: rearrange/reduce/repeat
pattern-string axis-name algebra.  Detects rank/group mismatch, duplicate axes,
undefined output axes, and composed-group decomposition mismatches; infers the
output shape for downstream flow; abstains soundly outside the modeled fragment.
"""

from src.symexec.engine import analyze_source
from src.symexec.bugs import SymBugKind
from src.symexec.interpreter import (
    _parse_einops_axes,
    _parse_einops_pattern,
    _einops_names,
)

EINOPS = SymBugKind.EINOPS_PATTERN_MISMATCH


def _kinds(src):
    return [b.kind for b in analyze_source(src).bugs]


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def test_rank_mismatch():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4)
    return rearrange(x, "b c -> c b")
"""
    assert EINOPS in _kinds(src)


def test_undefined_output_axis_rearrange():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3)
    return rearrange(x, "b c -> b c h")
"""
    assert EINOPS in _kinds(src)


def test_undefined_output_axis_reduce():
    src = """
import torch
from einops import reduce
def f():
    x = torch.zeros(2, 3)
    return reduce(x, "b c -> b k", "mean")
"""
    assert EINOPS in _kinds(src)


def test_duplicate_axis():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4)
    return rearrange(x, "b b c -> b c")
"""
    assert EINOPS in _kinds(src)


def test_decomposition_product_mismatch():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 10)
    return rearrange(x, "b (h w) -> b h w", h=3, w=3)
"""
    assert EINOPS in _kinds(src)


def test_decomposition_divisibility_mismatch():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 10)
    return rearrange(x, "b (h w) -> b h w", h=3)
"""
    assert EINOPS in _kinds(src)


def test_reduce_rank_mismatch():
    src = """
import torch
from einops import reduce
def f():
    x = torch.zeros(2, 3, 4)
    return reduce(x, "b c -> b", "mean")
"""
    assert EINOPS in _kinds(src)


# --------------------------------------------------------------------------
# No false positives
# --------------------------------------------------------------------------
def test_valid_rearrange_no_fp():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4, 5)
    return rearrange(x, "b c h w -> b (h w) c")
"""
    assert EINOPS not in _kinds(src)


def test_valid_decomposition_no_fp():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 12)
    return rearrange(x, "b (h w) -> b h w", h=3, w=4)
"""
    assert EINOPS not in _kinds(src)


def test_valid_decomposition_inferred_factor_no_fp():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 12)
    return rearrange(x, "b (h w) -> b h w", h=3)
"""
    assert EINOPS not in _kinds(src)


def test_repeat_new_axis_with_size_no_fp():
    src = """
import torch
from einops import repeat
def f():
    x = torch.zeros(2, 3)
    return repeat(x, "b c -> b c k", k=4)
"""
    assert EINOPS not in _kinds(src)


def test_qualified_einops_reduce_no_fp():
    src = """
import torch
import einops
def f():
    x = torch.zeros(2, 3, 4)
    return einops.reduce(x, "b c h -> b c", "mean")
"""
    assert EINOPS not in _kinds(src)


def test_ellipsis_abstains():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4)
    return rearrange(x, "... c -> ... c")
"""
    assert EINOPS not in _kinds(src)


def test_non_literal_pattern_abstains():
    src = """
import torch
from einops import rearrange
def f(pat):
    x = torch.zeros(2, 3, 4)
    return rearrange(x, pat)
"""
    assert EINOPS not in _kinds(src)


def test_symbolic_input_rank_no_fp():
    # unknown input rank -> no forced failure
    src = """
import torch
from einops import rearrange
def f(x):
    return rearrange(x, "b c -> c b")
"""
    assert EINOPS not in _kinds(src)


def test_functools_reduce_untouched():
    src = """
import functools
def f():
    return functools.reduce(lambda a, b: a + b, [1, 2, 3])
"""
    assert EINOPS not in _kinds(src)


# --------------------------------------------------------------------------
# Output-shape inference propagates downstream
# --------------------------------------------------------------------------
def test_output_shape_reshape_mismatch_flags():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4, 5)
    y = rearrange(x, "b c h w -> b (h w) c")  # (2, 20, 3) -> 120 elems
    return y.reshape(2, 7)  # 14 != 120
"""
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


def test_output_shape_reshape_ok_no_fp():
    src = """
import torch
from einops import rearrange
def f():
    x = torch.zeros(2, 3, 4, 5)
    y = rearrange(x, "b c h w -> b (h w) c")  # 120 elems
    return y.reshape(2, 60)
"""
    assert _kinds(src) == []


# --------------------------------------------------------------------------
# Parse-helper units
# --------------------------------------------------------------------------
def test_parse_axes_simple():
    assert _parse_einops_axes("b c h w") == [["b"], ["c"], ["h"], ["w"]]


def test_parse_axes_composition():
    assert _parse_einops_axes("b (h w) c") == [["b"], ["h", "w"], ["c"]]


def test_parse_axes_literal():
    assert _parse_einops_axes("b 1 c") == [["b"], [1], ["c"]]


def test_parse_axes_ellipsis_none():
    assert _parse_einops_axes("... c") is None


def test_parse_axes_nested_none():
    assert _parse_einops_axes("b ((h w))") is None


def test_parse_pattern_split():
    assert _parse_einops_pattern("b (h w) -> b h w") == (
        [["b"], ["h", "w"]],
        [["b"], ["h"], ["w"]],
    )


def test_parse_pattern_no_arrow_none():
    assert _parse_einops_pattern("b c h") is None


def test_einops_names_flatten():
    assert _einops_names([["b"], ["h", "w"], [1], ["c"]]) == ["b", "h", "w", "c"]
