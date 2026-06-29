"""Tests for Step 22: view/reshape element-count preservation.

Fires only when the receiver's element count and every requested size are
statically known, so the RuntimeError is forced.  Abstains otherwise.
"""

import pytest

from src.symexec import analyze_source, SymBugKind


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_reshape_product_mismatch_is_reported():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.view(2, 5)\n"
    )
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


def test_reshape_exact_match_is_silent():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.view(4, 6)\n"
    )
    assert _kinds(src) == []


def test_reshape_minus_one_divisible_is_silent():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.view(-1, 6)\n"
    )
    assert _kinds(src) == []


def test_reshape_minus_one_non_divisible_is_reported():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.view(-1, 5)\n"
    )
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


def test_reshape_tuple_arg_mismatch_is_reported():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.reshape((5, 5))\n"
    )
    assert SymBugKind.RESHAPE_SIZE_MISMATCH in _kinds(src)


def test_reshape_unknown_numel_abstains():
    src = "def f(x):\n    return x.view(2, 5)\n"
    assert _kinds(src) == []


def test_reshape_unknown_target_abstains():
    src = (
        "import torch\n"
        "def f(n):\n"
        "    x = torch.zeros(2, 3, 4)\n"
        "    return x.view(n, 6)\n"
    )
    assert _kinds(src) == []
