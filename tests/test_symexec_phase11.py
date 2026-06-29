"""Tests for Phase 11 concrete data/flow detectors.

Step 101 — builtin ``len()`` modeling + off-by-one indexing.
Step 102 — non-positive (negative) tensor dimension construction.
"""

import pytest

from src.symexec import analyze_source, SymBugKind


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


# ── Step 101: len() + off-by-one ─────────────────────────────────────────────
def test_list_index_at_len_is_off_by_one():
    src = "def f():\n    xs = [1, 2, 3]\n    return xs[len(xs)]\n"
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)


def test_list_index_len_minus_one_is_safe():
    src = "def f():\n    xs = [1, 2, 3]\n    return xs[len(xs) - 1]\n"
    assert _kinds(src) == []


def test_tuple_index_at_len_is_off_by_one():
    src = "def f():\n    t = (1, 2)\n    return t[len(t)]\n"
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)


def test_unknown_length_len_index_abstains():
    src = "def f(xs):\n    return xs[len(xs)]\n"
    assert _kinds(src) == []


def test_len_of_string_constant():
    # ``s[len(s)]`` would be out of range; here we just confirm len resolves and
    # there is no crash / false positive on a safe access.
    src = "def f():\n    xs = [0, 1, 2, 3]\n    return xs[len(xs) - 2]\n"
    assert _kinds(src) == []


# ── Step 102: negative dimension construction ────────────────────────────────
def test_negative_literal_dim_is_reported():
    src = "import torch\ndef f():\n    return torch.zeros(-1)\n"
    assert SymBugKind.NEGATIVE_DIMENSION in _kinds(src)


def test_zero_dim_is_legal():
    src = "import torch\ndef f():\n    return torch.zeros(0)\n"
    assert _kinds(src) == []


def test_positive_dims_are_silent():
    src = "import torch\ndef f():\n    return torch.zeros(3, 4)\n"
    assert _kinds(src) == []


def test_computed_negative_dim_is_reported():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = 2\n"
        "    b = 5\n"
        "    return torch.zeros(a - b)\n"
    )
    assert SymBugKind.NEGATIVE_DIMENSION in _kinds(src)


def test_negative_dim_in_tuple_arg():
    src = "import torch\ndef f():\n    return torch.zeros((2, -3))\n"
    assert SymBugKind.NEGATIVE_DIMENSION in _kinds(src)


def test_symbolic_dim_abstains():
    src = "import torch\ndef f(n):\n    return torch.zeros(n)\n"
    assert _kinds(src) == []


# ── Step 103: elementwise broadcasting mismatch ──────────────────────────────
def test_broadcast_trailing_mismatch_is_reported():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3, 4)\n"
        "    b = torch.zeros(3, 5)\n"
        "    return a + b\n"
    )
    assert SymBugKind.BROADCAST_MISMATCH in _kinds(src)


def test_broadcast_with_ones_is_silent():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(3, 4)\n"
        "    b = torch.zeros(1, 4)\n"
        "    return a + b\n"
    )
    assert _kinds(src) == []


def test_broadcast_same_shape_is_silent():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 3)\n"
        "    return a * b\n"
    )
    assert _kinds(src) == []


def test_broadcast_different_rank_compatible():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(5, 3, 4)\n"
        "    b = torch.zeros(4)\n"
        "    return a + b\n"
    )
    assert _kinds(src) == []


def test_broadcast_different_rank_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(5, 3, 4)\n"
        "    b = torch.zeros(3)\n"
        "    return a + b\n"
    )
    assert SymBugKind.BROADCAST_MISMATCH in _kinds(src)


def test_broadcast_symbolic_dim_abstains():
    src = (
        "import torch\n"
        "def f(n):\n"
        "    a = torch.zeros(3, n)\n"
        "    b = torch.zeros(3, 5)\n"
        "    return a + b\n"
    )
    assert _kinds(src) == []


# ── Step 104: nn.Linear in-feature mismatch ─────────────────────────────────
def test_linear_local_in_feature_mismatch():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    fc = nn.Linear(10, 20)\n"
        "    x = torch.zeros(4, 8)\n"
        "    return fc(x)\n"
    )
    assert SymBugKind.LAYER_DIM_MISMATCH in _kinds(src)


def test_linear_local_in_feature_ok():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    fc = nn.Linear(10, 20)\n"
        "    x = torch.zeros(4, 10)\n"
        "    return fc(x)\n"
    )
    assert _kinds(src) == []


def test_linear_output_feature_flows_to_next_layer():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    a = nn.Linear(10, 20)\n"
        "    b = nn.Linear(20, 5)\n"
        "    x = torch.zeros(4, 10)\n"
        "    return b(a(x))\n"
    )
    assert _kinds(src) == []


def test_linear_chained_mismatch_detected():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    a = nn.Linear(10, 20)\n"
        "    b = nn.Linear(99, 5)\n"
        "    x = torch.zeros(4, 10)\n"
        "    return b(a(x))\n"
    )
    assert SymBugKind.LAYER_DIM_MISMATCH in _kinds(src)


def test_linear_symbolic_input_abstains():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f(x):\n"
        "    fc = nn.Linear(10, 20)\n"
        "    return fc(x)\n"
    )
    assert _kinds(src) == []


def test_linear_init_forward_mismatch_via_main():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(10, 20)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "if __name__ == '__main__':\n"
        "    m = Net()\n"
        "    x = torch.zeros(4, 8)\n"
        "    m(x)\n"
    )
    assert SymBugKind.LAYER_DIM_MISMATCH in _kinds(src)


def test_linear_init_forward_ok_via_main():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(10, 20)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "if __name__ == '__main__':\n"
        "    m = Net()\n"
        "    x = torch.zeros(4, 10)\n"
        "    m(x)\n"
    )
    assert _kinds(src) == []
