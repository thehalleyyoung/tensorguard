"""Step 105 — matmul contracted-dimension mismatch.

Covers ``a @ b``, ``torch.matmul/mm/bmm``, tensor-method ``.matmul/.mm/.bmm``,
output-shape flow through chained products, and abstention on symbolic dims.
"""

from src.symexec import analyze_source, SymBugKind

MM = SymBugKind.MATMUL_DIM_MISMATCH


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_matmul_2d_inner_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(6, 7)\n"
        "    return a @ b\n"
    )
    assert MM in _kinds(src)


def test_matmul_2d_inner_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(5, 7)\n"
        "    return a @ b\n"
    )
    assert _kinds(src) == []


def test_torch_matmul_function_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(6, 7)\n"
        "    return torch.matmul(a, b)\n"
    )
    assert MM in _kinds(src)


def test_tensor_method_mm_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(6, 7)\n"
        "    return a.mm(b)\n"
    )
    assert MM in _kinds(src)


def test_matmul_output_shape_flows_chained_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(5, 7)\n"
        "    c = torch.zeros(7, 2)\n"
        "    return (a @ b) @ c\n"
    )
    assert _kinds(src) == []


def test_matmul_output_shape_flows_chained_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(4, 5)\n"
        "    b = torch.zeros(5, 7)\n"
        "    c = torch.zeros(9, 2)\n"
        "    return (a @ b) @ c\n"
    )
    assert MM in _kinds(src)


def test_bmm_batched_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(8, 4, 5)\n"
        "    b = torch.zeros(8, 5, 7)\n"
        "    return torch.bmm(a, b)\n"
    )
    assert _kinds(src) == []


def test_bmm_batched_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(8, 4, 5)\n"
        "    b = torch.zeros(8, 6, 7)\n"
        "    return torch.bmm(a, b)\n"
    )
    assert MM in _kinds(src)


def test_matmul_1d_inner_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(5)\n"
        "    b = torch.zeros(6)\n"
        "    return a @ b\n"
    )
    assert MM in _kinds(src)


def test_matmul_symbolic_abstains():
    src = "import torch\ndef f(a, b):\n    return a @ b\n"
    assert _kinds(src) == []


def test_linear_output_feeds_matmul_ok():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    fc = nn.Linear(10, 20)\n"
        "    x = torch.zeros(4, 10)\n"
        "    y = fc(x)\n"
        "    w = torch.zeros(20, 3)\n"
        "    return y @ w\n"
    )
    assert _kinds(src) == []


def test_linear_output_feeds_matmul_mismatch():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    fc = nn.Linear(10, 20)\n"
        "    x = torch.zeros(4, 10)\n"
        "    y = fc(x)\n"
        "    w = torch.zeros(99, 3)\n"
        "    return y @ w\n"
    )
    assert MM in _kinds(src)
