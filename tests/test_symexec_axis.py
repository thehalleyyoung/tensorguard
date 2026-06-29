"""Step 107 — axis/dim out-of-range for dim-taking tensor methods.

Most methods accept a dim in ``[-r, r-1]``; ``unsqueeze`` accepts ``[-r-1, r]``.
The detector fires only on a constant axis outside the valid range for a known
rank, and abstains on unknown rank or a non-constant axis.
"""

from src.symexec import analyze_source, SymBugKind

AX = SymBugKind.AXIS_OUT_OF_RANGE


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_squeeze_axis_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.squeeze(3)\n"
    assert AX in _kinds(src)


def test_squeeze_axis_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.squeeze(1)\n"
    assert _kinds(src) == []


def test_squeeze_negative_axis_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.squeeze(-3)\n"
    assert AX in _kinds(src)


def test_unsqueeze_allows_rank_position():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.unsqueeze(2)\n"
    assert _kinds(src) == []


def test_unsqueeze_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.unsqueeze(3)\n"
    assert AX in _kinds(src)


def test_transpose_axis_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.transpose(0, 2)\n"
    assert AX in _kinds(src)


def test_reduction_dim_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.sum(dim=5)\n"
    assert AX in _kinds(src)


def test_softmax_dim_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x.softmax(dim=1)\n"
    assert _kinds(src) == []


def test_permute_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3, 4)\n    return x.permute(0, 2, 1)\n"
    assert _kinds(src) == []


def test_permute_axis_out_of_range():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3, 4)\n    return x.permute(0, 3, 1)\n"
    assert AX in _kinds(src)


def test_unknown_rank_abstains():
    src = "import torch\ndef f(x):\n    return x.squeeze(9)\n"
    assert _kinds(src) == []
