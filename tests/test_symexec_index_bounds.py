"""Step 109 — tensor integer-index out of bounds on a known dim size.

Maps the leading run of plain integer indices to dims (a slice keeps its dim, a
``None`` insert is skipped, an ellipsis stops the scan) and flags a constant
index outside ``[-s, s-1]`` for a known dim size ``s``.  Abstains on symbolic
dims / non-constant indices / unknown rank.
"""

from src.symexec import analyze_source, SymBugKind

OOB = SymBugKind.TENSOR_INDEX_OOB


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_index_dim0_out_of_bounds():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[5]\n"
    assert OOB in _kinds(src)


def test_index_dim0_off_by_one():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[2]\n"
    assert OOB in _kinds(src)


def test_index_dim0_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[1]\n"
    assert _kinds(src) == []


def test_index_negative_out_of_bounds():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[-3]\n"
    assert OOB in _kinds(src)


def test_index_second_dim_out_of_bounds():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[0, 5]\n"
    assert OOB in _kinds(src)


def test_index_second_dim_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[0, 2]\n"
    assert _kinds(src) == []


def test_index_after_slice_checks_correct_dim():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    return x[:, 5]\n"
    assert OOB in _kinds(src)


def test_index_symbolic_dim_abstains():
    src = "import torch\ndef f(n):\n    x = torch.zeros(n, 3)\n    return x[5]\n"
    assert _kinds(src) == []


def test_index_unknown_rank_abstains():
    src = "import torch\ndef f(x):\n    return x[5]\n"
    assert _kinds(src) == []
