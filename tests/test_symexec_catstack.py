"""Step 106 — torch.cat / torch.stack shape mismatch.

``cat`` requires inputs to agree on every dim except the concat axis; ``stack``
requires identical shapes.  The detector fires only on statically-known unequal
dims that must match, and abstains on symbolic dims, differing/unknown rank, or
non-literal sequences.
"""

from src.symexec import analyze_source, SymBugKind

CAT = SymBugKind.CAT_SHAPE_MISMATCH


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_cat_default_dim_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 3)\n"
        "    return torch.cat([a, b])\n"
    )
    assert _kinds(src) == []


def test_cat_dim0_nonconcat_axis_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 4)\n"
        "    return torch.cat([a, b], dim=0)\n"
    )
    assert CAT in _kinds(src)


def test_cat_dim0_differing_concat_axis_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(5, 3)\n"
        "    return torch.cat([a, b], dim=0)\n"
    )
    assert _kinds(src) == []


def test_cat_dim1_nonconcat_axis_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(5, 3)\n"
        "    return torch.cat([a, b], dim=1)\n"
    )
    assert CAT in _kinds(src)


def test_stack_identical_ok():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 3)\n"
        "    return torch.stack([a, b], dim=1)\n"
    )
    assert _kinds(src) == []


def test_stack_shape_mismatch():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 4)\n"
        "    return torch.stack([a, b])\n"
    )
    assert CAT in _kinds(src)


def test_cat_symbolic_abstains():
    src = "import torch\ndef f(a, b):\n    return torch.cat([a, b])\n"
    assert _kinds(src) == []


def test_cat_differing_rank_abstains():
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(2, 3, 4)\n"
        "    return torch.cat([a, b])\n"
    )
    assert _kinds(src) == []


def test_cat_output_rank_flows():
    # cat preserves rank; result fed to a rank-based index check should be safe
    src = (
        "import torch\n"
        "def f():\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(4, 3)\n"
        "    c = torch.cat([a, b], dim=0)\n"
        "    return c[0, 0]\n"
    )
    assert _kinds(src) == []
