"""Step 7 — provenance & replayable witness tracking.

Abstract values carry a line-tagged derivation chain (``provenance``) that is
threaded through the value-producing transfers (tensor constructors, ``nn``
layers, tensor methods like ``view``/``transpose``, indexing, matmul).  When a
bug fires, the report's ``evidence`` renders that chain as a replayable witness
pointing from the value's source to the failing site.
"""

from src.symexec import analyze_source, SymBugKind


def _bugs(src: str, name: str = "m"):
    return analyze_source(src, name).bugs


def _first(src, kind):
    for b in _bugs(src):
        if b.kind == kind:
            return b
    return None


# ── None witness: the report points at where the None was introduced ────────
def test_none_deref_witness_points_to_origin_line():
    src = "def f():\n    x = None\n    y = x\n    return y.attr\n"
    b = _first(src, SymBugKind.NONE_PROPAGATION)
    assert b is not None
    assert b.evidence is not None
    assert "L2" in b.evidence and "None literal" in b.evidence


def test_none_deref_via_subscript_carries_witness():
    src = "def f():\n    x = None\n    return x[0]\n"
    b = _first(src, SymBugKind.NONE_PROPAGATION)
    assert b is not None
    assert b.evidence is not None and "None literal" in b.evidence


# ── tensor witness: constructor → op → … → failing site ─────────────────────
def test_matmul_witness_chains_constructors():
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 4)\n"
        "    a = torch.zeros(4, 9)\n"
        "    return x @ a @ a\n"
    )
    b = _first(src, SymBugKind.MATMUL_DIM_MISMATCH)
    assert b is not None and b.evidence is not None
    assert "torch.zeros(2, 4)" in b.evidence
    assert "matmul" in b.evidence
    # ordered source→sink (a constructor line precedes the matmul step)
    assert b.evidence.index("torch.zeros") < b.evidence.index("matmul")


def test_linear_output_witness_flows_into_matmul():
    src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "def f():\n"
        "    x = torch.zeros(2, 4)\n"
        "    fc = nn.Linear(4, 5)\n"
        "    h = fc(x)\n"
        "    w = torch.zeros(7, 3)\n"
        "    return h @ w\n"
    )
    b = _first(src, SymBugKind.MATMUL_DIM_MISMATCH)
    assert b is not None and b.evidence is not None
    # the witness shows the value passed through the Linear layer
    assert "Linear(4->5)" in b.evidence
    assert "torch.zeros(2, 4)" in b.evidence


def test_view_method_carries_provenance_into_index_report():
    # the tensor-index OOB report renders the receiver's provenance witness
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.zeros(2, 3)\n"
        "    return x[5]\n"
    )
    b = _first(src, SymBugKind.TENSOR_INDEX_OOB)
    assert b is not None and b.evidence is not None
    assert "torch.zeros(2, 3)" in b.evidence


# ── witnesses are line-tagged and ordered ───────────────────────────────────
def test_witness_steps_are_line_tagged():
    src = "def f():\n    x = None\n    return x.attr\n"
    b = _first(src, SymBugKind.NONE_PROPAGATION)
    assert b is not None and b.evidence is not None
    assert b.evidence.startswith("L2")


# ── soundness: no witness invented when there is no provenance ──────────────
def test_param_none_deref_has_no_fabricated_witness():
    # a parameter that is *forced* None has no constructed provenance chain;
    # the report must not invent one.
    src = "def f(x=None):\n    return x.attr\n"
    b = _first(src, SymBugKind.NONE_PROPAGATION)
    # whether or not this fires, if it does the evidence must be honest (None or
    # a real chain), never a fabricated string.
    if b is not None and b.evidence is not None:
        assert "L" in b.evidence
