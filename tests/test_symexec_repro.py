"""Tests for the runnable reproducer generator (even_more.md Tier 1, idea #1).

Two layers:
  * torch-free structural tests — a reproducer is generated for every covered
    kind, carries the right predicted exception, and is syntactically valid;
  * a torch-gated *executed-confirmation* test — each generated reproducer, when
    run, actually raises the predicted exception (the empirical soundness layer).
"""

from __future__ import annotations

import ast

import pytest

from src.symexec import (
    ReproScript,
    confirm,
    generate_repro,
    generate_repros,
)
from src.symexec.engine import analyze_source

try:  # torch gates only the execution test, not generation
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# (label, source, kind, expected_exception)
CASES = [
    ("matmul",
     "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(4,5); c=a@b\n",
     "matmul_dim_mismatch", "RuntimeError"),
    ("broadcast",
     "import torch\nif __name__=='__main__':\n a=torch.randn(3,4); b=torch.randn(3,5); c=a+b\n",
     "broadcast_mismatch", "RuntimeError"),
    ("reshape",
     "import torch\nif __name__=='__main__':\n x=torch.randn(2,3); y=x.reshape(5,5)\n",
     "reshape_size_mismatch", "RuntimeError"),
    ("axis",
     "import torch\nif __name__=='__main__':\n x=torch.randn(2,3); y=x.sum(dim=5)\n",
     "axis_out_of_range", "IndexError"),
    ("index",
     "if __name__=='__main__':\n xs=[1,2,3]; y=xs[5]\n",
     "rank_index_error", "IndexError"),
    ("divzero",
     "if __name__=='__main__':\n n=0; y=10//n\n",
     "division_by_zero", "ZeroDivisionError"),
    ("linear",
     "import torch, torch.nn as nn\nif __name__=='__main__':\n m=nn.Linear(10,5); x=torch.randn(2,7); y=m(x)\n",
     "layer_dim_mismatch", "RuntimeError"),
    ("cat",
     "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(2,5); c=torch.cat([a,b],dim=0)\n",
     "cat_shape_mismatch", "RuntimeError"),
    ("einsum",
     "import torch\nif __name__=='__main__':\n a=torch.randn(2,3); b=torch.randn(4,5); c=torch.einsum('ij,jk->ik',a,b)\n",
     "einsum_dim_mismatch", "RuntimeError"),
    ("negdim",
     "import torch\nif __name__=='__main__':\n x=torch.randn(2,-3)\n",
     "negative_dimension", "RuntimeError"),
]

_LABELS = [c[0] for c in CASES]


def _first_repro(source):
    r = analyze_source(source)
    assert r.bugs, "expected a bug"
    rep = generate_repro(r.bugs[0])
    assert rep is not None, "expected a reproducer"
    return r.bugs[0], rep


@pytest.mark.parametrize("label,source,kind,exc",
                         CASES, ids=_LABELS)
def test_repro_generated_with_right_prediction(label, source, kind, exc):
    bug, rep = _first_repro(source)
    assert isinstance(rep, ReproScript)
    assert rep.kind == kind
    assert rep.expected_exception == exc
    assert rep.fidelity in ("exact", "class")
    # provenance carried through
    assert rep.line == bug.line


@pytest.mark.parametrize("label,source,kind,exc", CASES, ids=_LABELS)
def test_repro_is_valid_python(label, source, kind, exc):
    _, rep = _first_repro(source)
    ast.parse(rep.script)  # raises SyntaxError if malformed
    assert rep.script.startswith('"""Auto-generated TensorGuard reproducer.')
    assert kind in rep.script


def test_unhandled_kind_returns_none():
    # none_propagation has no reproducer generator -> honest None.
    class _FakeKind:
        value = "none_propagation"

    class _FakeBug:
        kind = _FakeKind()
        message = "value may be None"
        line = 1
        col = 0
        function = ""

    assert generate_repro(_FakeBug()) is None


def test_generate_repros_skips_ungeneratable():
    r = analyze_source(CASES[0][1])
    reps = generate_repros(r)
    assert reps and all(isinstance(x, ReproScript) for x in reps)


def test_symresult_repros_method():
    r = analyze_source(CASES[0][1])
    reps = r.repros()
    assert reps and reps[0].kind == "matmul_dim_mismatch"


def test_clean_code_has_no_repros():
    clean = (
        "import torch\n"
        "if __name__ == '__main__':\n"
        "    a = torch.randn(2, 3); b = torch.randn(3, 5); c = a @ b\n"
    )
    assert analyze_source(clean).repros() == []


# --------------------------------------------------------------------------- #
# Empirical soundness: each reproducer actually raises when executed.          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="executed confirmation needs torch")
@pytest.mark.parametrize("label,source,kind,exc", CASES, ids=_LABELS)
def test_repro_actually_raises_predicted_exception(label, source, kind, exc):
    _, rep = _first_repro(source)
    res = confirm(rep)
    assert res.raised, f"{kind} reproducer did not raise: {res.detail}"
    assert res.confirmed, (
        f"{kind} raised {res.raised_exception}, expected {exc}: {res.detail}"
    )
    assert res.raised_exception is not None
