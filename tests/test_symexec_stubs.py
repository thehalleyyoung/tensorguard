"""Tests for the third-party stub library (roadmap Step 83).

Covers the declarative shape-summary registry (elementwise / factory / rank
transfers), alias-aware canonical-name resolution through ``import`` bindings, and
end-to-end shape propagation through stubbed library calls that lets downstream
detectors fire — with no false positives on correct code and a sound abstain when
the callee is shadowed by a local or the result shape is genuinely unknown.
"""

import src.symexec as s
from src.symexec import stubs
from src.symexec.interpreter import Interpreter
from src.symexec.values import IntVal, ListVal, TensorVal, TupleVal, int_const


def analyze(src):
    return s.analyze_source(src, "t.py")


def kinds(src):
    return [b.kind.name for b in analyze(src).bugs]


# --------------------------------------------------------------------------- #
# Registry & transfer units                                                   #
# --------------------------------------------------------------------------- #

def test_registry_has_common_entries():
    for name in (
        "torch.relu",
        "torch.nn.functional.relu",
        "torch.softmax",
        "torch.flatten",
        "torch.unsqueeze",
        "numpy.zeros",
        "numpy.exp",
        "torch.zeros_like",
    ):
        assert stubs.lookup(name) is not None, name
    assert stubs.lookup(None) is None
    assert stubs.lookup("torch.not_a_real_op") is None


def test_elementwise_preserves_shape():
    t = TensorVal(rank=2, shape=(int_const(4).sym, int_const(8).sym),
                  dtype="float32", device="cpu")
    out = stubs._elementwise([t], {})
    assert isinstance(out, TensorVal)
    assert out.rank == 2 and out.shape == t.shape
    assert out.dtype == "float32" and out.device == "cpu"


def test_elementwise_via_input_kw():
    t = TensorVal(rank=3)
    assert stubs._elementwise([], {"input": t}).rank == 3


def test_elementwise_abstains_on_non_tensor():
    assert stubs._elementwise([IntVal()], {}) is None
    assert stubs._elementwise([], {}) is None


def test_factory_from_tuple_and_int():
    out = stubs._factory([TupleVal(elems=(int_const(3), int_const(4)))], {})
    assert out.rank == 2
    out2 = stubs._factory([int_const(5)], {})
    assert out2.rank == 1
    out3 = stubs._factory([], {"shape": TupleVal(elems=(int_const(2),))})
    assert out3.rank == 1


def test_factory_abstains_on_unknown_shape():
    # np.zeros(n) with non-constant n is still soundly rank-1 (one unknown dim).
    out = stubs._factory([IntVal()], {})
    assert isinstance(out, TensorVal) and out.rank == 1 and out.shape == (None,)
    # a non-shape arg (a tensor) is not a shape spec
    assert stubs._factory([TensorVal(rank=2)], {}) is None


def test_flatten_rank_transform():
    t = TensorVal(rank=4)
    # default start=0,end=-1 -> fully flattened to rank 1
    assert stubs._flatten([t], {}).rank == 1
    # start_dim=1 -> collapse dims 1..3 into one -> rank 2
    assert stubs._flatten([t, int_const(1)], {}).rank == 2
    # start=1,end=2 -> collapse two dims -> rank 3
    assert stubs._flatten([t, int_const(1), int_const(2)], {}).rank == 3


def test_flatten_abstains_on_nonconstant_bound():
    t = TensorVal(rank=4)
    out = stubs._flatten([t, IntVal()], {})  # non-constant start_dim
    assert isinstance(out, TensorVal) and out.rank is None


def test_flatten_unknown_rank_input():
    assert stubs._flatten([TensorVal(rank=None)], {}).rank is None


def test_unsqueeze_and_squeeze():
    assert stubs._unsqueeze([TensorVal(rank=2), int_const(0)], {}).rank == 3
    assert stubs._unsqueeze([TensorVal(rank=None)], {}).rank is None
    # squeeze keeps only "it is a tensor" (rank unknown) — sound, never a guess
    sq = stubs._squeeze([TensorVal(rank=3)], {})
    assert isinstance(sq, TensorVal) and sq.rank is None


# --------------------------------------------------------------------------- #
# Alias-aware canonical resolution                                            #
# --------------------------------------------------------------------------- #

def _interp(src):
    import ast
    return Interpreter(ast.parse(src), filename="t.py")


def test_alias_import_as():
    interp = _interp("import numpy as np\nimport torch.nn.functional as F\n")
    assert interp._import_aliases["np"] == "numpy"
    assert interp._import_aliases["F"] == "torch.nn.functional"


def test_alias_plain_import():
    interp = _interp("import torch\nimport torch.nn\n")
    assert interp._import_aliases["torch"] == "torch"


def test_alias_from_import():
    interp = _interp("from torch import relu, nn\nfrom torch.nn import functional as F\n")
    assert interp._import_aliases["relu"] == "torch.relu"
    assert interp._import_aliases["nn"] == "torch.nn"
    assert interp._import_aliases["F"] == "torch.nn.functional"


def test_alias_relative_import_not_stubbed():
    interp = _interp("from . import helpers\nfrom .mod import thing\n")
    assert "helpers" not in interp._import_aliases
    assert "thing" not in interp._import_aliases


def test_canonical_callee_resolution():
    import ast
    interp = _interp(
        "import torch\nimport torch.nn.functional as F\nfrom torch import relu\n"
    )

    def chain(expr):
        return ast.parse(expr, mode="eval").body

    assert interp._canonical_callee(chain("F.relu")) == "torch.nn.functional.relu"
    assert interp._canonical_callee(chain("torch.softmax")) == "torch.softmax"
    assert interp._canonical_callee(chain("relu")) == "torch.relu"
    assert interp._canonical_callee(chain("unknown.thing")) is None
    assert interp._canonical_callee(chain("undefined_name")) is None


# --------------------------------------------------------------------------- #
# End-to-end shape propagation                                                #
# --------------------------------------------------------------------------- #

def test_functional_relu_propagates_to_matmul():
    src = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def f():\n"
        "    x = torch.randn(4, 8)\n"
        "    y = F.relu(x)\n"
        "    w = torch.randn(5, 3)\n"
        "    return torch.matmul(y, w)\n"
    )
    assert "MATMUL_DIM_MISMATCH" in kinds(src)


def test_numpy_zeros_propagates_to_matmul():
    src = (
        "import numpy as np\n"
        "import torch\n"
        "def g():\n"
        "    a = np.zeros((4, 8))\n"
        "    w = torch.randn(5, 3)\n"
        "    return torch.matmul(a, w)\n"
    )
    assert "MATMUL_DIM_MISMATCH" in kinds(src)


def test_unsqueeze_changes_rank_no_false_positive():
    # unsqueeze yields a known rank but unknown dim sizes; a downstream matmul
    # therefore cannot be proven wrong — the engine must stay silent (sound).
    src = (
        "import torch\n"
        "def h():\n"
        "    x = torch.randn(8)\n"
        "    y = torch.unsqueeze(x, 0)\n"  # rank 1 -> rank 2, shape unknown
        "    w = torch.randn(5, 3)\n"
        "    return torch.matmul(y, w)\n"
    )
    assert analyze(src).bugs == []


def test_correct_code_no_false_positive():
    src = (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def ok():\n"
        "    x = torch.randn(4, 8)\n"
        "    y = F.softmax(F.relu(x), dim=-1)\n"
        "    w = torch.randn(8, 3)\n"
        "    return torch.matmul(y, w)\n"
    )
    assert analyze(src).bugs == []


def test_flatten_unknown_dim_no_false_positive():
    # flatten loses exact dim sizes -> downstream matmul cannot be proven wrong.
    src = (
        "import torch\n"
        "def h():\n"
        "    x = torch.randn(2, 3, 4)\n"
        "    y = torch.flatten(x, 1)\n"
        "    w = torch.randn(7, 5)\n"
        "    return torch.matmul(y, w)\n"
    )
    assert analyze(src).bugs == []


def test_local_def_shadows_stub():
    # A user function named ``relu`` must be called, NOT the torch stub, so its
    # body is analysed and the stub never engages.
    src = (
        "from torch import relu\n"
        "import torch\n"
        "def relu(x):\n"
        "    return x\n"
        "def f():\n"
        "    a = torch.randn(2, 3)\n"
        "    return relu(a)\n"
    )
    # No crash, no spurious bug; the local def wins (stub is at the abstain point).
    assert analyze(src).bugs == []


def test_unimported_name_not_stubbed():
    # ``relu`` is never imported here, so the bare call must not resolve to a stub.
    src = (
        "import torch\n"
        "def f():\n"
        "    x = torch.randn(2, 3)\n"
        "    return softmax(x)\n"  # undefined free name -> stays Top, no stub
    )
    # softmax is unresolved; engine abstains (Top). No crash, no false bug.
    assert analyze(src).bugs == []


def test_fingerprint_unchanged_for_torch_only_corpus():
    # The matmul corpus file uses only literal torch ops handled before the stub
    # layer, so its proof fingerprint must be byte-identical.
    import os
    path = os.path.join(
        os.path.dirname(__file__), "symexec_corpus", "wild", "matmul_dim_mismatch.py"
    )
    r = s.analyze_source(open(path).read(), "matmul_dim_mismatch.py")
    assert r.fingerprint().startswith("de466b6f54018384")
