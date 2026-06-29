"""Step 36 — subscript-assignment bounds (``x[i] = v``).

The index on the LHS of an assignment must be in bounds for a known-size tensor
(``TENSOR_INDEX_OOB``) or list/tuple (``RANK_INDEX_ERROR``), mirroring the
read-side checks.  Abstains on symbolic sizes / non-constant indices.
"""

from src.symexec import analyze_source, SymBugKind


def _kinds(src, name="f.py"):
    return [b.kind for b in analyze_source(src, name).bugs]


def test_tensor_store_out_of_bounds():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    x[5] = 0\n"
    assert SymBugKind.TENSOR_INDEX_OOB in _kinds(src)


def test_tensor_store_ok():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    x[1] = 0\n"
    assert _kinds(src) == []


def test_tensor_store_second_dim_out_of_bounds():
    src = "import torch\ndef f():\n    x = torch.zeros(2, 3)\n    x[0, 5] = 0\n"
    assert SymBugKind.TENSOR_INDEX_OOB in _kinds(src)


def test_list_store_out_of_bounds():
    src = "def f():\n    xs = [1, 2, 3]\n    xs[5] = 0\n"
    assert SymBugKind.RANK_INDEX_ERROR in _kinds(src)


def test_list_store_ok():
    src = "def f():\n    xs = [1, 2, 3]\n    xs[2] = 0\n"
    assert _kinds(src) == []


def test_store_symbolic_abstains():
    src = "def f(xs):\n    xs[99] = 0\n"
    assert _kinds(src) == []


# ── value propagation: x[i] = v updates the container for later reads ────────
def test_list_store_propagates_value_to_later_read():
    # xs[0] becomes None, so xs[0].foo is a None-deref on the updated element
    src = "def f():\n    xs = [1, 2, 3]\n    xs[0] = None\n    return xs[0].foo\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_list_store_does_not_falsely_flag_other_index():
    # only xs[0] is updated; reading xs[1] (still an int) is clean
    src = "def f():\n    xs = [1, 2, 3]\n    xs[0] = None\n    return xs[1].foo\n"
    assert _kinds(src) == []


def test_dict_store_propagates_value_to_later_read():
    src = "def f():\n    d = {}\n    d['k'] = None\n    return d['k'].foo\n"
    assert SymBugKind.NONE_PROPAGATION in _kinds(src)


def test_list_store_preserves_length():
    # len(xs) is unchanged after a store, so xs[len(xs)-1] stays in bounds
    src = "def f():\n    xs = [1, 2, 3]\n    xs[0] = 9\n    return xs[len(xs) - 1]\n"
    assert _kinds(src) == []


def test_list_unknown_index_store_weakens_soundly():
    # storing at a symbolic index must not crash or falsely flag a later read
    src = "def f(i):\n    xs = [1, 2, 3]\n    xs[i] = None\n    return xs[0]\n"
    assert _kinds(src) == []
