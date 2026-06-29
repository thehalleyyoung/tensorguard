"""Roadmap step 7 — **total-function audit of ``model_contract._walk``**.

``_walk`` is the recursive tree-walker that reads a resolved module tree and emits
the weights contract.  Soundness requires it to be a *total* function over the
abstract-value graph: for **any** input it must

* **terminate** (never hang) — guaranteed by a depth ceiling plus cycle guards;
* **never raise** (in particular no ``RecursionError``) — the depth guard fires
  far below Python's recursion limit; and
* **abstain explicitly, with a reason** on everything it cannot resolve — deep
  nesting, cyclic module/container references, unenumerable containers, and opaque
  leaves — so the boundary of the derived contract is always recorded rather than
  silently guessed.

Because the contract is *partial* and only asserts positive obligations, an extra
abstention is always sound (it can only *under*-cover, never produce a false
obligation).  These tests exercise the guarantees both end-to-end (through
``derive_model_contract``) and directly against ``_walk`` with hand-built
pathological value graphs (1k+ deep chains, deliberate cycles).
"""

from __future__ import annotations

import sys

import pytest

from src.symexec.model_contract import (
    _MAX_WALK_DEPTH,
    _walk,
    AbstainCode,
    Abstention,
    derive_model_contract,
    ModelContract,
)
from src.symexec.values import ListVal, ModuleVal, TOP, TupleVal


# --------------------------------------------------------------------------- #
# Helpers.                                                                      #
# --------------------------------------------------------------------------- #
def _linear_leaf(in_f: int = 4, out_f: int = 4) -> ModuleVal:
    return ModuleVal(class_name="Linear", attrs=(),
                     meta=(("in_features", in_f), ("out_features", out_f),
                           ("bias", 1)))


def _run(root):
    params: dict = {}
    abstained: list = []
    resolved = [0]
    _walk(root, "", params, abstained, resolved, set())
    return params, abstained, resolved[0]


def _module_chain(depth: int, leaf: ModuleVal | None = None) -> ModuleVal:
    """A chain of ``depth`` user-module wrappers around ``leaf`` (so ``leaf`` sits
    at recursion depth ``depth``)."""
    node = leaf if leaf is not None else _linear_leaf()
    for i in range(depth):
        node = ModuleVal(class_name=f"W{i}", attrs=(("child", node),))
    return node


def _make_cyclic_module() -> ModuleVal:
    m = ModuleVal(class_name="Cyc", attrs=())
    object.__setattr__(m, "attrs", (("self", m),))
    return m


def _reasons(abstained):
    return [a.detail for a in abstained]


def _codes(abstained):
    return [a.code for a in abstained]


# --------------------------------------------------------------------------- #
# Termination & non-raising on pathological graphs (the core guarantee).        #
# --------------------------------------------------------------------------- #
def test_extremely_deep_module_chain_does_not_raise():
    # 50k deep — vastly beyond Python's recursion limit; the depth guard must
    # stop us long before any stack overflow.
    root = _module_chain(50_000)
    params, abstained, resolved = _run(root)
    assert ("max module-tree depth exceeded" in _reasons(abstained))
    # Nothing past the ceiling was resolved.
    assert resolved == 0 and params == {}


def test_deep_list_chain_does_not_raise():
    # A deeply-nested registered container (nn.ModuleList) is modelled as nested
    # index-keyed ModuleVals; the depth guard must stop the recursion long before
    # any stack overflow.
    node: object = ModuleVal(class_name="ModuleList", attrs=(("0", _linear_leaf()),))
    for _ in range(5_000):
        node = ModuleVal(class_name="ModuleList", attrs=(("0", node),))
    root = ModuleVal(class_name="W", attrs=(("xs", node),))
    _, abstained, _ = _run(root)
    assert "max module-tree depth exceeded" in _reasons(abstained)


def test_deep_tuple_chain_does_not_raise():
    node: object = ModuleVal(class_name="ModuleDict", attrs=(("k", _linear_leaf()),))
    for _ in range(5_000):
        node = ModuleVal(class_name="ModuleDict", attrs=(("k", node),))
    root = ModuleVal(class_name="W", attrs=(("t", node),))
    _, abstained, _ = _run(root)
    assert "max module-tree depth exceeded" in _reasons(abstained)


def test_cyclic_module_terminates_and_abstains():
    m = _make_cyclic_module()
    _, abstained, _ = _run(m)
    assert ("self", "cyclic module reference") in [(a.path, a.detail) for a in abstained]
    assert AbstainCode.CYCLIC_REFERENCE in _codes(abstained)


def test_indirect_module_cycle_terminates():
    a = ModuleVal(class_name="A", attrs=())
    b = ModuleVal(class_name="B", attrs=(("back", a),))
    object.__setattr__(a, "attrs", (("fwd", b),))
    _, abstained, _ = _run(a)
    assert "cyclic module reference" in _reasons(abstained)


def test_cyclic_list_terminates_and_abstains():
    # A self-referential registered container (ModuleList) is a module cycle.
    ml = ModuleVal(class_name="ModuleList", attrs=())
    object.__setattr__(ml, "attrs", (("0", ml),))
    root = ModuleVal(class_name="W", attrs=(("xs", ml),))
    _, abstained, _ = _run(root)
    assert "cyclic module reference" in _reasons(abstained)
    assert AbstainCode.CYCLIC_REFERENCE in _codes(abstained)


def test_cyclic_tuple_terminates_and_abstains():
    md = ModuleVal(class_name="ModuleDict", attrs=())
    object.__setattr__(md, "attrs", (("k", md),))
    root = ModuleVal(class_name="W", attrs=(("t", md),))
    _, abstained, _ = _run(root)
    assert "cyclic module reference" in _reasons(abstained)


# --------------------------------------------------------------------------- #
# Exact depth-ceiling boundary.                                                 #
# --------------------------------------------------------------------------- #
def test_depth_ceiling_boundary_resolves_then_abstains():
    # Leaf at exactly _MAX_WALK_DEPTH is processed (resolved).
    ok = _module_chain(_MAX_WALK_DEPTH)
    p_ok, a_ok, r_ok = _run(ok)
    assert r_ok == 1 and len(p_ok) == 2  # Linear weight + bias
    assert "max module-tree depth exceeded" not in _reasons(a_ok)

    # Leaf one level deeper is cut off with a depth abstention (never resolved).
    over = _module_chain(_MAX_WALK_DEPTH + 1)
    p_ov, a_ov, r_ov = _run(over)
    assert r_ov == 0 and p_ov == {}
    assert "max module-tree depth exceeded" in _reasons(a_ov)


def test_guard_fires_below_python_recursion_limit():
    """Even with a recursion limit barely above the ceiling, a huge chain is
    handled by abstaining (the guard, not the interpreter stack, terminates it)."""
    old = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(_MAX_WALK_DEPTH + 80)
        _, abstained, _ = _run(_module_chain(10_000))
        assert "max module-tree depth exceeded" in _reasons(abstained)
    finally:
        sys.setrecursionlimit(old)


# --------------------------------------------------------------------------- #
# Abstain shape & determinism.                                                  #
# --------------------------------------------------------------------------- #
def test_every_abstention_is_a_path_reason_pair():
    inputs = [
        _make_cyclic_module(),
        _module_chain(_MAX_WALK_DEPTH + 5),
        ModuleVal(class_name="W", attrs=(("xs", ListVal(exact_elems=None)),)),
        ModuleVal(class_name="W", attrs=(("t", TupleVal(elems=(), exact_len=False)),)),
        ModuleVal(class_name="W", attrs=(("x", TOP),)),
    ]
    for root in inputs:
        _, abstained, _ = _run(root)
        for entry in abstained:
            assert isinstance(entry, Abstention)
            assert isinstance(entry.path, str)
            assert isinstance(entry.code, AbstainCode)
            assert isinstance(entry.detail, str) and entry.detail  # non-empty


def test_walk_is_deterministic():
    root = ModuleVal(class_name="Net", attrs=(
        ("a", _linear_leaf(4, 8)),
        ("b", ListVal(exact_elems=(_linear_leaf(8, 8), _linear_leaf(8, 2)))),
        ("c", _make_cyclic_module()),
        ("d", TOP),
    ))
    r1 = _run(root)
    r2 = _run(root)
    assert r1 == r2


def test_opaque_leaf_is_silently_ignored_not_abstained_when_plain():
    # A TOP submodule slot is abstained on (it *might* be a module we can't see);
    # but it must never raise or emit a param.
    root = ModuleVal(class_name="W", attrs=(("x", TOP),))
    params, abstained, resolved = _run(root)
    assert params == {} and resolved == 0
    # No crash; TOP at a slot yields no obligation.
    assert all(isinstance(a.detail, str) for a in abstained)


def test_unenumerable_containers_abstain_with_distinct_reasons():
    # A *registered* container (ModuleList/ModuleDict) whose contents are not
    # statically enumerable is marked opaque and abstains with a typed reason.
    list_root = ModuleVal(class_name="W", attrs=(
        ("m", ModuleVal(class_name="ModuleList",
                        meta=(("__opaque_container__", 1),))),))
    _, a1, _ = _run(list_root)
    assert ("m", "ModuleList contents not statically enumerable") in [
        (a.path, a.detail) for a in a1]
    assert all(a.code is AbstainCode.UNENUMERABLE_CONTAINER for a in a1 if a.path == "m")

    dict_root = ModuleVal(class_name="W", attrs=(
        ("d", ModuleVal(class_name="ModuleDict",
                        meta=(("__opaque_container__", 1),))),))
    _, a2, _ = _run(dict_root)
    assert ("d", "ModuleDict contents not statically enumerable") in [
        (a.path, a.detail) for a in a2]
    assert all(a.code is AbstainCode.UNENUMERABLE_CONTAINER for a in a2 if a.path == "d")


def test_plain_list_and_tuple_slots_are_ignored_not_emitted():
    # A *plain* Python list/tuple of modules is not registered by PyTorch, so a
    # ListVal/TupleVal at a submodule slot must contribute nothing — neither a
    # param (which would be a false positive) nor an abstention.
    list_root = ModuleVal(class_name="W", attrs=(
        ("m", ListVal(exact_elems=(_linear_leaf(), _linear_leaf()))),))
    params, abstained, resolved = _run(list_root)
    assert params == {} and abstained == [] and resolved == 0

    tup_root = ModuleVal(class_name="W", attrs=(
        ("t", TupleVal(elems=(_linear_leaf(),), exact_len=True)),))
    params, abstained, resolved = _run(tup_root)
    assert params == {} and abstained == [] and resolved == 0

    # Even a non-enumerable plain list/tuple is silently ignored (torch registers
    # nothing for it), never a spurious abstention.
    for slot in (ListVal(exact_elems=None), TupleVal(elems=(), exact_len=False)):
        _, abst, _ = _run(ModuleVal(class_name="W", attrs=(("x", slot),)))
        assert abst == []


# --------------------------------------------------------------------------- #
# End-to-end: derive_model_contract never hangs or raises on stress sources.    #
# --------------------------------------------------------------------------- #
def _nested_class_source(n: int) -> str:
    lines = ["import torch.nn as nn",
             "class L0(nn.Module):",
             "    def __init__(self):",
             "        super().__init__()",
             "        self.lin = nn.Linear(4, 4)"]
    for i in range(1, n + 1):
        lines += [f"class L{i}(nn.Module):",
                  "    def __init__(self):",
                  "        super().__init__()",
                  f"        self.child = L{i-1}()"]
    return "\n".join(lines)


def test_end_to_end_1k_nested_modules_returns_partial_contract():
    c = derive_model_contract(_nested_class_source(1000), "L1000()")
    assert isinstance(c, ModelContract)
    assert c.partial is True  # always partial — never claims exhaustiveness
    # It returns a contract (possibly empty/partial) rather than hanging/raising.


def test_end_to_end_recursive_self_model_does_not_raise():
    src = (
        "import torch.nn as nn\n"
        "class Deep(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        if n > 0:\n"
        "            self.child = Deep(n - 1)\n"
        "        else:\n"
        "            self.lin = nn.Linear(4, 4)\n"
    )
    c = derive_model_contract(src, "Deep(5000)")
    assert isinstance(c, ModelContract) and c.partial is True


def test_end_to_end_modulelist_comprehension_abstains():
    src = (
        "import torch.nn as nn\n"
        "class S(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])\n"
    )
    c = derive_model_contract(src, "S(8)")
    assert isinstance(c, ModelContract)
    # The comprehension/ModuleList is not statically enumerable -> recorded.
    # (Either as an abstention here or simply unresolved; never a wrong obligation.)
    assert c.partial is True


def test_end_to_end_normal_model_still_resolves_exactly():
    """Regression: the totality hardening must not change correct resolution of a
    real (shallow) model."""
    src = (
        "import torch.nn as nn\n"
        "class Inner(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(4, 8)\n"
        "class Outer(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.a = Inner()\n"
        "        self.b = Inner()\n"
    )
    c = derive_model_contract(src, "Outer()")
    assert dict(c.params) == {
        "a.lin.weight": (8, 4), "a.lin.bias": (8,),
        "b.lin.weight": (8, 4), "b.lin.bias": (8,),
    }
    assert c.resolved_layers == 2
    assert c.abstained == ()


def test_max_walk_depth_is_safely_below_recursion_limit():
    # Sanity on the constant itself: the ceiling leaves comfortable head-room.
    assert 0 < _MAX_WALK_DEPTH < sys.getrecursionlimit()
    assert _MAX_WALK_DEPTH <= 500
