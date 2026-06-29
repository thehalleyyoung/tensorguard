"""Roadmap step 8 — **abstention taxonomy**.

Every reason ``model_contract`` declines to emit a parameter (or to descend a
subtree) is now a member of the *closed* :class:`AbstainCode` enum, attached to a
typed :class:`Abstention` record.  These tests prove:

* **No untyped abstention can exist** — an AST scan asserts every
  ``abstained.append(...)`` in ``model_contract.py`` constructs an ``Abstention``
  whose code is a literal ``AbstainCode.*`` member (so a future free-text abstain
  is rejected by CI).
* **Every code is reachable** — each enum member is triggered by a concrete
  input, and the union of triggered codes equals the whole enum (the spec's
  "a test enumerates all reasons reachable").
* The record/enum invariants: closed value set, unique stable string values,
  non-empty details, hashable/dedupable, and the ``abstain_codes`` summary.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.symexec.model_contract import (
    _MAX_WALK_DEPTH,
    _emit_layer,
    _walk,
    AbstainCode,
    Abstention,
    derive_model_contract,
    ModelContract,
)
from src.symexec.values import ModuleVal

_MODEL_CONTRACT_PY = (
    Path(__file__).resolve().parent.parent / "src" / "symexec" / "model_contract.py"
)


# --------------------------------------------------------------------------- #
# Per-code triggers (hand-built abstract values -> exact control).             #
# --------------------------------------------------------------------------- #
def _emit(mv: ModuleVal):
    abst: list = []
    _emit_layer(mv, "p", {}, abst)
    return abst


def _walk_collect(val):
    abst: list = []
    _walk(val, "p", {}, abst, [0], set())
    return abst


def _trigger_non_constant_dim():
    # Linear missing in_features -> required dim unresolved.
    mv = ModuleVal(class_name="Linear", meta=(("out_features", 4),))
    return _emit(mv)


def _trigger_unresolved_flag():
    # Linear with both dims but no statically-known bias flag.
    mv = ModuleVal(class_name="Linear",
                   meta=(("in_features", 4), ("out_features", 8)))
    return _emit(mv)


def _trigger_invalid_layer_config():
    # Conv2d with in_channels (3) not divisible by groups (2).
    mv = ModuleVal(class_name="Conv2d", meta=(
        ("in_channels", 3), ("out_channels", 4), ("groups", 2),
        ("k_len", 2), ("k0", 3), ("k1", 3),
    ))
    return _emit(mv)


def _trigger_unenumerable_container():
    root = ModuleVal(class_name="W", attrs=(
        ("m", ModuleVal(class_name="ModuleList",
                        meta=(("__opaque_container__", 1),))),))
    return _walk_collect(root)


def _trigger_cyclic_reference():
    m = ModuleVal(class_name="Cyc", attrs=())
    object.__setattr__(m, "attrs", (("self", m),))
    return _walk_collect(m)


def _trigger_max_depth_exceeded():
    leaf = ModuleVal(class_name="L0", attrs=())
    node = leaf
    for i in range(_MAX_WALK_DEPTH + 2):
        node = ModuleVal(class_name=f"W{i}", attrs=(("child", node),))
    return _walk_collect(node)


def _trigger_conditional_submodule():
    root = ModuleVal(class_name="W", attrs=(
        ("proj", ModuleVal(class_name="Linear",
                           meta=(("__conditional__", 1),))),))
    return _walk_collect(root)


_TRIGGERS = {
    AbstainCode.NON_CONSTANT_DIM: _trigger_non_constant_dim,
    AbstainCode.UNRESOLVED_FLAG: _trigger_unresolved_flag,
    AbstainCode.INVALID_LAYER_CONFIG: _trigger_invalid_layer_config,
    AbstainCode.UNENUMERABLE_CONTAINER: _trigger_unenumerable_container,
    AbstainCode.CONDITIONAL_SUBMODULE: _trigger_conditional_submodule,
    AbstainCode.CYCLIC_REFERENCE: _trigger_cyclic_reference,
    AbstainCode.MAX_DEPTH_EXCEEDED: _trigger_max_depth_exceeded,
}


@pytest.mark.parametrize("code", list(AbstainCode), ids=lambda c: c.name)
def test_each_code_is_reachable(code):
    assert code in _TRIGGERS, f"no trigger registered for {code.name}"
    abst = _TRIGGERS[code]()
    codes = {a.code for a in abst}
    assert code in codes, f"{code.name} not produced; got {codes}"


def test_triggers_cover_the_entire_enum():
    produced: set[AbstainCode] = set()
    for fn in _TRIGGERS.values():
        produced.update(a.code for a in fn())
    assert produced == set(AbstainCode), (
        f"unreachable codes: {set(AbstainCode) - produced}; "
        f"unexpected: {produced - set(AbstainCode)}"
    )


# --------------------------------------------------------------------------- #
# Static guarantee: no untyped abstention can be introduced.                    #
# --------------------------------------------------------------------------- #
def _abstain_append_calls(tree: ast.AST):
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "abstained"):
            out.append(node)
    return out


def test_every_abstain_append_constructs_a_typed_abstention():
    tree = ast.parse(_MODEL_CONTRACT_PY.read_text())
    calls = _abstain_append_calls(tree)
    assert calls, "expected to find abstained.append(...) sites"
    valid_names = {c.name for c in AbstainCode}
    for call in calls:
        assert len(call.args) == 1, "abstained.append expects one argument"
        arg = call.args[0]
        # Must be Abstention(path, AbstainCode.X, detail).
        assert isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) \
            and arg.func.id == "Abstention", \
            f"untyped abstention at line {call.lineno}"
        assert len(arg.args) == 3, f"Abstention needs 3 args at line {call.lineno}"
        code_arg = arg.args[1]
        assert (isinstance(code_arg, ast.Attribute)
                and isinstance(code_arg.value, ast.Name)
                and code_arg.value.id == "AbstainCode"
                and code_arg.attr in valid_names), \
            f"abstention code is not a literal AbstainCode member at line {call.lineno}"


def test_no_legacy_freetext_tuple_abstain_remains():
    """Guard against regressing to ``abstained.append((path, "text"))``."""
    tree = ast.parse(_MODEL_CONTRACT_PY.read_text())
    for call in _abstain_append_calls(tree):
        arg = call.args[0]
        assert not isinstance(arg, ast.Tuple), \
            f"raw-tuple abstention reintroduced at line {call.lineno}"


# --------------------------------------------------------------------------- #
# Enum / record invariants.                                                     #
# --------------------------------------------------------------------------- #
def test_enum_values_are_unique_and_stable_strings():
    values = [c.value for c in AbstainCode]
    assert len(values) == len(set(values))  # unique
    assert all(isinstance(v, str) and v.islower() and " " not in v for v in values)
    # Pin the closed set so adding/removing a code is a deliberate, reviewed change.
    assert {c.value for c in AbstainCode} == {
        "non_constant_dim", "unresolved_flag", "invalid_layer_config",
        "unenumerable_container", "conditional_submodule",
        "cyclic_reference", "max_depth_exceeded",
    }


def test_abstention_is_frozen_hashable_and_dedupes():
    a = Abstention("p", AbstainCode.UNRESOLVED_FLAG, "x")
    b = Abstention("p", AbstainCode.UNRESOLVED_FLAG, "x")
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1
    with pytest.raises(Exception):
        a.path = "q"  # frozen


def test_every_abstention_detail_is_nonempty():
    for fn in _TRIGGERS.values():
        for a in fn():
            assert isinstance(a.code, AbstainCode)
            assert isinstance(a.detail, str) and a.detail.strip()


# --------------------------------------------------------------------------- #
# Contract-level: typed codes flow through derive_model_contract end-to-end.    #
# --------------------------------------------------------------------------- #
def test_abstain_codes_property_summarises_and_sorts():
    c = ModelContract(
        model_class="M", construction="M()", params={},
        abstained=(
            Abstention("b", AbstainCode.UNRESOLVED_FLAG, "x"),
            Abstention("a", AbstainCode.NON_CONSTANT_DIM, "y"),
            Abstention("c", AbstainCode.NON_CONSTANT_DIM, "z"),  # dup code
        ),
        resolved_layers=0,
    )
    assert c.abstain_codes == (
        AbstainCode.NON_CONSTANT_DIM, AbstainCode.UNRESOLVED_FLAG,
    )  # deduped + sorted by value


def test_end_to_end_unenumerable_container_is_typed():
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])\n"
        "        self.norm = nn.LayerNorm(8)\n"
    )
    c = derive_model_contract(src, "M(n)")  # symbolic n -> unenumerable container
    assert all(isinstance(a, Abstention) for a in c.abstained)
    assert all(isinstance(a.code, AbstainCode) for a in c.abstained)
    assert AbstainCode.UNENUMERABLE_CONTAINER in c.abstain_codes
    # The resolvable sibling still resolves (sanity).
    assert c.params["norm.weight"] == (8,)


def test_end_to_end_invalid_groups_is_typed():
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.c = nn.Conv2d(3, 4, 3, groups=2)\n"
    )
    c = derive_model_contract(src, "M()")
    assert AbstainCode.INVALID_LAYER_CONFIG in c.abstain_codes
    assert not any(k.startswith("c.") for k in c.params)  # nothing invented


def test_derive_dedupes_and_sorts_abstentions():
    # Determinism: identical source yields identical, sorted, deduped abstentions.
    src = (
        "import torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        self.a = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])\n"
        "        self.b = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])\n"
    )
    c1 = derive_model_contract(src, "M(n=2)")
    c2 = derive_model_contract(src, "M(n=2)")
    assert c1.abstained == c2.abstained
    keys = [a._sort_key() for a in c1.abstained]
    assert keys == sorted(keys)
