"""Step 10 — serialization & pretty-printing tests.

Covers: stable pretty strings, canonical JSON, value & state round-trips
(``from_json(to_json(v))`` is lattice-equal to ``v``), deterministic ``dumps``
output for golden-test usage, and a couple of ``--explain``-style snapshots.
"""

from __future__ import annotations

import json

import pytest

from src.symexec.serialize import (
    dumps,
    from_json,
    pretty,
    pretty_state,
    state_from_json,
    state_to_json,
    to_json,
)
from src.symexec.state import State
from src.symexec.symdim import SymDim
from src.symexec.values import (
    BOTTOM,
    BoolVal,
    DictVal,
    FloatVal,
    ListVal,
    ModuleVal,
    NONE,
    SetVal,
    StrVal,
    TOP,
    TensorVal,
    TupleVal,
    int_const,
    int_range,
)


def _eq(a, b):
    return a.leq(b) and b.leq(a)


ROUND_TRIP_VALUES = [
    TOP,
    BOTTOM,
    NONE,
    int_const(5),
    int_const(-7),
    int_range(0, 10),
    int_range(None, 7),
    int_range(3, None),
    BoolVal(True),
    BoolVal(False),
    BoolVal(),
    FloatVal(1.5),
    FloatVal(),
    StrVal("hi"),
    StrVal(),
    TensorVal(rank=2, shape=(SymDim.const_dim(2), SymDim.const_dim(3))),
    TensorVal(rank=2, shape=(SymDim.const_dim(2), None), dtype="float32", device="cuda"),
    TensorVal(rank=None),
    TensorVal(rank=0, shape=()),
    TupleVal(elems=(int_const(1), NONE)),
    TupleVal(elems=(int_const(1),), exact_len=False),
    ListVal(elem=int_const(0), length=3),
    ListVal(exact_elems=(int_const(1), NONE)),
    SetVal(elem=int_const(0), length=2),
    DictVal(known=(("a", NONE), ("b", int_const(2))), exact_keys=True),
    DictVal(value=int_const(0), known=(), exact_keys=False),
    ModuleVal(class_name="Net"),
]


@pytest.mark.parametrize("v", ROUND_TRIP_VALUES, ids=lambda v: type(v).__name__)
def test_value_round_trip(v):
    back = from_json(to_json(v))
    assert _eq(back, v), f"{pretty(v)} != {pretty(back)}"


@pytest.mark.parametrize("v", ROUND_TRIP_VALUES, ids=lambda v: type(v).__name__)
def test_to_json_is_json_serializable(v):
    # Must be plain JSON (no exceptions, stable text).
    text = json.dumps(to_json(v), sort_keys=True)
    assert isinstance(text, str)


def test_dumps_is_deterministic_and_sorted():
    v = DictVal(known=(("a", NONE), ("b", int_const(2))), exact_keys=True)
    a = dumps(v)
    b = dumps(from_json(to_json(v)))
    assert a == b
    # canonical: keys sorted
    assert a == json.dumps(json.loads(a), sort_keys=True, ensure_ascii=False)


def test_dict_known_order_canonical_in_json():
    # Two dicts built with different insertion order serialize identically.
    d1 = DictVal(known=(("b", int_const(2)), ("a", NONE)), exact_keys=True)
    d2 = DictVal(known=(("a", NONE), ("b", int_const(2))), exact_keys=True)
    assert dumps(d1) == dumps(d2)


def test_pretty_snapshots():
    assert pretty(TOP) == "⊤"
    assert pretty(BOTTOM) == "⊥"
    assert pretty(NONE) == "None"
    assert pretty(int_const(5)) == "int=5"
    assert pretty(int_range(0, 10)) == "int[0, 10]"
    assert pretty(int_range(None, 7)) == "int[-∞, 7]"
    assert pretty(TensorVal(rank=2, shape=(SymDim.const_dim(2), None))) == "Tensor[2, ?]"
    assert pretty(TupleVal(elems=(int_const(1), NONE))) == "(int=1, None)"
    assert pretty(ListVal(exact_elems=(int_const(1), NONE))) == "[int=1, None]"
    assert pretty(DictVal(known=(("a", NONE),), exact_keys=True)) == "{a: None}"
    assert pretty(ModuleVal(class_name="Linear")) == "Module(Linear)"


def test_pretty_tensor_with_dtype_device():
    v = TensorVal(rank=1, shape=(SymDim.const_dim(4),), dtype="float32", device="cuda")
    assert pretty(v) == "Tensor[4] {dtype=float32, device=cuda}"


def test_inexact_containers_pretty():
    assert pretty(TupleVal(elems=(int_const(1),), exact_len=False)) == "(int=1, …)"
    assert pretty(DictVal(known=(("a", NONE),), exact_keys=False)) == "{a: None, …}"
    assert pretty(DictVal(known=(), exact_keys=False)) == "{…}"


def _sample_state():
    s = State()
    s.set("x", int_const(5))
    s.set("t", TensorVal(rank=2, shape=(SymDim.const_dim(2), None)))
    s.set_attr("self", "fc", ModuleVal(class_name="Linear"))
    return s


def test_state_round_trip():
    s = _sample_state()
    s2 = state_from_json(state_to_json(s))
    assert s2.env == s.env
    assert s2.store == s.store
    assert s2.reachable == s.reachable


def test_state_dumps_deterministic():
    s = _sample_state()
    assert dumps(s) == dumps(state_from_json(state_to_json(s)))


def test_pretty_state_sorted_and_readable():
    s = _sample_state()
    text = pretty_state(s)
    lines = text.splitlines()
    assert lines == ["t = Tensor[2, ?]", "x = int=5", "self.fc = Module(Linear)"]


def test_unreachable_state_pretty():
    s = State(reachable=False)
    assert pretty_state(s) == "<unreachable>"


def test_state_json_keys_sorted():
    s = State()
    s.set("zeta", int_const(1))
    s.set("alpha", int_const(2))
    j = state_to_json(s)
    assert list(j["env"].keys()) == ["alpha", "zeta"]


def test_nested_container_round_trip():
    v = ListVal(
        exact_elems=(
            DictVal(known=(("k", int_range(0, 3)),), exact_keys=True),
            TupleVal(elems=(TensorVal(rank=1, shape=(SymDim.const_dim(8),)), NONE)),
        )
    )
    back = from_json(to_json(v))
    assert _eq(back, v)
