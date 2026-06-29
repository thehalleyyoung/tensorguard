"""Step 19 — state diffing.

``diff_states``/``pretty_diff``/``diff_to_json`` render the delta between two
abstract :class:`State` snapshots (added / removed / changed env & store
entries, plus reachability), which is the primitive for loop/iteration debugging
and an ``--explain`` "what changed across this iteration" view.  Diffs are
deterministic (canonically sorted) and use lattice equality so a value is only
"changed" when it is genuinely not equal up to the lattice order.
"""

from __future__ import annotations

import json

from src.symexec.serialize import diff_states, diff_to_json, pretty_diff
from src.symexec.state import State
from src.symexec.symdim import SymDim
from src.symexec.values import (
    NONE,
    ModuleVal,
    TensorVal,
    int_const,
    int_range,
)


def _states():
    b, a = State(), State()
    b.set("x", int_const(5))
    a.set("x", int_range(0, 10))  # changed
    b.set("z", TensorVal(rank=1, shape=(SymDim.const_dim(3),)))  # removed
    a.set("y", NONE)  # added
    return b, a


def test_diff_classifies_added_removed_changed():
    b, a = _states()
    d = diff_states(b, a)
    assert set(d.added) == {"y"}
    assert set(d.removed) == {"z"}
    assert set(d.changed) == {"x"}
    old, new = d.changed["x"]
    assert old.const == 5 and new.lo() == 0 and new.hi() == 10


def test_diff_empty_for_identical_states():
    _, a = _states()
    d = diff_states(a, a)
    assert d.is_empty
    assert pretty_diff(a, a) == "(no change)"


def test_diff_uses_lattice_equality_not_structural():
    # Two values that are lattice-equal must not show up as "changed".
    b, a = State(), State()
    b.set("t", TensorVal(rank=2, shape=(SymDim.const_dim(2), None)))
    a.set("t", TensorVal(rank=2, shape=(SymDim.const_dim(2), None)))
    assert diff_states(b, a).is_empty


def test_store_attrs_are_diffed_with_dotted_keys():
    b, a = State(), State()
    b.set_attr("self", "fc", ModuleVal(class_name="Linear"))
    a.set_attr("self", "fc", ModuleVal(class_name="Conv2d"))
    d = diff_states(b, a)
    assert "self.fc" in d.changed


def test_reachability_change_is_reported():
    b, _ = _states()
    u = State(reachable=False)
    d = diff_states(b, u)
    assert d.reachable_before is True and d.reachable_after is False
    assert pretty_diff(b, u).startswith("! reachable: true → false")


def test_pretty_diff_markers_and_determinism():
    b, a = _states()
    text = pretty_diff(b, a)
    # markers present
    assert "+ y: None" in text
    assert "- z: " in text
    assert "~ x: int=5 → int[0, 10]" in text
    # deterministic across calls
    assert pretty_diff(b, a) == text


def test_diff_to_json_is_canonical_json():
    b, a = _states()
    j = diff_to_json(b, a)
    text = json.dumps(j, sort_keys=True)
    assert isinstance(text, str)
    assert j["added"]["y"]["k"] == "none"
    assert j["changed"]["x"]["before"]["const"] == 5
    assert j["removed"]["z"]["k"] == "tensor"


def test_diff_to_json_stable():
    b, a = _states()
    assert json.dumps(diff_to_json(b, a), sort_keys=True) == json.dumps(
        diff_to_json(b, a), sort_keys=True
    )
