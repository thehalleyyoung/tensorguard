"""Step 10 — stable serialization & pretty-printing of abstract values/states.

Two surfaces, both deterministic (canonically ordered, no fresh-symbol noise) so
they are usable for golden tests and an ``--explain`` view:

* :func:`pretty` / :func:`pretty_state` — compact human-readable strings
  (``Tensor[2, 3]``, ``int=5``, ``{a: None}``, ``⊤``/``⊥``).
* :func:`to_json` / :func:`from_json` (and ``*_state``) — a canonical
  JSON-compatible dict round-trippable back into an (lattice-)equal value.

Determinism is achieved by sorting every mapping (``env``/``store``/dict keys)
and rendering symbolic dims via :class:`SymDim`'s stable ``str`` (unknown dims —
``None`` in a shape — render as ``?``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .state import State
from .symdim import SymDim
from .values import (
    AbstractValue,
    BoolVal,
    Bottom,
    CallableVal,
    DictVal,
    FloatVal,
    IntVal,
    ListVal,
    ModuleVal,
    NoneVal,
    SetVal,
    StrVal,
    TensorVal,
    TOP,
    BOTTOM,
    NONE,
    Top,
    TupleVal,
    int_const,
    int_range,
)

__all__ = [
    "pretty",
    "pretty_state",
    "to_json",
    "from_json",
    "state_to_json",
    "state_from_json",
    "dumps",
    "StateDiff",
    "diff_states",
    "pretty_diff",
    "diff_to_json",
]


# ── dim helpers ─────────────────────────────────────────────────────────────
def _dim_str(d: Optional[SymDim]) -> str:
    return "?" if d is None else str(d)


def _dim_json(d: Optional[SymDim]):
    if d is None:
        return None
    return d.value if d.value is not None else str(d)


def _dim_from_json(x) -> Optional[SymDim]:
    if x is None:
        return None
    if isinstance(x, int):
        return SymDim.const_dim(x)
    if isinstance(x, str):
        return SymDim.var(x)  # best-effort: a single named variable
    return None


# ── pretty-printing ─────────────────────────────────────────────────────────
def pretty(v: AbstractValue) -> str:
    if isinstance(v, Top):
        return "⊤"
    if isinstance(v, Bottom):
        return "⊥"
    if isinstance(v, NoneVal):
        return "None"
    if isinstance(v, BoolVal):
        return "bool" if v.const is None else f"bool={v.const}"
    if isinstance(v, IntVal):
        return _pretty_int(v)
    if isinstance(v, FloatVal):
        return "float" if v.const is None else f"float={v.const}"
    if isinstance(v, StrVal):
        return "str" if v.const is None else f"str={v.const!r}"
    if isinstance(v, TensorVal):
        return _pretty_tensor(v)
    if isinstance(v, TupleVal):
        inner = ", ".join(pretty(e) for e in v.elems)
        return f"({inner}{'' if v.exact_len else ', …'})"
    if isinstance(v, ListVal):
        if v.exact_elems is not None:
            return "[" + ", ".join(pretty(e) for e in v.exact_elems) + "]"
        n = "" if v.length is None else f"×{v.length}"
        return f"[{pretty(v.elem)}]{n}"
    if isinstance(v, SetVal):
        n = "" if v.length is None else f"×{v.length}"
        return f"{{{pretty(v.elem)}}}{n}"
    if isinstance(v, DictVal):
        items = ", ".join(f"{k}: {pretty(val)}" for k, val in v.known)
        tail = "" if v.exact_keys else (", …" if items else "…")
        return "{" + items + tail + "}"
    if isinstance(v, ModuleVal):
        return f"Module({v.class_name})"
    if isinstance(v, CallableVal):
        return f"Callable({v.func_id})"
    return "?"


def _pretty_int(v: IntVal) -> str:
    if v.const is not None:
        return f"int={v.const}"
    lo, hi = v.lo(), v.hi()
    if lo is not None or hi is not None:
        a = str(lo) if lo is not None else "-∞"
        b = str(hi) if hi is not None else "+∞"
        return f"int[{a}, {b}]"
    if v.sym is not None:
        return f"int={v.sym}"
    return "int"


def _pretty_tensor(v: TensorVal) -> str:
    if v.rank is None:
        base = "Tensor"
    elif v.shape is not None:
        base = "Tensor[" + ", ".join(_dim_str(d) for d in v.shape) + "]"
    else:
        base = "Tensor[" + ", ".join("?" for _ in range(v.rank)) + "]" if v.rank else "Tensor[]"
    extra = []
    if v.dtype is not None:
        extra.append(f"dtype={v.dtype}")
    if v.device is not None:
        extra.append(f"device={v.device}")
    return base + (" {" + ", ".join(extra) + "}" if extra else "")


def pretty_state(s: State) -> str:
    if not s.reachable:
        return "<unreachable>"
    lines = [f"{k} = {pretty(v)}" for k, v in sorted(s.env.items())]
    for obj in sorted(s.store):
        for attr, val in sorted(s.store[obj].items()):
            lines.append(f"{obj}.{attr} = {pretty(val)}")
    return "\n".join(lines)


# ── canonical JSON ──────────────────────────────────────────────────────────
def to_json(v: AbstractValue):
    if isinstance(v, Top):
        return {"k": "top"}
    if isinstance(v, Bottom):
        return {"k": "bottom"}
    if isinstance(v, NoneVal):
        return {"k": "none"}
    if isinstance(v, BoolVal):
        return {"k": "bool", "const": v.const}
    if isinstance(v, IntVal):
        return {
            "k": "int",
            "const": v.const,
            "lo": v.lo(),
            "hi": v.hi(),
            "sym": str(v.sym) if (v.sym is not None and v.const is None) else None,
        }
    if isinstance(v, FloatVal):
        return {"k": "float", "const": v.const}
    if isinstance(v, StrVal):
        return {"k": "str", "const": v.const}
    if isinstance(v, TensorVal):
        return {
            "k": "tensor",
            "rank": v.rank,
            "shape": [_dim_json(d) for d in v.shape] if v.shape is not None else None,
            "dtype": v.dtype,
            "device": v.device,
        }
    if isinstance(v, TupleVal):
        return {"k": "tuple", "elems": [to_json(e) for e in v.elems], "exact": v.exact_len}
    if isinstance(v, ListVal):
        return {
            "k": "list",
            "elem": to_json(v.elem),
            "length": v.length,
            "exact_elems": [to_json(e) for e in v.exact_elems] if v.exact_elems is not None else None,
        }
    if isinstance(v, SetVal):
        return {"k": "set", "elem": to_json(v.elem), "length": v.length}
    if isinstance(v, DictVal):
        return {
            "k": "dict",
            "known": [[key, to_json(val)] for key, val in v.known],  # already canonically sorted
            "value": to_json(v.value),
            "exact_keys": v.exact_keys,
        }
    if isinstance(v, ModuleVal):
        return {"k": "module", "class_name": v.class_name}
    if isinstance(v, CallableVal):
        return {"k": "callable", "func_id": v.func_id}
    return {"k": "top"}


def from_json(d) -> AbstractValue:
    kind = d.get("k")
    if kind == "top":
        return TOP
    if kind == "bottom":
        return BOTTOM
    if kind == "none":
        return NONE
    if kind == "bool":
        return BoolVal(const=d.get("const"))
    if kind == "int":
        if d.get("const") is not None:
            return int_const(d["const"])
        lo, hi = d.get("lo"), d.get("hi")
        if lo is not None or hi is not None:
            return int_range(lo, hi)
        return IntVal()
    if kind == "float":
        return FloatVal(const=d.get("const"))
    if kind == "str":
        return StrVal(const=d.get("const"))
    if kind == "tensor":
        shape = d.get("shape")
        return TensorVal(
            rank=d.get("rank"),
            shape=tuple(_dim_from_json(x) for x in shape) if shape is not None else None,
            dtype=d.get("dtype"),
            device=d.get("device"),
        )
    if kind == "tuple":
        return TupleVal(elems=tuple(from_json(e) for e in d.get("elems", [])), exact_len=d.get("exact", True))
    if kind == "list":
        ex = d.get("exact_elems")
        return ListVal(
            elem=from_json(d.get("elem", {"k": "top"})),
            length=d.get("length"),
            exact_elems=tuple(from_json(e) for e in ex) if ex is not None else None,
        )
    if kind == "set":
        return SetVal(elem=from_json(d.get("elem", {"k": "top"})), length=d.get("length"))
    if kind == "dict":
        known = tuple((k, from_json(val)) for k, val in d.get("known", []))
        return DictVal(value=from_json(d.get("value", {"k": "top"})), known=known, exact_keys=d.get("exact_keys", False))
    if kind == "module":
        return ModuleVal(class_name=d.get("class_name", ""))
    if kind == "callable":
        return CallableVal(func_id=d.get("func_id", ""))
    return TOP


def state_to_json(s: State) -> dict:
    return {
        "reachable": s.reachable,
        "env": {k: to_json(v) for k, v in sorted(s.env.items())},
        "store": {
            obj: {attr: to_json(val) for attr, val in sorted(s.store[obj].items())}
            for obj in sorted(s.store)
        },
    }


def state_from_json(d: dict) -> State:
    st = State(reachable=d.get("reachable", True))
    for k, v in d.get("env", {}).items():
        st.env[k] = from_json(v)
    for obj, attrs in d.get("store", {}).items():
        st.store[obj] = {attr: from_json(val) for attr, val in attrs.items()}
    return st


def dumps(obj) -> str:
    """Canonical (sorted-key) JSON text for a value or state — golden-test ready."""
    payload = state_to_json(obj) if isinstance(obj, State) else to_json(obj)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ── state diffing (Step 19) ─────────────────────────────────────────────────
def _val_eq(a: AbstractValue, b: AbstractValue) -> bool:
    """Lattice equality (mutual ``leq``) — robust to the ``leq == (join==b)``
    definition, where structural ``==`` can spuriously differ."""
    return a.leq(b) and b.leq(a)


@dataclass(frozen=True)
class StateDiff:
    """Structural delta between two :class:`State` snapshots.

    ``changed`` maps a key to a ``(before, after)`` pair.  Keys are the variable
    name for env entries and ``"obj.attr"`` for store entries.  Everything is
    kept canonically sorted so the diff is deterministic (golden-test ready).
    """

    added: Dict[str, AbstractValue] = field(default_factory=dict)
    removed: Dict[str, AbstractValue] = field(default_factory=dict)
    changed: Dict[str, Tuple[AbstractValue, AbstractValue]] = field(default_factory=dict)
    reachable_before: bool = True
    reachable_after: bool = True

    @property
    def is_empty(self) -> bool:
        return (
            not self.added
            and not self.removed
            and not self.changed
            and self.reachable_before == self.reachable_after
        )


def _flatten(s: State) -> Dict[str, AbstractValue]:
    """Flatten a state's env + store into one ``name → value`` map.

    Store entries are keyed ``obj.attr``; plain variables keep their name."""
    out: Dict[str, AbstractValue] = dict(s.env)
    for obj in s.store:
        for attr, val in s.store[obj].items():
            out[f"{obj}.{attr}"] = val
    return out


def diff_states(before: State, after: State) -> StateDiff:
    """Compute the delta ``before → after`` over env and store entries."""
    fb, fa = _flatten(before), _flatten(after)
    added: Dict[str, AbstractValue] = {}
    removed: Dict[str, AbstractValue] = {}
    changed: Dict[str, Tuple[AbstractValue, AbstractValue]] = {}
    for k in sorted(set(fb) | set(fa)):
        in_b, in_a = k in fb, k in fa
        if in_a and not in_b:
            added[k] = fa[k]
        elif in_b and not in_a:
            removed[k] = fb[k]
        elif not _val_eq(fb[k], fa[k]):
            changed[k] = (fb[k], fa[k])
    return StateDiff(
        added=added,
        removed=removed,
        changed=changed,
        reachable_before=before.reachable,
        reachable_after=after.reachable,
    )


def pretty_diff(before: State, after: State) -> str:
    """Render ``before → after`` as a unified, deterministic, line-oriented diff.

    Markers: ``+`` added, ``-`` removed, ``~`` changed (``old → new``).  Returns
    ``"(no change)"`` when the states are equal."""
    d = diff_states(before, after)
    lines: List[str] = []
    if d.reachable_before != d.reachable_after:
        lines.append(
            f"! reachable: {str(d.reachable_before).lower()} → {str(d.reachable_after).lower()}"
        )
    for k in sorted(d.removed):
        lines.append(f"- {k}: {pretty(d.removed[k])}")
    for k in sorted(d.added):
        lines.append(f"+ {k}: {pretty(d.added[k])}")
    for k in sorted(d.changed):
        old, new = d.changed[k]
        lines.append(f"~ {k}: {pretty(old)} → {pretty(new)}")
    return "\n".join(lines) if lines else "(no change)"


def diff_to_json(before: State, after: State) -> dict:
    """Canonical JSON form of the state delta (sorted keys, round-trip-free)."""
    d = diff_states(before, after)
    return {
        "added": {k: to_json(v) for k, v in sorted(d.added.items())},
        "removed": {k: to_json(v) for k, v in sorted(d.removed.items())},
        "changed": {
            k: {"before": to_json(old), "after": to_json(new)}
            for k, (old, new) in sorted(d.changed.items())
        },
        "reachable_before": d.reachable_before,
        "reachable_after": d.reachable_after,
    }
