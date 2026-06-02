#!/usr/bin/env python3
"""Step 131 — differentially test the Python reduced-product reduction against a
*verified reference checker extracted from Lean*.

`lean/TensorGuard/ReducedProduct.lean` defines the reduced-product reduction and
proves it reductive, monotone, and concretization-preserving (Steps 126–128).
Because those definitions are executable, we *extract* a reference checker
directly from the verified model: enumerating all sixteen Tag×Nullity abstract
values and tabulating `reduce p` and `γ(p)` (committed verbatim in
`reproducibility/lean_reduction_extracted.txt`, regenerable by running Lean —
see `tests/test_verified_reduction_diff.py`).

This harness instantiates each abstract value as a concrete `ProductValue` and
runs the **real** Python reductions from `src/domains/product.py`
(`TypeTagToNullityReduction` then `NullityToTypeTagReduction`, the same order as
the Lean `reduce`), abstracts the result back, and diffs it against the verified
table. It reports, per input:

* whether the Python result **agrees** with the verified reduction, and
* whether the Python result is a **sound over-approximation** — its
  concretization contains the verified concretization (`γ_python ⊇ γ_lean`),
  the only thing soundness requires.

The audit is honest about what it finds: the Python reduction agrees with the
verified model on every *consistent* (reachable) abstract value, and is a sound
over-approximation on **all** sixteen, but it **diverges on contradictory
(unreachable) inputs** — where the verified model collapses to ⊥ but the Python
rule overwrites the conflicting nullity instead of detecting the contradiction.
Those states have empty concretization, so the divergence is a *precision* gap
(failure to prove unreachability), never an unsoundness.

Deterministic: byte-identical JSON/Markdown across runs; `--check` re-verifies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domains.intervals import Interval, IntervalValue  # noqa: E402
from src.domains.nullity import NullityValue  # noqa: E402
from src.domains.product import (  # noqa: E402
    NullityToTypeTagReduction,
    ProductValue,
    TypeTagToNullityReduction,
)
from src.domains.typetags import TypeTagSet, TypeTagValue  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE = os.path.join(_HERE, "lean_reduction_extracted.txt")
JSON_PATH = os.path.join(_HERE, "verified_reduction_diff.json")
MD_PATH = os.path.join(_HERE, "verified_reduction_diff.md")

# A representative non-None type tag used to realise Lean's `mayOther` bit.
_OTHER = "int"

_NUL_FROM_NAME = {
    "bot": lambda: NullityValue.bottom(),
    "null": lambda: NullityValue.definitely_null(),
    "notnull": lambda: NullityValue.definitely_not_null(),
    "top": lambda: NullityValue.maybe_null(),
}


def _tag_from_bits(may_none: bool, may_other: bool) -> TypeTagValue:
    names = set()
    if may_none:
        names.add("NoneType")
    if may_other:
        names.add(_OTHER)
    if not names:
        return TypeTagValue(TypeTagSet.bottom())
    return TypeTagValue(TypeTagSet(tags=frozenset(names)))


def _pval_from_code(code: str) -> ProductValue:
    """`code` is e.g. ``10:null`` (mayNone, mayOther, nullity-name)."""
    bits, nul_name = code.split(":")
    may_none, may_other = bits[0] == "1", bits[1] == "1"
    return ProductValue(
        interval=IntervalValue(Interval.top()),
        type_tag=_tag_from_bits(may_none, may_other),
        nullity=_NUL_FROM_NAME[nul_name](),
    )


def _abstract(pv: ProductValue) -> str:
    """Map a Python ProductValue back to the Lean ``mm:nul`` encoding."""
    if pv.is_bottom():
        return "00:bot"
    tags = pv.type_tag.tag_set
    if tags.is_top:
        may_none = may_other = True
    else:
        may_none = "NoneType" in tags.tags
        may_other = any(t != "NoneType" for t in tags.tags)
    if not may_none and not may_other:
        return "00:bot"
    nv = pv.nullity
    if nv.is_bottom():
        return "00:bot"
    if nv.is_definitely_null:
        nul = "null"
    elif nv.is_definitely_not_null:
        nul = "notnull"
    else:
        nul = "top"
    return f"{int(may_none)}{int(may_other)}:{nul}"


def _python_reduce(code: str) -> str:
    pv = _pval_from_code(code)
    # Same composition order as the verified Lean `reduce`.
    r1 = TypeTagToNullityReduction().apply(pv)
    r2 = NullityToTypeTagReduction().apply(r1)
    return _abstract(r2)


def _gamma(code: str) -> frozenset:
    """Concretization of an abstract code into {cnone, cobj}, matching Lean."""
    if code == "00:bot":
        return frozenset()
    bits, nul = code.split(":")
    may_none, may_other = bits[0] == "1", bits[1] == "1"
    allowed = set()
    if nul == "bot":
        return frozenset()
    for c, tag_ok in (("cnone", may_none), ("cobj", may_other)):
        nul_ok = (
            (nul == "top")
            or (nul == "null" and c == "cnone")
            or (nul == "notnull" and c == "cobj")
        )
        if tag_ok and nul_ok:
            allowed.add(c)
    return frozenset(allowed)


def _is_consistent(code: str) -> bool:
    """A reachable (non-contradictory) abstract value: non-empty tag, non-⊥
    nullity, and the tag/nullity do not assert opposite nullity facts."""
    bits, nul = code.split(":")
    may_none, may_other = bits[0] == "1", bits[1] == "1"
    if (not may_none and not may_other) or nul == "bot":
        return False
    # tag exactly {None} but nullity says notnull, or tag has no None but
    # nullity says null -> contradictory.
    if may_none and not may_other and nul == "notnull":
        return False
    if not may_none and may_other and nul == "null":
        return False
    return True


def _load_reference():
    rows = []
    with open(_REFERENCE) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            lhs, rest = line.split(" -> ")
            reduced, gamma_part = rest.split(" | gamma=")
            rows.append((lhs, reduced, gamma_part))
    return rows


def build():
    cases = []
    n_agree = 0
    n_sound = 0
    n_consistent = 0
    n_consistent_agree = 0
    divergent_inputs = []
    for code, lean_reduced, _gcode in _load_reference():
        py = _python_reduce(code)
        agree = py == lean_reduced
        g_lean = _gamma(lean_reduced)
        g_py = _gamma(py)
        sound = g_lean <= g_py  # python over-approximates the verified result
        consistent = _is_consistent(code)
        n_agree += int(agree)
        n_sound += int(sound)
        n_consistent += int(consistent)
        if consistent:
            n_consistent_agree += int(agree)
        if not agree:
            divergent_inputs.append(code)
        cases.append({
            "input": code,
            "lean_reduced": lean_reduced,
            "python_reduced": py,
            "agree": agree,
            "input_consistent": consistent,
            "gamma_lean": sorted(g_lean),
            "gamma_python": sorted(g_py),
            "python_overapproximates": sound,
        })

    n = len(cases)
    all_consistent_agree = n_consistent_agree == n_consistent
    all_sound = n_sound == n
    divergences_only_contradictory = all(
        not _is_consistent(c) for c in divergent_inputs)

    data = {
        "step": 131,
        "title": "Differential test of Python reduced-product reduction vs "
                 "Lean-extracted verified checker",
        "reference_source": "lean/TensorGuard/ReducedProduct.lean "
                            "(extracted to reproducibility/lean_reduction_extracted.txt)",
        "n_abstract_values": n,
        "n_agree": n_agree,
        "n_consistent_inputs": n_consistent,
        "n_consistent_agree": n_consistent_agree,
        "n_python_sound_overapproximation": n_sound,
        "divergent_inputs": sorted(divergent_inputs),
        "findings": {
            "python_matches_verified_on_all_consistent_inputs": all_consistent_agree,
            "python_is_sound_overapproximation_on_all_inputs": all_sound,
            "divergences_only_on_contradictory_unreachable_inputs":
                divergences_only_contradictory,
        },
        "interpretation": (
            "On every reachable (consistent) abstract value the Python reduction "
            "is byte-identical to the verified Lean reduction. On all sixteen "
            "values it is a sound over-approximation (gamma_python contains "
            "gamma_lean). The only divergences are on contradictory, unreachable "
            "inputs (empty concretization), where the verified model collapses to "
            "bottom but the Python rule overwrites the conflicting nullity rather "
            "than proving unreachability -- a precision gap, never an unsoundness."
        ),
        "cases": cases,
    }
    return data


def render_md(data) -> str:
    lines = [
        f"# Step {data['step']} — {data['title']}",
        "",
        f"Reference: `{data['reference_source']}`.",
        "",
        f"- Abstract values audited: **{data['n_abstract_values']}**",
        f"- Agree with verified reduction: **{data['n_agree']}**",
        f"- Consistent (reachable) inputs: **{data['n_consistent_inputs']}**, "
        f"of which agree: **{data['n_consistent_agree']}**",
        f"- Python a sound over-approximation: "
        f"**{data['n_python_sound_overapproximation']}** of "
        f"{data['n_abstract_values']}",
        "",
        "## Findings",
        "",
    ]
    for k, v in data["findings"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["", data["interpretation"], "", "## Per-input diff", "",
              "| input | lean reduce | python reduce | agree | consistent | "
              "γ(lean) | γ(python) | python ⊇ |",
              "|---|---|---|---|---|---|---|---|"]
    for c in data["cases"]:
        lines.append(
            f"| `{c['input']}` | `{c['lean_reduced']}` | `{c['python_reduced']}` | "
            f"{c['agree']} | {c['input_consistent']} | "
            f"{{{','.join(c['gamma_lean'])}}} | {{{','.join(c['gamma_python'])}}} | "
            f"{c['python_overapproximates']} |")
    lines.append("")
    return "\n".join(lines)


def _write(data):
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    with open(JSON_PATH, "w") as fh:
        fh.write(payload)
    with open(MD_PATH, "w") as fh:
        fh.write(render_md(data))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify committed artifacts are byte-identical")
    args = ap.parse_args()
    data = build()
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if args.check:
        ok = True
        if not os.path.exists(JSON_PATH):
            print("MISSING", JSON_PATH)
            return 1
        with open(JSON_PATH) as fh:
            if fh.read() != payload:
                print("DRIFT", JSON_PATH)
                ok = False
        if os.path.exists(MD_PATH):
            with open(MD_PATH) as fh:
                if fh.read() != render_md(data):
                    print("DRIFT", MD_PATH)
                    ok = False
        print("OK" if ok else "FAIL")
        return 0 if ok else 1
    _write(data)
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
