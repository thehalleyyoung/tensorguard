"""Reduced-product vs independent-domains ablation: precision gain (Step 118).

TensorGuard's Python-level analysis runs a *reduced* product of three abstract
domains -- interval, type-tag and nullity -- with inter-domain *reductions* that
propagate information between them (``src/domains/product.py``). The classic
abstract-interpretation question is: what does the reduction buy over the
*independent* (direct) product, in which the three domains never talk to each
other? This harness answers it on real code, two ways at once:

* **Real verifier internals.** We run the *actual* ``ProductInterpreter`` over
  small IR programs twice -- once with the real ``ReductionEngine`` (reduced
  product) and once with an empty reduction set (independent product) -- and
  compare the null-dereference verdicts the production ``NullDerefChecker``
  emits. The only thing that differs between the two runs is whether reductions
  fire.

* **Real CPython oracle.** Every IR program mirrors a concrete Python function.
  We *execute* that function under CPython over a battery of concrete inputs and
  record whether a null dereference is genuinely reachable. This is the ground
  truth that turns "the independent product emits a warning" into the stronger,
  checkable claim "the independent product emits a *false* warning".

The headline precision metric is the number of *spurious* null-deref warnings
the independent product raises that the reduced product eliminates and that the
CPython oracle confirms are unreachable. Two honesty guards accompany it:

1. **No recall loss.** On every program where the oracle shows a null deref is
   genuinely reachable, the reduced product must still warn (it must never
   suppress a real bug).
2. **Lattice refinement.** For every program and variable, the reduced abstract
   value must be ``leq`` the independent one (``γ(reduced) ⊆ γ(independent)``):
   the reduction only ever moves *down* the lattice, so it is a sound, monotone
   precision improvement, never an unsound shortcut.

Only counts and rounded rates are recorded, so the artifact is byte-identical
across machines.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.domains.base import AbstractState, IRNode  # noqa: E402
from src.domains.intervals import Interval, IntervalValue  # noqa: E402
from src.domains.nullity import NullityValue  # noqa: E402
from src.domains.product import (  # noqa: E402
    ProductInterpreter,
    ProductValue,
    ReducedProductDomain,
    ReductionEngine,
)
from src.domains.typetags import TypeTagSet, TypeTagValue  # noqa: E402

OUT_JSON = REPO / "reproducibility" / "reduced_product_ablation.json"
OUT_MD = REPO / "reproducibility" / "reduced_product_ablation.md"


# ---------------------------------------------------------------------------
# Scenario model
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """One labeled program exercised under both products + a CPython oracle."""

    sid: str
    family: str
    # IR program over a single variable ``x`` ending in a dereference check.
    program: List[IRNode]
    init_nullity: NullityValue
    # Concrete Python function (over ``x``) that mirrors the IR, plus the inputs
    # to drive it. The oracle reports whether a None-dereference is reachable.
    pyfunc: Callable[[Any], Any]
    inputs: List[Any]
    note: str = ""


def _initial_x(nv: NullityValue) -> ProductValue:
    return ProductValue(
        interval=IntervalValue(Interval.top()),
        type_tag=TypeTagValue(TypeTagSet.top()),
        nullity=nv,
    )


def _guard(type_name: str) -> List[IRNode]:
    return [IRNode("guard", condition=("isinstance", "x", type_name))]


# Concrete dereferences used by the CPython oracle. ``_deref`` performs a real
# attribute access that CPython resolves successfully on any genuine object but
# that raises ``AttributeError("'NoneType' object ...")`` the moment it is
# applied to ``None`` -- i.e. an authentic null dereference, executed (not
# hand-simulated) by the interpreter.
def _deref(x: Any) -> Any:
    if x is None:
        return x.__tensorguard_probe__()  # real NoneType AttributeError
    return type(x).__name__  # safe: x is a genuine object


def _deref_guarded(type_name: str) -> Callable[[Any], Any]:
    pytype = {
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple,
    }[type_name]

    def f(x: Any) -> Any:
        if isinstance(x, pytype):
            return _deref(x)  # reached only by non-None values of pytype
        return None

    return f


def _deref_unguarded(x: Any) -> Any:
    return _deref(x)  # reachable by None -> genuine null dereference


# A spread of concrete inputs covering None and several real types, so the
# oracle genuinely exercises the None path where it is reachable.
_PROBE_INPUTS = [None, 0, 1, -3, 3.5, "", "hi", True, [], [1], {}, {"a": 1}, (), (1,)]


def _build_scenarios() -> List[Scenario]:
    scen: List[Scenario] = []

    # Family G: an isinstance guard on a concrete non-None type. Under the guard
    # the value can never be None, so a dereference is null-safe -- but the
    # nullity domain alone does not learn this from isinstance, so the
    # independent product false-alarms. The reduced product's TypeTag->Nullity
    # reduction removes the warning.
    for t in ("int", "float", "str", "bool", "list", "dict", "tuple"):
        scen.append(
            Scenario(
                sid=f"guard_{t}",
                family="guarded_precise",
                program=_guard(t),
                init_nullity=NullityValue.maybe_null(),
                pyfunc=_deref_guarded(t),
                inputs=list(_PROBE_INPUTS),
                note=f"isinstance(x,{t}) then deref -> null-safe",
            )
        )

    # Family N: genuinely maybe-null, no refining guard. The deref IS reachable
    # on the None input, so BOTH products must warn (reduced loses no recall).
    scen.append(
        Scenario(
            sid="unguarded_maybe",
            family="genuine_null",
            program=[],
            init_nullity=NullityValue.maybe_null(),
            pyfunc=_deref_unguarded,
            inputs=list(_PROBE_INPUTS),
            note="unguarded deref of a maybe-null value",
        )
    )
    # An isinstance guard for a type the value is NOT (false branch keeps it
    # possibly-None) does not help; model the residual maybe-null deref.
    scen.append(
        Scenario(
            sid="unguarded_maybe_2",
            family="genuine_null",
            program=[IRNode("noop")],
            init_nullity=NullityValue.maybe_null(),
            pyfunc=_deref_unguarded,
            inputs=[None, 1, "x"],
            note="maybe-null deref after a no-op",
        )
    )

    # Family D: definitely null. Both products must report a definite error;
    # the oracle (a literal None) confirms the deref always raises.
    scen.append(
        Scenario(
            sid="definitely_null",
            family="definite_null",
            program=[],
            init_nullity=NullityValue.definitely_null(),
            pyfunc=_deref_unguarded,
            inputs=[None],
            note="definite null deref",
        )
    )
    return scen


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def _run_product(scen: Scenario, *, reduced: bool) -> Tuple[Optional[dict], ProductValue]:
    """Run the real ProductInterpreter; return (null-safety verdict, final x)."""
    engine = ReductionEngine() if reduced else ReductionEngine(reductions=[])
    domain = ReducedProductDomain(reduction_engine=engine)
    interp = ProductInterpreter(domain)
    state = AbstractState(env={"x": _initial_x(scen.init_nullity)}, domain=domain)
    state = interp.interpret_block(scen.program, state)
    verdict = interp.check_null_safety(state, "x", "deref")
    final_x = state.get("x")
    return verdict, final_x


def _warns(verdict: Optional[dict]) -> bool:
    return verdict is not None


def _oracle_null_deref_reachable(scen: Scenario) -> bool:
    """True iff executing the concrete function raises a None-attribute error."""
    for inp in scen.inputs:
        try:
            scen.pyfunc(inp)
        except AttributeError as exc:
            if "NoneType" in str(exc):
                return True
    return False


def _leq_refinement(scen: Scenario) -> bool:
    """reduced(x) ⊑ independent(x): the reduction is a sound refinement."""
    engine_r = ReductionEngine()
    engine_i = ReductionEngine(reductions=[])
    dom = ReducedProductDomain(reduction_engine=engine_r)
    interp_r = ProductInterpreter(ReducedProductDomain(reduction_engine=engine_r))
    interp_i = ProductInterpreter(ReducedProductDomain(reduction_engine=engine_i))
    st_r = interp_r.interpret_block(
        scen.program,
        AbstractState(env={"x": _initial_x(scen.init_nullity)}, domain=dom),
    )
    st_i = interp_i.interpret_block(
        scen.program,
        AbstractState(env={"x": _initial_x(scen.init_nullity)}, domain=dom),
    )
    xr, xi = st_r.get("x"), st_i.get("x")
    if xr is None or xi is None:
        return xr is xi
    return dom.leq(xr, xi)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def measure() -> dict:
    scenarios = _build_scenarios()

    independent_fp = 0  # independent warns but oracle says safe
    reduced_fp = 0  # reduced warns but oracle says safe
    precision_gain = 0  # independent warns, reduced does not, oracle safe
    reduced_misses = 0  # oracle unsafe but reduced does not warn
    independent_misses = 0
    refinements_hold = 0
    per_scenario = []

    for sc in scenarios:
        v_ind, _ = _run_product(sc, reduced=False)
        v_red, _ = _run_product(sc, reduced=True)
        unsafe = _oracle_null_deref_reachable(sc)
        leq_ok = _leq_refinement(sc)
        refinements_hold += int(leq_ok)

        ind_w = _warns(v_ind)
        red_w = _warns(v_red)

        if not unsafe and ind_w:
            independent_fp += 1
        if not unsafe and red_w:
            reduced_fp += 1
        if not unsafe and ind_w and not red_w:
            precision_gain += 1
        if unsafe and not red_w:
            reduced_misses += 1
        if unsafe and not ind_w:
            independent_misses += 1

        per_scenario.append(
            {
                "id": sc.sid,
                "family": sc.family,
                "oracle_null_deref_reachable": unsafe,
                "independent_warns": ind_w,
                "reduced_warns": red_w,
                "is_precision_gain": (not unsafe) and ind_w and (not red_w),
                "leq_refinement_holds": leq_ok,
            }
        )

    n = len(scenarios)
    n_guarded = sum(1 for s in scenarios if s.family == "guarded_precise")
    n_genuine = sum(1 for s in scenarios if s.family in ("genuine_null", "definite_null"))

    data = {
        "step": 118,
        "n_scenarios": n,
        "n_guarded_precise": n_guarded,
        "n_genuine_or_definite_null": n_genuine,
        "independent_false_positives": independent_fp,
        "reduced_false_positives": reduced_fp,
        "precision_gain_cases": precision_gain,
        "false_positives_eliminated": independent_fp - reduced_fp,
        "reduced_misses_real_null": reduced_misses,
        "independent_misses_real_null": independent_misses,
        "lattice_refinement_holds_all": refinements_hold == n,
        "n_refinements_checked": n,
        # Honest headline assertions.
        "reduced_product_strictly_more_precise": precision_gain > 0
        and reduced_fp == 0
        and independent_fp > 0,
        "no_recall_loss": reduced_misses == 0,
        "reduced_is_sound_refinement": refinements_hold == n,
        "per_scenario": per_scenario,
    }
    return data


def render_markdown(d: dict) -> str:
    lines = [
        "# Reduced product vs independent domains: precision gain (Step 118)",
        "",
        f"{d['n_scenarios']} labeled single-variable programs run through the "
        "*real* `ProductInterpreter` twice -- with the production "
        "`ReductionEngine` (reduced product) and with an empty reduction set "
        "(independent product) -- and cross-checked against a CPython execution "
        "oracle.",
        "",
        "## Headline",
        "",
        f"- false null-deref warnings under the independent product: "
        f"**{d['independent_false_positives']}**",
        f"- false null-deref warnings under the reduced product: "
        f"**{d['reduced_false_positives']}**",
        f"- spurious warnings the reduction eliminates (oracle-confirmed safe): "
        f"**{d['false_positives_eliminated']}**",
        f"- reduced product strictly more precise: "
        f"**{d['reduced_product_strictly_more_precise']}**",
        f"- real null derefs missed by the reduced product (recall loss): "
        f"**{d['reduced_misses_real_null']}**",
        f"- reduced value ⊑ independent value on every program "
        f"(γ(reduced) ⊆ γ(independent)): **{d['reduced_is_sound_refinement']}**",
        "",
        "## Per-scenario",
        "",
        "| id | family | oracle null-deref | independent warns | reduced warns "
        "| precision gain | refinement |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in d["per_scenario"]:
        lines.append(
            f"| {s['id']} | {s['family']} | {s['oracle_null_deref_reachable']} "
            f"| {s['independent_warns']} | {s['reduced_warns']} "
            f"| {s['is_precision_gain']} | {s['leq_refinement_holds']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(check: bool = False) -> int:
    data = measure()
    js = json.dumps(data, indent=2, sort_keys=True) + "\n"
    md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != js:
            print(f"MISMATCH: {OUT_JSON}")
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != md:
            print(f"MISMATCH: {OUT_MD}")
            ok = False
        if ok:
            print("reduced_product_ablation: byte-identical")
        return 0 if ok else 1
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
