"""
soundness_contract.py
=====================

The **precise soundness contract** for TensorGuard (100_STEPS.md Step 5).

This module is the single, importable source of truth for *exactly* what
TensorGuard guarantees: which programs it promises never to miss-pass, and
which constructs it over-approximates, under-approximates, or skips. The
companion document ``SOUNDNESS_CONTRACT.md`` at the repo root is generated
from this module (``render_markdown``) and kept in sync by
``tests/test_soundness_contract.py``.

Two independent directions
--------------------------
A static verifier has two soundness directions; we state both explicitly
because TensorGuard's guarantees differ between them:

  * **Refutation soundness** ("no false alarm"): if TensorGuard reports a
    bug (``Refuted-Proof``) for an in-fragment module, the bug is real. This
    is the direction backed by Z3 discharge and the 0%-false-positive audit.

  * **Verification soundness** ("never miss-pass"): if TensorGuard reports a
    module SAFE, the module is genuinely free of the *modeled* bug classes.
    This direction is **scoped**: it holds only for modules inside the
    *verifiable fragment* (``src/verifiable_fragment.py``), for the *modeled
    bug classes* (shape/device/gradient), and only over the *sound* operator
    transfer functions. Everything outside that scope is enumerated below as
    OVER_APPROXIMATED, UNDER_APPROXIMATED, or SKIPPED — including the
    currently-known unsoundness gaps (see ``KNOWN_UNSOUNDNESS``), which are
    documented rather than hidden.

Nothing here is aspirational: every clause cites a concrete code symbol or a
documented behaviour, and the empirical clauses are pinned by tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from src.verifiable_fragment import UnsupportedCategory


class SoundnessClass(Enum):
    """How a construct/behaviour relates to the soundness contract."""

    SOUND = "sound"
    """Verdict is a genuine proof within the declared fragment+scope; no
    false result in the relevant direction."""

    OVER_APPROXIMATED = "over_approximated"
    """May be conservative (abstain / extra constraints / spurious-looking
    caution) but is never unsound: it will not silently pass a real bug."""

    UNDER_APPROXIMATED = "under_approximated"
    """May MISS bugs (a false-negative is possible). Explicitly out of the
    "never miss-pass" guarantee; listed so users do not over-read SAFE."""

    SKIPPED = "skipped"
    """A construct outside the verifiable fragment. The fragment checker
    (``check_traceability``) detects it; see the per-clause note for how the
    current verifier handles it."""


# Directions a clause speaks to.
REFUTATION = "refutation (no false alarm)"
VERIFICATION = "verification (never miss-pass)"


class SoundnessMode(Enum):
    """Verification strictness mode (Step 7).

    Controls how a verdict is rendered when the verifier cannot fully prove a
    module is in-fragment. Does NOT change which bugs are reported (recall /
    precision are governed separately by ``high_confidence_only``); it only
    governs whether a non-refuted module is reported ``SAFE`` or ``UNKNOWN``.
    """

    SOUND = "sound"
    """Strictest. A ``SAFE`` verdict is the contract PyTorch could rely on: it
    is emitted ONLY when the module is fully inside the verifiable fragment —
    no opaque/out-of-fragment layers, no static fragment violations (e.g.
    data-dependent control flow), and no operators whose transfer function is
    merely ``heuristic``. Anything else becomes ``UNKNOWN`` (never a silent
    ``SAFE``)."""

    BALANCED = "balanced"
    """Default. A ``SAFE`` verdict is downgraded to ``UNKNOWN`` only when the
    verifier hit an opaque (out-of-fragment) layer it could not model."""

    HEURISTIC = "heuristic"
    """Most permissive / best-effort. Abstention is tolerated: a non-refuted
    module is reported ``SAFE`` even if parts were out of fragment."""

    @classmethod
    def from_str(cls, value: "str | SoundnessMode") -> "SoundnessMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"invalid soundness_mode {value!r}; expected one of: {valid}"
            )


@dataclass(frozen=True)
class ContractClause:
    construct: str
    soundness_class: SoundnessClass
    direction: str
    rationale: str
    evidence: str  # code symbol / file / documented behaviour


# ---------------------------------------------------------------------------
# The guarantee statement
# ---------------------------------------------------------------------------
SOUNDNESS_GUARANTEE = (
    "For a module M that lies inside the verifiable fragment V_TG "
    "(src/verifiable_fragment.py: FX-traceable, only supported layers / "
    "functions / methods, no out-of-fragment constructs) and is analysed with "
    "input shapes provided, TensorGuard guarantees:\n"
    "  (R) Refutation soundness: every reported shape/device/gradient "
    "Refuted-Proof is Z3-discharged and corresponds to a real conflict that "
    "makes M unrunnable / mistrained (no false alarm).\n"
    "  (V) Verification soundness (modeled scope): if M is reported SAFE for "
    "an enabled domain, then over the SOUND operator transfer functions and "
    "the modeled bug classes, no violating execution exists. This guarantee "
    "does NOT extend to UNDER_APPROXIMATED operators/bug-classes, nor to "
    "modules outside V_TG (see SKIPPED + KNOWN_UNSOUNDNESS)."
)


# ---------------------------------------------------------------------------
# Domain-level clauses
# ---------------------------------------------------------------------------
DOMAIN_CLAUSES: List[ContractClause] = [
    ContractClause(
        construct="Shape domain (refutation)",
        soundness_class=SoundnessClass.SOUND,
        direction=REFUTATION,
        rationale="Shape conflicts are encoded as Z3 constraints and a bug is "
                  "emitted only when the solver proves unsatisfiability.",
        evidence="src/model_checker.py:_encode_shape_safety + _z3_check_safety",
    ),
    ContractClause(
        construct="Shape domain (verification, in-fragment, shapes given)",
        soundness_class=SoundnessClass.SOUND,
        direction=VERIFICATION,
        rationale="A SAFE verdict means Z3 found no shape-violating model over "
                  "the modeled (sound) shape transfer functions.",
        evidence="src/model_checker.py ConstraintVerifier.verify",
    ),
    ContractClause(
        construct="Device domain (requires check_devices)",
        soundness_class=SoundnessClass.SOUND,
        direction=REFUTATION,
        rationale="Device-mismatch refutations are Z3-discharged; contributes "
                  "real bugs the shape view misses.",
        evidence="experiments_v5/run_domain_contribution.py; "
                 "tests/test_domain_contribution.py",
    ),
    ContractClause(
        construct="Gradient domain (requires check_gradients)",
        soundness_class=SoundnessClass.SOUND,
        direction=REFUTATION,
        rationale="Gradient-flow refutations (e.g. detach on the trainable "
                  "path) are Z3-discharged.",
        evidence="src/model_checker.py:_encode_gradient_safety; "
                 "tests/test_domain_contribution.py",
    ),
    ContractClause(
        construct="Phase domain (train/eval)",
        soundness_class=SoundnessClass.OVER_APPROXIMATED,
        direction=VERIFICATION,
        rationale="DIAGNOSTIC-ONLY: registers well-formedness constraints for "
                  "BatchNorm/Dropout but does not refute, so it never produces "
                  "a false alarm and never claims to verify a phase property.",
        evidence="experiments_v5/domain_corpus/phase_01_batchnorm_dropout.py; "
                 "tests/test_domain_contribution.py::"
                 "test_phase_domain_is_diagnostic_only",
    ),
]


# ---------------------------------------------------------------------------
# Bug classes that are explicitly out of the "never miss-pass" guarantee
# ---------------------------------------------------------------------------
UNDER_APPROXIMATED_BUG_CLASSES: List[ContractClause] = [
    ContractClause(
        construct="Value/data-dependent shape bugs (shape depends on tensor "
                  "*values*, not just declared shapes)",
        soundness_class=SoundnessClass.UNDER_APPROXIMATED,
        direction=VERIFICATION,
        rationale="TensorGuard reasons about shapes symbolically, not values; "
                  "a bug that only manifests for particular runtime values may "
                  "be missed.",
        evidence="documented silent-miss rows in "
                 "reproducibility/reproduce_headline_60bug.json (silent_miss)",
    ),
    ContractClause(
        construct="Numerical / dtype-precision bugs (overflow, NaN, precision "
                  "loss) that do not change shapes or devices",
        soundness_class=SoundnessClass.UNDER_APPROXIMATED,
        direction=VERIFICATION,
        rationale="Out of scope: the modeled domains are shape/device/phase/"
                  "gradient, not numerical value semantics.",
        evidence="src/api.py BugCategory enum (no numeric-value domain)",
    ),
    ContractClause(
        construct="Operators with heuristic transfer functions",
        soundness_class=SoundnessClass.UNDER_APPROXIMATED,
        direction=VERIFICATION,
        rationale="Operators whose transfer function is tagged heuristic (vs. "
                  "sound) may admit a violating execution that the model does "
                  "not capture. The per-operator tag is the subject of Step 6.",
        evidence="100_STEPS.md Step 6 (machine-readable sound/complete/"
                 "heuristic operator table)",
    ),
]


# ---------------------------------------------------------------------------
# Out-of-fragment constructs (one clause per UnsupportedCategory)
# ---------------------------------------------------------------------------
_CATEGORY_NOTES = {
    UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW:
        "branch taken depends on tensor values (e.g. `if x.sum() > 0:`)",
    UnsupportedCategory.DATA_DEPENDENT_ITERATION:
        "loop bound depends on a tensor value / dynamic length",
    UnsupportedCategory.DYNAMIC_ASSERTION:
        "runtime assert on tensor contents",
    UnsupportedCategory.TENSOR_TO_SCALAR:
        "tensor coerced to a Python scalar (`.item()`, `int(t)`)",
    UnsupportedCategory.CUSTOM_AUTOGRAD:
        "custom torch.autograd.Function with opaque shape semantics",
    UnsupportedCategory.INPLACE_MUTATION:
        "in-place mutation that the static model does not track",
    UnsupportedCategory.JIT_SCRIPT:
        "torch.jit.script / scripted submodule",
    UnsupportedCategory.OPAQUE_EXTERNAL_CALL:
        "call into a function the analyzer cannot resolve",
    UnsupportedCategory.DYNAMIC_MODULE_CONSTRUCTION:
        "modules built from data-dependent configuration at runtime",
    UnsupportedCategory.UNSUPPORTED_BUILTIN:
        "unsupported Python builtin in forward",
    UnsupportedCategory.OTHER:
        "any other construct outside V_TG",
}


def _out_of_fragment_clauses() -> List[ContractClause]:
    clauses: List[ContractClause] = []
    for cat in UnsupportedCategory:
        note = _CATEGORY_NOTES.get(cat, cat.name.lower())
        clauses.append(ContractClause(
            construct=f"Out-of-fragment: {cat.name} ({note})",
            soundness_class=SoundnessClass.SKIPPED,
            direction=VERIFICATION,
            rationale="Outside the verifiable fragment. `check_traceability` "
                      "detects it (in_verifiable_fragment=False). NOTE: the "
                      "current `verify_architecture` does not yet gate on the "
                      "fragment, so such a module may receive a silent SAFE "
                      "(see KNOWN_UNSOUNDNESS U1); Step 8 makes this an "
                      "explicit `unknown`/abstain.",
            evidence="src/verifiable_fragment.py UnsupportedCategory."
                     f"{cat.name}; check_traceability",
        ))
    return clauses


OUT_OF_FRAGMENT_CLAUSES: List[ContractClause] = _out_of_fragment_clauses()


# ---------------------------------------------------------------------------
# Known unsoundness gaps — surfaced, not hidden
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnownGap:
    id: str
    description: str
    affected_direction: str
    location: str
    remediation: str


KNOWN_UNSOUNDNESS: List[KnownGap] = [
    KnownGap(
        id="U1",
        description="verify_architecture does not gate on the verifiable "
                    "fragment: an out-of-fragment module (e.g. data-dependent "
                    "control flow) can be reported SAFE/Verified instead of "
                    "abstaining, so a real bug hidden by the unmodeled "
                    "construct can be missed.",
        affected_direction=VERIFICATION,
        location="src/api.py verify_architecture (no check_traceability gate)",
        remediation="100_STEPS.md Step 8: report out-of-fragment constructs as "
                    "`unknown`/abstain rather than silent pass.",
    ),
    KnownGap(
        id="U2",
        description="shape_cegar may return CEGARStatus.SAFE when the "
                    "accumulated refined predicates are jointly infeasible "
                    "(SAFE-on-infeasible), which is unsound for the refined "
                    "contract. Step 1 works around it by emitting a "
                    "cegar_refined_contract bug from the iteration-log union, "
                    "but the SAFE-on-infeasible return remains.",
        affected_direction=VERIFICATION,
        location="src/shape_cegar.py (SAFE return on infeasible accumulated "
                 "predicates, ~lines 2954-2962)",
        remediation="Return REFUTED/UNKNOWN when accumulated predicates are "
                    "infeasible (later phase).",
    ),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def all_clauses() -> List[ContractClause]:
    return (DOMAIN_CLAUSES + UNDER_APPROXIMATED_BUG_CLASSES
            + OUT_OF_FRAGMENT_CLAUSES)


def render_markdown() -> str:
    lines: List[str] = []
    lines.append("# TensorGuard Soundness Contract")
    lines.append("")
    lines.append("> Generated from `src/soundness_contract.py` — the single "
                 "source of truth. Do not edit by hand; run "
                 "`python -m src.soundness_contract > SOUNDNESS_CONTRACT.md` "
                 "and it is pinned by `tests/test_soundness_contract.py`.")
    lines.append("")
    lines.append("## The guarantee")
    lines.append("")
    for para in SOUNDNESS_GUARANTEE.split("\n"):
        lines.append(para)
    lines.append("")
    lines.append("## Domains")
    lines.append("")
    lines.append("| Construct | Class | Direction | Rationale | Evidence |")
    lines.append("|-----------|-------|-----------|-----------|----------|")
    for c in DOMAIN_CLAUSES:
        lines.append(f"| {c.construct} | `{c.soundness_class.value}` | "
                     f"{c.direction} | {c.rationale} | `{c.evidence}` |")
    lines.append("")
    lines.append("## Bug classes outside the 'never miss-pass' guarantee "
                 "(UNDER_APPROXIMATED)")
    lines.append("")
    lines.append("| Construct | Class | Rationale | Evidence |")
    lines.append("|-----------|-------|-----------|----------|")
    for c in UNDER_APPROXIMATED_BUG_CLASSES:
        lines.append(f"| {c.construct} | `{c.soundness_class.value}` | "
                     f"{c.rationale} | `{c.evidence}` |")
    lines.append("")
    lines.append("## Out-of-fragment constructs (SKIPPED)")
    lines.append("")
    lines.append("These are detected by `check_traceability` "
                 "(`in_verifiable_fragment=False`).")
    lines.append("")
    lines.append("| Construct | Class | Note |")
    lines.append("|-----------|-------|------|")
    for c in OUT_OF_FRAGMENT_CLAUSES:
        lines.append(f"| {c.construct} | `{c.soundness_class.value}` | "
                     f"{c.rationale} |")
    lines.append("")
    lines.append("## Known unsoundness gaps (surfaced, not hidden)")
    lines.append("")
    lines.append("| ID | Affected direction | Description | Location | "
                 "Remediation |")
    lines.append("|----|--------------------|-------------|----------|"
                 "-------------|")
    for g in KNOWN_UNSOUNDNESS:
        lines.append(f"| {g.id} | {g.affected_direction} | {g.description} | "
                     f"`{g.location}` | {g.remediation} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_markdown())
