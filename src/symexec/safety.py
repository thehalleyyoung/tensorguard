"""Positive safety reports — *"why is this safe?"* (even_more Tier 5 #14).

Every other surface in this engine answers *"why is this a bug?"*.  A sound
analyser can also answer the dual, far rarer question only sound tools can make:
*"what did you prove is safe?"*  This module renders that positive report from a
:class:`~src.symexec.engine.SymResult`.

It invents nothing: a safety report is a faithful re-presentation of facts the
analysis already computed —

* the **verdict**: whether any *sound* (forced-failure) bug was provable;
* the **covered fragment**: the coverage profile (:class:`CoverageMeter`) — how
  much of the file the engine actually reasoned about with non-``Top`` values;
* the **guarantee**: the relative-completeness clauses
  (:data:`~src.symexec.completeness_contract.COMPLETE_FOR`) — the bug kinds for
  which *absence of a report on the covered fragment is a positive guarantee*
  (any genuine forced failure on known operands would have been reported); and
* the **boundary**: the abstain ledger (:class:`AbstainLedger`) — exactly where
  the engine chose ``Top`` and so the safety claim does **not** extend.

The honesty discipline is the same one that keeps the engine sound: the safety
claim is scoped to the *covered* fragment for the *complete-for* kinds, and the
abstain ledger names every place that scope stops.  The module is pure and read
only — it never runs the engine, mutates a result, or emits a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

from .completeness_contract import COMPLETE_FOR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .engine import SymResult

__all__ = [
    "SafetyReport",
    "safety_report",
    "render_safety_report",
    "explain_safety",
]


# Sound (forced-failure) reports are emitted at error severity; heuristic /
# intent suspicions are emitted as warnings and never carry a safety claim.
_WARNING_SEVERITY = "warning"


@dataclass(frozen=True)
class SafetyReport:
    """A faithful, structured re-presentation of a result's *positive* facts.

    ``proven_safe`` is ``True`` exactly when the analysis derived no sound
    forced-failure bug; it is a guarantee **scoped to** ``complete_for_kinds`` on
    the covered fragment, with ``abstain_by_category`` marking where that scope
    stops.  ``coverage`` / ``value_coverage`` are the fractions reported by the
    :class:`CoverageMeter`.
    """

    filename: str
    proven_safe: bool
    sound_bug_count: int
    heuristic_bug_count: int
    functions_analyzed: int
    total_statements: int
    covered_statements: int
    coverage: float
    value_coverage: float
    complete_for_kinds: Tuple[str, ...]
    abstain_total: int
    abstain_by_category: Tuple[Tuple[str, int], ...]
    abstain_by_detector: Tuple[Tuple[str, int], ...]
    fingerprint: str

    def to_dict(self) -> dict:
        """A stable, JSON-ready snapshot of the safety report."""
        return {
            "filename": self.filename,
            "proven_safe": self.proven_safe,
            "sound_bug_count": self.sound_bug_count,
            "heuristic_bug_count": self.heuristic_bug_count,
            "functions_analyzed": self.functions_analyzed,
            "total_statements": self.total_statements,
            "covered_statements": self.covered_statements,
            "coverage": round(self.coverage, 6),
            "value_coverage": round(self.value_coverage, 6),
            "complete_for_kinds": list(self.complete_for_kinds),
            "abstain_total": self.abstain_total,
            "abstain_by_category": [
                {"category": cat, "count": cnt}
                for cat, cnt in self.abstain_by_category
            ],
            "abstain_by_detector": [
                {"detector": det, "count": cnt}
                for det, cnt in self.abstain_by_detector
            ],
            "fingerprint": self.fingerprint,
        }


def _complete_for_kinds() -> Tuple[str, ...]:
    """The sound bug kinds that carry a relative-completeness guarantee, in a
    stable order (deduplicated, source order preserved)."""
    seen: Dict[str, None] = {}
    for clause in COMPLETE_FOR:
        seen.setdefault(clause.kind, None)
    return tuple(seen.keys())


def safety_report(result: "SymResult", *, filename: str = "<unknown>") -> SafetyReport:
    """Build the structured :class:`SafetyReport` for an analysed ``result``.

    A *sound* bug is any report not emitted at ``warning`` severity (heuristic /
    intent suspicions are warnings and never bear on the safety verdict)."""
    sound_bugs = [b for b in result.bugs if b.severity != _WARNING_SEVERITY]
    heuristic_bugs = [b for b in result.bugs if b.severity == _WARNING_SEVERITY]

    cov = result.coverage
    ledger = result.abstentions

    by_cat = sorted(
        ((cat.value, cnt) for cat, cnt in ledger.coverage().items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    by_det = sorted(
        ledger.by_detector().items(), key=lambda kv: (-kv[1], kv[0])
    )

    return SafetyReport(
        filename=filename,
        proven_safe=not sound_bugs,
        sound_bug_count=len(sound_bugs),
        heuristic_bug_count=len(heuristic_bugs),
        functions_analyzed=result.functions_analyzed,
        total_statements=cov.total,
        covered_statements=cov.non_top,
        coverage=cov.coverage,
        value_coverage=cov.value_coverage,
        complete_for_kinds=_complete_for_kinds(),
        abstain_total=ledger.total,
        abstain_by_category=tuple(by_cat),
        abstain_by_detector=tuple(by_det),
        fingerprint=result.fingerprint(),
    )


def _verdict_line(r: SafetyReport) -> str:
    if not r.proven_safe:
        return (
            f"❌ **Not safe** — {r.sound_bug_count} sound forced-failure "
            f"bug(s) were proven on the covered fragment."
        )
    if r.heuristic_bug_count:
        return (
            "✅ **No forced-failure bug was provable** on the covered fragment "
            f"— but {r.heuristic_bug_count} heuristic suspicion(s) were raised "
            "(outside the proven fragment; review separately)."
        )
    return "✅ **No forced-failure bug was provable** on the covered fragment."


def render_safety_report(r: SafetyReport) -> str:
    """Render a :class:`SafetyReport` as a deterministic Markdown document."""
    lines: List[str] = []
    lines.append(f"# Why is `{r.filename}` safe?")
    lines.append("")
    lines.append(_verdict_line(r))
    lines.append("")

    lines.append("## What was analysed")
    lines.append("")
    lines.append(
        f"- Functions analysed: **{r.functions_analyzed}**"
    )
    lines.append(
        f"- Statement coverage: **{r.covered_statements}/{r.total_statements}** "
        f"reasoned about with known (non-⊤) values "
        f"(**{r.coverage:.0%}**); value coverage **{r.value_coverage:.0%}**."
    )
    lines.append("")

    lines.append("## What the absence of a report guarantees")
    lines.append("")
    if r.proven_safe:
        lines.append(
            "On the covered fragment, for the bug kinds below, a genuine forced "
            "failure on *known* operands would have been **reported** "
            "(relative-completeness, the ⇒ direction of the soundness/"
            "completeness contract). No such report means no such failure was "
            "provable here:"
        )
    else:
        lines.append(
            "These are the bug kinds the engine is relative-complete for; on the "
            "covered fragment any genuine forced failure on known operands is "
            "reported (see the verdict above):"
        )
    lines.append("")
    for kind in r.complete_for_kinds:
        lines.append(f"- `{kind}`")
    lines.append("")

    lines.append("## Where the guarantee stops (abstentions)")
    lines.append("")
    if r.abstain_total == 0:
        lines.append(
            "The engine did not abstain anywhere: every dispatched statement was "
            "reasoned about with non-⊤ values."
        )
    else:
        lines.append(
            f"The engine abstained **{r.abstain_total}** time(s) — these are the "
            "operations whose operands were ⊤ (unknown) or which fall outside the "
            "modeled fragment. The safety claim does **not** extend to them "
            "(abstaining loses coverage but never produces a false report):"
        )
        lines.append("")
        lines.append("| Abstain category | Count |")
        lines.append("| --- | --- |")
        for cat, cnt in r.abstain_by_category:
            lines.append(f"| `{cat}` | {cnt} |")
    lines.append("")

    lines.append("## Soundness contract")
    lines.append("")
    lines.append(
        "TensorGuard reports a covered bug **iff** a forced failure is provable "
        "on known operands. The ⇐ direction (every report is a real failure) is "
        "machine-checked in Lean; the ⇒ direction (a failure on known operands is "
        "reported) is the relative-completeness contract above. A report is "
        "therefore never a false positive, and — on the covered, complete-for "
        "fragment — the absence of a report is a positive safety guarantee."
    )
    lines.append("")
    lines.append(f"_Result fingerprint: `{r.fingerprint}`_")

    return "\n".join(lines).rstrip() + "\n"


def explain_safety(result: "SymResult", *, filename: str = "<unknown>") -> str:
    """Convenience: build and render the safety report for ``result``."""
    return render_safety_report(safety_report(result, filename=filename))
