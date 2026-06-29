"""``--explain`` derivations (Step 65).

Each report already accumulates a *provenance chain* (Step 7) — the line-tagged
source→…→sink derivation of the offending value — plus, where available, a
concrete counterexample (Step 54), an algebraic certificate, and the 1-minimal
failing conditions (Step 58), all packed into ``SymBug.evidence``.  By default a
report shows only its headline message; this module renders the *full
derivation chain on demand* (the engine's ``--explain`` view), so an owner can
see exactly **why** the engine believes the failure is forced and how to
reproduce it.

The renderer is presentation-only and never re-runs analysis: it parses the
already-computed ``evidence`` string into labelled sections (derivation,
counterexample, certificate, minimal conditions) and lays them out together with
the location, calibrated confidence, and mechanical fix suggestion.  It is
torch-free and pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

__all__ = ["Explanation", "explain_bug", "explain_bugs"]

# ``SymBug.evidence`` is assembled by joining independent fragments with these
# separators (``_witness2``/cert via " | "; ``_with_minimal_trace`` via "; ").
_FRAGMENT_SEPARATORS = (" | ", "; ")
# The provenance chain joins its steps with this arrow (``_witness``/``_witness2``).
_PROV_ARROW = " → "


def _split_fragments(evidence: str) -> List[str]:
    """Split a composed evidence string into its independent fragments."""
    parts = [evidence]
    for sep in _FRAGMENT_SEPARATORS:
        nxt: List[str] = []
        for p in parts:
            nxt.extend(p.split(sep))
        parts = nxt
    return [p.strip() for p in parts if p.strip()]


@dataclass
class Explanation:
    """A structured, renderable derivation for one report."""

    kind: str
    message: str
    line: int
    col: int
    function: str
    confidence: float
    filename: Optional[str] = None
    derivation: List[str] = field(default_factory=list)
    counterexample: Optional[str] = None
    certificate: Optional[str] = None
    minimal_conditions: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    fix_suggestion: Optional[str] = None

    def render(self) -> str:
        """A multi-line human-readable explanation."""
        loc = self.filename or "<unknown>"
        out: List[str] = []
        out.append(f"[{self.kind}] {self.message}")
        out.append(f"  at {loc}:{self.line}:{self.col} in {self.function or '<module>'}")
        out.append(f"  confidence: {self.confidence:.2f}")
        if self.derivation:
            out.append("  derivation:")
            for step in self.derivation:
                out.append(f"    {step}")
        if self.certificate:
            out.append(f"  certificate: {self.certificate}")
        if self.counterexample:
            out.append(f"  counterexample: {self.counterexample}")
        if self.minimal_conditions:
            out.append(f"  minimal failing conditions: {self.minimal_conditions}")
        for note in self.notes:
            out.append(f"  note: {note}")
        if self.fix_suggestion:
            out.append(f"  fix: {self.fix_suggestion}")
        return "\n".join(out)


def _classify(fragment: str, exp: Explanation) -> None:
    """Route one evidence fragment into the right structured field."""
    low = fragment.lower()
    if low.startswith("certified counterexample"):
        exp.certificate = _strip_prefix(fragment, "certified counterexample:")
    elif low.startswith("concrete counterexample"):
        exp.counterexample = _strip_prefix(fragment, "concrete counterexample:")
    elif low.startswith("minimal failing conditions"):
        exp.minimal_conditions = _strip_prefix(fragment, "minimal failing conditions:")
    elif _PROV_ARROW in fragment:
        exp.derivation = [s.strip() for s in fragment.split(_PROV_ARROW) if s.strip()]
    else:
        exp.notes.append(fragment)


def _strip_prefix(fragment: str, prefix: str) -> str:
    if fragment[: len(prefix)].lower() == prefix.lower():
        return fragment[len(prefix):].strip()
    return fragment.strip()


def explain_bug(bug, filename: Optional[str] = None) -> Explanation:
    """Build a structured :class:`Explanation` from a ``SymBug``.

    Parses the report's accumulated ``evidence`` into a derivation chain and the
    counterexample / certificate / minimal-conditions sections (presentation
    only — no analysis is re-run)."""
    kind = getattr(getattr(bug, "kind", None), "name", str(getattr(bug, "kind", "")))
    exp = Explanation(
        kind=kind,
        message=getattr(bug, "message", ""),
        line=int(getattr(bug, "line", 0)),
        col=int(getattr(bug, "col", 0)),
        function=getattr(bug, "function", "") or "",
        confidence=float(getattr(bug, "confidence", 0.0)),
        filename=filename,
        fix_suggestion=getattr(bug, "fix_suggestion", None),
    )
    evidence = getattr(bug, "evidence", None)
    if evidence:
        for frag in _split_fragments(evidence):
            _classify(frag, exp)
    return exp


def explain_bugs(bugs: Sequence, filename: Optional[str] = None) -> str:
    """Render the full ``--explain`` view for a sequence of reports."""
    if not bugs:
        return "no bugs found."
    blocks = [explain_bug(b, filename=filename).render() for b in bugs]
    return "\n\n".join(blocks)
