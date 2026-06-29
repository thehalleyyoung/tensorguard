"""Certificate replay / independent re-derivation (SYMEXEC_100_STEPS Step 95).

A :class:`~src.symexec.certificate.BugCertificate` is *replayable*: this module
re-derives the verdict **without re-running the analysis engine**.  It looks the
violated runtime precondition up in the fixed vocabulary
(:data:`~src.symexec.certificate.PRECONDITIONS`) and re-evaluates it on the
certificate's witness operands.  A genuine forced-failure report carries operands
on which the precondition is *false* — so replay confirms the report is sound.

This is the dual of the Lean ``refute`` lemma made executable: the certificate
claims "precondition P is violated on witness w"; replay checks ``not P(w)``.

Three outcomes (honest about what was actually checked):

* ``verified``   — operands present and the precondition is violated on them
                   (``not holds(operands)``): the report is independently
                   re-derived as a true forced failure.
* ``refuted``    — operands present but the precondition *holds* on them: the
                   certificate is internally inconsistent (it would correspond to
                   a false positive).  Tampering with a witness is caught here.
* ``unchecked``  — claim-only certificate (no recoverable witness): the named
                   precondition stands, but nothing numeric was verified.

Torch-free; depends only on the pure precondition vocabulary, never on the
interpreter, so it is a genuinely independent checker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .certificate import (
    BugCertificate,
    PRECONDITIONS,
    loads_certificates,
    precondition_holds,
)

__all__ = [
    "ReplayResult",
    "replay",
    "replay_all",
    "replay_text",
    "all_verified",
]


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of independently re-deriving one certificate."""

    kind: str
    line: int
    col: int
    predicate: str
    status: str  # "verified" | "refuted" | "unchecked"
    detail: str

    @property
    def ok(self) -> bool:
        """True unless the certificate was positively *refuted* (inconsistent)."""
        return self.status != "refuted"


def _render_ops(operands: Sequence[int]) -> str:
    return "(" + ", ".join(str(o) for o in operands) + ")"


def replay(cert: BugCertificate) -> ReplayResult:
    """Re-derive a single certificate's verdict from its witness."""
    base = dict(kind=cert.kind, line=cert.line, col=cert.col,
                predicate=cert.predicate)

    if cert.predicate not in PRECONDITIONS:
        return ReplayResult(
            **base, status="unchecked",
            detail=f"unknown precondition {cert.predicate!r}; cannot re-derive",
        )

    if cert.operands is None:
        return ReplayResult(
            **base, status="unchecked",
            detail=f"claim-only: {cert.claim} (no numeric witness recovered)",
        )

    try:
        holds = precondition_holds(cert.predicate, cert.operands)
    except (KeyError, ValueError) as exc:
        return ReplayResult(
            **base, status="unchecked",
            detail=f"could not evaluate precondition: {exc}",
        )

    ops = _render_ops(cert.operands)
    if not holds:
        return ReplayResult(
            **base, status="verified",
            detail=(f"precondition '{cert.predicate}' is violated on witness "
                    f"{ops}: {cert.claim} — forced failure re-derived"),
        )
    return ReplayResult(
        **base, status="refuted",
        detail=(f"precondition '{cert.predicate}' HOLDS on witness {ops}: the "
                f"certificate does not describe a forced failure"),
    )


def replay_all(certs: Sequence[BugCertificate]) -> List[ReplayResult]:
    """Replay every certificate in a batch."""
    return [replay(c) for c in certs]


def replay_text(text: str) -> List[ReplayResult]:
    """Replay certificates straight from their serialized JSON form.

    This is the fully-independent path: nothing but the certificate file and the
    precondition vocabulary participate in the re-derivation."""
    return replay_all(loads_certificates(text))


def all_verified(results: Sequence[ReplayResult]) -> bool:
    """True iff no certificate was refuted (verified or unchecked only)."""
    return all(r.ok for r in results)
