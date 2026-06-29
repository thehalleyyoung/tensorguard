"""Confidence calibration (Step 63).

Every report already carries a ``confidence`` — but historically each detector
hard-coded a magic number.  This module turns that into a *principled*,
evidence-driven calibration applied at the single ``_emit`` choke point.

The model is deliberately **monotone and corroborative**: a detector's declared
value is treated as a *prior* — the structural certainty of the check itself —
and independent *corroborating evidence* gathered during analysis can only
*raise* the confidence, never silently lower it.  This is sound by construction:
every emitted bug is already a Z3-proved / concretely-forced failure (confidence
is a presentation signal, not a reporting gate), so calibration never changes
*which* bugs report — only how strongly each is ranked.

Corroborating signals (each an independent line of evidence that the report is
real):

* ``unconditional`` — the fault is forced on *every* path (no path constraint
  was needed to force it); the simplest, most robust kind of forced failure.
* ``smt_forced`` — the failure was discharged against the path constraints via
  Z3 and survived the feasibility gate (the path is satisfiable and the failing
  condition is forced under it).
* ``witness`` — a concrete counterexample (a lifted Z3 model) was attached: the
  report is replayable.
* ``certificate`` — an independent algebraic certificate (e.g. a concretization
  oracle witness for a matmul) corroborates the symbolic proof.
* ``minimal_trace`` — the failing path facts were delta-minimised to a 1-minimal
  slice, pinpointing the exact responsible conditions.

Confidence is capped strictly below ``1.0``: absolute certainty is reserved for
"the file does not even parse / run" (handled outside this module).

The module is torch-free and pure (no z3, no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConfidenceSignals", "calibrate", "MAX_CONFIDENCE"]

#: Confidence ceiling for an analysis-derived report.  ``1.0`` is reserved for
#: syntactic "won't run" defects surfaced before interpretation.
MAX_CONFIDENCE = 0.99

# Per-signal corroboration weights (additive on top of the detector prior).
_W_UNCONDITIONAL = 0.03
_W_SMT_FORCED = 0.03
_W_WITNESS = 0.04
_W_CERTIFICATE = 0.04
_W_MINIMAL_TRACE = 0.01


@dataclass(frozen=True)
class ConfidenceSignals:
    """The corroborating evidence available for one report at emit time."""

    unconditional: bool = False
    smt_forced: bool = False
    witness: bool = False
    certificate: bool = False
    minimal_trace: bool = False

    @classmethod
    def from_evidence(
        cls,
        evidence,
        *,
        has_path_constraints: bool,
        smt_checked: bool,
    ) -> "ConfidenceSignals":
        """Derive the signals from a report's ``evidence`` string and the path
        context at the emit site.

        ``has_path_constraints`` is whether any symbolic path facts / failing
        conditions were in effect; ``smt_checked`` is whether the emit site ran
        the Z3 feasibility gate over them."""
        ev = evidence or ""
        return cls(
            unconditional=not has_path_constraints,
            smt_forced=has_path_constraints and smt_checked,
            witness="concrete counterexample" in ev,
            certificate="certified counterexample" in ev,
            minimal_trace="minimal failing conditions" in ev,
        )

    def boost(self) -> float:
        """Total additive corroboration from the present signals."""
        return (
            (_W_UNCONDITIONAL if self.unconditional else 0.0)
            + (_W_SMT_FORCED if self.smt_forced else 0.0)
            + (_W_WITNESS if self.witness else 0.0)
            + (_W_CERTIFICATE if self.certificate else 0.0)
            + (_W_MINIMAL_TRACE if self.minimal_trace else 0.0)
        )


def calibrate(prior: float, signals: ConfidenceSignals) -> float:
    """Calibrated confidence = ``prior`` raised by corroborating evidence.

    Monotone in evidence (more corroboration ⇒ higher), never below the prior
    (the detector's structural certainty is a floor), clamped to
    ``[prior, MAX_CONFIDENCE]``.  With no signals it returns the prior unchanged,
    so a report with no extra evidence keeps exactly its declared confidence."""
    raised = prior + signals.boost()
    if raised < prior:  # defensive; boosts are non-negative
        raised = prior
    return min(MAX_CONFIDENCE, raised)
