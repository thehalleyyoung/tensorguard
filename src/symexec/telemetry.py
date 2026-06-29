"""Telemetry & confidence feedback (Step 87).

Every report the engine emits carries a *calibrated* confidence (Step 63).  That
number is principled but, until it is checked against real outcomes, it is only
a *claim*.  This module closes the loop: it lets a consumer **opt in** to record
anonymized calibration data — one record per report — and then *measure* whether
the engine's confidences are well-calibrated and *suggest* per-detector prior
adjustments from labelled outcomes.

Three properties are load-bearing:

* **Opt-in.** Recording is off by default (:class:`TelemetrySink` starts
  ``enabled=False`` and every method is a no-op until enabled).  Nothing is
  collected unless a caller asks for it.
* **Anonymous.** A :class:`CalibrationRecord` carries *no source content* — no
  file path, no message text, no line/col — only the bug *kind*, the calibrated
  confidence, and the boolean evidence signals recoverable from the report's
  evidence string.  There is nothing to leak.
* **Advisory & side-effect-free.** Telemetry is *never* consulted by any
  detector and *never* mutates analysis.  The prior-adjustment suggestions are
  recommendations for a human to apply to :mod:`src.symexec.confidence`; they are
  *not* auto-applied, because changing a detector prior would change which bugs
  rank where (and could perturb reproducibility goldens).  So enabling telemetry
  cannot change which bugs report or any proof fingerprint.

The module is torch-free, has no z3 dependency, and performs **no I/O** itself:
serialization helpers return/accept strings so the caller owns any file writes
(keeping the opt-in boundary explicit).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "CalibrationRecord",
    "TelemetrySink",
    "ReliabilityBin",
    "CalibrationReport",
    "PriorSuggestion",
    "reliability_table",
    "expected_calibration_error",
    "brier_score",
    "precision_by_kind",
    "suggest_prior_adjustments",
    "calibration_report",
    "records_to_jsonl",
    "records_from_jsonl",
]

# Evidence-string markers (see src/symexec/confidence.py).  These are the
# corroboration signals recoverable from a finished report's ``evidence`` text.
_WITNESS_MARK = "concrete counterexample"
_CERTIFICATE_MARK = "certified counterexample"
_MINIMAL_TRACE_MARK = "minimal failing conditions"


@dataclass(frozen=True)
class CalibrationRecord:
    """One anonymized telemetry record for a single emitted report.

    Carries no source content: only the bug ``kind`` (its enum *value* string),
    the calibrated ``confidence``, the recoverable evidence flags, and an
    optional ground-truth ``outcome`` (``True`` = a confirmed real bug, ``False``
    = a false positive, ``None`` = not yet labelled)."""

    kind: str
    confidence: float
    has_witness: bool = False
    has_certificate: bool = False
    has_minimal_trace: bool = False
    outcome: Optional[bool] = None

    @classmethod
    def from_bug(cls, bug, outcome: Optional[bool] = None) -> "CalibrationRecord":
        """Build a record from a :class:`~src.symexec.bugs.SymBug` (or anything
        with ``kind``/``confidence``/``evidence``).  ``kind`` is taken as the
        enum value when present, else ``str(kind)``."""
        kind = getattr(bug.kind, "value", None) or str(bug.kind)
        ev = bug.evidence or ""
        return cls(
            kind=kind,
            confidence=float(bug.confidence),
            has_witness=_WITNESS_MARK in ev,
            has_certificate=_CERTIFICATE_MARK in ev,
            has_minimal_trace=_MINIMAL_TRACE_MARK in ev,
            outcome=outcome,
        )

    def with_outcome(self, outcome: Optional[bool]) -> "CalibrationRecord":
        """A copy with the ground-truth ``outcome`` set (for feedback labelling)."""
        return CalibrationRecord(
            kind=self.kind,
            confidence=self.confidence,
            has_witness=self.has_witness,
            has_certificate=self.has_certificate,
            has_minimal_trace=self.has_minimal_trace,
            outcome=outcome,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "has_witness": self.has_witness,
            "has_certificate": self.has_certificate,
            "has_minimal_trace": self.has_minimal_trace,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationRecord":
        return cls(
            kind=str(d["kind"]),
            confidence=float(d["confidence"]),
            has_witness=bool(d.get("has_witness", False)),
            has_certificate=bool(d.get("has_certificate", False)),
            has_minimal_trace=bool(d.get("has_minimal_trace", False)),
            outcome=d.get("outcome", None),
        )


@dataclass
class TelemetrySink:
    """An opt-in, in-memory collector of :class:`CalibrationRecord`s.

    Disabled by default: every ingestion method is a no-op until ``enabled`` is
    set ``True`` (or the sink is constructed with ``enabled=True``).  This is the
    privacy boundary — no data is gathered unless the consumer explicitly asks.
    """

    enabled: bool = False
    records: List[CalibrationRecord] = field(default_factory=list)

    def enable(self) -> "TelemetrySink":
        self.enabled = True
        return self

    def disable(self) -> "TelemetrySink":
        self.enabled = False
        return self

    def record_bug(self, bug, outcome: Optional[bool] = None) -> None:
        """Ingest one report.  No-op when disabled."""
        if not self.enabled:
            return
        self.records.append(CalibrationRecord.from_bug(bug, outcome=outcome))

    def record_result(self, result, outcome: Optional[bool] = None) -> int:
        """Ingest every bug in a ``SymResult``.  Returns the number ingested
        (``0`` when disabled)."""
        if not self.enabled:
            return 0
        n = 0
        for bug in result.bugs:
            self.records.append(CalibrationRecord.from_bug(bug, outcome=outcome))
            n += 1
        return n

    def clear(self) -> None:
        self.records.clear()

    def report(self, bins: int = 10) -> "CalibrationReport":
        """The full calibration report over the records gathered so far."""
        return calibration_report(self.records, bins=bins)


# --------------------------------------------------------------------------- #
# Calibration metrics (pure functions over a list of records)                 #
# --------------------------------------------------------------------------- #

def _labelled(records: List[CalibrationRecord]) -> List[CalibrationRecord]:
    """Records that carry a ground-truth outcome (the only ones that can score
    calibration)."""
    return [r for r in records if r.outcome is not None]


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bucket of a reliability diagram.

    ``lo``/``hi`` are the bucket bounds; ``count`` the number of *labelled*
    records in it; ``mean_confidence`` the average predicted confidence; and
    ``empirical_precision`` the fraction of those records that were confirmed
    real bugs (``outcome=True``).  A well-calibrated engine has
    ``mean_confidence ≈ empirical_precision`` in every populated bucket."""

    lo: float
    hi: float
    count: int
    mean_confidence: float
    empirical_precision: float


def reliability_table(
    records: List[CalibrationRecord], bins: int = 10
) -> List[ReliabilityBin]:
    """A reliability diagram: ``bins`` equal-width confidence buckets over
    ``[0, 1]``, each summarising the *labelled* records that fall in it.

    Only populated buckets are returned, in ascending order.  A record with
    confidence exactly ``1.0`` falls in the last bucket."""
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    labelled = _labelled(records)
    width = 1.0 / bins
    buckets: Dict[int, List[CalibrationRecord]] = defaultdict(list)
    for r in labelled:
        idx = min(int(r.confidence / width), bins - 1)
        buckets[idx].append(r)
    out: List[ReliabilityBin] = []
    for idx in sorted(buckets):
        group = buckets[idx]
        n = len(group)
        mean_conf = sum(r.confidence for r in group) / n
        precision = sum(1 for r in group if r.outcome) / n
        out.append(
            ReliabilityBin(
                lo=idx * width,
                hi=(idx + 1) * width,
                count=n,
                mean_confidence=mean_conf,
                empirical_precision=precision,
            )
        )
    return out


def expected_calibration_error(
    records: List[CalibrationRecord], bins: int = 10
) -> Optional[float]:
    """Expected Calibration Error: the count-weighted mean absolute gap between
    predicted confidence and empirical precision across buckets.

    ``0.0`` is perfect calibration; ``None`` when there are no labelled records.
    """
    table = reliability_table(records, bins=bins)
    total = sum(b.count for b in table)
    if total == 0:
        return None
    return sum(
        b.count * abs(b.mean_confidence - b.empirical_precision) for b in table
    ) / total


def brier_score(records: List[CalibrationRecord]) -> Optional[float]:
    """Mean squared error between predicted confidence and the binary outcome
    over labelled records (lower is better; ``None`` when none are labelled)."""
    labelled = _labelled(records)
    if not labelled:
        return None
    return sum(
        (r.confidence - (1.0 if r.outcome else 0.0)) ** 2 for r in labelled
    ) / len(labelled)


def precision_by_kind(
    records: List[CalibrationRecord],
) -> Dict[str, "KindStats"]:
    """Per-kind empirical precision and mean confidence over labelled records."""
    groups: Dict[str, List[CalibrationRecord]] = defaultdict(list)
    for r in _labelled(records):
        groups[r.kind].append(r)
    out: Dict[str, KindStats] = {}
    for kind in sorted(groups):
        g = groups[kind]
        n = len(g)
        out[kind] = KindStats(
            kind=kind,
            count=n,
            mean_confidence=sum(r.confidence for r in g) / n,
            empirical_precision=sum(1 for r in g if r.outcome) / n,
        )
    return out


@dataclass(frozen=True)
class KindStats:
    kind: str
    count: int
    mean_confidence: float
    empirical_precision: float


@dataclass(frozen=True)
class PriorSuggestion:
    """An *advisory* per-kind confidence-prior adjustment derived from labelled
    outcomes.  ``delta`` is ``empirical_precision - mean_confidence`` (positive =
    the detector is under-confident and could be raised; negative = over-confident
    and should be lowered).  Purely a recommendation — never auto-applied."""

    kind: str
    count: int
    mean_confidence: float
    empirical_precision: float
    delta: float


def suggest_prior_adjustments(
    records: List[CalibrationRecord], *, min_count: int = 5
) -> List[PriorSuggestion]:
    """Suggest a confidence-prior delta per bug kind from labelled outcomes.

    Kinds with fewer than ``min_count`` labelled records are skipped (too little
    evidence to act on).  Returned in descending order of absolute delta so the
    most mis-calibrated detectors surface first.  Advisory only."""
    out: List[PriorSuggestion] = []
    for kind, st in precision_by_kind(records).items():
        if st.count < min_count:
            continue
        out.append(
            PriorSuggestion(
                kind=kind,
                count=st.count,
                mean_confidence=st.mean_confidence,
                empirical_precision=st.empirical_precision,
                delta=st.empirical_precision - st.mean_confidence,
            )
        )
    out.sort(key=lambda s: abs(s.delta), reverse=True)
    return out


@dataclass(frozen=True)
class CalibrationReport:
    """A bundled, deterministic calibration summary over a record set."""

    total: int
    labelled: int
    reliability: List[ReliabilityBin]
    ece: Optional[float]
    brier: Optional[float]
    by_kind: Dict[str, KindStats]
    suggestions: List[PriorSuggestion]

    def summary(self) -> str:
        """A compact one-line, deterministic summary."""
        ece = "n/a" if self.ece is None else f"{self.ece:.3f}"
        brier = "n/a" if self.brier is None else f"{self.brier:.3f}"
        return (
            f"telemetry: {self.total} records ({self.labelled} labelled), "
            f"ECE={ece}, Brier={brier}, kinds={len(self.by_kind)}"
        )


def calibration_report(
    records: List[CalibrationRecord], *, bins: int = 10, min_count: int = 5
) -> CalibrationReport:
    """Compute the full :class:`CalibrationReport` over ``records``."""
    return CalibrationReport(
        total=len(records),
        labelled=len(_labelled(records)),
        reliability=reliability_table(records, bins=bins),
        ece=expected_calibration_error(records, bins=bins),
        brier=brier_score(records),
        by_kind=precision_by_kind(records),
        suggestions=suggest_prior_adjustments(records, min_count=min_count),
    )


# --------------------------------------------------------------------------- #
# Serialization (string in / string out — the caller owns any file I/O)       #
# --------------------------------------------------------------------------- #

def records_to_jsonl(records: List[CalibrationRecord]) -> str:
    """Serialize records as newline-delimited JSON (one record per line).

    Deterministic (sorted keys), so a fixed record set round-trips byte-stably.
    Returns the text; the caller decides whether/where to persist it."""
    return "\n".join(
        json.dumps(r.to_dict(), sort_keys=True) for r in records
    )


def records_from_jsonl(text: str) -> List[CalibrationRecord]:
    """Parse newline-delimited JSON produced by :func:`records_to_jsonl`.

    Blank lines are ignored, so a trailing newline is tolerated."""
    out: List[CalibrationRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(CalibrationRecord.from_dict(json.loads(line)))
    return out
