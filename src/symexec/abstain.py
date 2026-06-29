"""Abstain accounting (Step 59).

The symbolic-execution engine is **sound by abstention**: anywhere a detector
leaves the modeled fragment — an unknown rank, a dimension it cannot represent
as a single theory variable, an ``einops`` pattern with an ellipsis, a theory
backend that is unavailable, an unmodeled construct — it silently returns
``Top`` and emits *no* report.  That silence is what guarantees zero false
positives, but it also makes the engine's *coverage* invisible: there was no way
to ask "how often, and for what reasons, did we decline to reason?".

This module makes that explicit.  A detector, at each abstain site, records a
structured :class:`AbstainReason` (a category, the detector name, a short human
detail, and a source location) into an :class:`AbstainLedger`.  Recording is
**purely diagnostic and side-effect-only**: it never changes whether a bug
fires, so soundness and the zero-regression contract are preserved.  The ledger
then exposes coverage metrics — counts by category and by detector — used by the
test-suite (and, later, by ``--explain`` / telemetry) to measure how much of the
analyzed code the engine actually reasoned about versus abstained on.

The module is torch-free and has no dependency on z3, so importing it never
pulls a heavy backend at engine load time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

__all__ = ["AbstainCategory", "AbstainReason", "AbstainLedger"]


class AbstainCategory(Enum):
    """Why a detector declined to reason about an operation.

    The categories partition the modeled fragment's *boundary*: each value names
    a distinct reason the abstract state was too imprecise (or the construct too
    far outside the model) for a sound forced-failure judgment.
    """

    #: A tensor's rank (number of dims) is unknown (``Top`` rank).
    UNKNOWN_RANK = "unknown_rank"
    #: A tensor's shape tuple is unknown even though the rank may be known.
    UNKNOWN_SHAPE = "unknown_shape"
    #: An individual dimension is unknown / symbolic and unconstrained.
    UNKNOWN_DIM = "unknown_dim"
    #: An affine dim form (sum / scaled var) can't be one theory dim variable.
    UNREPRESENTABLE_AFFINE = "unrepresentable_affine"
    #: A reshape/view target (or other op argument) is not statically known.
    UNKNOWN_TARGET = "unknown_target"
    #: An operand is not a tensor (or the wrong value kind) for this detector.
    NON_TENSOR_OPERAND = "non_tensor_operand"
    #: An ``einops``/``einsum`` pattern uses an ellipsis (rank-polymorphic).
    ELLIPSIS_PATTERN = "ellipsis_pattern"
    #: A pattern/equation string is non-literal or otherwise unparseable.
    NON_LITERAL_PATTERN = "non_literal_pattern"
    #: A required theory backend (e.g. the z3 reshape theory) is unavailable.
    THEORY_UNAVAILABLE = "theory_unavailable"
    #: No path constraints are in effect, so a symbolic dim can't be forced.
    NO_PATH_CONSTRAINTS = "no_path_constraints"
    #: The construct itself is outside the modeled fragment (unmodeled op/call).
    UNMODELED_CONSTRUCT = "unmodeled_construct"
    #: Analysis stopped early because a wall-clock resource budget was exceeded
    #: (only ever recorded when an explicit budget is supplied; the remaining
    #: units are left un-analysed — sound: lost coverage, never a false report).
    RESOURCE_BUDGET = "resource_budget"


@dataclass(frozen=True)
class AbstainReason:
    """One recorded decision to abstain at a specific source location."""

    category: AbstainCategory
    detector: str
    detail: str = ""
    line: int = 0
    col: int = 0
    function: str = "<module>"

    def __str__(self) -> str:  # pragma: no cover - trivial rendering
        loc = f"{self.function}:{self.line}:{self.col}"
        detail = f" ({self.detail})" if self.detail else ""
        return f"[{self.category.value}] {self.detector} @ {loc}{detail}"


@dataclass
class AbstainLedger:
    """Append-only collection of :class:`AbstainReason`s with coverage metrics.

    The ledger is owned by an :class:`~src.symexec.interpreter.Interpreter` and
    accumulated across the whole analysis.  It is *append-only* and never
    consulted by any detector's control flow, so it cannot affect which bugs are
    reported — it only measures where the engine chose ``Top``.
    """

    reasons: List[AbstainReason] = field(default_factory=list)

    def record(self, reason: AbstainReason) -> None:
        """Append ``reason``.  Always returns ``None`` so a detector can write
        ``return self._abstain(...)`` at a site that abstains with ``return``."""
        self.reasons.append(reason)
        return None

    @property
    def total(self) -> int:
        return len(self.reasons)

    def coverage(self) -> Dict[AbstainCategory, int]:
        """Counts keyed by :class:`AbstainCategory` (only non-zero entries)."""
        c: Counter = Counter(r.category for r in self.reasons)
        return dict(c)

    def by_detector(self) -> Dict[str, int]:
        """Counts keyed by detector name (only non-zero entries)."""
        c: Counter = Counter(r.detector for r in self.reasons)
        return dict(c)

    def categories(self) -> set:
        """The distinct categories that occurred at least once."""
        return {r.category for r in self.reasons}

    def summary(self) -> str:
        """A compact, deterministic one-line coverage summary."""
        if not self.reasons:
            return "abstentions: 0"
        parts = [
            f"{cat.value}={cnt}"
            for cat, cnt in sorted(
                self.coverage().items(), key=lambda kv: kv[0].value
            )
        ]
        return f"abstentions: {self.total} (" + ", ".join(parts) + ")"
