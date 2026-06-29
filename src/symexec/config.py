"""Configuration & soundness modes (Step 86).

The symbolic-execution engine ships a single, principled default behaviour, but
different consumers sit at different points on the precision/recall curve:

* a release-gate that must *never* cry wolf wants **maximum precision** — only
  report a fault it can positively prove is reachable;
* the everyday default wants the engine's proven, zero-false-positive findings
  exactly as they are today;
* an exploratory "what might be wrong here?" pass is willing to trade some
  precision for recall and see lower-confidence *suspicions*.

:class:`SymConfig` exposes that as three named modes plus orthogonal knobs:

``balanced`` (the default)
    Byte-identical to the engine's historic behaviour.  Every emitted report is
    a Z3-proved / concretely-forced failure; nothing is suppressed and no
    best-effort suspicions are surfaced.  ``min_confidence=0.0``,
    ``require_feasibility=False``, ``enable_heuristics=False``.

``sound``
    Maximum precision — a strict **subset** of ``balanced``'s reports.  A report
    that depends on a symbolic path is kept only when the solver can *positively
    confirm* that path (conjoined with the failing condition) is satisfiable;
    when the solver is missing or returns ``unknown`` the report is dropped
    rather than kept.  A confidence floor (``min_confidence=0.85``) additionally
    discards any low-prior finding.  Unconditional, fully-concrete faults (no
    path constraints) are reported in every mode, so this never suppresses the
    engine's bread-and-butter shape proofs.

``heuristic``
    Maximum recall — a **superset** of ``balanced``'s reports.  Detectors are
    allowed to emit clearly-labelled, *lower-confidence* "suspected" findings at
    sites where ``balanced`` would soundly abstain.  These may be false
    positives by construction; the mode exists for triage, not gating.

The knobs are independent of the mode label and can be overridden:

* ``min_confidence`` — drop any report whose *calibrated* confidence is below
  this floor (a pure presentation/triage gate applied at the single ``_emit``
  choke point).
* ``require_feasibility`` — when a report is path-conditioned, keep it only if
  the path is provably satisfiable (suppress on ``unknown`` / missing solver).
* ``enable_heuristics`` — let detectors surface best-effort suspicions.
* ``budget_ms`` — default per-file wall-clock budget (``None`` = unbounded).

This module is torch-free and pure (no z3, no I/O); it only carries policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

__all__ = ["SymConfig", "MODES", "DEFAULT_CONFIG"]

#: The three named soundness modes, ordered from highest precision to highest
#: recall.  Report sets nest: ``sound`` ⊆ ``balanced`` ⊆ ``heuristic``.
MODES = ("sound", "balanced", "heuristic")

# Confidence floor used by the ``sound`` preset.  Below this a finding's
# structural prior is too weak to clear a release gate.
_SOUND_MIN_CONFIDENCE = 0.85


@dataclass(frozen=True)
class SymConfig:
    """An immutable analysis policy.

    The default instance is ``balanced`` and is byte-identical to the engine's
    historic behaviour, so constructing an :class:`~src.symexec.interpreter.\
Interpreter` without a config changes nothing.
    """

    mode: str = "balanced"
    min_confidence: float = 0.0
    require_feasibility: bool = False
    enable_heuristics: bool = False
    budget_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"unknown mode {self.mode!r}; expected one of {MODES}"
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence!r}"
            )
        if self.budget_ms is not None and self.budget_ms <= 0:
            raise ValueError(
                f"budget_ms must be positive or None, got {self.budget_ms!r}"
            )

    # -- presets -----------------------------------------------------------

    @classmethod
    def balanced(cls, **overrides) -> "SymConfig":
        """The default policy (current engine behaviour)."""
        return cls(
            mode="balanced",
            min_confidence=0.0,
            require_feasibility=False,
            enable_heuristics=False,
        ).with_overrides(**overrides)

    @classmethod
    def sound(cls, **overrides) -> "SymConfig":
        """Maximum precision: a strict subset of ``balanced``'s reports."""
        return cls(
            mode="sound",
            min_confidence=_SOUND_MIN_CONFIDENCE,
            require_feasibility=True,
            enable_heuristics=False,
        ).with_overrides(**overrides)

    @classmethod
    def heuristic(cls, **overrides) -> "SymConfig":
        """Maximum recall: a superset of ``balanced``'s reports."""
        return cls(
            mode="heuristic",
            min_confidence=0.0,
            require_feasibility=False,
            enable_heuristics=True,
        ).with_overrides(**overrides)

    @classmethod
    def for_mode(cls, mode: str, **overrides) -> "SymConfig":
        """Return the preset for ``mode`` (with optional knob overrides).

        ``mode`` is one of :data:`MODES`; raises ``ValueError`` otherwise."""
        if mode == "sound":
            return cls.sound(**overrides)
        if mode == "balanced":
            return cls.balanced(**overrides)
        if mode == "heuristic":
            return cls.heuristic(**overrides)
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    def with_overrides(self, **overrides) -> "SymConfig":
        """A copy with the given fields replaced (``mode`` is preserved unless
        explicitly overridden).  Re-validates via ``__post_init__``."""
        if not overrides:
            return self
        return replace(self, **overrides)

    # -- reporting policy --------------------------------------------------

    def allows_confidence(self, confidence: float) -> bool:
        """Whether a report at this calibrated ``confidence`` clears the floor."""
        return confidence >= self.min_confidence


#: The engine's default policy.  Identical to the historic behaviour.
DEFAULT_CONFIG = SymConfig()
