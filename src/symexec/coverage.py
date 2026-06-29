"""Statement-coverage metering for the symbolic-execution engine (Step 77).

The engine is *sound by abstention*: any construct outside the modeled fragment
evaluates to ``Top`` and emits no report.  That silence is what guarantees zero
false positives — but, on its own, it makes the engine's analytical *reach*
invisible.  Two files can both yield "no bugs": one because the engine fully
interpreted every statement and proved them safe, the other because the engine
went ``Top`` on the very first line and never reasoned about anything.  The
:class:`AbstainLedger` counts *where* a detector chose ``Top``; this module
answers the complementary question — *how much of the program did the engine
actually interpret with a non-``Top`` value?*

A :class:`CoverageMeter` is owned by an
:class:`~src.symexec.interpreter.Interpreter` and accumulated across the whole
analysis.  Exactly like the abstain ledger it is **append-only and never
consulted by any detector's control flow**, so it cannot affect which bugs are
reported — it is a pure, after-the-fact diagnostic.  It is deliberately *not*
folded into the proof fingerprint, so every existing reproducibility golden
value stays byte-identical.

A statement is recorded once per distinct AST node (keyed by ``id``); when the
same statement is interpreted on several paths or fixpoint passes the *best*
outcome is kept (modeled / non-``Top`` are monotone "did it ever happen"
signals), so loops and re-analysis never inflate or deflate the metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class StmtCoverage:
    """The coverage outcome for a single source statement.

    ``modeled`` is True when the interpreter had a transfer function for the
    statement's node type (an unmodeled type is a no-op that leaves the state
    untouched — a coverage gap).  ``is_binding`` marks value-producing
    statements (assignments) where ``Top`` vs non-``Top`` is meaningful.
    ``non_top`` is True when the engine interpreted the statement with concrete
    abstract information: for a binding statement, at least one target was bound
    to a non-``Top`` value; for any other modeled statement, that it was
    structurally interpreted (control flow / imports / returns / …)."""

    line: int
    kind: str
    modeled: bool
    is_binding: bool
    non_top: bool


@dataclass
class CoverageMeter:
    """Append-only per-statement coverage accumulator with derived metrics."""

    #: id(ast.stmt) -> StmtCoverage (latest, best-merged outcome).
    _records: Dict[int, StmtCoverage] = field(default_factory=dict)

    def record(
        self,
        stmt,
        modeled: bool,
        is_binding: bool,
        non_top: bool,
    ) -> None:
        """Record (or monotonically upgrade) the outcome for ``stmt``.

        Always returns ``None`` and never inspects engine state, so it is safe to
        call unconditionally from the statement-dispatch choke point."""
        key = id(stmt)
        line = getattr(stmt, "lineno", 0)
        kind = type(stmt).__name__
        prev = self._records.get(key)
        if prev is not None:
            modeled = modeled or prev.modeled
            is_binding = is_binding or prev.is_binding
            non_top = non_top or prev.non_top
            line = prev.line or line
        self._records[key] = StmtCoverage(
            line=line,
            kind=kind,
            modeled=modeled,
            is_binding=is_binding,
            non_top=non_top,
        )

    # -- raw counts ------------------------------------------------------
    @property
    def total(self) -> int:
        """Distinct source statements the interpreter dispatched on."""
        return len(self._records)

    @property
    def modeled(self) -> int:
        return sum(1 for r in self._records.values() if r.modeled)

    @property
    def non_top(self) -> int:
        return sum(1 for r in self._records.values() if r.non_top)

    @property
    def bindings(self) -> int:
        return sum(1 for r in self._records.values() if r.is_binding)

    @property
    def non_top_bindings(self) -> int:
        return sum(
            1 for r in self._records.values() if r.is_binding and r.non_top
        )

    # -- derived fractions ----------------------------------------------
    @property
    def coverage(self) -> float:
        """Headline metric: fraction of executed statements interpreted
        non-``Top`` (``1.0`` when there were no statements — vacuously full)."""
        return 1.0 if self.total == 0 else self.non_top / self.total

    @property
    def modeled_coverage(self) -> float:
        """Fraction of executed statements whose node type the engine models."""
        return 1.0 if self.total == 0 else self.modeled / self.total

    @property
    def value_coverage(self) -> float:
        """Of the value-binding statements, the fraction bound to a non-``Top``
        value — the sharpest signal of how much real information the abstract
        interpreter recovered (``1.0`` when there were no bindings)."""
        b = self.bindings
        return 1.0 if b == 0 else self.non_top_bindings / b

    # -- gap reporting ---------------------------------------------------
    def gaps(self) -> List[StmtCoverage]:
        """Every statement the engine did *not* interpret non-``Top`` (unmodeled
        node types and bindings that collapsed to ``Top``), in source order —
        the actionable list of where to extend the engine's reach."""
        out = [r for r in self._records.values() if not r.non_top]
        out.sort(key=lambda r: (r.line, r.kind))
        return out

    def unmodeled_kinds(self) -> Dict[str, int]:
        """Counts of statement node types the engine has no transfer function
        for, keyed by ``ast`` node name (only non-zero entries)."""
        counts: Dict[str, int] = {}
        for r in self._records.values():
            if not r.modeled:
                counts[r.kind] = counts.get(r.kind, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        """A stable, JSON-ready snapshot of the coverage profile."""
        return {
            "total_statements": self.total,
            "modeled_statements": self.modeled,
            "non_top_statements": self.non_top,
            "binding_statements": self.bindings,
            "non_top_bindings": self.non_top_bindings,
            "coverage": round(self.coverage, 6),
            "modeled_coverage": round(self.modeled_coverage, 6),
            "value_coverage": round(self.value_coverage, 6),
            "unmodeled_kinds": self.unmodeled_kinds(),
        }

    def summary(self) -> str:
        """A compact, deterministic one-line coverage summary."""
        return (
            f"coverage: {self.non_top}/{self.total} statements non-Top "
            f"({self.coverage:.0%}); value {self.non_top_bindings}/{self.bindings} "
            f"({self.value_coverage:.0%})"
        )
