"""Determinism & proof footprint (Step 60).

Two runs of the symbolic executor over the same source must produce *byte for
byte* the same findings — the report list is already canonically sorted
(Step 20) and every suppression/refinement is a Z3-*proved* fact, so the result
is a deterministic function of the input.  This module makes that determinism
**checkable**: it distils an analysis result into a compact, stable
``ProofFootprint`` — a SHA-256 *digest* over the canonical bug list plus the
abstain-coverage profile — that can be recorded as a golden value, compared
across runs/machines, or attached to a report as a reproducibility receipt.

Design choices for stability:

* The digest is taken over each bug's **identity** fields — kind, source
  location, function, message, severity — and *not* over diagnostic-only fields
  (``confidence``, ``fix_suggestion``, ``evidence``).  Those carry concrete
  counterexamples / minimal traces that depend on solver availability; excluding
  them keeps the footprint identical whether or not z3 is installed, so the
  footprint certifies *which forced failures were found*, not how they were
  explained.
* Records are sorted before hashing, so the digest is independent of internal
  traversal / fixpoint-pass order (it is a function of the *set* of findings).
* A ``FOOTPRINT_VERSION`` tag is folded into the payload so the format can evolve
  without silently colliding with old golden values.

The module is torch-free and depends only on the standard library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

__all__ = [
    "FOOTPRINT_VERSION",
    "ProofFootprint",
    "bug_record",
    "bug_fingerprint",
    "footprint",
]

FOOTPRINT_VERSION = 1


def bug_record(bug) -> dict:
    """The stable identity of a :class:`~src.symexec.bugs.SymBug` for hashing.

    Diagnostic-only fields (confidence, fix_suggestion, evidence) are excluded so
    the footprint is solver-availability-independent."""
    kind = getattr(bug.kind, "value", str(bug.kind))
    return {
        "kind": kind,
        "line": int(getattr(bug, "line", 0)),
        "col": int(getattr(bug, "col", 0)),
        "function": getattr(bug, "function", "") or "",
        "message": getattr(bug, "message", ""),
        "severity": getattr(bug, "severity", "error"),
    }


def _records(bugs: Sequence) -> List[dict]:
    recs = [bug_record(b) for b in bugs]
    # Sort defensively so the digest is a function of the *set* of findings,
    # independent of the order they were produced/collected in.
    recs.sort(key=lambda r: (r["line"], r["col"], r["kind"], r["function"], r["message"]))
    return recs


def _abstain_coverage(abstentions) -> Dict[str, int]:
    """Coverage profile ``{category_value: count}`` from a ledger or ``None``."""
    if abstentions is None:
        return {}
    cov = getattr(abstentions, "coverage", None)
    if callable(cov):
        return {cat.value: cnt for cat, cnt in cov().items()}
    return {}


def _canonical_text(records: List[dict], coverage: Dict[str, int]) -> str:
    payload = {
        "version": FOOTPRINT_VERSION,
        "bugs": records,
        "abstain_coverage": coverage,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def bug_fingerprint(bugs: Sequence) -> str:
    """SHA-256 hex digest over just the canonical bug list (no abstain profile)."""
    text = _canonical_text(_records(bugs), {})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProofFootprint:
    """A compact, deterministic receipt for one analysis result."""

    version: int
    digest: str
    bug_count: int
    abstain_count: int
    abstain_coverage: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "digest": self.digest,
            "bug_count": self.bug_count,
            "abstain_count": self.abstain_count,
            "abstain_coverage": dict(sorted(self.abstain_coverage.items())),
        }

    @property
    def short(self) -> str:
        """The first 12 hex chars of the digest — handy for logs / golden tags."""
        return self.digest[:12]


def footprint(bugs: Sequence, abstentions=None) -> ProofFootprint:
    """Build a :class:`ProofFootprint` from a bug list and an optional ledger.

    The ``digest`` folds in the abstain-coverage profile so the footprint
    certifies not only the forced failures found but the *coverage shape* of the
    run (how many sites, by category, the engine soundly declined to reason
    about).  Identical inputs ⇒ identical footprint, on any machine."""
    records = _records(bugs)
    coverage = _abstain_coverage(abstentions)
    text = _canonical_text(records, coverage)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    abstain_count = (
        getattr(abstentions, "total", 0) if abstentions is not None else 0
    )
    return ProofFootprint(
        version=FOOTPRINT_VERSION,
        digest=digest,
        bug_count=len(records),
        abstain_count=int(abstain_count),
        abstain_coverage=coverage,
    )
