"""
verdict_taxonomy.py — refined verdict set for TensorGuard v5.

Splits the legacy ``Refuted`` verdict into three semantically distinct
sub-verdicts to address the NeurIPS reviewer's concern about overloading:

* REFUTED_PROOF      — sound counterexample; Theorem 1 applies.
* CONTRACT_VIOLATION — caller's precondition (config.X / **kwargs) is
                       too loose; sound under the stated caller contract
                       assumption; not necessarily a library bug.
* LIBRARY_WARN       — refutation on library code that relies on dynamic
                       dispatch / runtime metaprogramming; NOT covered by
                       Theorem 1; treated as a conservative warning.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Iterable

# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

_CONTRACT_PATTERNS: list[str] = [
    r"self\.config\.",
    r"getattr\s*\(\s*self\.config",
    r"\*\*kwargs",
    r"nn\.ModuleList\s*\(.*for.*in.*range\s*\(\s*self\.config",
]

_CONTRACT_RE = re.compile("|".join(_CONTRACT_PATTERNS))


class Verdict(IntEnum):
    VERIFIED = 0
    REFUTED_PROOF = 1
    CONTRACT_VIOLATION = 2
    LIBRARY_WARN = 3
    ABSTAIN = 4
    NA = 5

    def __str__(self) -> str:
        _display = {
            Verdict.VERIFIED: "Verified",
            Verdict.REFUTED_PROOF: "Refuted-Proof",
            Verdict.CONTRACT_VIOLATION: "Contract-Violation",
            Verdict.LIBRARY_WARN: "Library-Warn",
            Verdict.ABSTAIN: "Abstain",
            Verdict.NA: "N/A",
        }
        return _display[self]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_proof_verdict(v: Verdict) -> bool:
    """Return True only for REFUTED_PROOF (used by Theorem 1 statements)."""
    return v is Verdict.REFUTED_PROOF


def summarize(verdicts: Iterable[Verdict]) -> dict[str, int]:
    """Return a count dict keyed by canonical string name."""
    counts: dict[str, int] = {}
    for v in verdicts:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# classify_refutation
# ---------------------------------------------------------------------------

def classify_refutation(
    item: dict,
    source: str | None = None,
) -> Verdict:
    """Classify a Refuted result into one of the three sub-verdicts.

    Parameters
    ----------
    item:
        A per-block or per-bug result dict (as found in
        ``v5_benchmark_results.json`` ``per_input`` arrays).
    source:
        Optional source code of the block.  When provided, pattern
        matching is used to distinguish CONTRACT_VIOLATION from
        LIBRARY_WARN.

    Returns
    -------
    Verdict
        One of REFUTED_PROOF, CONTRACT_VIOLATION, or LIBRARY_WARN.
    """
    # Bug-corpus items (is_buggy_gt=True or corpus=="bug") are always
    # true counterexamples.
    if item.get("is_buggy_gt") or item.get("corpus") == "bug":
        return Verdict.REFUTED_PROOF

    # If explicit source text is given, scan for contract-violation patterns.
    if source is not None:
        if _CONTRACT_RE.search(source):
            return Verdict.CONTRACT_VIOLATION

    # Fallback to LIBRARY_WARN (dynamic dispatch / unmodelled fragment).
    return Verdict.LIBRARY_WARN
