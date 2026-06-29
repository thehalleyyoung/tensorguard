"""Step 58 — trace minimization (shrink a counterexample to its minimal slice).

A forced-failure report is most actionable when it names the **smallest** set of
path conditions that still makes the fault unavoidable.  The broadcast detector,
for example, fires under the accumulated path facts ``a != b ∧ a != 1 ∧ b != 1``
— but some of those facts may be incidental.  This module shrinks such a
"trace" (the conjunction of path conditions that witness the failure) to a
**1-minimal** slice: a subset from which no single condition can be removed
without the failure ceasing to be forced.

The core is the classic delta-debugging reduction, parameterised by a
detector-supplied ``holds`` predicate (``does the failure still hold under this
subset?``).  It is deterministic and order-preserving, and makes at most
``O(n²)`` predicate calls over the (tiny) fact set.

Soundness note: minimization only affects *diagnostics* — the report fires on
the full path facts regardless.  The shrunk slice is a sound explanation because
``holds`` is re-checked on it (the fault is still forced under the slice alone).
"""

from __future__ import annotations

from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")

__all__ = ["minimize"]


def minimize(elements: Sequence[T], holds: Callable[[List[T]], bool]) -> List[T]:
    """Return a 1-minimal sublist of ``elements`` for which ``holds`` is still
    ``True``.

    Preconditions/semantics:

    * ``holds(list(elements))`` is assumed ``True`` (the full trace witnesses the
      fault).  If it is not, the full list is returned unchanged (nothing can be
      justified as removable).
    * If ``holds([])`` is ``True`` the fault is *unconditional* (independent of
      the trace) and the empty slice is returned.
    * The result is **1-minimal**: removing any single remaining element makes
      ``holds`` ``False``.  Order is preserved (deterministic output).
    """
    elements = list(elements)
    if not holds(elements):
        return elements
    if holds([]):
        return []
    kept = list(elements)
    i = 0
    while i < len(kept):
        candidate = kept[:i] + kept[i + 1 :]
        if holds(candidate):
            kept = candidate  # element i was incidental — drop it (don't advance)
        else:
            i += 1  # element i is necessary — keep it and move on
    return kept
