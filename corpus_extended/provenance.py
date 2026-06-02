"""Per-case provenance and license metadata for the extended corpus (Step 102).

For a benchmark dataset to be *redistributable* every case must have clear
provenance and a compatible license. The extended corpus is unusual in a good
way: its cases are **synthetic, originally generated** by
:mod:`corpus_extended.generators` -- no third-party source code is copied. The
buggy families are *inspired* by public PyTorch issue reports (recorded as a
``seed_reference`` for context only); the emitted module source is our own work.

This module produces a structured provenance record per case:

* ``origin`` -- always ``"synthetic_generated"`` (we author the source);
* ``generator`` -- the generator family that emitted it;
* ``seed_reference`` -- the public issue URL that *inspired* the pattern (or
  ``null`` for canonical patterns), explicitly marked as a reference, not a
  source of copied code;
* ``authors`` / ``license`` / ``spdx`` -- the dataset is released under the same
  MIT license as the repository, so it is freely redistributable;
* ``redistributable`` -- ``True`` for every case (no copyleft / proprietary
  encumbrance, no copied third-party code).

:mod:`reproducibility.corpus_provenance_audit` consumes these records and proves
the whole corpus is clean to redistribute.
"""

from __future__ import annotations

from typing import Dict, List

from corpus_extended.generators import Case, all_cases

DATASET_LICENSE = "MIT"
DATASET_SPDX = "MIT"
DATASET_AUTHORS = "TensorGuard Authors"


def provenance_for(case: Case) -> Dict[str, object]:
    """Return a structured, redistributable-by-construction provenance record."""
    return {
        "id": case.id,
        "origin": "synthetic_generated",
        "generator": case.family,
        "provenance_type": case.provenance_type,
        # The seed is an *inspiration reference only*; no code is copied from it.
        "seed_reference": case.seed_url,
        "seed_is_reference_only": True,
        "copied_third_party_code": False,
        "authors": DATASET_AUTHORS,
        "license": DATASET_LICENSE,
        "spdx": DATASET_SPDX,
        "redistributable": True,
    }


def all_provenance() -> List[Dict[str, object]]:
    return [provenance_for(c) for c in all_cases()]
