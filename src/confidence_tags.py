"""Shared confidence-tag vocabulary for TensorGuard operator transfers."""

from __future__ import annotations

import enum


class ConfidenceTag(str, enum.Enum):
    """Confidence level for an operator's transfer function."""

    COMPLETE = "complete"
    SOUND = "sound"
    HEURISTIC = "heuristic"
