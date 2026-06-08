"""Bridge the data-plane engine into TensorGuard's :class:`~src.api.Bug` model.

The data-plane engine (:mod:`src.dataplane.dataplane`) emits
:class:`DataPlaneObligation` objects across seven bug axes.  This module lowers
each *rejected* obligation into a first-class TensorGuard :class:`~src.api.Bug`
so data-plane findings flow through the same reporting surface as shape bugs, and
provides :func:`analyze_data_plane` / :func:`analyze_data_plane_file` as the
additive public entry points.

Only ``rejected`` obligations become bugs; ``admitted``/``unknown`` obligations
are dropped (precision-first, exactly as the underlying scanners behave).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .dataplane import DataPlaneObligation, DataPlaneReport, analyze_all, analyze_tree

# Imported lazily-safe: src.api imports are cheap and have no cycle with dataplane.
from ..api import Bug, BugCategory, SourceLocation

# data-plane axis -> the public BugCategory it maps to.
_AXIS_TO_CATEGORY = {
    "refinement": BugCategory.DATA_VALUE_DOMAIN,
    "non_interference": BugCategory.DATA_LEAKAGE,
    "temporal": BugCategory.DATA_TEMPORAL_LEAKAGE,
    "group": BugCategory.DATA_GROUP_LEAKAGE,
    "join": BugCategory.DATA_JOIN_CARDINALITY,
    "sampling": BugCategory.DATA_SAMPLING_DETERMINISM,
    "split": BugCategory.DATA_SPLIT_CONTRACT,
}

_AXIS_HEADLINE = {
    "refinement": "loss applied to an input outside its required value domain",
    "non_interference": "featuriser fitted before the train/test split (data leakage)",
    "temporal": "feature reads a future row (temporal lookahead leakage)",
    "group": "group straddles the train/test split (group leakage)",
    "join": "merge fans out rows before the split (join-cardinality leakage)",
    "sampling": "non-deterministic data sampling / DataLoader",
    "split": "overlapping or ill-formed train/test split",
}


def _site_to_location(site: str) -> SourceLocation:
    """Parse a ``"file:line"`` engine site into a :class:`SourceLocation`."""
    file, _, line = site.rpartition(":")
    if not file:
        file = site
    try:
        lineno = int(line)
    except ValueError:
        file, lineno = site, 0
    return SourceLocation(file=file, line=lineno, column=0)


def obligation_to_bug(ob: DataPlaneObligation) -> Bug:
    """Lower one rejected data-plane obligation into a TensorGuard :class:`Bug`."""
    category = _AXIS_TO_CATEGORY.get(ob.axis, BugCategory.TYPE_ERROR)
    headline = _AXIS_HEADLINE.get(ob.axis, ob.axis)
    message = f"{headline}: {ob.detail}" if ob.detail else headline
    return Bug(
        category=category,
        message=message,
        location=_site_to_location(ob.site),
        severity="error",
        confidence=1.0,
        guard_evidence=str(ob.witness) if ob.witness else None,
    )


def report_to_bugs(report: DataPlaneReport) -> List[Bug]:
    """Lower every *rejected* obligation in ``report`` into TensorGuard bugs."""
    return [obligation_to_bug(o) for o in report.violations]


def analyze_data_plane(source: str, filename: str = "<string>") -> List[Bug]:
    """Analyze ``source`` for data-plane bugs across all seven axes.

    Returns TensorGuard :class:`Bug` objects (one per rejected obligation), so the
    result composes with the model-plane :func:`src.api.analyze` bug list.
    """
    return report_to_bugs(analyze_all(source, filename))


def analyze_data_plane_file(path: str | Path) -> List[Bug]:
    """Read ``path`` and analyze it for data-plane bugs."""
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return analyze_data_plane(source, str(p))


def analyze_data_plane_tree(root: str | Path) -> List[Bug]:
    """Sweep every ``*.py`` file under ``root`` for data-plane bugs."""
    return report_to_bugs(analyze_tree(root, include_families=True))


__all__ = [
    "obligation_to_bug",
    "report_to_bugs",
    "analyze_data_plane",
    "analyze_data_plane_file",
    "analyze_data_plane_tree",
]
