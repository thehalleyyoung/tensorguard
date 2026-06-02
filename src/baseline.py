"""Step 72 — baseline & inline suppression for incremental adoption.

Legacy repositories often have a backlog of true findings that a team cannot fix
in one go.  Two mechanisms let them adopt TensorGuard without a wall of failures:

* **Baseline file** (``.tensorguard-baseline.json``): a snapshot of the findings
  that exist *today*.  Future runs suppress anything already in the baseline, so
  only *new* findings can fail the build.  The fingerprint is line-independent —
  a known finding that merely moves up or down the file stays baselined.

* **Inline suppression**: a ``# tensorguard: ignore`` comment on a finding's
  source line suppresses it; ``# tensorguard: ignore[shape,broadcast]`` suppresses
  only the listed rule tags.  ``# tg: ignore`` is accepted as a short alias.

Both operate at the annotation layer produced by :mod:`src.github_action`, so the
GitHub Action, the pre-commit hook and the pytest plugin all benefit at once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.tg_config import rule_tag

BASELINE_FILENAME = ".tensorguard-baseline.json"
BASELINE_VERSION = 1

# `# tensorguard: ignore` / `# tg: ignore` optionally `[rule1, rule2]`
_SUPPRESS_RE = re.compile(
    r"#\s*(?:tensorguard|tg)\s*:\s*ignore(?:\s*\[([^\]]*)\])?",
    re.IGNORECASE,
)


def _relpath(file: str, root: Optional[str]) -> str:
    """Normalised, root-relative POSIX path for stable fingerprints."""
    p = file
    if root:
        try:
            p = os.path.relpath(file, root)
        except ValueError:
            p = file
    return p.replace(os.sep, "/")


def finding_fingerprint(file: str, message: str, root: Optional[str] = None) -> str:
    """Deterministic, line-independent fingerprint for one finding.

    Built from the root-relative path, the finding's rule tag, and the message
    with digits normalised away, so a known finding that shifts lines (or whose
    numeric details change slightly) keeps the same fingerprint.
    """
    rel = _relpath(file, root)
    first = (message or "").splitlines()[0] if message else ""
    tag = rule_tag(first)
    norm = re.sub(r"\d+", "N", first).strip().lower()
    raw = f"{rel}|{tag}|{norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def annotation_fingerprint(ann: Any, root: Optional[str] = None) -> str:
    """Fingerprint for a :class:`src.github_action.Annotation`."""
    return finding_fingerprint(
        getattr(ann, "file", ""), getattr(ann, "message", ""), root
    )


# --------------------------------------------------------------------------- #
# Baseline file I/O
# --------------------------------------------------------------------------- #
def baseline_payload(action_result: Any, root: Optional[str] = None) -> Dict[str, Any]:
    """Serialisable baseline snapshot for an :class:`ActionResult`."""
    entries: Dict[str, Dict[str, str]] = {}
    for ann in getattr(action_result, "annotations", None) or []:
        fp = annotation_fingerprint(ann, root)
        # Keep the first human-readable example per fingerprint.
        entries.setdefault(
            fp,
            {
                "file": _relpath(getattr(ann, "file", ""), root),
                "message": (getattr(ann, "message", "") or "").splitlines()[0],
            },
        )
    return {
        "version": BASELINE_VERSION,
        "generated_at": int(time.time()),
        "fingerprints": entries,
    }


def write_baseline(
    path: str, action_result: Any, root: Optional[str] = None
) -> Dict[str, Any]:
    """Write a baseline JSON file and return the payload."""
    payload = baseline_payload(action_result, root)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return payload


def load_baseline_fingerprints(path: str) -> Set[str]:
    """Load the set of baselined fingerprints from a baseline file."""
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    fps = data.get("fingerprints", {})
    if isinstance(fps, dict):
        return set(fps.keys())
    return set(fps)


def find_baseline_file(start: str) -> Optional[str]:
    """Search *start* and its parents for a ``.tensorguard-baseline.json``."""
    cur = os.path.abspath(start)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        candidate = os.path.join(cur, BASELINE_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


# --------------------------------------------------------------------------- #
# Inline suppression
# --------------------------------------------------------------------------- #
def parse_inline_suppressions(source: str) -> Dict[int, Optional[Set[str]]]:
    """Map 1-based line number -> suppressed rule tags (None = all rules).

    A line carrying ``# tensorguard: ignore`` suppresses every finding reported
    on it; ``# tensorguard: ignore[shape, broadcast]`` suppresses only those tags.
    """
    out: Dict[int, Optional[Set[str]]] = {}
    for i, line in enumerate((source or "").splitlines(), start=1):
        m = _SUPPRESS_RE.search(line)
        if not m:
            continue
        rules_group = m.group(1)
        if rules_group is None or not rules_group.strip():
            out[i] = None  # suppress all rules on this line
        else:
            tags = {
                t.strip().lower()
                for t in rules_group.split(",")
                if t.strip()
            }
            existing = out.get(i)
            if existing is None and i in out:
                continue  # already suppress-all
            out[i] = (existing or set()) | tags
    return out


def is_suppressed_inline(
    ann: Any, suppressions: Dict[int, Optional[Set[str]]]
) -> bool:
    """True if an annotation is silenced by an inline suppression on its line."""
    line = getattr(ann, "line", None)
    if line not in suppressions:
        return False
    rules = suppressions[line]
    if rules is None:
        return True  # suppress-all on this line
    return rule_tag(getattr(ann, "message", "") or "") in rules


def filter_inline(
    annotations: Iterable[Any], source: str
) -> Tuple[List[Any], List[Any]]:
    """Split annotations into (kept, suppressed) by inline comments in *source*."""
    suppressions = parse_inline_suppressions(source)
    if not suppressions:
        return list(annotations), []
    kept, suppressed = [], []
    for a in annotations:
        (suppressed if is_suppressed_inline(a, suppressions) else kept).append(a)
    return kept, suppressed


# --------------------------------------------------------------------------- #
# Baseline application to an ActionResult
# --------------------------------------------------------------------------- #
def apply_baseline(
    action_result: Any,
    baseline_fingerprints: Set[str],
    root: Optional[str] = None,
):
    """Return a new ActionResult with baselined findings suppressed.

    ``failed`` is recomputed: a build that originally failed only fails now if
    *new* (non-baselined) findings remain.  ``fail_on=never`` (original
    ``failed`` already False) is preserved.
    """
    from src.github_action import ActionResult

    kept: List[Any] = []
    for ann in getattr(action_result, "annotations", None) or []:
        if annotation_fingerprint(ann, root) in baseline_fingerprints:
            continue
        kept.append(ann)

    files_with_issues = len({getattr(a, "file", "") for a in kept})
    total = len(kept)
    failed = bool(action_result.failed) and total > 0
    return ActionResult(
        annotations=kept,
        files_checked=action_result.files_checked,
        files_with_issues=files_with_issues,
        total_issues=total,
        failed=failed,
        results_by_file=getattr(action_result, "results_by_file", []),
    )
