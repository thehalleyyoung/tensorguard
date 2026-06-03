"""Per-operator confidence tags derived from the proof-footprint manifest.

``src.proof_footprint`` is the source of truth for operator trust metadata: each
manifest row records the confidence tag, confidence-specific rationale, and the
proof/evidence footprint that justifies it.  This module intentionally exposes
the historical ``operator-confidence`` API as a projection of that manifest so
the JSON table cannot drift from the audited footprint.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from src.confidence_tags import ConfidenceTag


def tag_for(op_name: str) -> ConfidenceTag:
    """Return the :class:`ConfidenceTag` for an operator name.

    Unknown / unregistered operators default to ``HEURISTIC`` through the
    proof-footprint confidence policy.
    """

    from src.proof_footprint import confidence_for

    return confidence_for(op_name)[0]


def rationale_for(op_name: str) -> str:
    """Return the human-readable justification for an operator's tag."""

    from src.proof_footprint import confidence_for

    return confidence_for(op_name)[1]


def confidence_table() -> List[Dict[str, str]]:
    """Return the full machine-readable confidence table, sorted by name.

    Covers every operator with a registered transfer function. Each row is
    ``{"operator", "confidence", "rationale"}``, projected from the generated
    proof-footprint rows.
    """

    from src.proof_footprint import proof_footprint_table

    rows: List[Dict[str, str]] = []
    for row in proof_footprint_table():
        rows.append({
            "operator": str(row["operator"]),
            "confidence": str(row["confidence"]),
            "rationale": str(row["confidence_rationale"]),
        })
    return rows


def to_json(indent: int = 2) -> str:
    """Serialize the confidence table (with a summary header) to JSON."""

    table = confidence_table()
    summary: Dict[str, int] = {t.value: 0 for t in ConfidenceTag}
    for row in table:
        summary[row["confidence"]] += 1
    payload = {
        "schema": "tensorguard.operator_confidence/v1",
        "default_tag": ConfidenceTag.HEURISTIC.value,
        "summary": summary,
        "total": len(table),
        "operators": table,
    }
    return json.dumps(payload, indent=indent, sort_keys=False)


def annotate_registry() -> int:
    """Stamp the manifest-derived confidence tag onto registered transfers.

    Returns the number of transfer functions annotated. Idempotent.
    """

    try:
        from src.graph_compiler import _UNIVERSAL_TRANSFER_REGISTRY
    except Exception:  # pragma: no cover - import guard
        return 0
    count = 0
    for name, tf in _UNIVERSAL_TRANSFER_REGISTRY.items():
        tf.confidence = tag_for(name).value
        count += 1
    return count


def _base_name(op_name: str) -> str:
    return op_name.rsplit(".", 1)[-1]


@lru_cache(maxsize=1)
def _heuristic_base_ops() -> frozenset[str]:
    """Base operator names whose manifest confidence is heuristic."""

    from src.proof_footprint import proof_footprint_table

    return frozenset(
        _base_name(str(row["operator"]))
        for row in proof_footprint_table()
        if row["confidence"] == ConfidenceTag.HEURISTIC.value
    )


def heuristic_ops_in_source(source: str) -> List[str]:
    """Return sorted qualified names of heuristic-tagged ops called in *source*.

    A best-effort static scan of call expressions (e.g. ``torch.unique(...)``,
    ``x.einsum(...)``, ``torch.linalg.lstsq(...)``). Used so ``sound`` mode can
    abstain rather than emit a confident SAFE when an inference would rely on a
    heuristic transfer function.
    """

    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found = set()
    heuristic_bases = _heuristic_base_ops()

    def _qualified(node: ast.AST) -> str:
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        func = n.func
        if not isinstance(func, ast.Attribute):
            continue
        base = func.attr
        qualified = _qualified(func)
        parents = qualified.split(".")
        if base in heuristic_bases:
            found.add(qualified)
        elif len(parents) >= 2 and parents[-2] == "linalg" and tag_for(qualified) is ConfidenceTag.HEURISTIC:
            found.add(qualified)
    return sorted(found)


if __name__ == "__main__":  # pragma: no cover
    print(to_json())
