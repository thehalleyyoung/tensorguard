"""Per-operator confidence tagging (Step 6 of 100_STEPS).

Every operator whose shape transfer function TensorGuard knows about is tagged
with one of three machine-readable confidence levels so that downstream users
know *how much to trust* an inference involving that operator:

    complete  - The transfer function is exact: it is both sound (never
                silently accepts an ill-typed program) and complete (it does
                not raise on a well-typed one). These are the shape-preserving
                pointwise families (activations, elementwise unary math,
                elementwise comparisons) whose output shape is, by
                construction, identical to the input.

    sound     - The transfer function is sound (no false "OK") and has an
                exact, well-defined shape rule that the verifier enforces, but
                we do not claim full completeness for every broadcasting /
                zero-dim / keepdim edge case. Structural ops (matmul family,
                reductions, gather/scatter, sort/topk, FFTs, sampling ops with
                a static shape) live here.

    heuristic - The output shape is genuinely data-dependent (e.g. the number
                of rows depends on runtime *values*, not just shapes) or the
                operator's rule is approximated generically. Inferences through
                these ops are best-effort and may be neither sound nor complete.

The default for any operator *not* explicitly classified here - including ops
with no registered transfer function at all - is ``HEURISTIC``. This is the
honest, conservative choice: we never claim more confidence than we can defend.

This module is the single source of truth for the tags. It exposes:

    tag_for(op_name)        -> ConfidenceTag
    rationale_for(op_name)  -> str
    confidence_table()      -> list[dict]   (machine-readable, sorted)
    to_json()               -> str
    annotate_registry()     -> int          (stamp tags onto TransferFunctions)
"""

from __future__ import annotations

import enum
import json
from typing import Dict, List, Tuple


class ConfidenceTag(str, enum.Enum):
    """Confidence level for an operator's shape transfer function."""

    COMPLETE = "complete"
    SOUND = "sound"
    HEURISTIC = "heuristic"


# ── Operator families ──────────────────────────────────────────────────────
# Names are the *base* operator (namespace prefix such as ``torch.`` / ``F.``
# stripped). ``torch.fft.*`` and ``torch.linalg.*`` are handled specially in
# :func:`_classify` because their family is encoded in the namespace.

_ACTIVATIONS = frozenset({
    "relu", "gelu", "silu", "mish", "hardswish", "hardsigmoid", "leaky_relu",
    "elu", "selu", "celu", "prelu", "rrelu", "softplus", "softsign",
    "tanhshrink", "softshrink", "hardshrink", "logsigmoid", "sigmoid", "tanh",
})

_ELEMENTWISE_UNARY = frozenset({
    "abs", "neg", "sign", "ceil", "floor", "round", "exp", "log", "log2",
    "log10", "sqrt", "rsqrt", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "erf", "erfc", "clamp", "clip", "nan_to_num",
})

_COMPARISON = frozenset({
    "eq", "ne", "gt", "ge", "lt", "le", "equal", "isnan", "isinf", "isfinite",
})

_MATMUL_FAMILY = frozenset({
    "matmul", "bmm", "mm", "mv", "outer", "kron", "tensordot", "cross",
})

_REDUCTIONS = frozenset({
    "sum", "mean", "prod", "max", "min", "std", "var", "norm", "logsumexp",
    "any", "all", "amax", "amin",
})

# Output shape is an exact function of shapes (and static integer args).
_STRUCTURAL_EXACT = frozenset({
    "gather", "scatter", "index_select", "sort", "argsort", "topk",
    "stack", "hstack", "vstack", "dstack", "column_stack", "row_stack",
    "squeeze", "unsqueeze", "movedim", "moveaxis", "swapaxes", "swapdims",
    "roll", "rot90", "flip",
    # Sampling ops whose output shape is a static argument / equals the input.
    "bernoulli", "poisson", "cdist",
})

# Output shape depends on runtime *values*, or the rule is approximated.
_DATA_DEPENDENT = frozenset({
    "unique", "multinomial", "einsum",
})


def _base_name(op_name: str) -> str:
    """Strip a leading ``torch.``/``F.``/namespace prefix, keeping the last part."""
    return op_name.rsplit(".", 1)[-1]


def _classify(op_name: str) -> Tuple[ConfidenceTag, str]:
    """Return the (tag, rationale) for a fully-qualified operator name."""
    # Namespace-encoded families first.
    if op_name.startswith("torch.linalg."):
        return (
            ConfidenceTag.HEURISTIC,
            "Linear-algebra decomposition with multiple outputs and "
            "value-dependent / approximated shape handling.",
        )
    if op_name.startswith("torch.fft."):
        return (
            ConfidenceTag.SOUND,
            "FFT family: exact, well-defined output-shape rule (e.g. rfft maps "
            "the last dim n -> n//2 + 1) enforced soundly.",
        )

    base = _base_name(op_name)

    if base in _ACTIVATIONS:
        return (
            ConfidenceTag.COMPLETE,
            "Pointwise activation: output shape is identical to the input, so "
            "the transfer is exact (sound and complete).",
        )
    if base in _ELEMENTWISE_UNARY:
        return (
            ConfidenceTag.COMPLETE,
            "Elementwise unary op: shape-preserving, so the transfer is exact "
            "(sound and complete).",
        )
    if base in _COMPARISON:
        return (
            ConfidenceTag.COMPLETE,
            "Elementwise comparison: shape-preserving boolean output, so the "
            "transfer is exact (sound and complete).",
        )
    if base in _MATMUL_FAMILY:
        return (
            ConfidenceTag.SOUND,
            "Matmul-family op with an exact, well-defined contraction rule that "
            "is enforced soundly (full completeness not claimed for every "
            "broadcasting / zero-dim edge case).",
        )
    if base in _REDUCTIONS:
        return (
            ConfidenceTag.SOUND,
            "Reduction with an exact dim/keepdim shape rule enforced soundly "
            "(full completeness not claimed for every keepdim edge case).",
        )
    if base in _STRUCTURAL_EXACT:
        return (
            ConfidenceTag.SOUND,
            "Structural op whose output shape is an exact function of the input "
            "shapes and static integer arguments; enforced soundly.",
        )
    if base in _DATA_DEPENDENT:
        return (
            ConfidenceTag.HEURISTIC,
            "Output shape depends on runtime values or is approximated "
            "generically; best-effort, neither sound nor complete in general.",
        )

    return (
        ConfidenceTag.HEURISTIC,
        "No explicit confidence classification; defaulting conservatively to "
        "heuristic (best-effort).",
    )


def tag_for(op_name: str) -> ConfidenceTag:
    """Return the :class:`ConfidenceTag` for an operator name.

    Unknown / unregistered operators default to ``HEURISTIC``.
    """
    return _classify(op_name)[0]


def rationale_for(op_name: str) -> str:
    """Return the human-readable justification for an operator's tag."""
    return _classify(op_name)[1]


def _registry_names() -> List[str]:
    """All operator names with a registered transfer function (best-effort)."""
    try:
        from src.graph_compiler import _UNIVERSAL_TRANSFER_REGISTRY
    except Exception:  # pragma: no cover - import guard
        return []
    return list(_UNIVERSAL_TRANSFER_REGISTRY.keys())


def confidence_table() -> List[Dict[str, str]]:
    """Return the full machine-readable confidence table, sorted by name.

    Covers every operator with a registered transfer function. Each row is
    ``{"operator", "confidence", "rationale"}``.
    """
    rows: List[Dict[str, str]] = []
    for name in sorted(set(_registry_names())):
        tag, rationale = _classify(name)
        rows.append({
            "operator": name,
            "confidence": tag.value,
            "rationale": rationale,
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
    """Stamp the confidence tag onto every registered ``TransferFunction``.

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


# Base operator names whose transfer function is only ``heuristic`` and which
# can be spotted by a lightweight source scan (used by ``sound`` mode to refuse
# a confident SAFE). ``torch.linalg.*`` is matched via the namespace.
_HEURISTIC_BASE_OPS = frozenset(_DATA_DEPENDENT)
_HEURISTIC_NAMESPACES = ("linalg",)


def heuristic_ops_in_source(source: str) -> List[str]:
    """Return sorted qualified names of heuristic-tagged ops called in *source*.

    A best-effort static scan of call expressions (e.g. ``torch.unique(...)``,
    ``x.einsum(...)``, ``torch.linalg.svd(...)``). Used so ``sound`` mode can
    abstain rather than emit a confident SAFE when an inference would rely on a
    heuristic transfer function.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found = set()

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
        if base in _HEURISTIC_BASE_OPS:
            found.add(qualified)
        elif len(parents) >= 2 and parents[-2] in _HEURISTIC_NAMESPACES:
            found.add(qualified)
    return sorted(found)


if __name__ == "__main__":  # pragma: no cover
    print(to_json())
