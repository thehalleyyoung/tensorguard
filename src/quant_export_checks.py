"""Quantization & export safety checks for PyTorch ``nn.Module`` source.

This module adds two static analyzers that catch a class of bugs the core
shape/device/dtype verifier does not target, because they concern *deployment*
rather than the forward dataflow of a float model:

1. **Quantization hazards** (``analyze_quantization``).  Quantization-aware
   training (QAT) and post-training quantization (PTQ) in PyTorch impose
   placement rules that, when violated, fail *at runtime* with opaque
   ``NotImplementedError``/backend-dispatch errors rather than at model
   construction.  The classic example is performing tensor arithmetic
   (``a + b``) on quantized tensors instead of routing it through
   :class:`torch.ao.nn.quantized.FloatFunctional`; real PyTorch raises

       NotImplementedError: Could not run 'aten::add.out' with arguments
       from the 'QuantizedCPU' backend

   We also flag asymmetric quant/dequant boundaries (a ``QuantStub`` with no
   matching ``DeQuantStub`` leaks quantized tensors into float-only code, and
   vice-versa).

2. **Export-safety hazards** (``analyze_export_safety``).  ``torch.export`` and
   ONNX export trace the module under symbolic shapes; constructs that branch
   or iterate on *data-dependent* tensor values, or that pull a Python scalar
   out of a tensor (``.item()``), make the export trace fail (e.g.
   ``GuardOnDataDependentSymNode``).  These are exactly the constructs that lie
   *outside* TensorGuard's verifiable fragment, so we reuse the fragment
   analyzer (:func:`src.verifiable_fragment.analyze_source`) as the single
   source of truth and surface only the subset of categories that genuinely
   break ``torch.export``.

Both analyzers are *instance-free*: they take module **source** and never
import or execute the user's model.  Each reported hazard carries a
:class:`Confidence` so downstream consumers can distinguish constructs that are
unconditionally unsupported by export (``SOUND``) from quantization heuristics
that may have false positives (``HEURISTIC``).
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.verifiable_fragment import UnsupportedCategory, analyze_source


class Confidence(Enum):
    """Whether a reported hazard is unconditionally true or a heuristic."""

    SOUND = "sound"
    HEURISTIC = "heuristic"


class QuantHazardKind(Enum):
    QUANT_ARITH_WITHOUT_FLOATFUNCTIONAL = "quant_arith_without_floatfunctional"
    MISSING_DEQUANTSTUB = "missing_dequantstub"
    MISSING_QUANTSTUB = "missing_quantstub"


class ExportHazardKind(Enum):
    DATA_DEPENDENT_CONTROL_FLOW = "export_data_dependent_control_flow"
    DATA_DEPENDENT_ITERATION = "export_data_dependent_iteration"
    TENSOR_TO_SCALAR = "export_tensor_to_scalar"
    DYNAMIC_ASSERTION = "export_dynamic_assertion"


# torch.export traces under symbolic shapes; these fragment categories provoke a
# data-dependent guard failure (or a Python-int coercion error) during tracing.
# Other out-of-fragment categories (e.g. in-place mutation, custom autograd) do
# not necessarily break export, so they are intentionally excluded here.
_EXPORT_BREAKING = {
    UnsupportedCategory.DATA_DEPENDENT_CONTROL_FLOW: (
        ExportHazardKind.DATA_DEPENDENT_CONTROL_FLOW
    ),
    UnsupportedCategory.DATA_DEPENDENT_ITERATION: (
        ExportHazardKind.DATA_DEPENDENT_ITERATION
    ),
    UnsupportedCategory.TENSOR_TO_SCALAR: ExportHazardKind.TENSOR_TO_SCALAR,
    UnsupportedCategory.DYNAMIC_ASSERTION: ExportHazardKind.DYNAMIC_ASSERTION,
}


@dataclass(frozen=True)
class QuantHazard:
    kind: QuantHazardKind
    confidence: Confidence
    location: str
    message: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence.value,
            "location": self.location,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class ExportHazard:
    kind: ExportHazardKind
    confidence: Confidence
    location: str
    message: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence.value,
            "location": self.location,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Quantization analyzer
# ---------------------------------------------------------------------------

_QUANTSTUB_NAMES = {"QuantStub", "QuantStub()"}
_DEQUANTSTUB_NAMES = {"DeQuantStub", "DeQuantStub()"}


class _QuantVisitor(ast.NodeVisitor):
    """Collect quantization-relevant facts from a module class body."""

    def __init__(self) -> None:
        self.has_quantstub = False
        self.has_dequantstub = False
        self.uses_floatfunctional = False
        # (location, op_symbol) for arithmetic BinOps inside any forward method
        self.arith_ops: List[tuple] = []
        self._in_forward = False

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node.func)
        if name is not None:
            base = name.split(".")[-1]
            if base == "QuantStub":
                self.has_quantstub = True
            elif base == "DeQuantStub":
                self.has_dequantstub = True
            elif base == "FloatFunctional":
                self.uses_floatfunctional = True
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == "forward":
            prev = self._in_forward
            self._in_forward = True
            self.generic_visit(node)
            self._in_forward = prev
        else:
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self._in_forward and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            # Only flag arithmetic where both operands are *names/attributes*
            # (i.e. tensors), not literal scalars like ``x * 2``.
            if _is_tensorish(node.left) and _is_tensorish(node.right):
                sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}[type(node.op)]
                self.arith_ops.append((f"forward:line {node.lineno}", sym))
        self.generic_visit(node)


def _callee_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = []
        cur: ast.AST = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _is_tensorish(node: ast.AST) -> bool:
    """True if the operand is a name/attribute/call (potential tensor),
    not a numeric/constant literal."""
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return True
    return False


def analyze_quantization(source: str) -> List[QuantHazard]:
    """Statically flag quantization placement hazards in *source*.

    Returns an empty list for source that does not use quantization stubs at
    all (no false positives on ordinary float models).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    v = _QuantVisitor()
    v.visit(tree)

    # Only treat the module as quantization-relevant if it references stubs.
    is_quant_module = v.has_quantstub or v.has_dequantstub
    hazards: List[QuantHazard] = []

    if is_quant_module and not v.uses_floatfunctional:
        for loc, sym in v.arith_ops:
            hazards.append(
                QuantHazard(
                    kind=QuantHazardKind.QUANT_ARITH_WITHOUT_FLOATFUNCTIONAL,
                    confidence=Confidence.HEURISTIC,
                    location=loc,
                    message=(
                        f"tensor arithmetic '{sym}' inside a quantized module "
                        "without FloatFunctional; quantized tensors have no "
                        "'aten::add/mul' kernel and raise at runtime"
                    ),
                    recommendation=(
                        "route the op through "
                        "torch.ao.nn.quantized.FloatFunctional()"
                    ),
                )
            )

    if v.has_quantstub and not v.has_dequantstub:
        hazards.append(
            QuantHazard(
                kind=QuantHazardKind.MISSING_DEQUANTSTUB,
                confidence=Confidence.HEURISTIC,
                location="module",
                message=(
                    "QuantStub present but no DeQuantStub; quantized tensors "
                    "leak into float-only downstream code"
                ),
                recommendation="add self.dequant = DeQuantStub() at the output boundary",
            )
        )
    if v.has_dequantstub and not v.has_quantstub:
        hazards.append(
            QuantHazard(
                kind=QuantHazardKind.MISSING_QUANTSTUB,
                confidence=Confidence.HEURISTIC,
                location="module",
                message=(
                    "DeQuantStub present but no QuantStub; the input is never "
                    "quantized at the entry boundary"
                ),
                recommendation="add self.quant = QuantStub() at the input boundary",
            )
        )

    return hazards


# ---------------------------------------------------------------------------
# Export-safety analyzer
# ---------------------------------------------------------------------------


def analyze_export_safety(source: str) -> List[ExportHazard]:
    """Flag constructs that make ``torch.export`` / ONNX export tracing fail.

    Built on the verifiable-fragment analyzer (single source of truth); only the
    subset of out-of-fragment categories that provoke a data-dependent guard
    failure during symbolic tracing is reported, each at SOUND confidence.
    """
    hazards: List[ExportHazard] = []
    for construct in analyze_source(source):
        mapped = _EXPORT_BREAKING.get(construct.category)
        if mapped is None:
            continue
        hazards.append(
            ExportHazard(
                kind=mapped,
                confidence=Confidence.SOUND,
                location=construct.location,
                message=(
                    f"{construct.description}; breaks torch.export symbolic "
                    "tracing (data-dependent guard)"
                ),
            )
        )
    return hazards


def summarize(source: str) -> dict:
    """Combined quant + export hazard summary for *source*."""
    q = analyze_quantization(source)
    e = analyze_export_safety(source)
    return {
        "quant_hazards": [h.to_dict() for h in q],
        "export_hazards": [h.to_dict() for h in e],
        "n_quant_hazards": len(q),
        "n_export_hazards": len(e),
        "export_safe": len(e) == 0,
        "quant_safe": len(q) == 0,
    }
