"""
Human-readable explanation generation for CEGAR verification results.

Transforms CEGAR refinement traces and verification outcomes into
developer-friendly explanations of why a model is safe or unsafe,
without requiring knowledge of SMT solving.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    VerificationResult,
    SafetyViolation,
    CounterexampleTrace,
    LayerDef,
    LayerKind,
    OpKind,
    extract_computation_graph,
    verify_model,
    Device,
    Phase,
)
from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    CEGARVerdict,
    ShapePredicate,
    PredicateKind,
    IterationRecord,
    run_shape_cegar,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer description helpers
# ═══════════════════════════════════════════════════════════════════════════════

_LAYER_TYPE_NAMES = {
    LayerKind.LINEAR: "Linear",
    LayerKind.CONV2D: "Conv2d",
    LayerKind.CONV1D: "Conv1d",
    LayerKind.BATCHNORM1D: "BatchNorm1d",
    LayerKind.BATCHNORM2D: "BatchNorm2d",
    LayerKind.LAYERNORM: "LayerNorm",
    LayerKind.DROPOUT: "Dropout",
    LayerKind.RELU: "ReLU",
    LayerKind.SOFTMAX: "Softmax",
    LayerKind.EMBEDDING: "Embedding",
    LayerKind.LSTM: "LSTM",
    LayerKind.GRU: "GRU",
    LayerKind.MULTIHEAD_ATTENTION: "MultiheadAttention",
    LayerKind.MAXPOOL2D: "MaxPool2d",
    LayerKind.AVGPOOL2D: "AvgPool2d",
    LayerKind.ADAPTIVE_AVGPOOL2D: "AdaptiveAvgPool2d",
    LayerKind.FLATTEN: "Flatten",
    LayerKind.SEQUENTIAL: "Sequential",
}


def _layer_description(layer: LayerDef) -> str:
    """One-line human-readable layer description."""
    name = _LAYER_TYPE_NAMES.get(layer.kind, layer.kind.name)
    if layer.kind == LayerKind.LINEAR:
        inf = layer.in_features or "?"
        outf = layer.out_features or "?"
        return f"{name} {inf}→{outf}"
    if layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
        inc = layer.in_channels or "?"
        outc = layer.out_channels or "?"
        ks = layer.kernel_size or "?"
        return f"{name} {inc}→{outc}, kernel={ks}"
    if layer.kind == LayerKind.EMBEDDING:
        ne = layer.num_embeddings or "?"
        ed = layer.embedding_dim or "?"
        return f"{name} {ne}×{ed}"
    if layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D, LayerKind.LAYERNORM):
        nf = layer.num_features or layer.params.get("normalized_shape", "?")
        return f"{name}({nf})"
    return name


def _predicate_provenance_label(pred: ShapePredicate) -> str:
    """Short label describing where a predicate came from."""
    prov = pred.provenance
    if "api_stub" in prov or "api" in prov:
        return "from API stub"
    if "guard" in prov:
        return "from guard harvesting"
    if "interpolation" in prov or "craig" in prov:
        return "from Craig interpolation"
    if "template" in prov:
        return "from template matching"
    return "from CEGAR discovery"


# ═══════════════════════════════════════════════════════════════════════════════
# Explanation dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationExplanation:
    """A structured, human-readable explanation of a verification result."""
    model_name: str
    verdict: str  # "SAFE", "UNSAFE", "UNKNOWN", "TIMEOUT"
    layer_explanations: List[str] = field(default_factory=list)
    refinement_trace: List[str] = field(default_factory=list)
    counterexample_path: List[str] = field(default_factory=list)
    summary: str = ""
    iterations: int = 0
    num_predicates: int = 0
    num_counterexamples: int = 0

    def render(self) -> str:
        """Render the full explanation as a formatted string."""
        lines: List[str] = []

        # Header
        if self.verdict == "SAFE":
            lines.append(f"Safety Explanation for {self.model_name}:")
        elif self.verdict == "UNSAFE":
            lines.append(f"Safety Violation Report for {self.model_name}:")
        else:
            lines.append(f"Verification Report for {self.model_name} ({self.verdict}):")

        # Layer-by-layer analysis
        if self.layer_explanations:
            for le in self.layer_explanations:
                lines.append(le)

        # Counterexample path (for unsafe models)
        if self.counterexample_path:
            lines.append("")
            lines.append("Counterexample Path:")
            for cp in self.counterexample_path:
                lines.append(cp)

        # CEGAR refinement trace
        if self.refinement_trace:
            lines.append("")
            lines.append("CEGAR Refinement Trace:")
            for rt in self.refinement_trace:
                lines.append(rt)

        # Result summary (skip if already in refinement trace)
        if self.summary and not self.refinement_trace:
            lines.append(f"  {self.summary}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "model_name": self.model_name,
            "verdict": self.verdict,
            "layer_explanations": self.layer_explanations,
            "refinement_trace": self.refinement_trace,
            "counterexample_path": self.counterexample_path,
            "summary": self.summary,
            "iterations": self.iterations,
            "num_predicates": self.num_predicates,
            "num_counterexamples": self.num_counterexamples,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Core explanation generation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_layer_explanations(
    graph: ComputationGraph,
    cegar_result: ShapeCEGARResult,
    is_safe: bool,
) -> List[str]:
    """Build per-layer explanation lines."""
    lines: List[str] = []
    predicates = cegar_result.discovered_predicates

    # Build a map from layer ref -> predicates that mention it
    layer_preds: Dict[str, List[ShapePredicate]] = {}
    for pred in predicates:
        tensor = pred.tensor
        # Try to associate with a layer
        for step in graph.steps:
            if step.layer_ref and (
                tensor in step.inputs or tensor == step.output
            ):
                layer_preds.setdefault(step.layer_ref, []).append(pred)
                break

    # Walk layers in order of computation steps
    seen_layers: set = set()
    for step in graph.steps:
        if step.op != OpKind.LAYER_CALL or not step.layer_ref:
            continue
        if step.layer_ref in seen_layers:
            continue
        seen_layers.add(step.layer_ref)

        layer = graph.layers.get(step.layer_ref)
        if layer is None:
            continue

        desc = _layer_description(layer)
        marker = "✓" if is_safe else "✗"

        # Build constraint description
        constraint_parts: List[str] = []
        if layer.kind == LayerKind.LINEAR and layer.in_features:
            constraint_parts.append(
                f"Input requires last dim = {layer.in_features}."
            )
        elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D) and layer.in_channels:
            constraint_parts.append(
                f"Input requires {layer.in_channels} channels."
            )
        elif layer.kind == LayerKind.EMBEDDING and layer.num_embeddings:
            constraint_parts.append(
                f"Input indices must be < {layer.num_embeddings}."
            )

        constraint_line = " ".join(constraint_parts)
        lines.append(
            f"{marker} Layer {step.layer_ref} ({desc}): {constraint_line}"
        )

        # Add invariant info from predicates
        preds_for_layer = layer_preds.get(step.layer_ref, [])
        if preds_for_layer and is_safe:
            for pred in preds_for_layer:
                lines.append(
                    f"  Invariant discovered: {pred.pretty()}"
                )

        # Output shape guarantees
        if is_safe and layer.kind == LayerKind.LINEAR and layer.out_features:
            lines.append(
                f"  Output shape: last dim is always {layer.out_features}."
            )

    return lines


def _build_refinement_trace(cegar_result: ShapeCEGARResult) -> List[str]:
    """Build the CEGAR refinement trace lines."""
    lines: List[str] = []
    for record in cegar_result.iteration_log:
        preds_added = record.predicates_added
        if preds_added:
            for pred in preds_added:
                prov = _predicate_provenance_label(pred)
                lines.append(
                    f"  Iteration {record.iteration + 1}: "
                    f"Added predicate {pred.pretty()} ({prov})"
                )
        else:
            lines.append(
                f"  Iteration {record.iteration + 1}: "
                f"No new predicates ({record.num_violations} violations checked)"
            )

    # Result summary
    verdict = cegar_result.verdict
    total_cex = sum(r.num_real for r in cegar_result.iteration_log)
    lines.append(
        f"  Result: {verdict.name} after {cegar_result.iterations} "
        f"iteration{'s' if cegar_result.iterations != 1 else ''} "
        f"({total_cex} counterexample{'s' if total_cex != 1 else ''})"
    )
    return lines


def _build_counterexample_path(
    cegar_result: ShapeCEGARResult,
    graph: Optional[ComputationGraph],
) -> List[str]:
    """Build counterexample path explanation for unsafe models."""
    lines: List[str] = []
    vresult = cegar_result.verification_result
    if vresult is None or vresult.counterexample is None:
        if cegar_result.real_bugs:
            for bug in cegar_result.real_bugs:
                lines.append(f"  ✗ Step {bug.step_index}: {bug.message}")
        return lines

    cex = vresult.counterexample

    # Show concrete dimension assignments if available
    if cex.concrete_dims:
        dims_str = ", ".join(f"{k}={v}" for k, v in cex.concrete_dims.items())
        lines.append(f"  Input dimensions: {dims_str}")

    # Show the violation chain
    for violation in cex.violations:
        step = violation.step
        layer_info = ""
        if step.layer_ref and graph:
            layer = graph.layers.get(step.layer_ref)
            if layer:
                layer_info = f" ({_layer_description(layer)})"

        lines.append(
            f"  ✗ Step {violation.step_index}{layer_info}: {violation.message}"
        )
        if violation.shape_a and violation.shape_b:
            lines.append(
                f"    Got shape {violation.shape_a}, "
                f"expected compatible with {violation.shape_b}"
            )

    return lines


def generate_explanation(
    cegar_result: ShapeCEGARResult,
    graph: Optional[ComputationGraph] = None,
    model_name: Optional[str] = None,
) -> VerificationExplanation:
    """Generate a human-readable explanation from a ShapeCEGARResult.

    Parameters
    ----------
    cegar_result : ShapeCEGARResult
        The result from running the CEGAR loop.
    graph : ComputationGraph, optional
        The computation graph (extracted from source). If not provided,
        layer-level explanations will be limited.
    model_name : str, optional
        Override the model name in the explanation.

    Returns
    -------
    VerificationExplanation
        A structured explanation that can be rendered as text or JSON.
    """
    # Determine model name
    if model_name is None:
        if graph:
            model_name = graph.class_name
        elif (cegar_result.verification_result
              and cegar_result.verification_result.graph):
            model_name = cegar_result.verification_result.graph.class_name
            graph = cegar_result.verification_result.graph
        else:
            model_name = "Model"

    # If we have a verification_result with a graph but no graph was passed
    if graph is None and cegar_result.verification_result:
        graph = cegar_result.verification_result.graph

    is_safe = cegar_result.is_safe
    verdict = cegar_result.verdict

    # Layer explanations
    layer_explanations: List[str] = []
    if graph:
        layer_explanations = _build_layer_explanations(
            graph, cegar_result, is_safe,
        )

    # Refinement trace
    refinement_trace = _build_refinement_trace(cegar_result)

    # Counterexample path (only for unsafe)
    counterexample_path: List[str] = []
    if not is_safe:
        counterexample_path = _build_counterexample_path(
            cegar_result, graph,
        )

    total_cex = sum(r.num_real for r in cegar_result.iteration_log)

    return VerificationExplanation(
        model_name=model_name,
        verdict=verdict.name,
        layer_explanations=layer_explanations,
        refinement_trace=refinement_trace,
        counterexample_path=counterexample_path,
        summary=(
            f"Result: {verdict.name} after {cegar_result.iterations} "
            f"iteration{'s' if cegar_result.iterations != 1 else ''} "
            f"({total_cex} counterexample{'s' if total_cex != 1 else ''})"
        ),
        iterations=cegar_result.iterations,
        num_predicates=len(cegar_result.discovered_predicates),
        num_counterexamples=total_cex,
    )


def explain_verification(
    model_source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
    model_name: Optional[str] = None,
) -> VerificationExplanation:
    """Run verification on a model and produce a human-readable explanation.

    This is the main entry point for explanation generation. It runs the
    CEGAR loop, extracts the computation graph, and generates a structured
    explanation of the verification result.

    Parameters
    ----------
    model_source : str
        Python source code containing an ``nn.Module`` subclass.
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.
    max_iterations : int
        Maximum CEGAR iterations.
    model_name : str, optional
        Override the model name in the explanation.

    Returns
    -------
    VerificationExplanation
        A structured explanation that can be rendered via ``.render()``.

    Examples
    --------
    >>> explanation = explain_verification('''
    ... import torch.nn as nn
    ... class MyModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(768, 256)
    ...     def forward(self, x):
    ...         return self.fc(x)
    ... ''', input_shapes={"x": ("batch", 768)})
    >>> print(explanation.render())
    """
    # Extract graph for layer-level explanations
    graph: Optional[ComputationGraph] = None
    try:
        graph = extract_computation_graph(model_source)
    except (ValueError, SyntaxError):
        pass

    # Run CEGAR loop
    cegar_result = run_shape_cegar(
        model_source,
        input_shapes=input_shapes,
        max_iterations=max_iterations,
    )

    return generate_explanation(cegar_result, graph=graph, model_name=model_name)
