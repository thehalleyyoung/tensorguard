"""
TorchDynamo-based computation graph extraction.

Uses ``torch._dynamo`` (PyTorch 2.x) to capture computation graphs,
handling data-dependent control flow via graph breaks — which
``torch.fx.symbolic_trace`` cannot do.

Falls back to :mod:`src.fx_extractor` when TorchDynamo is unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_DYNAMO = False
if HAS_TORCH:
    try:
        import torch._dynamo
        # Verify Dynamo is actually usable (not just importable)
        torch._dynamo.eval_frame.check_if_dynamo_supported()
        HAS_DYNAMO = True
    except (ImportError, RuntimeError):
        pass

from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    LayerDef,
    LayerKind,
    OpKind,
    Device,
    Phase,
    ConstraintVerifier,
    VerificationResult,
    Confidence,
)
from src.graph_break_attribution import (
    GraphBreakAttributionReport,
    classify_graph_break_failure,
)

from src.fx_extractor import (
    fx_trace_to_graph,
    _make_layer_def,
    _module_to_layer_kind,
    _extract_layer_params,
    _function_to_op,
    _collect_node_inputs,
    _extract_function_params,
    _extract_method_params,
    verify_module as fx_verify_module,
)


def _attach_graph_break_attribution(
    result: VerificationResult,
    report: GraphBreakAttributionReport,
) -> VerificationResult:
    """Attach graph-break attribution without changing verification semantics."""
    result.dynamic_features["graph_break_attribution"] = report.to_dict()
    if report.attributions:
        first = report.attributions[0]
        location = "" if first.line is None else f" at line {first.line}"
        result.dynamic_feature_warnings.append(
            f"{report.backend} graph capture failed{location}: "
            f"{first.category}. Minimal change: {first.minimal_change}"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Composition semantics for graph-break safety
# ═══════════════════════════════════════════════════════════════════════════════


class CompositionSemantics(Enum):
    """Safety semantics under TorchDynamo graph breaks.

    When a model triggers graph breaks, TorchDynamo produces multiple
    subgraphs that are verified independently.  The semantics of the
    overall safety verdict depends on whether cross-break shape
    dependencies exist.

    MONOLITHIC_SAFE
        No graph breaks, or a single subgraph.  The safety verdict
        covers all execution paths monolithically.

    PER_SUBGRAPH_SAFE
        Multiple subgraphs, each independently verified safe, with
        no detected cross-break shape dependencies.  Safety holds
        *per subgraph* but not necessarily across the full pipeline
        if intermediate Python code between breaks alters shapes.

    UNKNOWN
        Cross-break shape dependencies detected (output shape of
        subgraph N may constrain input of subgraph N+2 through
        intermediate Python).  Independent verification may miss
        these constraints.
    """
    MONOLITHIC_SAFE = auto()
    PER_SUBGRAPH_SAFE = auto()
    UNKNOWN = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamo backend that captures FX graphs
# ═══════════════════════════════════════════════════════════════════════════════

class _DynamoGraphCapture:
    """Custom TorchDynamo backend that intercepts FX graphs.

    Each graph break produces a separate ``torch.fx.GraphModule``.
    We collect them all and later compose into a single
    ``ComputationGraph``.
    """

    def __init__(self) -> None:
        self.captured_graphs: List["torch.fx.GraphModule"] = []

    def __call__(self, gm: "torch.fx.GraphModule", example_inputs):
        self.captured_graphs.append(gm)
        return gm.forward


# ═══════════════════════════════════════════════════════════════════════════════
# Convert a single Dynamo-captured FX graph to ComputationGraph
# ═══════════════════════════════════════════════════════════════════════════════

def _dynamo_fx_to_graph(
    gm: "torch.fx.GraphModule",
    root_module: "nn.Module",
    class_name: str,
    step_offset: int = 0,
    input_remap: Optional[Dict[str, str]] = None,
) -> ComputationGraph:
    """Convert a single Dynamo-produced FX graph to a ComputationGraph.

    Similar to :func:`fx_trace_to_graph` but handles Dynamo-specific node
    patterns (e.g. ``_operator`` references, guard placeholders).
    """
    graph = ComputationGraph(class_name=class_name)
    graph.dynamic_features["dynamo_traced"] = True

    # Extract layers from the root module
    for name, module in root_module.named_modules():
        if name == "":
            continue
        flat_name = name.replace(".", "_")
        if flat_name not in graph.layers:
            graph.layers[flat_name] = _make_layer_def(flat_name, module)

    # Also extract layers from the graph module itself (Dynamo may inline)
    for name, module in gm.named_modules():
        if name == "":
            continue
        flat_name = name.replace(".", "_")
        if flat_name not in graph.layers:
            graph.layers[flat_name] = _make_layer_def(flat_name, module)

    step_idx = step_offset
    node_to_tensor: Dict[str, str] = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            tensor_name = node.name
            if input_remap and node.name in input_remap:
                tensor_name = input_remap[node.name]
            node_to_tensor[node.name] = tensor_name
            graph.input_names.append(tensor_name)

        elif node.op == "get_attr":
            node_to_tensor[node.name] = f"_attr_{node.name}"

        elif node.op == "call_module":
            target = str(node.target)
            flat_target = target.replace(".", "_")
            input_names = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_names.append(node_to_tensor.get(arg.name, arg.name))
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, torch.fx.Node):
                            input_names.append(
                                node_to_tensor.get(a.name, a.name)
                            )

            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            # Resolve the submodule from the graph module
            try:
                submodule = gm.get_submodule(target)
            except AttributeError:
                try:
                    submodule = root_module.get_submodule(target)
                except AttributeError:
                    submodule = None

            if submodule is not None:
                kind = _module_to_layer_kind(submodule)
                if flat_target not in graph.layers:
                    graph.layers[flat_target] = _make_layer_def(
                        flat_target, submodule
                    )
                if kind in (LayerKind.RELU, LayerKind.DROPOUT,
                            LayerKind.IDENTITY):
                    op = (OpKind.ACTIVATION if kind != LayerKind.DROPOUT
                          else OpKind.DROPOUT)
                elif kind == LayerKind.FLATTEN:
                    op = OpKind.FLATTEN
                else:
                    op = OpKind.LAYER_CALL
                params = _extract_layer_params(submodule, kind)
            else:
                op = OpKind.LAYER_CALL
                params = {}

            step = ComputationStep(
                op=op,
                inputs=input_names,
                output=output_name,
                layer_ref=flat_target,
                params=params,
            )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "call_function":
            op_kind = _function_to_op(node.target)
            if op_kind is None:
                op_kind = OpKind.ACTIVATION

            input_names = _collect_node_inputs(node, node_to_tensor)
            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            params = _extract_function_params(node, op_kind)

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "call_method":
            method_name = str(node.target)
            input_names = _collect_node_inputs(node, node_to_tensor)
            output_name = f"_t{step_idx}"
            node_to_tensor[node.name] = output_name

            from src.fx_extractor import _METHOD_OP_MAP as _FX_METHOD_OP_MAP
            op_kind = _FX_METHOD_OP_MAP.get(method_name, OpKind.ACTIVATION)
            params = _extract_method_params(node, method_name, op_kind)

            step = ComputationStep(
                op=op_kind,
                inputs=input_names,
                output=output_name,
                params=params,
            )
            graph.steps.append(step)
            step_idx += 1

        elif node.op == "output":
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    tensor_name = node_to_tensor.get(arg.name, arg.name)
                    graph.output_names.append(tensor_name)
                elif isinstance(arg, (tuple, list)):
                    for a in arg:
                        if isinstance(a, torch.fx.Node):
                            tensor_name = node_to_tensor.get(a.name, a.name)
                            graph.output_names.append(tensor_name)

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Compose multiple subgraphs from graph breaks
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_cross_break_dependencies(
    subgraphs: List[ComputationGraph],
) -> bool:
    """Detect whether cross-break shape dependencies exist.

    Returns True when the output shape of subgraph N could constrain the
    input of subgraph N+2 (or later) through intermediate Python code
    that is invisible to any single subgraph's verification.
    """
    if len(subgraphs) <= 1:
        return False

    # Build the set of tensor names produced by each subgraph
    produced_by: Dict[str, int] = {}  # tensor_name -> subgraph index
    for sg_idx, sg in enumerate(subgraphs):
        for step in sg.steps:
            produced_by[step.output] = sg_idx

    for sg_idx, sg in enumerate(subgraphs):
        if sg_idx == 0:
            continue
        for step in sg.steps:
            for inp in step.inputs:
                producer = produced_by.get(inp)
                # If an input comes from a subgraph that is NOT the
                # immediately preceding one, there is a transitive
                # dependency that may flow through opaque Python code.
                if producer is not None and producer < sg_idx - 1:
                    return True

    # Also flag if any subgraph's input names are *not* found in the
    # previous subgraph's outputs (indicating intermediate Python).
    for sg_idx in range(1, len(subgraphs)):
        prev_out = set(subgraphs[sg_idx - 1].output_names)
        for inp_name in subgraphs[sg_idx].input_names:
            if inp_name not in prev_out:
                # This input comes from outside the previous subgraph,
                # potentially through Python code between breaks.
                return True

    return False


def _compose_subgraphs(
    subgraphs: List[ComputationGraph],
    class_name: str,
) -> Tuple[ComputationGraph, CompositionSemantics]:
    """Compose multiple subgraphs (from graph breaks) into one.

    Each subgraph's outputs feed into the next subgraph's inputs.

    Returns
    -------
    (ComputationGraph, CompositionSemantics)
        The composed graph and the applicable safety semantics:
        - MONOLITHIC_SAFE when there is only one subgraph.
        - PER_SUBGRAPH_SAFE when all subgraphs chain cleanly.
        - UNKNOWN when cross-break shape dependencies are detected.
    """
    if len(subgraphs) == 1:
        return subgraphs[0], CompositionSemantics.MONOLITHIC_SAFE

    has_cross_deps = _detect_cross_break_dependencies(subgraphs)
    semantics = (CompositionSemantics.UNKNOWN if has_cross_deps
                 else CompositionSemantics.PER_SUBGRAPH_SAFE)

    # Thread-modular verification: refine PER_SUBGRAPH_SAFE verdict
    thread_modular_result = None
    if semantics == CompositionSemantics.PER_SUBGRAPH_SAFE:
        try:
            from src.thread_modular import (
                ThreadModularVerifier,
                CompositionVerdict,
            )
            tmv = ThreadModularVerifier(subgraphs)
            thread_modular_result = tmv.verify()
        except Exception as exc:
            logger.debug("Thread-modular verification failed: %s", exc)

    composed = ComputationGraph(class_name=class_name)
    composed.dynamic_features["dynamo_traced"] = True
    composed.dynamic_features["graph_breaks"] = len(subgraphs) - 1
    composed.dynamic_features["composition_semantics"] = semantics.name
    if thread_modular_result is not None:
        composed.dynamic_features["thread_modular_verdict"] = (
            thread_modular_result.verdict.name
        )
        composed.dynamic_features["thread_modular_gaps"] = len(
            thread_modular_result.gaps
        )
    if has_cross_deps:
        composed.dynamic_features["cross_break_dependencies"] = True

    # Collect all layers
    for sg in subgraphs:
        composed.layers.update(sg.layers)

    # First subgraph's inputs are the composed graph's inputs
    composed.input_names = list(subgraphs[0].input_names)

    # Chain subgraphs together, renaming tensors for uniqueness
    step_counter = 0
    prev_outputs: Dict[str, str] = {}

    for sg_idx, sg in enumerate(subgraphs):
        for step in sg.steps:
            # Remap inputs from previous subgraph's outputs
            remapped_inputs = []
            for inp in step.inputs:
                remapped_inputs.append(prev_outputs.get(inp, inp))

            new_output = f"_t{step_counter}"
            new_step = ComputationStep(
                op=step.op,
                inputs=remapped_inputs,
                output=new_output,
                layer_ref=step.layer_ref,
                params=dict(step.params),
                line=step.line,
                col=step.col,
                condition=step.condition,
                true_branch=step.true_branch,
                false_branch=step.false_branch,
            )
            composed.steps.append(new_step)

            # Map this subgraph's output name to the composed name
            prev_outputs[step.output] = new_output
            step_counter += 1

        # Map subgraph outputs for the next subgraph's inputs
        for out_name in sg.output_names:
            if out_name in prev_outputs:
                pass  # already mapped

    # Last subgraph's outputs are the composed graph's outputs
    if subgraphs:
        last = subgraphs[-1]
        for out_name in last.output_names:
            composed.output_names.append(prev_outputs.get(out_name, out_name))

    return composed, semantics


# ═══════════════════════════════════════════════════════════════════════════════
# Main API: dynamo_trace_to_graph
# ═══════════════════════════════════════════════════════════════════════════════

def dynamo_trace_to_graph(
    module: "nn.Module",
    example_inputs: Optional[Tuple] = None,
    class_name: Optional[str] = None,
) -> ComputationGraph:
    """Capture a computation graph from *module* using TorchDynamo.

    Parameters
    ----------
    module : nn.Module
        The model to trace.
    example_inputs : tuple, optional
        Concrete example inputs for tracing.  If ``None`` a small
        random tensor is synthesised.
    class_name : str, optional
        Override class name in the resulting graph.

    Returns
    -------
    ComputationGraph

    Raises
    ------
    RuntimeError
        If TorchDynamo is not available.
    """
    if not HAS_DYNAMO:
        raise RuntimeError(
            "TorchDynamo not available (requires PyTorch >= 2.0). "
            "Use fx_trace_to_graph or verify_module as a fallback."
        )

    cname = class_name or type(module).__name__
    module.eval()

    if example_inputs is None:
        example_inputs = (torch.randn(2, 64),)

    capture = _DynamoGraphCapture()

    # Reset Dynamo state for clean capture
    torch._dynamo.reset()

    try:
        optimized = torch._dynamo.optimize(capture)(module)
        with torch.no_grad():
            optimized(*example_inputs)
    except Exception as exc:
        logger.warning("TorchDynamo capture failed: %s", exc)
        raise

    if not capture.captured_graphs:
        raise RuntimeError("TorchDynamo did not capture any graphs")

    # Convert each captured FX graph to a ComputationGraph
    subgraphs: List[ComputationGraph] = []
    step_offset = 0
    for gm in capture.captured_graphs:
        sg = _dynamo_fx_to_graph(
            gm, module, cname, step_offset=step_offset,
        )
        subgraphs.append(sg)
        step_offset += len(sg.steps)

    result, semantics = _compose_subgraphs(subgraphs, cname)
    result.dynamic_features["num_dynamo_subgraphs"] = len(subgraphs)
    result.dynamic_features["composition_semantics"] = semantics.name
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# High-level API: verify_module_dynamo
# ═══════════════════════════════════════════════════════════════════════════════

def verify_module_dynamo(
    module: "nn.Module",
    input_shapes: Optional[Dict[str, tuple]] = None,
    example_inputs: Optional[Tuple] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    constraints: Optional[Dict[str, Union[str, int]]] = None,
    high_confidence_only: bool = False,
    class_name: Optional[str] = None,
    fallback_to_fx: bool = True,
) -> VerificationResult:
    """Verify an ``nn.Module`` using TorchDynamo for graph capture.

    This is the Dynamo-specific entry point.  For automatic backend
    selection (dynamo → fx → AST), use
    :func:`src.fx_extractor.verify_module` with ``backend="auto"``.

    **Auto backend selection algorithm** (in ``verify_module``):

    The auto selector maximises *coverage* first, then *soundness*:

    1. **Dynamo** (this function): Preferred.  Captures all reachable
       subgraphs including across graph breaks.  Each subgraph is
       individually sound.  Gap: cross-break shape dependencies may be
       invisible (PER_SUBGRAPH_SAFE or UNKNOWN semantics).

    2. **FX** (``torch.fx.symbolic_trace``): Fallback when Dynamo fails.
       Sound within a single trace but misses data-dependent branches.

    3. **AST**: Always available.  Sound for statically analyzable
       patterns but incomplete for dynamic control flow.

    **Soundness**: Within each backend, a reported violation is genuine.
    The gap is in *completeness* — uncaptured paths are not checked.

    Falls back to :func:`src.fx_extractor.verify_module` when Dynamo
    is unavailable or fails, unless *fallback_to_fx* is ``False``.

    Parameters
    ----------
    module : nn.Module
        The model instance to verify.
    input_shapes : dict, optional
        Input shape specification.
    example_inputs : tuple, optional
        Concrete example inputs for Dynamo tracing.
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth.
    constraints : dict, optional
        Relational constraints between symbolic dimensions.
    high_confidence_only : bool
        When True, only report Z3-proven violations.
    class_name : str, optional
        Override class name.
    fallback_to_fx : bool
        Fall back to torch.fx when Dynamo is unavailable (default True).

    Returns
    -------
    VerificationResult
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for verify_module_dynamo")

    t0 = time.monotonic()

    # Build example inputs from input_shapes if not provided
    if example_inputs is None and input_shapes is not None:
        tensors = []
        for name, shape in input_shapes.items():
            concrete = []
            for d in shape:
                if isinstance(d, int):
                    concrete.append(d)
                elif isinstance(d, str):
                    concrete.append(2)
                else:
                    concrete.append(2)
            tensors.append(torch.randn(*concrete))
        if tensors:
            example_inputs = tuple(tensors)

    # Try Dynamo
    graph = None
    dynamo_error = None
    if HAS_DYNAMO:
        try:
            cname = class_name or type(module).__name__
            graph = dynamo_trace_to_graph(
                module,
                example_inputs=example_inputs,
                class_name=cname,
            )
        except Exception as exc:
            dynamo_error = str(exc)
            logger.info("Dynamo extraction failed: %s; trying fallback", exc)

    if graph is None and fallback_to_fx:
        result = fx_verify_module(
            module,
            input_shapes=input_shapes,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
            constraints=constraints,
            high_confidence_only=high_confidence_only,
            class_name=class_name,
        )
        if dynamo_error is not None:
            report = classify_graph_break_failure(
                module,
                dynamo_error,
                backend="dynamo",
                fallback_used="fx",
            )
            _attach_graph_break_attribution(result, report)
        return result

    if graph is None:
        msg = dynamo_error or "TorchDynamo not available"
        report = classify_graph_break_failure(
            module,
            msg,
            backend="dynamo",
        )
        return VerificationResult(
            safe=False,
            errors=[f"Dynamo graph capture failed: {msg}"],
            verification_time_ms=(time.monotonic() - t0) * 1000,
            dynamic_features={"graph_break_attribution": report.to_dict()},
            dynamic_feature_warnings=[
                (
                    f"dynamo graph capture failed: {report.attributions[0].category}. "
                    f"Minimal change: {report.attributions[0].minimal_change}"
                )
            ] if report.attributions else [],
        )

    # Verify
    checker = ConstraintVerifier(
        graph,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        constraints=constraints,
    )
    result = checker.verify()
    if high_confidence_only:
        result = result.filter_by_confidence(Confidence.HIGH)

    # Attach composition semantics to the result
    comp_sem = graph.dynamic_features.get("composition_semantics")
    if comp_sem is not None:
        result.dynamic_features["composition_semantics"] = comp_sem
        if result.safe:
            if comp_sem == CompositionSemantics.MONOLITHIC_SAFE.name:
                result.dynamic_features["safety_note"] = (
                    "SAFE: monolithic verification — no graph breaks, "
                    "safety covers all execution paths."
                )
            elif comp_sem == CompositionSemantics.PER_SUBGRAPH_SAFE.name:
                result.dynamic_features["safety_note"] = (
                    "SAFE (per-subgraph): each subgraph independently "
                    "verified safe.  No cross-break shape dependencies "
                    "detected, but intermediate Python between graph "
                    "breaks is not verified."
                )
            elif comp_sem == CompositionSemantics.UNKNOWN.name:
                result.dynamic_features["safety_note"] = (
                    "SAFE (subgraph-local only): cross-break shape "
                    "dependencies detected.  Independent subgraph "
                    "verification may miss transitive constraints."
                )
        if graph.dynamic_features.get("cross_break_dependencies"):
            result.dynamic_feature_warnings.append(
                "Cross-break shape dependencies detected: output shape "
                "of an earlier subgraph may constrain a later subgraph "
                "through intermediate Python code.  Composition semantics: "
                f"{comp_sem}."
            )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamoTraceStats:
    """Statistics about a TorchDynamo trace."""
    traceable: bool
    backend: str = "dynamo"  # "dynamo" | "fx_fallback" | "failed"
    trace_error: Optional[str] = None
    num_subgraphs: int = 0
    num_graph_breaks: int = 0
    num_steps: int = 0
    num_layers: int = 0
    num_inputs: int = 0
    num_outputs: int = 0
    layer_kinds: Dict[str, int] = field(default_factory=dict)
    op_kinds: Dict[str, int] = field(default_factory=dict)
    composition_semantics: Optional[str] = None


def dynamo_trace_stats(
    module: "nn.Module",
    example_inputs: Optional[Tuple] = None,
) -> DynamoTraceStats:
    """Get tracing statistics using TorchDynamo."""
    if not HAS_TORCH:
        return DynamoTraceStats(
            traceable=False, backend="failed",
            trace_error="PyTorch not available",
        )

    if not HAS_DYNAMO:
        return DynamoTraceStats(
            traceable=False, backend="failed",
            trace_error="TorchDynamo not available",
        )

    try:
        graph = dynamo_trace_to_graph(module, example_inputs=example_inputs)
    except Exception as exc:
        return DynamoTraceStats(
            traceable=False, backend="failed", trace_error=str(exc),
        )

    layer_counts: Dict[str, int] = {}
    for ldef in graph.layers.values():
        k = ldef.kind.name
        layer_counts[k] = layer_counts.get(k, 0) + 1

    op_counts: Dict[str, int] = {}
    for step in graph.steps:
        k = step.op.name
        op_counts[k] = op_counts.get(k, 0) + 1

    return DynamoTraceStats(
        traceable=True,
        backend="dynamo",
        num_subgraphs=graph.dynamic_features.get("num_dynamo_subgraphs", 1),
        num_graph_breaks=graph.dynamic_features.get("graph_breaks", 0),
        num_steps=graph.num_steps,
        num_layers=len(graph.layers),
        num_inputs=len(graph.input_names),
        num_outputs=len(graph.output_names),
        layer_kinds=layer_counts,
        op_kinds=op_counts,
        composition_semantics=graph.dynamic_features.get("composition_semantics"),
    )
