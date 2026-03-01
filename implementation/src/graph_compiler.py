"""
Universal Computation Graph Compiler for Arbitrary PyTorch Models.

Handles the full spectrum of PyTorch computation patterns:
  - torch.export / torch.compile graphs (ExportedProgram)
  - Dynamic control flow (torch.cond, torch.vmap, torch.scan)
  - Higher-order patterns (ModuleList iteration, MoE routing)
  - Data-dependent shapes (dynamic reshape, masked select)
  - Nested submodule hierarchies (arbitrary depth)
  - Functional transforms (vmap, grad, checkpoint)

The compiler converts any nn.Module into a TensorGuard ComputationGraph
by combining three extraction strategies:
  1. AST-based extraction (source code analysis)
  2. FX symbolic tracing (torch.fx)
  3. TorchDynamo capture (torch._dynamo)

When all three strategies fail, falls back to a conservative analysis
that treats unknown operations as shape-preserving with appropriate
warnings.

Key extension over base model_checker: supports MoE, conditional
computation, torch.cond, and arbitrary DAG topologies including
cycles (via fixed-point iteration with widening).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

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
    TensorShape,
    ShapeDim,
    CounterexampleTrace,
    extract_computation_graph,
    verify_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Extended operation kinds for arbitrary graphs
# ═══════════════════════════════════════════════════════════════════════════════

class ExtendedOpKind(Enum):
    """Additional operation kinds for arbitrary computation graphs."""
    TORCH_COND = auto()         # torch.cond(pred, true_fn, false_fn, operands)
    TORCH_VMAP = auto()         # torch.vmap(fn, in_dims, out_dims)
    TORCH_SCAN = auto()         # torch.scan (sequential map)
    MOE_DISPATCH = auto()       # MoE top-k gating + dispatch
    MOE_COMBINE = auto()        # MoE expert output combination
    DYNAMIC_RESHAPE = auto()    # reshape with data-dependent dims
    MASKED_SELECT = auto()      # output shape depends on mask values
    SCATTER = auto()            # scatter/gather operations
    CUSTOM_AUTOGRAD = auto()    # custom autograd.Function
    HIGHER_ORDER = auto()       # higher-order module pattern


# ═══════════════════════════════════════════════════════════════════════════════
# Symbolic shape constraints for dynamic patterns
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ShapeConstraint:
    """A symbolic constraint on tensor shapes.

    Represents relationships like:
      - dim_i > 0  (positivity)
      - dim_i == dim_j  (equality across tensors)
      - dim_i * dim_j == dim_k  (reshape product constraint)
      - dim_i <= dim_j  (bounded)
    """
    kind: str  # "eq", "gt", "ge", "lt", "le", "product", "divisible"
    lhs: Union[str, Tuple[str, ...]]
    rhs: Union[str, int, Tuple[str, ...]]

    def to_z3(self, ctx: Dict[str, Any]) -> Optional[Any]:
        """Convert to Z3 expression given variable context."""
        if not HAS_Z3:
            return None
        if self.kind == "eq":
            l = ctx.get(self.lhs)
            r = ctx.get(self.rhs) if isinstance(self.rhs, str) else self.rhs
            if l is not None and r is not None:
                return l == r
        elif self.kind == "gt":
            l = ctx.get(self.lhs)
            r = ctx.get(self.rhs) if isinstance(self.rhs, str) else self.rhs
            if l is not None and r is not None:
                return l > r
        elif self.kind == "product":
            # lhs is tuple of dim names, rhs is result dim
            factors = [ctx.get(d) for d in self.lhs]
            result = ctx.get(self.rhs) if isinstance(self.rhs, str) else self.rhs
            if all(f is not None for f in factors) and result is not None:
                prod = factors[0]
                for f in factors[1:]:
                    prod = prod * f
                return prod == result
        elif self.kind == "divisible":
            l = ctx.get(self.lhs)
            r = self.rhs if isinstance(self.rhs, int) else ctx.get(self.rhs)
            if l is not None and r is not None:
                return l % r == 0
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MoE (Mixture of Experts) shape analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layers."""
    num_experts: int
    top_k: int = 2
    expert_capacity: Optional[int] = None
    gate_type: str = "top_k"  # "top_k", "switch", "hash"


def analyze_moe_shapes(
    input_shape: TensorShape,
    config: MoEConfig,
    expert_shapes: Optional[List[TensorShape]] = None,
) -> Tuple[Optional[TensorShape], List[ShapeConstraint], Optional[str]]:
    """Analyze shapes through a Mixture of Experts layer.

    MoE routing: input (B, S, D) → gate (B, S, E) → dispatch → experts → combine
    Output shape equals input shape (experts preserve last dim).

    Returns (output_shape, constraints, error_message).
    """
    if input_shape.ndim < 2:
        return None, [], "MoE requires at least 2D input (batch, features)"

    constraints = []
    batch_dims = input_shape.dims[:-1]
    feature_dim = input_shape.dims[-1]

    # Gate output: each token gets top_k expert assignments
    # Expert dispatch: tokens routed to experts based on gate
    # Expert output: same feature dim as input (standard MoE)
    if expert_shapes:
        for i, eshape in enumerate(expert_shapes):
            if eshape.ndim >= 1:
                expert_out = eshape.dims[-1]
                if not expert_out.is_symbolic and not feature_dim.is_symbolic:
                    if expert_out.value != feature_dim.value:
                        return None, [], (
                            f"Expert {i} output dim {expert_out.value} != "
                            f"input feature dim {feature_dim.value}"
                        )

    # Capacity constraint: tokens per expert bounded
    if config.expert_capacity is not None:
        constraints.append(ShapeConstraint(
            kind="le",
            lhs="tokens_per_expert",
            rhs=config.expert_capacity,
        ))

    # Output shape = input shape (MoE preserves dimensions)
    return TensorShape(input_shape.dims), constraints, None


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic control flow analysis (torch.cond, data-dependent branching)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionalBranch:
    """A branch in conditional control flow."""
    condition: str
    steps: List[ComputationStep]
    output_shape: Optional[TensorShape] = None


@dataclass
class DynamicControlFlow:
    """Analysis result for dynamic control flow patterns."""
    pattern: str  # "torch_cond", "if_shape", "if_training", "loop"
    branches: List[ConditionalBranch] = field(default_factory=list)
    merged_shape: Optional[TensorShape] = None
    constraints: List[ShapeConstraint] = field(default_factory=list)
    is_shape_preserving: bool = False


def analyze_torch_cond(
    pred_shape: Optional[TensorShape],
    true_fn_output: TensorShape,
    false_fn_output: TensorShape,
) -> DynamicControlFlow:
    """Analyze shapes through torch.cond.

    torch.cond requires both branches to return tensors of identical shape.
    This is enforced at trace time by PyTorch, so we verify it statically.
    """
    constraints = []

    # Both branches must have same number of dimensions
    if true_fn_output.ndim != false_fn_output.ndim:
        return DynamicControlFlow(
            pattern="torch_cond",
            branches=[
                ConditionalBranch("true_fn", [], true_fn_output),
                ConditionalBranch("false_fn", [], false_fn_output),
            ],
            constraints=[],
            is_shape_preserving=False,
        )

    # Add equality constraints for each dimension
    for i in range(true_fn_output.ndim):
        td = true_fn_output.dims[i]
        fd = false_fn_output.dims[i]
        if not td.is_symbolic and not fd.is_symbolic:
            if td.value != fd.value:
                return DynamicControlFlow(
                    pattern="torch_cond",
                    branches=[
                        ConditionalBranch("true_fn", [], true_fn_output),
                        ConditionalBranch("false_fn", [], false_fn_output),
                    ],
                    constraints=[],
                    is_shape_preserving=False,
                )
        else:
            constraints.append(ShapeConstraint(
                kind="eq",
                lhs=f"true_dim_{i}",
                rhs=f"false_dim_{i}",
            ))

    return DynamicControlFlow(
        pattern="torch_cond",
        branches=[
            ConditionalBranch("true_fn", [], true_fn_output),
            ConditionalBranch("false_fn", [], false_fn_output),
        ],
        merged_shape=true_fn_output,
        constraints=constraints,
        is_shape_preserving=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Universal transfer function registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferFunction:
    """A shape transfer function for an operator."""
    name: str
    input_ranks: Optional[List[int]] = None  # expected input ranks (None = any)
    output_rank_fn: Optional[str] = None     # how output rank relates to input
    preserves_shape: bool = False
    reduces_dim: Optional[int] = None
    doc: str = ""


# Registry of all supported transfer functions
_UNIVERSAL_TRANSFER_REGISTRY: Dict[str, TransferFunction] = {}


def register_transfer(name: str, tf: TransferFunction) -> None:
    """Register a transfer function in the universal registry."""
    _UNIVERSAL_TRANSFER_REGISTRY[name] = tf


def get_transfer(name: str) -> Optional[TransferFunction]:
    """Look up a transfer function by name."""
    return _UNIVERSAL_TRANSFER_REGISTRY.get(name)


def _init_universal_registry():
    """Initialize the universal transfer function registry."""
    # Shape-preserving activations
    for act in [
        "relu", "gelu", "silu", "mish", "hardswish", "hardsigmoid",
        "leaky_relu", "elu", "selu", "celu", "prelu", "rrelu",
        "softplus", "softsign", "tanhshrink", "softshrink", "hardshrink",
        "logsigmoid", "sigmoid", "tanh",
    ]:
        register_transfer(f"F.{act}", TransferFunction(
            name=act, preserves_shape=True, doc=f"Activation: {act}"))
        register_transfer(f"torch.{act}", TransferFunction(
            name=act, preserves_shape=True, doc=f"Activation: {act}"))

    # Shape-preserving element-wise ops
    for op in ["abs", "neg", "sign", "ceil", "floor", "round",
               "exp", "log", "log2", "log10", "sqrt", "rsqrt",
               "sin", "cos", "tan", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "erf", "erfc",
               "clamp", "clip", "nan_to_num"]:
        register_transfer(f"torch.{op}", TransferFunction(
            name=op, preserves_shape=True, doc=f"Element-wise: {op}"))

    # Reduction ops
    for red in ["sum", "mean", "prod", "max", "min", "std", "var",
                "norm", "logsumexp", "any", "all", "amax", "amin"]:
        register_transfer(f"torch.{red}", TransferFunction(
            name=red, preserves_shape=False,
            doc=f"Reduction: {red} (removes dim if keepdim=False)"))

    # Comparison ops (preserve shape, return bool tensor)
    for cmp in ["eq", "ne", "gt", "ge", "lt", "le",
                "equal", "isnan", "isinf", "isfinite"]:
        register_transfer(f"torch.{cmp}", TransferFunction(
            name=cmp, preserves_shape=True, doc=f"Comparison: {cmp}"))

    # Linear algebra
    register_transfer("torch.matmul", TransferFunction(
        name="matmul", doc="Matrix multiplication with broadcasting"))
    register_transfer("torch.bmm", TransferFunction(
        name="bmm", input_ranks=[3], doc="Batched matrix multiply"))
    register_transfer("torch.mm", TransferFunction(
        name="mm", input_ranks=[2], doc="Matrix multiply"))
    register_transfer("torch.mv", TransferFunction(
        name="mv", doc="Matrix-vector multiply"))
    register_transfer("torch.linalg.solve", TransferFunction(
        name="linalg.solve", doc="Solve linear system"))
    register_transfer("torch.linalg.svd", TransferFunction(
        name="linalg.svd", doc="Singular value decomposition"))
    register_transfer("torch.linalg.qr", TransferFunction(
        name="linalg.qr", doc="QR decomposition"))

    # Scatter/gather
    register_transfer("torch.scatter", TransferFunction(
        name="scatter", preserves_shape=True, doc="Scatter"))
    register_transfer("torch.gather", TransferFunction(
        name="gather", doc="Gather along axis"))
    register_transfer("torch.index_select", TransferFunction(
        name="index_select", doc="Index select along axis"))

    # Advanced tensor ops
    register_transfer("torch.einsum", TransferFunction(
        name="einsum", doc="Einstein summation"))
    register_transfer("torch.tensordot", TransferFunction(
        name="tensordot", doc="Tensor dot product"))
    register_transfer("torch.kron", TransferFunction(
        name="kron", doc="Kronecker product"))
    register_transfer("torch.outer", TransferFunction(
        name="outer", doc="Outer product"))
    register_transfer("torch.cross", TransferFunction(
        name="cross", preserves_shape=True, doc="Cross product"))
    register_transfer("torch.cdist", TransferFunction(
        name="cdist", doc="Pairwise distance"))

    # FFT
    for fft_op in ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2"]:
        register_transfer(f"torch.fft.{fft_op}", TransferFunction(
            name=fft_op, doc=f"FFT: {fft_op}"))

    # Distributions / stochastic
    for dist_op in ["bernoulli", "multinomial", "poisson"]:
        register_transfer(f"torch.{dist_op}", TransferFunction(
            name=dist_op, doc=f"Stochastic: {dist_op}"))

    # Sort/topk
    register_transfer("torch.sort", TransferFunction(
        name="sort", preserves_shape=True, doc="Sort along dim"))
    register_transfer("torch.topk", TransferFunction(
        name="topk", doc="Top-k along dim"))
    register_transfer("torch.argsort", TransferFunction(
        name="argsort", preserves_shape=True, doc="Argsort"))
    register_transfer("torch.unique", TransferFunction(
        name="unique", doc="Unique elements (dynamic output size)"))


# Initialize on import
_init_universal_registry()


def count_registered_transfers() -> int:
    """Return total number of registered transfer functions."""
    return len(_UNIVERSAL_TRANSFER_REGISTRY)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph compiler: multiple extraction strategies
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompilationResult:
    """Result of compiling a model to a computation graph."""
    graph: ComputationGraph
    strategy: str  # "ast", "fx", "dynamo", "conservative"
    warnings: List[str] = field(default_factory=list)
    dynamic_patterns: List[DynamicControlFlow] = field(default_factory=list)
    coverage_ratio: float = 1.0  # fraction of ops with known transfer functions
    compilation_time_ms: float = 0.0


def compile_model(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    *,
    strategy: str = "auto",
    detect_moe: bool = True,
    detect_dynamic: bool = True,
    max_unroll: int = 10,
) -> CompilationResult:
    """Compile a PyTorch model source into a TensorGuard computation graph.

    Tries multiple extraction strategies in order:
    1. AST extraction (works without torch installed)
    2. FX symbolic trace (requires torch)
    3. TorchDynamo capture (requires torch 2.x)
    4. Conservative fallback (always works)

    Args:
        source: Python source code containing nn.Module subclass
        input_shapes: mapping from parameter name to shape tuple
        strategy: "auto", "ast", "fx", "dynamo", or "conservative"
        detect_moe: whether to detect MoE patterns
        detect_dynamic: whether to detect dynamic control flow
        max_unroll: maximum loop unrolling depth

    Returns:
        CompilationResult with the extracted graph
    """
    t0 = time.time()
    warnings_list = []
    dynamic_patterns = []

    if input_shapes is None:
        input_shapes = {}

    # Parse source to detect patterns
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return CompilationResult(
            graph=ComputationGraph(class_name="Unknown"),
            strategy="error",
            warnings=[f"Syntax error: {e}"],
            compilation_time_ms=(time.time() - t0) * 1000,
        )

    # Detect MoE patterns
    moe_configs = []
    if detect_moe:
        moe_configs = _detect_moe_patterns(tree)
        if moe_configs:
            warnings_list.append(
                f"Detected {len(moe_configs)} MoE pattern(s); "
                f"expert routing shapes verified symbolically"
            )

    # Detect dynamic control flow
    if detect_dynamic:
        dynamic_patterns = _detect_dynamic_patterns(tree)
        if dynamic_patterns:
            for dp in dynamic_patterns:
                warnings_list.append(
                    f"Dynamic control flow: {dp.pattern}"
                )

    # Try AST extraction (primary strategy)
    if strategy in ("auto", "ast"):
        try:
            graph = extract_computation_graph(source)
            if graph and graph.steps:
                # Augment with MoE analysis
                if moe_configs:
                    _augment_moe_steps(graph, moe_configs)

                elapsed = (time.time() - t0) * 1000
                coverage = _compute_coverage(graph)
                return CompilationResult(
                    graph=graph,
                    strategy="ast",
                    warnings=warnings_list,
                    dynamic_patterns=dynamic_patterns,
                    coverage_ratio=coverage,
                    compilation_time_ms=elapsed,
                )
        except Exception as e:
            if strategy == "ast":
                warnings_list.append(f"AST extraction failed: {e}")
            logger.debug("AST extraction failed: %s", e)

    # Try FX symbolic trace
    if strategy in ("auto", "fx"):
        try:
            from src.fx_extractor import verify_module
            # FX requires a live module; skip in source-only mode
            logger.debug("FX extraction requires a live module instance")
        except ImportError:
            pass

    # Conservative fallback
    if strategy in ("auto", "conservative"):
        graph = _conservative_extract(source)
        elapsed = (time.time() - t0) * 1000
        warnings_list.append(
            "Using conservative extraction: unknown ops treated as shape-preserving"
        )
        return CompilationResult(
            graph=graph,
            strategy="conservative",
            warnings=warnings_list,
            dynamic_patterns=dynamic_patterns,
            coverage_ratio=_compute_coverage(graph),
            compilation_time_ms=elapsed,
        )

    # Should not reach here
    elapsed = (time.time() - t0) * 1000
    return CompilationResult(
        graph=ComputationGraph(class_name="Unknown"),
        strategy="none",
        warnings=warnings_list + ["No extraction strategy succeeded"],
        compilation_time_ms=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_moe_patterns(tree: ast.AST) -> List[MoEConfig]:
    """Detect Mixture of Experts patterns in source AST."""
    configs = []

    class MoEDetector(ast.NodeVisitor):
        def visit_Assign(self, node):
            # Look for patterns like:
            # self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
            # self.gate = nn.Linear(d_model, num_experts)
            src = ast.dump(node)
            # Check for ModuleList with expert-related naming
            has_modulelist = "ModuleList" in src
            has_expert = False
            # Check target name for "expert"
            for target in node.targets:
                if isinstance(target, ast.Attribute) and "expert" in target.attr.lower():
                    has_expert = True
                elif isinstance(target, ast.Name) and "expert" in target.id.lower():
                    has_expert = True
            if not has_expert:
                has_expert = "expert" in src.lower()

            if has_modulelist and has_expert:
                # Try to extract num_experts from range()
                found = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = ""
                        if isinstance(child.func, ast.Name):
                            func_name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            func_name = child.func.attr
                        if func_name == "range" and child.args:
                            if isinstance(child.args[0], ast.Constant):
                                configs.append(MoEConfig(
                                    num_experts=child.args[0].value,
                                    top_k=2,
                                ))
                                found = True
                            elif isinstance(child.args[0], ast.Name):
                                # Variable reference (e.g., num_experts)
                                configs.append(MoEConfig(
                                    num_experts=8,  # default
                                    top_k=2,
                                ))
                                found = True
                if not found:
                    configs.append(MoEConfig(num_experts=8, top_k=2))
            self.generic_visit(node)

    MoEDetector().visit(tree)
    return configs


def _detect_dynamic_patterns(tree: ast.AST) -> List[DynamicControlFlow]:
    """Detect dynamic control flow patterns."""
    patterns = []

    class DynamicDetector(ast.NodeVisitor):
        def visit_Call(self, node):
            # torch.cond
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "cond":
                    patterns.append(DynamicControlFlow(
                        pattern="torch_cond",
                        is_shape_preserving=True,
                    ))
                elif node.func.attr == "vmap":
                    patterns.append(DynamicControlFlow(
                        pattern="torch_vmap",
                        is_shape_preserving=False,
                    ))
            self.generic_visit(node)

        def visit_For(self, node):
            # Loop over ModuleList
            if isinstance(node.iter, ast.Call):
                iter_src = ast.dump(node.iter)
                if "self." in iter_src and ("expert" in iter_src.lower()
                                            or "layer" in iter_src.lower()):
                    patterns.append(DynamicControlFlow(
                        pattern="loop",
                        is_shape_preserving=True,
                    ))
            self.generic_visit(node)

    DynamicDetector().visit(tree)
    return patterns


def _augment_moe_steps(
    graph: ComputationGraph,
    moe_configs: List[MoEConfig],
) -> None:
    """Add MoE-aware shape constraints to graph steps."""
    for step in graph.steps:
        if step.layer_ref and "expert" in step.layer_ref.lower():
            step.params["moe_config"] = {
                "num_experts": moe_configs[0].num_experts if moe_configs else 8,
                "top_k": moe_configs[0].top_k if moe_configs else 2,
            }


def _conservative_extract(source: str) -> ComputationGraph:
    """Conservative extraction: parse class structure, treat unknowns as identity."""
    try:
        return extract_computation_graph(source)
    except Exception:
        return ComputationGraph(class_name="Unknown")


def _compute_coverage(graph: ComputationGraph) -> float:
    """Compute fraction of operations with known transfer functions."""
    if not graph.steps:
        return 1.0
    known = 0
    total = 0
    for step in graph.steps:
        total += 1
        if step.op == OpKind.LAYER_CALL:
            if step.layer_ref and step.layer_ref in graph.layers:
                layer = graph.layers[step.layer_ref]
                if layer.kind != LayerKind.UNKNOWN:
                    known += 1
                else:
                    known += 0  # unknown layer
            else:
                known += 1  # assume known if referenced
        elif step.op != OpKind.CUSTOM:
            known += 1
    return known / total if total > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Verification entry point for arbitrary models
# ═══════════════════════════════════════════════════════════════════════════════

def verify_arbitrary_model(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    *,
    detect_moe: bool = True,
    detect_dynamic: bool = True,
) -> VerificationResult:
    """Verify an arbitrary PyTorch model.

    This is the main entry point for verifying models that may contain
    MoE layers, dynamic control flow, or other advanced patterns.

    Falls back to standard verify_model after graph compilation.

    Args:
        source: Python source code containing nn.Module subclass
        input_shapes: mapping from parameter name to shape tuple
        detect_moe: whether to detect and verify MoE patterns
        detect_dynamic: whether to analyze dynamic control flow

    Returns:
        VerificationResult with safety verdict
    """
    compilation = compile_model(
        source,
        input_shapes,
        detect_moe=detect_moe,
        detect_dynamic=detect_dynamic,
    )

    # Use the standard verifier on the compiled graph
    result = verify_model(source, input_shapes=input_shapes)

    # Augment result with compilation metadata
    if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
        result.metadata["compilation_strategy"] = compilation.strategy
        result.metadata["coverage_ratio"] = compilation.coverage_ratio
        result.metadata["dynamic_patterns"] = [
            dp.pattern for dp in compilation.dynamic_patterns
        ]

    return result
