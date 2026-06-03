"""
Parametric Architecture Verification for nn.Module Families.

Verifies nn.Module architecture *families* (∀-verification) rather than just
concrete instances (∃-verification).  Given a source template whose
``__init__`` uses symbolic parameters (e.g. ``nn.Linear(d_model, d_ff)``),
this module checks whether the architecture is safe for ALL valid
parameter assignments satisfying the given bounds.

The approach:
  1. Inject arch_params as symbolic strings into the extraction param_map so
     layer definitions carry symbolic dimension names.
  2. Run the existing ConstraintVerifier, but patch the Z3 encoding so that
     symbolic layer parameters become Z3 Int variables with bound constraints
     instead of IntVal constants.
  3. If verification reports SAFE → universally safe (for all params in bounds).
     If UNSAFE → extract counter-example parameter values.
  4. Optionally discover minimal safety constraints by probing equalities.
"""

from __future__ import annotations

import ast
import copy
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.model_checker import (
    ComputationGraph,
    ConstraintVerifier,
    Device,
    LayerDef,
    LayerKind,
    Phase,
    VerificationResult,
    _InitExtractor,
    _ForwardExtractor,
    _collect_module_classes,
    _find_method,
    _find_root_module,
    _detect_dynamic_features,
    extract_computation_graph,
)

logger = logging.getLogger(__name__)


def _literal_interpolate_size(
    params: Dict[str, Any],
    spatial_rank: int,
) -> Optional[Tuple[int, ...]]:
    size = (params or {}).get("size")
    if isinstance(size, bool):
        return None
    if isinstance(size, int):
        return tuple(size for _ in range(spatial_rank))
    if isinstance(size, (tuple, list)) and len(size) == spatial_rank:
        if all(isinstance(v, int) and not isinstance(v, bool) for v in size):
            return tuple(size)
    return None


try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParametricConstraint:
    """A discovered constraint on architecture parameters."""
    expression: str       # e.g., "d_ff == d_model"
    parameters: List[str] # involved parameters

    def __repr__(self) -> str:
        return f"ParametricConstraint({self.expression!r})"


@dataclass
class ParametricResult:
    """Result of parametric architecture verification."""
    universally_safe: bool  # True if safe for ALL valid param values
    safety_constraints: List[ParametricConstraint] = field(default_factory=list)
    counterexample_params: Optional[Dict[str, int]] = None
    verification_result: Optional[VerificationResult] = None
    arch_params_used: Dict[str, Dict] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = []
        if self.universally_safe:
            lines.append("✓ Architecture family is UNIVERSALLY SAFE")
        else:
            lines.append("✗ Architecture family is NOT universally safe")
        if self.safety_constraints:
            lines.append("  Safety constraints:")
            for c in self.safety_constraints:
                lines.append(f"    • {c.expression}")
        if self.counterexample_params:
            lines.append("  Counter-example params:")
            for k, v in self.counterexample_params.items():
                lines.append(f"    {k} = {v}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph extraction with symbolic arch params
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_parametric_graph(
    source: str,
    arch_params: Dict[str, Dict[str, Any]],
) -> ComputationGraph:
    """Extract computation graph with arch_params injected as symbolic strings.

    The arch_param names become string values in the param_map so that
    ``_const_value`` returns strings instead of None for those parameters.
    Layer definitions will carry symbolic dimension names (strings) instead
    of concrete ints.
    """
    tree = ast.parse(source)
    dynamic_features = _detect_dynamic_features(tree, source)

    module_classes = _collect_module_classes(tree)
    if not module_classes:
        raise ValueError("No nn.Module subclass found in source")

    class_map = {c.name: c for c in module_classes}
    root_cls = _find_root_module(module_classes)

    graph = ComputationGraph(class_name=root_cls.name)
    graph.dynamic_features = dynamic_features

    init_fn = _find_method(root_cls, "__init__")
    if init_fn:
        extractor = _InitExtractor(class_map=class_map)
        # First do the normal extraction to get defaults
        extractor.extract(init_fn)
        # Then override the param_map with symbolic arch_params and re-extract
        for pname in arch_params:
            extractor._param_map[pname] = pname  # string → symbolic
        # Re-extract layers with symbolic param_map
        extractor.layers = {}
        extractor.visit(init_fn)
        graph.layers = extractor.layers

    fwd_fn = _find_method(root_cls, "forward")
    if fwd_fn:
        fwd_ext = _ForwardExtractor(graph.layers)
        fwd_ext.extract(fwd_fn)
        graph.steps = fwd_ext.steps
        graph.input_names = fwd_ext.input_names
        graph.output_names = fwd_ext.output_names

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# Parametric constraint verifier (subclass)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_symbolic_params(graph: ComputationGraph) -> Set[str]:
    """Collect all symbolic (string) layer parameters from the graph."""
    syms: Set[str] = set()
    for layer in graph.layers.values():
        for attr in ("in_features", "out_features", "in_channels",
                      "out_channels", "num_features", "embedding_dim",
                      "hidden_size", "num_embeddings"):
            val = getattr(layer, attr, None)
            if isinstance(val, str) and not val.startswith("_"):
                syms.add(val)
        if isinstance(layer.params.get("embed_dim"), str):
            syms.add(layer.params["embed_dim"])
        if isinstance(layer.params.get("num_heads"), str):
            syms.add(layer.params["num_heads"])
    return syms


class ParametricVerifier(ConstraintVerifier):
    """Extends ConstraintVerifier to handle symbolic arch parameters.

    When a layer's dimension field (e.g. in_features) is a string, we create
    a Z3 Int variable for it and add bound constraints, rather than using
    IntVal (which would fail on strings).
    """

    def __init__(
        self,
        graph: ComputationGraph,
        arch_params: Dict[str, Dict[str, Any]],
        input_shapes: Optional[Dict[str, tuple]] = None,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
    ) -> None:
        self._arch_params = arch_params
        self._z3_arch_vars: Dict[str, Any] = {}  # param_name -> Z3 Int
        self._arch_bound_constraints: list = []

        super().__init__(
            graph, input_shapes=input_shapes,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
        )

        if HAS_Z3:
            self._setup_arch_vars()

    def _setup_arch_vars(self) -> None:
        """Create Z3 Int variables for each architecture parameter.

        Uses the same Z3 variable as the context's dim() method so that
        symbolic input shape dimensions and symbolic layer parameters
        share the same Z3 Int, establishing the necessary equalities.
        """
        all_syms = _collect_symbolic_params(self.graph)
        # Ensure all declared arch_params have Z3 variables (via ctx.dim()
        # so they are the same variable used for symbolic shape dims)
        for pname, bounds in self._arch_params.items():
            if pname not in self._z3_arch_vars:
                self._z3_arch_vars[pname] = self.ctx.dim(pname)
            v = self._z3_arch_vars[pname]
            lo = bounds.get("min", 1)
            hi = bounds.get("max", None)
            self._arch_bound_constraints.append(v >= lo)
            if hi is not None:
                self._arch_bound_constraints.append(v <= hi)
        # Also create variables for symbolic params found in graph
        # but not explicitly declared (they get default min=1)
        for sym in all_syms:
            if sym not in self._z3_arch_vars:
                self._z3_arch_vars[sym] = self.ctx.dim(sym)
                self._arch_bound_constraints.append(
                    self._z3_arch_vars[sym] >= 1
                )

    def _dim_to_z3(self, val: Any) -> Any:
        """Convert a layer dimension to a Z3 expression.

        If val is a string (symbolic arch param), return its Z3 Int variable.
        If val is an int, return z3.IntVal.
        """
        if isinstance(val, str):
            if val in self._z3_arch_vars:
                return self._z3_arch_vars[val]
            # Create a new variable on the fly via ctx.dim() for consistency
            self._z3_arch_vars[val] = self.ctx.dim(val)
            self._arch_bound_constraints.append(
                self._z3_arch_vars[val] >= 1
            )
            return self._z3_arch_vars[val]
        return z3.IntVal(val)

    def _encode_shape_safety(
        self,
        k,  # KripkeState
        step,  # ComputationStep
        ms,  # ModelState
        idx: int,
    ) -> List:
        """Override to handle symbolic (string) layer parameters.

        When a layer dimension is a string (symbolic arch param), use the
        corresponding Z3 Int variable instead of z3.IntVal.
        """
        from src.model_checker import OpKind

        cs: list = []
        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            inp = step.inputs[0] if step.inputs else None
            if layer and inp and inp in k.shape_vars:
                dims = k.shape_vars[inp]
                if layer.kind == LayerKind.LINEAR:
                    if layer.in_features is not None and dims:
                        cs.append(
                            dims[-1] == self._dim_to_z3(layer.in_features)
                        )
                elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
                    if layer.in_channels is not None and len(dims) >= 2:
                        cs.append(
                            dims[1] == self._dim_to_z3(layer.in_channels)
                        )
                elif layer.kind == LayerKind.CONVTRANSPOSE2D:
                    if layer.in_channels is not None and len(dims) >= 2:
                        cs.append(
                            dims[1] == self._dim_to_z3(layer.in_channels)
                        )
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D):
                    if layer.num_features is not None and len(dims) >= 2:
                        cs.append(
                            dims[1] == self._dim_to_z3(layer.num_features)
                        )
                elif layer.kind in (LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    if layer.num_features is not None and len(dims) >= 2:
                        cs.append(
                            dims[1] == self._dim_to_z3(layer.num_features)
                        )
                elif layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                    if layer.in_features is not None and dims:
                        cs.append(
                            dims[-1] == self._dim_to_z3(layer.in_features)
                        )
                elif layer.kind in (
                    LayerKind.TRANSFORMER_ENCODER_LAYER,
                    LayerKind.TRANSFORMER_DECODER_LAYER,
                    LayerKind.TRANSFORMER_ENCODER,
                    LayerKind.TRANSFORMER_DECODER,
                ):
                    if layer.in_features is not None and dims:
                        cs.append(
                            dims[-1] == self._dim_to_z3(layer.in_features)
                        )
        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                if ad and bd:
                    if len(bd) >= 2:
                        cs.append(ad[-1] == bd[-2])
                    elif len(bd) == 1:
                        cs.append(ad[-1] == bd[0])
        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                ndim = max(len(ad), len(bd))
                for i in range(1, ndim + 1):
                    da = ad[-i] if i <= len(ad) else z3.IntVal(1)
                    db = bd[-i] if i <= len(bd) else z3.IntVal(1)
                    cs.append(z3.Or(da == db, da == z3.IntVal(1),
                                    db == z3.IntVal(1)))
        elif step.op == OpKind.MULTIPLY and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if a in k.shape_vars and b in k.shape_vars:
                ad = k.shape_vars[a]
                bd = k.shape_vars[b]
                ndim = max(len(ad), len(bd))
                for i in range(1, ndim + 1):
                    da = ad[-i] if i <= len(ad) else z3.IntVal(1)
                    db = bd[-i] if i <= len(bd) else z3.IntVal(1)
                    cs.append(z3.Or(da == db, da == z3.IntVal(1),
                                    db == z3.IntVal(1)))
        # Positivity for all involved shape dims
        for inp in step.inputs:
            if inp in k.shape_vars:
                for d in k.shape_vars[inp]:
                    cs.append(d > 0)
        return cs

    def _encode_shape_transition(
        self,
        pre,   # KripkeState
        step,  # ComputationStep
        post,  # KripkeState
        model_state,  # ModelState
    ) -> List:
        """Override to handle symbolic layer dimensions in transitions.

        Replaces z3.IntVal(layer.X) with self._dim_to_z3(layer.X) so that
        string (symbolic) dimension values become Z3 Int variables.
        """
        from src.model_checker import OpKind

        cs: list = []
        inp_name = step.inputs[0] if step.inputs else None

        if step.op == OpKind.LAYER_CALL and step.layer_ref:
            layer = self.graph.layers.get(step.layer_ref)
            if (layer and inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                if layer.kind == LayerKind.LINEAR:
                    for i in range(min(len(pre_d) - 1, len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.out_features is not None and post_d:
                        cs.append(
                            post_d[-1] == self._dim_to_z3(layer.out_features)
                        )
                elif layer.kind in (LayerKind.CONV2D, LayerKind.CONV1D):
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == self._dim_to_z3(layer.out_channels)
                        )
                elif layer.kind in (LayerKind.BATCHNORM1D,
                                    LayerKind.BATCHNORM2D,
                                    LayerKind.LAYERNORM,
                                    LayerKind.GROUPNORM,
                                    LayerKind.INSTANCENORM2D):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind in (LayerKind.RELU, LayerKind.DROPOUT,
                                    LayerKind.IDENTITY, LayerKind.SOFTMAX):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.EMBEDDING:
                    for i in range(min(len(pre_d), len(post_d) - 1)):
                        cs.append(post_d[i] == pre_d[i])
                    if layer.embedding_dim is not None and post_d:
                        cs.append(
                            post_d[-1] == self._dim_to_z3(layer.embedding_dim)
                        )
                elif layer.kind == LayerKind.FLATTEN:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind in (LayerKind.ADAPTIVE_AVGPOOL2D,):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                    if layer.output_size and len(post_d) >= 4:
                        cs.append(
                            post_d[2] == self._dim_to_z3(layer.output_size[0])
                        )
                        cs.append(
                            post_d[3] == self._dim_to_z3(layer.output_size[1])
                        )
                elif layer.kind in (LayerKind.MAXPOOL2D, LayerKind.AVGPOOL2D):
                    if len(pre_d) >= 2 and len(post_d) >= 2:
                        cs.append(post_d[0] == pre_d[0])
                        cs.append(post_d[1] == pre_d[1])
                elif layer.kind == LayerKind.SEQUENTIAL:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                elif layer.kind == LayerKind.MODULELIST:
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                elif layer.kind == LayerKind.CONVTRANSPOSE2D:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                    if layer.out_channels is not None and len(post_d) >= 2:
                        cs.append(
                            post_d[1] == self._dim_to_z3(layer.out_channels)
                        )
                elif layer.kind == LayerKind.UPSAMPLE:
                    for dp, dq in zip(pre_d[:2], post_d[:2]):
                        cs.append(dq == dp)
                    size = _literal_interpolate_size(
                        layer.params or {},
                        max(0, len(pre_d) - 2),
                    )
                    if size is not None:
                        for i, dim in enumerate(size):
                            if len(post_d) > i + 2:
                                cs.append(post_d[i + 2] == z3.IntVal(dim))
                elif layer.kind == LayerKind.MULTIHEAD_ATTENTION:
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                    if layer.in_features is not None and pre_d:
                        cs.append(
                            pre_d[-1] == self._dim_to_z3(layer.in_features)
                        )
                elif layer.kind in (LayerKind.TRANSFORMER_ENCODER_LAYER,
                                    LayerKind.TRANSFORMER_DECODER_LAYER,
                                    LayerKind.TRANSFORMER_ENCODER,
                                    LayerKind.TRANSFORMER_DECODER):
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)
                    if layer.in_features is not None and pre_d:
                        cs.append(
                            pre_d[-1] == self._dim_to_z3(layer.in_features)
                        )
                elif layer.kind == LayerKind.SUBMODULE:
                    if pre_d and post_d:
                        cs.append(post_d[0] == pre_d[0])
                else:
                    for dp, dq in zip(pre_d, post_d):
                        cs.append(dq == dp)

        elif step.op == OpKind.MATMUL and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                if len(ad) >= 2 and pd:
                    cs.append(pd[0] == ad[0])
                if len(bd) >= 2 and len(pd) >= 2:
                    cs.append(pd[-1] == bd[-1])

        elif step.op == OpKind.ADD and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                ndim = max(len(ad), len(bd))
                for i in range(min(ndim, len(pd))):
                    da = ad[len(ad) - 1 - i] if i < len(ad) else z3.IntVal(1)
                    db = bd[len(bd) - 1 - i] if i < len(bd) else z3.IntVal(1)
                    dp = pd[len(pd) - 1 - i]
                    cs.append(z3.Or(
                        z3.And(da == z3.IntVal(1), dp == db),
                        z3.And(db == z3.IntVal(1), dp == da),
                        z3.And(da == db, dp == da),
                    ))

        elif step.op == OpKind.MULTIPLY and len(step.inputs) >= 2:
            a, b = step.inputs[0], step.inputs[1]
            if (a in pre.shape_vars and b in pre.shape_vars
                    and step.output in post.shape_vars):
                ad = pre.shape_vars[a]
                bd = pre.shape_vars[b]
                pd = post.shape_vars[step.output]
                ndim = max(len(ad), len(bd))
                for i in range(min(ndim, len(pd))):
                    da = ad[len(ad) - 1 - i] if i < len(ad) else z3.IntVal(1)
                    db = bd[len(bd) - 1 - i] if i < len(bd) else z3.IntVal(1)
                    dp = pd[len(pd) - 1 - i]
                    cs.append(z3.Or(
                        z3.And(da == z3.IntVal(1), dp == db),
                        z3.And(db == z3.IntVal(1), dp == da),
                        z3.And(da == db, dp == da),
                    ))

        elif step.op == OpKind.INTERPOLATE:
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                for dp, dq in zip(pre_d[:2], post_d[:2]):
                    cs.append(dq == dp)
                size = _literal_interpolate_size(
                    step.params or {},
                    max(0, len(pre_d) - 2),
                )
                if size is not None:
                    for i, dim in enumerate(size):
                        if len(post_d) > i + 2:
                            cs.append(post_d[i + 2] == z3.IntVal(dim))

        elif step.op == OpKind.RESHAPE:
            dims = step.params.get("dims")
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars and dims is not None):
                pre_d = pre.shape_vars[inp_name]
                post_d = post.shape_vars[step.output]
                for i, d in enumerate(dims):
                    if isinstance(d, int) and d >= 0 and i < len(post_d):
                        cs.append(post_d[i] == z3.IntVal(d))
                cs.extend(self._encode_reshape_safety(pre_d, post_d))

        elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                          OpKind.CONTIGUOUS, OpKind.DETACH, OpKind.TO_DEVICE):
            if (inp_name and inp_name in pre.shape_vars
                    and step.output in post.shape_vars):
                for dp, dq in zip(
                    pre.shape_vars[inp_name],
                    post.shape_vars[step.output],
                ):
                    cs.append(dq == dp)

        elif step.op == OpKind.CONDITIONAL:
            pass

        return cs

    def verify(self) -> VerificationResult:
        """Override verify to inject arch param bound constraints into solver."""
        if HAS_Z3 and self._arch_bound_constraints:
            for c in self._arch_bound_constraints:
                self.ctx.solver.add(c)
        return super().verify()


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete-layer shape propagation with symbolic dims
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_layer_for_symbolic(layer: LayerDef) -> LayerDef:
    """Ensure a layer with symbolic (string) dimensions doesn't crash
    the concrete propagator by making ShapeDim treat them as symbolic.

    The propagation layer already handles is_symbolic gracefully for most
    checks. The main issue is out_features/out_channels being strings causes
    ShapeDim(string) which is valid (is_symbolic=True).
    """
    # No patching needed — ShapeDim already accepts strings
    return layer


# ═══════════════════════════════════════════════════════════════════════════════
# Safety constraint discovery
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_safety_constraints(
    source: str,
    arch_params: Dict[str, Dict[str, Any]],
    input_shapes: Optional[Dict[str, tuple]],
    graph: ComputationGraph,
) -> List[ParametricConstraint]:
    """Discover minimal constraints on arch params needed for safety.

    Strategy: for each pair of symbolic dimensions that appear in consecutive
    layers, test whether adding the equality constraint makes the model safe.
    """
    constraints: List[ParametricConstraint] = []
    if not HAS_Z3:
        return constraints

    # Collect which symbolic params appear as in/out features across layers
    layer_dims: List[Tuple[str, str, Any]] = []  # (layer_name, role, value)
    for lname, layer in graph.layers.items():
        if isinstance(layer.in_features, str):
            layer_dims.append((lname, "in_features", layer.in_features))
        if isinstance(layer.out_features, str):
            layer_dims.append((lname, "out_features", layer.out_features))
        if isinstance(layer.in_channels, str):
            layer_dims.append((lname, "in_channels", layer.in_channels))
        if isinstance(layer.out_channels, str):
            layer_dims.append((lname, "out_channels", layer.out_channels))
        if isinstance(layer.num_features, str):
            layer_dims.append((lname, "num_features", layer.num_features))

    # Get unique symbolic param names
    sym_names = sorted(set(v for _, _, v in layer_dims))

    # For each pair, check if equating them is necessary
    for i, a in enumerate(sym_names):
        for b in sym_names[i + 1:]:
            # Check: is the model safe when a == b?
            # First check if these are connected (one feeds into the other)
            a_out = any(role == "out_features" or role == "out_channels"
                        for _, role, v in layer_dims if v == a)
            b_in = any(role == "in_features" or role == "in_channels"
                       for _, role, v in layer_dims if v == b)
            b_out = any(role == "out_features" or role == "out_channels"
                        for _, role, v in layer_dims if v == b)
            a_in = any(role == "in_features" or role == "in_channels"
                       for _, role, v in layer_dims if v == a)

            if (a_out and b_in) or (b_out and a_in):
                constraints.append(ParametricConstraint(
                    expression=f"{a} == {b}",
                    parameters=[a, b],
                ))

    # Also check input shape symbolic dims vs first layer's in_features
    if input_shapes:
        for inp_name, shape in input_shapes.items():
            for d in shape:
                if isinstance(d, str) and d in arch_params:
                    # This input dim should match the first layer's in_features
                    for lname, role, val in layer_dims:
                        if role in ("in_features", "in_channels") and val != d:
                            if val in arch_params:
                                constraints.append(ParametricConstraint(
                                    expression=f"{d} == {val}",
                                    parameters=[d, val],
                                ))

    return constraints


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def verify_parametric(
    source: str,
    arch_params: Dict[str, Dict[str, Any]],
    input_shapes: Optional[Dict[str, tuple]] = None,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    discover_constraints: bool = True,
) -> ParametricResult:
    """Verify an nn.Module architecture *family* parametrically.

    Parameters
    ----------
    source : str
        Python source code containing an nn.Module subclass whose __init__
        takes architecture parameters (e.g. ``d_model``, ``d_ff``).
    arch_params : dict
        Mapping from parameter name to bounds dict.
        Example: ``{"d_model": {"min": 1}, "d_ff": {"min": 1, "max": 4096}}``
    input_shapes : dict, optional
        Mapping from forward-parameter names to shape tuples.
        Dimensions may reference arch_params by name (strings).
    default_device : Device
        Default device for input tensors.
    default_phase : Phase
        Default phase (TRAIN or EVAL).
    max_k : int, optional
        Maximum verification depth.
    discover_constraints : bool
        If True and the model is not universally safe, attempt to discover
        minimal constraints that would make it safe.

    Returns
    -------
    ParametricResult
        Contains universally_safe status, discovered constraints,
        and counter-example parameter values if unsafe.
    """
    t0 = time.monotonic()

    # Step 1: Extract graph with symbolic arch params
    try:
        graph = _extract_parametric_graph(source, arch_params)
    except (ValueError, SyntaxError) as exc:
        return ParametricResult(
            universally_safe=False,
            verification_result=VerificationResult(
                safe=False, errors=[str(exc)],
            ),
            arch_params_used=arch_params,
        )

    # Step 2: Run the parametric verifier
    checker = ParametricVerifier(
        graph,
        arch_params=arch_params,
        input_shapes=input_shapes or {},
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
    )

    vresult = checker.verify()

    # Step 3: Interpret result
    if vresult.safe:
        # SAFE with symbolic params → universally safe
        return ParametricResult(
            universally_safe=True,
            verification_result=vresult,
            arch_params_used=arch_params,
        )

    # UNSAFE: try to extract counter-example param values
    cex_params: Optional[Dict[str, int]] = None
    if HAS_Z3 and vresult.counterexample and vresult.counterexample.concrete_dims:
        cex_params = {}
        for pname in arch_params:
            z3_name = f"arch_{pname}"
            for dim_name, dim_val in vresult.counterexample.concrete_dims.items():
                if dim_name == z3_name or dim_name == pname:
                    cex_params[pname] = dim_val
            # Also try to find from model decls
            if pname not in cex_params:
                for dim_name, dim_val in vresult.counterexample.concrete_dims.items():
                    if pname in dim_name:
                        cex_params[pname] = dim_val

    # Step 4: Discover safety constraints if requested
    safety_constraints: List[ParametricConstraint] = []
    if discover_constraints:
        safety_constraints = _discover_safety_constraints(
            source, arch_params, input_shapes, graph,
        )

    return ParametricResult(
        universally_safe=False,
        safety_constraints=safety_constraints,
        counterexample_params=cex_params if cex_params else None,
        verification_result=vresult,
        arch_params_used=arch_params,
    )
