"""
IC3/PDR Engine for Unbounded Tensor Shape Verification.

Implements the IC3 (Incremental Construction of Inductive Clauses for
Indubitable Correctness) / PDR (Property Directed Reachability) algorithm
adapted for tensor shape verification.

Key idea:
  - States are symbolic shape assignments: {tensor_name → (d1, d2, …, dn)}
  - Transitions model shape propagation through one nn.Module layer
  - Safety property: shape compatibility at each layer
  - "Unbounded" means: verify for ALL values of symbolic dimensions (batch_size,
    seq_len, etc.) rather than a fixed set of concrete values

The algorithm maintains a sequence of frames F_0, F_1, …, F_k where each
frame over-approximates the set of reachable states at depth ≤ i.  It uses
UNSAT-core–based generalization (via Craig interpolation when available)
to efficiently block sets of bad states and detect inductive invariants.

Usage::

    from src.ic3_pdr import ic3_verify, IC3Result

    result = ic3_verify(
        model_source=\"\"\"
        import torch.nn as nn
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
            def forward(self, x):
                return self.fc2(self.fc1(x))
        \"\"\",
        symbolic_dims={"batch": "batch_size"},
    )
    assert result.safe
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Z3 import
# ---------------------------------------------------------------------------
try:
    import z3

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
from src.model_checker import (
    ComputationGraph,
    ComputationStep,
    ConstraintVerifier,
    Device,
    LayerDef,
    LayerKind,
    ModelState,
    OpKind,
    Phase,
    VerificationResult,
    extract_computation_graph,
)
from src.tensor_shapes import TensorShape, ShapeDim

try:
    from src.craig_interpolation import (
        DimMapping,
        _compute_cvc5_interpolant,
        _compute_simulated_interpolant,
    )

    HAS_INTERPOLATION = True
except ImportError:
    HAS_INTERPOLATION = False

try:
    import cvc5 as _cvc5_mod
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

try:
    from src.unsat_core_cegar import IncrementalCEGARSolver

    HAS_UNSAT_CORE = True
except ImportError:
    HAS_UNSAT_CORE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class IC3Result:
    """Result of IC3/PDR verification.

    Attributes
    ----------
    safe : bool
        True if the model is safe for ALL values of symbolic dimensions.
    invariant : str or None
        Human-readable inductive invariant (when safe).
    counterexample_depth : int or None
        Depth at which a counterexample was found (when unsafe).
    frames_computed : int
        Number of frames computed before termination.
    verification_time_ms : float
        Wall-clock time in milliseconds.
    symbolic_dims : dict
        Symbolic dimensions used during verification.
    num_blocked_cubes : int
        Total number of cubes blocked across all frames.
    z3_queries : int
        Total number of Z3 solver queries.
    invariant_clauses : list of str
        Individual clauses forming the inductive invariant.
    counterexample_trace : list of dict or None
        Shape assignments at each step of the counterexample trace.
    """

    safe: bool
    invariant: Optional[str] = None
    counterexample_depth: Optional[int] = None
    frames_computed: int = 0
    verification_time_ms: float = 0.0
    symbolic_dims: Dict[str, str] = field(default_factory=dict)
    num_blocked_cubes: int = 0
    z3_queries: int = 0
    invariant_clauses: List[str] = field(default_factory=list)
    counterexample_trace: Optional[List[Dict[str, Any]]] = None
    frame_sequence: List[Dict[str, Any]] = field(default_factory=list)
    soundness_warning: Optional[str] = None


class IC3Status(Enum):
    """Internal status for IC3/PDR algorithm."""

    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Cube / Clause representation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ShapeCube:
    """A cube (conjunction of literals) over shape dimensions.

    Each literal is a constraint on a symbolic dimension variable,
    e.g., ``dim_0 == 10``, ``dim_1 >= 1``.
    """

    literals: FrozenSet[str]
    z3_expr: Any = None  # z3.ExprRef, stored as Any for optional z3

    def __repr__(self) -> str:
        return f"ShapeCube({', '.join(sorted(self.literals))})"

    def negate(self) -> "ShapeClause":
        """Negate this cube to get a clause (for blocking)."""
        if self.z3_expr is not None and HAS_Z3:
            return ShapeClause(
                literals=self.literals,
                z3_expr=z3.Not(self.z3_expr),
            )
        return ShapeClause(literals=self.literals, z3_expr=None)


@dataclass(frozen=True)
class ShapeClause:
    """A clause (disjunction of literals) used to block cubes in frames."""

    literals: FrozenSet[str]
    z3_expr: Any = None

    def __repr__(self) -> str:
        return f"ShapeClause({', '.join(sorted(self.literals))})"


# ═══════════════════════════════════════════════════════════════════════════════
# Proof obligation (for the IC3 queue)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProofObligation:
    """A proof obligation: a cube that must be blocked at a given frame level."""

    cube: ShapeCube
    frame_level: int
    depth: int = 0  # recursion depth for counterexample trace

    def __lt__(self, other: "ProofObligation") -> bool:
        return (self.frame_level, self.depth) < (other.frame_level, other.depth)


# ═══════════════════════════════════════════════════════════════════════════════
# Shape transition system
# ═══════════════════════════════════════════════════════════════════════════════


class ShapeTransitionSystem:
    """Encodes the tensor shape propagation as a transition system.

    Uses SSA (Static Single Assignment) naming: each computation step
    produces uniquely-named output variables (step-indexed) so that
    reassignment of the same tensor name in Python does not cause
    conflicting constraints.

    States: symbolic shape assignments (Z3 Int variables for each dimension)
    Transitions: shape propagation through one computation step
    Safety: shape compatibility at each step
    """

    def __init__(
        self,
        graph: ComputationGraph,
        input_shapes: Dict[str, tuple],
        symbolic_dims: Dict[str, str],
        solver_timeout_ms: int = 5000,
    ) -> None:
        if not HAS_Z3:
            raise RuntimeError("Z3 is required for IC3/PDR verification")

        self.graph = graph
        self.input_shapes = input_shapes
        self.symbolic_dims = symbolic_dims
        self.solver_timeout_ms = solver_timeout_ms

        # Z3 variables for shape dimensions
        self._dim_vars: Dict[str, z3.ArithRef] = {}
        self._z3_queries = 0

        # SSA version map: tensor_name -> current version index
        self._ssa_version: Dict[str, int] = {}
        # Maps (tensor_name, version) -> ndim
        self._ssa_ndim: Dict[Tuple[str, int], int] = {}

        # Build the transition system
        self._init_constraints: List[z3.ExprRef] = []
        self._transition_constraints: List[List[z3.ExprRef]] = []
        self._safety_constraints: List[List[z3.ExprRef]] = []
        self._bad_constraints: List[z3.ExprRef] = []

        self._build()

    def dim(self, name: str) -> z3.ArithRef:
        """Get or create a Z3 Int variable for a dimension."""
        if name not in self._dim_vars:
            self._dim_vars[name] = z3.Int(name)
        return self._dim_vars[name]

    def _ssa_input_name(self, tensor_name: str, axis: int) -> str:
        """Get SSA variable name for reading a tensor's dimension."""
        ver = self._ssa_version.get(tensor_name, 0)
        return f"sh_{tensor_name}_v{ver}_d{axis}"

    def _ssa_output_name(self, tensor_name: str, axis: int, step_idx: int) -> str:
        """Get SSA variable name for writing a tensor's dimension."""
        ver = self._ssa_version.get(tensor_name, 0) + 1
        return f"sh_{tensor_name}_v{ver}_d{axis}"

    def _bump_ssa(self, tensor_name: str, ndim: int) -> int:
        """Increment SSA version for a tensor and record its ndim."""
        ver = self._ssa_version.get(tensor_name, 0) + 1
        self._ssa_version[tensor_name] = ver
        self._ssa_ndim[(tensor_name, ver)] = ndim
        return ver

    def _snapshot_input_ver(self, tensor_name: str) -> int:
        """Snapshot the current SSA version for reading (before bumping)."""
        return self._ssa_version.get(tensor_name, 0)

    def _get_inp_var_at(self, tensor_name: str, axis: int, ver: int) -> z3.ArithRef:
        """Get Z3 var for tensor at a specific SSA version."""
        return self.dim(f"sh_{tensor_name}_v{ver}_d{axis}")

    def _current_ndim(self, tensor_name: str) -> int:
        """Get the current ndim for a tensor based on SSA tracking."""
        ver = self._ssa_version.get(tensor_name, 0)
        if (tensor_name, ver) in self._ssa_ndim:
            return self._ssa_ndim[(tensor_name, ver)]
        if tensor_name in self.input_shapes:
            return len(self.input_shapes[tensor_name])
        return 2

    def _build(self) -> None:
        """Build init, transition, and safety constraints from the graph."""
        # --- Initial state constraints ---
        for inp_name, shape_tuple in self.input_shapes.items():
            ver = 0
            self._ssa_version[inp_name] = ver
            self._ssa_ndim[(inp_name, ver)] = len(shape_tuple)
            for axis, dim_val in enumerate(shape_tuple):
                var_name = f"sh_{inp_name}_v{ver}_d{axis}"
                v = self.dim(var_name)
                if isinstance(dim_val, int):
                    self._init_constraints.append(v == dim_val)
                elif isinstance(dim_val, str):
                    sym_var = self.dim(dim_val)
                    self._init_constraints.append(v == sym_var)
                    self._init_constraints.append(sym_var > 0)

        # --- Transition + safety constraints per step ---
        for step_idx, step in enumerate(self.graph.steps):
            step_trans: List[z3.ExprRef] = []
            step_safety: List[z3.ExprRef] = []

            if step.op == OpKind.LAYER_CALL and step.layer_ref:
                layer = self.graph.layers.get(step.layer_ref)
                if layer:
                    t, s = self._encode_layer_step(step, layer, step_idx)
                    step_trans.extend(t)
                    step_safety.extend(s)
            elif step.op in (OpKind.ACTIVATION, OpKind.DROPOUT, OpKind.SOFTMAX,
                             OpKind.DETACH, OpKind.CONTIGUOUS):
                t = self._encode_identity_step(step, step_idx)
                step_trans.extend(t)
            elif step.op == OpKind.ADD or step.op == OpKind.MULTIPLY:
                t, s = self._encode_elementwise_step(step, step_idx)
                step_trans.extend(t)
                step_safety.extend(s)
            elif step.op == OpKind.MATMUL:
                t, s = self._encode_matmul_step(step, step_idx)
                step_trans.extend(t)
                step_safety.extend(s)
            elif step.op == OpKind.RESHAPE:
                t = self._encode_reshape_step(step, step_idx)
                step_trans.extend(t)
            elif step.op == OpKind.FLATTEN:
                t = self._encode_flatten_step(step, step_idx)
                step_trans.extend(t)
            elif step.op == OpKind.CAT:
                t, s = self._encode_cat_step(step, step_idx)
                step_trans.extend(t)
                step_safety.extend(s)
            elif step.op == OpKind.TRANSPOSE:
                t = self._encode_transpose_step(step, step_idx)
                step_trans.extend(t)
            elif step.op == OpKind.RETURN:
                pass
            else:
                t = self._encode_identity_step(step, step_idx)
                step_trans.extend(t)

            self._transition_constraints.append(step_trans)
            self._safety_constraints.append(step_safety)

        # --- Bad state: negation of all safety ---
        all_safety = []
        for sc in self._safety_constraints:
            all_safety.extend(sc)
        if all_safety:
            self._bad_constraints = [z3.Not(z3.And(*all_safety))]
        else:
            self._bad_constraints = [z3.BoolVal(False)]

    def _encode_layer_step(
        self, step: ComputationStep, layer: LayerDef, step_idx: int
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        """Encode shape propagation through a layer using SSA."""
        trans: List[z3.ExprRef] = []
        safety: List[z3.ExprRef] = []

        if not step.inputs:
            return trans, safety

        inp_name = step.inputs[0]
        out_name = step.output
        inp_ndim = self._current_ndim(inp_name)
        # Snapshot input version BEFORE bumping (critical when inp == out)
        inp_ver = self._snapshot_input_ver(inp_name)

        if layer.kind == LayerKind.LINEAR:
            in_f = layer.in_features
            out_f = layer.out_features
            if in_f is not None and out_f is not None:
                inp_last = self._get_inp_var_at(inp_name, inp_ndim - 1, inp_ver)
                in_f_val = self._to_z3(in_f)
                safety.append(inp_last == in_f_val)

                out_ndim = inp_ndim
                out_ver = self._bump_ssa(out_name, out_ndim)
                for d in range(inp_ndim - 1):
                    inp_d = self._get_inp_var_at(inp_name, d, inp_ver)
                    out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
                    trans.append(out_d == inp_d)
                out_last = self.dim(f"sh_{out_name}_v{out_ver}_d{inp_ndim - 1}")
                out_f_val = self._to_z3(out_f)
                trans.append(out_last == out_f_val)
            else:
                self._bump_ssa(out_name, inp_ndim)

        elif layer.kind == LayerKind.CONV2D:
            in_c = layer.in_channels
            out_c = layer.out_channels
            if in_c is not None and out_c is not None:
                inp_c = self._get_inp_var_at(inp_name, 1, inp_ver)
                in_c_val = self._to_z3(in_c)
                safety.append(inp_c == in_c_val)

                out_ver = self._bump_ssa(out_name, 4)
                batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
                batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
                trans.append(batch_out == batch_in)

                out_c_var = self.dim(f"sh_{out_name}_v{out_ver}_d1")
                out_c_val = self._to_z3(out_c)
                trans.append(out_c_var == out_c_val)

                h_in = self._get_inp_var_at(inp_name, 2, inp_ver)
                w_in = self._get_inp_var_at(inp_name, 3, inp_ver)
                h_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
                w_out = self.dim(f"sh_{out_name}_v{out_ver}_d3")

                ks = layer.kernel_size or (3, 3)
                stride = layer.params.get("stride", (1, 1))
                padding = layer.params.get("padding", (0, 0))
                if isinstance(ks, int):
                    ks = (ks, ks)
                if isinstance(stride, int):
                    stride = (stride, stride)
                if isinstance(padding, int):
                    padding = (padding, padding)

                # Use Z3 integer division (not Python float division)
                # for convolution output dimensions (fixes reviewer bug #3).
                if stride[0] != 1:
                    trans.append(
                        h_out == (h_in + 2 * padding[0] - ks[0]) / stride[0] + 1
                    )
                else:
                    trans.append(
                        h_out == h_in + 2 * padding[0] - ks[0] + 1
                    )
                if stride[1] != 1:
                    trans.append(
                        w_out == (w_in + 2 * padding[1] - ks[1]) / stride[1] + 1
                    )
                else:
                    trans.append(
                        w_out == w_in + 2 * padding[1] - ks[1] + 1
                    )
                safety.append(h_in + 2 * padding[0] >= ks[0])
                safety.append(w_in + 2 * padding[1] >= ks[1])
            else:
                self._bump_ssa(out_name, 4)

        elif layer.kind == LayerKind.CONV1D:
            in_c = layer.in_channels
            out_c = layer.out_channels
            if in_c is not None and out_c is not None:
                inp_c = self._get_inp_var_at(inp_name, 1, inp_ver)
                in_c_val = self._to_z3(in_c)
                safety.append(inp_c == in_c_val)

                out_ver = self._bump_ssa(out_name, 3)
                batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
                batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
                trans.append(batch_out == batch_in)

                out_c_var = self.dim(f"sh_{out_name}_v{out_ver}_d1")
                out_c_val = self._to_z3(out_c)
                trans.append(out_c_var == out_c_val)

                l_in = self._get_inp_var_at(inp_name, 2, inp_ver)
                l_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
                ks = layer.kernel_size or (3,)
                if isinstance(ks, int):
                    ks = (ks,)
                stride = layer.params.get("stride", (1,))
                padding = layer.params.get("padding", (0,))
                if isinstance(stride, int):
                    stride = (stride,)
                if isinstance(padding, int):
                    padding = (padding,)
                # Use Z3 integer division for Conv1d output dimension.
                if stride[0] != 1:
                    trans.append(
                        l_out == (l_in + 2 * padding[0] - ks[0]) / stride[0] + 1
                    )
                else:
                    trans.append(
                        l_out == l_in + 2 * padding[0] - ks[0] + 1
                    )
                safety.append(l_in + 2 * padding[0] >= ks[0])
            else:
                self._bump_ssa(out_name, 3)

        elif layer.kind in (LayerKind.BATCHNORM1D, LayerKind.BATCHNORM2D,
                            LayerKind.LAYERNORM, LayerKind.GROUPNORM,
                            LayerKind.INSTANCENORM2D, LayerKind.DROPOUT,
                            LayerKind.RELU, LayerKind.IDENTITY):
            t = self._encode_identity_step(step, step_idx)
            trans.extend(t)

        elif layer.kind == LayerKind.MAXPOOL2D or layer.kind == LayerKind.AVGPOOL2D:
            out_ver = self._bump_ssa(out_name, 4)
            batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
            batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
            trans.append(batch_out == batch_in)
            c_in = self._get_inp_var_at(inp_name, 1, inp_ver)
            c_out = self.dim(f"sh_{out_name}_v{out_ver}_d1")
            trans.append(c_out == c_in)

            h_in = self._get_inp_var_at(inp_name, 2, inp_ver)
            w_in = self._get_inp_var_at(inp_name, 3, inp_ver)
            h_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
            w_out = self.dim(f"sh_{out_name}_v{out_ver}_d3")
            ks = layer.kernel_size or (2, 2)
            if isinstance(ks, int):
                ks = (ks, ks)
            stride = layer.params.get("stride", ks)
            if isinstance(stride, int):
                stride = (stride, stride)
            padding = layer.params.get("padding", (0, 0))
            if isinstance(padding, int):
                padding = (padding, padding)
            trans.append(
                h_out == (h_in + 2 * padding[0] - ks[0]) / stride[0] + 1
            )
            trans.append(
                w_out == (w_in + 2 * padding[1] - ks[1]) / stride[1] + 1
            )
            safety.append(h_in + 2 * padding[0] >= ks[0])
            safety.append(w_in + 2 * padding[1] >= ks[1])

        elif layer.kind == LayerKind.ADAPTIVE_AVGPOOL2D:
            out_ver = self._bump_ssa(out_name, 4)
            batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
            batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
            trans.append(batch_out == batch_in)
            c_in = self._get_inp_var_at(inp_name, 1, inp_ver)
            c_out = self.dim(f"sh_{out_name}_v{out_ver}_d1")
            trans.append(c_out == c_in)
            if layer.output_size:
                oh, ow = layer.output_size[0], layer.output_size[1] if len(layer.output_size) > 1 else layer.output_size[0]
                h_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
                w_out = self.dim(f"sh_{out_name}_v{out_ver}_d3")
                trans.append(h_out == oh)
                trans.append(w_out == ow)

        elif layer.kind == LayerKind.FLATTEN:
            t = self._encode_flatten_step(step, step_idx)
            trans.extend(t)

        elif layer.kind == LayerKind.EMBEDDING:
            if layer.embedding_dim is not None:
                out_ver = self._bump_ssa(out_name, 3)
                batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
                batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
                trans.append(batch_out == batch_in)
                seq_in = self._get_inp_var_at(inp_name, 1, inp_ver)
                seq_out = self.dim(f"sh_{out_name}_v{out_ver}_d1")
                trans.append(seq_out == seq_in)
                embed_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
                embed_val = self._to_z3(layer.embedding_dim)
                trans.append(embed_out == embed_val)
            else:
                self._bump_ssa(out_name, self._current_ndim(inp_name))

        elif layer.kind == LayerKind.LSTM:
            if layer.hidden_size is not None:
                out_ver = self._bump_ssa(out_name, 3)
                batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
                batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
                trans.append(batch_out == batch_in)
                seq_in = self._get_inp_var_at(inp_name, 1, inp_ver)
                seq_out = self.dim(f"sh_{out_name}_v{out_ver}_d1")
                trans.append(seq_out == seq_in)
                h_out = self.dim(f"sh_{out_name}_v{out_ver}_d2")
                hs = self._to_z3(layer.hidden_size)
                mult = 2 if layer.bidirectional else 1
                trans.append(h_out == hs * mult)
            else:
                self._bump_ssa(out_name, self._current_ndim(inp_name))

        else:
            t = self._encode_identity_step(step, step_idx)
            trans.extend(t)

        return trans, safety

    def _encode_identity_step(
        self, step: ComputationStep, step_idx: int
    ) -> List[z3.ExprRef]:
        """Encode a shape-preserving step using SSA."""
        trans: List[z3.ExprRef] = []
        if not step.inputs:
            return trans
        inp_name = step.inputs[0]
        out_name = step.output
        ndim = self._current_ndim(inp_name)
        inp_ver = self._snapshot_input_ver(inp_name)
        out_ver = self._bump_ssa(out_name, ndim)
        for d in range(ndim):
            inp_d = self._get_inp_var_at(inp_name, d, inp_ver)
            out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
            trans.append(out_d == inp_d)
        return trans

    def _encode_elementwise_step(
        self, step: ComputationStep, step_idx: int
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        """Encode element-wise operation with broadcasting using SSA."""
        trans: List[z3.ExprRef] = []
        safety: List[z3.ExprRef] = []
        if len(step.inputs) < 2:
            return self._encode_identity_step(step, step_idx), safety

        a_name, b_name = step.inputs[0], step.inputs[1]
        out_name = step.output
        ndim_a = self._current_ndim(a_name)
        ndim_b = self._current_ndim(b_name)
        ndim_out = max(ndim_a, ndim_b)
        a_ver = self._snapshot_input_ver(a_name)
        b_ver = self._snapshot_input_ver(b_name)

        out_ver = self._bump_ssa(out_name, ndim_out)
        for d in range(ndim_out):
            a_d = self._get_inp_var_at(a_name, d, a_ver) if d < ndim_a else z3.IntVal(1)
            b_d = self._get_inp_var_at(b_name, d, b_ver) if d < ndim_b else z3.IntVal(1)
            out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
            safety.append(z3.Or(a_d == b_d, a_d == 1, b_d == 1))
            trans.append(out_d == z3.If(a_d >= b_d, a_d, b_d))
        return trans, safety

    def _encode_matmul_step(
        self, step: ComputationStep, step_idx: int
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        """Encode matrix multiplication using SSA."""
        trans: List[z3.ExprRef] = []
        safety: List[z3.ExprRef] = []
        if len(step.inputs) < 2:
            return trans, safety

        a_name, b_name = step.inputs[0], step.inputs[1]
        out_name = step.output
        a_ver = self._snapshot_input_ver(a_name)
        b_ver = self._snapshot_input_ver(b_name)

        a_d0 = self._get_inp_var_at(a_name, 0, a_ver)
        a_d1 = self._get_inp_var_at(a_name, 1, a_ver)
        b_d0 = self._get_inp_var_at(b_name, 0, b_ver)
        b_d1 = self._get_inp_var_at(b_name, 1, b_ver)

        safety.append(a_d1 == b_d0)

        out_ver = self._bump_ssa(out_name, 2)
        out_d0 = self.dim(f"sh_{out_name}_v{out_ver}_d0")
        out_d1 = self.dim(f"sh_{out_name}_v{out_ver}_d1")
        trans.append(out_d0 == a_d0)
        trans.append(out_d1 == b_d1)
        return trans, safety

    def _encode_reshape_step(
        self, step: ComputationStep, step_idx: int
    ) -> List[z3.ExprRef]:
        """Encode reshape operation using SSA."""
        trans: List[z3.ExprRef] = []
        target_shape = step.params.get("shape", step.params.get("dims", ()))
        out_name = step.output
        if not target_shape:
            out_ver = self._bump_ssa(out_name, 2)
            return trans
        out_ver = self._bump_ssa(out_name, len(target_shape))
        for d, dim_val in enumerate(target_shape):
            out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
            if isinstance(dim_val, int) and dim_val > 0:
                trans.append(out_d == dim_val)
            elif isinstance(dim_val, str):
                sym = self.dim(dim_val)
                trans.append(out_d == sym)
        return trans

    def _encode_flatten_step(
        self, step: ComputationStep, step_idx: int
    ) -> List[z3.ExprRef]:
        """Encode flatten operation using SSA."""
        trans: List[z3.ExprRef] = []
        if not step.inputs:
            return trans
        inp_name = step.inputs[0]
        out_name = step.output
        inp_ver = self._snapshot_input_ver(inp_name)
        out_ver = self._bump_ssa(out_name, 2)
        batch_in = self._get_inp_var_at(inp_name, 0, inp_ver)
        batch_out = self.dim(f"sh_{out_name}_v{out_ver}_d0")
        trans.append(batch_out == batch_in)
        flat_out = self.dim(f"sh_{out_name}_v{out_ver}_d1")
        trans.append(flat_out > 0)
        return trans

    def _encode_cat_step(
        self, step: ComputationStep, step_idx: int
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        """Encode concatenation using SSA."""
        trans: List[z3.ExprRef] = []
        safety: List[z3.ExprRef] = []
        if len(step.inputs) < 2:
            return trans, safety
        cat_dim = step.params.get("dim", 0)
        a_name = step.inputs[0]
        out_name = step.output
        ndim = self._current_ndim(a_name)
        # Snapshot all input versions before bumping
        inp_vers = {n: self._snapshot_input_ver(n) for n in step.inputs}
        out_ver = self._bump_ssa(out_name, ndim)

        for d in range(ndim):
            if d == cat_dim:
                total = z3.IntVal(0)
                for inp in step.inputs:
                    total = total + self._get_inp_var_at(inp, d, inp_vers[inp])
                out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
                trans.append(out_d == total)
            else:
                first_d = self._get_inp_var_at(a_name, d, inp_vers[a_name])
                for inp in step.inputs[1:]:
                    inp_d = self._get_inp_var_at(inp, d, inp_vers[inp])
                    safety.append(inp_d == first_d)
                out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
                trans.append(out_d == first_d)
        return trans, safety

    def _encode_transpose_step(
        self, step: ComputationStep, step_idx: int
    ) -> List[z3.ExprRef]:
        """Encode transpose using SSA."""
        trans: List[z3.ExprRef] = []
        if not step.inputs:
            return trans
        inp_name = step.inputs[0]
        out_name = step.output
        ndim = self._current_ndim(inp_name)
        dim0 = step.params.get("dim0", 0)
        dim1 = step.params.get("dim1", 1)
        inp_ver = self._snapshot_input_ver(inp_name)
        out_ver = self._bump_ssa(out_name, ndim)
        for d in range(ndim):
            inp_d_idx = d
            if d == dim0:
                inp_d_idx = dim1
            elif d == dim1:
                inp_d_idx = dim0
            inp_d = self._get_inp_var_at(inp_name, inp_d_idx, inp_ver)
            out_d = self.dim(f"sh_{out_name}_v{out_ver}_d{d}")
            trans.append(out_d == inp_d)
        return trans

    def _to_z3(self, val: Any) -> z3.ArithRef:
        """Convert a value to Z3: int -> IntVal, str -> Int(name)."""
        if isinstance(val, int):
            return z3.IntVal(val)
        if isinstance(val, str):
            return self.dim(val)
        return z3.IntVal(int(val))

    def get_init_constraints(self) -> List[z3.ExprRef]:
        """Return initial state constraints."""
        return list(self._init_constraints)

    def get_transition_constraints(self, step_idx: int) -> List[z3.ExprRef]:
        """Return transition constraints for a given step."""
        if step_idx < len(self._transition_constraints):
            return list(self._transition_constraints[step_idx])
        return []

    def get_safety_constraints(self, step_idx: int) -> List[z3.ExprRef]:
        """Return safety constraints for a given step."""
        if step_idx < len(self._safety_constraints):
            return list(self._safety_constraints[step_idx])
        return []

    def get_all_safety_constraints(self) -> List[z3.ExprRef]:
        """Return conjunction of all safety constraints."""
        all_s: List[z3.ExprRef] = []
        for sc in self._safety_constraints:
            all_s.extend(sc)
        return all_s

    def get_bad_constraints(self) -> List[z3.ExprRef]:
        """Return bad-state constraints (negation of safety)."""
        return list(self._bad_constraints)

    def get_all_transition_constraints(self) -> List[z3.ExprRef]:
        """Return conjunction of all transition constraints."""
        all_t: List[z3.ExprRef] = []
        for tc in self._transition_constraints:
            all_t.extend(tc)
        return all_t

    def num_steps(self) -> int:
        """Number of computation steps."""
        return len(self.graph.steps)

    def get_dim_vars(self) -> Dict[str, z3.ArithRef]:
        """Return all dimension variables."""
        return dict(self._dim_vars)


# ═══════════════════════════════════════════════════════════════════════════════
# IC3 Solver
# ═══════════════════════════════════════════════════════════════════════════════


class IC3Solver:
    """IC3/PDR solver for tensor shape verification.

    Implements the standard IC3/PDR algorithm:
    1. Maintain frames F_0, F_1, ..., F_k over-approximating reachable states
    2. F_0 = initial states
    3. Check if bad states are reachable from F_k via transition
    4. If reachable: recursively block predecessor states
    5. If not: strengthen frames; detect fixed point (F_i == F_{i+1})
    6. Use generalization (UNSAT cores / interpolation) to block cubes efficiently
    """

    def __init__(
        self,
        transition_system: ShapeTransitionSystem,
        max_frames: int = 100,
        solver_timeout_ms: int = 5000,
        use_interpolation: bool = True,
    ) -> None:
        if not HAS_Z3:
            raise RuntimeError("Z3 is required for IC3/PDR")

        self.ts = transition_system
        self.max_frames = max_frames
        self.solver_timeout_ms = solver_timeout_ms
        self.use_interpolation = use_interpolation and HAS_INTERPOLATION

        # Frames: each frame is a list of blocking clauses
        # F_0 = init, F_i over-approximates states reachable in ≤ i steps
        self._frames: List[List[ShapeClause]] = []
        self._frame_solvers: List[z3.Solver] = []

        # Statistics
        self._z3_queries = 0
        self._blocked_cubes = 0
        self._status = IC3Status.UNKNOWN
        self._invariant_level: Optional[int] = None
        self._cex_depth: Optional[int] = None
        self._cex_trace: Optional[List[Dict[str, Any]]] = None
        self._used_unsat_core_fallback = False

    def solve(self) -> IC3Status:
        """Run IC3/PDR algorithm. Returns SAFE, UNSAFE, or UNKNOWN."""
        # Initialize F_0
        self._init_frame0()

        # Check initiation: F_0 ∩ Bad = ∅
        if not self._check_initiation():
            self._status = IC3Status.UNSAFE
            self._cex_depth = 0
            return self._status

        # Main IC3 loop
        for k in range(1, self.max_frames + 1):
            self._new_frame()

            # Try to block all bad cubes reachable from F_k
            blocked = self._block_all_bad(k)
            if not blocked:
                self._status = IC3Status.UNSAFE
                return self._status

            # Propagate clauses forward and check for fixed point
            if self._propagate_clauses():
                self._status = IC3Status.SAFE
                return self._status

        self._status = IC3Status.UNKNOWN
        return self._status

    def _init_frame0(self) -> None:
        """Initialize F_0 with initial state constraints."""
        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)

        # F_0 = Init
        for c in self.ts.get_init_constraints():
            solver.add(c)
        # Add all transition constraints (the model's shape propagation)
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)

        self._frames.append([])
        self._frame_solvers.append(solver)

    def _new_frame(self) -> None:
        """Add a new frame F_{k+1}."""
        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)

        # Add init constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        # Add transition constraints
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        # Add all blocking clauses from previous frame
        if self._frames:
            for clause in self._frames[-1]:
                if clause.z3_expr is not None:
                    solver.add(clause.z3_expr)

        self._frames.append([])
        self._frame_solvers.append(solver)

    def _check_initiation(self) -> bool:
        """Check F_0 ∩ Bad = ∅ (initial states are safe)."""
        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        self._z3_queries += 1

        # Add init constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        # Add transition constraints
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        # Add bad state constraints
        for c in self.ts.get_bad_constraints():
            solver.add(c)

        result = solver.check()
        if result == z3.sat:
            # Initial state can reach a bad state
            model = solver.model()
            self._cex_trace = [self._extract_state(model)]
            return False
        return True

    def _check_consecution(self, frame_idx: int, cube: ShapeCube) -> bool:
        """Check if cube is already blocked by frame_idx.

        Returns True if F_{frame_idx} ∧ T ⊨ ¬cube (cube is blocked).
        Returns False if F_{frame_idx} can reach cube in one step.
        """
        if cube.z3_expr is None:
            return True

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        self._z3_queries += 1

        # Add frame constraints
        if frame_idx < len(self._frame_solvers):
            for c in self.ts.get_init_constraints():
                solver.add(c)
            for c in self.ts.get_all_transition_constraints():
                solver.add(c)
            # Add blocking clauses from this frame
            if frame_idx < len(self._frames):
                for clause in self._frames[frame_idx]:
                    if clause.z3_expr is not None:
                        solver.add(clause.z3_expr)

        # Check if cube is reachable
        solver.add(cube.z3_expr)

        result = solver.check()
        return result == z3.unsat

    def _generalize(self, cube: ShapeCube, frame_idx: int) -> ShapeClause:
        """Generalize a cube to a blocking clause using UNSAT core.

        Given that F_{frame_idx} ∧ T ⊨ ¬cube, find a minimal subset of
        cube's literals that is sufficient for blocking.
        """
        if cube.z3_expr is None or not HAS_Z3:
            return cube.negate()

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        solver.set("unsat_core", True)
        self._z3_queries += 1

        # Add frame + transition constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        if frame_idx < len(self._frames):
            for clause in self._frames[frame_idx]:
                if clause.z3_expr is not None:
                    solver.add(clause.z3_expr)

        # Track each literal of the cube for UNSAT core extraction
        if cube.z3_expr is not None:
            # Split the cube into individual literals and track them
            lits = self._split_conjunction(cube.z3_expr)
            tags = []
            for i, lit in enumerate(lits):
                tag = z3.Bool(f"__cube_lit_{i}")
                solver.assert_and_track(lit, tag)
                tags.append((tag, lit))

            result = solver.check()
            if result == z3.unsat:
                core = solver.unsat_core()
                core_names = {str(c) for c in core}

                # Keep only literals that appear in the core
                core_lits = []
                core_lit_strs = set()
                for tag, lit in tags:
                    if str(tag) in core_names:
                        core_lits.append(lit)
                        core_lit_strs.add(str(lit))

                if core_lits:
                    gen_expr = z3.Not(z3.And(*core_lits)) if len(core_lits) > 1 else z3.Not(core_lits[0])
                    return ShapeClause(
                        literals=frozenset(core_lit_strs),
                        z3_expr=gen_expr,
                    )

        # Fallback: negate the full cube
        return cube.negate()

    def _try_interpolation_generalize(
        self, cube: ShapeCube, frame_idx: int
    ) -> Optional[ShapeClause]:
        """Try Craig interpolation for generalization.

        Prefers CVC5 native get-interpolant (mathematically valid Craig
        interpolation) over Z3 UNSAT-core simulation.  Falls back to
        simulation only when CVC5 is unavailable.
        """
        if not self.use_interpolation or not HAS_INTERPOLATION:
            return None
        if cube.z3_expr is None:
            return None

        # A = frame constraints ∧ transition
        a_constraints = list(self.ts.get_init_constraints())
        a_constraints.extend(self.ts.get_all_transition_constraints())
        if frame_idx < len(self._frames):
            for clause in self._frames[frame_idx]:
                if clause.z3_expr is not None:
                    a_constraints.append(clause.z3_expr)

        # B = cube (bad state we want to block)
        b_constraints = [cube.z3_expr]

        # Interface variables
        interface_vars = set()
        for name in self.ts.get_dim_vars():
            interface_vars.add(name)

        # Prefer CVC5 native interpolation (valid Craig interpolation)
        interpolant = None
        if HAS_CVC5:
            interpolant = _compute_cvc5_interpolant(
                a_constraints, b_constraints, interface_vars,
                timeout_ms=self.solver_timeout_ms,
            )
        # Fall back to UNSAT-core simulation if CVC5 unavailable
        if interpolant is None:
            self._used_unsat_core_fallback = True
            logger.warning(
                "CVC5 interpolation unavailable or failed; falling back to "
                "UNSAT-core simulation.  The resulting interpolant satisfies "
                "separation (I ∧ B is UNSAT) but may not satisfy the full "
                "Craig vocabulary restriction (Vars(I) ⊆ Vars(A) ∩ Vars(B))."
            )
            interpolant = _compute_simulated_interpolant(
                a_constraints, b_constraints, interface_vars,
                timeout_ms=self.solver_timeout_ms,
            )
        if interpolant is not None:
            return ShapeClause(
                literals=frozenset({str(interpolant)}),
                z3_expr=interpolant,
            )
        return None

    def _block_all_bad(self, k: int) -> bool:
        """Try to block all bad cubes at frame k.

        Returns True if all bad cubes were blocked, False if a real
        counterexample was found.
        """
        import heapq

        # Find bad cubes reachable from F_k
        bad_cube = self._find_bad_cube(k)
        if bad_cube is None:
            return True  # No bad states reachable

        # Priority queue of proof obligations
        queue: List[ProofObligation] = []
        heapq.heappush(queue, ProofObligation(bad_cube, k, depth=0))

        while queue:
            obligation = heapq.heappop(queue)

            if obligation.frame_level == 0:
                # Reached F_0 — real counterexample
                self._cex_depth = obligation.depth
                self._cex_trace = self._reconstruct_trace(obligation)
                return False

            # Check if cube is already blocked at this level
            if self._check_consecution(obligation.frame_level - 1, obligation.cube):
                # Cube is blocked by frame below — generalize and add clause
                clause = self._generalize(obligation.cube, obligation.frame_level - 1)

                # Try interpolation-based generalization for stronger clause
                if self.use_interpolation:
                    interp_clause = self._try_interpolation_generalize(
                        obligation.cube, obligation.frame_level - 1
                    )
                    if interp_clause is not None:
                        clause = interp_clause

                # Add blocking clause to frames 1..obligation.frame_level
                for i in range(1, obligation.frame_level + 1):
                    if i < len(self._frames):
                        self._frames[i].append(clause)
                        if clause.z3_expr is not None and i < len(self._frame_solvers):
                            self._frame_solvers[i].add(clause.z3_expr)
                self._blocked_cubes += 1
            else:
                # Cube is NOT blocked — find predecessor and recurse
                pred_cube = self._find_predecessor(obligation.frame_level - 1, obligation.cube)
                if pred_cube is not None:
                    heapq.heappush(
                        queue,
                        ProofObligation(
                            pred_cube,
                            obligation.frame_level - 1,
                            depth=obligation.depth + 1,
                        ),
                    )
                    # Re-push current obligation
                    heapq.heappush(queue, obligation)
                else:
                    # No predecessor found, but not blocked either - block it
                    clause = obligation.cube.negate()
                    for i in range(1, obligation.frame_level + 1):
                        if i < len(self._frames):
                            self._frames[i].append(clause)
                    self._blocked_cubes += 1

        return True

    def _find_bad_cube(self, frame_idx: int) -> Optional[ShapeCube]:
        """Find a bad cube reachable from frame_idx, or None."""
        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        self._z3_queries += 1

        # Frame constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        if frame_idx < len(self._frames):
            for clause in self._frames[frame_idx]:
                if clause.z3_expr is not None:
                    solver.add(clause.z3_expr)

        # Bad state constraints
        for c in self.ts.get_bad_constraints():
            solver.add(c)

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return self._model_to_cube(model)
        return None

    def _find_predecessor(
        self, frame_idx: int, cube: ShapeCube
    ) -> Optional[ShapeCube]:
        """Find a predecessor state that can reach the cube in one step."""
        if cube.z3_expr is None:
            return None

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        self._z3_queries += 1

        # Frame constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        if frame_idx < len(self._frames):
            for clause in self._frames[frame_idx]:
                if clause.z3_expr is not None:
                    solver.add(clause.z3_expr)

        # Target cube
        solver.add(cube.z3_expr)

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            return self._model_to_cube(model)
        return None

    def _propagate_clauses(self) -> bool:
        """Push blocking clauses forward; detect fixed point.

        If F_i == F_{i+1} for some i, we have an inductive invariant.
        Returns True if fixed point detected (SAFE).
        """
        if len(self._frames) < 2:
            return False

        for i in range(1, len(self._frames) - 1):
            clauses_i = self._frames[i]
            clauses_next = self._frames[i + 1]

            # Try to push each clause from F_i to F_{i+1}
            new_clauses_pushed = []
            for clause in clauses_i:
                if clause not in clauses_next:
                    # Check if clause holds at F_{i+1}
                    if self._clause_holds_at(clause, i + 1):
                        new_clauses_pushed.append(clause)

            for clause in new_clauses_pushed:
                self._frames[i + 1].append(clause)
                if clause.z3_expr is not None and (i + 1) < len(self._frame_solvers):
                    self._frame_solvers[i + 1].add(clause.z3_expr)

            # Check fixed point: F_i ⊆ F_{i+1} and F_{i+1} ⊆ F_i
            clauses_i_set = set(self._frames[i])
            clauses_next_set = set(self._frames[i + 1])
            if clauses_i_set == clauses_next_set or clauses_i_set.issubset(clauses_next_set):
                self._invariant_level = i
                return True

        return False

    def _clause_holds_at(self, clause: ShapeClause, frame_idx: int) -> bool:
        """Check if a clause holds at frame_idx (is inductive relative to F_{frame_idx-1})."""
        if clause.z3_expr is None:
            return True

        solver = z3.Solver()
        solver.set("timeout", self.solver_timeout_ms)
        self._z3_queries += 1

        # Add frame constraints
        for c in self.ts.get_init_constraints():
            solver.add(c)
        for c in self.ts.get_all_transition_constraints():
            solver.add(c)
        if frame_idx - 1 < len(self._frames):
            for fc in self._frames[frame_idx - 1]:
                if fc.z3_expr is not None:
                    solver.add(fc.z3_expr)

        # Check if ¬clause is reachable
        solver.add(z3.Not(clause.z3_expr))

        result = solver.check()
        return result == z3.unsat

    def _model_to_cube(self, model: z3.ModelRef) -> ShapeCube:
        """Extract a cube from a Z3 model."""
        literals: List[str] = []
        conjuncts: List[z3.ExprRef] = []

        for name, var in self.ts.get_dim_vars().items():
            val = model.eval(var, model_completion=True)
            if val is not None:
                try:
                    int_val = val.as_long()
                    lit_str = f"{name} == {int_val}"
                    literals.append(lit_str)
                    conjuncts.append(var == int_val)
                except (AttributeError, z3.Z3Exception):
                    pass

        z3_expr = z3.And(*conjuncts) if conjuncts else z3.BoolVal(True)
        return ShapeCube(
            literals=frozenset(literals),
            z3_expr=z3_expr,
        )

    def _split_conjunction(self, expr: z3.ExprRef) -> List[z3.ExprRef]:
        """Split a Z3 conjunction into individual conjuncts."""
        if z3.is_and(expr):
            result = []
            for child in expr.children():
                result.extend(self._split_conjunction(child))
            return result
        return [expr]

    def _extract_state(self, model: z3.ModelRef) -> Dict[str, Any]:
        """Extract state (dimension assignments) from Z3 model."""
        state: Dict[str, Any] = {}
        for name, var in self.ts.get_dim_vars().items():
            val = model.eval(var, model_completion=True)
            if val is not None:
                try:
                    state[name] = val.as_long()
                except (AttributeError, z3.Z3Exception):
                    state[name] = str(val)
        return state

    def _reconstruct_trace(self, obligation: ProofObligation) -> List[Dict[str, Any]]:
        """Reconstruct counterexample trace from proof obligation chain."""
        # Return the cube's literals as a trace step
        trace: List[Dict[str, Any]] = []
        if obligation.cube.z3_expr is not None:
            solver = z3.Solver()
            solver.set("timeout", self.solver_timeout_ms)
            for c in self.ts.get_init_constraints():
                solver.add(c)
            for c in self.ts.get_all_transition_constraints():
                solver.add(c)
            solver.add(obligation.cube.z3_expr)
            if solver.check() == z3.sat:
                trace.append(self._extract_state(solver.model()))
        if not trace:
            trace.append({str(l): True for l in obligation.cube.literals})
        return trace

    def get_invariant_str(self) -> Optional[str]:
        """Return human-readable inductive invariant if safe."""
        if self._status != IC3Status.SAFE or self._invariant_level is None:
            return None

        clauses = self._frames[self._invariant_level]
        if not clauses:
            # No blocking clauses needed — extract safety properties from
            # the transition system as the proven invariant.
            safety = self.ts.get_all_safety_constraints()
            init = self.ts.get_init_constraints()
            trans = self.ts.get_all_transition_constraints()
            parts = []
            for c in init:
                parts.append(str(c))
            for c in trans:
                parts.append(str(c))
            for c in safety:
                parts.append(str(c))
            if parts:
                return " ∧ ".join(parts)
            return "True (trivially safe)"

        clause_strs = []
        for clause in clauses:
            clause_strs.append(str(clause.z3_expr) if clause.z3_expr is not None else str(clause.literals))

        return " ∧ ".join(clause_strs)

    def get_invariant_clauses(self) -> List[str]:
        """Return individual invariant clauses."""
        if self._status != IC3Status.SAFE or self._invariant_level is None:
            return []
        clauses = self._frames[self._invariant_level]
        if not clauses:
            # Return safety + init + transition constraints as the invariant
            result = []
            for c in self.ts.get_init_constraints():
                result.append(str(c))
            for c in self.ts.get_all_transition_constraints():
                result.append(str(c))
            for c in self.ts.get_all_safety_constraints():
                result.append(str(c))
            return result
        return [
            str(c.z3_expr) if c.z3_expr is not None else str(c.literals)
            for c in clauses
        ]

    @property
    def frames_computed(self) -> int:
        return len(self._frames)

    @property
    def z3_queries(self) -> int:
        return self._z3_queries

    @property
    def blocked_cubes(self) -> int:
        return self._blocked_cubes

    def get_frame_sequence_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of each frame for visualization.

        Returns a list of dicts with:
          - frame_index: int
          - num_clauses: int
          - clause_summaries: list of str (first 5 clauses as strings)
        """
        summary: List[Dict[str, Any]] = []
        for i, frame_clauses in enumerate(self._frames):
            clause_strs = []
            for c in frame_clauses[:5]:
                clause_strs.append(
                    str(c.z3_expr) if c.z3_expr is not None else str(c.literals)
                )
            summary.append({
                "frame_index": i,
                "num_clauses": len(frame_clauses),
                "clause_summaries": clause_strs,
            })
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def ic3_verify(
    model_source: str,
    symbolic_dims: Optional[Dict[str, str]] = None,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_frames: int = 100,
    solver_timeout_ms: int = 5000,
    use_interpolation: bool = True,
) -> IC3Result:
    """Run IC3/PDR verification on a PyTorch nn.Module.

    Parameters
    ----------
    model_source : str
        Python source code containing an nn.Module subclass.
    symbolic_dims : dict, optional
        Mapping from shape position names to symbolic parameter names.
        E.g., {"batch": "batch_size"} means the batch dimension is symbolic.
    input_shapes : dict, optional
        Mapping from forward parameter names to shape tuples.
        Dimensions may be ints (concrete) or strings (symbolic).
    max_frames : int
        Maximum number of IC3 frames before giving up.
    solver_timeout_ms : int
        Z3 solver timeout per query.
    use_interpolation : bool
        Whether to use Craig interpolation for generalization.

    Returns
    -------
    IC3Result
        Contains safety verdict, invariant (if safe), counterexample depth
        (if unsafe), and statistics.
    """
    t0 = time.monotonic()
    symbolic_dims = symbolic_dims or {}

    if not HAS_Z3:
        return IC3Result(
            safe=False,
            frames_computed=0,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    # Extract computation graph
    try:
        graph = extract_computation_graph(model_source)
    except (ValueError, SyntaxError) as exc:
        logger.error("Failed to extract computation graph: %s", exc)
        return IC3Result(
            safe=False,
            frames_computed=0,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    # Infer input shapes if not provided
    if input_shapes is None:
        input_shapes = {}
        for inp_name in graph.input_names:
            input_shapes[inp_name] = ("batch", 10)

    # Apply symbolic dims to input shapes
    resolved_shapes: Dict[str, tuple] = {}
    for inp_name, shape in input_shapes.items():
        new_shape = []
        for dim_val in shape:
            if isinstance(dim_val, str) and dim_val in symbolic_dims:
                new_shape.append(symbolic_dims[dim_val])
            else:
                new_shape.append(dim_val)
        resolved_shapes[inp_name] = tuple(new_shape)

    # Build transition system
    try:
        ts = ShapeTransitionSystem(
            graph, resolved_shapes, symbolic_dims, solver_timeout_ms
        )
    except Exception as exc:
        logger.error("Failed to build transition system: %s", exc)
        return IC3Result(
            safe=False,
            frames_computed=0,
            verification_time_ms=(time.monotonic() - t0) * 1000,
            symbolic_dims=symbolic_dims,
        )

    # Run IC3/PDR
    solver = IC3Solver(
        ts,
        max_frames=max_frames,
        solver_timeout_ms=solver_timeout_ms,
        use_interpolation=use_interpolation,
    )

    status = solver.solve()

    elapsed_ms = (time.monotonic() - t0) * 1000
    frame_seq = solver.get_frame_sequence_summary()

    # Build soundness warning when UNSAT-core fallback was used
    sw: Optional[str] = None
    if solver._used_unsat_core_fallback:
        sw = (
            "Craig interpolation fell back to UNSAT-core simulation "
            "(CVC5 unavailable or failed).  The simulated interpolant "
            "satisfies separation (I ∧ B is UNSAT) but may violate the "
            "Craig vocabulary restriction, so the inductive invariant "
            "may reference variables outside the interface.  Safety "
            "verdicts derived from these interpolants carry reduced "
            "soundness guarantees."
        )

    if status == IC3Status.SAFE:
        return IC3Result(
            safe=True,
            invariant=solver.get_invariant_str(),
            frames_computed=solver.frames_computed,
            verification_time_ms=elapsed_ms,
            symbolic_dims=symbolic_dims,
            num_blocked_cubes=solver.blocked_cubes,
            z3_queries=solver.z3_queries,
            invariant_clauses=solver.get_invariant_clauses(),
            frame_sequence=frame_seq,
            soundness_warning=sw,
        )
    elif status == IC3Status.UNSAFE:
        return IC3Result(
            safe=False,
            counterexample_depth=solver._cex_depth,
            frames_computed=solver.frames_computed,
            verification_time_ms=elapsed_ms,
            symbolic_dims=symbolic_dims,
            num_blocked_cubes=solver.blocked_cubes,
            z3_queries=solver.z3_queries,
            counterexample_trace=solver._cex_trace,
            frame_sequence=frame_seq,
            soundness_warning=sw,
        )
    else:
        return IC3Result(
            safe=False,
            frames_computed=solver.frames_computed,
            verification_time_ms=elapsed_ms,
            symbolic_dims=symbolic_dims,
            num_blocked_cubes=solver.blocked_cubes,
            z3_queries=solver.z3_queries,
            frame_sequence=frame_seq,
            soundness_warning=sw,
        )
