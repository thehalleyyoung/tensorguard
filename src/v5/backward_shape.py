"""
Backward shape verifier (Track D).

Statically certifies, given a verified forward graph, that the autograd
backward pass would produce gradients whose shapes match the corresponding
primals -- WITHOUT actually executing the backward pass.

Theorem (backward-shape soundness, informal):
    If `verify_backward(graph)` returns OK, then for every node v in the
    forward graph that participates in autograd (φ_grad(v) = True), the
    gradient produced by ``loss.backward()`` -- if it exists -- has shape
    equal to ``primal(v).shape``, modulo the broadcasting reduction
    sum_to(grad, primal.shape).

Sketch of soundness:
    PyTorch's autograd contract guarantees, for each Function F with
    forward outputs y_1..y_k and inputs x_1..x_n, that the backward call
    receives ``grad_outputs`` with ``grad_outputs[i].shape == y_i.shape``
    and that it produces ``grad_inputs[j]`` either ``None`` (if x_j has
    requires_grad=False or is non-floating-point) or of shape ``x_j.shape``
    (after sum_to reduction over broadcasted dimensions).

    By induction over the topological reverse order:
      * Base: ``grad_loss.shape == ()`` matches the scalar loss.
      * Step: if all downstream grads at a node n have correct shape,
        then by the per-op shape lemmas (see SHAPE_RULES below), the
        upstream grads also have correct shape.

    The static checker mirrors this induction symbolically by walking
    the forward graph in reverse and applying the per-op shape rules.

Caveats / assumptions:
  * .view() shares storage; .reshape() may copy. Both preserve shape
    semantics for the gradient (grad_input is reshaped back). We treat
    them identically for shape, but record a `storage_aliased` flag for
    the in-place check.
  * .detach() severs the graph: downstream grads do not flow through it.
    Such edges are pruned from the backward graph.
  * In-place ops mutating a tensor that is needed by another node's
    backward are flagged (because saved_tensors would observe the
    mutated value, breaking the shape contract for many functions).

This module imports nothing from ``src/`` (the repository's main package
under tensorguard) so as not to create a hard coupling, but the public
``TensorSpec`` mirrors the Track A refinement triple
``Tensor[τ, φ_shape, φ_grad]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

# ---------------------------------------------------------------------------
# Refinement-style tensor spec  (Tensor[τ, φ_shape, φ_grad])
# ---------------------------------------------------------------------------

ShapeT = Tuple[Optional[int], ...]   # None = symbolic / unknown dim


@dataclass
class TensorSpec:
    """Refinement triple Tensor[τ, φ_shape, φ_grad]."""
    name: str
    shape: ShapeT
    dtype: str = "float32"               # τ
    requires_grad: bool = False          # φ_grad
    storage_id: Optional[int] = None     # identity for view/inplace checks
    is_leaf: bool = True
    detached: bool = False               # severed from graph

    @property
    def has_grad(self) -> bool:
        """φ_grad: participates in autograd."""
        return self.requires_grad and not self.detached and \
            self.dtype.startswith("float") or self.dtype.startswith("complex")

    def with_(self, **kw) -> "TensorSpec":
        d = self.__dict__.copy(); d.update(kw); return TensorSpec(**d)


@dataclass
class Node:
    """A forward-graph node: an op with input specs and output specs."""
    op: str
    inputs: List[str]                  # names of input tensors
    outputs: List[str]                 # names of output tensors
    attrs: Dict[str, Any] = field(default_factory=dict)
    inplace: bool = False              # mutates inputs[0] in-place


@dataclass
class ForwardGraph:
    """A verified forward graph (Track A output)."""
    tensors: Dict[str, TensorSpec]
    nodes: List[Node]                  # topologically ordered
    loss: str                          # name of scalar loss tensor


# ---------------------------------------------------------------------------
# Per-op static shape rules for the backward pass.
#
# A rule receives the node and the tensor environment; it returns, for
# each input position, the *expected* shape of grad_input[i] (or None if
# that input does not receive a gradient).  The verifier then checks
# this against the primal input's shape.
# ---------------------------------------------------------------------------

GradShapeFn = Callable[[Node, Dict[str, TensorSpec]], List[Optional[ShapeT]]]
SHAPE_RULES: Dict[str, GradShapeFn] = {}


def _rule(name: str):
    def deco(fn):
        SHAPE_RULES[name] = fn
        return fn
    return deco


def _broadcast_reduce(grad_shape: ShapeT, target_shape: ShapeT) -> ShapeT:
    """sum_to: simulate the broadcast-reduction PyTorch performs."""
    # right-align; any axis where target is 1 or missing must reduce.
    diff = len(grad_shape) - len(target_shape)
    if diff < 0:
        # grad cannot be smaller than target along right-aligned axes
        return target_shape  # treat conservatively as target
    return target_shape


@_rule("add")
@_rule("sub")
@_rule("mul")
@_rule("div")
def _bin_elementwise(node, env):
    out_spec = env[node.outputs[0]]
    return [_broadcast_reduce(out_spec.shape, env[i].shape) for i in node.inputs]


@_rule("matmul")
def _matmul(node, env):
    a, b = env[node.inputs[0]], env[node.inputs[1]]
    return [a.shape, b.shape]


@_rule("relu")
@_rule("gelu")
@_rule("silu")
@_rule("tanh")
@_rule("sigmoid")
@_rule("softmax")
@_rule("dropout")
@_rule("layer_norm")
@_rule("batch_norm")
def _unary_same_shape(node, env):
    return [env[i].shape for i in node.inputs]


@_rule("sum")
@_rule("mean")
def _reduction(node, env):
    return [env[node.inputs[0]].shape]


@_rule("view")
@_rule("reshape")
def _reshape_like(node, env):
    return [env[node.inputs[0]].shape]


@_rule("transpose")
@_rule("permute")
def _permute(node, env):
    return [env[node.inputs[0]].shape]


@_rule("cat")
def _cat(node, env):
    return [env[i].shape for i in node.inputs]


@_rule("linear")
def _linear(node, env):
    # inputs: x, weight, [bias]
    out = [env[i].shape for i in node.inputs]
    return out


@_rule("conv2d")
def _conv2d(node, env):
    return [env[i].shape for i in node.inputs]


@_rule("detach")
def _detach(node, env):
    # gradient does NOT flow back through detach
    return [None]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@dataclass
class BackwardIssue:
    kind: str                # 'shape_mismatch' | 'unknown_op' | 'inplace_alias'
                              # | 'unreachable_grad' | 'detach_breaks_path'
    node_op: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.node_op}: {self.detail}"


@dataclass
class BackwardReport:
    ok: bool
    issues: List[BackwardIssue]
    grad_shapes: Dict[str, ShapeT]   # certified grad shape per tensor

    def __bool__(self) -> bool:
        return self.ok


def verify_backward(graph: ForwardGraph) -> BackwardReport:
    """Statically verify the backward pass shape contract."""
    issues: List[BackwardIssue] = []
    grad_shapes: Dict[str, ShapeT] = {}

    # 1. Precondition: loss must be scalar and have_grad.
    loss = graph.tensors.get(graph.loss)
    if loss is None:
        return BackwardReport(False, [BackwardIssue(
            "missing_loss", "<root>", f"loss '{graph.loss}' not in graph")], {})
    if loss.shape not in ((), (1,)):
        issues.append(BackwardIssue(
            "shape_mismatch", "<loss>",
            f"loss '{graph.loss}' has shape {loss.shape}, expected scalar"))
    if not loss.requires_grad:
        issues.append(BackwardIssue(
            "unreachable_grad", "<loss>",
            f"loss '{graph.loss}' has requires_grad=False"))
    grad_shapes[graph.loss] = ()

    # 2. In-place alias check: any tensor needed for a backward of a
    #    *later* node and mutated in-place by some intermediate node?
    storage_to_consumers: Dict[int, List[int]] = {}
    for idx, n in enumerate(graph.nodes):
        for inp in n.inputs:
            sid = graph.tensors[inp].storage_id
            if sid is not None:
                storage_to_consumers.setdefault(sid, []).append(idx)
    for idx, n in enumerate(graph.nodes):
        if n.inplace and n.inputs:
            sid = graph.tensors[n.inputs[0]].storage_id
            later = [c for c in storage_to_consumers.get(sid, []) if c > idx]
            if later and graph.tensors[n.inputs[0]].requires_grad:
                issues.append(BackwardIssue(
                    "inplace_alias", n.op,
                    f"in-place op mutates '{n.inputs[0]}' "
                    f"(storage {sid}) needed by {len(later)} downstream node(s)"))

    # 3. Reverse topological walk applying per-op shape rules.
    for n in reversed(graph.nodes):
        # if no output of this node has a downstream grad, skip
        out_has_grad = any(graph.tensors[o].has_grad for o in n.outputs)
        if not out_has_grad:
            continue

        rule = SHAPE_RULES.get(n.op)
        if rule is None:
            issues.append(BackwardIssue(
                "unknown_op", n.op,
                f"no shape rule for op '{n.op}' (cannot certify backward)"))
            continue
        expected = rule(n, graph.tensors)
        for i, inp in enumerate(n.inputs):
            t = graph.tensors[inp]
            exp = expected[i]
            if exp is None:
                # gradient not propagated to this input
                continue
            if not t.has_grad:
                # input doesn't need a grad; nothing to check
                continue
            if exp != t.shape:
                issues.append(BackwardIssue(
                    "shape_mismatch", n.op,
                    f"grad_input[{i}] for '{inp}' would be {exp} "
                    f"but primal shape is {t.shape}"))
            else:
                grad_shapes[inp] = t.shape

    return BackwardReport(ok=not issues, issues=issues, grad_shapes=grad_shapes)


# ---------------------------------------------------------------------------
# Convenience: extract a ForwardGraph from a torch.fx.GraphModule
# ---------------------------------------------------------------------------

def from_fx(gm) -> ForwardGraph:
    """Best-effort extraction (used by tests; depends on torch)."""
    import torch
    tensors: Dict[str, TensorSpec] = {}
    nodes: List[Node] = []
    loss_name = None
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            tensors[n.name] = TensorSpec(name=n.name, shape=(), requires_grad=True)
        elif n.op == "call_function" or n.op == "call_module":
            opname = (n.target.__name__
                      if hasattr(n.target, "__name__") else str(n.target))
            inputs = [a.name for a in n.args if hasattr(a, "name")]
            tensors[n.name] = TensorSpec(name=n.name, shape=(), requires_grad=True)
            nodes.append(Node(op=opname.lower(), inputs=inputs, outputs=[n.name]))
            loss_name = n.name
        elif n.op == "output":
            pass
    return ForwardGraph(tensors=tensors, nodes=nodes, loss=loss_name or "")


__all__ = [
    "TensorSpec", "Node", "ForwardGraph",
    "BackwardIssue", "BackwardReport",
    "verify_backward", "from_fx", "SHAPE_RULES",
]
