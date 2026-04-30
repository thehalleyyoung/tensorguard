"""
Grad-flag verifier (Track D, NEW per user request).

Statically certifies the autograd `requires_grad` / `.grad`-population
contract over a verified forward graph (Track A's
``Tensor[τ, φ_shape, φ_grad]``).

Soundness statement (informal):
    Let G be a verified forward graph and P = {p : p ∈ model.parameters()}.
    After ``loss.backward()`` (with ``loss`` the designated loss tensor),
    PyTorch sets ``p.grad`` to a non-None tensor IFF there exists a path
    in G from p to loss along edges whose source has φ_grad = True and
    whose target is *not* a `.detach()` or *not* inside `no_grad`.

    The verifier returns OK iff for every p ∈ P:
        (p.grad will be populated)  ⇔  (the user expects p.grad)

    Concretely it flags four bug classes:
        (B1) silently-None grad: parameter expected to learn but not
             reachable (used only inside ``with torch.no_grad():``,
             behind ``.detach()``, or never used).
        (B2) requires_grad=False on a parameter that *is* used.
        (B3) in-place op on a leaf that requires_grad (autograd error).
        (B4) ``.backward()`` precondition violated: no leaf in the graph
             has requires_grad=True (PyTorch raises RuntimeError).

Soundness sketch:
    Each forward node carries a `no_grad` flag (set by Track A when the
    op was traced inside ``torch.no_grad()``) and a per-input `detached`
    bit. Reachability is computed on the sub-graph excluding such edges.
    By PyTorch's autograd construction, only nodes on a non-pruned
    has_grad path enter the backward graph; hence ``.grad`` is set
    exactly on the leaves of that sub-graph. The verifier reproduces
    this reachability symbolically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .backward_shape import ForwardGraph, TensorSpec, Node


@dataclass
class GradIssue:
    kind: str            # B1 | B2 | B3 | B4 | unknown_param
    param: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.param}: {self.detail}"


@dataclass
class GradFlagReport:
    ok: bool
    issues: List[GradIssue]
    will_have_grad: Set[str]
    expected_grad: Set[str]

    def __bool__(self) -> bool:
        return self.ok


def _reverse_reachable(graph: ForwardGraph, sink: str) -> Set[str]:
    """Tensors that reach ``sink`` along non-detached, non-no_grad edges."""
    producer: Dict[str, Node] = {}
    for n in graph.nodes:
        for o in n.outputs:
            producer[o] = n

    reachable: Set[str] = set()
    stack = [sink]
    while stack:
        t = stack.pop()
        if t in reachable:
            continue
        reachable.add(t)
        n = producer.get(t)
        if n is None:
            continue
        if n.attrs.get("no_grad", False):
            # autograd edges below this node are pruned
            continue
        if n.op == "detach":
            continue
        for inp in n.inputs:
            spec = graph.tensors.get(inp)
            if spec is None or spec.detached:
                continue
            stack.append(inp)
    return reachable


def verify_grad_flags(
    graph: ForwardGraph,
    parameters: Iterable[str],
    *,
    expected_to_learn: Optional[Iterable[str]] = None,
) -> GradFlagReport:
    """Verify the autograd grad-flag contract.

    Args:
      graph: a verified forward graph.
      parameters: names of tensors corresponding to ``model.parameters()``.
      expected_to_learn: names the *user* expects to receive a gradient
        (defaults to ``parameters``). Useful when a model freezes some
        params on purpose; pass the trainable subset.
    """
    issues: List[GradIssue] = []
    params = list(parameters)
    expected = set(expected_to_learn) if expected_to_learn is not None else set(params)

    # B4: backward precondition.
    if not any(graph.tensors[p].requires_grad for p in params if p in graph.tensors):
        # also accept if any *leaf* tensor has requires_grad=True
        any_grad_leaf = any(
            t.is_leaf and t.requires_grad for t in graph.tensors.values()
        )
        if not any_grad_leaf:
            issues.append(GradIssue(
                "B4", "<graph>",
                ".backward() precondition violated: no leaf has requires_grad=True"))

    # Reachability from loss (autograd-aware).
    reachable = _reverse_reachable(graph, graph.loss)

    will_have_grad: Set[str] = set()
    for p in params:
        spec = graph.tensors.get(p)
        if spec is None:
            issues.append(GradIssue(
                "unknown_param", p, "parameter not in forward graph"))
            continue

        used = p in reachable
        rg = spec.requires_grad and not spec.detached

        # Determine what PyTorch will actually do.
        if used and rg:
            will_have_grad.add(p)

        wants = p in expected

        # B1: expected but won't get grad
        if wants and not (used and rg):
            why = []
            if not used:
                why.append("not reachable from loss "
                           "(possibly inside no_grad or after .detach())")
            if not spec.requires_grad:
                why.append("requires_grad=False")
            if spec.detached:
                why.append("tensor was .detach()'d")
            issues.append(GradIssue(
                "B1", p,
                "silently-None grad: " + "; ".join(why or ["unknown"])))

        # B2: requires_grad=False but used and expected to learn
        if wants and used and not spec.requires_grad:
            issues.append(GradIssue(
                "B2", p,
                "requires_grad=False on a tensor used in forward path"))

        # B3: in-place op on a leaf tensor that requires_grad
        for n in graph.nodes:
            if n.inplace and n.inputs and n.inputs[0] == p \
                    and spec.is_leaf and spec.requires_grad:
                issues.append(GradIssue(
                    "B3", p,
                    f"in-place op '{n.op}' on leaf with requires_grad=True "
                    f"(autograd will raise RuntimeError)"))

    return GradFlagReport(
        ok=not issues, issues=issues,
        will_have_grad=will_have_grad, expected_grad=expected,
    )


# ---------------------------------------------------------------------------
# optimizer.step preconditions
# ---------------------------------------------------------------------------

@dataclass
class OptStepReport:
    ok: bool
    silently_skipped: List[str]    # params whose update will be a no-op
    detail: Dict[str, str]


def verify_optimizer_step_preconditions(
    graph: ForwardGraph,
    parameters: Iterable[str],
) -> OptStepReport:
    """Predict which params optimizer.step() would silently leave unchanged.

    For SGD, Adam, etc., the per-param update is ``no-op if p.grad is None``
    (PyTorch's vanilla loop skips them). We flag every such param.
    """
    grad_report = verify_grad_flags(graph, parameters,
                                    expected_to_learn=parameters)
    skipped = [p for p in parameters
               if p not in grad_report.will_have_grad]
    detail = {iss.param: iss.detail for iss in grad_report.issues
              if iss.kind in ("B1", "B2")}
    return OptStepReport(ok=not skipped, silently_skipped=skipped, detail=detail)


# ---------------------------------------------------------------------------
# Runtime cross-check (for property tests).
# ---------------------------------------------------------------------------

def runtime_grad_flags(model, inputs) -> Dict[str, bool]:
    """Run model(*inputs).sum().backward() and return {pname: grad is not None}."""
    import torch
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    out = model(*inputs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    loss = out.sum()
    if loss.requires_grad:
        loss.backward()
    return {n: (p.grad is not None) for n, p in model.named_parameters()}


__all__ = [
    "GradIssue", "GradFlagReport", "OptStepReport",
    "verify_grad_flags", "verify_optimizer_step_preconditions",
    "runtime_grad_flags",
]
