"""Step 58 -- the "why" explainer (``--explain``).

When TensorGuard reports a shape bug, ``--explain`` prints the *inference chain*
that led to it: the step-by-step shape propagation from the forward inputs down
to the failing operation, so a developer can see exactly where a tensor first
acquired the shape that made the final op illegal.

The chain is reconstructed purely from the verifier's own counterexample trace
(``CounterexampleTrace``) and the computation graph -- no re-execution, no torch.
``states[i]`` is the symbolic shape environment *before* step ``i`` and
``states[i+1]`` the environment *after* it, so each link can show the op, its
input shapes, and the shape it produced.  The failing step is highlighted with
the expected-vs-actual shapes drawn from the violation.

This is deliberately model-agnostic and defensive (every field access is
guarded) so it can never raise from inside the verification pipeline; an
incomplete trace simply yields a shorter chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Reuse the shape pretty-printer so chain output matches the diagnostics.
try:
    from src.source_mapped_errors import _shape_str  # type: ignore
except Exception:  # pragma: no cover - fallback if import graph changes
    def _shape_str(shape: Any) -> str:  # type: ignore
        return "unknown" if shape is None else str(shape)


__all__ = [
    "ChainLink",
    "InferenceChain",
    "build_inference_chain",
    "format_chain_plain",
    "format_chain_ansi",
]


@dataclass
class ChainLink:
    """One step in the shape-inference chain."""
    step_index: int
    op: str                       # op name, e.g. "LAYER_CALL" or "RESHAPE"
    layer: Optional[str]          # attr name when the op is a layer call
    line: int
    inputs: List[str]             # input tensor names
    input_shapes: List[str]       # pretty-printed input shapes (aligned w/ inputs)
    output: str                   # output tensor name
    output_shape: str             # pretty-printed output shape ("?" if unknown)
    is_failing: bool = False
    expected_shape: Optional[str] = None   # only on the failing link
    actual_shape: Optional[str] = None     # only on the failing link


@dataclass
class InferenceChain:
    """The full inference chain leading to a reported bug."""
    model_name: str
    failing_step: int
    links: List[ChainLink] = field(default_factory=list)
    concrete_dims: Dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def __bool__(self) -> bool:
        return bool(self.links)


def _shape_env(state: Any) -> Dict[str, Any]:
    return dict(getattr(state, "shape_env", {}) or {})


def _op_name(step: Any) -> str:
    op = getattr(step, "op", None)
    return getattr(op, "name", str(op)) if op is not None else "unknown"


def build_inference_chain(
    graph: Any,
    counterexample: Any,
) -> InferenceChain:
    """Reconstruct the shape-inference chain from a counterexample trace.

    Parameters
    ----------
    graph:
        The ``ComputationGraph`` (provides ordered ``steps``).
    counterexample:
        The ``CounterexampleTrace`` (provides ``states``, ``failing_step``,
        ``violations`` and ``concrete_dims``).

    Returns
    -------
    InferenceChain
        Empty (falsey) when there is nothing to explain.
    """
    chain = InferenceChain(
        model_name=str(getattr(graph, "class_name", "") or "model"),
        failing_step=int(getattr(counterexample, "failing_step", -1) or -1),
        concrete_dims=dict(getattr(counterexample, "concrete_dims", {}) or {}),
    )
    if graph is None or counterexample is None:
        return chain

    steps = list(getattr(graph, "steps", []) or [])
    states = list(getattr(counterexample, "states", []) or [])
    if not steps or not states:
        return chain

    failing = chain.failing_step
    last = failing if failing >= 0 else len(steps) - 1
    last = min(last, len(steps) - 1)

    # Map the failing step's violation (if any) to expected/actual shapes.
    fail_expected: Optional[str] = None
    fail_actual: Optional[str] = None
    for v in getattr(counterexample, "violations", []) or []:
        if int(getattr(v, "step_index", -1)) == failing:
            sa = getattr(v, "shape_a", None)
            sb = getattr(v, "shape_b", None)
            if sa is not None:
                fail_actual = _shape_str(sa)
            if sb is not None:
                fail_expected = _shape_str(sb)
            break

    for i in range(0, last + 1):
        step = steps[i]
        before = _shape_env(states[i]) if i < len(states) else {}
        after = _shape_env(states[i + 1]) if (i + 1) < len(states) else {}

        inputs = list(getattr(step, "inputs", []) or [])
        in_shapes = [
            _shape_str(before.get(name)) if name in before else "?"
            for name in inputs
        ]
        output = str(getattr(step, "output", "") or "")
        out_shape = _shape_str(after.get(output)) if output in after else "?"

        link = ChainLink(
            step_index=i,
            op=_op_name(step),
            layer=getattr(step, "layer_ref", None),
            line=int(getattr(step, "line", 0) or 0),
            inputs=inputs,
            input_shapes=in_shapes,
            output=output,
            output_shape=out_shape,
            is_failing=(i == failing),
        )
        if link.is_failing:
            link.expected_shape = fail_expected
            link.actual_shape = fail_actual
        chain.links.append(link)

    # One-line summary: where the offending shape entered vs where it failed.
    if chain.links:
        fail_link = next((l for l in chain.links if l.is_failing), chain.links[-1])
        op_label = fail_link.layer or fail_link.op.lower()
        chain.summary = (
            f"The bug surfaces at step {fail_link.step_index} "
            f"({op_label}, line {fail_link.line}): "
        )
        if fail_link.expected_shape and fail_link.actual_shape:
            chain.summary += (
                f"it expected {fail_link.expected_shape} but the chain produced "
                f"{fail_link.actual_shape}."
            )
        else:
            chain.summary += "see the chain above for the propagated shapes."
    return chain


def _format_link(link: ChainLink) -> str:
    label = f"self.{link.layer}" if link.layer else link.op.lower()
    ins = ", ".join(
        f"{n}={s}" for n, s in zip(link.inputs, link.input_shapes)
    ) or "(inputs)"
    base = (
        f"[{link.step_index}] {label}  (line {link.line})\n"
        f"      in:  {ins}\n"
        f"      out: {link.output}={link.output_shape}"
    )
    if link.is_failing and link.expected_shape and link.actual_shape:
        base += (
            f"\n      !! expected {link.expected_shape}, "
            f"got {link.actual_shape}"
        )
    return base


def format_chain_plain(chain: InferenceChain) -> str:
    """Render the inference chain as plain text."""
    if not chain:
        return ""
    lines = [f"Why: inference chain for {chain.model_name}"]
    if chain.concrete_dims:
        dims = ", ".join(f"{k}={v}" for k, v in sorted(chain.concrete_dims.items()))
        lines.append(f"  (with concrete dimensions {dims})")
    for link in chain.links:
        marker = "  x " if link.is_failing else "  -> "
        rendered = _format_link(link)
        first, *rest = rendered.split("\n")
        lines.append(f"{marker}{first}")
        lines.extend(f"    {r}" for r in rest)
    if chain.summary:
        lines.append(f"  {chain.summary}")
    return "\n".join(lines)


# ANSI colours (kept local so the module has no hard dependency).
_R = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_DIM = "\033[2m"
_CYAN = "\033[36m"


def format_chain_ansi(chain: InferenceChain) -> str:
    """Render the inference chain with ANSI colour for terminals."""
    if not chain:
        return ""
    lines = [f"{_BOLD}{_CYAN}Why{_R}: inference chain for {chain.model_name}"]
    if chain.concrete_dims:
        dims = ", ".join(f"{k}={v}" for k, v in sorted(chain.concrete_dims.items()))
        lines.append(f"  {_DIM}(with concrete dimensions {dims}){_R}")
    for link in chain.links:
        rendered = _format_link(link)
        first, *rest = rendered.split("\n")
        if link.is_failing:
            lines.append(f"  {_RED}{_BOLD}x{_R} {first}")
            lines.extend(f"    {_RED}{r}{_R}" for r in rest)
        else:
            lines.append(f"  {_DIM}->{_R} {first}")
            lines.extend(f"    {_DIM}{r}{_R}" for r in rest)
    if chain.summary:
        lines.append(f"  {_BOLD}{chain.summary}{_R}")
    return "\n".join(lines)
