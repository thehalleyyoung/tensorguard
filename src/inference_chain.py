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

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
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
    "format_explain_html",
    "write_explain_html",
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


def _html(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _bug_rows(result: Any) -> str:
    rows = []
    for bug in list(getattr(result, "bugs", []) or []):
        loc = getattr(bug, "location", None)
        location = f"{getattr(loc, 'file', '')}:{getattr(loc, 'line', 0)}"
        rows.append(
            "<tr>"
            f"<td>{_html(getattr(bug, 'severity', ''))}</td>"
            f"<td>{_html(getattr(bug, 'category', ''))}</td>"
            f"<td>{_html(location)}</td>"
            f"<td>{_html(getattr(bug, 'message', ''))}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="4">No bugs reported.</td></tr>'
    return "\n".join(rows)


def _counterexample_html(counterexample: Any) -> str:
    if not counterexample:
        return "<p>No counterexample witness was produced for this run.</p>"
    dims = dict(counterexample.get("concrete_dims", {}) or {}) if isinstance(counterexample, dict) else {}
    violations = list(counterexample.get("violations", []) or []) if isinstance(counterexample, dict) else []
    parts = []
    if dims:
        parts.append("<h3>Concrete symbolic dimensions</h3><div class=\"chips\">")
        for name, value in sorted(dims.items()):
            parts.append(f"<span class=\"chip\">{_html(name)} = {_html(value)}</span>")
        parts.append("</div>")
    if violations:
        parts.append("<h3>Violations</h3><ol>")
        for v in violations:
            parts.append(
                "<li>"
                f"<strong>{_html(v.get('kind', 'violation'))}</strong>"
                f" at line {_html(v.get('line', 0))}: "
                f"{_html(v.get('message', ''))}"
                "</li>"
            )
        parts.append("</ol>")
    if not parts:
        parts.append("<pre>" + _html(json.dumps(counterexample, indent=2, default=str)) + "</pre>")
    return "\n".join(parts)


def _footprint_for_link(link: ChainLink) -> Dict[str, object]:
    op_name = link.layer or link.op.lower()
    try:
        from src.proof_footprint import footprint_for
        return footprint_for(op_name)
    except Exception:
        return {
            "operator": op_name,
            "proof_status": "unknown",
            "confidence": "heuristic",
            "rule": "proof footprint unavailable",
            "evidence": [],
            "rationale": "Could not load the proof-footprint manifest.",
        }


def _source_snippet(source_lines: List[str], line: int) -> str:
    if line <= 0 or line > len(source_lines):
        return ""
    start = max(1, line - 1)
    end = min(len(source_lines), line + 1)
    rendered = []
    for idx in range(start, end + 1):
        marker = ">>" if idx == line else "  "
        rendered.append(f"{marker} {idx:4d} | {source_lines[idx - 1]}")
    return "\n".join(rendered)


def _graph_svg(chain: Optional[InferenceChain]) -> str:
    if not chain:
        return '<p class="muted">No inference-chain graph is available.</p>'
    width = max(760, 210 * len(chain.links))
    height = 170
    nodes = []
    edges = []
    for idx, link in enumerate(chain.links):
        x = 40 + idx * 200
        y = 60
        cls = "node failing" if link.is_failing else "node"
        label = f"{link.step_index}: {link.layer or link.op}"
        shape = link.output_shape
        nodes.append(
            f'<g class="{cls}">'
            f'<rect x="{x}" y="{y}" width="150" height="70" rx="12"></rect>'
            f'<text x="{x + 12}" y="{y + 28}">{_html(label)}</text>'
            f'<text class="small" x="{x + 12}" y="{y + 52}">{_html(shape)}</text>'
            "</g>"
        )
        if idx:
            x1 = 40 + (idx - 1) * 200 + 150
            x2 = x
            edges.append(
                f'<line x1="{x1}" y1="{y + 35}" x2="{x2}" y2="{y + 35}" marker-end="url(#arrow)"></line>'
            )
    return (
        f'<svg class="chain-graph" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="TensorGuard inference-chain graph">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z"></path>'
        '</marker></defs>'
        + "".join(edges)
        + "".join(nodes)
        + "</svg>"
    )


def _chain_cards(chain: Optional[InferenceChain], source: str = "") -> str:
    if not chain:
        return '<p class="muted">No chain was reconstructed. The bug list and counterexample still describe the run.</p>'
    source_lines = source.splitlines()
    cards = []
    for link in chain.links:
        fp = _footprint_for_link(link)
        status = str(fp.get("proof_status", "unknown"))
        confidence = str(fp.get("confidence", "heuristic"))
        inputs = "".join(
            f"<li><code>{_html(name)}</code>: {_html(shape)}</li>"
            for name, shape in zip(link.inputs, link.input_shapes)
        ) or "<li>No tensor inputs</li>"
        fail = ""
        if link.is_failing:
            fail = (
                '<div class="failure">'
                f"Expected {_html(link.expected_shape or '?')}, "
                f"got {_html(link.actual_shape or '?')}."
                "</div>"
            )
        snippet = _source_snippet(source_lines, link.line)
        snippet_html = f"<pre>{_html(snippet)}</pre>" if snippet else ""
        evidence = ", ".join(str(e) for e in fp.get("evidence", []) or [])
        cards.append(
            '<article class="step-card">'
            f'<header><h3>Step {link.step_index}: {_html(link.layer or link.op)}</h3>'
            f'<span class="badge status-{_html(status)}">{_html(status)}</span>'
            f'<span class="badge">{_html(confidence)}</span></header>'
            f"<p><strong>Line:</strong> {_html(link.line)} "
            f"<strong>Op:</strong> {_html(link.op)}</p>"
            f"<ul>{inputs}</ul>"
            f"<p><strong>Output:</strong> <code>{_html(link.output)}</code> = {_html(link.output_shape)}</p>"
            f"{fail}"
            f"<p class=\"muted\"><strong>Proof footprint:</strong> {_html(fp.get('rule', ''))}</p>"
            f"<p class=\"muted\"><strong>Evidence:</strong> {_html(evidence or 'manifest fallback')}</p>"
            f"{snippet_html}"
            "</article>"
        )
    return "\n".join(cards)


def _fixes_html(result: Any) -> str:
    fixes = list(getattr(result, "autofixes", []) or [])
    if not fixes:
        return "<p>No mechanical fix is available for this bug class.</p>"
    parts = ["<ol>"]
    for fix in fixes:
        parts.append(
            "<li>"
            f"<p>{_html(getattr(fix, 'description', ''))}</p>"
            f"<pre>- {_html(getattr(fix, 'original', '').strip())}\n"
            f"+ {_html(getattr(fix, 'suggested', '').strip())}</pre>"
            "</li>"
        )
    parts.append("</ol>")
    return "\n".join(parts)


def format_explain_html(result: Any, source: str = "", title: str = "TensorGuard Explain Report") -> str:
    """Render a self-contained HTML explanation report for a verifier result."""
    chain = getattr(result, "inference_chain", None)
    verdict = getattr(result, "verdict", "UNSAFE" if getattr(result, "bugs", []) else "SAFE")
    summary = getattr(chain, "summary", "") if chain else ""
    unknown_reasons = list(getattr(result, "unknown_reasons", []) or [])
    unknown_html = "".join(f"<li>{_html(r)}</li>" for r in unknown_reasons) or "<li>None</li>"
    css = """
    :root { color-scheme: light; --bg:#f8fafc; --fg:#111827; --muted:#64748b; --card:#ffffff; --line:#cbd5e1; --bad:#dc2626; --ok:#16a34a; --warn:#ca8a04; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--fg); background:var(--bg); }
    main { max-width:1120px; margin:0 auto; padding:32px 24px 56px; }
    .hero, .step-card, section { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px; margin:18px 0; box-shadow:0 8px 24px rgba(15,23,42,.06); }
    .hero { border-left:8px solid var(--bad); }
    .verdict-SAFE { border-left-color:var(--ok); }
    .verdict-UNKNOWN { border-left-color:var(--warn); }
    h1, h2, h3 { margin-top:0; }
    table { border-collapse:collapse; width:100%; }
    th, td { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
    code, pre { background:#0f172a; color:#e2e8f0; border-radius:8px; }
    code { padding:2px 5px; }
    pre { padding:12px; overflow:auto; }
    .chain-graph { width:100%; height:auto; min-height:160px; }
    .chain-graph line { stroke:#64748b; stroke-width:2; }
    .chain-graph rect { fill:#eef2ff; stroke:#6366f1; stroke-width:2; }
    .chain-graph .failing rect { fill:#fee2e2; stroke:var(--bad); }
    .chain-graph text { font-size:14px; fill:#111827; }
    .chain-graph .small { font-size:12px; fill:#475569; }
    .badge, .chip { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 9px; margin:2px 4px 2px 0; font-size:12px; background:#f1f5f9; }
    .status-lean_theorem { background:#dcfce7; border-color:#86efac; }
    .status-pen_and_paper_rule { background:#e0f2fe; border-color:#7dd3fc; }
    .status-tested_only_rule { background:#fef9c3; border-color:#fde047; }
    .status-heuristic, .status-unknown { background:#fee2e2; border-color:#fca5a5; }
    .step-card header { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
    .step-card header h3 { margin-right:auto; }
    .failure { border-left:4px solid var(--bad); background:#fef2f2; padding:10px; border-radius:8px; }
    .muted { color:var(--muted); }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html(title)}</title>
<style>{css}</style>
</head>
<body>
<main>
<div class="hero verdict-{_html(verdict)}">
<h1>{_html(title)}</h1>
<p><strong>Verdict:</strong> {_html(verdict)} | <strong>Bugs:</strong> {_html(len(getattr(result, "bugs", []) or []))} | <strong>Soundness mode:</strong> {_html(getattr(result, "soundness_mode", "balanced"))}</p>
<p>{_html(summary or "TensorGuard generated a verification report for this module.")}</p>
</div>
<section>
<h2>Inference-chain graph</h2>
{_graph_svg(chain)}
</section>
<section>
<h2>Counterexample</h2>
{_counterexample_html(getattr(result, "counterexample", None))}
</section>
<section>
<h2>Inference steps and proof-footprint badges</h2>
{_chain_cards(chain, source)}
</section>
<section>
<h2>Suggested fixes</h2>
{_fixes_html(result)}
</section>
<section>
<h2>Reported bugs</h2>
<table><thead><tr><th>Severity</th><th>Category</th><th>Location</th><th>Message</th></tr></thead><tbody>
{_bug_rows(result)}
</tbody></table>
</section>
<section>
<h2>Unknown reasons</h2>
<ul>{unknown_html}</ul>
</section>
</main>
</body>
</html>
"""


def write_explain_html(path: str | Path, result: Any, source: str = "", title: str = "TensorGuard Explain Report") -> Path:
    """Write :func:`format_explain_html` to *path* and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_explain_html(result, source=source, title=title), encoding="utf-8")
    return out
