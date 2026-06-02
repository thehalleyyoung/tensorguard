"""
training_loop_checks.py — static detection of training-loop hazards
(100_STEPS.md Step 96, Phase 10).

TensorGuard's architecture verifier reasons about a single ``nn.Module``'s
forward pass. The *training loop* around it is the other half of where real
PyTorch programs break, and those bugs are silent: a detached loss trains
nothing, a missing ``zero_grad`` accumulates gradients across steps, an fp16
autocast without a ``GradScaler`` underflows. This module adds a sound-leaning,
AST-based static analyzer for a curated set of well-defined training-step
hazards. Each finding carries a confidence tag so the permissive/strict split
mirrors the rest of the tool.

Hazard categories (see ``HazardKind``):

  * GRADIENT_FLOW_BREAK — the tensor passed to ``.backward()`` is computed via
    ``.detach()`` / ``.item()`` / ``.numpy()`` or inside a ``torch.no_grad()`` /
    ``inference_mode()`` block, so ``backward`` either raises or trains nothing.
    (sound: the dataflow into the loss is followed on the AST.)
  * MISSING_ZERO_GRAD — a loop performs ``loss.backward()`` / ``optimizer.step()``
    but never ``optimizer.zero_grad()`` / ``model.zero_grad()``, so gradients
    accumulate across iterations. (sound for the structural pattern.)
  * MISSING_OPTIMIZER_STEP — ``loss.backward()`` with no ``optimizer.step()``:
    gradients are computed but parameters never update. (sound.)
  * BACKWARD_BEFORE_ZERO_GRAD — ``zero_grad()`` is called *after* ``backward()``
    within the same step, wiping the freshly computed gradients before
    ``step()``. (sound for the ordering pattern.)
  * AMP_MISSING_GRAD_SCALER — an fp16 ``autocast`` region drives ``backward()``
    with no ``GradScaler`` anywhere, risking gradient underflow. (heuristic.)

The analyzer never executes the code; it parses it with ``ast``.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class HazardKind(str, Enum):
    GRADIENT_FLOW_BREAK = "gradient_flow_break"
    MISSING_ZERO_GRAD = "missing_zero_grad"
    MISSING_OPTIMIZER_STEP = "missing_optimizer_step"
    BACKWARD_BEFORE_ZERO_GRAD = "backward_before_zero_grad"
    AMP_MISSING_GRAD_SCALER = "amp_missing_grad_scaler"


class Confidence(str, Enum):
    SOUND = "sound"          # the pattern is a real bug whenever it matches
    HEURISTIC = "heuristic"  # likely a bug, but context could exonerate it


@dataclass(frozen=True)
class TrainingHazard:
    kind: HazardKind
    message: str
    lineno: int
    confidence: Confidence
    evidence: str = ""

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "lineno": self.lineno,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
        }


_GRAD_BREAKING_METHODS = {"detach", "item", "numpy", "tolist"}
_NO_GRAD_CTX = {"no_grad", "inference_mode"}


def _attr_chain(node: ast.AST) -> str:
    """Return a dotted name for an attribute/call target, best-effort."""
    if isinstance(node, ast.Attribute):
        return _attr_chain(node.value) + "." + node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _attr_chain(node.func)
    return ""


def _method_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


class _Analyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.backward_calls: List[ast.Call] = []
        self.step_calls: List[ast.Call] = []
        self.zero_grad_calls: List[ast.Call] = []
        self.assignments: Dict[str, ast.AST] = {}      # var -> RHS
        self.assign_in_nograd: Set[str] = set()         # vars assigned in no_grad
        self._nograd_depth = 0
        self.autocast_dtypes: List[str] = []
        self.has_grad_scaler = False
        self.uses_autocast = False
        self._has_update_call = False
        self._has_scale_call = False

    # -- context tracking --------------------------------------------------
    def visit_With(self, node: ast.With) -> None:
        opened = 0
        for item in node.items:
            ctx = item.context_expr
            name = _method_name(ctx) if isinstance(ctx, ast.Call) else None
            full = _attr_chain(ctx)
            if name in _NO_GRAD_CTX or full.split(".")[-1] in _NO_GRAD_CTX:
                opened += 1
            if "autocast" in full:
                self.uses_autocast = True
                if isinstance(ctx, ast.Call):
                    for kw in ctx.keywords:
                        if kw.arg == "dtype":
                            self.autocast_dtypes.append(_attr_chain(kw.value)
                                                        or "?")
        self._nograd_depth += opened
        self.generic_visit(node)
        self._nograd_depth -= opened

    # -- assignments -------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                self.assignments[tgt.id] = node.value
                if self._nograd_depth > 0:
                    self.assign_in_nograd.add(tgt.id)
        self.generic_visit(node)

    # -- calls -------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        m = _method_name(node)
        full = _attr_chain(node)
        if m == "backward":
            self.backward_calls.append(node)
        elif m == "step":
            recv = full.lower().rsplit(".step", 1)[0]
            if "scheduler" not in recv and "lr_sched" not in recv:
                # optimizer.step() or scaler.step(opt); not scheduler.step()
                self.step_calls.append(node)
        elif m == "zero_grad":
            self.zero_grad_calls.append(node)
        elif m == "update":
            self._has_update_call = True
        elif m == "scale":
            self._has_scale_call = True
        if "GradScaler" in full:
            self.has_grad_scaler = True
        self.generic_visit(node)


def _expr_breaks_grad(node: ast.AST, assignments: Dict[str, ast.AST],
                      nograd_vars: Set[str], seen: Set[str]) -> Optional[str]:
    """Walk the expression feeding the loss; return an evidence string if a
    gradient-breaking construct is found, else None. Follows simple
    var = expr assignment chains."""
    found: Optional[str] = None

    class _V(ast.NodeVisitor):
        def visit_Call(self, n: ast.Call) -> None:
            nonlocal found
            m = _method_name(n)
            if m in _GRAD_BREAKING_METHODS:
                found = found or f".{m}()"
            self.generic_visit(n)

        def visit_Name(self, n: ast.Name) -> None:
            nonlocal found
            if n.id in nograd_vars and found is None:
                found = "computed inside no_grad/inference_mode"
            if n.id in assignments and n.id not in seen:
                seen.add(n.id)
                sub = _expr_breaks_grad(assignments[n.id], assignments,
                                        nograd_vars, seen)
                if sub and found is None:
                    found = sub

    _V().visit(node)
    return found


def analyze_training_loop(source: str) -> List[TrainingHazard]:
    """Parse ``source`` and return the list of detected training-loop hazards.

    ``source`` may be a full module, a training function, or a loop body; the
    analyzer scans the whole tree for the curated structural patterns.
    """
    tree = ast.parse(source)
    a = _Analyzer()
    a.visit(tree)
    hazards: List[TrainingHazard] = []

    # 1. Gradient-flow break into backward().
    for call in a.backward_calls:
        target = call.func.value if isinstance(call.func, ast.Attribute) \
            else None
        if target is None:
            continue
        ev = _expr_breaks_grad(target, a.assignments, a.assign_in_nograd,
                               set())
        if ev:
            hazards.append(TrainingHazard(
                kind=HazardKind.GRADIENT_FLOW_BREAK,
                message=("the tensor passed to .backward() is detached from the "
                         f"autograd graph ({ev}); backward() raises or trains "
                         "nothing"),
                lineno=call.lineno,
                confidence=Confidence.SOUND,
                evidence=ev,
            ))

    # 2. Missing zero_grad when there is a backward/step.
    if a.backward_calls and not a.zero_grad_calls:
        hazards.append(TrainingHazard(
            kind=HazardKind.MISSING_ZERO_GRAD,
            message=("loss.backward() is called but optimizer.zero_grad() / "
                     "model.zero_grad() never is; gradients accumulate across "
                     "iterations"),
            lineno=a.backward_calls[0].lineno,
            confidence=Confidence.SOUND,
            evidence="no zero_grad() call in scope",
        ))

    # 3. Missing optimizer.step().
    if a.backward_calls and not a.step_calls:
        hazards.append(TrainingHazard(
            kind=HazardKind.MISSING_OPTIMIZER_STEP,
            message=("loss.backward() computes gradients but optimizer.step() "
                     "is never called; parameters never update"),
            lineno=a.backward_calls[0].lineno,
            confidence=Confidence.SOUND,
            evidence="no optimizer.step() call in scope",
        ))

    # 4. zero_grad() ordered AFTER backward() (and before step), wiping grads.
    if a.backward_calls and a.zero_grad_calls and a.step_calls:
        bwd = min(c.lineno for c in a.backward_calls)
        stp = max(c.lineno for c in a.step_calls)
        zg_after = [z for z in a.zero_grad_calls if bwd < z.lineno <= stp]
        if zg_after and not any(z.lineno < bwd for z in a.zero_grad_calls):
            hazards.append(TrainingHazard(
                kind=HazardKind.BACKWARD_BEFORE_ZERO_GRAD,
                message=("zero_grad() is called after backward() and before "
                         "step(), erasing the gradients just computed"),
                lineno=zg_after[0].lineno,
                confidence=Confidence.SOUND,
                evidence=f"backward@L{bwd} < zero_grad@L{zg_after[0].lineno} "
                         f"<= step@L{stp}",
            ))

    # 5. fp16 autocast driving backward with no GradScaler (heuristic).
    scaler_in_use = (a.has_grad_scaler
                     or (a._has_scale_call and a._has_update_call))
    if a.uses_autocast and a.backward_calls and not scaler_in_use:
        fp16 = any("float16" in d or d.endswith(".half") or d == "torch.float16"
                   or "half" in d for d in a.autocast_dtypes)
        # default autocast dtype on CUDA is float16; flag unless bfloat16 named.
        bf16_named = any("bfloat16" in d for d in a.autocast_dtypes)
        if (fp16 or not a.autocast_dtypes) and not bf16_named:
            hazards.append(TrainingHazard(
                kind=HazardKind.AMP_MISSING_GRAD_SCALER,
                message=("an fp16 autocast region drives backward() but no "
                         "torch.cuda.amp.GradScaler is used; small gradients "
                         "can underflow to zero"),
                lineno=a.backward_calls[0].lineno,
                confidence=Confidence.HEURISTIC,
                evidence="autocast + backward, no GradScaler",
            ))

    hazards.sort(key=lambda h: (h.lineno, h.kind.value))
    return hazards


def summarize(source: str) -> Dict:
    hazards = analyze_training_loop(source)
    return {
        "n_hazards": len(hazards),
        "sound": sum(1 for h in hazards if h.confidence is Confidence.SOUND),
        "heuristic": sum(1 for h in hazards
                         if h.confidence is Confidence.HEURISTIC),
        "hazards": [h.to_dict() for h in hazards],
    }
