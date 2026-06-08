"""Static scanner for the PyTorch loss **value-domain refinement** contract.

This is the first contract in the lattice about the *numeric value domain* of a
tensor rather than its membership (`split_disjointness`, `group_disjointness`),
order (`temporal_causality`), effect (`sampling_independence`), or relational
multiplicity (`join_cardinality`).  It is the deliberate **complement to a
shape/dtype verifier** such as `thehalleyyoung/tensorguard`: TensorGuard proves
the *shape* refinement ``{v : Tensor | shape(v) == (B, C)}`` of a loss call with
shape-only models, but -- being shape-only -- it can never see the *value
domain* of the operand.  DataRefine fills exactly that gap, proving the
*value-domain* refinement ``{v | 0 <= v <= 1}`` of the **same** call site.  The
two compose with **no overlap**: TensorGuard certifies shapes/dtypes/reduction,
DataRefine certifies the numeric domain, and together they fully certify the
loss invocation.

Several PyTorch losses carry an unstated precondition on the *value domain* of
their input that, when violated, yields a silently-wrong loss with **no
exception** -- the runtime-silent footgun this contract targets:

* **V1 --- ``BCELoss`` / ``binary_cross_entropy`` on un-sigmoided logits.**
  The loss expects probabilities in ``[0, 1]``; fed a raw activation (logits) it
  computes a meaningless loss and unstable gradients.  The fix is
  ``BCEWithLogitsLoss`` (numerically stable, takes logits) or an explicit
  ``sigmoid`` -- the textbook ``BCELoss`` vs ``BCEWithLogitsLoss`` mistake.

* **V2 --- ``NLLLoss`` / ``nll_loss`` on non-log-probabilities.**
  The loss expects *log*-probabilities (``<= 0``, from ``log_softmax``); fed a
  ``softmax`` (probabilities ``> 0``) or a ``sigmoid`` output it silently
  optimises nonsense.  The fix is ``log_softmax`` (or ``CrossEntropyLoss``,
  which fuses ``log_softmax + nll_loss``).

As with every other scanner we **do not model Python / tensor semantics**: the
input's value domain is read from *local, syntactic* producer signals --- the
op that produced the input (``sigmoid`` / ``clamp(0, 1)`` -> ``[0, 1]``;
``log_softmax`` / ``log`` -> log-probs; ``softmax`` -> probs; a raw ``matmul`` /
``@`` -> logits) plus conventional variable-name hints (``logits`` is raw,
``probs`` is normalised).  Each candidate is lowered to a ``value_domain``
obligation that the z3-backed :class:`~datarefine.certification.StructuralCertifier`
decides over LRA (returning a concrete out-of-domain witness value), with an
independent re-check.

The precision boundary is honest and identical in spirit to the other scanners:
this is a *recall* (definite-signal) recognizer, not a soundness oracle.  It
fires only when the input's producer is *recognisably* outside the required
domain (a raw matmul / ``log_softmax`` feeding ``BCELoss``; a ``softmax``
feeding ``NLLLoss``; or a logit-named variable); a bare model output whose
activation lives inside an opaque ``forward`` is the honestly-stated recall gap
(classified ``unknown`` and never flagged).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .certification import StructuralCertifier
from .obligations import obligation

__all__ = [
    "ValueDomainFinding",
    "ValueDomainScanner",
    "scan_source",
    "scan_path",
    "scan_tree",
]

_CERT = StructuralCertifier()

# Functional losses whose input carries a value-domain precondition.
#   name -> (op, required_lo, required_hi)
_FUNCTIONAL_LOSSES = {
    "binary_cross_entropy": ("bce", 0.0, 1.0),
    "nll_loss": ("nll", None, 0.0),
}
# Module classes with the same preconditions (instance-call form).
_MODULE_LOSSES = {
    "BCELoss": ("bce", 0.0, 1.0),
    "NLLLoss": ("nll", None, 0.0),
}
# Losses that are *correct* by construction (handle the domain internally) --
# never flagged, so the contract does not collide with their valid use.
_SAFE_LOSSES = frozenset({
    "binary_cross_entropy_with_logits", "BCEWithLogitsLoss",
    "cross_entropy", "CrossEntropyLoss",
})

# Value-changing producers (anything not listed is a value-preserving passthrough
# we recurse through: dtype/device/shape ops do not change the value domain).
_PASSTHROUGH = frozenset({
    "float", "double", "half", "to", "detach", "clone", "contiguous",
    "cpu", "cuda", "type", "type_as", "view", "reshape", "flatten", "ravel",
    "squeeze", "unsqueeze", "t", "transpose", "permute", "reshape_as",
    "view_as", "requires_grad_",
})
_SIGMOID = frozenset({"sigmoid"})
_SOFTMAX = frozenset({"softmax"})
_LOG_SOFTMAX = frozenset({"log_softmax"})
_LOG = frozenset({"log"})
_MATMUL = frozenset({"matmul", "mm", "bmm", "addmm", "linear"})

# Conventional names for raw logits vs normalised probabilities / log-probs.
_LOGIT_NAMES = frozenset({
    "logit", "logits", "score", "scores", "raw", "z", "logits_", "pred_logits",
    "y_logits", "out_logits",
})
_PROB_NAMES = frozenset({
    "prob", "probs", "proba", "probas", "p", "p_hat", "phat", "yhat_prob",
    "y_prob", "y_proba", "prediction_prob",
})
_LOGPROB_NAMES = frozenset({
    "log_prob", "log_probs", "logp", "logprob", "logprobs", "log_softmax_out",
    "logp_hat",
})


@dataclass
class ValueDomainFinding:
    file: str
    line: int
    pattern: str
    constraint: str
    verdict: str
    detail: str
    snippet: str
    loss: str
    op: str
    producer: str
    witness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _clamps_to_unit(call: ast.Call) -> bool:
    """True if a ``clamp`` / ``clip`` call pins the value into ``[0, 1]``."""
    lows: list[float] = []
    highs: list[float] = []
    for kw in call.keywords:
        if kw.arg in ("min", "a_min") and isinstance(kw.value, ast.Constant):
            lows.append(float(kw.value.value) if isinstance(kw.value.value, (int, float)) else 99)
        if kw.arg in ("max", "a_max") and isinstance(kw.value, ast.Constant):
            highs.append(float(kw.value.value) if isinstance(kw.value.value, (int, float)) else 99)
    consts = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, (int, float))]
    vals = lows + highs + [float(c) for c in consts]
    has_zero = any(v == 0 for v in vals)
    has_one = any(v == 1 for v in vals)
    return has_zero and has_one


class ValueDomainScanner:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self.findings: list[ValueDomainFinding] = []
        self._assign: dict[str, ast.AST] = {}
        self._loss_vars: dict[str, tuple[str, object, object]] = {}

    def _certify(self, **payload) -> tuple[str, dict | None]:
        ob = obligation("domain", self.file, "value_domain",
                        constraint="value_domain", **payload)
        verdict = _CERT.certify(ob)
        witness = None
        diags = verdict.diagnostics or ()
        if diags and isinstance(diags[0], dict):
            witness = diags[0].get("model")
        return verdict.status, witness

    def _snippet(self, line: int) -> str:
        return self.src_lines[line - 1].strip() if 0 < line <= len(self.src_lines) else ""

    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is not None:
                for t in node.targets:
                    name = _name_of(t)
                    if name is None:
                        continue
                    self._assign[name] = node.value
                    # Track ``crit = nn.BCELoss()`` style loss-module bindings.
                    if isinstance(node.value, ast.Call):
                        cls = _name_of(node.value.func)
                        if cls in _MODULE_LOSSES:
                            self._loss_vars[name] = (cls, *_MODULE_LOSSES[cls][1:])

    # -- producer classification -------------------------------------------
    def _producer(self, node: ast.AST | None, depth: int = 0) -> str:
        """A coarse label for the value-domain-relevant producer of ``node``."""
        if node is None or depth > 6:
            return "unknown"
        # value-preserving passthroughs: recurse on the receiver.
        if isinstance(node, ast.Call):
            fname = _name_of(node.func)
            if fname in _PASSTHROUGH:
                recv = node.func.value if isinstance(node.func, ast.Attribute) else (
                    node.args[0] if node.args else None)
                return self._producer(recv, depth + 1)
            if fname in _SIGMOID:
                return "sigmoid"
            if fname in _LOG_SOFTMAX:
                return "log_softmax"
            if fname in _SOFTMAX:
                return "softmax"
            if fname in _LOG:
                return "log"
            if fname in ("clamp", "clip"):
                return "clamp01" if _clamps_to_unit(node) else "clamp"
            if fname in _MATMUL:
                return "matmul"
            return "unknown"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return "matmul"
        if isinstance(node, ast.Name):
            base = node.id.lower()
            if base in _LOGIT_NAMES:
                return "logit_name"
            if base in _PROB_NAMES:
                return "prob_name"
            if base in _LOGPROB_NAMES:
                return "logprob_name"
            # one-hop resolution through a local assignment
            if depth == 0 and node.id in self._assign:
                return self._producer(self._assign[node.id], depth + 1)
        return "unknown"

    @staticmethod
    def _classify(op: str, producer: str) -> str:
        """Map a producer to ``normalized`` / ``raw`` / ``unknown`` for ``op``."""
        if op == "bce":
            if producer in ("sigmoid", "clamp01", "softmax", "prob_name"):
                return "normalized"
            if producer in ("matmul", "log_softmax", "log", "logit_name", "logprob_name"):
                return "raw"
            return "unknown"
        if op == "nll":
            if producer in ("log_softmax", "log", "logprob_name"):
                return "normalized"
            if producer in ("softmax", "sigmoid", "clamp01", "prob_name", "logit_name", "matmul"):
                return "raw"
            return "unknown"
        return "unknown"

    # -- loss-site recognition ---------------------------------------------
    def _loss_site(self, call: ast.Call):
        """Return ``(loss_name, op, required_lo, required_hi, input_node)`` for a
        value-domain-bearing loss call, or ``None``."""
        fname = _name_of(call.func)
        if fname in _SAFE_LOSSES:
            return None
        # functional form: F.binary_cross_entropy(input, target, ...)
        if fname in _FUNCTIONAL_LOSSES and call.args:
            op, lo, hi = _FUNCTIONAL_LOSSES[fname]
            return fname, op, lo, hi, call.args[0]
        # inline module form: nn.BCELoss()(input, target)
        if isinstance(call.func, ast.Call):
            cls = _name_of(call.func.func)
            if cls in _MODULE_LOSSES and call.args:
                op, lo, hi = _MODULE_LOSSES[cls]
                return cls, op, lo, hi, call.args[0]
        # bound module form: crit(input, target) where crit = nn.BCELoss()
        if isinstance(call.func, ast.Name) and call.func.id in self._loss_vars and call.args:
            cls, lo, hi = self._loss_vars[call.func.id]
            op = _MODULE_LOSSES[cls][0]
            return cls, op, lo, hi, call.args[0]
        return None

    def scan(self) -> list[ValueDomainFinding]:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return []
        self._collect(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            site = self._loss_site(node)
            if site is None:
                continue
            loss, op, lo, hi, inp = site
            producer = self._producer(inp)
            klass = self._classify(op, producer)
            if klass == "unknown":
                continue  # precision-first: never flag an opaque producer
            established = klass == "normalized"
            status, witness = self._certify(
                op=op, loss=loss, producer=producer,
                domain_established=established, required_lo=lo, required_hi=hi)
            if status != "rejected":
                continue
            line = getattr(node, "lineno", 0)
            if op == "bce":
                pattern = "V1:bce_on_unnormalized_logits"
                detail = (
                    f"{loss} expects probabilities in [0, 1] but its input is "
                    f"produced by {producer!r} (not a sigmoid/clamp); feeding raw "
                    f"logits silently computes a wrong, unstable loss. Use "
                    f"BCEWithLogitsLoss (stable, takes logits) or apply "
                    f"torch.sigmoid first."
                )
            else:
                pattern = "V2:nll_on_non_log_probabilities"
                detail = (
                    f"{loss} expects log-probabilities (<= 0) but its input is "
                    f"produced by {producer!r} (probabilities/raw, not "
                    f"log_softmax); the loss is silently wrong. Use "
                    f"F.log_softmax(..., dim=-1) (or CrossEntropyLoss, which "
                    f"fuses log_softmax + nll_loss)."
                )
            self.findings.append(ValueDomainFinding(
                file=self.file, line=line, pattern=pattern,
                constraint="value_domain", verdict=status, detail=detail,
                snippet=self._snippet(line), loss=loss, op=op,
                producer=producer, witness=witness))
        return self.findings


def scan_source(source: str, filename: str = "<string>") -> list[ValueDomainFinding]:
    return ValueDomainScanner(source, filename).scan()


def scan_path(path: str | Path) -> list[ValueDomainFinding]:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_source(source, filename=str(p))


def scan_tree(root: str | Path) -> list[ValueDomainFinding]:
    root = Path(root)
    out: list[ValueDomainFinding] = []
    paths: Iterable[Path] = [root] if root.is_file() else root.rglob("*.py")
    for p in paths:
        out.extend(scan_path(p))
    return out
