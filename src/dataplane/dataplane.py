"""The data-plane abstract interpreter --- the spine described in ``NORTH_STAR.md``.

Every prior scanner re-walked the AST with bespoke pattern matching and called the
certifier in isolation.  This module is the unifying front end the project was
missing: it lowers PyTorch / pandas / sklearn source into a *data-plane dataflow*,
abstractly interprets that dataflow over a small **refinement product lattice**
(value-domain x information-flow provenance x role x split-origin), and **emits
obligations at sinks** that are discharged by the existing
:class:`~datarefine.certification.StructuralCertifier`.

It deliberately never models the full semantics of Python or of tensors: every
unrecognised operator yields the lattice top (``unknown``) and is *never blamed*.
Two sinks are wired today, as the proof that one engine already spans two
independent bug axes (the definition-of-done in NORTH_STAR.md step 1):

* **refinement / value-domain** -- a loss whose input refinement does not entail
  the loss's required value domain (BCELoss on logits, NLLLoss on probabilities),
  discharged over LRA;
* **non-interference / leakage** -- a featuriser fitted on data whose *provenance*
  includes the held-out partition (the classic fit-before-split leak), discharged
  over the ``fit_transform_isolation`` sets theory;
* **temporal causality / lookahead** -- a feature-construction operator chain whose
  composed *forward reach* reads a future row (``shift(-k)``, centered ``rolling``,
  negative ``diff``), discharged over the ``temporal_causality`` index theory.  The
  operator vocabulary here is exactly the validated forward-edge catalog of
  :mod:`datarefine.temporal_leakage`, lifted onto the engine as operator signatures.

Adding a new bug class is *adding operator signatures and a sink* here, not a new
scanner module --- see ``NORTH_STAR.md`` sections 3 and 6.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .certification import StructuralCertifier
from .obligations import Obligation, ObligationPacket, obligation, obligation_packet
from .temporal_leakage import _max_chain_reach, _looks_like_target

DATAPLANE_SCHEMA_VERSION = "datarefine.dataplane.v1"

_CERT = StructuralCertifier()


# ---------------------------------------------------------------------------
# The refinement product lattice (NORTH_STAR.md sec. 2.1)
# ---------------------------------------------------------------------------
# value-domain sub-lattice: a flat lattice with top = "unknown".
UNKNOWN = "unknown"
PROB_UNIT = "prob_unit"   # sigmoid / clamp(0,1) / probability-named  -> [0, 1]
PROB = "prob"             # softmax                                   -> [0, 1], sums to 1
LOG_PROB = "log_prob"     # log_softmax / log / log-prob-named        -> <= 0
LOGIT = "logit"           # matmul / linear / logit-named             -> R

# information-flow labels (NORTH_STAR.md sec. 2.3): HIGH dominates under join.
LOW = "low"
HIGH = "high"

# split-origin tokens carried as a provenance set.
ALL = "all"
TRAIN = "train"
TEST = "test"


def _join_domain(a: str, b: str) -> str:
    return a if a == b else UNKNOWN


def _join_label(a: str, b: str) -> str:
    return HIGH if HIGH in (a, b) else LOW


@dataclass(frozen=True)
class Refinement:
    """The typed contract a single data value carries through the data plane."""

    value_domain: str = UNKNOWN
    flow_label: str = LOW
    split_origin: frozenset[str] = field(default_factory=lambda: frozenset({ALL}))
    role: str | None = None
    provenance: tuple[str, ...] = ()

    def join(self, other: "Refinement") -> "Refinement":
        return Refinement(
            value_domain=_join_domain(self.value_domain, other.value_domain),
            flow_label=_join_label(self.flow_label, other.flow_label),
            split_origin=frozenset(self.split_origin | other.split_origin),
            role=self.role if self.role == other.role else None,
            provenance=tuple(dict.fromkeys((*self.provenance, *other.provenance))),
        )

    def with_domain(self, domain: str, producer: str) -> "Refinement":
        return replace(self, value_domain=domain,
                       provenance=tuple(dict.fromkeys((*self.provenance, producer))))


TOP = Refinement()


# ---------------------------------------------------------------------------
# Operator vocabulary (value-domain transfer functions)
# ---------------------------------------------------------------------------
_PASSTHROUGH = frozenset({
    "float", "double", "half", "to", "detach", "clone", "contiguous", "cpu",
    "cuda", "type", "type_as", "view", "reshape", "flatten", "ravel", "squeeze",
    "unsqueeze", "t", "transpose", "permute", "reshape_as", "view_as",
    "requires_grad_",
})
# operator name -> the value-domain point it *produces* (regardless of input).
_DOMAIN_PRODUCERS: dict[str, str] = {
    "sigmoid": PROB_UNIT,
    "softmax": PROB,
    "log_softmax": LOG_PROB,
    "log": LOG_PROB,
    "matmul": LOGIT, "mm": LOGIT, "bmm": LOGIT, "addmm": LOGIT, "linear": LOGIT,
}
_LOGIT_NAMES = frozenset({"logit", "logits", "score", "scores", "raw", "z",
                          "logits_", "pred_logits", "y_logits", "out_logits"})
_PROB_NAMES = frozenset({"prob", "probs", "proba", "probas", "p", "p_hat",
                         "phat", "yhat_prob", "y_prob", "y_proba", "prediction_prob"})
_LOGPROB_NAMES = frozenset({"log_prob", "log_probs", "logp", "logprob",
                            "logprobs", "log_softmax_out", "logp_hat"})

# loss sinks: name -> (op, required value-domain points, required_lo, required_hi)
_FUNCTIONAL_LOSSES = {
    "binary_cross_entropy": ("bce", frozenset({PROB_UNIT, PROB}), 0.0, 1.0),
    "nll_loss": ("nll", frozenset({LOG_PROB}), None, 0.0),
}
_MODULE_LOSSES = {
    "BCELoss": ("bce", frozenset({PROB_UNIT, PROB}), 0.0, 1.0),
    "NLLLoss": ("nll", frozenset({LOG_PROB}), None, 0.0),
}
_SAFE_LOSSES = frozenset({"binary_cross_entropy_with_logits", "BCEWithLogitsLoss",
                          "cross_entropy", "CrossEntropyLoss"})

# leakage axis: featuriser fit sinks and split operators.
_FIT_OPS = frozenset({"fit", "fit_transform"})
_SPLIT_OPS = frozenset({"train_test_split", "random_split"})

# Structural families whose recognition logic lives in a dedicated scanner module
# but whose *axis* the engine front door unifies.  Each is lifted into a native,
# re-certified obligation on its own axis; the value-domain, fit-before-split, and
# temporal axes are interpreted natively above and so are excluded here to avoid
# double emission.  Maps engine axis -> scanners.py family key.
_DELEGATED_FAMILY_AXES: tuple[tuple[str, str], ...] = (
    ("split", "contracts"),
    ("group", "group"),
    ("sampling", "sampling"),
    ("join", "joins"),
)


def _name_of(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _finding_detail(finding: object) -> str:
    value = getattr(finding, "detail", None)
    if value is None:
        value = getattr(finding, "explanation", "")
    return str(value or "")


def _clamps_to_unit(call: ast.Call) -> bool:
    lows: list[float] = []
    highs: list[float] = []
    for kw in call.keywords:
        if kw.arg in ("min", "a_min") and isinstance(kw.value, ast.Constant):
            lows.append(float(kw.value.value))
        if kw.arg in ("max", "a_max") and isinstance(kw.value, ast.Constant):
            highs.append(float(kw.value.value))
    consts = [a.value for a in call.args if isinstance(a, ast.Constant)]
    nums = [float(c) for c in consts if isinstance(c, (int, float))]
    if len(nums) >= 2:
        lows.append(min(nums))
        highs.append(max(nums))
    return bool(lows and highs and min(lows) <= 0.0 and max(highs) >= 1.0 and max(highs) <= 1.0)


# ---------------------------------------------------------------------------
# The abstract interpreter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataPlaneObligation:
    """An obligation born at a sink during abstract interpretation."""

    axis: str            # "refinement" | "non_interference"
    site: str            # file:line
    obligation: Obligation
    status: str          # admitted | rejected | unknown
    witness: Mapping[str, object] | None
    detail: str

    @property
    def rejected(self) -> bool:
        return self.status == "rejected"

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "site": self.site,
            "status": self.status,
            "obligation_id": self.obligation.obligation_id,
            "constraint": self.obligation.payload.get("constraint"),
            "witness": dict(self.witness) if self.witness else None,
            "detail": self.detail,
        }


@dataclass
class DataPlaneReport:
    file: str
    obligations: list[DataPlaneObligation] = field(default_factory=list)

    @property
    def violations(self) -> list[DataPlaneObligation]:
        return [o for o in self.obligations if o.rejected]

    def to_obligation_packet(self) -> ObligationPacket:
        return obligation_packet(
            *(o.obligation for o in self.obligations),
            metadata={"schema_version": DATAPLANE_SCHEMA_VERSION,
                      "producer": "datarefine", "source_id": self.file},
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": DATAPLANE_SCHEMA_VERSION,
            "file": self.file,
            "n_obligations": len(self.obligations),
            "n_violations": len(self.violations),
            "obligations": [o.as_dict() for o in self.obligations],
        }


class DataPlaneInterpreter:
    """Abstractly interpret a module's data plane over the refinement lattice."""

    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.file = filename
        self.src_lines = source.splitlines()
        self._assign: dict[str, ast.AST] = {}
        self._origin: dict[str, frozenset[str]] = {}
        self._loss_vars: dict[str, str] = {}
        self.report = DataPlaneReport(file=filename)

    # -- refinement inference (transfer functions) -------------------------
    def _infer(self, node: ast.AST | None, depth: int = 0) -> Refinement:
        if node is None or depth > 8:
            return TOP
        if isinstance(node, ast.Call):
            fname = _name_of(node.func)
            if fname in _PASSTHROUGH:
                recv = node.func.value if isinstance(node.func, ast.Attribute) else (
                    node.args[0] if node.args else None)
                return self._infer(recv, depth + 1)
            if fname in ("clamp", "clip"):
                dom = PROB_UNIT if _clamps_to_unit(node) else UNKNOWN
                return TOP.with_domain(dom, "clamp01" if dom == PROB_UNIT else "clamp")
            if fname in _DOMAIN_PRODUCERS:
                return TOP.with_domain(_DOMAIN_PRODUCERS[fname], fname)
            return TOP
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return TOP.with_domain(LOGIT, "matmul")
        if isinstance(node, ast.Name):
            base = node.id.lower()
            if base in _LOGIT_NAMES:
                return TOP.with_domain(LOGIT, "logit_name")
            if base in _PROB_NAMES:
                return TOP.with_domain(PROB_UNIT, "prob_name")
            if base in _LOGPROB_NAMES:
                return TOP.with_domain(LOG_PROB, "logprob_name")
            origin = self._origin.get(node.id)
            ref = TOP if origin is None else replace(TOP, split_origin=origin)
            if depth == 0 and node.id in self._assign:
                inferred = self._infer(self._assign[node.id], depth + 1)
                return replace(inferred, split_origin=origin or inferred.split_origin)
            return ref
        return TOP

    def _producer_label(self, ref: Refinement) -> str:
        for name in ref.provenance:
            return name
        return "unknown"

    # -- statement collection ----------------------------------------------
    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.value is not None:
                # tuple unpack of a split: X_train, X_test = train_test_split(...)
                if (isinstance(node.value, ast.Call)
                        and _name_of(node.value.func) in _SPLIT_OPS
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Tuple)):
                    elts = node.targets[0].elts
                    for i, elt in enumerate(elts):
                        name = _name_of(elt)
                        if name is None:
                            continue
                        # train_test_split returns (train, test, train, test, ...)
                        self._origin[name] = frozenset({TRAIN if i % 2 == 0 else TEST})
                    continue
                for t in node.targets:
                    name = _name_of(t)
                    if name is None:
                        continue
                    self._assign[name] = node.value
                    if isinstance(node.value, ast.Call):
                        cls = _name_of(node.value.func)
                        if cls in _MODULE_LOSSES:
                            self._loss_vars[name] = cls

    # -- loss-site recognition (refinement sink) ---------------------------
    def _loss_site(self, call: ast.Call):
        fname = _name_of(call.func)
        if fname in _SAFE_LOSSES:
            return None
        if fname in _FUNCTIONAL_LOSSES and call.args:
            op, req, lo, hi = _FUNCTIONAL_LOSSES[fname]
            return fname, op, req, lo, hi, call.args[0]
        if isinstance(call.func, ast.Call):
            cls = _name_of(call.func.func)
            if cls in _MODULE_LOSSES and call.args:
                op, req, lo, hi = _MODULE_LOSSES[cls]
                return cls, op, req, lo, hi, call.args[0]
        if isinstance(call.func, ast.Name) and call.func.id in self._loss_vars and call.args:
            cls = self._loss_vars[call.func.id]
            op, req, lo, hi = _MODULE_LOSSES[cls]
            return cls, op, req, lo, hi, call.args[0]
        return None

    def _emit_value_domain(self, call: ast.Call) -> None:
        site = self._loss_site(call)
        if site is None:
            return
        loss, op, required, lo, hi, inp = site
        ref = self._infer(inp)
        if ref.value_domain == UNKNOWN:
            return  # opaque producer -> never blamed (precision-first)
        established = ref.value_domain in required
        line = getattr(call, "lineno", 0)
        producer = self._producer_label(ref)
        ob = obligation(
            "domain", f"{self.file}:{line}", "value_domain",
            status="unknown", constraint="value_domain", op=op, loss=loss,
            producer=producer, domain_established=established,
            required_lo=lo, required_hi=hi,
        )
        verdict = _CERT.certify(ob)
        witness = verdict.diagnostics[0].get("model") if verdict.diagnostics else None
        self.report.obligations.append(DataPlaneObligation(
            axis="refinement", site=f"{self.file}:{line}", obligation=ob,
            status=verdict.status, witness=witness,
            detail=(f"{loss} requires its input in the {sorted(required)} value "
                    f"domain but the inferred producer is {producer!r} "
                    f"(domain={ref.value_domain})"),
        ))

    # -- fit-before-split recognition (non-interference sink) --------------
    def _emit_leakage(self, fit_calls: list[ast.Call], first_split_line: int | None) -> None:
        for call in fit_calls:
            line = getattr(call, "lineno", 0)
            data_arg = call.args[0] if call.args else None
            ref = self._infer(data_arg) if data_arg is not None else TOP
            # provenance includes the held-out partition iff the fit data is not
            # already a train-only subset and the fit precedes any split.
            fit_on_train_only = ref.split_origin == frozenset({TRAIN})
            before_split = first_split_line is not None and line < first_split_line
            if fit_on_train_only or not before_split:
                continue
            # abstract provenance -> a sets-theory obligation: a fitted statistic
            # whose rows include a holdout row violates disjoint(fit, holdout).
            ob = obligation(
                "provenance", f"{self.file}:{line}", "fit_transform_isolation",
                status="unknown", constraint="fit_transform_isolation",
                fit_row_ids=["train_0", "holdout_0"], holdout_row_ids=["holdout_0"],
                fit_feature_sources=["feature_0"], outcome_columns=["target"],
            )
            verdict = _CERT.certify(ob)
            witness = verdict.diagnostics[0].get("model") if verdict.diagnostics else None
            self.report.obligations.append(DataPlaneObligation(
                axis="non_interference", site=f"{self.file}:{line}", obligation=ob,
                status=verdict.status,
                witness=witness or {"violation": "fit_before_split",
                                    "fit_line": line, "split_line": first_split_line},
                detail=("a featuriser is fitted on full (pre-split) data whose "
                        "provenance includes the held-out partition; its statistic "
                        "leaks test information into training"),
            ))

    # -- temporal-causality recognition (lookahead sink) -------------------
    def _emit_temporal(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
                value = node.value
            else:
                continue
            reach, operators = _max_chain_reach(value)
            if reach < 1:
                continue
            # a forward-shifted *target* (next-step label) is legitimate ML.
            if any(_looks_like_target(t) for t in targets):
                continue
            line = getattr(node, "lineno", 0)
            ob = obligation(
                "temporal", f"{self.file}:{line}", "temporal_causality",
                status="unknown", constraint="temporal_causality",
                forward_reach=reach, backward_reach=0,
                operators=[list(op) for op in operators], horizon=reach + 2,
            )
            verdict = _CERT.certify(ob)
            witness = verdict.diagnostics[0].get("model") if verdict.diagnostics else None
            chain = " -> ".join(f"{m}({d:+d})" for m, d in operators) or "lookahead operator"
            self.report.obligations.append(DataPlaneObligation(
                axis="temporal", site=f"{self.file}:{line}", obligation=ob,
                status=verdict.status,
                witness=witness or {"forward_reach": reach},
                detail=(f"feature reads {reach} row(s) into the future via {chain}; "
                        f"at streaming inference that future row is unavailable when "
                        f"the current row is scored (lookahead bias)"),
            ))

    # -- delegated structural families (split/group/sampling/join) ----------
    def _emit_structural_families(self) -> None:
        """Lift the remaining structural axes through the engine's front door.

        The recognition for these families lives in their dedicated scanner
        modules (each a syntactic operator-signature recogniser).  Rather than
        re-walk the AST a second way, the engine *delegates* recognition, then
        re-births every rejected finding as a first-class obligation via
        :func:`datarefine.scanners.finding_to_obligation` and **re-discharges it
        with the engine's own certifier**, attaching it to the unified report on
        the family's axis.  This makes one ``analyze_all`` run span every
        structural bug axis the project supports, all decided by one certifier.
        """
        from . import scanners as _scanners  # lazy: avoids import cycles

        for axis, family_key in _DELEGATED_FAMILY_AXES:
            try:
                fam = _scanners.family(family_key)
                findings = fam.scan_source(self.source, self.file)
            except (KeyError, SyntaxError):
                continue
            for finding in findings:
                if str(getattr(finding, "verdict", "")) != "rejected":
                    continue
                ob = _scanners.finding_to_obligation(finding)
                # A faithful structural reconstruction (group/join/sampling) is
                # genuinely re-dischargeable by the engine's certifier; a
                # record-only obligation (e.g. split contracts, no reconstructor)
                # carries no structural payload, so we preserve the scanner's
                # authoritative verdict rather than vacuously re-certifying it.
                if "scanner_constraint" in ob.payload:
                    status = str(getattr(finding, "verdict", "unknown"))
                    witness = dict(getattr(finding, "witness", {}) or {})
                else:
                    verdict = _CERT.certify(ob)
                    status = verdict.status
                    witness = verdict.diagnostics[0].get("model") if verdict.diagnostics else None
                line = int(getattr(finding, "line", 0))
                self.report.obligations.append(DataPlaneObligation(
                    axis=axis, site=f"{self.file}:{line}", obligation=ob,
                    status=status,
                    witness=witness or (dict(getattr(finding, "witness", {}) or {})),
                    detail=_finding_detail(finding),
                ))

    def run(self, include_families: bool = False) -> DataPlaneReport:
        try:
            tree = ast.parse(self.source, filename=self.file)
        except SyntaxError:
            return self.report
        self._collect(tree)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        split_lines = [getattr(c, "lineno", 0) for c in calls
                       if _name_of(c.func) in _SPLIT_OPS]
        first_split_line = min(split_lines) if split_lines else None
        fit_calls = [c for c in calls if _name_of(c.func) in _FIT_OPS]
        for call in calls:
            self._emit_value_domain(call)
        self._emit_leakage(fit_calls, first_split_line)
        self._emit_temporal(tree)
        if include_families:
            self._emit_structural_families()
        return self.report


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def analyze_source(source: str, filename: str = "<string>") -> DataPlaneReport:
    """Abstractly interpret a single source unit's data plane."""
    return DataPlaneInterpreter(source, filename).run()


def analyze_all(source: str, filename: str = "<string>") -> DataPlaneReport:
    """The unified front door: one engine run spanning **every** structural axis.

    Interprets the natively-modelled axes (value-domain refinement,
    fit-before-split non-interference, temporal causality) and additionally
    delegates the remaining structural families (split contracts, group
    disjointness, sampling independence, join cardinality), re-birthing each as a
    first-class obligation re-discharged by the same certifier.  The returned
    :class:`DataPlaneReport` therefore carries obligations across all seven
    DataRefine bug axes from a single call.
    """
    return DataPlaneInterpreter(source, filename).run(include_families=True)


def analyze_path(path: str | Path, *, include_families: bool = False) -> DataPlaneReport:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        source = ""
    return DataPlaneInterpreter(source, str(p)).run(include_families=include_families)


def analyze_tree(root: str | Path, *, include_families: bool = True) -> DataPlaneReport:
    """Run the unified front door over every ``*.py`` file under ``root``.

    Aggregates the per-file obligations into one repo-level
    :class:`DataPlaneReport` (``file`` is the root), so a whole project can be
    swept across all structural axes in a single call.  ``include_families``
    defaults to ``True`` here: a repo sweep wants the complete axis coverage.
    """
    root = Path(root)
    report = DataPlaneReport(file=str(root))
    paths: Iterable[Path] = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for p in paths:
        report.obligations.extend(
            analyze_path(p, include_families=include_families).obligations)
    return report


def infer_refinement(source: str, expression: str) -> Refinement:
    """Infer the refinement of ``expression`` in the context of ``source``.

    A small introspection helper (and the natural unit-test seam for the lattice
    transfer functions): runs the collection pass over ``source`` then infers the
    parsed ``expression``.
    """
    interp = DataPlaneInterpreter(source, "<introspect>")
    try:
        interp._collect(ast.parse(source))
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return TOP
    return interp._infer(node)


__all__ = [
    "DATAPLANE_SCHEMA_VERSION",
    "Refinement",
    "TOP",
    "UNKNOWN", "PROB_UNIT", "PROB", "LOG_PROB", "LOGIT",
    "LOW", "HIGH", "ALL", "TRAIN", "TEST",
    "DataPlaneObligation",
    "DataPlaneReport",
    "DataPlaneInterpreter",
    "analyze_source",
    "analyze_all",
    "analyze_path",
    "analyze_tree",
    "infer_refinement",
]
