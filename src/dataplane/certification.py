"""Pluggable certifier architecture: SMT where it is honest, other paradigms elsewhere.

This module gives DataRefine a *certifier protocol* (``Certifier``) and a small family
of backends that turn obligations into structured, replayable verdicts. The design
principle is conservative: a backend only **admits** an obligation it can actually
decide, and otherwise returns ``empirical-required`` (a downstream empirical witness is
needed), ``unknown`` (the backend could not decide), ``rejected`` (the obligation is
violated), or ``skipped`` (out of this backend's scope). Every verdict carries a
human-readable explanation and the input hashes of what it looked at, so a reader can
tell *why* a claim was admitted and *over what input*.

The backends:

``StructuralCertifier``
    Lowers crisp, finite obligations (schema consistency, role constraints, split
    disjointness, column lineage, join safety, numeric bounds, no-outcome-in-feature)
    to an SMT-ready :class:`LoweredFormula` and decides them with a pure-Python decision
    procedure (or an optional solver). It refuses to opine on statistical/calibration/
    causal-effect/benchmark-utility obligations, returning ``empirical-required``.

``CausalGraphCertifier``
    Reasons over a declared causal graph: role consistency, forbidden outcome
    descendants in the feature set, adjustment-set records, collider warnings, and
    post-treatment leakage.

``ProbabilisticCertifier``
    Handles uncertainty coverage, calibration, bootstrap intervals and empirical
    confidence using *recorded empirical evidence*; with no evidence it stays honest and
    returns ``empirical-required`` rather than admitting.

``RuntimeReplayCertifier``
    Admits an obligation only when a replay record (command, fixture hashes, observed
    violations, metric keys, acceptance criteria, bounded environment) shows the
    acceptance criteria met and no observed violations.

:func:`combine_verdicts` merges verdicts from several certifiers under an explicit
precedence and reports conflicts (e.g. one backend rejects while another admits).
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .obligations import Obligation
from .provenance import stable_json_hash

CERTIFICATION_SCHEMA_VERSION = "datarefine.certification.v1"
CERTIFIER_PACKET_SCHEMA_VERSION = "datarefine.certifier_packet.v1"

# The closed set of verdict statuses a certifier may return (step 261).
VERDICT_STATUSES = (
    "admitted",
    "rejected",
    "unknown",
    "empirical-required",
    "skipped",
)

# Only ``admitted`` is an admission. Everything else (including ``unknown``,
# ``empirical-required`` and solver errors) is *not* an admission unless a downstream
# witness explicitly covers it (step 266).
ADMISSION_STATUS = "admitted"
NON_ADMISSION_STATUSES = ("rejected", "unknown", "empirical-required", "skipped")

# Structural constraint families the SMT-ready lowering understands (step 262). These
# are decidable, finite, quantifier-free obligations.
STRUCTURAL_CONSTRAINTS = (
    "schema_consistency",
    "role_constraint",
    "split_disjointness",
    "column_lineage",
    "join_safety",
    "bounds",
    "no_outcome_in_feature",
    "fit_transform_isolation",
    "partition_lengths",
    "temporal_causality",
    "group_disjointness",
    "sampling_independence",
    "join_cardinality",
    "value_domain",
)

# Obligation predicates that are *not* structural and must go to an empirical backend
# (step 263). The structural certifier returns ``empirical-required`` for these.
EMPIRICAL_CONSTRAINTS = (
    "statistical_significance",
    "calibration",
    "causal_effect",
    "benchmark_utility",
    "coverage",
    "bootstrap_interval",
    "empirical_confidence",
)

# Package extra that ships an optional SMT solver (step 264). The library never imports
# it eagerly; structural checks fall back to a pure-Python decision procedure.
SMT_EXTRA = "datarefine[smt]"

# Precedence for merging verdicts from different certifier kinds (step 270). A rejection
# from a higher-precedence kind dominates an admission from a lower one.
CERTIFIER_PRECEDENCE = (
    "structural",
    "causal",
    "privacy",
    "probabilistic",
    "runtime",
    "human_evidence",
)


@dataclass(frozen=True)
class CertifierVerdict:
    """A structured, replayable certifier decision (step 261)."""

    certifier: str
    kind: str
    status: str
    obligation_kind: str
    obligation_id: str
    explanation: str
    diagnostics: tuple[Mapping[str, object], ...] = ()
    input_hashes: Mapping[str, str] = field(default_factory=dict)
    covered_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in VERDICT_STATUSES:
            raise ValueError(f"unsupported verdict status {self.status!r}")

    @property
    def admitted(self) -> bool:
        return self.status == ADMISSION_STATUS

    @property
    def is_admission(self) -> bool:
        """An admission requires either ``admitted`` or an explicit covering witness."""

        if self.status == ADMISSION_STATUS:
            return True
        return bool(self.covered_by)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CERTIFICATION_SCHEMA_VERSION,
            "certifier": self.certifier,
            "kind": self.kind,
            "status": self.status,
            "obligation_kind": self.obligation_kind,
            "obligation_id": self.obligation_id,
            "explanation": self.explanation,
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
            "input_hashes": dict(self.input_hashes),
            "covered_by": list(self.covered_by),
        }


@runtime_checkable
class Certifier(Protocol):
    """The pluggable certifier interface (step 261)."""

    name: str
    kind: str

    def handles(self, obligation: Obligation) -> bool:
        ...

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        ...


@dataclass(frozen=True)
class LoweredFormula:
    """An SMT-ready lowering of a structural obligation (step 262)."""

    constraint: str
    theory: str
    variables: tuple[str, ...]
    assertions: tuple[str, ...]
    payload: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "constraint": self.constraint,
            "theory": self.theory,
            "variables": list(self.variables),
            "assertions": list(self.assertions),
            "payload": dict(self.payload),
        }

    @property
    def formula_hash(self) -> str:
        return stable_json_hash(self.as_dict())


def _constraint_of(obligation: Obligation) -> str | None:
    constraint = obligation.payload.get("constraint")
    if isinstance(constraint, str) and constraint:
        return constraint
    # Map obligation kinds to default structural constraint families.
    return {
        "schema": "schema_consistency",
        "role": "role_constraint",
        "split": "split_disjointness",
        "lineage": "column_lineage",
        "temporal": "temporal_causality",
        "group": "group_disjointness",
        "sampling": "sampling_independence",
        "join": "join_cardinality",
        "domain": "value_domain",
    }.get(obligation.kind)


def _seq(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.items())
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def lower_obligation(obligation: Obligation) -> LoweredFormula:
    """Lower a structural obligation into an SMT-ready :class:`LoweredFormula`.

    Raises ``ValueError`` if the obligation is not a structural constraint family this
    backend understands; callers should route such obligations to an empirical or
    causal backend instead.
    """

    constraint = _constraint_of(obligation)
    if constraint is None or constraint not in STRUCTURAL_CONSTRAINTS:
        raise ValueError(f"obligation {obligation.obligation_id} is not a structural constraint")
    payload = dict(obligation.payload)
    target = obligation.target

    if constraint == "schema_consistency":
        declared = tuple(str(name) for name in _seq(payload, "declared_columns"))
        observed = tuple(str(name) for name in _seq(payload, "observed_columns"))
        assertions = (f"set({list(observed)}) == set({list(declared)})",)
        return LoweredFormula(constraint, "sets", declared + observed, assertions, payload)

    if constraint == "role_constraint":
        roles = tuple(str(role) for role in _seq(payload, "roles"))
        allowed = tuple(str(role) for role in _seq(payload, "allowed_roles")) or roles
        assertions = tuple(f"role({target}.{i}) in {list(allowed)}" for i in range(len(roles)))
        return LoweredFormula(constraint, "EUF", roles, assertions, payload)

    if constraint == "split_disjointness":
        interval_raw = payload.get("interval_partitions") or {}
        if interval_raw:
            intervals = {str(k): [float(v[0]), float(v[1])] for k, v in interval_raw.items()
                         if isinstance(v, Sequence) and len(v) >= 2}
            names = tuple(sorted(intervals))
            assertions = tuple(
                f"disjoint_interval({a}, {b})"
                for i, a in enumerate(names)
                for b in names[i + 1 :]
            )
            return LoweredFormula(constraint, "LIA", names,
                                  assertions or (f"singleton({names})",),
                                  {**payload, "interval_partitions": intervals})
        partitions = {str(k): tuple(str(v) for v in _seq({"v": vs}, "v")) for k, vs in (payload.get("partitions") or {}).items()}
        names = tuple(sorted(partitions))
        assertions = tuple(
            f"disjoint({a}, {b})"
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        )
        return LoweredFormula(constraint, "sets", names, assertions or (f"singleton({names})",), {**payload, "partitions": partitions})

    if constraint == "column_lineage":
        edges = tuple((str(d), tuple(str(s) for s in _seq({"v": srcs}, "v"))) for d, srcs in (payload.get("lineage") or {}).items())
        variables = tuple(d for d, _ in edges)
        assertions = tuple(f"has_source({d})" for d, _ in edges)
        return LoweredFormula(constraint, "EUF", variables, assertions, {**payload, "lineage": {d: list(s) for d, s in edges}})

    if constraint == "join_safety":
        keys = tuple(str(k) for k in _seq(payload, "join_keys"))
        targets = tuple(str(c) for c in _seq(payload, "target_columns"))
        assertions = tuple(f"{k} not in {list(targets)}" for k in keys)
        return LoweredFormula(constraint, "sets", keys + targets, assertions, payload)

    if constraint == "bounds":
        lo = payload.get("lower")
        hi = payload.get("upper")
        values = tuple(_seq(payload, "values"))
        assertions = tuple(f"{lo} <= v_{i} <= {hi}" for i in range(len(values)))
        return LoweredFormula(constraint, "LRA", tuple(f"v_{i}" for i in range(len(values))), assertions, payload)

    if constraint == "no_outcome_in_feature":
        feature_sources = tuple(str(c) for c in _seq(payload, "feature_sources"))
        outcomes = tuple(str(c) for c in _seq(payload, "outcome_columns"))
        assertions = tuple(f"{c} not in {list(outcomes)}" for c in feature_sources)
        return LoweredFormula(constraint, "sets", feature_sources + outcomes, assertions, payload)

    if constraint == "partition_lengths":
        lengths = tuple(_seq(payload, "lengths"))
        total = payload.get("total")
        fractions = bool(payload.get("fractions"))
        target = total if total is not None else (1.0 if fractions else None)
        assertions = tuple(f"length_{i} > 0" for i in range(len(lengths)))
        if target is not None:
            assertions = assertions + (f"sum(lengths) == {target}",)
        return LoweredFormula(constraint, "LRA",
                              tuple(f"length_{i}" for i in range(len(lengths))),
                              assertions or ("no_lengths",), payload)

    if constraint == "temporal_causality":
        # No-lookahead: a feature computed at output row ``i`` must read only
        # rows at index <= i.  The composed forward reach R of the feature's
        # operator chain is the obligation's only datum; the property is the LIA
        # sentence ``forall i,j. not (i < j <= i + R)`` over the index axis (with
        # an optional chronological ``cut`` restricting i to the training region
        # and j to the held-out region).  z3 searches for a concrete violating
        # (feature_row, future_source_row); R == 0 is causal and admits.
        try:
            reach = int(float(payload.get("forward_reach", 0)))
        except (TypeError, ValueError):
            reach = 0
        cut = payload.get("cut")
        scope = f", cut={cut}" if cut is not None else ""
        assertions = (f"no_future_dependency(forward_reach={reach}{scope})",)
        return LoweredFormula(constraint, "LIA",
                              ("feature_row", "future_source_row"),
                              assertions, payload)

    if constraint == "group_disjointness":
        # Quotient disjointness: the train/test partitions must be disjoint in
        # the quotient set ``X/~`` induced by the group key (same patient/user/
        # speaker/molecule).  This is the disjointness contract *lifted along an
        # equivalence relation*: ``split_disjointness`` is the special case where
        # ``~`` is the identity (every row its own singleton group).  A
        # *group-aware* splitter (GroupKFold / ``groups=``) keeps every member of
        # a group inside one partition; a row-level group-blind split assigns the
        # members independently, so a group of size >= 2 can straddle the cut.
        # We lower the abstract member->partition assignment; z3 *searches* it
        # for a straddling group (a concrete leakage witness).
        try:
            members = max(int(float(payload.get("group_size", 2))), 2)
        except (TypeError, ValueError):
            members = 2
        try:
            parts = max(int(float(payload.get("partitions", 2))), 2)
        except (TypeError, ValueError):
            parts = 2
        aware = bool(payload.get("group_aware"))
        key = str(payload.get("group_key", "group"))
        scope = "group_aware" if aware else "group_blind"
        assertions = (
            f"same_group_same_partition(key={key}, members={members}, "
            f"partitions={parts}, {scope})",
        )
        return LoweredFormula(constraint, "LIA",
                              tuple(f"member_{i}" for i in range(members)),
                              assertions,
                              {**payload, "group_size": members,
                               "partitions": parts, "group_aware": aware,
                               "group_key": key})

    if constraint == "sampling_independence":
        # An *effect* contract on the Dataset/DataLoader sampling path (a strict
        # lift of the lattice from value properties to the path's stochastic
        # effect set).  Two violations are decidable from a handful of flags:
        #   * fork-correlation: a training path that draws from a *process-global*
        #     RNG (numpy / random) under ``num_workers >= 1`` *without* a
        #     ``worker_init_fn`` reseed -- forked workers inherit one RNG state,
        #     so the per-sample draws are NOT independent (correlated / duplicated
        #     augmentations across the epoch);
        #   * eval-nondeterminism: an evaluation path whose transform carries a
        #     stochastic augmentation (the test metric is then nondeterministic
        #     and biased).
        try:
            workers = int(float(payload.get("num_workers", 0)))
        except (TypeError, ValueError):
            workers = 0
        flags = {
            "global_rng": bool(payload.get("global_rng")),
            "worker_init_fn": bool(payload.get("worker_init_fn")),
            "is_eval": bool(payload.get("is_eval")),
            "stochastic_eval": bool(payload.get("stochastic_eval")),
        }
        assertions = (
            "independent_draws("
            f"global_rng={flags['global_rng']}, num_workers={workers}, "
            f"worker_init_fn={flags['worker_init_fn']}, is_eval={flags['is_eval']}, "
            f"stochastic_eval={flags['stochastic_eval']})",
        )
        return LoweredFormula(constraint, "QF_BV+LIA", ("num_workers",),
                              assertions, {**payload, "num_workers": workers, **flags})

    if constraint == "join_cardinality":
        # A *relational-multiplicity* contract -- the first one about cardinality
        # preservation rather than membership / order / effect.  A join
        # ``L ><_k R`` preserves left cardinality iff the key ``k`` is unique on
        # the right (an m:1 / 1:1 join); if ``k`` is non-unique on the right the
        # rows *fan out* (``|result| = sum_l mult_R(k_l)``), silently duplicating
        # samples and inflating every downstream count / mean -- the canonical
        # pandas ``merge`` fan-out bug that ``validate=`` exists to prevent.  We
        # lower the multiplicity arithmetic; z3 *searches* for a fan-out factor.
        validated = bool(payload.get("validated"))
        right_unique = bool(payload.get("right_key_unique"))
        consumed = bool(payload.get("cardinality_consumed", True))
        key = str(payload.get("join_key", "key"))
        how = str(payload.get("how", "inner"))
        try:
            left_rows = max(int(float(payload.get("left_rows", 100))), 1)
        except (TypeError, ValueError):
            left_rows = 100
        assertions = (
            f"cardinality_preserving(join_key={key}, how={how}, "
            f"validated={validated}, right_key_unique={right_unique}, "
            f"cardinality_consumed={consumed})",
        )
        return LoweredFormula(constraint, "LIA", ("right_mult", "result_rows"),
                              assertions,
                              {**payload, "validated": validated,
                               "right_key_unique": right_unique,
                               "cardinality_consumed": consumed,
                               "join_key": key, "how": how, "left_rows": left_rows})

    if constraint == "value_domain":
        # A *value-domain refinement* contract -- the numeric-domain complement to
        # a shape/dtype verifier (e.g. thehalleyyoung/tensorguard, which proves the
        # *shape* refinement {v | shape(v)==...} but, being shape-only, cannot see a
        # tensor's value domain).  Several PyTorch losses carry an unstated numeric
        # *precondition on the value domain of their input* that, when violated,
        # produces a silently-wrong loss with no exception:
        #   * BCELoss / binary_cross_entropy expects probabilities in [0,1]; fed an
        #     un-sigmoided activation (raw logits) it computes a meaningless loss
        #     (use BCEWithLogitsLoss or apply sigmoid);
        #   * NLLLoss / nll_loss expects log-probabilities (<= 0); fed a softmax
        #     (probabilities, > 0) or raw logits it silently optimises nonsense
        #     (use log_softmax, or CrossEntropyLoss which fuses it).
        # ``domain_established`` records whether a domain-normalising producer
        # (sigmoid/log_softmax) is locally visible on the input; if so the input is
        # provably inside [required_lo, required_hi] and the obligation holds.
        op = str(payload.get("op", "bce"))
        loss = str(payload.get("loss", op))
        producer = str(payload.get("producer", "unknown"))
        established = bool(payload.get("domain_established"))
        req_lo = payload.get("required_lo")
        req_hi = payload.get("required_hi")
        assertions = (
            f"value_domain(op={op}, producer={producer}, "
            f"domain_established={established}, required=[{req_lo},{req_hi}])",
        )
        return LoweredFormula(constraint, "LRA", ("v",), assertions,
                              {**payload, "op": op, "loss": loss,
                               "producer": producer,
                               "domain_established": established,
                               "required_lo": req_lo, "required_hi": req_hi})

    # fit_transform_isolation: a featurizer's fitted state must derive only from
    # training rows (no holdout rows) and must not be fit on the outcome column.
    fit_ids = tuple(str(v) for v in _seq(payload, "fit_row_ids"))
    holdout = tuple(str(v) for v in _seq(payload, "holdout_row_ids"))
    fit_sources = tuple(str(v) for v in _seq(payload, "fit_feature_sources"))
    outcomes = tuple(str(v) for v in _seq(payload, "outcome_columns"))
    assertions = (
        "disjoint(fit_row_ids, holdout_row_ids)",
        f"fit_feature_sources not in {list(outcomes)}",
    )
    return LoweredFormula(constraint, "sets", fit_ids + holdout + fit_sources + outcomes, assertions, payload)


def solver_available() -> bool:
    """Whether an optional SMT solver backend is importable (step 264)."""

    try:  # pragma: no cover - exercised only where z3 is installed
        import z3  # noqa: F401
    except Exception:
        return False
    return True


def solver_diagnostics() -> dict[str, object]:
    """Diagnostics describing solver availability and the pure-Python fallback."""

    available = solver_available()
    options = {"timeout_s": 1.0}
    version = "not-installed"
    if available:
        try:
            from . import smt_backend as _smt

            version = str(_smt.z3_version())
            options = _smt.z3_options()
        except Exception:  # pragma: no cover - defensive diagnostic only
            version = "unknown"
    return {
        "solver_available": available,
        "backend": "z3" if available else "pure-python-structural",
        "version": version,
        "options": options,
        "env_hash": solver_environment_hash(),
        "fallback": "pure-python-structural",
        "install_hint": None if available else f"pip install {SMT_EXTRA}",
        "note": (
            "structural obligations are decided by a pure-Python procedure; the solver is "
            "optional and only accelerates/cross-checks the same decidable fragment"
        ),
    }


def solver_environment_hash() -> str:
    """Hash the interpreter, platform, solver version, and solver options."""

    try:
        from . import smt_backend as _smt

        z3_version = _smt.z3_version()
        z3_options = _smt.z3_options()
    except Exception:  # pragma: no cover - optional dependency diagnostic only
        z3_version = None
        z3_options = {}
    return stable_json_hash(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "z3_version": z3_version,
            "z3_options": z3_options,
        }
    )[:16]


@dataclass(frozen=True)
class SolverReport:
    """Reproducible record of a solver run for a certifier packet (step 265)."""

    solver: str
    version: str
    timeout_s: float
    input_hash: str
    formula_hash: str
    result: str
    valid: bool | None = None
    model: Mapping[str, object] | None = None
    unsat_core: tuple[str, ...] = ()
    options: Mapping[str, object] = field(default_factory=dict)
    env_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "version": self.version,
            "timeout_s": self.timeout_s,
            "options": dict(self.options),
            "env_hash": self.env_hash or solver_environment_hash(),
            "input_hash": self.input_hash,
            "formula_hash": self.formula_hash,
            "result": self.result,
            "valid": self.valid,
            "model": dict(self.model) if self.model is not None else None,
            "unsat_core": list(self.unsat_core),
        }


def _decide_structural(constraint: str, payload: Mapping[str, object]) -> tuple[bool, list[Mapping[str, object]]]:
    """Pure-Python decision procedure for the decidable structural fragment."""

    diagnostics: list[Mapping[str, object]] = []
    if constraint == "schema_consistency":
        declared = set(str(c) for c in _seq(payload, "declared_columns"))
        observed = set(str(c) for c in _seq(payload, "observed_columns"))
        missing = sorted(declared - observed)
        extra = sorted(observed - declared)
        if missing or extra:
            diagnostics.append({"missing_columns": missing, "unexpected_columns": extra})
            return False, diagnostics
        return True, diagnostics
    if constraint == "role_constraint":
        roles = [str(r) for r in _seq(payload, "roles")]
        allowed = set(str(r) for r in _seq(payload, "allowed_roles")) or set(roles)
        bad = sorted(r for r in roles if r not in allowed)
        if bad:
            diagnostics.append({"disallowed_roles": bad, "allowed_roles": sorted(allowed)})
            return False, diagnostics
        return True, diagnostics
    if constraint == "split_disjointness":
        intervals = payload.get("interval_partitions") or {}
        if intervals:
            names = sorted(intervals)
            overlaps: list[Mapping[str, object]] = []
            for i, a in enumerate(names):
                la, ha = float(intervals[a][0]), float(intervals[a][1])
                for b in names[i + 1:]:
                    lb, hb = float(intervals[b][0]), float(intervals[b][1])
                    lo, hi = max(la, lb), min(ha, hb)
                    if lo < hi:  # half-open [lo, hi) overlap
                        overlaps.append({"partitions": sorted([a, b]),
                                         "overlap_range": [lo, hi]})
            if overlaps:
                diagnostics.extend(overlaps)
                return False, diagnostics
            return True, diagnostics
        partitions = payload.get("partitions") or {}
        seen: dict[str, str] = {}
        overlaps = []
        for name, ids in partitions.items():
            for ident in ids:
                ident = str(ident)
                if ident in seen and seen[ident] != str(name):
                    overlaps.append({"id": ident, "in": sorted([seen[ident], str(name)])})
                seen[ident] = str(name)
        if overlaps:
            diagnostics.extend(overlaps)
            return False, diagnostics
        return True, diagnostics
    if constraint == "column_lineage":
        lineage = payload.get("lineage") or {}
        orphans = sorted(d for d, srcs in lineage.items() if not srcs)
        if orphans:
            diagnostics.append({"columns_without_source": orphans})
            return False, diagnostics
        return True, diagnostics
    if constraint == "join_safety":
        keys = set(str(k) for k in _seq(payload, "join_keys"))
        targets = set(str(t) for t in _seq(payload, "target_columns"))
        unsafe = sorted(keys & targets)
        if unsafe:
            diagnostics.append({"target_derived_join_keys": unsafe})
            return False, diagnostics
        return True, diagnostics
    if constraint == "bounds":
        lo = payload.get("lower")
        hi = payload.get("upper")
        out = []
        # A bound that is provided but not a finite number (NaN/non-numeric)
        # must NOT silently disable that side of the check -- that would let the
        # certifier report `valid` while enforcing nothing. Treat it as malformed.
        for edge, raw in (("lower", lo), ("upper", hi)):
            if raw is None:
                continue
            try:
                fb = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                out.append({"invalid_bound": edge, "value": raw})
                continue
            if fb != fb:  # NaN bound
                out.append({"invalid_bound": edge, "not_a_number": True})
        if out:
            diagnostics.extend(out)
            return False, diagnostics
        for value in _seq(payload, "values"):
            try:
                v = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                out.append({"non_numeric_value": value})
                continue
            if v != v:  # NaN is never within any numeric bound
                out.append({"value": value, "not_a_number": True})
                continue
            if lo is not None and v < float(lo):
                out.append({"value": v, "below_lower": float(lo)})
            if hi is not None and v > float(hi):
                out.append({"value": v, "above_upper": float(hi)})
        if out:
            diagnostics.extend(out)
            return False, diagnostics
        return True, diagnostics
    if constraint == "no_outcome_in_feature":
        feature_sources = set(str(c) for c in _seq(payload, "feature_sources"))
        outcomes = set(str(c) for c in _seq(payload, "outcome_columns"))
        leaked = sorted(feature_sources & outcomes)
        if leaked:
            diagnostics.append({"outcome_columns_in_features": leaked})
            return False, diagnostics
        return True, diagnostics
    if constraint == "partition_lengths":
        raw = list(_seq(payload, "lengths"))
        nums: list[float] = []
        for x in raw:
            try:
                nums.append(float(x))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                diagnostics.append({"non_numeric_length": x})
                return False, diagnostics
        if not nums:
            diagnostics.append({"empty_partition_lengths": True})
            return False, diagnostics
        nonpos = [v for v in nums if v <= 0]
        if nonpos:
            diagnostics.append({"non_positive_partitions": nonpos})
            return False, diagnostics
        fractions = bool(payload.get("fractions"))
        total = payload.get("total")
        target = total if total is not None else (1.0 if fractions else None)
        if target is not None:
            tol = payload.get("tolerance")
            tol = float(tol) if tol is not None else (1e-9 if fractions else 0.0)
            if abs(sum(nums) - float(target)) > tol:
                diagnostics.append({"sum": sum(nums), "expected_total": float(target)})
                return False, diagnostics
        return True, diagnostics
    if constraint == "temporal_causality":
        try:
            reach = int(float(payload.get("forward_reach", 0)))
        except (TypeError, ValueError):
            diagnostics.append({"non_numeric_forward_reach": payload.get("forward_reach")})
            return False, diagnostics
        if reach >= 1:
            cut = payload.get("cut")
            try:
                base = int(float(cut)) if cut is not None else 0
            except (TypeError, ValueError):
                base = 0
            diagnostics.append({
                "lookahead_forward_reach": reach,
                "feature_row": base,
                "future_source_row": base + reach,
            })
            return False, diagnostics
        return True, diagnostics
    if constraint == "group_disjointness":
        # A group-aware split forces every member of a group into one partition,
        # so no group can straddle (valid).  A group-blind row-level split over
        # data carrying a group of size >= 2 with >= 2 partitions lets two
        # members land in different partitions (rejected, with a concrete pair).
        aware = bool(payload.get("group_aware"))
        if aware:
            return True, diagnostics
        try:
            members = max(int(float(payload.get("group_size", 2))), 2)
        except (TypeError, ValueError):
            members = 2
        try:
            parts = max(int(float(payload.get("partitions", 2))), 2)
        except (TypeError, ValueError):
            parts = 2
        key = str(payload.get("group_key", "group"))
        if members >= 2 and parts >= 2:
            diagnostics.append({
                "group_key": key,
                "member_a": 0, "partition_a": 0,
                "member_b": 1, "partition_b": 1,
                "group_size": members, "partitions": parts,
            })
            return False, diagnostics
        return True, diagnostics
    if constraint == "sampling_independence":
        try:
            workers = int(float(payload.get("num_workers", 0)))
        except (TypeError, ValueError):
            workers = 0
        global_rng = bool(payload.get("global_rng"))
        worker_init = bool(payload.get("worker_init_fn"))
        is_eval = bool(payload.get("is_eval"))
        stochastic_eval = bool(payload.get("stochastic_eval"))
        if is_eval and stochastic_eval:
            diagnostics.append({
                "violation": "eval_nondeterminism",
                "detail": "stochastic augmentation on an evaluation/test path",
            })
            return False, diagnostics
        if (not is_eval) and global_rng and workers >= 1 and not worker_init:
            diagnostics.append({
                "violation": "correlated_worker_rng",
                "num_workers": workers,
                "detail": ("forked workers inherit one global RNG state without a "
                           "worker_init_fn reseed; per-sample draws are not independent"),
            })
            return False, diagnostics
        return True, diagnostics
    if constraint == "join_cardinality":
        # A validated merge (``validate=``) or a right operand made unique on the
        # key (dedup / groupby / pivot) cannot silently fan out (valid).  An
        # un-guarded join whose result is consumed in a cardinality-sensitive way
        # (split / sample / count / mean) can duplicate rows (rejected, with a
        # concrete fan-out factor).
        validated = bool(payload.get("validated"))
        right_unique = bool(payload.get("right_key_unique"))
        consumed = bool(payload.get("cardinality_consumed", True))
        key = str(payload.get("join_key", "key"))
        how = str(payload.get("how", "inner"))
        if consumed and not validated and not right_unique:
            try:
                left_rows = max(int(float(payload.get("left_rows", 100))), 1)
            except (TypeError, ValueError):
                left_rows = 100
            diagnostics.append({
                "violation": "join_fanout",
                "join_key": key,
                "how": how,
                "right_multiplicity": 2,
                "left_rows": left_rows,
                "result_rows": left_rows * 2,
                "detail": ("join key is not unique on the right and the merge has "
                           "no validate= guard; rows fan out and silently inflate "
                           "the downstream count/sample"),
            })
            return False, diagnostics
        return True, diagnostics
    if constraint == "value_domain":
        # The input domain holds iff a domain-normalising producer guarantees the
        # input lies inside the loss's required value interval.  Absent that
        # producer the input can take a value outside the interval (rejected, with
        # a concrete out-of-domain witness value).
        op = str(payload.get("op", "bce"))
        loss = str(payload.get("loss", op))
        producer = str(payload.get("producer", "unknown"))
        established = bool(payload.get("domain_established"))
        if not established:
            req_lo = payload.get("required_lo")
            req_hi = payload.get("required_hi")

            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None

            flo, fhi = _f(req_lo), _f(req_hi)
            # Pick a concrete value the un-normalised input can take that breaches
            # the required interval (logit > 1 for BCE; positive prob for NLL).
            if fhi is not None:
                witness_value = fhi + 1.0
            elif flo is not None:
                witness_value = flo - 1.0
            else:
                witness_value = 0.0
            diagnostics.append({
                "violation": "value_domain_breach",
                "op": op,
                "loss": loss,
                "producer": producer,
                "out_of_domain_value": witness_value,
                "required_lo": req_lo,
                "required_hi": req_hi,
                "detail": (f"{loss} requires its input in [{req_lo}, {req_hi}] but "
                           f"no domain-normalising producer is applied "
                           f"(producer={producer!r}); the input value "
                           f"{witness_value} breaches that domain and the loss is "
                           "silently wrong"),
            })
            return False, diagnostics
        return True, diagnostics
    # fit_transform_isolation
    fit_ids = set(str(v) for v in _seq(payload, "fit_row_ids"))
    holdout = set(str(v) for v in _seq(payload, "holdout_row_ids"))
    leaked_rows = sorted(fit_ids & holdout)
    fit_sources = set(str(v) for v in _seq(payload, "fit_feature_sources"))
    outcomes = set(str(v) for v in _seq(payload, "outcome_columns"))
    leaked_cols = sorted(fit_sources & outcomes)
    if leaked_rows or leaked_cols:
        if leaked_rows:
            diagnostics.append({"holdout_rows_used_in_fit": leaked_rows[:50],
                                "leaked_row_count": len(leaked_rows)})
        if leaked_cols:
            diagnostics.append({"outcome_columns_used_in_fit": leaked_cols})
        return False, diagnostics
    return True, diagnostics


@dataclass(frozen=True)
class StructuralCertifier:
    """Admits only the decidable structural fragment; everything else is empirical."""

    name: str = "structural"
    kind: str = "structural"
    timeout_s: float = 1.0
    differential: bool = True

    def handles(self, obligation: Obligation) -> bool:
        constraint = obligation.payload.get("constraint")
        if isinstance(constraint, str) and constraint in EMPIRICAL_CONSTRAINTS:
            return True
        try:
            lower_obligation(obligation)
        except ValueError:
            return False
        return True

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        constraint = obligation.payload.get("constraint")
        if isinstance(constraint, str) and constraint in EMPIRICAL_CONSTRAINTS:
            return CertifierVerdict(
                certifier=self.name,
                kind=self.kind,
                status="empirical-required",
                obligation_kind=obligation.kind,
                obligation_id=obligation.obligation_id,
                explanation=(
                    f"constraint {constraint!r} is statistical/empirical; the structural "
                    "backend cannot admit it. Route to an empirical certifier."
                ),
                input_hashes={"obligation": obligation.content_hash},
            )
        try:
            formula = lower_obligation(obligation)
        except ValueError as exc:
            return CertifierVerdict(
                certifier=self.name,
                kind=self.kind,
                status="skipped",
                obligation_kind=obligation.kind,
                obligation_id=obligation.obligation_id,
                explanation=str(exc),
                input_hashes={"obligation": obligation.content_hash},
            )
        ok, diagnostics = _decide_structural(formula.constraint, formula.payload)

        # Differential SMT oracle (STEPS.md 332): decide the same lowered formula
        # with an independent z3 backend. Agreement strengthens the verdict and
        # yields a re-checkable proof object (STEPS.md 333); a disagreement is a
        # soundness signal -- we downgrade to ``unknown`` and never ``admit``.
        solver_name = "pure-python-structural"
        z3_result = None
        z3_model: Mapping[str, object] | None = None
        z3_core: tuple[str, ...] = ()
        disagreement: Mapping[str, object] | None = None
        if self.differential:
            try:
                from . import smt_backend as _smt
                dec = _smt.z3_decide(formula.constraint, formula.payload)
            except Exception:  # pragma: no cover - defensive
                dec = None
            if dec is not None and dec.available and dec.valid is not None:
                solver_name = f"z3+pure-python-differential (z3 {_smt.z3_version()})"
                z3_result = dec.result
                z3_model = dec.model
                z3_core = dec.unsat_core
                if dec.valid != ok:
                    disagreement = {
                        "differential_disagreement": True,
                        "pure_python_valid": ok,
                        "z3_valid": dec.valid,
                        "z3_result": dec.result,
                    }

        if disagreement is not None:
            status = "unknown"
            result_label = "differential-disagreement"
        else:
            status = "admitted" if ok else "rejected"
            result_label = "valid" if ok else "unsat-violation"

        # Proof object: a counterexample model on rejection, an unsat core on
        # admission. Re-checkable by datarefine.smt_backend.recheck.
        proof_model = z3_model if status == "rejected" else None
        if status == "admitted":
            proof_core = z3_core or tuple(formula.assertions)
        elif status == "rejected":
            proof_core = z3_core
        else:
            proof_core = ()
        report = SolverReport(
            solver=solver_name,
            version=CERTIFICATION_SCHEMA_VERSION,
            timeout_s=self.timeout_s,
            options=solver_diagnostics()["options"],  # type: ignore[index]
            env_hash=solver_environment_hash(),
            input_hash=obligation.content_hash,
            formula_hash=formula.formula_hash,
            result=result_label if z3_result is None else f"{result_label} (z3={z3_result})",
            valid=True if status == "admitted" else (False if status == "rejected" else None),
            model=proof_model,
            unsat_core=proof_core,
        )
        extra_diag: tuple[Mapping[str, object], ...] = (disagreement,) if disagreement else ()
        return CertifierVerdict(
            certifier=self.name,
            kind=self.kind,
            status=status,
            obligation_kind=obligation.kind,
            obligation_id=obligation.obligation_id,
            explanation=(
                f"structural constraint {formula.constraint!r} decided "
                f"{'valid' if status == 'admitted' else ('violated' if status == 'rejected' else 'UNKNOWN (differential disagreement)')} "
                f"by {solver_name} over theory {formula.theory}"
            ),
            diagnostics=(report.as_dict(), *extra_diag, *diagnostics),
            input_hashes={"obligation": obligation.content_hash, "formula": formula.formula_hash},
        )


@dataclass(frozen=True)
class CausalGraph:
    """A declared causal graph for the causal-graph certifier (step 267)."""

    roles: Mapping[str, str]
    edges: tuple[tuple[str, str], ...] = ()
    adjustment_set: tuple[str, ...] = ()

    def descendants(self, node: str) -> set[str]:
        out: set[str] = set()
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for src, dst in self.edges:
                if src == current and dst not in out:
                    out.add(dst)
                    frontier.append(dst)
        return out

    def parents(self, node: str) -> set[str]:
        return {src for src, dst in self.edges if dst == node}


@dataclass(frozen=True)
class CausalGraphCertifier:
    name: str = "causal_graph"
    kind: str = "causal"

    def handles(self, obligation: Obligation) -> bool:
        return obligation.kind == "role" and obligation.payload.get("constraint") == "causal_graph"

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        context = context or {}
        graph = context.get("causal_graph")
        if not isinstance(graph, CausalGraph):
            return CertifierVerdict(
                certifier=self.name,
                kind=self.kind,
                status="unknown",
                obligation_kind=obligation.kind,
                obligation_id=obligation.obligation_id,
                explanation="no CausalGraph supplied in context['causal_graph']",
                input_hashes={"obligation": obligation.content_hash},
            )
        diagnostics: list[Mapping[str, object]] = []
        features = {n for n, r in graph.roles.items() if r in {"feature", "covariate"}}
        outcomes = {n for n, r in graph.roles.items() if r == "outcome"}
        treatments = {n for n, r in graph.roles.items() if r == "treatment"}

        # Forbidden descendants: a feature must not be a descendant of an outcome.
        forbidden = sorted({f for o in outcomes for f in graph.descendants(o) if f in features})
        if forbidden:
            diagnostics.append({"outcome_descendants_used_as_features": forbidden})

        # Post-treatment leakage: a covariate descended from a treatment is post-treatment.
        post_treatment = sorted({f for t in treatments for f in graph.descendants(t) if f in features})
        if post_treatment:
            diagnostics.append({"post_treatment_features": post_treatment, "warning": "may bias effect estimates"})

        # Collider warnings: a node with >= 2 parents that is in the adjustment set.
        colliders = sorted(n for n in graph.roles if len(graph.parents(n)) >= 2 and n in graph.adjustment_set)
        if colliders:
            diagnostics.append({"conditioned_colliders": colliders, "warning": "opens a backdoor path"})

        diagnostics.append({"adjustment_set": list(graph.adjustment_set)})

        if forbidden:
            status = "rejected"
            explanation = "feature set contains descendants of an outcome (leakage)"
        elif post_treatment or colliders:
            status = "empirical-required"
            explanation = "structural roles consistent but post-treatment/collider risks need empirical checks"
        else:
            status = "admitted"
            explanation = "roles consistent, no outcome descendants, no conditioned colliders"
        return CertifierVerdict(
            certifier=self.name,
            kind=self.kind,
            status=status,
            obligation_kind=obligation.kind,
            obligation_id=obligation.obligation_id,
            explanation=explanation,
            diagnostics=tuple(diagnostics),
            input_hashes={"obligation": obligation.content_hash, "graph": stable_json_hash({"roles": dict(graph.roles), "edges": [list(e) for e in graph.edges]})},
        )


@dataclass(frozen=True)
class ProbabilisticCertifier:
    """Admits empirical obligations only with recorded evidence (step 268)."""

    name: str = "probabilistic"
    kind: str = "probabilistic"

    def handles(self, obligation: Obligation) -> bool:
        constraint = obligation.payload.get("constraint")
        return isinstance(constraint, str) and constraint in EMPIRICAL_CONSTRAINTS

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        context = context or {}
        evidence = context.get("evidence")
        constraint = str(obligation.payload.get("constraint"))
        oid = obligation.obligation_id
        base = {
            "certifier": self.name,
            "kind": self.kind,
            "obligation_kind": obligation.kind,
            "obligation_id": oid,
        }
        if not isinstance(evidence, Mapping):
            return CertifierVerdict(
                **base,
                status="empirical-required",
                explanation=f"{constraint}: no empirical evidence recorded; cannot admit",
                input_hashes={"obligation": obligation.content_hash},
            )
        observed = evidence.get("observed")
        target = evidence.get("target")
        if observed is None or target is None:
            return CertifierVerdict(
                **base,
                status="empirical-required",
                explanation=f"{constraint}: evidence missing 'observed'/'target' keys",
                diagnostics=({"evidence_keys": sorted(str(k) for k in evidence)},),
                input_hashes={"obligation": obligation.content_hash},
            )
        observed_v = float(observed)
        target_v = float(target)
        meets = observed_v >= target_v
        diagnostics = [{"observed": observed_v, "target": target_v, "meets_target": meets}]
        if "interval" in evidence:
            lo, hi = evidence["interval"]  # type: ignore[misc]
            diagnostics.append({"bootstrap_interval": [float(lo), float(hi)]})
            # honest failure: if the interval includes values below target, do not admit.
            if float(lo) < target_v:
                return CertifierVerdict(
                    **base,
                    status="empirical-required",
                    explanation=(
                        f"{constraint}: point estimate {observed_v} but interval lower bound "
                        f"{float(lo)} < target {target_v}; confidence insufficient"
                    ),
                    diagnostics=tuple(diagnostics),
                    input_hashes={"obligation": obligation.content_hash, "evidence": stable_json_hash(dict(evidence))},
                )
        status = "admitted" if meets else "rejected"
        explanation = (
            f"{constraint}: observed {observed_v} {'>=' if meets else '<'} target {target_v}"
        )
        return CertifierVerdict(
            **base,
            status=status,
            explanation=explanation,
            diagnostics=tuple(diagnostics),
            input_hashes={"obligation": obligation.content_hash, "evidence": stable_json_hash(dict(evidence))},
        )


@dataclass(frozen=True)
class RuntimeReplayCertifier:
    """Admits an obligation from a bounded, recorded replay (step 269)."""

    name: str = "runtime_replay"
    kind: str = "runtime"

    def handles(self, obligation: Obligation) -> bool:
        return obligation.payload.get("constraint") == "runtime_replay" or obligation.kind in {"provenance", "claim_scope"}

    def certify(self, obligation: Obligation, context: Mapping[str, object] | None = None) -> CertifierVerdict:
        context = context or {}
        record = context.get("replay")
        oid = obligation.obligation_id
        base = {
            "certifier": self.name,
            "kind": self.kind,
            "obligation_kind": obligation.kind,
            "obligation_id": oid,
        }
        if not isinstance(record, Mapping):
            return CertifierVerdict(
                **base,
                status="unknown",
                explanation="no replay record supplied in context['replay']",
                input_hashes={"obligation": obligation.content_hash},
            )
        required = ("command", "fixture_hashes", "observed_violations", "metric_keys", "acceptance", "environment")
        missing = [key for key in required if key not in record]
        if missing:
            return CertifierVerdict(
                **base,
                status="unknown",
                explanation=f"replay record missing fields: {', '.join(missing)}",
                diagnostics=({"missing_fields": missing},),
                input_hashes={"obligation": obligation.content_hash},
            )
        violations = list(record.get("observed_violations") or [])
        acceptance = bool(record.get("acceptance"))
        record_hash = stable_json_hash({str(k): record[k] for k in sorted(record)})
        diagnostics = [
            {
                "command": record.get("command"),
                "metric_keys": list(record.get("metric_keys") or []),
                "environment": dict(record.get("environment") or {}),
                "fixture_hashes": dict(record.get("fixture_hashes") or {}),
                "observed_violations": violations,
            }
        ]
        if violations:
            status, explanation = "rejected", f"replay observed {len(violations)} violation(s)"
        elif acceptance:
            status, explanation = "admitted", "replay met acceptance criteria with no observed violations"
        else:
            status, explanation = "unknown", "replay had no violations but acceptance criteria not met"
        return CertifierVerdict(
            **base,
            status=status,
            explanation=explanation,
            diagnostics=tuple(diagnostics),
            input_hashes={"obligation": obligation.content_hash, "replay": record_hash},
        )


def resolve_unknown(verdict: CertifierVerdict, covering: Sequence[CertifierVerdict] = ()) -> CertifierVerdict:
    """Apply the step-266 rule: unknown/empirical-required are non-admissions unless a
    downstream runtime/empirical witness *admits* the same obligation."""

    if verdict.status == ADMISSION_STATUS:
        return verdict
    covers = tuple(
        f"{c.certifier}:{c.status}"
        for c in covering
        if c.obligation_id == verdict.obligation_id and c.status == ADMISSION_STATUS
    )
    if covers:
        return CertifierVerdict(
            certifier=verdict.certifier,
            kind=verdict.kind,
            status=verdict.status,
            obligation_kind=verdict.obligation_kind,
            obligation_id=verdict.obligation_id,
            explanation=verdict.explanation + f" (covered by {', '.join(covers)})",
            diagnostics=verdict.diagnostics,
            input_hashes=verdict.input_hashes,
            covered_by=covers,
        )
    return verdict


@dataclass(frozen=True)
class CombinedVerdict:
    """Result of merging per-certifier verdicts under precedence (step 270)."""

    obligation_id: str
    status: str
    decided_by: str
    explanation: str
    conflicts: tuple[Mapping[str, object], ...] = ()
    members: tuple[CertifierVerdict, ...] = ()

    @property
    def admitted(self) -> bool:
        return self.status == ADMISSION_STATUS

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CERTIFICATION_SCHEMA_VERSION,
            "obligation_id": self.obligation_id,
            "status": self.status,
            "decided_by": self.decided_by,
            "explanation": self.explanation,
            "conflicts": [dict(conflict) for conflict in self.conflicts],
            "members": [member.as_dict() for member in self.members],
        }


def combine_verdicts(verdicts: Sequence[CertifierVerdict]) -> CombinedVerdict:
    """Merge verdicts for a single obligation under :data:`CERTIFIER_PRECEDENCE`.

    Rules: a ``rejected`` from any backend dominates (safety first). Otherwise the
    highest-precedence backend that produced an admission decides. If no backend
    admitted, the most informative non-admission (rejected > empirical-required >
    unknown > skipped) is reported. Conflicts (an admission and a rejection, or two
    backends disagreeing) are recorded explicitly.
    """

    if not verdicts:
        raise ValueError("combine_verdicts requires at least one verdict")
    oids = {v.obligation_id for v in verdicts}
    if len(oids) != 1:
        raise ValueError("combine_verdicts requires verdicts for a single obligation")
    obligation_id = next(iter(oids))

    def rank(kind: str) -> int:
        return CERTIFIER_PRECEDENCE.index(kind) if kind in CERTIFIER_PRECEDENCE else len(CERTIFIER_PRECEDENCE)

    rejections = [v for v in verdicts if v.status == "rejected"]
    admissions = [v for v in verdicts if v.is_admission]

    conflicts: list[Mapping[str, object]] = []
    if rejections and admissions:
        conflicts.append(
            {
                "type": "admission_vs_rejection",
                "rejected_by": sorted({v.certifier for v in rejections}),
                "admitted_by": sorted({v.certifier for v in admissions}),
            }
        )

    if rejections:
        decisive = sorted(rejections, key=lambda v: rank(v.kind))[0]
        return CombinedVerdict(
            obligation_id=obligation_id,
            status="rejected",
            decided_by=decisive.certifier,
            explanation=f"rejected by {decisive.certifier}: {decisive.explanation}",
            conflicts=tuple(conflicts),
            members=tuple(verdicts),
        )
    if admissions:
        decisive = sorted(admissions, key=lambda v: rank(v.kind))[0]
        return CombinedVerdict(
            obligation_id=obligation_id,
            status="admitted",
            decided_by=decisive.certifier,
            explanation=f"admitted by {decisive.certifier}: {decisive.explanation}",
            conflicts=tuple(conflicts),
            members=tuple(verdicts),
        )
    order = {"empirical-required": 0, "unknown": 1, "skipped": 2}
    decisive = sorted(verdicts, key=lambda v: (order.get(v.status, 3), rank(v.kind)))[0]
    return CombinedVerdict(
        obligation_id=obligation_id,
        status=decisive.status,
        decided_by=decisive.certifier,
        explanation=f"no admission; {decisive.certifier} reports {decisive.status}: {decisive.explanation}",
        conflicts=tuple(conflicts),
        members=tuple(verdicts),
    )


def certifier_packet(
    verdicts: Sequence[CertifierVerdict],
    *,
    combined: Sequence[CombinedVerdict] = (),
    solver_reports: Sequence[SolverReport] = (),
) -> dict[str, object]:
    """Assemble a reproducible certifier packet (step 265)."""

    return {
        "schema_version": CERTIFIER_PACKET_SCHEMA_VERSION,
        "solver": solver_diagnostics(),
        "verdicts": [verdict.as_dict() for verdict in verdicts],
        "combined": [item.as_dict() for item in combined],
        "solver_reports": [report.as_dict() for report in solver_reports],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "ADMISSION_STATUS",
    "CERTIFICATION_SCHEMA_VERSION",
    "CERTIFIER_PACKET_SCHEMA_VERSION",
    "CERTIFIER_PRECEDENCE",
    "EMPIRICAL_CONSTRAINTS",
    "NON_ADMISSION_STATUSES",
    "SMT_EXTRA",
    "STRUCTURAL_CONSTRAINTS",
    "VERDICT_STATUSES",
    "CausalGraph",
    "CausalGraphCertifier",
    "Certifier",
    "CertifierVerdict",
    "CombinedVerdict",
    "LoweredFormula",
    "ProbabilisticCertifier",
    "RuntimeReplayCertifier",
    "SolverReport",
    "StructuralCertifier",
    "certifier_packet",
    "combine_verdicts",
    "lower_obligation",
    "resolve_unknown",
    "solver_environment_hash",
    "solver_available",
    "solver_diagnostics",
]
