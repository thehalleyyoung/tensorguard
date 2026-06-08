"""A genuine z3 backend for DataRefine's structural obligation fragment.

This module makes the SMT backend *load-bearing*: instead of the optional solver
merely advertising its availability, ``z3_decide`` encodes the **violation**
predicate of each structural constraint family and asks z3 to decide it:

* ``sat``   -> a concrete violation exists -> the obligation is **invalid**
              (``rejected``); z3's model is a counterexample witness (e.g. the
              actual overlapping row index, the out-of-range value).
* ``unsat`` -> no violation exists -> the obligation is **valid** (``admitted``);
              the unsat core is a re-checkable proof of validity.
* ``unknown`` -> z3 could not decide; callers must not admit on this alone.

For the arithmetic / interval families (``bounds``, ``partition_lengths``,
interval ``split_disjointness``) z3 genuinely *searches* -- it discovers the
offending integer index or real value without enumeration. For the finite-set
families z3 differentially cross-checks the pure-Python decision procedure
(:func:`datarefine.certification._decide_structural`); a disagreement between the
two independent deciders is a soundness signal that the certifier surfaces as
``unknown`` rather than silently admitting (STEPS.md 332).

The proof objects produced here (a satisfying model, or an unsat core) are
independently re-validated by :func:`recheck` so admission never rests on
trusting a single solver run (STEPS.md 333-334).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence


# z3 is genuinely load-bearing for the arithmetic / interval families (bounds,
# partition_lengths, interval split_disjointness) where it *searches* for a
# witness. For the finite-set families the pure-Python procedure is already a
# complete decision procedure, so z3 only runs as a bounded *differential
# cross-check*: the naive O(n^2) set-membership encoding blows up on large
# payloads, so above ``_MAX_SET_TERMS`` literals we skip z3 and let the
# pure-Python decider stand alone (it is exact for these families). A per-solver
# timeout is a secondary safety net so no single check can hang.
_MAX_SET_TERMS = int(os.environ.get("DATAREFINE_Z3_MAX_SET_TERMS", "64"))
_TIMEOUT_MS = int(os.environ.get("DATAREFINE_Z3_TIMEOUT_MS", "1000"))

_SET_FAMILIES = frozenset({
    "schema_consistency", "role_constraint", "join_safety",
    "no_outcome_in_feature", "column_lineage", "fit_transform_isolation",
})


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------
def z3_available() -> bool:
    try:  # pragma: no cover - trivial
        import z3  # noqa: F401
    except Exception:
        return False
    return True


def z3_version() -> str | None:
    try:  # pragma: no cover - trivial
        import z3
        return z3.get_version_string()
    except Exception:
        return None


def z3_options() -> dict[str, object]:
        """Stable solver options that affect admission and proof objects."""

        return {
            "timeout_ms": _TIMEOUT_MS,
            "max_set_terms": _MAX_SET_TERMS,
            "unsat_core": True,
        }


# ---------------------------------------------------------------------------
# decision record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Z3Decision:
    """Outcome of deciding a constraint's violation predicate with z3."""

    constraint: str
    available: bool
    result: str  # "sat" | "unsat" | "unknown" | "skipped"
    theory: str = ""
    valid: bool | None = None  # constraint holds (no violation) iff result == "unsat"
    model: Mapping[str, object] | None = None  # counterexample witness when invalid
    unsat_core: tuple[str, ...] = ()  # proof labels when valid
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "constraint": self.constraint,
            "available": self.available,
            "result": self.result,
            "theory": self.theory,
            "valid": self.valid,
            "model": dict(self.model) if self.model is not None else None,
            "unsat_core": list(self.unsat_core),
            "error": self.error,
        }


def _skipped(constraint: str, reason: str) -> Z3Decision:
    return Z3Decision(constraint=constraint, available=z3_available(),
                      result="skipped", error=reason)


# ---------------------------------------------------------------------------
# payload helpers (kept independent of certification._seq for module isolation)
# ---------------------------------------------------------------------------
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


def _strs(payload: Mapping[str, object], key: str) -> list[str]:
    return [str(v) for v in _seq(payload, key)]


def _payload_term_count(payload: Mapping[str, object]) -> int:
    """Total number of literals across the set-valued payload keys (an upper
    bound on the size of the z3 membership encoding)."""
    keys = (
        "declared_columns", "observed_columns", "roles", "allowed_roles",
        "join_keys", "target_columns", "feature_sources", "outcome_columns",
        "fit_row_ids", "holdout_row_ids", "fit_feature_sources",
    )
    total = sum(len(_seq(payload, k)) for k in keys)
    for ids in (payload.get("partitions") or {}).values():
        total += len(_seq({"v": ids}, "v"))
    lineage = payload.get("lineage") or {}
    total += sum(1 for _ in lineage)
    return total


def _mk_solver(z3):
    """A solver configured for unsat cores and a bounded run time."""
    s = z3.Solver()
    s.set(unsat_core=True)
    if _TIMEOUT_MS > 0:
        s.set("timeout", _TIMEOUT_MS)
    return s


class _Interner:
    """Stable bijection between arbitrary labels and small non-negative ints so
    set membership can be encoded as ``Or(e == i, ...)`` in z3."""

    def __init__(self) -> None:
        self._to_int: dict[str, int] = {}
        self._to_label: dict[int, str] = {}

    def i(self, label: object) -> int:
        s = str(label)
        if s not in self._to_int:
            n = len(self._to_int)
            self._to_int[s] = n
            self._to_label[n] = s
        return self._to_int[s]

    def label(self, n: int) -> str:
        return self._to_label.get(int(n), f"<{n}>")


# ---------------------------------------------------------------------------
# the decision procedure
# ---------------------------------------------------------------------------
def z3_decide(constraint: str, payload: Mapping[str, object]) -> Z3Decision:
    """Decide a structural constraint's violation predicate with z3.

    Returns a :class:`Z3Decision`; ``valid is True`` means the obligation holds
    (no violation; ``unsat``), ``valid is False`` means a violation witness was
    found (``sat``). When z3 is unavailable the result is ``skipped`` and callers
    fall back to the pure-Python procedure.
    """

    if not z3_available():
        return _skipped(constraint, "z3 not importable")
    try:
        import z3
    except Exception as exc:  # pragma: no cover
        return _skipped(constraint, f"z3 import failed: {exc}")

    # Bounded differential: skip the finite-set encoding on large payloads where
    # z3's O(n^2) membership query would blow up (the pure-Python decider is
    # exact and instant for these families). Arithmetic / interval families do
    # not have this blowup and are never skipped.
    if constraint in _SET_FAMILIES or (
            constraint == "split_disjointness" and not (payload.get("interval_partitions"))):
        terms = _payload_term_count(payload)
        if terms > _MAX_SET_TERMS:
            return _skipped(constraint,
                            f"payload too large for z3 set encoding ({terms} > {_MAX_SET_TERMS} terms)")

    try:
        if constraint == "schema_consistency":
            return _decide_schema(z3, payload)
        if constraint == "role_constraint":
            return _decide_role(z3, payload)
        if constraint == "join_safety":
            return _decide_join(z3, payload)
        if constraint == "no_outcome_in_feature":
            return _decide_no_outcome(z3, payload)
        if constraint == "column_lineage":
            return _decide_lineage(z3, payload)
        if constraint == "fit_transform_isolation":
            return _decide_fit_isolation(z3, payload)
        if constraint == "bounds":
            return _decide_bounds(z3, payload)
        if constraint == "partition_lengths":
            return _decide_partition_lengths(z3, payload)
        if constraint == "split_disjointness":
            return _decide_split(z3, payload)
        if constraint == "temporal_causality":
            return _decide_temporal(z3, payload)
        if constraint == "group_disjointness":
            return _decide_group(z3, payload)
        if constraint == "sampling_independence":
            return _decide_sampling(z3, payload)
        if constraint == "join_cardinality":
            return _decide_join_cardinality(z3, payload)
        if constraint == "value_domain":
            return _decide_value_domain(z3, payload)
    except Exception as exc:  # pragma: no cover - defensive
        return Z3Decision(constraint=constraint, available=True, result="unknown",
                          error=f"encoding error: {exc}")
    return _skipped(constraint, f"no z3 encoding for {constraint!r}")


def _finish(z3, constraint: str, theory: str, solver, witness_terms) -> Z3Decision:
    """Run ``solver`` (asserting the *violation*) and package the verdict."""
    res = solver.check()
    if res == z3.sat:
        model = solver.model()
        wit: dict[str, object] = {}
        for name, term in witness_terms(model):
            wit[name] = term
        return Z3Decision(constraint=constraint, available=True, result="sat",
                          theory=theory, valid=False, model=wit)
    if res == z3.unsat:
        core = tuple(str(c) for c in solver.unsat_core())
        return Z3Decision(constraint=constraint, available=True, result="unsat",
                          theory=theory, valid=True, unsat_core=core)
    return Z3Decision(constraint=constraint, available=True, result="unknown",
                      theory=theory, valid=None)


# --- finite-set families ---------------------------------------------------
def _membership(z3, var, ints):
    if not ints:
        return z3.BoolVal(False)
    return z3.Or([var == n for n in ints])


def _decide_schema(z3, payload) -> Z3Decision:
    intern = _Interner()
    declared = [intern.i(c) for c in _strs(payload, "declared_columns")]
    observed = [intern.i(c) for c in _strs(payload, "observed_columns")]
    e = z3.Int("col")
    s = _mk_solver(z3)
    # violation: a column in exactly one of the two sets (missing or unexpected)
    in_d, in_o = _membership(z3, e, declared), _membership(z3, e, observed)
    s.assert_and_track(z3.Or(z3.And(in_d, z3.Not(in_o)),
                             z3.And(in_o, z3.Not(in_d))), "schema_symmetric_difference")
    # restrict e to the finite universe so the search is well-founded
    universe = set(declared) | set(observed)
    s.assert_and_track(_membership(z3, e, sorted(universe)) if universe else z3.BoolVal(False),
                       "column_in_universe")

    def wit(m):
        yield "column", intern.label(m[e].as_long())
    return _finish(z3, "schema_consistency", "sets", s, wit)


def _decide_role(z3, payload) -> Z3Decision:
    intern = _Interner()
    roles = [intern.i(r) for r in _strs(payload, "roles")]
    allowed_labels = _strs(payload, "allowed_roles") or _strs(payload, "roles")
    allowed = [intern.i(r) for r in allowed_labels]
    e = z3.Int("role")
    s = _mk_solver(z3)
    s.assert_and_track(_membership(z3, e, roles), "role_present")
    s.assert_and_track(z3.Not(_membership(z3, e, allowed)), "role_not_allowed")

    def wit(m):
        yield "disallowed_role", intern.label(m[e].as_long())
    return _finish(z3, "role_constraint", "sets", s, wit)


def _decide_join(z3, payload) -> Z3Decision:
    intern = _Interner()
    keys = [intern.i(k) for k in _strs(payload, "join_keys")]
    targets = [intern.i(t) for t in _strs(payload, "target_columns")]
    e = z3.Int("key")
    s = _mk_solver(z3)
    s.assert_and_track(_membership(z3, e, keys), "is_join_key")
    s.assert_and_track(_membership(z3, e, targets), "is_target_column")

    def wit(m):
        yield "target_derived_join_key", intern.label(m[e].as_long())
    return _finish(z3, "join_safety", "sets", s, wit)


def _decide_no_outcome(z3, payload) -> Z3Decision:
    intern = _Interner()
    feats = [intern.i(c) for c in _strs(payload, "feature_sources")]
    outs = [intern.i(c) for c in _strs(payload, "outcome_columns")]
    e = z3.Int("col")
    s = _mk_solver(z3)
    s.assert_and_track(_membership(z3, e, feats), "is_feature_source")
    s.assert_and_track(_membership(z3, e, outs), "is_outcome_column")

    def wit(m):
        yield "outcome_column_in_features", intern.label(m[e].as_long())
    return _finish(z3, "no_outcome_in_feature", "sets", s, wit)


def _decide_lineage(z3, payload) -> Z3Decision:
    lineage = payload.get("lineage") or {}
    orphans = [str(d) for d, srcs in lineage.items() if not srcs]
    intern = _Interner()
    orphan_ints = [intern.i(o) for o in orphans]
    e = z3.Int("column")
    s = _mk_solver(z3)
    s.assert_and_track(_membership(z3, e, orphan_ints), "column_without_source")

    def wit(m):
        yield "column_without_source", intern.label(m[e].as_long())
    return _finish(z3, "column_lineage", "EUF", s, wit)


def _decide_fit_isolation(z3, payload) -> Z3Decision:
    intern = _Interner()
    fit_ids = [intern.i(v) for v in _strs(payload, "fit_row_ids")]
    holdout = [intern.i(v) for v in _strs(payload, "holdout_row_ids")]
    fit_src = [intern.i(v) for v in _strs(payload, "fit_feature_sources")]
    outs = [intern.i(v) for v in _strs(payload, "outcome_columns")]
    row = z3.Int("row")
    col = z3.Int("col")
    s = _mk_solver(z3)
    # violation: a row used both in fit and held out, OR an outcome column used as a fit source
    s.assert_and_track(
        z3.Or(
            z3.And(_membership(z3, row, fit_ids), _membership(z3, row, holdout)),
            z3.And(_membership(z3, col, fit_src), _membership(z3, col, outs)),
        ),
        "holdout_or_outcome_leak",
    )

    def wit(m):
        r, c = m[row], m[col]
        if r is not None and any(r.as_long() == n for n in fit_ids):
            yield "holdout_row_used_in_fit", intern.label(r.as_long())
        if c is not None and any(c.as_long() == n for n in fit_src):
            yield "outcome_column_used_in_fit", intern.label(c.as_long())
    return _finish(z3, "fit_transform_isolation", "sets", s, wit)


# --- arithmetic / interval families ---------------------------------------
def _to_float(x):
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _decide_bounds(z3, payload) -> Z3Decision:
    lo, hi = payload.get("lower"), payload.get("upper")
    flo, fhi = _to_float(lo), _to_float(hi)
    # malformed bound (provided but non-finite/non-numeric) is a violation
    for edge, raw, fv in (("lower", lo, flo), ("upper", hi, fhi)):
        if raw is not None and (fv is None or fv != fv):
            return Z3Decision("bounds", True, "sat", "LRA", valid=False,
                              model={"invalid_bound": edge, "value": raw})
    values = [_to_float(v) for v in _seq(payload, "values")]
    raw_values = list(_seq(payload, "values"))
    # NaN / non-numeric value is never within any bound -> violation
    for raw, fv in zip(raw_values, values):
        if fv is None or fv != fv:
            return Z3Decision("bounds", True, "sat", "LRA", valid=False,
                              model={"non_numeric_or_nan_value": raw})
    v = z3.Real("v")
    s = _mk_solver(z3)
    # restrict v to the observed values; violation: v < lo or v > hi
    if values:
        s.assert_and_track(z3.Or([v == z3.RealVal(repr(x)) for x in values]), "v_in_values")
    else:
        return Z3Decision("bounds", True, "unsat", "LRA", valid=True,
                          unsat_core=("no_values",))
    viol = []
    if flo is not None:
        viol.append(v < z3.RealVal(repr(flo)))
    if fhi is not None:
        viol.append(v > z3.RealVal(repr(fhi)))
    if not viol:
        return Z3Decision("bounds", True, "unsat", "LRA", valid=True,
                          unsat_core=("no_active_bounds",))
    s.assert_and_track(z3.Or(viol), "value_out_of_bounds")

    def wit(m):
        val = m[v]
        try:
            yield "out_of_bounds_value", float(val.as_fraction())
        except Exception:
            yield "out_of_bounds_value", str(val)
    return _finish(z3, "bounds", "LRA", s, wit)


def _decide_partition_lengths(z3, payload) -> Z3Decision:
    raw_lengths = list(_seq(payload, "lengths"))
    lengths = [_to_float(x) for x in raw_lengths]
    if any(x is None for x in lengths) or not lengths:
        return Z3Decision("partition_lengths", True, "sat", "LRA", valid=False,
                          model={"non_numeric_lengths": raw_lengths})
    fractions = bool(payload.get("fractions"))
    total = payload.get("total")
    ftotal = _to_float(total)
    if ftotal is None:
        ftotal = 1.0 if fractions else None
    tol = _to_float(payload.get("tolerance")) or (1e-9 if fractions else 0.0)
    s = _mk_solver(z3)
    consts = [z3.RealVal(repr(x)) for x in lengths]
    total_expr = z3.Sum(consts) if len(consts) > 1 else consts[0]
    viol = []
    # any non-positive partition is a violation
    nonpos = z3.Or([c <= 0 for c in consts])
    viol.append(nonpos)
    if ftotal is not None:
        # |sum - total| > tol
        diff_hi = total_expr > z3.RealVal(repr(ftotal + tol))
        diff_lo = total_expr < z3.RealVal(repr(ftotal - tol))
        viol.append(z3.Or(diff_hi, diff_lo))
    s.assert_and_track(z3.Or(viol), "partition_lengths_violation")

    def wit(m):
        # report the computed sum and which rule fired
        ssum = sum(lengths)
        out: dict[str, object] = {"sum": ssum, "lengths": lengths}
        if ftotal is not None:
            out["expected_total"] = ftotal
        if any(x <= 0 for x in lengths):
            out["non_positive_partition"] = True
        yield "partition", out
    return _finish(z3, "partition_lengths", "LRA", s, wit)


def _interval_partitions(payload) -> dict[str, tuple[float, float]]:
    raw = payload.get("interval_partitions") or {}
    out: dict[str, tuple[float, float]] = {}
    for name, bounds in raw.items():
        seq = list(bounds)
        if len(seq) >= 2:
            lo, hi = _to_float(seq[0]), _to_float(seq[1])
            if lo is not None and hi is not None:
                out[str(name)] = (lo, hi)
    return out


def _decide_split(z3, payload) -> Z3Decision:
    intervals = _interval_partitions(payload)
    if intervals:
        # half-open [lo, hi) integer intervals; violation: an index in two of them
        names = sorted(intervals)
        idx = z3.Int("idx")
        s = _mk_solver(z3)
        clauses = []
        for i, a in enumerate(names):
            la, ha = intervals[a]
            in_a = z3.And(idx >= z3.IntVal(int(la)), idx < z3.IntVal(int(ha)))
            for b in names[i + 1:]:
                lb, hb = intervals[b]
                in_b = z3.And(idx >= z3.IntVal(int(lb)), idx < z3.IntVal(int(hb)))
                clauses.append(z3.And(in_a, in_b))
        if not clauses:
            return Z3Decision("split_disjointness", True, "unsat", "LIA",
                              valid=True, unsat_core=("single_partition",))
        s.assert_and_track(z3.Or(clauses), "interval_overlap")

        def wit(m):
            i = m[idx].as_long()
            members = [n for n in names if intervals[n][0] <= i < intervals[n][1]]
            yield "overlapping_index", i
            yield "in_partitions", members
        return _finish(z3, "split_disjointness", "LIA", s, wit)

    # set-of-ids partitions: violation = an id in two distinct partitions
    partitions = payload.get("partitions") or {}
    intern = _Interner()
    enc = {str(n): [intern.i(v) for v in ids] for n, ids in partitions.items()}
    names = sorted(enc)
    e = z3.Int("id")
    s = _mk_solver(z3)
    clauses = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            clauses.append(z3.And(_membership(z3, e, enc[a]), _membership(z3, e, enc[b])))
    if not clauses:
        return Z3Decision("split_disjointness", True, "unsat", "sets",
                          valid=True, unsat_core=("single_partition",))
    s.assert_and_track(z3.Or(clauses), "id_in_two_partitions")

    def wit(m):
        ident = intern.label(m[e].as_long())
        members = [n for n in names if ident in [str(x) for x in partitions[n]]]
        yield "shared_id", ident
        yield "in_partitions", members
    return _finish(z3, "split_disjointness", "sets", s, wit)


def _decide_temporal(z3, payload) -> Z3Decision:
    """Search the index axis for a feature row that reads a future source row.

    The feature's composed forward reach ``R`` (summed over its operator chain)
    is the only datum.  We assert the *violation* predicate over two integer
    index variables and let z3 find a concrete counterexample:

        0 <= i < N,  0 <= j < N,  i < j <= i + R

    plus, when a chronological ``cut`` is supplied, ``i <= cut < j`` (a training
    row reading a held-out row).  ``R == 0`` makes ``i < j <= i`` unsatisfiable,
    so a causal/trailing feature is *unsat* (valid).  This is genuine LIA search,
    exactly analogous to the S1 interval-overlap encoding.
    """
    try:
        reach = int(float(payload.get("forward_reach", 0)))
    except (TypeError, ValueError):
        return Z3Decision("temporal_causality", True, "sat", "LIA", valid=False,
                          model={"non_numeric_forward_reach": payload.get("forward_reach")})
    cut_raw = payload.get("cut")
    try:
        cut = int(float(cut_raw)) if cut_raw is not None else None
    except (TypeError, ValueError):
        cut = None
    horizon = payload.get("horizon")
    try:
        n = int(float(horizon)) if horizon is not None else None
    except (TypeError, ValueError):
        n = None
    if n is None:
        n = max(reach + 2, (cut + 2) if cut is not None else 0, 2)

    i = z3.Int("feature_row")
    j = z3.Int("future_source_row")
    s = _mk_solver(z3)
    base = [i >= 0, i < z3.IntVal(n), j >= 0, j < z3.IntVal(n),
            j > i, j <= i + z3.IntVal(reach)]
    if cut is not None:
        base += [i <= z3.IntVal(cut), j > z3.IntVal(cut)]
    s.assert_and_track(z3.And(*base), "future_dependency")

    def wit(m):
        yield "feature_row", m[i].as_long()
        yield "future_source_row", m[j].as_long()
        yield "forward_reach", reach
    return _finish(z3, "temporal_causality", "LIA", s, wit)


def _decide_group(z3, payload) -> Z3Decision:
    """Search the member->partition assignment for a group that straddles the cut.

    The group key induces an equivalence relation ``~`` on the rows; the contract
    is *quotient disjointness* --- the partitions must be disjoint in ``X/~``.  We
    model one representative group of ``m`` members and ``p`` partitions with one
    integer assignment variable per member, ``0 <= member_k < p``:

    * a **group-blind** (row-level) split assigns the members independently, so
      the violation predicate ``Exists i<j: member_i != member_j`` (two members
      of the *same* group in *different* partitions) is satisfiable --- z3 returns
      a concrete straddling pair (the leakage witness);
    * a **group-aware** split (GroupKFold / GroupShuffleSplit / ``groups=``) adds
      ``member_k == member_0`` for every member, forcing the whole group into one
      partition, so the violation is unsatisfiable (valid).

    This is genuine LIA search over the assignment space, exactly analogous to the
    S1 interval-overlap encoding --- the group-awareness flag is the load-bearing
    datum that flips sat/unsat.
    """
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
    a = [z3.Int(f"member_{k}") for k in range(members)]
    s = _mk_solver(z3)
    s.add(z3.And(*[z3.And(v >= 0, v < z3.IntVal(parts)) for v in a]))
    if aware:
        for v in a[1:]:
            s.add(v == a[0])
    viol = z3.Or([a[i] != a[j]
                  for i in range(members) for j in range(i + 1, members)])
    s.assert_and_track(viol, "members_split_across_partitions")

    def wit(m):
        assign = [int(m[v].as_long()) for v in a]
        pair = next(((i, j) for i in range(members)
                     for j in range(i + 1, members) if assign[i] != assign[j]),
                    (0, 1))
        yield "group_key", key
        yield "member_a", pair[0]
        yield "partition_a", assign[pair[0]]
        yield "member_b", pair[1]
        yield "partition_b", assign[pair[1]]
    return _finish(z3, "group_disjointness", "LIA", s, wit)


def _decide_sampling(z3, payload) -> Z3Decision:
    """Decide the Dataset/DataLoader stochastic-effect contract.

    Booleans model the sampling path's effects and an integer models the worker
    count; z3 searches for an assignment satisfying the *violation* predicate

        (is_eval AND stochastic_eval)
        OR (NOT is_eval AND global_rng AND num_workers >= 1 AND NOT worker_init_fn)

    The first disjunct is eval nondeterminism (a stochastic transform on the test
    path); the second is fork-correlation (forked workers share one process-global
    RNG state with no per-worker reseed).  All inputs are fixed by the payload, so
    this is genuine (if small) decision over LIA + booleans, decided independently
    of the pure-Python procedure for the differential oracle.
    """
    try:
        workers_val = int(float(payload.get("num_workers", 0)))
    except (TypeError, ValueError):
        workers_val = 0
    global_rng = bool(payload.get("global_rng"))
    worker_init = bool(payload.get("worker_init_fn"))
    is_eval = bool(payload.get("is_eval"))
    stochastic_eval = bool(payload.get("stochastic_eval"))

    workers = z3.Int("num_workers")
    s = _mk_solver(z3)
    s.add(workers == z3.IntVal(workers_val))
    eval_bad = z3.And(z3.BoolVal(is_eval), z3.BoolVal(stochastic_eval))
    fork_bad = z3.And(z3.BoolVal(not is_eval), z3.BoolVal(global_rng),
                      workers >= 1, z3.BoolVal(not worker_init))
    s.assert_and_track(z3.Or(eval_bad, fork_bad), "non_independent_sampling")

    def wit(m):
        w = m[workers].as_long()
        if is_eval and stochastic_eval:
            yield "violation", "eval_nondeterminism"
        else:
            yield "violation", "correlated_worker_rng"
            yield "num_workers", w
    return _finish(z3, "sampling_independence", "QF_LIA", s, wit)


def _decide_join_cardinality(z3, payload) -> Z3Decision:
    """Decide the relational-multiplicity (join fan-out) contract.

    Integers model the join's cardinality arithmetic and z3 searches for a
    right-key multiplicity that fans the result out past the left cardinality:

        result_rows == left_rows * right_mult
        AND right_mult >= 2 AND result_rows > left_rows
        AND NOT (validated OR right_key_unique OR NOT cardinality_consumed)

    ``left_rows`` is fixed by the payload so the product is *linear* in the
    searched multiplicity (decidable LIA, no nonlinear blow-up).  A ``validate=``
    guard, a right operand made unique on the key, or a result that is never
    consumed in a cardinality-sensitive way makes the violation unsatisfiable
    (admitted).  Decided independently of the pure-Python procedure for the
    differential oracle, with the fan-out factor as a concrete witness.
    """
    validated = bool(payload.get("validated"))
    right_unique = bool(payload.get("right_key_unique"))
    consumed = bool(payload.get("cardinality_consumed", True))
    try:
        left_rows_val = max(int(float(payload.get("left_rows", 100))), 1)
    except (TypeError, ValueError):
        left_rows_val = 100
    key = str(payload.get("join_key", "key"))
    how = str(payload.get("how", "inner"))

    left_rows = z3.Int("left_rows")
    right_mult = z3.Int("right_mult")
    result_rows = z3.Int("result_rows")
    s = _mk_solver(z3)
    s.add(left_rows == z3.IntVal(left_rows_val))
    s.add(result_rows == left_rows * right_mult)
    guard_safe = z3.Or(z3.BoolVal(validated), z3.BoolVal(right_unique),
                       z3.BoolVal(not consumed))
    viol = z3.And(z3.Not(guard_safe), right_mult >= 2, result_rows > left_rows)
    s.assert_and_track(viol, "unvalidated_join_fanout")

    def wit(m):
        yield "violation", "join_fanout"
        yield "join_key", key
        yield "how", how
        yield "right_multiplicity", m[right_mult].as_long()
        yield "left_rows", m[left_rows].as_long()
        yield "result_rows", m[result_rows].as_long()
    return _finish(z3, "join_cardinality", "LIA", s, wit)


def _decide_value_domain(z3, payload) -> Z3Decision:
    """Decide the value-domain refinement contract for a loss input.

    A real ``v`` models the loss input's value.  When a domain-normalising
    producer is established (``sigmoid`` for BCE, ``log_softmax`` for NLL) we
    constrain ``v`` to the required interval and assert the breach predicate

        v < required_lo  OR  v > required_hi

    which is then *unsat* (the input provably stays in domain -> admitted).
    Without the producer ``v`` is free and z3 returns a concrete out-of-domain
    witness (sat -> rejected).  This is the numeric-value complement to a
    shape-only verifier such as thehalleyyoung/tensorguard: TensorGuard proves
    the *shape* refinement of the same loss call, this proves the *value-domain*
    refinement -- decided over LRA, independently of the pure-Python procedure.
    """

    op = str(payload.get("op", "bce"))
    loss = str(payload.get("loss", op))
    producer = str(payload.get("producer", "unknown"))
    established = bool(payload.get("domain_established"))
    flo = _to_float(payload.get("required_lo"))
    fhi = _to_float(payload.get("required_hi"))

    v = z3.Real("v")
    s = _mk_solver(z3)
    if established:
        if flo is not None:
            s.add(v >= z3.RealVal(repr(flo)))
        if fhi is not None:
            s.add(v <= z3.RealVal(repr(fhi)))
    viol = []
    if flo is not None:
        viol.append(v < z3.RealVal(repr(flo)))
    if fhi is not None:
        viol.append(v > z3.RealVal(repr(fhi)))
    if not viol:
        # No active domain bound to violate -> nothing to reject.
        return Z3Decision("value_domain", True, "unsat", "LRA", valid=True,
                          unsat_core=("no_active_domain",))
    s.assert_and_track(z3.Or(viol), "value_domain_breach")

    def wit(m):
        yield "violation", "value_domain_breach"
        yield "op", op
        yield "loss", loss
        yield "producer", producer
        val = m[v]
        try:
            yield "out_of_domain_value", float(val.as_fraction())
        except Exception:
            yield "out_of_domain_value", str(val)
        yield "required_lo", payload.get("required_lo")
        yield "required_hi", payload.get("required_hi")
    return _finish(z3, "value_domain", "LRA", s, wit)


# ---------------------------------------------------------------------------
# independent proof re-checker (STEPS.md 334)
# ---------------------------------------------------------------------------
def recheck(constraint: str, payload: Mapping[str, object],
            proof: Mapping[str, object]) -> tuple[bool, str]:
    """Independently re-validate a proof object against the lowered constraint.

    A *rejection* proof carries a counterexample ``model``; we confirm the model
    actually violates the constraint. An *admission* proof carries an
    ``unsat_core``; we re-run z3 on the full violation predicate and confirm it
    is unsatisfiable (no violation exists). Returns ``(ok, detail)``.
    """

    valid_claim = proof.get("valid")
    if valid_claim is False:
        ok, detail = _recheck_counterexample(constraint, payload, proof.get("model") or {})
        return ok, detail
    if valid_claim is True:
        if not z3_available():
            return False, "cannot re-check admission without z3"
        dec = z3_decide(constraint, payload)
        if dec.result == "unsat" and dec.valid is True:
            return True, "re-derived unsat: no violation exists"
        return False, f"admission not reproduced (got result={dec.result})"
    return False, "proof carries no decidable validity claim"


def _recheck_counterexample(constraint: str, payload: Mapping[str, object],
                            model: Mapping[str, object]) -> tuple[bool, str]:
    """Confirm a counterexample genuinely violates the constraint, using a
    minimal evaluator independent of the main decision procedure."""

    if constraint == "split_disjointness":
        intervals = _interval_partitions(payload)
        if intervals and "overlapping_index" in model:
            i = float(model["overlapping_index"])
            members = [n for n, (lo, hi) in intervals.items() if lo <= i < hi]
            ok = len(members) >= 2
            return ok, f"index {i} in partitions {sorted(members)}"
        if "shared_id" in model:
            partitions = payload.get("partitions") or {}
            ident = str(model["shared_id"])
            members = [n for n, ids in partitions.items() if ident in [str(x) for x in ids]]
            ok = len(members) >= 2
            return ok, f"id {ident!r} in partitions {sorted(members)}"
        return False, "counterexample shape not recognised"
    if constraint == "temporal_causality":
        try:
            reach = int(float(payload.get("forward_reach", 0)))
        except (TypeError, ValueError):
            return ("non_numeric_forward_reach" in model), "non-numeric forward reach"
        i = _to_float(model.get("feature_row"))
        j = _to_float(model.get("future_source_row"))
        if i is None or j is None:
            return ("non_numeric_forward_reach" in model), "malformed temporal counterexample"
        ok = (j > i) and (j <= i + reach) and reach >= 1
        return ok, f"feature row {i} reads future source row {j} within reach {reach}"
    if constraint == "group_disjointness":
        if bool(payload.get("group_aware")):
            return False, "group-aware split keeps a group together; cannot straddle"
        pa, pb = model.get("partition_a"), model.get("partition_b")
        try:
            ok = pa is not None and pb is not None and int(pa) != int(pb)
        except (TypeError, ValueError):
            ok = False
        return ok, (f"group {model.get('group_key')!r} has members in partitions "
                    f"{pa} and {pb}")
    if constraint == "sampling_independence":
        v = model.get("violation")
        if v == "eval_nondeterminism":
            ok = bool(payload.get("is_eval")) and bool(payload.get("stochastic_eval"))
            return ok, "stochastic augmentation on an evaluation path"
        if v == "correlated_worker_rng":
            try:
                workers = int(float(payload.get("num_workers", 0)))
            except (TypeError, ValueError):
                workers = 0
            ok = (bool(payload.get("global_rng")) and workers >= 1
                  and not bool(payload.get("worker_init_fn"))
                  and not bool(payload.get("is_eval")))
            return ok, (f"global RNG under {workers} forked workers without "
                        "worker_init_fn reseed")
        return False, "sampling counterexample shape not recognised"
    if constraint == "join_cardinality":
        if model.get("violation") != "join_fanout":
            return False, "join counterexample shape not recognised"
        validated = bool(payload.get("validated"))
        right_unique = bool(payload.get("right_key_unique"))
        consumed = bool(payload.get("cardinality_consumed", True))
        try:
            mult = int(float(model.get("right_multiplicity", 0)))
        except (TypeError, ValueError):
            mult = 0
        ok = consumed and not validated and not right_unique and mult >= 2
        return ok, (f"join on {model.get('join_key')!r} fans out x{mult} without "
                    "validate= or a deduplicated right key")
    if constraint == "value_domain":
        if model.get("violation") != "value_domain_breach":
            return False, "value-domain counterexample shape not recognised"
        established = bool(payload.get("domain_established"))
        val = _to_float(model.get("out_of_domain_value"))
        flo = _to_float(payload.get("required_lo"))
        fhi = _to_float(payload.get("required_hi"))
        breaches = val is not None and (
            (flo is not None and val < flo) or (fhi is not None and val > fhi))
        ok = (not established) and breaches
        return ok, (f"{model.get('loss')} input value {model.get('out_of_domain_value')} "
                    f"breaches required domain [{payload.get('required_lo')}, "
                    f"{payload.get('required_hi')}] with no normalising producer "
                    f"({model.get('producer')!r})")
    if constraint == "partition_lengths":
        lengths = [float(x) for x in _seq(payload, "lengths")]
        fractions = bool(payload.get("fractions"))
        total = _to_float(payload.get("total"))
        if total is None:
            total = 1.0 if fractions else None
        tol = _to_float(payload.get("tolerance")) or (1e-9 if fractions else 0.0)
        bad = any(x <= 0 for x in lengths)
        if total is not None and abs(sum(lengths) - total) > tol:
            bad = True
        return bad, f"sum={sum(lengths)} expected={total}"
    if constraint == "bounds":
        lo, hi = _to_float(payload.get("lower")), _to_float(payload.get("upper"))
        if "out_of_bounds_value" in model:
            v = _to_float(model["out_of_bounds_value"])
            if v is None:
                return False, "non-numeric witness"
            bad = (lo is not None and v < lo) or (hi is not None and v > hi)
            return bad, f"value {v} vs [{lo}, {hi}]"
        if "invalid_bound" in model or "non_numeric_or_nan_value" in model:
            return True, "malformed bound / value witnessed"
        return False, "bounds counterexample shape not recognised"
    # finite-set families: confirm the named witness is in the offending sets
    pairs = {
        "schema_consistency": ("column", ("declared_columns", "observed_columns")),
        "role_constraint": ("disallowed_role", ("roles",)),
        "join_safety": ("target_derived_join_key", ("join_keys", "target_columns")),
        "no_outcome_in_feature": ("outcome_column_in_features", ("feature_sources", "outcome_columns")),
    }
    if constraint in pairs:
        key, sets = pairs[constraint]
        token = model.get(key)
        if token is None:
            return False, "no witness token"
        present = any(str(token) in _strs(payload, s) for s in sets)
        return present, f"witness {token!r} present in {sets}"
    if constraint == "fit_transform_isolation":
        fit = set(_strs(payload, "fit_row_ids"))
        hold = set(_strs(payload, "holdout_row_ids"))
        fsrc = set(_strs(payload, "fit_feature_sources"))
        outs = set(_strs(payload, "outcome_columns"))
        token_r = model.get("holdout_row_used_in_fit")
        token_c = model.get("outcome_column_used_in_fit")
        ok = (token_r is not None and str(token_r) in (fit & hold)) or \
             (token_c is not None and str(token_c) in (fsrc & outs))
        return ok, "fit/holdout or outcome overlap witnessed"
    return False, f"no re-checker for {constraint!r}"


__all__ = [
    "Z3Decision",
    "z3_available",
    "z3_version",
    "z3_options",
    "z3_decide",
    "recheck",
]
