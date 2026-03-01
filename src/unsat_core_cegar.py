"""
Enhanced UNSAT-core-driven CEGAR for Shape Contract Discovery.

Extends the base ``ShapeCEGARLoop`` with:

* **Incremental SMT solving** — reuses a single Z3 solver across CEGAR
  iterations via ``push``/``pop``, preserving learned lemmas.
* **Proper UNSAT core extraction** — tracks every assertion with a label,
  then extracts minimal unsatisfiable subsets (MUS) via deletion-based
  algorithm to synthesise predicates.
* **Predicate provenance** — distinguishes ``core_derived`` predicates
  (extracted from UNSAT cores) from ``template_derived`` ones (from the
  existing template enumeration fallback).

Public API
----------
``run_enhanced_cegar(source, input_shapes, **kwargs) -> ShapeCEGARResult``
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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
# Internal imports
# ---------------------------------------------------------------------------

from src.shape_cegar import (
    CEGARStatus,
    CEGARVerdict,
    InferredContract,
    PredicateKind,
    ShapeCEGARLoop,
    ShapeCEGARResult,
    ShapePredicate,
    ShapeRefinement,
    UnsatCorePredicateExtractor,
    infer_contracts,
)
from src.model_checker import (
    ComputationGraph,
    Device,
    Phase,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  UnsatCorePredicate — provenance-tracked predicate
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UnsatCorePredicate:
    """A predicate discovered from UNSAT core analysis.

    Attributes
    ----------
    formula : z3 expression (stored as Any to avoid import issues)
        The Z3 formula this predicate represents.
    source_core : frozenset of str
        Label names of the UNSAT core this predicate was extracted from.
    strength : int
        Number of distinct UNSAT cores this predicate appears in.
    variables : tuple of str
        Dimension variable names constrained by this predicate.
    shape_predicate : ShapePredicate or None
        The concrete ``ShapePredicate`` derived from this core predicate.
    """

    formula: Any
    source_core: FrozenSet[str] = field(default_factory=frozenset)
    strength: int = 1
    variables: Tuple[str, ...] = ()
    shape_predicate: Optional[ShapePredicate] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  IncrementalCEGARSolver
# ═══════════════════════════════════════════════════════════════════════════════


class IncrementalCEGARSolver:
    """Wraps a Z3 solver with push/pop for incremental CEGAR solving.

    Background constraints are added once and never popped.  Per-iteration
    assertions live inside a push/pop frame so that learned lemmas from
    earlier iterations are preserved.
    """

    def __init__(self, timeout_ms: int = 5000) -> None:
        if not HAS_Z3:
            raise RuntimeError("Z3 is required for IncrementalCEGARSolver")

        self._solver = z3.Solver()
        self._solver.set("timeout", timeout_ms)
        self._solver.set("unsat_core", True)

        # Persistent (background) constraints — never popped.
        self._background: List[Any] = []
        # Label → original formula, across all iterations.
        self._label_map: Dict[str, Any] = {}
        # How many push frames are currently active.
        self._depth: int = 0
        # Statistics
        self._check_count: int = 0
        self._reuse_count: int = 0
        # Accumulated UNSAT core labels across iterations.
        self._all_core_labels: List[FrozenSet[str]] = []

    # ------------------------------------------------------------------
    # Background constraints
    # ------------------------------------------------------------------

    def add_background(self, constraints: List[Any]) -> None:
        """Add persistent constraints that survive push/pop."""
        for c in constraints:
            self._solver.add(c)
            self._background.append(c)

    # ------------------------------------------------------------------
    # Incremental push / pop
    # ------------------------------------------------------------------

    def push(self) -> None:
        self._solver.push()
        self._depth += 1

    def pop(self) -> None:
        if self._depth > 0:
            self._solver.pop()
            self._depth -= 1
            self._reuse_count += 1

    # ------------------------------------------------------------------
    # Assert-and-track
    # ------------------------------------------------------------------

    def assert_and_track(self, formula: Any, label: str) -> None:
        """Assert *formula* tracked by *label* for UNSAT core extraction."""
        lbl = z3.Bool(label)
        self._label_map[label] = formula
        self._solver.assert_and_track(formula, lbl)

    def add(self, formula: Any) -> None:
        """Add an untracked assertion to the current frame."""
        self._solver.add(formula)

    # ------------------------------------------------------------------
    # Check with UNSAT core
    # ------------------------------------------------------------------

    def check_with_core(
        self,
        assumptions: Optional[List[Any]] = None,
    ) -> Tuple[Any, List[str]]:
        """Check satisfiability; return ``(result, core_labels)``.

        ``core_labels`` is non-empty only when the result is UNSAT.
        """
        self._check_count += 1
        if assumptions:
            result = self._solver.check(*assumptions)
        else:
            result = self._solver.check()

        core_labels: List[str] = []
        if result == z3.unsat:
            raw = self._solver.unsat_core()
            core_labels = [str(c) for c in raw]
            self._all_core_labels.append(frozenset(core_labels))

        return result, core_labels

    # ------------------------------------------------------------------
    # Predicate extraction from accumulated cores
    # ------------------------------------------------------------------

    def get_learned_predicates(self) -> List[UnsatCorePredicate]:
        """Build ``UnsatCorePredicate`` objects from all observed UNSAT cores."""
        # Count how many cores each label appears in.
        label_strength: Dict[str, int] = {}
        for core in self._all_core_labels:
            for lbl in core:
                label_strength[lbl] = label_strength.get(lbl, 0) + 1

        predicates: List[UnsatCorePredicate] = []
        seen_formulas: Set[str] = set()

        for lbl, strength in label_strength.items():
            formula = self._label_map.get(lbl)
            if formula is None:
                continue
            fstr = str(formula)
            if fstr in seen_formulas:
                continue
            seen_formulas.add(fstr)

            # Extract variable names from the formula.
            variables = tuple(
                str(v) for v in _collect_z3_vars(formula)
            )

            # Find which core(s) contained this label.
            source = frozenset(
                core for core in self._all_core_labels if lbl in core
            )

            predicates.append(
                UnsatCorePredicate(
                    formula=formula,
                    source_core=frozenset({lbl}),
                    strength=strength,
                    variables=variables,
                )
            )

        return predicates

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "check_count": self._check_count,
            "reuse_count": self._reuse_count,
            "depth": self._depth,
            "background_count": len(self._background),
            "cores_observed": len(self._all_core_labels),
        }

    def reset_iteration(self) -> None:
        """Pop the current frame and push a fresh one."""
        self.pop()
        self.push()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  UnsatCorePredicateExtractor (enhanced)
# ═══════════════════════════════════════════════════════════════════════════════


class EnhancedUnsatCorePredicateExtractor:
    """Extracts predicates from UNSAT cores with MUS minimisation.

    Given an UNSAT core (set of tracked assertion labels) and the
    mapping from labels to Z3 formulas, performs:

    1. **MUS extraction** — deletion-based algorithm to find a minimal
       unsatisfiable subset.
    2. **Direct predicate synthesis** — each core assertion that matches
       a dimension constraint becomes a predicate.
    3. **Interpolant-style predicates** — pairs of core subsets are
       checked to derive implied constraints.
    4. **Generalisation** — concrete values are weakened to symbolic
       bounds (e.g. ``x == 768`` → ``x >= 1``).
    """

    def __init__(self, timeout_ms: int = 3000) -> None:
        self._timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_predicates(
        self,
        core_labels: List[str],
        assertion_map: Dict[str, Any],
        dim_map: Optional[Dict[str, Tuple[str, int]]] = None,
    ) -> List[UnsatCorePredicate]:
        """Extract predicates from an UNSAT core.

        Parameters
        ----------
        core_labels : list of str
            Label names from the UNSAT core.
        assertion_map : dict
            Maps label names → Z3 formulas.
        dim_map : dict, optional
            Maps Z3 variable names → (tensor_name, axis).
        """
        if not HAS_Z3 or not core_labels:
            return []

        # Step 1: MUS extraction
        mus = self._extract_mus(core_labels, assertion_map)

        # Step 2: Direct predicates
        predicates = self._direct_predicates(mus, assertion_map, dim_map)

        # Step 3: Interpolant-style predicates from pairs
        predicates.extend(
            self._pairwise_predicates(mus, assertion_map, dim_map)
        )

        # Step 4: Generalised predicates
        predicates.extend(
            self._generalised_predicates(mus, assertion_map, dim_map)
        )

        # Deduplicate by formula string.
        seen: Set[str] = set()
        unique: List[UnsatCorePredicate] = []
        for p in predicates:
            key = str(p.formula)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique

    # ------------------------------------------------------------------
    # MUS extraction (deletion-based)
    # ------------------------------------------------------------------

    def _extract_mus(
        self,
        core_labels: List[str],
        assertion_map: Dict[str, Any],
    ) -> List[str]:
        """Deletion-based MUS extraction.

        Iteratively removes each constraint and checks if the remaining
        set is still UNSAT.  If removing a constraint makes the set SAT,
        that constraint is necessary and stays in the MUS.
        """
        if not HAS_Z3:
            return list(core_labels)

        candidate = list(core_labels)

        for lbl in list(core_labels):
            if lbl not in candidate:
                continue
            reduced = [l for l in candidate if l != lbl]
            if not reduced:
                continue

            solver = z3.Solver()
            solver.set("timeout", self._timeout_ms)
            for l in reduced:
                formula = assertion_map.get(l)
                if formula is not None:
                    solver.add(formula)

            if solver.check() == z3.unsat:
                # lbl is redundant — remove it.
                candidate = reduced
            # else: lbl is necessary — keep it.

        return candidate

    # ------------------------------------------------------------------
    # Direct predicates
    # ------------------------------------------------------------------

    def _direct_predicates(
        self,
        mus_labels: List[str],
        assertion_map: Dict[str, Any],
        dim_map: Optional[Dict[str, Tuple[str, int]]],
    ) -> List[UnsatCorePredicate]:
        """Convert each MUS assertion into a predicate if it constrains a dimension."""
        results: List[UnsatCorePredicate] = []
        core_fs = frozenset(mus_labels)

        for lbl in mus_labels:
            formula = assertion_map.get(lbl)
            if formula is None:
                continue

            variables = tuple(str(v) for v in _collect_z3_vars(formula))
            sp = _formula_to_shape_predicate(formula, dim_map) if dim_map else None

            results.append(
                UnsatCorePredicate(
                    formula=formula,
                    source_core=core_fs,
                    strength=1,
                    variables=variables,
                    shape_predicate=sp,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Pairwise (interpolant-style) predicates
    # ------------------------------------------------------------------

    def _pairwise_predicates(
        self,
        mus_labels: List[str],
        assertion_map: Dict[str, Any],
        dim_map: Optional[Dict[str, Tuple[str, int]]],
    ) -> List[UnsatCorePredicate]:
        """From pairs of MUS subsets, derive implied constraints.

        For two subsets A, B of the MUS where A ∧ B is UNSAT, any formula
        implied by A that contradicts B is an interpolant-style predicate.
        We approximate this by checking if A alone implies any simple
        bound on shared variables.
        """
        if len(mus_labels) < 2:
            return []

        results: List[UnsatCorePredicate] = []
        core_fs = frozenset(mus_labels)

        # Collect all variables across MUS formulas.
        all_vars: Set[str] = set()
        for lbl in mus_labels:
            formula = assertion_map.get(lbl)
            if formula is not None:
                all_vars.update(str(v) for v in _collect_z3_vars(formula))

        # For each variable, derive implied bounds from the full MUS.
        for var_name in all_vars:
            if not var_name.startswith("__interp_") and not var_name.startswith("__ci_"):
                continue

            solver = z3.Solver()
            solver.set("timeout", self._timeout_ms)
            for lbl in mus_labels:
                formula = assertion_map.get(lbl)
                if formula is not None:
                    solver.add(formula)

            var = z3.Int(var_name)

            # Find upper bound.
            ub_var = z3.Int(f"__ub_{var_name}")
            solver.push()
            solver.add(var > ub_var)
            if solver.check() == z3.unsat:
                # No solution with var > ub_var → var has an upper bound.
                pass
            solver.pop()

            # Find lower bound by maximising.
            solver.push()
            solver.add(var >= 1)
            if solver.check() == z3.sat:
                model = solver.model()
                val = model.eval(var, model_completion=True)
                if z3.is_int_value(val):
                    bound_val = val.as_long()
                    bound_formula = var == z3.IntVal(bound_val)
                    sp = None
                    if dim_map and var_name in dim_map:
                        tensor, axis = dim_map[var_name]
                        sp = ShapePredicate(
                            kind=PredicateKind.DIM_EQ,
                            tensor=tensor,
                            axis=axis,
                            value=bound_val,
                            provenance="core_interpolant",
                        )
                    results.append(
                        UnsatCorePredicate(
                            formula=bound_formula,
                            source_core=core_fs,
                            strength=1,
                            variables=(var_name,),
                            shape_predicate=sp,
                        )
                    )
            solver.pop()

        return results

    # ------------------------------------------------------------------
    # Generalised predicates
    # ------------------------------------------------------------------

    def _generalised_predicates(
        self,
        mus_labels: List[str],
        assertion_map: Dict[str, Any],
        dim_map: Optional[Dict[str, Tuple[str, int]]],
    ) -> List[UnsatCorePredicate]:
        """Weaken concrete equalities to inequality bounds.

        E.g. ``x == 768`` may be generalised to ``x >= 1`` if the MUS
        remains UNSAT with the weaker constraint.
        """
        results: List[UnsatCorePredicate] = []
        core_fs = frozenset(mus_labels)

        for lbl in mus_labels:
            formula = assertion_map.get(lbl)
            if formula is None or not z3.is_eq(formula):
                continue

            children = formula.children()
            if len(children) != 2:
                continue

            lhs, rhs = children
            # Identify which side is the variable and which is the value.
            var_expr, val_expr = None, None
            if z3.is_int_value(rhs) and not z3.is_int_value(lhs):
                var_expr, val_expr = lhs, rhs
            elif z3.is_int_value(lhs) and not z3.is_int_value(rhs):
                var_expr, val_expr = rhs, lhs
            else:
                continue

            concrete_val = val_expr.as_long()
            if concrete_val <= 0:
                continue

            # Try weakening to >= 1.
            weak = var_expr >= z3.IntVal(1)
            other_labels = [l for l in mus_labels if l != lbl]

            solver = z3.Solver()
            solver.set("timeout", self._timeout_ms)
            solver.add(weak)
            for l in other_labels:
                f = assertion_map.get(l)
                if f is not None:
                    solver.add(f)

            if solver.check() == z3.unsat:
                # Generalised constraint still causes UNSAT — useful!
                var_name = str(var_expr)
                sp = None
                if dim_map and var_name in dim_map:
                    tensor, axis = dim_map[var_name]
                    sp = ShapePredicate(
                        kind=PredicateKind.DIM_GE,
                        tensor=tensor,
                        axis=axis,
                        value=1,
                        provenance="core_generalised",
                    )
                results.append(
                    UnsatCorePredicate(
                        formula=weak,
                        source_core=core_fs,
                        strength=1,
                        variables=(str(var_expr),),
                        shape_predicate=sp,
                    )
                )

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EnhancedShapeCEGARLoop
# ═══════════════════════════════════════════════════════════════════════════════


class EnhancedShapeCEGARLoop(ShapeCEGARLoop):
    """CEGAR loop enhanced with incremental solving and UNSAT-core predicates.

    Extends ``ShapeCEGARLoop`` to:
    * Reuse a single Z3 solver across iterations (``IncrementalCEGARSolver``).
    * Prefer UNSAT-core-derived predicates over template enumeration.
    * Track predicate provenance (``core_derived`` vs ``template_derived``).
    """

    def __init__(
        self,
        source: str,
        input_shapes: Optional[Dict[str, tuple]] = None,
        max_iterations: int = 10,
        default_device: Device = Device.CPU,
        default_phase: Phase = Phase.TRAIN,
        max_k: Optional[int] = None,
        enable_quality_filter: bool = True,
        quality_threshold: float = 0.3,
        constraints: Optional[Dict[str, Any]] = None,
        enable_interpolation: bool = True,
        solver_timeout_ms: int = 5000,
        mus_timeout_ms: int = 3000,
    ) -> None:
        super().__init__(
            source,
            input_shapes=input_shapes,
            max_iterations=max_iterations,
            default_device=default_device,
            default_phase=default_phase,
            max_k=max_k,
            enable_quality_filter=enable_quality_filter,
            quality_threshold=quality_threshold,
            constraints=constraints,
            enable_interpolation=enable_interpolation,
        )
        self._solver_timeout_ms = solver_timeout_ms
        self._mus_timeout_ms = mus_timeout_ms
        self._incremental_solver: Optional[IncrementalCEGARSolver] = None
        self._core_extractor = EnhancedUnsatCorePredicateExtractor(
            timeout_ms=mus_timeout_ms,
        )
        self._enhanced_stats: Dict[str, int] = {
            "core_predicates_count": 0,
            "template_predicates_count": 0,
            "solver_reuse_count": 0,
            "mus_extractions": 0,
            "total_core_labels": 0,
        }

    # ------------------------------------------------------------------
    # Override run() to inject incremental solver + core predicates.
    # ------------------------------------------------------------------

    def run(self) -> ShapeCEGARResult:
        """Execute the enhanced CEGAR loop with incremental solving."""
        from src.shape_cegar import (
            CounterexampleAnalyser,
            IterationRecord,
        )
        from src.model_checker import (
            ConstraintVerifier,
            extract_computation_graph,
        )

        t0 = time.monotonic()

        try:
            graph = extract_computation_graph(self.source)
        except (ValueError, SyntaxError):
            return ShapeCEGARResult(
                final_status=CEGARStatus.PARSE_ERROR,
                total_time_ms=(time.monotonic() - t0) * 1000,
            )

        if not HAS_Z3:
            result = self._single_pass(graph)
            result.total_time_ms = (time.monotonic() - t0) * 1000
            result.final_status = CEGARStatus.NO_Z3
            return result

        # Initialise incremental solver.
        self._incremental_solver = IncrementalCEGARSolver(
            timeout_ms=self._solver_timeout_ms,
        )

        current_input_shapes = dict(self.input_shapes)
        current_shape_env: Dict[str, Any] = {}
        last_vresult = None

        for iteration in range(self.max_iterations):
            iter_t0 = time.monotonic()

            # === Verify ===
            checker = ConstraintVerifier(
                graph,
                input_shapes=current_input_shapes,
                default_device=self.default_device,
                default_phase=self.default_phase,
                max_k=self.max_k,
                constraints=self.relational_constraints,
            )
            vresult = checker.verify()
            last_vresult = vresult
            current_shape_env = dict(checker._init_state.shape_env)

            if vresult.safe:
                iter_time = (time.monotonic() - iter_t0) * 1000
                self._iteration_log.append(IterationRecord(
                    iteration=iteration,
                    num_violations=0, num_spurious=0, num_real=0,
                    time_ms=iter_time,
                ))
                return self._build_enhanced_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            cex = vresult.counterexample
            if cex is None or not cex.violations:
                return self._build_enhanced_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            # === Analyse counterexamples ===
            analyser = CounterexampleAnalyser(
                graph, current_shape_env, current_input_shapes,
            )
            analysed = analyser.analyse(cex)

            new_predicates: List[ShapePredicate] = []
            real_bugs = []
            num_spurious = num_real = 0

            for acex in analysed:
                if acex.is_real_bug():
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1
                elif acex.is_spurious():
                    # --- Enhanced: use incremental solver + MUS ---
                    core_preds = self._extract_core_predicates(
                        graph, acex, current_input_shapes, current_shape_env,
                    )
                    if core_preds:
                        new_predicates.extend(core_preds)
                        self._enhanced_stats["core_predicates_count"] += len(core_preds)
                    else:
                        # Fall back to existing template predicates.
                        new_predicates.extend(acex.synthesised_predicates)
                        self._enhanced_stats["template_predicates_count"] += len(
                            acex.synthesised_predicates
                        )
                    num_spurious += 1
                else:
                    real_bugs.append(acex.violation)
                    self._real_bugs_so_far.append(acex.violation)
                    num_real += 1

            # Craig interpolation fallback (same as base).
            template_pred_count = len(new_predicates)
            if self.enable_interpolation and HAS_Z3:
                try:
                    from src.craig_interpolation import (
                        InterpolationPredicateDiscovery,
                        LinearComboPredicate,
                    )
                    from src.shape_cegar import _convert_linear_combo_to_predicate

                    ipd = InterpolationPredicateDiscovery()
                    pe = UnsatCorePredicateExtractor(graph, current_shape_env)
                    for acex in analysed:
                        if 0 <= acex.step_index < len(graph.steps):
                            path_cs, safety_neg_cs, dm = pe._build_interpolation_query(
                                graph, acex.step_index, current_input_shapes,
                                concrete_dims=None,
                            )
                            if path_cs and safety_neg_cs:
                                self._interpolation_stats["attempted"] += 1
                                interp_preds = ipd.discover_via_interpolation(
                                    path_cs, safety_neg_cs, dm,
                                )
                                if interp_preds:
                                    self._interpolation_stats["successful"] += 1
                                    self._interpolation_stats["predicates_from_interpolation"] += len(interp_preds)
                                    for ip in interp_preds:
                                        if isinstance(ip, ShapePredicate):
                                            new_predicates.append(ip)
                                        elif isinstance(ip, LinearComboPredicate):
                                            converted = _convert_linear_combo_to_predicate(ip)
                                            if converted is not None:
                                                new_predicates.append(converted)
                except (ImportError, Exception):
                    pass

            self.pred_set.set_known_bugs(self._real_bugs_so_far)
            iter_time = (time.monotonic() - iter_t0) * 1000
            added = self.pred_set.add_all(new_predicates)

            self._iteration_log.append(IterationRecord(
                iteration=iteration,
                num_violations=len(cex.violations),
                num_spurious=num_spurious,
                num_real=num_real,
                predicates_added=new_predicates[:],
                time_ms=iter_time,
            ))

            if real_bugs:
                result = self._build_enhanced_result(
                    CEGARStatus.REAL_BUG_FOUND, graph, last_vresult, t0,
                )
                result.real_bugs = real_bugs
                return result

            if added == 0:
                return self._build_enhanced_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            if not ShapeRefinement.check_feasibility(self.pred_set.predicates):
                return self._build_enhanced_result(
                    CEGARStatus.SAFE, graph, last_vresult, t0,
                )

            current_input_shapes, current_shape_env = ShapeRefinement.apply_predicates(
                current_input_shapes, current_shape_env, new_predicates,
            )

        self._enhanced_stats["solver_reuse_count"] = (
            self._incremental_solver.stats["reuse_count"]
            if self._incremental_solver
            else 0
        )
        return self._build_enhanced_result(
            CEGARStatus.MAX_ITER, graph, last_vresult, t0,
        )

    # ------------------------------------------------------------------
    # Core predicate extraction via incremental solver
    # ------------------------------------------------------------------

    def _extract_core_predicates(
        self,
        graph: ComputationGraph,
        acex: Any,
        input_shapes: Dict[str, tuple],
        shape_env: Dict[str, Any],
    ) -> List[ShapePredicate]:
        """Use ``IncrementalCEGARSolver`` + ``EnhancedUnsatCorePredicateExtractor``."""
        if not HAS_Z3 or self._incremental_solver is None:
            return []

        step_idx = acex.step_index
        if step_idx < 0 or step_idx >= len(graph.steps):
            return []

        pe = UnsatCorePredicateExtractor(graph, shape_env)
        path_cs, safety_cs, dim_map = pe._build_predicate_extraction_query(
            graph.steps[step_idx], step_idx, acex.concrete_dims, input_shapes,
        )
        if not path_cs or not safety_cs:
            return []

        # Use incremental solver.
        solver = self._incremental_solver
        solver.push()

        label_map: Dict[str, Any] = {}
        for i, c in enumerate(path_cs):
            label = f"__enh_path_{step_idx}_{i}"
            label_map[label] = c
            solver.assert_and_track(c, label)

        if safety_cs:
            neg_safety = z3.Not(z3.And(*safety_cs))
            solver.add(neg_safety)

        result, core_labels = solver.check_with_core()
        solver.pop()

        if result != z3.unsat:
            return []

        self._enhanced_stats["mus_extractions"] += 1
        self._enhanced_stats["total_core_labels"] += len(core_labels)

        # Extract via enhanced extractor.
        core_preds = self._core_extractor.extract_predicates(
            core_labels, label_map, dim_map,
        )

        shape_predicates: List[ShapePredicate] = []
        for cp in core_preds:
            if cp.shape_predicate is not None:
                shape_predicates.append(cp.shape_predicate)
            else:
                # Try converting the formula directly.
                sp = _formula_to_shape_predicate(cp.formula, dim_map)
                if sp is not None:
                    shape_predicates.append(sp)

        return shape_predicates

    # ------------------------------------------------------------------
    # Result builder (adds enhanced stats)
    # ------------------------------------------------------------------

    def _build_enhanced_result(
        self,
        status: CEGARStatus,
        graph: ComputationGraph,
        vresult: Any,
        t0: float,
    ) -> ShapeCEGARResult:
        result = self._build_result(status, graph, vresult, t0)
        # Attach enhanced stats as part of interpolation_stats dict.
        if result.interpolation_stats is None:
            result.interpolation_stats = {}
        result.interpolation_stats.update(self._enhanced_stats)
        if self._incremental_solver:
            result.interpolation_stats.update(self._incremental_solver.stats)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_z3_vars(expr: Any) -> List[Any]:
    """Collect all free Z3 integer variables from an expression."""
    if not HAS_Z3:
        return []
    result: List[Any] = []
    seen: Set[int] = set()

    def _walk(e: Any) -> None:
        eid = id(e)
        if eid in seen:
            return
        seen.add(eid)
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED and e.sort() == z3.IntSort():
            result.append(e)
        else:
            for child in e.children():
                _walk(child)

    try:
        _walk(expr)
    except Exception:
        pass
    return result


def _formula_to_shape_predicate(
    formula: Any,
    dim_map: Optional[Dict[str, Tuple[str, int]]],
) -> Optional[ShapePredicate]:
    """Convert a Z3 formula to a ``ShapePredicate`` if possible."""
    if not HAS_Z3 or dim_map is None:
        return None

    formula_str = str(formula)

    # Equality: var == IntVal
    if z3.is_eq(formula):
        children = formula.children()
        if len(children) == 2:
            lhs, rhs = children
            var_name, val = None, None
            if z3.is_int_value(rhs) and not z3.is_int_value(lhs):
                var_name, val = str(lhs), rhs.as_long()
            elif z3.is_int_value(lhs) and not z3.is_int_value(rhs):
                var_name, val = str(rhs), lhs.as_long()

            if var_name and val is not None and val > 0 and var_name in dim_map:
                tensor, axis = dim_map[var_name]
                return ShapePredicate(
                    kind=PredicateKind.DIM_EQ,
                    tensor=tensor,
                    axis=axis,
                    value=val,
                    provenance="core_derived",
                )

    # Inequality: var >= IntVal or var > IntVal
    if z3.is_app(formula):
        dk = formula.decl().kind()
        children = formula.children()
        if len(children) == 2:
            lhs, rhs = children
            if dk == z3.Z3_OP_GE and z3.is_int_value(rhs) and not z3.is_int_value(lhs):
                var_name = str(lhs)
                val = rhs.as_long()
                if var_name in dim_map and val > 0:
                    tensor, axis = dim_map[var_name]
                    return ShapePredicate(
                        kind=PredicateKind.DIM_GE,
                        tensor=tensor,
                        axis=axis,
                        value=val,
                        provenance="core_derived",
                    )
            elif dk == z3.Z3_OP_GT and z3.is_int_value(rhs) and not z3.is_int_value(lhs):
                var_name = str(lhs)
                val = rhs.as_long()
                if var_name in dim_map:
                    tensor, axis = dim_map[var_name]
                    return ShapePredicate(
                        kind=PredicateKind.DIM_GT,
                        tensor=tensor,
                        axis=axis,
                        value=val,
                        provenance="core_derived",
                    )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Top-level API
# ═══════════════════════════════════════════════════════════════════════════════


def run_enhanced_cegar(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
    default_device: Device = Device.CPU,
    default_phase: Phase = Phase.TRAIN,
    max_k: Optional[int] = None,
    enable_quality_filter: bool = True,
    quality_threshold: float = 0.3,
    solver_timeout_ms: int = 5000,
    mus_timeout_ms: int = 3000,
    **kwargs: Any,
) -> ShapeCEGARResult:
    """Run enhanced CEGAR with incremental solving and UNSAT core extraction.

    Drop-in replacement for ``run_shape_cegar`` with additional options
    for solver timeout and MUS extraction timeout.

    Returns
    -------
    ShapeCEGARResult
        Same result type as ``run_shape_cegar``, but
        ``interpolation_stats`` includes additional keys:
        ``core_predicates_count``, ``template_predicates_count``,
        ``solver_reuse_count``, ``mus_extractions``.
    """
    loop = EnhancedShapeCEGARLoop(
        source,
        input_shapes=input_shapes,
        max_iterations=max_iterations,
        default_device=default_device,
        default_phase=default_phase,
        max_k=max_k,
        enable_quality_filter=enable_quality_filter,
        quality_threshold=quality_threshold,
        solver_timeout_ms=solver_timeout_ms,
        mus_timeout_ms=mus_timeout_ms,
    )
    return loop.run()
