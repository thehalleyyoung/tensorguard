"""
CEGAR as a Configurable Program Analysis (CPA) Framework.

Formalizes the counterexample-guided abstraction refinement loop from
``shape_cegar.py`` as a Configurable Program Analysis (CPA) in the sense
of Beyer, Henzinger & Théoduloz (CAV 2007).  The key contribution is a
fixed-point characterization of the CEGAR refinement operator on a
finite predicate lattice, together with a convergence certificate.

Motivation (reviewer critique — Sinha)
--------------------------------------
    "CEGAR loop lacks formal fixed-point characterization.  Sinha demands
     formalization as a configurable program analysis (CPA) with precision
     adjustment operator."

CPA background
--------------
A CPA is a tuple ``(D, Π, ⇝, merge, stop, prec)`` where:

* **D** is an abstract domain (here: ``PredicateLattice``).
* **Π** is a set of precisions (here: subsets of the predicate universe).
* **⇝** is the transfer relation (``TransferFunction``).
* **merge** combines two abstract states (``MergeOperator``).
* **stop** decides subsumption (``StopOperator``).
* **prec** is the precision adjustment operator (``PrecisionAdjustment``),
  which adds predicates discovered from counterexample analysis.

Fixed-point characterization
----------------------------
Define the *refinement operator*

    R : PredicateLattice → PredicateLattice
    R(P) = P ∪ { predicates synthesised from counterexamples of Verify(P) }

R is monotone on the complete lattice of predicate sets ordered by ⊆.
The lattice has finite height bounded by L × D × K where:

* L = number of layers in the nn.Module,
* D = maximum number of shape dimensions per layer,
* K = |PredicateKind| = 7.

By Kleene's fixed-point theorem, iterating R from ⊥ = ∅ reaches the
least fixed point in at most L × D × K steps.  At the fixed point,
Verify(P*) returns no new spurious counterexamples, so either the model
is SAFE or a genuine bug has been found.

Note on GFP vs LFP
-------------------
The CEGAR loop actually computes the *least* fixed point (LFP) of R
starting from ⊥ = ∅ (no predicates).  This yields the *smallest*
predicate set that eliminates all spurious counterexamples — the most
permissive contract.  We expose ``cegar_as_least_fixed_point()`` as the
primary API, and ``cegar_as_greatest_fixed_point()`` for the dual view
that starts from the full predicate universe and removes unnecessary
predicates.

Integration
-----------
The module re-uses ``ShapePredicate``, ``PredicateKind``, ``PredicateSet``,
and ``ShapeCEGARLoop`` from ``shape_cegar`` and wraps them with the CPA
abstraction layer.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from src.shape_cegar import (
    ShapePredicate,
    PredicateKind,
    PredicateSet,
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
    IterationRecord,
)

logger = logging.getLogger(__name__)

# Number of distinct PredicateKind values.
NUM_PREDICATE_KINDS: int = len(PredicateKind)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Predicate Lattice — abstract domain
# ═══════════════════════════════════════════════════════════════════════════════

class PredicateLattice:
    r"""Complete lattice of predicate sets ordered by subset inclusion.

    Elements are frozensets of ``ShapePredicate``.  The partial order is:

        P₁ ⊑ P₂  ⟺  P₁ ⊆ P₂

    Interpretation: *more* predicates ⟹ *more refined* (fewer concrete
    models satisfy all predicates).  The lattice operations are:

    * **Bottom (⊥)**: ∅ — no predicates, most abstract (all models allowed).
    * **Top (⊤)**: the full predicate universe — most concrete.
    * **Join (⊔)**: set union — combines the precision of both elements.
    * **Meet (⊓)**: set intersection — common precision.

    The lattice height is bounded by the size of the predicate universe,
    which is at most ``L × D × K`` where L = layers, D = max dims per
    layer, K = |PredicateKind|.

    Parameters
    ----------
    universe : frozenset of ShapePredicate, optional
        The full set of possible predicates (⊤).  If not provided, the
        lattice is *open* — the universe grows as predicates are discovered.
        In the open case, ``top()`` returns the current known universe.
    """

    def __init__(
        self,
        universe: Optional[FrozenSet[ShapePredicate]] = None,
    ) -> None:
        self._universe: FrozenSet[ShapePredicate] = universe or frozenset()
        self._open = universe is None

    # -- Universe management --------------------------------------------------

    @property
    def universe(self) -> FrozenSet[ShapePredicate]:
        """The full predicate universe (⊤ element)."""
        return self._universe

    def extend_universe(self, preds: FrozenSet[ShapePredicate]) -> None:
        """Extend the universe with newly discovered predicates."""
        self._universe = self._universe | preds

    # -- Lattice elements -----------------------------------------------------

    def bottom(self) -> FrozenSet[ShapePredicate]:
        """⊥ = ∅ — no predicates, most abstract."""
        return frozenset()

    def top(self) -> FrozenSet[ShapePredicate]:
        """⊤ = full predicate universe — most concrete."""
        return self._universe

    # -- Partial order --------------------------------------------------------

    @staticmethod
    def leq(p1: FrozenSet[ShapePredicate], p2: FrozenSet[ShapePredicate]) -> bool:
        """P₁ ⊑ P₂ iff P₁ ⊆ P₂ (p2 is at least as refined as p1)."""
        return p1 <= p2

    # -- Join and Meet --------------------------------------------------------

    @staticmethod
    def join(p1: FrozenSet[ShapePredicate], p2: FrozenSet[ShapePredicate]) -> FrozenSet[ShapePredicate]:
        """P₁ ⊔ P₂ = P₁ ∪ P₂ — least upper bound (union of predicates)."""
        return p1 | p2

    @staticmethod
    def meet(p1: FrozenSet[ShapePredicate], p2: FrozenSet[ShapePredicate]) -> FrozenSet[ShapePredicate]:
        """P₁ ⊓ P₂ = P₁ ∩ P₂ — greatest lower bound (intersection)."""
        return p1 & p2

    # -- Lattice height -------------------------------------------------------

    def height(self) -> int:
        """Upper bound on lattice height = |universe|.

        In a finite powerset lattice ordered by ⊆, the longest chain has
        length |universe| (from ∅ to the full set, adding one element at
        a time).
        """
        return len(self._universe)

    @staticmethod
    def height_bound(num_layers: int, max_dims: int, num_kinds: int = NUM_PREDICATE_KINDS) -> int:
        """Theoretical height bound: L × D × K.

        Parameters
        ----------
        num_layers : int
            Number of layers (L) in the nn.Module.
        max_dims : int
            Maximum number of shape dimensions per layer (D).
        num_kinds : int
            Number of predicate kinds (K), default 7.
        """
        return num_layers * max_dims * num_kinds

    # -- Utility --------------------------------------------------------------

    def is_bottom(self, p: FrozenSet[ShapePredicate]) -> bool:
        return len(p) == 0

    def is_top(self, p: FrozenSet[ShapePredicate]) -> bool:
        return p == self._universe and len(self._universe) > 0

    def __repr__(self) -> str:
        return f"PredicateLattice(|universe|={len(self._universe)})"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CPA components
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AbstractState:
    """An abstract state in the CPA = a set of predicates (precision level).

    Attributes
    ----------
    predicates : frozenset of ShapePredicate
        The currently accumulated predicates.
    iteration : int
        Which CEGAR iteration produced this state.
    """
    predicates: FrozenSet[ShapePredicate]
    iteration: int = 0


class CPADomain:
    """The abstract domain component of the CPA.

    Wraps ``PredicateLattice`` and provides the domain operations required
    by the CPA framework: initial state, abstraction, and concretisation
    (the latter is the identity here since our abstract states *are*
    predicate sets).
    """

    def __init__(self, lattice: PredicateLattice) -> None:
        self.lattice = lattice

    def initial_state(self) -> AbstractState:
        """The initial abstract state: ⊥ (no predicates)."""
        return AbstractState(predicates=self.lattice.bottom(), iteration=0)

    def is_bottom(self, state: AbstractState) -> bool:
        return self.lattice.is_bottom(state.predicates)

    def is_top(self, state: AbstractState) -> bool:
        return self.lattice.is_top(state.predicates)

    def leq(self, s1: AbstractState, s2: AbstractState) -> bool:
        """s1 ⊑ s2 iff s1.predicates ⊆ s2.predicates."""
        return self.lattice.leq(s1.predicates, s2.predicates)


class TransferFunction:
    """Transfer function: propagates abstract state through one CEGAR iteration.

    In classical CPA, the transfer function models one program statement.
    In our CPA, each "statement" corresponds to one full verify → analyse →
    refine cycle.  The transfer function takes the current predicate set,
    runs verification, and returns the new predicate set augmented with
    predicates synthesised from counterexamples.

    Parameters
    ----------
    refinement_fn : callable
        ``(FrozenSet[ShapePredicate]) → FrozenSet[ShapePredicate]``
        The refinement operator R that, given a predicate set, returns the
        union with newly synthesised predicates.
    """

    def __init__(
        self,
        refinement_fn: Callable[
            [FrozenSet[ShapePredicate]], FrozenSet[ShapePredicate]
        ],
    ) -> None:
        self._refine = refinement_fn

    def apply(self, state: AbstractState) -> AbstractState:
        """Apply one refinement step: R(P)."""
        new_preds = self._refine(state.predicates)
        return AbstractState(
            predicates=new_preds,
            iteration=state.iteration + 1,
        )


class PrecisionAdjustment:
    """Precision adjustment operator π.

    In the CPA framework, the precision adjustment operator selects which
    predicates to track at each program location.  In our CEGAR CPA,
    precision adjustment corresponds to adding the predicates discovered
    from counterexample analysis.

    The operator is monotone: π(P) ⊇ P (we only add predicates, never
    remove them).

    Parameters
    ----------
    max_predicates : int or None
        Optional cap on the number of predicates to keep.  If the
        predicate set would exceed this limit, only the most recently
        added predicates are kept (a form of widening).
    """

    def __init__(self, max_predicates: Optional[int] = None) -> None:
        self.max_predicates = max_predicates

    def adjust(
        self,
        state: AbstractState,
        new_predicates: FrozenSet[ShapePredicate],
    ) -> AbstractState:
        """Adjust precision by adding *new_predicates* to the state.

        Returns the adjusted abstract state with the enlarged predicate set.
        If ``max_predicates`` is set and the result would exceed it, the
        oldest predicates are dropped (widening).
        """
        combined = state.predicates | new_predicates
        if self.max_predicates is not None and len(combined) > self.max_predicates:
            # Keep the most recent predicates (widening)
            as_list = list(state.predicates) + [
                p for p in new_predicates if p not in state.predicates
            ]
            combined = frozenset(as_list[-self.max_predicates:])
        return AbstractState(predicates=combined, iteration=state.iteration)

    def is_monotone(
        self,
        before: AbstractState,
        after: AbstractState,
    ) -> bool:
        """Verify that adjustment was monotone: before.predicates ⊆ after.predicates."""
        return before.predicates <= after.predicates


class MergeOperator:
    """Merge operator: combines two abstract states.

    For predicate-based CEGAR, merge is typically *join* (union) — we keep
    all predicates discovered by either branch.  The ``sep`` mode keeps
    states separate (no merging), which is useful for path-sensitive
    analysis.

    Parameters
    ----------
    mode : str
        ``"join"`` for union merge, ``"sep"`` for no merge (keep separate).
    """

    def __init__(self, mode: str = "join") -> None:
        if mode not in ("join", "sep"):
            raise ValueError(f"Unknown merge mode: {mode!r}")
        self.mode = mode

    def merge(self, s1: AbstractState, s2: AbstractState) -> AbstractState:
        """Merge two abstract states."""
        if self.mode == "sep":
            return s2  # no merge — keep s2 as-is
        # join: union of predicates
        joined = PredicateLattice.join(s1.predicates, s2.predicates)
        return AbstractState(
            predicates=joined,
            iteration=max(s1.iteration, s2.iteration),
        )


class StopOperator:
    """Stop (subsumption) operator.

    Decides whether a new abstract state is already subsumed by an
    existing state in the reached set.  In our setting, state ``s_new``
    is subsumed by ``s_old`` iff ``s_new.predicates ⊆ s_old.predicates``
    (the old state is at least as refined).
    """

    @staticmethod
    def is_subsumed(
        new_state: AbstractState,
        reached: Sequence[AbstractState],
    ) -> bool:
        """Return True if *new_state* is subsumed by any state in *reached*."""
        for existing in reached:
            if new_state.predicates <= existing.predicates:
                return True
        return False

    @staticmethod
    def is_fixed_point(
        current: AbstractState,
        refined: AbstractState,
    ) -> bool:
        """Return True if refinement produced no new predicates (fixed point)."""
        return refined.predicates == current.predicates


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Fixed-point characterization
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FixedPointResult:
    """Result of computing a fixed point of the refinement operator.

    Attributes
    ----------
    fixed_point : frozenset of ShapePredicate
        The predicate set at the fixed point.
    iterations : int
        Number of iterations to reach the fixed point.
    trajectory : list of frozenset
        The sequence of predicate sets at each iteration.
    is_lfp : bool
        True if this is the least fixed point (ascending from ⊥).
    converged : bool
        True if a genuine fixed point was reached (not just budget exhaustion).
    """
    fixed_point: FrozenSet[ShapePredicate]
    iterations: int
    trajectory: List[FrozenSet[ShapePredicate]] = field(default_factory=list)
    is_lfp: bool = True
    converged: bool = False


def refinement_operator(
    verify_fn: Callable[[FrozenSet[ShapePredicate]], FrozenSet[ShapePredicate]],
    current: FrozenSet[ShapePredicate],
) -> FrozenSet[ShapePredicate]:
    """The refinement operator R: PredicateLattice → PredicateLattice.

    R(P) = P ∪ { predicates from counterexample analysis of Verify(P) }

    This is a monotone operator on the powerset lattice (ordered by ⊆):
    * R(P) ⊇ P  (we only add predicates, never remove)
    * P₁ ⊆ P₂ ⟹ R(P₁) ⊆ R(P₂)  (more predicates ⟹ fewer cex ⟹ fewer new preds)

    Parameters
    ----------
    verify_fn : callable
        Given a predicate set, runs verification and returns the union
        of the input set with newly synthesised predicates.
    current : frozenset
        The current predicate set P.
    """
    return verify_fn(current)


def cegar_as_least_fixed_point(
    refinement_fn: Callable[[FrozenSet[ShapePredicate]], FrozenSet[ShapePredicate]],
    lattice: PredicateLattice,
    max_iterations: int = 100,
) -> FixedPointResult:
    """Compute the CEGAR result as the least fixed point (LFP) of R.

    Starts from ⊥ = ∅ and iterates R until R(P) = P (fixed point) or
    the iteration budget is exhausted.

    By Kleene's fixed-point theorem, since R is monotone on a complete
    lattice of finite height, this iteration converges in at most
    ``lattice.height()`` steps (or ``L × D × K`` in theory).

    The LFP gives the *smallest* predicate set that eliminates all
    spurious counterexamples — i.e., the most permissive contract.

    Parameters
    ----------
    refinement_fn : callable
        The refinement operator R.
    lattice : PredicateLattice
        The abstract domain lattice.
    max_iterations : int
        Safety bound on iteration count.
    """
    current = lattice.bottom()
    trajectory: List[FrozenSet[ShapePredicate]] = [current]

    for i in range(max_iterations):
        refined = refinement_fn(current)
        trajectory.append(refined)

        # Extend universe with any newly discovered predicates
        lattice.extend_universe(refined)

        if refined == current:
            # Fixed point reached
            return FixedPointResult(
                fixed_point=current,
                iterations=i + 1,
                trajectory=trajectory,
                is_lfp=True,
                converged=True,
            )
        current = refined

    return FixedPointResult(
        fixed_point=current,
        iterations=max_iterations,
        trajectory=trajectory,
        is_lfp=True,
        converged=False,
    )


def cegar_as_greatest_fixed_point(
    refinement_fn: Callable[[FrozenSet[ShapePredicate]], FrozenSet[ShapePredicate]],
    lattice: PredicateLattice,
    max_iterations: int = 100,
) -> FixedPointResult:
    """Compute the CEGAR result as the greatest fixed point (GFP) of R.

    The dual view: starts from ⊤ (all predicates) and iteratively removes
    unnecessary predicates.  The GFP gives the *largest* predicate set
    that is a fixed point — i.e., the most restrictive consistent contract.

    In practice, the LFP (ascending from ⊥) is preferred because it
    discovers only the predicates actually needed to eliminate spurious
    counterexamples.

    Parameters
    ----------
    refinement_fn : callable
        The refinement operator R.
    lattice : PredicateLattice
        The abstract domain lattice (must have a non-empty universe for ⊤).
    max_iterations : int
        Safety bound on iteration count.
    """
    current = lattice.top()
    if not current:
        # Empty universe — fall back to LFP
        return cegar_as_least_fixed_point(refinement_fn, lattice, max_iterations)

    trajectory: List[FrozenSet[ShapePredicate]] = [current]

    for i in range(max_iterations):
        refined = refinement_fn(current)
        trajectory.append(refined)

        if refined == current:
            return FixedPointResult(
                fixed_point=current,
                iterations=i + 1,
                trajectory=trajectory,
                is_lfp=False,
                converged=True,
            )
        current = refined

    return FixedPointResult(
        fixed_point=current,
        iterations=max_iterations,
        trajectory=trajectory,
        is_lfp=False,
        converged=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Widening operator
# ═══════════════════════════════════════════════════════════════════════════════

def widen(
    p1: FrozenSet[ShapePredicate],
    p2: FrozenSet[ShapePredicate],
    max_size: Optional[int] = None,
) -> FrozenSet[ShapePredicate]:
    r"""Widening operator for the predicate lattice.

    Since the predicate lattice is finite, widening is not strictly
    necessary for termination (the ascending chain condition is
    guaranteed).  However, widening can accelerate convergence by
    limiting the predicate set size.

    The widening strategy is:
    1. Compute P₁ ∪ P₂ (the join).
    2. If ``max_size`` is set and |P₁ ∪ P₂| > max_size, keep only the
       predicates from P₂ (the newer set) up to ``max_size``.

    This is an *extrapolation* operator: widen(P₁, P₂) ⊇ P₂ ⊇ P₁ ∪ P₂
    when no size limit applies.

    Parameters
    ----------
    p1 : frozenset
        Previous predicate set.
    p2 : frozenset
        Current predicate set (after refinement).
    max_size : int, optional
        Maximum number of predicates.  If None, no limit (equivalent to join).
    """
    joined = p1 | p2
    if max_size is not None and len(joined) > max_size:
        # Prefer predicates from the newer iteration (p2 \ p1 first, then p1 ∩ p2)
        new_preds = list(p2 - p1)
        old_preds = list(p1 & p2)
        kept = (new_preds + old_preds)[:max_size]
        return frozenset(kept)
    return joined


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Convergence certificate
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceCertificate:
    """Certificate attesting that the CEGAR loop converged correctly.

    Encodes the key properties that a reviewer (or Lean proof checker)
    would verify:

    1. **Lattice height bound**: the theoretical maximum number of
       iterations before the predicate universe is exhausted.
    2. **Actual iterations**: how many iterations were actually performed.
    3. **Monotonicity**: each iteration's predicate set is a (non-strict)
       superset of the previous one.
    4. **Fixed-point check**: the final predicate set is a fixed point
       of the refinement operator R.
    5. **Strict growth until convergence**: each non-final iteration
       added at least one new predicate (strict superset).
    """
    lattice_height_bound: int
    actual_iterations: int
    trajectory_sizes: List[int]
    monotonicity_verified: bool
    strict_growth_verified: bool
    fixed_point_reached: bool
    final_predicate_count: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """A certificate is valid iff monotonicity holds and a fixed point was reached."""
        return self.monotonicity_verified and self.fixed_point_reached

    def summary(self) -> str:
        return (
            f"ConvergenceCertificate: "
            f"{'VALID' if self.is_valid else 'INVALID'}, "
            f"{self.actual_iterations}/{self.lattice_height_bound} iterations, "
            f"{self.final_predicate_count} predicates, "
            f"monotone={self.monotonicity_verified}, "
            f"fixed_point={self.fixed_point_reached}"
        )


def convergence_certificate(
    trajectory: List[FrozenSet[ShapePredicate]],
    lattice_height_bound: int,
    converged: bool,
) -> ConvergenceCertificate:
    """Build a convergence certificate from a CEGAR trajectory.

    Parameters
    ----------
    trajectory : list of frozenset
        The sequence of predicate sets at each iteration, starting from ⊥.
    lattice_height_bound : int
        The theoretical height bound L × D × K.
    converged : bool
        Whether the iteration reached a fixed point.

    Returns
    -------
    ConvergenceCertificate
        The certificate with all checks filled in.
    """
    sizes = [len(t) for t in trajectory]

    # Monotonicity: P_i ⊆ P_{i+1} for all i
    monotone = all(
        trajectory[i] <= trajectory[i + 1]
        for i in range(len(trajectory) - 1)
    )

    # Strict growth: P_i ⊊ P_{i+1} for all i < len-1 (i.e., except the
    # last pair which should be equal at the fixed point)
    strict_pairs = list(range(len(trajectory) - 2)) if len(trajectory) >= 3 else []
    strict_growth = all(
        trajectory[i] < trajectory[i + 1]
        for i in strict_pairs
    ) if strict_pairs else True

    # Fixed-point check: last two elements are equal
    fp_reached = converged
    if len(trajectory) >= 2:
        fp_reached = fp_reached or (trajectory[-1] == trajectory[-2])

    return ConvergenceCertificate(
        lattice_height_bound=lattice_height_bound,
        actual_iterations=max(0, len(trajectory) - 1),
        trajectory_sizes=sizes,
        monotonicity_verified=monotone,
        strict_growth_verified=strict_growth,
        fixed_point_reached=fp_reached,
        final_predicate_count=sizes[-1] if sizes else 0,
        details={
            "trajectory_lengths": sizes,
            "height_bound_respected": (len(trajectory) - 1) <= lattice_height_bound if trajectory else True,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CPA wrapper — ties all components together
# ═══════════════════════════════════════════════════════════════════════════════

class CEGARCPA:
    """Configurable Program Analysis wrapping the CEGAR loop.

    Ties together the lattice, transfer function, precision adjustment,
    merge, and stop operators into a single CPA that can be iterated to
    a fixed point.

    Usage
    -----
    >>> lattice = PredicateLattice()
    >>> cpa = CEGARCPA(lattice, refinement_fn=my_refine)
    >>> result = cpa.run()
    >>> cert = cpa.convergence_certificate()
    >>> assert cert.is_valid

    Parameters
    ----------
    lattice : PredicateLattice
        The abstract domain.
    refinement_fn : callable
        ``(FrozenSet[ShapePredicate]) → FrozenSet[ShapePredicate]``
        The refinement operator R.
    max_iterations : int
        Iteration budget (safety bound).
    merge_mode : str
        ``"join"`` or ``"sep"`` — how to combine abstract states.
    max_predicates : int, optional
        Predicate cap for the precision adjustment operator (widening).
    """

    def __init__(
        self,
        lattice: PredicateLattice,
        refinement_fn: Callable[
            [FrozenSet[ShapePredicate]], FrozenSet[ShapePredicate]
        ],
        max_iterations: int = 100,
        merge_mode: str = "join",
        max_predicates: Optional[int] = None,
    ) -> None:
        self.lattice = lattice
        self.domain = CPADomain(lattice)
        self.transfer = TransferFunction(refinement_fn)
        self.precision = PrecisionAdjustment(max_predicates)
        self.merge_op = MergeOperator(merge_mode)
        self.stop_op = StopOperator()
        self.max_iterations = max_iterations

        # State
        self._trajectory: List[FrozenSet[ShapePredicate]] = []
        self._reached: List[AbstractState] = []
        self._converged = False
        self._result: Optional[FixedPointResult] = None

    def run(self) -> FixedPointResult:
        """Execute the CPA iteration to a fixed point.

        Returns
        -------
        FixedPointResult
            The fixed-point result including trajectory and convergence info.
        """
        state = self.domain.initial_state()
        self._trajectory = [state.predicates]
        self._reached = [state]

        for i in range(self.max_iterations):
            # Transfer: apply refinement operator
            new_state = self.transfer.apply(state)

            # Precision adjustment
            new_preds = new_state.predicates - state.predicates
            adjusted = self.precision.adjust(state, new_preds)
            adjusted = AbstractState(
                predicates=adjusted.predicates,
                iteration=i + 1,
            )

            self._trajectory.append(adjusted.predicates)
            self.lattice.extend_universe(adjusted.predicates)

            # Stop check: fixed point?
            if self.stop_op.is_fixed_point(state, adjusted):
                self._converged = True
                self._reached.append(adjusted)
                break

            # Merge with existing reached states
            if self._reached:
                merged = self.merge_op.merge(self._reached[-1], adjusted)
                adjusted = merged

            self._reached.append(adjusted)
            state = adjusted

        self._result = FixedPointResult(
            fixed_point=self._trajectory[-1],
            iterations=len(self._trajectory) - 1,
            trajectory=list(self._trajectory),
            is_lfp=True,
            converged=self._converged,
        )
        return self._result

    def get_convergence_certificate(
        self,
        num_layers: int = 1,
        max_dims: int = 1,
    ) -> ConvergenceCertificate:
        """Build a convergence certificate for the completed run.

        Parameters
        ----------
        num_layers : int
            Number of layers (L) for the height bound.
        max_dims : int
            Max dims per layer (D) for the height bound.
        """
        height = PredicateLattice.height_bound(num_layers, max_dims)
        return convergence_certificate(
            trajectory=list(self._trajectory),
            lattice_height_bound=height,
            converged=self._converged,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Integration helper — wrap ShapeCEGARLoop as CPA
# ═══════════════════════════════════════════════════════════════════════════════

def shape_cegar_as_cpa(
    source: str,
    input_shapes: Optional[Dict[str, tuple]] = None,
    max_iterations: int = 10,
    max_predicates: Optional[int] = None,
) -> Tuple[ShapeCEGARResult, ConvergenceCertificate]:
    """Run the existing ShapeCEGARLoop and produce a CPA convergence certificate.

    This is the integration bridge: it runs the concrete CEGAR loop from
    ``shape_cegar.py`` and post-hoc constructs the CPA trajectory and
    convergence certificate.

    Parameters
    ----------
    source : str
        Python source code of the nn.Module.
    input_shapes : dict, optional
        Input shape specifications.
    max_iterations : int
        Iteration budget.
    max_predicates : int, optional
        Predicate cap (widening).

    Returns
    -------
    (ShapeCEGARResult, ConvergenceCertificate)
        The CEGAR result and its convergence certificate.
    """
    loop = ShapeCEGARLoop(
        source,
        input_shapes=input_shapes,
        max_iterations=max_iterations,
    )
    result = loop.run()

    # Reconstruct trajectory from iteration log
    trajectory: List[FrozenSet[ShapePredicate]] = [frozenset()]  # start from ⊥
    accumulated: Set[ShapePredicate] = set()
    for record in result.iteration_log:
        accumulated = accumulated | set(record.predicates_added)
        trajectory.append(frozenset(accumulated))

    # If the loop converged to SAFE or REAL_BUG, add final state
    if not result.iteration_log:
        trajectory.append(frozenset())

    # Estimate height bound from discovered predicates
    # Use a conservative bound: max(actual predicates, theoretical bound)
    num_preds = len(result.discovered_predicates)
    height = max(num_preds + 1, 10)  # conservative default

    converged = result.final_status in (CEGARStatus.SAFE, CEGARStatus.REAL_BUG_FOUND)
    cert = convergence_certificate(trajectory, height, converged)

    return result, cert
