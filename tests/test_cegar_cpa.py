"""
Tests for the CEGAR CPA (Configurable Program Analysis) formalization.

Covers:
  - PredicateLattice operations (join, meet, partial order, bottom, top)
  - CPA component interactions (domain, transfer, precision, merge, stop)
  - Fixed-point convergence (LFP and GFP)
  - Height bound verification
  - Widening operator
  - Convergence certificate validation
  - Integration with existing ShapeCEGARLoop
"""

from __future__ import annotations

import pytest

from src.shape_cegar import (
    ShapePredicate,
    PredicateKind,
    PredicateSet,
    ShapeCEGARLoop,
    ShapeCEGARResult,
    CEGARStatus,
)
from src.cegar_cpa import (
    PredicateLattice,
    AbstractState,
    CPADomain,
    TransferFunction,
    PrecisionAdjustment,
    MergeOperator,
    StopOperator,
    FixedPointResult,
    ConvergenceCertificate,
    CEGARCPA,
    NUM_PREDICATE_KINDS,
    refinement_operator,
    cegar_as_least_fixed_point,
    cegar_as_greatest_fixed_point,
    widen,
    convergence_certificate,
    shape_cegar_as_cpa,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: sample predicates
# ═══════════════════════════════════════════════════════════════════════════════

P_DIM_EQ_10 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=10)
P_DIM_EQ_20 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=-1, value=20)
P_DIM_GE_5 = ShapePredicate(PredicateKind.DIM_GE, "x", axis=0, value=5)
P_DIM_GT_0 = ShapePredicate(PredicateKind.DIM_GT, "x", axis=0, value=0)
P_NDIM_3 = ShapePredicate(PredicateKind.NDIM_EQ, "x", value=3)
P_DIV_8 = ShapePredicate(PredicateKind.DIM_DIVISIBLE, "y", axis=-1, divisor=8)
P_MATCH = ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=-1, match_tensor="w", match_axis=0)

ALL_SAMPLE = frozenset({P_DIM_EQ_10, P_DIM_EQ_20, P_DIM_GE_5, P_DIM_GT_0, P_NDIM_3, P_DIV_8, P_MATCH})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PredicateLattice — lattice operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredicateLattice:
    """Tests for the predicate lattice structure."""

    def test_bottom_is_empty(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert lat.bottom() == frozenset()

    def test_top_is_universe(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert lat.top() == ALL_SAMPLE

    def test_leq_subset(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        assert PredicateLattice.leq(p1, p2)
        assert not PredicateLattice.leq(p2, p1)

    def test_leq_reflexive(self):
        p = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        assert PredicateLattice.leq(p, p)

    def test_leq_bottom_leq_everything(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert PredicateLattice.leq(lat.bottom(), ALL_SAMPLE)
        assert PredicateLattice.leq(lat.bottom(), frozenset({P_DIM_EQ_10}))
        assert PredicateLattice.leq(lat.bottom(), lat.bottom())

    def test_join_is_union(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5})
        joined = PredicateLattice.join(p1, p2)
        assert joined == frozenset({P_DIM_EQ_10, P_DIM_GE_5})

    def test_join_idempotent(self):
        p = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        assert PredicateLattice.join(p, p) == p

    def test_join_commutative(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5})
        assert PredicateLattice.join(p1, p2) == PredicateLattice.join(p2, p1)

    def test_join_associative(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5})
        p3 = frozenset({P_NDIM_3})
        lhs = PredicateLattice.join(PredicateLattice.join(p1, p2), p3)
        rhs = PredicateLattice.join(p1, PredicateLattice.join(p2, p3))
        assert lhs == rhs

    def test_meet_is_intersection(self):
        p1 = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        p2 = frozenset({P_DIM_GE_5, P_NDIM_3})
        met = PredicateLattice.meet(p1, p2)
        assert met == frozenset({P_DIM_GE_5})

    def test_meet_with_bottom(self):
        p = frozenset({P_DIM_EQ_10})
        assert PredicateLattice.meet(p, frozenset()) == frozenset()

    def test_meet_idempotent(self):
        p = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        assert PredicateLattice.meet(p, p) == p

    def test_height_equals_universe_size(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert lat.height() == len(ALL_SAMPLE)

    def test_height_bound_formula(self):
        # L=3 layers, D=4 dims, K=7 kinds => 84
        assert PredicateLattice.height_bound(3, 4, 7) == 84

    def test_is_bottom(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert lat.is_bottom(frozenset())
        assert not lat.is_bottom(frozenset({P_DIM_EQ_10}))

    def test_is_top(self):
        lat = PredicateLattice(ALL_SAMPLE)
        assert lat.is_top(ALL_SAMPLE)
        assert not lat.is_top(frozenset({P_DIM_EQ_10}))

    def test_extend_universe(self):
        lat = PredicateLattice(frozenset({P_DIM_EQ_10}))
        assert lat.height() == 1
        lat.extend_universe(frozenset({P_DIM_GE_5, P_NDIM_3}))
        assert lat.height() == 3
        assert P_DIM_GE_5 in lat.universe

    def test_open_lattice_starts_empty(self):
        lat = PredicateLattice()  # open — no fixed universe
        assert lat.top() == frozenset()
        assert lat.height() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CPA component interactions
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPADomain:
    def test_initial_state_is_bottom(self):
        lat = PredicateLattice(ALL_SAMPLE)
        dom = CPADomain(lat)
        init = dom.initial_state()
        assert init.predicates == frozenset()
        assert init.iteration == 0
        assert dom.is_bottom(init)

    def test_domain_leq(self):
        lat = PredicateLattice(ALL_SAMPLE)
        dom = CPADomain(lat)
        s1 = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        s2 = AbstractState(frozenset({P_DIM_EQ_10, P_DIM_GE_5}), 1)
        assert dom.leq(s1, s2)
        assert not dom.leq(s2, s1)


class TestTransferFunction:
    def test_apply_adds_predicates(self):
        def refine(preds):
            return preds | frozenset({P_DIM_EQ_10})

        tf = TransferFunction(refine)
        state = AbstractState(frozenset(), 0)
        new_state = tf.apply(state)
        assert P_DIM_EQ_10 in new_state.predicates
        assert new_state.iteration == 1

    def test_apply_fixed_point(self):
        """If refinement adds nothing, the state is unchanged."""
        def refine(preds):
            return preds  # identity — already at fixed point

        tf = TransferFunction(refine)
        state = AbstractState(frozenset({P_DIM_EQ_10}), 3)
        new_state = tf.apply(state)
        assert new_state.predicates == state.predicates


class TestPrecisionAdjustment:
    def test_adjust_adds_predicates(self):
        pa = PrecisionAdjustment()
        state = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        new_preds = frozenset({P_DIM_GE_5})
        adjusted = pa.adjust(state, new_preds)
        assert P_DIM_EQ_10 in adjusted.predicates
        assert P_DIM_GE_5 in adjusted.predicates

    def test_adjust_monotone(self):
        pa = PrecisionAdjustment()
        state = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        new_preds = frozenset({P_DIM_GE_5})
        adjusted = pa.adjust(state, new_preds)
        assert pa.is_monotone(state, adjusted)

    def test_adjust_with_cap(self):
        pa = PrecisionAdjustment(max_predicates=2)
        state = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        new_preds = frozenset({P_DIM_GE_5, P_NDIM_3})
        adjusted = pa.adjust(state, new_preds)
        assert len(adjusted.predicates) <= 2

    def test_is_monotone_detects_removal(self):
        pa = PrecisionAdjustment()
        before = AbstractState(frozenset({P_DIM_EQ_10, P_DIM_GE_5}), 0)
        after = AbstractState(frozenset({P_DIM_EQ_10}), 1)
        assert not pa.is_monotone(before, after)


class TestMergeOperator:
    def test_join_merge(self):
        merge = MergeOperator("join")
        s1 = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        s2 = AbstractState(frozenset({P_DIM_GE_5}), 1)
        merged = merge.merge(s1, s2)
        assert merged.predicates == frozenset({P_DIM_EQ_10, P_DIM_GE_5})

    def test_sep_merge_keeps_second(self):
        merge = MergeOperator("sep")
        s1 = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        s2 = AbstractState(frozenset({P_DIM_GE_5}), 1)
        merged = merge.merge(s1, s2)
        assert merged.predicates == frozenset({P_DIM_GE_5})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown merge mode"):
            MergeOperator("invalid")


class TestStopOperator:
    def test_subsumed_by_superset(self):
        reached = [AbstractState(frozenset({P_DIM_EQ_10, P_DIM_GE_5}), 0)]
        new = AbstractState(frozenset({P_DIM_EQ_10}), 1)
        assert StopOperator.is_subsumed(new, reached)

    def test_not_subsumed_by_different(self):
        reached = [AbstractState(frozenset({P_DIM_EQ_10}), 0)]
        new = AbstractState(frozenset({P_DIM_GE_5}), 1)
        assert not StopOperator.is_subsumed(new, reached)

    def test_fixed_point_detection(self):
        s1 = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        s2 = AbstractState(frozenset({P_DIM_EQ_10}), 1)
        assert StopOperator.is_fixed_point(s1, s2)

    def test_not_fixed_point(self):
        s1 = AbstractState(frozenset({P_DIM_EQ_10}), 0)
        s2 = AbstractState(frozenset({P_DIM_EQ_10, P_DIM_GE_5}), 1)
        assert not StopOperator.is_fixed_point(s1, s2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fixed-point convergence
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixedPoint:
    def test_lfp_converges_identity(self):
        """R = identity from the start => LFP = ⊥ in one step."""
        lat = PredicateLattice()

        def refine(p):
            return p

        result = cegar_as_least_fixed_point(refine, lat, max_iterations=10)
        assert result.converged
        assert result.fixed_point == frozenset()
        assert result.iterations == 1
        assert result.is_lfp

    def test_lfp_converges_finite_steps(self):
        """R adds one predicate per step, then stops."""
        preds_to_add = [P_DIM_EQ_10, P_DIM_GE_5, P_NDIM_3]
        step = [0]

        def refine(p):
            if step[0] < len(preds_to_add):
                result = p | frozenset({preds_to_add[step[0]]})
                step[0] += 1
                return result
            return p

        lat = PredicateLattice()
        result = cegar_as_least_fixed_point(refine, lat, max_iterations=20)
        assert result.converged
        assert result.fixed_point == frozenset(preds_to_add)
        assert result.iterations == len(preds_to_add) + 1  # +1 for the check that confirms FP

    def test_lfp_respects_budget(self):
        """R always adds a new predicate => exhausts budget."""
        counter = [0]

        def refine(p):
            counter[0] += 1
            new_pred = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=counter[0])
            return p | frozenset({new_pred})

        lat = PredicateLattice()
        result = cegar_as_least_fixed_point(refine, lat, max_iterations=5)
        assert not result.converged
        assert result.iterations == 5

    def test_gfp_empty_universe_falls_back_to_lfp(self):
        """GFP with empty universe delegates to LFP."""
        lat = PredicateLattice()

        def refine(p):
            return p

        result = cegar_as_greatest_fixed_point(refine, lat, max_iterations=10)
        assert result.converged

    def test_gfp_converges_from_top(self):
        """GFP starting from full universe, R is identity => FP = ⊤."""
        lat = PredicateLattice(ALL_SAMPLE)

        def refine(p):
            return p

        result = cegar_as_greatest_fixed_point(refine, lat, max_iterations=10)
        assert result.converged
        assert result.fixed_point == ALL_SAMPLE
        assert not result.is_lfp

    def test_trajectory_is_monotone(self):
        """Each step in the LFP trajectory is a superset of the previous."""
        preds = [P_DIM_EQ_10, P_DIM_GE_5, P_NDIM_3, P_DIV_8]
        step = [0]

        def refine(p):
            if step[0] < len(preds):
                result = p | frozenset({preds[step[0]]})
                step[0] += 1
                return result
            return p

        lat = PredicateLattice()
        result = cegar_as_least_fixed_point(refine, lat, max_iterations=20)
        for i in range(len(result.trajectory) - 1):
            assert result.trajectory[i] <= result.trajectory[i + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Height bound verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeightBound:
    def test_num_predicate_kinds_matches(self):
        """NUM_PREDICATE_KINDS matches the actual enum."""
        assert NUM_PREDICATE_KINDS == len(PredicateKind)

    def test_height_bound_scales_linearly(self):
        h1 = PredicateLattice.height_bound(2, 3)
        h2 = PredicateLattice.height_bound(4, 3)
        assert h2 == 2 * h1

    def test_lfp_within_height_bound(self):
        """LFP converges within lattice height when universe is finite."""
        universe = frozenset({P_DIM_EQ_10, P_DIM_GE_5, P_NDIM_3})
        lat = PredicateLattice(universe)
        preds_iter = iter([P_DIM_EQ_10, P_DIM_GE_5, P_NDIM_3])

        def refine(p):
            try:
                return p | frozenset({next(preds_iter)})
            except StopIteration:
                return p

        result = cegar_as_least_fixed_point(refine, lat, max_iterations=100)
        assert result.converged
        # iterations <= height + 1 (one extra to confirm FP)
        assert result.iterations <= lat.height() + 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Widening operator
# ═══════════════════════════════════════════════════════════════════════════════

class TestWidening:
    def test_widen_no_limit_is_join(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5})
        assert widen(p1, p2) == frozenset({P_DIM_EQ_10, P_DIM_GE_5})

    def test_widen_with_limit(self):
        p1 = frozenset({P_DIM_EQ_10, P_DIM_GE_5})
        p2 = frozenset({P_NDIM_3, P_DIV_8})
        result = widen(p1, p2, max_size=2)
        assert len(result) <= 2

    def test_widen_prefers_new_predicates(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5, P_NDIM_3})
        result = widen(p1, p2, max_size=2)
        # New predicates (from p2 - p1) should be preferred
        new_preds = p2 - p1
        assert len(result & new_preds) > 0

    def test_widen_superset_of_join_when_no_limit(self):
        p1 = frozenset({P_DIM_EQ_10})
        p2 = frozenset({P_DIM_GE_5})
        joined = PredicateLattice.join(p1, p2)
        widened = widen(p1, p2)
        assert widened >= joined


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Convergence certificate
# ═══════════════════════════════════════════════════════════════════════════════

class TestConvergenceCertificate:
    def test_valid_certificate(self):
        trajectory = [
            frozenset(),
            frozenset({P_DIM_EQ_10}),
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),  # fixed point
        ]
        cert = convergence_certificate(trajectory, lattice_height_bound=10, converged=True)
        assert cert.is_valid
        assert cert.monotonicity_verified
        assert cert.fixed_point_reached
        assert cert.actual_iterations == 3
        assert cert.final_predicate_count == 2
        assert cert.lattice_height_bound == 10

    def test_invalid_certificate_non_monotone(self):
        trajectory = [
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),
            frozenset({P_DIM_EQ_10}),  # removed a predicate!
        ]
        cert = convergence_certificate(trajectory, lattice_height_bound=10, converged=False)
        assert not cert.monotonicity_verified
        assert not cert.is_valid

    def test_certificate_not_converged(self):
        trajectory = [
            frozenset(),
            frozenset({P_DIM_EQ_10}),
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),
        ]
        cert = convergence_certificate(trajectory, lattice_height_bound=10, converged=False)
        assert not cert.fixed_point_reached  # last two differ
        assert cert.monotonicity_verified

    def test_certificate_summary(self):
        trajectory = [frozenset(), frozenset(), ]
        cert = convergence_certificate(trajectory, lattice_height_bound=5, converged=True)
        s = cert.summary()
        assert "VALID" in s

    def test_certificate_trajectory_sizes(self):
        trajectory = [
            frozenset(),
            frozenset({P_DIM_EQ_10}),
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),
            frozenset({P_DIM_EQ_10, P_DIM_GE_5}),
        ]
        cert = convergence_certificate(trajectory, lattice_height_bound=10, converged=True)
        assert cert.trajectory_sizes == [0, 1, 2, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CEGARCPA integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCEGARCPA:
    def test_cpa_converges_trivial(self):
        """CPA with identity refinement converges immediately."""
        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=lambda p: p, max_iterations=10)
        result = cpa.run()
        assert result.converged
        assert result.fixed_point == frozenset()

    def test_cpa_converges_finite_steps(self):
        """CPA converges after adding a few predicates."""
        preds = [P_DIM_EQ_10, P_DIM_GE_5, P_NDIM_3]
        step = [0]

        def refine(p):
            if step[0] < len(preds):
                result = p | frozenset({preds[step[0]]})
                step[0] += 1
                return result
            return p

        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=refine, max_iterations=20)
        result = cpa.run()
        assert result.converged
        assert result.fixed_point == frozenset(preds)

    def test_cpa_certificate(self):
        """CPA produces a valid convergence certificate."""
        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=lambda p: p, max_iterations=10)
        cpa.run()
        cert = cpa.get_convergence_certificate(num_layers=2, max_dims=3)
        assert cert.is_valid
        assert cert.lattice_height_bound == PredicateLattice.height_bound(2, 3)

    def test_cpa_with_widening_cap(self):
        """CPA with max_predicates limits predicate set size."""
        counter = [0]

        def refine(p):
            counter[0] += 1
            new_pred = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=counter[0])
            return p | frozenset({new_pred})

        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=refine, max_iterations=10, max_predicates=3)
        result = cpa.run()
        # Due to widening cap, predicate set should be bounded
        assert len(result.fixed_point) <= 3 or not result.converged

    def test_cpa_merge_join(self):
        """CPA with join merge combines predicate sets."""
        step = [0]
        preds = [P_DIM_EQ_10, P_DIM_GE_5]

        def refine(p):
            if step[0] < len(preds):
                result = p | frozenset({preds[step[0]]})
                step[0] += 1
                return result
            return p

        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=refine, max_iterations=20, merge_mode="join")
        result = cpa.run()
        assert result.converged

    def test_cpa_merge_sep(self):
        """CPA with sep merge keeps states separate."""
        lat = PredicateLattice()
        cpa = CEGARCPA(lat, refinement_fn=lambda p: p, max_iterations=10, merge_mode="sep")
        result = cpa.run()
        assert result.converged


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Integration with ShapeCEGARLoop
# ═══════════════════════════════════════════════════════════════════════════════

SIMPLE_LINEAR = """\
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
"""


class TestShapeCEGARIntegration:
    def test_shape_cegar_as_cpa_runs(self):
        """shape_cegar_as_cpa runs and returns result + certificate."""
        result, cert = shape_cegar_as_cpa(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert isinstance(result, ShapeCEGARResult)
        assert isinstance(cert, ConvergenceCertificate)
        assert cert.actual_iterations >= 0
        assert cert.lattice_height_bound > 0

    def test_shape_cegar_as_cpa_certificate_monotonicity(self):
        """Certificate from shape_cegar_as_cpa has monotone trajectory."""
        result, cert = shape_cegar_as_cpa(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        assert cert.monotonicity_verified

    def test_shape_cegar_as_cpa_safe_model(self):
        """Safe model produces a converged certificate."""
        result, cert = shape_cegar_as_cpa(SIMPLE_LINEAR, input_shapes={"x": ("batch", 10)})
        if result.final_status == CEGARStatus.SAFE:
            assert cert.fixed_point_reached
