"""Tests for broadcast associativity correspondence between Lean proofs
and the BroadcastPropagator implementation.

Verifies that the pairwise-compatibility precondition assumed by the
Lean broadcast_assoc theorem is enforced by the implementation.
"""

from __future__ import annotations

import itertools

import pytest

from src.smt.broadcast_theory import (
    _are_dims_broadcast_compatible,
    _broadcast_result,
)

try:
    import z3
    from src.smt.broadcast_theory import (
        BroadcastPropagator,
        BroadcastTheoryPlugin,
        broadcast_result_dim,
    )

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

pytestmark = pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")


# -----------------------------------------------------------------------
# 1. Broadcast associativity for all valid dimension combos (1-10)
# -----------------------------------------------------------------------

class TestBroadcastAssociativity:
    """Verify broadcast_assoc holds for all compatible triples in 1-10."""

    def test_associativity_all_compatible_triples(self):
        """bc(bc(a,b),c) == bc(a,bc(b,c)) for all pairwise-compatible
        (a,b,c) with values in 1..10."""
        for a, b, c in itertools.product(range(1, 11), repeat=3):
            if not _are_dims_broadcast_compatible(a, b):
                continue
            if not _are_dims_broadcast_compatible(b, c):
                continue
            ab = _broadcast_result(a, b)
            bc = _broadcast_result(b, c)
            if not _are_dims_broadcast_compatible(ab, c):
                continue
            if not _are_dims_broadcast_compatible(a, bc):
                continue
            left = _broadcast_result(ab, c)
            right = _broadcast_result(a, bc)
            assert left == right, (
                f"Associativity failed for ({a},{b},{c}): "
                f"bc(bc({a},{b}),{c})={left} != bc({a},bc({b},{c}))={right}"
            )

    def test_associativity_with_ones(self):
        """Associativity holds when some dims are 1 (broadcast stretch)."""
        cases = [
            (1, 1, 1),
            (1, 5, 1),
            (1, 1, 7),
            (3, 1, 3),
            (1, 4, 4),
        ]
        for a, b, c in cases:
            ab = _broadcast_result(a, b)
            bc = _broadcast_result(b, c)
            assert _broadcast_result(ab, c) == _broadcast_result(a, bc)

    def test_associativity_equal_dims(self):
        """Associativity is trivial when all dims are equal."""
        for d in range(1, 11):
            assert _broadcast_result(
                _broadcast_result(d, d), d
            ) == _broadcast_result(d, _broadcast_result(d, d))

    def test_verify_method_returns_true_for_compatible(self):
        """BroadcastPropagator.verify_broadcast_associativity returns True
        for all pairwise-compatible triples in 1..10."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        for a, b, c in itertools.product(range(1, 11), repeat=3):
            result = prop.verify_broadcast_associativity(a, b, c)
            assert result is True, (
                f"verify_broadcast_associativity({a},{b},{c}) returned False"
            )

    def test_verify_method_vacuous_for_incompatible(self):
        """When preconditions fail, verify returns True (vacuously)."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        # (3, 5) are not compatible
        assert prop.verify_broadcast_associativity(3, 5, 2) is True
        assert prop.verify_broadcast_associativity(2, 3, 5) is True


# -----------------------------------------------------------------------
# 2. Correspondence check passes for well-formed queries
# -----------------------------------------------------------------------

class TestCorrespondenceWellFormed:
    """check_correspondence_preconditions passes on valid queries."""

    def test_single_compatible_triple(self):
        """No violations for a single compatible broadcast triple."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, out = z3.Ints("a b out")
        s.add(broadcast_result_dim(prop, a, b, out))
        s.add(a == 3, b == 1)
        assert s.check() == z3.sat
        violations = prop.check_correspondence_preconditions()
        assert violations == []

    def test_chained_compatible_triples(self):
        """No violations for chained bc(bc(a,b), c)."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, ab, c, abc = z3.Ints("a b ab c abc")
        s.add(broadcast_result_dim(prop, a, b, ab))
        s.add(broadcast_result_dim(prop, ab, c, abc))
        s.add(a == 1, b == 5, c == 5)
        assert s.check() == z3.sat
        violations = prop.check_correspondence_preconditions()
        assert violations == []

    def test_multiple_independent_triples(self):
        """No violations for independent (non-chained) triples."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, ab = z3.Ints("a b ab")
        c, d, cd = z3.Ints("c d cd")
        s.add(broadcast_result_dim(prop, a, b, ab))
        s.add(broadcast_result_dim(prop, c, d, cd))
        s.add(a == 3, b == 1, c == 4, d == 4)
        assert s.check() == z3.sat
        violations = prop.check_correspondence_preconditions()
        assert violations == []

    def test_no_triples_no_violations(self):
        """Empty propagator has no violations."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        violations = prop.check_correspondence_preconditions()
        assert violations == []


# -----------------------------------------------------------------------
# 3. Correspondence check catches precondition violations
# -----------------------------------------------------------------------

class TestCorrespondenceViolations:
    """check_correspondence_preconditions detects violations."""

    def test_incompatible_single_triple(self):
        """Detects incompatible dims in a single triple.

        Note: Z3 itself would report UNSAT, but the correspondence
        checker independently verifies the precondition.
        """
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, out = z3.Ints("a b out")
        # Manually register and fix without going through Z3
        prop._register_var(a)
        prop._register_var(b)
        prop._register_var(out)
        prop._broadcast_triples.append((a, b, out))
        # Simulate fixed values directly (bypassing Z3 solver)
        prop._fixed[a.get_id()] = 3
        prop._fixed[b.get_id()] = 5
        violations = prop.check_correspondence_preconditions()
        assert len(violations) == 1
        assert "not broadcast-compatible" in violations[0]

    def test_multiple_incompatible_triples(self):
        """Detects violations across multiple triples."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, ab = z3.Ints("a2 b2 ab2")
        c, d, cd = z3.Ints("c2 d2 cd2")
        prop._register_var(a)
        prop._register_var(b)
        prop._register_var(ab)
        prop._broadcast_triples.append((a, b, ab))
        prop._register_var(c)
        prop._register_var(d)
        prop._register_var(cd)
        prop._broadcast_triples.append((c, d, cd))
        prop._fixed[a.get_id()] = 3
        prop._fixed[b.get_id()] = 7
        prop._fixed[c.get_id()] = 2
        prop._fixed[d.get_id()] = 9
        violations = prop.check_correspondence_preconditions()
        assert len(violations) == 2

    def test_partially_fixed_no_violation(self):
        """No violation when dims are not yet fixed."""
        s = z3.Solver()
        prop = BroadcastPropagator(s)
        a, b, out = z3.Ints("a3 b3 out3")
        prop._register_var(a)
        prop._register_var(b)
        prop._register_var(out)
        prop._broadcast_triples.append((a, b, out))
        # Only fix one variable
        prop._fixed[a.get_id()] = 3
        violations = prop.check_correspondence_preconditions()
        assert violations == []
