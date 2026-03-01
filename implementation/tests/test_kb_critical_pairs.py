"""
Tests for KB critical pair enumeration.

Verifies:
  - All 28 pairs are enumerated (21 inter-rule + 7 self-overlaps)
  - Joinability of each critical pair
  - RPO orientation of each rule
  - K∘Z∘K idempotence on a range of expressions
"""

from __future__ import annotations

import pytest

from src.knuth_bendix import (
    CriticalPair,
    RewriteRule,
    Term,
    ac_normalize,
    compute_critical_pairs,
    full_normalize,
    get_completed_rules,
    normalize,
    rpo_gt,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rules():
    return get_completed_rules()


@pytest.fixture
def rule_pairs(rules):
    """All 28 (i, j) pairs with i <= j."""
    pairs = []
    n = len(rules)
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j))
    return pairs


# ======================================================================
# Helpers
# ======================================================================

def kzk_normalize(term: Term, rules) -> Term:
    step1 = ac_normalize(term)
    step2 = full_normalize(step1, rules)
    step3 = ac_normalize(step2)
    return step3


# ======================================================================
# Test: exactly 7 rules
# ======================================================================


class TestRuleCount:
    def test_exactly_7_rules(self, rules):
        assert len(rules) == 7

    def test_rule_ids(self, rules):
        ids = [r.id for r in rules]
        assert ids == [1, 2, 3, 4, 5, 6, 7]

    def test_rule_names(self, rules):
        names = [r.name for r in rules]
        assert "bc_identity_right" in names
        assert "bc_identity_left" in names
        assert "bc_idempotent" in names
        assert "double_transpose" in names
        assert "reshape_numel" in names
        assert "conv_basic" in names
        assert "pool_stride_eq_kernel" in names


# ======================================================================
# Test: exactly 28 pairs enumerated
# ======================================================================


class TestPairEnumeration:
    def test_28_pairs(self, rule_pairs):
        assert len(rule_pairs) == 28

    def test_7_self_overlaps(self, rule_pairs):
        self_pairs = [(i, j) for i, j in rule_pairs if i == j]
        assert len(self_pairs) == 7

    def test_21_inter_rule(self, rule_pairs):
        inter_pairs = [(i, j) for i, j in rule_pairs if i != j]
        assert len(inter_pairs) == 21

    def test_all_rules_appear(self, rule_pairs, rules):
        all_indices = set()
        for i, j in rule_pairs:
            all_indices.add(i)
            all_indices.add(j)
        assert all_indices == set(range(len(rules)))


# ======================================================================
# Test: RPO orientation of all rules
# ======================================================================


class TestRPOOrientation:
    def test_all_rules_rpo_oriented(self, rules):
        for r in rules:
            assert rpo_gt(r.lhs, r.rhs), (
                f"Rule {r.id} ({r.name}) not RPO-oriented: "
                f"{r.lhs!r} is not >_RPO {r.rhs!r}"
            )

    def test_r1_bc_identity_right(self, rules):
        r = rules[0]
        assert r.name == "bc_identity_right"
        assert rpo_gt(r.lhs, r.rhs)

    def test_r4_double_transpose(self, rules):
        r = rules[3]
        assert r.name == "double_transpose"
        assert rpo_gt(r.lhs, r.rhs)

    def test_r6_conv_basic(self, rules):
        r = rules[5]
        assert r.name == "conv_basic"
        assert rpo_gt(r.lhs, r.rhs)

    def test_r7_pool_stride(self, rules):
        r = rules[6]
        assert r.name == "pool_stride_eq_kernel"
        assert rpo_gt(r.lhs, r.rhs)


# ======================================================================
# Test: critical pair joinability
# ======================================================================


class TestCriticalPairJoinability:
    def test_all_self_overlap_cps_joinable(self, rules):
        for r in rules:
            cps = compute_critical_pairs(r, r)
            for cp in cps:
                nf1 = full_normalize(cp.term1, rules)
                nf2 = full_normalize(cp.term2, rules)
                assert nf1 == nf2, (
                    f"Non-joinable self-overlap CP for R{r.id}: "
                    f"{cp.term1!r} -> {nf1!r} vs {cp.term2!r} -> {nf2!r}"
                )

    def test_all_inter_rule_cps_joinable(self, rules):
        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules):
                if i == j:
                    continue
                cps = compute_critical_pairs(r1, r2)
                for cp in cps:
                    nf1 = full_normalize(cp.term1, rules)
                    nf2 = full_normalize(cp.term2, rules)
                    assert nf1 == nf2, (
                        f"Non-joinable CP for R{r1.id}×R{r2.id}: "
                        f"{cp.term1!r} -> {nf1!r} vs {cp.term2!r} -> {nf2!r}"
                    )

    def test_r1_r2_overlap(self, rules):
        """R1: bc(a, 1) -> a  and  R2: bc(1, b) -> b
        Overlap at bc(1, 1): both match, producing 1 and 1 -> joinable.
        """
        r1, r2 = rules[0], rules[1]
        cps = compute_critical_pairs(r1, r2) + compute_critical_pairs(r2, r1)
        # Any CPs found should be joinable
        for cp in cps:
            nf1 = full_normalize(cp.term1, rules)
            nf2 = full_normalize(cp.term2, rules)
            assert nf1 == nf2

    def test_r1_r3_overlap(self, rules):
        """R1: bc(a, 1) -> a  and  R3: bc(a, a) -> a
        Overlap at bc(1, 1): R1 gives 1, R3 gives 1 -> joinable.
        """
        r1, r3 = rules[0], rules[2]
        cps = compute_critical_pairs(r1, r3) + compute_critical_pairs(r3, r1)
        for cp in cps:
            nf1 = full_normalize(cp.term1, rules)
            nf2 = full_normalize(cp.term2, rules)
            assert nf1 == nf2

    def test_r2_r3_overlap(self, rules):
        """R2: bc(1, b) -> b  and  R3: bc(a, a) -> a
        Overlap at bc(1, 1): R2 gives 1, R3 gives 1 -> joinable.
        """
        r2, r3 = rules[1], rules[2]
        cps = compute_critical_pairs(r2, r3) + compute_critical_pairs(r3, r2)
        for cp in cps:
            nf1 = full_normalize(cp.term1, rules)
            nf2 = full_normalize(cp.term2, rules)
            assert nf1 == nf2


# ======================================================================
# Test: K∘Z∘K idempotence
# ======================================================================


class TestKZKIdempotence:
    def test_kzk_bc_identity_right(self, rules):
        a = Term.var("a")
        expr = Term.bc(a, Term.const(1))
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_bc_identity_left(self, rules):
        b = Term.var("b")
        expr = Term.bc(Term.const(1), b)
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_bc_idempotent(self, rules):
        a = Term.var("a")
        expr = Term.bc(a, a)
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_double_transpose(self, rules):
        s, d0, d1 = Term.var("s"), Term.var("d0"), Term.var("d1")
        expr = Term.transp(Term.transp(s, d0, d1), d0, d1)
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_reshape_numel(self, rules):
        s, t = Term.var("s"), Term.var("t")
        expr = Term.numel(Term.reshape(s, t))
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_conv(self, rules):
        h, k = Term.var("h"), Term.var("k")
        expr = Term.conv(h, k, Term.const(1), Term.const(0))
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_pool(self, rules):
        h, k = Term.var("h"), Term.var("k")
        expr = Term.pool(h, k, k, Term.const(0))
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_nested(self, rules):
        a, b = Term.var("a"), Term.var("b")
        expr = Term.bc(Term.bc(a, Term.const(1)), b)
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice

    def test_kzk_already_normal(self, rules):
        a = Term.var("a")
        once = kzk_normalize(a, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice
        assert once == a

    def test_kzk_constant(self, rules):
        c = Term.const(42)
        once = kzk_normalize(c, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice
        assert once == c

    def test_kzk_deep_nesting(self, rules):
        a = Term.var("a")
        _1 = Term.const(1)
        expr = Term.bc(Term.bc(Term.bc(Term.bc(a, _1), _1), _1), _1)
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice
        assert once == a

    def test_kzk_mixed_operations(self, rules):
        h, k = Term.var("h"), Term.var("k")
        d0, d1 = Term.var("d0"), Term.var("d1")
        _0, _1 = Term.const(0), Term.const(1)
        expr = Term.transp(
            Term.transp(Term.conv(Term.bc(h, _1), k, _1, _0), d0, d1),
            d0, d1,
        )
        once = kzk_normalize(expr, rules)
        twice = kzk_normalize(once, rules)
        assert once == twice
