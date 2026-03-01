"""
Tests for Knuth-Bendix completion of the tensor shape rewrite system.

Covers:
  - RPO ordering on concrete terms
  - Rule orientation
  - Critical pair computation
  - Joinability checking
  - Term normalization
  - Z3 expression normalization
  - Completion procedure convergence
  - Confluence of the completed system
  - Termination verification
  - Interaction with Z3's simplifier
"""

from __future__ import annotations

import pytest

from src.knuth_bendix import (
    CriticalPair,
    CompletionResult,
    Equation,
    RewriteRule,
    SymbolKind,
    Term,
    ac_normalize,
    apply_substitution,
    build_tensor_shape_trs,
    compute_critical_pairs,
    full_normalize,
    get_completed_rules,
    knuth_bendix_completion,
    match_term,
    normalize,
    orient_equation,
    rpo_ge,
    rpo_gt,
    tensor_shape_axioms,
    unify,
    verify_confluence,
    verify_termination,
)

try:
    import z3
    from src.knuth_bendix import normalize_z3_expr, term_to_z3, z3_to_term
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ======================================================================
# Helpers
# ======================================================================

a = Term.var("a")
b = Term.var("b")
c = Term.var("c")
s = Term.var("s")
t = Term.var("t")
d0 = Term.var("d0")
d1 = Term.var("d1")
h = Term.var("h")
k = Term.var("k")
_0 = Term.const(0)
_1 = Term.const(1)
_2 = Term.const(2)
_3 = Term.const(3)
_5 = Term.const(5)


# ======================================================================
# Test Term Construction
# ======================================================================


class TestTermConstruction:
    def test_variable_term(self):
        x = Term.var("x")
        assert x.is_var
        assert x.name == "x"
        assert not x.is_const
        assert not x.is_compound

    def test_constant_term(self):
        c = Term.const(42)
        assert c.is_const
        assert c.value == 42
        assert not c.is_var

    def test_compound_term(self):
        t = Term.bc(a, b)
        assert t.is_compound
        assert t.symbol == SymbolKind.BC
        assert len(t.children) == 2

    def test_nested_term(self):
        t = Term.bc(Term.bc(a, b), c)
        assert t.children[0].symbol == SymbolKind.BC
        assert t.children[0].children == (a, b)

    def test_variables_in_term(self):
        t = Term.bc(Term.bc(a, b), c)
        assert t.variables() == frozenset({"a", "b", "c"})

    def test_term_size(self):
        assert a.size() == 1
        assert _1.size() == 1
        assert Term.bc(a, b).size() == 3
        assert Term.bc(Term.bc(a, b), c).size() == 5

    def test_term_repr(self):
        t = Term.bc(a, _1)
        assert "bc" in repr(t)
        assert "a" in repr(t)


# ======================================================================
# Test RPO Ordering
# ======================================================================


class TestRPO:
    """Test Recursive Path Ordering on concrete terms."""

    def test_rpo_bc_gt_const(self):
        """bc(a, b) >_RPO 1 because bc > const in precedence."""
        assert rpo_gt(Term.bc(a, b), _1)

    def test_rpo_numel_gt_bc(self):
        """numel(s) >_RPO bc(a, b) because numel > bc in precedence."""
        assert rpo_gt(Term.numel(a), Term.bc(a, _1))

    def test_rpo_conv_gt_add(self):
        """conv(h,k,1,0) >_RPO add(sub(h,k), 1) because conv > add."""
        lhs = Term.conv(h, k, _1, _0)
        rhs = Term.add(Term.sub(h, k), _1)
        assert rpo_gt(lhs, rhs)

    def test_rpo_pool_gt_floor_div(self):
        """pool(h,k,k,0) >_RPO floor_div(h,k)."""
        lhs = Term.pool(h, k, k, _0)
        rhs = Term.floor_div(h, k)
        assert rpo_gt(lhs, rhs)

    def test_rpo_transp_transp_gt_var(self):
        """transp(transp(s,d0,d1), d0, d1) >_RPO s."""
        lhs = Term.transp(Term.transp(s, d0, d1), d0, d1)
        assert rpo_gt(lhs, s)

    def test_rpo_bc_identity_orientable(self):
        """bc(a, 1) >_RPO a because bc > var in precedence."""
        assert rpo_gt(Term.bc(a, _1), a)

    def test_rpo_numel_reshape_gt_numel(self):
        """numel(reshape(s,t)) >_RPO numel(s) by subterm/lex."""
        lhs = Term.numel(Term.reshape(s, t))
        rhs = Term.numel(s)
        assert rpo_gt(lhs, rhs)

    def test_rpo_var_not_gt_var(self):
        """Variables are incomparable."""
        assert not rpo_gt(a, b)

    def test_rpo_reflexive_false(self):
        """No term is strictly greater than itself."""
        t = Term.bc(a, b)
        assert not rpo_gt(t, t)

    def test_rpo_ge_reflexive(self):
        """≥_RPO is reflexive."""
        t = Term.bc(a, b)
        assert rpo_ge(t, t)

    def test_rpo_const_incomparable(self):
        """Different constants are incomparable in RPO."""
        assert not rpo_gt(_1, _2)
        assert not rpo_gt(_2, _1)


# ======================================================================
# Test Rule Orientation
# ======================================================================


class TestRuleOrientation:
    """Test orienting equations into rewrite rules via RPO."""

    def test_orient_bc_identity_right(self):
        eq = Equation(Term.bc(a, _1), a)
        rule = orient_equation(eq, 0)
        assert rule is not None
        assert rule.lhs == Term.bc(a, _1)
        assert rule.rhs == a

    def test_orient_bc_identity_left(self):
        eq = Equation(Term.bc(_1, b), b)
        rule = orient_equation(eq, 0)
        assert rule is not None
        assert rule.lhs == Term.bc(_1, b)
        assert rule.rhs == b

    def test_orient_bc_idempotent(self):
        eq = Equation(Term.bc(a, a), a)
        rule = orient_equation(eq, 0)
        assert rule is not None
        assert rule.lhs == Term.bc(a, a)
        assert rule.rhs == a

    def test_orient_double_transpose(self):
        eq = Equation(Term.transp(Term.transp(s, d0, d1), d0, d1), s)
        rule = orient_equation(eq, 0)
        assert rule is not None
        assert rule.rhs == s

    def test_orient_numel_reshape(self):
        eq = Equation(Term.numel(Term.reshape(s, t)), Term.numel(s))
        rule = orient_equation(eq, 0)
        assert rule is not None
        assert rule.lhs == Term.numel(Term.reshape(s, t))

    def test_orient_commutativity_fails(self):
        """Commutativity bc(a,b) = bc(b,a) cannot be oriented."""
        eq = Equation(Term.bc(a, b), Term.bc(b, a))
        rule = orient_equation(eq, 0)
        assert rule is None

    def test_orient_conv_basic(self):
        eq = Equation(
            Term.conv(h, k, _1, _0),
            Term.add(Term.sub(h, k), _1),
        )
        rule = orient_equation(eq, 0)
        assert rule is not None

    def test_orient_pool_basic(self):
        eq = Equation(
            Term.pool(h, k, k, _0),
            Term.floor_div(h, k),
        )
        rule = orient_equation(eq, 0)
        assert rule is not None


# ======================================================================
# Test Unification and Matching
# ======================================================================


class TestUnification:
    def test_unify_identical(self):
        subst = unify(a, a)
        assert subst is not None
        assert len(subst) == 0

    def test_unify_var_const(self):
        subst = unify(a, _1)
        assert subst is not None
        assert subst["a"] == _1

    def test_unify_compound(self):
        t1 = Term.bc(a, _1)
        t2 = Term.bc(_2, b)
        subst = unify(t1, t2)
        assert subst is not None
        assert subst["a"] == _2
        assert subst["b"] == _1

    def test_unify_occurs_check(self):
        """Unification should fail when occurs check fails."""
        subst = unify(a, Term.bc(a, _1))
        assert subst is None

    def test_match_pattern(self):
        pattern = Term.bc(a, _1)
        target = Term.bc(_5, _1)
        subst = match_term(pattern, target)
        assert subst is not None
        assert subst["a"] == _5

    def test_match_no_target_vars(self):
        """Matching should not bind target variables."""
        pattern = _1
        target = Term.bc(a, b)
        subst = match_term(pattern, target)
        assert subst is None


# ======================================================================
# Test Critical Pair Computation
# ======================================================================


class TestCriticalPairs:
    def test_critical_pairs_bc_rules(self):
        """Critical pairs between bc(a,1)→a and bc(a,a)→a."""
        r1 = RewriteRule(1, Term.bc(a, _1), a, "bc_id_right")
        r3 = RewriteRule(3, Term.bc(a, a), a, "bc_idemp")
        cps = compute_critical_pairs(r3, r1)
        # Overlap: bc(a, a) where inner a unifies with bc(x, 1)
        # This can produce pairs — the exact count depends on overlaps
        assert isinstance(cps, list)

    def test_critical_pairs_self_overlap(self):
        """No trivial self-overlap at root for the same rule."""
        r1 = RewriteRule(1, Term.bc(a, _1), a, "bc_id_right")
        cps = compute_critical_pairs(r1, r1)
        # Root overlap is excluded for same rule
        assert all(cp.overlap_position != () for cp in cps)

    def test_critical_pair_structure(self):
        r1 = RewriteRule(1, Term.bc(a, _1), a, "bc_id_right")
        r2 = RewriteRule(2, Term.bc(_1, b), b, "bc_id_left")
        cps = compute_critical_pairs(r1, r2)
        for cp in cps:
            assert isinstance(cp, CriticalPair)
            assert isinstance(cp.term1, Term)
            assert isinstance(cp.term2, Term)


# ======================================================================
# Test Normalization
# ======================================================================


class TestNormalization:
    def test_normalize_bc_identity_right(self):
        """bc(x, 1) normalizes to x."""
        rules = get_completed_rules()
        t = Term.bc(Term.var("x"), _1)
        nf = normalize(t, rules)
        assert nf == Term.var("x")

    def test_normalize_bc_identity_left(self):
        """bc(1, y) normalizes to y."""
        rules = get_completed_rules()
        t = Term.bc(_1, Term.var("y"))
        nf = normalize(t, rules)
        assert nf == Term.var("y")

    def test_normalize_bc_idempotent(self):
        """bc(x, x) normalizes to x."""
        rules = get_completed_rules()
        t = Term.bc(Term.var("x"), Term.var("x"))
        nf = normalize(t, rules)
        assert nf == Term.var("x")

    def test_normalize_double_transpose(self):
        """transp(transp(s, 0, 1), 0, 1) normalizes to s."""
        rules = get_completed_rules()
        t = Term.transp(Term.transp(s, d0, d1), d0, d1)
        nf = normalize(t, rules)
        assert nf == s

    def test_normalize_numel_reshape(self):
        """numel(reshape(s, t)) normalizes to numel(s)."""
        rules = get_completed_rules()
        term = Term.numel(Term.reshape(s, t))
        nf = normalize(term, rules)
        assert nf == Term.numel(s)

    def test_normalize_conv_basic(self):
        """conv(h, k, 1, 0) normalizes to add(sub(h, k), 1)."""
        rules = get_completed_rules()
        term = Term.conv(h, k, _1, _0)
        nf = normalize(term, rules)
        expected = Term.add(Term.sub(h, k), _1)
        assert nf == expected

    def test_normalize_pool_basic(self):
        """pool(h, k, k, 0) normalizes to floor_div(h, k)."""
        rules = get_completed_rules()
        term = Term.pool(h, k, k, _0)
        nf = normalize(term, rules)
        expected = Term.floor_div(h, k)
        assert nf == expected

    def test_normalize_nested_bc(self):
        """bc(bc(x, 1), y) normalizes to bc(x, y)."""
        rules = get_completed_rules()
        x, y = Term.var("x"), Term.var("y")
        term = Term.bc(Term.bc(x, _1), y)
        nf = normalize(term, rules)
        assert nf == Term.bc(x, y)

    def test_normalize_already_normal(self):
        """A term in normal form is unchanged."""
        rules = get_completed_rules()
        x = Term.var("x")
        nf = normalize(x, rules)
        assert nf == x

    def test_normalize_chained_reshape_numel(self):
        """numel(reshape(reshape(s, t1), t2)) → numel(s)."""
        rules = get_completed_rules()
        t1, t2 = Term.var("t1"), Term.var("t2")
        term = Term.numel(Term.reshape(Term.reshape(s, t1), t2))
        nf = normalize(term, rules)
        assert nf == Term.numel(s)


# ======================================================================
# Test AC-Normalization
# ======================================================================


class TestACNormalization:
    def test_ac_normalize_bc_sorted(self):
        """ac_normalize sorts bc arguments canonically."""
        t1 = ac_normalize(Term.bc(b, a))
        t2 = ac_normalize(Term.bc(a, b))
        assert t1 == t2

    def test_ac_normalize_nested(self):
        """AC normalization descends into subterms."""
        t = Term.numel(Term.bc(b, a))
        tn = ac_normalize(t)
        assert tn == Term.numel(Term.bc(a, b))


# ======================================================================
# Test Full Normalization
# ======================================================================


class TestFullNormalization:
    def test_full_normalize_combines_ac_and_kb(self):
        """full_normalize applies both AC and KB rules."""
        term = Term.bc(Term.bc(_1, a), b)
        nf = full_normalize(term)
        # bc(1, a) → a by KB, then bc(a, b) is the result
        assert nf == Term.bc(a, b) or nf == Term.bc(b, a)

    def test_full_normalize_idempotent(self):
        """Applying full_normalize twice gives the same result."""
        term = Term.bc(Term.bc(a, _1), Term.bc(b, b))
        nf1 = full_normalize(term)
        nf2 = full_normalize(nf1)
        assert nf1 == nf2


# ======================================================================
# Test Completion Procedure
# ======================================================================


class TestCompletion:
    def test_completion_terminates(self):
        """KB completion on tensor shape axioms terminates."""
        result = build_tensor_shape_trs()
        assert isinstance(result, CompletionResult)
        assert result.iterations < 100

    def test_completion_produces_rules(self):
        """Completion produces at least the directly orientable rules."""
        result = build_tensor_shape_trs()
        assert len(result.rules) >= 5

    def test_completion_converges(self):
        """Completion converges (no unresolved equations)."""
        result = build_tensor_shape_trs()
        assert result.converged

    def test_simple_completion(self):
        """Test completion on a simple two-equation system."""
        x, y = Term.var("x"), Term.var("y")
        axioms = [
            Equation(Term.bc(x, _1), x),
            Equation(Term.bc(x, x), x),
        ]
        result = knuth_bendix_completion(axioms)
        assert result.converged
        assert len(result.rules) >= 2


# ======================================================================
# Test Termination Verification
# ======================================================================


class TestTermination:
    def test_all_rules_terminate(self):
        """Every rule l → r satisfies l >_RPO r."""
        rules = get_completed_rules()
        terminates, failing = verify_termination(rules)
        assert terminates, f"Failing rules: {failing}"
        assert len(failing) == 0


# ======================================================================
# Test Confluence Verification
# ======================================================================


class TestConfluence:
    def test_completed_system_is_confluent(self):
        """The completed TRS is confluent (all CPs joinable)."""
        rules = get_completed_rules()
        is_confluent, non_joinable = verify_confluence(rules)
        assert is_confluent, f"Non-joinable pairs: {non_joinable}"


# ======================================================================
# Test Z3 Integration
# ======================================================================


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
class TestZ3Integration:
    def test_z3_to_term_intval(self):
        expr = z3.IntVal(42)
        term = z3_to_term(expr)
        assert term == Term.const(42)

    def test_z3_to_term_variable(self):
        x = z3.Int("x")
        term = z3_to_term(x)
        assert term.is_var
        assert term.name == "x"

    def test_z3_to_term_addition(self):
        x, y = z3.Int("x"), z3.Int("y")
        expr = x + y
        term = z3_to_term(expr)
        assert term.symbol == SymbolKind.ADD

    def test_z3_roundtrip_variable(self):
        x = z3.Int("x")
        term = z3_to_term(x)
        back = term_to_z3(term)
        assert z3.eq(back, x)

    def test_z3_roundtrip_arithmetic(self):
        x, y = z3.Int("x"), z3.Int("y")
        expr = x + y
        term = z3_to_term(z3.simplify(expr))
        back = term_to_z3(term)
        s = z3.Solver()
        s.add(back != expr)
        assert s.check() == z3.unsat

    def test_normalize_z3_bc_identity(self):
        """normalize_z3_expr simplifies bc(x, 1) → x."""
        bc = z3.Function("bc", z3.IntSort(), z3.IntSort(), z3.IntSort())
        x = z3.Int("x")
        expr = bc(x, z3.IntVal(1))
        result = normalize_z3_expr(expr)
        assert z3.eq(result, x)

    def test_normalize_z3_idempotent(self):
        """Applying normalize_z3_expr twice gives the same result."""
        bc = z3.Function("bc", z3.IntSort(), z3.IntSort(), z3.IntSort())
        x = z3.Int("x")
        expr = bc(x, z3.IntVal(1))
        r1 = normalize_z3_expr(expr)
        r2 = normalize_z3_expr(r1)
        assert z3.eq(r1, r2)

    def test_normalize_z3_preserves_semantics(self):
        """Normalization preserves semantic equivalence (checked via Z3)."""
        bc = z3.Function("bc", z3.IntSort(), z3.IntSort(), z3.IntSort())
        x = z3.Int("x")
        expr = bc(bc(x, z3.IntVal(1)), x)
        result = normalize_z3_expr(expr)
        # After normalization, bc(bc(x,1), x) → bc(x, x) → x
        assert z3.eq(result, x)

    def test_z3_simplify_then_kb_then_simplify(self):
        """The three-phase pipeline: z3.simplify → KB → z3.simplify."""
        x, y = z3.Int("x"), z3.Int("y")
        # (x + 0) is simplified by Z3, then KB has nothing to do
        expr = x + 0
        result = normalize_z3_expr(expr)
        assert z3.eq(result, x)


# ======================================================================
# Test Edge Cases
# ======================================================================


class TestEdgeCases:
    def test_normalize_constant(self):
        """Constants are already in normal form."""
        rules = get_completed_rules()
        nf = normalize(_1, rules)
        assert nf == _1

    def test_normalize_variable(self):
        """Variables are already in normal form."""
        rules = get_completed_rules()
        nf = normalize(a, rules)
        assert nf == a

    def test_substitution_identity(self):
        """Empty substitution is identity."""
        t = Term.bc(a, b)
        result = apply_substitution(t, {})
        assert result == t

    def test_substitution_applies(self):
        t = Term.bc(a, b)
        result = apply_substitution(t, {"a": _1, "b": _2})
        assert result == Term.bc(_1, _2)
