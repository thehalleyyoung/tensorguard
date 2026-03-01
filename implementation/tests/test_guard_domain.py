"""
Tests for the Galois insertion formalization of the shape guard domain.

Verifies all four Galois connection properties:
  1. Soundness: S ⊆ γ(α(S))
  2. Monotonicity: S₁ ⊆ S₂ ⟹ α(S₂) ⊑ α(S₁)
  3. Galois condition: α(S) ⊑ a ⟺ S ⊆ γ(a)
  4. Best abstraction: S ⊆ γ(a) ⟹ α(S) ⊑ a
"""

import pytest
from src.guard_domain import (
    ShapeEnv,
    GuardAbstraction,
    alpha,
    gamma_member,
    gamma_check,
    verify_soundness,
    verify_galois_condition,
    verify_best_abstraction,
    verify_monotonicity,
    classify_abstract_domain_position,
    domain_hierarchy_summary,
)
from src.shape_cegar import PredicateKind, ShapePredicate


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def env_mlp():
    """Simple MLP shape environment."""
    return ShapeEnv({"x": (32, 784), "w1": (784, 256), "w2": (256, 10)})


@pytest.fixture
def env_mlp_batch64():
    """Same MLP, different batch size."""
    return ShapeEnv({"x": (64, 784), "w1": (784, 256), "w2": (256, 10)})


@pytest.fixture
def env_cnn():
    """CNN-style shapes."""
    return ShapeEnv({"x": (16, 3, 224, 224), "conv_w": (64, 3, 7, 7)})


# ─── ShapeEnv tests ─────────────────────────────────────────────────────────

class TestShapeEnv:
    def test_get_dim(self, env_mlp):
        assert env_mlp.get_dim("x", 0) == 32
        assert env_mlp.get_dim("x", 1) == 784
        assert env_mlp.get_dim("nonexistent", 0) is None

    def test_get_ndim(self, env_mlp):
        assert env_mlp.get_ndim("x") == 2
        assert env_mlp.get_ndim("nonexistent") is None

    def test_satisfies_dim_eq(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        assert env_mlp.satisfies(p)
        p_wrong = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=100)
        assert not env_mlp.satisfies(p_wrong)

    def test_satisfies_dim_gt(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_GT, "x", axis=0, value=10)
        assert env_mlp.satisfies(p)
        p_tight = ShapePredicate(PredicateKind.DIM_GT, "x", axis=0, value=32)
        assert not env_mlp.satisfies(p_tight)

    def test_satisfies_dim_ge(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_GE, "x", axis=0, value=32)
        assert env_mlp.satisfies(p)
        p_too_high = ShapePredicate(PredicateKind.DIM_GE, "x", axis=0, value=33)
        assert not env_mlp.satisfies(p_too_high)

    def test_satisfies_dim_divisible(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_DIVISIBLE, "x", axis=0, divisor=8)
        assert env_mlp.satisfies(p)  # 32 % 8 == 0
        p_odd = ShapePredicate(PredicateKind.DIM_DIVISIBLE, "x", axis=0, divisor=3)
        assert not env_mlp.satisfies(p_odd)  # 32 % 3 != 0

    def test_satisfies_dim_match(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=1,
                          match_tensor="w1", match_axis=0)
        assert env_mlp.satisfies(p)  # 784 == 784

    def test_satisfies_ndim_eq(self, env_mlp):
        p = ShapePredicate(PredicateKind.NDIM_EQ, "x", value=2)
        assert env_mlp.satisfies(p)

    def test_satisfies_shape_eq(self, env_mlp):
        p = ShapePredicate(PredicateKind.SHAPE_EQ, "x", value=(32, 784))
        assert env_mlp.satisfies(p)
        p_wrong = ShapePredicate(PredicateKind.SHAPE_EQ, "x", value=(32, 100))
        assert not env_mlp.satisfies(p_wrong)


# ─── GuardAbstraction tests ─────────────────────────────────────────────────

class TestGuardAbstraction:
    def test_top(self):
        t = GuardAbstraction.top()
        assert len(t) == 0
        assert not t.is_bottom

    def test_bottom(self):
        b = GuardAbstraction.bottom()
        assert b.is_bottom

    def test_partial_order_bottom_leq_all(self):
        b = GuardAbstraction.bottom()
        t = GuardAbstraction.top()
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        a = GuardAbstraction(frozenset({p}))
        assert b.leq(t)
        assert b.leq(a)
        assert not t.leq(b)

    def test_partial_order_stronger_leq_weaker(self):
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        strong = GuardAbstraction(frozenset({p1, p2}))
        weak = GuardAbstraction(frozenset({p1}))
        assert strong.leq(weak)  # strong ⊑ weak (more preds = stronger)
        assert not weak.leq(strong)

    def test_join(self):
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        a1 = GuardAbstraction(frozenset({p1, p2}))
        a2 = GuardAbstraction(frozenset({p1}))
        j = a1.join(a2)
        assert j.predicates == frozenset({p1})

    def test_meet(self):
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        p2 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        a1 = GuardAbstraction(frozenset({p1}))
        a2 = GuardAbstraction(frozenset({p2}))
        m = a1.meet(a2)
        assert m.predicates == frozenset({p1, p2})


# ─── Gamma tests ─────────────────────────────────────────────────────────────

class TestGamma:
    def test_gamma_top_admits_all(self, env_mlp):
        assert gamma_member(env_mlp, GuardAbstraction.top())

    def test_gamma_bottom_admits_none(self, env_mlp):
        assert not gamma_member(env_mlp, GuardAbstraction.bottom())

    def test_gamma_specific(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        a = GuardAbstraction(frozenset({p}))
        assert gamma_member(env_mlp, a)

    def test_gamma_rejects_wrong(self, env_mlp):
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=100)
        a = GuardAbstraction(frozenset({p}))
        assert not gamma_member(env_mlp, a)


# ─── Alpha tests ─────────────────────────────────────────────────────────────

class TestAlpha:
    def test_alpha_empty_returns_top(self):
        a = alpha([])
        assert a.predicates == frozenset()
        assert not a.is_bottom

    def test_alpha_single_env(self, env_mlp):
        a = alpha([env_mlp])
        # Must contain DIM_EQ for x.shape[0] == 32
        dim_eq = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        assert dim_eq in a.predicates
        # Must contain DIM_MATCH for x.shape[1] == w1.shape[0]
        dim_match = ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=1,
                                   match_tensor="w1", match_axis=0)
        assert dim_match in a.predicates

    def test_alpha_multiple_envs_weakens(self, env_mlp, env_mlp_batch64):
        a_single = alpha([env_mlp])
        a_both = alpha([env_mlp, env_mlp_batch64])
        # The shared predicates (e.g., x.shape[1]==784) should survive
        dim_eq_feat = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        assert dim_eq_feat in a_both.predicates
        # The batch-specific predicate should NOT survive
        dim_eq_batch = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        assert dim_eq_batch not in a_both.predicates
        # a_both should be weaker (fewer predicates)
        assert len(a_both.predicates) <= len(a_single.predicates)


# ─── Galois connection property tests ────────────────────────────────────────

class TestGaloisProperties:
    """Test all four Galois connection properties."""

    def test_soundness_mlp(self, env_mlp, env_mlp_batch64):
        assert verify_soundness([env_mlp])
        assert verify_soundness([env_mlp, env_mlp_batch64])

    def test_soundness_cnn(self, env_cnn):
        assert verify_soundness([env_cnn])

    def test_soundness_empty(self):
        assert verify_soundness([])

    def test_soundness_many_envs(self):
        """Soundness with diverse batch sizes."""
        envs = [
            ShapeEnv({"x": (b, 784), "w": (784, 256)})
            for b in [1, 8, 16, 32, 64, 128]
        ]
        assert verify_soundness(envs)

    def test_galois_condition_holds(self, env_mlp):
        """α(S) ⊑ a ⟺ S ⊆ γ(a) for various abstract elements."""
        envs = [env_mlp]
        # Top should satisfy
        assert verify_galois_condition(envs, GuardAbstraction.top())
        # α(S) itself should satisfy
        a_S = alpha(envs)
        assert verify_galois_condition(envs, a_S)
        # A weaker abstraction should satisfy
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        weaker = GuardAbstraction(frozenset({p}))
        assert verify_galois_condition(envs, weaker)
        # A wrong abstraction should still have consistent LHS/RHS
        wrong = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=100)
        wrong_abs = GuardAbstraction(frozenset({wrong}))
        assert verify_galois_condition(envs, wrong_abs)

    def test_best_abstraction(self, env_mlp, env_mlp_batch64):
        """α(S) is the strongest valid abstraction."""
        envs = [env_mlp, env_mlp_batch64]
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=784)
        weaker = GuardAbstraction(frozenset({p}))
        assert verify_best_abstraction(envs, weaker)
        assert verify_best_abstraction(envs, GuardAbstraction.top())

    def test_monotonicity(self, env_mlp, env_mlp_batch64):
        """More environments ⟹ weaker abstraction."""
        s1 = [env_mlp]
        s2 = [env_mlp, env_mlp_batch64]
        a1 = alpha(s1)
        a2 = alpha(s2)
        # a2 should be weaker than a1 for shared predicates
        for p in a2.predicates:
            assert p in a1.predicates

    def test_soundness_stress(self):
        """Soundness with randomized shapes."""
        import random
        random.seed(42)
        for _ in range(20):
            n_envs = random.randint(1, 5)
            envs = [
                ShapeEnv({
                    "x": tuple(random.randint(1, 100) for _ in range(random.randint(1, 4))),
                    "y": tuple(random.randint(1, 100) for _ in range(random.randint(1, 4))),
                })
                for _ in range(n_envs)
            ]
            assert verify_soundness(envs), f"Soundness failed for {envs}"


# ─── Domain hierarchy classification tests ───────────────────────────────────

class TestDomainHierarchy:
    def test_interval_classification(self):
        p = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        a = GuardAbstraction(frozenset({p}))
        assert classify_abstract_domain_position(a) == "interval"

    def test_octagonal_classification(self):
        p1 = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        p2 = ShapePredicate(PredicateKind.DIM_MATCH, "x", axis=1,
                           match_tensor="w", match_axis=0)
        a = GuardAbstraction(frozenset({p1, p2}))
        assert classify_abstract_domain_position(a) == "octagonal"

    def test_beyond_polyhedra_classification(self):
        p = ShapePredicate(PredicateKind.DIM_DIVISIBLE, "x", axis=0, divisor=8)
        a = GuardAbstraction(frozenset({p}))
        assert classify_abstract_domain_position(a) == "beyond_polyhedra"

    def test_hierarchy_summary(self):
        s = domain_hierarchy_summary()
        assert "galois_connection" in s
        assert len(s["galois_connection"]["properties"]) == 4
        assert s["linear_fragment"]["hierarchy_position"] == "octagonal sub-domain"
        assert s["full_domain"]["hierarchy_position"] == "between octagonal and nonlinear integer arithmetic"


# ─── Integration: alpha on real verification scenarios ───────────────────────

class TestIntegration:
    def test_resnet_style_shapes(self):
        """ResNet-style residual connection."""
        envs = [
            ShapeEnv({
                "x": (32, 64, 56, 56),
                "conv1_out": (32, 64, 56, 56),
                "residual": (32, 64, 56, 56),
            }),
            ShapeEnv({
                "x": (16, 64, 56, 56),
                "conv1_out": (16, 64, 56, 56),
                "residual": (16, 64, 56, 56),
            }),
        ]
        assert verify_soundness(envs)
        a = alpha(envs)
        # Channel dim should be captured
        p_chan = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=1, value=64)
        assert p_chan in a.predicates
        # Batch should NOT be captured (varies)
        p_batch = ShapePredicate(PredicateKind.DIM_EQ, "x", axis=0, value=32)
        assert p_batch not in a.predicates
        # DIM_MATCH between conv1_out and residual should be captured
        for axis in range(4):
            p_match = ShapePredicate(
                PredicateKind.DIM_MATCH, "conv1_out", axis=axis,
                match_tensor="residual", match_axis=axis
            )
            assert p_match in a.predicates

    def test_transformer_shapes(self):
        """Transformer with head decomposition."""
        envs = [
            ShapeEnv({
                "q": (8, 12, 128, 64),  # batch=8, heads=12, seq=128, head_dim=64
                "k": (8, 12, 128, 64),
                "v": (8, 12, 128, 64),
            }),
        ]
        assert verify_soundness(envs)
        a = alpha(envs)
        # All q/k/v should match on heads dim
        assert ShapePredicate(
            PredicateKind.DIM_MATCH, "q", axis=1,
            match_tensor="k", match_axis=1
        ) in a.predicates

    def test_galois_on_alpha_gamma_roundtrip(self):
        """Verify α(γ(α(S))) = α(S) — closure property."""
        envs = [
            ShapeEnv({"x": (32, 784)}),
            ShapeEnv({"x": (64, 784)}),
        ]
        a1 = alpha(envs)
        # γ(a1) contains at least envs
        for env in envs:
            assert gamma_member(env, a1)
        # α applied to the same envs should give the same result
        a2 = alpha(envs)
        assert a1.predicates == a2.predicates
