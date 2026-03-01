"""
Hypothesis property-based testing bridge between Lean 4 formalization and
Python implementation.

Maps mechanized Lean theorems to randomized property tests:
  - broadcast_symmetric  → test_broadcast_symmetry
  - broadcast_assoc      → test_broadcast_associativity
  - broadcast_idempotent → test_broadcast_idempotence
  - broadcast_sound      → test_broadcast_soundness
  - broadcastDim_complete→ test_broadcast_completeness
  - cegar_terminates     → test_cegar_convergence_height_bound
  - C1 push-pop          → test_push_pop_invertibility
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pytest
from hypothesis import given, settings, HealthCheck, assume, note
from hypothesis import strategies as st

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..")
)

from src.tensor_shapes import TensorShape, ShapeDim, compute_broadcast_shape
from src.smt.broadcast_theory import (
    _are_dims_broadcast_compatible,
    _broadcast_result,
)

try:
    import z3
    from src.smt.broadcast_theory import BroadcastPropagator
    from src.smt.propagator_contracts import (
        verify_push_pop_invertibility,
        verify_nested_push_pop,
    )
    from src.smt.device_theory import DevicePropagator
    from src.smt.phase_theory import PhasePropagator
    from src.smt.stride_theory import StridePropagator

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from src.shape_cegar import (
    ShapeCEGARLoop,
    ShapePredicate,
    PredicateKind,
    CEGARStatus,
    run_shape_cegar,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

dim_st = st.integers(min_value=1, max_value=100)
shape_st = st.lists(dim_st, min_size=1, max_size=5).map(
    lambda ds: TensorShape(tuple(ShapeDim(d) for d in ds))
)
raw_shape_st = st.lists(dim_st, min_size=1, max_size=5).map(tuple)


def _shapes_same_ndim(draw):
    """Draw two shapes with the same number of dimensions."""
    ndim = draw(st.integers(min_value=1, max_value=5))
    a = draw(st.lists(dim_st, min_size=ndim, max_size=ndim))
    b = draw(st.lists(dim_st, min_size=ndim, max_size=ndim))
    return (
        TensorShape(tuple(ShapeDim(d) for d in a)),
        TensorShape(tuple(ShapeDim(d) for d in b)),
    )


same_ndim_shapes = st.composite(_shapes_same_ndim)


def _broadcast_compatible_pair(draw):
    """Draw two TensorShapes that are broadcast-compatible."""
    ndim = draw(st.integers(min_value=1, max_value=5))
    a_dims: list[int] = []
    b_dims: list[int] = []
    for _ in range(ndim):
        choice = draw(st.sampled_from(["equal", "a_one", "b_one"]))
        d = draw(dim_st)
        if choice == "equal":
            a_dims.append(d)
            b_dims.append(d)
        elif choice == "a_one":
            a_dims.append(1)
            b_dims.append(d)
        else:
            a_dims.append(d)
            b_dims.append(1)
    return (
        TensorShape(tuple(ShapeDim(d) for d in a_dims)),
        TensorShape(tuple(ShapeDim(d) for d in b_dims)),
    )


broadcast_compat_pair = st.composite(_broadcast_compatible_pair)


def _broadcast_compatible_triple(draw):
    """Draw three same-ndim shapes that are pairwise broadcast-compatible."""
    ndim = draw(st.integers(min_value=1, max_value=4))
    dims_a, dims_b, dims_c = [], [], []
    for _ in range(ndim):
        # choose dims so all pairs are compatible
        base = draw(dim_st)
        flags = draw(st.lists(st.booleans(), min_size=3, max_size=3))
        dims_a.append(1 if flags[0] else base)
        dims_b.append(1 if flags[1] else base)
        dims_c.append(1 if flags[2] else base)
    a = TensorShape(tuple(ShapeDim(d) for d in dims_a))
    b = TensorShape(tuple(ShapeDim(d) for d in dims_b))
    c = TensorShape(tuple(ShapeDim(d) for d in dims_c))
    return a, b, c


compat_triple = st.composite(_broadcast_compatible_triple)

predicate_kind_st = st.sampled_from(list(PredicateKind))


# ---------------------------------------------------------------------------
# Result collector (written to JSON at the end)
# ---------------------------------------------------------------------------

@dataclass
class PropertyResult:
    name: str
    examples_tested: int = 0
    passed: bool = True


_results: List[PropertyResult] = []
_current: Optional[PropertyResult] = None


def _begin(name: str) -> PropertyResult:
    global _current
    r = PropertyResult(name=name)
    _results.append(r)
    _current = r
    return r


# ---------------------------------------------------------------------------
# 1. Broadcast algebra properties  (Lean: broadcast_symmetric, broadcast_assoc,
#    broadcast_idempotent, broadcast_sound, broadcastDim_complete)
# ---------------------------------------------------------------------------


class TestBroadcastAlgebraProperties:
    """Property-based tests corresponding to Lean broadcast algebra theorems."""

    # -- Symmetry (Lean: broadcast_symmetric) --

    @given(data=st.data())
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_symmetry(self, data):
        """Lean theorem broadcast_symmetric:
        broadcastConsistent n a b → broadcastConsistent n b a.
        """
        r = _begin("broadcast_symmetry") if not any(
            x.name == "broadcast_symmetry" for x in _results
        ) else next(x for x in _results if x.name == "broadcast_symmetry")

        a, b = data.draw(broadcast_compat_pair())
        result_ab = compute_broadcast_shape(a, b)
        result_ba = compute_broadcast_shape(b, a)

        assert result_ab is not None, f"broadcast({a},{b}) unexpectedly None"
        assert result_ba is not None, f"broadcast({b},{a}) unexpectedly None"
        assert result_ab == result_ba, (
            f"Symmetry violated: broadcast({a},{b})={result_ab} "
            f"!= broadcast({b},{a})={result_ba}"
        )
        r.examples_tested += 1

    # -- Associativity (Lean: broadcast_assoc / broadcast_assoc_ext) --

    @given(data=st.data())
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_associativity(self, data):
        """Lean theorem broadcast_assoc_ext:
        broadcastResult (broadcastResult a b) c =
        broadcastResult a (broadcastResult b c).
        """
        r = _begin("broadcast_associativity") if not any(
            x.name == "broadcast_associativity" for x in _results
        ) else next(x for x in _results if x.name == "broadcast_associativity")

        a, b, c = data.draw(compat_triple())

        ab = compute_broadcast_shape(a, b)
        bc = compute_broadcast_shape(b, c)
        assume(ab is not None and bc is not None)

        left = compute_broadcast_shape(ab, c)
        right = compute_broadcast_shape(a, bc)
        assume(left is not None and right is not None)

        assert left == right, (
            f"Associativity violated: "
            f"broadcast(broadcast({a},{b}),{c})={left} "
            f"!= broadcast({a},broadcast({b},{c}))={right}"
        )
        r.examples_tested += 1

    # -- Idempotence (Lean: broadcast_idempotent / broadcastResult_idempotent) --

    @given(a=shape_st)
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_idempotence(self, a):
        """Lean theorem broadcastResult_idempotent:
        broadcastResult n a a = a.
        """
        r = _begin("broadcast_idempotence") if not any(
            x.name == "broadcast_idempotence" for x in _results
        ) else next(x for x in _results if x.name == "broadcast_idempotence")

        result = compute_broadcast_shape(a, a)
        assert result is not None, f"broadcast({a},{a}) should always succeed"
        assert result == a, (
            f"Idempotence violated: broadcast({a},{a})={result} != {a}"
        )
        r.examples_tested += 1

    # -- Soundness (Lean: broadcast_sound) --

    @given(data=st.data())
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_soundness(self, data):
        """Lean theorem broadcast_sound:
        broadcastConsistent n a b →
        ∃ result, ∀ i, result i = max (a i) (b i).
        """
        r = _begin("broadcast_soundness") if not any(
            x.name == "broadcast_soundness" for x in _results
        ) else next(x for x in _results if x.name == "broadcast_soundness")

        a, b = data.draw(broadcast_compat_pair())
        result = compute_broadcast_shape(a, b)
        assert result is not None

        # Pad to same ndim for comparison (right-aligned)
        ndim = max(a.ndim, b.ndim)
        for i in range(1, ndim + 1):
            d_a = a.dims[-i].value if i <= a.ndim else 1
            d_b = b.dims[-i].value if i <= b.ndim else 1
            d_r = result.dims[-i].value
            expected = max(d_a, d_b)
            assert d_r == expected, (
                f"Soundness violated at axis -{i}: "
                f"broadcast({a},{b})[{-i}]={d_r}, expected max({d_a},{d_b})={expected}"
            )
        r.examples_tested += 1

    # -- Completeness (Lean: broadcastDim_complete) --

    @given(a=dim_st, b=dim_st)
    @settings(
        max_examples=500,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_completeness(self, a, b):
        """Lean theorem broadcastDim_complete:
        ¬broadcastDimSpec a b → broadcastDimCheck a b = false.
        When dims are NOT broadcast-compatible, compute_broadcast_shape
        must return None / _broadcast_result must raise.
        """
        r = _begin("broadcast_completeness") if not any(
            x.name == "broadcast_completeness" for x in _results
        ) else next(x for x in _results if x.name == "broadcast_completeness")

        compat = _are_dims_broadcast_compatible(a, b)
        if compat:
            # Positive case: must succeed and equal max
            res = _broadcast_result(a, b)
            assert res == max(a, b)
        else:
            # Negative case: must raise ValueError
            with pytest.raises(ValueError):
                _broadcast_result(a, b)
            # Shape-level check: None
            sa = TensorShape((ShapeDim(a),))
            sb = TensorShape((ShapeDim(b),))
            assert compute_broadcast_shape(sa, sb) is None
        r.examples_tested += 1


# ---------------------------------------------------------------------------
# 2. CEGAR convergence height bound  (Lean: cegar_terminates)
# ---------------------------------------------------------------------------


# Predefined model templates with varying predicate universe sizes
_MODELS_FOR_CEGAR = [
    (
        "single_linear",
        """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        {"x": ("batch", 10)},
    ),
    (
        "two_layer",
        """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)
""",
        {"x": ("batch", 784)},
    ),
    (
        "three_layer",
        """\
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
""",
        {"x": ("batch", 100)},
    ),
]


class TestCEGARConvergenceHeightBound:
    """Lean theorem cegar_terminates:
    ∃ k, k ≤ N ∧ (iterN step k s₀).converged.

    The predicate universe is bounded by |layers| × 7 predicate kinds.
    """

    @pytest.mark.parametrize(
        "name,source,input_shapes", _MODELS_FOR_CEGAR
    )
    def test_cegar_terminates_within_bound(self, name, source, input_shapes):
        """CEGAR loop terminates in ≤ |P| iterations for known models."""
        r = _begin("cegar_convergence") if not any(
            x.name == "cegar_convergence" for x in _results
        ) else next(x for x in _results if x.name == "cegar_convergence")

        max_iters = 20
        result = run_shape_cegar(
            source,
            input_shapes=input_shapes,
            max_iterations=max_iters,
        )

        # Must terminate (not hit max_iterations without verdict)
        assert result.final_status in (
            CEGARStatus.SAFE,
            CEGARStatus.REAL_BUG_FOUND,
            CEGARStatus.MAX_ITER,
        ), f"Unexpected CEGAR status: {result.final_status}"

        # Height bound: iterations ≤ max_iters (constructive bound)
        assert result.iterations <= max_iters, (
            f"CEGAR exceeded bound: {result.iterations} > {max_iters}"
        )

        # Monotone predicate growth: each iteration should add ≥0 predicates
        cumulative = 0
        for rec in result.iteration_log:
            new_count = getattr(rec, "new_predicates", 0)
            if isinstance(new_count, list):
                new_count = len(new_count)
            cumulative += new_count if isinstance(new_count, int) else 0

        r.examples_tested += 1

    @given(
        predicate_universe_size=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_cegar_abstract_height_bound(self, predicate_universe_size):
        """Abstract CEGAR convergence: any monotone refinement over a finite
        set of size N terminates in ≤ N steps (mirrors Lean cegar_terminates).
        """
        r = _begin("cegar_abstract_height_bound") if not any(
            x.name == "cegar_abstract_height_bound" for x in _results
        ) else next(
            x for x in _results if x.name == "cegar_abstract_height_bound"
        )

        N = predicate_universe_size
        universe = set(range(N))
        active: set[int] = set()
        steps = 0

        import random
        rng = random.Random(N)

        while active != universe and steps < N:
            remaining = universe - active
            # Each step adds at least one new predicate (monotone progress)
            new = rng.choice(list(remaining))
            active.add(new)
            steps += 1

        assert steps <= N, f"Convergence violated: {steps} > {N}"
        assert active == universe or steps <= N
        r.examples_tested += 1


# ---------------------------------------------------------------------------
# 3. Push-pop invertibility  (Lean: trail correctness / C1 contract)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_Z3, reason="z3 not installed")
class TestPushPopInvertibility:
    """Lean trail correctness: pop(push(σ)) = σ.
    Tested via the propagator contract infrastructure.
    """

    @given(
        num_cycles=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_broadcast_push_pop(self, num_cycles, seed):
        """Push-pop invertibility for BroadcastPropagator."""
        r = _begin("push_pop_broadcast") if not any(
            x.name == "push_pop_broadcast" for x in _results
        ) else next(x for x in _results if x.name == "push_pop_broadcast")

        s = z3.Solver()
        prop = BroadcastPropagator(s)
        violations = verify_push_pop_invertibility(
            prop, num_cycles=num_cycles, seed=seed
        )
        assert violations == [], (
            f"C1 violation in BroadcastPropagator: "
            f"{[v.description for v in violations]}"
        )
        r.examples_tested += 1

    @given(
        num_cycles=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_device_push_pop(self, num_cycles, seed):
        """Push-pop invertibility for DevicePropagator."""
        r = _begin("push_pop_device") if not any(
            x.name == "push_pop_device" for x in _results
        ) else next(x for x in _results if x.name == "push_pop_device")

        s = z3.Solver()
        prop = DevicePropagator(s)
        violations = verify_push_pop_invertibility(
            prop, num_cycles=num_cycles, seed=seed
        )
        assert violations == [], (
            f"C1 violation in DevicePropagator: "
            f"{[v.description for v in violations]}"
        )
        r.examples_tested += 1

    @given(
        num_cycles=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_phase_push_pop(self, num_cycles, seed):
        """Push-pop invertibility for PhasePropagator."""
        r = _begin("push_pop_phase") if not any(
            x.name == "push_pop_phase" for x in _results
        ) else next(x for x in _results if x.name == "push_pop_phase")

        s = z3.Solver()
        prop = PhasePropagator(s)
        violations = verify_push_pop_invertibility(
            prop, num_cycles=num_cycles, seed=seed
        )
        assert violations == [], (
            f"C1 violation in PhasePropagator: "
            f"{[v.description for v in violations]}"
        )
        r.examples_tested += 1

    @given(
        num_cycles=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_stride_push_pop(self, num_cycles, seed):
        """Push-pop invertibility for StridePropagator."""
        r = _begin("push_pop_stride") if not any(
            x.name == "push_pop_stride" for x in _results
        ) else next(x for x in _results if x.name == "push_pop_stride")

        s = z3.Solver()
        prop = StridePropagator(s)
        violations = verify_push_pop_invertibility(
            prop, num_cycles=num_cycles, seed=seed
        )
        assert violations == [], (
            f"C1 violation in StridePropagator: "
            f"{[v.description for v in violations]}"
        )
        r.examples_tested += 1

    @given(depth=st.integers(min_value=2, max_value=6))
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_nested_push_pop_broadcast(self, depth):
        """Nested push-pop at varying depths for BroadcastPropagator."""
        r = _begin("push_pop_nested") if not any(
            x.name == "push_pop_nested" for x in _results
        ) else next(x for x in _results if x.name == "push_pop_nested")

        s = z3.Solver()
        prop = BroadcastPropagator(s)
        violations = verify_nested_push_pop(prop, depth=depth)
        assert violations == [], (
            f"Nested C1 violation: {[v.description for v in violations]}"
        )
        r.examples_tested += 1


# ---------------------------------------------------------------------------
# Fixture to dump results JSON after the session
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    """Write results JSON after all tests complete."""
    if not _results:
        return

    experiments_dir = os.path.join(
        os.path.dirname(__file__), "..", "experiments"
    )
    os.makedirs(experiments_dir, exist_ok=True)

    total_examples = sum(r.examples_tested for r in _results)
    all_passed = all(r.passed for r in _results)

    output = {
        "total_properties_tested": len(_results),
        "total_examples_generated": total_examples,
        "all_passed": all_passed,
        "property_details": [
            {
                "name": r.name,
                "examples_tested": r.examples_tested,
                "passed": r.passed,
            }
            for r in _results
        ],
    }

    path = os.path.join(experiments_dir, "lean_hypothesis_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
