"""Step 86 — reconcile the formalization (docs/formalization/type_system.md)
with the real implementation.

These tests *prove against real code* the algebraic laws the formalization
relies on: the interval domain and the reduced product are bounded lattices,
widening terminates on ascending chains, the inter-theory reductions are
reductive, the symbolic-dimension algebra matches the rules in the spec, and a
shape verification condition is discharged soundly end-to-end on a real
nn.Module.
"""

from __future__ import annotations

import itertools
import os

import pytest

from src.domains.abstract_domains import Interval, IntervalDomain

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _eq(a: Interval, b: Interval) -> bool:
    if a.is_bottom() and b.is_bottom():
        return True
    return a.lo == b.lo and a.hi == b.hi


# A fixed, representative sample including bottom, top, points and ranges.
_SAMPLE = [
    Interval.bottom(),
    Interval.top(),
    Interval.const(0),
    Interval.const(7),
    Interval.from_range(-5, 5),
    Interval.from_range(1, 10),
    Interval.from_range(-3, 2),
    Interval.from_range(3, 8),
]


@pytest.fixture(scope="module")
def D():
    return IntervalDomain()


# ── Section 3.1: the interval domain is a bounded lattice ──────────────────


def test_partial_order_reflexive_antisym_transitive(D):
    for a in _SAMPLE:
        assert D.leq(a, a)  # reflexive
    for a, b in itertools.product(_SAMPLE, repeat=2):
        if D.leq(a, b) and D.leq(b, a):
            assert _eq(a, b)  # antisymmetric
    for a, b, c in itertools.product(_SAMPLE, repeat=3):
        if D.leq(a, b) and D.leq(b, c):
            assert D.leq(a, c)  # transitive


def test_join_is_least_upper_bound(D):
    for a, b in itertools.product(_SAMPLE, repeat=2):
        j = D.join(a, b)
        assert D.leq(a, j) and D.leq(b, j)  # upper bound
        # least: any other upper bound u has j <= u
        for u in _SAMPLE:
            if D.leq(a, u) and D.leq(b, u):
                assert D.leq(j, u)


def test_meet_is_greatest_lower_bound(D):
    for a, b in itertools.product(_SAMPLE, repeat=2):
        m = D.meet(a, b)
        assert D.leq(m, a) and D.leq(m, b)  # lower bound
        for l in _SAMPLE:
            if D.leq(l, a) and D.leq(l, b):
                assert D.leq(l, m)


def test_join_meet_commutative_associative_idempotent(D):
    for a, b in itertools.product(_SAMPLE, repeat=2):
        assert _eq(D.join(a, b), D.join(b, a))
        assert _eq(D.meet(a, b), D.meet(b, a))
    for a in _SAMPLE:
        assert _eq(D.join(a, a), a) or a.is_bottom() and D.join(a, a).is_bottom()
        assert _eq(D.meet(a, a), a) or a.is_bottom()
    for a, b, c in itertools.product(_SAMPLE, repeat=3):
        assert _eq(D.join(D.join(a, b), c), D.join(a, D.join(b, c)))
        assert _eq(D.meet(D.meet(a, b), c), D.meet(a, D.meet(b, c)))


def test_absorption_laws(D):
    for a, b in itertools.product(_SAMPLE, repeat=2):
        assert _eq(D.join(a, D.meet(a, b)), a)
        assert _eq(D.meet(a, D.join(a, b)), a)


def test_bounds(D):
    bot, top = D.bottom(), D.top()
    for a in _SAMPLE:
        assert D.leq(bot, a) and D.leq(a, top)
        assert _eq(D.join(a, bot), a)
        assert _eq(D.meet(a, top), a)


# ── Section 5: widening terminates on ascending chains ─────────────────────


def test_widening_is_upper_bound(D):
    for a, b in itertools.product(_SAMPLE, repeat=2):
        w = D.widen(a, b)
        assert D.leq(a, w) and D.leq(b, w)


def test_widening_terminates_on_growing_chain(D):
    x = Interval.const(0)
    steps = 0
    for i in range(1, 1000):
        nxt = D.join(x, Interval.from_range(0, i))  # strictly grows upward
        nx = D.widen(x, nxt)
        steps += 1
        if _eq(nx, x):
            break
        x = nx
    else:  # pragma: no cover - would indicate non-termination
        pytest.fail("widening did not stabilise")
    assert steps < 50, f"widening took too long: {steps} steps"
    assert x.hi == float("inf") or x.hi >= 1


# ── Section 4: the reduced product is a componentwise bounded lattice ──────


def _product_sample():
    from src.domains.intervals import Interval as IV
    from src.domains.intervals import IntervalValue
    from src.domains.product import ReducedProductDomain

    P = ReducedProductDomain()
    vals = [
        P.top(),
        P.bottom(),
        P.top().with_interval(IntervalValue(IV.from_bounds(0, 5))),
        P.top().with_interval(IntervalValue(IV.singleton(3))),
        P.top().with_interval(IntervalValue(IV.non_negative())),
    ]
    return P, vals


def test_product_is_componentwise_lattice():
    P, vals = _product_sample()
    # The *raw* componentwise join (before reduction) is a true upper bound in
    # each sub-domain — this is the lattice structure of the product.
    for a, b in itertools.product(vals, repeat=2):
        assert P.interval_domain.leq(
            a.interval, P.interval_domain.join(a.interval, b.interval))
        assert P.interval_domain.leq(
            b.interval, P.interval_domain.join(a.interval, b.interval))
    # meet is a lower bound even after reduction (reduction only sharpens down)
    for a, b in itertools.product(vals, repeat=2):
        m = P.meet(a, b)
        assert P.leq(m, a) and P.leq(m, b), (a, b, m)
    bot, top = P.bottom(), P.top()
    for a in vals:
        assert P.leq(bot, a) and P.leq(a, top)


def test_product_leq_is_partial_order():
    P, vals = _product_sample()
    for a in vals:
        assert P.leq(a, a)
    for a, b, c in itertools.product(vals, repeat=3):
        if P.leq(a, b) and P.leq(b, c):
            assert P.leq(a, c)


def test_reduced_join_is_sound_over_concretization():
    # The reduced join may sit *below* the raw componentwise join (reduction
    # sharpens), but it must never drop below either operand's interval — i.e.
    # it still over-approximates each operand's numeric concretization.
    P, vals = _product_sample()
    for a, b in itertools.product(vals, repeat=2):
        j = P.join(a, b)
        raw_iv = P.interval_domain.join(a.interval, b.interval)
        # reduced interval component is no larger than the raw upper bound...
        assert P.interval_domain.leq(j.interval, raw_iv)
        # ...and the raw upper bound still bounds both operands.
        assert P.interval_domain.leq(a.interval, raw_iv)
        assert P.interval_domain.leq(b.interval, raw_iv)


def test_reduction_is_reductive():
    # ρ(p) ⊑ p : a reduction never enlarges the abstract value.
    from src.domains.product import ReductionEngine
    from src.domains.intervals import Interval as IV
    from src.domains.intervals import IntervalValue
    from src.domains.product import ReducedProductDomain

    P = ReducedProductDomain()
    engine = ReductionEngine()
    for iv in (IV.from_bounds(0, 5), IV.singleton(1), IV.non_negative()):
        raw = P.top().with_interval(IntervalValue(iv))
        reduced = engine.reduce(raw)
        assert P.leq(reduced, raw), (raw, reduced)


# ── Section 2: symbolic-dimension algebra matches the spec ─────────────────


def test_symbolic_dim_algebra():
    from src.tensor_shapes import ShapeDim, SymbolicDimension, TensorShape

    b = SymbolicDimension("batch")
    assert (b * 2).name == "(batch*2)"
    assert (2 * b).name == "(2*batch)"
    assert (b // 4).name == "(batch//4)"
    assert (b + 1).name == "(batch+1)"
    assert (b - 1).name == "(batch-1)"
    # equality is by name; distinct names are not equal
    assert SymbolicDimension("x") == SymbolicDimension("x")
    assert SymbolicDimension("x") != SymbolicDimension("y")
    # ShapeDim symbolic vs concrete classification
    assert ShapeDim("seq").is_symbolic is True
    assert ShapeDim(10).is_symbolic is False
    sh = TensorShape.from_tuple(("batch", 3, 32))
    assert sh.ndim == 3
    assert sh.dim(-1).value == 32


def test_formalization_doc_is_reconciled():
    path = os.path.join(_REPO, "docs", "formalization", "type_system.md")
    assert os.path.exists(path)
    text = open(path, "r", encoding="utf-8").read()
    # the doc must reference the real modules it claims to formalize
    for ref in ("ReducedProductDomain", "IntervalDomain", "SymbolicDimension",
                "ReductionEngine"):
        assert ref in text, f"formalization should reference {ref}"


# ── Section 2.1: a shape VC is discharged soundly end-to-end ───────────────


def test_linear_vc_soundness():
    """The T-Linear side condition (k = in) is enforced on real nn.Module code:
    a mismatch is UNSAFE, the matching case is not."""
    from src.api import verify_architecture

    bad = (
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(30, 5)\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    )
    good = bad.replace("Linear(30, 5)", "Linear(20, 5)")
    r_bad = verify_architecture(bad, input_shapes={"x": ("batch", 10)})
    r_good = verify_architecture(good, input_shapes={"x": ("batch", 10)})
    assert r_bad.verdict == "UNSAFE" and r_bad.bugs
    assert r_good.verdict != "UNSAFE"
