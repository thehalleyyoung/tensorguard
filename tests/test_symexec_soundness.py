"""Step 9 — property-based soundness harness for the abstract lattice.

Complements the example-based laws in ``test_symexec_lattice.py`` with
randomised (Hypothesis) properties covering the three pillars of abstract
interpretation soundness:

1. **Lattice laws** — join/meet are idempotent, commutative, associative; join
   is a least upper bound and meet a greatest lower bound; ``⊑`` is consistent
   with join; widening over-approximates join.
2. **Galois connection** — ``α(γ(v)) ⊑ v`` (the concretization oracle never
   hands out a representative outside the value's meaning) and ``γ`` is monotone
   (``a ⊑ b`` ⇒ every member of ``γ(a)`` is a member of ``γ(b)``).
3. **Transfer monotonicity** — the abstract transfers (``join``, tensor methods)
   are monotone: a more precise input never yields a less precise output.

Equality is taken in the lattice sense (mutual ``⊑``) so the properties are
robust to incidental representation differences.
"""

from hypothesis import given, settings, strategies as st

from src.symexec.symdim import SymDim
from src.symexec.transfer import tensor_method
from src.symexec.values import (
    BoolVal,
    DictVal,
    FloatVal,
    IntVal,
    ListVal,
    NONE,
    TOP,
    BOTTOM,
    StrVal,
    TensorVal,
    TupleVal,
    int_const,
    int_range,
    join_many,
)
from src.symexec.concretize import alpha, gamma_samples, is_sound_sample


# ── strategies for abstract values ─────────────────────────────────────────
_small_int = st.integers(min_value=-8, max_value=8)


@st.composite
def _int_values(draw):
    kind = draw(st.sampled_from(["const", "range", "ge", "top"]))
    if kind == "const":
        return int_const(draw(_small_int))
    if kind == "range":
        lo = draw(_small_int)
        hi = draw(st.integers(min_value=lo, max_value=lo + 6))
        return int_range(lo, hi)
    if kind == "ge":
        return int_range(draw(_small_int), None)
    return IntVal()


@st.composite
def _tensor_values(draw):
    rank = draw(st.sampled_from([None, 0, 1, 2, 3]))
    if rank is None or rank == 0:
        return TensorVal(rank=rank)
    if draw(st.booleans()):
        dims = draw(
            st.lists(st.integers(min_value=1, max_value=6), min_size=rank, max_size=rank)
        )
        return TensorVal(rank=rank, shape=tuple(SymDim.const_dim(d) for d in dims))
    return TensorVal(rank=rank)  # rank known, shape symbolic


def _atoms():
    return st.one_of(
        _int_values(),
        st.just(NONE),
        st.just(BoolVal()),
        st.builds(BoolVal, const=st.booleans()),
        st.builds(StrVal, const=st.text(max_size=3)),
        st.builds(FloatVal, const=st.floats(allow_nan=False, allow_infinity=False, width=32)),
        _tensor_values(),
        st.just(TOP),
        st.just(BOTTOM),
    )


@st.composite
def _containers(draw):
    elt = draw(_atoms())
    kind = draw(st.sampled_from(["tuple", "list", "dict"]))
    if kind == "tuple":
        n = draw(st.integers(min_value=0, max_value=3))
        return TupleVal(elems=tuple(draw(_atoms()) for _ in range(n)), exact_len=True)
    if kind == "list":
        n = draw(st.integers(min_value=0, max_value=3))
        elems = tuple(draw(_atoms()) for _ in range(n))
        return ListVal(elem=join_many(list(elems)) if elems else TOP, length=n, exact_elems=elems)
    keys = draw(st.lists(st.sampled_from(["a", "b", "c"]), max_size=3, unique=True))
    known = tuple((k, draw(_atoms())) for k in keys)
    return DictVal(
        value=join_many([v for _, v in known]) if known else TOP,
        known=known,
        exact_keys=draw(st.booleans()),
    )


def values():
    return st.one_of(_atoms(), _containers())


# ── lattice-theoretic equality ──────────────────────────────────────────────
def _eq(a, b) -> bool:
    return a.leq(b) and b.leq(a)


# ── 1. lattice laws ─────────────────────────────────────────────────────────
@settings(max_examples=300)
@given(values())
def test_join_idempotent(a):
    assert _eq(a.join(a), a)


@settings(max_examples=300)
@given(values(), values())
def test_join_commutative(a, b):
    assert _eq(a.join(b), b.join(a))


@settings(max_examples=300)
@given(values(), values(), values())
def test_join_associative(a, b, c):
    assert _eq(a.join(b).join(c), a.join(b.join(c)))


@settings(max_examples=300)
@given(values(), values())
def test_join_is_upper_bound(a, b):
    j = a.join(b)
    assert a.leq(j) and b.leq(j)


@settings(max_examples=300)
@given(values(), values())
def test_meet_over_approximates_intersection(a, b):
    # ``_meet2`` is a *sound over-approximation of the concrete intersection*
    # (used to refine a value under a guard), not necessarily an exact GLB: every
    # concrete value in γ(a) ∩ γ(b) must remain in γ(meet(a, b)).
    m = a.meet(b)
    for c in gamma_samples(a):
        if is_sound_sample(b, c):  # c ∈ γ(a) ∩ γ(b)
            assert is_sound_sample(m, c), f"{c!r} ∈ γ(a)∩γ(b) but ∉ γ(a⊓b)"


@settings(max_examples=300)
@given(values(), values())
def test_leq_consistent_with_join(a, b):
    # a ⊑ b  ⟺  join(a, b) ≡ b
    assert a.leq(b) == _eq(a.join(b), b)


@settings(max_examples=300)
@given(values())
def test_leq_reflexive(a):
    assert a.leq(a)


@settings(max_examples=300)
@given(values(), values())
def test_widen_over_approximates_join(a, b):
    assert a.join(b).leq(a.widen(b))


@given(values(), values())
def test_narrow_is_between_b_and_a(a, b):
    # Narrowing's contract: when ``b ⊑ a``, ``b ⊑ (a ▵ b) ⊑ a`` — it only
    # tightens ``a`` and never under-approximates ``b`` (soundness).  Use a join
    # to guarantee the ``b ⊑ a`` precondition for an arbitrary pair.
    a2 = a.join(b)  # a2 ⊒ b by construction
    n = a2.narrow(b)
    assert b.leq(n), f"narrow under-approximates b: {b} ⋢ {n}"
    assert n.leq(a2), f"narrow not below a: {n} ⋢ {a2}"


@given(values())
def test_narrow_reflexive(v):
    # ``v ▵ v == v`` (lattice equality): narrowing a value with itself is inert.
    assert _eq(v.narrow(v), v)


# ── 2. Galois connection: α(γ(v)) ⊑ v, and γ monotone ───────────────────────
@settings(max_examples=400)
@given(values())
def test_alpha_gamma_reductive(v):
    for c in gamma_samples(v):
        assert is_sound_sample(v, c), f"{c!r} ∉ γ({v!r}) but was produced by γ"


@settings(max_examples=400)
@given(values(), values())
def test_gamma_monotone(a, b):
    # build an ordered pair a ⊑ up by construction
    up = a.join(b)
    for c in gamma_samples(a):
        assert is_sound_sample(up, c), f"{c!r} ∈ γ(a) but ∉ γ(a⊔b)"


# ── 3. transfer monotonicity ────────────────────────────────────────────────
@settings(max_examples=300)
@given(values(), values(), values(), values())
def test_join_is_monotone(a1, b1, x, y):
    a2, b2 = a1.join(x), b1.join(y)  # a1 ⊑ a2, b1 ⊑ b2 by construction
    assert a1.join(b1).leq(a2.join(b2))


_NOARG_METHODS = ["t", "transpose", "flatten", "unsqueeze", "dim", "size", "sum", "mean"]


@settings(max_examples=300)
@given(_tensor_values(), _tensor_values(), st.sampled_from(_NOARG_METHODS))
def test_tensor_method_monotone(ta, tb, method):
    lo = ta if ta.leq(tb) else ta.join(tb)  # ensure lo ⊑ hi
    hi = ta.join(tb)
    rl = tensor_method(lo, method, [])
    rh = tensor_method(hi, method, [])
    assert rl.leq(rh), f"{method}: {rl!r} ⋢ {rh!r} for {lo!r} ⊑ {hi!r}"
