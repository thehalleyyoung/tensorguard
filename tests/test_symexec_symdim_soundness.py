"""Differential bridge: the affine ``SymDim`` oracle obeys the soundness
properties that are machine-checked in
``lean/TensorGuard/Symexec/Affine.lean``.

The Lean module proves, over the obvious concrete semantics (evaluate an affine
form ``const + Σ cᵢ·vᵢ`` under an arbitrary integer assignment to its variables):

  * ``definitely_eq_sound``        — if ``a.definitely_eq(b)`` then ``a`` and ``b``
                                     are equal under *every* assignment;
  * ``definitely_divisible_sound`` — if ``a.definitely_divisible_by(k)`` then ``k``
                                     divides ``a``'s value under *every*
                                     assignment;
  * ``eval_add`` / ``eval_smul``   — the ``+`` and ``·constant`` transfer
                                     functions are exact homomorphisms.

These tests pin the *Python* implementation to that proven model by brute-forcing
many concrete assignments, so the implementation cannot silently drift away from
the property the Lean kernel has certified.  (The Lean proof is the universal
guarantee; this is the empirical regression guard that the code still matches the
modeled algebra.)
"""

from __future__ import annotations

import itertools

from src.symexec.symdim import SymDim


def _evaluate(d: SymDim, env: dict) -> int:
    """Concrete semantics of an affine form under an integer assignment."""
    return d.const + sum(coeff * env[name] for name, coeff in d.terms)


# A small fixed library of symbolic dimensions covering constants, single vars,
# and genuinely affine forms (matching what flows through real shape reasoning).
def _b():
    return SymDim.var("b")


def _s():
    return SymDim.var("s")


_VARS = ["b", "s", "h"]
_ENVS = [
    dict(zip(_VARS, combo))
    for combo in itertools.product(range(-3, 6), repeat=len(_VARS))
]


def _forms():
    b, s, h = SymDim.var("b"), SymDim.var("s"), SymDim.var("h")
    return [
        SymDim.const_dim(0),
        SymDim.const_dim(12),
        b,
        s,
        b + s,
        b * 4,
        b * 4 + s * 2,
        b + 3,
        (b + s) - s,          # normalizes to b
        b * 6 + 12,
        h * 2 + b * 2,
        s - s,                # normalizes to const 0
    ]


def test_definitely_eq_is_sound():
    """Whenever the oracle certifies equality, the forms agree on every env."""
    forms = _forms()
    certified = 0
    for a, b in itertools.product(forms, forms):
        if a.definitely_eq(b):
            certified += 1
            for env in _ENVS:
                assert _evaluate(a, env) == _evaluate(b, env), (
                    f"definitely_eq({a}, {b}) is True but they differ at {env}"
                )
    # Non-vacuity: the oracle actually fires on some non-trivial pairs.
    assert certified >= len(forms), "definitely_eq never certified equality"


def test_definitely_eq_certifies_true_equalities():
    """The oracle is also relatively *complete*: structurally-equal affine forms
    (e.g. ``(b+s)-s`` vs ``b``) are certified, not just reflexive identities."""
    b, s = SymDim.var("b"), SymDim.var("s")
    assert ((b + s) - s).definitely_eq(b)
    assert (s - s).definitely_eq(SymDim.const_dim(0))
    assert (b * 2 + b).definitely_eq(b * 3)


def test_definitely_divisible_by_is_sound():
    """Whenever the oracle certifies divisibility, every concrete value of the
    form is divisible by ``k``."""
    forms = _forms()
    certified = 0
    for d in forms:
        for k in (2, 3, 4, 6, 12):
            if d.definitely_divisible_by(k) is True:
                certified += 1
                for env in _ENVS:
                    val = _evaluate(d, env)
                    assert val % k == 0, (
                        f"definitely_divisible_by: {d} certified div by {k} "
                        f"but value {val} at {env} is not"
                    )
    assert certified > 0, "divisibility oracle never certified anything"


def test_divisibility_abstains_when_not_provable():
    """Soundness's dual: the oracle returns None (abstain), never a false True,
    when divisibility is not guaranteed for all assignments."""
    b = SymDim.var("b")
    # b is divisible by 2 only for even b -> must NOT be certified True.
    assert b.definitely_divisible_by(2) is not True
    # b*2 + 1 is never divisible by 2 -> oracle may say False or None, never True.
    assert (b * 2 + SymDim.const_dim(1)).definitely_divisible_by(2) is not True


def test_add_transfer_is_exact():
    """``+`` is an exact homomorphism: eval(a+b) == eval(a)+eval(b)."""
    forms = _forms()
    for a, b in itertools.product(forms, forms):
        c = a + b
        for env in _ENVS:
            assert _evaluate(c, env) == _evaluate(a, env) + _evaluate(b, env)


def test_scalar_mul_transfer_is_exact():
    """``·constant`` is an exact homomorphism: eval(k*a) == k*eval(a)."""
    forms = _forms()
    for a in forms:
        for k in (-2, 0, 1, 3, 7):
            c = a * k
            for env in _ENVS:
                assert _evaluate(c, env) == k * _evaluate(a, env)
