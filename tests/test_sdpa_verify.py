"""Differential test: ``src.sdpa_verify`` vs real ``F.scaled_dot_product_attention``.

For each operand-shape case we run the *real* PyTorch SDPA op (default backend)
and compare raises-vs-succeeds and the success shape against the purely static
:func:`src.sdpa_verify.verify_sdpa`.  A randomised fuzzer drives hundreds of
extra cases (varying batch/head broadcasting, head dims, and mask shapes) so
the static model tracks real PyTorch semantics.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.sdpa_verify import verify_sdpa  # noqa: E402


def _real(q, k, v, mask=None, causal=False):
    tq = torch.randn(*q)
    tk = torch.randn(*k)
    tv = torch.randn(*v)
    tm = None
    if mask is not None:
        tm = torch.zeros(*mask, dtype=torch.float32)
    try:
        out = F.scaled_dot_product_attention(
            tq, tk, tv, attn_mask=tm, is_causal=causal
        )
        return "ok", tuple(out.shape)
    except Exception:
        return "err", None


def _check(q, k, v, mask=None, causal=False):
    real_status, real_shape = _real(q, k, v, mask, causal)
    verdict = verify_sdpa(q, k, v, attn_mask=mask, is_causal=causal)
    static_status = "ok" if verdict.ok else "err"
    assert static_status == real_status, (
        f"q={q} k={k} v={v} mask={mask}: real={real_status} "
        f"static={static_status} ({verdict.error})"
    )
    if real_status == "ok":
        assert tuple(verdict.output_shape) == real_shape, (
            f"q={q} k={k} v={v}: real={real_shape} static={verdict.output_shape}"
        )


VALID = [
    ((2, 4, 5, 8), (2, 4, 7, 8), (2, 4, 7, 8), None),
    ((2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 16), None),
    ((1, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 8), None),
    ((2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 5)),
    ((2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 8), (5, 5)),
    ((3, 5, 8), (3, 5, 8), (3, 5, 8), None),
]

INVALID = [
    ((2, 4, 5, 8), (2, 4, 5, 9), (2, 4, 5, 8), None),       # head dim mismatch
    ((2, 4, 5, 8), (2, 5, 5, 8), (2, 5, 5, 8), None),       # head count mismatch
    ((2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 8), (2, 4, 5, 7)),  # bad mask Lk
]


@pytest.mark.parametrize("q,k,v,mask", VALID)
def test_valid(q, k, v, mask):
    _check(q, k, v, mask)
    assert verify_sdpa(q, k, v, attn_mask=mask).ok


@pytest.mark.parametrize("q,k,v,mask", INVALID)
def test_invalid(q, k, v, mask):
    _check(q, k, v, mask)
    assert not verify_sdpa(q, k, v, attn_mask=mask).ok


def test_symbolic_not_refuted():
    v = verify_sdpa(("B", 4, "Lq", 8), ("B", 4, "Lk", 8), ("B", 4, "Lk", 16))
    assert v.ok
    assert v.output_shape == ("B", 4, "Lq", 16)


def test_fuzz_matches_real_torch():
    rng = random.Random(1234)
    checked = 0
    for _ in range(400):
        B = rng.choice([1, 2, 3])
        H = rng.choice([1, 2, 4])
        Lq = rng.randint(2, 6)
        Lk = rng.randint(2, 6)
        E = rng.choice([8, 16])
        Ev = rng.choice([8, 16])
        # randomly broadcast a leading dim to 1
        qb = 1 if rng.random() < 0.3 else B
        kh = H if rng.random() < 0.7 else rng.choice([1, 2, 4])
        q = (qb, H, Lq, E)
        k = (B, H, Lk, E if rng.random() < 0.8 else rng.choice([8, 16]))
        v = (B, kh, Lk, Ev)
        mask = None
        if rng.random() < 0.4:
            ml = rng.choice([Lk, rng.randint(2, 6)])
            mask = (B, H, Lq, ml)
        _check(q, k, v, mask)
        checked += 1
    assert checked == 400
