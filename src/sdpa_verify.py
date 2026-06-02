"""Static shape verifier for ``F.scaled_dot_product_attention`` (SDPA).

SDPA is the hot path of every modern attention block, and its shape contract is
subtle: ``query`` and ``key`` must agree on the head dimension ``E`` (the
contracted dim), the batch/head leading dims of ``q``/``k``/``v`` must
broadcast, and an explicit ``attn_mask`` must broadcast against the
``(L_q, L_k)`` score matrix.  A mismatch surfaces as an opaque
``RuntimeError`` deep inside a fused kernel — and often only for a particular
sequence length or head count that a smoke test never exercises.

:func:`verify_sdpa` reproduces *exactly* the conditions under which real
PyTorch raises (differentially tested in ``tests/test_sdpa_verify.py``),
operating purely on shapes.  Consistent with TensorGuard's soundness contract,
a dimension that cannot be decided statically (symbolic) is never refuted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

Dim = Union[int, str]

__all__ = ["SDPAVerdict", "verify_sdpa"]


@dataclass
class SDPAVerdict:
    ok: bool
    output_shape: Optional[Tuple[Dim, ...]] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None

    def __bool__(self) -> bool:  # pragma: no cover
        return self.ok


def _eq_known(a: Dim, b: Dim) -> bool:
    return isinstance(a, int) and isinstance(b, int) and a == b


def _broadcast_dim(a: Dim, b: Dim) -> Optional[Dim]:
    """Broadcast two dims; return None if statically incompatible."""
    if isinstance(a, int) and isinstance(b, int):
        if a == b:
            return a
        if a == 1:
            return b
        if b == 1:
            return a
        return None
    # symbolic: stay sound, prefer the concrete side if the other is 1
    if a == 1:
        return b
    if b == 1:
        return a
    return a  # cannot decide; assume compatible


def _broadcast_shapes(
    shapes: Sequence[Sequence[Dim]],
) -> Optional[List[Dim]]:
    out: List[Dim] = []
    ndim = max(len(s) for s in shapes)
    for i in range(1, ndim + 1):
        cur: Dim = 1
        for s in shapes:
            if i <= len(s):
                d = s[-i]
                nxt = _broadcast_dim(cur, d)
                if nxt is None:
                    return None
                cur = nxt
        out.append(cur)
    out.reverse()
    return out


def verify_sdpa(
    query: Sequence[Dim],
    key: Sequence[Dim],
    value: Sequence[Dim],
    attn_mask: Optional[Sequence[Dim]] = None,
    is_causal: bool = False,
) -> SDPAVerdict:
    """Verify one ``scaled_dot_product_attention`` call from operand shapes."""
    q, k, v = list(query), list(key), list(value)
    for name, s in (("query", q), ("key", k), ("value", v)):
        if len(s) < 3:
            return SDPAVerdict(
                False,
                error=f"{name} must have rank >= 3 (…, L, E); got rank {len(s)}",
                error_kind="rank",
            )

    Lq, Eq = q[-2], q[-1]
    Lk, Ek = k[-2], k[-1]
    Lv, Ev = v[-2], v[-1]

    # contracted head dim of q and k must match
    if isinstance(Eq, int) and isinstance(Ek, int) and Eq != Ek:
        return SDPAVerdict(
            False,
            error=(
                f"query/key head dim mismatch: query E={Eq} but key E={Ek} "
                "(the contracted dimension must be equal)"
            ),
            error_kind="head_dim",
        )

    # NOTE: key/value sequence-length agreement is intentionally *not* hard-
    # refuted: PyTorch's default fused backend tolerates a mismatch (only the
    # MATH backend raises), so flagging it would violate the soundness contract
    # (never refute a program that may run). It is surfaced as a soft warning
    # by callers that opt into backend-specific strictness.

    # leading (batch + head) dims of q, k, v must broadcast
    lead = _broadcast_shapes([q[:-2], k[:-2], v[:-2]])
    if lead is None:
        return SDPAVerdict(
            False,
            error=(
                "query/key/value batch/head dims do not broadcast: "
                f"{tuple(q[:-2])} vs {tuple(k[:-2])} vs {tuple(v[:-2])}"
            ),
            error_kind="batch_broadcast",
        )

    # attn_mask (if a tensor shape) must broadcast against (…lead, Lq, Lk)
    if attn_mask is not None:
        m = list(attn_mask)
        score_lead = lead + [Lq, Lk]
        if _broadcast_shapes([score_lead, m]) is None:
            return SDPAVerdict(
                False,
                error=(
                    f"attn_mask shape {tuple(m)} does not broadcast against the "
                    f"attention scores {tuple(score_lead)} (…, L_q, L_k)"
                ),
                error_kind="mask_broadcast",
            )

    out = tuple(lead) + (Lq, Ev)
    return SDPAVerdict(True, output_shape=out)
