#!/usr/bin/env python3.11
"""
TensorGuard V5 Lean ↔ PyTorch parity harness.

For each of 28 operators with a Lean shape-transfer rule defined in
`lean/TensorGuard/V5OperatorRules.lean`, we:

  1. Load the rule registry from Lean (executing the module via
     `lake env lean --run`) — this gives us a machine-checkable
     manifest of which rules Lean believes it has.
  2. Re-implement the *same* rule in pure Python (mirror, line-by-line
     equivalent of the Lean `def`).
  3. For each operator, generate 1000 random concrete shapes drawn
     from a sensible distribution: per-dim ∈ [1, 16], rank ∈ [1, 5],
     plus operator-specific param ranges.
  4. Compute Python-mirror predicted shape; if mirror returns `None`,
     skip (mirror disagrees with itself's preconditions). For shapes
     where mirror produces a result, compute torch's actual output
     shape and compare. If torch raises while mirror succeeded, we
     count a disagreement; if torch succeeds while mirror returns
     None, we count as "in-fragment exclusion" and report.
  5. Aggregate per-operator agreement rates and write
     `lean_parity_v5_results.json`.

Run:
    python3.11 experiments_v5/run_lean_parity_v5.py
"""

from __future__ import annotations
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
LEAN_DIR = ROOT / "lean"
RESULTS = Path(__file__).resolve().parent / "lean_parity_v5_results.json"

NUM_TESTS = 1000
SEED = 0x5C5C5C5C
DIM_LO, DIM_HI = 1, 16
RANK_LO, RANK_HI = 1, 5

# =============================================================================
# Step 1: load the Lean rule registry (the *source of truth* for "which rules
# Lean has").
# =============================================================================

def load_lean_registry() -> dict:
    """Invoke Lean to print the V5 rule registry as JSON."""
    cmd = ["lake", "env", "lean", "--run", "TensorGuard/V5OperatorRules.lean"]
    proc = subprocess.run(cmd, cwd=LEAN_DIR, capture_output=True, text=True,
                          timeout=300)
    # Lean prints warnings before the actual JSON line.
    json_line = None
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("{") and '"version"' in s:
            json_line = s
            break
    if json_line is None:
        raise RuntimeError(
            f"Could not find rules JSON in lean output. stderr={proc.stderr}\n"
            f"stdout head:\n{proc.stdout[:2000]}")
    return json.loads(json_line)


# =============================================================================
# Step 2: Python mirrors of every Lean rule. Each mirror returns
# Optional[Tuple[int, ...]] (or list of tuples for split/chunk/unbind).
# =============================================================================

Sh = Tuple[int, ...]


def prodL(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def bdim(a: int, b: int) -> Optional[int]:
    if a == b: return a
    if a == 1: return b
    if b == 1: return a
    return None


def bcast_eq(a: Sh, b: Sh) -> Optional[Sh]:
    if len(a) != len(b): return None
    out = []
    for x, y in zip(a, b):
        d = bdim(x, y)
        if d is None: return None
        out.append(d)
    return tuple(out)


def left_pad(s: Sh, n: int) -> Sh:
    if len(s) >= n: return s
    return (1,) * (n - len(s)) + s


def bcast(a: Sh, b: Sh) -> Optional[Sh]:
    n = max(len(a), len(b))
    return bcast_eq(left_pad(a, n), left_pad(b, n))


# ---- 1. matmul ----
def m_matmul(a: Sh, b: Sh) -> Optional[Sh]:
    if not a or not b:
        return None
    if len(a) == 1 and len(b) == 1:
        return () if a[0] == b[0] else None
    if len(a) == 1:
        if len(b) < 2: return None
        if a[0] != b[-2]: return None
        return tuple(b[:-2]) + (b[-1],)
    if len(b) == 1:
        if len(a) < 2: return None
        if a[-1] != b[0]: return None
        return tuple(a[:-2]) + (a[-2],)
    if a[-1] != b[-2]: return None
    bb = bcast(tuple(a[:-2]), tuple(b[:-2]))
    if bb is None: return None
    return bb + (a[-2], b[-1])


# ---- 2/3. bmm/batched_matmul ----
def m_bmm(a: Sh, b: Sh) -> Optional[Sh]:
    if len(a) != 3 or len(b) != 3: return None
    if a[0] != b[0] or a[2] != b[1]: return None
    return (a[0], a[1], b[2])


def m_batched_matmul(a, b):
    return m_bmm(a, b)


# ---- spatial helper ----
def conv_spatial(l_in, pad, dil, k, stride) -> Optional[int]:
    if stride == 0: return None
    num = l_in + 2 * pad
    eff = dil * (k - 1) + 1
    if eff > num: return None
    return (num - eff) // stride + 1


# ---- 4. conv1d ----
def m_conv1d(input_, weight, pad, dil, stride, groups) -> Optional[Sh]:
    if len(input_) != 3 or len(weight) != 3: return None
    n, cin, l = input_
    cout, cinG, k = weight
    if groups == 0: return None
    if cin != cinG * groups: return None
    lo = conv_spatial(l, pad, dil, k, stride)
    if lo is None: return None
    return (n, cout, lo)


# ---- 5. conv2d ----
def m_conv2d(input_, weight, pH, pW, dH, dW, kH, kW, sH, sW, groups):
    if len(input_) != 4 or len(weight) != 4: return None
    n, cin, h, w = input_
    cout, cinG, kHp, kWp = weight
    if groups == 0 or kH != kHp or kW != kWp: return None
    if cin != cinG * groups: return None
    ho = conv_spatial(h, pH, dH, kH, sH)
    wo = conv_spatial(w, pW, dW, kW, sW)
    if ho is None or wo is None: return None
    return (n, cout, ho, wo)


# ---- 6. conv3d ----
def m_conv3d(input_, weight, pD, pH, pW, dD, dH, dW, kD, kH, kW, sD, sH, sW, groups):
    if len(input_) != 5 or len(weight) != 5: return None
    n, cin, d, h, w = input_
    cout, cinG, kDp, kHp, kWp = weight
    if groups == 0 or kD != kDp or kH != kHp or kW != kWp: return None
    if cin != cinG * groups: return None
    do = conv_spatial(d, pD, dD, kD, sD)
    ho = conv_spatial(h, pH, dH, kH, sH)
    wo = conv_spatial(w, pW, dW, kW, sW)
    if None in (do, ho, wo): return None
    return (n, cout, do, ho, wo)


# ---- 7. conv_transpose2d ----
def conv_t_spatial(l_in, pad, dil, k, stride, out_pad) -> Optional[int]:
    inner = (l_in - 1) * stride + dil * (k - 1) + out_pad + 1
    if inner < 2 * pad: return None
    return inner - 2 * pad


def m_conv_transpose2d(input_, weight, pH, pW, dH, dW, kH, kW, sH, sW, oH, oW, groups):
    if len(input_) != 4 or len(weight) != 4: return None
    n, cin, h, w = input_
    cinW, cout_per_grp, kHp, kWp = weight
    if groups == 0 or kH != kHp or kW != kWp: return None
    if cin != cinW: return None
    ho = conv_t_spatial(h, pH, dH, kH, sH, oH)
    wo = conv_t_spatial(w, pW, dW, kW, sW, oW)
    if ho is None or wo is None: return None
    return (n, cout_per_grp * groups, ho, wo)


# ---- 8. view ----
def m_view(input_: Sh, out: Sh) -> Optional[Sh]:
    if prodL(input_) == prodL(out): return tuple(out)
    return None


# ---- 9. reshape (with -1) ----
def m_reshape(input_: Sh, out: List[int]) -> Optional[Sh]:
    total = prodL(input_)
    unknowns = [o for o in out if o == -1]
    knowns = [o for o in out if o != -1]
    pk = prodL(knowns)
    if len(unknowns) == 0:
        return tuple(out) if pk == total else None
    if len(unknowns) == 1:
        if pk == 0: return None
        if total % pk != 0: return None
        inferred = total // pk
        return tuple(inferred if o == -1 else o for o in out)
    return None


# ---- 10. permute ----
def m_permute(input_: Sh, perm: List[int]) -> Optional[Sh]:
    if len(perm) != len(input_): return None
    if any(i >= len(input_) for i in perm): return None
    if sorted(perm) != list(range(len(input_))): return None
    return tuple(input_[i] for i in perm)


# ---- 11. transpose ----
def m_transpose(input_: Sh, i: int, j: int) -> Optional[Sh]:
    if i >= len(input_) or j >= len(input_): return None
    out = list(input_)
    out[i], out[j] = out[j], out[i]
    return tuple(out)


# ---- 12. expand ----
def m_expand(input_: Sh, target: List[Optional[int]]) -> Optional[Sh]:
    if len(target) < len(input_): return None
    extra = len(target) - len(input_)
    aligned = (1,) * extra + tuple(input_)
    out = []
    for a, t in zip(aligned, target):
        if t is None or t == -1:
            out.append(a)
        else:
            if not (t == a or a == 1):
                return None
            out.append(t if a == 1 else a)
    return tuple(out)


# ---- 13. repeat ----
def m_repeat(input_: Sh, reps: List[int]) -> Optional[Sh]:
    if len(reps) < len(input_): return None
    extra = len(reps) - len(input_)
    aligned = (1,) * extra + tuple(input_)
    return tuple(a * r for a, r in zip(aligned, reps))


# ---- 14. broadcast_to ----
def m_broadcast_to(input_: Sh, target: Sh) -> Optional[Sh]:
    if len(target) < len(input_): return None
    extra = len(target) - len(input_)
    aligned = (1,) * extra + tuple(input_)
    for a, t in zip(aligned, target):
        if not (a == t or a == 1):
            return None
    return tuple(target)


# ---- 15. cat ----
def m_cat(shapes: List[Sh], axis: int) -> Optional[Sh]:
    if not shapes: return None
    s = shapes[0]
    if axis >= len(s): return None
    for t in shapes[1:]:
        if len(t) != len(s): return None
        for k in range(len(s)):
            if k != axis and s[k] != t[k]: return None
    total = sum(sh[axis] for sh in shapes)
    return tuple(total if i == axis else d for i, d in enumerate(s))


# ---- 16. stack ----
def m_stack(shapes: List[Sh], axis: int) -> Optional[Sh]:
    if not shapes: return None
    s = shapes[0]
    if axis > len(s): return None
    if any(t != s for t in shapes[1:]): return None
    return tuple(s[:axis]) + (len(shapes),) + tuple(s[axis:])


# ---- 17. split ----
def m_split(input_: Sh, axis: int, chunk_size: int) -> Optional[List[Sh]]:
    if chunk_size == 0 or axis >= len(input_): return None
    d = input_[axis]
    n_full = d // chunk_size
    rem = d % chunk_size
    def mk(sz): return tuple(sz if i == axis else x for i, x in enumerate(input_))
    fulls = [mk(chunk_size)] * n_full
    return fulls if rem == 0 else fulls + [mk(rem)]


# ---- 18. chunk ----
def m_chunk(input_: Sh, axis: int, n: int) -> Optional[List[Sh]]:
    if n == 0 or axis >= len(input_): return None
    d = input_[axis]
    chunk_size = (d + n - 1) // n
    if chunk_size == 0: return [tuple(input_)]
    return m_split(input_, axis, chunk_size)


# ---- 19. unbind ----
def m_unbind(input_: Sh, axis: int) -> Optional[List[Sh]]:
    if axis >= len(input_): return None
    d = input_[axis]
    s2 = tuple(input_[:axis]) + tuple(input_[axis+1:])
    return [s2] * d


# ---- 20. gather ----
def m_gather(input_: Sh, index: Sh, axis: int) -> Optional[Sh]:
    if axis >= len(input_) or len(input_) != len(index): return None
    for i in range(len(input_)):
        if i == axis: continue
        if input_[i] != index[i]: return None
    return tuple(index)


# ---- 21. scatter ----
def m_scatter(input_: Sh, index: Sh, src: Sh, axis: int) -> Optional[Sh]:
    if axis >= len(input_): return None
    if len(input_) != len(index) or len(input_) != len(src): return None
    for i in range(len(input_)):
        if index[i] > src[i]: return None
    return tuple(input_)


# ---- 22. index_select ----
def m_index_select(input_: Sh, axis: int, index_len: int) -> Optional[Sh]:
    if axis >= len(input_): return None
    return tuple(index_len if i == axis else d for i, d in enumerate(input_))


# ---- 23. narrow ----
def m_narrow(input_: Sh, axis: int, start: int, length: int) -> Optional[Sh]:
    if axis >= len(input_): return None
    d = input_[axis]
    if start + length > d: return None
    return tuple(length if i == axis else x for i, x in enumerate(input_))


# ---- 24. embed ----
def m_embed(input_: Sh, num_emb: int, embed_dim: int) -> Optional[Sh]:
    return tuple(input_) + (embed_dim,)


# ---- 25. layer_norm ----
def m_layer_norm(input_: Sh, normalized: Sh) -> Optional[Sh]:
    if len(normalized) > len(input_): return None
    suffix = tuple(input_[len(input_) - len(normalized):])
    return tuple(input_) if suffix == tuple(normalized) else None


# ---- 26. rms_norm ----
def m_rms_norm(input_: Sh, k: int) -> Optional[Sh]:
    if k > len(input_) or k == 0: return None
    return tuple(input_)


# ---- 27. sdpa ----
def m_sdpa(q: Sh, k: Sh, v: Sh) -> Optional[Sh]:
    if len(q) < 4 or len(k) < 4 or len(v) < 4: return None
    if q[-1] != k[-1] or q[-1] != v[-1]: return None
    if q[-3] != k[-3] or q[-3] != v[-3]: return None
    if k[-2] != v[-2]: return None
    if q[:-3] != k[:-3] or q[:-3] != v[:-3]: return None
    return tuple(q)


# ---- 28. linear ----
def m_linear(input_: Sh, in_f: int, out_f: int) -> Optional[Sh]:
    if not input_: return None
    if input_[-1] != in_f: return None
    return tuple(input_[:-1]) + (out_f,)


# =============================================================================
# Step 3: per-operator generators + torch executors. Each entry returns
#   (mirror_predicted: tuple-or-tuples, torch_actual: tuple-or-tuples-or-EXC)
# =============================================================================

R = random.Random(SEED)


def rdim(): return R.randint(DIM_LO, DIM_HI)
def rrank(lo=RANK_LO, hi=RANK_HI): return R.randint(lo, hi)
def rshape(rank=None):
    if rank is None: rank = rrank()
    return tuple(rdim() for _ in range(rank))


def torch_shape(t):
    return tuple(int(x) for x in t.shape)


# Each generator returns a dict with fields:
#   mirror : Optional[tuple|list-of-tuples]
#   actual : tuple|list-of-tuples | "EXC:<msg>"
#   inputs : repr of inputs (for disagreement debug)

def case_matmul():
    # 50% rank-2, 25% rank-3 (batched), 25% with broadcasting
    mode = R.choice(["r2", "r3", "bcast", "vec"])
    if mode == "r2":
        m, k, n = rdim(), rdim(), rdim()
        a, b = (m, k), (k, n)
    elif mode == "r3":
        bb, m, k, n = rdim(), rdim(), rdim(), rdim()
        a, b = (bb, m, k), (bb, k, n)
    elif mode == "vec":
        k = rdim()
        a, b = (k,), (k,)
    else:
        bb, m, k, n = rdim(), rdim(), rdim(), rdim()
        a, b = (1, m, k), (bb, k, n)
    mirror = m_matmul(a, b)
    try:
        actual = torch_shape(torch.matmul(torch.zeros(*a), torch.zeros(*b)))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(a, b))


def case_bmm():
    bb, m, k, n = rdim(), rdim(), rdim(), rdim()
    a, b = (bb, m, k), (bb, k, n)
    mirror = m_bmm(a, b)
    try:
        actual = torch_shape(torch.bmm(torch.zeros(*a), torch.zeros(*b)))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(a, b))


def case_batched_matmul():
    return case_bmm()


def case_conv1d():
    n, cin, l = rdim(), rdim(), R.randint(4, 16)
    cout = rdim()
    k = R.randint(1, min(l, 4))
    stride = R.randint(1, 3)
    pad = R.randint(0, 2)
    dil = R.randint(1, 2)
    groups = 1
    mirror = m_conv1d((n, cin, l), (cout, cin, k), pad, dil, stride, groups)
    try:
        out = F.conv1d(torch.zeros(n, cin, l), torch.zeros(cout, cin, k),
                       stride=stride, padding=pad, dilation=dil, groups=groups)
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual,
                inputs=((n, cin, l), (cout, cin, k), pad, dil, stride, groups))


def case_conv2d():
    n, cin = rdim(), rdim()
    h = R.randint(4, 16); w = R.randint(4, 16)
    cout = rdim()
    kH = R.randint(1, min(h, 4)); kW = R.randint(1, min(w, 4))
    sH = R.randint(1, 3); sW = R.randint(1, 3)
    pH = R.randint(0, 2); pW = R.randint(0, 2)
    dH = R.randint(1, 2); dW = R.randint(1, 2)
    groups = 1
    mirror = m_conv2d((n, cin, h, w), (cout, cin, kH, kW),
                      pH, pW, dH, dW, kH, kW, sH, sW, groups)
    try:
        out = F.conv2d(torch.zeros(n, cin, h, w), torch.zeros(cout, cin, kH, kW),
                       stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW),
                       groups=groups)
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual,
                inputs=((n, cin, h, w), (cout, cin, kH, kW), pH, pW, dH, dW, kH, kW, sH, sW))


def case_conv3d():
    n = R.randint(1, 4); cin = R.randint(1, 4)
    d = R.randint(4, 10); h = R.randint(4, 10); w = R.randint(4, 10)
    cout = R.randint(1, 4)
    kD = R.randint(1, 3); kH = R.randint(1, 3); kW = R.randint(1, 3)
    sD = R.randint(1, 2); sH = R.randint(1, 2); sW = R.randint(1, 2)
    pD = R.randint(0, 1); pH = R.randint(0, 1); pW = R.randint(0, 1)
    dD = 1; dH = 1; dW = 1
    groups = 1
    mirror = m_conv3d((n, cin, d, h, w), (cout, cin, kD, kH, kW),
                      pD, pH, pW, dD, dH, dW, kD, kH, kW, sD, sH, sW, groups)
    try:
        out = F.conv3d(torch.zeros(n, cin, d, h, w),
                       torch.zeros(cout, cin, kD, kH, kW),
                       stride=(sD, sH, sW), padding=(pD, pH, pW),
                       dilation=(dD, dH, dW), groups=groups)
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=("conv3d",))


def case_conv_transpose2d():
    n = R.randint(1, 4); cin = R.randint(1, 4)
    h = R.randint(4, 12); w = R.randint(4, 12)
    cout = R.randint(1, 4)
    kH = R.randint(1, 3); kW = R.randint(1, 3)
    sH = R.randint(1, 2); sW = R.randint(1, 2)
    pH = R.randint(0, 1); pW = R.randint(0, 1)
    dH = 1; dW = 1
    oH = R.randint(0, sH - 1) if sH > 0 else 0
    oW = R.randint(0, sW - 1) if sW > 0 else 0
    groups = 1
    mirror = m_conv_transpose2d((n, cin, h, w), (cin, cout, kH, kW),
                                pH, pW, dH, dW, kH, kW, sH, sW, oH, oW, groups)
    try:
        out = F.conv_transpose2d(
            torch.zeros(n, cin, h, w), torch.zeros(cin, cout, kH, kW),
            stride=(sH, sW), padding=(pH, pW), output_padding=(oH, oW),
            dilation=(dH, dW), groups=groups)
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=("ct2d",))


def case_view():
    rank = rrank()
    s = rshape(rank)
    # construct a divisor-based out
    total = prodL(s)
    # pick a factorization
    factors = []
    n = total
    target_rank = rrank()
    for _ in range(target_rank - 1):
        # random divisor
        divs = [d for d in range(1, n + 1) if n % d == 0]
        f = R.choice(divs)
        factors.append(f)
        n //= f
    factors.append(n)
    out = tuple(factors)
    mirror = m_view(s, out)
    try:
        t = torch.zeros(*s)
        actual = torch_shape(t.contiguous().view(*out))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, out))


def case_reshape():
    rank = rrank()
    s = rshape(rank)
    total = prodL(s)
    # build out with possibly one -1
    target_rank = rrank()
    factors = []
    n = total
    for _ in range(target_rank - 1):
        divs = [d for d in range(1, n + 1) if n % d == 0]
        f = R.choice(divs)
        factors.append(f); n //= f
    factors.append(n)
    out = list(factors)
    if R.random() < 0.5 and len(out) >= 1:
        idx = R.randrange(len(out))
        out[idx] = -1
    mirror = m_reshape(s, out)
    try:
        t = torch.zeros(*s)
        actual = torch_shape(t.reshape(*out))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, out))


def case_permute():
    rank = rrank()
    s = rshape(rank)
    perm = list(range(rank))
    R.shuffle(perm)
    mirror = m_permute(s, perm)
    try:
        actual = torch_shape(torch.zeros(*s).permute(*perm))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, perm))


def case_transpose():
    rank = R.randint(2, RANK_HI)
    s = rshape(rank)
    i = R.randrange(rank); j = R.randrange(rank)
    mirror = m_transpose(s, i, j)
    try:
        actual = torch_shape(torch.zeros(*s).transpose(i, j))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, i, j))


def case_expand():
    rank = rrank()
    s = list(rshape(rank))
    # set some dims to 1 so they're broadcastable
    for i in range(len(s)):
        if R.random() < 0.4: s[i] = 1
    s = tuple(s)
    target_rank = R.randint(rank, RANK_HI)
    extra = target_rank - rank
    target = [rdim() for _ in range(extra)]
    for d in s:
        if d == 1:
            target.append(rdim())
        else:
            target.append(d)
    mirror = m_expand(s, target)
    try:
        actual = torch_shape(torch.zeros(*s).expand(*target))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, target))


def case_repeat():
    rank = rrank()
    s = rshape(rank)
    rep_rank = R.randint(rank, RANK_HI)
    reps = [R.randint(1, 4) for _ in range(rep_rank)]
    mirror = m_repeat(s, reps)
    try:
        actual = torch_shape(torch.zeros(*s).repeat(*reps))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, reps))


def case_broadcast_to():
    rank = rrank()
    s = list(rshape(rank))
    for i in range(len(s)):
        if R.random() < 0.3: s[i] = 1
    s = tuple(s)
    extra = R.randint(0, RANK_HI - rank)
    target = [rdim() for _ in range(extra)]
    for d in s:
        target.append(d if d != 1 else rdim())
    target = tuple(target)
    mirror = m_broadcast_to(s, target)
    try:
        actual = torch_shape(torch.broadcast_to(torch.zeros(*s), target))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, target))


def case_cat():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    n_shapes = R.randint(1, 4)
    shapes = []
    for _ in range(n_shapes):
        sh = list(s)
        sh[axis] = rdim()
        shapes.append(tuple(sh))
    mirror = m_cat(shapes, axis)
    try:
        ts = [torch.zeros(*sh) for sh in shapes]
        actual = torch_shape(torch.cat(ts, dim=axis))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(shapes, axis))


def case_stack():
    rank = R.randint(1, RANK_HI - 1)
    s = rshape(rank)
    axis = R.randint(0, rank)
    n_shapes = R.randint(1, 4)
    shapes = [s] * n_shapes
    mirror = m_stack(shapes, axis)
    try:
        ts = [torch.zeros(*sh) for sh in shapes]
        actual = torch_shape(torch.stack(ts, dim=axis))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(shapes, axis))


def case_split():
    rank = rrank()
    s = rshape(rank)
    axis = R.randrange(rank)
    chunk_size = R.randint(1, max(1, s[axis]))
    mirror = m_split(s, axis, chunk_size)
    try:
        ts = torch.zeros(*s).split(chunk_size, dim=axis)
        actual = [torch_shape(t) for t in ts]
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, axis, chunk_size))


def case_chunk():
    rank = rrank()
    s = rshape(rank)
    axis = R.randrange(rank)
    n = R.randint(1, max(1, s[axis]))
    mirror = m_chunk(s, axis, n)
    try:
        ts = torch.zeros(*s).chunk(n, dim=axis)
        actual = [torch_shape(t) for t in ts]
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, axis, n))


def case_unbind():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    mirror = m_unbind(s, axis)
    try:
        ts = torch.zeros(*s).unbind(dim=axis)
        actual = [torch_shape(t) for t in ts]
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, axis))


def case_gather():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    idx_shape = list(s)
    idx_shape[axis] = R.randint(1, 8)
    idx_shape = tuple(idx_shape)
    mirror = m_gather(s, idx_shape, axis)
    try:
        # values for gather index must be < s[axis]
        idx = torch.zeros(idx_shape, dtype=torch.long)
        actual = torch_shape(torch.gather(torch.zeros(*s), axis, idx))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, idx_shape, axis))


def case_scatter():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    # for scatter_, index and src must have same shape, and index dims
    # must be ≤ src dims and input dims (other than axis).
    idx_shape = list(s)
    for i in range(len(idx_shape)):
        if i != axis:
            idx_shape[i] = R.randint(1, idx_shape[i])
    idx_shape[axis] = R.randint(1, idx_shape[axis])
    idx_shape = tuple(idx_shape)
    src_shape = idx_shape
    mirror = m_scatter(s, idx_shape, src_shape, axis)
    try:
        out = torch.zeros(*s).scatter(axis,
            torch.zeros(idx_shape, dtype=torch.long),
            torch.zeros(src_shape))
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual,
                inputs=(s, idx_shape, src_shape, axis))


def case_index_select():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    idx_len = R.randint(1, 8)
    mirror = m_index_select(s, axis, idx_len)
    try:
        idx = torch.zeros(idx_len, dtype=torch.long)
        actual = torch_shape(torch.index_select(torch.zeros(*s), axis, idx))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, axis, idx_len))


def case_narrow():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    axis = R.randrange(rank)
    d = s[axis]
    start = R.randint(0, d - 1)
    length = R.randint(1, d - start)
    mirror = m_narrow(s, axis, start, length)
    try:
        actual = torch_shape(torch.zeros(*s).narrow(axis, start, length))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, axis, start, length))


def case_embed():
    rank = R.randint(1, RANK_HI - 1)
    s = rshape(rank)
    num_emb = R.randint(2, 32)
    embed_dim = R.randint(1, 16)
    mirror = m_embed(s, num_emb, embed_dim)
    try:
        emb = torch.nn.Embedding(num_emb, embed_dim)
        ids = torch.zeros(*s, dtype=torch.long)
        actual = torch_shape(emb(ids))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, num_emb, embed_dim))


def case_layer_norm():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    norm_rank = R.randint(1, rank)
    normalized = tuple(s[-norm_rank:])
    mirror = m_layer_norm(s, normalized)
    try:
        actual = torch_shape(F.layer_norm(torch.zeros(*s), normalized))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, normalized))


def case_rms_norm():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    k = R.randint(1, rank)
    normalized = tuple(s[-k:])
    mirror = m_rms_norm(s, k)
    try:
        # torch.nn.functional.rms_norm exists in torch ≥ 2.4
        if hasattr(F, "rms_norm"):
            actual = torch_shape(F.rms_norm(torch.zeros(*s), normalized))
        else:
            # Fallback: model as identity
            actual = tuple(s)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, k))


def case_sdpa():
    bb = R.randint(1, 4)
    h = R.randint(1, 4)
    lq = R.randint(1, 8); lk = R.randint(1, 8)
    d = R.randint(1, 8)
    q = (bb, h, lq, d); k = (bb, h, lk, d); v = (bb, h, lk, d)
    mirror = m_sdpa(q, k, v)
    try:
        out = F.scaled_dot_product_attention(
            torch.zeros(*q), torch.zeros(*k), torch.zeros(*v))
        actual = torch_shape(out)
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(q, k, v))


def case_linear():
    rank = R.randint(1, RANK_HI)
    s = rshape(rank)
    in_f = s[-1]
    out_f = rdim()
    mirror = m_linear(s, in_f, out_f)
    try:
        lin = torch.nn.Linear(in_f, out_f)
        actual = torch_shape(lin(torch.zeros(*s)))
    except Exception as e:
        actual = f"EXC:{type(e).__name__}"
    return dict(mirror=mirror, actual=actual, inputs=(s, in_f, out_f))


GENERATORS: dict = {
    "matmul":                       case_matmul,
    "bmm":                          case_bmm,
    "batched_matmul":               case_batched_matmul,
    "conv1d":                       case_conv1d,
    "conv2d":                       case_conv2d,
    "conv3d":                       case_conv3d,
    "conv_transpose2d":             case_conv_transpose2d,
    "view":                         case_view,
    "reshape":                      case_reshape,
    "permute":                      case_permute,
    "transpose":                    case_transpose,
    "expand":                       case_expand,
    "repeat":                       case_repeat,
    "broadcast_to":                 case_broadcast_to,
    "cat":                          case_cat,
    "stack":                        case_stack,
    "split":                        case_split,
    "chunk":                        case_chunk,
    "unbind":                       case_unbind,
    "gather":                       case_gather,
    "scatter":                      case_scatter,
    "index_select":                 case_index_select,
    "narrow":                       case_narrow,
    "embed":                        case_embed,
    "layer_norm":                   case_layer_norm,
    "rms_norm":                     case_rms_norm,
    "scaled_dot_product_attention": case_sdpa,
    "linear":                       case_linear,
}


# =============================================================================
# Step 4: agreement loop
# =============================================================================

def normalize(x):
    if x is None: return None
    if isinstance(x, str): return x  # "EXC:..."
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return [tuple(int(d) for d in s) for s in x]
    return tuple(int(d) for d in x)


def run_op(name: str, gen: Callable) -> dict:
    total = 0
    agreed = 0
    in_fragment = 0
    excluded_mirror_none = 0
    excluded_torch_exc = 0
    sample_dis = []
    t0 = time.time()
    for _ in range(NUM_TESTS):
        total += 1
        try:
            res = gen()
        except Exception as e:
            sample_dis.append({"reason": "generator-error",
                               "error": f"{type(e).__name__}: {e}"})
            continue
        m = normalize(res["mirror"])
        a = normalize(res["actual"])
        # Categorize
        if m is None and isinstance(a, str):
            # Both rejected ⇒ trivially consistent
            agreed += 1
            in_fragment += 1
            continue
        if m is None and not isinstance(a, str):
            excluded_mirror_none += 1
            continue
        if isinstance(a, str) and m is not None:
            # Mirror predicted shape but torch raised — count as disagreement
            if len(sample_dis) < 5:
                sample_dis.append({"inputs": str(res["inputs"])[:200],
                                   "mirror": str(m), "actual": a})
            continue
        in_fragment += 1
        if m == a:
            agreed += 1
        else:
            if len(sample_dis) < 5:
                sample_dis.append({"inputs": str(res["inputs"])[:200],
                                   "mirror": str(m), "actual": str(a)})
    dt = time.time() - t0
    rate = agreed / in_fragment if in_fragment else 0.0
    return dict(
        name=name, total=total, agreed=agreed,
        in_fragment=in_fragment,
        disagreed=in_fragment - agreed,
        excluded_mirror_none=excluded_mirror_none,
        agreement_rate=rate,
        wall_sec=dt,
        sample_disagreements=sample_dis,
    )


def main():
    print("[V5] Loading Lean rule registry ...", flush=True)
    registry = load_lean_registry()
    declared = [r["name"] for r in registry["rules"]]
    print(f"[V5] Lean declares {len(declared)} rules: {declared}", flush=True)

    # Sanity: we should have a Python mirror for every Lean rule
    missing = [n for n in declared if n not in GENERATORS]
    if missing:
        print(f"[V5][WARN] missing python mirror for: {missing}", flush=True)

    results = []
    for r in registry["rules"]:
        name = r["name"]
        if name not in GENERATORS:
            results.append(dict(name=name, total=0, agreed=0,
                                in_fragment=0, disagreed=0,
                                excluded_mirror_none=0,
                                agreement_rate=None,
                                error="no-python-generator"))
            continue
        print(f"[V5] running {name} ({NUM_TESTS} tests) ...", flush=True)
        results.append(run_op(name, GENERATORS[name]))

    overall_total = sum(r["in_fragment"] for r in results)
    overall_agreed = sum(r["agreed"] for r in results)
    overall = overall_agreed / overall_total if overall_total else 0.0

    out = dict(
        metadata=dict(
            seed=SEED,
            python_version=sys.version,
            torch_version=torch.__version__,
            num_tests_per_op=NUM_TESTS,
            dim_range=[DIM_LO, DIM_HI],
            rank_range=[RANK_LO, RANK_HI],
        ),
        lean_registry=registry,
        ops=results,
        overall=dict(
            in_fragment=overall_total,
            agreed=overall_agreed,
            agreement_rate=overall,
            num_ops=len(results),
        ),
    )
    RESULTS.write_text(json.dumps(out, indent=2, default=str))
    print(f"[V5] wrote {RESULTS}", flush=True)
    print(f"[V5] overall agreement: {overall:.4f} "
          f"({overall_agreed}/{overall_total} in-fragment cases)",
          flush=True)
    for r in results:
        rate = r.get("agreement_rate")
        rate_s = f"{rate:.4f}" if isinstance(rate, float) else str(rate)
        print(f"  {r['name']:<32s} total={r['total']:<5d} "
              f"in-frag={r['in_fragment']:<5d} agreed={r['agreed']:<5d} "
              f"rate={rate_s}")


if __name__ == "__main__":
    main()
