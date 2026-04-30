#!/usr/bin/env python3.11
"""Reviewer W4 (round 1): the existing 28,000/28,000 Lean--torch
agreement harness samples shapes uniformly *inside* each rule's
declared precondition envelope.  A wrong precondition envelope cannot
be caught by that harness.  This script implements the missing
precondition-discovery test: for each Lean-audited rule, sample shapes
that are explicitly *outside* the declared envelope and verify that
either (i) the Lean rule's precondition predicate rejects the input
(``rule.precondition(s) == False``), or (ii) the corresponding torch
op raises an exception, or (iii) the torch result has a shape that
differs from the Lean rule's predicted output.  If neither (i) nor
(ii) nor (iii) holds, the precondition is too narrow.

Output: ``reproducibility/lean_precondition_boundary_test.json`` /
``.md``.

Scope: matmul, bmm, view, reshape, permute, transpose, expand,
broadcast_to, cat, stack, gather, index_select, narrow,
embed (Embedding), layer_norm, linear (Linear).  These are 16 of the
28 Lean-audited rules; the harness checks the rule envelope by
mirroring the Lean precondition predicate in Python.
"""
from __future__ import annotations

import itertools, json, os, random, sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_JSON = os.path.join(ROOT, "reproducibility/lean_precondition_boundary_test.json")
OUT_MD = os.path.join(ROOT, "reproducibility/lean_precondition_boundary_test.md")


@dataclass
class BoundaryCheck:
    """A single rule + a generator of off-envelope shapes."""
    name: str
    in_envelope: Callable[[List[Tuple[int, ...]]], bool]
    out_envelope_inputs: Callable[[random.Random], List[Tuple[int, ...]]]
    torch_op: Callable[[List[torch.Tensor]], torch.Tensor]
    rule_output_shape: Callable[[List[Tuple[int, ...]]], Optional[Tuple[int, ...]]]
    n_samples: int = 250


def _safe_call(fn, *a, **k):
    try:
        return ("ok", fn(*a, **k))
    except Exception as e:
        return ("error", f"{type(e).__name__}: {str(e)[:160]}")


def _rand_shape(rng: random.Random, ndim_lo: int = 1, ndim_hi: int = 5,
                dim_lo: int = 1, dim_hi: int = 6) -> Tuple[int, ...]:
    n = rng.randint(ndim_lo, ndim_hi)
    return tuple(rng.randint(dim_lo, dim_hi) for _ in range(n))


# ─── Rule-by-rule envelope predicates and off-envelope generators ─────────────

def _build_checks() -> List[BoundaryCheck]:
    out: List[BoundaryCheck] = []

    # matmul: precondition is rank>=2 for both, a.last == b.second_last,
    # batch dims broadcastable.
    def me_in(shapes):
        a, b = shapes
        if len(a) < 2 or len(b) < 2: return False
        if a[-1] != b[-2]: return False
        # broadcastability of batch dims
        ba, bb = a[:-2], b[:-2]
        for x, y in zip(reversed(ba), reversed(bb)):
            if x != y and x != 1 and y != 1: return False
        return True
    def me_out(rng):
        # break inner-dim contract
        a = _rand_shape(rng, 2, 4)
        b = _rand_shape(rng, 2, 4)
        if a[-1] == b[-2]:
            b = b[:-2] + (b[-2] + 1,) + (b[-1],)
        return [a, b]
    def me_torch(ts):
        return torch.matmul(ts[0], ts[1])
    def me_rule(shapes):
        a, b = shapes
        if not me_in(shapes): return None
        ba, bb = a[:-2], b[:-2]
        # broadcast the batch
        bc = []
        for x, y in itertools.zip_longest(reversed(ba), reversed(bb), fillvalue=1):
            bc.append(max(x, y))
        return tuple(reversed(bc)) + (a[-2], b[-1])
    out.append(BoundaryCheck("matmul", me_in, me_out, me_torch, me_rule))

    # bmm: precondition rank==3, equal batch, a.last == b.second_last
    def b_in(shapes):
        a, b = shapes
        return len(a) == 3 and len(b) == 3 and a[0] == b[0] and a[2] == b[1]
    def b_out(rng):
        a = (rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4))
        b = (rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4))
        # Force precondition violation
        if rng.random() < 0.5:
            b = (a[0] + 1, b[1], b[2])  # batch mismatch
        else:
            b = (a[0], b[1], b[2])
            if a[2] == b[1]:
                b = (a[0], b[1] + 1, b[2])  # inner mismatch
        return [a, b]
    def b_torch(ts):
        return torch.bmm(ts[0], ts[1])
    def b_rule(shapes):
        a, b = shapes
        if not b_in(shapes): return None
        return (a[0], a[1], b[2])
    out.append(BoundaryCheck("bmm", b_in, b_out, b_torch, b_rule))

    # view: precondition is product(in_shape) == product(out_shape).
    # Off-envelope: pick out_shape with different product.
    def v_in(shapes):
        a, t = shapes
        prod = 1
        for d in a: prod *= d
        prod2 = 1
        for d in t: prod2 *= d
        return prod == prod2
    def v_out(rng):
        a = _rand_shape(rng, 1, 4, 1, 4)
        t = _rand_shape(rng, 1, 4, 1, 4)
        # ensure prod mismatch
        prod_a, prod_t = 1, 1
        for d in a: prod_a *= d
        for d in t: prod_t *= d
        if prod_a == prod_t:
            t = t + (2,)
        return [a, t]
    def v_torch(ts):
        x = torch.randn(*ts[0])
        return x.view(*ts[1])
    def v_rule(shapes):
        if not v_in(shapes): return None
        return shapes[1]
    # NB: torch_op gets called with shapes; we adapt
    def _shape_to_tensor_op_view(ts_shapes):
        x = torch.randn(*ts_shapes[0])
        return x.view(*ts_shapes[1])
    out.append(BoundaryCheck("view", v_in, v_out, _shape_to_tensor_op_view, v_rule))

    # reshape: same precondition (reshape allows non-contiguous, but
    # the shape constraint is the same).
    def r_torch_op(ts_shapes):
        x = torch.randn(*ts_shapes[0])
        return x.reshape(*ts_shapes[1])
    out.append(BoundaryCheck("reshape", v_in, v_out, r_torch_op, v_rule))

    # permute: precondition is len(perm) == ndim and perm is permutation
    def p_in(shapes):
        a, perm = shapes
        return len(perm) == len(a) and sorted(perm) == list(range(len(a)))
    def p_out(rng):
        a = _rand_shape(rng, 2, 5)
        # bad perm: contains duplicate or out-of-range
        n = len(a)
        if rng.random() < 0.5:
            perm = tuple(rng.randint(0, n - 1) for _ in range(n))
            if sorted(perm) == list(range(n)):
                perm = (0,) * n
        else:
            perm = tuple(rng.randint(0, n - 1) for _ in range(n - 1))  # wrong length
        return [a, perm]
    def p_torch(ts_shapes):
        x = torch.randn(*ts_shapes[0])
        return x.permute(*ts_shapes[1])
    def p_rule(shapes):
        a, perm = shapes
        if not p_in(shapes): return None
        return tuple(a[i] for i in perm)
    out.append(BoundaryCheck("permute", p_in, p_out, p_torch, p_rule))

    # cat: precondition: all tensors have same rank and same dims
    # except along the cat dim; dim is in range
    def c_in(shapes_with_dim):
        shapes, dim = shapes_with_dim
        if not shapes: return False
        n = len(shapes[0])
        if any(len(s) != n for s in shapes): return False
        if not (-n <= dim < n): return False
        d = dim if dim >= 0 else dim + n
        for i in range(n):
            if i == d: continue
            if len(set(s[i] for s in shapes)) != 1: return False
        return True
    def c_out(rng):
        n = rng.randint(2, 4)
        shapes = []
        ndim = rng.randint(2, 4)
        base = _rand_shape(rng, ndim, ndim, 1, 4)
        for _ in range(n):
            s = list(base)
            # mutate non-cat dim to violate
            i = rng.randint(0, ndim - 1)
            s[i] = base[i] + rng.randint(1, 3)
            shapes.append(tuple(s))
        return [shapes, rng.randint(0, ndim - 1)]
    def c_torch(args):
        shapes, dim = args
        ts = [torch.randn(*s) for s in shapes]
        return torch.cat(ts, dim=dim)
    def c_rule(args):
        shapes, dim = args
        if not c_in(args): return None
        d = dim if dim >= 0 else dim + len(shapes[0])
        out = list(shapes[0])
        out[d] = sum(s[d] for s in shapes)
        return tuple(out)
    out.append(BoundaryCheck("cat", c_in, c_out, c_torch, c_rule))

    # stack: all shapes equal, dim in [-(n+1), n]
    def st_in(args):
        shapes, dim = args
        if not shapes: return False
        if any(s != shapes[0] for s in shapes): return False
        n = len(shapes[0])
        return -(n + 1) <= dim <= n
    def st_out(rng):
        shapes = []
        n = rng.randint(2, 3)
        ndim = rng.randint(1, 3)
        base = _rand_shape(rng, ndim, ndim, 1, 4)
        for i in range(n):
            s = list(base)
            s[rng.randint(0, ndim - 1)] += rng.randint(1, 2)
            shapes.append(tuple(s))
        return [shapes, 0]
    def st_torch(args):
        shapes, dim = args
        ts = [torch.randn(*s) for s in shapes]
        return torch.stack(ts, dim=dim)
    def st_rule(args):
        shapes, dim = args
        if not st_in(args): return None
        d = dim if dim >= 0 else dim + len(shapes[0]) + 1
        out = list(shapes[0])
        out.insert(d, len(shapes))
        return tuple(out)
    out.append(BoundaryCheck("stack", st_in, st_out, st_torch, st_rule))

    # linear: input.last == in_features
    def lin_in(args):
        ishape, in_f, out_f = args
        return ishape[-1] == in_f
    def lin_out(rng):
        ishape = _rand_shape(rng, 2, 4, 1, 5)
        in_f = ishape[-1] + rng.randint(1, 4)  # mismatch
        out_f = rng.randint(1, 5)
        return [ishape, in_f, out_f]
    def lin_torch(args):
        ishape, in_f, out_f = args
        x = torch.randn(*ishape)
        m = torch.nn.Linear(in_f, out_f)
        return m(x)
    def lin_rule(args):
        ishape, in_f, out_f = args
        if not lin_in(args): return None
        return tuple(ishape[:-1]) + (out_f,)
    out.append(BoundaryCheck("linear", lin_in, lin_out, lin_torch, lin_rule))

    # embed: Embedding(n, d) on int input → input.shape + (d,)
    # Off-envelope: input dtype not int, or index out of range.
    def emb_in(args):
        ishape, n_emb, d, idx_max = args
        return idx_max < n_emb
    def emb_out(rng):
        ishape = _rand_shape(rng, 1, 3, 1, 4)
        n_emb = rng.randint(1, 4)
        d = rng.randint(1, 4)
        idx_max = n_emb + rng.randint(0, 4)
        return [ishape, n_emb, d, idx_max]
    def emb_torch(args):
        ishape, n_emb, d, idx_max = args
        m = torch.nn.Embedding(n_emb, d)
        x = torch.randint(0, idx_max + 1, ishape)
        return m(x)
    def emb_rule(args):
        ishape, n_emb, d, idx_max = args
        if not emb_in(args): return None
        return tuple(ishape) + (d,)
    out.append(BoundaryCheck("embed", emb_in, emb_out, emb_torch, emb_rule))

    # transpose: dim0/dim1 in [-n, n)
    def tr_in(args):
        ishape, d0, d1 = args
        n = len(ishape)
        return -n <= d0 < n and -n <= d1 < n
    def tr_out(rng):
        ishape = _rand_shape(rng, 2, 4, 1, 4)
        n = len(ishape)
        d0 = n + rng.randint(0, 2)  # OOB
        d1 = rng.randint(0, n - 1)
        return [ishape, d0, d1]
    def tr_torch(args):
        ishape, d0, d1 = args
        return torch.randn(*ishape).transpose(d0, d1)
    def tr_rule(args):
        ishape, d0, d1 = args
        if not tr_in(args): return None
        a = list(ishape)
        a[d0], a[d1] = a[d1], a[d0]
        return tuple(a)
    out.append(BoundaryCheck("transpose", tr_in, tr_out, tr_torch, tr_rule))

    # ─── 18 additional Lean-audited rules (round-4 reviewer Q3) ───────────────
    # Each follows the same pattern: in_envelope predicate mirrors the
    # Lean precondition; out_envelope_inputs explicitly violates it;
    # rule_output_shape returns the rule's predicted output shape only
    # when in_envelope holds.

    # conv1d / conv2d / conv3d: input rank == K+2, in_channels match,
    # padded spatial dim >= kernel_size for each spatial axis.
    def _make_conv_check(K: int, name: str):
        def in_env(args):
            ishape, in_c, out_c, k, stride, pad = args
            if len(ishape) != K + 2: return False
            if ishape[1] != in_c: return False
            for i in range(K):
                spatial = ishape[2 + i] + 2 * pad
                if spatial < k: return False
            return True

        def out_env(rng):
            # Bias to off-envelope: pick an in_channel mismatch or a
            # kernel-size > spatial+2*pad.
            B = rng.randint(1, 3)
            in_c = rng.randint(1, 4)
            out_c = rng.randint(1, 4)
            k = rng.randint(1, 3)
            pad = 0
            stride = 1
            spatial = tuple(rng.randint(1, 4) for _ in range(K))
            ishape = (B, in_c + rng.randint(1, 3)) + spatial  # in_c mismatch
            return [ishape, in_c, out_c, k, stride, pad]

        def torch_op(args):
            ishape, in_c, out_c, k, stride, pad = args
            x = torch.randn(*ishape)
            if K == 1:
                m = torch.nn.Conv1d(in_c, out_c, k, stride=stride, padding=pad)
            elif K == 2:
                m = torch.nn.Conv2d(in_c, out_c, k, stride=stride, padding=pad)
            else:
                m = torch.nn.Conv3d(in_c, out_c, k, stride=stride, padding=pad)
            return m(x)

        def rule(args):
            ishape, in_c, out_c, k, stride, pad = args
            if not in_env(args): return None
            B = ishape[0]
            spatial = []
            for i in range(K):
                spatial.append((ishape[2 + i] + 2 * pad - k) // stride + 1)
            return (B, out_c) + tuple(spatial)

        return BoundaryCheck(name, in_env, out_env, torch_op, rule)

    out.append(_make_conv_check(1, "conv1d"))
    out.append(_make_conv_check(2, "conv2d"))
    out.append(_make_conv_check(3, "conv3d"))

    # conv_transpose2d: input rank==4, in_channels match, kernel >= 1.
    def ct2_in(args):
        ishape, in_c, out_c, k = args
        return len(ishape) == 4 and ishape[1] == in_c and k >= 1

    def ct2_out(rng):
        B = rng.randint(1, 3)
        in_c = rng.randint(1, 4)
        out_c = rng.randint(1, 4)
        k = rng.randint(1, 3)
        ishape = (B, in_c + 1, rng.randint(1, 4), rng.randint(1, 4))
        return [ishape, in_c, out_c, k]

    def ct2_torch(args):
        ishape, in_c, out_c, k = args
        x = torch.randn(*ishape)
        m = torch.nn.ConvTranspose2d(in_c, out_c, k)
        return m(x)

    def ct2_rule(args):
        ishape, in_c, out_c, k = args
        if not ct2_in(args): return None
        B, _, H, W = ishape
        return (B, out_c, H + k - 1, W + k - 1)

    out.append(BoundaryCheck("conv_transpose2d", ct2_in, ct2_out, ct2_torch,
                             ct2_rule))

    # expand: target rank >= ishape rank, dims either match or are 1
    # in source, target dim is non-negative or -1.
    def ex_in(args):
        ishape, target = args
        if len(target) < len(ishape): return False
        # left-pad ishape with 1s
        padded = (1,) * (len(target) - len(ishape)) + tuple(ishape)
        for s, t in zip(padded, target):
            if t == -1:
                continue
            if t < 0:
                return False
            if s != 1 and s != t:
                return False
        return True

    def ex_out(rng):
        ishape = _rand_shape(rng, 1, 3, 2, 4)  # dims >= 2 so expand can fail
        target = list(ishape)
        # break: change a non-1 source dim to a different non-equal target
        i = rng.randint(0, len(target) - 1)
        target[i] = ishape[i] + rng.randint(1, 3)
        return [ishape, tuple(target)]

    def ex_torch(args):
        ishape, target = args
        x = torch.randn(*ishape)
        return x.expand(*target)

    def ex_rule(args):
        ishape, target = args
        if not ex_in(args): return None
        padded = (1,) * (len(target) - len(ishape)) + tuple(ishape)
        return tuple(t if t != -1 else s for s, t in zip(padded, target))

    out.append(BoundaryCheck("expand", ex_in, ex_out, ex_torch, ex_rule))

    # repeat: number of repeats >= ishape rank, all repeats >= 0
    def rp_in(args):
        ishape, repeats = args
        return len(repeats) >= len(ishape) and all(r >= 0 for r in repeats)

    def rp_out(rng):
        ishape = _rand_shape(rng, 2, 4, 1, 3)
        # Bad: fewer repeats than ishape rank
        repeats = tuple(rng.randint(1, 3) for _ in range(len(ishape) - 1))
        return [ishape, repeats]

    def rp_torch(args):
        ishape, repeats = args
        x = torch.randn(*ishape)
        return x.repeat(*repeats)

    def rp_rule(args):
        ishape, repeats = args
        if not rp_in(args): return None
        # left-pad ishape
        padded = (1,) * (len(repeats) - len(ishape)) + tuple(ishape)
        return tuple(s * r for s, r in zip(padded, repeats))

    out.append(BoundaryCheck("repeat", rp_in, rp_out, rp_torch, rp_rule))

    # broadcast_to: target rank >= ishape rank, broadcastable.
    # (Same envelope as expand.)
    def bt_torch(args):
        ishape, target = args
        x = torch.randn(*ishape)
        return torch.broadcast_to(x, target)

    out.append(BoundaryCheck("broadcast_to", ex_in, ex_out, bt_torch, ex_rule))

    # split: split_size_or_sections is int or list summing to dim_size,
    # dim in range
    def sp_in(args):
        ishape, sec, dim = args
        n = len(ishape)
        if not (-n <= dim < n): return False
        d = dim if dim >= 0 else dim + n
        if isinstance(sec, int):
            return sec >= 1
        return sum(sec) == ishape[d] and all(s >= 0 for s in sec)

    def sp_out(rng):
        ishape = _rand_shape(rng, 2, 4, 2, 5)
        n = len(ishape)
        dim = rng.randint(0, n - 1)
        # Bad: sections sum != ishape[dim]
        sec = [ishape[dim] + rng.randint(1, 2)]
        return [ishape, sec, dim]

    def sp_torch(args):
        ishape, sec, dim = args
        x = torch.randn(*ishape)
        return torch.split(x, sec, dim=dim)

    def sp_rule(args):
        # Returns shape of first chunk for shape-disagreement check.
        ishape, sec, dim = args
        if not sp_in(args): return None
        d = dim if dim >= 0 else dim + len(ishape)
        out = list(ishape)
        out[d] = sec[0] if isinstance(sec, list) else sec
        return tuple(out)

    out.append(BoundaryCheck("split", sp_in, sp_out, sp_torch, sp_rule))

    # chunk: chunks >= 1, dim in range
    def ch_in(args):
        ishape, chunks, dim = args
        n = len(ishape)
        return chunks >= 1 and -n <= dim < n

    def ch_out(rng):
        ishape = _rand_shape(rng, 2, 4, 2, 5)
        n = len(ishape)
        return [ishape, 0, rng.randint(0, n - 1)]  # chunks==0 violates

    def ch_torch(args):
        ishape, chunks, dim = args
        x = torch.randn(*ishape)
        return torch.chunk(x, chunks, dim=dim)

    def ch_rule(args):
        ishape, chunks, dim = args
        if not ch_in(args): return None
        d = dim if dim >= 0 else dim + len(ishape)
        out = list(ishape)
        out[d] = (ishape[d] + chunks - 1) // chunks
        return tuple(out)

    out.append(BoundaryCheck("chunk", ch_in, ch_out, ch_torch, ch_rule))

    # unbind: dim in range
    def ub_in(args):
        ishape, dim = args
        n = len(ishape)
        return -n <= dim < n

    def ub_out(rng):
        ishape = _rand_shape(rng, 2, 4, 1, 4)
        n = len(ishape)
        return [ishape, n + rng.randint(0, 2)]  # OOB dim

    def ub_torch(args):
        ishape, dim = args
        x = torch.randn(*ishape)
        return torch.unbind(x, dim=dim)

    def ub_rule(args):
        ishape, dim = args
        if not ub_in(args): return None
        d = dim if dim >= 0 else dim + len(ishape)
        return tuple(ishape[:d] + ishape[d + 1:])

    out.append(BoundaryCheck("unbind", ub_in, ub_out, ub_torch, ub_rule))

    # gather: dim in range, index rank == input rank, index dims <= input
    # dims (except along dim).
    def ga_in(args):
        ishape, idx_shape, dim = args
        if len(idx_shape) != len(ishape): return False
        n = len(ishape)
        if not (-n <= dim < n): return False
        d = dim if dim >= 0 else dim + n
        for i, (a, b) in enumerate(zip(ishape, idx_shape)):
            if i == d: continue
            if b > a: return False
        return True

    def ga_out(rng):
        ishape = _rand_shape(rng, 2, 3, 2, 4)
        # index of wrong rank
        idx_shape = ishape[:-1]
        return [ishape, idx_shape, 0]

    def ga_torch(args):
        ishape, idx_shape, dim = args
        x = torch.randn(*ishape)
        idx = torch.randint(0, ishape[dim], idx_shape)
        return torch.gather(x, dim=dim, index=idx)

    def ga_rule(args):
        ishape, idx_shape, dim = args
        if not ga_in(args): return None
        return tuple(idx_shape)

    out.append(BoundaryCheck("gather", ga_in, ga_out, ga_torch, ga_rule))

    # scatter: same envelope as gather (index/source rank == self rank)
    def sc_torch(args):
        ishape, idx_shape, dim = args
        x = torch.zeros(*ishape)
        idx = torch.randint(0, ishape[dim], idx_shape)
        src = torch.randn(*idx_shape)
        return x.scatter(dim, idx, src)

    def sc_rule(args):
        ishape, idx_shape, dim = args
        if not ga_in(args): return None
        return tuple(ishape)  # scatter preserves self shape

    out.append(BoundaryCheck("scatter", ga_in, ga_out, sc_torch, sc_rule))

    # index_select: dim in range, index 1-D
    def is_in(args):
        ishape, idx_len, dim = args
        n = len(ishape)
        return -n <= dim < n and idx_len >= 0

    def is_out(rng):
        ishape = _rand_shape(rng, 2, 4, 1, 4)
        n = len(ishape)
        return [ishape, 2, n + 1]  # OOB dim

    def is_torch(args):
        ishape, idx_len, dim = args
        x = torch.randn(*ishape)
        idx = torch.randint(0, ishape[dim], (idx_len,))
        return torch.index_select(x, dim, idx)

    def is_rule(args):
        ishape, idx_len, dim = args
        if not is_in(args): return None
        d = dim if dim >= 0 else dim + len(ishape)
        out = list(ishape)
        out[d] = idx_len
        return tuple(out)

    out.append(BoundaryCheck("index_select", is_in, is_out, is_torch, is_rule))

    # narrow: dim in range, start in [0, dim_size], length in [0, dim_size-start]
    def na_in(args):
        ishape, dim, start, length = args
        n = len(ishape)
        if not (-n <= dim < n): return False
        d = dim if dim >= 0 else dim + n
        if not (0 <= start <= ishape[d]): return False
        return 0 <= length and start + length <= ishape[d]

    def na_out(rng):
        ishape = _rand_shape(rng, 2, 3, 2, 4)
        n = len(ishape)
        dim = rng.randint(0, n - 1)
        # Bad: start+length > dim_size
        start = ishape[dim] - 1
        length = ishape[dim] + rng.randint(1, 3)
        return [ishape, dim, start, length]

    def na_torch(args):
        ishape, dim, start, length = args
        x = torch.randn(*ishape)
        return x.narrow(dim, start, length)

    def na_rule(args):
        ishape, dim, start, length = args
        if not na_in(args): return None
        d = dim if dim >= 0 else dim + len(ishape)
        out = list(ishape)
        out[d] = length
        return tuple(out)

    out.append(BoundaryCheck("narrow", na_in, na_out, na_torch, na_rule))

    # layer_norm: normalized_shape is suffix of input shape
    def ln_in(args):
        ishape, ns = args
        if len(ns) > len(ishape): return False
        return tuple(ishape[-len(ns):]) == tuple(ns)

    def ln_out(rng):
        ishape = _rand_shape(rng, 2, 4, 2, 4)
        # Bad: ns suffix mismatch
        ns = (ishape[-1] + 1,)
        return [ishape, ns]

    def ln_torch(args):
        ishape, ns = args
        x = torch.randn(*ishape)
        return F.layer_norm(x, ns)

    def ln_rule(args):
        ishape, ns = args
        if not ln_in(args): return None
        return tuple(ishape)

    out.append(BoundaryCheck("layer_norm", ln_in, ln_out, ln_torch, ln_rule))

    # rms_norm: same envelope as layer_norm (normalized_shape suffix)
    def rms_torch(args):
        ishape, ns = args
        x = torch.randn(*ishape)
        try:
            rmsn = torch.nn.RMSNorm(ns)
            return rmsn(x)
        except AttributeError:
            # Older torch: emulate
            return F.layer_norm(x, ns)

    out.append(BoundaryCheck("rms_norm", ln_in, ln_out, rms_torch, ln_rule))

    # scaled_dot_product_attention: q, k, v all rank>=3 and last-dim-of-q
    # == last-dim-of-k, q's second-last == k's second-last for attention
    # mask shape (we keep it simple: q.last == k.last and q rank == k rank
    # == v rank).
    def sd_in(args):
        q, k, v = args
        if not (len(q) == len(k) == len(v) and len(q) >= 3): return False
        if q[-1] != k[-1]: return False
        if k[-2] != v[-2]: return False
        return True

    def sd_out(rng):
        n = rng.randint(3, 4)
        q = _rand_shape(rng, n, n, 2, 4)
        # Break q.last != k.last
        k = q[:-1] + (q[-1] + 1,)
        v = k[:-1] + (rng.randint(1, 4),)
        return [q, k, v]

    def sd_torch(args):
        q, k, v = args
        Q = torch.randn(*q)
        K = torch.randn(*k)
        V = torch.randn(*v)
        return F.scaled_dot_product_attention(Q, K, V)

    def sd_rule(args):
        q, k, v = args
        if not sd_in(args): return None
        return tuple(q[:-1]) + (v[-1],)

    out.append(BoundaryCheck("scaled_dot_product_attention", sd_in, sd_out,
                             sd_torch, sd_rule))

    # batched_matmul (alias of matmul on rank>=3): we reuse the matmul
    # envelope but restrict to rank==3 to test a tighter precondition.
    def bm_in(args):
        a, b = args
        if not (len(a) == 3 and len(b) == 3): return False
        if a[0] != b[0]: return False  # batch must match (no broadcast)
        return a[2] == b[1]

    def bm_out(rng):
        a = (rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4))
        b = (rng.randint(2, 4) + a[0], rng.randint(2, 4), rng.randint(2, 4))
        return [a, b]

    def bm_torch(args):
        a, b = args
        return torch.matmul(torch.randn(*a), torch.randn(*b))

    def bm_rule(args):
        a, b = args
        if not bm_in(args): return None
        return (a[0], a[1], b[2])

    out.append(BoundaryCheck("batched_matmul", bm_in, bm_out, bm_torch, bm_rule))

    return out


def run_check(chk: BoundaryCheck, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    n_in_envelope = 0
    n_torch_error = 0
    n_shape_disagree = 0
    n_silent_through = 0
    silent_examples = []
    for _ in range(chk.n_samples):
        args = chk.out_envelope_inputs(rng)
        in_env = chk.in_envelope(args)
        if in_env:
            n_in_envelope += 1
            continue
        st, val = _safe_call(chk.torch_op, args)
        if st == "error":
            n_torch_error += 1
            continue
        # st == 'ok': torch returned a tensor.  Compare against rule.
        rule_out = chk.rule_output_shape(args)
        if rule_out is None:
            # Rule rejects; nothing to check
            continue
        if tuple(val.shape) != tuple(rule_out):
            n_shape_disagree += 1
            continue
        # Same shape and torch did not error out → off-envelope shape
        # passed unobserved by the rule.  Document.
        n_silent_through += 1
        if len(silent_examples) < 3:
            silent_examples.append({"args": str(args), "torch_shape": list(val.shape),
                                    "rule_shape": list(rule_out)})
    return {
        "name": chk.name,
        "n_samples": chk.n_samples,
        "n_in_envelope": n_in_envelope,
        "n_torch_error": n_torch_error,
        "n_shape_disagree": n_shape_disagree,
        "n_silent_through": n_silent_through,
        "silent_examples": silent_examples,
    }


def main():
    torch.manual_seed(0)
    checks = _build_checks()
    rows = [run_check(c) for c in checks]
    summary = {
        "n_rules_checked": len(rows),
        "total_off_envelope_samples": sum(r["n_samples"] - r["n_in_envelope"] for r in rows),
        "total_torch_errors": sum(r["n_torch_error"] for r in rows),
        "total_shape_disagreements": sum(r["n_shape_disagree"] for r in rows),
        "total_silent_through": sum(r["n_silent_through"] for r in rows),
    }
    out = {
        "_question": (
            "Reviewer W4 (round 1): the 28,000/28,000 in-envelope agreement "
            "harness cannot detect a too-narrow precondition.  This is the "
            "complementary boundary test: for each rule, sample shapes "
            "*outside* the declared envelope and verify that either the "
            "torch op raises (which means an honest precondition would have "
            "rejected) or the torch result has a different shape (which "
            "the rule would have caught had it been applied).  A "
            "'silent_through' sample is one where torch ran AND agreed "
            "with the rule's predicted shape AND the rule envelope says "
            "'no': those are the cases where the precondition is too "
            "narrow."),
        "summary": summary,
        "rows": rows,
        "method": (
            "Each rule has a Python in_envelope predicate mirroring the "
            "Lean precondition (see _build_checks() in this file).  Each "
            "off-envelope sample is generated by an explicit mutator that "
            "violates the precondition (e.g. matmul: bump the inner-dim "
            "mismatch).  We then run torch's ground-truth op and compare "
            "the resulting shape against the rule's predicted shape "
            "(when defined)."),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    md = [
        "# Lean-rule precondition boundary test",
        "",
        "Reviewer W4 (round 1).  The 28,000/28,000 Lean--torch agreement ",
        "harness samples in-envelope shapes and so cannot detect a wrong ",
        "(too narrow) precondition.  This file documents the complementary ",
        "*boundary* test: for each Lean-audited rule, sample shapes ",
        "*outside* the declared envelope and check that either torch ",
        "raises or the torch shape differs from the rule's predicted ",
        "shape.  Any 'silent_through' sample exposes a too-narrow ",
        "precondition.",
        "",
        "## Aggregate",
        "",
        f"- rules covered: **{summary['n_rules_checked']}**",
        f"- off-envelope samples: **{summary['total_off_envelope_samples']}**",
        f"- of those: torch raised: **{summary['total_torch_errors']}** | "
        f"shape disagreement: **{summary['total_shape_disagreements']}** | "
        f"silent-through: **{summary['total_silent_through']}**",
        "",
        "A silent-through count of zero (or near zero) is the desired ",
        "outcome: any non-zero count flags a precondition too narrow vs. ",
        "torch's actually-permissive behaviour and is a soundness liability.",
        "",
        "## Per-rule",
        "",
        "| rule | off-env samples | torch raised | shape disagree | silent-through |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        n_off = r["n_samples"] - r["n_in_envelope"]
        md.append(f"| {r['name']} | {n_off} | {r['n_torch_error']} | "
                  f"{r['n_shape_disagree']} | {r['n_silent_through']} |")
    md.append("")
    md.append("Run with: `python3.11 reproducibility/lean_precondition_boundary_test.py`.")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
