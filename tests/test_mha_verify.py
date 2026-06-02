from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.fx_extractor import verify_module  # noqa: E402
from src.mha_verify import verify_multihead_attention  # noqa: E402


def test_public_exports():
    import src
    import tensorguard

    assert src.verify_multihead_attention is verify_multihead_attention
    assert tensorguard.verify_multihead_attention is verify_multihead_attention


def _rand(shape):
    return torch.randn(*shape)


def _attn_mask(shape):
    return torch.zeros(*shape, dtype=torch.bool)


def _key_padding_mask(shape):
    return torch.zeros(*shape, dtype=torch.bool)


def _real_module(
    q,
    k,
    v,
    *,
    embed_dim,
    num_heads,
    kdim=None,
    vdim=None,
    batch_first=False,
    attn_mask=None,
    key_padding_mask=None,
    need_weights=True,
    average_attn_weights=True,
):
    try:
        mod = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            out, weights = mod(
                _rand(q),
                _rand(k),
                _rand(v),
                attn_mask=_attn_mask(attn_mask) if attn_mask else None,
                key_padding_mask=(
                    _key_padding_mask(key_padding_mask)
                    if key_padding_mask else None
                ),
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )
        return "ok", tuple(out.shape), None if weights is None else tuple(weights.shape)
    except Exception:
        return "err", None, None


def _check_module(
    q,
    k,
    v,
    *,
    embed_dim,
    num_heads,
    kdim=None,
    vdim=None,
    batch_first=False,
    attn_mask=None,
    key_padding_mask=None,
    need_weights=True,
    average_attn_weights=True,
):
    real_status, real_out, real_weights = _real_module(
        q,
        k,
        v,
        embed_dim=embed_dim,
        num_heads=num_heads,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        average_attn_weights=average_attn_weights,
    )
    verdict = verify_multihead_attention(
        q,
        k,
        v,
        embed_dim,
        num_heads,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        average_attn_weights=average_attn_weights,
    )
    static_status = "ok" if verdict.ok else "err"
    assert static_status == real_status, (
        f"q={q} k={k} v={v} mask={attn_mask} kpm={key_padding_mask}: "
        f"real={real_status} static={static_status} ({verdict.error})"
    )
    if real_status == "ok":
        assert verdict.output_shape == real_out
        assert verdict.weights_shape == real_weights
    return verdict


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            q=(2, 5, 16),
            k=(2, 7, 16),
            v=(2, 7, 16),
            embed_dim=16,
            num_heads=4,
            batch_first=True,
            attn_mask=(5, 7),
            key_padding_mask=(2, 7),
        ),
        dict(
            q=(5, 2, 16),
            k=(7, 2, 16),
            v=(7, 2, 16),
            embed_dim=16,
            num_heads=4,
            attn_mask=(8, 5, 7),
            need_weights=True,
            average_attn_weights=False,
        ),
        dict(
            q=(2, 5, 16),
            k=(2, 7, 12),
            v=(2, 7, 20),
            embed_dim=16,
            num_heads=4,
            kdim=12,
            vdim=20,
            batch_first=True,
            need_weights=False,
        ),
        dict(
            q=(5, 16),
            k=(7, 16),
            v=(7, 16),
            embed_dim=16,
            num_heads=4,
            attn_mask=(4, 5, 7),
            key_padding_mask=(7,),
            average_attn_weights=False,
        ),
    ],
)
def test_valid_module_contracts_match_real_pytorch(kwargs):
    verdict = _check_module(**kwargs)
    assert verdict.ok


@pytest.mark.parametrize(
    "kwargs,kind",
    [
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 12),
                v=(2, 7, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
            ),
            "key_embed_dim",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 12),
                v=(2, 7, 21),
                embed_dim=16,
                num_heads=4,
                kdim=12,
                vdim=20,
                batch_first=True,
            ),
            "value_embed_dim",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 16),
                v=(2, 6, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
            ),
            "source_length",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(3, 7, 16),
                v=(3, 7, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
            ),
            "batch",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 16),
                v=(2, 7, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
                attn_mask=(5, 8),
            ),
            "attn_mask",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 16),
                v=(2, 7, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
                attn_mask=(7, 5, 7),
            ),
            "attn_mask",
        ),
        (
            dict(
                q=(2, 5, 16),
                k=(2, 7, 16),
                v=(2, 7, 16),
                embed_dim=16,
                num_heads=4,
                batch_first=True,
                key_padding_mask=(2, 8),
            ),
            "key_padding_mask",
        ),
    ],
)
def test_invalid_module_contracts_match_real_pytorch(kwargs, kind):
    verdict = _check_module(**kwargs)
    assert not verdict.ok
    assert verdict.error_kind == kind


def _real_functional(
    q,
    k,
    v,
    *,
    embed_dim,
    num_heads,
    kdim=None,
    vdim=None,
    use_separate_proj_weight=False,
    attn_mask=None,
    key_padding_mask=None,
    need_weights=True,
    average_attn_weights=True,
):
    try:
        tq, tk, tv = _rand(q), _rand(k), _rand(v)
        in_proj_weight = None
        q_proj_weight = k_proj_weight = v_proj_weight = None
        if use_separate_proj_weight:
            kdim = kdim if kdim is not None else k[-1]
            vdim = vdim if vdim is not None else v[-1]
            q_proj_weight = torch.randn(embed_dim, embed_dim)
            k_proj_weight = torch.randn(embed_dim, kdim)
            v_proj_weight = torch.randn(embed_dim, vdim)
        else:
            in_proj_weight = torch.randn(3 * embed_dim, embed_dim)
        with torch.no_grad():
            out, weights = F.multi_head_attention_forward(
                tq,
                tk,
                tv,
                embed_dim,
                num_heads,
                in_proj_weight,
                torch.zeros(3 * embed_dim),
                None,
                None,
                False,
                0.0,
                torch.randn(embed_dim, embed_dim),
                torch.zeros(embed_dim),
                training=False,
                key_padding_mask=(
                    _key_padding_mask(key_padding_mask)
                    if key_padding_mask else None
                ),
                need_weights=need_weights,
                attn_mask=_attn_mask(attn_mask) if attn_mask else None,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                average_attn_weights=average_attn_weights,
            )
        return "ok", tuple(out.shape), None if weights is None else tuple(weights.shape)
    except Exception:
        return "err", None, None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            q=(5, 2, 16),
            k=(7, 2, 16),
            v=(7, 2, 16),
            embed_dim=16,
            num_heads=4,
            attn_mask=(5, 7),
            key_padding_mask=(2, 7),
        ),
        dict(
            q=(5, 2, 16),
            k=(7, 2, 12),
            v=(7, 2, 20),
            embed_dim=16,
            num_heads=4,
            kdim=12,
            vdim=20,
            use_separate_proj_weight=True,
            need_weights=False,
        ),
        dict(
            q=(5, 16),
            k=(7, 16),
            v=(7, 16),
            embed_dim=16,
            num_heads=4,
            attn_mask=(4, 5, 7),
            key_padding_mask=(7,),
            average_attn_weights=False,
        ),
    ],
)
def test_functional_contracts_match_real_pytorch(kwargs):
    real_status, real_out, real_weights = _real_functional(**kwargs)
    verdict = verify_multihead_attention(
        kwargs["q"],
        kwargs["k"],
        kwargs["v"],
        kwargs["embed_dim"],
        kwargs["num_heads"],
        kdim=kwargs.get("kdim"),
        vdim=kwargs.get("vdim"),
        attn_mask=kwargs.get("attn_mask"),
        key_padding_mask=kwargs.get("key_padding_mask"),
        need_weights=kwargs.get("need_weights", True),
        average_attn_weights=kwargs.get("average_attn_weights", True),
        use_separate_proj_weight=kwargs.get("use_separate_proj_weight"),
    )
    assert ("ok" if verdict.ok else "err") == real_status
    if real_status == "ok":
        assert verdict.output_shape == real_out
        assert verdict.weights_shape == real_weights


def test_symbolic_and_nested_inputs_abstain_without_false_positives():
    verdict = verify_multihead_attention(
        ("N", "L", 16),
        ("N", "S", "K"),
        ("N", "S", "V"),
        16,
        4,
        kdim="K",
        vdim="V",
        batch_first=True,
        attn_mask=("N_times_H", "L", "S"),
        key_padding_mask=("N", "S"),
    )
    assert verdict.ok
    assert verdict.output_shape == ("N", "L", 16)
    assert verdict.weights_shape == ("N", "L", "S")

    nested = verify_multihead_attention((2, 5, 16), (2, 5, 16), (2, 5, 16), 16, 4, nested_tensor=True)
    assert nested.ok
    assert nested.abstained
    assert nested.output_shape is None


def test_random_valid_module_fuzz_matches_real_pytorch():
    rng = random.Random(189)
    checked = 0
    for _ in range(80):
        batch_first = rng.choice([False, True])
        batched = rng.choice([False, True])
        n = rng.randint(1, 3)
        lq = rng.randint(1, 5)
        lk = rng.randint(1, 5)
        heads = rng.choice([1, 2, 4])
        head_dim = rng.choice([2, 3, 4])
        embed = heads * head_dim
        separate = rng.choice([False, True])
        kdim = rng.choice([5, 6, 7]) if separate else None
        vdim = rng.choice([8, 9, 10]) if separate else None

        def shaped(seq, dim):
            if not batched:
                return (seq, dim)
            return (n, seq, dim) if batch_first else (seq, n, dim)

        q = shaped(lq, embed)
        k = shaped(lk, kdim or embed)
        v = shaped(lk, vdim or embed)
        attn_mask = None
        if rng.random() < 0.5:
            attn_mask = (lq, lk)
        elif rng.random() < 0.5:
            lead = heads if not batched else n * heads
            attn_mask = (lead, lq, lk)
        key_padding_mask = None
        if rng.random() < 0.5:
            key_padding_mask = (n, lk) if batched else (lk,)
        need_weights = rng.choice([False, True])
        average = rng.choice([False, True])
        verdict = _check_module(
            q,
            k,
            v,
            embed_dim=embed,
            num_heads=heads,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average,
        )
        assert verdict.ok
        checked += 1
    assert checked == 80


def _violation_kinds(result):
    if result.counterexample is None:
        return []
    return [v.kind for v in result.counterexample.violations]


def test_verify_module_catches_kdim_vdim_and_mask_mismatches():
    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(
                16, 4, kdim=12, vdim=20, batch_first=True
            )
            self.proj = nn.Linear(16, 3)

        def forward(self, q, k, v):
            out, _weights = self.mha(q, k, v, need_weights=False)
            return self.proj(out)

    good = verify_module(
        CrossAttention(),
        input_shapes={"q": (2, 5, 16), "k": (2, 7, 12), "v": (2, 7, 20)},
    )
    assert good.safe, _violation_kinds(good)

    bad = verify_module(
        CrossAttention(),
        input_shapes={"q": (2, 5, 16), "k": (2, 7, 13), "v": (2, 7, 20)},
    )
    assert not bad.safe
    assert "shape_incompatible" in _violation_kinds(bad)

    class MaskedSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(16, 4, batch_first=True)

        def forward(self, x, mask):
            out, _weights = self.mha(
                x, x, x, attn_mask=mask, need_weights=False
            )
            return out

    masked = verify_module(
        MaskedSelfAttention(),
        input_shapes={"x": (2, 5, 16), "mask": (5, 6)},
    )
    assert not masked.safe
    assert "shape_incompatible" in _violation_kinds(masked)
