"""End-to-end: the source-level einops checker catches bugs that real einops
models hit at runtime, and stays silent on correct models.

Each "buggy" source below is also *executed* with real torch + einops to prove
the runtime genuinely raises, so the static finding is grounded in real
behaviour rather than an invented rule.
"""

from __future__ import annotations

import textwrap

import pytest

torch = pytest.importorskip("torch")
einops = pytest.importorskip("einops")

from src.einops_source import find_einops_bugs, verify_einops_source  # noqa: E402


PATCH_EMBED_BUG = textwrap.dedent(
    '''
    import torch
    import torch.nn as nn
    from einops import rearrange

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(48, 64)

        def forward(self, x):
            # x: (batch, channels=3, H=14, W=14); patch=5 does NOT divide 14
            x = rearrange(
                x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=5, p2=5
            )
            return self.proj(x)
    '''
)

PATCH_EMBED_OK = textwrap.dedent(
    '''
    import torch
    import torch.nn as nn
    from einops import rearrange

    class PatchEmbed(nn.Module):
        def forward(self, x):
            # patch=7 divides 14 cleanly
            x = rearrange(
                x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=7, p2=7
            )
            return x
    '''
)

HEADS_BUG = textwrap.dedent(
    '''
    from einops import rearrange

    def attn(x):
        # x: (batch=2, seq=10, dim=63); 63 not divisible by num heads 8
        q = rearrange(x, "b s (h d) -> b h s d", h=8)
        return q
    '''
)

MISSING_LEN_BUG = textwrap.dedent(
    '''
    from einops import repeat

    def tile(x):
        # repeat needs the new axis length
        return repeat(x, "b c -> b c k")
    '''
)


def _runtime_raises(src: str, shape) -> bool:
    """Execute the module/function on a real tensor; return True iff it raises."""
    ns: dict = {}
    exec(compile(src, "<m>", "exec"), ns)
    x = torch.zeros(*shape)
    try:
        if "PatchEmbed" in ns:
            ns["PatchEmbed"]().forward(x)
        elif "attn" in ns:
            ns["attn"](x)
        elif "tile" in ns:
            ns["tile"](x)
        else:  # pragma: no cover
            raise AssertionError("no entrypoint")
        return False
    except Exception:
        return True


def test_patch_embed_divisibility_bug_is_caught_and_real():
    shape = (2, 3, 14, 14)
    assert _runtime_raises(PATCH_EMBED_BUG, shape)  # grounding
    bugs = find_einops_bugs(PATCH_EMBED_BUG, {"x": shape})
    assert len(bugs) == 1
    assert "rearrange" in bugs[0].message
    assert bugs[0].severity == "error"


def test_patch_embed_correct_is_silent_and_runs():
    shape = (2, 3, 14, 14)
    assert not _runtime_raises(PATCH_EMBED_OK, shape)  # grounding
    bugs = find_einops_bugs(PATCH_EMBED_OK, {"x": shape})
    assert bugs == []


def test_attention_head_split_bug():
    shape = (2, 10, 63)
    assert _runtime_raises(HEADS_BUG, shape)
    bugs = find_einops_bugs(HEADS_BUG, {"x": shape})
    assert len(bugs) == 1
    assert bugs[0].fix_suggestion


def test_missing_repeat_length_bug():
    shape = (2, 4)
    assert _runtime_raises(MISSING_LEN_BUG, shape)
    res = verify_einops_source(MISSING_LEN_BUG, {"x": shape})
    assert res.status == "UNSAFE"
    assert any("k" in b.message or "length" in (b.fix_suggestion or "")
               for b in res.bugs)


def test_unknown_shape_is_skipped_no_false_positive():
    # No seed shape for x -> nothing to check, no false positives.
    bugs = find_einops_bugs(HEADS_BUG, {})
    assert bugs == []


def test_symbolic_seq_dim_not_refuted():
    # Symbolic batch but concrete, divisible dim -> must stay silent.
    src = textwrap.dedent(
        '''
        from einops import rearrange
        def f(x):
            return rearrange(x, "b (h d) -> b h d", h=8)
        '''
    )
    bugs = find_einops_bugs(src, {"x": ("batch", 64)})
    assert bugs == []
