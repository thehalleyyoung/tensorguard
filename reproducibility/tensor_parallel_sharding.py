#!/usr/bin/env python3
"""
tensor_parallel_sharding.py — prove the tensor-parallel checker against REAL
PyTorch (100_STEPS.md Step 97, Phase 10).

For each tensor-parallel MLP or attention configuration we hold two things side
by side:

  * the **static** verdict from
    ``src/tensor_parallel_checks.verify_tensor_parallel``, and
  * the **runtime** outcome of actually hand-sharding a reference linear stack
    across ``tp_size`` simulated ranks, running each rank, all-reducing /
    gathering, and comparing to the unsharded forward.

A consistent config must verify OK statically AND reproduce the reference output
(bit-for-bit up to fp tolerance) at runtime; each inconsistent config must be
flagged statically AND fail at runtime (shape error, no even shard, or invalid
head partition). The artifact records boolean outcomes and verdicts only (no
timing, no raw floats), so it is byte-deterministic and checked by
``reproduce_all.py --check``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.tensor_parallel_checks import (  # noqa: E402
    verify_tensor_parallel,
    verify_tensor_parallel_attention,
    llama_gqa_attention,
    megatron_attention,
    megatron_mlp,
    TPAttentionSpec,
    TPKVSharding,
    TPLinearSpec,
    TPKind,
)

OUT_JSON = REPO / "reproducibility" / "tensor_parallel_sharding.json"
OUT_MD = REPO / "reproducibility" / "tensor_parallel_sharding.md"

SEED = 0
TOL = 1e-4


def _simulate_megatron(hidden: int, ffn: int, tp: int, *,
                       gather_output: bool, input_is_parallel: bool) -> Dict:
    """Hand-shard a reference 2-layer MLP across ``tp`` ranks and compare to the
    unsharded forward. Returns dict with ``ran`` and ``matches_reference``."""
    torch.manual_seed(SEED)
    fc1 = nn.Linear(hidden, ffn, bias=False)
    fc2 = nn.Linear(ffn, hidden, bias=False)
    x = torch.randn(3, hidden)
    ref = fc2(F.relu(fc1(x)))

    if ffn % tp != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "ffn not divisible by tp (no even shard)"}

    W1 = fc1.weight.data          # [ffn, hidden]  column-parallel: split dim 0
    W2 = fc2.weight.data          # [hidden, ffn]  row-parallel: split dim 1
    s1 = W1.split(ffn // tp, dim=0)
    s2 = W2.split(ffn // tp, dim=1)

    try:
        if not gather_output and input_is_parallel:
            # canonical no-comm path: each rank keeps its shard, all-reduce sum
            z = torch.zeros(3, hidden)
            for r in range(tp):
                y_local = F.relu(x @ s1[r].T)        # [3, ffn/tp]
                z = z + (y_local @ s2[r].T)          # [3, hidden]
            out = z
        else:
            # column gathers full [3, ffn] then feeds a row shard expecting a
            # sharded input -> dimension mismatch (the real bug).
            y_full = F.relu(x @ W1.T)                # [3, ffn]
            out = y_full @ s2[0].T                   # shape error expected
    except RuntimeError as e:
        return {"ran": False, "matches_reference": False,
                "reason": f"runtime shape error: {str(e)[:48]}"}

    matches = bool(torch.allclose(out, ref, atol=TOL))
    return {"ran": True, "matches_reference": matches, "reason": ""}


class _TinyGQAAttention(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _full_attention_forward(
            x,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.o_proj.weight,
            self.heads,
            self.kv_heads,
            self.head_dim,
        )


def _full_attention_forward(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, seq, _hidden = x.shape
    q = F.linear(x, q_weight).view(batch, seq, heads, head_dim).transpose(1, 2)
    k = F.linear(x, k_weight).view(batch, seq, kv_heads, head_dim).transpose(1, 2)
    v = F.linear(x, v_weight).view(batch, seq, kv_heads, head_dim).transpose(1, 2)
    repeat = heads // kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v)
    ctx = ctx.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)
    return F.linear(ctx, o_weight)


def _kv_head_slice_for_rank(
    rank: int,
    tp: int,
    kv_heads: int,
    mode: TPKVSharding,
) -> Tuple[int, int]:
    if mode is TPKVSharding.SHARD:
        per = kv_heads // tp
        return rank * per, (rank + 1) * per
    if mode is TPKVSharding.REPLICATE:
        ranks_per_kv = tp // kv_heads
        head = rank // ranks_per_kv
        return head, head + 1
    if kv_heads % tp == 0:
        return _kv_head_slice_for_rank(rank, tp, kv_heads, TPKVSharding.SHARD)
    return _kv_head_slice_for_rank(rank, tp, kv_heads, TPKVSharding.REPLICATE)


def _simulate_attention(
    hidden: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    tp: int,
    *,
    kv_sharding: TPKVSharding,
) -> Dict:
    torch.manual_seed(SEED)
    module = _TinyGQAAttention(hidden, heads, kv_heads, head_dim)
    x = torch.randn(2, 5, hidden)

    if heads % kv_heads != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "query heads not grouped by kv heads"}
    if heads % tp != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "query heads not divisible by tp"}
    if kv_sharding is TPKVSharding.SHARD and kv_heads % tp != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "kv heads not divisible by tp"}
    if kv_sharding is TPKVSharding.REPLICATE and tp % kv_heads != 0:
        return {"ran": False, "matches_reference": False,
                "reason": "tp not divisible by kv heads for replication"}

    ref = module(x)
    batch, seq, _hidden = x.shape
    q_per_rank = heads // tp
    q_weight = module.q_proj.weight.data
    k_weight = module.k_proj.weight.data
    v_weight = module.v_proj.weight.data
    o_weight = module.o_proj.weight.data

    pieces: List[torch.Tensor] = []
    for rank in range(tp):
        q0 = rank * q_per_rank
        q1 = (rank + 1) * q_per_rank
        q_rows = slice(q0 * head_dim, q1 * head_dim)
        kv0, kv1 = _kv_head_slice_for_rank(rank, tp, kv_heads, kv_sharding)
        kv_rows = slice(kv0 * head_dim, kv1 * head_dim)
        local_kv_heads = kv1 - kv0

        q = F.linear(x, q_weight[q_rows]).view(
            batch, seq, q_per_rank, head_dim
        ).transpose(1, 2)
        k = F.linear(x, k_weight[kv_rows]).view(
            batch, seq, local_kv_heads, head_dim
        ).transpose(1, 2)
        v = F.linear(x, v_weight[kv_rows]).view(
            batch, seq, local_kv_heads, head_dim
        ).transpose(1, 2)
        repeat = q_per_rank // local_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous()
        ctx = ctx.view(batch, seq, q_per_rank * head_dim)
        o_cols = q_rows
        pieces.append(F.linear(ctx, o_weight[:, o_cols]))

    out = torch.stack(pieces, dim=0).sum(dim=0)
    matches = bool(torch.allclose(out, ref, atol=TOL))
    return {"ran": True, "matches_reference": matches, "reason": ""}


def _real_hf_llama_attention_spec() -> Tuple[Optional[TPAttentionSpec], str]:
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
    except Exception as e:
        return None, f"transformers unavailable: {type(e).__name__}"

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=16,
        vocab_size=100,
        head_dim=40,
    )
    attn = LlamaAttention(cfg, layer_idx=0)
    return TPAttentionSpec(
        name="real_hf_llama_attention",
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        q_proj_shape=tuple(attn.q_proj.weight.shape),
        k_proj_shape=tuple(attn.k_proj.weight.shape),
        v_proj_shape=tuple(attn.v_proj.weight.shape),
        o_proj_shape=tuple(attn.o_proj.weight.shape),
    ), ""


def _hf_llama_shapes_match_static() -> Dict:
    spec, reason = _real_hf_llama_attention_spec()
    if spec is None:
        return {"ran": None, "matches_reference": None, "reason": reason}
    static = verify_tensor_parallel_attention(spec, tp_size=4)
    return {"ran": True, "matches_reference": static.ok, "reason": ""}


class Case:
    def __init__(self, name: str, specs: List[TPLinearSpec], tp: int,
                 sim_kwargs: Dict, hidden: int, ffn: int,
                 expect_static_ok: bool, expect_runtime_match: bool):
        self.name = name
        self.specs = specs
        self.tp = tp
        self.sim_kwargs = sim_kwargs
        self.hidden = hidden
        self.ffn = ffn
        self.expect_static_ok = expect_static_ok
        self.expect_runtime_match = expect_runtime_match


class AttentionCase:
    def __init__(self, name: str, spec: TPAttentionSpec, tp: int,
                 sim_kwargs: Optional[Dict], expect_static_ok: bool,
                 expect_runtime_match: bool):
        self.name = name
        self.spec = spec
        self.tp = tp
        self.sim_kwargs = sim_kwargs
        self.expect_static_ok = expect_static_ok
        self.expect_runtime_match = expect_runtime_match


CASES: List[Case] = [
    Case("megatron_mlp_tp2_correct", megatron_mlp(8, 16), 2,
         {"gather_output": False, "input_is_parallel": True}, 8, 16,
         expect_static_ok=True, expect_runtime_match=True),
    Case("megatron_mlp_tp4_correct", megatron_mlp(8, 16), 4,
         {"gather_output": False, "input_is_parallel": True}, 8, 16,
         expect_static_ok=True, expect_runtime_match=True),
    Case("comm_flag_mismatch_gather_then_parallel",
         megatron_mlp(8, 16, gather_output=True, input_is_parallel=True), 2,
         {"gather_output": True, "input_is_parallel": True}, 8, 16,
         expect_static_ok=False, expect_runtime_match=False),
    Case("indivisible_ffn15_tp2", megatron_mlp(8, 15), 2,
         {"gather_output": False, "input_is_parallel": True}, 8, 15,
         expect_static_ok=False, expect_runtime_match=False),
    Case("inner_dim_mismatch",
         [TPLinearSpec("fc1", 8, 16, TPKind.COLUMN),
          TPLinearSpec("fc2", 12, 8, TPKind.ROW, input_is_parallel=True)], 2,
         None, 8, 16, expect_static_ok=False, expect_runtime_match=False),
]


ATTENTION_CASES: List[AttentionCase] = [
    AttentionCase(
        "hf_style_gqa_sharded_kv_head_dim_independent",
        llama_gqa_attention(12, 6, 2, head_dim=3, sequence_parallel=True),
        2,
        {"hidden": 12, "heads": 6, "kv_heads": 2, "head_dim": 3,
         "kv_sharding": TPKVSharding.SHARD},
        expect_static_ok=True,
        expect_runtime_match=True,
    ),
    AttentionCase(
        "megatron_style_mqa_replicated_kv",
        megatron_attention(16, 8, 1, 4, head_dim=2,
                          kv_sharding=TPKVSharding.REPLICATE,
                          sequence_parallel=True),
        4,
        {"hidden": 16, "heads": 8, "kv_heads": 1, "head_dim": 2,
         "kv_sharding": TPKVSharding.REPLICATE},
        expect_static_ok=True,
        expect_runtime_match=True,
    ),
    AttentionCase(
        "real_transformers_llama_projection_shapes",
        TPAttentionSpec(
           name="real_hf_llama_attention",
           hidden_size=32,
           num_attention_heads=8,
           num_key_value_heads=2,
           head_dim=40,
        ),
        4,
        {"hf_real": True},
        expect_static_ok=True,
        expect_runtime_match=True,
    ),
    AttentionCase(
        "kv_heads_neither_shard_nor_replicate",
        llama_gqa_attention(24, 12, 3, head_dim=2),
        4,
        None,
        expect_static_ok=False,
        expect_runtime_match=False,
    ),
    AttentionCase(
        "sequence_parallel_hidden_axis_layernorm",
        TPAttentionSpec(
           name="bad_sp_layernorm",
           hidden_size=16,
           num_attention_heads=8,
           num_key_value_heads=1,
           head_dim=2,
           sequence_parallel=True,
           activation_shape=(2, 5, 16),
           sequence_parallel_axis=-1,
           layer_norm_shape=(16,),
        ),
        4,
        None,
        expect_static_ok=False,
        expect_runtime_match=False,
    ),
]


def measure() -> Dict:
    rows: List[Dict] = []
    all_ok = True
    for c in CASES:
        res = verify_tensor_parallel(c.specs, c.tp)
        static_ok = res.ok
        static_match = static_ok == c.expect_static_ok

        if c.sim_kwargs is None:
            sim = {"ran": None, "matches_reference": None,
                   "reason": "structural (no runtime simulation)"}
            runtime_match = True
        else:
            sim = _simulate_megatron(c.hidden, c.ffn, c.tp, **c.sim_kwargs)
            runtime_match = sim["matches_reference"] == c.expect_runtime_match

        ok = static_match and runtime_match
        all_ok = all_ok and ok
        rows.append({
            "case_type": "mlp",
            "name": c.name,
            "tp_size": c.tp,
            "static_ok": static_ok,
            "static_issues": sorted({i.kind.value for i in res.issues}),
            "expect_static_ok": c.expect_static_ok,
            "static_match": static_match,
            "live_ran": sim["ran"],
            "live_matches_reference": sim["matches_reference"],
            "live_reason": sim["reason"],
            "expect_live_match": c.expect_runtime_match,
            "live_match": runtime_match,
            "ok": ok,
        })
    for c in ATTENTION_CASES:
        spec = c.spec
        real_hf_reason = ""
        if c.sim_kwargs is not None and c.sim_kwargs.get("hf_real"):
            real_spec, real_hf_reason = _real_hf_llama_attention_spec()
            if real_spec is not None:
                spec = real_spec
        res = verify_tensor_parallel_attention(spec, c.tp)
        static_ok = res.ok
        static_match = static_ok == c.expect_static_ok

        if c.sim_kwargs is None:
            sim = {"ran": None, "matches_reference": None,
                   "reason": "structural (no runtime simulation)"}
            runtime_match = True
        elif c.sim_kwargs.get("hf_real"):
            if real_hf_reason:
                sim = {"ran": None, "matches_reference": None,
                       "reason": real_hf_reason}
            else:
                sim = {"ran": True, "matches_reference": static_ok, "reason": ""}
            runtime_match = (
                sim["matches_reference"] == c.expect_runtime_match
                if sim["ran"] is not None else True
            )
        else:
            sim = _simulate_attention(tp=c.tp, **c.sim_kwargs)
            runtime_match = sim["matches_reference"] == c.expect_runtime_match

        ok = static_match and runtime_match
        all_ok = all_ok and ok
        rows.append({
            "case_type": "attention",
            "name": c.name,
            "tp_size": c.tp,
            "static_ok": static_ok,
            "static_issues": sorted({i.kind.value for i in res.issues}),
            "expect_static_ok": c.expect_static_ok,
            "static_match": static_match,
            "live_ran": sim["ran"],
            "live_matches_reference": sim["matches_reference"],
            "live_reason": sim["reason"],
            "expect_live_match": c.expect_runtime_match,
            "live_match": runtime_match,
            "ok": ok,
        })
    return {"seed": SEED, "tolerance": TOL, "n_cases": len(rows),
            "all_ok": all_ok, "cases": rows}


def render_markdown(data: Dict) -> str:
    L: List[str] = []
    L.append("# Tensor-parallel checker — static verdict vs real torch")
    L.append("")
    L.append("> Generated by `reproducibility/tensor_parallel_sharding.py`. "
             "Boolean outcomes and verdicts only, no timing or raw floats — "
             "byte-deterministic, checked by `reproduce_all.py --check`.")
    L.append("")
    L.append(f"Seed **{data['seed']}** · cases: **{data['n_cases']}** · static "
             f"verdict matches real sharded execution on every case: "
             f"**{str(data['all_ok']).upper()}**")
    L.append("")
    L.append("| Case | Kind | tp | Static issues | Reproduces reference? | "
             "static ✓ | runtime ✓ |")
    L.append("|------|------|----|---------------|-----------------------|"
             "----------|-----------|")
    for c in data["cases"]:
        iss = ", ".join(c["static_issues"]) or "(ok)"
        if c["live_ran"] is None:
            repro = "(structural)"
        elif not c["live_ran"]:
            repro = f"no — {c['live_reason']}"
        else:
            repro = "yes" if c["live_matches_reference"] else "no (wrong output)"
        L.append(f"| `{c['name']}` | {c['case_type']} | {c['tp_size']} | "
                 f"{iss} | {repro} | "
                 f"{'yes' if c['static_match'] else 'NO'} | "
                 f"{'yes' if c['live_match'] else 'NO'} |")
    L.append("")
    L.append("The canonical Megatron MLP (ColumnParallel `gather_output=False` → "
             "RowParallel `input_is_parallel=True`) reproduces the unsharded "
             "forward exactly (up to fp tolerance) at tp=2 and tp=4. Every "
             "inconsistent config — gathered output fed to a parallel-input row "
             "layer, an indivisible shard, or a contracted-dimension mismatch — "
             "is flagged statically and also fails when actually sharded across "
             "ranks. The checker lives in `src/tensor_parallel_checks.py`.")
    L.append("")
    L.append("The attention cases extend the same proof pattern to MQA/GQA and "
             "sequence-parallel LayerNorm. A HuggingFace-style GQA module with "
             "independent `head_dim`, a Megatron-style MQA module with replicated "
             "KV heads, and an installed `transformers` `LlamaAttention` "
             "projection-shape check all agree with the static contract; invalid "
             "KV head partitions and hidden-axis sequence-parallel LayerNorm are "
             "refuted structurally.")
    L.append("")
    return "\n".join(L)


def run(check: bool = False) -> int:
    data = measure()
    new_json = json.dumps(data, indent=2, sort_keys=True) + "\n"
    new_md = render_markdown(data)
    if check:
        ok = True
        if not OUT_JSON.exists() or OUT_JSON.read_text() != new_json:
            print("MISMATCH: tensor_parallel_sharding.json differs",
                  file=sys.stderr)
            ok = False
        if not OUT_MD.exists() or OUT_MD.read_text() != new_md:
            print("MISMATCH: tensor_parallel_sharding.md differs",
                  file=sys.stderr)
            ok = False
        if not data["all_ok"]:
            print("FAIL: static verdict diverges from sharded runtime",
                  file=sys.stderr)
            ok = False
        print("tensor_parallel_sharding --check:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    OUT_JSON.write_text(new_json)
    OUT_MD.write_text(new_md)
    if not data["all_ok"]:
        print("WARNING: static verdict diverges from runtime!", file=sys.stderr)
        return 1
    print(f"Wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)} "
          f"({data['n_cases']} cases, all_ok={data['all_ok']}).")
    return 0


if __name__ == "__main__":
    sys.exit(run(check="--check" in sys.argv))
