# Real Bug Corpus — TensorGuard v5 Evaluation

Ten publicly-citeable PyTorch shape bugs mined from popular open-source repositories
(HuggingFace Transformers, EleutherAI GPT-NeoX, HuggingFace Diffusers, HuggingFace PEFT).
Each bug produces a **REFUTED-PROOF (RP)** verdict from TensorGuard's `verify_architecture`
at ≥0.99 confidence. Bugs were verified on their minimal reproducible forms with hardcoded
integer dimensions substituted from original issue/PR error messages.

## Summary Table

| ID     | Model / Library                  | Category           | TG Conf | Source                                  |
|--------|----------------------------------|--------------------|---------|------------------------------------------|
| rb_001 | xLSTM (mLSTM chunkwise)         | reshape_size       | 0.99    | HF transformers #43208, PR #43209        |
| rb_002 | xLSTM (mLSTM chunkwise)         | reshape_size       | 0.99    | HF transformers #43208, PR #43209        |
| rb_003 | GPT-NeoX                        | linear_view_chain  | 0.99    | HF transformers #23081                   |
| rb_004 | ConvBERT                        | reshape_size       | 0.99    | HF transformers #21523                   |
| rb_005 | Longformer                      | reshape_size       | 0.99    | HF transformers #5646                    |
| rb_006 | LongT5 / MT5 / Pop2Piano        | linear_view_chain  | 0.99    | HF transformers PR #45109                |
| rb_007 | GPT-NeoX (GQA)                  | reshape_size       | 0.99    | EleutherAI/gpt-neox #1314, PR #1315     |
| rb_008 | Diffusers UNet1DModel           | linear_inout       | 0.99    | HF diffusers #12110, PR #12111           |
| rb_009 | PEFT PrefixTuning (flan-t5-sm.) | reshape_size       | 0.99    | HF peft #385                             |
| rb_010 | PEFT DoRA (Conv2d groups=2)     | reshape_size       | 0.99    | HF peft #2549                            |

All 10 bugs verified: `$ PYTHONPATH=. python3.11 experiments_v5/v8/verify_real_bugs.py`

---

## Bug Details

### rb_001 — xLSTM matQ view: dqk not divided by nc

- **Repo / Issue**: huggingface/transformers [#43208](https://github.com/huggingface/transformers/issues/43208)
- **Fixed in**: [PR #43209](https://github.com/huggingface/transformers/pull/43209)
- **Error**: `shape '[12, 4, 8, 64, 768]' is invalid for input of size 2359296`
- **Root cause**: `matQ.view(batch_size, nh, nc, chunk_size, dqk)` uses `dqk=384`
  instead of `dqk//nc=192`, doubling the last dimension.
- **TG detection**: `INPUT=(2,4,128,192)` → `view(2,4,2,64,384)`:
  196 608 ≠ 393 216

### rb_002 — xLSTM matK view: same bug on key tensor

- **Repo / Issue**: huggingface/transformers [#43208](https://github.com/huggingface/transformers/issues/43208)
- **Fixed in**: [PR #43209](https://github.com/huggingface/transformers/pull/43209)
- **Error**: same error class as rb_001 but for the key matrix `matK`.
- **TG detection**: identical to rb_001 (separate tensor, same pattern).

### rb_003 — GPT-NeoX: odd num_heads causes QKV view mismatch

- **Repo / Issue**: huggingface/transformers [#23081](https://github.com/huggingface/transformers/issues/23081)
- **Error**: Integer division `1024 // 12 = 85` so view target `(1,5,12,255)` needs
  `12×255=3060` features/token but `Linear(1024,3072)` outputs 3072.
- **TG detection**: `Linear(1024,3072) → view(1,5,12,255)`: 3072 ≠ 3060

### rb_004 — ConvBERT: wrong head_ratio in mixed attention view

- **Repo / Issue**: huggingface/transformers [#21523](https://github.com/huggingface/transformers/issues/21523)
- **Error**: Mixed attention output has 192 features/token but view target uses 384.
- **TG detection**: `INPUT=(3,10,192)` → `view(3,10,384)`: 5760 ≠ 11520

### rb_005 — Longformer: global attention view dimension swap

- **Repo / Issue**: huggingface/transformers [#5646](https://github.com/huggingface/transformers/issues/5646)
- **Error**: `attn_probs` is `(batch, heads, seq, global)` but view targets
  `(batch, heads, global, seq)` with incompatible element counts.
- **TG detection**: `INPUT=(1,12,512,518)` → `view(1,12,5,512)`: 3 182 592 ≠ 30 720

### rb_006 — LongT5/MT5/Pop2Piano: query sharded under TP=4

- **Repo / PR**: huggingface/transformers [PR #45109](https://github.com/huggingface/transformers/pull/45109)
- **Error**: Under TP=4, `q = Linear(512, 128)` outputs 128/token; `view(batch,-1,8,64)`
  requires 512/token; 128/(8×64) = 0.25 (non-integer).
- **TG detection**: `Linear(512,128) → view(2,-1,8,64)`: non-integer −1 resolution

### rb_007 — EleutherAI GPT-NeoX: GQA fake-head-dim truncation

- **Repo / Issue**: EleutherAI/gpt-neox [#1314](https://github.com/EleutherAI/gpt-neox/issues/1314)
- **Fixed in**: [PR #1315](https://github.com/EleutherAI/gpt-neox/pull/1315) (commit 96c242eb)
- **Error**: `shape '[4096, 1, 5, 179]' is invalid for input of size 3670016`
- **Root cause**: `int(128 × (1 + 2 × (1/5))) = int(179.2) = 179`; 5×179=895 ≠ 7×128=896.
- **TG detection**: `INPUT=(4096,1,896)` → `view(4096,1,5,179)`: 3 670 016 ≠ 3 665 920

### rb_008 — Diffusers UNet1DModel: GaussianFourier hardcoded embedding_size=8

- **Repo / Issue**: huggingface/diffusers [#12110](https://github.com/huggingface/diffusers/issues/12110)
- **Fixed in**: [PR #12111](https://github.com/huggingface/diffusers/pull/12111) (commit 751e250f)
- **Error**: `mat1 and mat2 shapes cannot be multiplied (1x16 and 64x128)`
- **Root cause**: `GaussianFourierProjection(embedding_size=8)` → 16 features;
  `TimestepEmbedding.linear_1 = Linear(64,128)` expected 64.
- **TG detection**: `INPUT=(1,16)` → `Linear(64,128)`: last dim 16 ≠ 64

### rb_009 — PEFT PrefixTuning: flan-t5-small non-divisible d_model/num_heads

- **Repo / Issue**: huggingface/peft [#385](https://github.com/huggingface/peft/issues/385)
- **Error**: `shape '[8, 8, 16, 6, 85]' is invalid for input of size 524288`
- **Root cause**: `512 // 6 = 85` (integer division); `8×8×16×6×85 = 522240 ≠ 524288`.
- **TG detection**: `INPUT=(8,8,8192)` → `view(8,8,16,6,85)`: 524 288 ≠ 522 240

### rb_010 — PEFT DoRA: Conv2d groups=2 weight reshape mismatch

- **Repo / Issue**: huggingface/peft [#2549](https://github.com/huggingface/peft/issues/2549)
- **Error**: `shape '[192, 48, 3, 3]' is invalid for input of size 165888`
- **Root cause**: DoRA uses `in_channels//groups=48` for weight dim but actual weight has
  `in_channels=96` giving `(192,96,3,3)` = 165888 elements; view target = 82944.
- **TG detection**: `INPUT=(165888,)` → `view(192,48,3,3)`: 165 888 ≠ 82 944

---

## Methodology

1. **Search**: GitHub issue/PR search for "shape is invalid for input of size" and
   "mat1 and mat2 shapes cannot be multiplied" across popular ML repos.
2. **Filter**: Retained only bugs where: (a) the root cause is a static-integer
   view/reshape or Linear chain mismatch, (b) the error message contains concrete
   integer values, and (c) the bug is in model code (not user input misconfiguration).
3. **Substitute**: Where source code used `self.config.*`, replaced with concrete
   integers from the reported error message.
4. **Verify**: Ran TensorGuard `verify_architecture` on each minimal repro; confirmed
   ≥0.99 confidence RP verdict.

## Reproducibility

```bash
cd tensorguard
PYTHONPATH=. python3.11 -c "
from src.api import verify_architecture
import os, re

base = 'experiments_v5/v8/real_bugs'
for fname in sorted(os.listdir(base)):
    if not fname.endswith('.py'): continue
    src = open(os.path.join(base, fname)).read()
    shapes = eval(re.search(r'INPUT_SHAPES\s*=\s*(\{[^}]+\})', src).group(1))
    r = verify_architecture(src, input_shapes=shapes)
    conf = max((b.confidence for b in r.bugs), default=0)
    print(f'{fname}: {conf:.2f}')
"
```
