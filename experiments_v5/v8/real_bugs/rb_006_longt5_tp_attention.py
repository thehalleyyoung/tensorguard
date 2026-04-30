"""
Real Bug Repro: LongT5/MT5 attention shape mismatch under Tensor Parallelism

GitHub PR:   https://github.com/huggingface/transformers/pull/45109
Repository:  huggingface/transformers
Models:      LongT5, MT5, Pop2Piano

Bug: When using Tensor Parallelism (TP=4), the query projection linear layer
`q = nn.Linear(d_model, d_kv * num_heads)` is sharded across devices, reducing
its output from `d_kv * num_heads` to `d_kv * num_heads / TP`. The subsequent
view `q.view(batch, seq_len, num_heads, d_kv)` expects the FULL `d_kv * num_heads`
features but only receives the sharded portion.

Example: d_model=512, d_kv=64, num_heads=8, TP=4.
- Non-sharded q: nn.Linear(512, 512) → output 512 features/token
- Sharded q under TP=4: nn.Linear(512, 128) → output 128 features/token
- View target: (batch, seq, 8, 64) requires 512 features/token

With TP=4: 128 / (8 * 64) = 0.25 (non-integer) → view fails.

Substitution note: batch=2, seq_len=10, d_model=512, d_kv=64, num_heads=8, TP=4.
"""
import torch
import torch.nn as nn

# Input to the module: hidden states (batch=2, seq=10, d_model=512)
INPUT_SHAPES = {"x": (2, 10, 512)}


class BuggyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Sharded under TP=4: output is d_kv * num_heads / TP = 512 / 4 = 128
        self.q = nn.Linear(512, 128)

    def forward(self, x):
        # Bug: view expects 8 heads * 64 dim = 512 features,
        # but after TP sharding q only produces 128 features.
        # -1 resolves to 10 (sequence length), but 128 / (8 * 64) = 0.25 ← not integer
        # TG sees: (2, 10, 128) → view(2, -1, 8, 64) → 128 / (8*64) = 0.25 → error
        return self.q(x).view(2, -1, 8, 64)
