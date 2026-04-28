"""
L2-device stress case 02: positional embedding buffer mixed with CUDA attention.

Target feature: device-consistency check (L2).
Bug: pos_embed is a registered CPU buffer; attention output is on CUDA when
the model runs on GPU. Adding them causes a device mismatch.

Expected:
  WITHOUT L2: Verified
  WITH    L2: Refuted (DEVICE-MISMATCH on pos_embed + attention output)
"""
import torch
import torch.nn as nn


class PositionalAttention(nn.Module):
    def __init__(self, d_model: int = 128, seq_len: int = 64):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        # BUG: positional embedding stays on CPU
        self.register_buffer("pos_embed", torch.zeros(1, seq_len, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is on CUDA; pos_embed is on CPU → device mismatch
        x = x + self.pos_embed
        out, _ = self.attn(x, x, x)
        return self.norm(out)


FEATURE = "L2_device"
INPUT_SHAPES = {"x": ("batch", 64, 128)}
EXPECTED_WITHOUT = "Verified"
EXPECTED_WITH    = "Refuted"
