"""
Unfiltered post-freeze repro #13 (in-fragment, literal divisibility):
HF peft - VeRA adapter rank/dim mismatch.

GitHub PR  : https://github.com/huggingface/peft/pull/3208  (merged 2026-04-20)
Repository : huggingface/peft
Buggy file : src/peft/tuners/vera/layer.py
            (pre-#3208, VeRALayer.forward)

Root cause: the shared random projection matrix had shape
(in_features, r) but the per-task scaler vector was sized to
out_features (= 4 * r in this checkpoint), so the elementwise
multiply broadcast to a 4*r * r tensor instead of in_features.
The mismatch is on a literal product.

In-fragment, expected verdict: RP at >= 0.99.
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, in_features=64, r=16, out_features=64):
        super().__init__()
        # Shared random projection (frozen).
        self.A = nn.Parameter(torch.randn(in_features, r), requires_grad=False)
        # BUG (pre-#3208): per-task scaler sized to (out_features,)
        # but applied elementwise to (B, in_features) input.
        self.scaler = nn.Parameter(torch.ones(out_features))
        self.B_proj = nn.Linear(r, out_features)

    def forward(self, x):
        # x: (B, in_features)
        # BUG: elementwise multiply against scaler whose dim is
        # out_features, not in_features.
        scaled = x * self.scaler  # mismatched broadcast
        proj = scaled @ self.A
        return self.B_proj(proj)


INPUT_SHAPES = {"x": (4, 64)}
