"""
Upstream-faithful real-bug repro: EleutherAI gpt-neox GQA reshape.
GitHub Issue: https://github.com/EleutherAI/gpt-neox/issues/1314
Fixed in PR : https://github.com/EleutherAI/gpt-neox/pull/1315
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, np=5, kvp=1, hn=128):
        super().__init__()
        self.np = np
        self.kvp = kvp
        self.hn = hn

    def forward(self, mixed_x_layer):
        sq, b, _ = mixed_x_layer.shape
        # BUG (pre-#1315): silent int() truncation of fractional head dim.
        fake_head_dim = int(self.hn * (1 + 2 * self.kvp / self.np))
        return mixed_x_layer.view(sq, b, self.np, fake_head_dim)


INPUT_SHAPES = {"mixed_x_layer": (4096, 1, (5 + 2 * 1) * 128)}
