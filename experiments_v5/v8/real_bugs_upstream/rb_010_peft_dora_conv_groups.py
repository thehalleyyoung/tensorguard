"""
Upstream-faithful real-bug repro: PEFT DoRA Conv2d weight reshape under groups>1.
GitHub Issue: https://github.com/huggingface/peft/issues/2549
Buggy file  : peft/tuners/dora.py (DoraConv2dLayer pre-fix)
"""
import torch
import torch.nn as nn


class BuggyModule(nn.Module):
    def __init__(self, in_channels=96, out_channels=192, kernel=3, groups=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.groups = groups
        # Real Conv2d weight: (out, in // groups, k, k)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                              groups=groups, bias=False)

    def forward(self, _x):
        weight = self.conv.weight
        # BUG (pre-fix): view target multiplies in_channels (NOT in_channels // groups).
        return weight.view(self.out_channels, self.in_channels, self.kernel, self.kernel)


INPUT_SHAPES = {"_x": (1,)}
