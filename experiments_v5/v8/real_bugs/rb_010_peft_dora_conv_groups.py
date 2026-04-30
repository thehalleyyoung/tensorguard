"""
Real Bug Repro: PEFT DoRA Conv2d groups=2 weight reshape mismatch

GitHub Issue: https://github.com/huggingface/peft/issues/2549
Repository:   huggingface/peft (Hugging Face PEFT library)
Method:       DoRA (Weight-Decomposed Low-Rank Adaptation)

Bug: DoRA normalizes weight columns for Conv2d layers. It reshapes the weight
as `weight.view(weight.shape[0], -1)` to get a (out_channels, rest) matrix.
For a Conv2d with groups=1, in_channels=C, out_channels=O, kernel_size=k:
  weight shape = (O, C, k, k), view gives (O, C*k*k).

When groups=2 is used: the PyTorch Conv2d weight actually stores full filter rows
including the group-split info, and DoRA's view target assumed groups=1 dimensioning.
Specifically, DoRA computed the "expected" reshape as if all input channels were
available to each output filter, but the actual weight has a different organization
under groups=2, causing element count mismatch.

Original error:
  RuntimeError: shape '[192, 48, 3, 3]' is invalid for input of size 165888
  When groups=1: weight shape = (192, 96, 3, 3), total = 165888 ✓
  But the view target uses (192, 48, 3, 3) = 82944 ≠ 165888

Root cause: the view target used in_channels/groups=48 for the second dimension,
but the actual weight tensor has in_channels=96 giving 165888 total elements.

Substitution note: out_channels=192, in_channels=96, groups=2, kernel=3.
View target incorrectly uses in_channels//groups=48 instead of in_channels=96.
"""
import torch
import torch.nn as nn

# Conv2d(in_channels=96, out_channels=192, kernel=3) weight: (192, 96, 3, 3) = 165888 elements
# DoRA incorrectly treats as if weight were (192, 48, 3, 3) = 82944 elements
INPUT_SHAPES = {"x": (165888,)}


class BuggyModule(nn.Module):
    def forward(self, x):
        # Bug: view uses in_channels//groups=48 but actual dim is in_channels=96
        # 192 * 48 * 3 * 3 = 82,944 ≠ 165,888
        return x.view(192, 48, 3, 3)
