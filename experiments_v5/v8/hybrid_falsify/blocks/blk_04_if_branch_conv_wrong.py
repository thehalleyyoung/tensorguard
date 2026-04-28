import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 3, "H", "W")}
FT_INPUT_SHAPES = {"x": (2, 3, 8, 8)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "If-branch wrong conv (16 in but x has 3 channels). FT with batch=2<5 takes else."


class blk_04_if_branch_conv_wrong(nn.Module):
    def __init__(self):
        super().__init__()
        self.buggy_conv = nn.Conv2d(16, 32, 3, padding=1)
        self.safe_conv = nn.Conv2d(3, 32, 3, padding=1)

    def forward(self, x):
        if x.shape[0] > 5:
            return self.buggy_conv(x)
        else:
            return self.safe_conv(x)
