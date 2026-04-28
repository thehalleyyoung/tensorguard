import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 8, "H", "W")}
FT_INPUT_SHAPES = {"x": (2, 16, 8, 8)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = (
    "Else-branch wrong Conv2d(4,16) when C=8 (8>8 is False → else). "
    "TG with concrete C=8 catches mismatch; FT with C=16>8 takes safe branch."
)


class blk_12_else_branch_wrong_conv_channels(nn.Module):
    def __init__(self):
        super().__init__()
        self.safe = nn.Conv2d(16, 32, 1)
        self.wrong = nn.Conv2d(4, 16, 1)

    def forward(self, x):
        if x.shape[1] > 8:
            return self.safe(x)
        else:
            # Bug: wrong expects 4 input channels but x has 8
            return self.wrong(x)
