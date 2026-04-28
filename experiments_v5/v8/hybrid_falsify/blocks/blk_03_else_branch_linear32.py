import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 64)}
FT_INPUT_SHAPES = {"x": (8, 64)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Else-branch bug: TG checks all branches; FT with batch=8>4 takes safe branch."


class blk_03_else_branch_linear32(nn.Module):
    def __init__(self):
        super().__init__()
        self.safe = nn.Linear(64, 10)
        self.buggy = nn.Linear(32, 10)

    def forward(self, x):
        if x.shape[0] > 4:
            return self.safe(x)
        else:
            return self.buggy(x)
