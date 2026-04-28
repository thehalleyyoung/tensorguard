import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 64)}
FT_INPUT_SHAPES = {"x": (2, 64)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "If-branch wrong Linear(32,10); FT with batch=2<=4 takes else (safe)."


class blk_10_if_branch_wrong_fc_size(nn.Module):
    def __init__(self):
        super().__init__()
        self.wrong = nn.Linear(32, 10)
        self.safe = nn.Linear(64, 10)

    def forward(self, x):
        if x.shape[0] > 4:
            return self.wrong(x)
        else:
            return self.safe(x)
