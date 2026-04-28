import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", "seq", 32)}
FT_INPUT_SHAPES = {"x": (2, 16, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Else branch wrong embed dim. FT with seq=16>8 takes safe branch."


class blk_06_else_branch_wrong_embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.safe = nn.Linear(32, 10)
        self.wrong = nn.Linear(64, 10)

    def forward(self, x):
        if x.shape[1] > 8:
            return self.safe(x)
        else:
            return self.wrong(x)
