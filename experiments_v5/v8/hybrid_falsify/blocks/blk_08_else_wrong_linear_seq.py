import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", "seq", 16)}
FT_INPUT_SHAPES = {"x": (2, 32, 16)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Else-branch with wrong Linear(8,10); FT with seq=32>16 takes safe branch."


class blk_08_else_wrong_linear_seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.safe = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10))
        self.wrong = nn.Linear(8, 10)

    def forward(self, x):
        if x.shape[1] > 16:
            return self.safe(x)
        else:
            return self.wrong(x)
