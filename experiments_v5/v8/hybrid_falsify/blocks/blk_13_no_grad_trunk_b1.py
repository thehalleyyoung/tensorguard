import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B1"
REASON = "Trunk runs inside no_grad; trunk params silently get None grad."


class blk_13_no_grad_trunk_b1(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(32, 16)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            x = self.trunk(x)
        return self.head(x)
