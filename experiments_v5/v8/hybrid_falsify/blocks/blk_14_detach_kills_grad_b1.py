import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B1"
REASON = "x.detach() before fc severs all parameter gradients."


class blk_14_detach_kills_grad_b1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = x.detach()
        return self.fc(x)
