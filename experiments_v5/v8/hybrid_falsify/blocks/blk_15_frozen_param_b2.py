import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B2"
REASON = "fc.weight.requires_grad_(False) on a parameter expected to learn."


class blk_15_frozen_param_b2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)
        self.fc.weight.requires_grad_(False)

    def forward(self, x):
        return self.fc(x)
