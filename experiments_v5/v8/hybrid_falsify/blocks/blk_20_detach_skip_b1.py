import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B1"
REASON = "Skip connection routes through detach; residual params get no grad."


class blk_20_detach_skip_b1(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 32)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        residual = self.proj(x).detach()
        x = x + residual
        return self.fc(x)
