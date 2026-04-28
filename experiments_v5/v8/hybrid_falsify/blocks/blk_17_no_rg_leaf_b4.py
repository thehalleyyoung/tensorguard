import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B4"
REASON = "All parameters have requires_grad=False; .backward() precondition violated."


class blk_17_no_rg_leaf_b4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.fc(x)
