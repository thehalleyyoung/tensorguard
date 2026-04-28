import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B1"
REASON = "Two layers nested inside no_grad; head sees no grads downstream."


class blk_18_nested_no_grad_b1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32, 32)
        self.l2 = nn.Linear(32, 16)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            x = self.l1(x)
            with torch.no_grad():
                x = self.l2(x)
        return self.head(x)
