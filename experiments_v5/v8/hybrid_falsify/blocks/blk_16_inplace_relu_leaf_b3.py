import torch
import torch.nn as nn
import torch.nn.functional as F

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B3"
REASON = (
    "weight.data.zero_() in forward: in-place op on leaf with requires_grad=True. "
    "FT forward succeeds (data= bypasses autograd check); "
    "TG grad-flag verifier catches B3 (in-place on leaf parameter)."
)


class blk_16_inplace_relu_leaf_b3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # B3: silently corrupts weight every forward via .data (bypasses FT check)
        with torch.no_grad():
            self.fc.weight.data.zero_()
        return self.fc(x)
