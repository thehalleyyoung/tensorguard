import torch
import torch.nn as nn

CATEGORY = "C_ft_only"
TG_INPUT_SHAPES = {"x": (2, 8, 8)}
FT_INPUT_SHAPES = {"x": (2, 8, 8)}
EXPECTED_TG_VERDICT = "Verified"
EXPECTED_FT_VERDICT = "Refuted"
REASON = "size(0)/view(b,-1) arithmetic: TG misses; FT runs concretely and refutes."


class blk_21_view_size0_wrong_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)  # expects 100 but flattened input is 8*8=64

    def forward(self, x):
        b = x.size(0)
        return self.fc(x.view(b, -1))
