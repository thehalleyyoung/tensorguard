import torch
import torch.nn as nn

CATEGORY = "C_ft_only"
TG_INPUT_SHAPES = {"x": (2, 8, 8)}
FT_INPUT_SHAPES = {"x": (2, 8, 8)}
EXPECTED_TG_VERDICT = "Verified"
EXPECTED_FT_VERDICT = "Refuted"
REASON = (
    "b=x.size(0) view(-1): view(b,-1) flattens to (2,64) but fc expects 100. "
    "TG cannot evaluate x.size(0) assigned to variable; FT traces concretely and refutes."
)


class blk_25_adaptive_pool_size0_wrong_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)  # bug: actual flat size is 8*8=64

    def forward(self, x):
        # TG misses: b assigned from x.size(0) then passed to view
        b = x.size(0)
        return self.fc(x.view(b, -1))  # (B, 64) != Linear(100, 10)
