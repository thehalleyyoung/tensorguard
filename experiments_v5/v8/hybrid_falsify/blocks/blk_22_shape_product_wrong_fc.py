import torch
import torch.nn as nn

CATEGORY = "C_ft_only"
TG_INPUT_SHAPES = {"x": (2, 8, 8)}
FT_INPUT_SHAPES = {"x": (2, 8, 8)}
EXPECTED_TG_VERDICT = "Verified"
EXPECTED_FT_VERDICT = "Refuted"
REASON = "shape arithmetic d=shape[1]*shape[2]; TG misses, FT refutes."


class blk_22_shape_product_wrong_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        n = x.shape[0]
        d = x.shape[1] * x.shape[2]
        return self.fc(x.view(n, d))
