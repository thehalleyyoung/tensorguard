import torch
import torch.nn as nn

CATEGORY = "C_ft_only"
TG_INPUT_SHAPES = {"x": (2, 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Verified"
EXPECTED_FT_VERDICT = "Refuted"
REASON = "half = shape[1]//2 slice; TG misses // arithmetic; FT refutes."


class blk_23_halved_slice_wrong_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)  # expects 32 but receives 16

    def forward(self, x):
        half = x.shape[1] // 2
        return self.fc(x[:, :half])
