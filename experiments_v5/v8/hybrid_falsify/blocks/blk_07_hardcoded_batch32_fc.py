import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (32, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Hardcoded reshape(32,-1); FT with batch=32 succeeds."


class blk_07_hardcoded_batch32_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(32, -1)
        return self.fc(x)
