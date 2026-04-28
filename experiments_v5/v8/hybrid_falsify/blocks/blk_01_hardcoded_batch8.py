import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 8, 8)}
FT_INPUT_SHAPES = {"x": (8, 8, 8)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Hardcoded view(8,-1): TG with symbolic batch refutes; FT with batch=8 succeeds."


class blk_01_hardcoded_batch8(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(8, -1)
        return self.fc(x)
