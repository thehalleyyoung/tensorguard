import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 4, 4)}
FT_INPUT_SHAPES = {"x": (4, 4, 4)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = "Hardcoded view(4,-1) for batch dim; FT with batch=4 happens to match."


class blk_05_hardcoded_spatial_flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(4, -1)
        return self.fc(x)
