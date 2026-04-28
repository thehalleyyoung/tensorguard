import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 4, 4)}
FT_INPUT_SHAPES = {"x": (64, 4, 4)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = (
    "Hardcoded view(16,-1): requires batch*16=16*64 so batch must be 64. "
    "TG symbolic batch refutes; FT with batch=64 gives view(16,64)->Linear(64,10) OK."
)


class blk_02_hardcoded_batch16(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Bug: hardcoded first dim of view; valid only when batch=64
        x = x.view(16, -1)
        return self.fc(x)
