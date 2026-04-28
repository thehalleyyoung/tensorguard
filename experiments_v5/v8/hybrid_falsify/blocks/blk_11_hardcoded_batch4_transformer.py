import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 4, 8)}
FT_INPUT_SHAPES = {"x": (4, 4, 8)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = (
    "Hardcoded view(4,-1): requires batch*32/4=32, i.e. batch=4. "
    "TG symbolic batch refutes; FT with batch=4 gives view(4,32)->Linear(32,10) OK."
)


class blk_11_hardcoded_batch4_transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Bug: hardcoded first dim of view; valid only when batch=4
        x = x.view(4, -1)
        return self.fc(x)
