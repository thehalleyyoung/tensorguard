import torch
import torch.nn as nn

CATEGORY = "A_symbolic_shape"
TG_INPUT_SHAPES = {"x": ("batch", 4)}
FT_INPUT_SHAPES = {"x": (32, 4)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
REASON = (
    "Hardcoded view(8,-1): valid only when batch*4/8=16, i.e. batch=32. "
    "TG symbolic batch refutes; FT with batch=32 gives view(8,16)->Linear(16,10) OK."
)


class blk_09_hardcoded_pool_view(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Bug: hardcoded first dim; valid only when batch=32
        x = x.view(8, -1)
        return self.fc(x)
