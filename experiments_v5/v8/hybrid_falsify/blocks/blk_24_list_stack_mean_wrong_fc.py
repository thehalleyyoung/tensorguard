import torch
import torch.nn as nn

CATEGORY = "C_ft_only"
TG_INPUT_SHAPES = {"x": (2, 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Verified"
EXPECTED_FT_VERDICT = "Refuted"
REASON = (
    "Dynamic half-slice: half=x.size(1)//2=16; fc expects 32 but gets 16. "
    "TG cannot track integer-divide of dynamic size; FT traces concretely and refutes."
)


class blk_24_list_stack_mean_wrong_fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 10)  # bug: gets 16 after half-slice

    def forward(self, x):
        # TG misses: half is computed dynamically from x.size(1)
        half = x.size(1) // 2   # = 16 for x.shape[1]=32
        return self.fc(x[:, :half])  # (B, 16) != Linear(32, 10)
