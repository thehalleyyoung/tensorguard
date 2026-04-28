import torch
import torch.nn as nn

CATEGORY = "B_grad_flag"
TG_INPUT_SHAPES = {"x": ("batch", 32)}
FT_INPUT_SHAPES = {"x": (2, 32)}
EXPECTED_TG_VERDICT = "Refuted"
EXPECTED_FT_VERDICT = "Verified"
GRAD_BUG_KIND = "B2"
REASON = "Backbone params frozen by requires_grad_(False); user expects them to learn."


class blk_19_frozen_backbone_b2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(32, 16)
        self.head = nn.Linear(16, 10)
        self.backbone.requires_grad_(False)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
